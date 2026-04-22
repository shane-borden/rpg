//! Google Gemini provider implementation.

use super::{CompletionOptions, CompletionResult, LlmProvider, Message, Role};
#[cfg(not(target_arch = "wasm32"))]
use futures::StreamExt;

// ---------------------------------------------------------------------------
// GeminiProvider
// ---------------------------------------------------------------------------

/// LLM provider backed by the Google Gemini API.
#[derive(Debug)]
pub struct GeminiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
    /// Timeout in seconds for each HTTP request.
    timeout_secs: u64,
}

impl GeminiProvider {
    /// Create a new `GeminiProvider`.
    ///
    /// `base_url` defaults to `https://generativelanguage.googleapis.com` when `None`.
    /// `timeout_secs` sets the HTTP request timeout; must be > 0.
    pub fn new(api_key: String, base_url: Option<String>, timeout_secs: u64) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("failed to build HTTP client");
        #[cfg(target_arch = "wasm32")]
        let client = reqwest::Client::builder()
            .build()
            .expect("failed to build HTTP client");
        Self {
            api_key,
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_owned()),
            client,
            timeout_secs,
        }
    }

    /// Prepare the JSON body for the Gemini API.
    fn prepare_body(&self, messages: &[Message], options: &CompletionOptions) -> serde_json::Value {
        let mut contents: Vec<serde_json::Value> = Vec::new();
        let mut system_instruction = None;

        for m in messages {
            match m.role {
                Role::System => {
                    system_instruction = Some(serde_json::json!({
                        "parts": [{ "text": m.content }]
                    }));
                }
                Role::User | Role::Assistant => {
                    let role_str = if matches!(m.role, Role::User) {
                        "user"
                    } else {
                        "model"
                    };

                    if let Some(last) = contents.last_mut() {
                        if last["role"] == role_str {
                            // Merge with last message of the same role.
                            let last_text = last["parts"][0]["text"].as_str().unwrap_or("");
                            last["parts"][0]["text"] =
                                serde_json::json!(format!("{}\n\n{}", last_text, m.content));
                            continue;
                        }
                    }

                    contents.push(serde_json::json!({
                        "role": role_str,
                        "parts": [{ "text": m.content }]
                    }));
                }
            }
        }

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": options.max_tokens,
                "temperature": options.temperature,
            }
        });

        if let Some(si) = system_instruction {
            body["system_instruction"] = si;
        }

        body
    }

    /// Shared non-streaming completion logic.
    async fn complete_inner(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> Result<CompletionResult, String> {
        let model = if options.model.is_empty() {
            self.default_model().to_owned()
        } else {
            options.model.clone()
        };

        let body = self.prepare_body(&messages, &options);

        let timeout_secs = self.timeout_secs;
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );

        let resp = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    format!("AI request timed out after {timeout_secs}s")
                } else {
                    format!("Gemini API error: {e}")
                }
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(format!("Gemini API {status}: {body_text}"));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("Gemini response parse error: {e}"))?;

        let content = json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_owned();

        let input_tokens =
            u32::try_from(json["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0))
                .unwrap_or(u32::MAX);
        let output_tokens = u32::try_from(
            json["usageMetadata"]["candidatesTokenCount"]
                .as_u64()
                .unwrap_or(0),
        )
        .unwrap_or(u32::MAX);

        Ok(CompletionResult {
            content,
            input_tokens,
            output_tokens,
        })
    }

    /// Shared streaming completion logic (native only).
    #[cfg(not(target_arch = "wasm32"))]
    async fn complete_streaming_inner(
        &self,
        messages: Vec<Message>,
        options: CompletionOptions,
        on_token: Box<dyn Fn(&str) + Send>,
    ) -> Result<CompletionResult, String> {
        let model = if options.model.is_empty() {
            self.default_model().to_owned()
        } else {
            options.model.clone()
        };

        let body = self.prepare_body(&messages, &options);

        let timeout_secs = self.timeout_secs;
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        );

        let resp = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    format!("AI request timed out after {timeout_secs}s")
                } else {
                    format!("Gemini API error: {e}")
                }
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(format!("Gemini API {status}: {body_text}"));
        }

        let mut full_content = String::new();
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut input_tokens = 0;
        let mut output_tokens = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Stream error: {e}"))?;
            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buf.find('\n') {
                let line = buf[..newline_pos].trim_end().to_owned();
                buf = buf[newline_pos + 1..].to_owned();

                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(text) = json["candidates"][0]["content"]["parts"][0]["text"].as_str()
                        {
                            on_token(text);
                            full_content.push_str(text);
                        }
                        if let Some(usage) = json.get("usageMetadata") {
                            input_tokens = u32::try_from(usage["promptTokenCount"].as_u64().unwrap_or(0))
                                .unwrap_or(u32::MAX);
                            output_tokens = u32::try_from(
                                usage["candidatesTokenCount"]
                                    .as_u64()
                                    .unwrap_or(0),
                            )
                            .unwrap_or(u32::MAX);
                        }
                    }
                }
            }
        }

        Ok(CompletionResult {
            content: full_content,
            input_tokens,
            output_tokens,
        })
    }
}

// ---------------------------------------------------------------------------
// Native LlmProvider impl
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
impl LlmProvider for GeminiProvider {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn default_model(&self) -> &'static str {
        "gemini-3.1-pro-preview"
    }

    fn complete(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<CompletionResult, String>> + Send + '_>,
    > {
        let messages = messages.to_vec();
        let options = options.clone();
        Box::pin(self.complete_inner(messages, options))
    }

    fn complete_streaming(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
        on_token: Box<dyn Fn(&str) + Send>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<CompletionResult, String>> + Send + '_>,
    > {
        let messages = messages.to_vec();
        let options = options.clone();
        Box::pin(self.complete_streaming_inner(messages, options, on_token))
    }
}

// ---------------------------------------------------------------------------
// WASM LlmProvider impl
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
impl LlmProvider for GeminiProvider {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn default_model(&self) -> &'static str {
        "gemini-3.1-pro-preview"
    }

    fn complete(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CompletionResult, String>> + '_>>
    {
        let messages = messages.to_vec();
        let options = options.clone();
        Box::pin(self.complete_inner(messages, options))
    }

    fn complete_streaming(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
        on_token: Box<dyn Fn(&str)>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<CompletionResult, String>> + '_>>
    {
        let messages = messages.to_vec();
        let options = options.clone();
        Box::pin(async move {
            let result = self.complete_inner(messages, options).await?;
            on_token(&result.content);
            Ok(result)
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_name() {
        let p = GeminiProvider::new("key".to_owned(), None, 30);
        assert_eq!(p.name(), "gemini");
    }

    #[test]
    fn default_model() {
        let p = GeminiProvider::new("key".to_owned(), None, 30);
        assert_eq!(p.default_model(), "gemini-3.1-pro-preview");
    }

    #[test]
    fn default_base_url() {
        let p = GeminiProvider::new("key".to_owned(), None, 30);
        assert_eq!(p.base_url, "https://generativelanguage.googleapis.com");
    }

    #[test]
    fn timeout_stored() {
        let p = GeminiProvider::new("key".to_owned(), None, 45);
        assert_eq!(p.timeout_secs, 45);
    }

    #[test]
    fn consecutive_roles_merged() {
        let p = GeminiProvider::new("key".to_owned(), None, 30);
        let messages = vec![
            Message {
                role: Role::User,
                content: "part 1".to_owned(),
            },
            Message {
                role: Role::User,
                content: "part 2".to_owned(),
            },
            Message {
                role: Role::Assistant,
                content: "resp 1".to_owned(),
            },
            Message {
                role: Role::Assistant,
                content: "resp 2".to_owned(),
            },
        ];
        let options = CompletionOptions::default();
        let body = p.prepare_body(&messages, &options);

        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "part 1\n\npart 2");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["text"], "resp 1\n\nresp 2");
    }
}
