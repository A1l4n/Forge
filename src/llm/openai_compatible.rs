//! Provider for OpenAI-compatible chat completion endpoints.
//!
//! Works with: Ollama (`http://localhost:11434/v1`), OpenRouter, Groq,
//! LM Studio, vLLM, Together AI, and OpenAI itself. Tool-calling is best-effort —
//! many models silently ignore the `tools` parameter; Forge's main flow doesn't
//! depend on tool-calling in this provider.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{debug, warn};

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole};

use super::{LLMProvider, LLMResponse, StopReason, ToolCallInfo, Usage};

#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    stream: bool,
}

#[derive(Debug, Clone, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    model: String,
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallJson>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallJson {
    id: String,
    #[serde(default, rename = "type")]
    _kind: Option<String>,
    function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAIUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// OpenAI-compatible provider. Construct with [`OpenAICompatibleProvider::new`]
/// or one of the named convenience constructors.
pub struct OpenAICompatibleProvider {
    base_url: String,
    api_key: Option<String>,
    model: String,
    provider_name: String,
    http_client: HttpClient,
    max_tokens: Option<u32>,
}

impl OpenAICompatibleProvider {
    pub fn new(
        provider_name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            provider_name: provider_name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            model: model.into(),
            http_client,
            max_tokens: Some(4096),
        }
    }

    /// Local Ollama at `http://localhost:11434/v1`. No API key needed.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new("ollama", "http://localhost:11434/v1", None, model)
            .with_max_tokens(None) // Ollama handles context internally
    }

    /// LM Studio (default port 1234).
    pub fn lm_studio(model: impl Into<String>) -> Self {
        Self::new("lm-studio", "http://localhost:1234/v1", None, model)
    }

    /// OpenRouter — free + paid models, OpenAI-compatible.
    pub fn openrouter(api_key: String, model: impl Into<String>) -> Self {
        Self::new(
            "openrouter",
            "https://openrouter.ai/api/v1",
            Some(api_key),
            model,
        )
    }

    /// Groq — free, very fast Llama/Mixtral hosting.
    pub fn groq(api_key: String, model: impl Into<String>) -> Self {
        Self::new("groq", "https://api.groq.com/openai/v1", Some(api_key), model)
    }

    /// Together AI.
    pub fn together(api_key: String, model: impl Into<String>) -> Self {
        Self::new("together", "https://api.together.xyz/v1", Some(api_key), model)
    }

    /// OpenAI itself.
    pub fn openai(api_key: String, model: impl Into<String>) -> Self {
        Self::new("openai", "https://api.openai.com/v1", Some(api_key), model)
    }

    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl LLMProvider for OpenAICompatibleProvider {
    async fn generate(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Result<LLMResponse> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut chat_messages = Vec::with_capacity(messages.len() + 1);
        if !system.is_empty() {
            chat_messages.push(ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            });
        }
        for m in messages {
            let role = match m.role {
                MessageRole::User => "user",
                MessageRole::Luna | MessageRole::Agent => "assistant",
                MessageRole::System => "system",
            };
            chat_messages.push(ChatMessage {
                role: role.to_string(),
                content: m.content.clone(),
            });
        }

        // Translate Anthropic-style tool definitions to OpenAI tool schema.
        let openai_tools = tools.map(|defs| {
            defs.iter()
                .map(|t| {
                    let name = t.get("name").cloned().unwrap_or(json!(""));
                    let description = t.get("description").cloned().unwrap_or(json!(""));
                    let parameters = t
                        .get("input_schema")
                        .cloned()
                        .unwrap_or(json!({"type":"object","properties":{}}));
                    json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters,
                        }
                    })
                })
                .collect::<Vec<_>>()
        });

        let request = ChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            max_tokens: self.max_tokens,
            temperature: None,
            tools: openai_tools,
            stream: false,
        };

        debug!(provider = %self.provider_name, model = %self.model, "→ OpenAI-compatible /chat/completions");

        let mut req = self
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request);
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {}", key));
        }

        let response = req
            .send()
            .await
            .map_err(|e| Error::HTTP(format!("{} request: {}", self.provider_name, e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::HTTP(format!("read response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::ClaudeAPI(format!(
                "{} HTTP {}: {}",
                self.provider_name, status, body
            )));
        }

        let parsed: ChatResponse = serde_json::from_str(&body).map_err(|e| {
            warn!("{} parse failed: {}", self.provider_name, body);
            Error::Serialization(e)
        })?;

        let choice = parsed.choices.into_iter().next().ok_or_else(|| {
            Error::ClaudeAPI(format!("{}: no choices in response", self.provider_name))
        })?;

        let text = choice.message.content.unwrap_or_default();
        let mut tool_calls = Vec::new();
        if let Some(calls) = choice.message.tool_calls {
            for c in calls {
                let input: Value = serde_json::from_str(&c.function.arguments)
                    .unwrap_or_else(|_| Value::String(c.function.arguments.clone()));
                tool_calls.push(ToolCallInfo {
                    id: c.id,
                    name: c.function.name,
                    input,
                });
            }
        }

        let stop_reason = choice
            .finish_reason
            .as_deref()
            .map(StopReason::from_str)
            .unwrap_or(StopReason::EndTurn);

        let usage = parsed
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default();

        Ok(LLMResponse {
            text,
            tool_calls,
            stop_reason,
            model: if parsed.model.is_empty() {
                self.model.clone()
            } else {
                parsed.model
            },
            usage,
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        &self.provider_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_factory_sets_local_url() {
        let p = OpenAICompatibleProvider::ollama("llama3.1");
        assert_eq!(p.provider_name(), "ollama");
        assert_eq!(p.model(), "llama3.1");
    }

    #[test]
    fn base_url_trims_trailing_slash() {
        let p = OpenAICompatibleProvider::new(
            "test",
            "http://localhost:11434/v1/",
            None,
            "x",
        );
        assert!(!p.base_url.ends_with('/'));
    }
}
