//! Anthropic API provider (Claude).
//!
//! Uses the standard `x-api-key` auth scheme. For OAuth/Claude-Code-style
//! auth, swap `x-api-key` for `Authorization: Bearer ...` and add the
//! `anthropic-beta: oauth-2025-04-20` header — not done here because the
//! Pro/Max OAuth flow isn't an officially supported API path.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{debug, warn};

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole};

use super::{LLMProvider, LLMResponse, StopReason, ToolCallInfo, Usage};

const CLAUDE_API_VERSION: &str = "2024-06-01";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessageParam {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateMessageRequest {
    model: String,
    max_tokens: u32,
    system: Option<String>,
    messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateMessageResponse {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    role: String,
    content: Vec<ContentBlock>,
    model: String,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Anthropic provider — talks directly to the Claude Messages API.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    model: String,
    http_client: HttpClient,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            api_key,
            base_url: "https://api.anthropic.com".to_string(),
            model,
            http_client,
            max_tokens: 4096,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn generate(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Result<LLMResponse> {
        let url = format!("{}/v1/messages", self.base_url);

        let message_params: Vec<MessageParam> = messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::User => "user",
                    MessageRole::Luna | MessageRole::Agent => "assistant",
                    MessageRole::System => "user",
                };
                MessageParam {
                    role: role.to_string(),
                    content: msg.content.clone(),
                }
            })
            .collect();

        let request = CreateMessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: if system.is_empty() {
                None
            } else {
                Some(system.to_string())
            },
            messages: message_params,
            tools: tools.map(|t| t.to_vec()),
        };

        debug!(model = %self.model, "→ Anthropic /v1/messages");

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", CLAUDE_API_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| Error::HTTP(format!("anthropic request: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::HTTP(format!("read response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::ClaudeAPI(format!("HTTP {}: {}", status, body)));
        }

        let parsed: CreateMessageResponse = serde_json::from_str(&body).map_err(|e| {
            warn!("anthropic parse failed: {}", body);
            Error::Serialization(e)
        })?;

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for block in &parsed.content {
            match block {
                ContentBlock::Text { text: t } => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCallInfo {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                }
            }
        }

        Ok(LLMResponse {
            text,
            tool_calls,
            stop_reason: StopReason::from_str(&parsed.stop_reason),
            model: parsed.model,
            usage: Usage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            },
        })
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        "anthropic"
    }
}

// Helper used by tool execution loops to format a tool result for the next turn.
pub(crate) fn tool_result_block(tool_use_id: &str, content: &str, is_error: bool) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_creation() {
        let p = AnthropicProvider::new("test".into(), "claude-opus-4-7".into());
        assert_eq!(p.model(), "claude-opus-4-7");
        assert_eq!(p.provider_name(), "anthropic");
    }
}
