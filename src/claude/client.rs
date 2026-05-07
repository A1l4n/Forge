//! Claude API Client with streaming support

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole};
use reqwest::{Client as HttpClient, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{debug, warn};

const CLAUDE_API_VERSION: &str = "2024-06-01";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageParam {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub system: Option<String>,
    pub messages: Vec<MessageParam>,
    pub tools: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateMessageResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Claude API Client
pub struct ClaudeClient {
    api_key: String,
    base_url: String,
    model: String,
    http_client: HttpClient,
    timeout: Duration,
}

impl ClaudeClient {
    /// Create a new Claude client
    pub fn new(api_key: String, model: String) -> Self {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            api_key,
            base_url: "https://api.anthropic.com".to_string(),
            model,
            http_client,
            timeout: Duration::from_secs(30),
        }
    }

    /// Create a message (non-streaming)
    pub async fn create_message(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse> {
        let url = format!("{}/v1/messages", self.base_url);

        debug!(
            "Sending Claude API request to {} with model {}",
            url, self.model
        );

        let response = self
            .http_client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", CLAUDE_API_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| Error::HTTP(format!("Failed to send request: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::HTTP(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            return Err(Error::ClaudeAPI(format!(
                "API error {}: {}",
                status, body
            )));
        }

        serde_json::from_str(&body).map_err(|e| {
            warn!("Failed to parse response: {}", body);
            Error::Serialization(e)
        })
    }

    /// Create a message from conversation history
    pub async fn message_from_history(
        &self,
        system_prompt: String,
        messages: Vec<Message>,
        tools: Option<Vec<Value>>,
    ) -> Result<CreateMessageResponse> {
        let message_params: Vec<MessageParam> = messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::User => "user",
                    MessageRole::Luna | MessageRole::Agent => "assistant",
                    MessageRole::System => "user",
                };
                MessageParam {
                    role: role.to_string(),
                    content: msg.content,
                }
            })
            .collect();

        let request = CreateMessageRequest {
            model: self.model.clone(),
            max_tokens: 4096,
            system: Some(system_prompt),
            messages: message_params,
            tools,
        };

        self.create_message(request).await
    }

    /// Extract text from response
    pub fn extract_text(response: &CreateMessageResponse) -> String {
        response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract tool calls from response
    pub fn extract_tool_calls(response: &CreateMessageResponse) -> Vec<(String, String, Value)> {
        response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, name, input } => {
                    Some((id.clone(), name.clone(), input.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Check if response contains tool calls
    pub fn has_tool_calls(response: &CreateMessageResponse) -> bool {
        response
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
    }

    /// Get model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Set model
    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = ClaudeClient::new(
            "test-key".to_string(),
            "claude-opus-4-7".to_string(),
        );
        assert_eq!(client.model(), "claude-opus-4-7");
    }

    #[test]
    fn test_extract_text() {
        let response = CreateMessageResponse {
            id: "1".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text {
                    text: "Hello".to_string(),
                },
                ContentBlock::Text {
                    text: "World".to_string(),
                },
            ],
            model: "claude-opus-4-7".to_string(),
            stop_reason: "end_turn".to_string(),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let text = ClaudeClient::extract_text(&response);
        assert_eq!(text, "Hello\nWorld");
    }
}
