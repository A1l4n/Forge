use reqwest::Client;
use serde_json::json;
use crate::Result;
use crate::errors::ForgeError;

pub struct ClaudeClient {
    api_key: String,
    model: String,
    client: Client,
}

impl ClaudeClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: Client::new(),
        }
    }

    pub async fn chat(&self, messages: Vec<serde_json::Value>) -> Result<String> {
        let body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| ForgeError::ApiError(e.to_string()))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| ForgeError::ApiError(e.to_string()))?;

        response
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ForgeError::ApiError("Invalid response format".to_string()))
    }

    pub fn extract_text(response: &serde_json::Value) -> Result<String> {
        response
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ForgeError::ApiError("Invalid response format".to_string()))
    }
}
