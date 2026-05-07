//! LLM provider abstraction.
//!
//! Backends:
//! - [`AnthropicProvider`] — Claude via the Anthropic API (`x-api-key`).
//! - [`OpenAICompatibleProvider`] — any OpenAI-compatible endpoint (Ollama,
//!   OpenRouter, Groq, LM Studio, vLLM, Together, etc.).

pub mod anthropic;
pub mod openai_compatible;

pub use anthropic::AnthropicProvider;
pub use openai_compatible::OpenAICompatibleProvider;

use async_trait::async_trait;
use serde_json::Value;

use crate::models::Message;
use crate::Result;

/// Reason a generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Other(String),
}

impl StopReason {
    pub fn from_str(s: &str) -> Self {
        match s {
            "end_turn" | "stop" => StopReason::EndTurn,
            "tool_use" | "tool_calls" => StopReason::ToolUse,
            "max_tokens" | "length" => StopReason::MaxTokens,
            other => StopReason::Other(other.to_string()),
        }
    }
}

/// A tool call requested by the model.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// Token-usage info, when the provider returns it.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Normalized response shape across providers.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCallInfo>,
    pub stop_reason: StopReason,
    pub model: String,
    pub usage: Usage,
}

impl LLMResponse {
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Trait every LLM backend implements.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Generate a single response given a system prompt, conversation history,
    /// and an optional tool list (in Claude's tool-format JSON).
    async fn generate(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Result<LLMResponse>;

    fn model(&self) -> &str;

    /// Human-readable provider name (for status endpoints, logs).
    fn provider_name(&self) -> &str;
}
