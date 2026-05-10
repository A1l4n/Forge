//! Provider for OpenAI-compatible chat completion endpoints.
//!
//! Works with: Ollama (`http://localhost:11434/v1`), OpenRouter, Groq,
//! LM Studio, vLLM, Together AI, Mistral, Cerebras, GLM, and OpenAI itself.
//!
//! Two generation paths:
//! - [`OpenAICompatibleProvider::generate`] — single-shot text in/out.
//! - [`OpenAICompatibleProvider::agentic_step`] — multi-turn with proper
//!   `tool_calls` field on assistant messages and `role: "tool"` results.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{debug, warn};

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole};

use super::{AgenticTurn, LLMProvider, LLMResponse, StopReason, ToolCallInfo, Usage};

#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    stream: bool,
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
        Self::new("ollama", "http://localhost:11434/v1", None, model).with_max_tokens(None)
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

    /// Zhipu AI's GLM models. `glm-4-flash` is free and surprisingly capable.
    pub fn glm(api_key: String, model: impl Into<String>) -> Self {
        Self::new(
            "glm",
            "https://open.bigmodel.cn/api/paas/v4",
            Some(api_key),
            model,
        )
    }

    /// Mistral AI (la Plateforme — has a free tier).
    pub fn mistral(api_key: String, model: impl Into<String>) -> Self {
        Self::new("mistral", "https://api.mistral.ai/v1", Some(api_key), model)
    }

    /// Cerebras — fast free tier hosting Llama models.
    pub fn cerebras(api_key: String, model: impl Into<String>) -> Self {
        Self::new("cerebras", "https://api.cerebras.ai/v1", Some(api_key), model)
    }

    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Translate Anthropic-style tool defs to OpenAI tool schema.
    fn translate_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
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
            .collect()
    }

    async fn send_request(&self, request: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);
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

        serde_json::from_str(&body).map_err(|e| {
            warn!("{} parse failed: {}", self.provider_name, body);
            Error::Serialization(e)
        })
    }

    fn parse_response(&self, parsed: ChatResponse) -> Result<LLMResponse> {
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

        let model = if parsed.model.is_empty() {
            self.model.clone()
        } else {
            parsed.model
        };

        Ok(LLMResponse {
            text,
            tool_calls,
            stop_reason,
            model,
            usage,
        })
    }
}

/// Convert plain-text Messages into OpenAI chat messages.
fn messages_to_value(system: &str, messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    for m in messages {
        let role = match m.role {
            MessageRole::User => "user",
            MessageRole::Luna | MessageRole::Agent => "assistant",
            MessageRole::System => "system",
        };
        out.push(json!({"role": role, "content": m.content}));
    }
    out
}

/// Convert structured AgenticTurns into OpenAI chat messages with proper
/// tool_calls field and role:"tool" entries for results.
fn turns_to_value(system: &str, turns: &[AgenticTurn]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    for turn in turns {
        match turn {
            AgenticTurn::User(text) => {
                out.push(json!({"role": "user", "content": text}));
            }
            AgenticTurn::Assistant { text, tool_calls } => {
                let mut msg = serde_json::Map::new();
                msg.insert("role".into(), json!("assistant"));
                if text.is_empty() {
                    msg.insert("content".into(), Value::Null);
                } else {
                    msg.insert("content".into(), json!(text));
                }
                if !tool_calls.is_empty() {
                    let calls: Vec<Value> = tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": c.input.to_string(),
                                }
                            })
                        })
                        .collect();
                    msg.insert("tool_calls".into(), Value::Array(calls));
                }
                out.push(Value::Object(msg));
            }
            AgenticTurn::ToolResults(results) => {
                for r in results {
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": r.tool_use_id,
                        "content": r.content,
                    }));
                }
            }
        }
    }
    out
}

#[async_trait]
impl LLMProvider for OpenAICompatibleProvider {
    async fn generate(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Result<LLMResponse> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages_to_value(system, messages),
            max_tokens: self.max_tokens,
            temperature: None,
            tools: tools.map(Self::translate_tools),
            stream: false,
        };
        let parsed = self.send_request(request).await?;
        self.parse_response(parsed)
    }

    async fn agentic_step(
        &self,
        system: &str,
        turns: &[AgenticTurn],
        tools: &[Value],
    ) -> Result<LLMResponse> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: turns_to_value(system, turns),
            max_tokens: self.max_tokens,
            temperature: None,
            tools: if tools.is_empty() {
                None
            } else {
                Some(Self::translate_tools(tools))
            },
            stream: false,
        };
        let parsed = self.send_request(request).await?;
        self.parse_response(parsed)
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

    #[test]
    fn turns_emit_tool_role_messages() {
        use super::super::ToolResultEntry;
        let turns = vec![
            AgenticTurn::User("read main.rs".into()),
            AgenticTurn::Assistant {
                text: "".into(),
                tool_calls: vec![ToolCallInfo {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "main.rs"}),
                }],
            },
            AgenticTurn::ToolResults(vec![ToolResultEntry {
                tool_use_id: "call_1".into(),
                content: "fn main(){}".into(),
                is_error: false,
            }]),
        ];
        let v = turns_to_value("you are luna", &turns);
        // system + user + assistant + tool = 4
        assert_eq!(v.len(), 4);
        assert_eq!(v[0]["role"], "system");
        assert_eq!(v[3]["role"], "tool");
        assert_eq!(v[3]["tool_call_id"], "call_1");
    }
}
