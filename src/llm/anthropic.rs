//! Anthropic API provider (Claude).
//!
//! Supports two auth modes (auto-detected from token prefix):
//! - **API key** (`sk-ant-api03-...`) → standard `x-api-key` header.
//! - **OAuth token** (`sk-ant-oat01-...`) → `Authorization: Bearer ...` plus
//!   `anthropic-beta: oauth-2025-04-20` and a Claude-Code identity prefix in
//!   the system prompt. Tokens come from Claude Code's credentials file.
//!
//! Two generation paths:
//! - [`AnthropicProvider::generate`] — single-shot, plain string content.
//! - [`AnthropicProvider::agentic_step`] — multi-turn with structured
//!   `tool_use` / `tool_result` blocks for proper agentic loops.
//!
//! ⚠️ OAuth-token usage outside Claude Code is a gray area: it routes your
//! Pro/Max subscription's compute through a custom app. Anthropic could
//! restrict or revoke this at any time. Use API keys for production work.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, warn};

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole};

use super::{AgenticTurn, LLMProvider, LLMResponse, StopReason, ToolCallInfo, Usage};

const CLAUDE_API_VERSION: &str = "2023-06-01";
const OAUTH_BETA_HEADER: &str = "oauth-2025-04-20";
const CLAUDE_CODE_IDENTITY: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicAuth {
    ApiKey,
    OAuth,
}

impl AnthropicAuth {
    /// Detect auth type from token prefix.
    pub fn detect(token: &str) -> Self {
        if token.starts_with("sk-ant-oat") {
            AnthropicAuth::OAuth
        } else {
            AnthropicAuth::ApiKey
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CreateMessageRequest {
    model: String,
    max_tokens: u32,
    system: Option<String>,
    messages: Vec<Value>,
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

/// Shape of `~/.claude/.credentials.json` (only the fields we care about).
#[derive(Debug, Deserialize)]
struct ClaudeCredentials {
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: Option<ClaudeOAuthEntry>,
}

#[derive(Debug, Deserialize)]
struct ClaudeOAuthEntry {
    #[serde(rename = "accessToken")]
    access_token: String,
}

/// Anthropic provider — talks directly to the Claude Messages API.
pub struct AnthropicProvider {
    token: String,
    base_url: String,
    model: String,
    http_client: HttpClient,
    max_tokens: u32,
    auth: AnthropicAuth,
}

impl AnthropicProvider {
    pub fn new(token: String, model: String) -> Self {
        let auth = AnthropicAuth::detect(&token);
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            token,
            base_url: "https://api.anthropic.com".to_string(),
            model,
            http_client,
            max_tokens: 8192,
            auth,
        }
    }

    /// Read the OAuth access token from Claude Code's credentials file.
    /// Looks at `%USERPROFILE%\.claude\.credentials.json` (Windows) or
    /// `$HOME/.claude/.credentials.json` (Unix).
    pub fn from_claude_code_credentials(model: String) -> Result<Self> {
        let path = credentials_path()?;
        if !path.exists() {
            return Err(Error::Authentication(format!(
                "Claude Code credentials not found at {}. Run `claude login` first.",
                path.display()
            )));
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Authentication(format!("read {}: {}", path.display(), e)))?;
        let creds: ClaudeCredentials = serde_json::from_str(&text)
            .map_err(|e| Error::Authentication(format!("parse credentials: {}", e)))?;
        let entry = creds.claude_ai_oauth.ok_or_else(|| {
            Error::Authentication(
                "credentials file missing claudeAiOauth section. \
                 Run `claude login` in a terminal to complete the OAuth flow. \
                 Alternatively use --backend groq (free, needs GROQ_API_KEY from console.groq.com) \
                 or --backend glm (free, needs GLM_API_KEY from bigmodel.cn)."
                    .into(),
            )
        })?;
        Ok(Self::new(entry.access_token, model))
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn auth(&self) -> AnthropicAuth {
        self.auth
    }

    /// Inject the Claude-Code identity into the system prompt when using OAuth.
    fn finalize_system(&self, system: &str) -> Option<String> {
        match self.auth {
            AnthropicAuth::OAuth => {
                if system.is_empty() {
                    Some(CLAUDE_CODE_IDENTITY.to_string())
                } else if system.starts_with(CLAUDE_CODE_IDENTITY) {
                    Some(system.to_string())
                } else {
                    Some(format!("{}\n\n{}", CLAUDE_CODE_IDENTITY, system))
                }
            }
            AnthropicAuth::ApiKey => {
                if system.is_empty() {
                    None
                } else {
                    Some(system.to_string())
                }
            }
        }
    }

    /// Apply auth headers to a request builder.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.auth {
            AnthropicAuth::ApiKey => req.header("x-api-key", &self.token),
            AnthropicAuth::OAuth => req
                .header("authorization", format!("Bearer {}", self.token))
                .header("anthropic-beta", OAUTH_BETA_HEADER),
        }
    }

    async fn send_request(&self, request: CreateMessageRequest) -> Result<CreateMessageResponse> {
        let url = format!("{}/v1/messages", self.base_url);
        debug!(model = %self.model, auth = ?self.auth, "→ Anthropic /v1/messages");

        let req = self
            .http_client
            .post(&url)
            .header("anthropic-version", CLAUDE_API_VERSION)
            .header("content-type", "application/json")
            .json(&request);
        let req = self.apply_auth(req);

        let response = req
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

        serde_json::from_str(&body).map_err(|e| {
            warn!("anthropic parse failed: {}", body);
            Error::Serialization(e)
        })
    }

    fn parse_response(parsed: CreateMessageResponse) -> LLMResponse {
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

        LLMResponse {
            text,
            tool_calls,
            stop_reason: StopReason::from_str(&parsed.stop_reason),
            model: parsed.model,
            usage: Usage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            },
        }
    }
}

fn credentials_path() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map_err(|_| Error::Authentication("USERPROFILE/HOME not set".into()))?;
    Ok(PathBuf::from(home).join(".claude").join(".credentials.json"))
}

/// Convert plain-text Messages to Anthropic message JSON.
fn messages_to_value(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                MessageRole::User | MessageRole::System => "user",
                MessageRole::Luna | MessageRole::Agent => "assistant",
            };
            json!({ "role": role, "content": msg.content })
        })
        .collect()
}

/// Convert structured AgenticTurns to Anthropic message JSON with proper
/// tool_use / tool_result content blocks.
fn turns_to_value(turns: &[AgenticTurn]) -> Vec<Value> {
    turns
        .iter()
        .map(|t| match t {
            AgenticTurn::User(text) => json!({
                "role": "user",
                "content": text
            }),
            AgenticTurn::Assistant { text, tool_calls } => {
                let mut content: Vec<Value> = Vec::new();
                if !text.is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                }
                for c in tool_calls {
                    content.push(json!({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.input,
                    }));
                }
                if content.is_empty() {
                    // Anthropic requires non-empty content
                    content.push(json!({"type": "text", "text": ""}));
                }
                json!({"role": "assistant", "content": content})
            }
            AgenticTurn::ToolResults(results) => {
                let blocks: Vec<Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "type": "tool_result",
                            "tool_use_id": r.tool_use_id,
                            "content": r.content,
                            "is_error": r.is_error,
                        })
                    })
                    .collect();
                json!({"role": "user", "content": blocks})
            }
        })
        .collect()
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn generate(
        &self,
        system: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
    ) -> Result<LLMResponse> {
        let request = CreateMessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: self.finalize_system(system),
            messages: messages_to_value(messages),
            tools: tools.map(|t| t.to_vec()),
        };
        let parsed = self.send_request(request).await?;
        Ok(Self::parse_response(parsed))
    }

    async fn agentic_step(
        &self,
        system: &str,
        turns: &[AgenticTurn],
        tools: &[Value],
    ) -> Result<LLMResponse> {
        let request = CreateMessageRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: self.finalize_system(system),
            messages: turns_to_value(turns),
            tools: if tools.is_empty() { None } else { Some(tools.to_vec()) },
        };
        let parsed = self.send_request(request).await?;
        Ok(Self::parse_response(parsed))
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        match self.auth {
            AnthropicAuth::ApiKey => "anthropic",
            AnthropicAuth::OAuth => "anthropic-oauth",
        }
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
    fn detect_oauth_token() {
        assert_eq!(
            AnthropicAuth::detect("sk-ant-oat01-abc"),
            AnthropicAuth::OAuth
        );
        assert_eq!(
            AnthropicAuth::detect("sk-ant-api03-xyz"),
            AnthropicAuth::ApiKey
        );
    }

    #[test]
    fn provider_picks_correct_auth() {
        let api = AnthropicProvider::new("sk-ant-api03-x".into(), "claude-opus-4-7".into());
        assert_eq!(api.auth(), AnthropicAuth::ApiKey);
        assert_eq!(api.provider_name(), "anthropic");

        let oauth = AnthropicProvider::new("sk-ant-oat01-y".into(), "claude-opus-4-7".into());
        assert_eq!(oauth.auth(), AnthropicAuth::OAuth);
        assert_eq!(oauth.provider_name(), "anthropic-oauth");
    }

    #[test]
    fn turns_to_value_handles_tool_use_round_trip() {
        let turns = vec![
            AgenticTurn::User("read main.rs".into()),
            AgenticTurn::Assistant {
                text: "I'll read it.".into(),
                tool_calls: vec![ToolCallInfo {
                    id: "toolu_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "main.rs"}),
                }],
            },
            AgenticTurn::ToolResults(vec![super::super::ToolResultEntry {
                tool_use_id: "toolu_1".into(),
                content: "fn main(){}".into(),
                is_error: false,
            }]),
        ];
        let v = turns_to_value(&turns);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[1]["role"], "assistant");
        assert!(v[1]["content"].is_array());
        assert_eq!(v[2]["role"], "user");
        assert_eq!(v[2]["content"][0]["type"], "tool_result");
    }
}
