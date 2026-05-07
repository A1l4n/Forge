//! Message model for conversations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageRole {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "luna")]
    Luna,
    #[serde(rename = "agent")]
    Agent,
    #[serde(rename = "system")]
    System,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Luna => write!(f, "Luna"),
            MessageRole::Agent => write!(f, "Agent"),
            MessageRole::System => write!(f, "System"),
        }
    }
}

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub session_id: String,
    pub role: MessageRole,
    pub content: String,
    pub agent_name: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub tool_results: Option<Vec<serde_json::Value>>,
    pub created_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl Message {
    /// Create a new message from user
    pub fn user(session_id: String, content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            role: MessageRole::User,
            content,
            agent_name: None,
            tool_calls: None,
            tool_results: None,
            created_at: Utc::now(),
            metadata: None,
        }
    }

    /// Create a new message from Luna
    pub fn luna(session_id: String, content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            role: MessageRole::Luna,
            content,
            agent_name: None,
            tool_calls: None,
            tool_results: None,
            created_at: Utc::now(),
            metadata: None,
        }
    }

    /// Create a new message from an agent
    pub fn agent(session_id: String, agent_name: String, content: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            role: MessageRole::Agent,
            content,
            agent_name: Some(agent_name),
            tool_calls: None,
            tool_results: None,
            created_at: Utc::now(),
            metadata: None,
        }
    }

    /// Add tool calls to message
    pub fn with_tool_calls(mut self, tool_calls: Vec<serde_json::Value>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    /// Add tool results to message
    pub fn with_tool_results(mut self, tool_results: Vec<serde_json::Value>) -> Self {
        self.tool_results = Some(tool_results);
        self
    }
}
