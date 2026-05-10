//! Core data models for Forge

pub mod message;
pub mod task;
pub mod agent;
pub mod tool;

pub use message::{Message, MessageRole};
pub use task::{Task, TaskStatus};
pub use agent::AgentConfig;
pub use tool::{Tool, ToolCall, ToolInput, ToolResult};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Session represents a conversation between user and Luna
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Session {
    pub fn new(user_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// Request from user to Luna
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRequest {
    pub content: String,
    pub session_id: String,
    pub context: Option<HashMap<String, String>>,
}

/// Response from Luna to user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LunaResponse {
    pub id: String,
    pub session_id: String,
    pub content: String,
    pub agent_activity: Vec<AgentActivity>,
    pub created_at: DateTime<Utc>,
}

/// Track agent activity during request processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentActivity {
    pub agent_name: String,
    pub task_id: String,
    pub status: String,
    pub result: Option<String>,
    pub duration_ms: u64,
}

/// Execution context passed between agents
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub session_id: String,
    pub user_id: String,
    pub task_id: String,
    pub memory: HashMap<String, serde_json::Value>,
    pub available_tools: Vec<String>,
}
