//! Error types for Forge

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Claude API error: {0}")]
    ClaudeAPI(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Invalid configuration: {0}")]
    Configuration(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Agent not found: {0}")]
    AgentNotFound(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution failed: {0}")]
    ToolExecution(String),

    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Orchestration error: {0}")]
    Orchestration(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    HTTP(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
