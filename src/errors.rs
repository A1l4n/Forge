use thiserror::Error;

pub type Result<T> = std::result::Result<T, ForgeError>;

#[derive(Error, Debug)]
pub enum ForgeError {
    #[error("Claude API error: {0}")]
    ApiError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Agent error: {0}")]
    AgentError(String),

    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}
