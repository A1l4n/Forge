use serde::{Deserialize, Serialize};
use std::fs;
use crate::Result;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub claude_api_key: String,
    pub model: String,
    pub database_url: String,
    pub log_level: String,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| crate::errors::ForgeError::ConfigError(e.to_string()))?;
        toml::from_str(&content)
            .map_err(|e| crate::errors::ForgeError::ConfigError(e.to_string()))
    }

    pub fn from_env() -> Result<Self> {
        Ok(Config {
            claude_api_key: std::env::var("CLAUDE_API_KEY")
                .map_err(|_| crate::errors::ForgeError::ConfigError("CLAUDE_API_KEY not set".to_string()))?,
            model: std::env::var("CLAUDE_MODEL")
                .unwrap_or_else(|_| "claude-opus-4-6".to_string()),
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:forge.db".to_string()),
            log_level: std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string()),
        })
    }
}
