//! Configuration management

use crate::errors::{Error, Result};
use crate::models::AgentConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeConfig {
    pub server: ServerConfig,
    pub claude: ClaudeConfig,
    pub database: DatabaseConfig,
    pub agents: Vec<AgentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub api_key: Option<String>,
    pub api_url: String,
    pub default_model: String,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub path: String,
    pub max_connections: u32,
}

impl Default for ForgeConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
                env: "development".to_string(),
            },
            claude: ClaudeConfig {
                api_key: None,
                api_url: "https://api.anthropic.com".to_string(),
                default_model: "claude-opus-4-7".to_string(),
                timeout_secs: 30,
            },
            database: DatabaseConfig {
                path: "./forge.db".to_string(),
                max_connections: 10,
            },
            agents: vec![],
        }
    }
}

impl ForgeConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Configuration(format!("Failed to read config: {}", e)))?;
        toml::from_str(&content)
            .map_err(|e| Error::Configuration(format!("Failed to parse config: {}", e)))
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Configuration(format!("Failed to serialize config: {}", e)))?;
        std::fs::write(path, content)
            .map_err(|e| Error::Configuration(format!("Failed to write config: {}", e)))
    }

    pub fn merge(&mut self, other: ForgeConfig) {
        if !other.server.host.is_empty() && other.server.host != "127.0.0.1" {
            self.server.host = other.server.host;
        }
        if other.server.port != 0 && other.server.port != 8080 {
            self.server.port = other.server.port;
        }
        if let Some(key) = other.claude.api_key {
            self.claude.api_key = Some(key);
        }
        if !other.agents.is_empty() {
            self.agents = other.agents;
        }
    }

    pub fn get_agent(&self, name: &str) -> Option<&AgentConfig> {
        self.agents.iter().find(|a| a.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = ForgeConfig::default();
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.claude.default_model, "claude-opus-4-7");
    }
}
