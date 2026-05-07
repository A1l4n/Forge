//! Agent configuration model

use serde::{Deserialize, Serialize};

/// Configuration for a specialist agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub role: String,
    pub description: String,
    pub system_prompt: String,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Vec<String>,
    pub enabled: bool,
}

impl AgentConfig {
    pub fn new(name: String, role: String, description: String, system_prompt: String) -> Self {
        Self {
            name,
            role,
            description,
            system_prompt,
            model: None,
            temperature: None,
            max_tokens: None,
            tools: vec![],
            enabled: true,
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }
}
