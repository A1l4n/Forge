//! Tool model for Claude tool calling

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Definition of a tool available to agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

impl Tool {
    pub fn new(name: String, description: String, input_schema: Value) -> Self {
        Self {
            name,
            description,
            input_schema,
        }
    }

    /// Create a simple tool with basic properties
    pub fn simple(name: String, description: String) -> Self {
        Self {
            name,
            description,
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    /// Convert to Claude API tool format
    pub fn to_claude_format(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        })
    }
}

/// A tool call made by Claude
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: ToolInput,
}

/// Input parameters for tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    pub params: Value,
}

impl ToolInput {
    pub fn from_value(value: Value) -> Self {
        Self { params: value }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.params.get(key)
    }

    pub fn get_string(&self, key: &str) -> Option<String> {
        self.params.get(key).and_then(|v| v.as_str().map(String::from))
    }

    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.params.get(key).and_then(|v| v.as_bool())
    }

    pub fn get_array(&self, key: &str) -> Option<&Vec<Value>> {
        self.params.get(key).and_then(|v| v.as_array())
    }

    pub fn get_object(&self, key: &str) -> Option<&serde_json::Map<String, Value>> {
        self.params.get(key).and_then(|v| v.as_object())
    }
}

/// Result of tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub success: bool,
    pub output: Value,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn success(tool_call_id: String, tool_name: String, output: Value) -> Self {
        Self {
            tool_call_id,
            tool_name,
            success: true,
            output,
            error: None,
        }
    }

    pub fn error(tool_call_id: String, tool_name: String, error: String) -> Self {
        Self {
            tool_call_id,
            tool_name,
            success: false,
            output: Value::Null,
            error: Some(error),
        }
    }

    pub fn to_claude_format(&self) -> Value {
        json!({
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": if self.success {
                self.output.to_string()
            } else {
                format!("Error: {}", self.error.as_ref().unwrap_or(&"Unknown error".to_string()))
            }
        })
    }
}
