use async_trait::async_trait;
use crate::Result;
use crate::models::{Task, ExecutionContext};

#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn role(&self) -> &str;
    fn system_prompt(&self) -> String;
    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String>;
}

pub struct CodeAgent;

#[async_trait]
impl Agent for CodeAgent {
    fn name(&self) -> &str {
        "CodeAgent"
    }

    fn role(&self) -> &str {
        "Code generation and analysis specialist"
    }

    fn system_prompt(&self) -> String {
        "You are a code generation expert. Write clean, efficient, and well-documented code.".to_string()
    }

    async fn execute(&self, task: Task, _context: &ExecutionContext) -> Result<String> {
        Ok(format!("CodeAgent processing: {}", task.description))
    }
}

pub struct ResearchAgent;

#[async_trait]
impl Agent for ResearchAgent {
    fn name(&self) -> &str {
        "ResearchAgent"
    }

    fn role(&self) -> &str {
        "Research and analysis specialist"
    }

    fn system_prompt(&self) -> String {
        "You are a research expert. Provide thorough, well-sourced analysis.".to_string()
    }

    async fn execute(&self, task: Task, _context: &ExecutionContext) -> Result<String> {
        Ok(format!("ResearchAgent processing: {}", task.description))
    }
}
