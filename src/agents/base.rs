//! Base agent trait

use async_trait::async_trait;
use crate::{models::{ExecutionContext, Task}, Result};

/// Trait for specialist agents
#[async_trait]
pub trait Agent: Send + Sync {
    /// Get agent name
    fn name(&self) -> &str;

    /// Get agent role
    fn role(&self) -> &str;

    /// Get system prompt for this agent
    fn system_prompt(&self) -> String;

    /// Execute a task
    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String>;

    /// Check if agent can handle a task
    fn can_handle(&self, task_description: &str) -> bool {
        // Default: all agents can attempt any task
        true
    }
}
