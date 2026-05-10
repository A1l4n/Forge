//! Dynamic agents — specialists Luna recruits at runtime.
//!
//! Unlike the four built-in agents (Code/Research/Writing/Planning), dynamic
//! agents are stored in the SQLite [`MemoryStore`] and loaded on each startup.
//! They are created via the `spawn_agent` tool and can be renamed (`rename_agent`)
//! or removed (`delete_dynamic_agent`).
//!
//! Each dynamic agent is just a name + role description + system prompt; it
//! shares the same LLM provider as the rest of the team.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::info;

use crate::agents::Agent;
use crate::llm::LLMProvider;
use crate::memory::DynamicAgentRow;
use crate::models::{ExecutionContext, Message, Task};
use crate::Result;

/// A specialist agent created at runtime by Luna.
pub struct DynamicAgent {
    name: String,
    role: String,
    system_prompt: String,
    llm: Arc<dyn LLMProvider>,
}

impl DynamicAgent {
    pub fn new(
        name: String,
        role: String,
        system_prompt: String,
        llm: Arc<dyn LLMProvider>,
    ) -> Self {
        Self {
            name,
            role,
            system_prompt,
            llm,
        }
    }

    /// Build a DynamicAgent from a row loaded out of the memory store.
    pub fn from_row(row: DynamicAgentRow, llm: Arc<dyn LLMProvider>) -> Self {
        Self::new(row.name, row.role, row.system_prompt, llm)
    }
}

#[async_trait]
impl Agent for DynamicAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn role(&self) -> &str {
        &self.role
    }

    fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        info!(agent = %self.name, task_id = %task.id, "Dynamic agent executing");
        let messages = vec![Message::user(
            context.session_id.clone(),
            task.description.clone(),
        )];
        let response = self
            .llm
            .generate(&self.system_prompt, &messages, None)
            .await?;
        Ok(response.text)
    }
}
