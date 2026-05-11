//! Specialist agents module.
//!
//! Houses the `Agent` trait, the five built-in specialists Luna delegates to,
//! and a [`DynamicAgent`] type for runtime-recruited team members.
//!
//! Built-in roster:
//! - **CodeAgent** — software engineer, writes/reviews/debugs code.
//! - **ResearchAgent** — finds, analyzes, and synthesizes information.
//! - **WritingAgent** — copywriting, documentation, structured prose.
//! - **PlanningAgent** — project plans, task decomposition.
//! - **TradingAgent** (Nexus/Sigma) — market analysis, pre-trade checklist, trade execution.

pub mod base;
pub mod code_agent;
pub mod dynamic;
pub mod planning_agent;
pub mod research_agent;
pub mod trading_agent;
pub mod writing_agent;

pub use base::Agent;
pub use code_agent::CodeAgent;
pub use dynamic::DynamicAgent;
pub use planning_agent::{PlanStep, PlanningAgent};
pub use research_agent::{Finding, ResearchAgent};
pub use trading_agent::TradingAgent;
pub use writing_agent::{WritingAgent, WritingStyle};

use crate::llm::LLMProvider;
use crate::memory::MemoryStore;
use crate::Result;
use std::sync::Arc;

/// Build the default set of specialist agents wired to a shared LLM provider.
pub fn default_agents(llm: Arc<dyn LLMProvider>) -> Vec<Arc<dyn Agent>> {
    vec![
        Arc::new(CodeAgent::new(llm.clone())),
        Arc::new(ResearchAgent::new(llm.clone())),
        Arc::new(WritingAgent::new(llm.clone())),
        Arc::new(PlanningAgent::new(llm.clone())),
        // Trading specialists — always present when Binance keys are loaded.
        // Nexus = market scanner / analyst, Sigma = trade executor / tracker.
        Arc::new(TradingAgent::new(llm)),
    ]
}

/// Load all dynamic agents from the memory store and wrap them with the
/// shared LLM provider. Returns an empty vec if none are persisted.
pub async fn load_dynamic_agents(
    memory: &MemoryStore,
    llm: Arc<dyn LLMProvider>,
) -> Result<Vec<Arc<dyn Agent>>> {
    let rows = memory.list_dynamic_agents().await?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let agent: Arc<dyn Agent> = Arc::new(DynamicAgent::from_row(row, llm.clone()));
            agent
        })
        .collect())
}

/// Built-in + dynamic agents in one list.
pub async fn full_team(
    memory: &MemoryStore,
    llm: Arc<dyn LLMProvider>,
) -> Result<Vec<Arc<dyn Agent>>> {
    let mut team = default_agents(llm.clone());
    team.extend(load_dynamic_agents(memory, llm).await?);
    Ok(team)
}
