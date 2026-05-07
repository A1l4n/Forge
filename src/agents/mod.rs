//! Specialist agents module.
//!
//! Houses the `Agent` trait and the four built-in specialists that Luna delegates to:
//! [`CodeAgent`], [`ResearchAgent`], [`WritingAgent`], and [`PlanningAgent`].

pub mod base;
pub mod code_agent;
pub mod research_agent;
pub mod writing_agent;
pub mod planning_agent;

pub use base::Agent;
pub use code_agent::CodeAgent;
pub use research_agent::{Finding, ResearchAgent};
pub use writing_agent::{WritingAgent, WritingStyle};
pub use planning_agent::{PlanStep, PlanningAgent};

use crate::claude::ClaudeClient;
use std::sync::Arc;

/// Build the default set of specialist agents wired to a shared Claude client.
pub fn default_agents(claude: Arc<ClaudeClient>) -> Vec<Arc<dyn Agent>> {
    vec![
        Arc::new(CodeAgent::new(claude.clone())),
        Arc::new(ResearchAgent::new(claude.clone())),
        Arc::new(WritingAgent::new(claude.clone())),
        Arc::new(PlanningAgent::new(claude)),
    ]
}
