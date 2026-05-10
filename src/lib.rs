//! Forge - Agentic Operating System powered by Claude (and friends).
//!
//! Multi-backend: works against the Anthropic API, Ollama (local), or any
//! OpenAI-compatible endpoint (OpenRouter, Groq, LM Studio, vLLM, ...).

// Module declarations
pub mod agents;
pub mod claude;     // legacy Claude client kept for backward compat
pub mod config;
pub mod errors;
pub mod gateway;
pub mod llm;        // new multi-provider abstraction (preferred)
pub mod logging;
pub mod luna;
pub mod memory;
pub mod models;
pub mod tools;
pub mod utils;
pub mod webui;

// Re-exports for convenience
pub use errors::{Error, Result};
pub use llm::{LLMProvider, LLMResponse};
pub use luna::orchestrator::{Orchestrator, OrchestrationResult, UsageTotals};
pub use models::{Message, Task, TaskStatus};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const APP_NAME: &str = "Forge";
