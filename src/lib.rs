//! Forge - Agentic Operating System powered by Claude
//!
//! This library provides the core orchestration engine for coordinating
//! multiple specialist AI agents powered by Claude API.

// Module declarations
pub mod claude;
pub mod config;
pub mod errors;
pub mod gateway;
pub mod luna;
pub mod memory;
pub mod models;
pub mod tools;
pub mod agents;
pub mod utils;
pub mod logging;

// Re-exports for convenience
pub use errors::{Error, Result};
pub use models::{Message, Task, TaskStatus};
pub use luna::orchestrator::Orchestrator;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const APP_NAME: &str = "Forge";
