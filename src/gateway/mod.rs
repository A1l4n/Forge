//! Gateway module — message routing, HTTP server, and CLI shell.

pub mod cli;
pub mod http;
pub mod router;

pub use cli::CLIShell;
pub use http::{start_server, AppState};
pub use router::MessageRouter;
