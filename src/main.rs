//! Forge — Agentic Operating System powered by Claude.
//!
//! CLI entry point. Three modes:
//! - default (no subcommand): one-shot if --message is given, otherwise interactive REPL.
//! - `serve`: start the HTTP gateway.
//! - `chat`:  force the interactive REPL.

use clap::{Parser, Subcommand};
use std::sync::Arc;
use uuid::Uuid;

use forge::{
    agents::{default_agents, Agent},
    claude::ClaudeClient,
    gateway::{cli::CLIShell, http},
    logging,
    memory::MemoryStore,
    models::UserRequest,
    Orchestrator,
};

#[derive(Parser, Debug)]
#[command(name = "Forge")]
#[command(about = "Agentic OS powered by Claude", long_about = None)]
#[command(version)]
struct Args {
    /// API key for Claude (also reads CLAUDE_API_KEY / ANTHROPIC_API_KEY)
    #[arg(short, long, env = "CLAUDE_API_KEY")]
    api_key: Option<String>,

    /// Model to use
    #[arg(short, long, default_value = "claude-opus-4-7")]
    model: String,

    /// Path to SQLite database file
    #[arg(long, default_value = "./forge.db")]
    db: String,

    /// One-shot message — if provided, run once and exit
    #[arg(value_name = "MESSAGE")]
    message: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the HTTP API server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Start the interactive REPL
    Chat,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init_logging();
    let args = Args::parse();

    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("CLAUDE_API_KEY").ok())
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .ok_or("CLAUDE_API_KEY environment variable not set")?;

    let claude = Arc::new(ClaudeClient::new(api_key, args.model.clone()));
    let orchestrator = Arc::new(Orchestrator::new(claude.clone()));
    let memory = Arc::new(MemoryStore::open(args.db.clone()).await?);
    let agents: Arc<Vec<Arc<dyn Agent>>> = Arc::new(default_agents(claude.clone()));

    match args.command {
        Some(Command::Serve { host, port }) => {
            let state = http::AppState {
                orchestrator: orchestrator.clone(),
                memory: memory.clone(),
                agents: agents.clone(),
                model: args.model.clone(),
            };
            http::start_server(&host, port, state).await?;
        }
        Some(Command::Chat) => {
            let mut shell = CLIShell::new(orchestrator, memory, agents, args.model);
            shell.run().await?;
        }
        None => {
            if let Some(message) = args.message {
                let request = UserRequest {
                    content: message,
                    session_id: Uuid::new_v4().to_string(),
                    context: None,
                };

                let result = orchestrator
                    .process_with_agents(request, &agents, &memory)
                    .await?;

                if !result.activities.is_empty() {
                    println!("Agent activity:");
                    for a in &result.activities {
                        println!("  - {} → {} ({}ms)", a.agent_name, a.status, a.duration_ms);
                    }
                }
                println!("\nLuna: {}\n", result.response);
            } else {
                let mut shell = CLIShell::new(orchestrator, memory, agents, args.model);
                shell.run().await?;
            }
        }
    }

    Ok(())
}
