//! Interactive REPL for Forge.
//!
//! Slash commands:
//! - `/history`  — show messages stored for the current session.
//! - `/agents`   — list available specialists.
//! - `/status`   — show provider, model, and usage info.
//! - `/usage`    — show running token totals.
//! - `/clear`    — start a fresh session.
//! - `/exit`     — quit.

use std::io::{self, Write};
use std::sync::Arc;
use tracing::warn;
use uuid::Uuid;

use crate::agents::Agent;
use crate::memory::MemoryStore;
use crate::models::UserRequest;
use crate::utils::format_duration;
use crate::{Orchestrator, Result};

pub struct CLIShell {
    orchestrator: Arc<Orchestrator>,
    memory: Arc<MemoryStore>,
    agents: Arc<Vec<Arc<dyn Agent>>>,
    session_id: String,
    provider: String,
    model: String,
}

impl CLIShell {
    pub fn new(
        orchestrator: Arc<Orchestrator>,
        memory: Arc<MemoryStore>,
        agents: Arc<Vec<Arc<dyn Agent>>>,
        provider: String,
        model: String,
    ) -> Self {
        Self {
            orchestrator,
            memory,
            agents,
            session_id: Uuid::new_v4().to_string(),
            provider,
            model,
        }
    }

    /// Run the REPL until the user types `/exit` or sends EOF.
    pub async fn run(&mut self) -> Result<()> {
        self.banner();

        loop {
            print!("\n> ");
            io::stdout().flush().ok();

            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => {
                    println!("\nGoodbye.");
                    return Ok(());
                }
                Ok(_) => {}
                Err(e) => {
                    warn!(error = %e, "stdin error");
                    return Ok(());
                }
            }

            let input = line.trim();
            if input.is_empty() {
                continue;
            }

            if input.starts_with('/') {
                if self.handle_command(input).await? {
                    return Ok(());
                }
                continue;
            }

            self.dispatch(input).await;
        }
    }

    fn banner(&self) {
        println!("Welcome to Forge — agentic OS powered by LLMs");
        println!("Orchestrator: Luna");
        println!("Provider:     {}", self.provider);
        println!("Model:        {}", self.model);
        println!("Session:      {}", self.session_id);
        println!("\nType /help for commands. Type a message to talk to Luna.");
    }

    /// Returns Ok(true) if the shell should exit.
    async fn handle_command(&mut self, raw: &str) -> Result<bool> {
        let cmd = raw.trim();
        match cmd {
            "/exit" | "/quit" => {
                println!("Goodbye.");
                Ok(true)
            }
            "/help" => {
                println!("Commands:");
                println!("  /history   show session message history");
                println!("  /agents    list specialist agents");
                println!("  /status    show system info");
                println!("  /usage     show running token totals");
                println!("  /clear     start a new session");
                println!("  /exit      quit");
                Ok(false)
            }
            "/agents" => {
                println!("Specialist agents:");
                for a in self.agents.iter() {
                    println!("  - {} ({})", a.name(), a.role());
                }
                Ok(false)
            }
            "/status" => {
                let usage = self.orchestrator.current_usage().await;
                println!("Forge {}", crate::VERSION);
                println!("Provider: {}", self.provider);
                println!("Model:    {}", self.model);
                println!("Session:  {}", self.session_id);
                println!("DB:       {}", self.memory.db_path());
                println!("Tokens:   {} in / {} out", usage.input_tokens, usage.output_tokens);
                Ok(false)
            }
            "/usage" => {
                let usage = self.orchestrator.current_usage().await;
                println!("Tokens used so far: {} in / {} out (total {})",
                    usage.input_tokens, usage.output_tokens, usage.total());
                Ok(false)
            }
            "/history" => {
                let msgs = self.memory.get_session_messages(&self.session_id).await?;
                if msgs.is_empty() {
                    println!("(no messages yet)");
                } else {
                    for m in msgs {
                        println!(
                            "[{}] {}: {}",
                            m.created_at.format("%H:%M:%S"),
                            m.role,
                            m.content
                        );
                    }
                }
                Ok(false)
            }
            "/clear" => {
                self.session_id = Uuid::new_v4().to_string();
                println!("New session: {}", self.session_id);
                Ok(false)
            }
            other => {
                println!("Unknown command: {other}. Try /help.");
                Ok(false)
            }
        }
    }

    async fn dispatch(&self, input: &str) {
        let request = UserRequest {
            content: input.to_string(),
            session_id: self.session_id.clone(),
            context: None,
        };

        println!("\nProcessing with agents...");
        let started = std::time::Instant::now();

        match self
            .orchestrator
            .process_with_agents(request, &self.agents, &self.memory)
            .await
        {
            Ok(result) => {
                if !result.activities.is_empty() {
                    println!("\nAgent activity:");
                    for a in &result.activities {
                        println!(
                            "  - {} → {} ({})",
                            a.agent_name,
                            a.status,
                            format_duration(a.duration_ms)
                        );
                    }
                }
                println!(
                    "\nLuna ({}, {} in / {} out tokens):\n{}\n",
                    format_duration(started.elapsed().as_millis() as u64),
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.response
                );
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
}
