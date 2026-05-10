//! Forge — Agentic Operating System powered by Claude (and friends).
//!
//! Multi-backend CLI. Pick a provider with `--backend`:
//!
//!   --backend anthropic     (default, needs CLAUDE_API_KEY — sk-ant-api03-...)
//!   --backend claude-oauth  (uses Claude Code's OAuth token from ~/.claude/.credentials.json)
//!   --backend ollama        (free, local, talks to http://localhost:11434)
//!   --backend openrouter    (free + paid, needs OPENROUTER_API_KEY)
//!   --backend groq          (free, fast, needs GROQ_API_KEY)
//!   --backend glm           (FREE glm-4-flash, needs GLM_API_KEY from bigmodel.cn)
//!   --backend mistral       (free tier, needs MISTRAL_API_KEY)
//!   --backend cerebras      (free tier, needs CEREBRAS_API_KEY)
//!   --backend together      (paid + $1 free, needs TOGETHER_API_KEY)
//!   --backend lm-studio     (free, local LM Studio at localhost:1234)
//!   --backend openai        (paid, needs OPENAI_API_KEY)
//!   --backend custom        (any OpenAI-compatible endpoint via --base-url)
//!
//! Three run modes:
//!   forge "msg"              one-shot
//!   forge chat               interactive REPL
//!   forge serve --port 8080  HTTP gateway + embedded web UI
//!
//! Luna is fully **agentic** — at startup she loads the full tool registry
//! (file system, shell, self-modification, persistent memory, team management),
//! plus any dynamic agents previously recruited via `spawn_agent`.

use clap::{Parser, Subcommand, ValueEnum};
use std::sync::Arc;
use uuid::Uuid;

use forge::{
    agents::{full_team, Agent},
    gateway::{cli::CLIShell, http},
    llm::{AnthropicProvider, LLMProvider, OpenAICompatibleProvider},
    logging,
    memory::MemoryStore,
    models::UserRequest,
    tools::{PermissionMode, ToolRegistry},
    Orchestrator,
};

#[derive(Parser, Debug)]
#[command(name = "Forge")]
#[command(about = "Agentic OS powered by Claude (and other LLMs)", long_about = None)]
#[command(version)]
struct Args {
    /// Which LLM backend to use
    #[arg(long, value_enum, default_value_t = Backend::Anthropic)]
    backend: Backend,

    /// API key (auto-reads provider-specific env vars; see --help for the list)
    #[arg(long, env = "FORGE_API_KEY")]
    api_key: Option<String>,

    /// Model name. Defaults depend on backend (claude-opus-4-7 for anthropic, llama3.1 for ollama, ...)
    #[arg(short, long)]
    model: Option<String>,

    /// Base URL override (only meaningful for --backend custom)
    #[arg(long)]
    base_url: Option<String>,

    /// Path to SQLite database file
    #[arg(long, default_value = "./forge.db")]
    db: String,

    /// Hard cap on total tokens spent per request (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    token_budget: u64,

    /// Disable Luna's tool registry (file/shell/self-edit/memory/team).
    /// By default Luna has full local access.
    #[arg(long, default_value_t = false)]
    no_tools: bool,

    /// Strict mode — block all `Confirm`-tier tools (write_file, run_shell,
    /// self_edit_source, git_commit, spawn_agent, save_skill). Use this when
    /// Luna is exposed publicly or running unattended.
    #[arg(long, default_value_t = false)]
    strict: bool,

    /// Allow a specific tool regardless of mode (repeatable).
    /// Useful with --strict to whitelist only a few destructive tools.
    #[arg(long = "allow-tool")]
    allow_tools: Vec<String>,

    /// Cap on agentic-loop iterations per request (default 12).
    #[arg(long, default_value_t = 12)]
    max_iterations: usize,

    /// Auth token for the HTTP gateway. When set, every API request must
    /// include `Authorization: Bearer <token>` or `X-Forge-Token: <token>`.
    /// The web UI prompts for it on load and stores it locally.
    /// Strongly recommended when exposing Forge over the public internet.
    #[arg(long, env = "FORGE_AUTH_TOKEN")]
    auth_token: Option<String>,

    /// One-shot message — if provided, run once and exit
    #[arg(value_name = "MESSAGE")]
    message: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the HTTP API server + embedded web UI
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Start the interactive REPL
    Chat,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    Anthropic,
    #[value(name = "claude-oauth")]
    ClaudeOauth,
    Ollama,
    Openrouter,
    Groq,
    Glm,
    Mistral,
    Cerebras,
    #[value(name = "lm-studio")]
    LmStudio,
    Openai,
    Together,
    Custom,
}

impl Backend {
    fn default_model(self) -> &'static str {
        match self {
            Backend::Anthropic => "claude-opus-4-7",
            Backend::ClaudeOauth => "claude-opus-4-7",
            Backend::Ollama => "llama3.1",
            Backend::Openrouter => "meta-llama/llama-3.3-70b-instruct:free",
            Backend::Groq => "llama3-groq-70b-8192-tool-use-preview",
            Backend::Glm => "glm-4-flash",
            Backend::Mistral => "mistral-small-latest",
            Backend::Cerebras => "llama3.1-70b",
            Backend::LmStudio => "local-model",
            Backend::Openai => "gpt-4o-mini",
            Backend::Together => "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            Backend::Custom => "default",
        }
    }
}

fn pick_api_key(backend: Backend, explicit: Option<String>) -> Option<String> {
    if let Some(k) = explicit {
        return Some(k);
    }
    let candidates: &[&str] = match backend {
        Backend::Anthropic => &["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
        Backend::ClaudeOauth => &[],
        Backend::Openrouter => &["OPENROUTER_API_KEY"],
        Backend::Groq => &["GROQ_API_KEY"],
        Backend::Glm => &["GLM_API_KEY", "ZHIPU_API_KEY"],
        Backend::Mistral => &["MISTRAL_API_KEY"],
        Backend::Cerebras => &["CEREBRAS_API_KEY"],
        Backend::Openai => &["OPENAI_API_KEY"],
        Backend::Together => &["TOGETHER_API_KEY", "TOGETHERAI_API_KEY"],
        Backend::Custom => &["FORGE_API_KEY"],
        Backend::Ollama | Backend::LmStudio => &[],
    };
    for var in candidates {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}

fn build_provider(args: &Args) -> Result<Arc<dyn LLMProvider>, Box<dyn std::error::Error>> {
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| args.backend.default_model().to_string());
    let api_key = pick_api_key(args.backend, args.api_key.clone());

    let provider: Arc<dyn LLMProvider> = match args.backend {
        Backend::Anthropic => {
            let key = api_key
                .ok_or("anthropic backend requires CLAUDE_API_KEY (or --api-key)")?;
            Arc::new(AnthropicProvider::new(key, model))
        }
        Backend::ClaudeOauth => Arc::new(AnthropicProvider::from_claude_code_credentials(model)?),
        Backend::Ollama => Arc::new(OpenAICompatibleProvider::ollama(model)),
        Backend::LmStudio => Arc::new(OpenAICompatibleProvider::lm_studio(model)),
        Backend::Openrouter => {
            let key = api_key.ok_or("openrouter backend requires OPENROUTER_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::openrouter(key, model))
        }
        Backend::Groq => {
            let key = api_key.ok_or("groq backend requires GROQ_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::groq(key, model))
        }
        Backend::Glm => {
            let key = api_key
                .ok_or("glm backend requires GLM_API_KEY (get one at bigmodel.cn)")?;
            Arc::new(OpenAICompatibleProvider::glm(key, model))
        }
        Backend::Mistral => {
            let key = api_key.ok_or("mistral backend requires MISTRAL_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::mistral(key, model))
        }
        Backend::Cerebras => {
            let key = api_key.ok_or("cerebras backend requires CEREBRAS_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::cerebras(key, model))
        }
        Backend::Openai => {
            let key = api_key.ok_or("openai backend requires OPENAI_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::openai(key, model))
        }
        Backend::Together => {
            let key = api_key.ok_or("together backend requires TOGETHER_API_KEY")?;
            Arc::new(OpenAICompatibleProvider::together(key, model))
        }
        Backend::Custom => {
            let base_url = args
                .base_url
                .clone()
                .ok_or("--backend custom requires --base-url")?;
            Arc::new(OpenAICompatibleProvider::new(
                "custom", base_url, api_key, model,
            ))
        }
    };

    Ok(provider)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init_logging();
    let args = Args::parse();

    let llm = build_provider(&args)?;
    let model = llm.model().to_string();
    let provider_name = llm.provider_name().to_string();

    let memory = Arc::new(MemoryStore::open(args.db.clone()).await?);

    // Build the tool registry. Luna gets the full toolkit by default.
    let tools = if args.no_tools {
        None
    } else {
        let mode = if args.strict {
            PermissionMode::Strict
        } else {
            PermissionMode::Open
        };
        let mut reg = ToolRegistry::with_full_toolkit(memory.clone()).with_mode(mode);
        for name in &args.allow_tools {
            reg.allow_always(name.clone());
        }
        Some(Arc::new(reg))
    };

    let mut orchestrator = Orchestrator::new(llm.clone()).with_max_iterations(args.max_iterations);
    if args.token_budget > 0 {
        orchestrator = orchestrator.with_token_budget(args.token_budget);
    }
    if let Some(t) = tools.clone() {
        orchestrator = orchestrator.with_tools(t);
    }
    let orchestrator = Arc::new(orchestrator);

    // Build the team (built-in specialists + any persisted dynamic agents).
    let team = full_team(&memory, llm.clone()).await?;
    let agents: Arc<Vec<Arc<dyn Agent>>> = Arc::new(team);

    let tool_count = tools.as_ref().map(|t| t.all().len()).unwrap_or(0);
    let permission_label = if args.no_tools {
        "no-tools"
    } else if args.strict {
        "strict"
    } else {
        "open"
    };
    println!(
        "Luna ready · provider={} · model={} · {} agent(s) · {} tool(s) · permissions={} · db={}",
        provider_name,
        model,
        agents.len(),
        tool_count,
        permission_label,
        memory.db_path()
    );

    match args.command {
        Some(Command::Serve { host, port }) => {
            let auth_status = if args.auth_token.is_some() {
                "🔒 enabled"
            } else {
                "⚠ DISABLED — set --auth-token or FORGE_AUTH_TOKEN before exposing publicly"
            };
            let state = http::AppState {
                orchestrator: orchestrator.clone(),
                memory: memory.clone(),
                agents: agents.clone(),
                model: model.clone(),
                provider: provider_name.clone(),
                auth_token: args.auth_token.clone(),
            };
            println!(
                "🔥 Forge starting on http://{}:{} (provider={}, model={}, auth={})",
                host, port, provider_name, model, auth_status
            );
            http::start_server(&host, port, state).await?;
        }
        Some(Command::Chat) => {
            let mut shell = CLIShell::new(orchestrator, memory, agents, provider_name, model);
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
                    println!("Tool / agent activity:");
                    for a in &result.activities {
                        println!("  - {} → {} ({}ms)", a.agent_name, a.status, a.duration_ms);
                    }
                }
                println!("\nLuna: {}\n", result.response);
                println!(
                    "(usage: {} in / {} out · {} tool calls)",
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.tool_invocations.len()
                );
            } else {
                let mut shell = CLIShell::new(orchestrator, memory, agents, provider_name, model);
                shell.run().await?;
            }
        }
    }

    Ok(())
}
