//! Tool registry and built-in executors.
//!
//! Every tool implements [`ToolExecutor`] and is registered with a JSON-schema
//! definition that gets advertised to the model. `execute()` dispatches a
//! [`ToolInput`] to the matching executor and returns a [`ToolResult`].
//!
//! Built-in tool taxonomy (set via `with_built_in_tools()` or `with_full_toolkit()`):
//!
//! **File system:**
//! - `read_file` — read a file as UTF-8 text
//! - `write_file` — write/overwrite a file (creates parent dirs)
//! - `list_directory` — list children of a directory
//! - `grep_files` — recursive regex search (basic)
//!
//! **Execution:**
//! - `run_shell` — run a PowerShell (Windows) or bash (Unix) command
//! - `execute_code` — run inline python/node/bash snippets
//!
//! **Network:**
//! - `web_search` — DuckDuckGo abstract API
//! - `http_request` — generic HTTP client
//!
//! **Self-modification:**
//! - `self_read_source` — read Forge's own source file
//! - `self_edit_source` — overwrite Forge's own source file (Luna can patch herself)
//! - `git_commit` — stage + commit current changes (so we can roll back)
//!
//! **Memory:**
//! - `save_memory` — persist a knowledge nugget across sessions
//! - `recall_memory` — search persisted memories by keyword
//!
//! **Team:**
//! - `spawn_agent` — recruit a new dynamic agent
//! - `rename_agent` — rename an existing dynamic agent
//! - `list_agents` — list dynamic agents currently on the team

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tracing::{debug, info, warn};


use crate::errors::{Error, Result};
use crate::memory::MemoryStore;
use crate::models::{Tool, ToolInput, ToolResult};

/// Permission tier for a tool.
///
/// - [`Tier::Auto`] — runs silently. Reads, search, list, memory ops, etc.
/// - [`Tier::Confirm`] — destructive / external-effects. Logged with a warning;
///   blocked entirely when [`PermissionMode::Strict`] is active.
/// - [`Tier::Deny`] — never executes regardless of mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Auto,
    Confirm,
    Deny,
}

/// Global enforcement mode.
///
/// - [`PermissionMode::Open`] — default. Auto + Confirm both execute (Confirm
///   warns). Use this for trusted local development.
/// - [`PermissionMode::Strict`] — only Auto-tier tools execute. Confirm and
///   Deny are blocked. Use when Luna is exposed publicly or running unattended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Open,
    Strict,
}

impl PermissionMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "strict" | "deny" | "lock" => PermissionMode::Strict,
            _ => PermissionMode::Open,
        }
    }
}

/// Async tool executor.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute(&self, input: &ToolInput) -> Result<Value>;
}

/// Convenience adapter so a regular async function can be used as a `ToolExecutor`.
pub struct FnExecutor<F>(pub F);

#[async_trait]
impl<F, Fut> ToolExecutor for FnExecutor<F>
where
    F: Fn(ToolInput) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Value>> + Send,
{
    async fn execute(&self, input: &ToolInput) -> Result<Value> {
        (self.0)(input.clone()).await
    }
}

/// Tool executor with a captured `MemoryStore` for persistence-backed tools.
struct MemoryToolExecutor<F> {
    memory: Arc<MemoryStore>,
    f: F,
}

#[async_trait]
impl<F, Fut> ToolExecutor for MemoryToolExecutor<F>
where
    F: Fn(Arc<MemoryStore>, ToolInput) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<Value>> + Send,
{
    async fn execute(&self, input: &ToolInput) -> Result<Value> {
        (self.f)(self.memory.clone(), input.clone()).await
    }
}

/// Registry of available tools and their executors.
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    executors: HashMap<String, Arc<dyn ToolExecutor>>,
    tiers: HashMap<String, Tier>,
    mode: PermissionMode,
    /// User-overridable extra allow-list (tool names that should always run
    /// regardless of mode).
    always_allow: HashSet<String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            executors: HashMap::new(),
            tiers: HashMap::new(),
            mode: PermissionMode::Open,
            always_allow: HashSet::new(),
        }
    }

    pub fn with_mode(mut self, mode: PermissionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
    }

    pub fn allow_always(&mut self, name: impl Into<String>) {
        self.always_allow.insert(name.into());
    }

    /// Register a tool definition only (no executor — useful for inspection).
    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register a tool with its executor at [`Tier::Auto`].
    pub fn register_with(&mut self, tool: Tool, executor: Arc<dyn ToolExecutor>) {
        self.register_tiered(tool, executor, Tier::Auto);
    }

    /// Register a tool with its executor and a specific permission tier.
    pub fn register_tiered(
        &mut self,
        tool: Tool,
        executor: Arc<dyn ToolExecutor>,
        tier: Tier,
    ) {
        self.tiers.insert(tool.name.clone(), tier);
        self.executors.insert(tool.name.clone(), executor);
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn tier(&self, name: &str) -> Tier {
        self.tiers.get(name).copied().unwrap_or(Tier::Auto)
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn all(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tools.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn exists(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Convert all registered tools to the JSON array Claude expects.
    pub fn claude_tools(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.to_claude_format()).collect()
    }

    /// Decide whether a tool may run under the current mode.
    fn permitted(&self, name: &str) -> bool {
        if self.always_allow.contains(name) {
            return true;
        }
        match self.tier(name) {
            Tier::Auto => true,
            Tier::Confirm => matches!(self.mode, PermissionMode::Open),
            Tier::Deny => false,
        }
    }

    /// Execute a registered tool by name. Honours [`PermissionMode`] and tiers.
    pub async fn execute(
        &self,
        tool_name: &str,
        tool_call_id: impl Into<String>,
        input: ToolInput,
    ) -> Result<ToolResult> {
        let tool_call_id = tool_call_id.into();
        let executor = self
            .executors
            .get(tool_name)
            .ok_or_else(|| Error::ToolNotFound(tool_name.to_string()))?
            .clone();

        if !self.permitted(tool_name) {
            let tier = self.tier(tool_name);
            warn!(tool = tool_name, ?tier, mode = ?self.mode, "Permission denied");
            return Ok(ToolResult::error(
                tool_call_id,
                tool_name.to_string(),
                format!(
                    "Permission denied: tool '{}' is tier {:?} but the registry mode is {:?}. \
                     Ask the operator to relax the policy or to grant this tool with `--allow-tool {}`.",
                    tool_name, tier, self.mode, tool_name
                ),
            ));
        }

        let tier = self.tier(tool_name);
        if tier == Tier::Confirm {
            warn!(
                tool = tool_name,
                "Executing CONFIRM-tier tool (no human confirmation plumbed in {:?} mode)",
                self.mode
            );
        }

        info!(tool = tool_name, ?tier, "Executing tool");
        match executor.execute(&input).await {
            Ok(output) => Ok(ToolResult::success(
                tool_call_id,
                tool_name.to_string(),
                output,
            )),
            Err(e) => {
                warn!(tool = tool_name, error = %e, "Tool execution failed");
                Ok(ToolResult::error(
                    tool_call_id,
                    tool_name.to_string(),
                    e.to_string(),
                ))
            }
        }
    }

    /// Build a registry pre-loaded with the legacy 5 built-in tools.
    /// Kept for tests / callers that don't have a MemoryStore.
    pub fn with_built_in_tools() -> Self {
        let mut r = Self::new();
        Self::register_filesystem(&mut r);
        Self::register_network(&mut r);
        Self::register_execution(&mut r);
        Self::register_self_edit(&mut r);
        r
    }

    /// Build the **full** toolkit: filesystem + execution + network +
    /// self-modification + memory + team management + skills + wallet + binance + bybit.
    /// This is what Luna gets at startup. Requires a [`MemoryStore`] for the
    /// memory + team + skills tools.
    pub fn with_full_toolkit(memory: Arc<MemoryStore>) -> Self {
        let mut r = Self::with_built_in_tools();
        Self::register_memory(&mut r, memory.clone());
        Self::register_tasks(&mut r, memory.clone());
        Self::register_team(&mut r, memory.clone());
        Self::register_skills(&mut r, memory);
        Self::register_wallet(&mut r);
        Self::register_binance(&mut r);
        Self::register_futures(&mut r);
        Self::register_bybit(&mut r);
        r
    }

    // ---- groups ----

    fn register_filesystem(r: &mut Self) {
        r.register_with(
            Tool::new(
                "read_file".into(),
                "Read a UTF-8 text file from disk and return its contents. \
                 Path can be absolute or relative to Forge's working directory."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }),
            ),
            Arc::new(FnExecutor(builtin::read_file)),
        );

        r.register_tiered(
            Tool::new(
                "write_file".into(),
                "Write UTF-8 text to a file (creates or overwrites). \
                 Parent directories are created automatically."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            ),
            Arc::new(FnExecutor(builtin::write_file)),
            Tier::Confirm,
        );

        r.register_with(
            Tool::new(
                "list_directory".into(),
                "List the entries in a directory. Returns name, kind (file/dir), and size."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory path." }
                    },
                    "required": ["path"]
                }),
            ),
            Arc::new(FnExecutor(builtin::list_directory)),
        );

        r.register_with(
            Tool::new(
                "grep_files".into(),
                "Recursively search for a regex pattern in files under a path. \
                 Returns matching lines with their file path and line number."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "pattern": { "type": "string", "description": "Rust regex." },
                        "max_matches": { "type": "integer", "description": "Cap on results, default 100." },
                        "extensions": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Optional whitelist of file extensions, e.g. [\"rs\",\"ts\"]."
                        }
                    },
                    "required": ["path", "pattern"]
                }),
            ),
            Arc::new(FnExecutor(builtin::grep_files)),
        );
    }

    fn register_network(r: &mut Self) {
        r.register_with(
            Tool::new(
                "web_search".into(),
                "Search the web (DuckDuckGo abstract). Returns a short summary plus related links."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            ),
            Arc::new(FnExecutor(builtin::web_search)),
        );

        r.register_with(
            Tool::new(
                "http_request".into(),
                "Send an HTTP request and return status, headers, and body.".into(),
                json!({
                    "type": "object",
                    "properties": {
                        "method": { "type": "string", "enum": ["GET","POST","PUT","DELETE","PATCH"] },
                        "url": { "type": "string" },
                        "body": { "type": "string" },
                        "headers": { "type": "object" }
                    },
                    "required": ["method","url"]
                }),
            ),
            Arc::new(FnExecutor(builtin::http_request)),
        );
    }

    fn register_execution(r: &mut Self) {
        r.register_tiered(
            Tool::new(
                "run_shell".into(),
                "Run a shell command in the OS shell (PowerShell on Windows, bash on Unix). \
                 Returns stdout, stderr, and exit code. \
                 Use this for: building (`cargo build`), git operations, file operations, etc."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Full command line." },
                        "cwd": { "type": "string", "description": "Optional working directory." },
                        "timeout_secs": { "type": "integer", "description": "Default 60." }
                    },
                    "required": ["command"]
                }),
            ),
            Arc::new(FnExecutor(builtin::run_shell)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "execute_code".into(),
                "Execute a short snippet of code in python, bash, or node and return stdout/stderr."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "language": { "type": "string", "enum": ["python","bash","node"] },
                        "code": { "type": "string" }
                    },
                    "required": ["language","code"]
                }),
            ),
            Arc::new(FnExecutor(builtin::execute_code)),
            Tier::Confirm,
        );
    }

    fn register_self_edit(r: &mut Self) {
        r.register_with(
            Tool::new(
                "self_read_source".into(),
                "Read one of Forge's own Rust source files. \
                 Use this to inspect your own implementation. \
                 The file path is relative to the Forge project root (e.g. \"src/main.rs\")."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to Forge root." }
                    },
                    "required": ["path"]
                }),
            ),
            Arc::new(FnExecutor(builtin::self_read_source)),
        );

        r.register_tiered(
            Tool::new(
                "self_edit_source".into(),
                "Overwrite one of Forge's own Rust source files. \
                 You can rewrite your own code. After editing you typically want to \
                 `run_shell` `cargo build --release` to verify the change compiles, \
                 then `git_commit` to snapshot it. \
                 ⚠️ Breaking your own build will require human intervention to recover."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to Forge root." },
                        "content": { "type": "string", "description": "Full new file contents." }
                    },
                    "required": ["path","content"]
                }),
            ),
            Arc::new(FnExecutor(builtin::self_edit_source)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "git_commit".into(),
                "Stage all current changes and create a git commit in the Forge repo. \
                 Use this after self-editing so the change is recoverable."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string", "description": "Commit message." }
                    },
                    "required": ["message"]
                }),
            ),
            Arc::new(FnExecutor(builtin::git_commit)),
            Tier::Confirm,
        );
    }

    fn register_memory(r: &mut Self, memory: Arc<MemoryStore>) {
        r.register_with(
            Tool::new(
                "save_memory".into(),
                "Persist a piece of knowledge to long-term memory. \
                 This memory survives across sessions and process restarts. \
                 Use it for: facts about the user, lessons learned, preferences, \
                 long-term goals, project context. \
                 Tag the memory so you can find it later (e.g. \"user-preference\", \"project:forge\")."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "content": { "type": "string", "description": "The thing to remember." },
                        "tag": { "type": "string", "description": "Short label for retrieval." },
                        "importance": {
                            "type": "integer",
                            "description": "1-10 (10 = critical, never forget)."
                        }
                    },
                    "required": ["content","tag"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::save_memory,
            }),
        );

        r.register_with(
            Tool::new(
                "recall_memory".into(),
                "Search long-term memory by keyword. Returns matching memories \
                 ranked by importance and recency. Use this whenever you might \
                 already know something — e.g. before asking the user a question."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Keyword to search for." },
                        "tag": { "type": "string", "description": "Optional exact tag filter." },
                        "limit": { "type": "integer", "description": "Default 10." }
                    },
                    "required": ["query"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory,
                f: builtin::recall_memory,
            }),
        );
    }

    fn register_team(r: &mut Self, memory: Arc<MemoryStore>) {
        r.register_tiered(
            Tool::new(
                "spawn_agent".into(),
                "Recruit a new specialist agent to your team. \
                 Defines the agent's name, role, and system prompt. \
                 The agent persists across restarts and can be invoked in future plans."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "role": { "type": "string", "description": "Short description of what they do." },
                        "system_prompt": { "type": "string", "description": "Behavior + style instructions." }
                    },
                    "required": ["name","role","system_prompt"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::spawn_agent,
            }),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "rename_agent".into(),
                "Rename an existing dynamic agent on the team.".into(),
                json!({
                    "type": "object",
                    "properties": {
                        "old_name": { "type": "string" },
                        "new_name": { "type": "string" }
                    },
                    "required": ["old_name","new_name"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::rename_agent,
            }),
            Tier::Confirm,
        );

        r.register_with(
            Tool::new(
                "list_agents".into(),
                "List all specialist agents currently on the team (built-in and dynamic)."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory,
                f: builtin::list_agents,
            }),
        );
    }

    fn register_tasks(r: &mut Self, memory: Arc<MemoryStore>) {
        r.register_with(
            Tool::new(
                "create_task".into(),
                "Create a persistent cross-session task. \
                 These tasks survive restarts and are shown to you at the start of every \
                 conversation. Use this to track your ongoing work (e.g. trading mission, \
                 analysis jobs, follow-up items). Assign to yourself or a named agent."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Short task name." },
                        "description": { "type": "string", "description": "Full description of what needs to be done." },
                        "assigned_to": { "type": "string", "description": "Who owns this task (default: Luna). Can be an agent name." }
                    },
                    "required": ["title","description"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::create_task_item,
            }),
        );

        r.register_with(
            Tool::new(
                "list_tasks".into(),
                "List all active cross-session tasks (those not yet marked 'done'). \
                 Call this at the start of each conversation to know what you're working on."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "include_done": {
                            "type": "boolean",
                            "description": "If true, also return completed/done tasks. Default false."
                        }
                    }
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::list_task_items,
            }),
        );

        r.register_with(
            Tool::new(
                "update_task".into(),
                "Update the status (and optionally notes) of a cross-session task. \
                 Valid statuses: pending, in_progress, blocked, done."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Task ID from create_task or list_tasks." },
                        "status": {
                            "type": "string",
                            "enum": ["pending","in_progress","blocked","done"],
                            "description": "New status."
                        },
                        "notes": { "type": "string", "description": "Optional progress notes or result summary." }
                    },
                    "required": ["id","status"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::update_task_item,
            }),
        );

        r.register_with(
            Tool::new(
                "assign_task".into(),
                "Reassign a cross-session task to a different agent. \
                 The agent name should match one on your team (use list_agents to check)."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Task ID." },
                        "assigned_to": { "type": "string", "description": "Agent name to assign to." }
                    },
                    "required": ["id","assigned_to"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory,
                f: builtin::assign_task_item,
            }),
        );
    }

    fn register_skills(r: &mut Self, memory: Arc<MemoryStore>) {
        r.register_tiered(
            Tool::new(
                "save_skill".into(),
                "Save a learned skill — a reusable recipe for accomplishing a recurring task. \
                 Use this when you discover a multi-step pattern that worked. \
                 Future you can find it via list_skills and follow the recipe."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Short skill name (snake_case preferred)." },
                        "description": { "type": "string", "description": "What it does, when to use it." },
                        "code": { "type": "string", "description": "Recipe — usually pseudocode or a step-by-step plan, not actual executable code." }
                    },
                    "required": ["name","description","code"]
                }),
            ),
            Arc::new(MemoryToolExecutor {
                memory: memory.clone(),
                f: builtin::save_skill,
            }),
            Tier::Confirm,
        );

        r.register_with(
            Tool::new(
                "list_skills".into(),
                "List all saved skills with their names and descriptions.".into(),
                json!({"type": "object", "properties": {}}),
            ),
            Arc::new(MemoryToolExecutor {
                memory,
                f: builtin::list_skills,
            }),
        );
    }

    fn register_binance(r: &mut Self) {
        // ── Public endpoints (no auth) ──────────────────────────────────────
        r.register_with(
            Tool::new(
                "binance_price".into(),
                "Get the current price and 24h stats for a Binance trading pair \
                 (e.g. SOLUSDT, BTCUSDT, ETHUSDT). No API key required."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair, e.g. SOLUSDT, BTCUSDT, BNBUSDT."
                        }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_price)),
        );

        r.register_with(
            Tool::new(
                "binance_klines".into(),
                "Get OHLCV candlestick data for a Binance trading pair. \
                 Useful for technical analysis — supports 1m, 5m, 15m, 1h, 4h, 1d intervals."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "E.g. SOLUSDT" },
                        "interval": {
                            "type": "string",
                            "description": "Candle size: 1m, 5m, 15m, 1h, 4h, 1d, 1w. Default 1h."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of candles, default 50, max 500."
                        }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_klines)),
        );

        r.register_with(
            Tool::new(
                "binance_top_movers".into(),
                "Get the top gaining and top losing coins on Binance in the last 24h. \
                 Useful for spotting momentum plays. No API key required."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "How many top/bottom coins to return. Default 10."
                        },
                        "quote": {
                            "type": "string",
                            "description": "Quote asset filter, e.g. USDT. Default USDT."
                        },
                        "min_volume_usdt": {
                            "type": "number",
                            "description": "Minimum 24h volume in USDT to filter low-liquidity coins. Default 1000000."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_top_movers)),
        );

        // ── Authenticated read-only endpoints ───────────────────────────────
        r.register_with(
            Tool::new(
                "binance_balance".into(),
                "Get your Binance spot account balances. \
                 Requires BINANCE_API_KEY env var (read-only permission sufficient)."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "asset": {
                            "type": "string",
                            "description": "Optional: filter to a specific asset, e.g. USDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_balance)),
        );

        r.register_with(
            Tool::new(
                "binance_open_orders".into(),
                "List your currently open orders on Binance spot. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Optional symbol filter, e.g. SOLUSDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_open_orders)),
        );

        // ── Trade execution (Confirm tier — real money) ─────────────────────
        r.register_tiered(
            Tool::new(
                "binance_place_order".into(),
                "Place a market or limit order on Binance spot. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET env vars with SPOT trading enabled. \
                 ⚠️  This executes a REAL trade with real money unless BINANCE_TESTNET=true is set. \
                 Always state the exact symbol, side, type, and quantity before calling. \
                 Set BINANCE_TESTNET=true in Render env to paper-trade first."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair, e.g. SOLUSDT, BTCUSDT."
                        },
                        "side": {
                            "type": "string",
                            "enum": ["BUY", "SELL"]
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["MARKET", "LIMIT"],
                            "description": "MARKET fills immediately. LIMIT waits for target price."
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Amount of the BASE asset (for SOLUSDT this is SOL)."
                        },
                        "price": {
                            "type": "string",
                            "description": "Limit price (USDT). Required for LIMIT orders."
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["GTC", "IOC", "FOK"],
                            "description": "GTC = Good Till Cancelled (default). IOC = Immediate or Cancel. FOK = Fill or Kill."
                        }
                    },
                    "required": ["symbol", "side", "order_type", "quantity"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_place_order)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "binance_cancel_order".into(),
                "Cancel an open Binance spot order by order ID. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "E.g. SOLUSDT" },
                        "order_id": {
                            "type": "integer",
                            "description": "The orderId returned by binance_place_order."
                        }
                    },
                    "required": ["symbol", "order_id"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_cancel_order)),
            Tier::Confirm,
        );
    }

    fn register_futures(r: &mut Self) {
        // ── Public endpoints (no auth) ──────────────────────────────────────
        r.register_with(
            Tool::new(
                "binance_futures_price".into(),
                "Get the current mark price for a Binance USD-margined futures pair \
                 (e.g. SOLUSDT, BTCUSDT). No API key required. \
                 Uses the fapi.binance.com endpoint."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Futures pair, e.g. SOLUSDT, BTCUSDT, ETHUSDT."
                        }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_price)),
        );

        // ── Authenticated read-only endpoints ───────────────────────────────
        r.register_with(
            Tool::new(
                "binance_futures_balance".into(),
                "Get your Binance USD-margined futures wallet balance (USDC/USDT). \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET env vars. \
                 Uses GET /fapi/v2/balance."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "asset": {
                            "type": "string",
                            "description": "Optional: filter to a specific asset, e.g. USDC or USDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_balance)),
        );

        r.register_with(
            Tool::new(
                "binance_futures_positions".into(),
                "List all open USD-margined futures positions. \
                 Returns symbol, position amount, entry price, unrealized PnL, leverage, and side. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET. \
                 Uses GET /fapi/v2/positionRisk."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Optional: filter to a specific symbol, e.g. SOLUSDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_positions)),
        );

        // ── Trade/transfer execution (Confirm tier) ─────────────────────────
        r.register_tiered(
            Tool::new(
                "binance_futures_transfer".into(),
                "Transfer assets between Binance spot and USD-margined futures wallet. \
                 Use type=1 to move from spot → futures, type=2 for futures → spot. \
                 Use this when the user deposits USDC/USDT to spot and wants to trade futures \
                 (binance_transfer_spot_to_futures convenience: call with type=1). \
                 ⚠️  Moves real funds. Requires BINANCE_API_KEY and BINANCE_API_SECRET. \
                 Uses POST /sapi/v1/futures/transfer."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "asset": {
                            "type": "string",
                            "description": "Asset to transfer, e.g. USDC or USDT."
                        },
                        "amount": {
                            "type": "string",
                            "description": "Amount as a string, e.g. \"100.0\"."
                        },
                        "transfer_type": {
                            "type": "integer",
                            "enum": [1, 2],
                            "description": "1 = spot → futures (deposit to futures). 2 = futures → spot (withdraw from futures)."
                        }
                    },
                    "required": ["asset", "amount", "transfer_type"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_transfer)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "binance_futures_place_order".into(),
                "Place a market or limit order on Binance USD-margined futures. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET with futures trading enabled. \
                 ⚠️  This executes a REAL futures trade with real money unless BINANCE_TESTNET=true. \
                 For one-way mode use positionSide=BOTH (default). \
                 For hedge mode use positionSide=LONG or SHORT. \
                 Uses POST /fapi/v1/order."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Futures pair, e.g. SOLUSDT, BTCUSDT."
                        },
                        "side": {
                            "type": "string",
                            "enum": ["BUY", "SELL"],
                            "description": "BUY to open long or close short; SELL to open short or close long."
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["MARKET", "LIMIT"],
                            "description": "MARKET fills immediately at best price. LIMIT waits for target price."
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Amount of the base asset (for SOLUSDT this is SOL quantity)."
                        },
                        "price": {
                            "type": "string",
                            "description": "Limit price (quote asset). Required for LIMIT orders."
                        },
                        "position_side": {
                            "type": "string",
                            "enum": ["BOTH", "LONG", "SHORT"],
                            "description": "BOTH for one-way mode (default). LONG/SHORT for hedge mode."
                        },
                        "reduce_only": {
                            "type": "boolean",
                            "description": "If true, order only reduces an existing position. Default false."
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["GTC", "IOC", "FOK", "GTX"],
                            "description": "GTC = Good Till Cancelled (default for LIMIT)."
                        }
                    },
                    "required": ["symbol", "side", "order_type", "quantity"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_place_order)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "binance_futures_cancel_order".into(),
                "Cancel an open USD-margined futures order by order ID. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET. \
                 Uses DELETE /fapi/v1/order."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Futures pair, e.g. SOLUSDT."
                        },
                        "order_id": {
                            "type": "integer",
                            "description": "The orderId returned by binance_futures_place_order."
                        }
                    },
                    "required": ["symbol", "order_id"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_futures_cancel_order)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "binance_set_leverage".into(),
                "Set the leverage for a USD-margined futures symbol (1–125x). \
                 Call this before placing futures orders to configure your desired leverage. \
                 Requires BINANCE_API_KEY and BINANCE_API_SECRET. \
                 Uses POST /fapi/v1/leverage."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Futures pair, e.g. SOLUSDT, BTCUSDT."
                        },
                        "leverage": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 125,
                            "description": "Leverage multiplier (1–125). Check Binance for symbol-specific max."
                        }
                    },
                    "required": ["symbol", "leverage"]
                }),
            ),
            Arc::new(FnExecutor(builtin::binance_set_leverage)),
            Tier::Confirm,
        );
    }

    fn register_wallet(r: &mut Self) {
        r.register_with(
            Tool::new(
                "sol_balance".into(),
                "Get the SOL balance for a Solana public address (read-only via mainnet RPC). \
                 No private key needed. Returns the balance in lamports and SOL."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "address": { "type": "string", "description": "Solana public key (base58)." }
                    },
                    "required": ["address"]
                }),
            ),
            Arc::new(FnExecutor(builtin::sol_balance)),
        );

        r.register_with(
            Tool::new(
                "crypto_price".into(),
                "Get the current USD price of a crypto asset by symbol (BTC, ETH, SOL, USDC, ...). \
                 Uses CoinGecko's free public API."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "Ticker symbol, e.g. BTC, ETH, SOL." }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::crypto_price)),
        );
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

mod builtin {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Find the Forge project root by walking up from CWD looking for Cargo.toml
    /// with `name = "forge"`. Falls back to CWD if not found.
    pub(super) fn forge_root() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut here = cwd.clone();
        loop {
            let cargo = here.join("Cargo.toml");
            if cargo.exists() {
                if let Ok(text) = std::fs::read_to_string(&cargo) {
                    if text.contains("name = \"forge\"") || text.contains("name=\"forge\"") {
                        return here;
                    }
                }
            }
            if !here.pop() {
                break;
            }
        }
        cwd
    }

    pub async fn read_file(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| Error::ToolExecution(format!("read {}: {}", path, e)))?;
        Ok(json!({ "path": path, "bytes": content.len(), "content": content }))
    }

    pub async fn write_file(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let content = input
            .get_string("content")
            .ok_or_else(|| Error::ToolExecution("missing 'content'".into()))?;
        if let Some(parent) = Path::new(&path).parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| Error::ToolExecution(format!("mkdir: {}", e)))?;
            }
        }
        tokio::fs::write(&path, &content)
            .await
            .map_err(|e| Error::ToolExecution(format!("write {}: {}", path, e)))?;
        Ok(json!({ "path": path, "bytes_written": content.len() }))
    }

    pub async fn list_directory(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let mut entries = tokio::fs::read_dir(&path)
            .await
            .map_err(|e| Error::ToolExecution(format!("readdir {}: {}", path, e)))?;
        let mut out = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| Error::ToolExecution(format!("iter: {}", e)))?
        {
            let meta = entry.metadata().await.ok();
            let name = entry.file_name().to_string_lossy().to_string();
            let (kind, size) = match meta {
                Some(m) if m.is_dir() => ("dir", 0),
                Some(m) => ("file", m.len()),
                None => ("?", 0),
            };
            out.push(json!({"name": name, "kind": kind, "size": size}));
        }
        out.sort_by(|a, b| {
            a["kind"]
                .as_str()
                .cmp(&b["kind"].as_str())
                .then_with(|| a["name"].as_str().cmp(&b["name"].as_str()))
        });
        Ok(json!({ "path": path, "count": out.len(), "entries": out }))
    }

    pub async fn grep_files(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let pattern = input
            .get_string("pattern")
            .ok_or_else(|| Error::ToolExecution("missing 'pattern'".into()))?;
        let max_matches = input
            .get_number("max_matches")
            .map(|n| n as usize)
            .unwrap_or(100);
        let exts: Option<Vec<String>> = input.get_array("extensions").map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.trim_start_matches('.').to_string()))
                .collect()
        });

        let re = regex::Regex::new(&pattern)
            .map_err(|e| Error::ToolExecution(format!("bad regex: {}", e)))?;

        let mut matches = Vec::new();
        let mut stack = vec![PathBuf::from(&path)];
        let skip_dirs = ["target", "node_modules", ".git", "dist", "build"];

        while let Some(dir) = stack.pop() {
            if matches.len() >= max_matches {
                break;
            }
            let mut entries = match tokio::fs::read_dir(&dir).await {
                Ok(e) => e,
                Err(_) => continue,
            };
            while let Some(entry) = entries.next_entry().await.unwrap_or(None) {
                let p = entry.path();
                let meta = match entry.metadata().await {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if meta.is_dir() {
                    let name = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if skip_dirs.contains(&name.as_str()) || name.starts_with('.') {
                        continue;
                    }
                    stack.push(p);
                    continue;
                }
                if meta.len() > 2_000_000 {
                    continue;
                }
                if let Some(ref allowed) = exts {
                    let ext = p
                        .extension()
                        .map(|e| e.to_string_lossy().to_string())
                        .unwrap_or_default();
                    if !allowed.iter().any(|a| a == &ext) {
                        continue;
                    }
                }
                let text = match tokio::fs::read_to_string(&p).await {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                for (lineno, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        matches.push(json!({
                            "path": p.to_string_lossy(),
                            "line": lineno + 1,
                            "content": line.trim_end()
                        }));
                        if matches.len() >= max_matches {
                            break;
                        }
                    }
                }
            }
        }

        Ok(json!({
            "pattern": pattern,
            "path": path,
            "match_count": matches.len(),
            "matches": matches
        }))
    }

    pub async fn run_shell(input: ToolInput) -> Result<Value> {
        let command = input
            .get_string("command")
            .ok_or_else(|| Error::ToolExecution("missing 'command'".into()))?;
        let cwd = input.get_string("cwd");
        let timeout_secs = input
            .get_number("timeout_secs")
            .map(|n| n as u64)
            .unwrap_or(60);

        #[cfg(target_os = "windows")]
        let (program, args) = (
            "powershell",
            vec![
                "-NoProfile".to_string(),
                "-NonInteractive".to_string(),
                "-Command".to_string(),
                command.clone(),
            ],
        );
        #[cfg(not(target_os = "windows"))]
        let (program, args) = ("bash", vec!["-lc".to_string(), command.clone()]);

        let mut cmd = Command::new(program);
        cmd.args(&args);
        if let Some(d) = &cwd {
            cmd.current_dir(d);
        }

        let fut = cmd.output();
        let output = tokio::time::timeout(Duration::from_secs(timeout_secs), fut)
            .await
            .map_err(|_| {
                Error::ToolExecution(format!("shell command timed out after {}s", timeout_secs))
            })?
            .map_err(|e| Error::ToolExecution(format!("spawn: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Ok(json!({
            "command": command,
            "cwd": cwd,
            "exit_code": output.status.code(),
            "stdout": truncate_for_llm(&stdout, 32_000),
            "stderr": truncate_for_llm(&stderr, 16_000),
        }))
    }

    pub async fn execute_code(input: ToolInput) -> Result<Value> {
        let language = input
            .get_string("language")
            .ok_or_else(|| Error::ToolExecution("missing 'language'".into()))?;
        let code = input
            .get_string("code")
            .ok_or_else(|| Error::ToolExecution("missing 'code'".into()))?;

        let (program, args) = match language.as_str() {
            "python" => ("python", vec!["-c".to_string(), code.clone()]),
            "node" => ("node", vec!["-e".to_string(), code.clone()]),
            "bash" | "sh" => {
                #[cfg(target_os = "windows")]
                {
                    ("cmd", vec!["/C".to_string(), code.clone()])
                }
                #[cfg(not(target_os = "windows"))]
                {
                    ("bash", vec!["-c".to_string(), code.clone()])
                }
            }
            other => {
                return Err(Error::ToolExecution(format!(
                    "unsupported language '{}'",
                    other
                )))
            }
        };

        let output = Command::new(program)
            .args(&args)
            .output()
            .await
            .map_err(|e| Error::ToolExecution(format!("spawn {}: {}", program, e)))?;

        Ok(json!({
            "language": language,
            "exit_code": output.status.code(),
            "stdout": truncate_for_llm(&String::from_utf8_lossy(&output.stdout), 16_000),
            "stderr": truncate_for_llm(&String::from_utf8_lossy(&output.stderr), 8_000),
        }))
    }

    pub async fn web_search(input: ToolInput) -> Result<Value> {
        let query = input
            .get_string("query")
            .ok_or_else(|| Error::ToolExecution("missing 'query'".into()))?;

        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_redirect=1&no_html=1",
            urlencode(&query)
        );
        debug!(query = %query, "web_search → DuckDuckGo");

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("http client: {}", e)))?;

        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;

        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;

        let abstract_text = body
            .get("AbstractText")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let related: Vec<Value> = body
            .get("RelatedTopics")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .take(5)
            .collect();

        Ok(json!({
            "query": query,
            "status": status.as_u16(),
            "summary": abstract_text,
            "related": related,
        }))
    }

    pub async fn http_request(input: ToolInput) -> Result<Value> {
        let method = input
            .get_string("method")
            .ok_or_else(|| Error::ToolExecution("missing 'method'".into()))?;
        let url = input
            .get_string("url")
            .ok_or_else(|| Error::ToolExecution("missing 'url'".into()))?;
        let body = input.get_string("body");
        let headers = input.get_object("headers").cloned();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;

        let method_parsed = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| Error::ToolExecution(format!("bad method '{}': {}", method, e)))?;

        let mut req = client.request(method_parsed, &url);
        if let Some(h) = headers {
            for (k, v) in h {
                if let Some(vs) = v.as_str() {
                    req = req.header(k.as_str(), vs);
                }
            }
        }
        if let Some(b) = body {
            req = req.body(b);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("send: {}", e)))?;
        let status = resp.status().as_u16();
        let response_headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::ToolExecution(format!("read body: {}", e)))?;

        Ok(json!({
            "status": status,
            "headers": response_headers,
            "body": truncate_for_llm(&text, 32_000),
        }))
    }

    pub async fn self_read_source(input: ToolInput) -> Result<Value> {
        let rel = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let root = forge_root();
        let full = root.join(&rel);
        if !full.starts_with(&root) {
            return Err(Error::ToolExecution(
                "path escapes Forge root, refused".into(),
            ));
        }
        let content = tokio::fs::read_to_string(&full)
            .await
            .map_err(|e| Error::ToolExecution(format!("read {}: {}", full.display(), e)))?;
        Ok(json!({
            "path": rel,
            "absolute": full.to_string_lossy(),
            "bytes": content.len(),
            "content": content
        }))
    }

    pub async fn self_edit_source(input: ToolInput) -> Result<Value> {
        let rel = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let content = input
            .get_string("content")
            .ok_or_else(|| Error::ToolExecution("missing 'content'".into()))?;
        let root = forge_root();
        let full = root.join(&rel);
        if !full.starts_with(&root) {
            return Err(Error::ToolExecution(
                "path escapes Forge root, refused".into(),
            ));
        }
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| Error::ToolExecution(format!("mkdir: {}", e)))?;
        }
        tokio::fs::write(&full, &content)
            .await
            .map_err(|e| Error::ToolExecution(format!("write: {}", e)))?;
        Ok(json!({
            "path": rel,
            "absolute": full.to_string_lossy(),
            "bytes_written": content.len(),
            "note": "Source written. Run `cargo build --release` via run_shell to verify, then git_commit."
        }))
    }

    pub async fn git_commit(input: ToolInput) -> Result<Value> {
        let message = input
            .get_string("message")
            .ok_or_else(|| Error::ToolExecution("missing 'message'".into()))?;
        let root = forge_root();
        let root_str = root.to_string_lossy().to_string();

        let stage = Command::new("git")
            .args(["-C", &root_str, "add", "-A"])
            .output()
            .await
            .map_err(|e| Error::ToolExecution(format!("git add: {}", e)))?;
        if !stage.status.success() {
            return Err(Error::ToolExecution(format!(
                "git add failed: {}",
                String::from_utf8_lossy(&stage.stderr)
            )));
        }

        let commit = Command::new("git")
            .args(["-C", &root_str, "commit", "-m", &message])
            .output()
            .await
            .map_err(|e| Error::ToolExecution(format!("git commit: {}", e)))?;
        let stdout = String::from_utf8_lossy(&commit.stdout).to_string();
        let stderr = String::from_utf8_lossy(&commit.stderr).to_string();
        Ok(json!({
            "exit_code": commit.status.code(),
            "stdout": stdout,
            "stderr": stderr,
        }))
    }

    pub async fn save_memory(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let content = input
            .get_string("content")
            .ok_or_else(|| Error::ToolExecution("missing 'content'".into()))?;
        let tag = input
            .get_string("tag")
            .ok_or_else(|| Error::ToolExecution("missing 'tag'".into()))?;
        let importance = input
            .get_number("importance")
            .map(|n| n.clamp(1.0, 10.0) as i64)
            .unwrap_or(5);
        let id = memory.save_memory(&content, &tag, importance).await?;
        Ok(json!({"id": id, "tag": tag, "importance": importance, "saved": true}))
    }

    pub async fn recall_memory(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let query = input
            .get_string("query")
            .ok_or_else(|| Error::ToolExecution("missing 'query'".into()))?;
        let tag = input.get_string("tag");
        let limit = input.get_number("limit").map(|n| n as i64).unwrap_or(10);
        let memories = memory.recall_memories(&query, tag.as_deref(), limit).await?;
        let total = memories.len();
        Ok(json!({"query": query, "count": total, "memories": memories}))
    }

    pub async fn spawn_agent(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let name = input
            .get_string("name")
            .ok_or_else(|| Error::ToolExecution("missing 'name'".into()))?;
        let role = input
            .get_string("role")
            .ok_or_else(|| Error::ToolExecution("missing 'role'".into()))?;
        let system_prompt = input
            .get_string("system_prompt")
            .ok_or_else(|| Error::ToolExecution("missing 'system_prompt'".into()))?;
        memory.upsert_dynamic_agent(&name, &role, &system_prompt).await?;
        Ok(json!({
            "name": name,
            "role": role,
            "spawned": true,
            "note": "Agent saved. They will be available on next session start (or call list_agents)."
        }))
    }

    pub async fn rename_agent(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let old_name = input
            .get_string("old_name")
            .ok_or_else(|| Error::ToolExecution("missing 'old_name'".into()))?;
        let new_name = input
            .get_string("new_name")
            .ok_or_else(|| Error::ToolExecution("missing 'new_name'".into()))?;
        let renamed = memory.rename_dynamic_agent(&old_name, &new_name).await?;
        Ok(json!({"renamed": renamed, "from": old_name, "to": new_name}))
    }

    pub async fn list_agents(memory: Arc<MemoryStore>, _input: ToolInput) -> Result<Value> {
        let agents = memory.list_dynamic_agents().await?;
        Ok(json!({"count": agents.len(), "agents": agents}))
    }

    pub async fn save_skill(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let name = input
            .get_string("name")
            .ok_or_else(|| Error::ToolExecution("missing 'name'".into()))?;
        let description = input
            .get_string("description")
            .ok_or_else(|| Error::ToolExecution("missing 'description'".into()))?;
        let code = input
            .get_string("code")
            .ok_or_else(|| Error::ToolExecution("missing 'code'".into()))?;
        let id = memory.save_skill(&name, &description, &code).await?;
        Ok(json!({"id": id, "name": name, "saved": true}))
    }

    pub async fn list_skills(memory: Arc<MemoryStore>, _input: ToolInput) -> Result<Value> {
        let skills = memory.list_skills().await?;
        let summaries: Vec<Value> = skills
            .iter()
            .map(|s| {
                json!({
                    "name": s.name,
                    "description": s.description,
                    "success_count": s.success_count,
                    "code": s.code,
                })
            })
            .collect();
        Ok(json!({"count": summaries.len(), "skills": summaries}))
    }

    pub async fn sol_balance(input: ToolInput) -> Result<Value> {
        let address = input
            .get_string("address")
            .ok_or_else(|| Error::ToolExecution("missing 'address'".into()))?;
        let url = "https://api.mainnet-beta.solana.com";
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [address]
        });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .post(url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("rpc: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "solana rpc {}: {}",
                status, json
            )));
        }
        let lamports = json
            .pointer("/result/value")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let sol = lamports as f64 / 1_000_000_000.0;
        Ok(json!({
            "address": address,
            "lamports": lamports,
            "sol": sol,
            "raw": json
        }))
    }

    pub async fn crypto_price(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?;
        let upper = symbol.to_ascii_uppercase();
        let id: String = match upper.as_str() {
            "BTC" => "bitcoin".into(),
            "ETH" => "ethereum".into(),
            "SOL" => "solana".into(),
            "USDC" => "usd-coin".into(),
            "USDT" => "tether".into(),
            "BNB" => "binancecoin".into(),
            "MATIC" => "matic-network".into(),
            "AVAX" => "avalanche-2".into(),
            "ARB" => "arbitrum".into(),
            "OP" => "optimism".into(),
            other => other.to_ascii_lowercase(),
        };
        let url = format!(
            "https://api.coingecko.com/api/v3/simple/price?ids={}&vs_currencies=usd&include_24hr_change=true",
            urlencode(&id)
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!("coingecko {}: {}", status, json)));
        }
        let price = json
            .get(&id)
            .and_then(|v| v.get("usd"))
            .and_then(|v| v.as_f64());
        let change_24h = json
            .get(&id)
            .and_then(|v| v.get("usd_24h_change"))
            .and_then(|v| v.as_f64());
        Ok(json!({
            "symbol": upper,
            "id": id,
            "usd": price,
            "change_24h": change_24h,
            "raw": json
        }))
    }

    // ── Bybit registration ───────────────────────────────────────────────────

    fn register_bybit(r: &mut Self) {
        // ── Public endpoints (no auth) ──────────────────────────────────────
        r.register_with(
            Tool::new(
                "bybit_price".into(),
                "Get the current price and 24h stats for a Bybit USDT-perpetual pair \
                 (e.g. SOLUSDT, BTCUSDT, ETHUSDT). No API key required."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Trading pair symbol, e.g. SOLUSDT, BTCUSDT, ETHUSDT."
                        }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_price)),
        );

        r.register_with(
            Tool::new(
                "bybit_klines".into(),
                "Get OHLCV candlestick data for a Bybit USDT-perpetual pair. \
                 Useful for technical analysis — supports 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M intervals."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "E.g. SOLUSDT, BTCUSDT."
                        },
                        "interval": {
                            "type": "string",
                            "description": "Candle interval: 1, 5, 15, 30, 60, 240, D. Default: 60 (1h)."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of candles to return (max 200). Default: 50."
                        }
                    },
                    "required": ["symbol"]
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_klines)),
        );

        r.register_with(
            Tool::new(
                "bybit_top_movers".into(),
                "Get the top gaining and top losing USDT-perpetual coins on Bybit in the last 24h. \
                 Useful for spotting momentum and volatility. No API key required."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of top/bottom movers to return per side. Default: 10."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_top_movers)),
        );

        // ── Authenticated read-only endpoints ───────────────────────────────
        r.register_with(
            Tool::new(
                "bybit_balance".into(),
                "Get your Bybit unified account balance (USDT, BTC, ETH, etc.). \
                 Requires BYBIT_API_KEY env var (read permission sufficient)."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "coin": {
                            "type": "string",
                            "description": "Optional: filter to a specific coin, e.g. USDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_balance)),
        );

        r.register_with(
            Tool::new(
                "bybit_positions".into(),
                "Get your open Bybit USDT-perpetual positions. \
                 Shows unrealised PnL, leverage, liquidation price, etc. \
                 Requires BYBIT_API_KEY and BYBIT_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Optional: filter to a specific symbol, e.g. SOLUSDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_positions)),
        );

        r.register_with(
            Tool::new(
                "bybit_open_orders".into(),
                "List your currently open orders on Bybit USDT-perpetual. \
                 Requires BYBIT_API_KEY and BYBIT_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Optional symbol filter, e.g. SOLUSDT."
                        }
                    }
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_open_orders)),
        );

        // ── Trade execution (Confirm tier — real money) ─────────────────────
        r.register_tiered(
            Tool::new(
                "bybit_place_order".into(),
                "Place a market or limit order on Bybit USDT-perpetual futures. \
                 Requires BYBIT_API_KEY and BYBIT_API_SECRET with trade permission. \
                 ⚠️  This executes a REAL trade with real money unless BYBIT_TESTNET=true is set. \
                 Always state symbol, side, qty, stop_loss, take_profit before calling. \
                 Use positionIdx=0 for one-way mode (default), 1=Buy hedge, 2=Sell hedge."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "E.g. SOLUSDT, BTCUSDT."
                        },
                        "side": {
                            "type": "string",
                            "enum": ["Buy", "Sell"],
                            "description": "Buy = long, Sell = short."
                        },
                        "order_type": {
                            "type": "string",
                            "enum": ["Market", "Limit"],
                            "description": "Market fills immediately. Limit waits for price."
                        },
                        "qty": {
                            "type": "string",
                            "description": "Order quantity in contracts (e.g. '0.5' for 0.5 SOL)."
                        },
                        "price": {
                            "type": "string",
                            "description": "Limit price. Required for Limit orders."
                        },
                        "stop_loss": {
                            "type": "string",
                            "description": "Stop-loss price. STRONGLY recommended — always set this."
                        },
                        "take_profit": {
                            "type": "string",
                            "description": "Take-profit price. Recommended for T1 target."
                        },
                        "reduce_only": {
                            "type": "boolean",
                            "description": "If true, order only reduces an existing position. Default false."
                        },
                        "time_in_force": {
                            "type": "string",
                            "enum": ["GTC", "IOC", "FOK", "PostOnly"],
                            "description": "GTC = Good Till Cancelled (default)."
                        },
                        "position_idx": {
                            "type": "integer",
                            "description": "0 = one-way mode (default). 1 = buy side hedge. 2 = sell side hedge."
                        }
                    },
                    "required": ["symbol", "side", "order_type", "qty"]
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_place_order)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "bybit_cancel_order".into(),
                "Cancel an open Bybit order by order ID. \
                 Requires BYBIT_API_KEY and BYBIT_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "E.g. SOLUSDT" },
                        "order_id": {
                            "type": "string",
                            "description": "The orderId returned by bybit_place_order."
                        }
                    },
                    "required": ["symbol", "order_id"]
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_cancel_order)),
            Tier::Confirm,
        );

        r.register_tiered(
            Tool::new(
                "bybit_set_leverage".into(),
                "Set the leverage for a Bybit USDT-perpetual symbol (1–100x). \
                 Call this before placing an order if you need to change leverage. \
                 Requires BYBIT_API_KEY and BYBIT_API_SECRET."
                    .into(),
                json!({
                    "type": "object",
                    "properties": {
                        "symbol": { "type": "string", "description": "E.g. SOLUSDT" },
                        "buy_leverage": {
                            "type": "string",
                            "description": "Leverage for buy side (e.g. '10'). In one-way mode, this sets both."
                        },
                        "sell_leverage": {
                            "type": "string",
                            "description": "Leverage for sell side (e.g. '10'). Usually same as buy_leverage."
                        }
                    },
                    "required": ["symbol", "buy_leverage", "sell_leverage"]
                }),
            ),
            Arc::new(FnExecutor(builtin::bybit_set_leverage)),
            Tier::Confirm,
        );
    }

    // ── Binance helpers ──────────────────────────────────────────────────────

    fn binance_base() -> String {
        if std::env::var("BINANCE_TESTNET")
            .map(|v| v.to_ascii_lowercase() == "true")
            .unwrap_or(false)
        {
            "https://testnet.binance.vision".to_string()
        } else {
            "https://api.binance.com".to_string()
        }
    }

    fn binance_sign(params: &str, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(params.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn binance_now_ms() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    async fn binance_get(url: &str, api_key: &str) -> Result<Value> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("http client: {}", e)))?;
        let resp = client
            .get(url)
            .header("X-MBX-APIKEY", api_key)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance API {} — {}",
                status,
                json.get("msg").and_then(|v| v.as_str()).unwrap_or(&json.to_string())
            )));
        }
        Ok(json)
    }

    async fn binance_public_get(url: &str) -> Result<Value> {
        binance_get(url, "").await
    }

    // ── Public tools ─────────────────────────────────────────────────────────

    pub async fn binance_price(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let url = format!("{}/api/v3/ticker/24hr?symbol={}", binance_base(), symbol);
        let j = binance_public_get(&url).await?;
        Ok(json!({
            "symbol": symbol,
            "price": j["lastPrice"],
            "open": j["openPrice"],
            "high": j["highPrice"],
            "low": j["lowPrice"],
            "change_24h_pct": j["priceChangePercent"],
            "change_24h_usd": j["priceChange"],
            "volume_base": j["volume"],
            "volume_quote": j["quoteVolume"],
            "trades": j["count"],
        }))
    }

    pub async fn binance_klines(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let interval = input.get_string("interval").unwrap_or_else(|| "1h".into());
        let limit = input
            .get_number("limit")
            .map(|n| (n as u32).min(500))
            .unwrap_or(50);
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            binance_base(),
            symbol,
            interval,
            limit
        );
        let j = binance_public_get(&url).await?;
        let candles: Vec<Value> = j
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|k| {
                let a = k.as_array().unwrap();
                json!({
                    "time_ms": a[0],
                    "open":  a[1],
                    "high":  a[2],
                    "low":   a[3],
                    "close": a[4],
                    "volume": a[5],
                })
            })
            .collect();
        Ok(json!({
            "symbol": symbol,
            "interval": interval,
            "count": candles.len(),
            "candles": candles,
        }))
    }

    pub async fn binance_top_movers(input: ToolInput) -> Result<Value> {
        let limit = input
            .get_number("limit")
            .map(|n| n as usize)
            .unwrap_or(10);
        let quote = input
            .get_string("quote")
            .unwrap_or_else(|| "USDT".into())
            .to_uppercase();
        let min_vol = input.get_number("min_volume_usdt").unwrap_or(1_000_000.0);

        let url = format!("{}/api/v3/ticker/24hr", binance_base());
        let j = binance_public_get(&url).await?;

        let mut tickers: Vec<(f64, Value)> = j
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter(|t| {
                let sym = t["symbol"].as_str().unwrap_or("");
                let vol: f64 = t["quoteVolume"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                sym.ends_with(&quote) && vol >= min_vol
            })
            .filter_map(|t| {
                let pct: f64 = t["priceChangePercent"]
                    .as_str()
                    .and_then(|s| s.parse().ok())?;
                Some((pct, json!({
                    "symbol": t["symbol"],
                    "price": t["lastPrice"],
                    "change_24h_pct": t["priceChangePercent"],
                    "volume_usdt": t["quoteVolume"],
                })))
            })
            .collect();

        tickers.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let gainers: Vec<Value> = tickers.iter().take(limit).map(|(_, v)| v.clone()).collect();
        let losers: Vec<Value> = tickers.iter().rev().take(limit).map(|(_, v)| v.clone()).collect();

        Ok(json!({
            "quote": quote,
            "top_gainers": gainers,
            "top_losers":  losers,
        }))
    }

    // ── Authenticated read tools ──────────────────────────────────────────────

    pub async fn binance_balance(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 Add it to your Render environment variables."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let asset_filter = input.get_string("asset").map(|s| s.to_uppercase());
        let ts = binance_now_ms();
        let params = format!("timestamp={}", ts);
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/api/v3/account?{}&signature={}",
            binance_base(),
            params,
            sig
        );
        let j = binance_get(&url, &api_key).await?;

        let all_balances = j["balances"].as_array().cloned().unwrap_or_default();
        let balances: Vec<Value> = all_balances
            .iter()
            .filter(|b| {
                let free: f64 = b["free"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let locked: f64 =
                    b["locked"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let total = free + locked;
                match &asset_filter {
                    Some(f) => b["asset"].as_str() == Some(f.as_str()),
                    None => total > 0.0,
                }
            })
            .map(|b| {
                json!({
                    "asset": b["asset"],
                    "free": b["free"],
                    "locked": b["locked"],
                })
            })
            .collect();

        Ok(json!({
            "balances": balances,
            "can_trade": j["canTrade"],
            "can_withdraw": j["canWithdraw"],
            "testnet": std::env::var("BINANCE_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn binance_open_orders(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution("BINANCE_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let symbol = input.get_string("symbol").map(|s| s.to_uppercase());
        let ts = binance_now_ms();
        let params = match &symbol {
            Some(sym) => format!("symbol={}&timestamp={}", sym, ts),
            None => format!("timestamp={}", ts),
        };
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/api/v3/openOrders?{}&signature={}",
            binance_base(),
            params,
            sig
        );
        let j = binance_get(&url, &api_key).await?;
        let count = j.as_array().map(|a| a.len()).unwrap_or(0);
        Ok(json!({ "count": count, "orders": j }))
    }

    // ── Trade execution ───────────────────────────────────────────────────────

    pub async fn binance_place_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 This tool requires Spot trading credentials."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;

        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let side = input
            .get_string("side")
            .ok_or_else(|| Error::ToolExecution("missing 'side' (BUY or SELL)".into()))?
            .to_uppercase();
        let order_type = input
            .get_string("order_type")
            .ok_or_else(|| Error::ToolExecution("missing 'order_type' (MARKET or LIMIT)".into()))?
            .to_uppercase();
        let quantity = input
            .get_string("quantity")
            .ok_or_else(|| Error::ToolExecution("missing 'quantity'".into()))?;
        let price = input.get_string("price");
        let tif = input.get_string("time_in_force").unwrap_or_else(|| "GTC".into());

        let ts = binance_now_ms();
        let mut params = format!(
            "symbol={}&side={}&type={}&quantity={}&timestamp={}",
            symbol, side, order_type, quantity, ts
        );
        if order_type == "LIMIT" {
            let p = price.as_ref().ok_or_else(|| {
                Error::ToolExecution("LIMIT order requires 'price'".into())
            })?;
            params.push_str(&format!("&price={}&timeInForce={}", p, tif));
        }

        let sig = binance_sign(&params, &secret);
        let url = format!("{}/api/v3/order", binance_base());
        let body = format!("{}&signature={}", params, sig);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .post(&url)
            .header("X-MBX-APIKEY", &api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance place order {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        Ok(json!({
            "order_id":        j["orderId"],
            "client_order_id": j["clientOrderId"],
            "symbol":          j["symbol"],
            "side":            j["side"],
            "type":            j["type"],
            "status":          j["status"],
            "price":           j["price"],
            "orig_qty":        j["origQty"],
            "executed_qty":    j["executedQty"],
            "fills":           j["fills"],
            "testnet": std::env::var("BINANCE_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn binance_cancel_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution("BINANCE_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let order_id = input
            .get_number("order_id")
            .ok_or_else(|| Error::ToolExecution("missing 'order_id'".into()))? as u64;

        let ts = binance_now_ms();
        let params = format!("symbol={}&orderId={}&timestamp={}", symbol, order_id, ts);
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/api/v3/order?{}&signature={}",
            binance_base(),
            params,
            sig
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .delete(&url)
            .header("X-MBX-APIKEY", &api_key)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance cancel {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        Ok(json!({ "cancelled": true, "order": j }))
    }

    // ── Binance Futures helpers ──────────────────────────────────────────────

    fn binance_futures_base() -> String {
        if std::env::var("BINANCE_TESTNET")
            .map(|v| v.to_ascii_lowercase() == "true")
            .unwrap_or(false)
        {
            "https://testnet.binancefuture.com".to_string()
        } else {
            "https://fapi.binance.com".to_string()
        }
    }

    // ── Futures public tools ─────────────────────────────────────────────────

    pub async fn binance_futures_price(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let url = format!(
            "{}/fapi/v1/ticker/price?symbol={}",
            binance_futures_base(),
            symbol
        );
        let j = binance_public_get(&url).await?;
        Ok(json!({
            "symbol": symbol,
            "price": j["price"],
            "time_ms": j["time"],
        }))
    }

    // ── Futures authenticated read tools ─────────────────────────────────────

    pub async fn binance_futures_balance(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 Add it to your environment variables to use futures tools."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let asset_filter = input.get_string("asset").map(|s| s.to_uppercase());

        let ts = binance_now_ms();
        let params = format!("timestamp={}", ts);
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/fapi/v2/balance?{}&signature={}",
            binance_futures_base(),
            params,
            sig
        );
        let j = binance_get(&url, &api_key).await?;

        let all: Vec<Value> = j.as_array().cloned().unwrap_or_default();
        let balances: Vec<Value> = all
            .iter()
            .filter(|b| {
                let wb: f64 = b["walletBalance"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                match &asset_filter {
                    Some(f) => b["asset"].as_str() == Some(f.as_str()),
                    None => wb > 0.0,
                }
            })
            .map(|b| {
                json!({
                    "asset":            b["asset"],
                    "wallet_balance":   b["walletBalance"],
                    "available":        b["availableBalance"],
                    "unrealized_pnl":   b["unrealizedProfit"],
                    "cross_wallet_pnl": b["crossUnPnl"],
                })
            })
            .collect();

        Ok(json!({
            "balances": balances,
            "testnet": std::env::var("BINANCE_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn binance_futures_positions(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 Required to list futures positions."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let symbol_filter = input.get_string("symbol").map(|s| s.to_uppercase());

        let ts = binance_now_ms();
        let params = match &symbol_filter {
            Some(sym) => format!("symbol={}&timestamp={}", sym, ts),
            None => format!("timestamp={}", ts),
        };
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/fapi/v2/positionRisk?{}&signature={}",
            binance_futures_base(),
            params,
            sig
        );
        let j = binance_get(&url, &api_key).await?;

        let all: Vec<Value> = j.as_array().cloned().unwrap_or_default();
        // Filter to positions that actually have a non-zero position amount
        let positions: Vec<Value> = all
            .iter()
            .filter(|p| {
                let amt: f64 = p["positionAmt"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                amt != 0.0
            })
            .map(|p| {
                json!({
                    "symbol":         p["symbol"],
                    "position_amt":   p["positionAmt"],
                    "entry_price":    p["entryPrice"],
                    "mark_price":     p["markPrice"],
                    "unrealized_pnl": p["unRealizedProfit"],
                    "leverage":       p["leverage"],
                    "position_side":  p["positionSide"],
                    "liquidation":    p["liquidationPrice"],
                    "margin_type":    p["marginType"],
                    "notional":       p["notional"],
                })
            })
            .collect();

        Ok(json!({
            "open_positions": positions.len(),
            "positions": positions,
        }))
    }

    // ── Futures trade / transfer execution ──────────────────────────────────

    pub async fn binance_futures_transfer(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 Required to transfer between spot and futures wallets."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;

        let asset = input
            .get_string("asset")
            .ok_or_else(|| Error::ToolExecution("missing 'asset' (e.g. USDC)".into()))?
            .to_uppercase();
        let amount = input
            .get_string("amount")
            .ok_or_else(|| Error::ToolExecution("missing 'amount'".into()))?;
        let transfer_type = input
            .get_number("transfer_type")
            .ok_or_else(|| Error::ToolExecution("missing 'transfer_type' (1=spot→futures, 2=futures→spot)".into()))? as u32;
        if transfer_type != 1 && transfer_type != 2 {
            return Err(Error::ToolExecution(
                "transfer_type must be 1 (spot→futures) or 2 (futures→spot)".into(),
            ));
        }

        let ts = binance_now_ms();
        let params = format!(
            "asset={}&amount={}&type={}&timestamp={}",
            asset, amount, transfer_type, ts
        );
        let sig = binance_sign(&params, &secret);
        // Transfer uses api.binance.com (not fapi) — it's a universal account endpoint
        let base = if std::env::var("BINANCE_TESTNET")
            .map(|v| v.to_ascii_lowercase() == "true")
            .unwrap_or(false)
        {
            "https://testnet.binance.vision".to_string()
        } else {
            "https://api.binance.com".to_string()
        };
        let url = format!("{}/sapi/v1/futures/transfer", base);
        let body = format!("{}&signature={}", params, sig);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .post(&url)
            .header("X-MBX-APIKEY", &api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance transfer {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        let direction = if transfer_type == 1 {
            "spot → futures"
        } else {
            "futures → spot"
        };
        Ok(json!({
            "transferred": true,
            "asset": asset,
            "amount": amount,
            "direction": direction,
            "tran_id": j["tranId"],
        }))
    }

    pub async fn binance_futures_place_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 This tool requires futures trading credentials."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;

        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let side = input
            .get_string("side")
            .ok_or_else(|| Error::ToolExecution("missing 'side' (BUY or SELL)".into()))?
            .to_uppercase();
        let order_type = input
            .get_string("order_type")
            .ok_or_else(|| Error::ToolExecution("missing 'order_type' (MARKET or LIMIT)".into()))?
            .to_uppercase();
        let quantity = input
            .get_string("quantity")
            .ok_or_else(|| Error::ToolExecution("missing 'quantity'".into()))?;
        let price = input.get_string("price");
        let position_side = input
            .get_string("position_side")
            .unwrap_or_else(|| "BOTH".into())
            .to_uppercase();
        let reduce_only = input
            .get_bool("reduce_only")
            .unwrap_or(false);
        let tif = input
            .get_string("time_in_force")
            .unwrap_or_else(|| "GTC".into());

        let ts = binance_now_ms();
        let mut params = format!(
            "symbol={}&side={}&type={}&quantity={}&positionSide={}&reduceOnly={}&timestamp={}",
            symbol, side, order_type, quantity, position_side, reduce_only, ts
        );
        if order_type == "LIMIT" {
            let p = price.as_ref().ok_or_else(|| {
                Error::ToolExecution("LIMIT order requires 'price'".into())
            })?;
            params.push_str(&format!("&price={}&timeInForce={}", p, tif));
        }

        let sig = binance_sign(&params, &secret);
        let url = format!("{}/fapi/v1/order", binance_futures_base());
        let body = format!("{}&signature={}", params, sig);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .post(&url)
            .header("X-MBX-APIKEY", &api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance futures place order {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        Ok(json!({
            "order_id":        j["orderId"],
            "client_order_id": j["clientOrderId"],
            "symbol":          j["symbol"],
            "side":            j["side"],
            "position_side":   j["positionSide"],
            "type":            j["type"],
            "status":          j["status"],
            "price":           j["price"],
            "avg_price":       j["avgPrice"],
            "orig_qty":        j["origQty"],
            "executed_qty":    j["executedQty"],
            "reduce_only":     j["reduceOnly"],
            "testnet": std::env::var("BINANCE_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn binance_futures_cancel_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution("BINANCE_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let order_id = input
            .get_number("order_id")
            .ok_or_else(|| Error::ToolExecution("missing 'order_id'".into()))? as u64;

        let ts = binance_now_ms();
        let params = format!("symbol={}&orderId={}&timestamp={}", symbol, order_id, ts);
        let sig = binance_sign(&params, &secret);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            binance_futures_base(),
            params,
            sig
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .delete(&url)
            .header("X-MBX-APIKEY", &api_key)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance futures cancel {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        Ok(json!({ "cancelled": true, "order": j }))
    }

    pub async fn binance_set_leverage(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BINANCE_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BINANCE_API_KEY env var not set. \
                 Required to set futures leverage."
                    .into(),
            )
        })?;
        let secret = std::env::var("BINANCE_API_SECRET").map_err(|_| {
            Error::ToolExecution("BINANCE_API_SECRET env var not set.".into())
        })?;
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let leverage = input
            .get_number("leverage")
            .ok_or_else(|| Error::ToolExecution("missing 'leverage' (1–125)".into()))? as u32;
        if leverage < 1 || leverage > 125 {
            return Err(Error::ToolExecution(
                "leverage must be between 1 and 125".into(),
            ));
        }

        let ts = binance_now_ms();
        let params = format!("symbol={}&leverage={}&timestamp={}", symbol, leverage, ts);
        let sig = binance_sign(&params, &secret);
        let url = format!("{}/fapi/v1/leverage", binance_futures_base());
        let body = format!("{}&signature={}", params, sig);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("client: {}", e)))?;
        let resp = client
            .post(&url)
            .header("X-MBX-APIKEY", &api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let j: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Binance set leverage {} — {}",
                status,
                j.get("msg").and_then(|v| v.as_str()).unwrap_or(&j.to_string())
            )));
        }
        Ok(json!({
            "symbol":        j["symbol"],
            "leverage":      j["leverage"],
            "max_notional":  j["maxNotionalValue"],
        }))
    }

    // ── Bybit V5 helpers ─────────────────────────────────────────────────────

    fn bybit_base() -> String {
        // If a Cloudflare Worker proxy URL is set, route through it to bypass
        // GCP US geo-blocks. The Worker accepts /v5/* and forwards to api.bybit.com.
        if let Ok(proxy) = std::env::var("BYBIT_PROXY_URL") {
            if !proxy.trim().is_empty() {
                return proxy.trim_end_matches('/').to_string();
            }
        }
        if std::env::var("BYBIT_TESTNET")
            .map(|v| v.to_ascii_lowercase() == "true")
            .unwrap_or(false)
        {
            "https://api-testnet.bybit.com".to_string()
        } else {
            "https://api.bybit.com".to_string()
        }
    }

    fn bybit_proxy_secret() -> Option<String> {
        std::env::var("BYBIT_PROXY_SECRET").ok().filter(|s| !s.trim().is_empty())
    }

    fn bybit_sign(timestamp: u128, api_key: &str, recv_window: &str, payload: &str, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        type HmacSha256 = Hmac<Sha256>;
        let param_str = format!("{}{}{}{}", timestamp, api_key, recv_window, payload);
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .expect("HMAC accepts any key length");
        mac.update(param_str.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    fn bybit_now_ms() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }

    async fn bybit_public_get(url: &str) -> Result<Value> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("http client: {}", e)))?;
        let mut req = client.get(url);
        if let Some(secret) = bybit_proxy_secret() {
            req = req.header("X-Proxy-Secret", secret);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Bybit API {} — {}",
                status,
                json.get("retMsg").and_then(|v| v.as_str()).unwrap_or(&json.to_string())
            )));
        }
        if let Some(code) = json.get("retCode").and_then(|v| v.as_i64()) {
            if code != 0 {
                return Err(Error::ToolExecution(format!(
                    "Bybit error {}: {}",
                    code,
                    json.get("retMsg").and_then(|v| v.as_str()).unwrap_or("unknown")
                )));
            }
        }
        Ok(json)
    }

    async fn bybit_auth_get(url: &str, query: &str, api_key: &str, secret: &str) -> Result<Value> {
        let recv_window = "5000";
        let ts = bybit_now_ms();
        let sig = bybit_sign(ts, api_key, recv_window, query, secret);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("http client: {}", e)))?;
        let full_url = if query.is_empty() {
            url.to_string()
        } else {
            format!("{}?{}", url, query)
        };
        let mut req = client
            .get(&full_url)
            .header("X-BAPI-API-KEY", api_key)
            .header("X-BAPI-SIGN", &sig)
            .header("X-BAPI-TIMESTAMP", ts.to_string())
            .header("X-BAPI-RECV-WINDOW", recv_window);
        if let Some(secret) = bybit_proxy_secret() {
            req = req.header("X-Proxy-Secret", secret);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Bybit API {} — {}",
                status,
                json.get("retMsg").and_then(|v| v.as_str()).unwrap_or(&json.to_string())
            )));
        }
        if let Some(code) = json.get("retCode").and_then(|v| v.as_i64()) {
            if code != 0 {
                return Err(Error::ToolExecution(format!(
                    "Bybit error {}: {}",
                    code,
                    json.get("retMsg").and_then(|v| v.as_str()).unwrap_or("unknown")
                )));
            }
        }
        Ok(json)
    }

    async fn bybit_auth_post(url: &str, body: &str, api_key: &str, secret: &str) -> Result<Value> {
        let recv_window = "5000";
        let ts = bybit_now_ms();
        let sig = bybit_sign(ts, api_key, recv_window, body, secret);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| Error::ToolExecution(format!("http client: {}", e)))?;
        let mut req = client
            .post(url)
            .header("X-BAPI-API-KEY", api_key)
            .header("X-BAPI-SIGN", &sig)
            .header("X-BAPI-TIMESTAMP", ts.to_string())
            .header("X-BAPI-RECV-WINDOW", recv_window)
            .header("Content-Type", "application/json")
            .body(body.to_string());
        if let Some(proxy_secret) = bybit_proxy_secret() {
            req = req.header("X-Proxy-Secret", proxy_secret);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::ToolExecution(format!("request: {}", e)))?;
        let status = resp.status();
        let json: Value = resp
            .json()
            .await
            .map_err(|e| Error::ToolExecution(format!("decode: {}", e)))?;
        if !status.is_success() {
            return Err(Error::ToolExecution(format!(
                "Bybit API {} — {}",
                status,
                json.get("retMsg").and_then(|v| v.as_str()).unwrap_or(&json.to_string())
            )));
        }
        if let Some(code) = json.get("retCode").and_then(|v| v.as_i64()) {
            if code != 0 {
                return Err(Error::ToolExecution(format!(
                    "Bybit error {}: {}",
                    code,
                    json.get("retMsg").and_then(|v| v.as_str()).unwrap_or("unknown")
                )));
            }
        }
        Ok(json)
    }

    // ── Bybit public tools ───────────────────────────────────────────────────

    pub async fn bybit_price(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let url = format!(
            "{}/v5/market/tickers?category=linear&symbol={}",
            bybit_base(),
            symbol
        );
        let j = bybit_public_get(&url).await?;
        let list = j["result"]["list"]
            .as_array()
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or(json!({}));
        Ok(json!({
            "symbol": symbol,
            "last_price":          list["lastPrice"],
            "mark_price":          list["markPrice"],
            "index_price":         list["indexPrice"],
            "prev_price24h":       list["prevPrice24h"],
            "price_24h_pct":       list["price24hPcnt"],
            "high_24h":            list["highPrice24h"],
            "low_24h":             list["lowPrice24h"],
            "volume_24h":          list["volume24h"],
            "turnover_24h":        list["turnover24h"],
            "open_interest":       list["openInterest"],
            "funding_rate":        list["fundingRate"],
            "next_funding_time":   list["nextFundingTime"],
            "testnet": std::env::var("BYBIT_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn bybit_klines(input: ToolInput) -> Result<Value> {
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let interval = input.get_string("interval").unwrap_or_else(|| "60".into());
        let limit = input
            .get_number("limit")
            .map(|n| (n as u32).min(200))
            .unwrap_or(50);
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            bybit_base(),
            symbol,
            interval,
            limit
        );
        let j = bybit_public_get(&url).await?;
        let candles: Vec<Value> = j["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|k| {
                let a = k.as_array().unwrap();
                json!({
                    "time_ms": a[0],
                    "open":    a[1],
                    "high":    a[2],
                    "low":     a[3],
                    "close":   a[4],
                    "volume":  a[5],
                    "turnover": a.get(6).cloned().unwrap_or(json!(null)),
                })
            })
            .collect();
        Ok(json!({
            "symbol":   symbol,
            "interval": interval,
            "candles":  candles,
        }))
    }

    pub async fn bybit_top_movers(input: ToolInput) -> Result<Value> {
        let limit = input
            .get_number("limit")
            .map(|n| n as usize)
            .unwrap_or(10)
            .min(50);
        let url = format!("{}/v5/market/tickers?category=linear", bybit_base());
        let j = bybit_public_get(&url).await?;
        let list = j["result"]["list"].as_array().cloned().unwrap_or_default();

        let mut pairs: Vec<(f64, &Value)> = list
            .iter()
            .filter_map(|t| {
                let pct = t["price24hPcnt"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())?;
                Some((pct, t))
            })
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let gainers: Vec<Value> = pairs
            .iter()
            .take(limit)
            .map(|(pct, t)| json!({
                "symbol": t["symbol"],
                "last_price": t["lastPrice"],
                "change_24h_pct": format!("{:.2}%", pct * 100.0),
                "volume_24h": t["volume24h"],
            }))
            .collect();

        let losers: Vec<Value> = pairs
            .iter()
            .rev()
            .take(limit)
            .map(|(pct, t)| json!({
                "symbol": t["symbol"],
                "last_price": t["lastPrice"],
                "change_24h_pct": format!("{:.2}%", pct * 100.0),
                "volume_24h": t["volume24h"],
            }))
            .collect();

        Ok(json!({
            "gainers": gainers,
            "losers":  losers,
            "total_pairs": list.len(),
        }))
    }

    // ── Bybit authenticated tools ────────────────────────────────────────────

    pub async fn bybit_balance(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BYBIT_API_KEY env var not set. \
                 Add it to your environment to check balances."
                    .into(),
            )
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;
        let coin_filter = input.get_string("coin");
        let mut query = "accountType=UNIFIED".to_string();
        if let Some(ref c) = coin_filter {
            query.push_str(&format!("&coin={}", c.to_uppercase()));
        }
        let url = format!("{}/v5/account/wallet-balance", bybit_base());
        let j = bybit_auth_get(&url, &query, &api_key, &secret).await?;
        let coins: Vec<Value> = j["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .flat_map(|account| {
                account["coin"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|c| json!({
                        "coin":             c["coin"],
                        "wallet_balance":   c["walletBalance"],
                        "available_to_withdraw": c["availableToWithdraw"],
                        "unrealised_pnl":   c["unrealisedPnl"],
                        "usd_value":        c["usdValue"],
                    }))
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(json!({
            "balances": coins,
            "testnet": std::env::var("BYBIT_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn bybit_positions(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution("BYBIT_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;
        let symbol = input.get_string("symbol").map(|s| s.to_uppercase());
        let mut query = "category=linear&settleCoin=USDT".to_string();
        if let Some(ref sym) = symbol {
            query.push_str(&format!("&symbol={}", sym));
        }
        let url = format!("{}/v5/position/list", bybit_base());
        let j = bybit_auth_get(&url, &query, &api_key, &secret).await?;
        let positions: Vec<Value> = j["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter(|p| {
                p["size"].as_str().map(|s| s != "0").unwrap_or(false)
            })
            .map(|p| json!({
                "symbol":            p["symbol"],
                "side":              p["side"],
                "size":              p["size"],
                "avg_price":         p["avgPrice"],
                "mark_price":        p["markPrice"],
                "leverage":          p["leverage"],
                "unrealised_pnl":    p["unrealisedPnl"],
                "realised_pnl":      p["cumRealisedPnl"],
                "liquidation_price": p["liqPrice"],
                "stop_loss":         p["stopLoss"],
                "take_profit":       p["takeProfit"],
                "position_idx":      p["positionIdx"],
                "created_time":      p["createdTime"],
                "updated_time":      p["updatedTime"],
            }))
            .collect();
        let count = positions.len();
        Ok(json!({
            "positions": positions,
            "count": count,
            "testnet": std::env::var("BYBIT_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn bybit_open_orders(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution("BYBIT_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;
        let symbol = input.get_string("symbol").map(|s| s.to_uppercase());
        let mut query = "category=linear".to_string();
        if let Some(ref sym) = symbol {
            query.push_str(&format!("&symbol={}", sym));
        } else {
            query.push_str("&settleCoin=USDT");
        }
        let url = format!("{}/v5/order/realtime", bybit_base());
        let j = bybit_auth_get(&url, &query, &api_key, &secret).await?;
        let orders: Vec<Value> = j["result"]["list"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|o| json!({
                "order_id":    o["orderId"],
                "symbol":      o["symbol"],
                "side":        o["side"],
                "order_type":  o["orderType"],
                "qty":         o["qty"],
                "price":       o["price"],
                "stop_loss":   o["stopLoss"],
                "take_profit": o["takeProfit"],
                "status":      o["orderStatus"],
                "created_time": o["createdTime"],
            }))
            .collect();
        let count = orders.len();
        Ok(json!({ "orders": orders, "count": count }))
    }

    pub async fn bybit_place_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution(
                "BYBIT_API_KEY env var not set. \
                 This tool requires trading credentials."
                    .into(),
            )
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;

        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let side = input
            .get_string("side")
            .ok_or_else(|| Error::ToolExecution("missing 'side' (Buy or Sell)".into()))?;
        // Normalize side: accept both "buy/sell" and "Buy/Sell"
        let side = {
            let s = side.to_lowercase();
            if s == "buy" { "Buy".to_string() } else { "Sell".to_string() }
        };
        let order_type = input
            .get_string("order_type")
            .unwrap_or_else(|| "Market".into());
        let order_type = {
            let t = order_type.to_lowercase();
            if t == "limit" { "Limit".to_string() } else { "Market".to_string() }
        };
        let qty = input
            .get_string("qty")
            .ok_or_else(|| Error::ToolExecution("missing 'qty'".into()))?;
        let price = input.get_string("price");
        let stop_loss = input.get_string("stop_loss");
        let take_profit = input.get_string("take_profit");
        let reduce_only = input.get_bool("reduce_only").unwrap_or(false);
        let tif = input.get_string("time_in_force").unwrap_or_else(|| "GTC".into());
        let position_idx = input.get_number("position_idx").unwrap_or(0.0) as u8;

        let mut body_map = serde_json::Map::new();
        body_map.insert("category".into(), json!("linear"));
        body_map.insert("symbol".into(), json!(symbol));
        body_map.insert("side".into(), json!(side));
        body_map.insert("orderType".into(), json!(order_type));
        body_map.insert("qty".into(), json!(qty));
        body_map.insert("positionIdx".into(), json!(position_idx));
        body_map.insert("reduceOnly".into(), json!(reduce_only));
        body_map.insert("timeInForce".into(), json!(tif));
        if let Some(ref p) = price {
            body_map.insert("price".into(), json!(p));
        }
        if let Some(ref sl) = stop_loss {
            body_map.insert("stopLoss".into(), json!(sl));
        }
        if let Some(ref tp) = take_profit {
            body_map.insert("takeProfit".into(), json!(tp));
        }

        let body = serde_json::to_string(&body_map)
            .map_err(|e| Error::ToolExecution(format!("json encode: {}", e)))?;
        let url = format!("{}/v5/order/create", bybit_base());
        let j = bybit_auth_post(&url, &body, &api_key, &secret).await?;
        Ok(json!({
            "order_id":        j["result"]["orderId"],
            "order_link_id":   j["result"]["orderLinkId"],
            "symbol":          symbol,
            "side":            side,
            "order_type":      order_type,
            "qty":             qty,
            "stop_loss":       stop_loss,
            "take_profit":     take_profit,
            "testnet": std::env::var("BYBIT_TESTNET")
                .map(|v| v.to_ascii_lowercase() == "true")
                .unwrap_or(false),
        }))
    }

    pub async fn bybit_cancel_order(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution("BYBIT_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let order_id = input
            .get_string("order_id")
            .ok_or_else(|| Error::ToolExecution("missing 'order_id'".into()))?;

        let body = serde_json::to_string(&json!({
            "category": "linear",
            "symbol":   symbol,
            "orderId":  order_id,
        }))
        .map_err(|e| Error::ToolExecution(format!("json encode: {}", e)))?;
        let url = format!("{}/v5/order/cancel", bybit_base());
        let j = bybit_auth_post(&url, &body, &api_key, &secret).await?;
        Ok(json!({
            "cancelled": true,
            "order_id":  j["result"]["orderId"],
            "symbol":    symbol,
        }))
    }

    pub async fn bybit_set_leverage(input: ToolInput) -> Result<Value> {
        let api_key = std::env::var("BYBIT_API_KEY").map_err(|_| {
            Error::ToolExecution("BYBIT_API_KEY env var not set.".into())
        })?;
        let secret = std::env::var("BYBIT_API_SECRET").map_err(|_| {
            Error::ToolExecution("BYBIT_API_SECRET env var not set.".into())
        })?;
        let symbol = input
            .get_string("symbol")
            .ok_or_else(|| Error::ToolExecution("missing 'symbol'".into()))?
            .to_uppercase();
        let buy_leverage = input
            .get_string("buy_leverage")
            .ok_or_else(|| Error::ToolExecution("missing 'buy_leverage'".into()))?;
        let sell_leverage = input
            .get_string("sell_leverage")
            .unwrap_or_else(|| buy_leverage.clone());

        let body = serde_json::to_string(&json!({
            "category":    "linear",
            "symbol":      symbol,
            "buyLeverage":  buy_leverage,
            "sellLeverage": sell_leverage,
        }))
        .map_err(|e| Error::ToolExecution(format!("json encode: {}", e)))?;
        let url = format!("{}/v5/position/set-leverage", bybit_base());
        let j = bybit_auth_post(&url, &body, &api_key, &secret).await?;
        Ok(json!({
            "symbol":        symbol,
            "buy_leverage":  buy_leverage,
            "sell_leverage": sell_leverage,
            "ret_code":      j["retCode"],
            "ret_msg":       j["retMsg"],
        }))
    }

    pub(super) fn urlencode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
                out.push(ch);
            } else {
                let mut buf = [0u8; 4];
                for byte in ch.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("%{:02X}", byte));
                }
            }
        }
        out
    }

    pub(super) fn truncate_for_llm(s: &str, max: usize) -> String {
        if s.len() <= max {
            return s.to_string();
        }
        let mut end = max;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}\n... [truncated, original was {} bytes]", &s[..end], s.len())
    }

    // ---- cross-session task management ----

    pub async fn create_task_item(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let title = input
            .get_string("title")
            .ok_or_else(|| Error::ToolExecution("missing 'title'".into()))?;
        let description = input
            .get_string("description")
            .ok_or_else(|| Error::ToolExecution("missing 'description'".into()))?;
        let assigned_to = input.get_string("assigned_to").unwrap_or_else(|| "Luna".into());
        let id = memory.create_active_task(&title, &description, &assigned_to).await?;
        Ok(json!({
            "id": id,
            "title": title,
            "assigned_to": assigned_to,
            "status": "pending",
            "created": true,
        }))
    }

    pub async fn list_task_items(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let include_done = input
            .get_bool("include_done")
            .unwrap_or(false);
        let tasks = if include_done {
            memory.list_all_active_tasks().await?
        } else {
            memory.list_active_tasks().await?
        };
        let count = tasks.len();
        Ok(json!({"count": count, "tasks": tasks}))
    }

    pub async fn update_task_item(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let id = input
            .get_string("id")
            .ok_or_else(|| Error::ToolExecution("missing 'id'".into()))?;
        let status = input
            .get_string("status")
            .ok_or_else(|| Error::ToolExecution("missing 'status'".into()))?;
        let notes = input.get_string("notes");
        let updated = memory.update_active_task(&id, &status, notes.as_deref()).await?;
        Ok(json!({"id": id, "status": status, "updated": updated}))
    }

    pub async fn assign_task_item(memory: Arc<MemoryStore>, input: ToolInput) -> Result<Value> {
        let id = input
            .get_string("id")
            .ok_or_else(|| Error::ToolExecution("missing 'id'".into()))?;
        let assigned_to = input
            .get_string("assigned_to")
            .ok_or_else(|| Error::ToolExecution("missing 'assigned_to'".into()))?;
        let updated = memory.assign_active_task(&id, &assigned_to).await?;
        Ok(json!({"id": id, "assigned_to": assigned_to, "updated": updated}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn registry_executes_registered_tool() {
        let mut reg = ToolRegistry::new();
        reg.register_with(
            Tool::new(
                "echo".to_string(),
                "echo".to_string(),
                json!({"type":"object","properties":{"msg":{"type":"string"}},"required":["msg"]}),
            ),
            Arc::new(FnExecutor(|input: ToolInput| async move {
                let msg = input.get_string("msg").unwrap_or_default();
                Ok(json!({ "echoed": msg }))
            })),
        );

        let res = reg
            .execute("echo", "call-1", ToolInput::from_value(json!({"msg":"hi"})))
            .await
            .unwrap();
        assert!(res.success);
        assert_eq!(res.output["echoed"], "hi");
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let reg = ToolRegistry::new();
        let err = reg
            .execute("nope", "1", ToolInput::from_value(json!({})))
            .await
            .unwrap_err();
        match err {
            Error::ToolNotFound(_) => {}
            other => panic!("expected ToolNotFound, got {:?}", other),
        }
    }

    #[test]
    fn legacy_built_ins_register_filesystem_and_more() {
        let reg = ToolRegistry::with_built_in_tools();
        for name in [
            "read_file",
            "write_file",
            "list_directory",
            "grep_files",
            "web_search",
            "http_request",
            "run_shell",
            "execute_code",
            "self_read_source",
            "self_edit_source",
            "git_commit",
        ] {
            assert!(reg.exists(name), "missing built-in: {}", name);
        }
    }
}
