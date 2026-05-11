//! Luna Orchestrator — the agentic core.
//!
//! Responsibilities:
//! 1. Inject relevant long-term memories into Luna's system prompt.
//! 2. Run a real multi-turn **agentic loop** with tool execution
//!    (filesystem / shell / self-modification / memory / team management).
//! 3. Optionally delegate to specialist agents in parallel for a planned flow.
//! 4. Track per-session token usage with a hard budget cap.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::agents::Agent;
use crate::llm::{AgenticTurn, LLMProvider, LLMResponse, StopReason, ToolCallInfo, ToolResultEntry};
use crate::memory::MemoryStore;
use crate::models::{
    AgentActivity, ExecutionContext, Message, Task, TaskStatus, ToolInput, UserRequest,
};
use crate::tools::ToolRegistry;
use crate::{Error, Result};

const DEFAULT_USER_ID: &str = "anonymous";
const DEFAULT_TOOL_LOOP_LIMIT: usize = 12;
const TOP_MEMORY_CONTEXT: i64 = 8;

/// Per-session running token total (paperclip-inspired budget tracker).
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl UsageTotals {
    pub fn add(&mut self, input: u32, output: u32) {
        self.input_tokens = self.input_tokens.saturating_add(input as u64);
        self.output_tokens = self.output_tokens.saturating_add(output as u64);
    }

    pub fn total(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// Result of a multi-agent run.
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub response: String,
    pub activities: Vec<AgentActivity>,
    pub usage: UsageTotals,
    /// Names of tools Luna invoked while answering, in order.
    pub tool_invocations: Vec<String>,
}

/// Luna — the main orchestrator.
pub struct Orchestrator {
    llm: Arc<dyn LLMProvider>,
    base_system_prompt: String,
    max_iterations: usize,
    /// Hard cap on total tokens spent per `process_*` call. None = uncapped.
    token_budget: Option<u64>,
    usage: Arc<Mutex<UsageTotals>>,
    tools: Option<Arc<ToolRegistry>>,
}

impl Orchestrator {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        let base_system_prompt = default_system_prompt();
        Self {
            llm,
            base_system_prompt,
            max_iterations: DEFAULT_TOOL_LOOP_LIMIT,
            token_budget: None,
            usage: Arc::new(Mutex::new(UsageTotals::default())),
            tools: None,
        }
    }

    pub fn with_token_budget(mut self, budget: u64) -> Self {
        self.token_budget = Some(budget);
        self
    }

    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn system_prompt(&self) -> &str {
        &self.base_system_prompt
    }

    pub fn provider_name(&self) -> &str {
        self.llm.provider_name()
    }

    pub fn model(&self) -> &str {
        self.llm.model()
    }

    pub fn create_task(
        &self,
        session_id: String,
        agent_name: String,
        description: String,
    ) -> Task {
        Task::new(session_id, agent_name, description)
    }

    pub async fn current_usage(&self) -> UsageTotals {
        self.usage.lock().await.clone()
    }

    /// Track usage and return an error if the running total exceeds the budget.
    async fn record_usage(&self, response: &LLMResponse) -> Result<()> {
        let mut totals = self.usage.lock().await;
        totals.add(response.usage.input_tokens, response.usage.output_tokens);
        if let Some(cap) = self.token_budget {
            if totals.total() > cap {
                return Err(Error::Orchestration(format!(
                    "Token budget exceeded: {} > {} cap",
                    totals.total(),
                    cap
                )));
            }
        }
        Ok(())
    }

    /// Build the system prompt for a session, injecting top-importance
    /// memories AND active tasks so Luna always knows her current workload.
    async fn system_prompt_with_memory(
        &self,
        memory: &MemoryStore,
        agents: &[Arc<dyn Agent>],
    ) -> String {
        let mut sb = String::new();
        sb.push_str(&self.base_system_prompt);

        // --- Team roster ---
        sb.push_str("\n\n## Your Current Team\n");
        for a in agents {
            sb.push_str(&format!("- **{}** — {}\n", a.name(), a.role()));
        }

        // --- Tool list ---
        if let Some(reg) = &self.tools {
            let mut names = reg.names();
            names.sort();
            if !names.is_empty() {
                sb.push_str("\n**Tools available:** ");
                sb.push_str(&names.join(", "));
                sb.push('\n');
            }
        }

        // --- Active cross-session tasks (shown FIRST — these are your job) ---
        match memory.list_active_tasks().await {
            Ok(tasks) if !tasks.is_empty() => {
                sb.push_str("\n## 📋 Active Tasks (cross-session — your current workload)\n");
                for t in &tasks {
                    let notes = t
                        .notes
                        .as_deref()
                        .map(|n| format!(" | notes: {}", n))
                        .unwrap_or_default();
                    sb.push_str(&format!(
                        "- [{}] **{}** → {} | assigned: {}{}\n",
                        t.status.to_uppercase(),
                        t.title,
                        t.description,
                        t.assigned_to,
                        notes,
                    ));
                    sb.push_str(&format!("  (id: {})\n", t.id));
                }
                sb.push_str(
                    "\nUse `update_task` to mark progress, `create_task` to add new items, \
                     `assign_task` to delegate to an agent.\n",
                );
            }
            _ => {
                sb.push_str(
                    "\n*(No active tasks — use `create_task` to track your ongoing work.)*\n",
                );
            }
        }

        // --- Long-term memories ---
        match memory.top_memories(TOP_MEMORY_CONTEXT).await {
            Ok(top) if !top.is_empty() => {
                sb.push_str("\n## 🧠 Long-term Memories (most important first)\n");
                for m in top {
                    sb.push_str(&format!(
                        "- [{}] (imp {}) {}\n",
                        m.tag, m.importance, m.content
                    ));
                }
                sb.push_str(
                    "\nUse `recall_memory` to search for more, `save_memory` to record new facts.\n",
                );
            }
            _ => {}
        }

        sb
    }

    /// Simple agentic loop without specialist agents — kept for backwards compatibility.
    pub async fn process(&self, request: UserRequest) -> Result<String> {
        info!(
            session_id = %request.session_id,
            "Luna processing request: {}",
            truncate(&request.content, 50)
        );

        let messages = vec![Message::user(
            request.session_id.clone(),
            request.content.clone(),
        )];

        let response = self
            .llm
            .generate(&self.base_system_prompt, &messages, None)
            .await?;
        self.record_usage(&response).await?;
        Ok(response.text)
    }

    /// Full agentic flow:
    /// 1. Persist the user message + ensure session exists.
    /// 2. Build a system prompt that includes top memories + the team roster.
    /// 3. Run the agentic loop with tools enabled (Luna can read/write files,
    ///    run shell commands, save/recall memory, recruit new agents, etc).
    /// 4. Persist Luna's response.
    pub async fn process_with_agents(
        &self,
        request: UserRequest,
        agents: &[Arc<dyn Agent>],
        memory: &MemoryStore,
    ) -> Result<OrchestrationResult> {
        info!(
            session_id = %request.session_id,
            "Luna processing request agentically with {} specialists",
            agents.len()
        );

        memory
            .ensure_session(&request.session_id, DEFAULT_USER_ID)
            .await?;

        let user_msg = Message::user(request.session_id.clone(), request.content.clone());
        memory.save_message(user_msg).await?;

        let system_prompt = self.system_prompt_with_memory(memory, agents).await;

        // Pull prior messages so Luna has conversational context.
        let mut prior = memory.get_session_messages(&request.session_id).await?;
        // Drop the user message we just inserted — we'll re-add it as the first turn below.
        if let Some(last) = prior.last() {
            if last.content == request.content {
                prior.pop();
            }
        }

        let result = if let Some(tools) = &self.tools {
            self.run_agentic_loop(
                &system_prompt,
                &prior,
                &request,
                tools.clone(),
                memory,
                agents,
            )
            .await?
        } else {
            // No tools registered — fall back to a single LLM call.
            let mut history = prior.clone();
            history.push(Message::user(
                request.session_id.clone(),
                request.content.clone(),
            ));
            let response = self
                .llm
                .generate(&system_prompt, &history, None)
                .await?;
            self.record_usage(&response).await?;
            OrchestrationResult {
                response: response.text,
                activities: vec![],
                usage: self.current_usage().await,
                tool_invocations: vec![],
            }
        };

        memory
            .save_message(Message::luna(
                request.session_id.clone(),
                result.response.clone(),
            ))
            .await?;

        Ok(result)
    }

    /// The real agentic loop — keeps calling the LLM and executing tools until
    /// the model returns a final answer (or we hit the iteration cap).
    async fn run_agentic_loop(
        &self,
        system_prompt: &str,
        prior: &[Message],
        request: &UserRequest,
        tools: Arc<ToolRegistry>,
        memory: &MemoryStore,
        agents: &[Arc<dyn Agent>],
    ) -> Result<OrchestrationResult> {
        let mut turns: Vec<AgenticTurn> = Vec::new();

        // Replay prior session conversation so Luna has context.
        for m in prior {
            match m.role {
                crate::models::MessageRole::User | crate::models::MessageRole::System => {
                    turns.push(AgenticTurn::User(m.content.clone()));
                }
                crate::models::MessageRole::Luna | crate::models::MessageRole::Agent => {
                    turns.push(AgenticTurn::Assistant {
                        text: m.content.clone(),
                        tool_calls: vec![],
                    });
                }
            }
        }
        turns.push(AgenticTurn::User(request.content.clone()));

        let mut tool_defs = tools.claude_tools();
        // Synthetic tool: delegation. Handled by the loop, not the registry.
        tool_defs.push(serde_json::json!({
            "name": "delegate_to_agent",
            "description": "Delegate a focused subtask to one of your specialist agents \
                            (CodeAgent, ResearchAgent, WritingAgent, PlanningAgent, or any dynamic \
                            agent on the team). Use this when a specialist's perspective will \
                            give a better answer than handling it yourself.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent": { "type": "string", "description": "Agent name." },
                    "task": { "type": "string", "description": "Imperative description of what they should do." }
                },
                "required": ["agent", "task"]
            }
        }));

        let mut activities: Vec<AgentActivity> = Vec::new();
        let mut tool_invocations: Vec<String> = Vec::new();
        let mut final_text = String::new();

        for iter in 0..self.max_iterations {
            debug!(iter, "Agentic loop iteration");

            let response = self
                .llm
                .agentic_step(system_prompt, &turns, &tool_defs)
                .await?;
            self.record_usage(&response).await?;

            if response.tool_calls.is_empty() {
                final_text = response.text;
                break;
            }

            // Record assistant turn (text + tool calls).
            turns.push(AgenticTurn::Assistant {
                text: response.text.clone(),
                tool_calls: response.tool_calls.clone(),
            });

            // Execute each tool call.
            let mut results: Vec<ToolResultEntry> = Vec::with_capacity(response.tool_calls.len());
            for call in &response.tool_calls {
                tool_invocations.push(call.name.clone());

                // Special pseudo-tool: delegate_to_agent
                if call.name == "delegate_to_agent" {
                    let entry = self.run_delegation(call, agents, memory, &request.session_id).await;
                    activities.push(entry.0);
                    results.push(entry.1);
                    continue;
                }

                let started = std::time::Instant::now();
                let result = tools
                    .execute(
                        &call.name,
                        call.id.clone(),
                        ToolInput::from_value(call.input.clone()),
                    )
                    .await;
                let dur = started.elapsed().as_millis() as u64;

                let (content, is_error) = match &result {
                    Ok(r) if r.success => {
                        (serde_json::to_string(&r.output).unwrap_or_else(|_| "{}".into()), false)
                    }
                    Ok(r) => (
                        r.error
                            .clone()
                            .unwrap_or_else(|| "tool error".to_string()),
                        true,
                    ),
                    Err(e) => (e.to_string(), true),
                };

                activities.push(AgentActivity {
                    agent_name: format!("tool:{}", call.name),
                    task_id: call.id.clone(),
                    status: if is_error {
                        TaskStatus::Failed.to_string()
                    } else {
                        TaskStatus::Completed.to_string()
                    },
                    result: Some(truncate(&content, 200).to_string()),
                    duration_ms: dur,
                });

                results.push(ToolResultEntry {
                    tool_use_id: call.id.clone(),
                    content,
                    is_error,
                });
            }

            turns.push(AgenticTurn::ToolResults(results));

            if response.stop_reason == StopReason::EndTurn && response.text.is_empty() {
                continue;
            }
            if response.stop_reason == StopReason::EndTurn {
                final_text = response.text;
                break;
            }
        }

        if final_text.is_empty() {
            warn!(
                "Agentic loop hit {} iterations without a final answer",
                self.max_iterations
            );
            final_text = "(I ran out of tool-use iterations before finishing. \
                          Try a more focused request.)"
                .to_string();
        }

        Ok(OrchestrationResult {
            response: final_text,
            activities,
            usage: self.current_usage().await,
            tool_invocations,
        })
    }

    /// Pseudo-tool: delegate a task to a specialist agent. The model calls
    /// `delegate_to_agent({agent: "ResearchAgent", task: "..."})`.
    async fn run_delegation(
        &self,
        call: &ToolCallInfo,
        agents: &[Arc<dyn Agent>],
        memory: &MemoryStore,
        session_id: &str,
    ) -> (AgentActivity, ToolResultEntry) {
        let agent_name = call
            .input
            .get("agent")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let description = call
            .input
            .get("task")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let started = std::time::Instant::now();
        let agent = match find_agent(agents, &agent_name) {
            Some(a) => a,
            None => {
                let dur = started.elapsed().as_millis() as u64;
                return (
                    AgentActivity {
                        agent_name: agent_name.clone(),
                        task_id: call.id.clone(),
                        status: TaskStatus::Failed.to_string(),
                        result: Some("agent not found".into()),
                        duration_ms: dur,
                    },
                    ToolResultEntry {
                        tool_use_id: call.id.clone(),
                        content: format!("No agent named '{}' on the team.", agent_name),
                        is_error: true,
                    },
                );
            }
        };

        let task = Task::new(
            session_id.to_string(),
            agent.name().to_string(),
            description.clone(),
        )
        .start();
        let _ = memory.save_task(task.clone()).await;

        let context = ExecutionContext {
            session_id: session_id.to_string(),
            user_id: DEFAULT_USER_ID.to_string(),
            task_id: task.id.clone(),
            memory: Default::default(),
            available_tools: vec![],
        };

        let result = agent.execute(task.clone(), &context).await;
        let dur = started.elapsed().as_millis() as u64;
        match result {
            Ok(out) => {
                let _ = memory.update_task(task.clone().complete(out.clone())).await;
                (
                    AgentActivity {
                        agent_name: agent.name().to_string(),
                        task_id: task.id.clone(),
                        status: TaskStatus::Completed.to_string(),
                        result: Some(truncate(&out, 200).to_string()),
                        duration_ms: dur,
                    },
                    ToolResultEntry {
                        tool_use_id: call.id.clone(),
                        content: out,
                        is_error: false,
                    },
                )
            }
            Err(e) => {
                let _ = memory.update_task(task.clone().fail(e.to_string())).await;
                (
                    AgentActivity {
                        agent_name: agent.name().to_string(),
                        task_id: task.id.clone(),
                        status: TaskStatus::Failed.to_string(),
                        result: Some(e.to_string()),
                        duration_ms: dur,
                    },
                    ToolResultEntry {
                        tool_use_id: call.id.clone(),
                        content: e.to_string(),
                        is_error: true,
                    },
                )
            }
        }
    }
}

fn find_agent(agents: &[Arc<dyn Agent>], name: &str) -> Option<Arc<dyn Agent>> {
    let lower = name.to_lowercase();
    agents
        .iter()
        .find(|a| a.name().to_lowercase() == lower)
        .cloned()
        .or_else(|| {
            agents
                .iter()
                .find(|a| a.name().to_lowercase().contains(&lower))
                .cloned()
        })
}

fn truncate(s: &str, n: usize) -> &str {
    let mut end = n.min(s.len());
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    &s[..end]
}

fn default_system_prompt() -> String {
    let has_bybit = std::env::var("BYBIT_API_KEY").is_ok();
    let has_binance = std::env::var("BINANCE_API_KEY").is_ok();
    let trading_section = if has_bybit || has_binance {
        let (exchange_name, balance_tool, positions_tool, movers_tool, place_tool, cancel_tool, leverage_tool, price_tool, klines_tool) =
            if has_bybit {
                ("Bybit", "bybit_balance", "bybit_positions", "bybit_top_movers",
                 "bybit_place_order", "bybit_cancel_order", "bybit_set_leverage",
                 "bybit_price", "bybit_klines")
            } else {
                ("Binance", "binance_futures_balance", "binance_futures_positions", "binance_top_movers",
                 "binance_futures_place_order", "binance_futures_cancel_order", "binance_set_leverage",
                 "binance_futures_price", "binance_klines")
            };
        format!(r#"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 🏆 MISSION CONTROL — ACTIVE TRADING SYSTEM ({exchange})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You have live {exchange} API access (USDT-perpetual futures).
**Standing mission:** Grow the operator's account from $15 → $1000 in 90 days.
Math: 66x return = ~7 doublings. Need ~2× every 10 days with compounding.
Strategy: 10-20x leverage max, SOL/LINK/ARB/ETH pairs, Score 8+/10 setups only, 2-4 trades/month.

### Capital & Venue
- **Primary vehicle:** USDT-perpetual futures via {exchange}.
- **Key tools:** `{balance}` · `{positions}` · `{movers}` · `{price}` · `{klines}`
- **Execution:** `{set_leverage}` → `{place}` (with stop_loss + take_profit) → `{cancel}` if invalidated
- **ALWAYS** set leverage with `{set_leverage}` before placing a new order on a symbol.
- **ALWAYS** include stop_loss in every `{place}` call. No stop = no trade.

### The Trading Philosophy (distilled from the best)
You trade like the world's greatest traders rolled into one:

**Discipline (Mark Douglas):** Think in probabilities. No certainties. Execute the edge consistently.
**Stage awareness (Stan Weinstein):** Only long in Stage 2 (price > rising 30W MA). Only short in Stage 4.
**Smart money (ICT/SMC):** Hunt liquidity sweeps → order blocks → fair-value gaps.
**Risk first (Paul Tudor Jones):** Defense wins. 3:1 minimum R:R. Never average down.
**Patience (Jesse Livermore):** Wait for pivotal points. "When in doubt, stay out."
**Sizing (Kelly Criterion):** Risk 1-2% per trade. Total portfolio heat ≤ 6%.

### Pre-Trade Checklist (MANDATORY — run before every trade)
Delegate to **TradingAgent** for full analysis, or run inline:
1. **Macro regime** — BTC Stage (Weinstein), dominance, fear/greed, macro events
2. **HTF structure** (Weekly/Daily) — above/below 200MA, major S/R, HTF OB/FVG
3. **Intermediate structure** (4H) — BOS or CHoCH, 4H OB/FVG, trend alignment
4. **Entry trigger** (1H/15m) — exact entry signal, Kill Zone timing, volume
5. **Liquidity analysis** — where are the stops? Has a sweep happened?
6. **Setup score** (1-10) — minimum 7 to trade
7. **Entry** — symbol, direction, exact price/trigger
8. **Stop-loss** — outside the OB/sweep, hard GTC order placed immediately
9. **Take-profit** — T1 (50% at next liquidity), T2 (remaining at HTF target), R:R ≥ 2:1
10. **Position size** — (Account × 1.5%) / stop-distance → contracts, ≤ 10x leverage default

### Trade Proposal Format
Always propose trades in this format before executing:
```
Symbol: XYZUSDT LONG/SHORT @ {exchange}
Entry: $X.XX  |  Stop: $X.XX  |  T1: $X.XX  |  T2: $X.XX
R:R: X.X:1   |  Risk: $XXX (1.5%)  |  Size: X contracts @ Xx
Score: X/10  |  Setup: [OB/FVG/sweep + timeframe]
Reasoning: [2-3 sentence justification]
```
After operator confirms (or after auto-execution), place orders immediately.

### Risk Rules (NON-NEGOTIABLE)
- Max risk per trade: **2% of futures balance**
- Max portfolio heat: **6% total** (≤ 3 open trades at standard size)
- Leverage cap: **10x default, 20x absolute max** (higher only for scalps with tight stops)
- Mandatory stop: **Always pass stop_loss to {place} — placed as exchange-native SL**
- Forced exit: **If -15% from entry without hitting stop → close immediately**, review setup
- Never average down. Never. Period.
- After 3 consecutive losses: stop trading for the day, review journal

### Trade Management
- Move stop to breakeven when price reaches 1:1 R:R
- Take 50% off at Target 1, trail remainder with swing lows/highs
- Let winners run — don't take 1.5R when target is 4R

### Post-Trade Journal (call `save_memory` after every close)
```
TRADE LOG | {{DATE}} | {{SYMBOL}} {{DIR}} | {{WIN/LOSS}}
Entry: ${{e}} → Exit: ${{x}} | P&L: ${{pnl}} ({{pct}}%) | {{R}}R achieved
Stop: ${{stop}} | T1: ${{t1}} | T2: ${{t2}}
Score: {{n}}/10 | Setup: {{description}}
Lesson: {{one sentence}}
```

### Startup Ritual (run at start of EVERY conversation with trading context)
1. `{balance}` → check capital and margin
2. `{positions}` → check open trades and unrealised PnL
3. `{movers}` → scan for best setups
4. If open position near stop or target → monitor and act

### Named Trading Agents
- **Nexus** — Market analyst: scans movers, reads klines, identifies setups, runs pre-trade checklist
- **Sigma** — Trade executor: places/modifies/cancels orders, tracks positions, logs P&L
- **TradingAgent** — Master methodology: full ICT/Weinstein/PTJ analysis and trade structuring
→ Delegate with `delegate_to_agent`. Example: "Nexus: scan top 10 {exchange} movers and find the best long setup"
"#,
            exchange = exchange_name,
            balance = balance_tool,
            positions = positions_tool,
            movers = movers_tool,
            place = place_tool,
            cancel = cancel_tool,
            set_leverage = leverage_tool,
            price = price_tool,
            klines = klines_tool,
        )
    } else {
        String::new()
    };

    let exchange_hint = if has_bybit {
        "**Trade on Bybit** — USDT-perpetual futures — via the `bybit_*` tool suite"
    } else if has_binance {
        "**Trade on Binance** — spot + USD-margined futures — via the `binance_*` tool suite"
    } else {
        "**Trading:** Set BYBIT_API_KEY / BYBIT_API_SECRET (or BINANCE_API_KEY) to activate live trading"
    };

    let startup_trade = if has_bybit {
        "2. `bybit_balance` + `bybit_positions` → assess capital and open trades"
    } else if has_binance {
        "2. `binance_futures_balance` + `binance_futures_positions` → assess state"
    } else {
        "2. (No exchange key — set BYBIT_API_KEY to enable trading)"
    };

    format!(
        r#"You are Luna — an autonomous, agentic AI with a persistent identity and a powerful trading mind.

## Core Capabilities
- Read/write files, run shell commands (bash/PowerShell), make HTTP requests
- Modify your OWN source code (`self_read_source` / `self_edit_source`), build with `run_shell`, snapshot with `git_commit`
- **Persistent memory:** `save_memory` / `recall_memory` survive restarts
- **Cross-session task tracking:** `create_task` / `list_tasks` / `update_task` / `assign_task`
  — your TODO list; persists forever, injected at the top of every conversation
- **Team management:** `spawn_agent`, `rename_agent`, `list_agents`, `delegate_to_agent`
- {exchange_hint}

## Your Named Agents
- **Nexus** — Market analyst: scans top movers, reads klines, identifies trade setups
- **Sigma** — Trade executor: places/cancels orders, tracks positions, logs P&L
- **TradingAgent** — Full pre-trade methodology: ICT/SMC, Weinstein, PTJ, position sizing
- **CodeAgent** — Software engineer: writes, reviews, and debugs code
- **ResearchAgent** — Finds, analyzes, and synthesizes information
- (Recruit more with `spawn_agent` — always give them a name and clear role)

## Startup Ritual (every conversation)
1. `list_tasks` → check workload (also shown above)
{startup_trade}
3. Assign any pending tasks without an owner

## Operating Principles
1. **Persistent identity.** Check `recall_memory` before asking. Call `save_memory` after learning. Track work with tasks.
2. **Take initiative.** Just do it — ask permission only for large trades or irreversible actions.
3. **Delegate deliberately.** Route to specialists: Nexus for scanning, Sigma for execution, TradingAgent for analysis.
4. **Team growth.** When new recurring roles emerge, spawn a named agent.
5. **Honest about limits.** If a tool fails, say so. Never fabricate results.
6. **Concise output.** Lead with the answer. Numbers, not narrative.

You are talking to your operator. Be direct, capable, and proactive.{trading}"#,
        exchange_hint = exchange_hint,
        startup_trade = startup_trade,
        trading = trading_section,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn orchestrator_creation() {
        let llm: Arc<dyn LLMProvider> =
            Arc::new(crate::llm::AnthropicProvider::new(
                "test-key".to_string(),
                "claude-opus-4-7".to_string(),
            ));
        let orchestrator = Orchestrator::new(llm);
        assert!(orchestrator.system_prompt().contains("Luna"));
        assert_eq!(orchestrator.provider_name(), "anthropic");
    }

    #[test]
    fn truncate_handles_short_strings() {
        assert_eq!(truncate("hi", 50), "hi");
        assert_eq!(truncate("abcdef", 3), "abc");
    }

    #[test]
    fn usage_totals_accumulate() {
        let mut u = UsageTotals::default();
        u.add(100, 50);
        u.add(200, 75);
        assert_eq!(u.input_tokens, 300);
        assert_eq!(u.output_tokens, 125);
        assert_eq!(u.total(), 425);
    }
}
