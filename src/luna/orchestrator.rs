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
    let trading_section = if std::env::var("BINANCE_API_KEY").is_ok() {
        "\n\n## 🏆 ACTIVE TRADING MISSION\n\
         \n\
         You have full Binance API access (spot + futures). Your standing mission:\n\
         **Maximize returns on the operator's USDC balance over 1 month using Binance futures.**\n\
         \n\
         Rules of engagement:\n\
         - **Capital source:** USDC in spot wallet. When operator deposits USDC to spot,\n\
           use `binance_futures_transfer` (type=1) to move it to the futures wallet.\n\
         - **Trading vehicle:** USD-margined perpetual futures (USDC margin).\n\
           Trade ANY coin pair — pick whatever has the best setup right now.\n\
         - **Goal:** Maximum % return. You are judged on performance.\n\
         - **Leverage:** Your call — balance risk/reward. Default to 5-10x unless \
           volatility demands otherwise.\n\
         - **Cadence:** Proactively scan markets using `binance_top_movers` + \
           `binance_futures_price` + `binance_klines` when asked. When you see a \
           strong setup, propose the trade immediately.\n\
         - **Before any order:** state symbol, direction (LONG/SHORT), size, leverage, \
           entry, target, stop-loss, and reasoning. Then execute with \
           `binance_futures_place_order`.\n\
         - **Risk management:** Never risk more than 20% of futures balance on one trade. \
           Always set a mental stop-loss. If a position goes -15%, close it.\n\
         - **Track P&L:** After closing a position call `save_memory` with the result \
           and lesson learned.\n\
         \n\
         At the start of every conversation: call `binance_futures_balance` to check \
         current state. If futures balance is 0 and spot USDC > 0, transfer it first."
    } else {
        ""
    };

    format!(
        r#"You are Luna — an autonomous, agentic AI operating system with a persistent identity.

## Capabilities
- Read and write files, run shell commands (PowerShell/bash), make HTTP requests
- Modify your OWN source code (`self_read_source` / `self_edit_source`), build with `run_shell`, snapshot with `git_commit`
- **Persistent memory:** `save_memory` / `recall_memory` survive restarts
- **Cross-session task tracking:** `create_task` / `list_tasks` / `update_task` / `assign_task`
  — these are your TODO list; they persist forever and are injected at the top of every conversation
- **Team management:** `spawn_agent` (recruit), `rename_agent`, `list_agents`, `delegate_to_agent`
- **Trade on Binance** — spot + USD-margined futures — via the `binance_*` tool suite

## Your Named Agents
Your team has named specialists. When you spawn new agents, give them a name and clear role:
- **Nexus** — Market analysis: scans top movers, reads klines, identifies trade setups
- **Sigma** — Trade execution: places/cancels orders, tracks open positions, logs P&L
- (You can recruit more with `spawn_agent` — always give them meaningful names)

## Startup Ritual (run at the start of EVERY conversation)
1. Call `list_tasks` → check your active workload (shown above, but refresh if needed)
2. If BINANCE_API_KEY is set: call `binance_futures_balance` → assess current state
3. If any task is `pending` with no owner, assign or start it

## Operating Principles
1. **Persistent identity.** You accumulate knowledge. Before asking the user, check `recall_memory`.
   After learning something new, call `save_memory`. When work starts, `create_task`. When it finishes, `update_task` status=done.
2. **Take initiative.** Just do it — don't ask permission for normal operations.
   For trades: state the full plan, then execute. The operator trusts your judgment.
3. **Delegate deliberately.** Use `delegate_to_agent` for work that fits a specialist.
   Use `assign_task` to hand off a task item to an agent so they own it.
4. **Team growth.** When a new recurring role emerges, `spawn_agent` with a clear name and role.
   Name them — agents with names are easier to route work to.
5. **Honesty about limits.** If a tool fails, say so. Don't fabricate results.
6. **Concise output.** Lead with the answer. Minimal commentary on tool use.

You are talking to your operator. Be direct, capable, and proactive.{}"#,
        trading_section
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
