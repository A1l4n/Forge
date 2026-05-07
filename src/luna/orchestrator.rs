//! Luna Orchestrator — coordinates specialist agents to fulfill user requests.

use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::agents::Agent;
use crate::claude::client::ClaudeClient;
use crate::memory::MemoryStore;
use crate::models::{
    AgentActivity, ExecutionContext, Message, Task, TaskStatus, ToolInput, UserRequest,
};
use crate::tools::ToolRegistry;
use crate::{Error, Result};

const DEFAULT_USER_ID: &str = "anonymous";
const TOOL_LOOP_LIMIT: usize = 5;

/// Result of a multi-agent run, useful for surfacing per-agent activity in UIs.
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub response: String,
    pub activities: Vec<AgentActivity>,
}

/// Luna — the main orchestrator.
pub struct Orchestrator {
    claude: Arc<ClaudeClient>,
    system_prompt: String,
    max_iterations: usize,
}

impl Orchestrator {
    pub fn new(claude: Arc<ClaudeClient>) -> Self {
        let system_prompt = r#"You are Luna, an AI orchestrator that manages a team of specialist agents.

Your role:
1. Understand the user's request.
2. Break it down into focused subtasks.
3. Delegate to the right specialist (CodeAgent, ResearchAgent, WritingAgent, PlanningAgent).
4. Coordinate their work — run independent steps in parallel, sequence dependencies.
5. Synthesize results into a single, clear answer for the user.

Available specialists:
- CodeAgent — code, debugging, technical implementation.
- ResearchAgent — research, analysis, summarization.
- WritingAgent — documentation, content, editing.
- PlanningAgent — task breakdown, workflow orchestration.

Think systematically. Be concise. When you have enough information, give a complete answer."#
            .to_string();

        Self {
            claude,
            system_prompt,
            max_iterations: 5,
        }
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub fn create_task(
        &self,
        session_id: String,
        agent_name: String,
        description: String,
    ) -> Task {
        Task::new(session_id, agent_name, description)
    }

    /// Simple agentic loop without specialist agents — kept for backwards compatibility.
    pub async fn process(&self, request: UserRequest) -> Result<String> {
        info!(
            session_id = %request.session_id,
            "Luna processing request: {}",
            truncate(&request.content, 50)
        );

        let mut messages = vec![Message::user(
            request.session_id.clone(),
            request.content.clone(),
        )];

        for iteration in 0..self.max_iterations {
            debug!(iteration, "Orchestrator iteration");

            let response = self
                .claude
                .message_from_history(self.system_prompt.clone(), messages.clone(), None)
                .await?;

            let text = ClaudeClient::extract_text(&response);
            messages.push(Message::luna(request.session_id.clone(), text.clone()));

            if response.stop_reason == "end_turn" {
                return Ok(text);
            }

            if ClaudeClient::has_tool_calls(&response) {
                debug!("Luna made tool calls (no registry attached, breaking)");
                return Ok(text);
            }

            if iteration == self.max_iterations - 1 {
                return Ok(text);
            }
        }

        Ok("Orchestration completed".to_string())
    }

    /// Full agentic flow:
    /// 1. Persist the user message.
    /// 2. Ask Claude to produce a structured plan referencing available specialists.
    /// 3. Dispatch each step to its specialist agent in parallel (when independent).
    /// 4. Persist all task results.
    /// 5. Ask Claude to synthesize a final response.
    /// 6. Persist Luna's response and return it along with per-agent activity.
    pub async fn process_with_agents(
        &self,
        request: UserRequest,
        agents: &[Arc<dyn Agent>],
        memory: &MemoryStore,
    ) -> Result<OrchestrationResult> {
        info!(
            session_id = %request.session_id,
            "Luna routing request through {} specialists",
            agents.len()
        );

        memory
            .ensure_session(&request.session_id, DEFAULT_USER_ID)
            .await?;

        let user_msg = Message::user(request.session_id.clone(), request.content.clone());
        memory.save_message(user_msg).await?;

        // 1. Plan.
        let plan = self.plan(&request, agents).await?;
        debug!(steps = plan.steps.len(), "Planned");

        if plan.steps.is_empty() {
            // No specialists needed — answer directly.
            let response = self
                .direct_response(&request)
                .await?;
            memory
                .save_message(Message::luna(request.session_id.clone(), response.clone()))
                .await?;
            return Ok(OrchestrationResult {
                response,
                activities: vec![],
            });
        }

        // 2. Dispatch to specialists in parallel.
        let context = ExecutionContext {
            session_id: request.session_id.clone(),
            user_id: DEFAULT_USER_ID.to_string(),
            task_id: String::new(),
            memory: Default::default(),
            available_tools: vec![],
        };

        let mut handles = Vec::new();
        for step in &plan.steps {
            let agent = match find_agent(agents, &step.agent) {
                Some(a) => a,
                None => {
                    warn!(agent = %step.agent, "Unknown specialist in plan, skipping");
                    continue;
                }
            };

            let task = Task::new(
                request.session_id.clone(),
                agent.name().to_string(),
                step.description.clone(),
            )
            .start();

            memory.save_task(task.clone()).await?;

            let agent_clone = agent.clone();
            let task_clone = task.clone();
            let mut ctx = context.clone();
            ctx.task_id = task.id.clone();

            let memory_clone = memory.clone();
            handles.push(tokio::spawn(async move {
                let started = std::time::Instant::now();
                let result = agent_clone.execute(task_clone.clone(), &ctx).await;
                let duration_ms = started.elapsed().as_millis() as u64;

                let (final_task, activity_status, activity_result) = match &result {
                    Ok(out) => (
                        task_clone.clone().complete(out.clone()),
                        TaskStatus::Completed,
                        Some(out.clone()),
                    ),
                    Err(e) => (
                        task_clone.clone().fail(e.to_string()),
                        TaskStatus::Failed,
                        Some(e.to_string()),
                    ),
                };

                let _ = memory_clone.update_task(final_task.clone()).await;

                AgentActivity {
                    agent_name: agent_clone.name().to_string(),
                    task_id: task_clone.id,
                    status: activity_status.to_string(),
                    result: activity_result,
                    duration_ms,
                }
            }));
        }

        let mut activities = Vec::new();
        for h in handles {
            if let Ok(act) = h.await {
                activities.push(act);
            }
        }

        // 3. Synthesize.
        let synthesis = self.synthesize(&request, &activities).await?;

        memory
            .save_message(Message::luna(request.session_id.clone(), synthesis.clone()))
            .await?;

        Ok(OrchestrationResult {
            response: synthesis,
            activities,
        })
    }

    /// Run a single task through a specific agent, allowing it to call tools.
    pub async fn execute_with_tools(
        &self,
        task: Task,
        tools: &ToolRegistry,
    ) -> Result<String> {
        let mut messages = vec![Message::user(task.session_id.clone(), task.description.clone())];
        let claude_tools = tools.claude_tools();
        let claude_tools_opt = if claude_tools.is_empty() {
            None
        } else {
            Some(claude_tools)
        };

        for iteration in 0..TOOL_LOOP_LIMIT {
            debug!(iteration, "Tool-execution iteration");

            let response = self
                .claude
                .message_from_history(
                    self.system_prompt.clone(),
                    messages.clone(),
                    claude_tools_opt.clone(),
                )
                .await?;

            if !ClaudeClient::has_tool_calls(&response) {
                let text = ClaudeClient::extract_text(&response);
                return Ok(text);
            }

            // Persist Luna's intermediate (tool-calling) message so the next
            // turn can include the corresponding tool_result blocks.
            let intermediate = ClaudeClient::extract_text(&response);
            messages.push(Message::luna(task.session_id.clone(), intermediate));

            let calls = ClaudeClient::extract_tool_calls(&response);
            let mut tool_outputs: Vec<Value> = Vec::new();
            for (id, name, input) in calls {
                let result = tools
                    .execute(&name, id.clone(), ToolInput::from_value(input))
                    .await?;
                tool_outputs.push(result.to_claude_format());
            }

            // Feed tool results back as a user message so Claude can react.
            let tool_msg = Message::user(
                task.session_id.clone(),
                serde_json::to_string(&json!({ "tool_results": tool_outputs }))?,
            );
            messages.push(tool_msg);
        }

        Err(Error::Orchestration(
            "Tool-execution loop exceeded maximum iterations".to_string(),
        ))
    }

    // ---- internal helpers ----

    async fn direct_response(&self, request: &UserRequest) -> Result<String> {
        let messages = vec![Message::user(request.session_id.clone(), request.content.clone())];
        let response = self
            .claude
            .message_from_history(self.system_prompt.clone(), messages, None)
            .await?;
        Ok(ClaudeClient::extract_text(&response))
    }

    async fn plan(&self, request: &UserRequest, agents: &[Arc<dyn Agent>]) -> Result<Plan> {
        let agent_list = agents
            .iter()
            .map(|a| format!("- {} ({})", a.name(), a.role()))
            .collect::<Vec<_>>()
            .join("\n");

        let planner_prompt = format!(
            r#"You are Luna's planner. Decompose the user request into 0–4 focused subtasks
that can each be handled by ONE specialist. Independent subtasks will be run in parallel.

Available specialists:
{agents}

Respond with a JSON object ONLY (no prose, no fences) of the form:
{{
  "steps": [
    {{ "agent": "<AgentName>", "description": "<one short imperative sentence>" }}
  ]
}}

If the request is small-talk or trivially answerable without specialists, return {{"steps": []}}."#,
            agents = agent_list
        );

        let messages = vec![Message::user(request.session_id.clone(), request.content.clone())];
        let response = self
            .claude
            .message_from_history(planner_prompt, messages, None)
            .await?;
        let raw = ClaudeClient::extract_text(&response);
        let plan_json = extract_json_object(&raw).unwrap_or_else(|| raw.clone());

        match serde_json::from_str::<Plan>(&plan_json) {
            Ok(p) => Ok(p),
            Err(e) => {
                warn!(error = %e, raw = %raw, "Plan parse failed, falling back to empty plan");
                Ok(Plan { steps: Vec::new() })
            }
        }
    }

    async fn synthesize(
        &self,
        request: &UserRequest,
        activities: &[AgentActivity],
    ) -> Result<String> {
        let mut transcript = String::new();
        transcript.push_str("User request:\n");
        transcript.push_str(&request.content);
        transcript.push_str("\n\nSpecialist results:\n");
        for a in activities {
            transcript.push_str(&format!(
                "\n[{}] status={} duration={}ms\n{}\n",
                a.agent_name,
                a.status,
                a.duration_ms,
                a.result.clone().unwrap_or_default()
            ));
        }

        let synth_prompt = r#"You are Luna. Several specialist agents have produced output for the user.
Combine their outputs into a single, clear, friendly response addressed to the user.
- Lead with the answer.
- Preserve any code blocks verbatim.
- Resolve conflicts between specialists by trusting the most relevant one.
- Do NOT mention "agents" or internal coordination — just give the answer."#
            .to_string();

        let messages = vec![Message::user(request.session_id.clone(), transcript)];
        let response = self
            .claude
            .message_from_history(synth_prompt, messages, None)
            .await?;
        Ok(ClaudeClient::extract_text(&response))
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

fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0i32;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, serde::Deserialize)]
struct Plan {
    steps: Vec<PlanStepJson>,
}

#[derive(Debug, serde::Deserialize)]
struct PlanStepJson {
    agent: String,
    description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn orchestrator_creation() {
        let claude = Arc::new(ClaudeClient::new(
            "test-key".to_string(),
            "claude-opus-4-7".to_string(),
        ));
        let orchestrator = Orchestrator::new(claude);
        assert!(orchestrator.system_prompt().contains("Luna"));
    }

    #[test]
    fn extract_json_object_pulls_out_inner_json() {
        let s = r#"Here is the plan: {"steps":[{"agent":"X","description":"do"}]} and that's it."#;
        let out = extract_json_object(s).unwrap();
        assert!(out.contains("steps"));
        let plan: Plan = serde_json::from_str(&out).unwrap();
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn truncate_handles_short_strings() {
        assert_eq!(truncate("hi", 50), "hi");
        assert_eq!(truncate("abcdef", 3), "abc");
    }
}
