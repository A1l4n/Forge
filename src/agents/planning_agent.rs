//! Planning specialist agent — breaks down complex tasks and orchestrates workflows.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::agents::Agent;
use crate::claude::ClaudeClient;
use crate::models::{ExecutionContext, Message, Task, TaskStatus};
use crate::Result;

/// A single step in a planned workflow.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    pub agent_hint: Option<String>,
    pub dependencies: Vec<usize>,
}

impl PlanStep {
    pub fn new(id: usize, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            agent_hint: None,
            dependencies: Vec::new(),
        }
    }

    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent_hint = Some(agent.into());
        self
    }

    pub fn depends_on(mut self, deps: Vec<usize>) -> Self {
        self.dependencies = deps;
        self
    }
}

/// Planning specialist — decomposes goals, tracks dependencies, monitors progress.
pub struct PlanningAgent {
    name: String,
    role: String,
    claude: Arc<ClaudeClient>,
}

impl PlanningAgent {
    pub fn new(claude: Arc<ClaudeClient>) -> Self {
        Self {
            name: "PlanningAgent".to_string(),
            role: "Planning Specialist".to_string(),
            claude,
        }
    }

    /// Decompose a free-form goal into ordered plan steps using simple heuristics.
    /// For richer plans, call `execute` and parse the model's structured output.
    pub fn decompose(goal: &str) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        let mut id = 0;
        for line in goal.lines() {
            let trimmed = line.trim().trim_start_matches(['-', '*', '•']).trim();
            if trimmed.is_empty() {
                continue;
            }
            id += 1;
            steps.push(PlanStep::new(id, trimmed));
        }
        if steps.is_empty() && !goal.trim().is_empty() {
            steps.push(PlanStep::new(1, goal.trim()));
        }
        steps
    }

    /// Compute completion percentage given a slice of tasks.
    pub fn progress(tasks: &[Task]) -> f32 {
        if tasks.is_empty() {
            return 0.0;
        }
        let done = tasks
            .iter()
            .filter(|t| matches!(t.status, TaskStatus::Completed | TaskStatus::Skipped))
            .count() as f32;
        (done / tasks.len() as f32) * 100.0
    }

    /// Topologically order steps so dependencies come first. Cycles are reported as an error.
    pub fn topo_sort(steps: &[PlanStep]) -> Result<Vec<PlanStep>> {
        use std::collections::{HashMap, HashSet};
        let mut by_id: HashMap<usize, &PlanStep> = HashMap::new();
        for s in steps {
            by_id.insert(s.id, s);
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut visiting: HashSet<usize> = HashSet::new();
        let mut order: Vec<PlanStep> = Vec::new();

        fn visit(
            id: usize,
            by_id: &std::collections::HashMap<usize, &PlanStep>,
            visited: &mut std::collections::HashSet<usize>,
            visiting: &mut std::collections::HashSet<usize>,
            order: &mut Vec<PlanStep>,
        ) -> crate::Result<()> {
            if visited.contains(&id) {
                return Ok(());
            }
            if !visiting.insert(id) {
                return Err(crate::Error::Orchestration(format!(
                    "Cycle detected in plan at step {}",
                    id
                )));
            }
            if let Some(step) = by_id.get(&id) {
                for dep in &step.dependencies {
                    visit(*dep, by_id, visited, visiting, order)?;
                }
                order.push((*step).clone());
            }
            visiting.remove(&id);
            visited.insert(id);
            Ok(())
        }

        for step in steps {
            visit(step.id, &by_id, &mut visited, &mut visiting, &mut order)?;
        }
        Ok(order)
    }
}

#[async_trait]
impl Agent for PlanningAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn role(&self) -> &str {
        &self.role
    }

    fn system_prompt(&self) -> String {
        r#"You are an expert planning specialist. Your responsibilities:

- Break complex goals into a clear, minimal sequence of executable steps.
- Identify dependencies between steps and surface critical paths.
- Suggest the right specialist agent for each step (CodeAgent, ResearchAgent, WritingAgent).
- Track progress and flag stalled or blocked work.

Output format when planning:
1. **Goal** — one-sentence restatement.
2. **Steps** — numbered list. Each: short imperative, agent hint in [brackets], dependencies if any.
3. **Risks** — what could derail the plan.

Be ruthless about scope. The smallest plan that achieves the goal is the best plan."#
            .to_string()
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        info!(agent = %self.name, task_id = %task.id, "Executing planning task");
        debug!(task_description = %task.description);

        let messages = vec![Message::user(
            context.session_id.clone(),
            task.description.clone(),
        )];

        let response = self
            .claude
            .message_from_history(self.system_prompt(), messages, None)
            .await?;

        Ok(ClaudeClient::extract_text(&response))
    }

    fn can_handle(&self, task_description: &str) -> bool {
        let lower = task_description.to_lowercase();
        const KEYWORDS: &[&str] = &[
            "plan", "break down", "breakdown", "decompose", "roadmap",
            "milestones", "schedule", "workflow", "orchestrate", "sequence",
            "dependencies", "steps", "outline", "strategy", "approach",
        ];
        KEYWORDS.iter().any(|k| lower.contains(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_splits_bullet_list() {
        let goal = "- Set up repo\n- Add tests\n- Deploy";
        let steps = PlanningAgent::decompose(goal);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].description, "Set up repo");
    }

    #[test]
    fn decompose_falls_back_to_single_step() {
        let steps = PlanningAgent::decompose("ship the thing");
        assert_eq!(steps.len(), 1);
    }

    #[test]
    fn progress_computes_percentage() {
        let mut a = Task::new("s".into(), "X".into(), "do a".into());
        let mut b = Task::new("s".into(), "X".into(), "do b".into());
        a.status = TaskStatus::Completed;
        b.status = TaskStatus::Pending;
        let pct = PlanningAgent::progress(&[a, b]);
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn topo_sort_orders_dependencies_first() {
        let s1 = PlanStep::new(1, "compile").depends_on(vec![2]);
        let s2 = PlanStep::new(2, "write code");
        let order = PlanningAgent::topo_sort(&[s1, s2]).unwrap();
        assert_eq!(order[0].id, 2);
        assert_eq!(order[1].id, 1);
    }

    #[test]
    fn topo_sort_detects_cycles() {
        let s1 = PlanStep::new(1, "a").depends_on(vec![2]);
        let s2 = PlanStep::new(2, "b").depends_on(vec![1]);
        assert!(PlanningAgent::topo_sort(&[s1, s2]).is_err());
    }
}
