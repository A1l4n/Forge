//! Research specialist agent — finds, analyzes, and synthesizes information.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::agents::Agent;
use crate::llm::LLMProvider;
use crate::models::{ExecutionContext, Message, Task};
use crate::Result;

/// A piece of evidence collected during research, with optional source.
#[derive(Debug, Clone)]
pub struct Finding {
    pub claim: String,
    pub source: Option<String>,
}

impl Finding {
    pub fn new(claim: impl Into<String>) -> Self {
        Self {
            claim: claim.into(),
            source: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Research specialist — searches, analyzes, synthesizes, and summarizes.
pub struct ResearchAgent {
    name: String,
    role: String,
    llm: Arc<dyn LLMProvider>,
}

impl ResearchAgent {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            name: "ResearchAgent".to_string(),
            role: "Research Specialist".to_string(),
            llm,
        }
    }

    /// Format a list of findings as a readable bulleted report.
    pub fn format_findings(findings: &[Finding]) -> String {
        if findings.is_empty() {
            return "No findings.".to_string();
        }
        findings
            .iter()
            .enumerate()
            .map(|(i, f)| match &f.source {
                Some(src) => format!("{}. {}\n   Source: {}", i + 1, f.claim, src),
                None => format!("{}. {}", i + 1, f.claim),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Agent for ResearchAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn role(&self) -> &str {
        &self.role
    }

    fn system_prompt(&self) -> String {
        r#"You are an expert research specialist. Your responsibilities:

- Find and analyze information on any topic with precision.
- Synthesize evidence into clear, structured conclusions.
- Distinguish between confirmed facts, well-supported claims, and speculation.
- Cite sources when known; be explicit about uncertainty when not.
- Summarize complex material without losing nuance.

Structure your output:
1. **Key findings** — bulleted, each with confidence level if relevant.
2. **Sources / evidence** — referenced inline or listed at the end.
3. **Open questions** — what remains uncertain or worth investigating further.

Be rigorous, balanced, and honest about the limits of your knowledge."#
            .to_string()
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        info!(agent = %self.name, task_id = %task.id, "Executing research task");
        debug!(task_description = %task.description);

        let messages = vec![Message::user(
            context.session_id.clone(),
            task.description.clone(),
        )];

        let response = self
            .llm
            .generate(&self.system_prompt(), &messages, None)
            .await?;

        Ok(response.text)
    }

    fn can_handle(&self, task_description: &str) -> bool {
        let lower = task_description.to_lowercase();
        const KEYWORDS: &[&str] = &[
            "research", "find", "search", "analyze", "summarize", "summary",
            "investigate", "explain", "compare", "evaluate", "what is",
            "what are", "who is", "history of", "background", "trends",
            "data", "statistics", "study", "report", "evidence", "source",
        ];
        KEYWORDS.iter().any(|k| lower.contains(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_findings_handles_empty() {
        assert_eq!(ResearchAgent::format_findings(&[]), "No findings.");
    }

    #[test]
    fn format_findings_includes_sources() {
        let findings = vec![
            Finding::new("Rust has zero-cost abstractions").with_source("rust-lang.org"),
            Finding::new("Ownership prevents data races"),
        ];
        let out = ResearchAgent::format_findings(&findings);
        assert!(out.contains("rust-lang.org"));
        assert!(out.contains("Ownership"));
    }
}
