//! Writing specialist agent — crafts documentation, content, and clear communication.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::agents::Agent;
use crate::llm::LLMProvider;
use crate::models::{ExecutionContext, Message, Task};
use crate::Result;

/// Writing styles available to the WritingAgent.
#[derive(Debug, Clone, Copy)]
pub enum WritingStyle {
    Technical,
    Conversational,
    Marketing,
    Academic,
    Concise,
}

impl WritingStyle {
    pub fn describe(self) -> &'static str {
        match self {
            WritingStyle::Technical => "precise, structured, jargon-aware technical writing",
            WritingStyle::Conversational => "warm, plain-spoken, second-person prose",
            WritingStyle::Marketing => "engaging, benefit-driven copy with clear calls to action",
            WritingStyle::Academic => "formal, citation-aware, rigorous prose",
            WritingStyle::Concise => "tight, low-fat prose with no padding",
        }
    }
}

/// Writing specialist — drafts, edits, and improves prose, docs, and copy.
pub struct WritingAgent {
    name: String,
    role: String,
    llm: Arc<dyn LLMProvider>,
    default_style: WritingStyle,
}

impl WritingAgent {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            name: "WritingAgent".to_string(),
            role: "Writing Specialist".to_string(),
            llm,
            default_style: WritingStyle::Concise,
        }
    }

    pub fn with_style(mut self, style: WritingStyle) -> Self {
        self.default_style = style;
        self
    }

    /// Quick heuristic quality check on a piece of generated content.
    /// Returns a list of warnings; an empty list means "no obvious issues".
    pub fn quality_check(content: &str) -> Vec<String> {
        let mut warnings = Vec::new();

        if content.trim().is_empty() {
            warnings.push("Content is empty.".to_string());
            return warnings;
        }
        if content.len() < 30 {
            warnings.push("Content is suspiciously short.".to_string());
        }
        let unmatched_fences = content.matches("```").count() % 2;
        if unmatched_fences != 0 {
            warnings.push("Unmatched code fence (```) detected.".to_string());
        }
        let open_brackets = content.matches('[').count();
        let close_brackets = content.matches(']').count();
        if open_brackets != close_brackets {
            warnings.push("Unbalanced markdown link brackets.".to_string());
        }
        if content.contains("TODO") || content.contains("FIXME") {
            warnings.push("Contains TODO/FIXME placeholders.".to_string());
        }

        warnings
    }
}

#[async_trait]
impl Agent for WritingAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn role(&self) -> &str {
        &self.role
    }

    fn system_prompt(&self) -> String {
        format!(
            r#"You are an expert writing specialist. Your responsibilities:

- Produce clear, well-structured documentation, content, and communication.
- Edit and improve existing prose for clarity, flow, and tone.
- Preserve formatting (markdown, code fences, lists, tables) when refining content.
- Match the requested style and audience.

Default style: {style}.
Adapt tone and depth to the audience. Prefer plain language over jargon. Cut filler.
When editing, return the improved text only — no meta-commentary unless explicitly asked."#,
            style = self.default_style.describe(),
        )
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        info!(agent = %self.name, task_id = %task.id, "Executing writing task");
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
            "write", "draft", "edit", "rewrite", "rephrase", "proofread",
            "documentation", "docs", "readme", "blog", "article", "essay",
            "summary", "summarize", "email", "letter", "post", "copy",
            "tone", "style", "narrative", "story", "content",
        ];
        KEYWORDS.iter().any(|k| lower.contains(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_check_flags_empty() {
        let w = WritingAgent::quality_check("   ");
        assert!(w.iter().any(|s| s.contains("empty")));
    }

    #[test]
    fn quality_check_flags_unmatched_fence() {
        let w = WritingAgent::quality_check("Here is code: ```rust\nfn x() {}");
        assert!(w.iter().any(|s| s.contains("Unmatched code fence")));
    }

    #[test]
    fn quality_check_passes_clean_content() {
        let content = "This is a perfectly reasonable paragraph with enough words to clear the threshold.";
        assert!(WritingAgent::quality_check(content).is_empty());
    }
}
