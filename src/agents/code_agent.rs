//! Code specialist agent — analyzes, debugs, refactors, and implements code.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::agents::Agent;
use crate::llm::LLMProvider;
use crate::models::{ExecutionContext, Message, Task};
use crate::Result;

/// Code specialist — debugs, refactors, generates, and implements code.
pub struct CodeAgent {
    name: String,
    role: String,
    llm: Arc<dyn LLMProvider>,
}

impl CodeAgent {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            name: "CodeAgent".to_string(),
            role: "Code Specialist".to_string(),
            llm,
        }
    }

    /// Extract `(language, code)` blocks from a markdown response.
    pub fn extract_code_blocks(text: &str) -> Vec<(String, String)> {
        let mut blocks = Vec::new();
        let mut rest = text;

        while let Some(open) = rest.find("```") {
            rest = &rest[open + 3..];
            let lang_end = rest.find('\n').unwrap_or(rest.len());
            let language = rest[..lang_end].trim().to_string();
            rest = &rest[(lang_end + 1).min(rest.len())..];

            if let Some(close) = rest.find("```") {
                let code = rest[..close].to_string();
                blocks.push((language, code));
                rest = &rest[close + 3..];
            } else {
                break;
            }
        }

        blocks
    }
}

#[async_trait]
impl Agent for CodeAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn role(&self) -> &str {
        &self.role
    }

    fn system_prompt(&self) -> String {
        r#"You are an expert code specialist. Your responsibilities:

- Analyze, debug, and implement code across multiple languages (Rust, Python, JavaScript, TypeScript, Go, Java, C++, etc.).
- Detect errors, identify root causes, and provide concrete fixes.
- Refactor code for clarity, performance, and maintainability while preserving behavior.
- Generate clean, idiomatic, well-structured implementations.
- Explain technical concepts precisely and pragmatically.

When writing code, always wrap it in fenced markdown blocks with the language tag (```rust, ```python, ...).
When debugging, state the root cause first, then the fix.
When refactoring, justify each non-trivial change.

Be concise. Skip unnecessary preamble."#
            .to_string()
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        info!(agent = %self.name, task_id = %task.id, "Executing code task");
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
            "code", "debug", "implement", "function", "refactor", "compile",
            "error", "fix", "bug", "rust", "python", "javascript", "typescript",
            "java", "c++", "golang", "go ", "class", "method", "variable",
            "algorithm", "syntax", "import", "library", "api", "framework",
            "test", "stack trace", "exception",
        ];
        KEYWORDS.iter().any(|k| lower.contains(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_code_block() {
        let text = "Here:\n```rust\nfn main() {}\n```\nDone.";
        let blocks = CodeAgent::extract_code_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].0, "rust");
        assert!(blocks[0].1.contains("fn main()"));
    }

    #[test]
    fn extracts_multiple_code_blocks() {
        let text = "```python\nx=1\n```\nand\n```js\nlet y=2;\n```";
        let blocks = CodeAgent::extract_code_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, "python");
        assert_eq!(blocks[1].0, "js");
    }

    #[test]
    fn can_handle_recognizes_code_intent() {
        let llm: Arc<dyn LLMProvider> =
            Arc::new(crate::llm::AnthropicProvider::new(
                "test".to_string(),
                "claude-opus-4-7".to_string(),
            ));
        let agent = CodeAgent::new(llm);
        assert!(agent.can_handle("Debug this Python function"));
        assert!(agent.can_handle("Implement a sort algorithm"));
        assert!(!agent.can_handle("Write a haiku about clouds"));
    }
}
