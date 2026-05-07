//! Tool registry and built-in executors.
//!
//! The registry holds tool definitions (for advertising to Claude) plus an
//! executor closure per tool. `execute()` dispatches a `ToolInput` to the
//! matching executor and returns a `ToolResult`.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::errors::{Error, Result};
use crate::models::{Tool, ToolInput, ToolResult};

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

/// Registry of available tools and their executors.
pub struct ToolRegistry {
    tools: HashMap<String, Tool>,
    executors: HashMap<String, Arc<dyn ToolExecutor>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            executors: HashMap::new(),
        }
    }

    /// Register a tool definition only (no executor — useful for inspection).
    pub fn register(&mut self, tool: Tool) {
        self.tools.insert(tool.name.clone(), tool);
    }

    /// Register a tool with its executor.
    pub fn register_with(&mut self, tool: Tool, executor: Arc<dyn ToolExecutor>) {
        self.executors.insert(tool.name.clone(), executor);
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Tool> {
        self.tools.get(name)
    }

    pub fn all(&self) -> Vec<&Tool> {
        self.tools.values().collect()
    }

    pub fn exists(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Convert all registered tools to the JSON array Claude expects.
    pub fn claude_tools(&self) -> Vec<Value> {
        self.tools.values().map(|t| t.to_claude_format()).collect()
    }

    /// Execute a registered tool by name.
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

        info!(tool = tool_name, "Executing tool");
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

    /// Build a registry pre-loaded with built-in tools (web_search, read_file,
    /// write_file, execute_code, http_request).
    pub fn with_built_in_tools() -> Self {
        let mut r = Self::new();

        r.register_with(
            Tool::new(
                "web_search".to_string(),
                "Search the web for a query and return summarized results.".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query." }
                    },
                    "required": ["query"]
                }),
            ),
            Arc::new(FnExecutor(builtin::web_search)),
        );

        r.register_with(
            Tool::new(
                "read_file".to_string(),
                "Read a file from disk and return its contents as UTF-8 text.".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative file path." }
                    },
                    "required": ["path"]
                }),
            ),
            Arc::new(FnExecutor(builtin::read_file)),
        );

        r.register_with(
            Tool::new(
                "write_file".to_string(),
                "Write UTF-8 text to a file (creates or overwrites).".to_string(),
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
        );

        r.register_with(
            Tool::new(
                "execute_code".to_string(),
                "Execute a short snippet of code in a supported language (python, bash, node) and return stdout/stderr.".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "language": { "type": "string", "enum": ["python", "bash", "node"] },
                        "code": { "type": "string" }
                    },
                    "required": ["language", "code"]
                }),
            ),
            Arc::new(FnExecutor(builtin::execute_code)),
        );

        r.register_with(
            Tool::new(
                "http_request".to_string(),
                "Send an HTTP request and return status, headers, and body.".to_string(),
                json!({
                    "type": "object",
                    "properties": {
                        "method": { "type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"] },
                        "url": { "type": "string" },
                        "body": { "type": "string" },
                        "headers": { "type": "object" }
                    },
                    "required": ["method", "url"]
                }),
            ),
            Arc::new(FnExecutor(builtin::http_request)),
        );

        r
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

mod builtin {
    use super::*;

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

    pub async fn read_file(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| Error::ToolExecution(format!("read {}: {}", path, e)))?;
        Ok(json!({ "path": path, "content": content }))
    }

    pub async fn write_file(input: ToolInput) -> Result<Value> {
        let path = input
            .get_string("path")
            .ok_or_else(|| Error::ToolExecution("missing 'path'".into()))?;
        let content = input
            .get_string("content")
            .ok_or_else(|| Error::ToolExecution("missing 'content'".into()))?;
        if let Some(parent) = std::path::Path::new(&path).parent() {
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
                let prog = "cmd";
                #[cfg(not(target_os = "windows"))]
                let prog = "bash";

                #[cfg(target_os = "windows")]
                let args = vec!["/C".to_string(), code.clone()];
                #[cfg(not(target_os = "windows"))]
                let args = vec!["-c".to_string(), code.clone()];

                (prog, args)
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
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
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
            "body": text,
        }))
    }

    fn urlencode(s: &str) -> String {
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
            .execute(
                "echo",
                "call-1",
                ToolInput::from_value(json!({"msg":"hi"})),
            )
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
    fn built_ins_register_all_five() {
        let reg = ToolRegistry::with_built_in_tools();
        for name in ["web_search", "read_file", "write_file", "execute_code", "http_request"] {
            assert!(reg.exists(name), "missing built-in: {}", name);
        }
    }
}
