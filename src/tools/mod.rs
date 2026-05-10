use std::collections::HashMap;
use async_trait::async_trait;
use crate::Result;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn execute(&self, args: HashMap<String, String>) -> Result<String>;
}

pub use registry::{PermissionMode, Tier, ToolRegistry};
