use crate::{
    claude::ClaudeClient,
    models::{UserRequest, AgentResponse},
    agents::Agent,
    Result,
};
use std::sync::Arc;
use tracing::{info, error};
use serde_json::json;

pub struct Orchestrator {
    claude: Arc<ClaudeClient>,
}

impl Orchestrator {
    pub fn new(claude: Arc<ClaudeClient>) -> Self {
        Self { claude }
    }

    pub async fn process(&self, request: UserRequest) -> Result<String> {
        info!("Processing request: {}", request.session_id);

        let messages = vec![
            json!({
                "role": "user",
                "content": request.content
            })
        ];

        match self.claude.chat(messages).await {
            Ok(response) => {
                info!("Request processed successfully");
                Ok(response)
            }
            Err(e) => {
                error!("Failed to process request: {}", e);
                Err(e)
            }
        }
    }
}
