//! Message router for gateway

use crate::{models::UserRequest, Result};

/// Routes user messages to Luna and other interfaces
pub struct MessageRouter;

impl MessageRouter {
    pub fn new() -> Self {
        Self
    }

    pub async fn route(&self, _request: UserRequest) -> Result<String> {
        // TODO: Implement routing logic
        Ok("Message routed".to_string())
    }
}
