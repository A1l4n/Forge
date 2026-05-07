use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
};
use serde_json::json;
use std::sync::Arc;

pub struct Gateway {
    // Add gateway state here
}

impl Gateway {
    pub fn new() -> Self {
        Self {}
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health_check))
            .route("/process", post(process_request))
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(json!({ "status": "ok" }))
}

async fn process_request(
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    Json(payload)
}
