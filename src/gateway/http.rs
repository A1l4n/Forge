//! HTTP gateway built on Axum.
//!
//! Routes:
//! - `POST /api/message`            — send a message to Luna (multi-agent flow).
//! - `GET  /api/status`             — system info: model, agents, version.
//! - `GET  /api/history/:session_id`— retrieve persisted messages for a session.
//! - `GET  /healthz`                — liveness probe.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use crate::agents::Agent;
use crate::errors::{Error, Result};
use crate::memory::MemoryStore;
use crate::models::{AgentActivity, Message, UserRequest};
use crate::Orchestrator;

/// Shared state carried by every request handler.
#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<Orchestrator>,
    pub memory: Arc<MemoryStore>,
    pub agents: Arc<Vec<Arc<dyn Agent>>>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct MessageRequest {
    pub content: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub session_id: String,
    pub response: String,
    pub activities: Vec<AgentActivity>,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub app: &'static str,
    pub version: &'static str,
    pub model: String,
    pub agents: Vec<AgentInfo>,
}

#[derive(Debug, Serialize)]
pub struct AgentInfo {
    pub name: String,
    pub role: String,
}

#[derive(Debug, Serialize)]
pub struct HistoryResponse {
    pub session_id: String,
    pub messages: Vec<Message>,
}

/// Build the Axum router. Useful for tests and embedding.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/api/status", get(status))
        .route("/api/message", post(message))
        .route("/api/history/:session_id", get(history))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Bind and serve on `host:port`.
pub async fn start_server(host: &str, port: u16, state: AppState) -> Result<()> {
    let ip = IpAddr::from_str(host)
        .map_err(|e| Error::Configuration(format!("invalid host '{}': {}", host, e)))?;
    let addr = SocketAddr::from((ip, port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Error::HTTP(format!("bind {}: {}", addr, e)))?;
    info!("Forge HTTP gateway listening on http://{}", addr);
    axum::serve(listener, router(state).into_make_service())
        .await
        .map_err(|e| Error::HTTP(format!("server: {}", e)))?;
    Ok(())
}

// ---- handlers ----

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let agents = state
        .agents
        .iter()
        .map(|a| AgentInfo {
            name: a.name().to_string(),
            role: a.role().to_string(),
        })
        .collect();

    Json(StatusResponse {
        app: crate::APP_NAME,
        version: crate::VERSION,
        model: state.model.clone(),
        agents,
    })
}

async fn message(
    State(state): State<AppState>,
    Json(payload): Json<MessageRequest>,
) -> std::result::Result<Json<MessageResponse>, ApiError> {
    let session_id = payload
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let request = UserRequest {
        content: payload.content,
        session_id: session_id.clone(),
        context: None,
    };

    let result = state
        .orchestrator
        .process_with_agents(request, &state.agents, &state.memory)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(MessageResponse {
        session_id,
        response: result.response,
        activities: result.activities,
    }))
}

async fn history(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> std::result::Result<Json<HistoryResponse>, ApiError> {
    let messages = state
        .memory
        .get_session_messages(&session_id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(HistoryResponse {
        session_id,
        messages,
    }))
}

// ---- error mapping ----

struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": {
                "type": format!("{}", self.0),
            }
        });
        (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
    }
}
