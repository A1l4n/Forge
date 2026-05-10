//! HTTP gateway built on Axum.
//!
//! Routes:
//! - `GET  /`                       — embedded web UI (chat interface).
//! - `POST /api/message`            — send a message to Luna (multi-agent flow).
//! - `GET  /api/status`             — system info: provider, model, agents, version, usage.
//! - `GET  /api/history/:session_id`— retrieve persisted messages for a session.
//! - `GET  /healthz`                — liveness probe.

use axum::{
    extract::{Path, Query, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

use crate::agents::Agent;
use crate::errors::{Error, Result};
use crate::memory::{DashboardStats, MemoryStore, SessionSummary};
use crate::models::{AgentActivity, Message, Task, UserRequest};
use crate::webui;
use crate::{Orchestrator, UsageTotals};

/// Shared state carried by every request handler.
#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<Orchestrator>,
    pub memory: Arc<MemoryStore>,
    pub agents: Arc<Vec<Arc<dyn Agent>>>,
    pub model: String,
    pub provider: String,
    /// Optional shared secret. When `Some`, every request (except `/healthz`
    /// and `/`) must include `Authorization: Bearer <token>` or
    /// `X-Forge-Token: <token>`. The login UI stores the token in
    /// `localStorage` after the user enters it once.
    pub auth_token: Option<String>,
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
    pub usage: UsageTotals,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub app: &'static str,
    pub version: &'static str,
    pub provider: String,
    pub model: String,
    pub agents: Vec<AgentInfo>,
    pub usage: UsageTotals,
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

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub limit: Option<i64>,
    pub session_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TasksResponse {
    pub tasks: Vec<Task>,
}

#[derive(Debug, Serialize)]
pub struct SessionsResponse {
    pub sessions: Vec<SessionSummary>,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    #[serde(flatten)]
    pub stats: DashboardStats,
    pub usage: UsageTotals,
    pub provider: String,
    pub model: String,
}

/// Build the Axum router. Useful for tests and embedding.
pub fn router(state: AppState) -> Router {
    // Public routes — never auth-checked.
    let public = Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/auth/check", get(auth_check));

    // Protected routes — gated by auth middleware (only when a token is set).
    let protected = Router::new()
        .route("/api/status", get(status))
        .route("/api/message", post(message))
        .route("/api/history/:session_id", get(history))
        .route("/api/tasks", get(list_tasks))
        .route("/api/sessions", get(list_sessions))
        .route("/api/stats", get(stats))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_auth,
        ));

    public
        .merge(protected)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Auth middleware. If `auth_token` is set on AppState, every protected route
/// requires a matching `Authorization: Bearer <token>` or `X-Forge-Token: <token>`
/// header. When the token is `None`, this passes through unconditionally.
async fn require_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    let Some(expected) = state.auth_token.as_ref() else {
        return Ok(next.run(request).await);
    };

    let provided = extract_token(request.headers());
    match provided {
        Some(t) if t == *expected => Ok(next.run(request).await),
        _ => {
            warn!(path = %request.uri().path(), "Auth rejected");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

fn extract_token(headers: &HeaderMap) -> Option<String> {
    if let Some(v) = headers.get("authorization") {
        if let Ok(s) = v.to_str() {
            if let Some(rest) = s.strip_prefix("Bearer ") {
                return Some(rest.trim().to_string());
            }
            if let Some(rest) = s.strip_prefix("bearer ") {
                return Some(rest.trim().to_string());
            }
        }
    }
    if let Some(v) = headers.get("x-forge-token") {
        if let Ok(s) = v.to_str() {
            return Some(s.trim().to_string());
        }
    }
    None
}

/// Lightweight endpoint the UI hits on load: returns whether auth is required
/// and (if a token was sent) whether it's valid. The UI uses this to decide
/// whether to show the login screen.
async fn auth_check(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let required = state.auth_token.is_some();
    let valid = match (&state.auth_token, extract_token(&headers)) {
        (None, _) => true,
        (Some(expected), Some(provided)) => &provided == expected,
        _ => false,
    };
    Json(serde_json::json!({ "required": required, "valid": valid }))
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
    info!("Open the web UI at http://{}/", addr);
    axum::serve(listener, router(state).into_make_service())
        .await
        .map_err(|e| Error::HTTP(format!("server: {}", e)))?;
    Ok(())
}

// ---- handlers ----

async fn index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        Html(webui::INDEX_HTML),
    )
}

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

    let usage = state.orchestrator.current_usage().await;

    Json(StatusResponse {
        app: crate::APP_NAME,
        version: crate::VERSION,
        provider: state.provider.clone(),
        model: state.model.clone(),
        agents,
        usage,
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
        usage: result.usage,
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

async fn list_tasks(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> std::result::Result<Json<TasksResponse>, ApiError> {
    let limit = q.limit.unwrap_or(100).clamp(1, 500);
    let tasks = match q.session_id {
        Some(sid) => state.memory.get_session_tasks(&sid).await.map_err(ApiError::from)?,
        None => state
            .memory
            .list_recent_tasks(limit)
            .await
            .map_err(ApiError::from)?,
    };
    Ok(Json(TasksResponse { tasks }))
}

async fn list_sessions(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> std::result::Result<Json<SessionsResponse>, ApiError> {
    let limit = q.limit.unwrap_or(50).clamp(1, 200);
    let sessions = state
        .memory
        .list_recent_sessions(limit)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(SessionsResponse { sessions }))
}

async fn stats(
    State(state): State<AppState>,
) -> std::result::Result<Json<StatsResponse>, ApiError> {
    let stats = state.memory.stats().await.map_err(ApiError::from)?;
    let usage = state.orchestrator.current_usage().await;
    Ok(Json(StatsResponse {
        stats,
        usage,
        provider: state.provider.clone(),
        model: state.model.clone(),
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
