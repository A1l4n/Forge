//! Task model for work units

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "skipped")]
    Skipped,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "Pending"),
            TaskStatus::Running => write!(f, "Running"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed => write!(f, "Failed"),
            TaskStatus::Skipped => write!(f, "Skipped"),
        }
    }
}

/// A task assigned to an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub session_id: String,
    pub parent_id: Option<String>,
    pub agent_name: String,
    pub description: String,
    pub status: TaskStatus,
    pub result: Option<String>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub metadata: Option<serde_json::Value>,
}

impl Task {
    /// Create a new task for an agent
    pub fn new(session_id: String, agent_name: String, description: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            parent_id: None,
            agent_name,
            description,
            status: TaskStatus::Pending,
            result: None,
            error: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            started_at: None,
            completed_at: None,
            metadata: None,
        }
    }

    /// Create a sub-task
    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Mark task as running
    pub fn start(mut self) -> Self {
        self.status = TaskStatus::Running;
        self.started_at = Some(Utc::now());
        self.updated_at = Utc::now();
        self
    }

    /// Mark task as completed with result
    pub fn complete(mut self, result: String) -> Self {
        self.status = TaskStatus::Completed;
        self.result = Some(result);
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
        self
    }

    /// Mark task as failed with error
    pub fn fail(mut self, error: String) -> Self {
        self.status = TaskStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
        self
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        let end = self.completed_at.or(self.started_at).unwrap_or(Utc::now());
        let start = self.started_at.unwrap_or(self.created_at);
        (end - start).num_milliseconds() as u64
    }

    /// Check if task is terminal (won't change further)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Skipped
        )
    }
}
