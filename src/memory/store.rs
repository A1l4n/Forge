//! SQLite-backed memory store for sessions, messages, and tasks.
//!
//! Uses `sqlx` with the bundled tokio-rustls runtime. The connection string
//! is normalised so callers can pass either a plain path or a `sqlite://` URL.

use chrono::{DateTime, TimeZone, Utc};
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use tracing::{debug, info};
use uuid::Uuid;

use crate::errors::{Error, Result};
use crate::models::{Message, MessageRole, Session, Task, TaskStatus};

/// A reusable skill saved by the system (Phase 2 hook — schema is provisioned
/// upfront so future agents can persist without a migration).
#[derive(Debug, Clone)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub code: String,
    pub success_count: i64,
    pub created_at: DateTime<Utc>,
}

/// Persistent state for sessions, messages, and tasks.
#[derive(Clone)]
pub struct MemoryStore {
    db_path: String,
    pool: SqlitePool,
}

impl MemoryStore {
    /// Open (or create) a SQLite database at `db_path` and run migrations.
    pub async fn open(db_path: impl Into<String>) -> Result<Self> {
        let db_path = db_path.into();
        let connect_path = strip_sqlite_prefix(&db_path);

        if let Some(parent) = Path::new(&connect_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Memory(format!("create dir {}: {}", parent.display(), e)))?;
            }
        }

        let opts = SqliteConnectOptions::from_str(&connect_path)
            .map_err(|e| Error::Memory(format!("invalid db path {}: {}", connect_path, e)))?
            .create_if_missing(true)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect_with(opts)
            .await?;

        let store = Self { db_path, pool };
        store.init().await?;
        Ok(store)
    }

    /// Backwards-compatible constructor — prefer [`MemoryStore::open`].
    pub async fn new(db_path: String) -> Result<Self> {
        Self::open(db_path).await
    }

    /// Database path used by this store (without any `sqlite://` prefix).
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    /// Underlying connection pool, exposed for advanced queries.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Create tables if missing.
    pub async fn init(&self) -> Result<()> {
        info!("Initialising memory store at {}", self.db_path);

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                agent_name TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                parent_id TEXT,
                agent_name TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (parent_id) REFERENCES tasks(id)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                code TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);")
            .execute(&self.pool)
            .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);")
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // ---------- Sessions ----------

    pub async fn save_session(&self, session: &Session) -> Result<()> {
        let metadata = serde_json::to_string(&session.metadata)?;
        sqlx::query(
            r#"
            INSERT INTO sessions (id, user_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id=excluded.user_id,
                updated_at=excluded.updated_at,
                metadata=excluded.metadata
            "#,
        )
        .bind(&session.id)
        .bind(&session.user_id)
        .bind(session.created_at.to_rfc3339())
        .bind(session.updated_at.to_rfc3339())
        .bind(metadata)
        .execute(&self.pool)
        .await?;
        debug!(session_id = %session.id, "Session saved");
        Ok(())
    }

    pub async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        let row = sqlx::query(
            "SELECT id, user_id, created_at, updated_at, metadata FROM sessions WHERE id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(match row {
            None => None,
            Some(r) => {
                let metadata_str: Option<String> = r.try_get("metadata").ok();
                let metadata = metadata_str
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                Some(Session {
                    id: r.try_get("id")?,
                    user_id: r.try_get("user_id")?,
                    created_at: parse_dt(r.try_get("created_at")?)?,
                    updated_at: parse_dt(r.try_get("updated_at")?)?,
                    metadata,
                })
            }
        })
    }

    /// Ensure a session row exists for the given id (creating it for an anonymous user
    /// if none is found). Returns the session.
    pub async fn ensure_session(&self, session_id: &str, user_id: &str) -> Result<Session> {
        if let Some(s) = self.get_session(session_id).await? {
            return Ok(s);
        }
        let session = Session {
            id: session_id.to_string(),
            user_id: user_id.to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: Default::default(),
        };
        self.save_session(&session).await?;
        Ok(session)
    }

    // ---------- Messages ----------

    pub async fn save_message(&self, message: Message) -> Result<()> {
        // Make sure the parent session exists so the FK is satisfied.
        self.ensure_session(&message.session_id, "anonymous").await?;

        let role = message_role_str(message.role);
        let tool_calls = match &message.tool_calls {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };
        let tool_results = match &message.tool_results {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };
        let metadata = match &message.metadata {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };

        sqlx::query(
            r#"
            INSERT INTO messages (id, session_id, role, content, agent_name, tool_calls, tool_results, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&message.id)
        .bind(&message.session_id)
        .bind(role)
        .bind(&message.content)
        .bind(&message.agent_name)
        .bind(tool_calls)
        .bind(tool_results)
        .bind(metadata)
        .bind(message.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        debug!(message_id = %message.id, session_id = %message.session_id, "Message saved");
        Ok(())
    }

    pub async fn get_session_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, role, content, agent_name, tool_calls, tool_results, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY datetime(created_at) ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let role_str: String = r.try_get("role")?;
            let tool_calls_str: Option<String> = r.try_get("tool_calls").ok();
            let tool_results_str: Option<String> = r.try_get("tool_results").ok();
            let metadata_str: Option<String> = r.try_get("metadata").ok();

            out.push(Message {
                id: r.try_get("id")?,
                session_id: r.try_get("session_id")?,
                role: parse_role(&role_str),
                content: r.try_get("content")?,
                agent_name: r.try_get("agent_name").ok(),
                tool_calls: parse_json_vec(tool_calls_str)?,
                tool_results: parse_json_vec(tool_results_str)?,
                metadata: parse_json_value(metadata_str)?,
                created_at: parse_dt(r.try_get("created_at")?)?,
            });
        }
        Ok(out)
    }

    pub async fn delete_session_messages(&self, session_id: &str) -> Result<u64> {
        let res = sqlx::query("DELETE FROM messages WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(res.rows_affected())
    }

    // ---------- Tasks ----------

    pub async fn save_task(&self, task: Task) -> Result<()> {
        self.ensure_session(&task.session_id, "anonymous").await?;
        let metadata = match &task.metadata {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };

        sqlx::query(
            r#"
            INSERT INTO tasks (id, session_id, parent_id, agent_name, description, status, result, error, metadata, created_at, updated_at, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                result=excluded.result,
                error=excluded.error,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at,
                started_at=excluded.started_at,
                completed_at=excluded.completed_at
            "#,
        )
        .bind(&task.id)
        .bind(&task.session_id)
        .bind(&task.parent_id)
        .bind(&task.agent_name)
        .bind(&task.description)
        .bind(task_status_str(task.status))
        .bind(&task.result)
        .bind(&task.error)
        .bind(metadata)
        .bind(task.created_at.to_rfc3339())
        .bind(task.updated_at.to_rfc3339())
        .bind(task.started_at.map(|d| d.to_rfc3339()))
        .bind(task.completed_at.map(|d| d.to_rfc3339()))
        .execute(&self.pool)
        .await?;

        debug!(task_id = %task.id, status = %task.status, "Task saved");
        Ok(())
    }

    pub async fn update_task(&self, task: Task) -> Result<()> {
        // save_task is upsert-friendly via ON CONFLICT.
        self.save_task(task).await
    }

    pub async fn get_session_tasks(&self, session_id: &str) -> Result<Vec<Task>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, parent_id, agent_name, description, status, result, error, metadata, created_at, updated_at, started_at, completed_at
            FROM tasks
            WHERE session_id = ?
            ORDER BY datetime(created_at) ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let status_str: String = r.try_get("status")?;
            let metadata_str: Option<String> = r.try_get("metadata").ok();
            let started_at: Option<String> = r.try_get("started_at").ok();
            let completed_at: Option<String> = r.try_get("completed_at").ok();

            out.push(Task {
                id: r.try_get("id")?,
                session_id: r.try_get("session_id")?,
                parent_id: r.try_get("parent_id").ok(),
                agent_name: r.try_get("agent_name")?,
                description: r.try_get("description")?,
                status: parse_status(&status_str),
                result: r.try_get("result").ok(),
                error: r.try_get("error").ok(),
                metadata: parse_json_value(metadata_str)?,
                created_at: parse_dt(r.try_get("created_at")?)?,
                updated_at: parse_dt(r.try_get("updated_at")?)?,
                started_at: started_at.map(parse_dt).transpose()?,
                completed_at: completed_at.map(parse_dt).transpose()?,
            });
        }
        Ok(out)
    }

    // ---------- Skills (Phase 2 hook) ----------

    pub async fn save_skill(
        &self,
        name: &str,
        description: &str,
        code: &str,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        sqlx::query(
            r#"
            INSERT INTO skills (id, name, description, code, success_count, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT(name) DO UPDATE SET
                description=excluded.description,
                code=excluded.code
            "#,
        )
        .bind(&id)
        .bind(name)
        .bind(description)
        .bind(code)
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(id)
    }

    pub async fn list_skills(&self) -> Result<Vec<Skill>> {
        let rows = sqlx::query(
            "SELECT id, name, description, code, success_count, created_at FROM skills",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(Skill {
                id: r.try_get("id")?,
                name: r.try_get("name")?,
                description: r.try_get("description")?,
                code: r.try_get("code")?,
                success_count: r.try_get("success_count")?,
                created_at: parse_dt(r.try_get("created_at")?)?,
            });
        }
        Ok(out)
    }
}

// ---- helpers ----

fn strip_sqlite_prefix(s: &str) -> String {
    s.strip_prefix("sqlite://").unwrap_or(s).to_string()
}

fn message_role_str(r: MessageRole) -> &'static str {
    match r {
        MessageRole::User => "user",
        MessageRole::Luna => "luna",
        MessageRole::Agent => "agent",
        MessageRole::System => "system",
    }
}

fn parse_role(s: &str) -> MessageRole {
    match s {
        "user" => MessageRole::User,
        "luna" => MessageRole::Luna,
        "agent" => MessageRole::Agent,
        "system" => MessageRole::System,
        _ => MessageRole::System,
    }
}

fn task_status_str(s: TaskStatus) -> &'static str {
    match s {
        TaskStatus::Pending => "pending",
        TaskStatus::Running => "running",
        TaskStatus::Completed => "completed",
        TaskStatus::Failed => "failed",
        TaskStatus::Skipped => "skipped",
    }
}

fn parse_status(s: &str) -> TaskStatus {
    match s {
        "pending" => TaskStatus::Pending,
        "running" => TaskStatus::Running,
        "completed" => TaskStatus::Completed,
        "failed" => TaskStatus::Failed,
        "skipped" => TaskStatus::Skipped,
        _ => TaskStatus::Pending,
    }
}

fn parse_dt(s: String) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&s)
        .map(|d| d.with_timezone(&Utc))
        .or_else(|_| {
            // Fallback: SQLite default CURRENT_TIMESTAMP format "YYYY-MM-DD HH:MM:SS".
            let naive = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| Error::Memory(format!("parse datetime {}: {}", s, e)))?;
            Ok::<_, Error>(Utc.from_utc_datetime(&naive).into())
        })
}

fn parse_json_vec(s: Option<String>) -> Result<Option<Vec<Value>>> {
    Ok(match s {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(&s)?),
        _ => None,
    })
}

fn parse_json_value(s: Option<String>) -> Result<Option<Value>> {
    Ok(match s {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(&s)?),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_store() -> MemoryStore {
        MemoryStore::open(":memory:").await.expect("open store")
    }

    #[tokio::test]
    async fn round_trip_message() {
        let store = make_store().await;
        let msg = Message::user("s1".into(), "hello".into());
        store.save_message(msg.clone()).await.unwrap();
        let msgs = store.get_session_messages("s1").await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[0].role, MessageRole::User);
    }

    #[tokio::test]
    async fn round_trip_task() {
        let store = make_store().await;
        let task = Task::new("s1".into(), "CodeAgent".into(), "do thing".into()).start();
        store.save_task(task.clone()).await.unwrap();
        let done = task.complete("ok".into());
        store.update_task(done).await.unwrap();
        let tasks = store.get_session_tasks("s1").await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, TaskStatus::Completed);
        assert_eq!(tasks[0].result.as_deref(), Some("ok"));
    }
}
