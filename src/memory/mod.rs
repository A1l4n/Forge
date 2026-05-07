use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use crate::Result;
use crate::errors::ForgeError;

pub struct MemoryStore {
    pool: SqlitePool,
}

impl MemoryStore {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| ForgeError::DatabaseError(e.to_string()))?;

        Ok(Self { pool })
    }

    pub async fn store(&self, key: &str, value: &str) -> Result<()> {
        // Implement storage logic
        Ok(())
    }

    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        // Implement retrieval logic
        Ok(None)
    }
}
