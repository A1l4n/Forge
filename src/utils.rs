use uuid::Uuid;
use chrono::Utc;

pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn get_timestamp() -> String {
    Utc::now().to_rfc3339()
}

pub fn truncate(s: &str, length: usize) -> String {
    if s.len() > length {
        format!("{}...", &s[..length])
    } else {
        s.to_string()
    }
}
