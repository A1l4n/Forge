//! Utility functions

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

/// Format a duration in milliseconds to human-readable string
pub fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else if ms < 60000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{:.1}m", ms as f64 / 60000.0)
    }
}

/// Truncate text to max length
pub fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len - 3])
    }
}

/// Pretty print JSON
pub fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "Invalid JSON".to_string())
}

/// Merge two JSON objects
pub fn merge_json(base: &mut Value, override_val: &Value) {
    if let (Some(base_obj), Some(override_obj)) = (base.as_object_mut(), override_val.as_object()) {
        for (key, value) in override_obj {
            base_obj.insert(key.clone(), value.clone());
        }
    }
}

/// Create error JSON response
pub fn error_json(code: &str, message: &str) -> Value {
    json!({
        "error": {
            "code": code,
            "message": message
        }
    })
}

/// Parse ISO datetime string
pub fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Get current timestamp as ISO string
pub fn current_timestamp() -> String {
    Utc::now().to_rfc3339()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500ms");
        assert_eq!(format_duration(5000), "5.0s");
        assert_eq!(format_duration(120000), "2.0m");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "he...");
    }
}
