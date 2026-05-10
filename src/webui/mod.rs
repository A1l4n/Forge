//! Embedded single-page web UI for Forge.
//!
//! The HTML/CSS/JS is compiled into the binary via `include_str!` so the
//! resulting executable is self-contained — no separate static directory.

pub const INDEX_HTML: &str = include_str!("index.html");
