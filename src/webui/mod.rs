//! Embedded single-page web UI for Forge.
//!
//! The HTML/CSS/JS is compiled into the binary via `include_str!` so the
//! resulting executable is self-contained — no separate static directory.

pub const INDEX_HTML: &str = include_str!("index.html");

/// PWA web-app manifest served at /manifest.json so mobile browsers can
/// offer "Add to Home Screen" with Luna branding.
pub const MANIFEST_JSON: &str = r#"{
  "name": "Luna",
  "short_name": "Luna",
  "description": "Luna AI Assistant — chat, trade, research.",
  "start_url": "/",
  "display": "standalone",
  "orientation": "portrait-primary",
  "background_color": "#0a0d10",
  "theme_color": "#0a0d10",
  "icons": [
    {
      "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌙</text></svg>",
      "sizes": "any",
      "type": "image/svg+xml",
      "purpose": "any maskable"
    }
  ]
}"#;
