use tracing_subscriber::{
    fmt,
    prelude::*,
    EnvFilter,
};

pub fn init() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_writer(std::io::stdout))
        .init();
}
