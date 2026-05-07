use forge::{
    claude::ClaudeClient,
    luna::Orchestrator,
    models::UserRequest,
    logging,
};
use std::sync::Arc;
use uuid::Uuid;
use clap::Parser;

#[derive(Parser)]
#[command(name = "Forge")]
#[command(about = "Agentic Operating System powered by Claude", long_about = None)]
struct Args {
    /// The message to process
    #[arg(value_name = "MESSAGE")]
    message: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init();

    let args = Args::parse();

    let message = if let Some(msg) = args.message {
        msg
    } else {
        // Read from stdin if no argument provided
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };

    let api_key = std::env::var("CLAUDE_API_KEY")
        .expect("CLAUDE_API_KEY environment variable not set");

    let claude = Arc::new(ClaudeClient::new(
        api_key,
        "claude-opus-4-6".to_string(),
    ));

    let orchestrator = Orchestrator::new(claude);

    let request = UserRequest {
        content: message,
        session_id: Uuid::new_v4().to_string(),
        context: None,
    };

    match orchestrator.process(request).await {
        Ok(response) => println!("Luna: {}", response),
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}
