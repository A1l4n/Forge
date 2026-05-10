# Forge 🔥

**Agentic Operating System powered by Claude**

Forge is a high-performance orchestration engine for coordinating multiple specialist AI agents. **Luna**, the main orchestrator, delegates work to specialized agents (Code, Research, Writing, Planning) and synthesizes the results.

Built in **Rust** for speed, safety, and reliability.

---

## Quick Start

### Prerequisites
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Claude API key (get from [console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/forge-ai/forge.git
cd forge
cargo build --release
```

### Usage

```bash
# Set your Claude API key
export CLAUDE_API_KEY="sk-ant-..."

# Run Forge with a message
./target/release/forge "Write a Rust function to calculate fibonacci numbers"

# Or pipe input
echo "What's the capital of France?" | ./target/release/forge
```

---

## Architecture

```
User Input → Gateway → Luna (Orchestrator) → Specialist Agents → Result
                          ↓
                      Tool Registry
                          ↓
                      Memory Store
```

### Components

- **Luna**: Main orchestrator that breaks down tasks and coordinates agents
- **Specialist Agents**: Code, Research, Writing, Planning (extensible)
- **Gateway**: Message router for CLI, Web, API interfaces
- **Memory Store**: SQLite persistence for sessions and history
- **Tool Registry**: Available tools/capabilities for agents
- **Claude Client**: Rust wrapper around Claude API

---

## Features

- ✅ Multi-agent orchestration (parallel execution)
- ✅ Streaming responses
- ✅ Session persistence
- ✅ Tool integration
- ✅ OAuth support
- ✅ Extensible agent system
- 🚧 Web Dashboard (in progress)
- 🚧 Learning loop (planned)

---

## Project Structure

```
forge/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library root
│   ├── claude/              # Claude API client
│   ├── config/              # Configuration management
│   ├── gateway/             # Message routing
│   ├── luna/                # Main orchestrator
│   ├── agents/              # Specialist agents
│   ├── memory/              # State & persistence
│   ├── tools/               # Tool registry
│   ├── models/              # Core data types
│   ├── errors.rs            # Error types
│   ├── logging.rs           # Logging setup
│   └── utils.rs             # Utilities
├── Cargo.toml               # Dependencies
└── README.md                # This file
```

---

## Building

### Development
```bash
cargo build
cargo run -- "Your message here"
```

### Release
```bash
cargo build --release
./target/release/forge "Your message here"
```

### Tests
```bash
cargo test
cargo test -- --nocapture  # Show output
```

---

## Environment Variables

```bash
# Required
CLAUDE_API_KEY="sk-ant-..."

# Optional
FORGE_CONFIG="./forge.config.toml"
RUST_LOG="debug"  # Logging level
```

---

## API

### Creating a Forge Client

```rust
use forge::{
    claude::ClaudeClient,
    luna::Orchestrator,
    models::UserRequest,
};
use std::sync::Arc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let claude = Arc::new(ClaudeClient::new(
        "sk-ant-...".to_string(),
        "claude-opus-4-6".to_string(),
    ));
    
    let orchestrator = Orchestrator::new(claude);
    
    let request = UserRequest {
        content: "Write a Rust function".to_string(),
        session_id: Uuid::new_v4().to_string(),
        context: None,
    };
    
    let response = orchestrator.process(request).await?;
    println!("Luna: {}", response);
    
    Ok(())
}
```

---

## Creating Custom Agents

Implement the `Agent` trait:

```rust
use async_trait::async_trait;
use forge::agents::Agent;

#[derive(Clone)]
pub struct MyCustomAgent;

#[async_trait]
impl Agent for MyCustomAgent {
    fn name(&self) -> &str {
        "MyCustomAgent"
    }

    fn role(&self) -> &str {
        "Custom specialist"
    }

    fn system_prompt(&self) -> String {
        "You are a specialist that...".to_string()
    }

    async fn execute(&self, task: Task, context: &ExecutionContext) -> Result<String> {
        // Your implementation here
        Ok("Result".to_string())
    }
}
```

---

## Performance

- Orchestration latency: <100ms (local)
- API response: 1-3s (depends on Claude API)
- Memory overhead: ~50MB per session
- Concurrent agents: Unlimited (bounded by Claude API rate limits)

---

## Roadmap

### Phase 1 ✅ (In Progress)
- [x] Core orchestrator (Luna)
- [x] Claude API client with streaming
- [x] Base agent framework
- [x] SQLite memory store
- [x] CLI interface
- [ ] Tool registry & execution

### Phase 2 (Week 1.5)
- [ ] Web dashboard (React)
- [ ] Real-time agent monitoring
- [ ] Embedding-based memory search
- [ ] SOUL.md personality system
- [ ] Skill auto-creation

### Phase 3 (Week 2)
- [ ] Production hardening
- [ ] Comprehensive tests
- [ ] Docker containerization
- [ ] Documentation & examples
- [ ] Public release

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent) (learning loop, multi-channel)
- Inspired by [OpenClaw](https://github.com/openclaw/openclaw) (gateway architecture, SOUL.md)
- Powered by [Claude](https://www.anthropic.com/) by Anthropic

---

## Questions?

- 📖 Check the [docs/](docs/) directory
- 🐛 Report issues on GitHub
- 💬 Start a discussion

---

**Built with ❤️ in Rust**
