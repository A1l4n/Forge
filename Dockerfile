# Multi-stage build: compile Forge in a Rust container, run in a slim Debian.
# Fly.io builds this remotely, so you don't need a Linux toolchain locally.

FROM rust:1.86-slim-bookworm AS builder
WORKDIR /usr/src/forge

# System deps for sqlite + reqwest TLS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Cache deps separately so source-only edits don't rebuild the world.
# BuildKit --mount=type=cache keeps the cargo registry and compiled deps
# across ALL builds (even different tags) — no more 90-min dep recompiles.
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/forge/target \
    mkdir -p src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -rf src

COPY src ./src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/forge/target \
    touch src/main.rs && cargo build --release \
    && cp target/release/forge /tmp/forge-bin

# ---- runtime image ----
FROM debian:bookworm-slim
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      libssl3 \
      sqlite3 \
      git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /tmp/forge-bin /usr/local/bin/forge

# Persistent data lives at /data — Fly mounts a volume here.
ENV FORGE_DB=/data/forge.db
RUN mkdir -p /data

EXPOSE 8080

# Default: serve on 0.0.0.0:8080. The container reads:
#   FORGE_AUTH_TOKEN  — required (protects the endpoint)
#   FORGE_BACKEND     — which LLM to use (default: groq)
#                       options: groq, gemini, sambanova, pollinations,
#                                deepseek, xai, cerebras, nvidia, hyperbolic
#   <BACKEND>_API_KEY — matching key (pollinations needs none)
#   BINANCE_API_KEY / BINANCE_API_SECRET — Binance trading tools
#   BINANCE_TESTNET=true — paper-trade mode (strongly recommended first)

# Shell form lets Docker expand $FORGE_BACKEND at container start time,
# so you can change the backend by updating the env var — no rebuild needed.
ENV FORGE_BACKEND=groq
CMD forge --db /data/forge.db --backend ${FORGE_BACKEND} serve --host 0.0.0.0 --port 8080
