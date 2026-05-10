# Multi-stage build: compile Forge in a Rust container, run in a slim Debian.
# Fly.io builds this remotely, so you don't need a Linux toolchain locally.

FROM rust:1.85-slim-bookworm AS builder
WORKDIR /usr/src/forge

# System deps for sqlite + reqwest TLS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Cache deps separately so source-only edits don't rebuild the world.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

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

COPY --from=builder /usr/src/forge/target/release/forge /usr/local/bin/forge

# Persistent data lives at /data — Fly mounts a volume here.
ENV FORGE_DB=/data/forge.db
RUN mkdir -p /data

EXPOSE 8080

# Default: serve on 0.0.0.0:8080. The container reads:
#   FORGE_AUTH_TOKEN  — required (or you'll get a public Luna, don't do that)
#   GROQ_API_KEY (or any other backend's key) — required
# Both are set on Fly via `flyctl secrets set ...`.
CMD ["forge", "--db", "/data/forge.db", "--backend", "groq", "serve", "--host", "0.0.0.0", "--port", "8080"]
