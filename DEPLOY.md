# Deploying Forge

Forge is a single Rust binary + an embedded web UI + a SQLite database.
You have a few hosting options.

## Option A — Fly.io (recommended, ~5 min, free tier)

Best for "Luna lives in the cloud and I can talk to her from anywhere."

```powershell
# 1. Install flyctl
iwr https://fly.io/install.ps1 -useb | iex

# 2. Sign up (asks for credit card but won't charge on free tier)
flyctl auth signup

# 3. Edit fly.toml: change `app = "forge-luna"` to something globally unique
#    (e.g. "luna-apo-2025"). Pick a region near you (`lax`, `iad`, `fra`, ...).

# 4. Create the persistent volume for /data (sqlite + memories survive restarts)
flyctl volumes create forge_data --region iad --size 1

# 5. Set the LLM provider key as a secret
flyctl secrets set GROQ_API_KEY=gsk_your_key_here
# or:  flyctl secrets set CLAUDE_API_KEY=sk-ant-api03-...
# or:  flyctl secrets set GLM_API_KEY=...
# or:  flyctl secrets set OPENAI_API_KEY=sk-...

# 6. Deploy
flyctl deploy

# Watch logs:
flyctl logs

# Get URL:
flyctl status   # shows the public hostname like https://luna-apo-2025.fly.dev
```

After step 6 you get a permanent HTTPS URL like `https://luna-apo-2025.fly.dev`.
Add it to your phone home screen — voice + chat work natively in mobile Safari/Chrome.

**Backups:** the sqlite file lives on the persistent volume. Snapshot it with:
```
flyctl ssh console -C "sqlite3 /data/forge.db .dump" > backup.sql
```

**Costs:** with `min_machines_running = 0` Fly stops the VM when idle and starts
on demand (~1s cold start). Free tier covers ~3 small VMs running 24/7. Volumes
are $0.15/GB/mo (1GB = $0.15/mo). Practically free.

---

## Option B — Cloudflared tunnel (no signup, only when laptop is on)

```powershell
# Terminal 1 — start forge
$env:GROQ_API_KEY = "gsk_..."
.\target\release\forge.exe --backend groq serve --host 0.0.0.0 --port 8080

# Terminal 2 — public HTTPS URL
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8080
```

You get a `https://*.trycloudflare.com` URL. Works only while your laptop is
running and tunnel is open.

---

## Option C — Self-host on a VPS (Hetzner / DigitalOcean / Linode, ~$5/mo)

Same Dockerfile works. SCP the binary or build on the VPS:

```bash
ssh user@your-vps
git clone <your-fork>
cd forge
docker build -t forge .
docker run -d \
  --name luna \
  -p 8080:8080 \
  -v /opt/forge/data:/data \
  -e GROQ_API_KEY=gsk_... \
  forge
```

Then point a subdomain at the VPS IP and put Caddy or Nginx in front for HTTPS.

---

## Strict mode for public deployment

Once Luna lives on a public URL, lock down destructive tools:

```toml
# fly.toml — extend the CMD in Dockerfile or override here
[processes]
  app = "forge --db /data/forge.db --backend groq --strict --allow-tool save_memory --allow-tool save_skill serve --host 0.0.0.0 --port 8080"
```

`--strict` blocks `write_file`, `run_shell`, `self_edit_source`, `git_commit`,
`spawn_agent`. `--allow-tool` selectively re-enables specific ones.
