#!/bin/bash
# Luna EU deploy to Fly.io (Amsterdam) — run this ONCE after your meeting.
# Prerequisite: flyctl must be authenticated (flyctl auth login)
#
# Usage: bash deploy-fly.sh
# Time : ~10 minutes (Fly builds remotely, you just wait)

set -e
FLYCTL="$HOME/.fly/bin/flyctl"

echo "=== Luna EU Deployment to Fly.io (Amsterdam) ==="
echo ""

# 1. First-time app creation (idempotent — skipped if app already exists)
if ! "$FLYCTL" apps list 2>/dev/null | grep -q "luna-forge-eu"; then
  echo "Creating app luna-forge-eu in region ams..."
  "$FLYCTL" apps create luna-forge-eu --machines
fi

# 2. Create persistent volume (idempotent)
if ! "$FLYCTL" volumes list -a luna-forge-eu 2>/dev/null | grep -q "luna_data"; then
  echo "Creating persistent volume for Luna's memory + trade journal..."
  "$FLYCTL" volumes create luna_data --app luna-forge-eu --region ams --size 1
fi

echo ""
echo "=== Setting secrets (edit these with your actual keys) ==="
echo "Run the following command with your real keys:"
echo ""
echo '  flyctl secrets set -a luna-forge-eu \'
echo '    FORGE_AUTH_TOKEN=4qUPrcRi7T6oebfFp0GQ2s3ntKgSMzuXBxwjyAmY \'
echo '    FORGE_BACKEND=gemini \'
echo '    GEMINI_API_KEY=<YOUR_NEW_GEMINI_KEY> \'
echo '    BINANCE_API_KEY=MM4mjkmzzXmmo0Y7015ACXshAN3lLSdk2zpON0z1UUweNryXAUVKbZi1QlH8gCxr \'
echo '    BINANCE_API_SECRET=Itv2f2HgrGDyykvdvjTipogGMwLv5AiSc1Z2UPSZIPb32707zlsRyrMzOTkMDXKj \'
echo '    BINANCE_TESTNET=false'
echo ""
read -p "Press Enter after you've set secrets above to continue with deploy..."

# 3. Deploy
echo "Deploying Luna to Amsterdam..."
"$FLYCTL" deploy --app luna-forge-eu --config fly.toml

echo ""
echo "=== Deployment complete! ==="
"$FLYCTL" status -a luna-forge-eu
echo ""
echo "=== Your Luna EU IP (whitelist on Binance) ==="
"$FLYCTL" ips list -a luna-forge-eu
echo ""
echo "Luna URL: https://luna-forge-eu.fly.dev"
echo "Auth token: 4qUPrcRi7T6oebfFp0GQ2s3ntKgSMzuXBxwjyAmY"
echo ""
echo "NEXT STEP: Go to Binance → API Management → Edit your key → add the IP above"
