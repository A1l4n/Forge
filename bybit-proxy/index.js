/**
 * Cloudflare Worker — Exchange API Proxy (Binance + Bybit)
 *
 * Routes:
 *   /binance/api/*    → https://api.binance.com/api/*
 *   /binance/fapi/*   → https://fapi.binance.com/fapi/*
 *   /binance/sapi/*   → https://api.binance.com/sapi/*
 *   /bybit/v5/*       → https://api.bybit.com/v5/*
 *
 * WHY: Binance and Bybit block direct API calls from GCP US IPs (geo-block).
 * This Worker runs on Cloudflare's edge (global IPs, not US-flagged), so
 * requests arrive at the exchange from Cloudflare IPs which are not blocked.
 *
 * SECURITY: Protected by PROXY_SECRET header. Only Luna knows this secret.
 * Luna sends:  X-Proxy-Secret: <your-secret>
 * Worker checks it and rejects anything else.
 *
 * FREE TIER: 100,000 req/day — plenty for a trading bot (~50-200/day).
 *
 * DEPLOY (one-time, ~3 min):
 *   1. Sign up free: https://dash.cloudflare.com/sign-up
 *   2. npm i -g wrangler
 *   3. wrangler login
 *   4. cd bybit-proxy && wrangler deploy
 *   5. wrangler secret put PROXY_SECRET   ← enter any strong random string
 *   6. Copy the worker URL shown after deploy (e.g. https://exchange-proxy.xxx.workers.dev)
 *
 * Then on the GCP VM, edit /opt/forge/luna.env and add:
 *   BINANCE_PROXY_URL=https://exchange-proxy.xxx.workers.dev/binance
 *   BINANCE_PROXY_SECRET=<the secret from step 5>
 *
 * That's it — Luna will automatically route Binance calls through the proxy.
 */

export default {
  async fetch(request, env) {
    // ── Auth check ──────────────────────────────────────────────────────────
    const proxySecret = env.PROXY_SECRET;
    if (proxySecret) {
      const incoming = request.headers.get("X-Proxy-Secret");
      if (incoming !== proxySecret) {
        return new Response(
          JSON.stringify({ error: "Unauthorized", hint: "Set X-Proxy-Secret header" }),
          { status: 401, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    const url = new URL(request.url);
    const path = url.pathname; // e.g. /binance/fapi/v1/order
    const query = url.search;  // e.g. ?symbol=BTCUSDT&...

    // ── Route: /binance/* ───────────────────────────────────────────────────
    if (path.startsWith("/binance/")) {
      const subpath = path.slice("/binance".length); // e.g. /fapi/v1/order

      let targetBase;
      if (subpath.startsWith("/fapi/")) {
        // Futures (USD-M)
        targetBase = env.BINANCE_TESTNET === "true"
          ? "https://testnet.binancefuture.com"
          : "https://fapi.binance.com";
      } else {
        // Spot / SAPI
        targetBase = env.BINANCE_TESTNET === "true"
          ? "https://testnet.binance.vision"
          : "https://api.binance.com";
      }

      const targetUrl = targetBase + subpath + query;
      return proxyTo(request, targetUrl, "binance");
    }

    // ── Route: /bybit/* ─────────────────────────────────────────────────────
    if (path.startsWith("/bybit/")) {
      const subpath = path.slice("/bybit".length); // e.g. /v5/market/tickers
      const bybitBase = env.BYBIT_TESTNET === "true"
        ? "https://api-testnet.bybit.com"
        : "https://api.bybit.com";
      const targetUrl = bybitBase + subpath + query;
      return proxyTo(request, targetUrl, "bybit");
    }

    // ── Health check ────────────────────────────────────────────────────────
    if (path === "/" || path === "/health") {
      return new Response(
        JSON.stringify({ status: "ok", routes: ["/binance/*", "/bybit/*"] }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }

    return new Response(
      JSON.stringify({ error: "Unknown route. Use /binance/* or /bybit/*" }),
      { status: 400, headers: { "Content-Type": "application/json" } }
    );
  },
};

async function proxyTo(request, targetUrl, exchange) {
  // Forward only exchange-specific auth headers + Content-Type
  const forwardHeaders = new Headers();
  for (const [key, value] of request.headers.entries()) {
    const k = key.toLowerCase();
    const isBinanceHeader = exchange === "binance" && k === "x-mbx-apikey";
    const isBybitHeader  = exchange === "bybit"   && k.startsWith("x-bapi-");
    if (isBinanceHeader || isBybitHeader || k === "content-type") {
      forwardHeaders.set(key, value);
    }
  }
  forwardHeaders.set("User-Agent", "Luna-TradingBot/3.0");

  let body = undefined;
  if (request.method === "POST" || request.method === "PUT") {
    body = await request.text();
  }

  const resp = await fetch(targetUrl, {
    method: request.method,
    headers: forwardHeaders,
    body: body,
  });

  const responseBody = await resp.text();
  return new Response(responseBody, {
    status: resp.status,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "X-Proxy-Via": `Cloudflare-Luna-${exchange}`,
    },
  });
}
