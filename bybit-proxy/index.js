/**
 * Cloudflare Worker — Bybit V5 API Proxy
 *
 * Routes: POST/GET /v5/* → https://api.bybit.com/v5/*
 *
 * WHY: Bybit blocks direct API calls from GCP US IPs (geo-block).
 * This Worker runs on Cloudflare's edge (including EU PoPs), so requests
 * arrive at Bybit from Cloudflare IP ranges, which are not geo-blocked.
 *
 * SECURITY: Protected by PROXY_SECRET header. Only Luna knows this secret.
 * Luna sends:  X-Proxy-Secret: <your-secret>
 * Worker checks it and rejects anything else.
 *
 * FREE TIER: 100,000 req/day — plenty for a trading bot (Luna sends ~50-200/day).
 *
 * DEPLOY:
 *   1. Install wrangler: npm i -g wrangler
 *   2. wrangler login
 *   3. cd bybit-proxy && wrangler deploy
 *   4. wrangler secret put PROXY_SECRET   (enter a strong random secret)
 *   5. Set BYBIT_PROXY_URL in luna.env:   https://bybit-proxy.<your-subdomain>.workers.dev
 *   6. Set BYBIT_PROXY_SECRET in luna.env: <the secret you set above>
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

    // ── Build target URL ────────────────────────────────────────────────────
    const url = new URL(request.url);
    const path = url.pathname + url.search; // e.g. /v5/market/tickers?category=linear

    // Only allow /v5/ paths — no proxying to arbitrary URLs
    if (!path.startsWith("/v5/")) {
      return new Response(
        JSON.stringify({ error: "Only /v5/* paths are allowed" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const bybitBase = env.BYBIT_TESTNET === "true"
      ? "https://api-testnet.bybit.com"
      : "https://api.bybit.com";

    const targetUrl = bybitBase + path;

    // ── Forward the request ─────────────────────────────────────────────────
    // Copy all Bybit auth headers (X-BAPI-*) from the incoming request
    const forwardHeaders = new Headers();
    for (const [key, value] of request.headers.entries()) {
      // Forward Bybit auth headers and Content-Type
      if (
        key.toLowerCase().startsWith("x-bapi-") ||
        key.toLowerCase() === "content-type"
      ) {
        forwardHeaders.set(key, value);
      }
    }

    // Add a proper User-Agent so Bybit doesn't block based on that
    forwardHeaders.set("User-Agent", "Luna-TradingBot/3.0");

    let body = undefined;
    if (request.method === "POST" || request.method === "PUT") {
      body = await request.text();
    }

    const bybitResponse = await fetch(targetUrl, {
      method: request.method,
      headers: forwardHeaders,
      body: body,
    });

    // ── Return Bybit's response ─────────────────────────────────────────────
    const responseBody = await bybitResponse.text();
    return new Response(responseBody, {
      status: bybitResponse.status,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "X-Proxy-Via": "Cloudflare-Luna-Worker",
      },
    });
  },
};
