# Render Deployment (Free Tier)

This repo includes `render.yaml` for a zero-cost API deployment on Render.

## What You Get
- API service (`trader-api`) built from the repo `Dockerfile`
- Free plan by default (`plan: free`)
- Health check at `/health`
- Auto-generated `TRADER_API_TOKEN`
- Low-resource defaults for smaller instances:
  - `TRADER_OPTIMIZER_ENABLED=false`
  - `TRADER_TOP_COMBOS_BACKTEST_ENABLED=false`
  - `TRADER_BOT_AUTOSTART=false`
  - `TRADER_API_MAX_EPOCHS=30`

## Deploy
1. Push this repo to GitHub.
2. In Render, create a new **Blueprint** and select your repo.
3. Render detects `render.yaml`; keep `trader-api` on `plan: free`.
4. Deploy.
5. In Render env vars, set optional secrets as needed:
   - `BINANCE_API_KEY` / `BINANCE_API_SECRET`
   - `COINBASE_API_KEY` / `COINBASE_API_SECRET` / `COINBASE_API_PASSPHRASE`
   - `TRADER_DB_URL` (optional external Postgres for durable ops/async persistence)
6. Copy your service URL (`https://<service>.onrender.com`) and generated `TRADER_API_TOKEN`.

## Verify
```bash
API_URL="https://<service>.onrender.com"
API_TOKEN="<render-generated-token>"

curl -s "${API_URL}/health"
curl -s -H "Authorization: Bearer ${API_TOKEN}" "${API_URL}/version"
```

## Free-Tier Limits
- Render free web services spin down after 15 minutes of inactivity and cold-start on the next request.
- The local filesystem is ephemeral; state under local paths (for example `TRADER_STATE_DIR`) can be lost after restarts/redeploys.

## Cheapest Always-On Fallback
If you need always-on behavior, use a paid plan:
- Render: use any paid web-service instance type (paid instances do not spin down).
- Railway: `Hobby` is currently $5/month and includes usage up to that amount.

Pricing references (checked on 2026-02-17):
- https://render.com/pricing
- https://render.com/docs/free
- https://docs.railway.com/pricing
