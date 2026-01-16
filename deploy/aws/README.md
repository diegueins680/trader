# AWS Deployment (App Runner)

This project exposes a REST API when run with `--serve` (default port `8080`).

Quick start: see `DEPLOY_AWS_QUICKSTART.md` and `DEPLOY_AWS.md` in the repo root.

## Build & Run Locally (Docker)

```bash
docker build -t trader-api:local .

# Optional: set an API token (recommended if you expose the port beyond localhost)
TRADER_API_TOKEN="$(openssl rand -hex 32)"

docker run --rm -p 8080:8080 -e TRADER_API_TOKEN="$TRADER_API_TOKEN" trader-api:local
curl -s http://127.0.0.1:8080/health
```

Async job persistence (recommended if you use the `*/async` endpoints behind a non-sticky load balancer):
- Mount a shared volume and set `TRADER_STATE_DIR` (recommended) or `TRADER_API_ASYNC_DIR` to your shared mount.
- If you only want async persistence, point `TRADER_API_ASYNC_DIR` at your shared mount; otherwise, use `TRADER_STATE_DIR` to persist journal/bot state/optimizer combos/LSTM weights alongside async jobs.

Example (named Docker volume):

```bash
docker volume create trader-async
docker run --rm -p 8080:8080 \
  -e TRADER_API_TOKEN="$TRADER_API_TOKEN" \
  -v trader-async:/var/lib/trader/async \
  trader-api:local
```

When `TRADER_API_TOKEN` is set, all endpoints except `/health` require either:
- `Authorization: Bearer <token>` or
- `X-API-Key: <token>`

The web UI (`haskell/web`) supports this via deploy-time config: set `apiToken` in `haskell/web/public/trader-config.js` (or `haskell/web/dist/trader-config.js` after build).

Build info:
- `GET /` and `GET /health` include `version` and optional `commit` (from env `TRADER_GIT_COMMIT` / `TRADER_COMMIT` / `GIT_COMMIT` / `COMMIT_SHA`).

## Deploy to App Runner (ECR)

### 1) Create an ECR repository

- Option A (Console): AWS Console → **ECR** → **Repositories** → **Create repository** → Name: `trader-api`
- Option B (AWS CLI):
  - Note: ECR uses regions (Tokyo is `ap-northeast-1`), not availability zones (like `ap-northeast-1a`).

Or directly:

```bash
aws ecr create-repository \
  --repository-name trader-api \
  --image-scanning-configuration scanOnPush=true \
  --region ap-northeast-1
```

### 2) Build and push the image

```bash
AWS_REGION=ap-northeast-1
AWS_ACCOUNT_ID=123456789012
ECR_REPO=trader-api
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# In zsh, prefer "${VAR}:latest" (not "$VAR:latest") to avoid zsh's ":<modifier>" expansion.
docker build -t "${ECR_REPO}:latest" .
docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
```

### 3) Create the App Runner service

- AWS Console → **App Runner** → **Create service**
- Source: **Container registry** → **Amazon ECR**
- Image: `trader-api:latest`
- Port: `8080`
- Health check path: `/health`
- Scaling: set **min=1 / max=1** unless you configure shared async-job storage (see below)
- Environment variables:
  - `TRADER_API_TOKEN` (recommended)
  - `BINANCE_API_KEY` / `BINANCE_API_SECRET` (only if you will call `/trade`)
  - `TRADER_BOT_SYMBOLS` / `TRADER_BOT_TRADE` (optional; used by the cron watchdog to build `/bot/start`)
  - Required: PostgreSQL persistence for ops/combos:
    - `TRADER_DB_URL=postgresql://user:pass@host:5432/trader?sslmode=require`
  - Required: S3 state persistence (App Runner has no EFS support):
    - `TRADER_STATE_S3_BUCKET=<bucket>`
    - `TRADER_STATE_S3_PREFIX=trader`
    - `TRADER_STATE_S3_REGION=ap-northeast-1`
  - Required: shared state directory (ECS/EKS/Docker) to persist across deploys:
    - `TRADER_STATE_DIR=/var/lib/trader/state` (mount durable storage)
  - Optional safety limits (to avoid OOM / timeouts on small instances):
    - `TRADER_API_MAX_ASYNC_RUNNING` (default: `1`)
    - `TRADER_API_MAX_BARS_LSTM` (default: `1000`)
    - `TRADER_API_MAX_EPOCHS` (default: `100`)
    - `TRADER_API_MAX_HIDDEN_SIZE` (default: `32`; set to `50` to allow larger LSTM hidden sizes)
  - Async-job persistence (recommended if you run multiple instances behind a non-sticky load balancer):
    - `TRADER_API_ASYNC_DIR` (shared mount; App Runner has no volume support). Docker image default: `/var/lib/trader/async`.
    - Or set `TRADER_STATE_DIR` to a shared mount to persist async jobs plus journal/bot state/optimizer combos/LSTM weights.
      - For multi-instance deployments, ensure this path is a shared writable mount across all instances (otherwise polling can still return “Not found”).

App Runner note: EFS volumes are not supported; use S3 (`TRADER_STATE_S3_BUCKET`) for persistence on App Runner.

Security note: if you set Binance keys and expose the service publicly, protect it (at minimum set `TRADER_API_TOKEN`, and ideally restrict ingress or put it behind an authenticated gateway).

Note (AWS CLI): when creating an App Runner service from a **private ECR** image, you must provide `AuthenticationConfiguration.AccessRoleArn` (an IAM role trusted by `build.apprunner.amazonaws.com` with the managed policy `AWSAppRunnerServicePolicyForECRAccess`). The repo’s `deploy-aws-quick.sh` script creates/reuses this role automatically. It can also create or reuse the S3 state bucket + App Runner instance role (`--ensure-resources`) and, if requested, the UI bucket + CloudFront distribution (`--cloudfront`).

### Scaling note (important)

This API includes:
- **Stateful** endpoints (`/bot/*`) that assume a single running instance, and
- **Async job** endpoints (`/signal/async`, `/backtest/async`, `/trade/async`) that need a shared `TRADER_API_ASYNC_DIR` mount for polling to work across instances (a local-only directory won’t help behind a non-sticky load balancer).

If you run multiple instances behind a non-sticky load balancer (including the optional CloudFront `/api/*` proxy), the UI can fail async polling with “Async job not found”.

Recommendation: run **single-instance** (min=1 / max=1) unless you have shared async-job storage.

Helper (AWS CLI):

`deploy-aws-quick.sh` sets min=1/max=1 automatically on API deploy. If you need to adjust it later, use the App Runner console.

## Web UI (S3/CloudFront)

The web UI is a static app (`haskell/web`). For production, it can call your deployed API directly (no CloudFront `/api/*` proxy required).

Build the UI pointing at your API:

```bash
cd haskell/web
TRADER_API_TARGET="https://<your-api-host>" npm run build
```

Then upload `haskell/web/dist` to your static hosting origin (S3/CloudFront).

### UI deploy helper (S3 + optional CloudFront invalidation)

If you already have an S3 bucket (and optionally a CloudFront distribution), you can deploy the UI and write `trader-config.js` in one step:

```bash
AWS_REGION=ap-northeast-1
S3_BUCKET="trader-ui-..."
APP_RUNNER_SERVICE_ARN="arn:aws:apprunner:..."
CLOUDFRONT_DISTRIBUTION_ID="E123..."

bash deploy-aws-quick.sh --ui-only \
  --region "$AWS_REGION" \
  --bucket "$S3_BUCKET" \
  --service-arn "$APP_RUNNER_SERVICE_ARN" \
  --distribution-id "$CLOUDFRONT_DISTRIBUTION_ID"
```

The script:
- Builds `haskell/web`
- Writes `haskell/web/dist/trader-config.js` (apiBaseUrl + apiToken)
- Syncs `dist/` to S3 and (optionally) invalidates CloudFront
- Tip: use `--cloudfront-domain d123.cloudfront.net` (or `TRADER_UI_CLOUDFRONT_DOMAIN`) to reuse an existing distribution without manually supplying the S3 bucket.
- The script loads `.env.deploy` (or `TRADER_DEPLOY_ENV_FILE`) for persistent deploy defaults.

### What’s the “API host”?

It’s the public base URL where your backend is reachable (the service running `trader-hs -- --serve`), for example:
- your **App Runner** service URL
- an **ALB** / API Gateway URL in front of your container
- an EC2 public URL (prefer putting it behind HTTPS)

It is not your CloudFront static site URL.

### API token

If you set `TRADER_API_TOKEN` on the backend, all endpoints except `/health` require it via:
- `Authorization: Bearer <token>` or
- `X-API-Key: <token>`

Generate a random token locally (example):

```bash
openssl rand -hex 32
```

Then set it as `TRADER_API_TOKEN` on the backend and set the same value in the UI’s deploy config (`haskell/web/public/trader-config.js` → `apiToken`).

### CloudFront `/api/*` proxy (optional)

If you prefer the UI calling `/api/*` on the same domain, configure a CloudFront behavior:
- Path pattern: `/api/*`
- Origin: your API service (App Runner/ALB/etc)
- Allowed methods: include `POST` (and `OPTIONS`) — the UI uses `POST` for async job polling (with `GET` fallback)
- Forward headers: at least `Authorization`, `X-API-Key`, `Content-Type`
- Forward query strings: include all (the UI uses `/bot/status?tail=...` to keep responses small)
- Cache: disable caching for `/api/*`

Notes:
- The UI config defaults `apiBaseUrl` to `/api` when a CloudFront distribution is set. Use `--ui-api-direct` (or `TRADER_UI_API_MODE=direct`) to keep the direct API URL (CORS required; the quick deploy script can auto-fill `TRADER_CORS_ORIGIN` from the CloudFront domain, and defaults `apiFallbackUrl` to `/api` for same-origin fallback).
- You can set the API base URL at deploy time via `haskell/web/public/trader-config.js` (`apiBaseUrl`). Set it to `/api` only when using the same-origin CloudFront `/api/*` behavior.
- If you run multiple backend instances, either keep it single-instance or ensure `TRADER_API_ASYNC_DIR` points to a shared writable directory (CloudFront itself is not sticky, so async jobs can return “Not found” when polling hits a different instance).
- If you *do* prefer same-origin `/api/*` routing, see “CloudFront `/api/*` proxy (optional)” above.
- After uploading a new UI build to S3, invalidate CloudFront so clients fetch the new hashed JS/CSS assets.
