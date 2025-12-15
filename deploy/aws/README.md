# AWS Deployment (App Runner)

This project exposes a REST API when run with `--serve` (default port `8080`).

## Build & Run Locally (Docker)

```bash
docker build -t trader-api:local .
docker run --rm -p 8080:8080 -e TRADER_API_TOKEN=change-me trader-api:local
curl -s http://127.0.0.1:8080/health
```

When `TRADER_API_TOKEN` is set, all endpoints except `/health` require either:
- `Authorization: Bearer <token>` or
- `X-API-Key: <token>`

The web UI (`haskell/web`) supports this: paste the token into the UI’s “API token” field.

## Deploy to App Runner (ECR)

### 1) Create an ECR repository

- Option A (Console): AWS Console → **ECR** → **Repositories** → **Create repository** → Name: `trader-api`
- Option B (AWS CLI):
  - Note: ECR uses regions (Tokyo is `ap-northeast-1`), not availability zones (like `ap-northeast-1a`).

```bash
AWS_REGION=ap-northeast-1
ECR_REPO=trader-api
bash deploy/aws/create-ecr-repo.sh
```

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
- Environment variables:
  - `TRADER_API_TOKEN` (recommended)
  - `BINANCE_API_KEY` / `BINANCE_API_SECRET` (only if you will call `/trade`)

Security note: if you set Binance keys and expose the service publicly, protect it (at minimum set `TRADER_API_TOKEN`, and ideally restrict ingress or put it behind an authenticated gateway).

## Web UI (S3/CloudFront)

The web UI is a static app (`haskell/web`). For production, it can call your deployed API directly (no CloudFront `/api/*` proxy required).

Build the UI pointing at your API:

```bash
cd haskell/web
TRADER_API_TARGET="https://<your-api-host>" npm run build
```

Then upload `haskell/web/dist` to your static hosting origin (S3/CloudFront).

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

Then set it as `TRADER_API_TOKEN` on the backend and paste the same value into the UI’s “API token” field.

### CloudFront `/api/*` proxy (optional)

If you prefer the UI calling `/api/*` on the same domain, configure a CloudFront behavior:
- Path pattern: `/api/*`
- Origin: your API service (App Runner/ALB/etc)
- Allowed methods: include `POST` (and `OPTIONS`)
- Forward headers: at least `Authorization`, `X-API-Key`, `Content-Type`
- Cache: disable caching for `/api/*`

Notes:
- You can also override the API base at runtime from the UI (stored in local storage) via the “API base URL” field.
- If you *do* prefer same-origin `/api/*` routing, see “CloudFront `/api/*` proxy (optional)” above.
