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
