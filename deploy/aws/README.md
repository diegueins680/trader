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

## Deploy to App Runner (ECR)

### 1) Create an ECR repository

- AWS Console → **ECR** → **Repositories** → **Create repository**
- Name: `trader-api`

### 2) Build and push the image

```bash
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
ECR_REPO=trader-api

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -t "$ECR_REPO:latest" .
docker tag "$ECR_REPO:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"
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

