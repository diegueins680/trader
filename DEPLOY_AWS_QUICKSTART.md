# AWS Deployment - Quick Start (5 Minutes)

## Prerequisites
- AWS account with credentials configured: `aws configure`
- Docker installed and running
- ~$1-2 to test the deployment

## Option 1: Automated Deployment (Recommended)

```bash
cd /path/to/trader

# Generate a random API token
API_TOKEN=$(openssl rand -hex 32)

# Save this token somewhere safe (you'll need it for the web UI deploy config: `trader-config.js`).

# Run the automated deployment script (auto-creates S3 state bucket)
bash deploy-aws-quick.sh --ensure-resources --region ap-northeast-1 --api-token "$API_TOKEN"

# Optional: auto-provision S3 state + CloudFront (reuses existing resources if present)
# bash deploy-aws-quick.sh --ensure-resources --cloudfront --region ap-northeast-1 --api-token "$API_TOKEN"
#   # defaults: trader-api-state-<account>-<region>, trader-ui-<account>-<region>
#
# Optional: override the state directory (default: /var/lib/trader/state)
# bash deploy-aws-quick.sh --region ap-northeast-1 --api-token "$API_TOKEN" --state-dir "/var/lib/trader/state"
#
# Required: enable S3 state persistence for App Runner (script enforces this)
# bash deploy-aws-quick.sh --region ap-northeast-1 --api-token "$API_TOKEN" \
#   --state-s3-bucket "trader-api-state-..." --state-s3-prefix "trader" --instance-role-arn "arn:aws:iam::123:role/TraderAppRunnerS3Role"
```

The script will:
1. ✅ Create ECR repository
2. ✅ Build Docker image
3. ✅ Push to ECR
4. ✅ Create (or reuse) the App Runner ECR access IAM role
5. ✅ Create/update App Runner service (single-instance)
6. ✅ Return the public API URL

With `--ensure-resources`, it also creates or reuses the state S3 bucket and App Runner instance role (and `--cloudfront` will create or reuse the UI bucket + CloudFront distribution).

**Total time: 5-10 minutes**

---

## Persist state with S3 (required for App Runner)

Checklist (App Runner + S3):
1. Create an S3 bucket for state (private).
2. Create an IAM role for App Runner with `s3:GetObject`/`s3:PutObject` on the bucket/prefix.
3. Pass `--state-s3-bucket` (plus optional `--state-s3-prefix`, `--state-s3-region`) and `--instance-role-arn` to the deploy script.
4. Or use `--ensure-resources` to create/reuse the bucket + instance role automatically.
5. The quick deploy script will fail without S3 state configured.
6. App Runner does **not** support EFS volumes; S3 is the supported persistence option.

---

## Option 2: Manual Steps (Step by Step)

### 1. Get AWS Account ID
```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=ap-northeast-1
export ECR_REPO=trader-api
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "ECR URI: $ECR_URI"
```

### 2. Create ECR Repository
```bash
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null
```

### 3. Build & Push Docker Image
```bash
# Login to ECR
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build
docker build -t "${ECR_REPO}:latest" .

# Tag
docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"

# Push
docker push "${ECR_URI}:latest"
```

### 4. Create App Runner Service (Console)
1. AWS Console → **App Runner** → **Create service**
2. Source: **ECR** → select `trader-api:latest`
3. Port: `8080`
4. Health check path: `/health`
5. Environment variables:
   ```
   TRADER_API_TOKEN=<your-api-token>
   TRADER_STATE_DIR=/var/lib/trader/state
   TRADER_STATE_S3_BUCKET=<s3-bucket>
   TRADER_STATE_S3_PREFIX=trader
   TRADER_STATE_S3_REGION=ap-northeast-1
   ```
6. Click **Create & deploy** (wait 5-10 min)

---

## Test the API

```bash
export API_URL=https://xxxx.ap-northeast-1.awsapprunner.com
export TRADER_API_TOKEN=<your-token>

# No auth needed
curl -s "${API_URL}/health"

# With auth (verifies token wiring; /health is always public)
curl -s -H "Authorization: Bearer ${TRADER_API_TOKEN}" "${API_URL}/health"
```

---

## Deploy Web UI (Optional)

### Option A: Use the same deploy script (recommended)

From the repo root:

```bash
AWS_REGION=ap-northeast-1
S3_BUCKET="trader-ui-..."
CLOUDFRONT_DISTRIBUTION_ID="E123..."   # optional

bash deploy-aws-quick.sh --ui-only \
  --region "$AWS_REGION" \
  --ui-bucket "$S3_BUCKET" \
  --api-url "$API_URL" \
  --api-token "$TRADER_API_TOKEN" \
  --distribution-id "$CLOUDFRONT_DISTRIBUTION_ID"
```

Notes:
- When `--distribution-id` is set, the script defaults `apiBaseUrl` to the API URL (direct). Use `--ui-api-proxy`/`TRADER_UI_API_MODE=proxy` to force `/api` (CloudFront `/api/*` behavior required). When `/api` is used and the App Runner URL is known, the script fills `apiFallbackUrl` to the API URL; for direct bases, set `--ui-api-fallback`/`TRADER_UI_API_FALLBACK_URL` explicitly if you want a fallback (CORS required).
- Use `--cloudfront` (and optionally `--ensure-resources`) to auto-create or reuse a CloudFront distribution and set the UI bucket policy.
- CloudFront is non-sticky. Keep App Runner min=1/max=1 unless you have shared async job storage (`TRADER_API_ASYNC_DIR` or `TRADER_STATE_DIR`).

### Option B: Manual deploy (S3 website hosting)

```bash
cd haskell/web

# Build the UI
TRADER_API_TARGET="${API_URL}" npm run build

# Configure deploy-time API settings (edit this file before uploading to S3)
cat > dist/trader-config.js <<EOF
globalThis.__TRADER_CONFIG__ = {
  // Use the API URL directly (default). Set "/api" only if CloudFront proxies /api/* to your API origin.
  apiBaseUrl: "${API_URL}",
  apiToken: "${TRADER_API_TOKEN}",
};
EOF

# Create S3 bucket
export S3_BUCKET=trader-ui-$(date +%s)
aws s3 mb "s3://${S3_BUCKET}" --region ap-northeast-1

# Enable static website hosting (SPA-friendly)
aws s3 website "s3://${S3_BUCKET}" \
  --index-document index.html \
  --error-document index.html

# Upload UI
aws s3 sync dist/ "s3://${S3_BUCKET}/" --delete

# Make public (optional, if you aren't using CloudFront/OAC)
aws s3api put-public-access-block \
  --bucket "$S3_BUCKET" \
  --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false \
  --region ap-northeast-1 || true

aws s3api put-bucket-policy --bucket "$S3_BUCKET" --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::'"$S3_BUCKET"'/*"
  }]
}'

echo "UI: http://${S3_BUCKET}.s3-website-ap-northeast-1.amazonaws.com"
```

If you are using a CloudFront `/api/*` proxy and want same-origin calls, set `apiBaseUrl` to `/api` instead.

---

## Cleanup (Delete Everything)

```bash
export AWS_REGION=ap-northeast-1
export ECR_REPO=trader-api
export S3_BUCKET=trader-ui-xxxxx

# Delete App Runner service
aws apprunner list-services --region "$AWS_REGION" --query 'ServiceSummaryList[?ServiceName==`'"${ECR_REPO}"'`].ServiceArn' --output text | xargs -I {} aws apprunner delete-service --service-arn {} --region "$AWS_REGION"

# Delete ECR repository
aws ecr delete-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" --force

# Delete S3 bucket
aws s3 rb "s3://${S3_BUCKET}" --force
```

---

## Useful Links

- Full guide: `DEPLOY_AWS.md`
- [App Runner docs](https://docs.aws.amazon.com/apprunner/)
- [ECR docs](https://docs.aws.amazon.com/ecr/)

---

## Common Issues

**API returns "API unreachable":**
- Wait for App Runner to finish deploying (check AWS Console)
- Verify service is ACTIVE: `aws apprunner describe-service --service-arn <arn>`

**Docker push fails:**
- Check ECR login: `aws ecr get-login-password --region ap-northeast-1 | docker login ...`
- Verify credentials: `aws sts get-caller-identity`

**Service stays in CREATING state:**
- Check logs in App Runner console
- Common cause: insufficient compute (1 vCPU / 2 GB minimum)
