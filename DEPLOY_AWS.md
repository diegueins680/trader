# AWS Deployment Guide for Trader

This guide walks you through deploying the Trader API and UI to AWS using App Runner and S3/CloudFront.

## Prerequisites

1. **AWS Account** with credentials configured locally
   ```bash
   aws configure
   ```

2. **Docker** installed locally

3. **AWS CLI** installed
   ```bash
   # macOS
   brew install awscli
   
   # or from: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
   ```

4. **Repository cloned** locally
   ```bash
   cd /path/to/trader
   ```

---

## Step 1: Create ECR Repository

Create an Amazon Elastic Container Registry (ECR) repository for the Docker image.

```bash
export AWS_REGION=ap-northeast-1          # Change to your region (e.g., us-east-1, eu-west-1)
export ECR_REPO=trader-api

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --image-scanning-configuration scanOnPush=true \
    --region "$AWS_REGION" >/dev/null
```

**Or manually via AWS Console:**
- AWS Console → **ECR** → **Create repository**
- Name: `trader-api`
- Click **Create repository**

---

## Step 2: Build & Push Docker Image

Login to ECR and push the image:

```bash
export AWS_REGION=ap-northeast-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=trader-api
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

# Login to ECR
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Build the image
docker build -t "${ECR_REPO}:latest" .

# Tag for ECR
docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"

# Push to ECR
docker push "${ECR_URI}:latest"

echo "Image pushed to: $ECR_URI:latest"
```

This builds the Haskell backend and packages it in a Docker container.

---

## Step 3: Create App Runner Service

Create an AWS App Runner service to run your API.

**Option A: AWS Console**

1. Go to **AWS Console** → **App Runner**
2. Click **Create service**
3. Configure:
   - **Source:** Container registry → **Amazon ECR**
   - **Repository:** Select `trader-api`
   - **Image tag:** `latest`
   - **Image URI:** Should auto-populate
   - **Port:** `8080`
   - **Health check path:** `/health`
4. Click **Next**
5. **Service settings:**
   - **Service name:** `trader-api`
   - **CPU:** 1 vCPU
   - **Memory:** 2 GB (or higher if needed)
   - **Scaling:**
     - **Min size:** 1
     - **Max size:** 1  (keep single-instance unless you have shared async storage)
6. **Environment variables** (optional but recommended):
   ```
   TRADER_API_TOKEN=<your-random-token>
   TRADER_STATE_S3_BUCKET=<s3-bucket>
   TRADER_STATE_S3_PREFIX=trader
   TRADER_STATE_S3_REGION=ap-northeast-1
   ```
   Generate a token:
   ```bash
   openssl rand -hex 32
   ```
7. Click **Create & deploy**

**Option B: AWS CLI** (faster after setup)

```bash
export AWS_REGION=ap-northeast-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/trader-api:latest"

# One-time (per account): App Runner needs an IAM role to pull from private ECR
ROLE_NAME=AppRunnerECRAccessRole
POLICY_ARN=arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess

cat > apprunner-ecr-trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "build.apprunner.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://apprunner-ecr-trust.json >/dev/null
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN" >/dev/null
ECR_ACCESS_ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

# Create the service (adjust CPU/memory as needed)
cat > apprunner-source.json <<EOF
{
  "AuthenticationConfiguration": { "AccessRoleArn": "${ECR_ACCESS_ROLE_ARN}" },
  "AutoDeploymentsEnabled": true,
  "ImageRepository": {
    "ImageIdentifier": "${ECR_URI}",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": { "Port": "8080" }
  }
}
EOF

aws apprunner create-service \
  --region "$AWS_REGION" \
  --service-name trader-api \
  --source-configuration file://apprunner-source.json \
  --health-check-configuration 'Protocol=HTTP,Path=/health' \
  --instance-configuration 'Cpu=1024,Memory=2048' \
  --tags Key=Name,Value=trader-api
```

**Note the Service URL** (you'll see it after deployment completes; `aws apprunner describe-service --service-arn <arn> --query 'Service.ServiceUrl' --output text` returns a host you can prefix with `https://`.)

---

## Step 4: Set Environment Variables (Optional but Recommended)

If you didn't set `TRADER_API_TOKEN` during creation, add it now:

1. Go to **App Runner** → **Services** → **trader-api**
2. Click **Configuration** → **Edit**
3. Scroll to **Environment variables**
4. Add:
   ```
   TRADER_API_TOKEN=<your-random-token>
   ```
5. Click **Save changes**

The service will redeploy.

---

## Optional: Enable S3 State Persistence (App Runner)

App Runner does **not** support EFS volumes. To persist bot snapshots and optimizer top-combos, store them in S3:

1. Create a private S3 bucket (e.g. `trader-api-state-...`).
2. Create an App Runner instance role with S3 access (trust `tasks.apprunner.amazonaws.com`).
3. Set the instance role on the service and add env vars:
   ```
   TRADER_STATE_S3_BUCKET=<s3-bucket>
   TRADER_STATE_S3_PREFIX=trader
   TRADER_STATE_S3_REGION=ap-northeast-1
   ```

If you use `deploy-aws-quick.sh`, pass `--state-s3-bucket ... --state-s3-prefix ... --instance-role-arn ...`.

---

## Step 5: Test the API

Once the service is deployed and running:

```bash
export APP_RUNNER_URL=https://xxxx.ap-northeast-1.awsapprunner.com
export TRADER_API_TOKEN=<your-token>

# Test health endpoint (no auth needed)
curl -s "${APP_RUNNER_URL}/health" | jq .

# Verify token wiring (/health is always public)
curl -s -H "Authorization: Bearer ${TRADER_API_TOKEN}" \
  "${APP_RUNNER_URL}/health" | jq .
```

Expected response notes:
- `authRequired` is `true` when `TRADER_API_TOKEN` is configured on the service.
- `authOk` is `false` without the auth header, and `true` when you pass the correct token.

Example response:
```json
{
  "status": "ok",
  "authRequired": true,
  "authOk": true,
  "computeLimits": { ... },
  "asyncJobs": { ... }
}
```

---

## Step 6: Build & Deploy Web UI

The web UI is a static React app. Build it pointing at your deployed API:

```bash
cd haskell/web

export TRADER_API_TARGET="https://xxxx.ap-northeast-1.awsapprunner.com"

# Install dependencies (if needed)
npm install

# Build with your API URL
TRADER_API_TARGET="$TRADER_API_TARGET" npm run build

# Verify dist/ folder was created
ls -la dist/
```

---

## Step 7: Deploy Web UI to S3

Create an S3 bucket and upload the UI:

```bash
export AWS_REGION=ap-northeast-1
export S3_BUCKET=trader-ui-$(date +%s)  # Unique bucket name

# Create S3 bucket
aws s3 mb "s3://${S3_BUCKET}" --region "$AWS_REGION"

# Enable static website hosting
aws s3 website "s3://${S3_BUCKET}" \
  --index-document index.html \
  --error-document index.html

# Upload the UI
aws s3 sync dist/ "s3://${S3_BUCKET}/" --delete

# Make objects public (optional, if you don't use CloudFront)
#
# Note: new buckets default to "Block Public Access", which will prevent public bucket policies.
# Only do this if you intentionally want a public website bucket.
aws s3api put-public-access-block \
  --bucket "$S3_BUCKET" \
  --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false \
  --region "$AWS_REGION" || true

aws s3api put-bucket-policy --bucket "$S3_BUCKET" --policy '{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::'"$S3_BUCKET"'/*"
  }]
}'

echo "UI deployed to: http://${S3_BUCKET}.s3-website-${AWS_REGION}.amazonaws.com"
```

---

## Step 8: (Optional) Setup CloudFront for HTTPS

For production, use CloudFront to serve the UI over HTTPS:

1. Go to **CloudFront** → **Create distribution**
2. **Origin domain:** Select your S3 bucket
3. **Viewer protocol policy:** Redirect HTTP to HTTPS
4. **Default root object:** `index.html`
5. **Custom error responses:**
   - Add: 403 error → `index.html` (for SPA routing)
   - Add: 404 error → `index.html`
6. Create the distribution

After it's deployed (5-10 minutes):
- Access your UI at: `https://dxxx.cloudfront.net`
- Create a CNAME for your domain (e.g., `trader.example.com`)

---

## Step 9: Update UI to Point to API

The UI can discover the API in two ways:

**Option A: CloudFront `/api/*` proxy (recommended)**
- Configure CloudFront to forward `/api/*` to your API origin (App Runner/ALB/etc)
- The UI must use `/api` (no extra UI config needed); `deploy-aws-quick.sh` forces this when a distribution ID is provided unless `--ui-api-direct`/`TRADER_UI_API_MODE=direct` is set (which keeps the full API URL and relies on CORS). Use `apiFallbackUrl` only if your API explicitly supports cross-origin requests.

**Option B: Deploy-time config file**
- Edit `haskell/web/public/trader-config.js` (or `haskell/web/dist/trader-config.js` after build) before uploading to S3:
  - `apiBaseUrl`: `/api` when CloudFront proxies `/api/*`, otherwise `https://<your-api-host>`
  - `apiToken`: the same value as backend `TRADER_API_TOKEN` (optional)

CloudFront is non-sticky. If you run multiple backend instances, either keep it single-instance or ensure `TRADER_API_ASYNC_DIR` (or `TRADER_STATE_DIR`) points to a shared writable directory so async job polling works across instances.

---

## Complete Deployment Checklist

- [ ] AWS credentials configured (`aws configure`)
- [ ] ECR repository created
- [ ] Docker image built and pushed
- [ ] App Runner service deployed
- [ ] API tested with `/health` endpoint
- [ ] Web UI built with `TRADER_API_TARGET`
- [ ] UI uploaded to S3
- [ ] (Optional) CloudFront distribution created for HTTPS
- [ ] Both API and UI tested end-to-end

---

## Useful Commands

```bash
# View App Runner service status
aws apprunner describe-service --service-arn <arn> --region ap-northeast-1

# View recent logs
aws apprunner describe-service --service-arn <arn> --region ap-northeast-1 \
  --query 'Service.ServiceSummary' --output json

# Redeploy (force pull latest image)
aws apprunner start-deployment --service-arn <arn> --region ap-northeast-1

# Set single-instance scaling
# Note: `deploy-aws-quick.sh` sets min=1/max=1 automatically on API deploy.
# If you need to adjust scaling manually, do it in the App Runner console.

# View ECR images
aws ecr describe-images --repository-name trader-api --region ap-northeast-1

# Invalidate CloudFront cache (after UI update)
aws cloudfront create-invalidation --distribution-id <id> --paths "/*"
```

---

## Troubleshooting

**API not responding:**
- Check App Runner service is running: `aws apprunner describe-service --service-arn <arn>`
- View logs in the Console or via CloudWatch
- Ensure `TRADER_API_TOKEN` is set if you require auth

**UI says "API unreachable":**
- Verify `trader-config.js` (`apiBaseUrl`) and/or your CloudFront `/api/*` proxy
- Ensure the API is publicly accessible (App Runner gives you a public URL by default)
- Check CORS if using CloudFront (allow `Content-Type`, `Authorization`, `X-API-Key` headers)

**Docker build fails:**
- Ensure Docker daemon is running: `docker ps`
- Check available disk space: `df -h`
- Try: `docker build --progress=plain .` for verbose output

**Image push fails:**
- Verify ECR login: `aws ecr get-login-password ... | docker login ...`
- Check AWS credentials: `aws sts get-caller-identity`
- Ensure ECR repository exists

---

## Cost Estimation (AWS App Runner)

- **Compute:** ~$0.10–0.20 per hour per vCPU
- **Memory:** ~$0.01–0.02 per hour per GB
- **Example (1 vCPU, 2 GB):** ~$50–70/month (if always running)
- **S3:** ~$0.023 per GB stored (UI is ~1 MB)
- **CloudFront:** ~$0.085 per GB outbound

Use CloudWatch to monitor and set up auto-scaling if you expect variable traffic.

---

## Next Steps

1. **Monitor the API:** Set up CloudWatch dashboards and alarms
2. **Persist state:** For App Runner, use S3-backed persistence (`TRADER_STATE_S3_BUCKET`) and grant the service an instance role with S3 access.
   - App Runner does **not** support EFS volumes; S3 is the supported persistence option.
3. **Setup CI/CD:** Use GitHub Actions to automatically build & push on commit
4. **Enable HTTPS:** Use CloudFront + ACM certificate for your domain
5. **Security:** Use IAM roles, VPC security groups, and API Gateway if needed

---

## Support

For detailed info, see:
- `deploy/aws/README.md` in this repo
- [AWS App Runner docs](https://docs.aws.amazon.com/apprunner/)
- [AWS ECR docs](https://docs.aws.amazon.com/ecr/)
