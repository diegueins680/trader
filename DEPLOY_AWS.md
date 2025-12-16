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
   cd /Users/diegosaa/GitHub/trader
   ```

---

## Step 1: Create ECR Repository

Create an Amazon Elastic Container Registry (ECR) repository for the Docker image.

```bash
export AWS_REGION=ap-northeast-1          # Change to your region (e.g., us-east-1, eu-west-1)
export ECR_REPO=trader-api

# Run the helper script
bash deploy/aws/create-ecr-repo.sh

# Expected output:
# 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/trader-api
```

Save the output (ECR URI) — you'll need it in the next step.

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
   ```
   Generate a token:
   ```bash
   openssl rand -hex 32
   ```
7. Click **Create & deploy**

**Option B: AWS CLI** (faster after setup)

```bash
export AWS_REGION=ap-northeast-1
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/trader-api:latest"

# Create the service (adjust CPU/memory as needed)
aws apprunner create-service \
  --region "$AWS_REGION" \
  --service-name trader-api \
  --source-configuration '{"ImageRepository":{"ImageIdentifier":"'"$ECR_URI"'","ImageRepositoryType":"ECR","ImageConfiguration":{"Port":"8080"}}}' \
  --instance-configuration '{"Cpu":"1024","Memory":"2048"}' \
  --tags Key=Name,Value=trader-api
```

**Note the Service URL** (you'll see it after deployment completes, format: `https://xxxx.ap-northeast-1.apprunner.amazonaws.com`)

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

## Step 5: Test the API

Once the service is deployed and running:

```bash
export APP_RUNNER_URL=https://xxxx.ap-northeast-1.apprunner.amazonaws.com
export TRADER_API_TOKEN=<your-token>

# Test health endpoint (no auth needed)
curl -s "${APP_RUNNER_URL}/health" | jq .

# Test a protected endpoint (requires token)
curl -s -H "Authorization: Bearer ${TRADER_API_TOKEN}" \
  "${APP_RUNNER_URL}/health" | jq .
```

Expected response:
```json
{
  "healthy": true,
  "computeLimits": { ... }
}
```

---

## Step 6: Build & Deploy Web UI

The web UI is a static React app. Build it pointing at your deployed API:

```bash
cd haskell/web

export TRADER_API_TARGET="https://xxxx.ap-northeast-1.apprunner.amazonaws.com"

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

**Option A: Built-in (already done in Step 6)**
- The UI knows the API URL from the build: `TRADER_API_TARGET`

**Option B: Runtime Override (in the UI)**
- Users can paste the API base URL into the "API base URL" field in the UI
- This is stored in local storage and persists across sessions

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
AWS_REGION=ap-northeast-1 bash deploy/aws/set-app-runner-single-instance.sh \
  --service-arn <arn> --min 1 --max 1

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
- Verify the API URL in the UI's "API base URL" field
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
2. **Backup async jobs:** If using stateful endpoints (`/bot/*`), ensure `TRADER_API_ASYNC_DIR` is on persistent storage (EFS)
3. **Setup CI/CD:** Use GitHub Actions to automatically build & push on commit
4. **Enable HTTPS:** Use CloudFront + ACM certificate for your domain
5. **Security:** Use IAM roles, VPC security groups, and API Gateway if needed

---

## Support

For detailed info, see:
- `deploy/aws/README.md` in this repo
- [AWS App Runner docs](https://docs.aws.amazon.com/apprunner/)
- [AWS ECR docs](https://docs.aws.amazon.com/ecr/)
