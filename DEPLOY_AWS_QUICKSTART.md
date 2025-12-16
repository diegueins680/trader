# AWS Deployment - Quick Start (5 Minutes)

## Prerequisites
- AWS account with credentials configured: `aws configure`
- Docker installed and running
- ~$1-2 to test the deployment

## Option 1: Automated Deployment (Recommended)

```bash
cd /Users/diegosaa/GitHub/trader

# Generate a random API token
API_TOKEN=$(openssl rand -hex 32)

# Run the automated deployment script
bash deploy-aws-quick.sh ap-northeast-1 "$API_TOKEN"
```

The script will:
1. ✅ Create ECR repository
2. ✅ Build Docker image
3. ✅ Push to ECR
4. ✅ Create (or reuse) the App Runner ECR access IAM role
5. ✅ Create/update App Runner service (single-instance)
6. ✅ Return the public API URL

**Total time: 5-10 minutes**

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
AWS_REGION=$AWS_REGION ECR_REPO=$ECR_REPO bash deploy/aws/create-ecr-repo.sh
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
   ```
6. Click **Create & deploy** (wait 5-10 min)

---

## Test the API

```bash
export API_URL=https://xxxx.ap-northeast-1.apprunner.amazonaws.com
export TRADER_API_TOKEN=<your-token>

# No auth needed
curl -s "${API_URL}/health"

# With auth
curl -s -H "Authorization: Bearer ${TRADER_API_TOKEN}" "${API_URL}/health"
```

---

## Deploy Web UI (Optional)

```bash
cd haskell/web

# Build pointing at your API
TRADER_API_TARGET="${API_URL}" npm run build

# Create S3 bucket
export S3_BUCKET=trader-ui-$(date +%s)
aws s3 mb "s3://${S3_BUCKET}" --region ap-northeast-1

# Upload UI
aws s3 sync dist/ "s3://${S3_BUCKET}/" --delete

# Make public (optional)
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
