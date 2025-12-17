#!/usr/bin/env bash
# Quick AWS deployment script for Trader
# Usage: bash deploy-aws-quick.sh [region] [api-token]
# Example: bash deploy-aws-quick.sh ap-northeast-1 $(openssl rand -hex 32)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="${1:-ap-northeast-1}"
TRADER_API_TOKEN="${2:-}"
ECR_REPO="trader-api"
APP_RUNNER_SERVICE_NAME="${APP_RUNNER_SERVICE_NAME:-$ECR_REPO}"
APP_RUNNER_ECR_ACCESS_ROLE_NAME="${APP_RUNNER_ECR_ACCESS_ROLE_NAME:-AppRunnerECRAccessRole}"

echo -e "${GREEN}=== Trader AWS Deployment Script ===${NC}\n"

ECR_URI=""
APP_RUNNER_SERVICE_URL=""

# Check prerequisites
check_prerequisites() {
  echo "Checking prerequisites..."
  
  if ! command -v aws >/dev/null 2>&1; then
    echo -e "${RED}✗ AWS CLI not found${NC}"
    echo "Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
  fi
  echo -e "${GREEN}✓ AWS CLI found${NC}"
  
  if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker not found${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ Docker found${NC}"
  
  # Check AWS credentials
  if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo -e "${RED}✗ AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
  fi
  echo -e "${GREEN}✓ AWS credentials valid${NC}\n"
}

# Get AWS account ID
get_account_id() {
  aws sts get-caller-identity --query Account --output text
}

wait_for_apprunner_running() {
  local service_arn="$1"
  local max_attempts="${2:-90}"
  local attempt=0
  local status=""

  while [[ $attempt -lt $max_attempts ]]; do
    status="$(
      aws apprunner describe-service \
        --service-arn "$service_arn" \
        --region "$AWS_REGION" \
        --query 'Service.Status' \
        --output text 2>/dev/null || echo ""
    )"

    if [[ "$status" == "RUNNING" ]]; then
      echo -e "${GREEN}✓ Service is RUNNING${NC}" >&2
      return 0
    fi
    if [[ "$status" == "CREATE_FAILED" || "$status" == "DELETE_FAILED" ]]; then
      echo -e "${RED}✗ Service status: ${status}${NC}" >&2
      echo "Check App Runner events/logs in the AWS Console for details." >&2
      return 1
    fi

    echo -n "." >&2
    sleep 5
    attempt=$((attempt + 1))
  done
  echo "" >&2
  echo "Timed out waiting for App Runner service to be RUNNING." >&2
  return 1
}

# Create ECR repository
create_ecr_repo() {
  set -euo pipefail
  echo "Creating ECR repository..." >&2
  
  local account_id=$(get_account_id)
  local ecr_uri="${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
  
  if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo -e "${YELLOW}✓ ECR repository already exists${NC}" >&2
  else
    aws ecr create-repository \
      --repository-name "$ECR_REPO" \
      --image-scanning-configuration scanOnPush=true \
      --region "$AWS_REGION" >/dev/null
    echo -e "${GREEN}✓ ECR repository created${NC}" >&2
  fi
  
  ECR_URI="$ecr_uri"
}

# Build and push Docker image
build_and_push() {
  local ecr_uri="$1"
  
  echo "Building Docker image..."
  docker build -t "${ECR_REPO}:latest" .
  echo -e "${GREEN}✓ Image built${NC}"
  
  echo "Logging in to ECR..."
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ecr_uri%/*}" >/dev/null 2>&1
  echo -e "${GREEN}✓ Logged in to ECR${NC}"
  
  echo "Tagging and pushing image..."
  docker tag "${ECR_REPO}:latest" "${ecr_uri}:latest"
  docker push "${ecr_uri}:latest" >/dev/null
  echo -e "${GREEN}✓ Image pushed to ${ecr_uri}:latest${NC}\n"
}

# Create App Runner service
create_app_runner() {
  set -euo pipefail
  local ecr_uri="$1"
  local image_identifier="${ecr_uri}:latest"
  
  echo "Creating App Runner service..." >&2
  
  # Find existing service (if any)
  local existing_service_arn=""
  existing_service_arn="$(
    aws apprunner list-services \
      --region "$AWS_REGION" \
      --query 'ServiceSummaryList[?ServiceName==`'"${APP_RUNNER_SERVICE_NAME}"'`].ServiceArn | [0]' \
      --output text 2>/dev/null || true
  )"
  if [[ "$existing_service_arn" == "None" ]]; then
    existing_service_arn=""
  fi

  echo "Ensuring App Runner ECR access role..." >&2
  local access_role_arn=""
  access_role_arn="$(ensure_apprunner_ecr_access_role)"
  echo -e "${GREEN}✓ Using ECR access role: ${access_role_arn}${NC}" >&2

  # Create source-configuration JSON (file:// is the most reliable for AWS CLI JSON input)
  local src_cfg
  src_cfg="$(mktemp)"

  local runtime_env_json='"TRADER_API_ASYNC_DIR":"/var/lib/trader/async"'
  if [[ -n "$TRADER_API_TOKEN" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_API_TOKEN\":\"${TRADER_API_TOKEN}\""
  fi

  cat >"$src_cfg" <<EOF
{
  "AuthenticationConfiguration": { "AccessRoleArn": "${access_role_arn}" },
  "AutoDeploymentsEnabled": true,
  "ImageRepository": {
    "ImageIdentifier": "${image_identifier}",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8080",
      "RuntimeEnvironmentVariables": { ${runtime_env_json} }
    }
  }
}
EOF

  local health_cfg="Protocol=HTTP,Path=/health"

  local service_arn=""
  if [[ -n "$existing_service_arn" ]]; then
    echo -e "${YELLOW}✓ App Runner service already exists (${APP_RUNNER_SERVICE_NAME})${NC}" >&2
    service_arn="$existing_service_arn"

    echo "Waiting for any in-progress operation to finish..." >&2
    wait_for_apprunner_running "$service_arn" >/dev/null

    echo "Updating service configuration..." >&2
    aws apprunner update-service \
      --region "$AWS_REGION" \
      --service-arn "$service_arn" \
      --source-configuration "file://${src_cfg}" \
      --instance-configuration "Cpu=1024,Memory=2048" \
      --health-check-configuration "$health_cfg" \
      >/dev/null

    echo "Waiting for update deployment to complete..." >&2
    wait_for_apprunner_running "$service_arn" >/dev/null
  else
    echo "Creating service..." >&2
    service_arn="$(aws apprunner create-service \
      --region "$AWS_REGION" \
      --service-name "$APP_RUNNER_SERVICE_NAME" \
      --source-configuration "file://${src_cfg}" \
      --instance-configuration "Cpu=1024,Memory=2048" \
      --health-check-configuration "$health_cfg" \
      --tags Key=Name,Value=trader-api \
      --query 'Service.ServiceArn' \
      --output text)"
    echo -e "${GREEN}✓ App Runner service created${NC}" >&2

    echo "Waiting for service to be RUNNING (this may take a few minutes)..." >&2
    wait_for_apprunner_running "$service_arn" >/dev/null
  fi

  rm -f "$src_cfg"

  echo "Setting single-instance scaling (min=1, max=1)..." >&2
  AWS_REGION="$AWS_REGION" bash deploy/aws/set-app-runner-single-instance.sh --service-arn "$service_arn" --min 1 --max 1 >/dev/null
  echo -e "${GREEN}✓ Scaling updated${NC}" >&2

  echo "Waiting for service to be RUNNING (this may take a few minutes)..." >&2
  wait_for_apprunner_running "$service_arn" >/dev/null

  local service_host
  service_host="$(aws apprunner describe-service --service-arn "$service_arn" --region "$AWS_REGION" --query 'Service.ServiceUrl' --output text)"
  if [[ "$service_host" == http* ]]; then
    APP_RUNNER_SERVICE_URL="$service_host"
  else
    APP_RUNNER_SERVICE_URL="https://${service_host}"
  fi
}

# Ensure an IAM role exists for App Runner to pull from private ECR.
# https://docs.aws.amazon.com/apprunner/latest/dg/security_iam_service-role.html
ensure_apprunner_ecr_access_role() {
  set -euo pipefail
  local role_name="$APP_RUNNER_ECR_ACCESS_ROLE_NAME"
  local policy_arn="arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"

  local role_arn=""
  role_arn="$(aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || true)"
  if [[ -n "$role_arn" && "$role_arn" != "None" ]]; then
    aws iam attach-role-policy --role-name "$role_name" --policy-arn "$policy_arn" >/dev/null 2>&1 || true
    echo "$role_arn"
    return 0
  fi

  echo "Creating IAM role: ${role_name}" >&2

  local trust_doc
  trust_doc="$(mktemp)"
  cat >"$trust_doc" <<'EOF'
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

  aws iam create-role \
    --role-name "$role_name" \
    --assume-role-policy-document "file://${trust_doc}" \
    --description "App Runner ECR access role (created by trader deploy script)" \
    >/dev/null

  aws iam attach-role-policy --role-name "$role_name" --policy-arn "$policy_arn" >/dev/null

  rm -f "$trust_doc"

  # IAM is eventually consistent; give it a moment.
  for _ in {1..12}; do
    role_arn="$(aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || true)"
    if [[ -n "$role_arn" && "$role_arn" != "None" ]]; then
      echo "$role_arn"
      return 0
    fi
    sleep 2
  done

  echo "Error: failed to create or read IAM role '${role_name}'." >&2
  exit 1
}

# Main execution
main() {
  check_prerequisites
  
  echo "Configuration:"
  echo "  Region: $AWS_REGION"
  echo "  API Token: ${TRADER_API_TOKEN:-(not set)}"
  echo ""
  
  # Create ECR repo
  create_ecr_repo
  local ecr_uri="$ECR_URI"
  
  # Build and push image
  build_and_push "$ecr_uri"
  
  # Create App Runner service and get URL
  create_app_runner "$ecr_uri"
  local service_url="$APP_RUNNER_SERVICE_URL"
  
  echo -e "${GREEN}=== Deployment Complete ===${NC}\n"
  echo "API URL: ${service_url}"
  echo ""
  echo "Next steps:"
  echo "1. Test the API:"
  echo "   curl -s ${service_url}/health | jq ."
  echo ""
  echo "2. Build and deploy the web UI:"
  echo "   cd haskell/web"
  echo "   TRADER_API_TARGET='${service_url}' npm run build"
  echo "   # Then upload dist/ to S3 or CloudFront"
  echo ""
  if [[ -z "$TRADER_API_TOKEN" ]]; then
    echo -e "${YELLOW}⚠ No API token set. For security, set TRADER_API_TOKEN in App Runner environment.${NC}"
  else
    echo "3. Use this token in the UI: ${TRADER_API_TOKEN}"
  fi
}

main "$@"
