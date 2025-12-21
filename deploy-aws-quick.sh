#!/usr/bin/env bash
# AWS deployment script for Trader (API + optional UI)
#
# Quick usage (API only):
#   bash deploy-aws-quick.sh [region] [api-token]
#   TRADER_API_TOKEN=... bash deploy-aws-quick.sh [region]
#
# Optional: also deploy the web UI (S3 + optional CloudFront invalidation):
#   bash deploy-aws-quick.sh --region ap-northeast-1 --api-token "$API_TOKEN" --ui-bucket "$S3_BUCKET" --distribution-id "$CF_ID"
#
# Notes:
#   - If region is omitted, uses AWS_REGION/AWS_DEFAULT_REGION/aws-cli config, then ap-northeast-1.
#   - If api-token is omitted, uses env TRADER_API_TOKEN (or reuses an existing App Runner token when updating).

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration (inputs)
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
TRADER_API_TOKEN="${TRADER_API_TOKEN:-}"
TRADER_API_MAX_BARS_LSTM="${TRADER_API_MAX_BARS_LSTM:-1000}"
UI_BUCKET="${TRADER_UI_BUCKET:-${S3_BUCKET:-}}"
UI_DISTRIBUTION_ID="${TRADER_UI_CLOUDFRONT_DISTRIBUTION_ID:-${CLOUDFRONT_DISTRIBUTION_ID:-}}"
UI_SKIP_BUILD="${TRADER_UI_SKIP_BUILD:-false}"
UI_DIST_DIR="${TRADER_UI_DIST_DIR:-haskell/web/dist}"
UI_ONLY="false"
API_ONLY="false"
UI_API_URL="${TRADER_UI_API_URL:-}"
UI_SERVICE_ARN="${TRADER_UI_SERVICE_ARN:-}"

# Configuration (defaults)
ECR_REPO="trader-api"
APP_RUNNER_SERVICE_NAME="${APP_RUNNER_SERVICE_NAME:-$ECR_REPO}"
APP_RUNNER_ECR_ACCESS_ROLE_NAME="${APP_RUNNER_ECR_ACCESS_ROLE_NAME:-AppRunnerECRAccessRole}"

ECR_URI=""
APP_RUNNER_SERVICE_URL=""

usage() {
  cat <<'EOF'
Usage:
  # API deploy (ECR + App Runner)
  bash deploy-aws-quick.sh [region] [api-token]

  # API deploy (named flags)
  bash deploy-aws-quick.sh --region <region> [--api-token <token>]

  # API + UI deploy (S3 + optional CloudFront invalidation)
  bash deploy-aws-quick.sh --region <region> [--api-token <token>] --ui-bucket <s3-bucket> [--distribution-id <cf-distribution-id>]

  # UI-only deploy (no App Runner changes)
  bash deploy-aws-quick.sh --ui-only --region <region> --ui-bucket <s3-bucket> --api-url <https://api-host> [--api-token <token>] [--distribution-id <cf-distribution-id>] [--skip-ui-build]

Flags:
  --region <region>                 AWS region (e.g. ap-northeast-1)
  --api-token <token>               API token (TRADER_API_TOKEN)
  --api-only                         Deploy API only
  --ui-only                          Deploy UI only (requires --ui-bucket and --api-url or --service-arn)
  --ui-bucket|--bucket <bucket>     S3 bucket to upload UI to
  --distribution-id <id>            CloudFront distribution ID (optional)
  --api-url <url>                   API base URL for UI config (UI-only; otherwise auto from App Runner)
  --service-arn <arn>               App Runner service ARN to auto-discover API URL/token (UI-only convenience)
  --skip-ui-build                   Skip `npm run build` (uses existing dist dir)
  --ui-dist-dir <dir>               UI dist dir (default: haskell/web/dist)
  -h|--help                         Show help

Environment variables (equivalents):
  AWS_REGION / AWS_DEFAULT_REGION
  TRADER_API_TOKEN
  TRADER_API_MAX_BARS_LSTM
  TRADER_UI_BUCKET / S3_BUCKET
  TRADER_UI_CLOUDFRONT_DISTRIBUTION_ID / CLOUDFRONT_DISTRIBUTION_ID
  TRADER_UI_SKIP_BUILD
  TRADER_UI_DIST_DIR
  TRADER_UI_API_URL
  TRADER_UI_SERVICE_ARN
EOF
}

mask_token() {
  local tok="${1:-}"
  if [[ -z "$tok" ]]; then
    echo "(not set)"
    return 0
  fi
  local n=${#tok}
  if [[ $n -le 12 ]]; then
    echo "${tok:0:2}…${tok:n-2:2}"
    return 0
  fi
  echo "${tok:0:6}…${tok:n-4:4}"
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --region)
      AWS_REGION="${2:-}"
      shift 2
      ;;
    --api-token)
      TRADER_API_TOKEN="${2:-}"
      shift 2
      ;;
    --api-only)
      API_ONLY="true"
      shift
      ;;
    --ui-only)
      UI_ONLY="true"
      shift
      ;;
    --ui-bucket|--bucket)
      UI_BUCKET="${2:-}"
      shift 2
      ;;
    --distribution-id)
      UI_DISTRIBUTION_ID="${2:-}"
      shift 2
      ;;
    --skip-ui-build)
      UI_SKIP_BUILD="true"
      shift
      ;;
    --ui-dist-dir)
      UI_DIST_DIR="${2:-}"
      shift 2
      ;;
    --api-url)
      UI_API_URL="${2:-}"
      shift 2
      ;;
    --service-arn)
      UI_SERVICE_ARN="${2:-}"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 && -z "${AWS_REGION:-}" ]]; then
  AWS_REGION="${POSITIONAL[0]}"
fi
if [[ ${#POSITIONAL[@]} -ge 2 && -z "${TRADER_API_TOKEN:-}" ]]; then
  TRADER_API_TOKEN="${POSITIONAL[1]}"
fi

DEPLOY_API="true"
DEPLOY_UI="false"
if [[ -n "${UI_BUCKET:-}" || "$UI_ONLY" == "true" ]]; then
  DEPLOY_UI="true"
fi
if [[ "$API_ONLY" == "true" ]]; then
  DEPLOY_UI="false"
fi
if [[ "$UI_ONLY" == "true" ]]; then
  DEPLOY_API="false"
  DEPLOY_UI="true"
fi

echo -e "${GREEN}=== Trader AWS Deployment Script ===${NC}\n"

# Check prerequisites
check_prerequisites() {
  local require_docker="${1:-true}"
  local require_npm="${2:-false}"

  echo "Checking prerequisites..."
  
  if ! command -v aws >/dev/null 2>&1; then
    echo -e "${RED}✗ AWS CLI not found${NC}"
    echo "Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
  fi
  echo -e "${GREEN}✓ AWS CLI found${NC}"

  # Resolve region early so AWS CLI commands that require a region (including STS) work.
  if [[ -z "${AWS_REGION:-}" ]]; then
    AWS_REGION="$(aws configure get region 2>/dev/null || true)"
  fi
  AWS_REGION="${AWS_REGION:-ap-northeast-1}"
  if [[ "$AWS_REGION" =~ [a-z]$ ]]; then
    echo -e "${RED}✗ AWS_REGION '$AWS_REGION' looks like an Availability Zone (e.g. ap-northeast-1a). Use a region like ap-northeast-1.${NC}" >&2
    exit 1
  fi
  
  if [[ "$require_docker" == "true" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      echo -e "${RED}✗ Docker not found${NC}"
      exit 1
    fi
    echo -e "${GREEN}✓ Docker found${NC}"

    # Check Docker daemon connectivity early (otherwise docker build/push errors are confusing).
    if ! docker ps >/dev/null 2>&1; then
      echo -e "${RED}✗ Docker daemon not reachable${NC}"
      echo "Start Docker Desktop (or ensure dockerd is running) and retry."
      exit 1
    fi
    echo -e "${GREEN}✓ Docker daemon reachable${NC}"
  fi

  if [[ "$require_npm" == "true" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo -e "${RED}✗ npm not found${NC}" >&2
      echo "Install Node.js (>=18) and retry." >&2
      exit 1
    fi
    echo -e "${GREEN}✓ npm found${NC}"
  fi
  
  # Check AWS credentials
  if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
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

discover_apprunner_service_url() {
  local service_arn="$1"
  local host=""
  host="$(
    aws apprunner describe-service \
      --service-arn "$service_arn" \
      --region "$AWS_REGION" \
      --query 'Service.ServiceUrl' \
      --output text
  )"
  if [[ -z "$host" || "$host" == "None" ]]; then
    echo "" >&2
    return 1
  fi
  if [[ "$host" =~ ^https?:// ]]; then
    echo "$host"
  else
    echo "https://${host}"
  fi
}

discover_apprunner_trader_api_token() {
  local service_arn="$1"
  local token=""
  token="$(
    aws apprunner describe-service \
      --service-arn "$service_arn" \
      --region "$AWS_REGION" \
      --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_API_TOKEN' \
      --output text 2>/dev/null || true
  )"
  if [[ "$token" == "None" ]]; then
    token=""
  fi
  echo "$token"
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

set_apprunner_scaling() {
  local service_arn="$1"
  local min_size="${2:-1}"
  local max_size="${3:-1}"

  local current_cfg_arn=""
  current_cfg_arn="$(
    aws apprunner describe-service \
      --service-arn "$service_arn" \
      --region "$AWS_REGION" \
      --query 'Service.AutoScalingConfigurationSummary.AutoScalingConfigurationArn' \
      --output text 2>/dev/null || true
  )"
  if [[ "$current_cfg_arn" == "None" ]]; then
    current_cfg_arn=""
  fi

  local current_min=""
  local current_max=""
  if [[ -n "$current_cfg_arn" ]]; then
    current_min="$(
      aws apprunner describe-auto-scaling-configuration \
        --auto-scaling-configuration-arn "$current_cfg_arn" \
        --region "$AWS_REGION" \
        --query 'AutoScalingConfiguration.MinSize' \
        --output text 2>/dev/null || true
    )"
    current_max="$(
      aws apprunner describe-auto-scaling-configuration \
        --auto-scaling-configuration-arn "$current_cfg_arn" \
        --region "$AWS_REGION" \
        --query 'AutoScalingConfiguration.MaxSize' \
        --output text 2>/dev/null || true
    )"
  fi
  if [[ "$current_min" == "None" ]]; then
    current_min=""
  fi
  if [[ "$current_max" == "None" ]]; then
    current_max=""
  fi

  if [[ -n "$current_min" && -n "$current_max" && "$current_min" != "None" && "$current_max" != "None" ]]; then
    if [[ "$current_min" == "$min_size" && "$current_max" == "$max_size" ]]; then
      return 0
    fi
  fi

  local ts
  ts="$(date -u +%s)"
  local name_prefix="trader-api-single"
  local cfg_name="${name_prefix}-${min_size}-${max_size}-${ts}"
  local max_len=32
  if (( ${#cfg_name} > max_len )); then
    local suffix_len=$(( ${#min_size} + ${#max_size} + ${#ts} + 3 ))
    local allowed_prefix_len=$(( max_len - suffix_len ))
    if (( allowed_prefix_len < 1 )); then
      name_prefix="t"
    else
      name_prefix="${name_prefix:0:${allowed_prefix_len}}"
    fi
    cfg_name="${name_prefix}-${min_size}-${max_size}-${ts}"
  fi

  local cfg_arn
  cfg_arn="$(
    aws apprunner create-auto-scaling-configuration \
      --region "$AWS_REGION" \
      --auto-scaling-configuration-name "$cfg_name" \
      --min-size "$min_size" \
      --max-size "$max_size" \
      --query 'AutoScalingConfiguration.AutoScalingConfigurationArn' \
      --output text
  )"

  aws apprunner update-service \
    --region "$AWS_REGION" \
    --service-arn "$service_arn" \
    --auto-scaling-configuration-arn "$cfg_arn" \
    >/dev/null
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
    | docker login --username AWS --password-stdin "${ecr_uri%/*}" >/dev/null
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

  if [[ -z "$TRADER_API_TOKEN" && -n "$existing_service_arn" ]]; then
    local existing_token=""
    existing_token="$(
      aws apprunner describe-service \
        --service-arn "$existing_service_arn" \
        --region "$AWS_REGION" \
        --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_API_TOKEN' \
        --output text 2>/dev/null || true
    )"
    if [[ "$existing_token" == "None" ]]; then
      existing_token=""
    fi
    if [[ -n "$existing_token" ]]; then
      TRADER_API_TOKEN="$existing_token"
      echo -e "${YELLOW}✓ Reusing existing TRADER_API_TOKEN from service${NC}" >&2
    fi
  fi

  echo "Ensuring App Runner ECR access role..." >&2
  local access_role_arn=""
  access_role_arn="$(ensure_apprunner_ecr_access_role)"
  echo -e "${GREEN}✓ Using ECR access role: ${access_role_arn}${NC}" >&2

  # Create source-configuration JSON (file:// is the most reliable for AWS CLI JSON input)
  local src_cfg
  src_cfg="$(mktemp)"

  local runtime_env_json='"TRADER_API_ASYNC_DIR":"/var/lib/trader/async"'
  if [[ -n "${TRADER_API_MAX_BARS_LSTM:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_API_MAX_BARS_LSTM\":\"${TRADER_API_MAX_BARS_LSTM}\""
  fi
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
  set_apprunner_scaling "$service_arn" 1 1
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

deploy_ui() {
  local api_url="$1"
  local api_token="$2"

  if [[ -z "${UI_BUCKET:-}" ]]; then
    echo -e "${RED}✗ Missing UI bucket. Provide --ui-bucket (or TRADER_UI_BUCKET / S3_BUCKET).${NC}" >&2
    usage >&2
    exit 2
  fi
  if [[ -z "${api_url:-}" ]]; then
    echo -e "${RED}✗ Missing API URL for UI config. Provide --api-url (or deploy the API in the same run).${NC}" >&2
    usage >&2
    exit 2
  fi

  local skip_build="false"
  if is_true "$UI_SKIP_BUILD"; then
    skip_build="true"
  fi

  if [[ "$skip_build" != "true" ]]; then
    echo "Building UI..." >&2
    if [[ ! -d "haskell/web/node_modules" ]]; then
      echo "Installing UI dependencies (npm ci)..." >&2
      (cd haskell/web && npm ci --no-audit --no-fund)
    fi
    (cd haskell/web && TRADER_API_TARGET="$api_url" npm run build)
    echo -e "${GREEN}✓ UI built${NC}" >&2
  fi

  if [[ ! -d "$UI_DIST_DIR" ]]; then
    echo -e "${RED}✗ UI dist dir not found: ${UI_DIST_DIR}${NC}" >&2
    echo "Build the UI first (or pass --ui-dist-dir)." >&2
    exit 1
  fi

  echo "Writing ${UI_DIST_DIR}/trader-config.js..." >&2
  cat > "${UI_DIST_DIR}/trader-config.js" <<EOF
globalThis.__TRADER_CONFIG__ = {
  apiBaseUrl: "${api_url}",
  apiToken: "${api_token}",
};
EOF

  if ! aws s3api head-bucket --bucket "$UI_BUCKET" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo -e "${RED}✗ S3 bucket not found or not accessible: ${UI_BUCKET}${NC}" >&2
    echo "Create it first, e.g.:" >&2
    echo "  aws s3 mb s3://${UI_BUCKET} --region ${AWS_REGION}" >&2
    exit 1
  fi

  echo "Uploading UI to s3://${UI_BUCKET}/ ..." >&2
  aws s3 sync "${UI_DIST_DIR}/" "s3://${UI_BUCKET}/" --delete --region "$AWS_REGION" >/dev/null
  aws s3 cp "${UI_DIST_DIR}/trader-config.js" "s3://${UI_BUCKET}/trader-config.js" --region "$AWS_REGION" >/dev/null
  echo -e "${GREEN}✓ UI uploaded${NC}" >&2

  if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
    echo "Invalidating CloudFront..." >&2
    aws cloudfront create-invalidation \
      --distribution-id "$UI_DISTRIBUTION_ID" \
      --paths "/" "/index.html" "/trader-config.js" \
      >/dev/null
    echo -e "${GREEN}✓ CloudFront invalidated${NC}" >&2
  fi
}

# Main execution
main() {
  local need_docker="false"
  local need_npm="false"
  if [[ "$DEPLOY_API" == "true" ]]; then
    need_docker="true"
  fi
  if [[ "$DEPLOY_UI" == "true" ]] && ! is_true "$UI_SKIP_BUILD"; then
    need_npm="true"
  fi

  check_prerequisites "$need_docker" "$need_npm"

  echo "Configuration:"
  echo "  Region: $AWS_REGION"
  echo "  API Token: $(mask_token "$TRADER_API_TOKEN")"
  echo "  API Max Bars (LSTM): ${TRADER_API_MAX_BARS_LSTM}"
  if [[ "$DEPLOY_UI" == "true" ]]; then
    echo "  UI Bucket: ${UI_BUCKET:-"(not set)"}"
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      echo "  UI CF Dist: ${UI_DISTRIBUTION_ID}"
    fi
    echo "  UI Dist Dir: ${UI_DIST_DIR}"
    echo "  UI Skip Build: ${UI_SKIP_BUILD}"
  fi
  echo ""

  local api_url=""
  local api_token="$TRADER_API_TOKEN"
  local ui_api_url_override="${UI_API_URL:-}"

  if [[ "$DEPLOY_API" == "true" ]]; then
    create_ecr_repo
    local ecr_uri="$ECR_URI"

    build_and_push "$ecr_uri"

    create_app_runner "$ecr_uri"
    api_url="$APP_RUNNER_SERVICE_URL"
    api_token="$TRADER_API_TOKEN"
  else
    if [[ -n "${ui_api_url_override:-}" ]]; then
      api_url="$ui_api_url_override"
    elif [[ -n "${UI_SERVICE_ARN:-}" ]]; then
      api_url="$(discover_apprunner_service_url "$UI_SERVICE_ARN")"
    fi
    if [[ -z "${api_token:-}" && -n "${UI_SERVICE_ARN:-}" ]]; then
      api_token="$(discover_apprunner_trader_api_token "$UI_SERVICE_ARN")"
    fi
  fi

  if [[ "$DEPLOY_UI" == "true" ]]; then
    local ui_api_url="$api_url"
    if [[ -z "${ui_api_url_override:-}" && -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      ui_api_url="/api"
    fi
    deploy_ui "$ui_api_url" "$api_token"
  fi

  echo -e "${GREEN}=== Deployment Complete ===${NC}\n"
  if [[ "$DEPLOY_API" == "true" ]]; then
    echo "API URL: ${api_url}"
  else
    echo "API URL (configured): ${ui_api_url}"
  fi
  if [[ "$DEPLOY_UI" == "true" ]]; then
    if [[ "$DEPLOY_API" == "true" ]]; then
      echo "UI API base: ${ui_api_url}"
    fi
    echo "UI uploaded to: s3://${UI_BUCKET}/"
  fi
  echo ""

  if [[ "$DEPLOY_API" == "true" ]]; then
    echo "Test the API:"
    echo "  curl -s ${api_url}/health | jq ."
    echo ""
  fi
  if [[ "$DEPLOY_UI" == "true" ]]; then
    echo "UI config token (for trader-config.js): $(mask_token "$api_token") (use full token value)"
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      echo "CloudFront invalidation: done (paths: /, /index.html, /trader-config.js)"
    fi
    echo ""
  fi
}

main "$@"
