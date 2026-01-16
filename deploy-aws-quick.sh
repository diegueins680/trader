#!/usr/bin/env bash
# AWS deployment script for Trader (API + optional UI)
#
# Quick usage (API only):
#   bash deploy-aws-quick.sh [region] [api-token]
#   TRADER_API_TOKEN=... bash deploy-aws-quick.sh [region]
#
# Optional: also deploy the web UI (S3 + optional CloudFront invalidation):
#   bash deploy-aws-quick.sh --region ap-northeast-1 --api-token "$API_TOKEN" --ui-bucket "$S3_BUCKET" --distribution-id "$CF_ID"
#   - When --distribution-id is set, the UI config apiBaseUrl defaults to /api (same-origin) unless --ui-api-direct is used.
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

DEPLOY_ENV_FILE="${TRADER_DEPLOY_ENV_FILE:-.env.deploy}"
if [[ -f "$DEPLOY_ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$DEPLOY_ENV_FILE"
  set +a
fi

# Configuration (inputs)
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
TRADER_API_TOKEN="${TRADER_API_TOKEN:-}"
TRADER_CORS_ORIGIN="${TRADER_CORS_ORIGIN:-}"
TRADER_API_MAX_BARS_LSTM="${TRADER_API_MAX_BARS_LSTM:-1000}"
TRADER_API_MAX_HIDDEN_SIZE="${TRADER_API_MAX_HIDDEN_SIZE:-50}"
TRADER_DB_URL="${TRADER_DB_URL:-${DATABASE_URL:-}}"
TRADER_STATE_S3_BUCKET_SET="${TRADER_STATE_S3_BUCKET+true}"
TRADER_STATE_S3_BUCKET="${TRADER_STATE_S3_BUCKET:-}"
TRADER_STATE_S3_PREFIX="${TRADER_STATE_S3_PREFIX:-}"
TRADER_STATE_S3_REGION="${TRADER_STATE_S3_REGION:-}"
TRADER_BOT_SYMBOLS="${TRADER_BOT_SYMBOLS:-}"
TRADER_BOT_SYMBOL="${TRADER_BOT_SYMBOL:-}"
TRADER_BOT_TRADE="${TRADER_BOT_TRADE:-true}"
TRADER_OPTIMIZER_ENABLED="${TRADER_OPTIMIZER_ENABLED:-}"
TRADER_OPTIMIZER_EVERY_SEC="${TRADER_OPTIMIZER_EVERY_SEC:-}"
TRADER_OPTIMIZER_TRIALS="${TRADER_OPTIMIZER_TRIALS:-}"
TRADER_OPTIMIZER_TIMEOUT_SEC="${TRADER_OPTIMIZER_TIMEOUT_SEC:-}"
TRADER_OPTIMIZER_OBJECTIVE="${TRADER_OPTIMIZER_OBJECTIVE:-}"
TRADER_OPTIMIZER_LOOKBACK_WINDOW="${TRADER_OPTIMIZER_LOOKBACK_WINDOW:-}"
TRADER_OPTIMIZER_BACKTEST_RATIO="${TRADER_OPTIMIZER_BACKTEST_RATIO:-}"
TRADER_OPTIMIZER_TUNE_RATIO="${TRADER_OPTIMIZER_TUNE_RATIO:-}"
TRADER_OPTIMIZER_MAX_POINTS="${TRADER_OPTIMIZER_MAX_POINTS:-}"
TRADER_OPTIMIZER_SYMBOLS="${TRADER_OPTIMIZER_SYMBOLS:-}"
TRADER_OPTIMIZER_INTERVALS="${TRADER_OPTIMIZER_INTERVALS:-}"
BINANCE_API_KEY="${BINANCE_API_KEY:-}"
BINANCE_API_SECRET="${BINANCE_API_SECRET:-}"
UI_BUCKET="${TRADER_UI_BUCKET:-${S3_BUCKET:-}}"
UI_DISTRIBUTION_ID="${TRADER_UI_CLOUDFRONT_DISTRIBUTION_ID:-${CLOUDFRONT_DISTRIBUTION_ID:-}}"
UI_CLOUDFRONT_DOMAIN="${TRADER_UI_CLOUDFRONT_DOMAIN:-}"
UI_SKIP_BUILD="${TRADER_UI_SKIP_BUILD:-false}"
UI_DIST_DIR="${TRADER_UI_DIST_DIR:-haskell/web/dist}"
UI_API_MODE="${TRADER_UI_API_MODE:-direct}"
UI_API_MODE_DEFAULT="true"
if [[ -n "${TRADER_UI_API_MODE:-}" ]]; then
  UI_API_MODE_DEFAULT="false"
fi
UI_ONLY="false"
API_ONLY="false"
UI_API_URL="${TRADER_UI_API_URL:-}"
UI_API_FALLBACK_URL="${TRADER_UI_API_FALLBACK_URL:-}"
UI_SERVICE_ARN="${TRADER_UI_SERVICE_ARN:-}"
APP_RUNNER_INSTANCE_ROLE_ARN="${APP_RUNNER_INSTANCE_ROLE_ARN:-${TRADER_APP_RUNNER_INSTANCE_ROLE_ARN:-}}"
APP_RUNNER_CPU="${APP_RUNNER_CPU:-${TRADER_APP_RUNNER_CPU:-}}"
APP_RUNNER_MEMORY="${APP_RUNNER_MEMORY:-${TRADER_APP_RUNNER_MEMORY:-}}"
ENSURE_RESOURCES="${TRADER_AWS_ENSURE_RESOURCES:-false}"
UI_CLOUDFRONT_AUTO="${TRADER_UI_CLOUDFRONT_AUTO:-false}"
UI_CLOUDFRONT_OAC_NAME="${TRADER_UI_CLOUDFRONT_OAC_NAME:-trader-ui-oac}"
APP_RUNNER_INSTANCE_ROLE_NAME="${TRADER_APP_RUNNER_INSTANCE_ROLE_NAME:-TraderAppRunnerS3Role}"
APP_RUNNER_STATE_POLICY_NAME="${TRADER_APP_RUNNER_STATE_POLICY_NAME:-TraderAppRunnerS3StatePolicy}"

# Configuration (defaults)
ECR_REPO="trader-api"
APP_RUNNER_SERVICE_NAME="${APP_RUNNER_SERVICE_NAME:-$ECR_REPO}"
APP_RUNNER_ECR_ACCESS_ROLE_NAME="${APP_RUNNER_ECR_ACCESS_ROLE_NAME:-AppRunnerECRAccessRole}"

ECR_URI=""
APP_RUNNER_SERVICE_URL=""
AWS_ACCOUNT_ID=""
LAST_S3_BUCKET_CREATED="false"

if [[ -z "${TRADER_STATE_DIR+x}" ]]; then
  TRADER_STATE_DIR="/var/lib/trader/state"
fi

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
  --db-url <url>                    Database URL for ops persistence (TRADER_DB_URL / DATABASE_URL)
  --state-dir <path>                State dir (default: /var/lib/trader/state; mount durable storage)
  --state-s3-bucket <bucket>        S3 bucket for App Runner state (required unless TRADER_DB_URL is set)
  --state-s3-prefix <prefix>        S3 key prefix for state (TRADER_STATE_S3_PREFIX)
  --state-s3-region <region>        S3 region override (TRADER_STATE_S3_REGION)
  --instance-role-arn <arn>         App Runner instance role ARN (for S3 access)
  --ensure-resources                Create/reuse AWS resources (defaults state bucket; CloudFront when --cloudfront or --distribution-id)
  --api-only                         Deploy API only
  --ui-only                          Deploy UI only (requires --ui-bucket and --api-url or --service-arn)
  --ui-bucket|--bucket <bucket>     S3 bucket to upload UI to
  --cloudfront                      Auto-create/reuse CloudFront distribution for the UI bucket
  --cloudfront-domain <domain>      Reuse a CloudFront distribution by domain (auto-detects UI bucket)
  --distribution-id <id>            CloudFront distribution ID (optional; defaults UI apiBaseUrl to /api unless --ui-api-direct)
  --api-url <url>                   API origin URL for UI-only deploys (also configures CloudFront /api/* behavior)
  --ui-api-fallback <url>           Optional UI fallback API URL (CORS required)
  --ui-api-direct                   Use the full API URL in trader-config.js even with CloudFront
  --ui-api-proxy                    Force apiBaseUrl to /api (requires CloudFront /api/* behavior)
  --service-arn <arn>               App Runner service ARN to auto-discover API URL/token (UI-only convenience)
  --skip-ui-build                   Skip `npm run build` (uses existing dist dir)
  --ui-dist-dir <dir>               UI dist dir (default: haskell/web/dist)
  -h|--help                         Show help

Environment variables (equivalents):
  AWS_REGION / AWS_DEFAULT_REGION
  TRADER_API_TOKEN
  TRADER_CORS_ORIGIN
  TRADER_DB_URL / DATABASE_URL
  TRADER_STATE_DIR
  TRADER_STATE_S3_BUCKET
  TRADER_STATE_S3_PREFIX
  TRADER_STATE_S3_REGION
  TRADER_API_MAX_BARS_LSTM
  TRADER_API_MAX_HIDDEN_SIZE
  TRADER_BOT_SYMBOLS
  TRADER_BOT_SYMBOL
  TRADER_BOT_TRADE
  TRADER_OPTIMIZER_ENABLED
  TRADER_OPTIMIZER_EVERY_SEC
  TRADER_OPTIMIZER_TRIALS
  TRADER_OPTIMIZER_TIMEOUT_SEC
  TRADER_OPTIMIZER_OBJECTIVE
  TRADER_OPTIMIZER_LOOKBACK_WINDOW
  TRADER_OPTIMIZER_BACKTEST_RATIO
  TRADER_OPTIMIZER_TUNE_RATIO
  TRADER_OPTIMIZER_MAX_POINTS
  TRADER_OPTIMIZER_SYMBOLS
  TRADER_OPTIMIZER_INTERVALS
  TRADER_AWS_ENSURE_RESOURCES
  BINANCE_API_KEY
  BINANCE_API_SECRET
  TRADER_UI_BUCKET / S3_BUCKET
  TRADER_UI_CLOUDFRONT_AUTO
  TRADER_UI_CLOUDFRONT_DOMAIN
  TRADER_UI_CLOUDFRONT_DISTRIBUTION_ID / CLOUDFRONT_DISTRIBUTION_ID
  TRADER_UI_CLOUDFRONT_OAC_NAME
  TRADER_UI_SKIP_BUILD
  TRADER_UI_DIST_DIR
  TRADER_UI_API_URL
  TRADER_UI_API_FALLBACK_URL
  TRADER_UI_API_MODE (direct|proxy)
  TRADER_UI_SERVICE_ARN
  TRADER_DEPLOY_ENV_FILE
  TRADER_APP_RUNNER_INSTANCE_ROLE_NAME
  TRADER_APP_RUNNER_STATE_POLICY_NAME
  APP_RUNNER_INSTANCE_ROLE_ARN / TRADER_APP_RUNNER_INSTANCE_ROLE_ARN
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

health_check_api() {
  local api_url="${1:-}"
  local health_url=""
  local resp=""
  local attempt=1
  local max_attempts=5

  if [[ -z "$api_url" ]]; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: curl not found; skipping API health check${NC}"
    return 0
  fi

  health_url="${api_url%/}/health"
  while [[ $attempt -le $max_attempts ]]; do
    if resp="$(curl -fsS --connect-timeout 5 --max-time 10 "$health_url" 2>/dev/null)"; then
      if [[ "$resp" == *"\"status\":\"ok\""* ]]; then
        echo -e "${GREEN}✓ API /health ok${NC}"
      else
        echo -e "${YELLOW}Warning: API /health returned unexpected payload${NC}"
        echo "  ${resp}"
      fi
      return 0
    fi
    if [[ $attempt -lt $max_attempts ]]; then
      sleep 3
    fi
    attempt=$((attempt + 1))
  done

  echo -e "${YELLOW}Warning: API /health check failed after ${max_attempts} attempts (${health_url})${NC}"
}

get_cloudfront_domain() {
  local dist_id="${1:-}"
  local domain=""

  if [[ -z "$dist_id" ]]; then
    return 1
  fi

  domain="$(
    aws cloudfront get-distribution \
      --id "$dist_id" \
      --query 'Distribution.DomainName' \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$domain" || "$domain" == "None" ]]; then
    return 1
  fi
  echo "$domain"
}

resolve_ui_base_url() {
  local dist_id="${1:-}"
  local domain_override="${2:-}"
  local domain=""

  if [[ -n "$domain_override" ]]; then
    domain="$(normalize_cloudfront_domain "$domain_override")"
  elif [[ -n "$dist_id" ]]; then
    domain="$(get_cloudfront_domain "$dist_id" || true)"
  fi

  if [[ -z "$domain" ]]; then
    return 1
  fi

  echo "https://${domain}"
}

normalize_origin_url() {
  local url="${1:-}"
  if [[ "$url" =~ ^https?://[^/]+ ]]; then
    echo "${BASH_REMATCH[0]}"
    return 0
  fi
  return 1
}

resolve_ui_origin() {
  local dist_id="${1:-}"
  local domain_override="${2:-}"
  local base=""
  base="$(resolve_ui_base_url "$dist_id" "$domain_override" || true)"
  if [[ -z "$base" ]]; then
    return 1
  fi
  normalize_origin_url "$base"
}

smoke_check_ui() {
  local ui_url="${1:-}"
  local ui_api_url="${2:-}"
  local api_url="${3:-}"
  local base=""
  local body=""
  local config=""

  if [[ -z "$ui_url" ]]; then
    echo -e "${YELLOW}Warning: UI URL unavailable; skipping UI smoke checks${NC}"
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: curl not found; skipping UI smoke checks${NC}"
    return 0
  fi

  base="${ui_url%/}"

  if body="$(curl -fsSL --connect-timeout 5 --max-time 10 "${base}/" 2>/dev/null)"; then
    if [[ "$body" == *'id="root"'* || "$body" == *'trader-config.js'* ]]; then
      echo -e "${GREEN}✓ UI index.html ok${NC}"
    else
      echo -e "${YELLOW}Warning: UI index.html returned unexpected content${NC}"
    fi
  else
    echo -e "${YELLOW}Warning: UI index.html fetch failed (${base}/)${NC}"
  fi

  if config="$(curl -fsSL --connect-timeout 5 --max-time 10 "${base}/trader-config.js" 2>/dev/null)"; then
    if [[ "$config" == *"__TRADER_CONFIG__"* && "$config" == *"apiBaseUrl"* ]]; then
      echo -e "${GREEN}✓ UI trader-config.js ok${NC}"
    else
      echo -e "${YELLOW}Warning: UI trader-config.js missing expected keys${NC}"
    fi
    if [[ -n "$ui_api_url" && "$config" != *"apiBaseUrl: \"${ui_api_url}\""* ]]; then
      echo -e "${YELLOW}Warning: UI trader-config.js apiBaseUrl does not match expected${NC}"
    fi
  else
    echo -e "${YELLOW}Warning: UI trader-config.js fetch failed (${base}/trader-config.js)${NC}"
  fi

  if [[ "$ui_api_url" == "/api" ]]; then
    local health_url="${base}/api/health"
    local resp=""
    if resp="$(curl -fsSL --connect-timeout 5 --max-time 10 "$health_url" 2>/dev/null)"; then
      if [[ "$resp" == *"\"status\":\"ok\""* ]]; then
        echo -e "${GREEN}✓ UI /api/health ok${NC}"
      else
        echo -e "${YELLOW}Warning: UI /api/health returned unexpected payload${NC}"
      fi
    else
      echo -e "${YELLOW}Warning: UI /api/health fetch failed (${health_url})${NC}"
    fi
  elif [[ "$ui_api_url" == http* && "$ui_api_url" != "$api_url" ]]; then
    health_check_api "$ui_api_url"
  fi
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
    --db-url)
      TRADER_DB_URL="${2:-}"
      shift 2
      ;;
    --state-dir)
      TRADER_STATE_DIR="${2:-}"
      shift 2
      ;;
    --state-s3-bucket)
      TRADER_STATE_S3_BUCKET="${2:-}"
      TRADER_STATE_S3_BUCKET_SET="true"
      shift 2
      ;;
    --state-s3-prefix)
      TRADER_STATE_S3_PREFIX="${2:-}"
      shift 2
      ;;
    --state-s3-region)
      TRADER_STATE_S3_REGION="${2:-}"
      shift 2
      ;;
    --instance-role-arn)
      APP_RUNNER_INSTANCE_ROLE_ARN="${2:-}"
      shift 2
      ;;
    --ensure-resources)
      ENSURE_RESOURCES="true"
      shift
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
    --cloudfront|--cloudfront-auto)
      UI_CLOUDFRONT_AUTO="true"
      shift
      ;;
    --cloudfront-domain)
      UI_CLOUDFRONT_DOMAIN="${2:-}"
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
    --ui-api-fallback)
      UI_API_FALLBACK_URL="${2:-}"
      shift 2
      ;;
    --ui-api-direct)
      UI_API_MODE="direct"
      UI_API_MODE_DEFAULT="false"
      shift
      ;;
    --ui-api-proxy)
      UI_API_MODE="proxy"
      UI_API_MODE_DEFAULT="false"
      shift
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

if [[ "${UI_API_MODE}" != "direct" && "${UI_API_MODE}" != "proxy" ]]; then
  UI_API_MODE="direct"
fi

DEPLOY_API="true"
DEPLOY_UI="false"
if [[ -n "${UI_BUCKET:-}" || -n "${UI_DISTRIBUTION_ID:-}" || -n "${UI_CLOUDFRONT_DOMAIN:-}" || "$UI_ONLY" == "true" ]] || is_true "$UI_CLOUDFRONT_AUTO"; then
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

ensure_account_id() {
  if [[ -z "${AWS_ACCOUNT_ID:-}" ]]; then
    AWS_ACCOUNT_ID="$(get_account_id)"
  fi
}

ensure_s3_bucket() {
  local bucket="$1"
  local region="${2:-$AWS_REGION}"
  LAST_S3_BUCKET_CREATED="false"

  if aws s3api head-bucket --bucket "$bucket" --region "$region" >/dev/null 2>&1; then
    echo -e "${YELLOW}✓ S3 bucket exists: ${bucket}${NC}" >&2
    return 0
  fi

  local args=(--bucket "$bucket" --region "$region")
  if [[ "$region" != "us-east-1" ]]; then
    args+=(--create-bucket-configuration "LocationConstraint=${region}")
  fi

  if aws s3api create-bucket "${args[@]}" >/dev/null; then
    LAST_S3_BUCKET_CREATED="true"
    echo -e "${GREEN}✓ S3 bucket created: ${bucket}${NC}" >&2
    return 0
  fi

  echo -e "${RED}✗ Unable to create S3 bucket: ${bucket}${NC}" >&2
  echo "Ensure the bucket name is available and you have permissions." >&2
  exit 1
}

discover_s3_bucket_region() {
  local bucket="$1"
  local region=""
  region="$(
    aws s3api get-bucket-location \
      --bucket "$bucket" \
      --region "$AWS_REGION" \
      --query 'LocationConstraint' \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$region" || "$region" == "None" || "$region" == "null" ]]; then
    region="us-east-1"
  elif [[ "$region" == "EU" ]]; then
    region="eu-west-1"
  fi
  if [[ -z "$region" ]]; then
    region="$AWS_REGION"
  fi
  echo "$region"
}

apply_bucket_private_defaults() {
  local bucket="$1"
  local region="${2:-$AWS_REGION}"
  aws s3api put-public-access-block \
    --bucket "$bucket" \
    --region "$region" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true >/dev/null
  aws s3api put-bucket-ownership-controls \
    --bucket "$bucket" \
    --region "$region" \
    --ownership-controls Rules=[{ObjectOwnership=BucketOwnerEnforced}] >/dev/null || true
}

write_state_policy_doc() {
  local bucket="$1"
  local prefix_raw="$2"
  local out="$3"
  local prefix="${prefix_raw#/}"
  prefix="${prefix%/}"
  local list_condition=""
  if [[ -n "$prefix" ]]; then
    list_condition="$(cat <<EOF
,
      "Condition": {
        "StringLike": {
          "s3:prefix": ["${prefix}", "${prefix}/*"]
        }
      }
EOF
)"
  fi
  local object_arn="arn:aws:s3:::${bucket}"
  if [[ -n "$prefix" ]]; then
    object_arn="${object_arn}/${prefix}/*"
  else
    object_arn="${object_arn}/*"
  fi

  cat > "$out" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListStateBucket",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${bucket}"${list_condition}
    },
    {
      "Sid": "ReadWriteStateObjects",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "${object_arn}"
    }
  ]
}
EOF
}

ensure_state_instance_role() {
  local bucket="$1"
  local prefix="$2"
  local role_name="$APP_RUNNER_INSTANCE_ROLE_NAME"
  local policy_name="$APP_RUNNER_STATE_POLICY_NAME"

  local role_arn=""
  role_arn="$(aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || true)"
  if [[ -z "$role_arn" || "$role_arn" == "None" ]]; then
    echo "Creating App Runner instance role: ${role_name}" >&2
    local trust_doc
    trust_doc="$(mktemp)"
    cat >"$trust_doc" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "tasks.apprunner.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
    aws iam create-role \
      --role-name "$role_name" \
      --assume-role-policy-document "file://${trust_doc}" \
      --description "App Runner instance role for Trader S3 state" \
      >/dev/null
    rm -f "$trust_doc"
  else
    echo -e "${YELLOW}✓ Reusing App Runner instance role: ${role_name}${NC}" >&2
  fi

  local policy_doc
  policy_doc="$(mktemp)"
  write_state_policy_doc "$bucket" "$prefix" "$policy_doc"
  aws iam put-role-policy \
    --role-name "$role_name" \
    --policy-name "$policy_name" \
    --policy-document "file://${policy_doc}" \
    >/dev/null
  rm -f "$policy_doc"

  if [[ -z "$role_arn" || "$role_arn" == "None" ]]; then
    for _ in {1..12}; do
      role_arn="$(aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text 2>/dev/null || true)"
      if [[ -n "$role_arn" && "$role_arn" != "None" ]]; then
        break
      fi
      sleep 2
    done
  fi

  if [[ -z "$role_arn" || "$role_arn" == "None" ]]; then
    echo -e "${RED}✗ Failed to resolve App Runner instance role ARN${NC}" >&2
    exit 1
  fi

  echo "$role_arn"
}

ensure_cloudfront_oac() {
  local name="$1"
  local oac_id=""
  oac_id="$(
    aws cloudfront list-origin-access-controls \
      --query "OriginAccessControlList.Items[?Name=='${name}'].Id | [0]" \
      --output text 2>/dev/null || true
  )"
  if [[ -n "$oac_id" && "$oac_id" != "None" ]]; then
    echo "$oac_id"
    return 0
  fi

  local cfg
  cfg="$(mktemp)"
  cat >"$cfg" <<EOF
{
  "Name": "${name}",
  "Description": "Trader UI CloudFront OAC",
  "SigningProtocol": "sigv4",
  "SigningBehavior": "always",
  "OriginAccessControlOriginType": "s3"
}
EOF
  oac_id="$(
    aws cloudfront create-origin-access-control \
      --origin-access-control-config "file://${cfg}" \
      --query OriginAccessControl.Id \
      --output text
  )"
  rm -f "$cfg"
  echo "$oac_id"
}

normalize_cloudfront_domain() {
  local domain="${1:-}"
  domain="${domain#https://}"
  domain="${domain#http://}"
  domain="${domain%%/*}"
  echo "$domain"
}

discover_cloudfront_distribution_id_for_domain() {
  local domain_raw="$1"
  local domain
  domain="$(normalize_cloudfront_domain "$domain_raw")"
  if [[ -z "$domain" ]]; then
    return 1
  fi
  local dist_id=""
  dist_id="$(
    aws cloudfront list-distributions \
      --query "DistributionList.Items[?DomainName=='${domain}'].Id | [0]" \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$dist_id" || "$dist_id" == "None" ]]; then
    return 1
  fi
  echo "$dist_id"
}

discover_cloudfront_ui_bucket() {
  local dist_id="$1"
  if [[ -z "$dist_id" ]]; then
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  aws cloudfront get-distribution-config \
    --id "$dist_id" \
    --query 'DistributionConfig.Origins.Items' \
    --output json 2>/dev/null \
    | python3 -c 'import json, re, sys
def bucket_from_origin(origin):
    domain = (origin.get("DomainName") or "").lower()
    origin_path = origin.get("OriginPath") or ""
    candidate = ""
    if origin_path.startswith("/"):
        candidate = origin_path.lstrip("/").split("/")[0]
    patterns = [
        r"^(?P<bucket>[^.]+)\.s3[.-][^.]+\.amazonaws\.com$",
        r"^(?P<bucket>[^.]+)\.s3\.amazonaws\.com$",
        r"^(?P<bucket>[^.]+)\.s3-website[.-][^.]+\.amazonaws\.com$",
        r"^(?P<bucket>[^.]+)\.s3-website\.amazonaws\.com$",
    ]
    for pat in patterns:
        match = re.match(pat, domain)
        if match:
            return match.group("bucket")
    if candidate and domain.startswith("s3"):
        return candidate
    return ""

try:
    origins = json.load(sys.stdin)
except Exception:
    sys.exit(0)

for origin in origins or []:
    bucket = bucket_from_origin(origin)
    if bucket:
        print(bucket)
        sys.exit(0)
'
}

discover_cloudfront_distribution_id_for_bucket() {
  local bucket="$1"
  local bucket_region=""
  bucket_region="$(discover_s3_bucket_region "$bucket")"
  local domain_primary="${bucket}.s3.${bucket_region}.amazonaws.com"
  local domain_legacy="${bucket}.s3-${bucket_region}.amazonaws.com"
  local domain_alt="${bucket}.s3.amazonaws.com"
  local domain_dualstack="${bucket}.s3.dualstack.${bucket_region}.amazonaws.com"
  local domain_website_dash="${bucket}.s3-website-${bucket_region}.amazonaws.com"
  local domain_website_dot="${bucket}.s3-website.${bucket_region}.amazonaws.com"
  local path_domain_primary="s3.${bucket_region}.amazonaws.com"
  local path_domain_legacy="s3-${bucket_region}.amazonaws.com"
  local path_domain_alt="s3.amazonaws.com"
  local path_domain_dualstack="s3.dualstack.${bucket_region}.amazonaws.com"
  local path_origin="/${bucket}"
  local ids
  ids="$(
    aws cloudfront list-distributions \
      --query "DistributionList.Items[?Origins.Items[?DomainName=='${domain_primary}' || DomainName=='${domain_legacy}' || DomainName=='${domain_alt}' || DomainName=='${domain_dualstack}' || DomainName=='${domain_website_dash}' || DomainName=='${domain_website_dot}' || (DomainName=='${path_domain_primary}' && OriginPath=='${path_origin}') || (DomainName=='${path_domain_legacy}' && OriginPath=='${path_origin}') || (DomainName=='${path_domain_alt}' && OriginPath=='${path_origin}') || (DomainName=='${path_domain_dualstack}' && OriginPath=='${path_origin}')]].Id" \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$ids" || "$ids" == "None" ]]; then
    return 1
  fi
  local id_list=()
  read -r -a id_list <<<"$ids"
  if (( ${#id_list[@]} == 0 )); then
    return 1
  fi
  if (( ${#id_list[@]} > 1 )); then
    echo -e "${YELLOW}Warning: multiple CloudFront distributions found for ${bucket}; using ${id_list[0]}${NC}" >&2
  fi
  echo "${id_list[0]}"
}

ensure_cloudfront_distribution() {
  local bucket="$1"
  local existing=""
  existing="$(discover_cloudfront_distribution_id_for_bucket "$bucket" || true)"
  if [[ -n "$existing" ]]; then
    echo -e "${YELLOW}✓ Reusing CloudFront distribution ${existing}${NC}" >&2
    echo "$existing"
    return 0
  fi

  local oac_id
  oac_id="$(ensure_cloudfront_oac "$UI_CLOUDFRONT_OAC_NAME")"
  local cache_policy_id
  cache_policy_id="$(get_managed_cache_policy_id "Managed-CachingOptimized")"
  if [[ -z "$cache_policy_id" || "$cache_policy_id" == "None" ]]; then
    cache_policy_id="$(get_managed_cache_policy_id "Managed-CachingDisabled")"
  fi
  if [[ -z "$cache_policy_id" || "$cache_policy_id" == "None" ]]; then
    echo -e "${RED}✗ CloudFront managed cache policy not found${NC}" >&2
    exit 1
  fi

  local bucket_region=""
  bucket_region="$(discover_s3_bucket_region "$bucket")"
  local domain="${bucket}.s3.${bucket_region}.amazonaws.com"
  local cfg
  cfg="$(mktemp)"
  local ref="trader-ui-$(date +%s)"
  cat >"$cfg" <<EOF
{
  "CallerReference": "${ref}",
  "Comment": "trader-ui",
  "Enabled": true,
  "IsIPV6Enabled": true,
  "PriceClass": "PriceClass_100",
  "DefaultRootObject": "index.html",
  "Aliases": {"Quantity": 0},
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "trader-ui-origin",
        "DomainName": "${domain}",
        "OriginPath": "",
        "CustomHeaders": {"Quantity": 0},
        "S3OriginConfig": {"OriginAccessIdentity": ""},
        "OriginAccessControlId": "${oac_id}"
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "trader-ui-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 3,
      "Items": ["GET", "HEAD", "OPTIONS"],
      "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}
    },
    "Compress": true,
    "CachePolicyId": "${cache_policy_id}",
    "TrustedSigners": {"Enabled": false, "Quantity": 0},
    "TrustedKeyGroups": {"Enabled": false, "Quantity": 0}
  },
  "CustomErrorResponses": {
    "Quantity": 2,
    "Items": [
      {"ErrorCode": 403, "ResponsePagePath": "/index.html", "ResponseCode": "200", "ErrorCachingMinTTL": 0},
      {"ErrorCode": 404, "ResponsePagePath": "/index.html", "ResponseCode": "200", "ErrorCachingMinTTL": 0}
    ]
  },
  "Restrictions": {
    "GeoRestriction": {"RestrictionType": "none", "Quantity": 0}
  },
  "ViewerCertificate": {"CloudFrontDefaultCertificate": true}
}
EOF
  local dist_id
  dist_id="$(
    aws cloudfront create-distribution \
      --distribution-config "file://${cfg}" \
      --query Distribution.Id \
      --output text
  )"
  rm -f "$cfg"
  echo -e "${GREEN}✓ CloudFront distribution created: ${dist_id}${NC}" >&2
  echo "$dist_id"
}

ensure_cloudfront_bucket_policy() {
  local bucket="$1"
  local dist_id="$2"
  if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}✗ python3 not found (required to update CloudFront bucket policy).${NC}" >&2
    exit 1
  fi
  ensure_account_id

  local tmp_in
  local tmp_out
  tmp_in="$(mktemp)"
  tmp_out="$(mktemp)"
  aws s3api get-bucket-policy --bucket "$bucket" --output json > "$tmp_in" 2>/dev/null || true
  python3 - "$tmp_in" "$tmp_out" "$bucket" "$AWS_ACCOUNT_ID" "$dist_id" <<'PY'
import json
import sys

src, out, bucket, account_id, dist_id = sys.argv[1:6]
raw = ""
try:
    with open(src) as f:
        raw = f.read().strip()
except FileNotFoundError:
    raw = ""

policy = {"Version": "2012-10-17", "Statement": []}
if raw:
    data = json.loads(raw)
    if isinstance(data, dict) and "Policy" in data:
        policy = json.loads(data["Policy"])
    else:
        policy = data

statements = policy.get("Statement") or []
if isinstance(statements, dict):
    statements = [statements]

statements = [s for s in statements if s.get("Sid") != "AllowCloudFrontServiceRead"]
statements.append(
    {
        "Sid": "AllowCloudFrontServiceRead",
        "Effect": "Allow",
        "Principal": {"Service": "cloudfront.amazonaws.com"},
        "Action": "s3:GetObject",
        "Resource": f"arn:aws:s3:::{bucket}/*",
        "Condition": {
            "StringEquals": {
                "AWS:SourceArn": f"arn:aws:cloudfront::{account_id}:distribution/{dist_id}"
            }
        },
    }
)

policy["Statement"] = statements
with open(out, "w") as f:
    json.dump(policy, f, indent=2)
PY
  aws s3api put-bucket-policy --bucket "$bucket" --policy "file://${tmp_out}" >/dev/null
  rm -f "$tmp_in" "$tmp_out"
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

discover_apprunner_service_arn_by_name() {
  local service_name="$1"
  if [[ -z "$service_name" ]]; then
    return 1
  fi
  local arn=""
  arn="$(
    aws apprunner list-services \
      --region "$AWS_REGION" \
      --query 'ServiceSummaryList[?ServiceName==`'"${service_name}"'`].ServiceArn | [0]' \
      --output text 2>/dev/null || true
  )"
  if [[ "$arn" == "None" ]]; then
    arn=""
  fi
  if [[ -n "$arn" ]]; then
    echo "$arn"
    return 0
  fi
  return 1
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

  if [[ -z "${TRADER_DB_URL:-}" && -n "$existing_service_arn" ]]; then
    local existing_db_url=""
    existing_db_url="$(
      aws apprunner describe-service \
        --service-arn "$existing_service_arn" \
        --region "$AWS_REGION" \
        --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_DB_URL' \
        --output text 2>/dev/null || true
    )"
    if [[ "$existing_db_url" == "None" ]]; then
      existing_db_url=""
    fi
    if [[ -n "$existing_db_url" ]]; then
      TRADER_DB_URL="$existing_db_url"
      echo -e "${YELLOW}✓ Reusing existing TRADER_DB_URL from service${NC}" >&2
    fi
  fi

  if [[ -z "${BINANCE_API_KEY:-}" && -n "$existing_service_arn" ]]; then
    local existing_binance_key=""
    existing_binance_key="$(
      aws apprunner describe-service \
        --service-arn "$existing_service_arn" \
        --region "$AWS_REGION" \
        --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.BINANCE_API_KEY' \
        --output text 2>/dev/null || true
    )"
    if [[ "$existing_binance_key" == "None" ]]; then
      existing_binance_key=""
    fi
    if [[ -n "$existing_binance_key" ]]; then
      BINANCE_API_KEY="$existing_binance_key"
      echo -e "${YELLOW}✓ Reusing existing BINANCE_API_KEY from service${NC}" >&2
    fi
  fi

  if [[ -z "${BINANCE_API_SECRET:-}" && -n "$existing_service_arn" ]]; then
    local existing_binance_secret=""
    existing_binance_secret="$(
      aws apprunner describe-service \
        --service-arn "$existing_service_arn" \
        --region "$AWS_REGION" \
        --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.BINANCE_API_SECRET' \
        --output text 2>/dev/null || true
    )"
    if [[ "$existing_binance_secret" == "None" ]]; then
      existing_binance_secret=""
    fi
    if [[ -n "$existing_binance_secret" ]]; then
      BINANCE_API_SECRET="$existing_binance_secret"
      echo -e "${YELLOW}✓ Reusing existing BINANCE_API_SECRET from service${NC}" >&2
    fi
  fi

  if [[ -n "$existing_service_arn" ]]; then
    if [[ -z "${APP_RUNNER_INSTANCE_ROLE_ARN:-}" ]]; then
      local existing_instance_role=""
      existing_instance_role="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.InstanceConfiguration.InstanceRoleArn' \
          --output text 2>/dev/null || true
      )"
      if [[ "$existing_instance_role" == "None" ]]; then
        existing_instance_role=""
      fi
      if [[ -n "$existing_instance_role" ]]; then
        APP_RUNNER_INSTANCE_ROLE_ARN="$existing_instance_role"
        echo -e "${YELLOW}✓ Reusing existing App Runner instance role${NC}" >&2
      fi
    fi

    if [[ -z "${APP_RUNNER_CPU:-}" || -z "${APP_RUNNER_MEMORY:-}" ]]; then
      local existing_cpu=""
      local existing_memory=""
      existing_cpu="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.InstanceConfiguration.Cpu' \
          --output text 2>/dev/null || true
      )"
      existing_memory="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.InstanceConfiguration.Memory' \
          --output text 2>/dev/null || true
      )"
      if [[ "$existing_cpu" == "None" ]]; then
        existing_cpu=""
      fi
      if [[ "$existing_memory" == "None" ]]; then
        existing_memory=""
      fi
      if [[ -z "${APP_RUNNER_CPU:-}" && -n "$existing_cpu" ]]; then
        APP_RUNNER_CPU="$existing_cpu"
        echo -e "${YELLOW}✓ Reusing existing App Runner CPU (${APP_RUNNER_CPU})${NC}" >&2
      fi
      if [[ -z "${APP_RUNNER_MEMORY:-}" && -n "$existing_memory" ]]; then
        APP_RUNNER_MEMORY="$existing_memory"
        echo -e "${YELLOW}✓ Reusing existing App Runner memory (${APP_RUNNER_MEMORY})${NC}" >&2
      fi
    fi

    if [[ -z "${TRADER_STATE_S3_BUCKET:-}" && -z "${TRADER_STATE_S3_BUCKET_SET:-}" ]]; then
      local existing_bucket=""
      existing_bucket="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_STATE_S3_BUCKET' \
          --output text 2>/dev/null || true
      )"
      if [[ "$existing_bucket" == "None" ]]; then
        existing_bucket=""
      fi
      if [[ -n "$existing_bucket" ]]; then
        TRADER_STATE_S3_BUCKET="$existing_bucket"
        echo -e "${YELLOW}✓ Reusing existing TRADER_STATE_S3_BUCKET from service${NC}" >&2
      fi

      local existing_prefix=""
      existing_prefix="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_STATE_S3_PREFIX' \
          --output text 2>/dev/null || true
      )"
      if [[ "$existing_prefix" == "None" ]]; then
        existing_prefix=""
      fi
      if [[ -n "$existing_prefix" ]]; then
        TRADER_STATE_S3_PREFIX="$existing_prefix"
      fi

      local existing_region=""
      existing_region="$(
        aws apprunner describe-service \
          --service-arn "$existing_service_arn" \
          --region "$AWS_REGION" \
          --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_STATE_S3_REGION' \
          --output text 2>/dev/null || true
      )"
      if [[ "$existing_region" == "None" ]]; then
        existing_region=""
      fi
      if [[ -n "$existing_region" ]]; then
        TRADER_STATE_S3_REGION="$existing_region"
      fi
    fi
  fi

  echo "Ensuring App Runner ECR access role..." >&2
  local access_role_arn=""
  access_role_arn="$(ensure_apprunner_ecr_access_role)"
  echo -e "${GREEN}✓ Using ECR access role: ${access_role_arn}${NC}" >&2

  local instance_cfg="Cpu=${APP_RUNNER_CPU:-1024},Memory=${APP_RUNNER_MEMORY:-2048}"
  if [[ -n "${APP_RUNNER_INSTANCE_ROLE_ARN:-}" ]]; then
    instance_cfg="${instance_cfg},InstanceRoleArn=${APP_RUNNER_INSTANCE_ROLE_ARN}"
  fi

  # Create source-configuration JSON (file:// is the most reliable for AWS CLI JSON input)
  local src_cfg
  src_cfg="$(mktemp)"

  local runtime_env_json='"TRADER_API_ASYNC_DIR":"/var/lib/trader/async"'
  if [[ -n "${TRADER_STATE_DIR}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_STATE_DIR\":\"${TRADER_STATE_DIR}\""
  fi
  if [[ -n "${TRADER_STATE_S3_BUCKET:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_STATE_S3_BUCKET\":\"${TRADER_STATE_S3_BUCKET}\""
    if [[ -n "${TRADER_STATE_S3_PREFIX:-}" ]]; then
      runtime_env_json="${runtime_env_json},\"TRADER_STATE_S3_PREFIX\":\"${TRADER_STATE_S3_PREFIX}\""
    fi
    if [[ -n "${TRADER_STATE_S3_REGION:-}" ]]; then
      runtime_env_json="${runtime_env_json},\"TRADER_STATE_S3_REGION\":\"${TRADER_STATE_S3_REGION}\""
    fi
  fi
  if [[ -n "${TRADER_DB_URL:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_DB_URL\":\"${TRADER_DB_URL}\""
  fi
  if [[ -n "${TRADER_CORS_ORIGIN:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_CORS_ORIGIN\":\"${TRADER_CORS_ORIGIN}\""
  fi
  if [[ -n "${TRADER_API_MAX_BARS_LSTM:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_API_MAX_BARS_LSTM\":\"${TRADER_API_MAX_BARS_LSTM}\""
  fi
  if [[ -n "${TRADER_API_MAX_HIDDEN_SIZE:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_API_MAX_HIDDEN_SIZE\":\"${TRADER_API_MAX_HIDDEN_SIZE}\""
  fi
  if [[ -n "${TRADER_BOT_SYMBOLS:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_BOT_SYMBOLS\":\"${TRADER_BOT_SYMBOLS}\""
  fi
  if [[ -n "${TRADER_BOT_SYMBOL:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_BOT_SYMBOL\":\"${TRADER_BOT_SYMBOL}\""
  fi
  if [[ -n "${TRADER_BOT_TRADE:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_BOT_TRADE\":\"${TRADER_BOT_TRADE}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_ENABLED:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_ENABLED\":\"${TRADER_OPTIMIZER_ENABLED}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_EVERY_SEC:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_EVERY_SEC\":\"${TRADER_OPTIMIZER_EVERY_SEC}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_TRIALS:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_TRIALS\":\"${TRADER_OPTIMIZER_TRIALS}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_TIMEOUT_SEC:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_TIMEOUT_SEC\":\"${TRADER_OPTIMIZER_TIMEOUT_SEC}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_OBJECTIVE:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_OBJECTIVE\":\"${TRADER_OPTIMIZER_OBJECTIVE}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_LOOKBACK_WINDOW:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_LOOKBACK_WINDOW\":\"${TRADER_OPTIMIZER_LOOKBACK_WINDOW}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_BACKTEST_RATIO:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_BACKTEST_RATIO\":\"${TRADER_OPTIMIZER_BACKTEST_RATIO}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_TUNE_RATIO:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_TUNE_RATIO\":\"${TRADER_OPTIMIZER_TUNE_RATIO}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_MAX_POINTS:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_MAX_POINTS\":\"${TRADER_OPTIMIZER_MAX_POINTS}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_SYMBOLS:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_SYMBOLS\":\"${TRADER_OPTIMIZER_SYMBOLS}\""
  fi
  if [[ -n "${TRADER_OPTIMIZER_INTERVALS:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"TRADER_OPTIMIZER_INTERVALS\":\"${TRADER_OPTIMIZER_INTERVALS}\""
  fi
  if [[ -n "${BINANCE_API_KEY:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"BINANCE_API_KEY\":\"${BINANCE_API_KEY}\""
  fi
  if [[ -n "${BINANCE_API_SECRET:-}" ]]; then
    runtime_env_json="${runtime_env_json},\"BINANCE_API_SECRET\":\"${BINANCE_API_SECRET}\""
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
      --instance-configuration "$instance_cfg" \
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
      --instance-configuration "$instance_cfg" \
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
  local api_fallback_url="${3:-}"

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
  apiFallbackUrl: "${api_fallback_url}",
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
  aws s3 cp "${UI_DIST_DIR}/trader-config.js" "s3://${UI_BUCKET}/trader-config.js" \
    --cache-control "no-cache, no-store, must-revalidate" \
    --content-type "application/javascript" \
    --region "$AWS_REGION" >/dev/null
  if [[ -f "${UI_DIST_DIR}/index.html" ]]; then
    aws s3 cp "${UI_DIST_DIR}/index.html" "s3://${UI_BUCKET}/index.html" \
      --cache-control "no-cache, no-store, must-revalidate" \
      --content-type "text/html" \
      --region "$AWS_REGION" >/dev/null
  fi
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

describe_cloudfront_distribution() {
  local dist_id="$1"
  local ui_api_url="${2:-}"
  local domain=""
  domain="$(
    aws cloudfront get-distribution \
      --id "$dist_id" \
      --query 'Distribution.DomainName' \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$domain" || "$domain" == "None" ]]; then
    echo "CloudFront: unable to describe distribution ${dist_id}" >&2
    return 0
  fi
  echo "CloudFront domain: https://${domain}"
  local api_origin_id=""
  api_origin_id="$(
    aws cloudfront get-distribution \
      --id "$dist_id" \
      --query 'Distribution.DistributionConfig.CacheBehaviors.Items[?PathPattern==`/api/*`].TargetOriginId | [0]' \
      --output text 2>/dev/null || true
  )"
  if [[ -z "$api_origin_id" || "$api_origin_id" == "None" ]]; then
    echo "CloudFront warning: /api/* behavior not found; UI should use the full API URL instead of /api." >&2
  else
    echo "CloudFront /api/* behavior: ${api_origin_id}"
  fi
  if [[ "$ui_api_url" == "/api" && ( -z "$api_origin_id" || "$api_origin_id" == "None" ) ]]; then
    echo "CloudFront warning: UI is configured to /api but no /api/* behavior is set. Update the distribution or set apiBaseUrl to the API host." >&2
  fi
}

get_managed_cache_policy_id() {
  local name="$1"
  aws cloudfront list-cache-policies \
    --type managed \
    --query "CachePolicyList.Items[?CachePolicy.CachePolicyConfig.Name=='${name}'].CachePolicy.Id | [0]" \
    --output text 2>/dev/null || true
}

get_managed_origin_request_policy_id() {
  local name="$1"
  aws cloudfront list-origin-request-policies \
    --type managed \
    --query "OriginRequestPolicyList.Items[?OriginRequestPolicy.OriginRequestPolicyConfig.Name=='${name}'].OriginRequestPolicy.Id | [0]" \
    --output text 2>/dev/null || true
}

ensure_cloudfront_api_behavior() {
  local dist_id="$1"
  local api_url="$2"

  if [[ -z "$api_url" || "$api_url" == "/api" ]]; then
    echo -e "${RED}✗ CloudFront /api/* behavior needs a full API URL (got '${api_url:-"(empty)"}').${NC}" >&2
    echo "Set --api-url or --service-arn, or deploy the API in the same run so the URL can be discovered." >&2
    return 1
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}✗ python3 not found (required to update CloudFront config).${NC}" >&2
    return 1
  fi

  local api_domain="${api_url#https://}"
  api_domain="${api_domain#http://}"
  api_domain="${api_domain%%/*}"
  if [[ -z "$api_domain" ]]; then
    echo -e "${RED}✗ Unable to parse API domain from '${api_url}'.${NC}" >&2
    return 1
  fi

  local cache_policy_id
  local origin_request_policy_id
  cache_policy_id="$(get_managed_cache_policy_id "Managed-CachingDisabled")"
  origin_request_policy_id="$(get_managed_origin_request_policy_id "Managed-AllViewerExceptHostHeader")"
  if [[ -z "$origin_request_policy_id" || "$origin_request_policy_id" == "None" ]]; then
    origin_request_policy_id="$(get_managed_origin_request_policy_id "Managed-AllViewer")"
  fi
  if [[ -z "$cache_policy_id" || "$cache_policy_id" == "None" || -z "$origin_request_policy_id" || "$origin_request_policy_id" == "None" ]]; then
    echo -e "${YELLOW}CloudFront warning: managed cache/origin request policy not found; using legacy forwarding.${NC}" >&2
    cache_policy_id=""
    origin_request_policy_id=""
  fi

  local tmp_json
  local tmp_cfg
  tmp_json="$(mktemp)"
  tmp_cfg="$(mktemp)"
  aws cloudfront get-distribution-config --id "$dist_id" > "$tmp_json"
  local etag
  etag="$(python3 - "$tmp_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
print(data["ETag"])
PY
)"

  local changed
  changed="$(python3 - "$tmp_json" "$tmp_cfg" "$api_domain" "$cache_policy_id" "$origin_request_policy_id" <<'PY'
import copy
import json
import sys

config_path, out_path, api_domain, cache_policy_id, origin_request_policy_id = sys.argv[1:6]
cache_policy_id = cache_policy_id or None
origin_request_policy_id = origin_request_policy_id or None

with open(config_path) as f:
    data = json.load(f)

original = data["DistributionConfig"]
config = copy.deepcopy(original)

origins = config.setdefault("Origins", {"Quantity": 0, "Items": []})
origin_items = origins.get("Items") or []
origins["Items"] = origin_items

origin_id = None
for origin in origin_items:
    if origin.get("DomainName") == api_domain:
        origin_id = origin.get("Id")
        break

if origin_id is None:
    base_id = "trader-api-origin"
    used_ids = {origin.get("Id") for origin in origin_items}
    origin_id = base_id
    counter = 1
    while origin_id in used_ids:
        origin_id = f"{base_id}-{counter}"
        counter += 1
    origin_items.append(
        {
            "Id": origin_id,
            "DomainName": api_domain,
            "OriginPath": "",
            "CustomHeaders": {"Quantity": 0},
            "CustomOriginConfig": {
                "HTTPPort": 80,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "https-only",
                "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]},
                "OriginReadTimeout": 30,
                "OriginKeepaliveTimeout": 5,
            },
        }
    )

origins["Quantity"] = len(origin_items)

cache_behaviors = config.setdefault("CacheBehaviors", {"Quantity": 0})
behavior_items = cache_behaviors.get("Items") or []
cache_behaviors["Items"] = behavior_items

api_behavior = None
for behavior in behavior_items:
    if behavior.get("PathPattern") == "/api/*":
        api_behavior = behavior
        break

if api_behavior is None:
    api_behavior = copy.deepcopy(config["DefaultCacheBehavior"])
    api_behavior["PathPattern"] = "/api/*"
    behavior_items.append(api_behavior)

cache_behaviors["Quantity"] = len(behavior_items)

api_behavior["TargetOriginId"] = origin_id
api_behavior["ViewerProtocolPolicy"] = "redirect-to-https"
api_behavior["AllowedMethods"] = {
    "Quantity": 7,
    "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
    "CachedMethods": {"Quantity": 3, "Items": ["GET", "HEAD", "OPTIONS"]},
}

if cache_policy_id and origin_request_policy_id:
    api_behavior["CachePolicyId"] = cache_policy_id
    api_behavior["OriginRequestPolicyId"] = origin_request_policy_id
    api_behavior.pop("ForwardedValues", None)
    api_behavior.pop("MinTTL", None)
    api_behavior.pop("DefaultTTL", None)
    api_behavior.pop("MaxTTL", None)
else:
    api_behavior.pop("CachePolicyId", None)
    api_behavior.pop("OriginRequestPolicyId", None)
    api_behavior["ForwardedValues"] = {
        "QueryString": True,
        "Cookies": {"Forward": "all"},
        "Headers": {
            "Quantity": 4,
            "Items": ["Authorization", "X-API-Key", "Content-Type", "Accept"],
        },
    }
    api_behavior["MinTTL"] = 0
    api_behavior["DefaultTTL"] = 0
    api_behavior["MaxTTL"] = 0

if config == original:
    print("false")
    sys.exit(0)

with open(out_path, "w") as f:
    json.dump(config, f, indent=2, sort_keys=False)

print("true")
PY
)"

  if [[ "$changed" != "true" ]]; then
    echo -e "${GREEN}✓ CloudFront /api/* behavior already configured${NC}" >&2
    rm -f "$tmp_json" "$tmp_cfg"
    return 0
  fi

  echo "Updating CloudFront /api/* behavior..." >&2
  aws cloudfront update-distribution \
    --id "$dist_id" \
    --if-match "$etag" \
    --distribution-config "file://${tmp_cfg}" >/dev/null
  echo -e "${GREEN}✓ CloudFront /api/* behavior updated${NC}" >&2

  rm -f "$tmp_json" "$tmp_cfg"
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

  if [[ -z "${UI_DISTRIBUTION_ID:-}" && -n "${UI_CLOUDFRONT_DOMAIN:-}" ]]; then
    UI_DISTRIBUTION_ID="$(discover_cloudfront_distribution_id_for_domain "$UI_CLOUDFRONT_DOMAIN" || true)"
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      echo -e "${YELLOW}✓ Using CloudFront distribution ${UI_DISTRIBUTION_ID} for ${UI_CLOUDFRONT_DOMAIN}${NC}" >&2
    elif [[ "$DEPLOY_UI" == "true" ]]; then
      echo -e "${RED}✗ CloudFront distribution not found for domain ${UI_CLOUDFRONT_DOMAIN}${NC}" >&2
      exit 2
    fi
  fi

  if [[ -n "${UI_DISTRIBUTION_ID:-}" && -z "${UI_BUCKET:-}" ]]; then
    UI_BUCKET="$(discover_cloudfront_ui_bucket "$UI_DISTRIBUTION_ID" || true)"
    if [[ -n "${UI_BUCKET:-}" ]]; then
      echo -e "${YELLOW}✓ Using UI bucket from CloudFront distribution: ${UI_BUCKET}${NC}" >&2
    elif [[ "$DEPLOY_UI" == "true" && "$UI_ONLY" == "true" ]]; then
      echo -e "${RED}✗ Unable to detect UI bucket from CloudFront distribution ${UI_DISTRIBUTION_ID}${NC}" >&2
      exit 2
    fi
  fi

  local ui_cloudfront_enabled="false"
  if [[ -n "${UI_DISTRIBUTION_ID:-}" ]] || is_true "$UI_CLOUDFRONT_AUTO"; then
    ui_cloudfront_enabled="true"
  fi
  if [[ "$DEPLOY_UI" == "true" && "$UI_API_MODE_DEFAULT" == "true" && "$ui_cloudfront_enabled" == "true" ]]; then
    UI_API_MODE="proxy"
    echo -e "${YELLOW}✓ CloudFront detected; defaulting UI API mode to /api (set TRADER_UI_API_MODE=direct to use the API URL).${NC}" >&2
  fi

  if [[ -z "${TRADER_CORS_ORIGIN:-}" && "$UI_API_MODE" == "direct" ]]; then
    local ui_origin=""
    ui_origin="$(resolve_ui_origin "$UI_DISTRIBUTION_ID" "${UI_CLOUDFRONT_DOMAIN:-}" || true)"
    if [[ -n "$ui_origin" ]]; then
      TRADER_CORS_ORIGIN="$ui_origin"
      echo -e "${YELLOW}✓ Defaulting TRADER_CORS_ORIGIN to ${TRADER_CORS_ORIGIN} for direct UI API calls.${NC}" >&2
    elif [[ "$DEPLOY_API" == "true" ]]; then
      echo -e "${YELLOW}Warning: UI API mode is direct but UI origin is unknown; set TRADER_CORS_ORIGIN to your UI origin to avoid CORS errors.${NC}" >&2
    fi
  fi

  if [[ "$DEPLOY_UI" == "true" && "$ui_cloudfront_enabled" == "true" && -z "${UI_BUCKET:-}" ]]; then
    if is_true "$ENSURE_RESOURCES" || is_true "$UI_CLOUDFRONT_AUTO"; then
      ensure_account_id
      UI_BUCKET="trader-ui-${AWS_ACCOUNT_ID}-${AWS_REGION}"
      echo -e "${YELLOW}✓ Using default UI bucket: ${UI_BUCKET}${NC}" >&2
    else
      echo -e "${RED}✗ Missing UI bucket. Provide --ui-bucket or enable --cloudfront with --ensure-resources.${NC}" >&2
      exit 2
    fi
  fi

  if [[ "$DEPLOY_API" == "true" && -z "${TRADER_STATE_S3_BUCKET:-}" && -z "${TRADER_DB_URL:-}" ]] && is_true "$ENSURE_RESOURCES"; then
    ensure_account_id
    TRADER_STATE_S3_BUCKET="trader-api-state-${AWS_ACCOUNT_ID}-${AWS_REGION}"
    echo -e "${YELLOW}✓ Using default state bucket: ${TRADER_STATE_S3_BUCKET}${NC}" >&2
  fi
  if [[ "$DEPLOY_API" == "true" && -z "${TRADER_STATE_S3_BUCKET:-}" && -z "${TRADER_DB_URL:-}" ]]; then
    echo -e "${RED}✗ Missing TRADER_STATE_S3_BUCKET or TRADER_DB_URL. Persistence is required for API deploys; pass --state-s3-bucket, --db-url, or enable --ensure-resources to create a default bucket.${NC}" >&2
    exit 2
  fi
  if [[ "$DEPLOY_API" == "true" && -n "${TRADER_DB_URL:-}" && -z "${TRADER_STATE_S3_BUCKET:-}" ]]; then
    echo -e "${YELLOW}⚠ TRADER_DB_URL set without TRADER_STATE_S3_BUCKET; bot snapshots and top-combos will not persist across deploys.${NC}" >&2
  fi

  if [[ -n "${TRADER_STATE_S3_BUCKET:-}" && -z "${TRADER_STATE_S3_REGION:-}" ]]; then
    TRADER_STATE_S3_REGION="$AWS_REGION"
  fi

  if [[ -n "${TRADER_STATE_S3_BUCKET:-}" ]] && is_true "$ENSURE_RESOURCES"; then
    ensure_s3_bucket "$TRADER_STATE_S3_BUCKET" "$TRADER_STATE_S3_REGION"
    if [[ "$LAST_S3_BUCKET_CREATED" == "true" ]]; then
      apply_bucket_private_defaults "$TRADER_STATE_S3_BUCKET" "$TRADER_STATE_S3_REGION"
    fi
    if [[ -z "${APP_RUNNER_INSTANCE_ROLE_ARN:-}" ]]; then
      APP_RUNNER_INSTANCE_ROLE_ARN="$(ensure_state_instance_role "$TRADER_STATE_S3_BUCKET" "$TRADER_STATE_S3_PREFIX")"
      echo -e "${GREEN}✓ Using App Runner instance role: ${APP_RUNNER_INSTANCE_ROLE_ARN}${NC}" >&2
    fi
  fi

  if [[ "$DEPLOY_UI" == "true" && "$ui_cloudfront_enabled" == "true" ]]; then
    if is_true "$ENSURE_RESOURCES" || is_true "$UI_CLOUDFRONT_AUTO"; then
      ensure_s3_bucket "$UI_BUCKET" "$AWS_REGION"
      if [[ "$LAST_S3_BUCKET_CREATED" == "true" ]]; then
        apply_bucket_private_defaults "$UI_BUCKET" "$AWS_REGION"
      fi
    fi
    if [[ -z "${UI_DISTRIBUTION_ID:-}" ]] && is_true "$UI_CLOUDFRONT_AUTO"; then
      UI_DISTRIBUTION_ID="$(ensure_cloudfront_distribution "$UI_BUCKET")"
    fi
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]] && (is_true "$ENSURE_RESOURCES" || is_true "$UI_CLOUDFRONT_AUTO"); then
      ensure_cloudfront_bucket_policy "$UI_BUCKET" "$UI_DISTRIBUTION_ID"
    fi
  fi

  echo "Configuration:"
  echo "  Region: $AWS_REGION"
  echo "  Ensure AWS Resources: ${ENSURE_RESOURCES}"
  echo "  API Token: $(mask_token "$TRADER_API_TOKEN")"
  echo "  CORS Origin: ${TRADER_CORS_ORIGIN:-"(not set)"}"
  if [[ -n "${TRADER_DB_URL:-}" ]]; then
    echo "  Ops DB URL: (set)"
  else
    echo "  Ops DB URL: (not set)"
  fi
  if [[ -n "${TRADER_STATE_DIR:-}" ]]; then
    echo "  State Dir: ${TRADER_STATE_DIR}"
  else
    echo "  State Dir: (disabled)"
  fi
  if [[ -n "${TRADER_STATE_S3_BUCKET:-}" ]]; then
    echo "  State S3 Bucket: ${TRADER_STATE_S3_BUCKET}"
    echo "  State S3 Prefix: ${TRADER_STATE_S3_PREFIX:-"(none)"}"
    echo "  State S3 Region: ${TRADER_STATE_S3_REGION:-"(default)"}"
  fi
  if [[ -n "${APP_RUNNER_INSTANCE_ROLE_ARN:-}" ]]; then
    echo "  App Runner Instance Role: ${APP_RUNNER_INSTANCE_ROLE_ARN}"
  fi
  echo "  API Max Bars (LSTM): ${TRADER_API_MAX_BARS_LSTM}"
  echo "  API Max Hidden Size: ${TRADER_API_MAX_HIDDEN_SIZE}"
  if [[ "$DEPLOY_UI" == "true" ]]; then
    echo "  UI Bucket: ${UI_BUCKET:-"(not set)"}"
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      echo "  UI CF Dist: ${UI_DISTRIBUTION_ID}"
    fi
    if is_true "$UI_CLOUDFRONT_AUTO"; then
      echo "  UI CF Auto: ${UI_CLOUDFRONT_AUTO}"
    fi
    echo "  UI API Mode: ${UI_API_MODE}"
    echo "  UI Dist Dir: ${UI_DIST_DIR}"
    echo "  UI Skip Build: ${UI_SKIP_BUILD}"
  fi
  echo ""

  local api_url=""
  local api_token="$TRADER_API_TOKEN"
  local ui_api_url_override="${UI_API_URL:-}"
  local ui_api_url=""
  local api_url_from_service="false"
  local ui_api_fallback=""

  if [[ "$DEPLOY_API" == "true" ]]; then
    create_ecr_repo
    local ecr_uri="$ECR_URI"

    build_and_push "$ecr_uri"

    create_app_runner "$ecr_uri"
    api_url="$APP_RUNNER_SERVICE_URL"
    api_token="$TRADER_API_TOKEN"
    api_url_from_service="true"
  else
    if [[ -z "${UI_SERVICE_ARN:-}" ]]; then
      UI_SERVICE_ARN="$(discover_apprunner_service_arn_by_name "$APP_RUNNER_SERVICE_NAME" || true)"
      if [[ -n "$UI_SERVICE_ARN" ]]; then
        echo -e "${YELLOW}✓ Discovered App Runner service '${APP_RUNNER_SERVICE_NAME}' for UI config${NC}" >&2
      fi
    fi
    if [[ -n "${ui_api_url_override:-}" ]]; then
      api_url="$ui_api_url_override"
    elif [[ -n "${UI_SERVICE_ARN:-}" ]]; then
      api_url="$(discover_apprunner_service_url "$UI_SERVICE_ARN")"
      api_url_from_service="true"
    fi
    if [[ -z "${api_token:-}" && -n "${UI_SERVICE_ARN:-}" ]]; then
      api_token="$(discover_apprunner_trader_api_token "$UI_SERVICE_ARN")"
    fi
  fi

  if [[ "$DEPLOY_UI" == "true" ]]; then
    if [[ -n "${UI_DISTRIBUTION_ID:-}" && -n "${api_url:-}" && "${api_url}" != "/api" ]]; then
      ensure_cloudfront_api_behavior "$UI_DISTRIBUTION_ID" "$api_url"
    fi
    if [[ -n "${UI_DISTRIBUTION_ID:-}" ]]; then
      if [[ "$UI_API_MODE" == "direct" ]]; then
        if [[ -n "${ui_api_url_override:-}" ]]; then
          ui_api_url="$ui_api_url_override"
        else
          ui_api_url="$api_url"
        fi
      else
        if [[ -n "${ui_api_url_override:-}" && "${ui_api_url_override}" != "/api" ]]; then
          echo -e "${YELLOW}Warning: --api-url override is ignored in proxy mode (apiBaseUrl is /api). Use --ui-api-direct to use the API URL instead.${NC}" >&2
        fi
        ui_api_url="/api"
      fi
    elif [[ -n "${ui_api_url_override:-}" ]]; then
      ui_api_url="$ui_api_url_override"
    else
      ui_api_url="$api_url"
    fi
    ui_api_fallback="$UI_API_FALLBACK_URL"
    if [[ -z "$ui_api_fallback" && "$ui_api_url" == "/api" && -n "$api_url" ]]; then
      ui_api_fallback="$api_url"
      echo -e "${YELLOW}✓ Using apiFallbackUrl from API URL${NC}" >&2
    fi
    deploy_ui "$ui_api_url" "$api_token" "$ui_api_fallback"
  fi

  echo -e "${GREEN}=== Deployment Complete ===${NC}\n"
  if [[ "$DEPLOY_API" == "true" ]]; then
    echo "API URL: ${api_url}"
  else
    echo "API URL (configured): ${ui_api_url}"
  fi
  if [[ "$DEPLOY_UI" == "true" ]]; then
    echo "UI API base: ${ui_api_url}"
    if [[ -n "${ui_api_fallback:-}" ]]; then
      echo "UI API fallback: ${ui_api_fallback}"
    fi
    echo "UI uploaded to: s3://${UI_BUCKET}/"
  fi
  echo ""

  if [[ "$DEPLOY_UI" == "true" && -n "${UI_DISTRIBUTION_ID:-}" ]]; then
    describe_cloudfront_distribution "$UI_DISTRIBUTION_ID" "$ui_api_url"
    echo ""
  fi

  if [[ "$DEPLOY_API" == "true" ]]; then
    echo "Post-deploy health check:"
    health_check_api "$api_url"
    echo ""
    echo "Test the API:"
    echo "  curl -s ${api_url}/health | jq ."
    echo ""
  fi
  if [[ "$DEPLOY_UI" == "true" ]]; then
    local ui_public_url=""
    ui_public_url="$(resolve_ui_base_url "$UI_DISTRIBUTION_ID" "${UI_CLOUDFRONT_DOMAIN:-}")" || true
    echo "Post-deploy UI smoke checks:"
    smoke_check_ui "$ui_public_url" "$ui_api_url" "$api_url"
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
