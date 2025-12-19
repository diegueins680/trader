#!/usr/bin/env bash
# Deploys the web UI (haskell/web) to S3, writes trader-config.js, and optionally invalidates CloudFront.
#
# Usage:
#   bash deploy/aws/deploy-ui.sh --bucket <s3-bucket> --service-arn <apprunner-arn> [--distribution-id <cf-id>] [--region <aws-region>]
#   bash deploy/aws/deploy-ui.sh --bucket <s3-bucket> --api-url <https://api-host> --api-token <token> [--distribution-id <cf-id>] [--region <aws-region>]
#
# Notes:
#   - The UI reads deploy-time config from `trader-config.js` (apiBaseUrl, apiToken).
#   - This script avoids printing the full token; it only prints a masked version.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
BUCKET=""
DISTRIBUTION_ID=""
SERVICE_ARN=""
API_URL=""
API_TOKEN="${TRADER_API_TOKEN:-}"
SKIP_BUILD="false"
DIST_DIR="haskell/web/dist"

usage() {
  cat <<EOF
Usage:
  bash deploy/aws/deploy-ui.sh --bucket <s3-bucket> --service-arn <apprunner-arn> [--distribution-id <cf-id>] [--region <aws-region>] [--skip-build]
  bash deploy/aws/deploy-ui.sh --bucket <s3-bucket> --api-url <https://api-host> --api-token <token> [--distribution-id <cf-id>] [--region <aws-region>] [--skip-build]
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

need() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo -e "${RED}✗ Missing dependency: ${cmd}${NC}" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      AWS_REGION="${2:-}"; shift 2 ;;
    --bucket)
      BUCKET="${2:-}"; shift 2 ;;
    --distribution-id)
      DISTRIBUTION_ID="${2:-}"; shift 2 ;;
    --service-arn)
      SERVICE_ARN="${2:-}"; shift 2 ;;
    --api-url)
      API_URL="${2:-}"; shift 2 ;;
    --api-token)
      API_TOKEN="${2:-}"; shift 2 ;;
    --skip-build)
      SKIP_BUILD="true"; shift ;;
    --dist-dir)
      DIST_DIR="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}" >&2
      usage
      exit 1
      ;;
  esac
done

need aws
need npm

if [[ -z "$AWS_REGION" ]]; then
  AWS_REGION="$(aws configure get region 2>/dev/null || true)"
fi
AWS_REGION="${AWS_REGION:-ap-northeast-1}"

if [[ -z "$BUCKET" ]]; then
  echo -e "${RED}✗ Missing --bucket${NC}" >&2
  usage
  exit 1
fi

if [[ -z "$API_URL" ]]; then
  if [[ -z "$SERVICE_ARN" ]]; then
    echo -e "${RED}✗ Provide --api-url (or --service-arn to auto-discover it).${NC}" >&2
    usage
    exit 1
  fi
  host="$(
    aws apprunner describe-service \
      --service-arn "$SERVICE_ARN" \
      --region "$AWS_REGION" \
      --query 'Service.ServiceUrl' \
      --output text
  )"
  if [[ -z "$host" || "$host" == "None" ]]; then
    echo -e "${RED}✗ Unable to discover App Runner service URL from --service-arn${NC}" >&2
    exit 1
  fi
  if [[ "$host" =~ ^https?:// ]]; then
    API_URL="$host"
  else
    API_URL="https://${host}"
  fi
fi

if [[ -z "$API_TOKEN" && -n "$SERVICE_ARN" ]]; then
  # Fetch without printing (still visible to AWS APIs, but keep console clean).
  API_TOKEN="$(
    aws apprunner describe-service \
      --service-arn "$SERVICE_ARN" \
      --region "$AWS_REGION" \
      --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables.TRADER_API_TOKEN' \
      --output text 2>/dev/null || true
  )"
  if [[ "$API_TOKEN" == "None" ]]; then
    API_TOKEN=""
  fi
fi

echo -e "${GREEN}=== Deploy Trader UI ===${NC}"
echo "Region: ${AWS_REGION}"
echo "Bucket: ${BUCKET}"
echo "API:    ${API_URL}"
echo "Token:  $(mask_token "$API_TOKEN")"
if [[ -n "$DISTRIBUTION_ID" ]]; then
  echo "CF:     ${DISTRIBUTION_ID}"
fi
echo ""

if [[ "$SKIP_BUILD" != "true" ]]; then
  echo "Building UI..."
  (cd haskell/web && npm run build)
  echo -e "${GREEN}✓ UI built${NC}"
fi

if [[ ! -d "$DIST_DIR" ]]; then
  echo -e "${RED}✗ dist dir not found: ${DIST_DIR}${NC}" >&2
  exit 1
fi

echo "Writing ${DIST_DIR}/trader-config.js..."
cat > "${DIST_DIR}/trader-config.js" <<EOF
globalThis.__TRADER_CONFIG__ = {
  apiBaseUrl: "${API_URL}",
  apiToken: "${API_TOKEN}",
};
EOF

echo "Uploading UI to s3://${BUCKET}/ ..."
aws s3 sync "${DIST_DIR}/" "s3://${BUCKET}/" --delete --region "$AWS_REGION"
# Ensure config is definitely the deployed one (avoid stale placeholder overwrites).
aws s3 cp "${DIST_DIR}/trader-config.js" "s3://${BUCKET}/trader-config.js" --region "$AWS_REGION" >/dev/null
echo -e "${GREEN}✓ Uploaded${NC}"

if [[ -n "$DISTRIBUTION_ID" ]]; then
  echo "Invalidating CloudFront..."
  aws cloudfront create-invalidation --distribution-id "$DISTRIBUTION_ID" --paths "/*" >/dev/null
  echo -e "${GREEN}✓ Invalidated${NC}"
fi

echo ""
echo -e "${GREEN}Done.${NC}"
