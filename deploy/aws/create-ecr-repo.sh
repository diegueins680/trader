#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  AWS_REGION=ap-northeast-1 ECR_REPO=trader-api bash deploy/aws/create-ecr-repo.sh

Environment variables:
  AWS_REGION             AWS region (default: ap-northeast-1)
  ECR_REPO               ECR repository name (default: trader-api)
  ECR_TAG_MUTABILITY     MUTABLE|IMMUTABLE (default: MUTABLE)
  ECR_SCAN_ON_PUSH       true|false (default: true)

Output:
  Prints the repository URI (e.g., 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/trader-api)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: AWS CLI ('aws') not found. Install and configure it first." >&2
  exit 1
fi

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-northeast-1}}"
ECR_REPO="${ECR_REPO:-trader-api}"
ECR_TAG_MUTABILITY="${ECR_TAG_MUTABILITY:-MUTABLE}"
ECR_SCAN_ON_PUSH="${ECR_SCAN_ON_PUSH:-true}"

if [[ "$AWS_REGION" =~ [a-z]$ ]]; then
  echo "Error: AWS_REGION '$AWS_REGION' looks like an Availability Zone (e.g. ap-northeast-1a). Use a region like ap-northeast-1." >&2
  exit 2
fi

if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  :
else
  aws ecr create-repository \
    --repository-name "$ECR_REPO" \
    --image-scanning-configuration "scanOnPush=$ECR_SCAN_ON_PUSH" \
    --image-tag-mutability "$ECR_TAG_MUTABILITY" \
    --region "$AWS_REGION" \
    >/dev/null
fi

aws ecr describe-repositories \
  --repository-names "$ECR_REPO" \
  --region "$AWS_REGION" \
  --query 'repositories[0].repositoryUri' \
  --output text
