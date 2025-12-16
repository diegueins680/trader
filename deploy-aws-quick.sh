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

echo -e "${GREEN}=== Trader AWS Deployment Script ===${NC}\n"

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

# Create ECR repository
create_ecr_repo() {
  echo "Creating ECR repository..."
  
  local account_id=$(get_account_id)
  local ecr_uri="${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
  
  if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo -e "${YELLOW}✓ ECR repository already exists${NC}"
  else
    aws ecr create-repository \
      --repository-name "$ECR_REPO" \
      --image-scanning-configuration scanOnPush=true \
      --region "$AWS_REGION" >/dev/null
    echo -e "${GREEN}✓ ECR repository created${NC}"
  fi
  
  echo "$ecr_uri"
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
  local ecr_uri="$1"
  
  echo "Creating App Runner service..."
  
  # Check if service already exists
  if aws apprunner describe-service --service-arn "arn:aws:apprunner:${AWS_REGION}:$(get_account_id):service/${ECR_REPO}" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo -e "${YELLOW}✓ App Runner service already exists${NC}"
    aws apprunner describe-service \
      --service-arn "arn:aws:apprunner:${AWS_REGION}:$(get_account_id):service/${ECR_REPO}" \
      --region "$AWS_REGION" \
      --query 'Service.ServiceUrl' \
      --output text
    return
  fi
  
  # Create environment variables JSON
  local env_vars='[{"Name":"TRADER_API_ASYNC_DIR","Value":"/var/lib/trader/async"}'
  if [[ -n "$TRADER_API_TOKEN" ]]; then
    env_vars="${env_vars},$(printf '{"Name":"TRADER_API_TOKEN","Value":"%s"}' "$TRADER_API_TOKEN")"
  fi
  env_vars="${env_vars}]"
  
  # Create service
  local service_arn=$(aws apprunner create-service \
    --region "$AWS_REGION" \
    --service-name "$ECR_REPO" \
    --source-configuration '{"ImageRepository":{"ImageIdentifier":"'"$ecr_uri"'","ImageRepositoryType":"ECR","ImageConfiguration":{"Port":"8080"}}}' \
    --instance-configuration '{"Cpu":"1024","Memory":"2048"}' \
    --environment-variables "$env_vars" \
    --tags Key=Name,Value=trader-api \
    --query 'Service.ServiceArn' \
    --output text)
  
  echo -e "${GREEN}✓ App Runner service created${NC}"
  
  # Get service URL
  echo "Waiting for service to be active (this may take a few minutes)..."
  local max_attempts=60
  local attempt=0
  
  while [[ $attempt -lt $max_attempts ]]; do
    local status=$(aws apprunner describe-service --service-arn "$service_arn" --region "$AWS_REGION" --query 'Service.Status' --output text 2>/dev/null || echo "")
    
    if [[ "$status" == "ACTIVE" ]]; then
      echo -e "${GREEN}✓ Service is active${NC}"
      break
    fi
    
    if [[ "$status" == "FAILED" ]]; then
      echo -e "${RED}✗ Service failed to start${NC}"
      exit 1
    fi
    
    echo -n "."
    sleep 5
    ((attempt++))
  done
  
  aws apprunner describe-service --service-arn "$service_arn" --region "$AWS_REGION" --query 'Service.ServiceUrl' --output text
}

# Main execution
main() {
  check_prerequisites
  
  echo "Configuration:"
  echo "  Region: $AWS_REGION"
  echo "  API Token: ${TRADER_API_TOKEN:-(not set)}"
  echo ""
  
  # Create ECR repo
  local ecr_uri=$(create_ecr_repo)
  
  # Build and push image
  build_and_push "$ecr_uri"
  
  # Create App Runner service and get URL
  local service_url=$(create_app_runner "$ecr_uri")
  
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
