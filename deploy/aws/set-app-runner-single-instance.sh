#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy/aws/set-app-runner-single-instance.sh --service-arn <arn> [--min 1] [--max 1] [--name-prefix trader-api-single]

Environment variables (optional):
  AWS_REGION                 AWS region to use (otherwise aws-cli default)
  APP_RUNNER_SERVICE_ARN     Alternative to --service-arn

Example:
  AWS_REGION=ap-northeast-1 \
    deploy/aws/set-app-runner-single-instance.sh --service-arn arn:aws:apprunner:...:service/... \
    --min 1 --max 1
EOF
}

service_arn="${APP_RUNNER_SERVICE_ARN:-}"
min_size="1"
max_size="1"
name_prefix="trader-api-single"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --service-arn)
      service_arn="${2:-}"
      shift 2
      ;;
    --min)
      min_size="${2:-}"
      shift 2
      ;;
    --max)
      max_size="${2:-}"
      shift 2
      ;;
    --name-prefix)
      name_prefix="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${service_arn}" ]]; then
  echo "Missing --service-arn (or APP_RUNNER_SERVICE_ARN)." >&2
  usage >&2
  exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install it first: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" >&2
  exit 127
fi

aws_args=()
if [[ -n "${AWS_REGION:-}" ]]; then
  aws_args+=(--region "$AWS_REGION")
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
cfg_name="${name_prefix}-${min_size}-${max_size}-${ts}"

cfg_arn="$(
  aws "${aws_args[@]}" apprunner create-auto-scaling-configuration \
    --auto-scaling-configuration-name "$cfg_name" \
    --min-size "$min_size" \
    --max-size "$max_size" \
    --query 'AutoScalingConfiguration.AutoScalingConfigurationArn' \
    --output text
)"

aws "${aws_args[@]}" apprunner update-service \
  --service-arn "$service_arn" \
  --auto-scaling-configuration-arn "$cfg_arn" \
  >/dev/null

echo "Updated App Runner service scaling."
echo "Service ARN: $service_arn"
echo "Auto scaling config ARN: $cfg_arn"
aws "${aws_args[@]}" apprunner describe-service \
  --service-arn "$service_arn" \
  --query 'Service.AutoScalingConfigurationSummary' \
  --output json

