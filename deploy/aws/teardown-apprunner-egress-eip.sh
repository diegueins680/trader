#!/usr/bin/env bash
# Teardown resources created by setup-apprunner-egress-eip.sh.
#
# This script is destructive. Use --confirm to proceed.

set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  cat <<'USAGE'
Usage:
  bash deploy/aws/teardown-apprunner-egress-eip.sh --confirm [options]

Options:
  --region <region>            AWS region (default: AWS_REGION/AWS_DEFAULT_REGION, then ap-northeast-1)
  --resource-prefix <prefix>   Resource name prefix (default: trader-apprunner-egress)
  --service-name <name>        App Runner service name to detach VPC egress (optional)
  --service-arn <arn>          App Runner service ARN to detach VPC egress (optional)
  --vpc-id <vpc-id>            VPC ID (optional; inferred from tags if omitted)
  --confirm                    Required. Perform deletions.
  -h, --help                   Show this help

Notes:
  - If the App Runner service still uses the VPC connector, pass --service-arn or --service-name
    so the script can switch egress back to DEFAULT.
  - Only resources tagged ManagedBy=trader and Purpose=apprunner-egress are targeted.
USAGE
}

log_info() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err() { echo -e "${RED}[ERR]${NC} $*"; }

die() {
  log_err "$*"
  exit 1
}

is_none() {
  [[ -z "$1" || "$1" == "None" || "$1" == "null" ]]
}

aws_cli() {
  aws --no-cli-pager --region "$AWS_REGION" "$@"
}

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
RESOURCE_PREFIX="${RESOURCE_PREFIX:-trader-apprunner-egress}"
SERVICE_NAME="${SERVICE_NAME:-}"
SERVICE_ARN="${SERVICE_ARN:-}"
VPC_ID="${VPC_ID:-}"
CONFIRM="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      AWS_REGION="$2"
      shift 2
      ;;
    --resource-prefix)
      RESOURCE_PREFIX="$2"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --service-arn)
      SERVICE_ARN="$2"
      shift 2
      ;;
    --vpc-id)
      VPC_ID="$2"
      shift 2
      ;;
    --confirm)
      CONFIRM="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

if ! command -v aws >/dev/null 2>&1; then
  die "aws CLI not found. Install it before running this script."
fi

if [[ -z "${AWS_REGION:-}" ]]; then
  AWS_REGION="$(aws configure get region 2>/dev/null || true)"
fi
AWS_REGION="${AWS_REGION:-ap-northeast-1}"

TAG_MANAGED_BY="trader"
TAG_PURPOSE="apprunner-egress"

VPC_NAME="${RESOURCE_PREFIX}"
PUBLIC_SUBNET_NAME="${RESOURCE_PREFIX}-public"
PRIVATE_SUBNET_NAME="${RESOURCE_PREFIX}-private"
IGW_NAME="${RESOURCE_PREFIX}-igw"
NAT_NAME="${RESOURCE_PREFIX}-nat"
EIP_NAME="${RESOURCE_PREFIX}-eip"
PUBLIC_RT_NAME="${RESOURCE_PREFIX}-public-rt"
PRIVATE_RT_NAME="${RESOURCE_PREFIX}-private-rt"
SG_NAME="${RESOURCE_PREFIX}-sg"
VPC_CONNECTOR_NAME="${RESOURCE_PREFIX}-connector"

resolve_service_arn() {
  if is_none "$SERVICE_ARN" && ! is_none "$SERVICE_NAME"; then
    SERVICE_ARN="$(aws_cli apprunner list-services \
      --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" --output text)"
    if [[ "$SERVICE_ARN" == "None" ]]; then
      SERVICE_ARN=""
    fi
  fi
}

maybe_detach_service() {
  if is_none "$SERVICE_ARN"; then
    return
  fi

  local ingress_public
  ingress_public="$(aws_cli apprunner describe-service --service-arn "$SERVICE_ARN" --query 'Service.NetworkConfiguration.IngressConfiguration.IsPubliclyAccessible' --output text)"
  local ip_address_type
  ip_address_type="$(aws_cli apprunner describe-service --service-arn "$SERVICE_ARN" --query 'Service.NetworkConfiguration.IpAddressType' --output text)"

  local ingress_value=""
  if [[ "$ingress_public" == "True" || "$ingress_public" == "true" ]]; then
    ingress_value="true"
  elif [[ "$ingress_public" == "False" || "$ingress_public" == "false" ]]; then
    ingress_value="false"
  fi

  local network_config="EgressConfiguration={EgressType=DEFAULT}"
  if [[ -n "$ingress_value" ]]; then
    network_config+=" ,IngressConfiguration={IsPubliclyAccessible=${ingress_value}}"
  fi
  if ! is_none "$ip_address_type"; then
    network_config+=" ,IpAddressType=${ip_address_type}"
  fi
  network_config="${network_config// /}"

  log_warn "Updating App Runner service to use DEFAULT egress"
  aws_cli apprunner update-service --service-arn "$SERVICE_ARN" --network-configuration "$network_config" >/dev/null

  for _ in {1..60}; do
    status="$(aws_cli apprunner describe-service --service-arn "$SERVICE_ARN" --query 'Service.Status' --output text)"
    if [[ "$status" == "RUNNING" ]]; then
      log_info "Service is RUNNING"
      return
    fi
    if [[ "$status" == "OPERATION_FAILED" || "$status" == "FAILED" ]]; then
      die "Service update failed; check App Runner events/logs."
    fi
    sleep 10
  done

  log_warn "Timed out waiting for service to reach RUNNING"
}

find_resource_id() {
  local resource_type="$1"
  local name="$2"
  local query="$3"
  local id
  id="$(aws_cli ec2 "describe-${resource_type}" \
    --filters "Name=tag:Name,Values=${name}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
    --query "$query" --output text)"
  if [[ "$id" == "None" ]]; then
    id=""
  fi
  echo "$id"
}

find_eip_alloc_id() {
  local id
  id="$(aws_cli ec2 describe-addresses \
    --filters "Name=tag:Name,Values=${EIP_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
    --query 'Addresses[0].AllocationId' --output text)"
  if [[ "$id" == "None" ]]; then
    id=""
  fi
  echo "$id"
}

find_vpc_connector_arn() {
  local arn
  arn="$(aws_cli apprunner list-vpc-connectors \
    --query "VpcConnectors[?VpcConnectorName=='${VPC_CONNECTOR_NAME}'].VpcConnectorArn | [0]" --output text)"
  if [[ "$arn" == "None" ]]; then
    arn=""
  fi
  echo "$arn"
}

discover_resources() {
  if is_none "$VPC_ID"; then
    VPC_ID="$(find_resource_id vpcs "$VPC_NAME" 'Vpcs[0].VpcId')"
  fi

  PUBLIC_SUBNET_ID="$(find_resource_id subnets "$PUBLIC_SUBNET_NAME" 'Subnets[0].SubnetId')"
  PRIVATE_SUBNET_ID="$(find_resource_id subnets "$PRIVATE_SUBNET_NAME" 'Subnets[0].SubnetId')"
  IGW_ID="$(find_resource_id internet-gateways "$IGW_NAME" 'InternetGateways[0].InternetGatewayId')"
  PUBLIC_RT_ID="$(find_resource_id route-tables "$PUBLIC_RT_NAME" 'RouteTables[0].RouteTableId')"
  PRIVATE_RT_ID="$(find_resource_id route-tables "$PRIVATE_RT_NAME" 'RouteTables[0].RouteTableId')"
  SG_ID="$(find_resource_id security-groups "$SG_NAME" 'SecurityGroups[0].GroupId')"
  NAT_ID="$(aws_cli ec2 describe-nat-gateways \
    --filter "Name=tag:Name,Values=${NAT_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
    --query 'NatGateways[0].NatGatewayId' --output text)"
  if [[ "$NAT_ID" == "None" ]]; then
    NAT_ID=""
  fi
  EIP_ALLOC_ID="$(find_eip_alloc_id)"
  VPC_CONNECTOR_ARN="$(find_vpc_connector_arn)"
}

print_plan() {
  echo "Resources to remove (if present):"
  echo "  VPC Connector: ${VPC_CONNECTOR_ARN:-<none>}"
  echo "  NAT Gateway:   ${NAT_ID:-<none>}"
  echo "  Elastic IP:    ${EIP_ALLOC_ID:-<none>}"
  echo "  Route tables:  ${PUBLIC_RT_ID:-<none>} ${PRIVATE_RT_ID:-<none>}"
  echo "  Subnets:       ${PUBLIC_SUBNET_ID:-<none>} ${PRIVATE_SUBNET_ID:-<none>}"
  echo "  IGW:           ${IGW_ID:-<none>}"
  echo "  Security grp:  ${SG_ID:-<none>}"
  echo "  VPC:           ${VPC_ID:-<none>}"
}

remove_vpc_connector() {
  if is_none "$VPC_CONNECTOR_ARN"; then
    return
  fi
  log_warn "Deleting VPC connector ${VPC_CONNECTOR_ARN}"
  aws_cli apprunner delete-vpc-connector --vpc-connector-arn "$VPC_CONNECTOR_ARN" >/dev/null
  for _ in {1..60}; do
    if ! aws_cli apprunner describe-vpc-connector --vpc-connector-arn "$VPC_CONNECTOR_ARN" >/dev/null 2>&1; then
      log_info "VPC connector deleted"
      return
    fi
    status="$(aws_cli apprunner describe-vpc-connector --vpc-connector-arn "$VPC_CONNECTOR_ARN" --query 'VpcConnector.Status' --output text 2>/dev/null || true)"
    if [[ "$status" == "DELETED" ]]; then
      log_info "VPC connector deleted"
      return
    fi
    sleep 5
  done
  log_warn "Timed out waiting for VPC connector deletion"
}

remove_nat_gateway() {
  if is_none "$NAT_ID"; then
    return
  fi
  log_warn "Deleting NAT Gateway ${NAT_ID}"
  aws_cli ec2 delete-nat-gateway --nat-gateway-id "$NAT_ID" >/dev/null
  aws_cli ec2 wait nat-gateway-deleted --nat-gateway-ids "$NAT_ID" || true
  log_info "NAT Gateway deleted"
}

remove_route_table() {
  local rt_id="$1"
  if is_none "$rt_id"; then
    return
  fi
  local assoc_ids
  assoc_ids="$(aws_cli ec2 describe-route-tables --route-table-ids "$rt_id" \
    --query 'RouteTables[0].Associations[?Main==`false`].RouteTableAssociationId' --output text)"
  for assoc in $assoc_ids; do
    aws_cli ec2 disassociate-route-table --association-id "$assoc" >/dev/null || true
  done
  log_warn "Deleting route table ${rt_id}"
  aws_cli ec2 delete-route-table --route-table-id "$rt_id" >/dev/null || true
}

remove_subnet() {
  local subnet_id="$1"
  if is_none "$subnet_id"; then
    return
  fi
  log_warn "Deleting subnet ${subnet_id}"
  aws_cli ec2 delete-subnet --subnet-id "$subnet_id" >/dev/null || true
}

remove_igw() {
  if is_none "$IGW_ID"; then
    return
  fi
  if ! is_none "$VPC_ID"; then
    aws_cli ec2 detach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID" >/dev/null || true
  fi
  log_warn "Deleting IGW ${IGW_ID}"
  aws_cli ec2 delete-internet-gateway --internet-gateway-id "$IGW_ID" >/dev/null || true
}

remove_security_group() {
  if is_none "$SG_ID"; then
    return
  fi
  log_warn "Deleting security group ${SG_ID}"
  aws_cli ec2 delete-security-group --group-id "$SG_ID" >/dev/null || true
}

release_eip() {
  if is_none "$EIP_ALLOC_ID"; then
    return
  fi
  log_warn "Releasing Elastic IP ${EIP_ALLOC_ID}"
  aws_cli ec2 release-address --allocation-id "$EIP_ALLOC_ID" >/dev/null || true
}

remove_vpc() {
  if is_none "$VPC_ID"; then
    return
  fi
  log_warn "Deleting VPC ${VPC_ID}"
  aws_cli ec2 delete-vpc --vpc-id "$VPC_ID" >/dev/null || true
}

main() {
  resolve_service_arn
  discover_resources
  print_plan

  if [[ "$CONFIRM" != "true" ]]; then
    echo
    log_warn "Dry run only. Re-run with --confirm to delete these resources."
    exit 0
  fi

  maybe_detach_service
  remove_vpc_connector
  remove_nat_gateway
  remove_route_table "$PRIVATE_RT_ID"
  remove_route_table "$PUBLIC_RT_ID"
  remove_subnet "$PRIVATE_SUBNET_ID"
  remove_subnet "$PUBLIC_SUBNET_ID"
  remove_igw
  remove_security_group
  release_eip
  remove_vpc

  log_info "Teardown complete."
}

main "$@"
