#!/usr/bin/env bash
# Provision a fixed egress IP for AWS App Runner by routing outbound traffic
# through a VPC connector + NAT Gateway with an Elastic IP.
#
# This script creates (or reuses) a small VPC footprint and attaches the
# App Runner service to it. It prints the Elastic IP to allowlist.
#
# Costs: NAT Gateways incur hourly + data charges. Remove resources when
# no longer needed.

set -euo pipefail
IFS=$'\n\t'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  cat <<'USAGE'
Usage:
  bash deploy/aws/setup-apprunner-egress-eip.sh --service-name <name> [--region <region>]
  bash deploy/aws/setup-apprunner-egress-eip.sh --service-arn <arn> [--region <region>]

Options:
  --region <region>            AWS region (default: AWS_REGION/AWS_DEFAULT_REGION, then ap-northeast-1)
  --service-name <name>        App Runner service name (used to resolve ARN)
  --service-arn <arn>          App Runner service ARN
  --resource-prefix <prefix>   Resource name prefix (default: trader-apprunner-egress)
  --vpc-id <vpc-id>            Use an existing VPC (script will create tagged subnets/routes/NAT inside it)
  --public-subnet-id <id>      Use an existing public subnet (route table will be attached)
  --private-subnet-id <id>     Use an existing private subnet (route table will be attached)
  --security-group-id <id>     Use an existing security group for the VPC connector
  --connector-name <name>      App Runner VPC connector name override
  --az <availability-zone>     Availability Zone for created subnets
  --no-update                  Do not update the App Runner service (just create/reuse infra)
  -h, --help                   Show this help

Environment:
  AWS_REGION / AWS_DEFAULT_REGION
  RESOURCE_PREFIX, VPC_CIDR, PUBLIC_SUBNET_CIDR, PRIVATE_SUBNET_CIDR (optional overrides)

Notes:
  - If you pass existing subnets, this script will associate new route tables
    to them to ensure NAT egress.
  - App Runner inbound traffic remains public by default; this script preserves
    the existing ingress setting.
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
VPC_CIDR="${VPC_CIDR:-10.250.0.0/16}"
PUBLIC_SUBNET_CIDR="${PUBLIC_SUBNET_CIDR:-10.250.0.0/24}"
PRIVATE_SUBNET_CIDR="${PRIVATE_SUBNET_CIDR:-10.250.1.0/24}"

SERVICE_NAME="${SERVICE_NAME:-}"
SERVICE_ARN="${SERVICE_ARN:-}"
VPC_ID="${VPC_ID:-}"
PUBLIC_SUBNET_ID="${PUBLIC_SUBNET_ID:-}"
PRIVATE_SUBNET_ID="${PRIVATE_SUBNET_ID:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
EIP_ALLOC_ID="${EIP_ALLOC_ID:-}"
VPC_CONNECTOR_NAME="${VPC_CONNECTOR_NAME:-}"
AVAILABILITY_ZONE="${AVAILABILITY_ZONE:-}"
NO_UPDATE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      AWS_REGION="$2"
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
    --resource-prefix)
      RESOURCE_PREFIX="$2"
      shift 2
      ;;
    --vpc-id)
      VPC_ID="$2"
      shift 2
      ;;
    --public-subnet-id)
      PUBLIC_SUBNET_ID="$2"
      shift 2
      ;;
    --private-subnet-id)
      PRIVATE_SUBNET_ID="$2"
      shift 2
      ;;
    --security-group-id)
      SECURITY_GROUP_ID="$2"
      shift 2
      ;;
    --connector-name)
      VPC_CONNECTOR_NAME="$2"
      shift 2
      ;;
    --az)
      AVAILABILITY_ZONE="$2"
      shift 2
      ;;
    --no-update)
      NO_UPDATE="true"
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

VPC_NAME="${VPC_NAME:-$RESOURCE_PREFIX}"
PUBLIC_SUBNET_NAME="${PUBLIC_SUBNET_NAME:-${RESOURCE_PREFIX}-public}"
PRIVATE_SUBNET_NAME="${PRIVATE_SUBNET_NAME:-${RESOURCE_PREFIX}-private}"
IGW_NAME="${IGW_NAME:-${RESOURCE_PREFIX}-igw}"
NAT_NAME="${NAT_NAME:-${RESOURCE_PREFIX}-nat}"
EIP_NAME="${EIP_NAME:-${RESOURCE_PREFIX}-eip}"
PUBLIC_RT_NAME="${PUBLIC_RT_NAME:-${RESOURCE_PREFIX}-public-rt}"
PRIVATE_RT_NAME="${PRIVATE_RT_NAME:-${RESOURCE_PREFIX}-private-rt}"
SG_NAME="${SG_NAME:-${RESOURCE_PREFIX}-sg}"
VPC_CONNECTOR_NAME="${VPC_CONNECTOR_NAME:-${RESOURCE_PREFIX}-connector}"

TAG_MANAGED_BY="trader"
TAG_PURPOSE="apprunner-egress"

vpc_tag_spec() {
  local name="$1"
  echo "ResourceType=vpc,Tags=[{Key=Name,Value=${name}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]"
}

subnet_tag_spec() {
  local name="$1"
  echo "ResourceType=subnet,Tags=[{Key=Name,Value=${name}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]"
}

rt_tag_spec() {
  local name="$1"
  echo "ResourceType=route-table,Tags=[{Key=Name,Value=${name}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]"
}

igw_tag_spec() {
  local name="$1"
  echo "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${name}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]"
}

nat_tag_spec() {
  local name="$1"
  echo "ResourceType=natgateway,Tags=[{Key=Name,Value=${name}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]"
}

ensure_vpc() {
  if is_none "$VPC_ID"; then
    VPC_ID="$(aws_cli ec2 describe-vpcs \
      --filters "Name=tag:Name,Values=${VPC_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" "Name=tag:Purpose,Values=${TAG_PURPOSE}" \
      --query 'Vpcs[0].VpcId' --output text)"
  fi

  if is_none "$VPC_ID"; then
    log_warn "Creating VPC ${VPC_NAME} (${VPC_CIDR})"
    VPC_ID="$(aws_cli ec2 create-vpc \
      --cidr-block "$VPC_CIDR" \
      --tag-specifications "$(vpc_tag_spec "$VPC_NAME")" \
      --query 'Vpc.VpcId' --output text)"
    aws_cli ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-support '{"Value":true}' >/dev/null
    aws_cli ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-hostnames '{"Value":true}' >/dev/null
    aws_cli ec2 wait vpc-available --vpc-ids "$VPC_ID"
    log_info "Created VPC ${VPC_ID}"
  else
    log_info "Using VPC ${VPC_ID}"
  fi
}

ensure_igw() {
  IGW_ID="$(aws_cli ec2 describe-internet-gateways \
    --filters "Name=attachment.vpc-id,Values=${VPC_ID}" \
    --query 'InternetGateways[0].InternetGatewayId' --output text)"

  if is_none "$IGW_ID"; then
    log_warn "Creating Internet Gateway"
    IGW_ID="$(aws_cli ec2 create-internet-gateway \
      --tag-specifications "$(igw_tag_spec "$IGW_NAME")" \
      --query 'InternetGateway.InternetGatewayId' --output text)"
    aws_cli ec2 attach-internet-gateway --internet-gateway-id "$IGW_ID" --vpc-id "$VPC_ID" >/dev/null
    log_info "Created and attached IGW ${IGW_ID}"
  else
    log_info "Using IGW ${IGW_ID}"
  fi
}

ensure_subnets() {
  if is_none "$AVAILABILITY_ZONE"; then
    AVAILABILITY_ZONE="$(aws_cli ec2 describe-availability-zones --query 'AvailabilityZones[0].ZoneName' --output text)"
  fi

  if is_none "$PUBLIC_SUBNET_ID"; then
    PUBLIC_SUBNET_ID="$(aws_cli ec2 describe-subnets \
      --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=${PUBLIC_SUBNET_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
      --query 'Subnets[0].SubnetId' --output text)"
  fi

  if is_none "$PUBLIC_SUBNET_ID"; then
    log_warn "Creating public subnet ${PUBLIC_SUBNET_NAME} (${PUBLIC_SUBNET_CIDR}) in ${AVAILABILITY_ZONE}"
    PUBLIC_SUBNET_ID="$(aws_cli ec2 create-subnet \
      --vpc-id "$VPC_ID" \
      --cidr-block "$PUBLIC_SUBNET_CIDR" \
      --availability-zone "$AVAILABILITY_ZONE" \
      --tag-specifications "$(subnet_tag_spec "$PUBLIC_SUBNET_NAME")" \
      --query 'Subnet.SubnetId' --output text)"
    aws_cli ec2 modify-subnet-attribute --subnet-id "$PUBLIC_SUBNET_ID" --map-public-ip-on-launch >/dev/null
    log_info "Created public subnet ${PUBLIC_SUBNET_ID}"
  else
    log_info "Using public subnet ${PUBLIC_SUBNET_ID}"
  fi

  if is_none "$PRIVATE_SUBNET_ID"; then
    PRIVATE_SUBNET_ID="$(aws_cli ec2 describe-subnets \
      --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=${PRIVATE_SUBNET_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
      --query 'Subnets[0].SubnetId' --output text)"
  fi

  if is_none "$PRIVATE_SUBNET_ID"; then
    log_warn "Creating private subnet ${PRIVATE_SUBNET_NAME} (${PRIVATE_SUBNET_CIDR}) in ${AVAILABILITY_ZONE}"
    PRIVATE_SUBNET_ID="$(aws_cli ec2 create-subnet \
      --vpc-id "$VPC_ID" \
      --cidr-block "$PRIVATE_SUBNET_CIDR" \
      --availability-zone "$AVAILABILITY_ZONE" \
      --tag-specifications "$(subnet_tag_spec "$PRIVATE_SUBNET_NAME")" \
      --query 'Subnet.SubnetId' --output text)"
    log_info "Created private subnet ${PRIVATE_SUBNET_ID}"
  else
    log_info "Using private subnet ${PRIVATE_SUBNET_ID}"
  fi
}

ensure_route_tables() {
  PUBLIC_RT_ID="$(aws_cli ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=${PUBLIC_RT_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
    --query 'RouteTables[0].RouteTableId' --output text)"

  if is_none "$PUBLIC_RT_ID"; then
    log_warn "Creating public route table"
    PUBLIC_RT_ID="$(aws_cli ec2 create-route-table \
      --vpc-id "$VPC_ID" \
      --tag-specifications "$(rt_tag_spec "$PUBLIC_RT_NAME")" \
      --query 'RouteTable.RouteTableId' --output text)"
  fi

  aws_cli ec2 replace-route --route-table-id "$PUBLIC_RT_ID" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW_ID" >/dev/null 2>&1 \
    || aws_cli ec2 create-route --route-table-id "$PUBLIC_RT_ID" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW_ID" >/dev/null

  PUBLIC_ASSOC_ID="$(aws_cli ec2 describe-route-tables --route-table-ids "$PUBLIC_RT_ID" \
    --query "RouteTables[0].Associations[?SubnetId=='${PUBLIC_SUBNET_ID}'].RouteTableAssociationId | [0]" --output text)"
  if is_none "$PUBLIC_ASSOC_ID"; then
    aws_cli ec2 associate-route-table --route-table-id "$PUBLIC_RT_ID" --subnet-id "$PUBLIC_SUBNET_ID" >/dev/null
  fi

  PRIVATE_RT_ID="$(aws_cli ec2 describe-route-tables \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=${PRIVATE_RT_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
    --query 'RouteTables[0].RouteTableId' --output text)"

  if is_none "$PRIVATE_RT_ID"; then
    log_warn "Creating private route table"
    PRIVATE_RT_ID="$(aws_cli ec2 create-route-table \
      --vpc-id "$VPC_ID" \
      --tag-specifications "$(rt_tag_spec "$PRIVATE_RT_NAME")" \
      --query 'RouteTable.RouteTableId' --output text)"
  fi

  aws_cli ec2 replace-route --route-table-id "$PRIVATE_RT_ID" --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "$NAT_ID" >/dev/null 2>&1 \
    || aws_cli ec2 create-route --route-table-id "$PRIVATE_RT_ID" --destination-cidr-block 0.0.0.0/0 --nat-gateway-id "$NAT_ID" >/dev/null

  PRIVATE_ASSOC_ID="$(aws_cli ec2 describe-route-tables --route-table-ids "$PRIVATE_RT_ID" \
    --query "RouteTables[0].Associations[?SubnetId=='${PRIVATE_SUBNET_ID}'].RouteTableAssociationId | [0]" --output text)"
  if is_none "$PRIVATE_ASSOC_ID"; then
    aws_cli ec2 associate-route-table --route-table-id "$PRIVATE_RT_ID" --subnet-id "$PRIVATE_SUBNET_ID" >/dev/null
  fi
}

ensure_eip_and_nat() {
  NAT_ID="$(aws_cli ec2 describe-nat-gateways \
    --filter "Name=vpc-id,Values=${VPC_ID}" "Name=tag:Name,Values=${NAT_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
    --query 'NatGateways[0].NatGatewayId' --output text)"

  if ! is_none "$NAT_ID"; then
    NAT_STATE="$(aws_cli ec2 describe-nat-gateways --nat-gateway-ids "$NAT_ID" --query 'NatGateways[0].State' --output text)"
    if [[ "$NAT_STATE" == "deleted" || "$NAT_STATE" == "failed" ]]; then
      log_warn "Existing NAT Gateway ${NAT_ID} is in state ${NAT_STATE}; creating a new one"
      NAT_ID=""
    fi
  fi

  if is_none "$NAT_ID"; then
    if is_none "$EIP_ALLOC_ID"; then
      EIP_ALLOC_ID="$(aws_cli ec2 describe-addresses \
        --filters "Name=tag:Name,Values=${EIP_NAME}" "Name=tag:ManagedBy,Values=${TAG_MANAGED_BY}" \
        --query 'Addresses[0].AllocationId' --output text)"
    fi

    if is_none "$EIP_ALLOC_ID"; then
      log_warn "Allocating Elastic IP"
      EIP_ALLOC_ID="$(aws_cli ec2 allocate-address --domain vpc \
        --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=${EIP_NAME}},{Key=ManagedBy,Value=${TAG_MANAGED_BY}},{Key=Purpose,Value=${TAG_PURPOSE}}]" \
        --query 'AllocationId' --output text)"
    fi

    log_warn "Creating NAT Gateway"
    NAT_ID="$(aws_cli ec2 create-nat-gateway \
      --subnet-id "$PUBLIC_SUBNET_ID" \
      --allocation-id "$EIP_ALLOC_ID" \
      --tag-specifications "$(nat_tag_spec "$NAT_NAME")" \
      --query 'NatGateway.NatGatewayId' --output text)"
  fi

  log_info "Waiting for NAT Gateway ${NAT_ID} to become available"
  aws_cli ec2 wait nat-gateway-available --nat-gateway-ids "$NAT_ID"

  NAT_PUBLIC_IP="$(aws_cli ec2 describe-nat-gateways --nat-gateway-ids "$NAT_ID" --query 'NatGateways[0].NatGatewayAddresses[0].PublicIp' --output text)"
  NAT_EIP_ALLOC_ID="$(aws_cli ec2 describe-nat-gateways --nat-gateway-ids "$NAT_ID" --query 'NatGateways[0].NatGatewayAddresses[0].AllocationId' --output text)"
}

ensure_security_group() {
  if is_none "$SECURITY_GROUP_ID"; then
    SECURITY_GROUP_ID="$(aws_cli ec2 describe-security-groups \
      --filters "Name=vpc-id,Values=${VPC_ID}" "Name=group-name,Values=${SG_NAME}" \
      --query 'SecurityGroups[0].GroupId' --output text)"
  fi

  if is_none "$SECURITY_GROUP_ID"; then
    log_warn "Creating security group ${SG_NAME}"
    SECURITY_GROUP_ID="$(aws_cli ec2 create-security-group \
      --vpc-id "$VPC_ID" \
      --group-name "$SG_NAME" \
      --description "App Runner egress connector" \
      --query 'GroupId' --output text)"
    aws_cli ec2 create-tags --resources "$SECURITY_GROUP_ID" \
      --tags Key=Name,Value="$SG_NAME" Key=ManagedBy,Value="$TAG_MANAGED_BY" Key=Purpose,Value="$TAG_PURPOSE" >/dev/null
  fi
  log_info "Using security group ${SECURITY_GROUP_ID}"
}

ensure_vpc_connector() {
  VPC_CONNECTOR_ARN="$(aws_cli apprunner list-vpc-connectors \
    --query "VpcConnectors[?VpcConnectorName=='${VPC_CONNECTOR_NAME}'].VpcConnectorArn | [0]" --output text)"

  VPC_CONNECTOR_STATUS="$(aws_cli apprunner list-vpc-connectors \
    --query "VpcConnectors[?VpcConnectorName=='${VPC_CONNECTOR_NAME}'].Status | [0]" --output text)"

  if ! is_none "$VPC_CONNECTOR_ARN" && [[ "$VPC_CONNECTOR_STATUS" == "ACTIVE" ]]; then
    log_info "Using existing VPC connector ${VPC_CONNECTOR_NAME}"
    return
  fi

  if ! is_none "$VPC_CONNECTOR_ARN" && [[ "$VPC_CONNECTOR_STATUS" != "ACTIVE" ]]; then
    log_warn "Existing VPC connector ${VPC_CONNECTOR_NAME} is ${VPC_CONNECTOR_STATUS}; creating a new one"
    VPC_CONNECTOR_NAME="${VPC_CONNECTOR_NAME}-$(date +%Y%m%d%H%M%S)"
  fi

  log_warn "Creating VPC connector ${VPC_CONNECTOR_NAME}"
  VPC_CONNECTOR_ARN="$(aws_cli apprunner create-vpc-connector \
    --vpc-connector-name "$VPC_CONNECTOR_NAME" \
    --subnets "$PRIVATE_SUBNET_ID" \
    --security-groups "$SECURITY_GROUP_ID" \
    --tags Key=Name,Value="$VPC_CONNECTOR_NAME" Key=ManagedBy,Value="$TAG_MANAGED_BY" Key=Purpose,Value="$TAG_PURPOSE" \
    --query 'VpcConnector.VpcConnectorArn' --output text)"

  for _ in {1..60}; do
    VPC_CONNECTOR_STATUS="$(aws_cli apprunner describe-vpc-connector --vpc-connector-arn "$VPC_CONNECTOR_ARN" --query 'VpcConnector.Status' --output text)"
    if [[ "$VPC_CONNECTOR_STATUS" == "ACTIVE" ]]; then
      log_info "VPC connector is ACTIVE"
      return
    fi
    if [[ "$VPC_CONNECTOR_STATUS" == "FAILED" || "$VPC_CONNECTOR_STATUS" == "INACTIVE" ]]; then
      die "VPC connector is ${VPC_CONNECTOR_STATUS}. Check App Runner events for details."
    fi
    sleep 5
  done

  die "Timed out waiting for VPC connector to become ACTIVE"
}

resolve_service_arn() {
  if [[ "$NO_UPDATE" == "true" ]]; then
    return
  fi

  if is_none "$SERVICE_ARN"; then
    if is_none "$SERVICE_NAME"; then
      die "Provide --service-arn or --service-name (or use --no-update)."
    fi
    SERVICE_ARN="$(aws_cli apprunner list-services \
      --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" --output text)"
  fi

  if is_none "$SERVICE_ARN"; then
    die "Unable to resolve App Runner service ARN. Check the name/region."
  fi
}

update_service_network() {
  if [[ "$NO_UPDATE" == "true" ]]; then
    return
  fi

  INGRESS_PUBLIC="$(aws_cli apprunner describe-service --service-arn "$SERVICE_ARN" --query 'Service.NetworkConfiguration.IngressConfiguration.IsPubliclyAccessible' --output text)"
  IP_ADDRESS_TYPE="$(aws_cli apprunner describe-service --service-arn "$SERVICE_ARN" --query 'Service.NetworkConfiguration.IpAddressType' --output text)"

  ingress_value=""
  if [[ "$INGRESS_PUBLIC" == "True" || "$INGRESS_PUBLIC" == "true" ]]; then
    ingress_value="true"
  elif [[ "$INGRESS_PUBLIC" == "False" || "$INGRESS_PUBLIC" == "false" ]]; then
    ingress_value="false"
  fi

  network_config="EgressConfiguration={EgressType=VPC,VpcConnectorArn=${VPC_CONNECTOR_ARN}}"
  if [[ -n "$ingress_value" ]]; then
    network_config+=" ,IngressConfiguration={IsPubliclyAccessible=${ingress_value}}"
  fi
  if ! is_none "$IP_ADDRESS_TYPE"; then
    network_config+=" ,IpAddressType=${IP_ADDRESS_TYPE}"
  fi

  # Remove spaces introduced above for CLI shorthand
  network_config="${network_config// /}"

  log_warn "Updating App Runner service to use VPC connector for egress"
  aws_cli apprunner update-service --service-arn "$SERVICE_ARN" --network-configuration "$network_config" >/dev/null

  log_info "Waiting for App Runner service update to complete"
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

  log_warn "Timed out waiting for service to reach RUNNING; check App Runner console"
}

main() {
  ensure_vpc
  ensure_igw
  ensure_subnets
  ensure_eip_and_nat
  ensure_route_tables
  ensure_security_group
  ensure_vpc_connector
  resolve_service_arn
  update_service_network

  echo
  echo "Egress setup complete."
  echo "Elastic IP to allowlist: ${NAT_PUBLIC_IP}"
  echo "NAT Gateway: ${NAT_ID}"
  echo "VPC Connector ARN: ${VPC_CONNECTOR_ARN}"
  echo "VPC: ${VPC_ID}"
  echo "Private subnet: ${PRIVATE_SUBNET_ID}"
  echo
  echo "Note: App Runner internal traffic (image pulls/logs/secrets) does not use your VPC."
  echo "Your application outbound traffic should now egress via the NAT gateway above."
}

main "$@"
