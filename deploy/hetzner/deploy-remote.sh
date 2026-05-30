#!/usr/bin/env bash
set -euo pipefail

# Deploy the trader stack to a Hetzner host over SSH.
#
# The host must already have:
#   * this repository cloned (origin -> this repo) at TRADER_HETZNER_REPO_DIR
#   * a populated, secret-filled env file at TRADER_HETZNER_ENV_FILE
#   * docker + the compose plugin installed
#
# This script updates the checkout to the requested commit, refreshes the build
# commit marker, and runs `docker compose ... up -d --build` on the host.
#
# Usage:
#   deploy/hetzner/deploy-remote.sh <host>
#
# Environment overrides:
#   TRADER_GIT_COMMIT            Commit to deploy (default: local `git rev-parse HEAD`)
#   TRADER_HETZNER_SSH_USER      SSH user (default: root)
#   TRADER_HETZNER_SSH_PORT      SSH port (default: 22)
#   TRADER_HETZNER_REPO_DIR      Repo path on the host (default: /opt/trader)
#   TRADER_HETZNER_ENV_FILE      Env file relative to repo dir (default: deploy/hetzner/trader.env)
#   TRADER_HETZNER_COMPOSE_FILE  Compose file relative to repo dir (default: deploy/hetzner/docker-compose.yml)
#   TRADER_HETZNER_GIT_REMOTE    Remote to fetch from (default: origin)
#   TRADER_HETZNER_SSH_KEY_FILE  SSH identity file (optional; for CI)
#   TRADER_HETZNER_KNOWN_HOSTS   known_hosts file (optional; for CI)
#   TRADER_HETZNER_SSH_EXTRA_OPTS Extra raw ssh options (optional)

usage() {
  sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'
}

host="${1:-}"
if [[ -z "$host" || "$host" == "-h" || "$host" == "--help" ]]; then
  usage
  [[ -n "$host" ]] && exit 0 || exit 1
fi

commit="${TRADER_GIT_COMMIT:-$(git rev-parse HEAD)}"
ssh_user="${TRADER_HETZNER_SSH_USER:-root}"
ssh_port="${TRADER_HETZNER_SSH_PORT:-22}"
repo_dir="${TRADER_HETZNER_REPO_DIR:-/opt/trader}"
env_file="${TRADER_HETZNER_ENV_FILE:-deploy/hetzner/trader.env}"
compose_file="${TRADER_HETZNER_COMPOSE_FILE:-deploy/hetzner/docker-compose.yml}"
git_remote="${TRADER_HETZNER_GIT_REMOTE:-origin}"

ssh_opts=(-p "$ssh_port" -o BatchMode=yes)
if [[ -n "${TRADER_HETZNER_SSH_KEY_FILE:-}" ]]; then
  ssh_opts+=(-i "$TRADER_HETZNER_SSH_KEY_FILE" -o IdentitiesOnly=yes)
fi
if [[ -n "${TRADER_HETZNER_KNOWN_HOSTS:-}" ]]; then
  ssh_opts+=(-o "UserKnownHostsFile=$TRADER_HETZNER_KNOWN_HOSTS" -o StrictHostKeyChecking=yes)
fi
if [[ -n "${TRADER_HETZNER_SSH_EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  ssh_opts+=($TRADER_HETZNER_SSH_EXTRA_OPTS)
fi

echo "Deploying commit ${commit} to ${ssh_user}@${host}:${repo_dir}"

# Env assignments are single-quoted so the remote login shell applies them to the
# `bash -s` process that consumes the heredoc on stdin.
ssh "${ssh_opts[@]}" "${ssh_user}@${host}" \
  "REPO_DIR='${repo_dir}' ENV_FILE='${env_file}' COMPOSE_FILE='${compose_file}'" \
  "GIT_REMOTE='${git_remote}' TRADER_GIT_COMMIT='${commit}' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REPO_DIR"

echo "==> Fetching ${GIT_REMOTE}"
git fetch --prune "$GIT_REMOTE"

echo "==> Checking out ${TRADER_GIT_COMMIT}"
git -c advice.detachedHead=false checkout --force "$TRADER_GIT_COMMIT"

printf '%s\n' "$TRADER_GIT_COMMIT" > haskell/.build-commit

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found at ${REPO_DIR}/${ENV_FILE}" >&2
  exit 1
fi

echo "==> Building and starting containers"
export TRADER_GIT_COMMIT
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build --remove-orphans

docker image prune -f >/dev/null 2>&1 || true
echo "==> Deploy complete ($TRADER_GIT_COMMIT)"
REMOTE
