#!/usr/bin/env bash
set -euo pipefail

# Deploy the trader stack to a Hetzner host over SSH.
#
# The boxes run the stack from a plain (non-git) copy of this repo at
# TRADER_HETZNER_REPO_DIR, with the built frontend in haskell/web/dist and a
# secret-filled env file at TRADER_HETZNER_ENV_FILE. This script rsyncs the
# local working tree to the host (overlay, never deleting the box's secrets or
# runtime data) and runs `docker compose ... up -d --build`.
#
# Build haskell/web/dist locally (or in CI) BEFORE running this; it is gitignored
# and Caddy serves it from the box.
#
# The host must have: docker + the compose plugin, rsync, the deploy SSH key in
# authorized_keys, and a populated TRADER_HETZNER_ENV_FILE.
#
# Usage:
#   deploy/hetzner/deploy-remote.sh <host>
#
# Environment overrides:
#   TRADER_GIT_COMMIT            Commit being deployed (default: local `git rev-parse HEAD`)
#   TRADER_HETZNER_SSH_USER      SSH user (default: root)
#   TRADER_HETZNER_SSH_PORT      SSH port (default: 22)
#   TRADER_HETZNER_REPO_DIR      Repo path on the host (default: /opt/trader)
#   TRADER_HETZNER_ENV_FILE      Env file relative to repo dir (default: deploy/hetzner/trader.env)
#   TRADER_HETZNER_MANAGED_ENV_FILE  Optional checked-in KEY=VALUE overlay relative to the
#                                repo dir; each key is updated-or-appended into the box's
#                                env file before compose up, so non-secret tuning ships
#                                from the repo instead of by hand. Missing file = no-op.
#   TRADER_HETZNER_COMPOSE_FILE  Compose file relative to repo dir (default: deploy/hetzner/docker-compose.yml)
#   TRADER_HETZNER_SSH_KEY_FILE  SSH identity file (optional; for CI)
#   TRADER_HETZNER_KNOWN_HOSTS   known_hosts file (optional; for CI)
#   TRADER_HETZNER_SSH_EXTRA_OPTS Extra raw ssh options (optional)

usage() {
  sed -n '4,32p' "$0" | sed 's/^# \{0,1\}//'
}

host="${1:-}"
if [[ -z "$host" || "$host" == "-h" || "$host" == "--help" ]]; then
  usage
  [[ -n "$host" ]] && exit 0 || exit 1
fi

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

commit="${TRADER_GIT_COMMIT:-$(git -C "$repo_root" rev-parse HEAD)}"
ssh_user="${TRADER_HETZNER_SSH_USER:-root}"
ssh_port="${TRADER_HETZNER_SSH_PORT:-22}"
repo_dir="${TRADER_HETZNER_REPO_DIR:-/opt/trader}"
env_file="${TRADER_HETZNER_ENV_FILE:-deploy/hetzner/trader.env}"
managed_env_file="${TRADER_HETZNER_MANAGED_ENV_FILE:-}"
compose_file="${TRADER_HETZNER_COMPOSE_FILE:-deploy/hetzner/docker-compose.yml}"

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

if [[ ! -d "${repo_root}/haskell/web/dist" ]]; then
  echo "ERROR: ${repo_root}/haskell/web/dist is missing. Build the frontend first" >&2
  echo "       (npm --prefix haskell/web ci && npm --prefix haskell/web run build)." >&2
  exit 1
fi

# Stamp the build commit so the running API reports the deployed SHA.
printf '%s\n' "$commit" > "${repo_root}/haskell/.build-commit"

echo "==> Syncing ${commit} to ${ssh_user}@${host}:${repo_dir}"

# Overlay sync (NO --delete): updates code + built web/dist, but never removes
# the box's secrets (trader.env), Caddyfile, or runtime data. Excluded paths are
# protected from deletion regardless.
rsync -az --human-readable \
  --exclude '.git/' \
  --exclude 'node_modules/' \
  --exclude '**/node_modules/' \
  --exclude 'dist-newstyle*/' \
  --exclude 'haskell/dist-newstyle*/' \
  --exclude '.env' \
  --exclude '*.env' \
  --exclude 'tmp/' \
  --exclude 'data/' \
  --exclude 'artifacts/' \
  --exclude 'reports/' \
  -e "ssh ${ssh_opts[*]}" \
  "${repo_root}/" "${ssh_user}@${host}:${repo_dir}/"

echo "==> Building and starting containers on ${host}"
ssh "${ssh_opts[@]}" "${ssh_user}@${host}" \
  "REPO_DIR='${repo_dir}' ENV_FILE='${env_file}' MANAGED_ENV_FILE='${managed_env_file}' COMPOSE_FILE='${compose_file}'" \
  "TRADER_GIT_COMMIT='${commit}' bash -s" <<'REMOTE'
set -euo pipefail
cd "$REPO_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found at ${REPO_DIR}/${ENV_FILE}" >&2
  exit 1
fi

# Merge the checked-in managed overlay into the box's env file: update each
# overlay key in place, append keys the box doesn't have yet, leave everything
# else (secrets, operator overrides of unmanaged keys) untouched.
if [[ -n "$MANAGED_ENV_FILE" && -f "$MANAGED_ENV_FILE" ]]; then
  merged="$(mktemp)"
  awk -F= '
    NR == FNR { if ($0 ~ /^[A-Za-z_][A-Za-z0-9_]*=/) managed[$1] = $0; next }
    /^[A-Za-z_][A-Za-z0-9_]*=/ && ($1 in managed) { print managed[$1]; delete managed[$1]; next }
    { print }
    END { for (k in managed) print managed[k] }
  ' "$MANAGED_ENV_FILE" "$ENV_FILE" > "$merged"
  if ! cmp -s "$merged" "$ENV_FILE"; then
    echo "==> Applying managed env overlay ${MANAGED_ENV_FILE} -> ${ENV_FILE}"
    cat "$merged" > "$ENV_FILE"
  fi
  rm -f "$merged"
fi

export TRADER_GIT_COMMIT
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build --remove-orphans
docker image prune -f >/dev/null 2>&1 || true
echo "==> Deploy complete ($TRADER_GIT_COMMIT)"
REMOTE
