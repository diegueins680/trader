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
#   TRADER_HETZNER_ENV_OVERRIDES_FILE Optional local KEY=VALUE overlay copied over SSH
#                                and merged into the remote env file before compose up.
#                                Use for CI-supplied secrets; the file is not rsynced.
#   TRADER_HETZNER_COMPOSE_FILE  Compose file relative to repo dir (default: deploy/hetzner/docker-compose.yml)
#   TRADER_HETZNER_SSH_KEY_FILE  SSH identity file (optional; for CI)
#   TRADER_HETZNER_KNOWN_HOSTS   pinned known_hosts file (optional only when the
#                                host key is already pinned in ~/.ssh/known_hosts)
#   TRADER_HETZNER_SSH_CONNECT_TIMEOUT  SSH ConnectTimeout seconds (default: 10)
#   TRADER_HETZNER_SSH_CONNECTION_ATTEMPTS  SSH ConnectionAttempts count (default: 3)
#   TRADER_HETZNER_HEALTH_TIMEOUT  Seconds to wait for API health (default: 180)
#   TRADER_HETZNER_ROLLBACK_IMAGE  Local Docker tag retaining the previous API
#                                release (default: trader-api:rollback)
#   TRADER_HETZNER_SSH_EXTRA_OPTS Extra raw ssh options (optional)

usage() {
  sed -n '4,39p' "$0" | sed 's/^# \{0,1\}//'
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
env_overrides_file="${TRADER_HETZNER_ENV_OVERRIDES_FILE:-}"
compose_file="${TRADER_HETZNER_COMPOSE_FILE:-deploy/hetzner/docker-compose.yml}"
ssh_connect_timeout="${TRADER_HETZNER_SSH_CONNECT_TIMEOUT:-10}"
ssh_connection_attempts="${TRADER_HETZNER_SSH_CONNECTION_ATTEMPTS:-3}"
health_timeout="${TRADER_HETZNER_HEALTH_TIMEOUT:-180}"
rollback_image="${TRADER_HETZNER_ROLLBACK_IMAGE:-trader-api:rollback}"

if [[ ! "$commit" =~ ^[0-9A-Fa-f]{7,64}$ ]]; then
  echo "ERROR: TRADER_GIT_COMMIT must be a 7-64 character hexadecimal commit, got: $commit" >&2
  exit 1
fi
if [[ ! "$health_timeout" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TRADER_HETZNER_HEALTH_TIMEOUT must be a positive integer, got: $health_timeout" >&2
  exit 1
fi
if [[ ! "$rollback_image" =~ ^[A-Za-z0-9][A-Za-z0-9._/:@-]*$ ]]; then
  echo "ERROR: invalid TRADER_HETZNER_ROLLBACK_IMAGE: $rollback_image" >&2
  exit 1
fi

# Keepalives: the remote `docker compose ... --build` step compiles the full
# Haskell tree and can sit minutes without output; without these the session
# dies with "Broken pipe" (observed on the research box, 2026-06-11).
ssh_opts=(
  -p "$ssh_port"
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o "ConnectTimeout=${ssh_connect_timeout}"
  -o "ConnectionAttempts=${ssh_connection_attempts}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)
if [[ -n "${TRADER_HETZNER_SSH_KEY_FILE:-}" ]]; then
  ssh_opts+=(-i "$TRADER_HETZNER_SSH_KEY_FILE" -o IdentitiesOnly=yes)
fi
if [[ -n "${TRADER_HETZNER_KNOWN_HOSTS:-}" ]]; then
  if [[ ! -s "$TRADER_HETZNER_KNOWN_HOSTS" ]]; then
    echo "ERROR: TRADER_HETZNER_KNOWN_HOSTS is empty or missing: $TRADER_HETZNER_KNOWN_HOSTS" >&2
    exit 1
  fi
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

if [[ -n "$env_overrides_file" && ! -f "$env_overrides_file" ]]; then
  echo "ERROR: TRADER_HETZNER_ENV_OVERRIDES_FILE does not exist: $env_overrides_file" >&2
  exit 1
fi

remote_env_overrides_file=""
cleanup_remote_env_overrides() {
  if [[ -n "$remote_env_overrides_file" ]]; then
    ssh "${ssh_opts[@]}" "${ssh_user}@${host}" "rm -f '$remote_env_overrides_file'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_env_overrides EXIT

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
  --exclude '.cabal/' \
  --exclude 'haskell/.cabal/' \
  --exclude 'haskell/.stack-root/' \
  --exclude 'haskell/.stack-work/' \
  --exclude '.venv/' \
  --exclude '.tmp/' \
  --exclude 'haskell/.tmp/' \
  --include '*.env.example' \
  --include '*.env.managed' \
  --exclude '.env' \
  --exclude '*.env' \
  --exclude 'tmp/' \
  --exclude 'data/' \
  --exclude 'artifacts/' \
  --exclude 'reports/' \
  -e "ssh ${ssh_opts[*]}" \
  "${repo_root}/" "${ssh_user}@${host}:${repo_dir}/"

# The general overlay deliberately never deletes server files, but hashed web
# bundles are disposable. Prune this directory so an old cached HTML document
# cannot keep loading a historical UI bundle after the API has been replaced.
rsync -az --delete --human-readable \
  -e "ssh ${ssh_opts[*]}" \
  "${repo_root}/haskell/web/dist/" "${ssh_user}@${host}:${repo_dir}/haskell/web/dist/"

if [[ -n "$env_overrides_file" ]]; then
  echo "==> Uploading env overrides to ${host}"
  remote_env_overrides_file="$(ssh "${ssh_opts[@]}" "${ssh_user}@${host}" "mktemp")"
  ssh "${ssh_opts[@]}" "${ssh_user}@${host}" "cat > '$remote_env_overrides_file'" < "$env_overrides_file"
fi

echo "==> Building and starting containers on ${host}"
# Send one explicit remote command. Passing the assignments as separate ssh
# arguments made a partial rsync appear successful in CI while the remote
# compose/attestation phase produced no output and did not replace the API.
quote_remote_arg() {
  printf '%q' "$1"
}

remote_command="env"
for assignment in \
  "REPO_DIR=${repo_dir}" \
  "ENV_FILE=${env_file}" \
  "MANAGED_ENV_FILE=${managed_env_file}" \
  "ENV_OVERRIDES_FILE=${remote_env_overrides_file}" \
  "COMPOSE_FILE=${compose_file}" \
  "TRADER_GIT_COMMIT=${commit}" \
  "DEPLOY_HEALTH_TIMEOUT_SEC=${health_timeout}" \
  "ROLLBACK_IMAGE=${rollback_image}"; do
  remote_command+=" $(quote_remote_arg "$assignment")"
done
remote_command+=" bash -s"

ssh "${ssh_opts[@]}" "${ssh_user}@${host}" "$remote_command" <<'REMOTE'
set -Eeuo pipefail
echo "==> Remote deployment started for ${TRADER_GIT_COMMIT}"
cd "$REPO_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found at ${REPO_DIR}/${ENV_FILE}" >&2
  exit 1
fi

merge_env_overlay() {
  local overlay_file="$1"
  local target_file="$2"
  local label="$3"
  local merged
  [[ -n "$overlay_file" && -f "$overlay_file" ]] || return 0

  merged="$(mktemp)"
  awk -F= '
    NR == FNR { if ($0 ~ /^[A-Za-z_][A-Za-z0-9_]*=/) managed[$1] = $0; next }
    /^[A-Za-z_][A-Za-z0-9_]*=/ && ($1 in managed) { print managed[$1]; delete managed[$1]; next }
    { print }
    END { for (k in managed) print managed[k] }
  ' "$overlay_file" "$target_file" > "$merged"
  if ! cmp -s "$merged" "$target_file"; then
    echo "==> Applying ${label} -> ${target_file}"
    cat "$merged" > "$target_file"
  fi
  rm -f "$merged"
}

# Merge the checked-in managed overlay into the box's env file: update each
# overlay key in place, append keys the box doesn't have yet, leave everything
# else (secrets, operator overrides of unmanaged keys) untouched.
merge_env_overlay "$MANAGED_ENV_FILE" "$ENV_FILE" "managed env overlay ${MANAGED_ENV_FILE}"

# Merge optional CI-supplied overrides after the managed overlay. This is for
# secrets such as shared Binance credentials and is uploaded over SSH stdin, not
# rsynced from the working tree.
merge_env_overlay "$ENV_OVERRIDES_FILE" "$ENV_FILE" "runtime env overrides"
rm -f "${ENV_OVERRIDES_FILE:-}"

export TRADER_GIT_COMMIT
compose=(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE")

api_health_json() {
  # This remote program is itself streamed to `bash -s` over SSH. Compose exec
  # must not inherit that stream or it consumes the remaining deployment steps
  # after the first health probe and exits without rebuilding or attesting.
  "${compose[@]}" exec -T api curl -fsS --max-time 5 http://127.0.0.1:8080/health </dev/null
}

health_commit() {
  sed -n 's/.*"commit"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

wait_for_api_health() {
  local deadline state container_id
  deadline=$((SECONDS + DEPLOY_HEALTH_TIMEOUT_SEC))
  while ((SECONDS < deadline)); do
    container_id="$("${compose[@]}" ps -q api 2>/dev/null || true)"
    if [[ -n "$container_id" ]]; then
      state="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)"
      if [[ "$state" == "healthy" ]]; then
        return 0
      fi
      if [[ "$state" == "exited" || "$state" == "dead" ]]; then
        echo "ERROR: API container entered state ${state}." >&2
        return 1
      fi
    fi
    sleep 5
  done
  echo "ERROR: API did not become healthy within ${DEPLOY_HEALTH_TIMEOUT_SEC}s." >&2
  "${compose[@]}" ps >&2 || true
  "${compose[@]}" logs --tail 100 api >&2 || true
  return 1
}

previous_container="$("${compose[@]}" ps -q api 2>/dev/null || true)"
previous_image_id=""
previous_commit=""
if [[ -n "$previous_container" ]]; then
  previous_image_id="$(docker inspect --format '{{.Image}}' "$previous_container" 2>/dev/null || true)"
  previous_commit="$(api_health_json 2>/dev/null | health_commit || true)"
fi
if [[ -n "$previous_image_id" ]]; then
  docker image tag "$previous_image_id" "$ROLLBACK_IMAGE"
  echo "==> Retained previous API image as ${ROLLBACK_IMAGE}${previous_commit:+ (${previous_commit})}"
fi

rollback_deployment() {
  local failed_status="$1" rollback_health rollback_commit
  trap - ERR
  set +e
  echo "ERROR: deployment of ${TRADER_GIT_COMMIT} failed." >&2
  "${compose[@]}" ps >&2
  "${compose[@]}" logs --tail 100 api >&2
  if [[ -z "$previous_image_id" ]]; then
    echo "ERROR: no previously running API image is available for automatic rollback." >&2
    exit "$failed_status"
  fi

  echo "==> Rolling API back to ${ROLLBACK_IMAGE}${previous_commit:+ (${previous_commit})}"
  TRADER_API_IMAGE="$ROLLBACK_IMAGE" "${compose[@]}" up -d --no-build --remove-orphans
  if wait_for_api_health; then
    rollback_health="$(api_health_json 2>/dev/null)"
    rollback_commit="$(printf '%s' "$rollback_health" | health_commit)"
    if [[ -n "$previous_commit" && "$rollback_commit" != "$previous_commit" ]]; then
      echo "ERROR: rollback health reported commit '${rollback_commit:-missing}', expected '${previous_commit}'." >&2
    else
      echo "==> Rollback healthy${rollback_commit:+ and attested at ${rollback_commit}}"
    fi
  else
    echo "ERROR: rollback image did not become healthy." >&2
  fi
  exit "$failed_status"
}

trap 'rollback_deployment $?' ERR

# Build and recreate the API explicitly. `up -d --build` is allowed to retain
# an already-running container when Compose decides its configuration is
# unchanged; that would update the static UI but leave the API's commit stale.
"${compose[@]}" build api
"${compose[@]}" up -d --no-deps --force-recreate api
"${compose[@]}" up -d --remove-orphans
wait_for_api_health

health_json="$(api_health_json)"
reported_commit="$(printf '%s' "$health_json" | health_commit)"
if [[ "$reported_commit" != "$TRADER_GIT_COMMIT" ]]; then
  echo "ERROR: /health reported commit '${reported_commit:-missing}', expected '${TRADER_GIT_COMMIT}'." >&2
  false
fi

trap - ERR
docker image prune -f >/dev/null 2>&1 || true
echo "==> Deploy healthy and commit-attested ($TRADER_GIT_COMMIT)"
REMOTE
