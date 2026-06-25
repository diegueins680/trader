# Cross-instance combo synchronization (shared S3 bus)

Goal: every running instance — **fly**, **hetzner-research**, **hetzner-trading**, and
**local** — automatically converges on the same set of best-performing combos. When any
instance discovers a new combo, it propagates to all the others without manual steps.

Storage backend: **Tigris** (S3-compatible object storage, provisioned through fly).
Plain AWS S3 also works — leave `TRADER_STATE_S3_ENDPOINT` empty and set a real region.

## How it works (no new algorithm — this already ships)

Each instance runs a background **anti-entropy reconcile loop** (`topCombosSyncLoop` in
`haskell/app/Main.hs`, enabled by default via `TRADER_TOP_COMBOS_SYNC_ENABLED`). Every
`TRADER_TOP_COMBOS_SYNC_EVERY_SEC` seconds it:

1. **Pulls** the shared leaderboard from S3 (`<prefix>/optimizer/top-combos.json`), plus
   any local file / DB / peer replicas it has.
2. **Merges** them with `mergeTopCombosPayloads` — a dedup-by-key, rank-by-performance,
   keep-top-N merge.
3. **Pushes** the merged result back to every replica that is behind, including S3
   (with a versioned copy under `<prefix>/optimizer/history/`).

This is a CRDT-style design: the combo set is a grow-only set with a top-N projection, so
the merge is **commutative, associative, and idempotent**. Consequences:

- No locking, quorum, or leader election across instances.
- Any instance may write concurrently; if two writes race, the next cycle re-merges and
  re-pushes — a lost write is at most a one-cycle delay, never data loss.
- An offline/NAT'd instance (e.g. local) just needs outbound HTTPS; it re-syncs whenever
  it comes back online. It never needs to be reachable *inbound*.

The object store (Tigris/S3) is the single durable rendezvous; both are highly available,
so there is **no primary/fallback to manage**.

> The existing Hetzner research→trading `/state/sync` HTTP push can be kept as
> Hetzner-internal redundancy, but the managed research env leaves it disabled by
> default. S3/Tigris is the primary combo bus, and avoiding the HTTP push keeps the
> trading API from spending time on long `/state/sync` requests.

## The one critical rule

`TRADER_STATE_S3_PREFIX` **must be identical on all four instances.** The combos key is
`<prefix>/optimizer/top-combos.json` and is **not** tenant-scoped, so a shared prefix =
one shared global leaderboard. (Per-bot *snapshots* — live trading state — *are*
tenant-scoped within the prefix, so fly's live state will not collide with research's.)

This repo ships with the placeholder prefix `trader-prod`. Keep it the same everywhere.

## S3-compatible endpoint support

`haskell/app/Trader/S3.hs` supports non-AWS S3-compatible stores via:

| Env var | Tigris value | Notes |
|---|---|---|
| `TRADER_STATE_S3_ENDPOINT` | `https://fly.storage.tigris.dev` | Empty = AWS (virtual-hosted). Set = path-style. |
| `TRADER_STATE_S3_REGION` | `auto` | Used for SigV4 scope. |
| `TRADER_STATE_S3_BUCKET` | *(generated)* | Bucket from `fly storage create`. |
| `TRADER_STATE_S3_PREFIX` | `trader-prod` | Identical everywhere. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | *(from Tigris)* | Credentials. |
| `TRADER_STATE_S3_FORCE_PATH_STYLE` | *(unset)* | Defaults to path-style when an endpoint is set. |
| `TRADER_TOP_COMBOS_SYNC_MAX_COMBOS` | `5000` | Anti-entropy retention cap. Defaults to `max(TRADER_OPTIMIZER_MAX_COMBOS, 5000)` so live/read-only replicas do not shrink the shared leaderboard. |

---

## Setup

### 1. Provision the Tigris bucket (from the repo root)

```sh
fly storage create
```

This creates a Tigris bucket and **sets the credentials as secrets on the fly app**
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, plus the bucket name and endpoint). It
prints the access key + secret **once** — copy them; you'll need them for Hetzner + local.

> ⚠️ Setting secrets triggers a release/restart of the fly app (the live trading bot).
> Run this at a moment a brief restart is acceptable. Non-interactive variant:
> `fly storage create --name trader-combos-prod --yes` (name must be globally unique).

Note the **bucket name** it reports (e.g. `trader-combos-prod` or a generated name).

### 2. fly

The credentials are already set as secrets by step 1. Edit `fly.toml [env]` and set the
bucket name (endpoint/region/prefix are already filled in):

```toml
TRADER_STATE_S3_BUCKET = "<bucket-name-from-step-1>"
```

Then deploy:

```sh
fly deploy
```

### 3. Hetzner (research and trading)

Each box copies `trader.<role>.env.example` → `trader.env` and fills in, in **both**
`trader.research.env` and `trader.trading.env`:

```ini
TRADER_STATE_S3_ENDPOINT=https://fly.storage.tigris.dev
TRADER_STATE_S3_REGION=auto
TRADER_STATE_S3_BUCKET=<bucket-name-from-step-1>
TRADER_STATE_S3_PREFIX=trader-prod
AWS_ACCESS_KEY_ID=<tigris-access-key-from-step-1>
AWS_SECRET_ACCESS_KEY=<tigris-secret-from-step-1>
TRADER_TOP_COMBOS_SYNC_ENABLED=true
TRADER_TOP_COMBOS_SYNC_EVERY_SEC=60
TRADER_TOP_COMBOS_SYNC_MAX_COMBOS=5000
```

`docker-compose.yml` already passes these through to the `api` service. Redeploy:

```sh
TRADER_HETZNER_ENV_FILE=deploy/hetzner/trader.env ./deploy/hetzner/deploy-remote.sh <host>
```

### 4. local

Copy `.env.example` → `.env` and set the same endpoint/region/bucket/prefix + the Tigris
creds. Keep `TRADER_STATE_S3_PREFIX=trader-prod`.

> Lost the Tigris keys? Create a fresh access key in the Tigris dashboard
> (`https://console.tigris.dev` → your bucket → Access Keys), or re-read them on fly with
> `fly ssh console -C "printenv AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY"`.

---

## Verify

After all instances restart, on each you should see in the logs:

```
Top combos sync enabled: everySec=60 maxCombos=5000 path=...
Top combos sync reconciled s3 (<N> combos).
```

Inspect the shared object directly with any S3 client pointed at Tigris, e.g.:

```sh
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  aws s3 --endpoint-url https://fly.storage.tigris.dev \
  cp s3://<bucket>/trader-prod/optimizer/top-combos.json - | jq '.combos | length'
```

End-to-end check: note the combo count, wait one optimizer cycle on any instance
(`TRADER_OPTIMIZER_EVERY_SEC`, ~15 min on fly/research), and confirm the count/top combo
appears on the *other* instances' `GET /optimizer/combos` within ~1–2 sync cycles.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `S3 ... status 403` | Wrong creds, or `TRADER_STATE_S3_BUCKET` not owned by these keys. |
| `S3 ... status 404` on PUT | Bucket name typo, or bucket not created. |
| Instances don't converge | Prefixes differ between instances — they must be byte-identical. |
| S3 never written | `TRADER_STATE_S3_BUCKET` empty on that instance (sync silently skips S3). |
| `SignatureDoesNotMatch` | `TRADER_STATE_S3_ENDPOINT` or region mismatch vs. the creds' provider. |

## Possible future hardening (not implemented)

- **Conditional PUT (`If-Match` on ETag).** Would make concurrent writes lost-update-safe
  immediately rather than relying on next-cycle self-healing. Requires adding the signed
  header to `s3PutObject` in `haskell/app/Trader/S3.hs`. Low value at current cadence.
