#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
fi

DB_URL="${TRADER_DB_URL:-${DATABASE_URL:-}}"
if [[ -z "${DB_URL}" ]]; then
  echo "TRADER_DB_URL or DATABASE_URL is required." >&2
  exit 1
fi

commit_sql=""
if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  commits_file="$(mktemp /tmp/trader_commits.XXXXXX)"
  git -C "${ROOT_DIR}" log --all --format='%H|%ct' | awk -F'|' 'NF==2 {print $1","($2*1000)}' > "${commits_file}"
  commit_sql=$(cat <<SQL
CREATE TEMP TABLE tmp_commits (commit_hash text, committed_at_ms bigint);
\\copy tmp_commits FROM '${commits_file}' CSV
INSERT INTO git_commits (commit_hash, committed_at_ms, created_at_ms)
SELECT commit_hash, committed_at_ms, :now_ms
FROM tmp_commits
ON CONFLICT (commit_hash) DO UPDATE
SET committed_at_ms = COALESCE(EXCLUDED.committed_at_ms, git_commits.committed_at_ms),
    version = COALESCE(git_commits.version, EXCLUDED.version);
DROP TABLE tmp_commits;
SQL
)
fi

psql "${DB_URL}" -v ON_ERROR_STOP=1 <<SQL
SELECT CAST(EXTRACT(EPOCH FROM NOW())*1000 AS BIGINT) AS now_ms \gset

DROP VIEW IF EXISTS performance_combo_deltas;
DROP VIEW IF EXISTS performance_commit_deltas;
DROP TABLE IF EXISTS performance_commit_summary;
DROP TABLE IF EXISTS performance_rollups;

CREATE TABLE performance_rollups (
  git_commit_id BIGINT NOT NULL REFERENCES git_commits(id),
  commit_hash TEXT,
  committed_at_ms BIGINT,
  symbol TEXT,
  market TEXT,
  interval TEXT,
  combo_uuid UUID,
  start_at_ms BIGINT,
  end_at_ms BIGINT,
  first_equity DOUBLE PRECISION,
  last_equity DOUBLE PRECISION,
  return DOUBLE PRECISION,
  max_drawdown DOUBLE PRECISION,
  status_points INTEGER,
  order_count INTEGER,
  sample_points INTEGER,
  updated_at_ms BIGINT
);

CREATE INDEX IF NOT EXISTS performance_rollups_commit_idx ON performance_rollups(git_commit_id);
CREATE INDEX IF NOT EXISTS performance_rollups_symbol_idx ON performance_rollups(symbol);
CREATE INDEX IF NOT EXISTS performance_rollups_combo_idx ON performance_rollups(combo_uuid);

$commit_sql

CREATE TEMP VIEW ops_inferred AS
SELECT
  id,
  NULLIF(lower(trim(COALESCE(
    args_json->>'platform',
    params_json->>'platform',
    result_json->>'platform',
    CASE
      WHEN lower(kind) LIKE 'binance.%' THEN 'binance'
      WHEN lower(kind) LIKE 'coinbase.%' THEN 'coinbase'
      WHEN lower(kind) LIKE 'kraken.%' THEN 'kraken'
      WHEN lower(kind) LIKE 'poloniex.%' THEN 'poloniex'
      ELSE NULL
    END
  ))), '') AS platform_code,
  NULLIF(lower(trim(COALESCE(
    args_json->>'market',
    params_json->>'market',
    result_json->>'market'
  ))), '') AS market_text,
  NULLIF(upper(trim(COALESCE(
    symbol,
    args_json->>'binanceSymbol', args_json->>'symbol',
    params_json->>'binanceSymbol', params_json->>'symbol',
    result_json->>'binanceSymbol', result_json->>'symbol'
  ))), '') AS symbol_norm
FROM ops;

UPDATE ops o
SET platform_id = p.id
FROM ops_inferred i
JOIN platforms p ON p.code = i.platform_code
WHERE o.id = i.id
  AND o.platform_id IS NULL
  AND i.platform_code IS NOT NULL;

UPDATE ops o
SET platform_id = p.id
FROM ops_inferred i
JOIN platforms p ON p.code = 'binance'
WHERE o.id = i.id
  AND o.platform_id IS NULL
  AND i.platform_code IS NULL
  AND i.symbol_norm ~ '^[A-Z0-9]{3,30}$';

WITH ops_syms_distinct AS (
  SELECT DISTINCT o.platform_id, COALESCE(i.platform_code, 'binance') AS platform_code, i.symbol_norm AS symbol, i.market_text AS market
  FROM ops o
  JOIN ops_inferred i ON i.id = o.id
  WHERE o.platform_id IS NOT NULL AND i.symbol_norm IS NOT NULL
),
expanded AS (
  SELECT
    os.platform_id,
    os.symbol,
    os.market,
    NULLIF(
      CASE
        WHEN os.platform_code = 'coinbase' AND position('-' in os.symbol) > 0 THEN split_part(os.symbol, '-', 1)
        WHEN os.platform_code = 'poloniex' AND position('_' in os.symbol) > 0 THEN split_part(os.symbol, '_', 1)
        ELSE
          CASE
            WHEN os.symbol LIKE '%USDT' THEN left(os.symbol, length(os.symbol) - 4)
            WHEN os.symbol LIKE '%USDC' THEN left(os.symbol, length(os.symbol) - 4)
            WHEN os.symbol LIKE '%FDUSD' THEN left(os.symbol, length(os.symbol) - 5)
            WHEN os.symbol LIKE '%TUSD' THEN left(os.symbol, length(os.symbol) - 4)
            WHEN os.symbol LIKE '%BUSD' THEN left(os.symbol, length(os.symbol) - 4)
            WHEN os.symbol LIKE '%BTC' THEN left(os.symbol, length(os.symbol) - 3)
            WHEN os.symbol LIKE '%ETH' THEN left(os.symbol, length(os.symbol) - 3)
            WHEN os.symbol LIKE '%BNB' THEN left(os.symbol, length(os.symbol) - 3)
            ELSE left(os.symbol, GREATEST(length(os.symbol) - 3, 0))
          END
      END, ''
    ) AS base_asset,
    NULLIF(
      CASE
        WHEN os.platform_code = 'coinbase' AND position('-' in os.symbol) > 0 THEN split_part(os.symbol, '-', 2)
        WHEN os.platform_code = 'poloniex' AND position('_' in os.symbol) > 0 THEN split_part(os.symbol, '_', 2)
        ELSE
          CASE
            WHEN os.symbol LIKE '%USDT' THEN 'USDT'
            WHEN os.symbol LIKE '%USDC' THEN 'USDC'
            WHEN os.symbol LIKE '%FDUSD' THEN 'FDUSD'
            WHEN os.symbol LIKE '%TUSD' THEN 'TUSD'
            WHEN os.symbol LIKE '%BUSD' THEN 'BUSD'
            WHEN os.symbol LIKE '%BTC' THEN 'BTC'
            WHEN os.symbol LIKE '%ETH' THEN 'ETH'
            WHEN os.symbol LIKE '%BNB' THEN 'BNB'
            ELSE right(os.symbol, 3)
          END
      END, ''
    ) AS quote_asset
  FROM ops_syms_distinct os
)
INSERT INTO platform_symbols (platform_id, symbol, market, base_asset, quote_asset, created_at_ms, updated_at_ms)
SELECT platform_id, symbol, market, base_asset, quote_asset, :now_ms, :now_ms
FROM expanded
ON CONFLICT (platform_id, symbol, market) DO UPDATE
SET base_asset = COALESCE(EXCLUDED.base_asset, platform_symbols.base_asset),
    quote_asset = COALESCE(EXCLUDED.quote_asset, platform_symbols.quote_asset),
    updated_at_ms = EXCLUDED.updated_at_ms;

WITH ops_syms_update AS (
  SELECT o.id, o.platform_id, i.symbol_norm AS symbol, i.market_text AS market
  FROM ops o
  JOIN ops_inferred i ON i.id = o.id
  WHERE o.platform_id IS NOT NULL AND i.symbol_norm IS NOT NULL
)
UPDATE ops o
SET symbol_id = ps.id
FROM ops_syms_update os
JOIN platform_symbols ps
  ON ps.platform_id = os.platform_id
  AND ps.symbol = os.symbol
  AND ((ps.market IS NULL AND os.market IS NULL) OR ps.market = os.market)
WHERE o.id = os.id
  AND o.symbol_id IS NULL;

WITH ops_syms_update AS (
  SELECT o.id, o.platform_id, i.symbol_norm AS symbol
  FROM ops o
  JOIN ops_inferred i ON i.id = o.id
  WHERE o.platform_id IS NOT NULL AND i.symbol_norm IS NOT NULL
)
UPDATE ops o
SET symbol_id = ps.id
FROM ops_syms_update os
JOIN platform_symbols ps
  ON ps.platform_id = os.platform_id
  AND ps.symbol = os.symbol
WHERE o.id = os.id
  AND o.symbol_id IS NULL
  AND ps.market = 'spot';

WITH ordered_commits AS (
  SELECT id, committed_at_ms,
         LEAD(committed_at_ms) OVER (ORDER BY committed_at_ms, id) AS next_at
  FROM git_commits WHERE committed_at_ms IS NOT NULL
), bounds AS (
  SELECT id, committed_at_ms, COALESCE(next_at, 9223372036854775807) AS next_at
  FROM ordered_commits
)
UPDATE ops
SET git_commit_id = bounds.id
FROM bounds
WHERE ops.git_commit_id IS NULL
  AND ops.at_ms >= bounds.committed_at_ms
  AND ops.at_ms < bounds.next_at;

WITH ops_filtered AS (
  SELECT
    o.git_commit_id,
    COALESCE(
      NULLIF(o.symbol, ''),
      NULLIF(upper(trim(COALESCE(
        o.args_json->>'binanceSymbol', o.args_json->>'symbol',
        o.params_json->>'binanceSymbol', o.params_json->>'symbol',
        o.result_json->>'binanceSymbol', o.result_json->>'symbol'
      ))), '')
    ) AS symbol,
    NULLIF(lower(trim(COALESCE(
      o.args_json->>'market',
      o.params_json->>'market',
      o.result_json->>'market'
    ))), '') AS market,
    NULLIF(trim(COALESCE(
      o.args_json->>'interval',
      o.params_json->>'interval',
      o.result_json->>'interval'
    )), '') AS interval,
    o.combo_uuid,
    o.at_ms,
    o.equity,
    o.kind
  FROM ops o
  WHERE o.git_commit_id IS NOT NULL
    AND o.equity IS NOT NULL
    AND o.kind IN ('bot.status', 'bot.order')
),
base AS (
  SELECT
    *,
    first_value(equity) OVER (
      PARTITION BY git_commit_id, symbol, market, interval, combo_uuid
      ORDER BY at_ms
    ) AS first_equity,
    first_value(at_ms) OVER (
      PARTITION BY git_commit_id, symbol, market, interval, combo_uuid
      ORDER BY at_ms
    ) AS first_at_ms,
    first_value(equity) OVER (
      PARTITION BY git_commit_id, symbol, market, interval, combo_uuid
      ORDER BY at_ms DESC
    ) AS last_equity,
    first_value(at_ms) OVER (
      PARTITION BY git_commit_id, symbol, market, interval, combo_uuid
      ORDER BY at_ms DESC
    ) AS last_at_ms,
    max(equity) OVER (
      PARTITION BY git_commit_id, symbol, market, interval, combo_uuid
      ORDER BY at_ms
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS peak_equity
  FROM ops_filtered
),
drawdowns AS (
  SELECT
    *,
    CASE WHEN peak_equity > 0 THEN equity / peak_equity - 1 ELSE NULL END AS drawdown
  FROM base
),
summary AS (
  SELECT
    git_commit_id,
    symbol,
    market,
    interval,
    combo_uuid,
    max(first_at_ms) AS start_at_ms,
    max(last_at_ms) AS end_at_ms,
    max(first_equity) AS first_equity,
    max(last_equity) AS last_equity,
    min(drawdown) AS max_drawdown,
    count(*) FILTER (WHERE kind = 'bot.status') AS status_points,
    count(*) FILTER (WHERE kind = 'bot.order') AS order_count,
    count(*) AS sample_points
  FROM drawdowns
  GROUP BY git_commit_id, symbol, market, interval, combo_uuid
)
INSERT INTO performance_rollups (
  git_commit_id,
  commit_hash,
  committed_at_ms,
  symbol,
  market,
  interval,
  combo_uuid,
  start_at_ms,
  end_at_ms,
  first_equity,
  last_equity,
  return,
  max_drawdown,
  status_points,
  order_count,
  sample_points,
  updated_at_ms
)
SELECT
  s.git_commit_id,
  g.commit_hash,
  g.committed_at_ms,
  s.symbol,
  s.market,
  s.interval,
  s.combo_uuid,
  s.start_at_ms,
  s.end_at_ms,
  s.first_equity,
  s.last_equity,
  CASE WHEN s.first_equity > 0 THEN s.last_equity / s.first_equity - 1 ELSE NULL END AS return,
  s.max_drawdown,
  s.status_points,
  s.order_count,
  s.sample_points,
  :now_ms AS updated_at_ms
FROM summary s
LEFT JOIN git_commits g ON g.id = s.git_commit_id;

CREATE TABLE performance_commit_summary (
  git_commit_id BIGINT NOT NULL REFERENCES git_commits(id),
  commit_hash TEXT,
  committed_at_ms BIGINT,
  start_at_ms BIGINT,
  end_at_ms BIGINT,
  symbols INTEGER,
  combos INTEGER,
  rollups INTEGER,
  avg_return DOUBLE PRECISION,
  median_return DOUBLE PRECISION,
  min_return DOUBLE PRECISION,
  max_return DOUBLE PRECISION,
  avg_drawdown DOUBLE PRECISION,
  median_drawdown DOUBLE PRECISION,
  worst_drawdown DOUBLE PRECISION,
  status_points INTEGER,
  order_count INTEGER,
  sample_points INTEGER,
  updated_at_ms BIGINT
);

CREATE INDEX IF NOT EXISTS performance_commit_summary_commit_idx ON performance_commit_summary(git_commit_id);
CREATE INDEX IF NOT EXISTS performance_commit_summary_committed_idx ON performance_commit_summary(committed_at_ms);

INSERT INTO performance_commit_summary (
  git_commit_id,
  commit_hash,
  committed_at_ms,
  start_at_ms,
  end_at_ms,
  symbols,
  combos,
  rollups,
  avg_return,
  median_return,
  min_return,
  max_return,
  avg_drawdown,
  median_drawdown,
  worst_drawdown,
  status_points,
  order_count,
  sample_points,
  updated_at_ms
)
SELECT
  pr.git_commit_id,
  MAX(pr.commit_hash) AS commit_hash,
  MAX(pr.committed_at_ms) AS committed_at_ms,
  MIN(pr.start_at_ms) AS start_at_ms,
  MAX(pr.end_at_ms) AS end_at_ms,
  COUNT(DISTINCT pr.symbol) AS symbols,
  COUNT(DISTINCT pr.combo_uuid) AS combos,
  COUNT(*) AS rollups,
  AVG(pr.return) FILTER (WHERE pr.return IS NOT NULL) AS avg_return,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pr.return) FILTER (WHERE pr.return IS NOT NULL) AS median_return,
  MIN(pr.return) FILTER (WHERE pr.return IS NOT NULL) AS min_return,
  MAX(pr.return) FILTER (WHERE pr.return IS NOT NULL) AS max_return,
  AVG(pr.max_drawdown) FILTER (WHERE pr.max_drawdown IS NOT NULL) AS avg_drawdown,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pr.max_drawdown) FILTER (WHERE pr.max_drawdown IS NOT NULL) AS median_drawdown,
  MIN(pr.max_drawdown) FILTER (WHERE pr.max_drawdown IS NOT NULL) AS worst_drawdown,
  SUM(pr.status_points) AS status_points,
  SUM(pr.order_count) AS order_count,
  SUM(pr.sample_points) AS sample_points,
  :now_ms AS updated_at_ms
FROM performance_rollups pr
GROUP BY pr.git_commit_id;

CREATE VIEW performance_commit_deltas AS
SELECT
  s.*,
  LAG(s.commit_hash) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id) AS prev_commit_hash,
  LAG(s.median_return) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id) AS prev_median_return,
  (s.median_return - LAG(s.median_return) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id)) AS delta_median_return,
  LAG(s.median_drawdown) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id) AS prev_median_drawdown,
  (s.median_drawdown - LAG(s.median_drawdown) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id)) AS delta_median_drawdown,
  LAG(s.worst_drawdown) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id) AS prev_worst_drawdown,
  (s.worst_drawdown - LAG(s.worst_drawdown) OVER (ORDER BY s.committed_at_ms NULLS LAST, s.git_commit_id)) AS delta_worst_drawdown
FROM performance_commit_summary s;

CREATE VIEW performance_combo_deltas AS
SELECT
  pr.*,
  LAG(pr.commit_hash) OVER (
    PARTITION BY pr.symbol, pr.market, pr.interval, pr.combo_uuid
    ORDER BY pr.committed_at_ms NULLS LAST, pr.git_commit_id
  ) AS prev_commit_hash,
  LAG(pr.return) OVER (
    PARTITION BY pr.symbol, pr.market, pr.interval, pr.combo_uuid
    ORDER BY pr.committed_at_ms NULLS LAST, pr.git_commit_id
  ) AS prev_return,
  (pr.return - LAG(pr.return) OVER (
    PARTITION BY pr.symbol, pr.market, pr.interval, pr.combo_uuid
    ORDER BY pr.committed_at_ms NULLS LAST, pr.git_commit_id
  )) AS delta_return,
  LAG(pr.max_drawdown) OVER (
    PARTITION BY pr.symbol, pr.market, pr.interval, pr.combo_uuid
    ORDER BY pr.committed_at_ms NULLS LAST, pr.git_commit_id
  ) AS prev_max_drawdown,
  (pr.max_drawdown - LAG(pr.max_drawdown) OVER (
    PARTITION BY pr.symbol, pr.market, pr.interval, pr.combo_uuid
    ORDER BY pr.committed_at_ms NULLS LAST, pr.git_commit_id
  )) AS delta_drawdown
FROM performance_rollups pr;
SQL

echo "performance_rollups updated."
