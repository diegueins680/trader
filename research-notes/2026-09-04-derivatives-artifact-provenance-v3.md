# Binance derivatives collection artifacts v3 — infrastructure result

Date: 2026-09-04.

## Result

The scheduled public collector now writes `last-run.json` with status schema 3
and artifact identity `binance_derivatives_collection_artifacts_v3`. Each
successful symbol result records the exact ordered bar-cache columns and binds
the bar CSV plus funding, open-interest, basis, and taker-flow first-seen
ledgers by absolute path, data-row count, and SHA-256. The status also records
the Git code commit, a witness that both collector files and the source-license
manifest match that commit, Python/numpy/pandas runtime versions,
`feature_availability_v2`,
`binance_derivatives_first_seen_v2`, and the canonical committed data-source
and license manifest. Status writes remain atomic and occur while the collector
owns the existing cache lock.

`collect_datafeed.py verify-artifacts` is a read-only admission check. It
requires a complete-pass status with no failed symbols, rejects duplicate JSON
keys and malformed scopes, verifies every recorded digest, checks exact raw CSV
headers, reloads and validates each ledger, recomputes row counts, time order,
and v2 coverage, requires every fresh-tail row to be versioned, and causally
realigns every ledger to the cache grid. Verification shares an existing
collector lock without writing to it and fails when a writer is active. Every
present v2 value, observed/fresh
mask, event time, and availability time must equal that reconstruction.
Changing a cache byte fails its digest; changing the digest as well still fails
when the cell disagrees with the bound ledger.

A byte-identical relocated cache can be verified with `--cache-dir`; the
recorded logical paths and all hashes remain fixed. The verifier returns the
SHA-256 of the status file itself. Because `last-run.json` is replaced on each
scheduled collection, a decisive experiment must freeze those bytes or record
that returned digest in its preregistered receipt before fitting. Schema-2
statuses remain valid historical operational evidence but cannot pass artifact
admission because they predate file binding.

## Compatibility and promotion boundary

No downloaded market data or model artifact is committed. No return, forecast,
position, PnL, risk statistic, development outcome, or final-holdout outcome is
calculated. The verifier is not imported by Haskell feature construction,
model loading, predictors, bots, champions, combos, or execution. It changes no
saved configuration, model identifier, environment authorization, deployment,
paper mode, live mode, or order behavior. It supplies only the missing frozen
input-provenance boundary for later separately registered offline research.

`FEATURE-MISSINGNESS-001` remains open. A future candidate still needs a
versioned feature/artifact schema, prospective evidence, all cost/statistical
and risk gates, disabled-by-default challenger isolation, and explicit rollback
before it can advance beyond offline research.

## Verification coverage

The scheduler regression covers atomic schema-3 emission, hashes for all five
files per symbol, successful direct and CLI verification, byte-identical cache
relocation, active-writer rejection, hash mismatch, cache rehash with failed
ledger reconstruction, non-pass status rejection, and duplicate manifest keys. The datafeed regression
also rejects non-canonical v2 column order. Existing lock, partial-failure,
timeout, interruption, and secret-free LaunchAgent checks remain in force.
