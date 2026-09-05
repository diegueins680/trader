# Binance derivatives first-seen schema v2 — infrastructure result

Date: 2026-09-04.

## Result

The public research collector now writes a separate raw ledger for each
`(symbol, interval, feature)` under `data/research/.observations/`. Each row is
bound to `binance_derivatives_first_seen_v2` and records the symbol, interval,
feature, event time, collector availability time, observed flag, and finite
dense value. Funding,
open interest, basis, and taker-flow history returned by Binance has an event
timestamp but no independent revision/publication timestamp, so a newly
retrieved value becomes available only when that fetch completes.

Unchanged refetches retain their earliest first-seen time. A changed value for
the same event becomes a later revision, including a change back to a prior
value. A revision cannot be inserted earlier than an already recorded release;
conflicting values at the same event and availability time fail closed. Ledger
scope, exact columns, finite values, integer timestamps, and
`eventTime <= availabilityTime` are validated before every merge. A missing
expected grid observation is retained as an explicit `observed=0`, `value=0`
tombstone so it clears older state without becoming a numeric signal. Each ledger
replacement is atomic and runs under the existing shared cache-writer lock.

The bar cache keeps its historical `funding`, `oi`, `basis`, and `taker`
columns unchanged for compatibility. It additionally emits five v2 columns per
family: `V2Value`, `V2Observed`, `V2Fresh`, `V2EventTime`, and
`V2AvailabilityTime`. Alignment admits only releases available by the bar
close. Funding uses the existing nine-hour maximum age; the other three
families use the existing two-bar maximum age. An observed-but-stale value
retains its timestamp witnesses but has `V2Fresh=0` and dense `V2Value=0`.
Pre-coverage rows have both masks at zero and no fabricated timestamps, so an
observed zero remains distinguishable from unavailable evidence.

This change advanced the scheduled collector status schema to version 2. A
later additive artifact-provenance change advances it to version 3; schema 2
remains historical operational evidence but is not a cryptographically bound
artifact manifest. A fresh run is degraded unless every derivatives ledger and
freshly written v2 tail passes schema, mask, finiteness, timestamp-causality,
and stale-neutrality checks.

## Compatibility and migration boundary

Existing legacy cache cells are not assigned inferred availability times and
are not relabeled as v2 evidence. On the first upgraded refresh, historical
endpoint rows are first seen at that refresh time; they cannot affect an
earlier bar. This deliberately sacrifices retrospective coverage rather than
manufacturing causal history. The raw ledgers and added bar columns remain
outside Git and may be regenerated only by future public collection.

No candidate was fit, no registered date moved, and no development or final
holdout outcome was inspected. The v2 columns are not connected to a Haskell
predictor, saved artifact, champion, combo, paper bot, live bot, or order path.
`FEATURE-MISSINGNESS-001` remains open until a separately versioned candidate
builder consumes these masks, remaining source families have equally explicit
availability, artifact compatibility is enforced, and prospective evidence
passes the frozen gates.

## Verification

`test/research-datafeed.test.mjs` covers unchanged refetches, revisions and
value reversions, observed-zero separation, staleness, future perturbation
isolation, malformed grids, non-zero unavailable values, raw-ledger persistence,
and first-refresh non-backdating. `test/research-datafeed-scheduler.test.mjs`
covers the versioned collector status boundary while retaining lock,
interruption, timeout, and secret-free LaunchAgent behavior.
