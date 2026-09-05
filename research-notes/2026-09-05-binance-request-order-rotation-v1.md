# Binance public-data request-order rotation v1

Date: 2026-09-05

## Demonstrated gap

The run-level provider-rate-limit circuit prevents wasteful requests after a
known throttle, but it originally traversed the canonical universe in the same
order every hour. The first installed circuit observation completed BTC,
failed on ETH, and skipped the remaining eight symbols. A later clean
merged-main run from `2026-09-05T14:10:06.680961Z` through
`2026-09-05T14:10:48.777169Z` completed BTC, ETH, and SOL, failed on BNB with
HTTP 429, and skipped the remaining six symbols. That mutable schema-3 status
was 17,245 bytes with SHA-256
`adcd25d5d6666ce759c1900973b77d19859ccdd714685fbf4a54916d869a521d`
and recorded clean commit
`81b51a869f3d6659c4bf07cb84b591a18599cc43` with no provenance issues.

These observations do not establish a stable provider capacity or prove that
this collector caused the shared-IP throttle. They do demonstrate that a
fixed prefix receives every available attempt when partial runs recur, while
tail symbols can be systematically starved.

## Implemented interpretation

The canonical configured `symbols` list remains unchanged. For each run, the
collector derives a separate request permutation:

`offset = floor(UTC start epoch seconds / 3600) mod symbol count`

`requestOrder = symbols[offset:] + symbols[:offset]`

The policy identifier is `utc_epoch_hour_rotation_v1`. Both fields are written
to schema-3 status. Repeated invocations in the same UTC hour retain the same
order, so a manual retry cannot silently choose a favorable leader. Across a
complete `N`-hour cycle, every member of an `N`-symbol universe leads exactly
once. A provider throttle still stops the run immediately and marks only the
remainder of that exact request order as circuit-skipped.

Artifact verification requires the recorded order to be a complete,
duplicate-free permutation and to match the recorded UTC start time. Existing
schema-3 statuses created before this additive field remain valid under their
legacy fixed-order interpretation; a status containing only one of the two new
fields, an unknown policy, or a time-incoherent permutation fails closed.

## Evidence boundary and limitations

Rotation makes partial acquisition opportunities fairer without increasing
the number of requests, changing endpoint budgets, altering the symbol
universe, backdating availability, or admitting a partial run. It does not
prevent shared-IP throttling, guarantee that any one run completes, repair an
already missed first-seen interval, or make asynchronously refreshed symbols a
complete cross-sectional panel. Complete-pass artifact admission and all
point-in-time availability rules remain unchanged.

No return, rank, weight, position, PnL, forecast/economic/risk metric, model
fit, holdout, order, credential, or live-authorization state was read or
changed. The current champion and no-adoption decision remain unchanged.
