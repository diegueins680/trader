# Binance request-order rotation operational receipt v1

Date: 2026-09-06

## Observation

The first scheduled collector run after request-order rotation reached merged
`main` started at `2026-09-06T06:10:06.639250Z` and finished at
`2026-09-06T06:10:36.384908Z` after 29.745 seconds. Status schema 3 recorded:

- code commit `41fd825021ab4786885be36c29b6ece7f548736e`;
- `provenanceTrackedClean: true` and no provenance issues;
- policy `utc_epoch_hour_rotation_v1`;
- request order XRP, DOGE, ADA, AVAX, LINK, LTC, BTC, ETH, SOL, BNB;
- successful acquisition for XRP and DOGE;
- a typed provider-rate-limit failure for ADA with HTTP 429; and
- seven later symbols skipped by the provider-rate-limit circuit without a
  request.

The mutable local status file was 12,733 bytes with SHA-256
`a3f23fb35e72d4dbf1f050af7ae0b6edb8c8f023e07509a84c12f57bca5d6ba8`
when this receipt was prepared. The status file and all market artifacts remain
outside Git.

The request order is the exact registered-symbol rotation for UTC epoch hour
06: XRP is canonical index 4 and `floor(epoch seconds / 3600) mod 10 = 4`.
The collector stopped after the first throttle and did not attempt AVAX, LINK,
LTC, BTC, ETH, SOL, or BNB. No manual retry was launched.

## Interpretation

This observation demonstrates the intended operational mechanism on clean
merged code: an available partial-run prefix moved away from BTC/ETH/SOL and
allowed XRP and DOGE to refresh before throttling. It also demonstrates that
rotation did not weaken the provider-wide circuit or turn a partial run into an
admissible complete pass.

It does **not** establish a stable provider capacity, prove that this collector
caused shared-IP throttling, demonstrate all-symbol fairness over a complete
ten-hour cycle, repair missed first-seen observations, or authorize a partial
cross-sectional panel. `RESEARCH-RATE-LIMIT-001` therefore remains
`HIGH / MITIGATED`; stable approved egress is still the required closure path.

## Research and safety boundary

Only collection metadata, provenance, request outcomes, and the status-file
hash were inspected. No registered return, rank, weight, position, PnL,
forecast, economic or risk statistic, model input, holdout, credential, order,
or live-authorization state was read or changed. The current champion and the
no-adoption decision remain unchanged.
