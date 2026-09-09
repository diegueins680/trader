# Market-context linear artifact v1 — research infrastructure result

Date: 2026-09-09.

## Result

`Trader.Predictors.MarketContextLinearArtifactV1` adds a pure, research-only
fitted-component boundary between the causal
`point_in_time_market_context_factor_v2` row and any future registered shallow
candidate. Its distinct semantic identity is
`point_in_time_market_context_linear_ols_v1`; it does not replace or alias the
legacy `MarketContext` implementation.

The fit requires the complete ordered training grid, a target event exactly at
the declared horizon, target availability no later than the fit cutoff, a
purge at least as long as that horizon, and the declared embargo before the
first validation event. The request binds the registration, code commit,
training-data/source/split digests, academic origins, symbol, universe and peer
scope, interval and horizon, residual-variance floor, runtime versions, cost
model reference, and creation time.

Unavailable factor rows remain in the grid and row count, but their dense zero
is excluded from OLS. Fitting requires at least three observed rows and
non-degenerate finite factor dispersion. Inference checks the exact symbol,
interval, horizon, feature schema, causal timestamps, post-training boundary,
payload digest, compatibility version, and finite positive variance. Failure
returns no estimate. The only successful output is a conditional mean and
residual variance; it is not an actionable signal.

## Artifact and authority boundary

The strict JSON envelope stores a canonical SHA-256 of its payload. Unknown
fields, payload corruption, semantic/schema/version drift, non-finite fit
values, and any research-admission, experiment, holdout, model, promotion,
deployment, order, or live-trading authority fail decoding. A future external
candidate manifest must still bind the exact artifact bytes; the internal hash
detects accidental corruption but is not an authenticity signature.

The component explicitly records validation metrics as `not_evaluated`, the
final holdout as `untouched`, and promotion state as
`offline_research_only`. It is compiled only into the test component and is
absent from the trader executable's module graph and all predictor, bot,
champion, execution, and live paths.

## Evidence and disposition

Synthetic tests recover a known intercept and slope, prove that an unavailable
row with a deliberately adverse label cannot influence the fit, check exact
round-trip bytes and scope-compatible inference, and reject incomplete grids,
insufficient purge, late labels, degenerate factors, non-finite targets,
corruption, rehashed semantic drift, and rehashed live-authority escalation.

No market data was acquired. No forecast or economic metric, trial, ablation,
development result, or final-holdout result was produced. The experiment
registry therefore receives no trial entry. No candidate passed, the current
champion remains unchanged, and `FEATURE-MISSINGNESS-001` remains open pending
prospective data, separate admission, remaining timestamp-preserving source
contracts, a complete candidate artifact, production builders, and every
registered statistical, economic, cost, risk, and operational gate.
