# Missingness-aware shallow feature panel v1 — implementation note

Date: 2026-09-09

Status: isolated, future-only research infrastructure; no model fit, experiment, holdout access, prediction, position, order, promotion, deployment, or live authority.

## Purpose

The preregistered `missingness_aware_calibrated_shallow_v1` candidate could not be implemented reproducibly while its price features were described only as returns, volatility, and trend and its tree settings referred to mutable “current defaults.” This outcome-blind amendment freezes those details before the registered data window begins on 2027-01-21.

The Haskell boundary `missingness_aware_calibrated_shallow_feature_panel_v1` composes three existing causal contracts:

1. required `complete_ohlcv_feature_inputs_v2` evidence;
2. optional `binance_derivatives_model_features_v2` evidence; and
3. optional `coinbase_cross_exchange_model_features_v2` evidence.

It accepts only the ten registered Binance USD-M symbols and the registered 1h, 4h, and 8h intervals. Every present optional panel must match the exact target symbol and bar grid and must have been decided no later than the primary OHLCV decision. The first 24 bars are causal lookback only. Registration bounds and phase alignment apply to the grid's exact bar-open timestamps: opens before the registered start or at/after the final-holdout boundary are rejected, while the final registered development open is admitted.

## Exact feature order

The first twelve fields are required and derived from completed close-to-close simple returns:

1. total returns over 1, 3, 6, and 24 bars;
2. population mean and standard deviation of the latest 6 one-bar returns;
3. population mean and standard deviation of the latest 24 one-bar returns;
4. 6-bar minus 24-bar total return;
5. one-bar return minus the 6-bar mean;
6. 6-bar divided by 24-bar return volatility, with zero only for an observed denominator at or below `1e-12`; and
7. 6-bar minus 24-bar mean return.

They are followed by the five registered derivatives fields and the five registered same-symbol Coinbase fields. These ten fields are optional individually and retain value, observed mask, event time, and availability time. An observed numeric zero keeps mask `true`; an unavailable value has dense compatibility value zero and mask `false`.

The panel deliberately does not expose the legacy dense `featureRowModelInputs` projection as an imputation policy. The future fitted artifact must estimate imputation means and normalization scales on each training prefix only, impute unavailable optional values to those training means, append the unchanged masks, and abstain when the compatible fitted preprocessor is absent or invalid.

## Cross-exchange provenance correction

`CrossExchangeInputsV2` previously validated Binance/Coinbase instrument mapping and then discarded both identifiers. It now retains and exposes the validated scope and immutable grid, allowing the composite panel to reject cross-symbol contamination. Existing formula outputs and legacy paths are unchanged.

## Frozen model defaults

The registration now records the numeric defaults rather than a moving source-code reference: ridge lambda `0.001`; GBDT 60 depth-one stumps at learning rate `0.1`; decision tree maximum depth 6 and minimum leaf size 12; linear quantiles q10/q50/q90 with 20 epochs, learning rate `0.05`, and L2 `0.001`; split-conformal alpha `0.2`; calibration fraction `0.2`; and seed `20270904`. Candidate and baseline counts remain 36, 81, and 117 total.

## Evidence and limitations

Deterministic synthetic tests cover exact ordering and formulas, source scope/grid alignment, observed-zero/missing distinctions, registered symbol and time boundaries, insufficient history, and causal future invariance. No public or private market row was read.

This does not close `FEATURE-MISSINGNESS-001`: production predictors remain on their compatible legacy dense schemas, no future Coinbase first-seen source has been admitted, and the candidate still lacks fold-local fitted preprocessing/artifact persistence. It also provides no forecast, calibration, economic, cost, drawdown, tail-risk, robustness, ablation, statistical, or performance evidence. The decision remains **no candidate passed; preserve the current champion**.
