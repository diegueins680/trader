# HAR-RV risk-gate artifact v1

## Decision boundary

`bar_har_rv_ridge_risk_gate_v1` is an isolated offline prototype of the preregistered `har_rv_risk_gate_v1` candidate. It is not a production predictor, an admitted research model, a champion challenger, or evidence of predictive or economic value. No market outcome, development result, final holdout, order path, deployment setting, credential, or live authorization was read or changed while implementing it. The registered data start remains `2027-01-21T00:00:00Z`; the final holdout remains untouched.

The Haskell module consumes only the abstract `complete_ohlcv_feature_inputs_v2` boundary for one exact symbol and registered interval. It accepts only the registration's exact ten Binance USD-M symbols; any syntactically valid symbol outside that frozen universe fails before evidence hashing or fitting. It remains absent from the legacy predictor registry, ensemble, optimizer, bot, API, CLI, artifact loader, execution, and deployment paths. Synthetic deterministic fixtures are the only fitted data in its tests.

## Frozen interpretation

- A completed-bar return is `log(close[t] / close[t-1])`.
- The three features are the natural log of summed squared returns over trailing 1-, 6-, and 24-bar blocks ending at the decision bar.
- The target is the natural log of summed squared returns over the next separately registered 1, 3, or 6 bars.
- A zero, incomplete, missing, stale, malformed, or non-finite variance block is unavailable; inference then returns absent and the public fail-closed scale is zero.
- Feature scaling uses training-only population means and population standard deviations. The unpenalized intercept is the training-target mean; the three standardized slopes solve `X'X + 1e-6 I` ridge regression.
- Forecast volatility is `exp(expected_log_variance / 2)`. The risk scale is `min(1, training_median_fitted_forecast_volatility / current_forecast_volatility)`.
- The optional 95% volatility bounds transform a Gaussian residual interval on the log-variance scale. They are explicitly uncalibrated diagnostics and cannot satisfy interval, calibration, or promotion gates without future out-of-sample evidence.

These details were clarified in the registration on 2026-09-09, before the future dataset begins and before any outcome evaluation. They do not add a configuration or expand the fixed 36-experiment budget.

## Artifact and safety contract

The artifact has an explicit schema, semantic model identifier, and compatibility version. Its payload records the registration, source papers, code revision, exact training-data and training-evidence hashes, source/split manifests, symbol, interval, horizon, train/validation/final-holdout boundaries, purge and embargo, fit/creation clocks, the fixed registered seed `20270904`, runtime versions, cost-model identity, training-only normalization, coefficients, residual dispersion, fit metrics, and training median forecast volatility. A SHA-256 digest covers the canonical payload.

The separately computed evidence digest covers every ordered complete OHLCV field, mask, event clock, first-seen clock, decision clock, and open time that can affect the 24-bar lookback or forward target. Changed fitted evidence therefore cannot be accepted by merely reusing the old request. Evidence after that causal slice does not change the digest or an earlier forecast.

Decoding rejects missing or unknown fields, unsupported versions or semantics, non-finite/invalid fit state, and digest mismatch. The serialized safety block is fixed to `offline_research_only`, `not_evaluated`, and `untouched`; every admission, experiment, holdout, model, promotion, deployment, order, and live-authority flag is false. Inference is restricted to the declared development-validation window, exact grid phase, pre-holdout targets, and fitted scope. The exposure helper multiplies the champion exposure only by a finite scale in `[0,1]`, so it cannot flip direction or increase absolute exposure; any invalid input produces zero exposure.

## Evidence and remaining gates

Deterministic Haskell tests cover fit shape and finiteness, artifact round-trip, valid-JSON digest tampering, malformed bytes, scope mismatch, off-grid and out-of-window inference, final-holdout refusal, flat-market abstention, non-finite champion exposure, same-sign exposure reduction, fitted-evidence mutation, and invariance to a later unused bar.

No forecast metric, economic metric, transaction-cost result, statistical diagnostic, drawdown comparison, ablation, robustness result, or performance benchmark has been calculated. The candidate remains `continue research`; the current champion is unchanged. Before any development evaluation, the prospective source artifact, split manifest, cost model, champion snapshot, complete baseline implementation, experiment registry, and runner must be independently frozen and reviewed. Promotion still requires every preregistered economic, DSR, PBO, confidence, drawdown, symbol/regime, cost, delay, ablation, failure, and CPU/memory/latency gate.
