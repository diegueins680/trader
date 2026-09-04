# Existing predictor and trading-component fidelity audit

Audit date: 2026-09-04. Code baseline: `1563f63d6a0a29f22456bd53aa81cabc648fd740`.

## Inventory and contract map

| Surface | Current implementation | Contract / persistence |
|---|---|---|
| Predictor IDs | `gbdt`, `knn`, `decision_tree`, `tcn`, `patch_tst`, `transformer`, `hmm`, `quantile`, `conformal` | Parsed by `Trader.Predictors.Types`; CSV serialization emits legacy canonical IDs |
| Primary strategy methods | Kalman-only `10`, LSTM-only `01`, both/blends/routers, online NN, TA families, residual “cross-sectional momentum” | Parsed by `Trader.Method`; saved combo JSON and bot configuration depend on stable strings |
| Feature path | OHLCV returns/volatility/trend; market context; derivative fields; Coinbase; round-level psychology; external feature families | `FeatureInputs` to deterministic row builders; exogenous values aligned by publication time where metadata exists |
| Regime/context | Three-state Gaussian HMM; market-basket OLS context; TA regime switch; router state | HMM posterior is filtered causally; market universe can be point-in-time or a current-volume fallback |
| Alternative data | JSON/CSV families plus public acquisition for market and derivative series | Event/publication-time alignment exists; optional inputs often collapse unavailable values to neutral zero |
| Backtest | Bar-close decision path with explicit position state, fees/slippage/spread-like cost inputs, funding options, turnover and delay controls | `BacktestResult` stores equity, exposure, changes, trade records, and cost attribution |
| Optimization | Threshold tuning plus large outer search; five-fold default walk-forward and one-bar fold-edge embargo; minimum round trips | Ordinary random CV is not used, but production folds are not a full nested purged research protocol for every model |
| Overfit diagnostics | Production emits empirical best-score p-value, `deflatedSharpeProxy`, and `pboProxy`; research scripts implement formal DSR and CSCV-PBO | Proxy fields are scoring/admission heuristics and are not substitutes for the research promotion gates |
| Champion/top combos | Eligible trials filtered then scored by validated return, walk-forward multiplier, drawdown, live evidence, and optional freshness/diversity | PostgreSQL/top-combo JSON payloads preserve legacy method/predictor strings; missing walk-forward evidence receives a penalty rather than automatic proof of deployability |
| Promotion lifecycle | Offline optimization, stored combos, shadow selector, paper/live runtime with explicit flags and risk gates | New predictor IDs are not automatically promoted; deployment source revision and environment gates are separate |
| Persistence | LSTM flat parameter persistence uses a `v1` cache key; top combos and bot configs persist JSON/string semantics | LSTM artifacts do not contain the full provenance manifest required by this mission; tabular/sequence sensors are generally trained in memory |
| API/CLI/web | `--predictors` is a CSV string; APIs expose configs/metrics; web surfaces strategy selection and optimizer evidence | No versioned model-card or general artifact-provenance API exists; changing legacy predictor output strings would break compatibility |

The reviewed live fleet in the checked-in Hetzner trading profile is AVAXUSDT, UNIUSDT, SUIUSDT, ETCUSDT, and ADAUSDT. The research optimizer universe is broader (22 symbols). The historical research campaigns use BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, and LTCUSDT. BTC and ETH are therefore the liquid benchmarks, but the deployed fleet and research panels are not identical.

## Predictor fidelity classifications

| Identifier/component | Classification | What it actually does | Material gap or caveat |
|---|---|---|---|
| GBDT | Faithful with documented simplifications | Deterministic squared-error gradient boosting over depth-one regression stumps and a bounded threshold grid | Not a general depth-controlled/categorical/missing-aware GBDT; weak interactions; no probabilistic objective |
| Decision tree | Faithful with documented simplifications | Recursive deterministic regression tree with max depth 6, minimum leaf 12, and bounded split candidates | Single tree; no pruning/uncertainty; non-finite rows are dropped rather than modeled as missing |
| KNN | Faithful with documented simplifications | Normalized inverse-squared-distance regression with `k=15` and bounded evenly sampled history | Distance instability and sparse analogues in high dimension; no learned metric |
| HMM | Faithful with documented simplifications | Three-state Gaussian HMM fit by Baum–Welch; causal filtered posterior and one-step state forecast | Mapping latent states to “trend,” “mean reversion,” and “high volatility” is a post-fit heuristic, not identified by the likelihood |
| LSTM (`01`) | Faithful with documented simplifications | A real vanilla LSTM with input/forget/output/candidate gates, recurrent state, AD gradients, and Adam | Raw-price target, limited normalization/validation, point output, and incomplete artifact provenance are economically material |
| Online neural (`online_nn`) | Faithful | Causally updated two-hidden-layer tanh MLP over tabular/context features | It is not recurrent; callers must not describe it as an online LSTM; adaptation can chase drift without a promotion boundary |
| Kalman (`10`) | Faithful with documented simplifications | Recursive linear-Gaussian state estimates with numerical floors plus optional physics-error and multi-sensor fusion | Model/state interpretation and noise calibration are heuristic; open numerical-stability risk remains in the canonical register |
| TCN (`tcn`) | Lightweight proxy / mislabeled | Builds a fixed vector of one-bar returns at hand-selected power-of-two dilated lags, then fits dense ridge regression | No learned causal convolutions, convolution kernels, nonlinear blocks, residual/skip connections, or neural optimization |
| PatchTST (`patch_tst`) | Lightweight proxy / mislabeled | Computes five scalar summaries for several trailing patch lengths and fits dense ridge regression | No patch-token embedding, self-attention, positional handling, channel independence, or neural optimization |
| Transformer (`transformer`) | Lightweight proxy / mislabeled | Stores feature/target rows and returns a temperature-softmax weighted target average from raw dot-product similarity | No learned Q/K/V projections, multi-head attention, positional encoding, feed-forward layers, or end-to-end Transformer training |
| Quantile | Faithful with documented simplifications | Three linear models optimized by SGD pinball loss for q10/q50/q90; prediction reorders/clamps the median | No joint non-crossing objective; calibration and serial dependence are not established |
| Conformal | Inspired by the named method | Split residual radius around a GBDT point forecast plus an adaptive multiplicative radius heuristic | Static split logic resembles conformal calibration; the adaptive rule is not a faithful implementation of Gibbs–Candès and has no time-series conditional-coverage guarantee |
| Technical analysis sensors | Inspired / heuristic | Deterministic moving-average, trend, reversal, breakout, volatility, cloud, and price-action rules | Names describe rules rather than papers; many degrees of freedom require full search accounting and cost-controlled OOS tests |
| Sensor fusion / ensembles | Inspired by forecast combination and online experts | Confidence picks/blends, robust medians/harmonic rules, routers, hedge/meta-hedge weights, uncertainty and disagreement gates | Several methods select on recent realized errors, but no single formal ensemble paper is reproduced; all adaptation must remain delayed and bounded |
| Market-context model | Faithful simple linear factor with caveat | Volume-weighted market lag and OLS target relation; optional Coinbase context | Current-volume fallback can induce survivor/universe leakage unless point-in-time membership is required |
| `CrossSectionalMomentum` | Mislabeled | A single target asset's return is residualized against a market basket; the residual is used as time-series momentum and scaled by funding/basis/taker/OI | It does not rank multiple contemporaneously investable assets. True cross-sectional ranking exists only in offline research scripts |
| Prediction-bias logic | Incomplete / unverified | Issue #119 proposes rolling bias neutralization, but no current `scripts/model_health.py` implementation exists at the audited revision | Acceptance criteria in the issue are not a preregistered net-OOS test; an adaptive correction could erase valid regime signal or introduce feedback |

## Required sequence-model confirmation

The preliminary concern is confirmed from the current source:

- `Trader.Predictors.TCN.trainTCNWithLambda` enumerates fixed dilated lags and calls a local dense `ridgeFit`. Its model holds dilations, kernel-size metadata, ridge weights, and residual sigma.
- `Trader.Predictors.PatchTST.trainPatchTSTWithLambda` chooses patch lengths, converts each patch to total return, mean, standard deviation, last return, and directional agreement, then calls dense ridge regression.
- `Trader.Predictors.Transformer.predictTransformer` computes raw similarity between a query and stored rows, applies a temperature softmax, and averages stored targets. Training mainly validates and bounds the memory.

These implementations are deterministic, CPU-cheap, and generally fail unavailable on malformed model/query shapes. Those virtues do not make them faithful namesake architectures. The compatibility-safe response is:

1. Preserve `tcn`, `patch_tst`, and `transformer` serialization and behavior.
2. Accept accurate input aliases—`dilated_lag_ridge_v1`, `patch_summary_ridge_v1`, and `similarity_attention_v1`—that resolve to the unchanged legacy implementations.
3. Expose an implementation-identity helper for diagnostics/model cards without changing legacy `predictorCode` output.
4. Require any future faithful model to use an explicit new semantic identifier (for example `patch_tst_neural_v1`) and a versioned artifact contract.

## Feature causality and missingness

Strong points:

- Price features use historical indices and completed input vectors.
- External observation alignment has event/publication-time concepts.
- Predictor training drops malformed rows, dimensions are checked, and predictions are finite or absent in most model paths.
- Research campaign code hashes registered inputs and contains direct no-lookahead, nested-selection, and holdout-registry tests.

Material gaps:

- `Features.hs` encodes many unavailable or non-finite optional external values as `0`. That is neutral for a hand-written multiplier only if zero is explicitly the neutral element, but in a fitted model it is indistinguishable from an observed economic zero unless a coverage/missingness field accompanies it.
- Some predictor training paths drop malformed rows while returning a configured zero-weight model when training is empty. A numeric zero forecast must remain neutral and low confidence; it cannot be interpreted as observed bearish/bullish evidence.
- `MarketContext` may use a current top-volume universe when point-in-time membership is not required. That fallback is useful operationally but unsuitable for a survivorship-controlled research claim.
- Labels are predominantly one-step returns. The current public predictor contract does not independently version 1-, 3-, and 6-bar distributional targets.

Accordingly, no new external-data predictor should become trading-eligible until the feature schema carries availability/coverage explicitly and required missing inputs force abstention.

## Validation and economic-fidelity audit

The production optimizer's walk-forward score partitions the backtest range and trims a configured embargo from both edges. It is stronger than random CV but is not equivalent to the research harness's nested rolling-origin selection with purged overlapping labels. The production `deflatedSharpeProxy` subtracts a normal-quantile multiple of cross-trial Sharpe dispersion; `pboProxy` transforms the selected walk-forward instability. Both are accurately labeled proxies in JSON and must not be presented as formal Bailey/López de Prado DSR or CSCV-PBO.

The offline campaign harness provides the more defensible research standard:

- chronological nested selection;
- full aligned trial-return matrices;
- formal DSR and CSCV-PBO;
- lifetime Bonferroni correction;
- circular/moving-block confidence intervals;
- cost and delay stresses;
- one-shot overlap-aware final-holdout control;
- stop-on-breach risk evidence without clipping or synthetic continuation.

The backtester charges configurable entry/exit and rebalance costs, can attribute funding, records turnover/exposure, and supports added delay. It remains a bar model. Queue position, partial/missed fills, nonlinear market impact, exact maker selection, intrabar liquidation, exchange outage/counterparty loss, and infrastructure cost are not established by historical bar output. These limitations preclude microstructure or high-turnover promotion without additional replay evidence.

## Champion, shadow, paper, and live controls

Eligible optimizer trials feed top-combo selection; scoring blends validated annualized return with walk-forward, drawdown, live-evidence, and freshness terms. Stored top-combo payloads and bot configurations make method/predictor string compatibility operationally significant. The checked-in trading role already contains live authorization for its existing reviewed fleet, while example and research profiles are false/read-only. This audit neither changes those flags nor authorizes a new candidate.

New research candidates must remain isolated challengers. No offline/shadow artifact may automatically replace the adopted combo, enable a predictor, change the bot method, change environment authorization, or reach private order endpoints. A human-approved micro-live trial is explicitly outside this task.

## Persistence and provenance

Current LSTM persistence uses a deterministic key beginning with `v1` and serialized parameters, while combo stores retain JSON configuration and metrics. Neither is a complete mission-grade model artifact. Before a future persisted challenger is eligible even for shadow deployment, its artifact must bind model family/version, semantic ID, source papers, commit, training-data hashes, source manifest, schema/target/horizon, universe, split manifest, hyperparameters, seeds, runtime versions, train/validation/holdout metrics, cost model, creation time, promotion state, and compatibility version. Corruption, schema mismatch, non-finite data, unsupported versions, missing required features, or hash failure must yield an unavailable prediction and safe abstention.

## Conclusion

No existing predictor is promoted or behaviorally replaced. The three neural-sounding sequence IDs are confirmed proxies. The main economically relevant gaps are not architectural novelty: they are truthful semantic identity, missingness-aware features, formal research diagnostics at the promotion boundary, complete artifact provenance, point-in-time universes, and execution fidelity.
