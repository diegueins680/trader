# Threshold-factor simulation contract audit

Date: 2026-09-06 local / 2026-09-07 UTC

Risk: `THRESHOLD-FACTOR-001`

Disposition: closed as a stale wiring concern after deterministic integration
evidence

## Finding

The risk register retained an older concern that `thresholdFactor` might not
reach simulation configuration. Current code does wire the feature through the
complete offline evaluation path:

1. `Trader.App.Args` parses every threshold-factor option, checks all numeric
   values for finiteness, constrains alpha to `[0, 1]`, requires non-negative
   bounds/floor, and requires maximum to be no smaller than minimum.
2. `Trader.Optimizer.Optimize` stores the factor fields in `TrialParams`,
   normalizes them, emits explicit enabled/disabled CLI flags, and serializes
   the values into trial evidence.
3. Both historical `EnsembleConfig` constructors in `Main` copy the enabled
   flag, alpha, bounds, floor, eight feature weights, and LSTM health evidence
   into the simulator configuration.
4. `Trader.Trading.simulateEnsembleWithHLChecked` uses the prior causal factor
   state to scale open and close thresholds, minimum edge, and minimum
   signal-to-noise. It updates the bounded EMA only after the current decision,
   so that update can affect only later bars. Disabled mode fixes both factors
   at one and is behaviorally neutral.

The separate latest-signal history replay uses the same clamping, floor, EMA,
and feature-weight interpretation. This audit does not assert parity for every
floating-point operation between the two independently implemented paths; it
establishes that the named simulation configuration is present and operative.

## Deterministic witness

`testThresholdFactorChangesSimulatorAdmission` runs two simulations with
identical prices, predictions, costs, exposure, and base thresholds. The
forecast is two percent above a flat price and the base entry threshold is one
percent.

- With threshold factor disabled, deliberately configured factor bounds remain
  neutral and the simulator opens exposure.
- With threshold factor enabled and both bounds fixed at three, the effective
  entry threshold is three percent and the identical two-percent forecast
  remains flat with no trade.

The fixture is deliberately mechanistic. It does not optimize a parameter,
inspect a development or final holdout, or claim that a factor of three is
economically desirable. It proves that changing the enabled simulation
contract changes admission in the expected direction under controlled inputs.

## Decision

The exact registered next action—confirm the contract and add a simulation
integration witness—is satisfied, so the stale wiring risk can close. This
does not validate threshold-factor forecasting value. Existing candidates,
champions, defaults, artifacts, configurations, deployments, and live-order
authorization remain unchanged; no order is placed.

Reproduce from the repository root:

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```
