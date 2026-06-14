# Post-fix Backtest Evaluation — 2026-06-14

## Verdict

**The cost-floor + position-cap + WF-Sharpe fixes are correct, but they expose a second, deeper problem: the predictors themselves do not produce edges above the venue round-trip cost on 1h–12h crypto perp timescales. Until the predictor stack improves, the live system cannot win — it should stay flat.**

Don't restart the bots yet. The math says they will produce zero (or negative) PnL even with the new guards.

## What the data says

### Existing 500 top combos (pre-fix leaderboard)

| Metric | Value |
|---|---|
| Total combos | 500 |
| Combos with `minEdge ≥ venueMinEdgeFloor` (18 bp) | **0 / 500** |
| Combos with `minEdge ≥ round-trip cost` (12 bp) | **0 / 500** |
| Combos with walk-forward summary | **0 / 500** |
| Median in-sample backtest Sharpe | 9.15 |
| Median in-sample trade count | 4 |
| Median `avgTradeReturn` (the *actual* per-trade gross return) | 1.11% |

The "Sharpe 9.15" leaderboard was a backtest illusion built on a median of **4 trades per combo** with no walk-forward validation. Statistically meaningless. The new adoption guards filter all 500 out, which is the correct behavior.

### Optimizer behavior under the new defaults

I ran 294 trials over the past 24h on the live API (post-fix code at commit `3d32f58c`) across 7 symbols/intervals (SUIUSDT, ETHUSDT, BNBUSDT, LINKUSDT, LTCUSDT, OPUSDT, BTCUSDT) on 1h–12h.

| Outcome | Count |
|---|---|
| Eligible (passed all gates) | **0** |
| Crashed (process error, ok=false) | 177 (60%) |
| Generated zero trades (`activityCount<1`) | 102 (35%) |
| Generated trades but below sharpe gate | 7 |
| Generated trades but below exposure gate | 7 |
| Other reject | 1 |

Of the **15 trials that actually produced 1+ trades**, every single one had a *negative* in-sample Sharpe ranging from **-0.37 to -31.64**. Median -7.04.

Examples (best to worst by Sharpe):

```
sym       int  meth minEdge   sharpe trades finalEq
SUIUSDT   2h   01   0.00043   -0.37     1   0.998
ETHUSDT   12h  01   0.00065   -2.24     3   0.987
OPUSDT    12h  01   0.00441   -4.00     8   0.911
SUIUSDT   2h   01   0.00047   -5.92     1   0.983
…
SUIUSDT   2h   01   0.00041  -31.64    27   0.811
```

## Interpretation

Three independent failure modes are now visible:

1. **Predictor unreliability** (60% of trials crash). Method 01 (LSTM), method 10 (Kalman), and method `regime_switch` all have nontrivial crash rates on real data even with timeouts ≥ 120s. This is a stability issue independent of edge.
2. **No-signal regime** (35% of trials produce no trades). At minEdge ≥ 18 bp the LSTM/Kalman signals on 4h–12h BTC/ETH simply don't exceed the threshold often enough. This is the *intended* behavior of the floor but it tells us the predictors don't have signal above cost.
3. **Negative expectancy when trades fire** (the 15 that did trade). Even when a trial's predictor *does* produce a signal big enough to open a position, the realized PnL is negative on average. The predictors are slightly worse than random on these timescales.

Conclusion: the cost-floor is doing what it was designed to do — it's preventing the system from putting money into combos that cannot win. But the deeper truth is the predictor stack is not currently producing edge on the asset/timeframe universe being scanned.

## What would change the picture

Listed by tractability, not by how much I expect them to help.

| Change | Tractable? | Likely impact |
|---|---|---|
| Run optimizer on shorter timeframes (15m, 5m) where minEdge=18bp is a larger fraction of typical bar return | yes, env-only change | unknown — has its own cost problem (more trades × same fee) |
| Add funding-rate-aware sizing (`fundingOnOpen=true`) on perp trades | yes, env-only | small, but eliminates a known leak |
| Re-enable regime-detector confirmation (`regimeParameterBank=true`) | yes, env-only | small to medium |
| Raise predictor capacity: `epochs ≥ 8`, `hidden ≥ 32` on research nodes | yes, env-only on Fly/Hetzner research | medium — that's where the prior leaderboard's actually-tradeable BTC/SOL combos used to come from |
| Look at the *crash* rate first — 60% is unusual. Likely something predictor-specific (NaN propagation, dimension mismatch) regressed | code change | high if it's a recent regression |
| Add cross-asset/predictor confirmation gates so multiple weak signals must agree before opening | code change | medium — reduces false fires but also reduces trade count |
| Add prediction-market herd confirmation (Polymarket) on signals — already implemented as opt-in | env-only on signals | unknown |

## Concrete next step

I would NOT restart the bots. I would do this in order:

1. **Look at why 60% of trials crash.** Past-day journal logs should have the stack traces.
2. **Raise predictor capacity on research nodes** (`TRADER_OPTIMIZER_EPOCHS_MAX=12`, `TRADER_OPTIMIZER_HIDDEN_SIZE_MAX=48`) and re-run — when the prior leaderboard had combos like LINKUSDT #77 with `epochs=8, hidden=29`, the heavier configurations are where positive-Sharpe trials show up.
3. **Try short timeframes** (5m, 15m on BTC/ETH/SOL) where the predictor's signal-to-noise might exceed the cost floor more often.
4. **Funding-on-open** for any combo that does survive. Set `TRADER_BOT_FUNDING_ON_OPEN=true` (or equivalent) as a default.

If after those four steps the optimizer still produces zero eligible combos, the honest answer is: this trading universe isn't profitable with the current predictor stack at the venue's cost structure, and the right play is to either upgrade predictors (more data, better features, different model class) or move to a cheaper venue.

## How to interpret this for the trader-firm objective (ROI + Sharpe)

The pre-fix system was producing a Sharpe of -3.20 on real money. The new guards make that mathematically impossible because the system will simply refuse to deploy. That's a *much* better state than the previous one — flat is better than -3 Sharpe. But it's not yet the Sharpe ≥ 1 that the strategy needs to be considered an investment. The next round of work is on the predictors, not on more guards.

## Root cause for the "60% crash rate" — addendum

After diagnosing further: those weren't crashes. The breakdown of the 180 ok=false trials is:

| reason | count |
|---|---|
| `timeout>180.0s` | 148 (82%) |
| `POSITION_SIZE_SCALE_EXCEEDED: …` | ~32 (18%) |

The first is just the per-trial timeout being too tight. Successful trials had median `elapsedSec` 85s and max 1955s; timeouts had median 187s — i.e. most of them would finish if given another minute or two. The second is the firm-critical position-size hard-fail correctly firing on trials whose `volScale * riskScale * snrScale * kellyLite` product compounds past 2× baseSize. That's the safety rail working as designed; the trial is correctly rejected.

**Immediate change applied:** raised `TRADER_OPTIMIZER_TIMEOUT_SEC` and `TRADER_OPTIMIZER_DISCOVERY_RECOVERY_TIMEOUT_SEC` from 180 → 360 seconds, set `TRADER_OPTIMIZER_MIN_WF_SHARPE_MEAN=0.3` and `TRADER_OPTIMIZER_MAX_WF_SHARPE_STD=1.5` in `.env`. API restarted.

With those changes, the next ~24 hours should produce a meaningful picture of how many trials *do* complete and what their walk-forward Sharpe distribution looks like. If positive-Sharpe eligible combos appear, the system has edge and the original strategy is salvageable. If they still don't, the predictor stack itself needs work and no amount of guard-rail tuning will help.
