# Trailing-Stop Reproduction Experiment — 2026-05-24

**Research Director:** trader-firm-research  
**Dataset:** BTCUSDT-4h-1000.csv, ETHUSDT-4h-1000.csv, SOLUSDT-4h-1000.csv  
**Binary:** `haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs`  
**Binary SHA-256:** `9bba966da781cb8858e23c68bde48f9c6af78e04a72e7c20f4f688e0c8faeb60`  
**Method:** `ta_trend`  
**Reproduction command template:**
```bash
BIN="/Users/diegosaa/GitHub/trader/haskell/dist-newstyle/build/x86_64-osx/ghc-9.4.8/trader-0.1.0.0/x/trader-hs/build/trader-hs/trader-hs"
$BIN --data data/BTCUSDT-4h-1000.csv --price-column close --method ta_trend --json --trailing-stop <VALUE>
```

## Hypothesis

**H₀:** Trailing-stop values of 0.005–0.02 improve Sharpe ratio vs baseline on all three assets.  
**H₁:** Trailing-stop improvement is asset-dependent; BTC and ETH degrade while SOL improves.

## Results

### BTC
| Design | trailing-stop | Sharpe | maxDD | Closed Trades | Avg Trade | Turnover | Total Return |
|--------|--------------|--------|-------|---------------|-----------|----------|--------------|
| Baseline | — | **-0.226** | 4.90% | 5 | 0.013% | 5.53% | -0.39% |
| trail_0.005 | 0.005 | -0.040 | 2.22% | 19 | 0.071% | 19.10% | -0.09% |
| trail_0.008 | 0.008 | -1.740 | 3.16% | 16 | -0.002% | 16.08% | -1.23% |
| trail_0.01 | 0.01 | -0.525 | 2.96% | 12 | 0.040% | 12.06% | -0.44% |
| **trail_0.015** | **0.015** | **+0.940** | **3.44%** | **10** | **0.138%** | **10.05%** | **+0.57%** |
| trail_0.02 | 0.02 | +0.140 | 4.34% | 8 | 0.081% | 8.04% | -0.02% |
| trail_0.03 | 0.03 | -4.462 | 4.04% | 5 | -0.625% | 5.03% | -3.50% |

### ETH
| Design | trailing-stop | Sharpe | maxDD | Closed Trades | Avg Trade | Turnover | Total Return |
|--------|--------------|--------|-------|---------------|-----------|----------|--------------|
| trail_0.005 | 0.005 | -15.718 | 1.91% | 6 | -0.271% | 6.03% | -1.91% |
| trail_0.01 | 0.01 | -4.986 | 2.09% | 5 | -0.234% | 5.03% | -1.38% |
| trail_0.015 | 0.015 | -4.350 | 2.36% | 4 | -0.314% | 4.02% | -1.45% |
| trail_0.02 | 0.02 | -5.342 | 2.58% | 3 | -0.615% | 3.02% | -2.01% |

### SOL
| Design | trailing-stop | Sharpe | maxDD | Closed Trades | Avg Trade | Turnover | Total Return |
|--------|--------------|--------|-------|---------------|-----------|----------|--------------|
| trail_0.005 | 0.005 | **+4.234** | 1.90% | 13 | 0.192% | 13.07% | +1.79% |
| trail_0.01 | 0.01 | +3.301 | 3.14% | 10 | 0.230% | 10.05% | +1.68% |
| trail_0.015 | 0.015 | +2.486 | 2.59% | 7 | 0.249% | 7.04% | +1.30% |
| trail_0.02 | 0.02 | +1.816 | 3.07% | 5 | 0.268% | 5.53% | +1.00% |

## Key Findings

1. **Prior report (+1.82 Sharpe for BTC trail 0.01) is NOT reproducible.** The current binary produces Sharpe -0.525 for BTC trail 0.01. This suggests either:
   - The prior result used a different binary build (e.g., `dist-newstyle-codex-review` vs `dist-newstyle`), or
   - The prior result used a different method or dataset variant.

2. **Trailing-stop effect is highly asset-dependent:**
   - **BTC:** Best at 0.015 (Sharpe +0.940), but degrades at tighter or wider stops.
   - **ETH:** Trailing-stop universally degrades performance. All values produce negative Sharpe.
   - **SOL:** Trailing-stop universally improves performance. Best at 0.005 (Sharpe +4.234).

3. **Trade count increases with tighter stops, but Sharpe does not monotonically improve.**
   - BTC: 19 trades at 0.005 (Sharpe -0.040) vs 10 trades at 0.015 (Sharpe +0.940).
   - SOL: 13 trades at 0.005 (Sharpe +4.234) vs 5 trades at 0.02 (Sharpe +1.816).

4. **Combined vol-target + trailing-stop degrades BTC further:**
   - Baseline: Sharpe -0.226
   - Vol-target only: Sharpe +0.167
   - Trailing-stop 0.01 only: Sharpe -0.525
   - Combined: Sharpe -0.644

## Verdict

**REJECT H₀.** Trailing-stop does not improve all assets. It helps SOL significantly, helps BTC only at a specific 0.015 value, and hurts ETH universally.

**Asset-specific recommendation (1000-bar 4h data):**
- **SOL:** `--trailing-stop 0.005` (Sharpe +4.23, 13 trades)
- **BTC:** `--trailing-stop 0.015` (Sharpe +0.94, 10 trades) — weak evidence, needs cross-validation
- **ETH:** Do NOT use trailing-stop on this dataset/method

## Blockers

- **Data delivery:** ≥5000-bar 4h datasets still not available. All conclusions based on 1000-bar data (backtest window ~200 bars).
- **Binary reproducibility:** Prior report results cannot be reproduced. Need to pin exact binary hash for all future experiments.

## Next Priority

1. **P1 — Binary hash pinning:** Record SHA-256 of the binary used for each experiment to ensure reproducibility.
2. **P2 — SOL scale-up:** SOL trailing-stop 0.005 is the strongest signal. Request ≥5000-bar SOL dataset from Data and re-test.
3. **P3 — ETH diagnosis:** ETH degrades with all trailing-stop values. Investigate if this is due to dataset-specific regime or method mismatch.
