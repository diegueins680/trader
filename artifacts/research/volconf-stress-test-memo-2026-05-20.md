# Vol/Conf Gate Stress-Test Memo — 2026-05-20

## Objective
Produce a scorecard showing trade retention < 100% when vol/conf gate filters low-confidence or high-volatility bars.

## Findings

### Real Data (BTCUSDT-4h-1000)
- ta_trend: 5 trades, all presets retain 100% trades
- ta_breakout: 7 trades, all presets retain 100% trades
- Entry bars have vol 0.20-0.47, conf 0.77-0.90 (all within gate limits)

### Synthetic Datasets
- Multiple synthetic datasets created with injected high-vol/low-conf windows
- Entries consistently occur outside high-vol windows due to signal disruption
- Gate reduces position size (e.g., 0.739 -> 0.554) but does not eliminate trades
- No preset achieved retention < 100% on any tested dataset

### Gate Behavior
- Gate IS applied during backtest (positions differ between disabled and active presets)
- Effect is primarily position-size reduction, not trade elimination
- Trade elimination requires VolHigh + ConfidenceWeak at entry bar
- This combination is rare because high-vol bars typically have strong directional moves

## Blocker / Next Step
To observe retention < 100%, we need either:
1. A dataset where an entry naturally occurs on a VolHigh + ConfidenceWeak bar, OR
2. A unit test that directly injects gate decisions at specific bars

Recommendation: Create a minimal Haskell unit test that mocks the gate cell to Block at a specific bar and verifies the trade is skipped.

## Artifacts
- Scorecards: artifacts/research/volconf-scorecard-*.md
- Synthetic datasets: data/stress-*.csv
