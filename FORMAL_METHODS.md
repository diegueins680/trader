Update the ROI formal-methods section with one additional proved property.

1. In `What is proved`, add a bullet after the ROI monotonicity/penalty bullets:
- malformed ROI submetrics fail closed: replacing a modeled finite annualized return, drawdown, tail loss, turnover, expectancy, payback duration, activity count, or exposure input with malformed data (NaN, Infinity, or impossible negative penalty inputs) never increases the ROI score

2. In the ROI proof-sketch text, add a short paragraph explaining:
- reward-like malformed inputs normalize to a large negative sentinel
- penalty-like malformed or impossible-negative inputs normalize to a large positive sentinel
- malformed exposure/activity/payback normalize to their most conservative no-reward / maximum-penalty forms
- because `TuneRoi` in `haskell/app/Trader/Optimization.hs` delegates to `roiImplementationScore`, threshold sweeps and optimizer winner selection inherit the same fail-closed contract without a separate production ranking branch

3. If you keep the state-count prose, mention that the malformed-input invariant is checked as an extra bounded replacement matrix over the existing ROI state space.