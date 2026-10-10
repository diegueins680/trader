# Training-budget sensitivity (2026-10-10)

The preregistration commit is `ff3cce38`, preceding implementation tests and
market outcomes. This is an adaptive continuation of the rejected v2 package,
from verified main `354c853176b7ee2ac6570951f7f6f2d0aed6909d`. Prior evidence and
protected holdouts remain unchanged. There is no current-champion comparison.

The sequential task remains bounded target exposure with inventory, trading
costs, funding and next-close fills. The v2 environment, exact ledger, safety
shield, twelve observations, network architecture and learning rates are fixed.
Only the training budget changes from 512 to 4,096. PPO collects fresh rollouts
and makes 128 actor updates; Double DQN collects 4,096 simulator transitions and
makes 4,096 replay updates. Discrete CQL, its alpha-zero diagnostic and behavior
cloning make 4,096 updates on the original 512-transition simulated datasets.
Both buffer content identities must match before any fit. This separates a
larger optimizer budget from additional offline data; training regimes do not
have identical numbers of unique interactions.

All five seeds and last checkpoints are retained. There is no checkpoint tuning,
best-budget selection or claim of convergence. Cash, constant-long and both
normalized observation-vector controls are included before outcome inspection.
The normalized momentum rule uses deviation from burn-in mean, not raw trend.
The original 20% action-change gate remains: unsupported proposals can be
aliased to the same zero target, but no mask or reward change conceals that issue.
PPO is trained with sampled proposals; final evaluation uses the registered
argmax extraction. All families evaluate through the same shield and ledger.

Data are the two previously exposed BTC/ETH/SOL eight-hour development windows,
with assumed close+1ms availability, 96-bar causal burn-in and immutable
per-episode scaling. Training episodes have 32 transitions; evaluation episodes
have 64. Funding is assigned to its next close; old inventory earns price/funding
changes before the due target fills. Seven inherited execution stresses apply.
No real behavior logs or exchange propensities exist. Historical replay is
model-dependent counterfactual evidence. DSR/PBO remain insufficient evidence.
No new holdout is created by giving this continuation a new campaign identifier.

Reproduction uses the pinned research environment and an empty inherited
environment except the executable PATH and single-thread BLAS settings:

```sh
python scripts/research/run_rl_budget_v3.py \
  --run-registered-budget-v3 \
  --panel /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-panel.csv \
  --settlements /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1/registered-development-settlements.csv \
  --output /private/tmp/trader-rl-budget-v3-evidence
python test/rl_composition_test.py
bash scripts/verify.sh full
```

The [registration](../registrations/rl-training-budget-v3.json) specifies all
seeds, hashes, comparisons and limits. This runner preserves the v1/v2 entrypoint
defaults. Reproducing the frozen v2 supplemental controls still requires their
recorded commit, because their source checks intentionally reject later runners.
Research artifact environment version remains v2; the new campaign and
registration hash identify this budget. No production loader is added.

Formal scope is unchanged: seven composition SMT lemmas, two zero-target
admission lemmas, a 43-state/107-edge bounded lifecycle model, eight Haskell exact
ledger witnesses and the turnover counterexample. Numerical and runtime trust
assumptions remain. The new orchestration is source-locked with targeted tests;
this is not a new theorem about learning efficacy or production integration.
