# Observation causality audit (obligation 17)

Engineering preregistration, 2026-10-06. Read-only audit of unchanged source.

## Criterion and finding

Obligation 17 requires that the current policy observation is invariant under
changes to data first available after the decision. The frozen runner fits
`Scale` once on the whole training prefix `[0, trainStop)`. `collect` then
starts training episodes anywhere in `[24, L − 96)` of that prefix and
normalizes each observation, and the `supported()` range check that neutralizes
unsupported actions, with that scale. A training-time observation at
t < trainStop therefore depends on bars after t.

**CE-RL-025** (executable witness): a synthetic 300-bar series with decision
t = 100. Every bar after t is rewritten and the scale refitted, as the runner
does. The prefix up to t is unchanged, yet the observation at t changes.

## Scope of the defect

- **Affected:** training-time observations and the `supported()` gating applied
  during collection. These are inputs to every trained policy.
- **Not affected:** evaluation-time observations. They lie in
  `[testStart, testStop)` with `testStart ≥ trainStop`, so their scale depends
  only on earlier bars. Obligation 3 (causal normalization at fold level) stays
  closed: its criterion concerns training rows and evaluation immutability, not
  per-decision invariance.
- **Economic effect:** not established. The frozen screen was rejected anyway,
  and no decision changes.

## Obligations / methods

F-RL-OBS-TRAIN-SCALE (refuted): the claim "every training-time observation in the
frozen path is invariant under changes to later data" is refuted by CE-RL-025.
The checker binds the source (single prefix-wide `Scale.fit` passed to training;
`collect` episode start range; `observation()` normalization through
`self.scale`) and re-executes the witness on every formal run.

F-RL-OBS-REPLAY-CAUSAL (property tested): with a fixed scale, the actual
`Replay` observation, `supported()` flag and every receipt up to step t are
unchanged when all prices and funding after t are rewritten. This is checked at
every step of 24 seeded episodes. It isolates the defect to the scale.

Obligation 17 stays open. A repair requires a newly registered causal
normalization, for example a scale fitted on a burn-in segment that ends before
the earliest training episode start, or an expanding-window scale. Archived code
is not edited.
