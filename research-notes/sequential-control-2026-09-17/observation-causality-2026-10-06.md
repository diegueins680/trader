# Observation causality audit (obligation 17) — 2026-10-06

**Obligation 17 stays open, with a new counterexample: CE-RL-025.** This is a
read-only audit: no source, registration, archived result or market data
changed. Counts stay 13 scoped closures / 24 partial / 1 open.

## Finding

The frozen runner fits `Scale` (mean/std and support ranges) once on the whole
training prefix `[0, trainStop)`. `collect` starts training episodes anywhere in
`[24, L − 96)` of that prefix and normalizes each observation, and the
`supported()` gate that neutralizes unsupported actions, with that scale. So a
training-time observation at t depends on bars after t.

Witness (synthetic, 300 bars, decision t = 100): every bar after t is rewritten
and the scale refitted, as the runner does. Prices up to t are identical, yet
the observation at t changes by up to **1.050554799**.

With a **fixed** scale, the actual `Replay` observation, `supported()` flag and
receipts up to t are unchanged at all 250 decisions of 24 seeded episodes. The
defect is therefore entirely the prefix-wide scale.

## What it affects

- **Affected:** training inputs of every frozen policy (PPO, Double DQN, CQL).
- **Not affected:**
  - Evaluation observations, which lie in the test window and use a scale
    fitted strictly before it.
  - Obligation 3 (closed), which concerns fold-level fitting and evaluation
    immutability, not per-decision invariance.
- **Economic effect:** not established. The screen was already rejected, so no
  decision changes.

## Next step

Register a causal-scale successor, for example a scale fitted on a burn-in
segment that ends before the earliest training-episode start, or an
expanding-window fit. Compose it with the exact replay runner, then address
provider availability, which is still only an assumption. Archived code is not
edited.
