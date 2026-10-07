# Split isolation closure (obligation 2) — 2026-10-06

**Obligation 2 closes for the delivered offline research path.** Counts move
from 12/25/1 to **13 scoped closures / 24 partial / 1 open**. This is a
read-only audit: no source, registration, archived result or market data changed.

## Resolved ambiguity (owner decision)

The frozen screen registers `purgeBars: 6` and `embargoBars: 6` but uses one
six-bar gap per fold. Earlier audits deliberately left this open. The owner chose
the **standard definition** (López de Prado, 2018, ch. 7):

- **Purge:** no training label window overlaps the test label window.
- **Embargo:** constrains only training rows after a fold's test window, so it
  is vacuous for these forward-only folds.

The six-bar gap is still **not** described as two additive regions. An *access*
means a label or outcome read. Observation lookback is governed by causality
obligations 1 and 17.

## What was certified

| Requirement | Status | Content |
|---|---|---|
| F-RL-SPLIT-SOURCE | exhaustively_checked | AST binding of the unchanged runner: `[:trainStop]` slices are the only training inputs; per-trial `net` reset; evaluation receives only its own split; all 5 full-panel uses in the fold loop reviewed; one hash-gated loader; Replay (6 reads, clock writes, window guard), Baselines label range and short-OPE window bound. |
| F-RL-SPLIT-REGIONS | smt_verified | 6 integer pairs: training labels ≤ trainStop−1; replay and short-OPE outcomes in [testStart+1, testStop−1] with no read at testStop; purge; vacuous embargo. Plus registered arithmetic: all 3 folds inside the 4,910-bar panel, and the last development close is 1 ms before the sealed holdout. |
| F-RL-SPLIT-CONFORMANCE | property_tested | 18 checks on a synthetic 400-bar panel: rewriting every value at or after `testStop` leaves `replay_policy` and `short_ope` unchanged, and every receipt index lies in the validation region. A leaky evaluator (stop + 3) is caught. |

Closure also requires the existing F-RL-SPLIT, F-RL-FIT-PREFIX,
F-RL-COLLECT-PREFIX and F-RL-DATA-COMPOSITION, plus the complete research-module
certificates used for obligations 3 and 5. All reproduce in the same run.

One nuance is stated rather than hidden. `short_ope` first validates that whole
series are finite, which is an accept/reject read over the full arrays. On the
hash-fixed panel its outcome is constant, so no future values reach any estimate.

## Verification

| Command | Result |
|---|---|
| `scripts/formal/test_integrity.py` | 310/310 OK |
| `scripts/formal/verify.py --record`, then plain `verify.py` | both exit 0; 13 closed / 25 unresolved |
| `node scripts/verify-formal-specs.mjs` | valid |

## Limits

A-SPLIT-ISOLATION trusts the Python interpreter, NumPy slicing, indexing and
`rng.integers` semantics, the AST translation and Z3. Production ingestion is
outside the scope. The closure reopens on any changed runner, loader, fold
design, evaluation helper or new caller of a v2 kernel. No economic claim follows.
