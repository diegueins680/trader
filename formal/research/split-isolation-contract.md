# Split isolation (obligation 2)

Engineering preregistration, 2026-10-06; base 5a5ba6245ba3ab13a813bb1a9d7ac5618ecbd647.
Read-only audit: no source, registration, archived result or data changes.

## Resolved interpretation (owner decision, 2026-10-06)

The frozen screen registers `purgeBars: 6` and `embargoBars: 6`, while each fold
uses one six-bar gap (`testStart = trainStop + 6`). Earlier audits deliberately
left the relation open. It is now resolved explicitly by the repository owner as
the **standard definition** (López de Prado, *Advances in Financial Machine
Learning*, 2018, ch. 7):

- **Purge:** no training observation may have a label window that overlaps the
  test label window.
- **Embargo:** only training observations that come **after** a fold's test
  window are dropped. Every registered fold trains strictly before its test
  window, so the embargo is vacuous here.

The six-bar gap is therefore **not** described as two additive regions. That
statement from the 2026-09-28 audit stands. Under the standard definition no
additive gap is required.

**Access** means a label or outcome read: a price or funding value that
determines a training label, reward or realized evaluation outcome. Observation
lookback into earlier bars (`market_features` reads `[t−24, t]`) is past
information, governed by causality obligations 1 and 17, and is not a split
access.

## Regions

For fold (trainStop, testStart, testStop) over the registered 4,910-bar panel:
training labels lie in [0, trainStop); validation (test) outcomes lie in
[testStart + 1, testStop − 1]. There is no calibration region (no inner
selection), and the final holdout is not part of the panel: the last development
close (1742227199999) precedes `holdoutStartOpenTime` (1742227200000).

## Obligations / methods

F-RL-SPLIT-SOURCE (exhaustively checked, AST source binding of unchanged code):

- **Fold loop:** `train`/`funds` are exactly the `[:trainStop]` slices. `Scale.fit`,
  `Baselines`, `train_ppo` and `train_q` receive only those slices, the fitted
  scale, horizon, seed and constants. `net` is reset per trial and assigned only
  from that fold's training call. `short_ope` and `replay_policy` receive only
  that same split's `testStart`/`testStop`.
- **Market data:** `load_development` is the only market-data reader, and it is
  hash-gated to the registered files.
- **Helper index expressions:** the bound expressions in `Replay` (constructor
  guard, `end`, outcome reads), `Baselines` (label index and range) and
  `short_ope` (start sampling and episode stop).

F-RL-SPLIT-REGIONS (SMT, integers): for those bound expressions,

- every training label index is ≤ trainStop − 1;
- every `replay_policy` outcome index is in [testStart + 1, testStop − 1];
- every `short_ope` outcome index is in [testStart + 1, testStop − 1];
- purge holds (training label windows end before the first test outcome);
- embargo is vacuous (trainStop ≤ testStart);
- each registered fold lies within the panel;
- the panel ends before the holdout.

F-RL-SPLIT-CONFORMANCE (property tested): the actual helpers run on a
synthetic 400-bar panel. Every replay receipt's outcome index lies in its
region, and rewriting all values at or after `testStop` with finite numbers
leaves `replay_policy` and `short_ope` results unchanged.

Admission validation: `short_ope` first runs `_admit_ope_window`, which checks
that each whole series is real and finite before any episode. This is an
accept/reject read over the full arrays, not a label read. On the frozen run the
panel is fixed by its registered hash, so the check's outcome is a constant and
no information about values at or after `testStop` reaches any estimate. The
conformance probe rewrites those values with finite numbers and requires
identical results.

Closure of obligation 2 requires F-RL-SPLIT-SOURCE and F-RL-SPLIT-REGIONS plus the existing
F-RL-SPLIT, F-RL-FIT-PREFIX, F-RL-COLLECT-PREFIX and F-RL-DATA-COMPOSITION, and
the complete research-module source certificates used by the obligation 3/5
closures. Disconnected v2 kernels take caller-supplied arrays, have no region
access of their own and have no caller. A-SPLIT-ISOLATION trusts the Python
interpreter, NumPy slicing/indexing semantics, the AST translation and Z3.
Production ingestion is outside the scope and is not claimed.

## Closure certificates versus tested evidence (review correction, 2026-10-06)

The conformance requirement is reproduced on every formal run as supporting
tested evidence. It is **not** a closure certificate. The verifier admits only
proof-class statuses (proved, model/SMT/refinement-verified, exhaustively
checked) as closure certificates, because tests are not proofs. Earlier wording
said closure required all three certificates. That is corrected here, and the
ledger was already consistent with this.
