# Economic completion audit — 2026-09-18

Baseline: merged main `d6ae3404ab34b0de48a09ac45566917243e3de52`.
Only synthetic engineering fixtures are used. No market archive, protected
holdout, training experiment, external data or production action is involved.

## Reproduced defect

`economic()` used the presence of a failure reason to choose complete/failed
status, without requiring replay to have stopped. Calling it after one valid step
could publish a complete result while inventory was still open and the episode
was unfinished. An unstarted replay returned a failed row with no failure reason.
Inconsistent terminal state was also ignored by the report.

The two pre-repair rejection methods failed all nineteen cases: two unfinished
states and seventeen completion/reason mutations. Mutations include nonboolean
termination markers, retained/non-finite inventory, a pending action, an early
outcome index, missing rows, invalid/mismatched final equity and invalid failure
reasons. This does not show that historical runner calls reported unfinished
paths; the standard replay driver already waits until its loop stops.

## Repair and compatibility

Require `done is True` before computing any metrics. If a failure reason is present,
require a nonblank string distinct from the reserved completion label. A successful
path additionally requires:

- The final outcome index, `stop - 1`.
- Finite zero inventory and no pending action.
- Finite positive equity equal to the final ledger row's equity.
- Exactly `stop - start - 1` recorded outcome bars.

Failed stopped paths retain their existing reporting behavior. A rejected initial
proposal still reports zero observations with an explicit failure. A market-data
failure after a valid fill still reports the recorded stopped loss and failure,
even if inventory remains open. A terminal capital-floor failure remains failed
with its loss and liquidation accounted for. The guard does not fabricate a fill,
reclassify failure as success or discard an adverse path.

The existing metric formulas and aggregate ledger reconciliation are unchanged.
Successful valid report bytes remain identical. There is no new schema, model,
configuration, dependency, training rule or production path. This checks completion
state, not every ledger field, the truth of source data or recovery/liquidation
value of inventory on failed paths. It does not establish economic superiority.

## Executable evidence

Four added methods bring the research suite to 102:

- Unstarted and active inventory-bearing replays reject before metric reporting.
- Seventeen inconsistent successful/reason states reject explicitly.
- Initial action rejection, partial invalid-market transition and capital-floor
  failure retain their failure reasons, observations and recorded losses.
- Completed cash paths at horizons 1/3/6 retain exact report bytes.

The completed-report fixture SHA-256, captured on pre-repair code and checked after
repair, is `e4c2a5f1da24364904426b42b94817524b42aa0ef2bde1f3bde31b707c7df6bb`.
Existing fee/funding/terminal, replay, runner/export, OPE and exact legacy-report
checks remain applicable. No generated artifact is committed; these are engineering
fixtures, not new financial trials or out-of-sample evidence.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k economic
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The new rejection tests with the baseline evaluation module reproduce the defect.
The PR records actual verification results and final-head CI.
`A-SEQUENTIAL-RESEARCH-R22` links four witnesses; Haskell/Markdown risk mitigations
remain synchronized and canonical `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, prior seeds/results, costs, statistical
diagnostics, OPE/support limitations, protected holdouts and deployed release.
Continue offline research without adoption. No live exploration, order authority,
production setting change or deployment is introduced.
