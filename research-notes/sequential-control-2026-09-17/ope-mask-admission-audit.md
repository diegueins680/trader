# OPE mask-admission audit — 2026-09-18

Baseline: merged main `a56cbe9d5e249fb74880cc9b22ac0423557495a1`.
Synthetic engineering fixtures only. No historical archive, market-data request,
protected holdout, policy training experiment or production action is used.

## Reproduced defect

`ope_estimates` converted its six trajectory/value inputs with `np.asarray` before
checking shapes, types, finite values and probability/action domains. That
conversion strips a NumPy masked array's mask. Consequently, hidden backing values
could enter OPE as observed rewards, actions, behavior/target probabilities or
Q/V estimates. A hash or finite underlying value does not restore missing evidence.

All eighteen pre-repair regression cases were accepted instead of rejected:
partial, full and all-false masks for each of the six inputs. This demonstrates an
estimator admission defect, not that historical runner inputs were masked. The
historical OPE results remain invalid and no promotion conclusion changes.

## Repair and limits

Inspect the six supplied inputs before any array coercion. A NumPy masked array
raises `ValueError: masked OPE input`, including all-false masks. This representation
rule is consistent with the existing environment, scaler, policy and baseline
boundaries. No mask is stripped, missing value filled or trajectory excluded.
The caller's data and mask are not mutated.

Ordinary supported NumPy arrays, lists and tuples retain the existing conversion,
shape/domain checks, formulas, bootstrap seed and output schema. Read-only arrays
remain valid. The runner's existing OPE exception boundary records rejection as
an explicit failed OPE outcome, retains the completed fit and allows the compact
report to expose that failure. It does not relabel the fit or fabricate an estimate.

The patch adds input admission only. It does not establish logged-policy support,
correctness of model-based control variates, estimator reliability, adequate ESS,
market-generalization uncertainty or economic advantage. It neither changes
weights nor clips them. No new model, dependency, configuration or production
service is introduced.

## Executable evidence

Three added methods bring the sequential research suite to 85:

- Eighteen mask/field combinations reject before `np.asarray` is called; original
  data and masks remain unchanged.
- Ordinary arrays, lists, tuples and read-only arrays give exactly equal complete
  result dictionaries, including bootstrap intervals. The known-policy fixture
  has IS, PDIS, WIS and DR equal to 5 and effective sample size equal to 2.
- A synthetic runner invokes the actual estimator with masked rewards. The OPE
  failure survives archive/export reconciliation, its fit has exactly one complete
  terminal record, and report promotion remains false.

Existing enumerated-tree OPE, zero-support, overflow, ledger, exact baseline-output
and exact legacy-report tests remain applicable. No artifact or dataset is added.
These tests are engineering fixtures, not financial research trials.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k ope
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

Use the new rejection fixture with the baseline evaluation module to reproduce the
defect. The PR records actual canonical verification and final-head CI.
`A-SEQUENTIAL-RESEARCH-R18` links three witnesses. Haskell/Markdown risk mitigations
remain synchronized; canonical `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, prior seeds/results, cost assumptions,
statistical diagnostics, OPE/support limitations, protected holdouts and deployed
release. Continue offline research without adoption. No live exploration, direct
order authority, production setting change or deployment is introduced.
