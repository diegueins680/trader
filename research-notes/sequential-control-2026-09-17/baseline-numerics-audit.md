# Baseline numerical-admission audit — 2026-09-18

Baseline: merged main `b0432f66066506be52088a64a297f41cfaed27ed`.
Synthetic engineering fixtures only; no market archive, protected holdout, fitted
market experiment, policy training or production action is used.

## Reproduced defect

Observation/name admission did not validate fitted parameters or intermediate
arithmetic. Non-finite bandit scores could enter `argmax`; infinite logistic scores
were clipped to finite values before thresholding. Corrupted coefficients and
behavior-clone actions could produce actionable outputs or raise exceptions.
Even finite coefficients could overflow during dot products or feature restoration.

The forty malformed-parameter cases produced 26 assertion failures and 13 errors
on baseline; an absent clone action already returned absence. Six overflow cases
also failed: ridge, logistic, bandit and three feature-based forecasts. This is an
interface/numerics defect; it does not show that historical fitted controls had
invalid coefficients or that previous research conclusions were wrong.

## Repair and limits

Validate only fitted parameters used by the selected rule. Ridge/logistic require
seven finite real coefficients; the contextual bandit requires a finite real
13-by-3 coefficient matrix. Reject masked, boolean, complex, object, missing and
wrong-shaped parameter representations. Historical mean must be a finite real
scalar; the behavior clone must hold one of the three allowed real actions.

Forecast/action arithmetic runs with overflow, invalid arithmetic and division
errors rejected. Explicit finite checks precede logistic clipping, bandit argmax
and utility selection, covering operations that return non-finite values without
raising a NumPy floating-point exception. Public forecasts are finite or absent;
actions are allowed real targets or absent. Expected numerical/type/shape failures
return absence rather than escaping into the runner's generic replay-error path.
Replay shielding rejects absence without a fill; absence does not mean flatten.

Only relevant components are evaluated. Constant and uniform rules remain usable
when unrelated fitted components are invalid. The formulas, cash-first utility
ties, valid behavior RNG sequence and clipping of large finite logits are retained.
No new rule, model identifier, artifact schema, dependency, configuration or
production path is added.

Numerical admission does not establish coefficient provenance, training-data
causality, calibration, forecast value, market generalization or an inference-time
guarantee. It does not repair corrupted fits or automatically retrain them.

## Executable evidence

Four added methods bring the research suite to 89:

- Forty malformed parameter cases return absence without consuming randomness.
- Six finite-input overflow cases return absence before selection/clipping.
- A replay with corrupted logistic coefficients has zero observations, zero fills,
  unchanged equity and an explicit `invalid_action` failure.
- Independent fixed/uniform rules, a large finite logistic score and valid NumPy
  scalar parameters retain their expected behavior and RNG state.

The existing exact valid-output fixture still covers twelve action rules and five
forecasts over four observations. Its SHA-256 remains
`1866180923c03a7716705d8dc021ccbac977c77992586f63acf697f5ffc2dc35`.
Existing observation, OPE, policy and exact legacy-report tests remain applicable.
These are synthetic engineering checks, not new financial trials or benchmarks.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k baseline
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records actual wrapper results and final-head CI. The pre-repair evaluation
module with the new rejection tests reproduces the defect. Formal contract
`A-SEQUENTIAL-RESEARCH-R19` links four witnesses; Haskell/Markdown risk mitigations
remain synchronized and canonical `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, prior results/seeds, costs, statistical
diagnostics, OPE/support limitations, protected holdouts and deployed release.
Continue offline research without adoption. No live exploration, order authority,
production setting change or deployment is introduced.
