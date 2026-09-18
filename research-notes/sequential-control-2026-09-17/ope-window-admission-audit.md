# Short-OPE window admission audit — 2026-09-18

Baseline: merged main `6eaec0b45821072fb68db6d755855dd0a32c3d85`.
Synthetic engineering fixtures only; no historical market archive, protected
holdout, training experiment, data request or production action is used.

## Reproduced defect

`short_ope` constructed an RNG and sampled before validating its full request.
Checks in individual Replay instances only covered sampled episodes. An oversized
requested window or an unavailable unsampled symbol could therefore escape that
validation. Extra funding symbols could be ignored, and malformed scalar controls
could be coerced or fail incidentally after sampling.

The pre-repair control regression produced 23 failures and six errors across 29
cases. The series/universe regression produced 19 failures and two errors across
21 cases, including a short unsampled symbol. These counts include early-rejection
contract failures, not only cases that returned estimates. They do not establish
that the registered historical run used malformed requests.

## Repair and contract

A small admission helper runs before RNG construction or policy inference:

- Require genuine Python or NumPy integer controls, excluding booleans, arrays,
  strings and floating-point representations. Convert to Python integers before
  multiplication/subtraction, avoiding NumPy fixed-width arithmetic overflow.
- Require horizon 1, 3 or 6, a nonnegative seed and a positive episode budget.
- Require `24 <= start < stop - 6*horizon`, preserving a completed-feature prefix
  and room for six decisions with outcomes below the exclusive stop boundary.
- Require price/funding dictionaries with the same nonempty set of nonempty string
  symbols and aligned one-dimensional real unmasked arrays for every symbol.
- Require every array to cover the entire requested exclusive stop, even when a
  particular symbol or late episode would not be sampled.

Admission inspects only representation, shape and length. It does not scan market
values after a decision or inspect unrelated values outside the requested window.
Replay continues to validate actual values when consumed. This does not verify
scale-training provenance, point-in-time data identity or dataset authenticity.

Valid native-integer requests retain the original sampling, action sequence,
rewards, terminal accounting, failure handling and estimate schema. Accepted NumPy
integers normalize explicitly. No episode is imputed, silently omitted or retried.
The helper does not register an experiment budget or impose a new arbitrary maximum;
registered budget enforcement remains a separate runner/research responsibility.
No dependency, configuration, model identifier or production path changes.

## Executable evidence

Four new methods bring the research suite to 98:

- Twenty-nine invalid control cases reject before RNG construction or policy calls.
- Twenty-one invalid series/universe cases reject at the same boundary, including
  full-window coverage for an otherwise unsampled symbol.
- Python, signed NumPy and unsigned NumPy integer requests yield identical complete
  result dictionaries. Replacing unused history/future values with NaN does not
  change the fixture result.
- The smallest valid window completes six decisions at each horizon 1/3/6, with
  both Replay instances exactly inside the admitted boundary.

Existing valid action-trace SHA-256 remains
`f104f3cd18e1919ded7ca5bbb242e70b046491d5b7e59d3a24b4e0af64db4e4a`.
Failed-episode, policy-admission, analytical OPE, runner/export and exact legacy
report tests remain applicable. No generated dataset or artifact is committed.
These are engineering fixtures, not new financial research trials.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k short_ope
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

Run the rejection tests with the baseline evaluation module to reproduce the
admission failures. The PR records actual wrapper results and final-head CI.
`A-SEQUENTIAL-RESEARCH-R21` links four witnesses. Haskell/Markdown risk mitigations
remain synchronized; canonical `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, prior seeds/results, costs, statistical
diagnostics, OPE/support limitations, protected holdouts and deployed release.
Continue offline research without adoption. No live exploration, order authority,
production setting change or deployment is introduced.
