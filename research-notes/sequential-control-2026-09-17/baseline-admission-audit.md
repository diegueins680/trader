# Baseline input-admission audit — 2026-09-18

Baseline: merged main `428e4a6b91ca319e1621516a9f7ccf444a0938c3`.
Only deterministic synthetic engineering fixtures are used. No market archive,
protected holdout, policy-training experiment or production action is involved.

## Reproduced defect

RL inference already rejected invalid observations with an absent proposal. The
offline baseline interface lacked that admission check. Constant and behavior
rules could emit actions without a valid observation; other branches could return
forecasts, apply argmax to non-finite values or raise representation errors.
Unknown action or forecast names silently fell through to the ridge forecast.
A misspelled name could therefore evaluate a different rule instead of failing.

The pre-repair input/name regressions reported 141 assertion failures and 34
errors. The invalid-observation matrix covers ten representations across all
twelve action rules and all five forecasting rules. Separate name cases include
a misspelling, empty name, null and integer; nonforecast action rules are also
rejected at the forecast interface. This demonstrates an interface defect, not
that any historical runner invocation used these inputs. The existing replay
shield already rejects an absent action or invalid replay observation.

## Repair

Both entry points require a recognized string name and an unmasked one-dimensional
NumPy array of exactly twelve finite real numbers. Boolean, complex and object
arrays, lists, missing values, wrong shapes and non-finite observations return
`None` before computation or random sampling. Forecasts are admitted only for
historical mean, last return, momentum, reversal and ridge optimizer.

An absent proposal is not a request to trade to zero. When it reaches replay,
the independent shield stops the path with `invalid_action`, zero observations,
zero fills and unchanged equity. Unsupported market-feature states retain the
existing separately defined neutral-target behavior in the replay driver.

Valid inputs use the original formulas, tie-breaking and random generator. No
training logic, action space, execution rule, reward, identifier, configuration,
dependency or production boundary changes. This is admission of observations and
names; it does not validate all fitted parameters, guarantee execution time or
complete the simulator/OPE audit. No statistical or economic improvement is claimed.

## Executable evidence

Four new test methods bring the research suite to 82:

- Invalid-observation matrix with RNG-state preservation for rejected actions.
- Unknown/nonforecast names return absence without consuming randomness.
- Actual replay rejects an absent baseline action without any fill.
- All twelve action rules and five forecasts retain exact outputs over four valid
  observations with a fixed behavior seed.

The compatibility fixture uses exact binary-representable inputs and coefficients,
not fitted market data. Its sorted JSON output SHA-256, captured from pre-repair
code and checked after repair, is
`1866180923c03a7716705d8dc021ccbac977c77992586f63acf697f5ffc2dc35`.
The legacy report-byte regression remains unchanged. No generated artifact is
committed. These are engineering fixtures, not additional research trials.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k baseline
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

To reproduce the rejection failures, run the new input/name tests with the
pre-repair evaluation module in an isolated checkout. The PR records the actual
full-wrapper result and final-head CI. `A-SEQUENTIAL-RESEARCH-R17` links all four
witnesses. Haskell/Markdown risk mitigations remain synchronized, while canonical
`RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, historical results, costs, seeds,
OPE/support limitations, statistical conclusions and protected holdouts. Continue
offline research without adoption. No live exploration, direct order authority,
production setting change or deployment is introduced.
