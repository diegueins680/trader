# Short-OPE policy admission audit — 2026-09-18

Baseline: merged main `4bbd17c25343325499144da5096c648c6232efa1`.
Synthetic engineering fixtures only; no market archive, protected holdout, trained
policy experiment, external data or production action is used.

## Reproduced defect

Ordinary RL inference checks observations and outputs, but `short_ope` called
`net.forward` directly. It could apply argmax or extract value estimates from
malformed outputs before an explicit admission check. The direct-policy path in
particular could turn a malformed output into a valid action index without that
output reaching the estimator's later Q/V validation.

The two new rejection methods produced 23 assertion failures and two errors on
baseline. They cover ten output representations on both logged and direct paths,
and five invalid observations. Some malformed values failed later or with an
incidental error; others produced estimates. This does not establish that the
original historical OPE calls encountered malformed outputs.

## Repair and compatibility

Both paths use one helper before selection/transition. Every observation sent to
the policy must be an unmasked one-dimensional 12-element finite real NumPy array.
Every returned value vector must be an unmasked one-dimensional three-element
finite real NumPy array. Reject missing, boolean, complex, object, masked,
wrong-shaped and non-finite representations without coercion.

Invalid observations raise before forward inference. Invalid outputs raise before
the affected replay step. A failure on the first direct-policy call may follow
six completed logged-behavior steps; those earlier steps do not become a usable
estimate. The whole OPE call fails. There is no imputation, cash substitution,
clipping, failed-episode exclusion or retry. Existing forward exceptions remain
explicit, and the runner's OPE exception boundary records them as failures.

Valid calls retain their action space, argmax tie-breaking, behavior RNG, reward,
terminal accounting and unsupported-feature fallback. The logged path still needs
policy values for its control variate even when its target falls back to cash;
the direct unsupported-state branch retains its existing no-inference behavior.
No learning, identifier, schema, dependency, configuration or production path changes.

Finite outputs do not calibrate PPO logits as Q values, resolve state-action
support, establish estimator reliability or provide an inference-time guarantee.
Those simulator/OPE limits and the no-adoption conclusion remain in force.

## Executable evidence

Five new methods bring the research suite to 94:

- Twenty output/phase cases reject before the affected step: zero steps for a
  first logged-path failure and six prior logged steps for a first direct failure.
- Five invalid observations reject before any forward call or transition.
- A valid cash-target policy retains the original twelve-step action trace, a
  zero direct simulator value and explicitly unreliable OPE classification.
- A forward exception propagates explicitly without an estimate.
- The original synthetic seed-11 path retains its explicit failed-episode outcome;
  seed 0 supplies the separate successful-path fixture.

The initial successful-path test incorrectly expected twelve steps from seed 11;
that sequence actually triggers the existing turnover limit and produced eight
steps across the two paths. Its Python-suite and targeted-automation failures led to the separate
failure-accounting regression above, without weakening the limit. These are
engineering-fixture seeds, not selected or omitted financial research trials.

The deterministic successful action-trace SHA-256, captured from baseline code and
checked after repair, is
`f104f3cd18e1919ded7ca5bbb242e70b046491d5b7e59d3a24b4e0af64db4e4a`.
Existing analytical OPE, failed-OPE runner/export, exact baseline-output and legacy
report-byte tests remain applicable. No generated artifact is committed. These are
engineering fixtures, not new financial experiments or generalization evidence.

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k short_ope
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

Run the new rejection tests with the baseline evaluation module to reproduce the
defect. The PR records actual wrapper results and final-head CI.
`A-SEQUENTIAL-RESEARCH-R20` links five witnesses; Haskell/Markdown risk mitigations
remain synchronized and canonical `RL-OFFLINE-001` stays HIGH/OPEN.

## Decision

No candidate passed. Preserve the champion, prior seeds/results, costs, statistical
diagnostics, OPE/support limitations, protected holdouts and deployed release.
Continue offline research without adoption. No live exploration, direct order
authority, production setting change or deployment is introduced.
