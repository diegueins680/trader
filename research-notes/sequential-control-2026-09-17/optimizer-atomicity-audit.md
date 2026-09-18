# Offline optimizer atomicity audit — 2026-09-18

Baseline: merged/deployed main `0f551a86bbc8e213e3fe1ec6981e8b9a01a8af45`.
Synthetic engineering fixtures only; no market archive, protected holdout,
registered market experiment or persisted trained policy is read or produced.

## Reproduced defects

Finite gradients are not enough to make clipping safe. With two zero observations
and finite output derivatives of `1e200`, all gradient arrays were finite, but
squaring them overflowed the global norm to infinity. Division by that norm made
all clipped gradients zero. Adam silently advanced its counter with no parameter
change. This could misrepresent a failed numerical update as completed training.

Adam also published moments and parameters one array at a time before checking
all parameters. A NaN learning rate or a failure at the last parameter could leave
the counter, earlier parameters and moments mutated after raising an exception.

The four added test methods produced 22 assertion failures against the baseline:
four actor/critic cold/warm norm cases, 15 invalid-control cases and three late
failure/partial-mutation cases. A non-finite gradient already rejected without
mutation, and the valid-update golden test already passed. This reproduces a
boundary defect, not evidence that the historical experiments suffered it.

## Repair and limitations

Require a positive finite real scalar learning rate and a nonnegative integer
counter, excluding booleans, before computing gradients. Normalize the counter
to a Python integer before incrementing. Reject floating-point overflow, invalid
arithmetic, non-finite gradients, non-finite norms and non-finite candidate arrays.
Do not turn an overflowing norm into a zero update or retry with changed tuning.

Compute all candidate parameters and Adam first/second moments locally. Publish
all three dictionaries and the incremented counter only after all checks pass.
Failure leaves existing state unchanged, even when that state was already
corrupted. It does not restore or sanitize a corrupt checkpoint. Existing runner
failure handling retains the failed fit rather than treating it as a valid model.

Normal gradient clipping, Adam coefficients, bias correction, arithmetic order,
learning rates used by the registered algorithms and policy artifact format are
unchanged. This boundary does not comprehensively validate input/state shapes,
masked representations or every hidden gradient intermediate; it does not certify
convergence, provenance, numerical conditioning, preemptive timeouts or economic
value. It is an offline optimizer change with no production inference integration.

## Executable evidence

Four new methods bring the research suite to 114 tests:

- Actor and critic networks reject finite-gradient norm overflow without changing
  cold or previously updated state, including under a caller's ignore policy.
- Invalid learning rates and counters reject before gradient computation.
- Late moment overflow, negative variance, NaN parameters and NaN gradients leave
  every parameter, moment and counter unchanged; the caller error policy restores.
- Three normal updates match stored pre-repair parameters and both moments at
  rtol `1e-13`, atol `1e-15`. The small deterministic fixture identifies its source
  commit. Floating-point bytes are not used as a cross-platform contract.

Existing finite-difference gradient, multi-seed synthetic training, inference,
serialization, replay, OPE and report fixtures remain part of verification.

```bash
python3 -m unittest discover -s test -p sequential_screen_test.py -k optimizer_
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

A local 1,000-update measurement (Network(11), four synthetic rows, same fixed
input/derivatives) on the shared macOS Intel host measured before/after median
0.13656/0.17508 ms, p99 0.24547/0.27792 ms and maximum 0.40709/1.21089 ms.
This is an optimizer microbenchmark, not end-to-end training, production inference
or a resource/timeout guarantee. No inference budget or acceptance gate changes.
External logs and benchmark output are retained outside Git; actual wrapper and
CI results are recorded in the PR. The only added fixture is small and synthetic.

## Decision and contracts

`A-SEQUENTIAL-RESEARCH-R25` links four executable witnesses. Haskell and Markdown
risk mitigations are synchronized; canonical `RL-OFFLINE-001` stays HIGH/OPEN.
README, CHANGELOG and reproduction instructions describe the failure contract.

No candidate passed. Preserve the current champion, all prior economic results,
failed seeds, OPE/statistical limitations and protected confirmation boundaries.
No new model, dependency, configuration, live exploration, order authorization or
automatic promotion is introduced. Deployment of this research-code repair does
not promote a research policy. Existing live execution settings must be preserved;
they are not represented as globally disabled.
