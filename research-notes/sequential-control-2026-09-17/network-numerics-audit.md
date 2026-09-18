# Neural forward arithmetic audit — 2026-09-18

Baseline: merged/deployed main `3edef3deadefc88ec0997d2fb9ee0e58612a4629`.
Synthetic engineering fixtures only. No market archive, protected holdout,
training experiment, saved checkpoint or production trading action is involved.

## Reproduced defect

`Network.forward` evaluated `tanh(x @ w1 + b1)` without checking its preactivation.
An overflow to infinity or an infinite hidden bias could become a finite saturated
activation. Final output checks could then accept finite scores and return a
directional proposal. Finite observation and final-output checks alone were not
sufficient. OPE could similarly consume the apparently finite scores.

After capturing the valid baseline hash, the four-method pre-repair run produced
28 assertion failures: 24 forward cases (six arithmetic faults × actor/critic ×
vector/batch), three inference cases, and one OPE case. This demonstrates an
arithmetic boundary defect, not that historical fits contained these faults.

## Repair and limits

Wrap hidden/output arithmetic in local NumPy overflow/invalid/divide error checks.
Require finite hidden preactivations *before* tanh and finite output values before
return. Floating-point errors become explicit ValueError failures. The local
error policy overrides a caller's ignore setting and is restored on exit.

The existing inference adapter catches failure and returns an absent proposal;
the independent shield rejects it without a fill. OPE and training propagate the
failure into existing runner failure accounting. Failed evaluations are not
converted to cash or silently omitted. Finite large preactivations may still
saturate: mathematical saturation itself is not an error.

Normal arithmetic order, architecture, output shape, model identifiers and policy
persistence are unchanged. Actor and critic evaluation both use this boundary.
This is not a complete validation of parameter representations, gradients,
optimizer state, provenance, numerical conditioning or preemptive timeouts.
No inference path or policy acquires order authority.

## Executable evidence

Four added methods bring the research suite to 110 tests:

- Hidden multiplication/addition overflow, explicit hidden NaN/infinity, output
  multiplication overflow and explicit infinite output reject in actor/critic
  vector and batch evaluation.
- Invalid hidden arithmetic produces absence, finite elapsed timing and no fill.
- OPE rejects an infinite hidden state before the first simulated transition.
- Seeds 11/23/47, one/three-output networks, vector/batch calls and ordinary/large
  finite inputs retain exact forward bytes.

The valid-forward SHA-256 captured before repair is
`1ca2422f7c23b9247715afc33d4d3d5cd5903cb28c41b4769b1460931f8d1d85`.
Existing numerical-gradient, actor/critic, multi-seed synthetic training,
serialization, OPE and reporting fixtures also pass.

```bash
python3 -m unittest discover -s test -p sequential_screen_test.py -k network_
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

A local 1,000-call forward-only measurement used Network(11) and
`np.arange(12)/12`. Before/after median latency was 0.00947/0.03135 ms, p99 was
0.04946/0.34695 ms and maximum was 0.11776/2.92346 ms. This shared macOS Intel host
measurement is not an isolated deployment benchmark or a timeout guarantee;
the observed maximum exceeds the prior 2 ms target. Do not use the median or p99
to claim that every invocation meets that target. No budget is relaxed.

`A-SEQUENTIAL-RESEARCH-R24` links four witnesses. Haskell/Markdown mitigations are
synchronized; canonical `RL-OFFLINE-001` remains HIGH/OPEN. README, CHANGELOG and
reproduction instructions describe the guard. The PR records actual wrapper,
CI and deployment results. Generated fixtures/logs remain outside Git.

## Decision

No candidate passed. Preserve the champion, prior economic results, all seeds,
statistical/OPE limitations and protected confirmation boundaries. This improves
research failure handling only. No new model, dependency, configuration,
production integration, live exploration or automatic policy promotion.
