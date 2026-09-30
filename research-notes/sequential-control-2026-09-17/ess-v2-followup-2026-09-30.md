# Isolated exact ESS diagnostic — 2026-09-30

**Decision: retain a disabled research diagnostic; no candidate adoption.** This
continuation addresses numerical moment underflow in a separately versioned pure
kernel. It does not repair the frozen OPE helper, establish reliable OPE, change a
policy, rerun a market experiment or open protected data. The champion is preserved
and all 38 broader mission obligations remain open or partially verified.

Specification/registration commit: `494cdddb`, before implementation or probes.
Verification source freeze: `36fbaf33ee5fdf72ebf0694f39fac62b19808910`.
Starting head: `2f8181dac9e4cba324c30fc40c165f0a6b7297e6`; its
[CI passed](https://github.com/diegueins680/trader/actions/runs/36704069686).
Latest main remains `dbd45e2691cb37f1421a23306c43b676fa82e6fc`. Work continues on
`research/sequential-review-2026-09-28`, draft PR #284 stacked on #281. Open issues
were empty; related feature/artifact and dependency PRs do not provide this kernel.

## Canonical behavior and isolation

The [contract](../../formal/research/ess-v2-contract.md) and
[engineering registration](../registrations/ess-v2-engineering.json) define
`effective_sample_size_v2` in
[ess_rational_v2.py](../../scripts/research/ess_rational_v2.py). It requires exact
`enabled=True`, native version string `ess-rational-v2`, and a native tuple of
1..256 native finite nonnegative floats. Invalid inputs return `None` before any
conversion. Signed zero is admitted as zero. No coercion, iterator or subclass is
accepted. No existing API, CLI, artifact, identifier or environment variable changes.

After full admission, the kernel converts each weight exactly to `Fraction`,
accumulates S=sum(w) and Q=sum(w*w), then returns Fraction(0) when Q=0 or S*S/Q
otherwise. No float result conversion, clamp, approximate normalization, clipping
or reliability label is introduced. A ghost processed-count k connects the source
loop to its mathematical recurrence. No side effects or partial result are exposed.

The full module AST is bound to the checker. Only the accumulator and result
expressions are translated; their exact identities are checked in addition to
bounds. The translator respects sequential assignment order. Source mutants that
preserve bounds while changing the diagnostic formula must still be rejected.
The final checker explicitly enforces those identities; this strengthens fidelity
within the preregistered formula without changing the kernel, domain or solver budget.

Static import checks find no existing research consumer. This is engineering
isolation evidence, not a proof that arbitrary dynamic Python cannot import a
module. The new module has no exchange or authorization interface, and no Haskell
production adapter was added. It is ordinary callable offline infrastructure,
not a candidate model or integrated trading behavior.

## Verification scope

| Requirement | Status | Scope |
|---|---|---|
| F-RL-ESS-V2-ACCUMULATE | smt_verified | Source recurrence equals S'=S+w, Q'=Q+w²; zero base and inductive step preserve k>=0, S>=0, Q>=0 and Q<=S²<=kQ. |
| F-RL-ESS-V2-BOUNDS | smt_verified | Exact source result equals the specified diagnostic, is zero for Q=0 and in [1,k] for positive mass, k in [1,256]. |
| F-RL-ESS-V2-PUBLISH | model_checked | Full admission precedes arithmetic, full accumulation precedes publication, terminals are stable, progress decreases, and no persistence/order-authorizing transition exists in the finite call model. |

Three independent SAT-premise / UNSAT-violation pairs cover the two SMT IDs. The
inductive-step lemma uses real k>=0, a stronger domain than the loop's integer
count. These base/step lemmas support the recurrence argument; they are not a
machine-checked Python interpreter or Fraction implementation proof.

The model covers all sizes 1..256: **66,309 reachable states, 99,718 transitions,
maximum shortest depth 515, strict rank bound 517**. Its phases are entry, shape
admission, element validation, accumulation and absent/zero/positive return.
Terminal stuttering is permitted. Liveness assumes terminating primitives and
normal resources; no physical runtime or memory-exhaustion recovery theorem is
claimed. Numeric value invariants and publication order are separate checks.

Pinned Python 3.13.3, Z3 4.15.4, solver seed 0 and 10,000 ms/query are unchanged.
Native-type/isfinite checks, exact conversion, rational primitives, stable bindings,
AST translation and solver/runtime remain explicit A-ESS-V2 assumptions. The
[Python documentation](https://docs.python.org/3.13/library/fractions.html) specifies
exact float conversion; the [pinned CPython source](https://github.com/python/cpython/blob/v3.13.3/Lib/fractions.py)
uses the float's integer ratio. Inspection is not a formal proof of those primitives.
No third-party code or new dependency is imported into the repository.

## Conformance and the old witness

Eight new integrity tests cover 13 source mutants, registration drift, two model
bypasses, solver-unknown refusal, an unexpected consumer import, hostile/native-type
boundaries, all 256 invalid element positions, extreme values, 340 fixed grid
cases and 64 seeded cases with exact scaling, permutation and replay properties.
The reference converts float integer ratios to a common integer denominator,
then computes integer sums and moments independently of the kernel's Fraction loop.
These tests supplement the conditional SMT/model evidence.

For CE-RL-018's two supplied weights 2^-600, v2 returns exactly **2**. The old
helper still returns ESS 0 with ignored underflow and the existing fixture still
checks that behavior. Equal minimum-subnormal or maximum-finite weights also yield
their exact count in v2; mixed extremes match the integer reference. This fixes
the prescribed moment-arithmetic mechanism only in the isolated diagnostic.

If upstream cumulative products already became zero, v2 receives zero and cannot
recover the lost probability mass. Invalid/infinite inputs reject. Exact arithmetic
does not prove overlap, unbiasedness, independent observations, confidence coverage,
correct rewards, calibration or trustworthy behavior propensities. The original
108 invalid OPE batches remain invalid and untouched.

## Statistical interpretation and current literature

Elvira, Martino and Robert, **Rethinking the Effective Sample Size** (International
Statistical Review 90(3), 525–550, 2022; DOI 10.1111/insr.12500), distinguish
variance-based efficiency from the common weight-only approximation. The former
depends on the integrand; the latter can miss poor sample diversity and infinite
variance. We checked the accepted manuscript's definitions and limitations, not an
independent replication. This supports retaining the output as a diagnostic only.
[Author manuscript](https://www.pure.ed.ac.uk/ws/files/265522588/ESS_final.pdf).

Mousavi and Elvira, **Beyond Effective Sample Size: Effective Number of Proposals
for Adaptive Importance Sampling** (arXiv:2608.15154, August 2026), propose a
proposal-level redundancy diagnostic. The abstract explains that balanced sample
weights may coexist with duplicated proposal components. We verified primary
metadata/abstract only; no full proof, code, license or replication audit is claimed.
Population AIS is not the current trading OPE setup, so this is a monitored idea,
not a new implementation candidate. [Preprint](https://arxiv.org/abs/2608.15154).

This literature screening does not alter the committed preregistration or metric.
No paper PDF, third-party implementation or dataset is committed.

## Synthetic CPU benchmark and reproduction

Registered cases: sizes 1,2,200,256 × zero/minimum-subnormal/maximum-finite/mixed
weights × 25 repetitions = 400 calls. Observed maximum **8.764765 ms**, below the
registered 500 ms diagnostic ceiling. At 256 rows, median/max milliseconds were
zero 0.840395/0.996608; tiny 2.665683/4.207131; maximum 2.065195/2.909411; mixed
7.877403/8.764765. Results are from macOS Intel, one run, with no dedicated CPU
isolation; they are not policy inference benchmarks or a proved deadline. Output
numerator/denominator bit lengths are recorded, not a heap-usage measurement.
No training, learned artifact, startup/reload or service benchmark is introduced.

Use the pinned environment in the [formal runbook](../../formal/research/README.md).
Run `python scripts/formal/test_integrity.py ExactESSV2Tests`, then
`bash scripts/verify.sh formal` and `bash scripts/verify.sh full`.
The exact benchmark command follows; its output stays outside Git:

```sh
python - <<'PY' > /private/tmp/trader-essv2-benchmark.json
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, 'scripts/research')
from ess_rational_v2 import effective_sample_size_v2, VERSION
reg = json.loads(Path('research-notes/registrations/ess-v2-engineering.json').read_text())
extremes = [float.fromhex(x) for x in reg['extremeHexValues']]
rows = []
for size in reg['benchmarkLengths']:
    cases = {'zero': (0.,)*size, 'tiny': (extremes[0],)*size,
             'maximum': (extremes[-1],)*size,
             'mixed': tuple(extremes[i % 3] for i in range(size))}
    for name in reg['benchmarkScenarios']:
        weights = cases[name]
        samples = []
        for _ in range(reg['benchmarkRepetitions']):
            start = time.perf_counter_ns()
            result = effective_sample_size_v2(weights, enabled=True, version=VERSION)
            samples.append((time.perf_counter_ns()-start)/1e6)
            if result is None or not (result == 0 if name == 'zero' else 1 <= result <= size):
                raise RuntimeError('benchmark result failed')
        rows.append({'size': size, 'scenario': name, 'calls': len(samples),
                     'medianMilliseconds': statistics.median(samples), 'maximumMilliseconds': max(samples),
                     'outputNumeratorBits': result.numerator.bit_length(),
                     'outputDenominatorBits': result.denominator.bit_length()})
passed = all(r['maximumMilliseconds'] <= reg['benchmarkMaximumMilliseconds'] for r in rows)
print(json.dumps({'version': VERSION, 'python': platform.python_version(), 'machine': platform.machine(),
                  'platform': platform.platform(), 'rows': rows, 'passedEngineeringBudget': passed,
                  'maximumMillisecondsBudget': reg['benchmarkMaximumMilliseconds'],
                  'sourceSHA256': hashlib.sha256(Path('scripts/research/ess_rational_v2.py').read_bytes()).hexdigest(),
                  'scope': 'synthetic diagnostic only; no policy inference or proved wall-clock bound'}, indent=2))
if not passed:
    raise SystemExit('registered engineering budget exceeded')
PY
```

The [evidence receipt](ess-v2-evidence-receipt-2026-09-30.json) records the
verification source freeze, exact commands, exit statuses, log hashes and timings.
`bash scripts/verify.sh formal` and `bash scripts/verify.sh full` both exited 0:
96 formal integrity tests, 34 scoped SMT requirements, Haskell checks, 241 web
tests and 185 automation tests passed. The web bundle-size advisory remains;
no test or assertion was disabled. [Source-freeze GitHub CI also passed](https://github.com/diegueins680/trader/actions/runs/36710073468)
for formal, Haskell, web and automation; Docker/deployment jobs were skipped.
Subsequent changes are this documentation and its evidence receipt.
`python scripts/formal/verify.py --require-complete` remains an acceptance refusal
while broader obligations are open. A green scoped gate does not authorize use.

General recommendation: **no candidate passed**. RL recommendation: **continue
offline research** while rejecting the frozen tested configurations for integration.
The diagnostic may be retained as independently useful engineering infrastructure;
any future estimator/financial integration needs its own registration and gates.
No market/holdout, champion, live authorization, ownership, fleet, leverage, margin,
risk limit, deployment or production-learning change was made.
