# Independent numerical proof queries — 2026-09-28

**Retain a verification-tooling improvement; no trading candidate adoption.**
The change separates premise/non-vacuity and universal violation queries in two
existing numerical proof drivers. Trading code, training code, gae-targets-v2,
formulas, arithmetic domains, rounding, solver seed and 10-second per-query limit
are unchanged. No market data, financial fit, OPE, holdout or live endpoint is used.

Latest main was rechecked at `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.
Specification and engineering registration commit `e51cccde` precedes implementation
freeze `525f9b82`. This remains draft #284 on `research/sequential-review-2026-09-28`,
stacked on #281. CI at the preceding receipt `449d39fd` passed all four check jobs
([run 36458020460](https://github.com/diegueins680/trader/actions/runs/36458020460)).
No new dependency or formal tool is introduced.

## Problem, consistency and refinement

Prior local verification sometimes returned unknown/canceled on valid binary64
queries, including an all-zero premise witness. Those failures are preserved in
[the target-v2 receipt](target-v2-followup-2026-09-28.md). A passing retry is not a
certificate for the failed attempt and supplies no timing guarantee.

The [contract](../../formal/research/query-isolation-contract.md) and
[registration](../registrations/numerical-query-isolation-engineering.json) specify:

- A fresh query checks SAT(P AND W), or SAT(P) when no witness is supplied.
- A different fresh query checks UNSAT(P AND NOT C), without witness constraints.
- Only the pair SAT/UNSAT succeeds. Invalid premises, counterexamples, unknown,
  canceled results and exceptions fail closed. There is no automatic retry.

The old witness-removal requirement describes absence from the universal query;
it does not require push/pop. Independent contexts preserve that semantics and
allow each query to start with all its assertions. The source translations,
source-skeleton checks and mathematical expressions are unchanged. The entire
reproduced `formal/research/results.json` equals the prior receipt after removing
only its `sourceHashes` field. No new theorem or broader proof coverage is claimed.

The [official Z3Py introduction](https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/)
and [solver API](https://z3prover.github.io/api/html/classz3py_1_1_solver.html),
reviewed 2026-09-28, describe assertion scopes, solver instances and result handling.
These are tool-semantics references, not financial evidence. Z3/runtime/compiler
trust remains assumption A-SOLVER. No independently checked solver certificate or
universal Python refinement is added.

`F-RL-INTEGRITY` maps the two changed drivers to their actual implementation tests,
canonical clause, proof ledger, source lock and `bash scripts/verify.sh formal`.
Its status remains `property_tested`. The 22 existing SMT requirements keep their
exact domains and outcomes; 38 broader mission obligations remain open/partial.
`RL-OFFLINE-001` remains HIGH/OPEN, with verification timing and trusted solver
semantics explicit. CE-RL-010/011 remain defects of the unchanged frozen learner.

## Conformance and complete failure table

Three added tests bring the integrity suite from 49 to 52. Recording solver doubles
exercise all 18 SAT/UNSAT/UNKNOWN pairs across the two actual helpers. They assert
exact formulas, distinct solver objects, one check per solver, no witness leakage,
unchanged timeout/seed and refusal before creating query B when A is not SAT.
Four injected check exceptions propagate; unknown reasons and SAT counterexample
diagnostics are retained. Actual Z3 checks reject a conclusion valid only at the
witness, false premises and inconsistent witnesses, while proving a simple real
implication. Existing binary64 proofs, source mutants and numerical counterexamples
remain covered. This finite protocol exercise is conformance evidence, not a
universal proof of the checker or of future verification completion time.

Existing bounds remain explicit in the [certificate](../../formal/research/results.json):
lifecycle 75 states/349 transitions, two callers, depth 6; artifact path 27 states/
40 transitions, depth 13; target-batch publication 33,410 states/66,562 transitions,
1..256 rows, depth/progress bound 258. Compiled Haskell conformance covers 20,480
cases; replay conformance covers 180 exact-rational traces. These remain separate
abstractions and conformance checks, not a composed production-system proof.

## Registered engineering comparison

All 12 attempts are retained in [the small engineering result registry](query-isolation-engineering-results-2026-09-28.json).
Three rounds use old/new, new/old, old/new order. Each variant runs the existing
terminal and target-v2 checks; the latter includes its existing bounded batch model.
Old helpers are extracted from `449d39fd`; only the proof-driver globals differ.
Successful payloads must equal the frozen certificate. No hyperparameter search,
retry, omitted attempt or financial experiment occurred.

| Round | Driver | Case | Outcome | Seconds |
| ---: | --- | --- | --- | ---: |
| 1 | old | terminal | pass | 4.722186 |
| 1 | old | target | failure: F-RL-TARGET-V2-FINITE: premise witness failed: unknown | 11.833255 |
| 1 | new | terminal | pass | 2.348854 |
| 1 | new | target | pass | 0.415048 |
| 2 | new | terminal | pass | 2.133847 |
| 2 | new | target | pass | 0.403197 |
| 2 | old | terminal | failure: F-RL-TERMINAL-FP-MASK: violating terminal claim: canceled | 10.774829 |
| 2 | old | target | failure: F-RL-TARGET-V2-FINITE: premise witness failed: unknown | 12.301531 |
| 3 | old | terminal | pass | 10.240459 |
| 3 | old | target | failure: F-RL-TARGET-V2-FINITE: premise witness failed: unknown | 12.019642 |
| 3 | new | terminal | pass | 3.563919 |
| 3 | new | target | pass | 0.249634 |

Revised queries passed 6/6; old queries passed 2/6. Four old failures were
unknown/canceled under the unchanged limit, not counterexamples. All successful
payloads matched. Per-case elapsed time includes several queries and may exceed
10 seconds even on a pass; 10 seconds is the limit for each solver check.
Measurements used Python 3.13.3, Z3 4.15.4 (distribution 4.15.4.0), Darwin x86_64,
and process-local numeric-library thread limits. End-of-run host load was 41.348
on 16 logical CPUs. The small interleaved comparison supports retaining the simpler
independent contexts; it does not establish a causal explanation for all prior
failures, statistical reliability, hard deadlines or general solver superiority.

## Reproduction and verification receipt

From the repository root, with pinned Node/Python/GHC dependencies installed:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py QueryIsolationTests TerminalNumericsTests TargetV2Tests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

These exports are limited to the verification shell, not deployment configuration.
No Node worker scheduling or test deadline changes. Reproduce the registered timing
comparison at frozen implementation `525f9b82` by saving this script outside Git and
running it with the same pinned Python and process prefix:

```python
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

root = Path.cwd()
sys.path.insert(0,str(root/'scripts/formal'))
import terminal_numerics as terminal
import target_v2 as target
import z3

baseline = '449d39fd0e24a22179b161f51655eb85f9284a12'
def previous(path, name):
    source = subprocess.check_output(['git','show',baseline+':'+path],text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace = {'z':z3}
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),namespace)
    return namespace[name]

def clone(function, replacements):
    return types.FunctionType(function.__code__, dict(function.__globals__,**replacements),
                              argdefs=function.__defaults__)

old_prove = previous('scripts/formal/terminal_numerics.py','prove')
old_certify = previous('scripts/formal/target_v2.py','certify')
fixtures = json.loads((root/'formal/research/terminal-counterexamples.json').read_text())
expected = json.loads(subprocess.check_output(['git','show',baseline+':formal/research/results.json']))
functions = {
 'old': {'terminal':clone(terminal.check_terminal,{'prove':old_prove}),
         'target':clone(target.check_targets,{'prove':old_prove,'certify':old_certify})},
 'new': {'terminal':terminal.check_terminal,'target':target.check_targets}}
rows=[]
for repeat,order in enumerate((('old','new'),('new','old'),('old','new')),1):
    for variant in order:
        for case in ('terminal','target'):
            start=time.perf_counter()
            row={'round':repeat,'variant':variant,'case':case}
            try:
                value=functions[variant][case](fixtures) if case=='terminal' else functions[variant][case]()
                key='terminalNumerics' if case=='terminal' else 'targetV2'
                if value != expected[key]:
                    raise ValueError('certificate payload differs from frozen baseline')
                row.update(outcome='pass',payloadEqual=True)
            except Exception as exc:
                row.update(outcome='failure',error=type(exc).__name__+': '+str(exc))
            row['seconds']=time.perf_counter()-start
            rows.append(row)
            print(json.dumps(row),file=sys.stderr,flush=True)
output={'schemaVersion':1,'baselineCommit':baseline,
        'testedCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'python':sys.version.split()[0],'z3':z3.get_version_string(),
        'sourceHashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                        ('scripts/formal/terminal_numerics.py','scripts/formal/target_v2.py')},
        'hostLoadAtEnd':os.getloadavg(),'attempts':rows,
        'scope':'Synthetic proof timing; no trading or training; shared-host timing is not a deadline guarantee'}
print(json.dumps(output,indent=2))
```

- Targeted checks: **exit 0**, 17 tests in 4.510 seconds.
- Explicit certificate recording: **exit 0**, 8.536 seconds; payload equality checked
  against the prior receipt excluding reviewed source hashes only.
- Formal wrapper: **exit 0**, 52 tests in 22.524 seconds; verifier 7.070 seconds.
- Full wrapper: **exit 0** on its first attempt for this change. Formal passed
  52 tests in 36.540 seconds and all scoped checks (verifier 10.715 seconds).
  Haskell build/format/lint/smoke/tests, web typecheck/241 tests/build and all 185
  automation tests passed, none skipped. Automation took 67.975 seconds. No
  assertion, deadline, source or solver limit changed during verification.
- Acceptance diagnostic: expected **exit 1**, after reproducing scoped certificates
  in 11.224 seconds; `ValueError: research acceptance blocked by open obligations`.
  All 38 mission obligations remain open/partial.
- Remote [CI run 36466045482](https://github.com/diegueins680/trader/actions/runs/36466045482)
  at `525f9b82`: formal, Haskell, web and automation passed; Docker build/deployment
  skipped. This is separate from the local full-wrapper result.

Raw logs remain outside Git; SHA-256 receipts below use prefix
`/private/tmp/trader-query-isolation-`, suffix `-20260928.log` unless noted.

| Artifact | SHA-256 |
| --- | --- |
| targeted.log | `4ec2bd85c13b3bf0c440e9225570203b908bc7233547cfe9ef59e00a5702b728` |
| record.log | `606769fe7f317ecb8731b850178272281a2cebf0b3c22a4585be6af96bba4292` |
| formal.log | `53e452e5f98e05b24a1c8b0a479489d7d5d2f98b511ddcb9ea66228355615688` |
| benchmark.log | `ae311d57c62f7e86b2bf189a520fa2cea84df909089cb83356806a6dde45ec45` |
| benchmark.json | `a49928dff09bd30f4f84d5167c3531de4e2a26677e5f57de84f23795b24bd083` |
| full.log | `7d6f3aa77912496543bab6649cf2f98c3d7e635f0a297ba65a777ce902d68cec` |
| acceptance.log | `a66edca330c50e5b1b5b29b3228e0207c2f49d73b075b041cb0f17a97650897b` |
| ci.json | `bcfb8ccb85163820d1daf49612e3ac4fed4e6dae5151c5d9d0a2523a804e523c` |

No policy-inference or economic benchmark is claimed. Previous return, cost,
drawdown, tail-risk, OPE and holdout evidence is unchanged. No live authorization,
order, live exploration, authenticated trading call, fleet/champion change,
deployment, merge or proof placeholder was introduced. General recommendation:
no trading adoption. RL recommendation: continue only offline research under the
existing gates; this verification improvement does not repair or promote a policy.
