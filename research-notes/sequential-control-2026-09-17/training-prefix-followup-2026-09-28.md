# Training-prefix verification continuation — 2026-09-28

**No adoption. The broader mission remains incomplete.** This follow-up connects
normalization and training episode bounds to the existing runner/environment
source. It introduces no policy, dataset, reward, artifact, production or live
behavior change. No financial trial or protected holdout is accessed.

This receipt records the training-prefix stage. The later
[artifact-admission continuation](artifact-admission-followup-2026-09-28.md) adds
one SMT requirement and a separate finite gate model; its own receipt records
the current counts without changing these earlier results.

The previous revision `63bc0a51` passed all remote CI jobs in
[run 36422872047](https://github.com/diegueins680/trader/actions/runs/36422872047).
Build and deployment were skipped. Latest main is still `dbd45e26`; the branch
remains `research/sequential-review-2026-09-28`, stacked on #281 in draft #284.

## Specification and source connection

The [contract](../../formal/research/training-prefix-contract.md) was committed
before implementation (`15ce021b`). `F-RL-FIT-PREFIX` checks the actual runner's
first three fold statements, preserving symbol keys in price/funding prefixes
and passing only the price prefixes to Scale.fit. Its audited source skeleton
binds the class method's row comprehension and reductions. Z3 receives arithmetic
extracted from the actual AST, including `split['trainStop']` and
`range(24, len(p))`; it does not merely check a separately written index example.
Together with the existing feature footprint, every fitted feature index is
strictly before the exclusive training stop under the stated preconditions.
The Scale pools symbols' training prefixes, as the registration specifies for
shared policy fitting; key preservation is not a claim of per-symbol scaling.

`F-RL-COLLECT-PREFIX` derives `rng.integers(24, len(p)-96)` and `start+97` from the
actual collector episode construction. For every admitted prefix length L>=121,
the start interval is nonempty and each constructed episode stop is at most L.
Outcomes permitted by the existing half-open Replay contract remain inside the
training prefix. Full Replay transition, reward and runtime correctness is not
proved by this index certificate.

There are six new SAT-premise/UNSAT-violation queries across these two requirement
IDs, each with a ten-second timeout, bringing the scoped requirement count to 15.
The checker also validates all nine frozen fold/horizon combinations (three folds
and horizons 1/3/6), integer domains and exclusive bounds. The actual six-bar gap
is not described as two additive six-bar purge and embargo regions. A stronger
partition design or independent confirmation remains outside these results.

## Conformance and counterexamples

[CE-RL-005/006](../../formal/research/training-counterexamples.json) are deliberately
introduced mutants, not existing implementation defects:

- A prefix ending at trainStop+1 lets the first excluded price alter fitted
  normalization. The checker rejects it, and actual-source setup execution
  demonstrates the parameter change at T=160 with a 256-row synthetic array.
- Replacing start+97 with start+98 overruns the shortest admitted prefix at
  start=24, length=121. SMT rejects that bound; the actual Replay constructor and
  a compiled copy of the mutant collector reject the episode before execution.

Forty-eight cases execute the actual three setup statements with two symbols and
corrupt all post-training prices with NaN, infinity, negative or very large values.
All four fitted fields (mean/std/low/high) remain unchanged. Twenty-seven paired
collector cases span three prefix lengths, three horizons and all seeds 11/23/47;
each pair collects twelve decisions. Later price/funding corruption leaves every
training-array output identical. Instrumentation records actual feature windows
and episode endpoints inside the prefix. These are deterministic property tests,
not a universal implementation proof.

An additional actual-runner conformance test covers four future corruptions,
checks the arrays and Scale passed to the optimizer and verifies persisted scale
provenance. Its loader, optimization and evaluation are stubbed; it is an
engineering fixture, not another financial experiment or empirical replication.

## Assumptions and unresolved scope

`A-TRAIN-PREFIX` names ordinary base NumPy arrays, stable symbol/split bindings,
no concurrent writers or hostile subclasses, and trusted Python slicing/range/RNG,
NumPy reductions, Scale construction and the restricted source-skeleton checker.
Earlier admission and arbitrary external/aliased mutations are not newly proved.
The source/fixture hashes and bidirectional ledger connect the claims to actual
implementation, checks and CI, but do not prove the checker or interpreter sound.

The fit cutoff T is distinct from a simulated training episode's historical
clock t. Scale.fit uses the whole training prefix, including rows later than an
early training episode. The new theorem excludes indices at/after T; it does not
certify as-of-t normalization for each training episode. Evaluation starts after T,
and the actual feature/observation metamorphic tests hold the supplied Scale fixed.
Do not present training episode performance as independently causal OOS evidence.

The real historical loader scans the entire registered panel for validity before
fitting. Future-invalid values can prevent admission of a complete run. Therefore
this is **conditional training-input isolation, not online admission causality**.
The actual-runner fixture explicitly bypasses that loader; no broader claim is
inferred. Availability/revision timing, universe selection, full state-history
causality, production normalization and independent economic superiority remain
unestablished. All 38 whole-mission obligations remain open/partial. The proof
ledger adds evidence to normalization, split, causality and replay-determinism
obligations without upgrading them to complete.

No current champion comparison, OPE result, statistical gate, failed seed or
financial conclusion changes. All previous RL configurations remain rejected;
no challenger becomes eligible for shadow/paper/live use. `RL-OFFLINE-001`
remains HIGH/OPEN. No dependencies or configuration flags are added.

## Verification receipt

Executable source is frozen at `db06ea4c`, with report revision `c4ea8f67`.
The receipt and cross-links change documentation only. Commands ran from the
isolated worktree using GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8,
Node 20.19.0 and formal Python 3.13.3 / NumPy 2.3.5 / z3-solver 4.15.4.0
(solver reports 4.15.4).

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
bash scripts/verify.sh formal
python3 -m unittest discover -s test -p sequential_screen_test.py \
  -k test_runner_passes_only_training_prefixes_to_fit_and_optimizer
bash scripts/verify.sh full
bash scripts/verify.sh automation
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

- Scoped formal wrapper: **exit 0**, 23 integrity tests, 15 scoped SMT requirement
  IDs, unchanged 75-state/349-transition lifecycle abstraction, 20,480 Haskell
  conformance cases and 180 exact-rational accounting traces. The standalone
  verifier reported 18.896 seconds. No claim of complete implementation refinement.
- Actual-runner test: **exit 0**, one test with four corruption cases, 0.712 seconds.
  Its mocked training output is not financial trial evidence.
- Initial full wrapper: **exit 1**. Formal, Haskell and web passed; automation
  passed 183/185. The scheduled-collector Python subprocess reached its unchanged
  30-second timeout, and the edge-campaign subprocess reached its unchanged
  60-second timeout (`spawnSync` status null). These synthetic fixture timeouts
  are preserved; host contention is plausible but not established as their cause.
- Unchanged automation retry: **exit 0**, 185/185, none skipped, 64.866 seconds.
  No assertion, timeout, concurrency setting or test selection was changed.
- Full-wrapper retry: **exit 0**. Formal checks, Haskell build/format/lint/smoke/
  tests, web typecheck/241 tests/build, and all 185 automation tests passed, none
  skipped. Automation took 55.765 seconds; the scoped verifier reported 7.450
  seconds. No source, test, timeout or gate changed between attempts.
- Acceptance diagnostic: expected **exit 1**, `ValueError: research acceptance
  blocked by open obligations`. All 38 broader obligations remain open/partial.
- Remote [CI run 36425743197](https://github.com/diegueins680/trader/actions/runs/36425743197)
  at `c4ea8f67`: formal, Haskell, web and automation passed. Docker build and
  deployment were skipped. This does not erase the initial local full-run failure.

Logs remain outside Git. SHA-256 receipts (prefix
`/private/tmp/trader-training-`, suffix `-20260928.log`):

| Log | SHA-256 |
| --- | --- |
| formal | `fd7eb0f697f0db570349cbd35e80593c1b320060ed61d66341bc1c302ae9e6d0` |
| acceptance | `0cc9ac4a02857bf0b612e5f698bb33d0571eb4040369dd78444a89d63091a077` |
| full (initial failure) | `d72668683fc2d2ba32db59986c92d5021cb2ea50b89665379dfe1172d8951d72` |
| automation-retry | `3a0360ac26a7327ff0190a6737e7d6052080a1514c176de32514f55993500760` |
| full-retry | `10d3a829c7f2ff5135d210cca6ec548725e6cdd0090c8e116ab67af2c37892ae` |

The branch remains a draft. No new live authorization, order, authenticated
exchange call, exploration, merge, deployment or champion change occurred.
Protected holdouts remain untouched. Passing these scoped checks cannot satisfy
the unresolved empirical, operational or whole-system proof requirements.
