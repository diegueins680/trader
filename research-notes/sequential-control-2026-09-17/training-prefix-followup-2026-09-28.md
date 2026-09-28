# Training-prefix verification continuation — 2026-09-28

**No adoption. The broader mission remains incomplete.** This follow-up connects
normalization and training episode bounds to the existing runner/environment
source. It introduces no policy, dataset, reward, artifact, production or live
behavior change. No financial trial or protected holdout is accessed.

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

## Verification status

Final scoped/full commands, results and log hashes will be recorded after those
runs finish. Previous CI passes are not a claim that this new implementation has
already passed full verification. `--require-complete` must continue to fail for
the explicitly unresolved mission obligations.
