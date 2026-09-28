# Training-prefix and episode-bound contract — 2026-09-28 continuation

Specified before implementation. This extends verification of the existing
rejected offline screen, without changing research semantics, registration,
market data, policy artifacts, production behavior or holdout permissions.

## Intended source-linked properties

**F-RL-FIT-PREFIX.** Let N be each admitted source length and T the exclusive
registered training stop, with 25 <= T <= N. The runner's actual price and funding
comprehensions must preserve symbol keys and slice each value at an upper bound
U satisfying 25 <= U <= T. Its immediate normalization fit must consume only
those price prefixes. Scale.fit must enumerate feature times in [24,U), and the
already checked feature footprint then implies every fit read i satisfies
0 <= i < T. The same fields of the resulting Scale must be unchanged when values
at indices T and later are modified, with unchanged inputs/metadata before T.

**F-RL-COLLECT-PREFIX.** Let L >= 121 be an admitted training-prefix length, bounded
by T. Derive the random-start interval and Replay stop expression from collect's
source. For every sampled start, the episode is valid and has stop <= L <= T;
all observations and outcomes permitted by that half-open episode are before T.
This proves index containment under the existing Replay boundary contract, not
all Replay transition, reward or NumPy behavior.

## Verification design

Use the existing pinned AST parser and Z3. Extract and inspect the actual runner's
fold setup, Scale.fit row comprehension and collect's episode construction.
Unsupported shapes, alternate data inputs, symbol remapping, changed fit dataflow,
extra setup statements or changed admission contracts must fail closed. Audited
source skeletons bind the supported dataflow; arithmetic expressions within the
specified slots are translated to unbounded mathematical integers. SAT premises
are mandatory and negated claims must be UNSAT within ten seconds. No source
hash alone is called a semantic proof.

The frozen registration supplies three folds, horizon values 1/3/6 and a six-bar
train/test gap. Check all nine combinations against the existing registered
horizon-separation rule, including valid integer types and exclusive stops. Do
not revise the historical registration or describe the gap as two additive
six-bar gaps: its purge/embargo labels do not establish two separately allocated
regions. This check certifies the existing index inequality, not compliance with
a stronger independent purge-plus-embargo design or untouched confirmation.

Counterexample fixtures must preserve deliberate normalization leakage and
one-bar episode-overrun mutations. Actual-source setup execution and actual Scale
and collect conformance use only synthetic fixtures. Modify future values,
including missing/non-finite values, and check normalized parameters and training
samples; log actual reads and reject any index at/after T. Property tests and
finite exhaustive cases must be labeled separately from the SMT checks.

## Assumptions and limitations

A-TRAIN-PREFIX: admitted arrays are ordinary fixed-shape base numeric NumPy arrays,
with stable symbol identities, source bindings and no concurrent writers. Python
slicing/range/RNG semantics, NumPy reductions, the audited Scale constructor and
snapshot behavior, and the restricted AST checker are trusted. The runner enters
the checked fold setup with its registered arrays and split; earlier provenance
admission and arbitrary external/aliased mutations are not newly proved. The fit
method is invoked on Scale, not a hostile subclass. Unsupported source changes
require explicit review.

The abstraction maps concrete array lengths, source index expressions and registered
split fields to exact integer bounds. Source-skeleton recognition plus index SMT
is not a verified compiler, a universal program analysis or a full runner proof.
Existing feature-footprint, immutable-scale and Replay tests remain necessary.

Publication/revision timing, leakage through universe selection, policy optimization
correctness, hidden market state, production normalization, full Replay refinement,
and independent economic evidence remain outside this certificate. Existing
whole-mission obligations remain open/partial; no candidate gains eligibility.

## Traceability and delivery

Each requirement maps to its source-derived expression, assumption, verifier,
actual implementation, mutation/conformance tests, deterministic receipt and
`bash scripts/verify.sh formal`, also invoked by `full`. Pin every new critical
source and fixture. Update the proof ledger, canonical specifications, risk record,
README/CHANGELOG and continuation report. Preserve all prior negative results,
failed paths, champion behavior, and the prohibition on live authorization.
