# Funding timestamp and arithmetic audit — 2026-10-01

**Decision: preserve negative engineering evidence; no adoption.** This audit
checks the unchanged development loader. No prices, settlements, archived policy
results, protected holdout or prospective campaign outcomes were read. Synthetic
fixtures alone exercise the source. The champion, production configuration and
frozen learner/economic results remain unchanged.

The engineering [registration](../registrations/funding-boundary-audit-engineering.json)
and [contract](../../formal/research/funding-boundary-contract.md) were committed
as `7d0b9141` before probes. Starting head `38896e1506ca4ffb08d638d49733f26d35935299`
passed [CI](https://github.com/diegueins680/trader/actions/runs/36711003428).
Latest fetched main is `dbd45e2691cb37f1421a23306c43b676fa82e6fc`.

## Findings and canonical interpretation

The loader assigns an event to the first close no earlier than that event. On
an exact ascending grid this puts j>0 events in (close[j-1], close[j]]. An event
exactly at a close belongs to that endpoint. Events after the final close reject.
Events at/before the first close collect in bucket zero; replay reads funding
at `left+1` after its start, so that bucket is not a prehistory cash debit.

A-SEQUENTIAL-RESEARCH-R1/E3 establish hash-bound byte snapshots, not the numerical
validity of every derived coefficient. R7 deliberately validates transition
values when consumed. E1's funding accounting needs the endpoint convention.
Those clauses do not establish a universally finite loader output. The stronger
claim is now separately marked **refuted**, rather than silently reinterpreting
a hash check as a numeric theorem or weakening downstream rejection.

The input checks admit finite rate/mark scalars but do not check the product or
accumulated bucket for finiteness. Prescribed **CE-RL-019** has two variants:

| Synthetic inputs | Overflow ignored | NumPy overflow raised |
|---|---|---|
| One rate 2, mark maximum finite binary64 | Loader returns positive infinity | Loader returns positive infinity |
| Two rates 0.75, each mark maximum finite binary64, same bucket | Loader returns positive infinity | FloatingPointError |

In these tested pandas tuples, multiplication uses Python floats; setting NumPy's
error policy does not trap that scalar product. Addition into the NumPy funding
array does obey the NumPy policy. The witnesses have strictly ordered timestamps,
finite rates and positive finite marks. Each full-loader fixture uses its own
matching synthetic byte hashes; no production registration or hash gate is bypassed.
The fixture magnitude is deliberately extreme, not a plausible calibrated market
scenario or evidence that any archived funding observation overflowed.

The returned array is read-only but can still contain infinity. In the tested
replay, the invalid next funding coefficient triggers `invalid_market_transition`
before a bar is booked: terminal failure, unchanged equity and no ledger row.
This is a deterministic downstream regression, not a proof of every replay path,
full-loader fail-closed behavior or production capability isolation.

## Formal evidence and refinement boundary

| Requirement | Status | Scope |
|---|---|---|
| F-RL-FUNDING-GRID | smt_verified | Source-derived grid expressions over all 4,910 registered indices: positive spacing, exact open/close relation, strict ordering and int64/binary64 representability bounds. |
| F-RL-FUNDING-ENDPOINT | smt_verified | Uniqueness and endpoint bounds conditional on NumPy left-insertion semantics, integer event time, count>=1 and positive spacing. |
| F-RL-FUNDING-FINITE | refuted | Two prescribed IEEE binary64 SAT witnesses, reproduced in the extracted loop and full synthetic CSV loader. |

The grid is metadata only: first open 1600819200000, last open 1742198400000,
first close 1600847999999, last close 1742227199999 milliseconds. Checking those
integers is not accessing the historical observations. The endpoint lemma covers
arbitrary integer times and positive spacing under the stated search relation;
it does not verify NumPy's binary-search implementation or infer actual provider
release time. Each positive theorem uses an independent satisfiable-premise and
unsatisfiable-violation query; each counterexample uses a prescribed SAT query.
Pinned solver seed 0 and 10,000 ms timeout are unchanged.

The full normalized loader AST is pinned to the preregistration. The checker
extracts the actual close expression and settlement loop. Conformance executes
that loop with ordinary synthetic records. Separate automation tests invoke the
whole existing CSV loader. Source binding and these tests are not universal
interpreter refinement. A-FUNDING-BOUNDARY names trusted Python/NumPy, AST
translation, exact sorted grids and stable ordinary bindings. No new dependency,
production path, order capability, model artifact or feature flag is introduced.

Eight new integrity tests cover six source mutants, four arithmetic/grid mutants,
solver-unknown refusal, registration drift, both numeric witnesses, signed/zero
funding and linear-reference comparison. Cases include 32 synthetic timestamps
and 14,730 close +/-1 ms neighbors on the registered grid. Two additional full-loader
tests cover seven endpoints and four witness/error-policy combinations, including
replay refusal. No randomized financial search or seed selection occurred.

No additional lifecycle model is introduced: this change is a read-only audit
of timestamp/numeric semantics, using the existing non-authorizing research
boundary and lifecycle gate. Production lifecycle, ownership, persistence,
shutdown and reconciliation proofs remain open. All 38 broader mission obligations
retain their open/partially verified status.

## Source semantics and reproduction

[NumPy searchsorted documentation](https://numpy.org/doc/2.3/reference/generated/numpy.searchsorted.html)
specifies the left-insertion endpoint relation on sorted arrays, including exterior
indices. [NumPy error handling](https://numpy.org/doc/2.3/reference/generated/numpy.seterr.html)
describes ignored, warned and raised NumPy arithmetic errors. Checked on 2026-10-01;
these references support primitive assumptions, not financial efficacy or a full
implementation proof. No paper, third-party code or dataset was copied.

Using the existing pinned [toolchain](../../formal/research/README.md):

```sh
python scripts/formal/test_integrity.py FundingBoundaryTests
python3 test/sequential_screen_test.py SequentialContracts.test_full_loader_funding_endpoint_contract SequentialContracts.test_preserved_funding_overflow_full_loader_and_replay_refusal
bash scripts/verify.sh formal
bash scripts/verify.sh full
python scripts/formal/verify.py --require-complete
```

The final command must refuse acceptance while critical broader obligations remain
open. The formal Python environment needs the existing pinned NumPy/Z3 stack;
the full-loader automation uses existing pinned pandas 2.3.3 as well. No new
pandas dependency was added to the formal gate.

At source freeze `fc0ab4142e6bc7761488e0c11fa69ec11919485e`,
`bash scripts/verify.sh formal` and `bash scripts/verify.sh full` both exited 0.
The full run passed 104 formal integrity tests, 36 scoped SMT requirements,
Haskell checks, 241 web tests and 185 automation tests. The existing web
bundle-size advisory remains; no tests/assertions were disabled. [Source-freeze
CI](https://github.com/diegueins680/trader/actions/runs/36801075943) passed formal,
Haskell, web and automation; Docker/deployment were skipped. Subsequent edits
record this report and evidence only.

The [machine-readable receipt](funding-boundary-evidence-receipt-2026-10-01.json)
records commands, statuses, log hashes, environment, proof/conformance counts and
verification timings. Standalone formal integrity/certificate times were
18.715/4.892 seconds; full-run formal times were 30.616/7.854 seconds on a shared
host. These are verification observations, not policy inference benchmarks or
guaranteed deadlines. The acceptance command exited 1 with
`ValueError: research acceptance blocked by open obligations`, as required.
Initial engineering errors are retained in that receipt: generated source string
syntax, a test invoked before its fixture existed, an incorrect unittest module
path, and two metadata-edit helper key errors. They were corrected without changing
the registered numeric cases, proof limits, funding source or financial protocol.
No initial failed invocation is presented as a passing verification.

Recommendation remains **no candidate passed**; reject the frozen tested RL
configurations for integration and continue offline research only under explicit
registration. This audit neither repairs the loader nor establishes reliable OPE,
untouched confirmation, champion superiority, cost realism or operational readiness.
