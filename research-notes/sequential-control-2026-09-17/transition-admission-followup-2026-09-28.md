# Training-transition admission continuation — 2026-09-28

**No adoption. The broader mission remains incomplete.** This continuation checks
the existing training-transition admission boundary without changing replay,
training algorithms, data, policies, artifacts, registrations, current champion or
live configuration. It uses synthetic engineering fixtures only. No financial
trial, protected holdout or independent out-of-sample period is accessed.

Latest remote main remains `dbd45e26`; work continues on the isolated branch
`research/sequential-review-2026-09-28` in draft #284, stacked on #281. Existing
[transition admission rules](transition-admission-audit.md) remain authoritative;
this continuation fills their proof/traceability gap, not a simulator behavior gap.

## Specification, source and SMT

The [contract](../../formal/research/transition-admission-contract.md) was committed
at `bcc816e0` before implementation. `F-RL-TRANSITION-ADMISSION` derives the base,
terminal and nonterminal validity expressions from the actual helper source.
An audited source skeleton binds its branch and raising behavior. The entire
collector source is recognized to preserve step → admission → append ordering;
unknown source/control-flow changes require review rather than silently passing.

Three SAT-premise/UNSAT-violation checks establish necessary admission conditions:

- Reward is finite, equity is finite and positive, the failure tag is permitted,
  and time advances within the supplied horizon and episode endpoint.
- An admitted terminal has absent successor, zero units, no pending action, and
  either a recognized risk-failure tag or final-bar endpoint.
- An admitted nonterminal has native false done flag, no failure, exactly one
  supplied horizon of progress before the endpoint, and a successor passing the
  existing real-vector representation, shape and finiteness predicates.

Indices are unbounded mathematical integers. Reward, equity and units use IEEE
binary64 predicates, including NaN, infinities and signed zero. The theorem does
not silently apply real-number equality to floating-point execution. Non-binary64
numeric types are outside this SMT domain. Array validity and absence predicates
are abstraction atoms under trusted primitive semantics. The scoped SMT count is
now **17**; the existing 75-state lifecycle and 27-state artifact models are unchanged.

## Counterexample and actual implementation tests

[CE-RL-009](../../formal/research/transition-counterexamples.json) deliberately
removes the zero-inventory terminal guard. At left=30, t=33, stop=34, horizon=3,
equity=1 and units=0.25, the original helper rejects the terminal and the compiled
mutant admits it. SMT rejects the mutated source. This is a regression mutant,
not an existing loader/replay defect or evidence of financial performance.

Six new regression tests cover source drift, missing reward/pending/finite-vector
checks, partial-horizon admission, vacuous premises and append-order bypasses.
Thirty invalid helper fixtures include non-finite scalars, invalid done types,
wrong indices, incomplete failures, missing/malformed successors, outstanding
inventory and pending actions. Valid terminal signed zeros and all four recognized
risk-failure categories are also exercised.

Thirty-six actual replay cases cross horizons 1/3/6, targets -0.25/0/0.25, costs
1x/2x and full/half fills on a small flat-price fixture. Every transition passes
the actual admission helper, and each episode terminates with zero units and no
pending action. An injected unliquidated terminal makes the real collector raise
before it can publish training output. These are deterministic regression tests,
not proofs of all traces or market realism. No trained artifact is retained.

## Assumptions and remaining blockers

`A-TRANSITION-ADMISSION` names ordinary stable Replay fields, integer indices,
binary64 scalars, primitive identity/comparison semantics, ordinary arrays and
stable helper bindings; no concurrent writes, hostile subclasses or monkeypatches.
Python exception/control flow, `_finite_real`, vector representation/shape/finite
checks, the AST parser and restricted translator are trusted. The abstraction maps
successful actual helper calls to satisfying assignments of the extracted predicate;
a machine-checked interpreter/compiler refinement is not provided.

These necessary admission conditions do not establish complete liquidation fees,
fill prices, funding, cash accounting, truthful failure classification, future
market safety, Bellman/GAE target correctness, finite optimizer updates or
preemptive timeouts. An earlier recognized risk failure remains failed economic
evidence; its accounted training terminal is not a successful evaluation path.

Canonical clauses, proof ledger, source locks, tests, CI and risk documentation
map this requirement bidirectionally. Mission obligations 6/18/19 gain partial
evidence, but **all 38 broader obligations remain open/partial**. `RL-OFFLINE-001`
remains HIGH/OPEN. The tested RL policies remain rejected; contaminated development
results, invalid OPE, missing matched champion comparison and sealed holdouts are
unchanged. Neither production authorization nor candidate promotion follows.

## Verification receipt

The targeted six-test suite passed. Final formal/full commands and log hashes
will be recorded after completion; no full-verification success is claimed yet.
