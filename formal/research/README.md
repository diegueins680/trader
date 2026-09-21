# Offline research verification runbook

This is a **scoped assurance gate**, not research acceptance. The rejected
sequential-control screen, champion, protected datasets and production permissions
are unchanged. Start with [the canonical contract and consistency resolutions](contract.md).

## Install and reproduce

Use Python 3.13.3 and GHC 9.4.8. The only new dependency is the MIT-licensed
Z3 Python distribution 4.15.4.0 (solver reports 4.15.4). Supported wheel hashes
are pinned for Linux/macOS x86-64/ARM64. Install once:

```sh
python3 -m venv /private/tmp/trader-formal-tools
/private/tmp/trader-formal-tools/bin/python -m pip install --require-hashes --only-binary=:all: -r scripts/formal/requirements.txt
export TRADER_FORMAL_PYTHON=/private/tmp/trader-formal-tools/bin/python
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

The formal command subsequently needs no network, market data, training artifacts,
exchange credentials or production services. It compiles a base-only Haskell
fixture driver in temporary storage. The existing full wrapper additionally needs
its normal Haskell, web and research dependencies. `verify:formal` is the npm
alias. CI uses the same formal wrapper, exact Python/GHC pins and hash-locked Z3.
The new CI job is a prerequisite for the existing build/deploy jobs; no deployment
command, identity, permission or live setting is changed.

A separate acceptance diagnostic deliberately fails today:

```sh
"${TRADER_FORMAL_PYTHON:-python3}" scripts/formal/verify.py --require-complete
```

It reports all 38 requested whole-system obligations as open/partially verified.
A green scoped gate does not make this draft ready for candidate integration.
There is no switch in this work to authorize a candidate, experiment or order.

## What is checked

- [Nine SMT obligations](../../scripts/formal/proofs.py): IEEE binary64 target
  bounds, evidence precedence, malformed/timeout fallback, disabled behavior,
  stable rescreening, constant lack of authority, integer slice/purge lemmas,
  and exact-real accounting identity. Every negated claim must be UNSAT within
  10 seconds. SAT and UNKNOWN are failures. Acceptance satisfiability is also
  checked so a gate rejecting everything cannot satisfy the suite vacuously.
- [Finite protocol](../../scripts/formal/lifecycle.py): all 75 reachable states,
  349 directed transitions, maximum shortest-path depth 6, two callers. Search
  reaches a fixed point, not a depth cutoff. Safety is invariant checking;
  conditional quiescence uses a rank of at most two outstanding calls. No fairness
  means no unconditional termination claim. The model is an abstract client,
  not the server's concurrency implementation.
- [Compiled Haskell conformance](../../scripts/formal/conformance.py): 16,384
  combinations from a declared representative domain plus 4,096 generated cases
  with seed 20260920. Binary64 values are transported as Word64 bits, covering
  signed zero, NaN, infinities, subnormals and adjacent threshold values. All
  240 accepted cases retain exact target bits and return false order authority.
  A negative compilation fixture also checks that the proposal constructor remains
  private. Random/generated testing does not prove universal implementation refinement.
- [Integrity regressions](../../scripts/formal/test_integrity.py): malformed and
  missing traceability, unsupported proof upgrades, source/receipt drift checks,
  duplicate/non-finite JSON, and deliberate unsafe state-model mutations.

The [results](results.json) contain deterministic outputs and source hashes;
wall time is printed per invocation and deliberately excluded from equality.
The [toolchain lock](toolchain.json) pins both manually translated source and
checker inputs. Hash matching binds the review to bytes; it is not a proof that
translation or a trusted compiler is correct. No production module was rewritten.

## Traceability and status

The [proof ledger](proof-ledger.json) is the bidirectional requirement/model/
assumption/artifact/code/test/CI index. Canonical IDs also appear in
[formal/specifications.json](../specifications.json). Every critical introduced
source and materially changed verification entry point has an inverse mapping.
Operational evidence is confined to reproducible offline checks and the prior
rejected research reports, with no order or deployment witness.

The ledger separates `smt_verified`, `model_checked`, `property_tested`, and
`open`; the schema accepts the requested vocabulary but the verifier refuses to
upgrade a claim to an unsupported verification class. The 38-item mission map
is deliberately more demanding than the narrow completed lemmas. It records
full implementation obligations as unresolved, even where a lemma is useful.
No broad “formally verified system” claim is made.

`CE-RL-001` is a preserved [counterexample](counterexamples.json) to an overly
strong specification: enable, start admissible call, disable, finish call. The
caller can retain a non-authorizing immutable proposal. The regression replays
this trace; the contract requires no order authority, not magical object erasure.
No unsafe exchange transition or production bug was demonstrated by this trace.

## Updating evidence

When a pinned file changes, investigate the proof/refinement impact, update the
contract and assumptions if necessary, and deliberately review updated source
hashes. Then use `scripts/formal/verify.py --record` to write results only after
all obligations check, inspect the diff, and rerun the normal wrapper. CI never
records or refreshes expected evidence. Never use pin updates to bypass a failed
property. Runtime timings are diagnostic, not reproducible performance claims.

## Tool choice, security and limitations

Explicit-state enumeration is sufficient for 75 states and avoids another runtime.
Z3 directly represents IEEE binary64; an exact-real theorem alone would miss
NaNs and infinities. A symbolic Haskell/SBV or Liquid Haskell core is a valuable
future way to reduce the translation gap, but adding a production dependency
without a validated challenger is unwarranted here. Lean/Coq/Isabelle can provide
smaller-kernel algebraic proofs; none is installed or claimed. PRISM/Storm need
credible transition probabilities before numerical market-risk claims are useful.
No probabilistic model-checking result is reported. Neural verifiers are not
added: no policy passed evidence gates, and no certified tanh policy domain is
claimed. The independent rejection gate remains the safety boundary.

Z3's [official security page](https://github.com/Z3Prover/z3/security) had no
published advisories when checked on 2026-09-20; this is not a vulnerability-free
guarantee. Its [release and license metadata](https://pypi.org/project/z3-solver/4.15.4.0/)
were checked. The pinned release is older than the latest release, chosen for a
stable reproducible binary64 verification environment; upgrades require rechecking.
Z3 runs on repository-controlled formulas, with a per-obligation solver timeout.
No native solver is exposed to network requests or model-supplied formulas.

Unproved: production ownership, server draining/recovery, real availability and
revisions, full simulator floating-point accounting, exchange rounding/margin,
artifact authenticity, end-to-end capability isolation, universal compiler
refinement, neural robustness, loss bounds through gaps, and future performance.
All empirical acceptance gates remain binding. `RL-OFFLINE-001` stays HIGH/OPEN.
