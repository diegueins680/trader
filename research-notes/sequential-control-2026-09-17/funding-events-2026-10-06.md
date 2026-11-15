# Exact funding events v2 — 2026-10-06

**Decision: retain a disabled, disconnected engineering kernel; no adoption.**
No prices, settlements, archived policy results, protected holdout or prospective
campaign outcomes were read. Synthetic fixtures alone exercise the source. The
champion, production configuration, frozen loader and all frozen learner and
economic results are unchanged.

The [registration](../registrations/funding-events-v2-engineering.json) and
[contract](../../formal/research/funding-events-v2-contract.md) were written
against base `fccd49914baf6d44f9949031d484797cded64c76` before any probe.

## Why this increment

Obligation 10 (overflow and underflow) is the only fully open mission obligation.
Its recorded next action names funding as a remaining path. CE-RL-019 showed
that the frozen development loader turns finite rate/mark scalars into an
infinite coefficient through a binary64 product or sum. `replay-accounting-v2`
already consumes exact (mark, rate) events but had no exact producer.

## What changed

`scripts/research/funding_events_v2.py` (`load_v2`, `enabled=False` by default):

- Admits only canonical ASCII fixed-point decimal text (at most 20 whole and 20
  fraction digits) and canonical integer times below 2^63. Floats, exponents,
  `inf`/`nan`, whitespace, `+` and Unicode digits reject.
- Decodes exactly to `Fraction`; marks must be positive and times strictly increasing.
- Puts each event in the first close no earlier than its time, using a
  two-pointer sweep. Events at or before the first close go to bucket 0, as in the
  frozen loader. An event after the final close, or a bucket with more than 128
  events, rejects the whole load.
- Publishes an immutable `Buckets` value (per-bar `(mark, rate)` events plus exact
  `per_unit` sums), or `None`. Nothing is published partially.

There is no file, network, runner, learner, promotion or order interface.

## Evidence

| Requirement | Status | Scope |
|---|---|---|
| F-RL-FUNDING-V2-SOURCE | exhaustively_checked | Complete reviewed AST lock, constants/regex domain, imports, immutable output, activation-first default and checked primitives. |
| F-RL-FUNDING-V2-ARITH | smt_verified | 11 SAT-premise/UNSAT-violation pairs: decoded, product, 128-term accumulated and reduced sizes plus Fraction intermediates stay far below 2^8192; quantifier-free sweep exit, body and carry lemmas. A 256-case exhaustive sweep check (grid 1,3,5,7; times 0..8; up to 4 events) runs alongside. |
| F-RL-FUNDING-V2-FLOW | model_checked | 8 states, 12 transitions: publication only after every stage succeeds; a mutant that publishes after a failure is caught. |
| F-RL-FUNDING-V2-CONFORMANCE | property_tested | 146 loads (128 seeded, 18 registered edge cases) agree with an independent compiled Haskell Rational oracle that parses by characters and buckets by prefix count; outputs compose exactly with `replay_accounting_v2.advance_v2`. |

Consequence on the admitted domain: no overflow, no underflow and no rounding;
the 8192-bit guard never rejects admitted input. Both CE-RL-019 witnesses, as
decimal text of the maximum finite binary64, are outside the domain and reject
before any arithmetic. The largest admitted 128-event bucket publishes an exact
coefficient above 10^40 in magnitude.

Six new integrity tests cover the end-to-end check, seven source mutants, a model
mutant, solver SAT/UNKNOWN refusal, 20 malformed-row and 8 malformed-grid cases,
endpoint and bucket-limit semantics, the CE-RL-019 witnesses, immutability and
replay composition.

## Engineering corrections (retained, not hidden)

- The first sweep lemma used a quantified array sortedness axiom. It returned
  `unknown` under the full suite with the pinned 10 s timeout, so the gate failed
  closed. It was replaced by quantifier-free lemmas with explicit sortedness
  instances (each now under 0.1 s), and the contract wording was updated to match.
- A test fixture initially placed events after the final close. The fixture was
  corrected; the semantics were not.
- Registration JSONs are not allowed in specification implementation lists. The
  entry was removed, as in earlier increments.

## Verification on this host

| Command | Result |
|---|---|
| `scripts/formal/test_integrity.py` (pre-record run) | 296 tests OK, 208.6 s |
| `scripts/formal/verify.py --record` | exit 0; 12 closed / 26 open obligations; `missionComplete` false |
| `bash scripts/verify.sh automation` | exit 0; 185/185 |
| `bash scripts/verify.sh formal` (final head) | **exit 1**: `PPOProcessBridgeTests` failed with `no actual inference for trained policy` |
| Same test on untouched `origin/main` | **fails identically** |

The PPO bridge failure is environmental. That test accepts `(Absent,True)`
deadline misses and fails only when all three trained-policy calls miss. The host
load average was 30–37 from unrelated builds and processes, and the test fails
identically on unmodified main. It is not presented as passing. Clean CI on the
draft PR is the authoritative formal/full verdict. `bash scripts/verify.sh full`
was not run locally: this change touches no Haskell or web source, and a from-scratch
cabal build under this load was not attempted.

## Limitations

A-FUNDING-EVENTS-V2 trusts pinned CPython re/int/Fraction/dataclass semantics, the
reviewed AST translation, Z3, GHC Rational for differential testing, stable
bindings and sufficient resources. The decimal domain is a numeric admission
contract, not evidence that any provider emits canonical text or that a record is
causally available. Provider release/first-seen timing, revisions, symbol scoping,
the frozen loader and runner composition stay unverified. CE-RL-019 stays refuted
in the frozen source. Obligation 10 stays **open**; the counts stay 12 scoped
closures / 25 partial / 1 open.

## Recommendation

Unchanged: **no candidate passed**. Reject the frozen tested RL configurations
for integration. Next admissible step for obligation 10: compose
replay-accounting-v2 and funding-events-v2 into a checked runner, then repair the
Q/CQL successor paths behind CE-RL-014/015. No live flag, order, authenticated
endpoint, market data, holdout, champion or fleet setting was touched.
