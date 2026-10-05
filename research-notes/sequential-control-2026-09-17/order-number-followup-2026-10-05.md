# Binance finite order-number admission — 2026-10-05

Baseline a772ca0ffcde120134bda927e38387258bd94435. Registration commit 1964ad46
preceded implementation. No market-data trial, final-holdout access, exchange
endpoint invocation, order, live setting change or deployment.

CE-ORDER-NUM-001: a comparison rejecting only x<=0 allows NaN and positive
infinity. Four futures constructors had this guard. CE-ORDER-NUM-002: the general
market constructor reached credential access with an invalid selected number.
The new base-only OrderNumeric module provides pure finite-positive and market
selection validators. All five constructors validate before the first credential
read. The public signatures and valid request-building tails remain unchanged.
Base quantity retains priority; a bad present base never falls back to quote.
Unused quote values retain legacy semantics. Missing-amount error messages are
preserved, but now take precedence over missing-credential errors.

The source-bound SMT scope is five SAT-premise/UNSAT-violation pairs over all
binary64 values and Boolean presence/market flags. The local transition model
has 40 states, 20 initial, 20 terminal, 20 transitions and depth one, covering
five constructors and numeric/context validity. The credential boundary is
terminal in the model; it is not an order capability or authorization proof.

Compiled actual current and preserved legacy prefixes cover 5344 rows: 1248
boundary combinations from 13 bit patterns and 4096 seeded generated rows
(seed 20261005). Each row traverses all five constructors, yielding 26720 current
and 26720 legacy prefix checks. Spot/margin/futures, both modes, optional-value
presence, quantity/quote priority and empty/nonempty order type are represented.
The driver's terminal marker replaces the first credential read; no Binance
module, credential, request, signing or HTTP code is linked. The source-derived
local trim helpers are compiled too. An independent Python oracle and Haskell
properties over 1000 generated words supplement the proofs.

Initial targeted run: solver/model/compiled conformance passed; the separate
source-mutation test failed because its harness omitted a JSON import. The import
was corrected without changing the numeric implementation or solver obligations.
Final targeted suite: 2 tests passed in 26.848 seconds.

Limitations: pinned GHC/base and IO primitive semantics, immutable non-bottom
inputs, source extraction and reviewed abstraction remain assumptions. This is
not a compiler proof, full IO refinement, or proof of caller retry behavior.
Tiny positive values can still become zero in existing eight-decimal formatting.
Venue filters, exact tick/lot representation, final wire caps, upper exposure
bounds, fill behavior and other exchanges remain unresolved. No future-profit
or financial acceptance claim follows from numeric admission.

Status remains 5 scoped closures, 27 partial, 6 open. Obligation 21 gains narrow
boundary evidence but remains partial. The frozen financial trials, all seeds,
costs, OPE failures, contamination status and sealed holdout remain unchanged.
No candidate is adopted. Canonical wrapper and final CI evidence will be recorded
after frozen-source reproduction.

Local full receipt reproduction (`python scripts/formal/verify.py --record`)
failed in the unchanged artifact v4 worker probe: `--snapshot-contract-v3`
exceeded its existing three-second subprocess timeout. No new receipt was
written and no timeout was widened. Pinned CI must reproduce the frozen sources;
targeted numeric passes are not a substitute for the full verification gates.

The first ordinary CI automation check rejected an uncovered implementation
file: OrderNumeric was in the proof ledger but absent from the canonical
specification's implementation roster. Added the exact module path and direct
conformance/property evidence links; no coverage rule or proof was weakened.
The superseded pinned run was canceled; the corrected source revision must
repeat receipt reproduction and both canonical wrappers before merge.
