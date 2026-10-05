# Binance order-number admission v1

The [wire-admission extension](order-wire-contract.md) supersedes the current
wire behavior and conformance bounds below. This v1 text records the original
repair; the current ledger and source-bound verifier include the extension.

Baseline a772ca0ffcde120134bda927e38387258bd94435. Engineering safety repair;
no financial trial, exchange endpoint invocation, live configuration or deployment.

Conflict: four typed futures constructors reject only x<=0, admitting NaN and
positive infinity; the general market constructor performs no finite check.
The intended positive order quantities/prices require finite positive binary64
values. Numeric invalidity must reject before credential access, timestamps,
signing, request construction or HTTP. Resolve against NaN-bypassing comparisons.

Affected complete roster: placeMarketOrder, placeFuturesPostOnlyLimitOrder,
placeFuturesMarketOrderWithPositionSide, placeFuturesTriggerMarketOrder,
placeFuturesAlgoTriggerMarketOrder. Preserve public signatures and valid request
construction. Existing market/mode/type restrictions remain. The selected base
quantity has priority over quoteOrderQty for spot/margin. Futures requires base;
an invalid selected base must not fall back to a valid quote. Ignored quote
values retain legacy semantics and are not serialized. Missing required amount
keeps its existing message, now before credentials; invalid numeric messages
explicitly require a finite positive number. No caps or permissions change.

Define valid(x) = not NaN(x) and not Infinite(x) and x>0.
A small total base-only Haskell module provides pure Either validators. For
market selection B=base present, Q=quote present, F=futures:
accept = if B then valid(base) else (not F and Q and valid(quote)).
Zero, negative, NaN and infinite selected values reject. No numeric conversion
or arithmetic is required for this gate. Existing serialization is unchanged:
positive tiny inputs may still serialize as zero; precision, exact tick/lot
membership, size caps and order effects remain separate unresolved obligations.

F-ORDER-NUMBER-FINITE: SMT over all binary64 values verifies finite positive
admission. F-ORDER-NUMBER-SELECTION: SMT verifies selected-value validity,
base priority and futures requirements over both optional-presence flags.
F-ORDER-NUMBER-FLOW: source-bound finite model, five constructors and two
Boolean classes each for numeric/context validity; no invalid-number transition
to the credential boundary. F-ORDER-NUMBER-CONFORMANCE: compile actual current
and preserved legacy guard prefixes against an inert credential-boundary marker,
plus pure validator oracle and Haskell generated properties. No constructor
body past the first credential read is executed in research tests.

A-ORDER-NUMBER: pinned GHC/base binary64 comparisons, Maybe/Either and ordinary
IO exception/sequence semantics, source extraction and reviewed model mapping;
inputs are ordinary immutable values, not bottoms; sufficient runtime resources.
Source binding is not a compiler or full IO refinement proof. The model abstracts
only local guards, not authorization, concurrency, fills, retries or lifecycle.
No complete safety-shield, wire formatting or venue-acceptance claim.

Preserve CE-ORDER-NUM-001 (NaN bypasses <=0) and CE-ORDER-NUM-002 (general
market constructor reaches credentials with non-finite selected quantity).
Use 13 boundary bit patterns and 4096 deterministic generated words, seed
20261005, optional-presence/market/priority cases, current/legacy source-derived
prefixes and source mutations. All five adapters must be covered independently
of the editable registry. Formal and full wrappers must pass on frozen sources.
Broader obligations remain unchanged; no extra closure follows from this gate.
