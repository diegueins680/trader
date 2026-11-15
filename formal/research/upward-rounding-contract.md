# Upward rounding and invalid maker-price rejection v1

Baseline: 906ad455bef2b6706bd44667679596aa75766aae. Engineering repair;
no market data, financial trial, deployment, order call or configuration change.

The two Main quantizeUp definitions subtract 1e-9 before ceiling, returning 1
for nextUp(1) on the unit grid. This contradicts their minimum-quantity and
away-from-spread intent. Non-finite inputs also reach partial ceiling. Resolve
in favor of exact ceiling with a finite publication guard, preserving signatures.

For finite positive binary64 x=n/d and positive Integer scale s and increment k,
N=n*s, D=d*k, u=(N+D-1) div D and r=u*k/s. Prove (u-1)D<N<=uD and
x<=r<x+k/s. Publish y=fromRational(r) only if finite and y>=x; otherwise
return zero. Invalid x/s/k returns zero before rational conversion/division.
Integer arithmetic is unbounded subject to resources. The Double result is an
approximation, not a certified exchange-grid decimal; no fixed-width overflow.

Both local helpers must delegate to this pure core. Probe normalization rejects
zero; entry minimum retry receives Nothing for zero and retains its error.
The existing isLongSpot fallback remains unchanged: failure to establish a
tradable minimum treats any positive balance as a position. This is conservative
inventory classification, not proof of all downstream sell/exit behavior.
Unfiltered minimum metadata and full caller arithmetic remain unresolved.

A separate pure validOrderPrice predicate requires finite positive price.
At the maker price branch, false must return a pure unsent No order result;
it must not call the configured market fallback. Book-fetch failures and valid
post-only timeout/rejection fallback retain current behavior. A source-bound
one-call transition model explores validity and fallback flag combinations:
observed -> rejected (invalid) or eligible (valid). Rejected has no outgoing
order/fallback transition. Model terminality is local, not server liveness.
Compile the actual bound Main price branch against effect-recording stubs to
check both invalid and valid dispatch independently of the Boolean abstraction.
No exchange endpoint is invoked by any verification.

Requirements: F-ROUND-UP-INTEGER (SMT exact quotient/reconstruction),
F-ROUND-UP-FINITE (SMT binary64 guard and price predicate),
F-ROUND-UP-FLOW (finite gate model, no invalid-price fallback),
F-ROUND-UP-CONFORMANCE (compiled core, both actual delegates and dispatch).
Assumption A-ROUND-UP: pinned GHC/base Integer/Rational and IEEE comparisons,
ordinary IO/pure/record-update semantics and reviewed source-to-model mapping;
sufficient resources. These are not compiler proofs or whole-Main refinement.
No proof of future economics, final decimal wire caps, tick acceptance, all
minimum increases, ownership, or all venue adapters. Obligation 9 remains partial.

Preserve CE-ROUND-003: nextUp(1) under-rounding. Preserve CE-ROUND-004:
invalid maker price with fallback enabled used the market branch; now unsent.
Use 130 boundary and 4096 seeded binary64 cases (20261005), compiled exact
Fraction oracle and caller conformance, deterministic Haskell properties and
source mutation rejection. Keep prior downward obligations intact. Formal and
full wrappers must pass on frozen sources before merge; no scope relabeling.
