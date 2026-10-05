# Downward quantity rounding: exact core v1

Baseline 1911aca834df47849d394913e8660d35a892e054. Engineering repair,
not a new predictor or financial experiment. No order calls or deployment.

Conflict: the inherited quantizeDown adds 1e-9 before floor. With scale=step=1
and x=nextDown(1), it returns 1 > x, contradicting G-FINITE and the downward
rounding intent. Invalid Step constructors and non-finite x also reach partial
floor/division. The authoritative behavior is non-increasing finite output;
the undocumented upward tolerance must be removed.

For positive finite binary64 x and positive Integer scale s and increment k,
let x=n/d be its exact rational value, d>0; u=(n*s) div (d*k), r=u*k/s.
Compute y=fromRational(r). Publish y only when finite and 0<=y<=x; otherwise
return positive zero. Invalid x/s/k returns zero before any rational conversion
or division. No floating multiplication, epsilon, or fixed-width integer math.
No cap, permission, live configuration, order constructor, or model ID changes.
Public Binance Step and quantizeDown types remain compatible; Binance delegates
to a small base-only pure core. Decimal boundary behavior intentionally becomes
conservative: binary64 0.3 lies below rational 3/10 and may lose one step.

F-ROUND-DOWN-INTEGER: SMT verifies Euclidean quotient bounds for N>=0,D>0
and exact reconstruction under rational abstraction. F-ROUND-DOWN-FINITE:
SMT binary64 verifies the actual publication guard for arbitrary intermediate
y, including NaN/infinity, and invalid-input zero fallback. Source review/binding
and compiled differential tests connect these lemmas to the Haskell core; this
is not a proof of the GHC compiler or its Rational conversion primitives.
F-ROUND-DOWN-CONFORMANCE: compiled actual core and Binance delegation match
an independent exact Fraction oracle over deterministic boundary, malformed,
extreme/subnormal and generated bit-pattern cases; production Haskell property
tests assert finiteness/non-increase and adapter equality. Source mutants reject.

A-ROUND-DOWN: pinned GHC/base IEEE binary64 comparisons, isNaN/isInfinite,
Integer, toRational/fromRational and division semantics; sufficient runtime
resources for finite Integer inputs. No arithmetic-overflow or conversion-accuracy
claim is inferred from a real-number proof. Final guard bounds Double output
independently of conversion rounding. No bounded latency for adversarial huge
Integers. Tests do not constitute proofs.

Broader obligation 9 remains unresolved: two Main quantizeUp helpers, minimum
quantity/notional increases, side-specific prices, other venue adapters and
8-decimal renderDouble wire rounding are not verified by this repair. Output
Double may approximate the exact grid rational. No exchange acceptance, complete
order exposure, or final decimal wire-cap theorem is claimed. Existing lifecycle
models remain unchanged; this pure helper has no lifecycle or capability effects.

Validation: pinned SMT, source mutation rejection, compiled golden/generated
conformance, existing Haskell tests, bash scripts/verify.sh formal and full.
Counterexample CE-ROUND-001 must remain reproducible. No financial trials,
market data, final holdout, network calls or policy artifacts are required.
