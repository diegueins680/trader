# Sizing input admission v1

Baseline a3cfc824af48ab5799b728538ffcfe3df2c37e06. Engineering repair;
no financial experiment, endpoint invocation, configuration or deployment.

Conflict: Main normalizeQty/normalizeProbeQty skip notional checks for an
invalid or unavailable price; NaN quantity bounds bypass comparisons. Probe
minimum expansion can return infinity without a grid, and validateProbeQuote
can return a non-finite amount. Numeric order-constructor guards are later
boundaries and do not establish sizing correctness or minimum-retry safety.

Before sizing arithmetic, validate every present effective minimum/maximum
quantity and minimum notional as finite and nonnegative. When both quantity
bounds exist require minimum <= maximum. Zero retains its existing comparison
semantics; missing optional filters retain existing meaning. Every supplied
price must be finite and strictly positive. A missing price rejects when a
positive minimum-notional filter requires it. Do not manufacture a price or
turn a malformed present value into an absent filter.

Use small pure validators in QuantityRounding. Keep its existing raw-quantity
and grid preflight first, then validate metadata/price before either normalizer
uses them. New error messages must not match Main's minimum-size retry classifier.
normalizeEntryQty must propagate invalid-input errors without reaching
minTradeQty. Probe normalization also checks the promoted quantity's finiteness
before publication. Quote-only probe sizing validates its raw amount and minimum
notional without requesting or manufacturing a base-asset price. Preserve all
finite outputs for previously admissible inputs, all normal minimum-size policy,
public API/CLI/JSON signatures, configured limits and wire formatting.

SMT obligations: over binary64 and Maybe-presence Booleans, accepted metadata
has finite nonnegative bounds, ordered present quantity bounds, finite positive
present prices, and required price availability. Prove all new validation
messages non-retryable under the actual classifier's string predicate. Prove
conditional quantity publication finite/positive and within present bounds
using the actual final comparison semantics. Do not reinterpret binary64
notional multiplication as an exact-real accounting proof.

Machine-check a finite minimum-retry control model: invalid quantity/metadata
admission is terminal and cannot reach minimum expansion; only the existing
three too-small error classes permit retry. Record actual state count, edge
count and depth. This is local pure sizing control, not order authority or
whole-server lifecycle.

Compile actual current and preserved Main sizing functions against pure Step
and SymbolFilters representations and production rounding/validation kernels.
Use generated properties, an independent Python admission oracle and
bit-preserving old/current compatibility checks. Capture pre-change witnesses
for invalid price, NaN maximum quantity, missing required price, overflowed
probe expansion and non-finite quote amounts. No exchange module/effect is
linked into the driver.

Budget: 13 binary64 boundary words substituted into five numeric slots; all
16 metadata/price presence masks, both filter-presence states and both no-grid/
valid-grid states (4160 boundary rows); 2048 generated rows, seed 20261005;
plus explicit counterexample fixtures. No data, training, checkpoint or holdout
access. Formal/full wrappers and final CI must pass on frozen sources.

Assumptions: pinned GHC/base binary64/Maybe/Either/Integer/Rational semantics,
ordinary immutable non-bottom inputs, source extraction/model correspondence,
and adequate resources. Source binding and property tests are not compiler or
whole-Main refinement proofs. Filter parsing, provider freshness/completeness,
missing optional filters, direct minTradeQty consumers, notional rounding,
wire caps/tick membership and other venue adapters remain outside this repair.
No new broad closure: obligations 6/7/9/10/21 remain at their existing status.

## Preregistered negative-input amendment

Before material implementation, extend raw-quantity admission to reject finite
negative quantities using the existing non-retryable Invalid quantity input
error. Clamping a negative value to zero previously allowed minimum-size retry
to create a positive quantity. Zero and negative zero retain their existing
minimum-size semantics. Preserve the historical validator alongside Main
fragments; add CE-SIZING-006 and check the real retry path. Accepted raw inputs
are finite and nonnegative; update the binary64 SMT premise and independent
conformance oracle. Budget becomes 6208 registered rows plus six witnesses.
This remains a local engineering repair with no broad-obligation closure.
