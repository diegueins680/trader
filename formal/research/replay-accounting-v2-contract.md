# Exact replay accounting v2

Engineering preregistration, 2026-10-06; base 30f76ac18fee3f02d31b1276f25fa2e42c619e83.

## Scope and interpretation

An independent, disabled-by-default one-bar futures replay transition. Frozen
sequential_replay_v1 and its reported evidence are unchanged. No new policy,
artifact selection, data loader, production consumer or order interface is added.
The unit of account is normalized initial equity. Inventory is signed base units;
prices and funding marks use quote/base units; rates and costs are dimensionless.
All admitted numbers are exact Fraction values with numerator and denominator
at most 8192 bits. Every intermediate rational operation is checked. Primitive
intermediate integer products require at most 16384 bits and an addition at most
16385 bits before reduction. Resource failures reject the whole transition.
There is no binary64 conversion or promise of compatibility with v1 rounding.

## State and transition

State is (tick, price, units, equity, peak, terminal). Tick is 0..4096, price and
peak are positive, peak >= equity, and a continuing state has equity > 0.
Initial state is (0, initial price, 0, 1, 1, false). Inputs are next completed
price, up to 128 (positive mark, signed rate) settlement events, a target in
{-1/4,0,1/4}, terminal flag, fill fraction in (0,1], cost multiplier >=0 and
impact coefficient >=0. Empty funding means no funding event, never missing data.
Callers must establish complete, causal events; absence of that evidence blocks
financial use. No source timestamp or availability theorem is claimed here.

Gross = units * (next price - price); funding = -units * sum(mark * rate).
Marked equity = old equity + gross + funding. Peak includes marked equity.
Before entry/rebalance, and after costs, test: equity >=4/5, drawdown <=3/20,
and |units*price/equity| <=7/20. A breach ends the episode. Price gaps can violate
these thresholds: detection and liquidation are guaranteed, not a capital floor.
For a continuing rebalance: desired = target*marked equity/price;
new units = old units + fill fraction*(desired-old units). Turnover greater than
half marked equity is rejected and ends the episode. A terminal input skips new
entry, cancels the proposal, and liquidates current units instead.

Fees, half-spread, slippage are respectively 5, 0.5, 4.5 basis points times
traded notional and cost multiplier. Impact = traded notional * impact
coefficient * sqrt(turnover). The square root is rounded UP to the grid 2^-32
by exact integer square root, with error <2^-32. This deliberately conservative
versioned impact assumption is not a calibrated market-impact estimate.
Terminal liquidation has full fill; an insolvent marked state cannot execute a
liquidation and records failed liquidation with outstanding inventory. Debit
that exhausts equity is accounted exactly; flat terminal inventory remains flat.

Receipt records all debit components, gross, funding, total turnover, old/new
state, termination reason, and liquidation status (not_required, flat, failed).
Identity: new equity = old equity + gross + funding - fee - spread - slippage -
impact. No reward clipping, learned penalty or discount enters accounting.
A rejected input/arithmetic/resource error returns None, leaving the immutable
input state unchanged and publishing no partial receipt. A terminal state cannot
advance. Continuation is bounded by tick4096, which forces liquidation.

## Obligations / methods

F-RL-ACCOUNT-V2-ARITH: source-derived exact accounting/debit and risk predicates;
SAT-premise/UNSAT-violation SMT queries, bounded integer arithmetic and conservative
sqrt lemmas. F-RL-ACCOUNT-V2-SOURCE: complete reviewed AST, effect/export/default
checks and source-to-expression binding. F-RL-ACCOUNT-V2-FLOW: finite transition
model of marked, rebalanced, closing, rejected and published states; no partial
publication, no continuation after termination, explicit insolvency. F-RL-ACCOUNT-
V2-CONFORMANCE: deterministic and generated exact reference cases, terminal
counterexamples, actual Haskell Rational oracle for wealth/debits; tests are not
proofs. CI runs through formal and full wrappers; no placeholders.

A-REPLAY-ACCOUNT-V2 trusts pinned CPython Fraction/int/isqrt/dataclass semantics,
solver and reviewed AST translator, GHC Rational for differential testing,
ordinary immutable exact values/stable bindings and sufficient resources.
Bit limits bound arithmetic size, not wall-clock execution or allocation recovery.
The model is finite and not whole-interpreter refinement. Full old/new replay,
observation, learner, funding-loader, latency, partial/missed real fills, margin,
position ownership and production accounting remain unverified. The 38 original
criteria and scopes are unchanged; no broad obligation closes on this kernel.
