# Replay event ordering — contract before verification

`replay-order-audit-v1`, 2026-10-01. Audit the unchanged Replay.step/_trade/_risk
and Execution.__post_init__ source. Classification: lifecycle, accounting order,
causality, action safety and conditional liveness. No market data or training.

## Canonical interpretation

A-SEQUENTIAL-RESEARCH-E1's phrase “every simulated fill occurs after its decision”
is ambiguous about mandatory terminal reconciliation. Existing code and the
registered environment use delayed policy targets but immediate liquidation at
the terminal endpoint. Preserve that distinction explicitly: every policy-target
fill is later than its originating decision; terminal liquidation occurs at the
endpoint after old inventory has earned that interval's price/funding. This is
not a new policy decision with privileged future information. Do not silently
change the simulator or infer that liquidation can prevent the preceding loss.

The exact existing sequence is: shield, observation admission, schedule or retain
pending target, validate next bar, book old-position P&L/funding, advance time,
check risk, conditionally fill a due pending target, recheck risk when appropriate,
then determine terminal state, clear pending, attempt solvent liquidation, mark
done, reconcile risk when appropriate and append one bar row. Early invalid gate
or invalid data returns can retain inventory/pending state and have no new row.
Those are incomplete failures, not successful liquidated episodes. Existing
training admission remains authoritative and must reject incomplete terminals.

## Obligations and abstraction

- F-RL-REPLAY-DUE: extract the pending due expression and fill predicate from
  source. For integer originating decision d and delay in {0,1}, due=d+1+delay>d;
  a policy-target fill requires no prior risk failure, a present pending target,
  and current endpoint>=due. This does not place a delay on compulsory liquidation.
- F-RL-REPLAY-ORDER: explicitly explore a source-bound single-call abstraction
  with remaining bars 1..6, horizons {1,3,6}, delays {0,1}, and initial pending
  offsets absent/1/2. Abstract each data/observation gate and each risk outcome
  nondeterministically. Check no simulated fill/mark/row before admission,
  mark-and-time-advance before risk/fill, due/risk precedence, at most one
  policy-target fill per call, terminal pending cancellation before liquidation
  or row publication, one row per advanced bar, no further bars after done,
  deadlock freedom and eventual return under terminating primitive assumptions.
  Terminal states stutter. Enumerate the complete finite reachable graph; no
  truncated search or state-space bound may be described as unbounded proof.

The abstraction records phase, elapsed bars, rows, pending due, fill count, gate
admission, risk and solvent tags, and terminal state. Risk/solvency are trusted
outcomes, not numeric theorems. Liquidation means the helper was called, not a
proof that numeric cash/quantity output is correct. No real-order capability is
modeled or introduced. An early rejected call is allowed to retain prior simulated
inventory and pending data; it cannot proceed to another simulated fill in the call.

Full source AST hashes are preregistered. Source extraction checks due/guard syntax;
mutation tests must reject changed source ordering and unsafe model transitions.
A-REPLAY-ORDER explicitly trusts stable ordinary attributes, Python control flow,
source-to-model abstraction, shield/observation/market predicates and returning
_trade/_risk/array primitives. No exceptions, concurrent writers, hidden callee
effects, crash recovery, physical deadlines or full interpreter refinement are
proved. Numeric overflow and insufficient execution realism remain blockers.

## Implementation conformance and test budget

Use actual Replay with synthetic prices/funding and causal prefix-fitted Scale.
Observe _risk/_trade calls and row append without replacing their arithmetic or
injecting exceptions. Verify the first risk check sees old units and old-position
P&L/funding, then any due target trade, then terminal liquidation and row append.
The projection checks ordering, not a universal equivalence to Python execution.

Registered grid: three targets x three horizons x two delays x three remaining
lengths x three initial-unit values x three next-bar returns = 486 traces.
Explicit additional cases cover invalid gate, invalid next market data, solvent
risk stop, insolvency, inherited pending target, partial and missed fills. Preserve
an endpoint round-trip witness: a target can fill on the terminal endpoint and
then liquidate there, with both costs recorded. This is simulator semantics,
not free profit or evidence of realistic execution. No new financial trial.

Pinned Python 3.13.3 / NumPy 2.3.5 / Z3 4.15.4. SMT uses separate SAT premise and
UNSAT violation queries, seed 0 and 10,000 ms limits. UNKNOWN fails. Record model
states, edges, depth, progress bound, all tests and any counterexample. Existing
admission, gap/accounting and lifecycle certificates remain scoped; this audit
does not close all 38 mission obligations or authorize candidate promotion.

Python statement and Boolean short-circuit semantics were checked against the
[language reference](https://docs.python.org/3.13/reference/compound_stmts.html#the-while-statement)
and [expression reference](https://docs.python.org/3.13/reference/expressions.html#boolean-operations).
They support explicit assumptions, not runtime refinement or market claims.

The model final node means an early explicit return or reaching the final reward
check. A non-finite reward can still raise there; no finite reward theorem is
claimed. Visible target-fill events count helper attempts (including zero, partial
or missed fills), not guaranteed filled quantity. Ghost observation/data/risk and
terminal-handling tags record which checks precede effects in the abstraction.
