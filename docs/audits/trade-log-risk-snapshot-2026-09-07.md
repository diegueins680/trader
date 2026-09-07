# Trade-log risk-state audit — 2026-09-07

## Decision

Close `TRADE-LOG-GAP-002` by assigning exact live risk-state ownership to an
additive schema-1.2 event record. Preserve schema-1.1 backtest records and every
legacy live field unchanged. Do not fabricate a backtest daily/weekly snapshot
from insufficient close-only evidence.

This is an observability correction, not a model, strategy, threshold, order,
promotion, deployment, or live-authorization change.

## Finding before the change

The risk was opened on 2026-05-27 as “Missing native drawdown, daily/weekly
loss, expectancy fields.” The imported contract was dated 2026-05-24, labeled
itself v1.0, and described fields and cost-attribution objects that the current
NDJSON writer does not emit. Production had subsequently advanced its two
version aliases to `1.1` and added a typed `exitReason`, but still emitted no
risk-state fields.

The omission could not be repaired by a trustworthy downstream derivation:

- live rows exist only for applied position OPEN/CLOSE transitions, not every
  marked bar;
- those compatibility rows intentionally store zero quantity, P&L, and fees;
- the file did not contain peak, day-start, or week-start equity references;
- it did not preserve whether expectancy was configured, sufficiently
  observed, or unavailable; and
- `timestamp` is processing time for live events, while the causal market-event
  time was absent.

Consequently, exact drawdown, daily loss, weekly loss, or decision-time
expectancy could not be reconstructed from the prior log. Schema ownership is
required.

The audit also confirmed that legacy live `exitReason` contains the event type
`OPEN` or `CLOSE`, not the actual close cause. Changing that existing meaning
would break consumers. Version 1.2 therefore preserves it and adds nullable
`closeReason`/`close_reason` aliases sourced from the canonical close reason.

## Implemented contract

`Trader.TradeLogRiskState` now owns the live encoder and schema version. An
applied live event carries equal `riskState` and `risk_state` objects with:

- separate market-event and processing timestamps;
- evaluated equity and its peak/day/week reference equities;
- UTC epoch day and seven-day bucket keys;
- exact drawdown, daily-loss, and weekly-loss values supplied to the canonical
  pre-execution halt decision;
- configured expectancy lookback, finite observation count, required/available
  status, and nullable evaluated expectancy; and
- the pre-execution halt reason.

The snapshot is explicitly labeled `pre_execution_decision`. It excludes the
current order's later execution fee and does not claim post-fill accounting.
Any non-finite or out-of-domain numeric input becomes JSON `null`. Non-finite
input sets `finite=false`; invalid domains or unavailable required expectancy
set `valid=false`. Unavailable expectancy remains `null`, never an invented
zero.

Backtest closed-trade rows remain version 1.1. The simulator's internal risk
state is not carried in `BacktestSummary`; reconstructing boundary-sensitive
daily and weekly values afterward would be an approximation. A future backtest
schema version may add an exact snapshot only by carrying it natively from the
simulation transition.

## Verification evidence

The regression suite binds:

1. every finite snapshot field to exact input values;
2. NaN/infinite evidence to `null` plus `finite=false` and `valid=false`;
3. unavailable required expectancy to `null` plus `valid=false`, while
   remaining distinct from non-finite evidence;
4. absence of `NaN` and `Infinity` tokens in encoded JSON;
5. both schema aliases at `1.2`;
6. equal camel/snake risk-state and close-reason aliases; and
7. continued presence of all 24 legacy v1.1 live fields; and
8. tested transition classification that attaches the canonical cause only to
   a completed transition to flat while preserving legacy reversal behavior.

The production `Main.hs` call site uses the tested transition classifier and
compiles against the typed encoder, so a missing close reason or risk-state
argument is a build failure. Canonical formal verification binds the module and
regressions to `H-EXECUTION`.

Commands recorded for this change:

```sh
cd haskell && cabal build trader-tests trader-hs
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

Only commands that complete successfully are reported as passing in the pull
request and final handoff.

Final local results on 2026-09-07:

- `cabal test trader-tests --test-show-details=direct`: pass;
- `bash scripts/verify.sh haskell`: pass; and
- `bash scripts/verify.sh full`: pass after correcting a production-only
  `Int`/`Double` classifier signature mismatch and one HLint `isNothing` hint
  exposed by earlier failed full runs.

The passing full run included 241 web tests, 161 root automation/research
tests, and a formal registry with 38 specifications, 324 named features, 246
clauses, 262 implementation files, 103 evidence links, and 32 canonical risks.

## Remaining limitations

- A live event row is not exchange fill evidence. Its legacy quantity/P&L/fee
  values remain compatibility markers and must not be used for accounting.
- Reversals retain their existing logging behavior; this change does not create
  new event rows or modify execution.
- UUID generation intentionally prevents byte-identical log files. Tests cover
  deterministic schema and value semantics instead.
- Existing consumers that reject unknown schema versions must explicitly admit
  the additive 1.2 record before consuming new live rows.
- No historical log is rewritten or upgraded.
