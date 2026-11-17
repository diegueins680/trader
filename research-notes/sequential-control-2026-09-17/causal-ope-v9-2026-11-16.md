# Causal OPE v9 — engineering increment, 2026-11-16

Engineering evidence only: no market data was read, no financial trial was run
and no holdout was opened.

## What was added

`scripts/research/causal_ope_v9.py` is causal-ope-v8 calling ope-rational-v3
instead of ope-rational-v2. Six token occurrences differ from the reviewed v8
module. `scripts/formal/causal_ope_successor.py` binds it to v8's review and to
the certified estimator.

## Findings

1. **Full-horizon evaluation of many runs now publishes.** 48 fixture runs (24
   per learner) at 32 decisions are evaluated and match an independent exact
   oracle; v8 rejects six.
2. **No admitted evaluation is rejected for size.** Every episode the adapter
   builds lies in the estimator's certified domain, so only resource failure
   can reject after admission. A model with subnormal target probabilities,
   which v8 rejects for size, is published.
3. **Numbers are unchanged where v8 publishes.** At horizons 1, 7 and 12 every
   estimate field equals v8's.

## Evidence

- Source: equal to the hash-pinned v8 module up to six token occurrences; the
  estimator is the hash-pinned ope-rational-v3; frozen-free closure of 11
  modules.
- SMT: v8's four implications and control, two domain-inclusion implications
  and one control.
- Model: v8's 48-state stage model, three mutants reject.
- Conformance: 10 evaluations, 72 oracle episodes, 3 predecessor agreements,
  one extreme model, 19 rejections.

## What this does not show

Capacity is not statistical validity: seeds share the same synthetic outcome
path, target models are synthetic, and the estimate stays marked unreliable.
Obligation 10 stays open: network and optimizer arithmetic is guarded rather
than bounded, and frozen paths are not retired. Ledger counts stay 20 scoped
closures / 17 partial / 1 open.
