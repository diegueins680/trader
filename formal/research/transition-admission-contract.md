# Training-transition admission — 2026-09-28 continuation

Specified before implementation. Preserve the frozen replay, training algorithms,
registration, datasets, artifacts and no-adoption decision. This is verification
of the existing admission rule, not a replacement simulator or new experiment.

## F-RL-TRANSITION-ADMISSION

Translate the actual `_admit_training_transition` validity assignments and branch
condition into a restricted SMT expression. Assume ordinary Replay fields, native
Boolean done flags (with other values modeled as a rejecting tag), mathematical
integer indices and binary64 reward/equity/units. Finite-vector predicates are
trusted Boolean atoms evaluated by the existing helpers.

Successful admission must imply:

1. A finite reward, finite positive equity, permitted failure category, and
   `left < t <= min(left+horizon, stop-1)`.
2. For `done is True`: absent successor, zero units, absent pending action, and
   either a recognized accounted risk failure or `t == stop-1`.
3. Otherwise: `done is False`, no failure, `t == left+horizon < stop-1`, and a
   correctly represented, correctly shaped, finite successor observation.

Check satisfiable admitted terminal and nonterminal witnesses, and UNSAT violations
of each implication. Use IEEE binary64 predicates for finite/positive/zero checks;
do not transfer a real-arithmetic equality theorem to floating-point execution.
Unknown syntax or altered control-flow structure must fail certificate admission.

## Source connection and refinement boundary

Bind the helper's full control-flow skeleton and the collector's full loop to
explicit audited AST structures. Derive the admission expressions from the actual
helper source. The collector's `env.step`, admission call and `rows.append` must
occur in that order; unsupported changes require review. This checks the admission
boundary, not every Replay transition or all learning updates.

A-TRANSITION-ADMISSION: stable ordinary Replay attributes, integer indices, binary64
scalars, built-in identity/comparison semantics, ordinary arrays and stable helper
bindings; no concurrent mutation, hostile subclasses or monkeypatching. The parser,
restricted translator, `_finite_real`, array metadata/finite checks and Python
exception/control-flow semantics are trusted. Successful actual helper calls are
mapped to satisfying assignments of the extracted validity expression. Full Python,
NumPy, compiler and simulation refinement is not established.

Zero terminal inventory and absence of pending actions are necessary admission
conditions, not a proof of correct liquidation prices, fees, complete cash flows or
market realism. The theorem does not prove profitability, finite neural updates,
preemptive timeouts or all caller behavior. Existing terminal mask/learning-target
semantics are unchanged and are not newly proved by this certificate.

## Counterexample and conformance

Preserve a deliberate mutant removing the zero-inventory terminal guard. It must
fail SMT and admit a deterministic nonzero-inventory fixture when the actual mutant
helper is compiled; the original must reject it. Label it as a mutation, not a bug
in current source or an economic experiment.

Test actual helper terminal/nonterminal admission and rejection with deterministic
boundary/non-finite fixtures. Connect to actual replay traces across all horizons
and actions, and verify that an injected terminal-inventory failure cannot reach
collector append/return. These tests supplement the proof, not replace it.

Register canonical statements, assumptions, code/test/CI traceability, source hashes,
proof results and risk limitations. Whole-mission transition/terminal-accounting
obligations can gain partial evidence, but remain blockers to candidate integration.
