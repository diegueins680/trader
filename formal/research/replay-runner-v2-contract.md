# Exact replay runner v2

Engineering preregistration, 2026-10-06; base 9c23673ba3e86b4e3a138bc47ecfd88d4b3513de.

## Scope and interpretation

A disabled-by-default episode runner that composes `funding-events-v2` (exact
per-bar funding buckets) with `replay-accounting-v2` (exact one-bar transition).
It replays a **fixed, caller-supplied target schedule**. It contains no policy,
learner, observation builder, OPE, artifact, file, network, promotion or order
interface. Frozen `sequential_replay_v1`, the frozen loader and all reported
evidence are unchanged.

## Inputs

- `prices`: tuple of positive exact `Fraction` completed close prices, one per
  bar of the funding grid (`len(prices) == len(buckets.closes)`, at most 8192).
- `buckets`: a `funding_events_v2.Buckets` value with the exact version.
- `left`: start bar index, an int with `0 <= left`.
- `targets`: tuple of 1..4096 targets, each in {-1/4, 0, 1/4}, with
  `left + len(targets) < len(prices)`.
- Cost parameters forwarded unchanged to `advance_v2`: fill, multiplier, impact.

## Semantics

Initial state is `initial_v2(prices[left])`. Step k = 1..H (H = len(targets))
is the transition from bar `left+k-1` to bar `left+k`. It reads exactly
`prices[left+k]`, `buckets.events[left+k]` and `targets[k-1]`, and step H is
forced terminal. This matches the frozen replay convention, where `funding[left+1]`
applies to the first transition, so bucket `left` (events at or before the start
close) is never charged. The run stops at the first terminal receipt (risk stop,
turnover limit, insolvency or step H). Termination is guaranteed within H steps.

The published `Episode` is immutable: version, start, receipts tuple and final
state. Load is all-or-nothing: an inactive call, invalid input, `None` from any
transition, or an arithmetic or resource error returns `None`.

## Obligations / methods

F-RL-RUNNER-V2-SOURCE: complete reviewed AST lock; activation first; reads of
price, events and target index only at `left + k` / `k - 1`; terminal forced at
the final step; immutable output; the only imports are the two v2 kernels.

F-RL-RUNNER-V2-ARITH: SMT. (a) Index alignment: every read index lies in
[left+1, left+H] within bounds, and step k reads no index above `left+k`.
(b) Episode telescoping: given the per-step identity already proved for
`advance_v2`, final equity equals 1 + Σgross + Σfunding − Σdebits by induction.
(c) Funding causality relative to the grid: an event charged at step k has
time in (close[left+k-1], close[left+k]], so it is no later than the decision
close of that step and strictly after the previous one.

F-RL-RUNNER-V2-FLOW: a finite model over (step, terminal, published) up to the
registered bound shows publication only of complete, terminal-ended episodes,
termination within H steps and no continuation after terminal.

F-RL-RUNNER-V2-CONFORMANCE: seeded synthetic episodes check the telescoping
identity, deterministic repetition and default rejection. Every receipt is
reconciled by the existing Haskell Rational oracle `ReplayAccountingV2.hs`. A
metamorphic test changes all prices and funding after step k and checks that
receipts 1..k are unchanged.

A-REPLAY-RUNNER-V2 trusts the assumptions of both composed kernels
(A-REPLAY-ACCOUNT-V2, A-FUNDING-EVENTS-V2) and treats a settlement's grid time as
its availability time. That is an assumption, not a provider release-time proof.
A fixed target schedule is not a policy: no observation causality, learning,
OPE, shield precedence or economic claim follows. The 38 original criteria and
scopes are unchanged.
