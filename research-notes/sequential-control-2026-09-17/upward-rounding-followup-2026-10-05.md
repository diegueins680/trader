# Upward rounding and numeric maker rejection — 2026-10-05

Baseline 906ad455bef2b6706bd44667679596aa75766aae. Registration commit 6c5889eb
preceded implementation. No financial trial, data acquisition, holdout evaluation,
order, production configuration change or deployment.

CE-ROUND-003: nextUp(1) = 1.0000000000000002 used to produce 1 on a unit grid.
Both local Main helpers now delegate to exact rational ceiling and a checked
binary64 publication boundary. CE-ROUND-004: invalid maker price previously
called the configured market fallback. The invalid branch now returns an unsent
No order result. Other fallback reasons are unchanged.

The independent Fraction oracle covers 130 boundary and 4096 generated binary64
cases (seed 20261005), including non-finite, subnormal, extreme and malformed-grid
inputs. Both actual Main delegates are compiled. The actual invalid-price branch
is compiled against effect-recording stubs, alongside its preserved legacy branch;
no exchange call occurs. Haskell properties exercise another 1000 generated words
and explicit edge values across eight grids. These are conformance tests, not proofs.

Four SAT-premise/UNSAT-violation checks cover exact ceiling/reconstruction and
binary64 admission. The finite model covers one price dispatch, two validity
classes and both fallback flags: 8 states, 4 initial, 4 terminal, 4 edges, depth 1.
Terminality is local; no server liveness or complete order authorization theorem.
Pinned runtime numeric primitives and source-to-model mapping remain assumptions.

The minTradeQty zero fallback remains Nothing; entry minimum retry preserves its
original error. isLongSpot still treats any positive balance as a position when
a minimum cannot be established. This conservative inventory classification is
not a proof of downstream exit behavior. Other unfiltered metadata, decimal wire
rounding, exchange grid acceptance and complete order-cap composition remain open.

Broader status: 5 scoped closures, 27 partial, 6 open. Obligation 9 remains partial.
The frozen financial screen and its failed acceptance/OPE gates are unchanged;
final returns remain sealed. No candidate adoption or RL promotion is justified.

Validation in progress: local downward tests passed (2 tests, 31.697 seconds);
local upward tests passed (2 tests, 12.485 seconds). Canonical wrapper results
and final reviewed proof-source identity will be recorded after reproduction.

Local full receipt reproduction (`python scripts/formal/verify.py --record`)
failed in the unchanged PPO bridge probe: `--snapshot-contract-v3` exceeded
its existing three-second subprocess timeout. No receipt was written and no
timeout was widened. Reproduce frozen sources on the pinned CI runner; local
targeted rounding passes do not imply the complete formal wrapper passed.
