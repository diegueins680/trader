# Development funding boundary audit — specified before verification

Version `funding-boundary-audit-v1`, 2026-10-01. Affected source is the unchanged
`load_development` function in `scripts/research/run_sequential_screen.py`.
Classification: data integrity, numeric/accounting correctness and causality.
This is an engineering audit with synthetic data only, not a financial trial.

The canonical environment defines funding[j] as the sum of mark times rate for
events in (close[j-1], close[j]]. The loader uses left insertion into a validated
regular close grid. Bucket zero also collects events at/before the first close;
Replay consumes only later buckets after its start, so that bucket is not a
prehistory cash debit. This interpretation does not prove provider availability.

## Obligations

- F-RL-FUNDING-GRID: for the committed development registration, each open and
  close timestamp computed by the source is strictly ordered, exactly representable
  in signed int64 and binary64, and equals the registered regular grid. All 4,910
  indices are covered by SMT integer bounds; no historical file is read.
- F-RL-FUNDING-ENDPOINT: assuming NumPy left-insertion semantics on a sorted grid,
  a finite integer event maps to bucket zero at/before the first close, to the
  unique j>0 satisfying close[j-1] < event <= close[j], or to n after the last
  close, which the source rejects. Distinct buckets cannot both contain an event.
  This is conditional timestamp arithmetic, not a proof of NumPy binary search.
- F-RL-FUNDING-FINITE: audit the stronger claim that finite rates, positive finite
  marks and ordered finite event times force all returned coefficients finite.
  Prescribed CE-RL-019 has separate product and accumulation overflow variants.
  If the current source accepts these fixtures, record the claim as refuted.
  Each test supplies matching hashes for temporary synthetic CSVs only; the
  production registration, hash gate and historical result remain unchanged.

The source function's full normalized AST is pinned before experiments. The
checker extracts the close expression and settlement loop. Integer arithmetic is
translated to SMT; searchsorted is an explicit trusted semantic assumption.
Executing the extracted loop is conformance evidence, not full-loader refinement.
Separate automation tests exercise the complete loader with synthetic byte hashes.
No model/policy can gain a capability from this audit. No behavior is replaced.

## Bounds, assumptions and verification plan

Use existing Python 3.13.3, NumPy 2.3.5, Z3 4.15.4; no new dependency. Each SMT
requirement needs independent SAT-premise and UNSAT-violation checks, seed 0,
10,000 ms timeout. All timestamps in the grid theorem remain between 0 and 2^53.
The endpoint lemma uses exact integer timestamps and arbitrary positive spacing;
the loader does not itself enforce all those generic premises on arbitrary
registrations. Full CSV dtype/schema admission remains open.

Compare the actual extracted loop against a simple linear-search reference over
32 synthetic event times (-1..30) on closes [9,19,29], plus each registered grid
close and its +/-1 millisecond neighbors (14,730 cases). Test exact-endpoint,
prehistory, after-end rejection and multi-event addition. Source mutants must
reject altered side, boundary arithmetic, range guard and accumulation. Check
both registered overflow witnesses under ignored and raised NumPy overflow;
record distinctions between Python scalar multiplication and NumPy accumulation.
Retain caller error policy. No output is silently reinterpreted as a valid signal.

Trusted assumptions A-FUNDING-BOUNDARY: stable ordinary source bindings, Python
and NumPy primitives, AST extraction, sorted exact grids and correctly matched
symbol arrays. Hashes identify bytes, not economic truth. The contract does not
prove funding publication/revision times, exchange settlement cash semantics,
replay accounting, arbitrary floating timestamp comparisons, rounding error,
provider coverage, live safety or historical incidence of the witness.

Primary semantics checked at execution time: [NumPy searchsorted](https://numpy.org/doc/2.3/reference/generated/numpy.searchsorted.html)
and [NumPy error handling](https://numpy.org/doc/2.3/reference/generated/numpy.seterr.html).
No new prediction or RL method is selected. Critical broader gaps stay open;
negative evidence cannot authorize integration or protected-data access.
