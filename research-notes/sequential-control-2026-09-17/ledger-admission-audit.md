# Economic ledger admission audit — 2026-09-18

Baseline: merged/deployed main `d2dabc2e66c694f47d8a0d2c821244c7075b35f0`.
Synthetic engineering fixtures only; no market archive or protected holdout was
read, no policy was trained and no financial trial was added.

## Reproduced defect

Economic reporting checked completion state and aggregate equity accounting but
trusted each recorded net return independently. Replacing every cash-path return
with a positive varying series could change Sharpe, win rate and tail metrics
without changing final equity or aggregate P&L. Intermediate equity, chronology
and offsetting gross/cost changes also escaped consistent admission.

Across 34 negative cases, baseline tests produced 24 assertion failures and three
unexpected type/key errors; seven cases were already rejected by aggregate
accounting. This does not show that a standard historical replay produced a bad
ledger. It shows an unchecked evidence boundary that callers or corrupted records
could violate. The original screen's economic claims are not recomputed.

## Repair

Before metric computation, require a list of contiguous rows from `start + 1`
through the stopped replay index, within the episode bounds. Require genuine
integer row indices, finite real values (excluding booleans) in the economic
fields, and nonnegative turnover, fees, spread, slippage and impact.

For every bar, equity equals prior equity plus gross P&L and funding less costs,
within relative/absolute tolerance 1e-10 (the existing aggregate tolerance).
Recorded net return equals equity divided by prior equity minus one, within
relative/absolute tolerance 1e-12. Prior equity must be positive. The final
recorded equity must equal replay equity exactly, as produced by Replay.

A finite nonpositive *final* equity remains reportable on an explicitly failed
path. Losses beyond initial equity are not clipped or silently dropped. Initial
rejections and partial failures with open inventory retain their existing failure
semantics. Invalid evidence raises before scores; existing runner publication
failure handling remains in force. There is no cash substitution or reconstruction
of an unobserved liquidation fill.

This checks internal consistency, not source authenticity, fill realism,
action-counter integrity or the truth of gross P&L. It does not validate unused
reward/drawdown ledger fields, prove market generalization or authorize promotion.
Existing output formulas and schema are unchanged.

## Executable evidence

Four new methods bring the suite to 106 tests:

- A fabricated positive return path over unchanged cash equity is rejected.
- Invalid field domains, chronology, row shape and offsetting ledger edits reject.
- Explicit exhausted-equity and partial-action failures retain recorded losses.
- Nonzero fee/spread/slippage/impact/funding reports at horizons 1/3/6 remain exact.

The nonzero report SHA-256 captured before repair is
`35ddd6f259ca53167ca707787af972b888878db9cdad8bc8f604d0dbe9036606`.
The earlier exact cash/legacy fixtures and terminal-accounting tests also pass.

```bash
python3 -m unittest discover -s test -p sequential_screen_test.py -k economic
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

A local 30-iteration timing of a synthetic 599-bar cash report measured median
1.093 ms before and 5.474 ms after admission (maxima 1.943/7.286 ms). Reports were
asserted equal. This measures offline reporting on a shared macOS Intel host,
not production inference or an isolated performance guarantee. It can be repeated
by replaying constant price 100, zero funding, start 30, stop 630, horizon 1,
training-only Scale fitted on the first 30 bars, and timing `economic(env)` after
termination. The PR records verification and deployment results.

`A-SEQUENTIAL-RESEARCH-R23` links the four witnesses. Haskell/Markdown risk
mitigations remain synchronized; canonical `RL-OFFLINE-001` stays HIGH/OPEN.
No dependency, configuration, model identifier, production integration or data
schema changes. No candidate passed; preserve the champion and all protected
confirmation boundaries. Continue offline research without adoption.
