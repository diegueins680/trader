# Adopted-champion screen v1 — retrospective result

Run date: 2026-10-09. Registration: [`adopted-champion-screen-v1.json`](../registrations/adopted-champion-screen-v1.json) (sha256 `92311fad…beae1d`), committed and pushed in `53df64be` before any evaluation-window result existed.

## Verdict

**FAIL — fleet REJECTED (3 of 3 evaluable combos), reason `no_trades`.**

Replayed through the live adoption path on 6.5 months of data the optimizer never saw, the live fleet's UNI, SUI and ETC combos never open a position, on any seed or chunk. They earn nothing and cost nothing, so there's no edge to protect. By design the retrospective phase could not pass. It also doesn't evaluate AVAX or ADA (see Scope).

Recommendation (for human decision; this campaign authorizes nothing): treat the pinned fleet as **not having a demonstrated edge**. The UNI/SUI/ETC workers are flat by construction, so they cost nothing but also earn nothing. The real exposure is ADA, plus AVAX if it trades, and neither has any lawful out-of-sample evidence. Don't raise the 1% adoption cap. Before the next pin review, re-select combos under gates that reject degenerate thresholds (below).

## What was tested

| Combo | Window (bar opens, UTC) | Bars | Chunks | Adopted per-side cost | Adopted max size |
|---|---|---|---|---|---|
| UNIUSDT 4h `da78765a` (Kalman, long-flat) | 2026-03-25 20:00 → 2026-10-08 20:00 | 1183 | 7 | 11.41 bps | 0.648 |
| SUIUSDT 6h `62c95834` (LSTM, long-short) | 2026-03-25 18:00 → 2026-10-08 18:00 | 789 | 8 | 10.21 bps | 0.691 |
| ETCUSDT 4h `76a9c22a` (LSTM, long-short) | 2026-03-25 20:00 → 2026-10-08 20:00 | 1183 | 10 | 9.44 bps | 0.766 |

How the replay works:
- **Same parameters as live.** Each frozen combo goes through `trader-hs --adopt-combo-file/--adopt-combo-uuid`, which applies `applyTopComboForStart`, the function live bots use. That covers parameters, stored thresholds and venue cost floors.
- **Causal refits.** The model is refit on the combo's own `bars` window ending before each chunk. Parameters never change.
- **Coverage.** Seeds 42 (the live seed), 1, 2, 3 and 4. Data is Binance USD-M closed klines, hash-pinned in [`retrospective-manifest.json`](../../data/research/adopted-champion-screen-v1/retrospective-manifest.json).

## Results

Every seed of every combo made 0 round trips, so net return, drawdown and Sharpe are all 0 and PSR is 0.5. The post-pin subwindow (from 2026-08-26) is also all zero.

The baselines below use identical rows, the adopted size and the adopted per-side cost. They are context only, not decision inputs.

| Combo | Buy-and-hold net / Sharpe / MaxDD | 20-bar momentum net / Sharpe / MaxDD |
|---|---|---|
| UNIUSDT | +60.3% / 1.77 / 28.6% | +45.0% / 1.65 / 24.2% |
| SUIUSDT | +9.3% / 0.57 / 38.8% | +17.5% / 0.82 / 27.1% |
| ETCUSDT | −4.0% / 0.07 / 31.5% | −30.6% / −1.20 / 48.7% |

Prices moved enough to trade. The strategies simply never act.

## Why they never trade (post-registration diagnosis, not a decision input)

On the final chunk of each combo:

| Combo | Stored open threshold (per bar) | Forecast edge \|pred/price − 1\| median / max | Latest action |
|---|---|---|---|
| UNIUSDT (Kalman) | 0.49% (cost-aware min edge 0.27%) | 0.10% / 0.15% | HOLD (Kalman neutral) |
| SUIUSDT (LSTM) | **38.6%** | 34.7% / 42.0% (forecast far from price) | HOLD (LSTM neutral) |
| ETCUSDT (LSTM) | **91.8%** | 89.4% / 90.5% (forecast far from price) | HOLD (LSTM neutral) |

- **UNI:** a sane forecaster whose predicted move never reaches its own cost hurdle.
- **SUI and ETC:** degenerate. Their LSTM forecasts sit 35–90% away from the price, and the optimizer's threshold sweep fitted thresholds of the same size. With adoption's sanity and confidence gates the result is permanent neutrality.

The thresholds are stored on the combos themselves (`openThreshold`/`closeThreshold` in the frozen snapshot), not introduced by the replay.

The production board already flags all five pinned combos as tier `candidate`, with `min-edge-below-floor` and `walk-forward-missing`. They reached live through `TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS` with relaxed gates.

Live corroboration: in the production snapshot only ADA (stored threshold 0.13%) has a `metrics_json.live` record. Its operational record, which is not evidence here, shows 22 operations from 2026-07-21 to 2026-09-20 ending at equity 1.0009. UNI, SUI, ETC and AVAX have none.

**Gate change worth considering:** reject any combo whose stored open threshold exceeds a few multiples of its round-trip cost, or whose forecasts track price this badly. `predictorLiveness` already measures tracking.

## Scope and limitations

- **AVAX and ADA excluded.** They are in the sealed 1,227-return holdout universe (through 2026-05-01) and in the funding-carry prospective window (through 2027-01-20 13:00). From 2027-01-21 the frozen champion is already a registered HAR-RV baseline. ADA, the one combo that actually trades, therefore has no lawful out-of-sample evaluation today.
- **Upward selection bias.** Rows before the 2026-08-26 pin informed the pin decision, which biases the window upward. That only strengthens a rejection.
- **Not modeled.** Maker-first (GTX) entries are not modeled; the replay charges adopted taker-floor costs. Costs × 2 and a one-bar delay were not run. With zero trades none of these can change the outcome.
- **Chunk resets.** Each chunk starts flat, which differs slightly from a continuously running live bot.
- **Trial count.** It is only a lower bound (3,261 combos on the board); the true count is unrecorded.
- **Process incident** (disclosed in the registration before commit): a dry run accidentally started on the real window. It was killed before writing or showing any result.

## Gate status and next steps

- `economic_evidence` stays **open**. Nothing here can satisfy it.
- Proposed ledger blocker text (not applied: recording the formal certificate needs maintainer approval):
  > Matched adopted-champion replay (research-notes/registrations/adopted-champion-screen-v1.json) rejected the live UNI/SUI/ETC combos on lawful post-creation data: zero round trips from 2026-03-25 to 2026-10-09 because their stored open thresholds (0.49%, 38.6%, 91.8% per bar) exceed every forecast. Its one-shot prospective phase cannot be read before 2027-01-21; AVAX/ADA remain unevaluated under the holdout and carry rules.
- **Prospective phase** (one shot, 2026-10-10 → 2027-01-21, readable from 2027-01-21 06:00 UTC). Before then, commit the 2×-cost and one-bar-delay stress implementation, tested only on pre-`createdAtMs` data. Then run:
  1. `python3 scripts/research/champion_screen.py fetch --end-ms 1800489600000 --manifest prospective-manifest.json`
  2. `python3 scripts/research/champion_screen.py run --phase prospective`

  Run nothing on that window earlier. If the fleet is re-pinned before then, these combos are no longer the live champion. The phase still runs as registered, but it then only describes the retired fleet.

## Reproduce

```bash
cd haskell && cabal build exe:trader-hs && cd ..
# champion-snapshot.json: the five pinned UUIDs from s3://<bucket>/trader-prod/optimizer/top-combos.json (not committed; repo is public)
python3 scripts/research/champion_screen.py fetch --manifest retrospective-manifest.json   # must reproduce the pinned hashes
TRADER_HS_BIN=$(cd haskell && cabal list-bin trader-hs) python3 scripts/research/champion_screen.py run --phase retrospective
```
