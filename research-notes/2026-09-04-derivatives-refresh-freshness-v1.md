# Binance derivatives refresh freshness v1 — operational result

Date: 2026-09-04.

## Finding

The installed hourly collector had completed 141 runs, but its latest schema-2
status was `partial_failure` for all ten fixed symbols. The common failure was
not an invented or non-finite feature value: the taker-volume endpoint's final
requested hourly bucket was still unavailable shortly after the corresponding
kline closed. The collector padded that bucket with an explicit unavailable
row, yet `_series_refresh_result` classified any such trailing row as
`missing_tail` before applying the already fixed two-hour freshness limit.

[Binance's official USD-M taker-volume documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume)
defines its timestamp as the period start and documents a 1,000-request per
five-minute IP limit. The corresponding
[open-interest documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
defines that timestamp as the period end. A read-only public check at Binance
server time `1788577111987` found the latest closed kline at open time
`1788570000000` and the taker bucket for that period available then; the
scheduled receipt about 46 minutes earlier had recorded the same bucket as
unavailable. No returned market value was committed.

## Correction

Source health now follows the existing family freshness boundary. A trailing
unavailable bucket is counted as `trailingUnavailable` and can coexist with
`status=ok` only while the latest finite observation is no older than the
unchanged limit. Once that finite observation exceeds the limit, the refresh
is `missing_tail`; a finite but old tail remains `stale`; an entirely empty
response remains `empty`.

This does not fill, forward-date, backdate, or reinterpret the unavailable
bucket. The first-seen ledger still records a zero-valued tombstone with
`observed=0`; the v2 aligned row remains neutral with its explicit false mask.
The schema-3 artifact verifier additionally requires integer observation,
finite, timestamp, lag, and trailing-unavailable fields; it recomputes the lag
from the bound cache tail, enforces the family limit, and verifies that the
timestamp distance equals the reported number of trailing periods.

## Live metadata-only validation

A ten-symbol public refresh with the correction updated every registered
symbol and reported all funding, OI, basis, and taker source families healthy.
Each taker response retained exactly one explicit trailing unavailable bucket;
all other families retained zero. The overall status deliberately remained
`partial_failure` because the code was uncommitted during validation and schema
3 correctly recorded `provenance_files_differ_from_commit`. That run is not an
admissible experiment receipt and no digest from it is frozen.

A second refresh after the executable changes were committed ran from
`2026-09-05T03:51:46.880601Z` through `2026-09-05T03:52:44.637228Z` at code
commit `10e3a310ddbc735ba995bf692f99ef9d66a3ad8c`. It completed `pass` for all
ten symbols with no failed symbol or provenance issue. The independent
verifier passed both the live cache and a byte-identical relocated local
archive and returned status SHA-256
`2d292d53a20b64a99684e9bbce4adf4f233d31e65f3a9eca8f83720b1bb70f00`.
The ignored archive contains 51 files: ten bar caches, forty first-seen
ledgers, and the exact status. Bar row counts range from 2,751 to 3,572;
per-symbol ledger counts are 500 funding rows, 721 OI rows, 719–721 basis
rows, and 727 taker rows. These are acquisition metadata, not forecast or
economic results. The 5.8 MiB archive remains outside Git.

No return, rank, target weight, position, PnL, risk statistic, forecast,
development result, or holdout result was calculated. No feature builder,
predictor, champion, combo, bot, order path, deployment setting, or live
authorization changed.

## Final-main receipt

After merge, commit `b24f321bf6b45cc09053e41e315cebb8da5a66cf`
produced a second complete-pass 10-symbol status from
`2026-09-05T04:06:06.975606Z` through `2026-09-05T04:07:06.009210Z`.
Verification passed in place and against a frozen relocated 51-file archive;
the exact status SHA-256 is
`83e22c3dd453ab5ee4730b5c05734c318e31a8d0cf48fd198f253c35ffe2b278`.
The committed metadata-only receipt at
`market-prediction-2026-09-04/receipts/binance-derivatives-main-2026-09-05T040706Z.json`
binds that status plus all 50 artifact hashes and row counts. Market-data bytes
remain outside Git.

Every future receipt must pass `verify-artifacts` before its bytes or returned
status digest can be frozen as acquisition evidence. Provider errors, an empty
source, an out-of-bound finite observation, malformed lag/count evidence,
missing v2 provenance, or any artifact mismatch still fail closed. This first
receipt does not authorize any outcome calculation or early prospective read.
