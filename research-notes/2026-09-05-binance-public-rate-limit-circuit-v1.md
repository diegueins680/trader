# Binance public-data rate-limit circuit v1

Date: 2026-09-05

## Operational evidence

The hourly collector run from `2026-09-05T05:11:05.617516Z` through
`2026-09-05T05:16:35.422785Z` ended `partial_failure` with clean tracked-code
provenance. Four of ten symbols received Binance error `-1003` from the basis
endpoint. The returned ban expiries ranged from
`2026-09-05T05:12:58.020Z` through `2026-09-05T05:20:58.200Z`; the remaining
six symbols happened to complete. No raw IP address from the provider error is
retained in this note.

This is evidence of shared-IP provider throttling, not evidence that this
collector alone exhausted the quota. The egress path may carry unrelated
traffic. The old client paced futures-stat calls by 0.3 seconds and retried
HTTP errors briefly, but it did not observe the IP-wide used-weight header and
did not distinguish HTTP 418/429 or JSON `-1003` from an ordinary isolated
feature failure. It could therefore continue calling other features and
symbols after a ban was known.

Binance's official
[USD-M general information](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info)
defines IP-based request limits, response weight headers, throttling, and ban
responses. The official
[USD-M market-data catalog](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api)
is authoritative for endpoint-specific limits. Provider limits remain
changeable external state; the code uses deliberately lower local budgets and
still treats the server response as authoritative.

## Implemented boundary

- Reuse the bounded historical downloader's tested sliding-window limiter.
- Budget at most 1,200 recorded request-weight units per minute in this
  process, 450 funding requests per five minutes, and 450 `/futures/data`
  requests per five minutes.
- Reconcile `X-MBX-USED-WEIGHT-1M` into the local limiter so observed
  shared-IP traffic can delay this process before its own count reaches the
  budget.
- Encode query parameters with the standard URL encoder.
- Continue bounded retries for transient transport and HTTP 5xx failures.
- Do not retry HTTP 418/429 or Binance JSON error `-1003` in the same run.
  Convert them to sanitized typed evidence containing only status, ban-until,
  and retry-after values when available.
- Re-raise that evidence through the per-feature isolation boundary. The
  scheduler records the affected symbol as `provider_rate_limit`, marks every
  later symbol `provider_rate_limit_circuit_open`, makes no request for those
  symbols, and ends `partial_failure`.
- Extend the collector clean-commit witness to the unchanged historical file
  that supplies the limiter. Frozen historical campaign registrations and
  implementation manifests are not rewritten.

Malformed or absent rate-limit headers remain unknown rather than being
invented. A throttled run cannot pass artifact verification or become a frozen
receipt. Existing cache and first-seen evidence remain isolated from model and
trading paths, and unavailable values remain masked/neutral rather than
directional.

## Validation and limitations

Synthetic tests cover request weights, shared-IP header observation, immediate
HTTP 429 and JSON `-1003` circuit opening, raw-IP redaction, one-symbol-only
scheduler attempts, and explicit remaining-symbol skips. The existing
historical downloader tests still pass unchanged. The previously frozen
10-symbol receipt also reconstructs successfully after this change.

The process-local limiter cannot coordinate unrelated programs or machines on
the same egress IP. An hourly run may still fail before receiving a usable
weight header, and any resulting acquisition gap is irrecoverable for
first-seen evidence. Long-term mitigation still requires a stable persistent
collector, monitored gaps, and controlled egress. This change reduces harm
after throttling; it does not guarantee uninterrupted collection.

No return, rank, position, PnL, forecast metric, economic metric, holdout,
model fit, order, credential, or live-authorization state was read or changed.
