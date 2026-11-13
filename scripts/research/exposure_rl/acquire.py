"""Bounded public-data acquisition for the exposure-control RL campaign.

Only public, unauthenticated Binance USD-M history endpoints are read. The
requested window is fixed by a named partition and is refused when it overlaps
a period reserved by another registration. No credentials are read or sent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import historical_datafeed as feed  # noqa: E402

INTERVAL = "6h"
BAR_MS = 6 * 60 * 60 * 1000
HOUR_MS = feed.MARK_PRICE_INTERVAL_MS
# BTC and ETH benchmarks plus the reviewed production fleet symbols.
SYMBOLS = (
    "ARBUSDT",
    "BCHUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "NEARUSDT",
    "SOLUSDT",
)

# Inclusive first/last bar open times (UTC milliseconds).
PARTITIONS = {
    # Calendar span of the previously exposed development panel:
    # 2020-09-23T00:00Z .. 2025-03-17T06:00Z (outcomes end 2025-03-17T12:00Z).
    "development": (1_600_819_200_000, 1_742_191_200_000),
    # 2026-05-01T00:00Z .. 2026-07-16T18:00Z: claimed by no earlier registration.
    "confirmation": (1_777_593_600_000, 1_784_224_800_000),
}

# Half-open [start, end) open-time ranges that this campaign must never fetch.
RESERVED = {
    "sealed-edge-campaign-holdout": (1_742_227_200_000, 1_777_593_600_000),
    "prospective-funding-carry-window": (1_784_246_400_000, 1_800_446_400_000),
}

PANEL_COLUMNS = (
    "symbol",
    "openTime",
    "closeTime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quoteVolume",
    "trades",
    "takerBuyQuoteVolume",
)
FUNDING_COLUMNS = ("symbol", "fundingTime", "fundingRate", "markPrice", "markSource")


def window(partition: str) -> tuple[int, int]:
    """Return the registered inclusive bar window, refusing reserved overlap."""
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    start, last = PARTITIONS[partition]
    if start % BAR_MS or last % BAR_MS or last < start:
        raise ValueError("partition is not aligned to the 8h grid")
    for name, (lo, hi) in RESERVED.items():
        if start < hi and last + BAR_MS > lo:
            raise ValueError(f"partition {partition} overlaps reserved window {name}")
    return start, last


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _fetch_bars(symbol: str, start: int, last: int, deadline: float) -> list[dict]:
    """Fetch the contiguous 6h PERPETUAL bars listed inside [start, last]."""
    bars = feed._paginate_window(
        "/fapi/v1/continuousKlines",
        start_time=start,
        end_time=last,
        page_limit=feed.KLINE_MAX_PAGE_LIMIT,
        maximum_page_limit=feed.KLINE_MAX_PAGE_LIMIT,
        timestamp_key="openTime",
        parse_row=feed._parse_contract_kline,
        request_params={"pair": symbol, "contractType": "PERPETUAL", "interval": INTERVAL},
        deadline=deadline,
    )
    if not bars or bars[-1]["openTime"] != last:
        raise feed.SnapshotIntegrityError(f"{symbol} bars do not reach the window end")
    for index, bar in enumerate(bars):
        expected_open = bars[0]["openTime"] + index * BAR_MS
        if bar["openTime"] != expected_open or bar["openTime"] % BAR_MS:
            raise feed.SnapshotIntegrityError(f"{symbol} bar grid has a gap at {expected_open}")
        if bar["closeTime"] != bar["openTime"] + BAR_MS - 1:
            raise feed.SnapshotIntegrityError(f"{symbol} bar close time is off-grid")
    return bars


def acquire(partition: str, output: Path, max_seconds: int = 3600) -> dict:
    start, last = window(partition)
    outcome_end = last + BAR_MS  # exclusive close of the last bar
    funding_end = outcome_end - 1
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + max_seconds
    panel: list[dict] = []
    funding: list[dict] = []
    fallbacks = 0
    first_open: dict[str, int] = {}
    for symbol in SYMBOLS:
        bars = _fetch_bars(symbol, start, last, deadline)
        # A symbol listed after the window start enters at its first listed bar.
        listed = int(bars[0]["openTime"])
        first_open[symbol] = listed
        for bar in bars:
            panel.append({"symbol": symbol, **{k: bar[k] for k in PANEL_COLUMNS[1:]}})
        mark_start = (listed // HOUR_MS) * HOUR_MS
        marks = {
            int(row["openTime"]): str(row["open"])
            for row in feed._paginate_window(
                "/fapi/v1/markPriceKlines",
                start_time=mark_start,
                end_time=funding_end,
                page_limit=feed.KLINE_MAX_PAGE_LIMIT,
                maximum_page_limit=feed.KLINE_MAX_PAGE_LIMIT,
                timestamp_key="openTime",
                parse_row=feed._parse_mark_price_kline,
                request_params={"symbol": symbol, "interval": feed.MARK_PRICE_INTERVAL},
                deadline=deadline,
            )
        }
        events = feed.fetch_funding_events(symbol, listed, funding_end, deadline=deadline)
        previous = -1
        for event in events:
            when = int(event["fundingTime"])
            if not listed <= when <= funding_end or when <= previous:
                raise feed.SnapshotIntegrityError(f"{symbol} funding events are unordered")
            previous = when
            mark = str(event.get("markPrice", "")).strip()
            source = "funding_event_mark"
            if not mark:
                # Causal fallback: open of the 1h mark bar containing the event.
                bucket = (when // HOUR_MS) * HOUR_MS
                if bucket not in marks:
                    raise feed.SnapshotIntegrityError(f"{symbol} funding event has no causal mark")
                mark = marks[bucket]
                source = "containing_1h_mark_open"
                fallbacks += 1
            if float(mark) <= 0:
                raise feed.SnapshotIntegrityError(f"{symbol} funding mark is not positive")
            funding.append(
                {
                    "symbol": symbol,
                    "fundingTime": when,
                    "fundingRate": event["fundingRate"],
                    "markPrice": mark,
                    "markSource": source,
                }
            )
    panel_path = output / f"{partition}-panel.csv"
    funding_path = output / f"{partition}-funding.csv"
    _write_csv(panel_path, PANEL_COLUMNS, panel)
    _write_csv(funding_path, FUNDING_COLUMNS, funding)
    manifest = {
        "schema": "exposure_rl_snapshot_v1",
        "partition": partition,
        "source": feed.BASE_URL,
        "endpoints": ["/fapi/v1/continuousKlines", "/fapi/v1/markPriceKlines", "/fapi/v1/fundingRate"],
        "credentialsUsed": False,
        "interval": INTERVAL,
        "symbols": list(SYMBOLS),
        "startOpenTime": start,
        "lastOpenTime": last,
        "outcomeEndExclusive": outcome_end,
        "firstOpenTime": first_open,
        "barsPerSymbol": {k: (last - v) // BAR_MS + 1 for k, v in first_open.items()},
        "panelRows": len(panel),
        "fundingRows": len(funding),
        "fundingMarkFallbacks": fallbacks,
        "panelSha256": sha256_file(panel_path),
        "fundingSha256": sha256_file(funding_path),
        "acquiredAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "availabilityNote": "Historical timestamps only; no first-seen witnesses. Provider may revise history.",
    }
    (output / f"{partition}-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", required=True, choices=sorted(PARTITIONS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.partition, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
