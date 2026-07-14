#!/usr/bin/env python3
"""Run one locked, observable refresh of the research market-data cache."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import datafeed as feed


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
)
INTERVAL = "1h"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{3,24}USDT$")
STATE_DIR_NAME = ".collector"
FRESHNESS_GRACE_MS = 300_000
DEFAULT_MAX_RUN_SECONDS = 3000
MAX_RUN_SECONDS_LIMIT = 3500


class CollectorStopped(BaseException):
    """Carry a terminal collector state out of an asynchronous signal."""

    def __init__(self, state: str, reason: str, exit_code: int):
        super().__init__(reason)
        self.state = state
        self.reason = reason
        self.exit_code = exit_code


class CollectorInterrupted(CollectorStopped):
    """Raised by SIGINT/SIGTERM."""


class CollectorTimedOut(CollectorStopped):
    """Raised when the one-shot collector exceeds its wall-clock budget."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _symbols_from_environment() -> list[str]:
    raw = os.environ.get("TRADER_RESEARCH_SYMBOLS", " ".join(DEFAULT_SYMBOLS))
    symbols = raw.replace(",", " ").split()
    if not symbols:
        raise ValueError("TRADER_RESEARCH_SYMBOLS must contain at least one symbol")
    invalid = [symbol for symbol in symbols if not SYMBOL_PATTERN.fullmatch(symbol)]
    if invalid:
        raise ValueError(
            "research symbols must be uppercase USDT contracts: "
            + ", ".join(invalid)
        )
    if len(set(symbols)) != len(symbols):
        raise ValueError("TRADER_RESEARCH_SYMBOLS contains duplicates")
    return symbols


def _max_run_seconds_from_environment() -> int:
    raw = os.environ.get(
        "TRADER_RESEARCH_MAX_RUN_SECONDS", str(DEFAULT_MAX_RUN_SECONDS)
    )
    try:
        seconds = int(raw)
    except ValueError as error:
        raise ValueError("TRADER_RESEARCH_MAX_RUN_SECONDS must be an integer") from error
    if not 1 <= seconds <= MAX_RUN_SECONDS_LIMIT:
        raise ValueError(
            f"TRADER_RESEARCH_MAX_RUN_SECONDS must be between 1 and "
            f"{MAX_RUN_SECONDS_LIMIT}"
        )
    return seconds


def _repository_commit() -> str | None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _cache_result(
    symbol: str, refresh: dict[str, object] | None
) -> dict[str, object]:
    if not refresh or refresh.get("status") != "updated":
        state = refresh.get("status") if refresh else "missing"
        raise RuntimeError(f"{symbol} refresh did not update klines ({state})")
    fresh_rows = refresh.get("freshRows")
    fresh_latest = refresh.get("freshLatestOpenTime")
    if not isinstance(fresh_rows, int) or fresh_rows <= 0 or not isinstance(fresh_latest, int):
        raise RuntimeError(f"{symbol} refresh evidence is malformed")
    now_ms = int(time.time() * 1000)
    interval_ms = feed.INTERVAL_MS[INTERVAL]
    latest_close = fresh_latest + interval_ms
    lag_ms = now_ms - latest_close
    if lag_ms < -FRESHNESS_GRACE_MS or lag_ms > interval_ms + FRESHNESS_GRACE_MS:
        raise RuntimeError(f"{symbol} newest fetched kline is stale (lagMs={lag_ms})")

    path = Path(feed._cache_path(symbol, INTERVAL))
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{symbol} cache is empty after refresh")
    timestamps = pd.to_numeric(frame["openTime"], errors="raise").astype(np.int64)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{symbol} cache timestamps are not sorted and unique")
    if int(timestamps.iloc[-1]) != fresh_latest:
        raise ValueError(f"{symbol} cache tail does not match the fetched kline")
    fields = ("funding", "oi", "basis", "taker")
    finite = {
        field: int(
            np.isfinite(
                pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
            ).sum()
        )
        for field in fields
    }
    joint = np.isfinite(
        frame.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ).all(axis=1)
    series = refresh.get("series")
    if not isinstance(series, dict):
        raise RuntimeError(f"{symbol} refresh has no derivative-series evidence")
    issues = [
        field
        for field in fields
        if not isinstance(series.get(field), dict)
        or series[field].get("status") != "ok"
    ]
    return {
        "status": "ok" if not issues else "degraded",
        "rows": len(frame),
        "latestOpenTime": int(timestamps.iloc[-1]),
        "freshRows": fresh_rows,
        "freshLatestOpenTime": fresh_latest,
        "freshnessLagMs": lag_ms,
        "refreshSeries": series,
        "issues": issues,
        "finite": finite,
        "jointFinite": int(joint.sum()),
    }


def _run_locked(cache_dir: Path, symbols: list[str], status_path: Path) -> int:
    started_monotonic = time.monotonic()
    results: dict[str, dict[str, object]] = {}
    status: dict[str, object] = {
        "schemaVersion": 1,
        "state": "starting",
        "cache": str(cache_dir),
        "interval": INTERVAL,
        "symbols": symbols,
        "results": results,
    }
    failures = 0
    try:
        started_at = _utc_now()
        status.update(
            {
                "state": "running",
                "startedAt": started_at,
                "updatedAt": started_at,
                "commit": _repository_commit(),
            }
        )
        _write_json_atomic(status_path, status)
        for symbol in symbols:
            try:
                refreshes = feed.update_cache(
                    [symbol], INTERVAL, acquire_lock=False
                )
                results[symbol] = _cache_result(symbol, refreshes.get(symbol))
                if results[symbol]["status"] != "ok":
                    failures += 1
                    print(
                        f"{symbol}: derivative refresh degraded "
                        f"({', '.join(results[symbol]['issues'])})",
                        file=sys.stderr,
                    )
            except Exception as error:
                failures += 1
                results[symbol] = {
                    "status": "error",
                    "error": str(error).replace("\n", " ")[:240],
                }
                print(f"{symbol}: refresh failed ({results[symbol]['error']})", file=sys.stderr)
            status["results"] = results
            status["updatedAt"] = _utc_now()
            _write_json_atomic(status_path, status)

        finished_at = _utc_now()
        status.update(
            {
                "state": "pass" if failures == 0 else "partial_failure",
                "finishedAt": finished_at,
                "updatedAt": finished_at,
                "durationSeconds": round(time.monotonic() - started_monotonic, 3),
                "failedSymbols": [
                    symbol
                    for symbol, result in results.items()
                    if result["status"] != "ok"
                ],
                "results": results,
            }
        )
        _write_json_atomic(status_path, status)
        print(
            f"research data collector {status['state']}: "
            f"{len(symbols) - failures}/{len(symbols)} symbols refreshed"
        )
        return 0 if failures == 0 else 1
    except CollectorStopped as stopped:
        stopped_at = _utc_now()
        status.update(
            {
                "state": stopped.state,
                "finishedAt": stopped_at,
                "updatedAt": stopped_at,
                "durationSeconds": round(time.monotonic() - started_monotonic, 3),
                "stopReason": stopped.reason,
                "failedSymbols": [
                    symbol
                    for symbol, result in results.items()
                    if result["status"] != "ok"
                ],
                "results": results,
            }
        )
        _write_terminal_status(status_path, status)
        return stopped.exit_code


def main() -> int:
    try:
        symbols = _symbols_from_environment()
        max_run_seconds = _max_run_seconds_from_environment()
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    cache_dir = Path(
        os.environ.get("TRADER_RESEARCH_CACHE") or feed.CACHE_DIR
    ).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    feed.CACHE_DIR = str(cache_dir)
    state_dir = cache_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "last-run.json"
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    }
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGALRM, _deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, max_run_seconds)
    try:
        try:
            with feed.cache_writer_lock(blocking=False):
                return _run_locked(cache_dir, symbols, status_path)
        except feed.CacheWriterBusy:
            print(f"research data collector already running for {cache_dir}")
            return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def _interrupt(signum: int, _frame: object) -> None:
    raise CollectorInterrupted("interrupted", signal.Signals(signum).name, 130)


def _deadline(_signum: int, _frame: object) -> None:
    raise CollectorTimedOut("timeout", "wall_clock_deadline", 124)


def _write_terminal_status(path: Path, status: dict[str, object]) -> None:
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    previous_handlers = {
        handled_signal: signal.getsignal(handled_signal)
        for handled_signal in handled_signals
    }
    try:
        for handled_signal in handled_signals:
            signal.signal(handled_signal, signal.SIG_IGN)
        _write_json_atomic(path, status)
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
