"""Disabled exact funding-settlement bucketing. No file, network, policy or order interface."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import re

VERSION = "funding-events-v2"
MAX_BITS = 8192
MAX_BARS = 8192
MAX_RECORDS = 65536
MAX_EVENTS = 128
MAX_TIME = 2**63
DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]{0,19})(?:\.[0-9]{1,20})?", re.ASCII)
INTEGER = re.compile(r"0|[1-9][0-9]{0,18}", re.ASCII)
__all__ = ["VERSION", "Buckets", "load_v2"]


@dataclass(frozen=True, slots=True)
class Buckets:
    version: str
    closes: tuple
    events: tuple
    per_unit: tuple


def _valid(x: object) -> bool:
    return (type(x) is F and x.numerator.bit_length() <= MAX_BITS and
            x.denominator.bit_length() <= MAX_BITS)


def _bounded(x: F) -> F:
    if not _valid(x):
        raise ArithmeticError("funding rational size limit")
    return x


def _add(a: F, b: F) -> F:
    return _bounded(a + b)


def _mul(a: F, b: F) -> F:
    return _bounded(a * b)


def _decimal(text: object) -> F:
    # Exact decimal text only: binary64 values, exponents and Unicode digits reject.
    if type(text) is not str or DECIMAL.fullmatch(text) is None:
        raise ValueError("funding decimal domain")
    sign = -1 if text.startswith("-") else 1
    whole, _, fraction = text.lstrip("-").partition(".")
    return _bounded(F(sign * int(whole + fraction), 10 ** len(fraction)))


def _time(text: object) -> int:
    if type(text) is not str or INTEGER.fullmatch(text) is None:
        raise ValueError("funding time domain")
    value = int(text)
    if value >= MAX_TIME:
        raise ValueError("funding time domain")
    return value


def _grid(closes: object) -> tuple:
    if (type(closes) is not tuple or not 1 <= len(closes) <= MAX_BARS or
        not all(type(c) is int and 0 <= c < MAX_TIME for c in closes) or
        not all(a < b for a, b in zip(closes, closes[1:]))):
        raise ValueError("funding close grid")
    return closes


def _records(rows: object) -> tuple:
    if type(rows) is not tuple or len(rows) > MAX_RECORDS:
        raise ValueError("funding record domain")
    parsed = []
    for row in rows:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError("funding record shape")
        time, rate, mark = _time(row[0]), _decimal(row[1]), _decimal(row[2])
        if mark <= 0:
            raise ValueError("funding mark")
        if parsed and time <= parsed[-1][0]:
            raise ValueError("funding order")
        parsed.append((time, rate, mark))
    return tuple(parsed)


def _bucket(closes: tuple, records: tuple) -> tuple:
    # Two-pointer sweep: j is the first close no earlier than the event time.
    events = [[] for _ in closes]
    j = 0
    for time, rate, mark in records:
        while j < len(closes) and closes[j] < time:
            j += 1
        if j == len(closes):
            raise ValueError("funding beyond final close")
        if len(events[j]) == MAX_EVENTS:
            raise ValueError("funding bucket size")
        events[j].append((mark, rate))
    return tuple(tuple(bucket) for bucket in events)


def _per_unit(events: tuple) -> tuple:
    out = []
    for bucket in events:
        total = F(0)
        for mark, rate in bucket:
            total = _add(total, _mul(mark, rate))
        out.append(total)
    return tuple(out)


def load_v2(closes: object, rows: object, *, enabled: object = False,
            version: object = VERSION) -> Buckets | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        grid = _grid(closes)
        events = _bucket(grid, _records(rows))
        return Buckets(VERSION, grid, events, _per_unit(events))
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
