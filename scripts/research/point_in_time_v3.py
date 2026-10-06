"""Witnessed, default-disabled training entry; no historical or live authority.

Values are frozen at each bar's decision vintage. Timestamps are caller evidence,
not authenticated provider facts. This entry never silently upgrades v1/v2 data.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np
from ppo_successor_v2 import TrainingResult, train_ppo_v2

VERSION = "point-in-time-training-v3"
MAX_TIME = 2**63 - 1
MAX_RECORDS = 262144


@dataclass(frozen=True, slots=True)
class Record:
    symbol: str
    kind: str
    close: int
    revision: int
    released: int | None
    first_seen: int
    collected: int
    revised: int | None
    value: float


@dataclass(frozen=True, slots=True)
class Witness:
    symbol: str
    kind: str
    close: int
    decision: int
    revision: int
    released: int | None
    first_seen: int
    collected: int
    revised: int | None
    available: int


@dataclass(frozen=True, slots=True)
class VintageBatch:
    symbols: tuple[str, ...]
    closes: tuple[int, ...]
    decisions: tuple[int, ...]
    processing_us: int
    prices: tuple[bytes, ...]
    funding: tuple[bytes, ...]
    witnesses: tuple[Witness, ...]


@dataclass(frozen=True, slots=True)
class TrainingV3:
    version: str
    admitted: VintageBatch
    training: TrainingResult


def _time(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_TIME


def _available(record: Record, processing_us: int) -> int:
    return max(record.close, record.first_seen, record.collected,
               record.released if record.released is not None else record.first_seen,
               record.revised if record.revised is not None else record.first_seen) + processing_us


def _header(record: object, symbols: tuple[str, ...]) -> bool:
    if type(record) is not Record or type(record.symbol) is not str or record.symbol not in symbols:
        return False
    if type(record.kind) is not str or record.kind not in ("price", "funding"):
        return False
    if not all(_time(t) for t in (record.close, record.first_seen, record.collected)):
        return False
    if not record.close <= record.first_seen <= record.collected:
        return False
    if type(record.revision) is not int or not 0 <= record.revision <= MAX_TIME:
        return False
    if record.revision > 0 and record.revised is None:
        return False
    for witness in (record.released, record.revised):
        if witness is not None and (not _time(witness) or not record.close <= witness <= record.first_seen):
            return False
    return True


def _grid(symbols: object, closes: object, decisions: object, processing_us: object) -> bool:
    if (type(symbols) is not tuple or not 1 <= len(symbols) <= 8 or
        any(type(s) is not str or not 1 <= len(s) <= 32 for s in symbols) or
        len(set(symbols)) != len(symbols)):
        return False
    if (type(closes) is not tuple or type(decisions) is not tuple or
        not 121 <= len(closes) <= 4096 or len(closes) != len(decisions) or not _time(processing_us)):
        return False
    if not all(_time(t) for t in closes + decisions):
        return False
    return (all(c <= d for c, d in zip(closes, decisions)) and
            all(d < c for d, c in zip(decisions[:-1], closes[1:])))


def admit_v3(records: object, symbols: object, closes: object, decisions: object,
             processing_us: object = 0, *, enabled: object = False,
             version: object = VERSION) -> VintageBatch | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if not _grid(symbols, closes, decisions, processing_us):
        return None
    if type(records) is not tuple or not 1 <= len(records) <= MAX_RECORDS:
        return None
    try:
        selected = {}
        ambiguous = set()
        deadlines = dict(zip(closes, decisions))
        for record in records:
            if not _header(record, symbols):
                return None
            decision = deadlines.get(record.close)
            if decision is None:
                continue
            available = _available(record, processing_us)
            if available > decision:
                continue
            key = (record.symbol, record.kind, record.close)
            old = selected.get(key)
            if old is None or record.revision > old.revision:
                selected[key] = record
                ambiguous.discard(key)
            elif record.revision == old.revision:
                ambiguous.add(key)
        if ambiguous:
            return None
        values = {"price": [], "funding": []}
        witnesses = []
        for symbol in symbols:
            for kind in ("price", "funding"):
                row = []
                for close, decision in zip(closes, decisions):
                    record = selected.get((symbol, kind, close))
                    if record is None or type(record.value) is not float or not isfinite(record.value):
                        return None
                    if kind == "price" and record.value <= 0:
                        return None
                    row.append(record.value)
                    witnesses.append(Witness(symbol, kind, close, decision, record.revision,
                                             record.released, record.first_seen, record.collected,
                                             record.revised, _available(record, processing_us)))
                values[kind].append(np.asarray(row, dtype="<f8").tobytes())
        return VintageBatch(symbols, closes, decisions, processing_us,
                        tuple(values["price"]), tuple(values["funding"]), tuple(witnesses))
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def train_point_in_time_v3(records: object, symbols: object, closes: object, decisions: object,
                           horizon: object, seed: object, steps: object = 4096,
                           processing_us: object = 0, *, enabled: object = False,
                           version: object = VERSION) -> TrainingV3 | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    admitted = admit_v3(records, symbols, closes, decisions, processing_us, enabled=True)
    if admitted is None:
        return None
    try:
        prices = {s: np.frombuffer(b, dtype="<f8").astype("float64") for s, b in zip(admitted.symbols, admitted.prices)}
        funding = {s: np.frombuffer(b, dtype="<f8").astype("float64") for s, b in zip(admitted.symbols, admitted.funding)}
        trained = train_ppo_v2(prices, funding, horizon, seed, steps, enabled=True)
        if trained is None:
            return None
        return TrainingV3(VERSION, admitted, trained)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
