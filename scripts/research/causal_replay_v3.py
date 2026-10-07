"""Prefix-only exact research replay. No policy callback, persistence or order API."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import replay_accounting_v2 as a

VERSION = "causal-replay-v3"
OBSERVATION = "causal-close-inventory-v3"
MAX_HISTORY = 8192


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    close: int
    available: int
    price: F


@dataclass(frozen=True, slots=True)
class Session:
    version: str
    history: tuple[Bar, ...]
    burn: int
    scale: tuple
    state: a.State


@dataclass(frozen=True, slots=True)
class Observation:
    version: str
    decision: int
    values: tuple[F, ...]
    supported: bool


def _bar(value: object) -> bool:
    return (type(value) is Bar and type(value.symbol) is str and 1 <= len(value.symbol) <= 32 and
            type(value.close) is int and type(value.available) is int and
            0 <= value.close <= value.available < 2**63 and a._valid(value.price) and value.price > 0)


def _history(bars: object) -> bool:
    return (type(bars) is tuple and 26 <= len(bars) <= MAX_HISTORY and all(_bar(b) for b in bars) and
            all(x.symbol == y.symbol and x.available < y.close for x, y in zip(bars[:-1], bars[1:])))


def _mean(values: tuple) -> F:
    total = F(0)
    for value in values:
        total = a._add(total, value)
    return a._div(total, F(len(values)))


def _features(bars: tuple) -> tuple:
    prices = tuple(b.price for b in bars[-25:])
    returns = tuple(a._add(a._div(y, x), F(-1)) for x, y in zip(prices[:-1], prices[1:]))
    return tuple(a._add(a._div(prices[-1], prices[-1 - h]), F(-1)) for h in (1, 3, 6, 24)) + (
        _mean(tuple(abs(r) for r in returns[-6:])), _mean(tuple(abs(r) for r in returns)))


def _fit(bars: tuple) -> tuple:
    rows = tuple(_features(bars[:stop]) for stop in range(25, len(bars) + 1))
    columns = tuple(zip(*rows))
    means = tuple(_mean(c) for c in columns)
    widths = tuple(max(F(1, 10**8), max(abs(a._add(x, -m)) for x in c)) for m, c in zip(means, columns))
    return means, widths, tuple(min(c) for c in columns), tuple(max(c) for c in columns)


def _session(value: object) -> bool:
    if (type(value) is not Session or type(value.version) is not str or value.version != VERSION or
        not _history(value.history) or type(value.burn) is not int or not 25 <= value.burn <= 4095 or
        type(value.state) is not a.State or type(value.state.tick) is not int or
        not 0 <= value.state.tick <= 4096 or len(value.history) != value.burn + 1 + value.state.tick or
        type(value.state.terminal) is not bool or value.state.price != value.history[-1].price):
        return False
    if (type(value.scale) is not tuple or len(value.scale) != 4 or
        any(type(c) is not tuple or len(c) != 6 or not all(a._valid(x) for x in c) for c in value.scale)):
        return False
    return (value.scale == _fit(value.history[:value.burn]) and
            all(w > 0 for w in value.scale[1]) and
            all(lo <= hi for lo, hi in zip(value.scale[2], value.scale[3])) and
            all(a._valid(x) for x in (value.state.price, value.state.units, value.state.equity, value.state.peak)) and
            0 < value.state.equity <= value.state.peak)


def _observe(session: Session) -> Observation:
    raw = _features(session.history)
    mean, width, low, high = session.scale
    normalized = tuple(a._div(a._add(x, -m), w) for x, m, w in zip(raw, mean, width))
    s = session.state
    inventory = (a._div(a._mul(s.units, s.price), s.equity),
                 a._add(F(1), -a._div(s.equity, s.peak)), s.equity)
    return Observation(OBSERVATION, session.history[-1].available, normalized + inventory,
                       all(lo <= x <= hi for lo, x, hi in zip(low, raw, high)))


def start_v3(bars: object, *, enabled: object = False, version: object = VERSION) -> Session | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _history(bars) or len(bars) > 4096:
            return None
        scale = _fit(bars[:-1])
        state = a.initial_v2(bars[-1].price, enabled=True)
        if state is None:
            return None
        session = Session(VERSION, bars, len(bars) - 1, scale, state)
        _observe(session)
        return session
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def observe_v3(session: object, *, enabled: object = False, version: object = VERSION) -> Observation | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _session(session) or session.state.terminal:
            return None
        return _observe(session)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def step_v3(session: object, bar: object, events: object, target: object, *,
            terminal: object = False, fill: object = F(1), multiplier: object = F(1),
            impact: object = F(0), enabled: object = False,
            version: object = VERSION) -> tuple[Session, a.Receipt] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        observation = observe_v3(session, enabled=True)
        if (observation is None or not _bar(bar) or len(session.history) >= MAX_HISTORY or
            session.history[-1].symbol != bar.symbol or
            session.history[-1].available >= bar.close or not a._valid(target) or
            target not in (F(-1, 4), F(0), F(1, 4))):
            return None
        chosen = target if observation.supported else F(0)
        receipt = a.advance_v2(session.state, bar.price, events, chosen, terminal=terminal,
                               fill=fill, multiplier=multiplier, impact=impact, enabled=True)
        if receipt is None:
            return None
        following = Session(VERSION, session.history + (bar,), session.burn, session.scale, receipt.after)
        return following, receipt
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
