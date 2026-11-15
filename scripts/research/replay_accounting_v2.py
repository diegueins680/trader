"""Disabled exact futures replay accounting. No data, policy or order interface."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

VERSION = "replay-accounting-v2"
MAX_BITS = 8192
MAX_TICKS = 4096
MAX_EVENTS = 128
SQRT_GRID = 2**32
__all__ = ["VERSION", "State", "Costs", "Receipt", "initial_v2", "advance_v2"]


@dataclass(frozen=True, slots=True)
class State:
    tick: int
    price: F
    units: F
    equity: F
    peak: F
    terminal: bool


@dataclass(frozen=True, slots=True)
class Costs:
    turnover: F = F(0)
    fee: F = F(0)
    spread: F = F(0)
    slippage: F = F(0)
    impact: F = F(0)


@dataclass(frozen=True, slots=True)
class Receipt:
    version: str
    before: State
    after: State
    gross: F
    funding: F
    costs: Costs
    reason: str | None
    liquidation: str


def _valid(x: object) -> bool:
    return (type(x) is F and x.numerator.bit_length() <= MAX_BITS and
            x.denominator.bit_length() <= MAX_BITS)


def _bounded(x: F) -> F:
    if not _valid(x):
        raise ArithmeticError("replay rational size limit")
    return x


def _add(a: F, b: F) -> F:
    return _bounded(a + b)


def _mul(a: F, b: F) -> F:
    return _bounded(a * b)


def _div(a: F, b: F) -> F:
    return _bounded(a / b)


def _sqrt_up(x: F) -> F:
    # n = floor(sqrt(x)*grid). Equality determines whether to round up.
    scaled = _mul(x, F(SQRT_GRID**2))
    n = isqrt(scaled.numerator // scaled.denominator)
    if n * n * scaled.denominator != scaled.numerator:
        n += 1
    return _bounded(F(n, SQRT_GRID))


def _debit(costs: Costs) -> F:
    return _add(_add(costs.fee, costs.spread), _add(costs.slippage, costs.impact))


def _merge(a: Costs, b: Costs) -> Costs:
    return Costs(*(_add(x, y) for x, y in zip(
        (a.turnover, a.fee, a.spread, a.slippage, a.impact),
        (b.turnover, b.fee, b.spread, b.slippage, b.impact))))


def _cost(old: F, new: F, price: F, equity: F, multiplier: F, impact: F) -> Costs:
    cash = _mul(abs(_add(new, -old)), price)
    turnover = _div(cash, equity)
    charged = _mul(cash, multiplier)
    return Costs(cash, _mul(charged, F(1, 2000)),
                 _mul(charged, F(1, 20000)), _mul(charged, F(9, 20000)),
                 _mul(_mul(cash, impact), _sqrt_up(turnover)))


def _risk(units: F, equity: F, price: F, peak: F) -> str | None:
    if equity <= 0:
        return "equity_exhausted"
    if equity < F(4, 5):
        return "capital_floor"
    if _add(F(1), -_div(equity, peak)) > F(3, 20):
        return "drawdown_limit"
    if abs(_div(_mul(units, price), equity)) > F(7, 20):
        return "endpoint_exposure"
    return None


def initial_v2(price: object, *, enabled: object = False,
               version: object = VERSION) -> State | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if not _valid(price) or price <= 0:
        return None
    return State(0, price, F(0), F(1), F(1), False)


def _admissible(state: object, price: object, events: object, target: object,
                terminal: object, fill: object, multiplier: object, impact: object) -> bool:
    if (type(state) is not State or type(state.tick) is not int or
        not 0 <= state.tick < MAX_TICKS or state.terminal is not False):
        return False
    if (not all(_valid(x) for x in (state.price, state.units, state.equity, state.peak,
                                    price, target, fill, multiplier, impact)) or
        min(state.price, price, state.equity, state.peak) <= 0 or state.peak < state.equity):
        return False
    if (type(terminal) is not bool or target not in (F(-1, 4), F(0), F(1, 4)) or
        not 0 < fill <= 1 or multiplier < 0 or impact < 0):
        return False
    return (type(events) is tuple and len(events) <= MAX_EVENTS and
            all(type(e) is tuple and len(e) == 2 and all(_valid(x) for x in e) and
                e[0] > 0 for e in events))


def _advance(state: State, price: F, events: tuple, target: F, terminal: bool,
             fill: F, multiplier: F, impact: F) -> Receipt:
    funding_per_unit = F(0)
    for mark, rate in events:
        funding_per_unit = _add(funding_per_unit, _mul(mark, rate))
    gross = _mul(state.units, _add(price, -state.price))
    funding = -_mul(state.units, funding_per_unit)
    equity = _add(_add(state.equity, gross), funding)
    peak = max(state.peak, equity)
    units = state.units
    reason = _risk(units, equity, price, peak)
    end = terminal or state.tick + 1 == MAX_TICKS or reason is not None
    costs = Costs()
    if not end:
        desired = _div(_mul(target, equity), price)
        new = _add(units, _mul(fill, _add(desired, -units)))
        notional = _mul(abs(_add(new, -units)), price)
        if _div(notional, equity) > F(1, 2):
            reason = "turnover_limit"
            end = True
        else:
            costs = _cost(units, new, price, equity, multiplier, impact)
            equity = _add(equity, -_debit(costs))
            units = new
            reason = _risk(units, equity, price, peak)
            end = reason is not None
    liquidation = "not_required"
    if end:
        if units == 0:
            liquidation = "flat"
        elif equity <= 0:
            liquidation = "failed"
        else:
            closing = _cost(units, F(0), price, equity, multiplier, impact)
            equity = _add(equity, -_debit(closing))
            costs = _merge(costs, closing)
            units = F(0)
            liquidation = "flat"
        reason = reason or _risk(units, equity, price, peak)
    after = State(state.tick + 1, price, units, equity, peak, end)
    return Receipt(VERSION, state, after, gross, funding, costs, reason, liquidation)


def advance_v2(state: object, price: object, events: object, target: object, *,
               terminal: object = False, fill: object = F(1), multiplier: object = F(1),
               impact: object = F(0), enabled: object = False,
               version: object = VERSION) -> Receipt | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _admissible(state, price, events, target, terminal, fill, multiplier, impact):
            return None
        return _advance(state, price, events, target, terminal, fill, multiplier, impact)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
