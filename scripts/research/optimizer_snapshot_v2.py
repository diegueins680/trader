"""Default-disabled offline optimizer with immutable, conditional publication.

Independent engineering successor. No historical runner, policy artifact or
production selector imports this module. No persistence or order authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import threading
import numpy as np

VERSION = "optimizer-snapshot-v2"
STEP_CAP = 2**31 - 1
KEYS = ("w1", "b1", "w2", "b2")
__all__ = ["VERSION", "Snapshot", "create_v2", "update_v2", "forward_v2"]


@dataclass(frozen=True, slots=True)
class Snapshot:
    version: str
    outputs: int
    step: int
    p: tuple[bytes, ...]
    m: tuple[bytes, ...]
    v: tuple[bytes, ...]


def _runtime_supported():
    return (sys.implementation.name == "cpython" and sys.version_info[:3] == (3, 13, 3) and
            np.__version__ == "2.3.5" and sys._is_gil_enabled())


def _shapes(outputs):
    return ((12, 16), (16,), (16, outputs), (outputs,))


def _pack(outputs, step, p, m, v):
    if type(outputs) is not int or outputs not in (1, 3) or type(step) is not int or not 0 <= step <= STEP_CAP:
        return None
    encoded = []
    for group_index, group in enumerate((p, m, v)):
        if type(group) is not tuple or len(group) != 4:
            return None
        buffers = []
        for value, shape in zip(group, _shapes(outputs)):
            if (type(value) is not np.ndarray or value.dtype != np.dtype("float64") or value.shape != shape or
                not np.isfinite(value).all() or (group_index == 2 and np.any(value < 0))):
                return None
            buffers.append(value.astype("<f8", copy=False).tobytes(order="C"))
        encoded.append(tuple(buffers))
    return Snapshot(VERSION, outputs, step, *encoded)


def _arrays(buffers, outputs):
    return tuple(np.frombuffer(raw, dtype="<f8").reshape(shape)
                 for raw, shape in zip(buffers, _shapes(outputs)))


def _working(buffers, outputs):
    # Native owned arrays preserve the baseline BLAS layout; snapshot bytes remain immutable.
    return tuple(value.copy() for value in _arrays(buffers, outputs))


class _Optimizer:
    __slots__ = ("_state", "_lock")

    def __init__(self, state):
        self._state = state
        self._lock = threading.Lock()

    def snapshot(self):
        return self._state

    def _publish(self, expected, candidate):
        if (candidate.step != expected.step + 1 or not 0 <= expected.step < STEP_CAP or
            candidate.outputs != expected.outputs):
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._state is not expected:
                return None
            self._state = candidate
            return candidate
        finally:
            self._lock.release()


def create_v2(seed: int, outputs: int = 3, *, enabled: bool = False) -> _Optimizer | None:
    if enabled is not True or type(seed) is not int or not 0 <= seed < 2**32 or type(outputs) is not int or outputs not in (1, 3) or not _runtime_supported():
        return None
    rng = np.random.default_rng(seed)
    p = (rng.normal(0, 1 / np.sqrt(12), (12, 16)), np.zeros(16),
         rng.normal(0, 0.01, (16, outputs)), np.zeros(outputs))
    zeros = tuple(np.zeros_like(value) for value in p)
    state = _pack(outputs, 0, p, zeros, zeros)
    return None if state is None else _Optimizer(state)


def _batch(value, width):
    if (type(value) is not np.ndarray or value.dtype != np.dtype("float64") or
        value.ndim != 2 or value.shape[1] != width or not 1 <= value.shape[0] <= 256 or
        not np.isfinite(value).all()):
        return None
    # Stable caller buffers during this copy are part of the admission contract.
    return value.copy()


def _stage(base, x, dz, lr):
    p = dict(zip(KEYS, _working(base.p, base.outputs)))
    m0 = dict(zip(KEYS, _working(base.m, base.outputs)))
    v0 = dict(zip(KEYS, _working(base.v, base.outputs)))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        h = np.tanh(x @ p["w1"] + p["b1"])
        dh = (dz @ p["w2"].T) * (1 - h**2)
        grads = {"w1": x.T @ dh, "b1": dh.sum(0), "w2": h.T @ dz, "b2": dz.sum(0)}
        if not all(np.isfinite(g).all() for g in grads.values()):
            return None
        norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        if not np.isfinite(norm):
            return None
        steps = base.step + 1
        params, moments, variances = {}, {}, {}
        for k, grad in grads.items():
            g = grad / max(1.0, norm)
            moments[k] = 0.9 * m0[k] + 0.1 * g
            variances[k] = 0.999 * v0[k] + 0.001 * g**2
            m = moments[k] / (1 - 0.9**steps)
            v = variances[k] / (1 - 0.999**steps)
            params[k] = p[k] - lr * m / (np.sqrt(v) + 1e-8)
        return _pack(base.outputs, steps, *(tuple(group[k] for k in KEYS)
                                         for group in (params, moments, variances)))


def update_v2(net: _Optimizer, x: np.ndarray, dz: np.ndarray, lr: float, *, enabled: bool = False) -> Snapshot | None:
    if enabled is not True or type(net) is not _Optimizer or not _runtime_supported():
        return None
    base = net.snapshot()
    if (type(lr) not in (int, float) or not 0 < lr <= 1 or not math.isfinite(lr) or
        base.step >= STEP_CAP):
        return None
    try:
        inputs = _batch(x, 12)
        residuals = _batch(dz, base.outputs)
        if inputs is None or residuals is None or len(inputs) != len(residuals):
            return None
        candidate = _stage(base, inputs, residuals, lr)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
    if candidate is None:
        return None
    return net._publish(base, candidate)


def forward_v2(net: _Optimizer, x: np.ndarray, *, enabled: bool = False) -> np.ndarray | None:
    if enabled is not True or type(net) is not _Optimizer or not _runtime_supported():
        return None
    base = net.snapshot()
    try:
        inputs = _batch(x, 12)
        if inputs is None:
            return None
        w1, b1, w2, b2 = _working(base.p, base.outputs)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hidden = inputs @ w1 + b1
            if not np.isfinite(hidden).all():
                return None
            out = np.tanh(hidden) @ w2 + b2
            if not np.isfinite(out).all():
                return None
        return out
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
