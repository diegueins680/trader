"""Pure transient PPO request encoding; no persistence or execution capability."""
from __future__ import annotations

import numpy as np
from optimizer_snapshot_v2 import Snapshot
from ppo_successor_v2 import TrainingResult, _configuration

VERSION = "PPO-SNAPSHOT-V3"
__all__ = ["VERSION", "encode_request_v3"]


def _words(value: np.ndarray) -> list[int]:
    if not bool(np.isfinite(value).all()) or bool(np.any(np.abs(value) > 1000)):
        raise ValueError("non-finite or unbounded tensor")
    return [int(word) for word in value.astype("<f8").view("<u8")]


def encode_request_v3(result: object, observation: object, *, enabled: object = False,
                      version: object = VERSION) -> bytes | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    if (type(result) is not TrainingResult or type(result.version) is not str or
        result.version != "ppo-successor-v2" or
        not _configuration(result.horizon, result.seed, result.steps)):
        return None
    actor = result.actor
    if (type(actor) is not Snapshot or type(actor.version) is not str or
        actor.version != "optimizer-snapshot-v2" or type(actor.outputs) is not int or
        actor.outputs != 3 or type(actor.step) is not int or
        actor.step != 4 * ((result.steps + 255) // 256) or
        type(actor.p) is not tuple or len(actor.p) != 4 or
        any(type(b) is not bytes or len(b) != size * 8
            for b, size in zip(actor.p, (192, 16, 48, 3)))):
        return None
    if (type(observation) is not np.ndarray or observation.dtype != np.dtype("float64") or
        observation.shape != (12,)):
        return None
    try:
        copied = np.frombuffer(observation.tobytes(), dtype="float64")
        obs = _words(copied)
        parameters = _words(np.frombuffer(b"".join(actor.p), dtype="<f8"))
        frame = f'("{VERSION}",{actor.step},{obs},{parameters})\n'.encode("ascii")
        return frame if len(frame) < 32768 else None
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
