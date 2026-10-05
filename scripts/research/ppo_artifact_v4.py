"""Pure default-disabled PPO artifact bytes; no persistence or authority effects."""
from __future__ import annotations

import hashlib
import json
import math
import re
import numpy as np
from optimizer_snapshot_v2 import Snapshot
from ppo_successor_v2 import TrainingResult, _configuration
from ppo_inference_v3 import encode_request_v3

VERSION = "ppo-artifact-v4"
CONTRACTS = ["ppo-successor-v2", "optimizer-snapshot-v2", "sequential_replay_v1",
             "close_inventory_v1", "signed-quarter-v1", "reward-v1", "PPO-SNAPSHOT-V3"]
__all__ = ["encode_artifact_v4", "decode_artifact_v4", "request_from_artifact_v4"]


def _check(ok: bool) -> None:
    if not ok:
        raise ValueError("invalid PPO artifact")


def _provenance(value: object) -> dict:
    fields = {"codeCommit": 40, "registrationSha256": 64, "dataSha256": 64,
              "fundingSha256": 64, "splitSha256": 64, "proofSha256": 64}
    _check(type(value) is dict and set(value) == set(fields) | {"datasetRole"})
    _check(type(value["datasetRole"]) is str and value["datasetRole"] == "development")
    for name, width in fields.items():
        _check(type(value[name]) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value[name]) is not None)
    return value.copy()


def _buffer(value: object, count: int, nonnegative: bool = False) -> bytes:
    _check(type(value) is str and len(value) == 16 * count and re.fullmatch("[0-9a-f]+", value) is not None)
    raw = bytes.fromhex(value)
    vector = np.frombuffer(raw, dtype="<f8")
    _check(bool(np.isfinite(vector).all()) and (not nonnegative or bool((vector >= 0).all())))
    return raw


def _snapshot(value: object, outputs: int, step: int) -> Snapshot:
    _check(type(value) is dict and set(value) == {"version", "outputs", "step", "p", "m", "v"})
    _check(type(value["version"]) is str and value["version"] == "optimizer-snapshot-v2" and
           type(value["outputs"]) is int and value["outputs"] == outputs and
           type(value["step"]) is int and value["step"] == step)
    groups = []
    for name in ("p", "m", "v"):
        _check(type(value[name]) is list and len(value[name]) == 4)
        groups.append(tuple(_buffer(b, n, name == "v")
                            for b, n in zip(value[name], (192, 16, 16 * outputs, outputs))))
    return Snapshot(value["version"], outputs, step, *groups)


def _restore(value: object) -> TrainingResult:
    _check(type(value) is dict and set(value) == {"version", "seed", "horizon", "steps", "symbols", "scale", "actor", "critic", "losses"})
    _check(type(value["version"]) is str and value["version"] == "ppo-successor-v2" and
           _configuration(value["horizon"], value["seed"], value["steps"]))
    symbols = value["symbols"]
    _check(type(symbols) is list and 1 <= len(symbols) <= 8 and
           all(type(s) is str and 1 <= len(s) <= 32 for s in symbols))
    _check(symbols == sorted(set(symbols)))
    _check(type(value["scale"]) is list and len(value["scale"]) == 4)
    scale = tuple(_buffer(b, 6) for b in value["scale"])
    step = 4 * ((value["steps"] + 255) // 256)
    actor, critic = _snapshot(value["actor"], 3, step), _snapshot(value["critic"], 1, step)
    losses = value["losses"]
    _check(type(losses) is list and len(losses) == step and all(type(v) is str and len(v) <= 32 for v in losses))
    parsed = tuple(float.fromhex(v) for v in losses)
    _check(all(math.isfinite(v) and v.hex() == text for v, text in zip(parsed, losses)))
    return TrainingResult(value["version"], value["seed"], value["horizon"], value["steps"],
                          tuple(symbols), scale, actor, critic, parsed)


def _pack_snapshot(value: object) -> dict:
    _check(type(value) is Snapshot and type(value.outputs) is int and value.outputs in (1, 3))
    groups = {}
    for name in ("p", "m", "v"):
        group = getattr(value, name)
        _check(type(group) is tuple and len(group) == 4 and all(type(b) is bytes and len(b) == n * 8 for b, n in
               zip(group, (192, 16, 16 * value.outputs, value.outputs))))
        groups[name] = [b.hex() for b in group]
    return {"version": value.version, "outputs": value.outputs, "step": value.step, **groups}


def _pack(value: object) -> dict:
    _check(type(value) is TrainingResult and type(value.symbols) is tuple and
           type(value.scale) is tuple and len(value.scale) == 4 and
           all(type(b) is bytes and len(b) == 48 for b in value.scale) and
           type(value.losses) is tuple and 4 <= len(value.losses) <= 64 and
           all(type(v) is float and math.isfinite(v) for v in value.losses))
    result = {"version": value.version, "seed": value.seed, "horizon": value.horizon,
              "steps": value.steps, "symbols": list(value.symbols), "scale": [b.hex() for b in value.scale],
              "actor": _pack_snapshot(value.actor), "critic": _pack_snapshot(value.critic),
              "losses": [v.hex() for v in value.losses]}
    _restore(result)
    return result


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _unique(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        _check(key not in result)
        result[key] = value
    return result


def encode_artifact_v4(result: object, provenance: object, *, enabled: object = False) -> bytes | None:
    if enabled is not True:
        return None
    try:
        value = {"schema": VERSION, "contracts": CONTRACTS, "enabled": False, "promotion": "research-only",
                 "provenance": _provenance(provenance), "result": _pack(result)}
        raw = _json(value)
        return raw if 0 < len(raw) <= 65536 else None
    except (ArithmeticError, ValueError, TypeError, MemoryError, RecursionError):
        return None


def decode_artifact_v4(raw: object, expected_sha256: object, expected_provenance: object,
                       *, enabled: object = False) -> TrainingResult | None:
    if enabled is not True:
        return None
    if type(raw) is not bytes or not 0 < len(raw) <= 65536:
        return None
    try:
        expected = _provenance(expected_provenance)
        _check(type(expected_sha256) is str and re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            return None
        value = json.loads(raw, object_pairs_hook=_unique)
        _check(type(value) is dict and set(value) == {"schema", "contracts", "enabled", "promotion", "provenance", "result"})
        if (value["schema"] != VERSION or value["contracts"] != CONTRACTS or
            value["enabled"] is not False or value["promotion"] != "research-only" or
            _provenance(value["provenance"]) != expected):
            return None
        restored = _restore(value["result"])
        if _json(value) != raw:
            return None
        return restored
    except (ArithmeticError, ValueError, TypeError, MemoryError, RecursionError):
        return None


def request_from_artifact_v4(raw: object, expected_sha256: object, expected_provenance: object,
                             observation: object, *, enabled: object = False) -> bytes | None:
    if enabled is not True:
        return None
    result = decode_artifact_v4(raw, expected_sha256, expected_provenance, enabled=True)
    return encode_request_v3(result, observation, enabled=True)
