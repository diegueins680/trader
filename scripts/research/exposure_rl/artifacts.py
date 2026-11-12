"""Versioned, hash-checked policy artifacts (strict JSON only; no executable serialisation)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .agents import Policy
from .env import ENV_VERSION, OBS_DIM
from .market import FEATURE_VERSION
from .nn import MLP
from .shield import VERSION as SHIELD_VERSION

SCHEMA = "exposure_policy_v1"
MAX_BYTES = 1 << 20
MODES = ("research_replay",)  # the only operating mode this campaign may write


class ArtifactError(ValueError):
    """Raised for any artifact that must not be used; callers fall back to hold."""


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_policy(path: Path, policy: Policy, metadata: dict) -> str:
    body = {
        "schema": SCHEMA,
        "envVersion": ENV_VERSION,
        "featureVersion": FEATURE_VERSION,
        "shieldVersion": SHIELD_VERSION,
        "obsDim": OBS_DIM,
        "actionSemantics": "raw proposal in [-1, 1]; absolute target; decoded and projected by the shield",
        "permittedMode": "research_replay",
        "liveAuthorization": False,
        "algorithm": policy.name,
        "sizes": policy.actor.sizes,
        "outTanh": policy.actor.out_tanh,
        "obsMask": None if policy.obs_mask is None else policy.obs_mask.tolist(),
        "params": [p.tolist() for p in policy.actor.params],
        "metadata": metadata,
    }
    digest = _digest(body)
    Path(path).write_text(json.dumps({"sha256": digest, "body": body}, sort_keys=True), encoding="utf-8")
    return digest


def load_policy(path: Path, expected_sha256: str | None = None) -> Policy:
    """Load a policy, refusing anything not exactly compatible with this code."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) > MAX_BYTES:
            raise ArtifactError("artifact too large")
        document = json.loads(raw.decode("utf-8"))
        body = document["body"]
        if document["sha256"] != _digest(body) or (expected_sha256 is not None and document["sha256"] != expected_sha256):
            raise ArtifactError("artifact hash mismatch")
        expected = {"schema": SCHEMA, "envVersion": ENV_VERSION, "featureVersion": FEATURE_VERSION, "shieldVersion": SHIELD_VERSION, "obsDim": OBS_DIM, "liveAuthorization": False}
        for key, value in expected.items():
            if body[key] != value or type(body[key]) is not type(value):
                raise ArtifactError(f"incompatible artifact field: {key}")
        if body["permittedMode"] not in MODES:
            raise ArtifactError("artifact mode is not permitted")
        sizes = body["sizes"]
        if sizes[0] != OBS_DIM or sizes[-1] != 1 or body["outTanh"] is not True:
            raise ArtifactError("artifact network shape is not a bounded scalar policy")
        actor = MLP(sizes, np.random.default_rng(0), out_tanh=True)
        params = [np.asarray(p, dtype=np.float64) for p in body["params"]]
        if len(params) != len(actor.params) or any(a.shape != b.shape or not np.isfinite(a).all() for a, b in zip(params, actor.params)):
            raise ArtifactError("artifact parameters are malformed")
        actor.params = params
        return Policy(actor, str(body["algorithm"]), body["obsMask"])
    except ArtifactError:
        raise
    except Exception as error:  # corrupt JSON, missing keys, wrong types
        raise ArtifactError(f"unreadable artifact: {type(error).__name__}") from error
