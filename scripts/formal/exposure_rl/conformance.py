"""Haskell <-> Python conformance for the exposure shield, plus a forgery check.

Every case is evaluated by the Python reference (the module the experiments
use) and by the compiled Haskell module, and the results must be identical.
The grid is exhaustive over a bounded set of snapshots and raw binary64 edge
values; a seeded random sample extends it. This is exhaustive bounded checking
plus property testing, not a proof about all inputs.
"""

from __future__ import annotations

import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from exposure_rl import shield as S  # noqa: E402

MODULE = "haskell/research/ExposureShieldV1.hs"
DRIVER = "formal/research/exposure-rl/Conformance.hs"
FORGERY = "formal/research/exposure-rl/Forgery.hs"
SEED = 20261111
EXPOSURES = [-100001, -100000, -3000, -1691, -1690, -1667, -1666, -1665, -800, -26, -25, -24, -1, 0, 1, 24, 25, 26, 800, 1665, 1666, 1667, 1690, 1691, 3000, 100000, 100001]
DRAWDOWNS = [-1, 0, 999, 1000, 1001, 10000, 10001]
HEADROOMS = [-1, 0, 1, 24, 25, 1666, 10000, 10001]
RAWS = [
    0.0, -0.0, 5e-324, 1e-300, 0.0006, 0.000600240096, 0.0150060024, 0.015, 0.0149, 0.5, 0.9993997599, 0.9994, 0.99999999, 1.0,
    1.0000000000000002, 1.5, 1e300, float("inf"), float("nan"),
]  # fmt: skip


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def shield_cases():
    raws = RAWS + [-value for value in RAWS if value == value]
    for exposure in EXPOSURES:
        for drawdown in DRAWDOWNS:
            for fresh in (True, False):
                for headroom in HEADROOMS:
                    for raw in raws:
                        yield exposure, drawdown, fresh, headroom, raw
    rng = random.Random(SEED)
    for _ in range(8192):
        raw = rng.choice([rng.uniform(-1.0, 1.0), rng.uniform(-1.2, 1.2), struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]])
        yield rng.randint(-4000, 4000), rng.randint(0, 2000), rng.random() < 0.8, rng.randint(0, 2000), raw


def reserve_cases():
    rng = random.Random(SEED + 1)
    for _ in range(4096):
        held = [rng.randint(0, 2000) for _ in range(rng.randint(1, 4))]
        version = rng.randint(0, 5)
        yield rng.randint(0, 5000), held, version, version if rng.random() < 0.7 else version - 1, rng.randint(-1, len(held)), rng.randint(-5, 2500)


def expected_shield(exposure, drawdown, fresh, headroom, raw) -> str:
    out = S.shield(S.Snapshot(exposure, drawdown, fresh, headroom), raw)
    return f"({out.kind},{out.target},False)"


def expected_reserve(budget, held, version, seen, index, wanted) -> str:
    new, new_version, granted = S.reserve(budget, tuple(held), version, seen, index, wanted)
    return f"({list(new)},{new_version},{granted})".replace(" ", "")


def compile_haskell(source: str, directory: Path, name: str):
    exe = directory / name
    done = subprocess.run(["ghc", "-v0", "-O0", "-Wall", "-Werror", "-ihaskell/research", "-outputdir", str(directory / (name + "-o")), source, "-o", str(exe)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    return done, exe


def check() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        done, exe = compile_haskell(DRIVER, directory, "conformance")
        if done.returncode != 0:
            raise ValueError("exposure conformance: Haskell driver failed to compile: " + done.stderr[-400:])
        shields, reserves = list(shield_cases()), list(reserve_cases())
        lines = [f"S({e},{d},{f},{h},{bits(r)})" for e, d, f, h, r in shields] + [f"R({b},{h},{v},{s},{i},{w})" for b, h, v, s, i, w in reserves]
        run = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
        if run.returncode != 0:
            raise ValueError("exposure conformance: driver exited with an error")
        actual = [line.replace(" ", "") for line in run.stdout.splitlines()]
        expected = [expected_shield(*case) for case in shields] + [expected_reserve(*case) for case in reserves]
        if len(actual) != len(expected):
            raise ValueError("exposure conformance: output length mismatch")
        for index, (got, want) in enumerate(zip(actual, expected)):
            if got != want:
                raise ValueError(f"exposure conformance: mismatch at {lines[index]}: haskell {got} python {want}")
        kinds = [0, 0, 0, 0]
        for line in actual[: len(shields)]:
            kinds[int(line[1])] += 1
        if min(kinds) == 0:
            raise ValueError("exposure conformance: an outcome kind is never exercised")
        if any(line.endswith("True)") for line in actual[: len(shields)]):
            raise ValueError("exposure conformance: an outcome claims order authority")
        forged, _ = compile_haskell(FORGERY, directory, "forgery")
        if forged.returncode == 0 or "ResearchIntent" not in forged.stderr:
            raise ValueError("exposure conformance: a research intent could be constructed outside the shield")
    return {"conformance": {"status": "exhaustively_checked", "shieldCases": len(shields), "reserveCases": len(reserves), "outcomeKinds": kinds, "seed": SEED}, "forgery": {"status": "exhaustively_checked", "constructorExported": False}}


if __name__ == "__main__":
    print(check())
