#!/usr/bin/env python3
"""Adopted-champion screen: replay the frozen live fleet causally on lawful data.

Registration: research-notes/registrations/adopted-champion-screen-v1.json.

Each frozen combo is replayed with ``trader-hs --adopt-combo-file`` so the
backtest applies exactly the parameters, venue cost floors and adoption caps a
live bot would.  The evaluation window is cut into consecutive chunks; every
chunk is the backtest slice of a run whose input is the combo's own ``bars``
window ending at the chunk end, so the model is refit only on data strictly
before each chunk (the optimizer's own protocol, rolled forward).  Baselines
are computed on the identical rows with the identical per-side cost.

Subcommands:
  fetch  - download Binance USD-M klines and write a hash manifest
  run    - replay one registered phase and write per-bar returns + summary
  selftest - synthetic checks of chunking, chaining and baselines (no network)
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "research-notes/registrations/adopted-champion-screen-v1.json"
AMENDMENT = ROOT / "research-notes/registrations/adopted-champion-screen-v1-prospective-amendment.json"
# Annotated tag on GitHub that fixes the runner, registration, amendment and
# haskell/ tree for one-shot phases; the trust root the repo files cannot forge.
TRUST_TAG = "adopted-champion-screen-v1-prospective"
# The tag is checked on this repository, never on whatever the local origin points to.
CANONICAL_REMOTE = "https://github.com/diegueins680/trader.git"
FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL_MS = {"1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000}
CSV_FIELDS = ["openTimeMs", "open", "high", "low", "close", "volume", "closeTimeMs"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registration(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRATION).read_text())


# ---------------------------------------------------------------- data


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    """All klines with start_ms <= openTime and closeTime < end_ms (closed bars only)."""
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms - 1, "limit": 1500})
        with urllib.request.urlopen(f"{FAPI_KLINES}?{query}", timeout=30) as resp:
            batch = json.loads(resp.read())
        if not batch:
            break
        rows.extend(k for k in batch if int(k[6]) < end_ms)
        nxt = int(batch[-1][0]) + INTERVAL_MS[interval]
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.2)
    return rows


def check_contiguous(open_times: list[int], step: int) -> None:
    for a, b in zip(open_times, open_times[1:]):
        if b - a != step:
            raise ValueError(f"kline gap or duplicate between {a} and {b}")


def render_csv(rows: list[list]) -> bytes:
    """The exact CSV bytes the screen stores for a list of klines."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(CSV_FIELDS)
    for r in rows:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
    return buffer.getvalue().encode()


def phase_data_dir(reg: dict, phase_name: str) -> Path:
    """Retrospective files live at the data root (as preregistered); other phases get their own directory."""
    root = ROOT / reg["data"]["directory"]
    return root if phase_name == "retrospective" else root / phase_name


def phase_fetch_end(reg: dict, phase_name: str) -> int:
    return int(reg["data"]["fetchEndExclusiveMs"]) if phase_name == "retrospective" else int(reg["phases"][phase_name]["endExclusiveMs"])


def cmd_fetch(args: argparse.Namespace) -> None:
    reg = load_registration()
    phase = reg["phases"][args.phase]
    check_embargo(phase, int(time.time() * 1000))
    out_dir = phase_data_dir(reg, args.phase)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "adopted_champion_screen_data_v1", "source": FAPI_KLINES, "files": {}}
    end_ms = phase_fetch_end(reg, args.phase)
    for combo in reg["combos"]:
        if combo["symbol"] not in phase["symbols"]:
            continue
        sym, interval = combo["symbol"], combo["interval"]
        rows = fetch_klines(sym, interval, int(reg["data"]["fetchStartMs"]), end_ms)
        check_contiguous([int(r[0]) for r in rows], INTERVAL_MS[interval])
        path = out_dir / f"{sym}-{interval}.csv"
        path.write_bytes(render_csv(rows))
        manifest["files"][path.name] = {"sha256": sha256_file(path), "rows": len(rows), "firstOpenMs": int(rows[0][0]), "lastOpenMs": int(rows[-1][0])}
        print(f"{path.name}: {len(rows)} rows", file=sys.stderr)
    (out_dir / phase["dataManifest"]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_phase_blobs(reg: dict, phase_name: str, data_dir: Path, manifest: dict) -> dict[str, bytes]:
    """Read every phase CSV exactly once and check those bytes against the manifest.

    The run parses only these buffers, so a file replaced later cannot reach the backtest.
    """
    phase = reg["phases"][phase_name]
    blobs: dict[str, bytes] = {}
    for combo in reg["combos"]:
        if combo["symbol"] not in phase["symbols"]:
            continue
        name = f"{combo['symbol']}-{combo['interval']}.csv"
        data = (data_dir / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]["sha256"]:
            raise SystemExit(f"{name} drifted from manifest")
        blobs[name] = data
    return blobs


def verify_against_source(reg: dict, phase_name: str, blobs: dict[str, bytes]) -> None:
    """Re-download every phase file from Binance and require byte-identical CSVs."""
    for combo in reg["combos"]:
        name = f"{combo['symbol']}-{combo['interval']}.csv"
        if name not in blobs:
            continue
        rows = fetch_klines(combo["symbol"], combo["interval"], int(reg["data"]["fetchStartMs"]), phase_fetch_end(reg, phase_name))
        if render_csv(rows) != blobs[name]:
            raise SystemExit(f"{name} does not match a fresh download from {FAPI_KLINES}")


def parse_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode())))


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- protocol


def chunk_plan(open_times: list[int], start_ms: int, end_ms: int, window: int, test_len: int) -> list[tuple[int, int]]:
    """Consecutive [s, e) row-index chunks covering every bar opening in [start_ms, end_ms).

    Each chunk is at most ``test_len`` bars.  Its run's backtest slice is rows
    ``s-1 .. e-1`` (the extra leading row anchors the first return), so the run
    input is rows ``e-window .. e-1`` and the model never sees a row >= s-1
    while fitting.
    """
    idx = [i for i, t in enumerate(open_times) if start_ms <= t < end_ms]
    if not idx:
        raise ValueError("evaluation window has no bars")
    first, last = idx[0], idx[-1] + 1
    chunks = []
    s = first
    while s < last:
        e = min(s + test_len, last)
        if e - window < 0 or (e - s) + 1 >= window:
            raise ValueError("not enough history before the evaluation window")
        chunks.append((s, e))
        s = e
    return chunks


def equity_to_returns(equity: list[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(equity, equity[1:])]


def chain(returns: list[float]) -> float:
    eq = 1.0
    for r in returns:
        eq *= 1.0 + r
    return eq - 1.0


def max_drawdown(returns: list[float]) -> float:
    eq, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        eq *= 1.0 + r
        peak = max(peak, eq)
        worst = max(worst, 1.0 - eq / peak)
    return worst


def sharpe_per_period(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    sd = statistics.stdev(returns)
    return 0.0 if sd == 0 else statistics.fmean(returns) / sd


def psr(returns: list[float]) -> float:
    """Probabilistic Sharpe ratio P(SR > 0) with skew/kurtosis correction (single trial)."""
    n = len(returns)
    if n < 3:
        return 0.5
    sr = sharpe_per_period(returns)
    mean = statistics.fmean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.5
    skew = sum(((r - mean) / sd) ** 3 for r in returns) / n
    kurt = sum(((r - mean) / sd) ** 4 for r in returns) / n
    var = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if var <= 0:
        return 0.5
    z = sr * math.sqrt(n - 1) / math.sqrt(var)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def baseline_returns(closes: list[float], s: int, e: int, size: float, cost_side: float, kind: str, long_short: bool, lookback: int) -> list[float]:
    """Per-bar net returns for bars s..e-1 (return of bar i = close[i]/close[i-1]-1)."""
    out: list[float] = []
    pos = 0.0
    for i in range(s, e):
        if kind == "flat":
            target = 0.0
        elif kind == "buy_and_hold":
            target = size
        elif kind == "momentum":
            j = i - 1 - lookback
            past = closes[i - 1] / closes[j] - 1.0 if j >= 0 else 0.0
            target = size if past > 0 else (-size if long_short and past < 0 else 0.0)
        else:
            raise ValueError(kind)
        # decide on information up to close[i-1], hold over bar i
        turnover = abs(target - pos)
        pos = target
        bar = closes[i] / closes[i - 1] - 1.0
        out.append(pos * bar - turnover * cost_side)
    if out and pos != 0.0:
        out[-1] -= abs(pos) * cost_side  # liquidate at window end
    return out


# ---------------------------------------------------------------- replay


def run_chunk(binary: Path, snapshot: Path, uuid: str, rows: list[dict], s: int, e: int, window: int, seed: int, cap: float, extra: list[str]) -> dict:
    lo = e - window
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows[lo:e])
        ratio = (e - s + 1) / window
        cmd = [
            str(binary), "--data", str(path), "--price-column", "close", "--high-column", "high", "--low-column", "low",
            "--adopt-combo-file", str(snapshot), "--adopt-combo-uuid", uuid,
            "--adoption-max-position-size-cap", repr(cap), "--bars", str(window),
            "--backtest-ratio", repr(ratio), "--seed", str(seed), "--json", *extra,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(f"trader-hs failed ({uuid} seed {seed} chunk {s}-{e}): {proc.stderr[-2000:]}")
        return json.loads(proc.stdout)["backtest"]


def chunk_returns(out: dict, expected: int) -> tuple[list[float], dict]:
    """``expected`` net returns from a backtest slice of ``expected + 1`` bars."""
    split = out["split"]
    if int(split["backtest"]) != expected + 1:
        raise RuntimeError(f"backtest slice {split['backtest']} != registered chunk {expected} + 1 anchor bar")
    curve = [float(x) for x in out["equityCurve"]]
    if len(curve) != expected + 1:
        raise RuntimeError(f"equityCurve length {len(curve)} != chunk {expected} + 1")
    realized = out.get("costs", {}).get("attribution", {}).get("realized", {})
    return equity_to_returns(curve), {"metrics": out.get("metrics", {}), "realizedCosts": realized, "perSideCost": out.get("costs", {}).get("perSideCost"), "maxPositionSize": out.get("maxPositionSize")}


def daily_returns(open_times: list[int], returns: list[float]) -> dict[int, float]:
    """Compound per-bar returns into UTC calendar days keyed by day index."""
    days: dict[int, float] = {}
    for t, r in zip(open_times, returns):
        day = t // 86_400_000
        days[day] = (1.0 + days.get(day, 0.0)) * (1.0 + r) - 1.0
    return days


def fleet_daily(per_combo: list[dict[int, float]]) -> list[float]:
    """Equal-weight daily fleet returns over the days every combo covers."""
    common = sorted(set.intersection(*(set(d) for d in per_combo)))
    return [statistics.fmean(d[day] for d in per_combo) for day in common]


def prospective_gates(combo_decisions: list[str], fleet: dict, trial_bound: float) -> dict:
    """Evaluate the amendment's pass gates; any failed gate is a rejection."""
    base = fleet["net"]
    lifetime = max(0.0, 1.0 - min(1.0, trial_bound * (1.0 - base["psr"])))
    gates = {
        "everyComboNotRejected": all(d == "NOT_REJECTED" for d in combo_decisions),
        "lifetimeAdjustedPsrAtLeast095": lifetime >= 0.95,
        "maxDrawdownAtMost20pct": base["maxDrawdown"] <= 0.20,
        "sharpeAboveBuyAndHold": base["sharpeAnnualized"] > fleet["buy_and_hold"]["sharpeAnnualized"],
        "sharpeAboveMomentum": base["sharpeAnnualized"] > fleet["momentum"]["sharpeAnnualized"],
        "doubledCostReturnPositive": fleet["doubledCosts"]["netReturn"] > 0.0,
        "oneBarDelayReturnPositive": fleet["oneBarDelay"]["netReturn"] > 0.0,
    }
    return {"lifetimeAdjustedPsr": lifetime, "gates": gates, "decision": "PASS" if all(gates.values()) else "REJECTED"}


def summarize(returns: list[float], periods_per_year: float) -> dict:
    sr = sharpe_per_period(returns)
    return {
        "bars": len(returns),
        "netReturn": chain(returns),
        "maxDrawdown": max_drawdown(returns),
        "sharpeAnnualized": sr * math.sqrt(periods_per_year),
        "psr": psr(returns),
    }


def check_embargo(phase: dict, now_ms: int) -> None:
    """Refuse a phase before its registered release time, before any row is read."""
    release = phase.get("minimumEvaluationTimeUtc")
    if release and now_ms < int(datetime.fromisoformat(release.replace("Z", "+00:00")).timestamp() * 1000):
        raise SystemExit(f"phase is embargoed until {release}; no rows were read")


def check_manifest(reg: dict, phase_name: str, manifest_path: Path) -> dict:
    """The retrospective manifest must be the preregistered one, byte for byte."""
    pinned = reg["data"].get(f"{phase_name}ManifestSha256")
    if pinned and sha256_file(manifest_path) != pinned:
        raise SystemExit(f"{manifest_path.name} does not match the registered sha256")
    return json.loads(manifest_path.read_text())


def pinned_binary(pinned_tree: str) -> dict:
    """Build trader-hs from a clean haskell/ tree equal to the preregistered one.

    One-shot evidence must come from preregistered code, so a dirty tree or a
    different tree hash is refused before any row is read.
    """
    git = ["git", "-C", str(ROOT)]
    if subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "haskell"]).returncode != 0:
        raise SystemExit("haskell/ has uncommitted changes; one-shot phases need the pinned tree")
    tree = subprocess.check_output([*git, "rev-parse", "HEAD:haskell"], text=True).strip()
    if tree != pinned_tree:
        raise SystemExit(f"haskell/ tree {tree} is not the preregistered {pinned_tree}; check out a commit with that tree")
    haskell = ROOT / "haskell"
    subprocess.run(["cabal", "build", "exe:trader-hs"], cwd=haskell, check=True, capture_output=True)
    binary = Path(subprocess.check_output(["cabal", "list-bin", "trader-hs"], cwd=haskell, text=True).strip())
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    # Every chunk runs this private read-only copy, so a rebuild during the run cannot mix engines.
    private = Path(tempfile.mkdtemp(prefix="champion-trader-hs-"))
    frozen = private / "trader-hs"
    shutil.copy2(binary, frozen)
    os.chmod(frozen, 0o500)
    os.chmod(private, 0o500)
    atexit.register(shutil.rmtree, private, ignore_errors=True)
    return {"path": frozen, "haskellTree": tree, "commit": commit, "sha256": sha256_file(frozen)}


def publish(out_dir: Path, payloads: list[tuple[str, str]], seal: bool) -> None:
    """Write every payload durably under a temporary name, then rename into place.

    The last payload (results.json) is the completion marker, so an interrupted
    run leaves no marker and may be repeated with identical inputs.
    """
    staged = []
    for name, text in payloads:
        tmp = out_dir / f".{name}.{os.getpid()}.partial"
        with tmp.open("w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged.append((tmp, out_dir / name))
    for tmp, final in staged:
        if seal:
            os.chmod(tmp, 0o444)
        os.replace(tmp, final)


def reserve(out_dir: Path) -> None:
    """Exclusively reserve a one-shot output directory before any row is read.

    A leftover reservation means a run is in progress or was interrupted; an
    operator must inspect it and remove the file before retrying.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(out_dir / ".reservation", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise SystemExit(f"{out_dir} is reserved by another run (or an interrupted one); inspect and remove .reservation to retry") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "startedAtMs": int(time.time() * 1000)}) + "\n")


def check_trust_root(tag: str = TRUST_TAG) -> str:
    """HEAD must carry the tagged runner, registration, amendment and haskell/ tree.

    The local tag must equal the tag on origin, so a rewritten local tag or a
    later commit that changes these files together is refused.
    """
    git = ["git", "-C", str(ROOT)]
    local = subprocess.run([*git, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"], capture_output=True, text=True).stdout.strip()
    remote_lines = subprocess.run([*git, "ls-remote", "--tags", CANONICAL_REMOTE, f"refs/tags/{tag}^{{}}"], capture_output=True, text=True).stdout.split()
    remote = remote_lines[0] if remote_lines else ""
    if not local or local != remote:
        raise SystemExit(f"tag {tag} is missing, or differs from {CANONICAL_REMOTE} ({local or 'none'} vs {remote or 'none'})")
    paths = [str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(), REGISTRATION, AMENDMENT)] + ["haskell"]
    for path in paths:
        tagged = subprocess.run([*git, "rev-parse", f"{local}:{path}"], capture_output=True, text=True).stdout.strip()
        current = subprocess.run([*git, "rev-parse", f"HEAD:{path}"], capture_output=True, text=True).stdout.strip()
        if not tagged or tagged != current:
            raise SystemExit(f"{path} at HEAD differs from tag {tag}; run from a checkout of the tag")
    return local


def check_runner_pin(amendment: dict) -> None:
    """The runner, registration and amendment must be committed, unmodified, and the runner preregistered."""
    git = ["git", "-C", str(ROOT)]
    tracked = [str(p.relative_to(ROOT)) for p in (Path(__file__).resolve(), REGISTRATION, AMENDMENT)]
    if subprocess.run([*git, "diff", "--quiet", "HEAD", "--", *tracked]).returncode != 0:
        raise SystemExit("runner, registration or amendment has uncommitted changes; one-shot phases need the pinned sources")
    if sha256_file(Path(__file__).resolve()) != amendment["pinnedRunnerSha256"]:
        raise SystemExit("champion_screen.py is not the preregistered runner (pinnedRunnerSha256)")


def freeze_snapshot(path: Path, expected_sha256: str) -> Path:
    """Copy the verified snapshot bytes to a private read-only file used by every chunk.

    Each chunk is a new process; reading the shared file again could pick up a
    replaced snapshot after the one-time hash check.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise SystemExit("frozen champion snapshot drifted from registration")
    private = Path(tempfile.mkdtemp(prefix="champion-snapshot-"))
    frozen = private / "snapshot.json"
    frozen.write_bytes(data)
    os.chmod(frozen, 0o400)
    os.chmod(private, 0o500)
    atexit.register(shutil.rmtree, private, ignore_errors=True)
    return frozen


def check_coverage(open_times: list[int], step: int, first_needed_ms: int, last_open_ms: int) -> None:
    """Rows must be contiguous from the first training row through the window's final bar."""
    try:
        check_contiguous(open_times, step)
    except ValueError as err:
        raise SystemExit(f"phase dataset is not contiguous: {err}") from err
    if not open_times or open_times[0] > first_needed_ms or open_times[-1] < last_open_ms:
        raise SystemExit("phase dataset does not cover the registered window and its training history")


def cmd_run(args: argparse.Namespace) -> None:
    reg = load_registration()
    phase = reg["phases"][args.phase]
    check_embargo(phase, int(time.time() * 1000))
    out_dir = ROOT / phase["outputDirectory"]  # absolute paths (tests) override ROOT
    if phase.get("oneShot") and (out_dir / "results.json").exists():
        raise SystemExit(f"{args.phase} is one-shot and already has sealed results; no rows were read")
    stress = "passGates" in phase
    build = None
    if phase.get("oneShot"):
        amendment = json.loads(AMENDMENT.read_text())
        if amendment["amendsSha256"] != sha256_file(REGISTRATION):
            raise SystemExit("prospective amendment does not amend this registration")
        check_runner_pin(amendment)
        trust_commit = check_trust_root()
        build = pinned_binary(amendment["pinnedHaskellTree"])
    data_dir = phase_data_dir(reg, args.phase)
    manifest = check_manifest(reg, args.phase, data_dir / phase["dataManifest"])
    blobs = load_phase_blobs(reg, args.phase, data_dir, manifest)
    if build:
        verify_against_source(reg, args.phase, blobs)
        build["trustTag"] = TRUST_TAG
        build["trustCommit"] = trust_commit
        reserve(out_dir)
    snapshot = freeze_snapshot(ROOT / reg["snapshot"]["path"], reg["snapshot"]["sha256"])
    binary = build["path"] if build else Path(args.binary)
    results = {"registration": REGISTRATION.name, "registrationSha256": sha256_file(REGISTRATION), "phase": args.phase, "combos": {}}
    if build:
        results["build"] = {k: str(v) for k, v in build.items()}
        results["amendmentSha256"] = sha256_file(AMENDMENT)
        results["runnerSha256"] = sha256_file(Path(__file__).resolve())
    series_out: dict[str, dict] = {}
    fleet_inputs: list[dict[str, dict[int, float]]] = []
    for combo in reg["combos"]:
        if combo["symbol"] not in phase["symbols"]:
            continue
        rows = parse_rows(blobs[f"{combo['symbol']}-{combo['interval']}.csv"])
        open_times = [int(r["openTimeMs"]) for r in rows]
        closes = [float(r["close"]) for r in rows]
        start_ms = max(int(phase["startMs"]), int(combo["createdAtMs"]))
        step = INTERVAL_MS[combo["interval"]]
        first_eval_ms = -(-start_ms // step) * step
        last_open_ms = (int(phase["endExclusiveMs"]) - 1) // step * step
        check_coverage(open_times, step, first_eval_ms - combo["windowBars"] * step, last_open_ms)
        chunks = chunk_plan(open_times, start_ms, int(phase["endExclusiveMs"]), combo["windowBars"], combo["chunkBars"])
        per_seed: dict[str, list[float]] = {}
        doubled_seed: dict[str, list[float]] = {}
        delayed_seed: dict[str, list[float]] = {}
        diag: dict[str, list] = {}
        for seed in reg["protocol"]["seeds"]:
            rets: list[float] = []
            doubled: list[float] = []
            delayed: list[float] = []
            diag[str(seed)] = []
            cap = reg["protocol"]["adoptionMaxPositionSizeCap"]
            for s, e in chunks:
                out = run_chunk(binary, snapshot, combo["uuid"], rows, s, e, combo["windowBars"], seed, cap, [])
                r, d = chunk_returns(out, e - s)
                rets.extend(r)
                diag[str(seed)].append({"chunk": [open_times[s], open_times[e - 1]], **d})
                if stress:
                    costly = run_chunk(binary, snapshot, combo["uuid"], rows, s, e, combo["windowBars"], seed, cap, ["--backtest-cost-multiplier", "2"])
                    doubled.extend(chunk_returns(costly, e - s)[0])
                    # Start the delayed run one bar earlier and drop that bar's return, so the
                    # chunk's first evaluated decision uses the previous bar's forecast
                    # instead of being forced flat at the slice boundary.
                    late = run_chunk(binary, snapshot, combo["uuid"], rows, s - 1, e, combo["windowBars"], seed, cap, ["--backtest-signal-delay-bars", "1"])
                    delayed.extend(chunk_returns(late, e - s + 1)[0][1:])
            per_seed[str(seed)] = rets
            doubled_seed[str(seed)] = doubled
            delayed_seed[str(seed)] = delayed
            print(f"{combo['symbol']} seed {seed}: {len(rets)} bars", file=sys.stderr)
        ppy = combo["periodsPerYear"]
        seed_summaries = {k: summarize(v, ppy) for k, v in per_seed.items()}
        for k, chunks_diag in diag.items():
            seed_summaries[k]["roundTrips"] = sum(int(c["metrics"].get("roundTrips") or 0) for c in chunks_diag)
        median_seed = sorted(seed_summaries, key=lambda k: seed_summaries[k]["netReturn"])[len(seed_summaries) // 2]
        s0, e_last = chunks[0][0], chunks[-1][1]
        adopted = diag[str(reg["protocol"]["liveSeed"])][0]
        size, cost_side = float(adopted["maxPositionSize"]), float(adopted["perSideCost"])
        base_returns = {
            kind: baseline_returns(closes, s0, e_last, size, cost_side, kind, combo["positioning"] == "long-short", reg["protocol"]["momentumLookbackBars"])
            for kind in ("flat", "buy_and_hold", "momentum")
        }
        bases = {kind: summarize(r, ppy) for kind, r in base_returns.items()}
        pin_idx = next(i for i, t in enumerate(open_times[s0:e_last]) if t >= int(reg["protocol"]["pinnedAtMs"]))
        med = per_seed[median_seed]
        rule = reg["protocol"]["rejectionRule"]
        rejected_return = seed_summaries[median_seed]["netReturn"] <= rule["maxNetReturnForRejection"]
        rejected_dd = seed_summaries[median_seed]["maxDrawdown"] > rule["maxDrawdown"]
        no_trades = seed_summaries[median_seed]["roundTrips"] == 0
        results["combos"][combo["symbol"]] = {
            "uuid": combo["uuid"],
            "window": [open_times[s0], open_times[e_last - 1]],
            "chunks": len(chunks),
            "adoptedMaxPositionSize": size,
            "adoptedPerSideCost": cost_side,
            "seeds": seed_summaries,
            "medianSeed": median_seed,
            "liveSeed": seed_summaries.get(str(reg["protocol"]["liveSeed"])),
            "postPinSubwindow": summarize(med[pin_idx:], ppy),
            "baselines": bases,
            "decision": "REJECTED" if (rejected_return or rejected_dd) else "NOT_REJECTED",
            "rejectionReasons": [n for n, hit in (("no_trades", no_trades), ("net_return_not_above_zero", rejected_return and not no_trades), ("drawdown_above_limit", rejected_dd)) if hit],
            "lifetimeAdjustedPsr": max(0.0, 1.0 - min(1.0, reg["protocol"]["optimizerTrialLowerBound"] * (1.0 - seed_summaries[median_seed]["psr"]))),
        }
        series_out[combo["symbol"]] = {"openTimeMs": open_times[s0:e_last], "seeds": per_seed}
        if stress:
            times = open_times[s0:e_last]
            med_net = per_seed[median_seed]
            fleet_inputs.append(
                {
                    "net": daily_returns(times, med_net),
                    "doubledCosts": daily_returns(times, doubled_seed[median_seed]),
                    "oneBarDelay": daily_returns(times, delayed_seed[median_seed]),
                    "buy_and_hold": daily_returns(times, base_returns["buy_and_hold"]),
                    "momentum": daily_returns(times, base_returns["momentum"]),
                }
            )
            series_out[combo["symbol"]]["stress"] = {"doubledCosts": doubled_seed[median_seed], "oneBarDelay": delayed_seed[median_seed], "medianSeed": median_seed}
        results.setdefault("diagnostics", {})[combo["symbol"]] = diag
    decisions = [c["decision"] for c in results["combos"].values()]
    rejected = decisions.count("REJECTED")
    results["fleetDecision"] = "REJECTED" if rejected >= reg["protocol"]["fleetRejectionMinCombos"] else "INCONCLUSIVE"
    if stress:
        fleet = {key: summarize(fleet_daily([c[key] for c in fleet_inputs]), 365.0) for key in fleet_inputs[0]}
        verdict = prospective_gates(decisions, fleet, reg["protocol"]["optimizerTrialLowerBound"])
        results["fleet"] = {**fleet, **verdict}
        results["fleetDecision"] = verdict["decision"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if phase.get("oneShot") and (out_dir / "results.json").exists():
        raise SystemExit(f"{args.phase} results appeared during the run; refusing to overwrite")
    publish(
        out_dir,
        [("returns.json", json.dumps(series_out, sort_keys=True) + "\n"), ("results.json", json.dumps(results, indent=1, sort_keys=True) + "\n")],
        seal=bool(phase.get("oneShot")),
    )
    print(json.dumps({k: {"decision": v["decision"], "median": v["seeds"][v["medianSeed"]]} for k, v in results["combos"].items()}, indent=1))
    print("fleet:", results["fleetDecision"])


# ---------------------------------------------------------------- selftest


def cmd_selftest(_: argparse.Namespace) -> None:
    times = [i * 10 for i in range(100)]
    plan = chunk_plan(times, 500, 1000, window=40, test_len=8)
    assert plan[0] == (50, 58) and plan[-1] == (98, 100), plan
    assert all(b[0] == a[1] for a, b in zip(plan, plan[1:]))
    try:
        chunk_plan(times, 0, 100, window=40, test_len=8)
        raise AssertionError("missing history must fail")
    except ValueError:
        pass
    assert abs(chain([0.1, -0.1]) - (-0.01)) < 1e-12
    assert abs(max_drawdown([0.1, -0.5, 0.2]) - 0.5) < 1e-12
    closes = [100.0, 110.0, 99.0]
    flat = baseline_returns(closes, 1, 3, 1.0, 0.001, "flat", False, 1)
    assert flat == [0.0, 0.0]
    bh = baseline_returns(closes, 1, 3, 1.0, 0.001, "buy_and_hold", False, 1)
    assert abs(bh[0] - (0.1 - 0.001)) < 1e-12 and abs(bh[1] - (-0.1 - 0.001)) < 1e-12
    assert psr([0.01, 0.02, 0.015, 0.012, 0.018]) > 0.9
    for call in (
        lambda: check_embargo({"minimumEvaluationTimeUtc": "2027-01-21T06:00:00Z"}, 1800489600000),
        lambda: check_coverage([0, 10, 20], 10, 0, 30),
        lambda: check_coverage([10, 20, 30], 10, 0, 30),
        lambda: check_coverage([0, 10, 30], 10, 0, 30),
    ):
        try:
            call()
            raise AssertionError("guard must refuse")
        except SystemExit:
            pass
    check_embargo({"minimumEvaluationTimeUtc": "2027-01-21T06:00:00Z"}, 1800511200000)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "returns.json").write_text("stale")
        publish(out, [("returns.json", "r"), ("results.json", "s")], seal=True)
        assert (out / "returns.json").read_text() == "r" and (out / "results.json").read_text() == "s"
        assert not list(out.glob(".*.partial"))
        assert (out / "results.json").stat().st_mode & 0o222 == 0, "sealed results must have no write bits"
        reserve(out)
        try:
            reserve(out)
            raise AssertionError("a second reservation must be refused")
        except SystemExit:
            pass
    day = 86_400_000
    d = daily_returns([0, day // 2, day], [0.1, 0.1, -0.5])
    assert abs(d[0] - 0.21) < 1e-12 and d[1] == -0.5
    assert fleet_daily([{0: 0.02, 1: 0.0}, {0: 0.0, 1: 0.04, 2: 0.1}]) == [0.01, 0.02]
    good = {"psr": 1.0, "maxDrawdown": 0.05, "sharpeAnnualized": 3.0, "netReturn": 0.2}
    fleet = {"net": good, "buy_and_hold": {"sharpeAnnualized": 1.0}, "momentum": {"sharpeAnnualized": 1.0},
             "doubledCosts": {"netReturn": 0.1}, "oneBarDelay": {"netReturn": 0.05}}
    assert prospective_gates(["NOT_REJECTED"] * 3, fleet, 3261)["decision"] == "PASS"
    assert prospective_gates(["NOT_REJECTED", "REJECTED", "NOT_REJECTED"], fleet, 3261)["decision"] == "REJECTED"
    assert prospective_gates(["NOT_REJECTED"] * 3, {**fleet, "oneBarDelay": {"netReturn": -0.01}}, 3261)["decision"] == "REJECTED"
    assert prospective_gates(["NOT_REJECTED"] * 3, {**fleet, "net": {**good, "psr": 0.9999}}, 3261)["decision"] == "REJECTED"
    check_coverage([0, 10, 20, 30], 10, 0, 30)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "snap.json"
        src.write_bytes(b'{"combos": []}')
        frozen = freeze_snapshot(src, hashlib.sha256(b'{"combos": []}').hexdigest())
        src.write_bytes(b"replaced")
        assert frozen.read_bytes() == b'{"combos": []}' and frozen.stat().st_mode & 0o222 == 0
        try:
            freeze_snapshot(src, hashlib.sha256(b'{"combos": []}').hexdigest())
            raise AssertionError("a drifted snapshot must be refused")
        except SystemExit:
            pass
    print("selftest ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--phase", default="retrospective", choices=["retrospective", "prospective"])
    r = sub.add_parser("run")
    r.add_argument("--phase", required=True, choices=["retrospective", "prospective"])
    r.add_argument("--binary", default=os.environ.get("TRADER_HS_BIN", "trader-hs"))
    sub.add_parser("selftest")
    args = parser.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
