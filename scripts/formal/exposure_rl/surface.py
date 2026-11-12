"""Exhaustive source-surface audit of the exposure research path.

Reads every file of the path and checks, by syntax tree and text, that the
research code cannot reach an order capability and is not linked into any
executable. This is an audit of the listed sources; it is not an operating
system sandbox and says nothing about code outside the listed files.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "scripts/research/exposure_rl"
HASKELL = "haskell/research/ExposureShieldV1.hs"
ALLOWED_IMPORTS = {
    "__future__", "argparse", "csv", "dataclasses", "datetime", "hashlib", "json", "math", "multiprocessing", "os", "pathlib",
    "subprocess", "sys", "tempfile", "time", "traceback", "typing", "numpy", "historical_datafeed", "exposure_rl",
}  # fmt: skip
NETWORK_MODULES = {"acquire.py"}  # the only module allowed to import the public datafeed
PROCESS_MODULES = {"run_campaign.py"}  # the only module allowed to start a process
FORBIDDEN_TEXT = (
    "BINANCE_API", "API_SECRET", "TRADER_BINANCE_LIVE", "TRADER_BOT_TRADE", "--binance-live", "--serve", "/fapi/v1/order", "/api/v3/order",
    "signature=", "X-MBX-APIKEY", "pickle", "eval(", "exec(", "liveAuthorization\": True",
)  # fmt: skip
ENV_NAMES = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("exposure surface audit: " + reason)


def audit_python(sources: dict[str, str]) -> dict:
    imports, processes = set(), 0
    for name, source in sorted(sources.items()):
        tree = ast.parse(source)
        for token in FORBIDDEN_TEXT:
            require(token not in source, f"forbidden text {token!r} in {name}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else ["exposure_rl" if node.level else (node.module or "")]
                for module in modules:
                    top = module.split(".")[0] if module else "exposure_rl"
                    require(top in ALLOWED_IMPORTS, f"unreviewed import {module} in {name}")
                    require(top != "historical_datafeed" or name in NETWORK_MODULES, f"network import outside acquire in {name}")
                    require(top != "subprocess" or name in PROCESS_MODULES, f"process import outside the runner in {name}")
                    imports.add(top)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                processes += 1
                require(name in PROCESS_MODULES and node.func.attr == "run", f"unreviewed process call in {name}")
                head = node.args[0]
                reviewed = (isinstance(head, ast.List) and isinstance(head.elts[0], ast.Constant) and head.elts[0].value == "git") or (isinstance(head, ast.Name) and head.id == "command")
                require(reviewed, f"unreviewed process command in {name}")
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                parent = next((p for p in ast.walk(tree) if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.value is node), None)
                require(parent is not None and parent.func.attr == "setdefault", f"environment read in {name}")
        for literal in (n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) is str):
            require(not re.fullmatch(r"[A-Z][A-Z0-9_]*(KEY|SECRET|TOKEN|PASSWORD)", literal), f"credential-like name in {name}")
    return {"modules": len(sources), "imports": sorted(imports), "processCalls": processes}


def audit_haskell(source: str, cabal: str, other_haskell: dict[str, str]) -> dict:
    imports = re.findall(r"^import\s+(?:qualified\s+)?([A-Z][\w.]*)", source, flags=re.M)
    require(imports == [], "the Haskell shield must import nothing beyond the Prelude")
    require("unsafePerformIO" not in source and "foreign " not in source and "IO " not in source, "the Haskell shield must be pure")
    export = re.search(r"module ExposureShieldV1 \((.*?)\) where", source, flags=re.S)
    require(export is not None, "missing explicit export list")
    exports = [item.strip() for item in export.group(1).replace("\n", " ").split(",") if item.strip()]
    require("ResearchIntent" in exports and "Snapshot" in exports, "abstract types must be exported")
    require(not any(item.startswith(("ResearchIntent (", "ResearchIntent(", "Snapshot (", "Snapshot(")) for item in exports), "a constructor is exported")
    require(re.search(r"^orderAuthorized _ = False$", source, flags=re.M) is not None, "orderAuthorized must be constantly False")
    source_dirs = {token for line in re.findall(r"hs-source-dirs:\s*(.*)", cabal) for token in re.split(r"[,\s]+", line) if token}
    require("haskell/research" not in cabal and "ExposureShieldV1" not in cabal and "research" not in source_dirs, "research Haskell is reachable from a cabal component")
    users = sorted(name for name, text in other_haskell.items() if "ExposureShieldV1" in text)
    require(users == [], "production or test Haskell imports the research shield: " + ", ".join(users))
    return {"exports": len(exports), "imports": 0, "cabalComponentsReaching": 0, "haskellImporters": 0}


def check() -> dict:
    sources = {path.name: path.read_text(encoding="utf-8") for path in sorted((ROOT / PACKAGE).glob("*.py"))}
    require({"shield.py", "env.py", "agents.py", "run_campaign.py", "acquire.py", "artifacts.py"} <= set(sources), "path sources are missing")
    runner = sources["run_campaign.py"]
    flags = re.search(r"CHAMPION_FLAGS = \[(.*?)\]", runner, flags=re.S)
    require(flags is not None and "--symbol" not in flags.group(1) and "--futures" not in flags.group(1), "champion export must be an offline CSV backtest")
    other = {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8", errors="replace") for base in ("haskell/app", "haskell/test", "haskell/bench") for path in sorted((ROOT / base).rglob("*.hs"))}
    return {
        "surface": {
            "status": "exhaustively_checked",
            "python": audit_python(sources),
            "haskell": audit_haskell((ROOT / HASKELL).read_text(encoding="utf-8"), (ROOT / "haskell/trader.cabal").read_text(encoding="utf-8"), other),
            "scope": "listed sources only; not a sandbox",
        }
    }


if __name__ == "__main__":
    print(check())
