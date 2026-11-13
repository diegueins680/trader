#!/usr/bin/env python3
"""Scoped formal verification of the exposure-control RL path.

Runs the SMT proofs, the explicit-state model, Haskell conformance and the
source-surface audit, validates the assurance ledger against what was actually
reproduced in this run, and compares the result with the reviewed receipt.
Solver timeout, UNKNOWN, a failed check or a missing tool is a failure.

  python3 scripts/formal/exposure_rl/verify.py            # verify against the receipt
  python3 scripts/formal/exposure_rl/verify.py --record   # rewrite the receipt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import conformance  # noqa: E402
import ledger_proofs  # noqa: E402
import lifecycle_model  # noqa: E402
import shield_proofs  # noqa: E402
import surface  # noqa: E402
import z3  # noqa: E402

LEDGER = "formal/research/exposure-rl/assurance-ledger.json"
RECEIPT = "formal/research/exposure-rl/results.json"
TOOLCHAIN = "formal/research/toolchain.json"
STATUSES = {"smt_verified", "model_checked", "exhaustively_checked", "property_tested", "assumption", "open"}
VERIFIED = {"smt_verified", "model_checked", "exhaustively_checked"}
ENTRY_FIELDS = (
    "requirementId", "claim", "criticality", "formalStatement", "assumptions", "tool", "toolVersion", "sourceModel", "artifact",
    "implementationFiles", "testFiles", "ciCommand", "result", "bounds", "limitations", "counterexample", "status",
)  # fmt: skip
OBLIGATIONS = tuple(f"FM-{n:02d}" for n in range(1, 13))
PROOF_SOURCES = ("shield_proofs.py", "ledger_proofs.py", "lifecycle_model.py", "conformance.py", "surface.py")
FORBIDDEN = re.compile(r"\b(sorry|admit|Admitted|axiom)\b|@skip|assert (True|False)\b")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("exposure formal verification: " + reason)


def sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def reproduce() -> dict:
    """Run every check. Returns the certificate for this run."""
    toolchain = json.loads((ROOT / TOOLCHAIN).read_text())
    require(z3.get_version_string() == toolchain["z3"], "unreviewed Z3 version")
    ghc = subprocess.run(["ghc", "--numeric-version"], capture_output=True, text=True, timeout=60).stdout.strip()
    require(ghc == toolchain["ghc"], "unreviewed GHC version")
    for name in PROOF_SOURCES:
        require(FORBIDDEN.search((HERE / name).read_text()) is None, "admission or disabled check in " + name)
    shield, ledger, model, conform, audit = shield_proofs.check(), ledger_proofs.check(), lifecycle_model.check(), conformance.check(), surface.check()
    smt = {**shield["smt"], **ledger["smt"]}
    require(all(value == "unsat" for value in smt.values()), "an SMT claim was not discharged")
    return {
        "schemaVersion": 1,
        "solver": z3.get_version_string(),
        "ghc": ghc,
        "smt": dict(sorted(smt.items())),
        "lifecycle": model,
        "conformance": conform["conformance"],
        "forgery": conform["forgery"],
        "surface": audit["surface"],
    }


def reproduced_statuses(certificate: dict) -> dict:
    statuses = {name: "smt_verified" for name in certificate["smt"]}
    model = certificate["lifecycle"]["model"]
    if model["states"] > 0 and model["transitions"] > 0 and certificate["lifecycle"]["mutantCounterexample"]["violation"] == "budget":
        statuses["F-RLX-LIFECYCLE-MODEL"] = "model_checked"
    for key, name in (("conformance", "F-RLX-CONFORMANCE"), ("forgery", "F-RLX-FORGERY"), ("surface", "F-RLX-SURFACE")):
        if certificate[key]["status"] == "exhaustively_checked":
            statuses[name] = "exhaustively_checked"
    return statuses


def validate_ledger(ledger: dict, certificate: dict) -> dict:
    """Check the ledger against this run. Returns the per-obligation closure."""
    require(ledger.get("schemaVersion") == 1 and isinstance(ledger.get("entries"), list), "malformed ledger")
    assumptions = {item["id"] for item in ledger["assumptions"]}
    reproduced = reproduced_statuses(certificate)
    entries = {}
    for entry in ledger["entries"]:
        require(set(entry) == set(ENTRY_FIELDS), "ledger entry fields: " + str(entry.get("requirementId")))
        name = entry["requirementId"]
        require(name not in entries and re.fullmatch(r"F-RLX-[A-Z0-9-]+", name) is not None, "duplicate or malformed requirement " + name)
        require(entry["status"] in STATUSES and entry["criticality"] in {"critical", "high", "medium"}, "status/criticality of " + name)
        require(set(entry["assumptions"]) <= assumptions, "unknown assumption in " + name)
        require(entry["ciCommand"] == "bash scripts/verify.sh formal" or entry["status"] in {"property_tested", "open", "assumption"}, "ci command of " + name)
        for path in [entry["sourceModel"], entry["artifact"], *entry["implementationFiles"], *entry["testFiles"]]:
            # The receipt is this run's own output; it need not exist before the first record.
            require(path is None or path == RECEIPT or (ROOT / path).is_file(), f"missing file {path} for {name}")
        if entry["status"] in VERIFIED:
            # A machine-checked status must have been reproduced in this very run.
            require(reproduced.get(name) == entry["status"], "status not reproduced: " + name)
        if entry["status"] == "property_tested":
            require(entry["testFiles"], "property-tested entry without a test: " + name)
            marker = entry["result"].split("`")[1] if "`" in entry["result"] else None
            require(marker is not None and any(marker in (ROOT / path).read_text() for path in entry["testFiles"]), "test marker not found for " + name)
        if entry["status"] == "open":
            require(len(entry["limitations"]) > 20, "open entry must state its blocker: " + name)
        entries[name] = entry
    require(set(reproduced) <= set(entries), "reproduced claim missing from the ledger: " + ", ".join(sorted(set(reproduced) - set(entries))))
    require({n for n, e in entries.items() if e["status"] in VERIFIED} == set(reproduced), "ledger and certificate rosters differ")
    require(set(ledger["criticalFiles"]) >= {path for e in entries.values() for path in [e["sourceModel"], *e["implementationFiles"]] if path}, "critical files do not cover the entries")
    closure = {}
    require([o["id"] for o in ledger["obligations"]] == list(OBLIGATIONS), "obligations must be FM-01..FM-12 in order")
    for obligation in ledger["obligations"]:
        require(set(obligation) == {"id", "title", "requiredCertificates", "supportingEvidence", "status", "scope", "openItems"}, "obligation fields: " + obligation["id"])
        required = obligation["requiredCertificates"]
        require(set(required) | set(obligation["supportingEvidence"]) <= set(entries), "unknown certificate in " + obligation["id"])
        machine_checked = bool(required) and all(entries[name]["status"] in VERIFIED for name in required)
        expected = "discharged_in_scope" if machine_checked and not obligation["openItems"] else "partially_verified" if required or obligation["supportingEvidence"] else "open"
        require(obligation["status"] == expected, f"{obligation['id']} status must be {expected}")
        closure[obligation["id"]] = obligation["status"]
    return closure


def run(record: bool) -> dict:
    started = time.time()
    certificate = reproduce()
    ledger = json.loads((ROOT / LEDGER).read_text())
    certificate["obligations"] = validate_ledger(ledger, certificate)
    certificate["sourceHashes"] = {path: sha256(path) for path in sorted(ledger["criticalFiles"])}
    payload = json.dumps(certificate, indent=1, sort_keys=True) + "\n"
    receipt = ROOT / RECEIPT
    if record:
        receipt.write_text(payload, encoding="utf-8")
    else:
        require(receipt.is_file() and receipt.read_text(encoding="utf-8") == payload, "certificate differs from the reviewed receipt")
    return {"claims": len(certificate["smt"]), "obligations": certificate["obligations"], "seconds": round(time.time() - started, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.record), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
