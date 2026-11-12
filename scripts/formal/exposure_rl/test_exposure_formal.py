#!/usr/bin/env python3
"""Mutation tests: the exposure-path checkers must reject broken sources and ledgers."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import lifecycle_model  # noqa: E402
import shield_proofs  # noqa: E402
import surface  # noqa: E402
import verify  # noqa: E402

SOURCE = (ROOT / shield_proofs.SOURCE).read_text(encoding="utf-8")


class ShieldMutants(unittest.TestCase):
    def mutant(self, old: str, new: str) -> str:
        self.assertEqual(SOURCE.count(old), 1, old)
        return SOURCE.replace(old, new)

    def test_real_source_is_certified(self):
        self.assertEqual(len(shield_proofs.check(SOURCE)["smt"]), 11)

    def test_broken_projections_are_refuted(self):
        mutants = {
            "cap removed": ("    limit = min(CAP, min(abs(current), CAP) + snapshot.headroom)", "    limit = abs(current) + snapshot.headroom"),
            "risk-off inverted": ("    if (not snapshot.fresh) or snapshot.drawdown >= DD_LIMIT:", "    if snapshot.fresh and snapshot.drawdown >= DD_LIMIT:"),
            "reduce-only crosses zero": ("            reduced = _clamp(target, 0, current)", "            reduced = _clamp(target, -current, current)"),
            "headroom ignored": ("    limit = min(CAP, min(abs(current), CAP) + snapshot.headroom)", "    limit = CAP"),
            "cap raised": ("CAP = 1_666  # maximum admitted |target exposure|", "CAP = 1_666  # maximum admitted |target exposure|\nMAX = 2_000"),
        }
        for name, (old, new) in mutants.items():
            source = self.mutant(old, new)
            if name == "cap removed":
                source = source.replace("    target = _clamp(proposal, -CAP, CAP)\n", "    target = proposal\n")
            if name == "cap raised":
                source = source.replace("    target = _clamp(proposal, -CAP, CAP)\n", "    target = _clamp(proposal, -MAX, MAX)\n").replace("    limit = min(CAP, min(abs(current), CAP) + snapshot.headroom)", "    limit = min(MAX, min(abs(current), MAX) + snapshot.headroom)")
            with self.assertRaises(ValueError, msg=name):
                shield_proofs.check(source)

    def test_unsupported_or_control_reading_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "logged-only"):
            shield_proofs.check(self.mutant("    limited = _clamp(target, -limit, limit)\n", "    limited = _clamp(target, -limit, limit) + reasons\n"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            shield_proofs.check(self.mutant("    limited = _clamp(target, -limit, limit)\n", "    limited = _clamp(target, -limit, limit) * 2\n"))

    def test_vacuous_premise_is_rejected(self):
        import z3 as z

        x = z.Int("x")
        with self.assertRaisesRegex(ValueError, "vacuous"):
            shield_proofs.certify(z.And(x > 0, x < 0), x == 7)
        with self.assertRaisesRegex(ValueError, "vacuous"):
            shield_proofs.certify(x > 0, x > 1)


class ModelAndSurfaceMutants(unittest.TestCase):
    def test_non_atomic_reservation_counterexample_is_preserved(self):
        result = lifecycle_model.explore(mutant=True)
        self.assertEqual(result["violation"], "budget")
        self.assertEqual(result["trace"][-1], "p1:submit_unsafe")
        self.assertIsNone(lifecycle_model.explore()["violation"])

    def test_violation_detectors_fire(self):
        state = lifecycle_model.initial()
        live = state[:5] + (True,)
        self.assertIn("no-live", set(lifecycle_model.violations(live)))
        promoted = ("research_replay",) + state[1:]
        self.assertIn("no-self-promo", set(lifecycle_model.violations(promoted, state, "p0:observe")))
        self.assertIn("disabled", set(lifecycle_model.violations(state, state, "p0:observe")))

    def test_surface_rejects_capability_reach(self):
        base = {"x.py": "import json\n"}
        self.assertEqual(surface.audit_python(base)["modules"], 1)
        for bad in ("import requests\n", "import os\nKEY = os.environ['HOME']\n", "import subprocess\nsubprocess.run(['rm'])\n", "NAME = 'BINANCE_API_KEY'\n", "import historical_datafeed\n", "FLAG = '--binance-live'\n"):
            with self.assertRaises(ValueError, msg=bad):
                surface.audit_python({"x.py": bad})
        good = (ROOT / surface.HASKELL).read_text()
        cabal = (ROOT / "haskell/trader.cabal").read_text()
        self.assertEqual(surface.audit_haskell(good, cabal, {})["imports"], 0)
        for bad in (good.replace("    ResearchIntent,\n", "    ResearchIntent (..),\n"), good.replace("orderAuthorized _ = False", "orderAuthorized _ = True"), good.replace("module ExposureShieldV1 (", "import System.IO\nmodule ExposureShieldV1 (")):
            with self.assertRaises(ValueError):
                surface.audit_haskell(bad, cabal, {})
        with self.assertRaises(ValueError):
            surface.audit_haskell(good, cabal + "\n    other-modules: ExposureShieldV1\n", {})
        with self.assertRaises(ValueError):
            surface.audit_haskell(good, cabal, {"haskell/app/Main.hs": "import ExposureShieldV1\n"})


class LedgerIntegrity(unittest.TestCase):
    def setUp(self):
        self.ledger = json.loads((ROOT / verify.LEDGER).read_text())
        self.certificate = {k: v for k, v in json.loads((ROOT / verify.RECEIPT).read_text()).items() if k not in ("obligations", "sourceHashes")}

    def test_reviewed_ledger_validates(self):
        closure = verify.validate_ledger(self.ledger, self.certificate)
        self.assertEqual(sorted(closure), list(verify.OBLIGATIONS))

    def test_overclaims_fail(self):
        def entry(ledger, name):
            return next(e for e in ledger["entries"] if e["requirementId"] == name)

        mutations = {
            "property test called a proof": lambda l: entry(l, "F-RLX-LEDGER-IMPL").__setitem__("status", "smt_verified"),
            "open item called verified": lambda l: entry(l, "F-RLX-PRODUCTION-INTEGRATION").__setitem__("status", "exhaustively_checked"),
            "proved claim dropped": lambda l: l["entries"].remove(entry(l, "F-RLX-SHIELD-CAP")),
            "obligation overstated": lambda l: l["obligations"][0].__setitem__("status", "discharged_in_scope"),
            "open items hidden": lambda l: l["obligations"][2].__setitem__("status", "discharged_in_scope"),
            "missing test marker": lambda l: entry(l, "F-RLX-DETERMINISM").__setitem__("result", "Passes `test_that_does_not_exist`."),
            "missing file": lambda l: entry(l, "F-RLX-SURFACE")["implementationFiles"].append("scripts/research/exposure_rl/absent.py"),
            "unknown assumption": lambda l: entry(l, "F-RLX-SHIELD-CAP")["assumptions"].append("A-UNKNOWN"),
        }
        for name, mutate in mutations.items():
            ledger = copy.deepcopy(self.ledger)
            mutate(ledger)
            with self.assertRaises(ValueError, msg=name):
                verify.validate_ledger(ledger, self.certificate)

    def test_unreproduced_certificate_fails(self):
        certificate = copy.deepcopy(self.certificate)
        del certificate["smt"]["F-RLX-SHIELD-REDUCE"]
        with self.assertRaises(ValueError):
            verify.validate_ledger(self.ledger, certificate)
        certificate = copy.deepcopy(self.certificate)
        certificate["lifecycle"]["mutantCounterexample"]["violation"] = None
        with self.assertRaises(ValueError):
            verify.validate_ledger(self.ledger, certificate)


if __name__ == "__main__":
    unittest.main()
