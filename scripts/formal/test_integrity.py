"""Regression tests for fail-closed certificate admission and model mutations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lifecycle
import gap_risk
import gap_conformance
import z3
from verify import ROOT, read_json, validate_ledger


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.ledger = read_json(ROOT / 'formal/research/proof-ledger.json')

    def test_valid_ledger_keeps_critical_blockers(self):
        self.assertGreater(validate_ledger(self.ledger, ROOT), 0)

    def test_mutated_ledgers_fail(self):
        changes = [lambda x: x['entries'].append(copy.deepcopy(x['entries'][0])),
                   lambda x: x['entries'][0].update(status='fully_verified'),
                   lambda x: x['entries'][0].update(formalStatement='a different theorem'),
                   lambda x: x['entries'][11].update(relatedRequirements=[]),
                   lambda x: x['criticalFiles'].remove('scripts/formal/proofs.py'),
                   lambda x: x['entries'][9].update(status='proved'),
                   lambda x: x['missionObligations'][0].update(status='proved'),
                   lambda x: x['entries'][0].update(implementationFiles=[]),
                   lambda x: x['entries'][0].update(assumptions=['unknown']),
                   lambda x: x['entries'][0].update(artifact='missing-result.json'),
                   lambda x: x['missionObligations'].pop(),
                   lambda x: x['criticalFiles'].append('unmapped-critical.py')]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(self.ledger)
                change(data)
                with self.assertRaises(ValueError):
                    validate_ledger(data, ROOT)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            for source in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
                path.write_text(source)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_live_authority_mutation_detected(self):
        with patch.object(lifecycle, 'authority', lambda state: state[0]):
            with self.assertRaisesRegex(RuntimeError, 'unsafe state'):
                lifecycle.check_model()

    def test_start_during_drain_mutation_detected(self):
        original = lifecycle.successors
        def mutated(state):
            yield from original(state)
            if state[1]:
                yield 'start0:1', (False, True, 1, state[3])
        with patch.object(lifecycle, 'successors', mutated):
            with self.assertRaisesRegex(RuntimeError, 'start while draining'):
                lifecycle.check_model()

    def test_disable_does_not_revoke_immutable_proposal(self):
        state = lifecycle.INITIAL
        for event in read_json(ROOT / 'formal/research/counterexamples.json')['entries'][0]['trace']:
            state = dict(lifecycle.successors(state))[event]
        self.assertFalse(state[0])
        self.assertIn(3, state[2:])
        self.assertFalse(lifecycle.authority(state))

    def test_gap_witnesses_replay_and_accounting_grid(self):
        document = read_json(ROOT / 'formal/research/gap-counterexamples.json')
        self.assertEqual(len(gap_risk.check_counterexamples(document)), 2)
        self.assertEqual(gap_conformance.check_replay(document)['boundedAccountingTraces'], 180)

    def test_invalid_gap_witness_fails(self):
        document = read_json(ROOT / 'formal/research/gap-counterexamples.json')
        document['entries'][0]['endingEquity'] = '1'
        with self.assertRaisesRegex(ValueError, 'invalid loss-floor witness'):
            gap_risk.check_counterexamples(document)

    def test_gap_smt_cannot_accept_false_or_vacuous_claims(self):
        for premise, conclusion in ((z3.BoolVal(True), z3.BoolVal(False)),
                                    (z3.BoolVal(False), z3.BoolVal(True))):
            with self.subTest(premise=premise), patch.object(
                    gap_risk, 'obligations', lambda: {'mutant': (premise, conclusion)}):
                with self.assertRaises(RuntimeError):
                    gap_risk.check_obligations()

    def test_gap_conformance_detects_dropped_costs(self):
        original = gap_conformance.Replay._trade
        def mutated(replay, target, terminal=False):
            before = replay.equity
            result = original(replay, target, terminal)
            replay.equity = before
            return result
        with patch.object(gap_conformance.Replay, '_trade', mutated):
            with self.assertRaisesRegex(RuntimeError, 'counterexample mismatch'):
                gap_conformance.check_replay(read_json(ROOT / 'formal/research/gap-counterexamples.json'))


if __name__ == '__main__':
    unittest.main()
