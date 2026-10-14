"""Failure-oriented tests for archive identity and replication comparisons."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
import compare_rl_replication as r
import compare_rl_penalty_replication as penalty


class ReplicationTests(unittest.TestCase):
    def test_explicit_legacy_horizon_is_checked(self):
        self.assertEqual(r.legacy_horizon({}), r.legacy_horizon({'trainingEpisodeTransitions': 32}))
        for value in (64, 32., True, None, '32'):
            with self.assertRaisesRegex(ValueError, 'horizon changed'):
                r.legacy_horizon({'trainingEpisodeTransitions': value})

    def test_strict_json(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e400}'):
            with self.assertRaises(ValueError):
                r.decode(raw)

    def test_types_discrete_actions_and_numeric_tolerance(self):
        r.same({'return': 0.1}, {'return': 0.1 + 1e-14})
        for a, b in ((True, 1), (1, 1.0), (0, 2), ([1], [1, 2]),
                     ({'status': 'failed'}, {'status': 'complete'}),
                     ({'return': 0.1}, {'return': 0.11}),
                     (float('nan'), float('nan'))):
            with self.assertRaises(ValueError):
                r.same(a, b)

    def test_snapshot_rejects_tampering_and_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b'{"status":"complete"}'
            (root/'data.json').write_bytes(data)
            raw = json.dumps({'data.json': hashlib.sha256(data).hexdigest()}).encode()
            (root/'index.json').write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            admitted = r.snapshot(root, digest)
            (root/'data.json').write_bytes(b'{"status":"failed"}')
            self.assertEqual(r.decode(admitted['data.json'])['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'entry hash'):
                r.snapshot(root, digest)
            (root/'data.json').write_bytes(data)
            (root/'unindexed').write_text('x')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                r.snapshot(root, digest)
            with self.assertRaisesRegex(ValueError, 'index hash'):
                r.snapshot(root, '0'*64)

    def test_parent_path_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').write_bytes(b'{}')
            (root/'alias').symlink_to(root/'data')
            for name in ('../escape', '/absolute', 'alias'):
                raw = json.dumps({name: hashlib.sha256(b'{}').hexdigest()}).encode()
                (root/'index.json').write_bytes(raw)
                with self.assertRaises(ValueError):
                    r.snapshot(root, hashlib.sha256(raw).hexdigest())

    def test_parameters_decoded_not_hidden_by_artifact_identity(self):
        a = {'mode': 'research_only', 'sourceCommit': 'old',
             'assuranceLedgerSha256': 'old',
             'parametersLittleEndianHex': [np.zeros(n, dtype='<f8').tobytes().hex()
                                           for n in (192, 16, 48, 3)]}
        b = {**a, 'sourceCommit': 'new', 'assuranceLedgerSha256': 'new'}
        self.assertEqual(r.policy_parameters(a, b), 259)
        self.assertEqual(r.policy_parameters(a, {**b, 'trainingEpisodeTransitions': 32}), 259)
        with self.assertRaisesRegex(ValueError, 'horizon changed'):
            r.policy_parameters(a, {**b, 'trainingEpisodeTransitions': 64})
        b['parametersLittleEndianHex'] = list(a['parametersLittleEndianHex'])
        b['parametersLittleEndianHex'][3] = np.ones(3, dtype='<f8').tobytes().hex()
        with self.assertRaisesRegex(ValueError, 'parameter mismatch'):
            r.policy_parameters(a, b)
        b['parametersLittleEndianHex'][3] = np.zeros(4, dtype='<f8').tobytes().hex()
        with self.assertRaisesRegex(ValueError, 'parameter shape'):
            r.policy_parameters(a, b)

    def test_incomplete_roster_cannot_pass(self):
        data = {'manifest.json': json.dumps({'campaign': 'rl-training-budget-v3',
                    'promotionAllowed': False, 'holdoutOpened': False}).encode(),
                'summary.json': json.dumps({'promotionAllowed': False, 'holdoutOpened': False,
                    'fits': [], 'records': []}).encode(), 'planned.json': b'[]'}
        with self.assertRaisesRegex(ValueError, 'roster'):
            r.reconcile(data)


class PenaltyReplicationTests(unittest.TestCase):
    def policy(self):
        return {'mode': 'research_only', 'algorithm': 'ppo', 'sourceCommit': 'old',
                'assuranceLedgerSha256': 'old', 'trainingEpisodeTransitions': 64,
                'trainingRewardVersion': penalty.REWARD_VERSION,
                'trainingPenaltyPerChangedProposal': .02,
                'parametersLittleEndianHex': [np.zeros(n, dtype='<f8').tobytes().hex()
                                              for n in (192, 16, 48, 3)]}

    def test_v6_parameters_and_reward_identity(self):
        p = self.policy()
        self.assertEqual(penalty.parameters(p, {**p, 'sourceCommit': 'new'}), 259)
        for key, value in [('trainingEpisodeTransitions', 32), ('trainingEpisodeTransitions', 64.),
                           ('trainingRewardVersion', 'other'), ('mode', 'live'),
                           ('trainingPenaltyPerChangedProposal', 0.), ('algorithm', 'cql')]:
            with self.assertRaises(ValueError): penalty.parameters(p, {**p, key: value})
        for value in (1., float('nan'), float('inf')):
            q = self.policy()
            q['parametersLittleEndianHex'][3] = np.full(3, value, dtype='<f8').tobytes().hex()
            with self.assertRaises(ValueError): penalty.parameters(p, q)

    def test_training_and_economic_trace_fields_not_excluded(self):
        row = {'reward': 1., 'trainingReward': .98, 'trainingPenalty': .02,
               'actionChanged': True, 'proposal': 2, 'terminal': False,
               'trainingEpisode': 1, 'equity': 1.01}
        raw = json.dumps([row]).encode()
        near = json.dumps([{**row, 'trainingReward': .98 + 1e-14}]).encode()
        self.assertEqual(penalty.compare_trace('training-test.json', raw, near), 1)
        for key, value in [('reward', 1.1), ('trainingReward', 1.), ('trainingPenalty', 0.),
                           ('actionChanged', 1), ('proposal', 2.), ('terminal', True),
                           ('trainingEpisode', 2), ('equity', 1.02)]:
            for name in ('training-test.json', 'paths/test.json'):
                with self.assertRaises(ValueError):
                    penalty.compare_trace(name, raw, json.dumps([{**row, key: value}]).encode())
        with self.assertRaises(ValueError): penalty.compare_trace('training-test.json', raw, b'[]')
        with self.assertRaises(ValueError): penalty.compare_trace('training-test.json', b'{}', b'{}')

    def test_only_registered_summary_fields_can_differ(self):
        base = {'elapsedSeconds': 1., 'maxRss': 1, 'rssUnit': 'bytes', 'promotionAllowed': False,
                'fits': [{'id': '0/ppo/11/training', 'seconds': 1., 'inference': {},
                          'artifactSha256': 'a', 'artifactBytes': 1, 'trainingRewardTraceSha256': 'a',
                          'lossLast': .2, 'trainingRewardAudit': {'economicRewardSum': 2.}}],
                'records': [{'netReturn': .01}]}
        from copy import deepcopy
        timing = deepcopy(base)
        timing.update(elapsedSeconds=9., maxRss=9)
        timing['fits'][0].update(seconds=9., artifactSha256='b', trainingRewardTraceSha256='b')
        r.same(penalty.normalized_summary(base), penalty.normalized_summary(timing))
        for key, value in [('lossLast', .3), ('trainingRewardAudit', {'economicRewardSum': 3.})]:
            changed = deepcopy(base); changed['fits'][0][key] = value
            with self.assertRaises(ValueError):
                r.same(penalty.normalized_summary(base), penalty.normalized_summary(changed))
        changed = deepcopy(base); changed['records'][0]['netReturn'] = .02
        with self.assertRaises(ValueError):
            r.same(penalty.normalized_summary(base), penalty.normalized_summary(changed))

    def test_v6_incomplete_roster_and_changed_sources_reject(self):
        reg = r.decode(penalty.REGISTRATION.read_bytes())
        manifest = {'campaign': 'rl-proposal-penalty-v6', 'algorithms': ['ppo'],
                    'trainingEpisodeTransitions': 64, 'promotionAllowed': False, 'holdoutOpened': False,
                    'registrationSha256': penalty.digest(penalty.REG.read_bytes()),
                    'sourceHashes': reg['requiredSourceHashes'], 'python': '3.13.3', 'numpy': '2.3.5'}
        summary = {'promotionAllowed': False, 'holdoutOpened': False, 'fits': [], 'records': []}
        def data():
            return {'manifest.json': json.dumps(manifest).encode(),
                    'summary.json': json.dumps(summary).encode(), 'planned.json': b'[]'}
        with self.assertRaisesRegex(ValueError, 'roster'): penalty.reconcile(data(), reg)
        manifest['sourceHashes'] = {**manifest['sourceHashes'], 'unreviewed': '0'*64}
        with self.assertRaisesRegex(ValueError, 'sources'): penalty.reconcile(data(), reg)


if __name__ == '__main__':
    unittest.main()
