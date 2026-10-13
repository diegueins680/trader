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


class ReplicationTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
