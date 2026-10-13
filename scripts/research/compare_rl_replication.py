"""Compare two complete RL archives; replication is not independent evidence.

Read-only input snapshots, explicit index trust anchors, no policy loading or
exchange client. Only the caller-selected new report file is written.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np

ATOL = RTOL = 1e-12
HEX = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def decode(raw: bytes):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=number)


def snapshot(root: Path, expected: str) -> dict[str, bytes]:
    require(type(expected) is str and HEX.fullmatch(expected), "invalid index digest")
    root = root.resolve(strict=True)
    index_path = root / "index.json"
    require(not index_path.is_symlink() and index_path.stat().st_size <= 2**20,
            "unsafe or oversized index")
    raw = index_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "index hash mismatch")
    index = decode(raw)
    require(type(index) is dict and 1 <= len(index) <= 10000, "invalid index")
    out = {}
    size = 0
    for name, digest in index.items():
        relative = Path(name)
        require(name != 'index.json' and str(relative) == name and
                not relative.is_absolute() and '..' not in relative.parts and
                type(digest) is str and HEX.fullmatch(digest), "unsafe index entry")
        path = root / relative
        require(all(not p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents),
                "symlink in archive")
        require(root in path.resolve(strict=True).parents, "path escapes archive")
        require(path.stat().st_size <= 8 * 2**20, "oversized archive entry")
        content = path.read_bytes()
        size += len(content)
        require(len(content) <= 8 * 2**20 and size <= 256 * 2**20, "archive size limit")
        require(hashlib.sha256(content).hexdigest() == digest, "entry hash mismatch: " + name)
        out[name] = content
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(index) | {'index.json'}, "archive inventory mismatch")
    # All subsequent parsing uses the same verified byte snapshots.
    return out


def same(left, right, location: str = 'value') -> None:
    require(type(left) is type(right), "type mismatch: " + location)
    if type(left) is dict:
        require(left.keys() == right.keys(), "field mismatch: " + location)
        for key in left:
            same(left[key], right[key], location + '/' + key)
    elif type(left) is list:
        require(len(left) == len(right), "length mismatch: " + location)
        for i, (a, b) in enumerate(zip(left, right)):
            same(a, b, location + '/' + str(i))
    elif type(left) is float:
        require(math.isfinite(left) and math.isfinite(right) and
                math.isclose(left, right, abs_tol=ATOL, rel_tol=RTOL),
                "numeric mismatch: " + location)
    else:
        require(left == right, "value mismatch: " + location)


def without(value: dict, fields: tuple[str, ...]) -> dict:
    return {k: v for k, v in value.items() if k not in fields}


def policy_parameters(left: dict, right: dict) -> int:
    same(without(left, ('sourceCommit', 'assuranceLedgerSha256', 'parametersLittleEndianHex')),
         without(right, ('sourceCommit', 'assuranceLedgerSha256', 'parametersLittleEndianHex')),
         'policy metadata')
    a, b = left['parametersLittleEndianHex'], right['parametersLittleEndianHex']
    require(len(a) == len(b) == 4, "parameter group mismatch")
    count = 0
    for x, y, width in zip(a, b, (192, 16, 48, 3)):
        x = np.frombuffer(bytes.fromhex(x), dtype='<f8')
        y = np.frombuffer(bytes.fromhex(y), dtype='<f8')
        require(x.shape == y.shape == (width,) and np.isfinite(x).all() and np.isfinite(y).all(),
                "invalid parameter shape or value")
        require(np.allclose(x, y, atol=ATOL, rtol=RTOL), "trained parameter mismatch")
        count += width
    return count


def reconcile(data: dict[str, bytes]) -> None:
    manifest, summary, planned = (decode(data[n]) for n in ('manifest.json', 'summary.json', 'planned.json'))
    require(manifest['campaign'] == 'rl-training-budget-v3' and
            manifest['promotionAllowed'] is False and manifest['holdoutOpened'] is False and
            summary['promotionAllowed'] is False and summary['holdoutOpened'] is False,
            "wrong research campaign or authority")
    rows = summary['fits'] + summary['records']
    ids = [r['id'] for r in rows]
    require(len(ids) == len(set(ids)) == len(planned) == len(set(planned)) == 1268 and
            set(ids) == set(planned) and summary['planned'] == summary['terminal'] == 1268 and
            len(summary['fits']) == 50 and len(summary['records']) == 1218,
            "incomplete registered roster")
    events = [decode(line) for line in data['events.jsonl'].splitlines()]
    require(len(events) == len(rows) and len({e['id'] for e in events}) == len(rows),
            "duplicate or missing terminal events")
    same(sorted(events, key=lambda r: r['id']), sorted(rows, key=lambda r: r['id']), 'events')
    for fit in summary['fits']:
        require(fit['status'] == 'complete', "failed fit cannot pass replication acceptance")
        name = f"policies/{fit['fold']}-{fit['algorithm']}-{fit['seed']}.json"
        require(hashlib.sha256(data[name]).hexdigest() == fit['artifactSha256'], "fit artifact identity")
    for row in summary['records']:
        require(row['status'] == 'complete', "failed replay cannot pass replication acceptance")
        name = 'paths/' + row['id'].replace('/', '-') + '.json'
        path = decode(data[name])
        require(len(path) == row['observations'] == 64 and path[-1]['terminal'] is True,
                "incomplete economic path")
        same(float(sum(r['increment'] for r in path)), row['incrementSum'], 'path sum')


def compare(parent: Path, replica: Path, parent_sha: str, replica_sha: str) -> dict:
    a, b = snapshot(parent, parent_sha), snapshot(replica, replica_sha)
    require(a.keys() == b.keys(), "archive file set differs")
    reconcile(a)
    reconcile(b)
    am, bm = decode(a['manifest.json']), decode(b['manifest.json'])
    same(without(am, ('commit', 'sourceHashes', 'proofLedgerSha256', 'platform',
                     'behaviorIdentityPrefix', 'algorithms', 'pairedAlgorithms')),
         without(bm, ('commit', 'sourceHashes', 'proofLedgerSha256', 'platform',
                     'behaviorIdentityPrefix', 'algorithms', 'pairedAlgorithms')), 'manifest')
    # Later runner revisions added explicit default subset metadata. Verify its
    # precise legacy meaning; do not ignore arbitrary algorithm changes.
    defaults = {'algorithms': ['ppo', 'double_dqn', 'cql', 'cql_alpha0', 'behavior_cloning'],
                'pairedAlgorithms': ['ppo', 'double_dqn', 'cql'], 'behaviorIdentityPrefix': None}
    for key, default in defaults.items():
        same(am.get(key, default), default, 'parent/'+key)
        same(bm.get(key, default), default, 'replica/'+key)
    sums = [decode(d['summary.json']) for d in (a, b)]
    for value in sums:
        for fit in value['fits']:
            for field in ('seconds', 'inference', 'artifactSha256', 'artifactBytes'):
                fit.pop(field, None)
    same(without(sums[0], ('elapsedSeconds', 'maxRss', 'rssUnit')),
         without(sums[1], ('elapsedSeconds', 'maxRss', 'rssUnit')), 'summary')
    parameters = paths = rows = 0
    for name in a:
        if name.startswith('policies/'):
            parameters += policy_parameters(decode(a[name]), decode(b[name]))
        elif name.startswith('paths/'):
            path = decode(a[name])
            same(path, decode(b[name]), name)
            paths += 1
            rows += len(path)
        elif name.endswith('.npz'):
            with np.load(io.BytesIO(a[name]), allow_pickle=False) as x, np.load(io.BytesIO(b[name]), allow_pickle=False) as y:
                require(set(x.files) == set(y.files), "buffer fields differ")
                for key in x.files:
                    require(x[key].dtype == y[key].dtype and np.array_equal(x[key], y[key]), "frozen dataset differs")
        elif name.startswith('behavior-'):
            left, right = decode(a[name]), decode(b[name])
            # v4 records full-buffer identity explicitly when no prefix is used.
            for value in (left, right):
                value.setdefault('identityPrefixTransitions', None)
                value.setdefault('identitySha256', value['sha256'])
            same(left, right, name)
        elif name == 'planned.json':
            same(decode(a[name]), decode(b[name]), name)
    return {'schema': 'rl-replication-v1', 'status': 'reproduced_within_tolerance',
            'parentIndexSha256': parent_sha, 'replicaIndexSha256': replica_sha,
            'parentCommit': am['commit'], 'replicaCommit': bm['commit'],
            'parentProofLedgerSha256': am['proofLedgerSha256'],
            'replicaProofLedgerSha256': bm['proofLedgerSha256'],
            'parentPlatform': am['platform'], 'replicaPlatform': bm['platform'],
            'sourceDifferences': {k: [am['sourceHashes'].get(k), bm['sourceHashes'].get(k)]
                for k in sorted(am['sourceHashes'].keys() | bm['sourceHashes'].keys())
                if am['sourceHashes'].get(k) != bm['sourceHashes'].get(k)},
            'absoluteTolerance': ATOL, 'relativeTolerance': RTOL, 'fits': 50,
            'parametersCompared': parameters, 'pathsCompared': paths, 'rowsCompared': rows,
            'newIndependentMarketSamples': 0, 'newSelectionTrials': 0,
            'promotionAllowed': False, 'holdoutOpened': False,
            'scope': 'Same-runtime numerical and full-path replication, not mathematical proof or financial confirmation.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'replica', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('parent-sha256', 'replica-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    require(all(destination != root.resolve() and root.resolve() not in destination.parents
                for root in (args.parent, args.replica)), 'report must be outside both archives')
    result = compare(args.parent, args.replica, args.parent_sha256, args.replica_sha256)
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
