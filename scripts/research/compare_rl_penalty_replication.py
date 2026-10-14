"""Strict v6 numerical replication, including every shaped/economic training row.

No training, network, policy inference or production interface. Archive hashes
are caller-supplied trust anchors, not signatures or evidence of profitability.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from compare_rl_replication import ATOL, RTOL, decode, require, same, snapshot, without
from rl_composition import buffer_hash
from run_rl_penalty_v6 import REG, REWARD_VERSION, RULES
from summarize_rl_composition import audit_rows
from summarize_rl_penalty_v6 import training_audit

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / 'research-notes/registrations/rl-penalty-reproduction-2026-10-14.json'


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parameters(left: dict, right: dict) -> int:
    for value in (left, right):
        require(type(value['trainingEpisodeTransitions']) is int and
                value['trainingEpisodeTransitions'] == 64, 'wrong v6 horizon')
        require(value['mode'] == 'research_only' and value['algorithm'] == 'ppo' and
                value['trainingRewardVersion'] == REWARD_VERSION and
                type(value['trainingPenaltyPerChangedProposal']) is float and
                value['trainingPenaltyPerChangedProposal'] == .02, 'wrong reward policy')
    same(without(left, ('sourceCommit', 'assuranceLedgerSha256', 'parametersLittleEndianHex')),
         without(right, ('sourceCommit', 'assuranceLedgerSha256', 'parametersLittleEndianHex')),
         'policy metadata')
    groups = [v['parametersLittleEndianHex'] for v in (left, right)]
    require(all(type(g) is list and len(g) == 4 for g in groups), 'parameter groups')
    for a, b, width in zip(*groups, (192, 16, 48, 3)):
        a, b = (np.frombuffer(bytes.fromhex(x), dtype='<f8') for x in (a, b))
        require(a.shape == b.shape == (width,) and np.isfinite(a).all() and
                np.isfinite(b).all(), 'parameter shape/finiteness')
        require(np.allclose(a, b, atol=ATOL, rtol=RTOL), 'trained parameter mismatch')
    return 259


def reconcile(data: dict[str, bytes], registration: dict) -> tuple[dict, dict, dict]:
    manifest, summary, planned = (decode(data[n]) for n in ('manifest.json', 'summary.json', 'planned.json'))
    for value in (manifest, summary):
        require(value['promotionAllowed'] is False and value['holdoutOpened'] is False,
                'research authority mismatch')
    require(manifest['campaign'] == 'rl-proposal-penalty-v6' and
            manifest['algorithms'] == ['ppo'] and manifest['trainingEpisodeTransitions'] == 64 and
            manifest['registrationSha256'] == digest(REG.read_bytes()), 'wrong campaign')
    same(manifest['sourceHashes'], registration['requiredSourceHashes'], 'frozen experiment sources')
    require(manifest['python'] == '3.13.3' and manifest['numpy'] == '2.3.5', 'runtime version drift')
    original = decode(REG.read_bytes())
    expected = {f'{fold}/ppo/{seed}/training' for fold in (0, 1) for seed in original['seeds']}
    expected |= {f'{fold}/{alg}/{seed}/{stress}/{symbol}' for fold in (0, 1)
                 for alg in ('ppo', *RULES) for seed in (original['seeds'] if alg == 'ppo' else [0])
                 for stress in original['stresses'] for symbol in original['data']['symbols']}
    rows = summary['fits'] + summary['records']
    ids = [r['id'] for r in rows]
    require(len(ids) == len(set(ids)) == len(planned) == len(set(planned)) == 388 and
            set(ids) == set(planned) == expected and
            summary['planned'] == summary['terminal'] == 388 and
            len(summary['fits']) == 10 and len(summary['records']) == 378, 'incomplete v6 roster')
    events = [decode(line) for line in data['events.jsonl'].splitlines()]
    same(sorted(events, key=lambda r: r['id']), sorted(rows, key=lambda r: r['id']), 'terminal events')
    audit = {'fits': 10, 'parameters': 0, 'trainingRows': 0, 'trainingEpisodes': 0,
             'evaluationPaths': 0, 'evaluationRows': 0, 'detectedViolations': 0}
    for fit in summary['fits']:
        require(fit['status'] == 'complete', 'failed fit cannot pass replication')
        fold, seed = fit['fold'], fit['seed']
        require(fit['algorithm'] == 'ppo' and fit['id'] == f'{fold}/ppo/{seed}/training' and
                fold in (0, 1) and seed in original['seeds'] and
                fit['steps'] == fit['collectedTransitions'] == 4096 and fit['updates'] == 128,
                'fit identity or budget mismatch')
        raw = data[f'policies/{fold}-ppo-{seed}.json']; policy = decode(raw)
        require(digest(raw) == fit['artifactSha256'] and policy['sourceCommit'] == manifest['commit'] and
                policy['assuranceLedgerSha256'] == manifest['proofLedgerSha256'] and
                policy['registrationSha256'] == manifest['registrationSha256'] and
                policy['fold'] == fold and policy['seed'] == seed and policy['updates'] == 128,
                'policy identity mismatch')
        audit['parameters'] += parameters(policy, policy)
        raw = data[f'training-{fold}-ppo-{seed}.json']
        require(digest(raw) == fit['trainingRewardTraceSha256'], 'training trace identity mismatch')
        trace = decode(raw)
        require(all(r['trainingSymbol'] in original['data']['symbols'] for r in trace), 'training universe')
        checked = training_audit(trace, 4096, original['folds'][fold]['trainStop'])
        same(checked, fit['trainingRewardAudit'], 'training reward audit')
        audit['trainingRows'] += checked['transitions']
        audit['trainingEpisodes'] += checked['episodes']
    for row in summary['records']:
        require(row['status'] == 'complete', 'failed evaluation cannot pass replication')
        path = decode(data['paths/' + row['id'].replace('/', '-') + '.json'])
        require(len(path) == row['observations'] == 64 and path[-1]['terminal'] is True,
                'incomplete evaluation path')
        same(float(sum(r['increment'] for r in path)), row['incrementSum'], 'evaluation sum')
        checked = audit_rows(path)
        require(not checked['violations'], 'economic audit violation')
        audit['evaluationPaths'] += 1
        audit['evaluationRows'] += len(path)
    for fold, wanted in enumerate(original['parentArchive']['behaviorHashes']):
        with np.load(io.BytesIO(data[f'behavior-fold{fold}.npz']), allow_pickle=False) as loaded:
            require(buffer_hash(dict(loaded)) == wanted, 'frozen behavior buffer changed')
    return manifest, summary, audit


def normalized_summary(value: dict) -> dict:
    return {**without(value, ('elapsedSeconds', 'maxRss', 'rssUnit', 'fits')),
            'fits': [without(r, ('seconds', 'inference', 'artifactSha256', 'artifactBytes',
                                 'trainingRewardTraceSha256')) for r in value['fits']]}


def compare_trace(name: str, left: bytes, right: bytes) -> int:
    value = decode(left)
    require(type(value) is list, 'trace must be a list')
    same(value, decode(right), name)
    return len(value)


def compare(parent: Path, replica: Path, parent_sha: str, replica_sha: str) -> dict:
    reg = decode(REGISTRATION.read_bytes())
    require(parent.resolve() != replica.resolve(), 'replica must be a separate archive')
    require(parent_sha == reg['parentIndexSha256'], 'unregistered parent')
    a, b = snapshot(parent, parent_sha), snapshot(replica, replica_sha)
    require(a.keys() == b.keys(), 'archive inventory differs')
    am, left, aa = reconcile(a, reg)
    bm, right, ba = reconcile(b, reg)
    require(am['commit'] == reg['parentTrainingCommit'] and am['commit'] != bm['commit'],
            'source provenance does not identify a new run')
    same(without(am, ('commit', 'proofLedgerSha256', 'platform')),
         without(bm, ('commit', 'proofLedgerSha256', 'platform')), 'manifest')
    same(normalized_summary(left), normalized_summary(right), 'summary')
    same(aa, ba, 'audited counts')
    parameter_count = training_rows = evaluation_rows = 0
    for name in a:
        if name.startswith('policies/'):
            parameter_count += parameters(decode(a[name]), decode(b[name]))
        elif name.startswith('paths/') or name.startswith('training-'):
            count = compare_trace(name, a[name], b[name])
            if name.startswith('paths/'): evaluation_rows += count
            else: training_rows += count
        elif (name.startswith('behavior-') and name.endswith('.json')) or name == 'planned.json':
            same(decode(a[name]), decode(b[name]), name)
        elif name.endswith('.npz'):
            with np.load(io.BytesIO(a[name]), allow_pickle=False) as x, np.load(io.BytesIO(b[name]), allow_pickle=False) as y:
                require(set(x.files) == set(y.files), 'buffer fields differ')
                for key in x.files:
                    require(x[key].dtype == y[key].dtype and np.array_equal(x[key], y[key]), 'buffer differs')
    require(parameter_count == 2590 and training_rows == 40960 and evaluation_rows == 24192,
            'comparison coverage mismatch')
    return dict(schema='rl-penalty-replication-v1', status='reproduced_within_tolerance',
                parentIndexSha256=parent_sha, replicaIndexSha256=replica_sha,
                parentCommit=am['commit'], replicaCommit=bm['commit'],
                parentProofLedgerSha256=am['proofLedgerSha256'], replicaProofLedgerSha256=bm['proofLedgerSha256'],
                parentPlatform=am['platform'], replicaPlatform=bm['platform'],
                sourceDifferences={}, registrationSha256=digest(REGISTRATION.read_bytes()),
                absoluteTolerance=ATOL, relativeTolerance=RTOL, audits=[aa, ba],
                parametersCompared=parameter_count, trainingRowsCompared=training_rows,
                evaluationRowsCompared=evaluation_rows, newIndependentMarketSamples=0,
                newSelectionTrials=0, promotionAllowed=False, holdoutOpened=False,
                scope='Same-runtime fresh-process retraining and full-trace numerical comparison; no cross-hardware guarantee, mathematical proof or financial confirmation.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'replica', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    for name in ('parent-sha256', 'replica-sha256'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    require(all(destination != p.resolve() and p.resolve() not in destination.parents
                for p in (args.parent, args.replica)), 'report must be outside archives')
    result = compare(args.parent, args.replica, args.parent_sha256, args.replica_sha256)
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
