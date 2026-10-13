"""Hash-verified horizon comparison; retain every seed and unchanged controls."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import numpy as np
from compare_rl_observations_v2 import archive
from compare_rl_replication import decode, reconcile, same, snapshot
from rl_composition import buffer_hash
from run_rl_horizon_v5 import REG, RULES, configuration
from run_rl_composition import ALGORITHMS, require, write
from summarize_rl_composition import export, read, sha
from summarize_rl_offline_data_v4 import aggregates, coverage


def reference(current, previous):
    prior = {r['id']: r for r in previous['records']}
    controls = [r for r in current['records'] if r['algorithm'] in RULES]
    require(len(controls) == 168 and len({r['id'] for r in controls}) == 168,
            'missing or duplicate controls')
    require(all(r == prior.get(r['id']) for r in controls), 'deterministic baseline changed')
    return prior


def summarize(source, parent, output, expected):
    reg = read(REG); configuration(reg)
    data = snapshot(source, expected)
    reconcile(snapshot(parent, reg['parentArchive']['indexSha256']))
    current, manifest = archive(source, expected)
    previous, parent_manifest = archive(parent, reg['parentArchive']['indexSha256'])
    events = [decode(line) for line in data['events.jsonl'].splitlines()]
    rows = current['fits'] + current['records']
    same(sorted(events, key=lambda r:r['id']), sorted(rows, key=lambda r:r['id']), 'terminal events')
    expected_replays = {f'{f}/{a}/{s}/{stress}/{symbol}' for f in (0,1)
        for a in (*ALGORITHMS,*RULES) for s in (reg['seeds'] if a in ALGORITHMS else [0])
        for stress in reg['stresses'] for symbol in reg['data']['symbols']}
    require(len(current['records']) == 1218 and
            {r['id'] for r in current['records']} == expected_replays, 'replay roster mismatch')
    for row in current['records']:
        if row['status'] == 'complete':
            path = decode(data['paths/'+row['id'].replace('/','-')+'.json'])
            require(len(path) == row['observations'] == 64 and path[-1]['terminal'] is True,
                    'incomplete economic path')
            same(sum(r['increment'] for r in path),row['incrementSum'],'path sum')
    require(manifest['campaign'] == reg['id'] and
            manifest['registrationSha256'] == sha(REG.read_bytes()) and
            manifest['trainingEpisodeTransitions'] == 64 and
            manifest['version'] == parent_manifest['version'], 'wrong horizon packages')
    require(output.resolve() != parent.resolve() and parent.resolve() not in output.resolve().parents,
            'output must be outside parent archive')
    prior = reference(current, previous)
    for row in current['records']:
        if row['algorithm'] in RULES:
            name = '-'.join(map(str, (row['fold'],row['algorithm'],row['seed'],row['stress'],row['symbol'])))+'.json'
            require((source/'paths'/name).read_bytes() == (parent/'paths'/name).read_bytes(),
                    'deterministic baseline trace changed')
    expected_fits = {f'{f}/{a}/{s}/training' for f in (0,1) for a in ALGORITHMS for s in reg['seeds']}
    require(len(current['fits']) == 50 and {r['id'] for r in current['fits']} == expected_fits,
            'missing or duplicate training trials')
    for fit in current['fits']:
        if fit['status'] == 'complete':
            count = 4096 if fit['algorithm'] in ('ppo','double_dqn') else 0
            require(fit['collectedTransitions'] == count, 'training regime mismatch')
            raw = data[f"policies/{fit['fold']}-{fit['algorithm']}-{fit['seed']}.json"]
            artifact = decode(raw)
            require(sha(raw) == fit['artifactSha256'] and
                    artifact['trainingEpisodeTransitions'] == 64 and
                    artifact['registrationSha256'] == manifest['registrationSha256'] and
                    artifact['sourceCommit'] == manifest['commit'] and
                    artifact['mode'] == 'research_only', 'policy provenance mismatch')
    export(source, output, expected)
    buffers = []
    for fold in (0,1):
        with np.load(source/f'behavior-fold{fold}.npz', allow_pickle=False) as loaded:
            data = dict(loaded)
        meta = read(source/f'behavior-fold{fold}.json')
        require(buffer_hash(data) == meta['sha256'] and len(data['a']) == 512, 'buffer identity mismatch')
        buffers.append(dict(fold=fold, sha256=meta['sha256'], coverage=coverage(data),
                            generationRiskFailures=meta['trainingRiskFailures']))
    complete = all(r['status'] == 'complete' for r in current['records'])
    deltas = [dict(id=r['id'], algorithm=r['algorithm'], seed=r['seed'],
                  difference=r['netReturn']-prior[r['id']]['netReturn']
                  if r['status'] == prior[r['id']]['status'] == 'complete' else None)
              for r in current['records'] if r['algorithm'] in ALGORITHMS and r['stress'] == 'base']
    counts = Counter()
    for file in (source/'paths').glob('*.json'):
        for row in read(file):
            for key in ('actionChanged','supportChangedAction','turnoverRejected','schedulingDeferred','newProposalIgnored'):
                counts[key] += row[key]
    result = dict(campaign=reg['id'], indexSha256=expected,
        parentIndexSha256=reg['parentArchive']['indexSha256'], buffers=buffers,
        aggregates=aggregates(current['records']) if complete else [],
        comparisonStatus='complete' if complete else 'insufficient_failed_paths',
        parentHorizonBasePathDeltas=deltas, interventions=dict(counts),
        deterministicControlsMatched=168,
        fitTrainingFailures={r['id']:r.get('trainingFailures') for r in current['fits'] if r.get('trainingFailures')},
        criticMse={a:statistics.mean(r['criticMseFinitePath'] for r in current['records']
                                    if r['algorithm']==a and r['stress']=='base')
                   for a in ('double_dqn','cql','cql_alpha0')} if complete else {},
        interpretation='Adaptive horizon deltas are descriptive; matched horizon does not establish continuing-operation efficacy.',
        promotionAllowed=False, holdoutOpened=False)
    write(output/'horizon-comparison.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-index-sha256', required=True)
    args = parser.parse_args()
    summarize(args.source, args.parent, args.output, args.expected_index_sha256)
