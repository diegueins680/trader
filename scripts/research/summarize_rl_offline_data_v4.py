"""Verify larger frozen buffers, unchanged controls and descriptive parent comparisons."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import numpy as np
from compare_rl_observations_v2 import archive
from rl_composition import buffer_hash
from run_rl_offline_data_v4 import ALGORITHMS, REG, RULES, configuration
from run_rl_composition import require, write
from summarize_rl_composition import export, read, sha


def coverage(data):
    observations = data['s']
    return dict(transitions=len(data['a']), actionCounts=np.bincount(data['a'], minlength=3).tolist(),
        supportedFraction=float(observations[:, 9].mean()),
        uniqueObservations=len(np.unique(observations, axis=0)),
        observationMin=observations.min(axis=0).tolist(), observationMax=observations.max(axis=0).tolist(),
        terminalTransitions=int(data['done'].sum()),
        scope='simulated raw proposals; support flag is causal burn-in feature range, not behavior overlap')


def aggregates(records):
    output=[]
    for algorithm, stress in sorted({(r['algorithm'],r['stress']) for r in records}):
        rows=[r for r in records if (r['algorithm'],r['stress']) == (algorithm,stress)]
        require(all(r['status']=='complete' for r in rows), 'incomplete comparison paths')
        means={str(s):statistics.mean(r['netReturn'] for r in rows if r['seed']==s)
               for s in sorted({r['seed'] for r in rows})}
        output.append(dict(algorithm=algorithm,stress=stress,meanNetReturn=statistics.mean(means.values()),
            seedMeanReturns=means, seedMeanSampleSD=statistics.stdev(means.values()) if len(means)>1 else None,
            **{name:statistics.mean(r[key] for r in rows) for name,key in
               (('meanGross','grossPnl'),('meanFunding','fundingPnl'),('meanCosts','feesSpreadSlippage'),
                ('meanIntervention','interventionRate'),('meanTurnover','turnover'),
                ('meanAbsExposure','meanAbsExposure'),('meanFillCount','fillsIncludingLiquidation'))},
            worstDrawdown=max(r['maxDrawdown'] for r in rows), worstES95=max(r['ES95'] for r in rows),
            worstDrawdownDurationBars=max(r['drawdownDurationBars'] for r in rows),
            bySymbol={s:statistics.mean(r['netReturn'] for r in rows if r['symbol']==s)
                      for s in sorted({r['symbol'] for r in rows})},
            byFold={str(f):statistics.mean(r['netReturn'] for r in rows if r['fold']==f) for f in (0,1)}))
    return output


def summarize(source, parent, output, expected):
    reg=read(REG); configuration(reg)
    current, manifest=archive(source, expected)
    previous, parent_manifest=archive(parent, reg['parentArchive']['indexSha256'])
    require(manifest['campaign']==reg['id'] and manifest['registrationSha256']==sha(REG.read_bytes()) and
            manifest['version']==parent_manifest['version'] and
            parent_manifest['commit']==reg['parentArchive']['frozenCodeCommit'], 'wrong offline data packages')
    require(output.resolve()!=parent.resolve() and parent.resolve() not in output.resolve().parents,
            'output must be outside parent archive')
    buffers=[]
    for fold, digest in enumerate(reg['parentArchive']['behaviorHashes']):
        with np.load(source/f'behavior-fold{fold}.npz', allow_pickle=False) as loaded:
            data=dict(loaded)
        prefix={key:value[:512] for key,value in data.items()}
        metadata=read(source/f'behavior-fold{fold}.json')
        require(buffer_hash(prefix)==digest==read(parent/f'behavior-fold{fold}.json')['sha256'] and
                buffer_hash(data)==metadata['sha256'] and len(data['a'])==4096, 'offline extension identity mismatch')
        buffers.append(dict(fold=fold,sha256=metadata['sha256'],prefixSha256=digest,
                            full=coverage(data),prefix=coverage(prefix),
                            generationRiskFailures=metadata['trainingRiskFailures']))
    prior={r['id']:r for r in previous['records']}
    for row in current['records']:
        if row['algorithm'] in RULES:
            require(row==prior[row['id']], 'deterministic baseline changed')
    require(len(current['fits'])==30 and len(current['records'])==798 and
            all(r['status']=='complete' and r['collectedTransitions']==0 for r in current['fits']),
            'offline trial accounting mismatch')
    export(source, output, expected)
    deltas=[dict(id=r['id'],algorithm=r['algorithm'],seed=r['seed'],
                 difference=r['netReturn']-prior[r['id']]['netReturn'])
            for r in current['records'] if r['algorithm'] in ALGORITHMS and r['stress']=='base']
    counts=Counter()
    for file in (source/'paths').glob('*.json'):
        for row in read(file):
            for key in ('actionChanged','supportChangedAction','turnoverRejected','schedulingDeferred','newProposalIgnored'):
                counts[key]+=row[key]
    result=dict(campaign=reg['id'], indexSha256=expected,
        parentIndexSha256=reg['parentArchive']['indexSha256'], buffers=buffers,
        aggregates=aggregates(current['records']), parentDataSizeBasePathDeltas=deltas,
        frozenSimulatorLearners=dict(sourceIndexSha256=reg['parentArchive']['indexSha256'],
            retrained=False, aggregates=aggregates([r for r in previous['records']
                                                   if r['algorithm'] in ('ppo','double_dqn')])),
        interventions=dict(counts),
        criticMse={a:statistics.mean(r['criticMseFinitePath'] for r in current['records']
                                    if r['algorithm']==a and r['stress']=='base') for a in ('cql','cql_alpha0')},
        interpretation='Data-size deltas descriptive only; no best-budget/seed selection or confirmation claim.',
        promotionAllowed=False, holdoutOpened=False)
    write(output/'offline-data-comparison.json', result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    args=parser.parse_args()
    summarize(args.source,args.parent,args.output,args.expected_index_sha256)
