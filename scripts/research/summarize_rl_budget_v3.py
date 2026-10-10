"""Hash-verified budget comparison. Never writes inside either frozen archive."""
import argparse
import collections
from pathlib import Path
import statistics
from compare_rl_observations_v2 import archive
from run_rl_budget_v3 import REG, configuration
from run_rl_composition import write
from summarize_rl_composition import export, read, sha


def summarize(source, parent, output, expected):
    reg = read(REG); configuration(reg)
    current, manifest = archive(source, expected)
    previous, parent_manifest = archive(parent, reg['parentArchive']['indexSha256'])
    if (manifest['campaign'] != reg['id'] or manifest['registrationSha256'] != sha(REG.read_bytes()) or
            manifest['version'] != parent_manifest['version'] or
            parent_manifest['commit'] != reg['parentArchive']['frozenCodeCommit']):
        raise ValueError('wrong budget packages')
    for fold, digest in enumerate(reg['parentArchive']['behaviorHashes']):
        if any(read(root/f'behavior-fold{fold}.json')['sha256'] != digest for root in (source,parent)):
            raise ValueError('offline data changed')
    if output.resolve() == parent.resolve() or parent.resolve() in output.resolve().parents:
        raise ValueError('output must be outside parent archive')
    export(source, output, expected)
    groups = collections.defaultdict(list)
    for row in current['records']: groups[row['algorithm'],row['stress']].append(row)
    aggregates=[]
    for (algorithm,stress), rows in sorted(groups.items()):
        if any(r['status'] != 'complete' for r in rows):
            aggregates.append(dict(algorithm=algorithm,stress=stress,status='insufficient_failed_paths'))
            continue
        seeds = sorted({r['seed'] for r in rows})
        means = [statistics.mean(r['netReturn'] for r in rows if r['seed']==seed) for seed in seeds]
        aggregates.append(dict(algorithm=algorithm,stress=stress,status='complete',
            meanNetReturn=statistics.mean(means), seedMeanReturns=dict(zip(map(str,seeds),means)),
            seedMeanSampleSD=statistics.stdev(means) if len(means)>1 else None,
            meanGross=statistics.mean(r['grossPnl'] for r in rows),
            meanFunding=statistics.mean(r['fundingPnl'] for r in rows),
            meanCosts=statistics.mean(r['feesSpreadSlippage'] for r in rows),
            worstDrawdown=max(r['maxDrawdown'] for r in rows), worstES95=max(r['ES95'] for r in rows),
            meanIntervention=statistics.mean(r['interventionRate'] for r in rows),
            meanTurnoverRejection=statistics.mean(r['turnoverRejectionRate'] for r in rows),
            meanTurnover=statistics.mean(r['turnover'] for r in rows),
            bySymbol={s:statistics.mean(r['netReturn'] for r in rows if r['symbol']==s)
                      for s in reg['data']['symbols']},
            byFold={str(f):statistics.mean(r['netReturn'] for r in rows if r['fold']==f) for f in (0,1)}))
    prior = {r['id']:r for r in previous['records']}
    deltas=[]
    for r in current['records']:
        if r['id'] in prior and r['algorithm'] in reg['algorithms'] and r['stress']=='base':
            old=prior[r['id']]
            deltas.append(dict(id=r['id'],algorithm=r['algorithm'],seed=r['seed'],
                difference=r['netReturn']-old['netReturn']
                if r['status']==old['status']=='complete' else None))
    # Include the registered alpha-zero diagnostic, whose name is not an algorithm-description key.
    for r in current['records']:
        if r['algorithm']=='cql_alpha0' and r['stress']=='base':
            old=prior[r['id']]
            deltas.append(dict(id=r['id'],algorithm=r['algorithm'],seed=r['seed'],
                difference=r['netReturn']-old['netReturn']
                if r['status']==old['status']=='complete' else None))
    counts=collections.Counter()
    for file in (source/'paths').glob('*.json'):
        for row in read(file):
            for key in ('actionChanged','supportChangedAction','turnoverRejected','schedulingDeferred','newProposalIgnored'):
                counts[key]+=row[key]
    critic={}
    for algorithm in ('double_dqn','cql','cql_alpha0'):
        values=[r['criticMseFinitePath'] for r in current['records']
                if r['algorithm']==algorithm and r['stress']=='base' and 'criticMseFinitePath' in r]
        critic[algorithm]=statistics.mean(values) if values else None
    result=dict(campaign=reg['id'],indexSha256=expected,
        parentIndexSha256=reg['parentArchive']['indexSha256'],aggregates=aggregates,
        parentBudgetBasePathDeltas=deltas,interventions=dict(counts),
        fitTrainingFailures={r['id']:r.get('trainingFailures') for r in current['fits'] if r.get('trainingFailures')},
        criticMse=critic,
        interpretation='Parent-budget deltas descriptive only; no best-budget/seed selection or confirmation claim.',
        promotionAllowed=False,holdoutOpened=False)
    write(output/'budget-comparison.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    args=parser.parse_args()
    summarize(args.source,args.parent,args.output,args.expected_index_sha256)
