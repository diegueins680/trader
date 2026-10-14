"""Verify the reward ablation, including economic and shaped training traces."""
import argparse
from collections import Counter
from itertools import groupby
import math
from pathlib import Path
import numpy as np
from compare_rl_observations_v2 import archive
from compare_rl_replication import decode, same, snapshot
from rl_composition import buffer_hash
from run_rl_composition import require, write
from run_rl_penalty_v6 import REG, RULES, configuration
from summarize_rl_composition import audit_rows, export, read, sha
from summarize_rl_horizon_v5 import reference
from summarize_rl_offline_data_v4 import aggregates


def training_audit(rows, expected, cutoff):
    require(len(rows) == expected, 'training transition count')
    groups = [(key,list(values)) for key,values in groupby(rows,key=lambda r:r['trainingEpisode'])]
    require([key for key,_ in groups] == list(range(1,len(groups)+1)), 'training episode order')
    violations=[]
    for number, episode in groups:
        first=episode[0]; start,stop=first['trainingStart'],first['trainingStop']
        require(type(start) is int and type(stop) is int and 96 <= start and
                stop-start == 65 and stop <= cutoff, 'training maturity/horizon')
        require(len(episode) <= 64 and
                all(row['index'] == start+i+1 and row['index'] < cutoff and
                    row['trainingStart'] == start and row['trainingStop'] == stop and
                    row['trainingSymbol'] == first['trainingSymbol'] for i,row in enumerate(episode)),
                'training episode identity')
        require(number == len(groups) or episode[-1]['terminal'], 'free training reset')
        violations.extend(dict(episode=number,**v) for v in audit_rows(episode)['violations'])
        for row in episode:
            require(type(row['actionChanged']) is bool, 'invalid training intervention')
            penalty = .02 if row['actionChanged'] else 0.
            require(math.isclose(row['trainingPenalty'],penalty,abs_tol=1e-12,rel_tol=0) and
                    math.isclose(row['trainingReward'],row['reward']-penalty,abs_tol=1e-12,rel_tol=1e-12),
                    'shaped/economic reward mismatch')
    require(not violations, 'training economic trace violation: '+str(violations[:3]))
    changed=sum(row['actionChanged'] for row in rows)
    return dict(transitions=len(rows),episodes=len(groups),changedProposals=changed,
        economicRewardSum=math.fsum(row['reward'] for row in rows),
        trainingRewardSum=math.fsum(row['trainingReward'] for row in rows),
        penaltySum=.02*changed,finalEpisodeTerminal=rows[-1]['terminal'])


def summarize(source,parent,output,expected):
    reg=read(REG);configuration(reg)
    data=snapshot(source,expected)
    current,manifest=archive(source,expected)
    previous,parent_manifest=archive(parent,reg['parentArchive']['indexSha256'])
    require(manifest['campaign']==reg['id'] and manifest['registrationSha256']==sha(REG.read_bytes()) and
            parent_manifest['campaign']=='rl-matched-horizon-v5' and
            manifest['algorithms']==['ppo'] and manifest['trainingEpisodeTransitions']==64,
            'wrong reward packages')
    require(output.resolve()!=parent.resolve() and parent.resolve() not in output.resolve().parents,
            'output must be outside parent archive')
    planned={f'{fold}/ppo/{seed}/training' for fold in (0,1) for seed in reg['seeds']}
    planned|={f'{fold}/{a}/{seed}/{stress}/{symbol}' for fold in (0,1)
        for a in ('ppo',*RULES) for seed in (reg['seeds'] if a=='ppo' else [0])
        for stress in reg['stresses'] for symbol in reg['data']['symbols']}
    rows=current['fits']+current['records']
    require(len(rows)==len(planned)==388 and {r['id'] for r in rows}==planned,
            'reward trial roster')
    same(sorted((decode(line) for line in data['events.jsonl'].splitlines()),key=lambda r:r['id']),
         sorted(rows,key=lambda r:r['id']),'terminal events')
    prior=reference(current,previous)
    for row in current['records']:
        name='paths/'+row['id'].replace('/','-')+'.json'
        if name not in data:
            require(row['status']=='failed' and row.get('failure','').startswith('training_failed:'),
                    'missing evaluation trace')
            continue
        path=decode(data[name])
        if row['algorithm'] in RULES:
            require(data[name]==(parent/name).read_bytes(),'baseline trace changed')
        if row['status']=='complete':
            require(len(path)==row['observations']==64 and path[-1]['terminal'],'incomplete evaluation')
            same(sum(r['increment'] for r in path),row['incrementSum'],'evaluation sum')
    for fold,digest in enumerate(reg['parentArchive']['behaviorHashes']):
        with np.load(source/f'behavior-fold{fold}.npz',allow_pickle=False) as loaded:
            require(buffer_hash(dict(loaded))==digest,'reference buffer changed')
    audits=[]
    for fit in current['fits']:
        if fit['status']!='complete':continue
        fold,seed=fit['fold'],fit['seed']
        raw=data[f'training-{fold}-ppo-{seed}.json']
        require(sha(raw)==fit['trainingRewardTraceSha256'],'training trace identity')
        trace=decode(raw)
        require(all(r['trainingSymbol'] in reg['data']['symbols'] for r in trace),'training universe')
        audit=training_audit(trace,4096,reg['folds'][fold]['trainStop'])
        same(audit,fit['trainingRewardAudit'],'training aggregates')
        require(fit['collectedTransitions']==4096 and fit['updates']==128,'PPO update count')
        raw=data[f'policies/{fold}-ppo-{seed}.json'];artifact=decode(raw)
        require(sha(raw)==fit['artifactSha256'] and artifact['mode']=='research_only' and
                artifact['trainingRewardVersion']==reg['environment']['trainingRewardVersion'] and
                artifact['trainingPenaltyPerChangedProposal']==.02 and
                artifact['sourceCommit']==manifest['commit'] and
                artifact['registrationSha256']==manifest['registrationSha256'],'reward artifact identity')
        audits.append(dict(id=fit['id'],**audit))
    export(source,output,expected)
    complete=all(r['status']=='complete' for r in rows)
    deltas=[dict(id=r['id'],seed=r['seed'],
                netReturnDifference=r['netReturn']-prior[r['id']]['netReturn'],
                interventionDifference=r['interventionRate']-prior[r['id']]['interventionRate'])
            for r in current['records'] if r['algorithm']=='ppo' and r['stress']=='base'] if complete else []
    counts=Counter()
    for name,raw in data.items():
        if name.startswith('paths/'):
            for row in decode(raw):
                for key in ('actionChanged','supportChangedAction','turnoverRejected','schedulingDeferred','newProposalIgnored'):
                    counts[key]+=row[key]
    write(output/'penalty-comparison.json',dict(campaign=reg['id'],indexSha256=expected,
        parentIndexSha256=reg['parentArchive']['indexSha256'],
        aggregates=aggregates(current['records']) if complete else [],
        frozenReferences={'retrained':False,'aggregates':aggregates(previous['records'])},
        unshapedPpoBasePathDeltas=deltas,trainingAudits=audits,
        trainingEconomicAuditViolations=0,trainingAuditScope='Selected archived transitions; binary64 tolerance1e-12. Not universal safety proof.',
        interventions=dict(counts),deterministicControlsMatched=168,
        interpretation='All financial metrics use the unshaped ledger. Reward/intervention deltas are adaptive development evidence.',
        promotionAllowed=False,holdoutOpened=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    args=parser.parse_args()
    summarize(args.source,args.parent,args.output,args.expected_index_sha256)
