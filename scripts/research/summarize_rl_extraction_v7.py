"""Read-only full-trace audit and small evidence export for frozen extraction."""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import statistics
import numpy as np
from compare_rl_replication import decode, require, same, snapshot
from run_rl_composition import sha, write
from run_rl_extraction_v7 import REG, roster
from summarize_rl_composition import audit_rows
from summarize_rl_offline_data_v4 import aggregates


def summarize(source, output, expected):
    require(source.resolve()!=output.resolve() and source.resolve() not in output.resolve().parents,
            'output must be outside archive')
    data=snapshot(source,expected); reg=decode(REG.read_bytes())
    summary,manifest=(decode(data[n]) for n in ('summary.json','manifest.json'))
    require(manifest['campaign']==reg['id'] and manifest['registrationSha256']==sha(REG.read_bytes()) and
            manifest['promotionAllowed'] is False and manifest['holdoutOpened'] is False and
            summary['trainingSteps']==0, 'campaign identity')
    plan=roster(reg);same(decode(data['planned.json']),plan,'registered roster')
    rows=summary['records']
    require(len(rows)==len(plan)==summary['terminal']==summary['planned']==2688 and
            [r['id'] for r in rows]==[r['id'] for r in plan], 'incomplete roster')
    same([decode(line) for line in data['events.jsonl'].splitlines()],rows,'event registry')
    audit_count=0;violations=[];counts=Counter();diagnostics={};grouped={}
    for r in rows:
        path=decode(data['paths/'+r['id'].replace('/','-')+'.json'])
        checked=audit_rows(path);audit_count+=len(path)
        violations.extend(dict(id=r['id'],**v) for v in checked['violations'])
        if r['status']=='complete':
            require(len(path)==r['observations']==64 and path[-1]['terminal'],'complete trace truncated')
        if path and 'incrementSum' in r:
            same(float(np.sum([p['increment'] for p in path])),r['incrementSum'],'increment sum')
            same(path[-1]['equity']-1,r['netReturn'],'endpoint return')
        sampled=r['algorithm'].endswith('_sampled')
        rng=np.random.default_rng(np.random.SeedSequence([r['actionSeed'],r['fold'],r['seed'],
                                                         reg['data']['symbols'].index(r['symbol'])]))
        for p in path:
            for key in ('actionChanged','supportChangedAction','turnoverRejected','schedulingDeferred','newProposalIgnored'):
                counts[key]+=p[key]
            if sampled:
                probs=np.array(p['proposalProbabilities'],dtype=float); u=float(rng.random())
                require(probs.shape==(3,) and np.isfinite(probs).all() and (probs>=0).all() and
                        abs(probs.sum()-1)<1e-12,'invalid probability trace')
                same(u,p['uniform'],'registered uniform stream')
                # Separate scalar interval implementation from runner searchsorted.
                a=next((i for i in range(3) if u < math.fsum(probs[:i+1])),2)
                require(a==p['actionIndex'] and probs[a]>0 and p['proposal']==(-.25,0.,.25)[a],
                        'inverse CDF trace mismatch')
                same(math.log(probs[a]),p['proposalLogProbability'],'proposal log probability')
                d=diagnostics.setdefault(r['algorithm'],dict(rows=0,entropySum=0.,maxMassSum=0.,greedyActions=0,actionCounts=[0,0,0]))
                d['rows']+=1;d['entropySum']-=sum(q*math.log(q) for q in probs if q>0)
                d['maxMassSum']+=float(max(probs));d['greedyActions']+=a==int(np.argmax(probs));d['actionCounts'][a]+=1
        grouped.setdefault((r['algorithm'],r['stress']),[]).append(r)
    require(not violations,'independent economic audit failed: '+str(violations[:3]))
    groups=[]
    for (algorithm,stress),values in sorted(grouped.items()):
        if all(v['status']=='complete' for v in values):
            aggregate=aggregates(values)[0]
            aggregate['actionSeedMeanReturns']={str(s):statistics.mean(v['netReturn'] for v in values if v['actionSeed']==s)
                                              for s in sorted({v['actionSeed'] for v in values})}
            aggregate['trainingByActionSeedMeanReturns']={f'{s}/{a}':statistics.mean(v['netReturn'] for v in values if v['seed']==s and v['actionSeed']==a)
                for s,a in sorted({(v['seed'],v['actionSeed']) for v in values})}
            groups.append(aggregate)
        else:
            groups.append(dict(algorithm=algorithm,stress=stress,status='insufficient_evidence_failed_paths',
                failures=[v['id'] for v in values if v['status']!='complete'],
                realizedMeanStoppedReturn=statistics.mean(v['netReturn'] for v in values) if all('netReturn'in v for v in values) else None))
    for d in diagnostics.values():
        d['meanEntropyNats']=d.pop('entropySum')/d['rows'];d['meanMaxProbability']=d.pop('maxMassSum')/d['rows']
        d['greedyActionFraction']=d['greedyActions']/d['rows']
    output.mkdir(parents=True,exist_ok=False)
    result=dict(campaign=reg['id'],archiveIndexSha256=expected,manifest=manifest,
        planned=summary['planned'],terminal=summary['terminal'],trainingSteps=0,
        independentAudit=dict(rows=audit_count,detectedViolations=len(violations),
            scope='Archived binary64 scalar checks at1e-12; not universal safety proof. Terminal fill/liquidation reconstruction remains partial.'),
        runnerAudit=summary['audit'],aggregates=groups,paired=summary['paired'],sampling=diagnostics,
        interventions=dict(counts),runtime={k:summary[k] for k in ('elapsedSeconds','maxRss','rssUnit','inference')},
        promotionAllowed=False,holdoutOpened=False,DSR='insufficient_evidence',PBO='insufficient_evidence',
        realOPE='unsupported',newIndependentMarketHistories=0,
        failureCounts=dict(Counter(v['failure'] for v in rows if v['status']!='complete')))
    write(output/'evidence.json',result)
    fields=['id','fold','algorithm','seed','actionSeed','stress','symbol','status','failure','observations',
            'netReturn','grossPnl','fundingPnl','feesSpreadSlippage','maxDrawdown','drawdownDurationBars',
            'ES95','turnover','fillsIncludingLiquidation','meanAbsExposure','interventionRate',
            'unsupportedRate','turnoverRejectionRate','supportActionChangeRate','schedulingDeferredRate',
            'newProposalIgnoredRate','incrementSum']
    with (output/'all-paths.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(output/'trial-registry.json',[{k:r.get(k) for k in ('id','status','failure')} for r in rows])
    print(dict(rows=audit_count,paths=len(rows),failures=result['failureCounts']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args();summarize(args.source,args.output,args.sha256)
