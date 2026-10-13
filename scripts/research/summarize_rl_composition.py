"""Export small, hash-verified pilot evidence without changing archived trials."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out: raise ValueError('duplicate JSON key')
            out[key]=value
        return out
    def invalid(value): raise ValueError('invalid JSON number '+value)
    def finite(value):
        result=float(value)
        if not math.isfinite(result): raise ValueError('overflowing JSON number')
        return result
    return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=invalid,parse_float=finite)


def audit_rows(rows):
    """Independent binary64 trace checks, not an executable-intent safety proof."""
    violations=[]; changed=0; previous=None
    def check(ok, index, reason):
        if not ok: violations.append({'row':index,'reason':reason})
    for i,row in enumerate(rows):
        check(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()),i,'nonfinite')
        check(row['proposal'] in (-.25,0,.25) and row['permitted'] in (-.25,0,.25),i,'action_domain')
        check(row['applied'] in (None,-.25,0,.25) and row['submitted'] in (None,-.25,0,.25),i,'intent_domain')
        check(row['supported'] or row['permitted']==0,i,'unsupported_target')
        check(row['decisionTime']<row['fillClose'],i,'decision_before_fill')
        if previous is not None:
            check(not previous['terminal'],i,'transition_after_terminal')
            check(abs(previous['equity']-row['equityBefore'])<=1e-12,i,'equity_continuity')
        cost=row['fee']+row['spread']+row['slippage']
        check(abs(row['equity']-row['equityBefore']-row['gross']-row['funding']+cost)<=1e-12,i,'ledger_identity')
        check(abs(row['reward']/100-row['increment'])<=1e-12,i,'reward_identity')
        check(abs(row['increment']-row['equity']+row['equityBefore'])<=1e-12,i,'increment_identity')
        check(cost>=0 and row['turnover']>=0,i,'negative_cost')
        if not row['terminal']:
            prefill=row['equityBefore']+row['gross']+row['funding']
            check(row['turnover']<=.5*prefill+1e-12,i,'nonterminal_turnover')
            check(row['exposure'] is not None and abs(row['exposure'])<=.35+1e-12,i,'nonterminal_exposure')
        if row['liquidation']=='flat': check(row['exposure']==0,i,'terminal_flat')
        if 'finalTarget' in row:
            # V2 adds enough primitive ledger fields for independent price,
            # units, funding and nonterminal fill reconstruction.
            near=lambda a,b: math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
            units,price=row['unitsBefore'],row['priceAfter']
            prefill=row['equityBefore']+row['gross']+row['funding']
            check(price>0 and row['priceBefore']>0,i,'positive_price')
            check(near(row['gross'],units*(price-row['priceBefore'])),i,'unit_price_gross')
            check(near(row['funding'],-units*row['fundingPerUnit']),i,'unit_funding')
            check(near(row['filledUnits'],row['unitsAfter']-units),i,'unit_delta')
            if previous is not None:
                check(near(previous['unitsAfter'],units) and near(previous['priceAfter'],row['priceBefore']),i,'unit_price_continuity')
            for key,rate in (('fee',.0005),('spread',.00005),('slippage',.00045)):
                check(near(row[key],row['turnover']*row['costMultiplier']*rate),i,'linear_'+key)
            check(row['finalTarget']==row['applied'],i,'final_target_identity')
            check(row['actionChanged']==((row['proposal']!=row['permitted']) or row['turnoverRejected'])
                  and row['intervention']==row['actionChanged'],i,'intervention_definition')
            if row['turnoverRejected']:
                due=row['dueTarget']
                check(due in (-.25,.25) and row['finalTarget']==0,i,'rejection_projection')
                if due is not None:
                    check(abs(due*prefill-units*price)*row['fillFraction']>.5*prefill-1e-12,i,'rejected_turnover_witness')
            if not row['terminal']:
                target=row['finalTarget']
                expected=units if target is None else units+row['fillFraction']*(target*prefill/price-units)
                check(near(expected,row['unitsAfter']),i,'fill_reconstruction')
                check(near(abs(expected-units)*price,row['turnover']),i,'turnover_reconstruction')
            if row['liquidation']=='flat':check(row['unitsAfter']==0,i,'terminal_units_flat')
        changed+=row['applied'] is None or row['applied']!=row['proposal']
        previous=row
    return {'rows':len(rows),'violations':violations,'changedOrDeferredProposals':changed}


def export(source,output,expected):
    source=source.resolve(); output=output.resolve()
    if output==source or source in output.parents: raise ValueError('output must be outside archive')
    index=source/'index.json'
    if sha(index.read_bytes())!=expected: raise ValueError('index identity mismatch')
    inventory=read(index)
    for name,digest in inventory.items():
        path=source/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or source not in path.resolve().parents:
            raise ValueError('unsafe archive path')
        if sha(path.read_bytes())!=digest: raise ValueError('archive hash mismatch: '+name)
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual!=set(inventory)|{'index.json'}: raise ValueError('archive inventory mismatch')
    summary=read(source/'summary.json');manifest=read(source/'manifest.json');planned=read(source/'planned.json')
    records=summary['fits']+summary['records']
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(planned) or summary['planned']!=len(ids) or summary['terminal']!=len(ids):
        raise ValueError('trial reconciliation failed')
    if summary['promotionAllowed'] or summary['holdoutOpened']: raise ValueError('wrong research mode')
    audits={name:audit_rows(read(source/name)) for name in inventory if name.startswith('paths/')}
    v2=manifest['version']=='rl-admission-continuation-v2'
    audit={'scope':'Post-hoc binary64 archived trace checks, tolerance 1e-12. Not a production safety proof. '+('V2 independently reconstructs units/prices/funding/linear costs and nonterminal fills.' if v2 else 'V1 lacks independent unit/price reconstruction.'),
           'paths':len(audits),'rows':sum(a['rows'] for a in audits.values()),
           'violationCount':sum(len(a['violations']) for a in audits.values()),
           'violations':{n:a['violations'] for n,a in audits.items() if a['violations']},
           'changedOrDeferredProposals':sum(a['changedOrDeferredProposals'] for a in audits.values()),
           'legacyCounter':'Archived unsafeExecutableIntents=0 was a constant and is NOT evidence. This independent audit supersedes that reporting claim.'}
    output.mkdir(parents=True,exist_ok=False)
    fields=['id','fold','algorithm','seed','stress','symbol','status','failure','observations','netReturn',
            'grossPnl','fundingPnl','feesSpreadSlippage','maxDrawdown','drawdownDurationBars','ES95',
            'turnover','fillsIncludingLiquidation','meanAbsExposure','interventionRate','unsupportedRate',
            'turnoverRejectionRate','supportActionChangeRate','schedulingDeferredRate','newProposalIgnoredRate',
            'proposalCounts','holdingPeriodsBars','trainingFailures','regime','steps','frozenDatasetSha256',
            'criticMseFinitePath','seconds','updates','collectedTransitions','artifactSha256']
    with (output/'trial-registry.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    groups=collections.defaultdict(list)
    for row in summary['records']:groups[row['algorithm'],row['seed'],row['stress']].append(row)
    aggregate=[]
    for (algorithm,seed,stress),rows in sorted(groups.items()):
        values=[r for r in rows if 'netReturn' in r]
        aggregate.append({'algorithm':algorithm,'seed':seed,'stress':stress,'paths':len(rows),
            'complete':sum(r['status']=='complete' for r in rows),'failed':sum(r['status']=='failed' for r in rows),
            'meanTerminalOrStoppedReturn':statistics.mean(r['netReturn'] for r in values) if values else None,
            'worstTerminalOrStoppedReturn':min((r['netReturn'] for r in values),default=None),
            'worstDrawdown':max((r['maxDrawdown'] for r in values),default=None),
            'worstES95':max((r['ES95'] for r in values),default=None),
            'meanInterventionRate':statistics.mean(r['interventionRate'] for r in values) if values else None,
            'meanFees':statistics.mean(r['feesSpreadSlippage'] for r in values) if values else None,
            'comparison':'incomplete paths are NOT equal-horizon economic comparisons'})
    with (output/'all-seed-results.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    receipt={'source':str(source),'indexSha256':expected,'manifest':manifest,'planned':len(planned),
        'fitCount':len(summary['fits']),'replayCount':len(summary['records']),
        'failedFits':sum(r['status']=='failed' for r in summary['fits']),
        'failedReplays':sum(r['status']=='failed' for r in summary['records']),
        'failureReasons':dict(collections.Counter(r.get('failure') for r in summary['records'] if r['status']=='failed')),
        'trainingSeconds':sum(r['seconds'] for r in summary['fits']),
        'elapsedSeconds':summary['elapsedSeconds'],'maxRss':summary['maxRss'],'rssUnit':summary['rssUnit'],
        'inferenceP99MaxMs':max((r['inference']['p99Ms'] for r in summary['fits'] if r['status']=='complete'),default=None),
        'artifactBytesMax':max((r['artifactBytes'] for r in summary['fits'] if r['status']=='complete'),default=None),
        'paired':summary['paired'],'DSR':summary['DSR'],'PBO':summary['PBO'],'OPE':summary['OPE'],
        'holdoutOpened':False,'promotionAllowed':False,'blockers':summary['blockers'],
        'parameterArtifactsInGit':False,'traceAudit':audit,
        'artifactProvenanceClarification':'The archived trainingDatasetSha256 field names the common simulated reference buffer. Only CQL/alpha0/BC train on that buffer; PPO and DDQN collect fresh simulator transitions. The manifest panel/settlement hashes, fold and source commit identify their market training inputs. These research artifacts have no production loader.'}
    (output/'evidence.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('fitCount','replayCount','failedFits','failedReplays','trainingSeconds','elapsedSeconds','inferenceP99MaxMs')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--expected-index-sha256',required=True)
    a=p.parse_args();export(a.source,a.output,a.expected_index_sha256)
