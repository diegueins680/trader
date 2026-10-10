"""Registered supplemental observation controls. Frozen RL traces; no fitting."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import rl_admission_v2 as v2
import run_rl_composition as driver
from summarize_rl_composition import read, sha, audit_rows

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT/'research-notes/registrations/rl-observation-controls-v2.json'
RULES = ('normalized_momentum3', 'normalized_reversal3')


def observation_rule(name, observation):
    driver.require(name in RULES and type(observation) is np.ndarray and
                   observation.shape == (12,) and observation.dtype == np.dtype('float64') and
                   np.isfinite(observation).all(), 'invalid observation control')
    sign = int(observation[1] > 0) - int(observation[1] < 0)
    return 1 + sign * (1 if name == RULES[0] else -1)


def archive(source, expected):
    source = source.resolve()
    driver.require(sha((source/'index.json').read_bytes()) == expected, 'archive identity')
    inventory = read(source/'index.json')
    for name, digest in inventory.items():
        p = source/name
        driver.require(not Path(name).is_absolute() and '..' not in Path(name).parts and
                       not p.is_symlink() and source in p.resolve().parents, 'archive path')
        driver.require(sha(p.read_bytes()) == digest, 'archive bytes')
    driver.require({str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
                   == set(inventory)|{'index.json'}, 'archive inventory')
    return read(source/'summary.json'), read(source/'manifest.json')


def paired(records):
    result=[]
    for algorithm in ('ppo','double_dqn','cql'):
        for rule in ('cash',RULES[0]):
            chosen=[r for r in records if r['algorithm'] in (algorithm,rule) and r['stress']=='base']
            driver.require(len(chosen)==36, 'missing paired paths')
            if any(r['status']!='complete' for r in chosen):
                result.append(dict(algorithm=algorithm,baseline=rule,status='insufficient_failed_paths'))
                continue
            differences=[]
            for fold in (0,1):
                arrays=[np.mean([r['increments'] for r in chosen if r['fold']==fold and r['algorithm']==a],axis=0)
                        for a in (algorithm,rule)]
                driver.require(all(a.shape==(64,) for a in arrays),'window mismatch')
                differences.append(arrays[0]-arrays[1])
            for block in (3,9,21):
                rng=np.random.default_rng(20261010);draws=np.zeros(2000)
                for difference in differences:
                    starts=rng.integers(64,size=(2000,int(np.ceil(64/block))))
                    idx=((starts[:,:,None]+np.arange(block))%64).reshape(2000,-1)[:,:64]
                    draws+=difference[idx].sum(axis=1)/2
                result.append(dict(algorithm=algorithm,baseline=rule,blockBars=block,
                    meanPairedNetChange=float(np.mean([sum(d) for d in differences])),
                    simultaneous95Interval=np.quantile(draws,[.05/12,1-.05/12]).tolist(),
                    status='adaptive_descriptive_only_not_original_primary'))
    return result


def run(source, panel, settlements, output):
    driver.require(output.resolve()!=source.resolve() and source.resolve() not in output.resolve().parents,
                   'output must be outside frozen archive')
    reg=read(REG)
    parent=read(ROOT/'research-notes/registrations'/reg['parentRegistration'])
    frozen, manifest=archive(source,reg['archiveIndexSha256'])
    driver.require(manifest['version']==v2.VERSION and manifest['commit']==reg['frozenPolicyCommit']
                   and not manifest['holdoutOpened'] and not manifest['promotionAllowed'], 'wrong frozen package')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=['scripts/research/compare_rl_observations_v2.py','scripts/research/rl_admission_v2.py',
           'scripts/research/rl_composition.py','scripts/research/run_rl_composition.py',
           'scripts/research/summarize_rl_composition.py',str(REG.relative_to(ROOT))]
    hashes={}
    for name in names:
        raw=(ROOT/name).read_bytes()
        driver.require(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)==raw,'uncommitted diagnostic')
        hashes[name]=sha(raw)
    # The supplement uses the exact existing environment, not a revised shield.
    for name,digest in manifest['sourceHashes'].items():
        driver.require(sha((ROOT/name).read_bytes())==digest, 'frozen environment source drift')
    bars,events=driver.load(panel,settlements,parent)
    output.mkdir(parents=True,exist_ok=False);(output/'paths').mkdir()
    planned=[f'{f}/{rule}/0/{stress}/{symbol}' for f in (0,1) for rule in RULES
             for stress in driver.STRESSES for symbol in parent['data']['symbols']]
    driver.write(output/'planned.json',planned)
    records=[];audits=[]
    for f,fold in enumerate(parent['folds']):
        for rule in RULES:
            for stress in driver.STRESSES:
                for symbol in parent['data']['symbols']:
                    ep=v2.Episode(bars[symbol][:fold['evaluationStop']],events[symbol][:fold['evaluationStop']],
                        fold['evaluationStart'],fold['evaluationStop'],stress,enabled=True)
                    row=dict(id=f'{f}/{rule}/0/{stress}/{symbol}',fold=f,algorithm=rule,seed=0,stress=stress,symbol=symbol)
                    try:
                        while not ep.done:ep.step(observation_rule(rule,ep.observation()))
                        row.update(driver.metrics(ep))
                        row['increments']=[r['increment'] for r in ep.rows]
                    except (ValueError,ArithmeticError,MemoryError) as exc:
                        row.update(status='failed',failure=str(exc),observations=len(ep.rows))
                    driver.write(output/'paths'/f'{f}-{rule}-0-{stress}-{symbol}.json',ep.rows)
                    audits.append(audit_rows(ep.rows));records.append(row)
    driver.require({r['id'] for r in records}==set(planned) and len(records)==84,'incomplete diagnostic')
    reused=[]
    for row in frozen['records']:
        if row['algorithm'] in ('ppo','double_dqn','cql','cash') and row['stress']=='base':
            name=f"{row['fold']}-{row['algorithm']}-{row['seed']}-base-{row['symbol']}.json"
            rows=read(source/'paths'/name)
            reused.append({**row,'increments':[r['increment'] for r in rows]})
    result=dict(version=reg['id'],commit=commit,sourceHashes=hashes,registrationSha256=sha(REG.read_bytes()),
        parentIndexSha256=reg['archiveIndexSha256'],trainedPolicies=0,planned=len(planned),
        records=[{k:v for k,v in r.items() if k!='increments'} for r in records],
        paired=paired(reused+records),auditRows=sum(a['rows'] for a in audits),
        auditViolations=sum(len(a['violations']) for a in audits),promotionAllowed=False,holdoutOpened=False)
    driver.write(output/'summary.json',result)
    driver.write(output/'index.json',{str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(dict(complete=True,planned=len(planned),failed=sum(r['status']!='complete' for r in records),
                         auditViolations=result['auditViolations'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-registered-observation-controls',action='store_true')
    for name in ('source','panel','settlements','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if not args.run_registered_observation_controls:p.error('disabled: registered controls opt-in required')
    run(args.source,args.panel,args.settlements,args.output)
