"""Default-disabled frozen PPO decoder diagnostic; no training or live clients."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
from compare_rl_replication import decode, require, same, snapshot
from optimizer_snapshot_v2 import Snapshot
from rl_composition import Fit, forward, softmax
from rl_admission_v2 import Episode, VERSION, OBSERVATION
from run_rl_budget_v3 import baseline
from run_rl_composition import ROOT, SOURCES, load, metrics, paired, sha, write
from summarize_rl_composition import audit_rows

REG = ROOT / 'research-notes/registrations/rl-policy-extraction-v7.json'
EXTRA = ('rl_admission_v2.py', 'run_rl_budget_v3.py', 'compare_rl_replication.py',
         'summarize_rl_composition.py', 'run_rl_extraction_v7.py')


def categorical(probabilities, uniform):
    """Half-open inverse CDF; reject invalid input before any environment step."""
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.shape == (3,) and np.isfinite(p).all() and np.all(p >= 0) and
            abs(float(p.sum()) - 1) <= 1e-12, 'invalid proposal probabilities')
    require(type(uniform) is float and math.isfinite(uniform) and 0 <= uniform < 1,
            'invalid uniform')
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.  # Closed numerical normalization; uniform is strictly below 1.
    action = int(np.searchsorted(cdf, uniform, side='right'))
    require(0 <= action < 3 and p[action] > 0, 'invalid sampled action')
    return action, p


def admit(data, package, fold, seed):
    """Identity is anchored to the registered archive, not arbitrary JSON claims."""
    manifest, summary = (decode(data[n]) for n in ('manifest.json', 'summary.json'))
    name = f'policies/{fold}-ppo-{seed}.json'
    raw = data[name]; value = decode(raw)
    require(manifest['campaign'] == {'v5': 'rl-matched-horizon-v5',
            'v6': 'rl-proposal-penalty-v6'}[package] and
            manifest['promotionAllowed'] is False and manifest['holdoutOpened'] is False,
            'wrong parent campaign')
    fits = [f for f in summary['fits'] if f['id'] == f'{fold}/ppo/{seed}/training']
    require(len(fits) == 1 and fits[0]['status'] == 'complete' and
            fits[0]['artifactSha256'] == sha(raw), 'unbound or failed policy')
    require(value['version'] == VERSION and value['observation'] == OBSERVATION and
            value['mode'] == 'research_only' and value['algorithm'] == 'ppo' and
            type(value['trainingEpisodeTransitions']) is int and value['trainingEpisodeTransitions'] == 64 and
            type(value['fold']) is int and value['fold'] == fold and
            type(value['seed']) is int and value['seed'] == seed and
            type(value['updates']) is int and value['updates'] == 128 and
            value['sourceCommit'] == manifest['commit'] and
            value['registrationSha256'] == manifest['registrationSha256'] and
            value['assuranceLedgerSha256'] == manifest['proofLedgerSha256'], 'incompatible policy')
    if package == 'v6':
        require(value['trainingRewardVersion'] == 'equity-minus-proposal-intervention-v6' and
                type(value['trainingPenaltyPerChangedProposal']) is float and
                value['trainingPenaltyPerChangedProposal'] == .02, 'wrong penalty reward')
    else:
        require('trainingRewardVersion' not in value and 'trainingPenaltyPerChangedProposal' not in value,
                'unexpected unshaped reward')
    groups = value['parametersLittleEndianHex']
    require(type(groups) is list and len(groups) == 4, 'invalid parameter groups')
    buffers = []
    for text, width in zip(groups, (192, 16, 48, 3)):
        require(type(text) is str and len(text) == width * 16, 'invalid parameter size')
        raw = bytes.fromhex(text)
        require(np.isfinite(np.frombuffer(raw, dtype='<f8')).all(), 'nonfinite parameter')
        buffers.append(raw)
    zeros = tuple(bytes(len(b)) for b in buffers)
    actor = Snapshot('optimizer-snapshot-v2', 3, 128, tuple(buffers), zeros, zeros)
    return Fit('ppo', seed, actor, {}), sha(data[name])


def roster(reg):
    rows = []
    for fold in range(2):
        for package in ('v5', 'v6'):
            for mode in ('greedy', 'sampled'):
                for seed in reg['trainingSeeds']:
                    for action_seed in reg['actionSeeds'] if mode == 'sampled' else [0]:
                        for stress in reg['stresses']:
                            for symbol in reg['data']['symbols']:
                                rows.append(dict(fold=fold, algorithm=f'{package}_{mode}', seed=seed,
                                                 actionSeed=action_seed, stress=stress, symbol=symbol))
        for rule in reg['rules']:
            for stress in reg['stresses']:
                for symbol in reg['data']['symbols']:
                    rows.append(dict(fold=fold, algorithm=rule, seed=0, actionSeed=0, stress=stress, symbol=symbol))
    for row in rows:
        row['id'] = '/'.join(str(row[k]) for k in ('fold','algorithm','seed','actionSeed','stress','symbol'))
    require(len(rows) == len({r['id'] for r in rows}) == 2688, 'invalid roster')
    return rows


def checked_path(rows, sampled):
    audit = audit_rows(rows)
    require(not audit['violations'], 'economic trace violation: ' + str(audit['violations'][:3]))
    if sampled:
        for r in rows:
            action, probabilities = categorical(r['proposalProbabilities'], r['uniform'])
            require(action == r['actionIndex'] and r['proposal'] == (-.25, 0., .25)[action] and
                    math.isclose(r['proposalLogProbability'], math.log(probabilities[action]),
                                 rel_tol=1e-12, abs_tol=1e-12), 'sampling trace violation')
    return audit


def continuation(root, digest, reg, identities):
    data=snapshot(root,digest)
    manifest=decode(data['manifest.json'])
    require(manifest['campaign']==reg['id'] and
            manifest['registrationSha256']==sha(REG.read_bytes()) and
            manifest['commit']=='9bd904a66e82c5829ddfb7f463e1ece8939a7955' and
            manifest['promotionAllowed'] is False and manifest['holdoutOpened'] is False and
            manifest['policyHashes']==identities and 'continuation' not in manifest,
            'continuation identity or phase mismatch')
    plan=roster(reg);same(decode(data['planned.json']),plan,'continuation roster')
    stop=decode(data['budget-stop.json'])
    require(stop['reason']=='registered_wall_budget' and stop['exitCode']==1 and
            stop['elapsedSecondsLowerBound']>=1200 and 'summary.json' not in data, 'unregistered continuation stop')
    rows=[decode(line) for line in data['events.jsonl'].splitlines()]
    require(0<len(rows)<len(plan) and [r['id'] for r in rows]==[r['id'] for r in plan[:len(rows)]],
            'continuation must be a strict registered prefix')
    require(set(data)=={'manifest.json','planned.json','events.jsonl','budget-stop.json'} |
            {'paths/'+r['id'].replace('/','-')+'.json' for r in rows}, 'orphan or unexpected prefix file')
    restored=[]
    for row in rows:
        path=decode(data['paths/'+row['id'].replace('/','-')+'.json'])
        require(row['status'] in ('complete','failed') and len(path)==row['observations'],
                'continuation event/path mismatch')
        checked_path(path,row['algorithm'].endswith('_sampled'))
        if row['status']=='complete':
            require(len(path)==64 and path[-1]['terminal'],'incomplete successful parent path')
        restored.append({**row,'increments':[p['increment'] for p in path]})
    return data,manifest,restored


def run(panel, settlements, output, resume=None, resume_sha=None):
    began = time.perf_counter(); raw_reg = REG.read_bytes(); reg = decode(raw_reg)
    require(platform.python_version() == '3.13.3' and np.__version__ == '2.3.5', 'unregistered runtime')
    require(reg['id'] == 'rl-policy-extraction-v7' and reg['promotionAllowed'] is False and
            reg['enabledByDefault'] is False and reg['liveAuthorization'] is False and
            reg['trainingSeeds'] == [11,23,47,71,101] and reg['actionSeeds'] == [19,37,59,83,109],
            'registration drift')
    output = output.resolve()
    require(ROOT / '.tmp' in output.parents, 'output must be in isolated repository .tmp')
    sources = {f'scripts/research/{n}': (ROOT/'scripts/research'/n).read_bytes() for n in (*SOURCES,*EXTRA)}
    sources[str(REG.relative_to(ROOT))] = raw_reg
    amendment=ROOT/'research-notes/registrations/rl-policy-extraction-v7-continuation.json'
    sources[str(amendment.relative_to(ROOT))]=amendment.read_bytes()
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for name, raw in sources.items():
        require(subprocess.check_output(['git','show',f'{commit}:{name}'], cwd=ROOT) == raw,
                'uncommitted experiment source')
    archives, policies, identities = {}, {}, {}
    for package, reference in reg['packages'].items():
        root = (ROOT/reference['path']).resolve()
        require(root != output and root not in output.parents, 'output overlaps parent')
        archives[package] = snapshot(root, reference['indexSha256'])
        parent = decode(archives[package]['manifest.json'])
        for name in ('rl_composition.py','rl_admission_v2.py','causal_replay_v3.py',
                     'replay_accounting_v2.py','optimizer_snapshot_v2.py','sequential_learning.py',
                     'run_rl_budget_v3.py'):
            key = 'scripts/research/' + name
            require(parent['sourceHashes'][key] == sha(sources[key]), 'parent semantics changed')
        for fold in range(2):
            for seed in reg['trainingSeeds']:
                fit, digest = admit(archives[package], package, fold, seed)
                policies[package, fold, seed] = fit
                identities[f'{package}/{fold}/{seed}'] = digest
    bars, funding = load(panel, settlements, reg)
    plan = roster(reg)
    previous,previous_manifest,restored = continuation(resume,resume_sha,reg,identities) if resume else ({},None,[])
    require(resume is None or (output!=resume.resolve() and resume.resolve() not in output.parents), 'output overlaps partial archive')
    output.mkdir(parents=True, exist_ok=False); (output/'paths').mkdir()
    write(output/'planned.json', plan)
    write(output/'manifest.json', dict(campaign=reg['id'], commit=commit,
        registrationSha256=sha(raw_reg), sourceHashes={k:sha(v) for k,v in sources.items()},
        policyHashes=identities, parentArchives=reg['packages'], python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), promotionAllowed=False, holdoutOpened=False,
        proofLedgerSha256=sha((ROOT/'formal/research/proof-ledger.json').read_bytes()),
        **(dict(continuation=dict(parentIndexSha256=resume_sha,parentManifest=previous_manifest,
                                 reusedPaths=len(restored),amendmentSha256=sha(amendment.read_bytes()))) if resume else {})))
    records=[]; audits=0; economic_rows=0; parity=0; timing=[]
    with (output/'events.jsonl').open('x') as events:
        for number, spec in enumerate(plan):
            if number<len(restored):
                row=restored[number]; path_name='paths/'+row['id'].replace('/','-')+'.json'
                (output/path_name).write_bytes(previous[path_name])
                records.append(row); audits+=1;economic_rows+=len(row['increments'])
                if not row['algorithm'].endswith('_sampled'):
                    learned=row['algorithm'].endswith('_greedy');pkg=row['algorithm'].split('_')[0] if learned else 'v5'
                    old_alg='ppo' if learned else row['algorithm']
                    old_name=f"paths/{row['fold']}-{old_alg}-{row['seed']}-{row['stress']}-{row['symbol']}.json"
                    same(decode(previous[path_name]),decode(archives[pkg][old_name]),'continued frozen parity')
                    parity+=1
                events.write(json.dumps({k:v for k,v in row.items() if k!='increments'},sort_keys=True,allow_nan=False)+'\n');events.flush()
                continue
            row=dict(spec); fold=reg['folds'][row['fold']]; symbol=row['symbol']
            sampled=row['algorithm'].endswith('_sampled')
            ep=Episode(bars[symbol][:fold['evaluationStop']],funding[symbol][:fold['evaluationStop']],
                       fold['evaluationStart'],fold['evaluationStop'],row['stress'],enabled=True)
            rng=np.random.default_rng(np.random.SeedSequence([row['actionSeed'],row['fold'],row['seed'],
                                                             reg['data']['symbols'].index(symbol)]))
            package=row['algorithm'].split('_')[0]
            fit=policies[package,row['fold'],row['seed']] if package in ('v5','v6') else None
            try:
                while not ep.done:
                    info={}
                    if fit is None:
                        action=baseline(row['algorithm'],ep)
                    else:
                        start=time.perf_counter_ns(); logits=forward(fit.actor,ep.observation()[None,:])[0]
                        if sampled:
                            uniform=float(rng.random()); action,p=categorical(softmax(logits[None,:])[0],uniform)
                            info=dict(proposalProbabilities=p.tolist(),uniform=uniform,actionIndex=action,
                                      proposalLogProbability=math.log(p[action]))
                        else: action=int(np.argmax(logits))
                        if sampled and len(timing)<10000: timing.append((time.perf_counter_ns()-start)/1e6)
                    ep.step(action); ep.rows[-1].update(info)
                row.update(metrics(ep)); row['increments']=[r['increment'] for r in ep.rows]
                checked_path(ep.rows,sampled); audits+=1; economic_rows+=len(ep.rows)
                if not sampled:
                    old_alg='ppo' if fit else row['algorithm']
                    path=f"paths/{row['fold']}-{old_alg}-{row['seed']}-{row['stress']}-{symbol}.json"
                    same(ep.rows, decode(archives[package if fit else 'v5'][path]), 'frozen greedy/control trace')
                    parity+=1
            except (ValueError,ArithmeticError,MemoryError) as exc:
                row.update(status='failed',failure=str(exc),observations=len(ep.rows))
            with (output/'paths'/(row['id'].replace('/','-')+'.json')).open('x') as handle:
                json.dump(ep.rows,handle,separators=(',',':'),allow_nan=False)
            records.append(row)
            events.write(json.dumps({k:v for k,v in row.items() if k!='increments'},sort_keys=True,allow_nan=False)+'\n')
            events.flush()
            if (number+1)%84==0: print(json.dumps(dict(terminal=number+1,planned=len(plan))),flush=True)
            require(time.perf_counter()-began < reg['budget']['campaignWallMinutes']*60,'campaign wall budget exhausted')
    contrasts=[]
    for package in ('v5','v6'):
        contrasts.extend(paired(records,(f'{package}_greedy','cash','normalized_momentum3'),(f'{package}_sampled',)))
    result=dict(campaign=reg['id'],planned=len(plan),terminal=len(records),trainingSteps=0,
                records=[{k:v for k,v in r.items() if k!='increments'} for r in records],paired=contrasts,
                audit=dict(paths=audits,rows=economic_rows,parityPaths=parity,
                           failedPaths=sum(r['status']!='complete' for r in records)),
                inference=dict(samples=len(timing),p99Ms=float(np.quantile(timing,.99)),
                               scope='Python observation+numeric decoding, includes first call; no hard latency/Haskell guarantee'),
                elapsedSeconds=time.perf_counter()-began,continuedPaths=len(restored),maxRss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                rssUnit='bytes' if platform.system()=='Darwin' else 'KiB',promotionAllowed=False,holdoutOpened=False,
                DSR='insufficient_evidence',PBO='insufficient_evidence',realOPE='unsupported')
    write(output/'summary.json',result)
    write(output/'index.json',{str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k:result[k] for k in ('planned','terminal','audit','elapsedSeconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-extraction-v7',action='store_true')
    for name in ('panel','settlements','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-prefix',type=Path)
    parser.add_argument('--resume-sha256')
    args=parser.parse_args()
    if not args.run_registered_extraction_v7: parser.error('disabled: registered extraction opt-in required')
    if bool(args.resume_prefix)!=bool(args.resume_sha256):parser.error('both resume prefix and digest required')
    run(args.panel,args.settlements,args.output,args.resume_prefix,args.resume_sha256)
