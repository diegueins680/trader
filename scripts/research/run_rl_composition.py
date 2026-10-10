"""Explicit development-only driver; admits only two registered immutable exports."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
from rl_composition import (VERSION, OBSERVATION, STRESSES, Episode, Collector, baseline,
                            buffer_hash, forward, policy, require, train)
from causal_replay_v3 import Bar

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / 'research-notes/registrations/rl-causal-composition-v1.json'
SOURCES = ('rl_composition.py', 'run_rl_composition.py', 'causal_replay_v3.py',
           'replay_accounting_v2.py', 'optimizer_snapshot_v2.py', 'gae_targets_v2.py',
           'value_objective_v2.py', 'sequential_learning.py', 'sequential_env.py')
ALGORITHMS = ('ppo', 'double_dqn', 'cql', 'cql_alpha0', 'behavior_cloning')
RULES = ('cash', 'constant_long', 'momentum3', 'reversal3')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def load(panel, settlements, registration):
    snapshots = [p.read_bytes() for p in (panel, settlements)]
    for raw, key in zip(snapshots, ('panelSha256', 'settlementsSha256')):
        require(sha(raw) == registration['data'][key], 'unregistered development bytes')
    symbols = registration['data']['symbols']
    bars, buckets = {s: [] for s in symbols}, {s: [] for s in symbols}
    for row in csv.DictReader(io.StringIO(snapshots[0].decode())):
        if row['symbol'] in bars:
            close = int(row['closeTime'])
            bars[row['symbol']].append(Bar(row['symbol'], close, close + 1, F(row['close'])))
    for s in symbols:
        require(len(bars[s]) == 4910 and all(y.close - x.close == 28800000 for x, y in zip(bars[s], bars[s][1:])), 'invalid chronology')
        buckets[s] = [[] for _ in bars[s]]
    import bisect
    clocks = {s: [b.close for b in bars[s]] for s in symbols}
    seen = set()
    for row in csv.DictReader(io.StringIO(snapshots[1].decode())):
        s = row['symbol']
        if s not in bars: continue
        clock = int(row['fundingTime'])
        key = (s, clock)
        require(key not in seen, 'duplicate funding event'); seen.add(key)
        j = bisect.bisect_left(clocks[s], clock)
        if 0 < j < len(bars[s]):
            require(clocks[s][j-1] < clock <= clocks[s][j], 'funding bucket mismatch')
            mark, rate = F(row['resolvedMarkPrice']), F(row['fundingRate'])
            require(mark > 0, 'invalid funding mark')
            buckets[s][j].append((mark, rate))
    return {s: tuple(v) for s, v in bars.items()}, {s: tuple(tuple(b) for b in v) for s, v in buckets.items()}


def metrics(episode):
    require(episode.done, 'unfinished episode')
    rows = episode.rows
    equity = np.array([1.] + [r['equity'] for r in rows])
    increments = np.array([r['increment'] for r in rows])
    returns = np.diff(equity) / equity[:-1]
    peak = np.maximum.accumulate(equity)
    drawdown = 1 - equity / peak
    durations, run = [], 0
    for d in drawdown:
        run = run + 1 if d > 0 else 0; durations.append(run)
    holds, run, previous = [], 0, 0
    for row in rows:
        sign = np.sign(row['exposure'] or 0)
        if sign != previous:
            if run: holds.append(run)
            run = 0
        if sign: run += 1
        previous = sign
    if run: holds.append(run)
    fees = sum(r['fee'] + r['spread'] + r['slippage'] for r in rows)
    require(abs(sum(r['reward'] for r in rows) / 100 - (equity[-1] - 1)) < 1e-12, 'reward mismatch')
    return {'status': 'failed' if episode.failure else 'complete', 'failure': episode.failure,
        'observations': len(rows), 'netReturn': float(equity[-1] - 1),
        'grossPnl': sum(r['gross'] for r in rows), 'fundingPnl': sum(r['funding'] for r in rows),
        'feesSpreadSlippage': fees, 'maxDrawdown': float(max(drawdown)),
        'drawdownDurationBars': max(durations), 'ES95': float(-np.sort(returns)[:max(1, int(np.ceil(.05*len(returns))))].mean()),
        'turnover': sum(r['turnover'] for r in rows), 'fillsIncludingLiquidation': sum(r['filledUnits'] != 0 for r in rows),
        'meanAbsExposure': float(np.mean([abs(r['exposure'] or 0) for r in rows])),
        'holdingPeriodsBars': holds, 'proposalCounts': [sum(r['proposal'] == a for r in rows) for a in (-.25, 0, .25)],
        'interventionRate': float(np.mean([r['intervention'] for r in rows])),
        'unsupportedRate': float(np.mean([not r['supported'] for r in rows])),
        'incrementSum': float(increments.sum()), 'unsafeExecutableIntents': 0,
        'riskWarning': 'Admission checks are not future risk guarantees; failures retain stopped endpoints.'}


def benchmark(fit, observation):
    times = []
    for _ in range(10): policy(fit, observation)
    for _ in range(100):
        start = time.perf_counter_ns(); policy(fit, observation)
        times.append((time.perf_counter_ns()-start)/1e6)
    return {'medianMs': float(np.median(times)), 'p99Ms': float(np.quantile(times, .99)),
            'maxMs': max(times), 'calls':100, 'scope':'warm Python numeric inference only; no hard deadline/Haskell artifact parity claim'}


def paired(records):
    result = []
    for alg in ('ppo', 'double_dqn', 'cql'):
        for rule in ('cash', 'momentum3'):
            chosen = [r for r in records if r['algorithm'] in (alg, rule) and r['stress'] == 'base']
            if any(r.get('status') != 'complete' for r in chosen):
                result.append({'algorithm': alg, 'baseline': rule, 'status':'insufficient_evidence_failed_paths'}); continue
            groups = {}
            for row in chosen:
                groups.setdefault((row['fold'], row['algorithm']), []).append(row['increments'])
            differences = [np.mean(groups[f, alg], axis=0) - np.mean(groups[f, rule], axis=0) for f in (0, 1)]
            require(all(d.shape == (64,) for d in differences), 'paired window mismatch')
            # One market path per fold: seeds and symbols averaged before resampling.
            for block in (3, 9, 21):
                rng = np.random.default_rng(20261010)
                draws = np.zeros(2000)
                for d in differences:
                    starts = rng.integers(len(d), size=(2000, int(np.ceil(len(d)/block))))
                    idx = ((starts[:, :, None] + np.arange(block)) % len(d)).reshape(2000, -1)[:, :len(d)]
                    draws += d[idx].sum(axis=1) / len(differences)
                result.append({'algorithm': alg, 'baseline': rule, 'blockBars':block,
                    'status':'descriptive_development_only', 'meanPairedNetChange':float(np.mean([sum(d) for d in differences])),
                    'simultaneous95Interval':np.quantile(draws, [.05/12, 1-.05/12]).tolist(),
                    'sampling':'circular blocks within each 64-bar fold; 6 comparisons; short dependent exposed windows, no confirmation'})
    return result


def run(panel: Path, settlements: Path, output: Path):
    raw = REG.read_bytes(); reg = json.loads(raw)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    sources = {f'scripts/research/{name}': (ROOT/'scripts/research'/name).read_bytes() for name in SOURCES}
    sources[str(REG.relative_to(ROOT))] = raw
    for name, source in sources.items():
        require(subprocess.check_output(['git','show',f'{commit}:{name}'], cwd=ROOT) == source, 'uncommitted experiment source')
    bars, events = load(panel, settlements, reg)
    output.mkdir(parents=True, exist_ok=False)
    (output/'policies').mkdir(); (output/'paths').mkdir()
    manifest = {'version': VERSION, 'observation': OBSERVATION, 'commit':commit,
        'registrationSha256':sha(raw), 'sourceHashes':{k:sha(v) for k,v in sources.items()},
        'python':platform.python_version(), 'numpy':np.__version__, 'platform':platform.platform(),
        'data':reg['data'], 'promotionAllowed':False, 'holdoutOpened':False,
        'proofLedgerSha256':sha((ROOT/'formal/research/proof-ledger.json').read_bytes())}
    write(output/'manifest.json', manifest)
    planned = []
    for fold in range(len(reg['folds'])):
        for alg in (*ALGORITHMS, *RULES):
            for seed in reg['seeds'] if alg in ALGORITHMS else [0]:
                if alg in ALGORITHMS: planned.append(f'{fold}/{alg}/{seed}/training')
                for stress in STRESSES:
                    for symbol in reg['data']['symbols']: planned.append(f'{fold}/{alg}/{seed}/{stress}/{symbol}')
    write(output/'planned.json', planned)
    records, fits, terminal_ids = [], [], []
    start_run = time.perf_counter()
    with (output/'events.jsonl').open('x') as ledger:
        def event(value):
            require(value['id'] not in terminal_ids, 'duplicate trial')
            ledger.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n'); ledger.flush()
            terminal_ids.append(value['id'])
        for fi, fold in enumerate(reg['folds']):
            # Training factories receive physically sliced prefixes, including funding.
            train_bars = {s:v[:fold['trainStop']] for s,v in bars.items()}
            train_events = {s:v[:fold['trainStop']] for s,v in events.items()}
            def factory(rng):
                symbol = reg['data']['symbols'][int(rng.integers(3))]
                start = int(rng.integers(96, fold['trainStop']-32))
                return Episode(train_bars[symbol], train_events[symbol], start, start+33, enabled=True)
            collector = Collector(factory, 20261010+fi)
            frozen = collector.collect(lambda _: np.full(3, 1/3), reg['steps'])
            for value in frozen.values(): value.setflags(write=False)
            dataset_sha = buffer_hash(frozen)
            np.savez(output/f'behavior-fold{fi}.npz', **frozen)
            write(output/f'behavior-fold{fi}.json', {'sha256':dataset_sha, 'transitions':len(frozen['a']),
                'actionCounts':np.bincount(frozen['a'], minlength=3).tolist(), 'proposalPropensity':1/3,
                'origin':'simulator_generated_not_exchange_logs', 'trainingRiskFailures':collector.failures})
            for alg in (*ALGORITHMS, *RULES):
                for seed in reg['seeds'] if alg in ALGORITHMS else [0]:
                    fit, failure = None, None
                    if alg in ALGORITHMS:
                        trial = f'{fi}/{alg}/{seed}/training'; began = time.perf_counter()
                        try:
                            fit = train(alg, factory, seed, reg['steps'], frozen, enabled=True)
                            # No production loader accepts this schema. Parameter bytes are only research evidence.
                            artifact = {'version':VERSION, 'observation':OBSERVATION, 'algorithm':alg,
                                'seed':seed, 'fold':fi, 'mode':'research_only', 'sourceCommit':commit,
                                'registrationSha256':sha(raw), 'trainingDatasetSha256':dataset_sha,
                                'parametersLittleEndianHex':[p.hex() for p in fit.actor.p],
                                'updates':fit.actor.step, 'performanceEvidence':'summary.json',
                                'assuranceLedgerSha256':manifest['proofLedgerSha256']}
                            ap = output/'policies'/f'{fi}-{alg}-{seed}.json'; write(ap, artifact)
                            timing = benchmark(fit, frozen['s'][0])
                            row = {'id':trial,'status':'complete','fold':fi,'algorithm':alg,'seed':seed,
                                'seconds':time.perf_counter()-began, 'artifactSha256':sha(ap.read_bytes()),
                                'artifactBytes':ap.stat().st_size, 'inference':timing, **fit.metadata}
                        except (ValueError, ArithmeticError, MemoryError) as exc:
                            failure = str(exc)
                            row = {'id':trial,'status':'failed','fold':fi,'algorithm':alg,'seed':seed,
                                   'seconds':time.perf_counter()-began,'failure':failure}
                        fits.append(row); event(row)
                        print(json.dumps({k:row[k] for k in ('id','status','seconds')}), flush=True)
                    for stress in STRESSES:
                        for symbol in reg['data']['symbols']:
                            trial = f'{fi}/{alg}/{seed}/{stress}/{symbol}'
                            row = {'id':trial, 'fold':fi,'algorithm':alg,'seed':seed,'stress':stress,'symbol':symbol}
                            if failure:
                                row.update(status='failed', failure='training_failed:'+failure)
                            else:
                                ep = Episode(bars[symbol][:fold['evaluationStop']], events[symbol][:fold['evaluationStop']],
                                    fold['evaluationStart'],fold['evaluationStop'],stress,enabled=True)
                                qpred = []
                                try:
                                    while not ep.done:
                                        x = ep.observation()
                                        action = baseline(alg, ep) if fit is None else policy(fit,x)
                                        if alg in ('double_dqn','cql','cql_alpha0'): qpred.append(float(forward(fit.actor,x[None,:])[0,action]))
                                        ep.step(action)
                                    row.update(metrics(ep))
                                    row['increments'] = [r['increment'] for r in ep.rows]
                                    if qpred:
                                        realized = np.cumsum([r['reward'] for r in ep.rows][::-1])[::-1]
                                        row['criticMseFinitePath'] = float(np.mean((np.asarray(qpred)-realized)**2))
                                except (ValueError, ArithmeticError, MemoryError) as exc:
                                    row.update(status='failed', failure=str(exc), observations=len(ep.rows))
                                write(output/'paths'/f'{fi}-{alg}-{seed}-{stress}-{symbol}.json', ep.rows)
                            records.append(row); event({k:v for k,v in row.items() if k != 'increments'})
            require(dataset_sha == buffer_hash(frozen), 'offline buffer mutation')
    require(set(terminal_ids) == set(planned) and len(terminal_ids) == len(planned), 'registry incomplete')
    summary = {'version':VERSION,'elapsedSeconds':time.perf_counter()-start_run,
        'maxRss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rssUnit':'bytes' if platform.system() == 'Darwin' else 'KiB',
        'planned':len(planned),'terminal':len(terminal_ids),'fits':fits,
        'records':[{k:v for k,v in r.items() if k != 'increments'} for r in records], 'paired':paired(records),
        'promotionAllowed':False,'holdoutOpened':False,'adoption':'continue_research',
        'DSR':'insufficient_evidence','PBO':'insufficient_evidence',
        'OPE':'No real logged behavior/support. Historical replay is model-based counterfactual evidence. No valid real IS/DR/FQE estimate.',
        'blockers':['matched current champion','authentic first-seen data','untouched confirmation',
                    'full production safety composition','realistic precision and min-notional','short pilot compute and horizon']}
    write(output/'summary.json',summary)
    write(output/'index.json',{str(p.relative_to(output)):sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({'complete':True,'planned':len(planned),'elapsedSeconds':summary['elapsedSeconds']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered-development-v1', action='store_true')
    parser.add_argument('--panel',type=Path); parser.add_argument('--settlements',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if not args.run_registered_development_v1 or not all((args.panel,args.settlements,args.output)):
        parser.error('disabled: explicit development opt-in and registered input/output paths required')
    run(args.panel,args.settlements,args.output)
