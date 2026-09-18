#!/usr/bin/env python3
"""Export small review evidence from a hash-verified completed offline screen.

No market input is read and no policy is trained, selected or promoted.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from sequential_registry import RL_FAMILIES, reconcile, reconcile_disposition


REPORT_INPUTS = frozenset({'manifest.json', 'summary.json', 'evaluation.json',
                          'training.json', 'planned-registry.json', 'events.jsonl', 'ope.json'})


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def decode_evidence(raw: bytes) -> object:
    """Decode verified JSON without discarded keys or non-finite numeric values."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate evidence JSON key')
            result[key] = value
        return result
    def reject_constant(_):
        raise ValueError('non-finite evidence JSON constant')
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError('non-finite evidence JSON number')
        return value
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant,
                      parse_float=finite_float)


def export(source, output, *, rss_unit, platform_label, expected_index_sha256):
    index_bytes = (source / 'evidence-index.json').read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != expected_index_sha256:
        raise ValueError('external evidence index hash mismatch')
    index = decode_evidence(index_bytes)
    if not isinstance(index, dict) or not REPORT_INPUTS <= index.keys():
        raise ValueError('external evidence index missing report inputs')
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual != set(index) | {'evidence-index.json'}:
        raise ValueError('external evidence file inventory mismatch')
    snapshots = {}
    for name, sha in index.items():
        p = source / name
        if not p.resolve().is_relative_to(source.resolve()):
            raise ValueError('external evidence hash/path mismatch')
        # Keep only report inputs in memory. Large return paths and policy
        # artifacts are still hash-verified with bounded streaming reads.
        if name in REPORT_INPUTS:
            raw = p.read_bytes()
            actual_sha = hashlib.sha256(raw).hexdigest()
            snapshots[name] = raw
        else:
            actual_sha = digest(p)
        if actual_sha != sha:
            raise ValueError('external evidence hash/path mismatch')
    read = lambda name: decode_evidence(snapshots.pop(name))
    manifest, summary = read('manifest.json'), read('summary.json')
    records, training, planned = read('evaluation.json'), read('training.json'), read('planned-registry.json')
    events = [decode_evidence(line) for line in snapshots.pop('events.jsonl').splitlines()]
    ope = read('ope.json')
    try:
        reconcile_disposition(manifest, summary)
        terminal = reconcile(planned, events, training, records, ope, summary, index)
        reports = render_reports(manifest, summary, records, training, planned, terminal, ope, index,
                                 rss_unit, platform_label, expected_index_sha256)
        reports = {name: content.encode('utf-8') for name, content in reports.items()}
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('malformed registry evidence') from exc
    output.mkdir(parents=True, exist_ok=False)
    for name, content in reports.items():
        (output / name).write_bytes(content)


def render_reports(manifest, summary, records, training, planned, terminal, ope, index,
                   rss_unit, platform_label, expected_index_sha256):
    """Prepare every compact report before creating output; no filesystem IO."""
    reports = {}
    def js(name, value):
        reports[name] = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    def csvfile(name, rows, fields):
        with io.StringIO(newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
            reports[name] = stream.getvalue()
    reasons = sorted({r.get('reason') for r in terminal.values() if r.get('reason')})
    codes = {reason: i+1 for i, reason in enumerate(reasons)}
    csvfile('experiment-registry.csv', [dict(id=p['id'], kind=p['kind'], status=terminal[p['id']]['status'],
        reasonCode=codes.get(terminal[p['id']].get('reason'), 0), observations=terminal[p['id']].get('observations',''))
        for p in planned], ['id','kind','status','reasonCode','observations'])
    groups = summary.pop('groups')
    csvfile('all-seed-results.csv', groups, sorted(groups[0]))
    fields = ['algorithm','horizon','seed','fold','symbol','status','reason','observations','netReturn','sharpe',
              'maxDrawdown','expectedShortfall95','oodObservationRate','latencyP99Ms']
    csvfile('symbol-fold-base-results.csv', [{k: ({**r, **r['result']}).get(k) for k in fields}
            for r in records if r['stress']=='base'], fields)
    compact = []
    for info in training:
        t = {k:v for k,v in info.items() if k not in ('episodes','losses')}
        episodes = info.get('episodes',[])
        t['trainingEpisodes'] = len(episodes)
        t['trainingFailures'] = dict(Counter(e.get('failure') or 'complete' for e in episodes))
        values = sorted(e['return'] for e in episodes)
        t['trainingEpisodeReturns'] = {'min':min(values), 'mean':sum(values)/len(values), 'max':max(values)} if values else None
        losses=info.get('losses',[])
        t['initialAndFinalLoss'] = [losses[0],losses[-1]] if losses else []
        compact.append(t)
    js('multi-seed-training.json',compact)
    js('ope-report.json',ope)
    rl = [r for r in records if r['algorithm'] in RL_FAMILIES]
    values = [r['result'] for r in rl if 'netReturn' in r['result']]
    sizes = [t['artifactBytes'] for t in training if 'artifactBytes' in t]
    summary.update(trainingSecondsSum=sum(t.get('seconds',0) for t in training),
        artifactByteRange=[min(sizes),max(sizes)] if sizes else None,
        rlAllPathsMaxDrawdown=max((v['maxDrawdown'] for v in values),default=None),
        rlAllPathsWorstES95=max((v['expectedShortfall95'] for v in values),default=None),
        rlObservedMaxP99InferenceMs=max((v['latencyP99Ms'] for v in values),default=None),
        rlOodObservationRateRange=[min(v['oodObservationRate'] for v in values),max(v['oodObservationRate'] for v in values)] if values else None,
        totalFailuresByReason=dict(Counter(r['result'].get('reason') or 'complete' for r in records)),
        rlBaseFailuresByReason=dict(Counter(r['result'].get('reason') or 'complete' for r in rl if r['stress']=='base')))
    summary['rlStressFailures']={stress:dict(failed=sum(r['result']['status']!='complete' for r in rl if r['stress']==stress),
        total=sum(r['stress']==stress for r in rl)) for stress in sorted({r['stress'] for r in rl})}
    # Explicit run-host units; never infer them from this export host.
    summary['platform']=platform_label
    summary['peakResidentMemoryMiB']=summary['processPeakRssPlatformUnits']/(1048576 if rss_unit=='bytes' else 1024)
    js('evaluation-summary.json',summary)
    manifest.update(externalEvidenceFiles=index,externalEvidenceIndexSha256=expected_index_sha256,
        registryReasonCodes={str(v):k for k,v in codes.items()},registryRows=len(planned),allPlannedEntriesTerminal=True,
        policyParametersNotCommitted=True,sourceDataNotCommitted=True,
        priorTrialAccounting=dict(earlierResidualFundingAttempts=46,
        thisScreenSeededCandidateTrials=len({(t["algorithm"],t["horizon"],t["seed"]) for t in training}),
        thisScreenOuterCandidateFits=len(training),
        thisScreenBaselineHorizonConfigurations=len({(r["algorithm"],r["horizon"]) for r in records if r["algorithm"] not in RL_FAMILIES}),
        baselineOuterRefits=len({(r["horizon"],r["fold"]) for r in records if r["algorithm"] not in RL_FAMILIES}),replayPaths=len(records)))
    js('experiment-manifest.json',manifest)
    return reports


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rss-unit',choices=['bytes','kib'],required=True)
    parser.add_argument('--platform',required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    args=parser.parse_args()
    export(args.source,args.output,rss_unit=args.rss_unit,platform_label=args.platform,
           expected_index_sha256=args.expected_index_sha256)
