import collections
import datetime
import json
import math
import re
import subprocess
import sys

expected = sys.argv[1]
if not re.fullmatch('[0-9a-f]{40}', expected):
    raise SystemExit('Expected commit required')

def run(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=40)
    if r.returncode:
        raise RuntimeError('Probe unavailable')
    return r.stdout

try:
    js = 'fetch("http://127.0.0.1:8080/health",{signal:AbortSignal.timeout(5000)}).then(r=>r.json()).then(h=>console.log(JSON.stringify({commit:h.commit,ready:h.ready}))).catch(()=>{process.exitCode=1})'
    health = json.loads(run(['docker', 'exec', 'trader-api-1', 'node', '-e', js]))
    if health['commit'] != expected:
        raise RuntimeError('Commit mismatch')
    started = run(['docker', 'inspect', '--format', '{{.State.StartedAt}}', 'trader-api-1']).strip()
    logs = subprocess.run(['docker', 'logs', '--timestamps', '--since', started, 'trader-api-1'], capture_output=True, text=True, timeout=40)
    if logs.returncode:
        raise RuntimeError('Log unavailable')
    events, malformed = [], 0
    warnings = {name: 0 for name in ['retention', 'local', 'db', 's3', 'state-sync']}
    settled = reconciled = 0
    keys = {'event', 'origin', 'attemptId', 'atMs', 'phase', 'waitMs', 'heldMs'}
    for line in (logs.stdout + '\n' + logs.stderr).splitlines():
        text = line.split(' ', 1)[-1]
        settled += text.startswith('Top combos sync: replicas unchanged since the last reconciled pass;')
        reconciled += text.startswith('Top combos sync reconciled ')
        for name in warnings:
            warnings[name] += text.startswith('Top combos sync ' + name + ' failed:')
        try:
            event = json.loads(text)
        except (ValueError, TypeError):
            malformed += 'top_combos_build_gate' in text
            continue
        if not isinstance(event, dict) or event.get('event') != 'top_combos_build_gate':
            continue
        valid = set(event) == keys and event['origin'] in ['sync', 'response'] and event['phase'] in ['waiting', 'acquired', 'finished'] and type(event['atMs']) is int and isinstance(event['attemptId'], str) and re.fullmatch(r'ThreadId \d+:\d+', event['attemptId'])
        valid = valid and all(event[key] is None or (type(event[key]) in [int, float] and math.isfinite(event[key]) and event[key] >= 0) for key in ['waitMs', 'heldMs'])
        if valid:
            events.append(event)
        else:
            malformed += 1
    attempts = collections.defaultdict(list)
    for event in events:
        attempts[event['attemptId']].append(event)
    sync = []
    for attempt_id, rows in attempts.items():
        if rows[0]['origin'] != 'sync':
            continue
        start = next((r for r in rows if r['phase'] == 'acquired'), None)
        end = next((r for r in rows if r['phase'] == 'finished'), None)
        sync.append({'attemptId': attempt_id, 'acquiredAtMs': start['atMs'] if start else None, 'finishedAtMs': end['atMs'] if end else None, 'waitMs': end['waitMs'] if end else None, 'heldMs': end['heldMs'] if end else None, 'complete': end is not None})
    if run(['docker', 'inspect', '--format', '{{.State.StartedAt}}', 'trader-api-1']).strip() != started:
        raise RuntimeError('Container changed')
    out = {'commit': expected, 'collectedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'containerStartedAt': started, 'ready': health['ready'], 'gateEvents': len(events), 'malformedGateRecords': malformed, 'syncAttempts': sync, 'settledSkipLogCount': settled, 'reconcileLogCount': reconciled, 'replicaWarningCounts': warnings, 'scope': 'Existing gate events since container startup; includes ordinary reads and read-only probe effects. Gate completion does not certify action success.'}
    print(json.dumps(out, separators=(',', ':')))
except Exception:
    raise SystemExit('Sync gate probe unavailable')
