import collections
import datetime
import json
import math
import re
import subprocess
import sys

def graduation_summary(log_text, sync_attempts):
    phases = ['connection', 'baseline', 'daily', 'orders', 'status', 'latest']
    records, malformed = [], 0
    pending_counts = {"missingBaseline": 0, "otherPending": 0}
    expected_keys = {'event', 'attemptId', 'atMs', 'phase', 'state', 'elapsedMs'}
    for line in log_text.splitlines():
        text = line.split(' ', 1)[-1]
        prefix = 'Automatic portfolio graduation remains pending: '
        if text.startswith(prefix):
            missing = text == prefix + 'graduation baseline is missing for a reviewed UUID'
            pending_counts['missingBaseline' if missing else 'otherPending'] += 1
        try:
            event = json.loads(text)
        except (ValueError, TypeError):
            malformed += 'portfolio_graduation_query' in text
            continue
        if not isinstance(event, dict) or event.get('event') != 'portfolio_graduation_query':
            continue
        valid = (set(event) == expected_keys and isinstance(event['attemptId'], str)
                 and re.fullmatch(r'ThreadId \d+:\d+', event['attemptId'])
                 and type(event['atMs']) is int and event['atMs'] > 0
                 and event['phase'] in phases and event['state'] in ['started', 'finished'])
        elapsed = event.get('elapsedMs')
        valid = valid and ((event['state'] == 'started' and elapsed is None)
                          or (event['state'] == 'finished' and type(elapsed) in [int, float]
                              and math.isfinite(elapsed) and elapsed >= 0))
        if valid:
            records.append(event)
        else:
            malformed += 1
    grouped = collections.defaultdict(list)
    for event in records:
        grouped[event['attemptId']].append(event)
    attempts = []
    for attempt_id, rows in grouped.items():
        details = []
        for phase in phases:
            starts = [r for r in rows if r['phase'] == phase and r['state'] == 'started']
            ends = [r for r in rows if r['phase'] == phase and r['state'] == 'finished']
            complete = len(starts) == len(ends) == 1 and starts[0]['atMs'] <= ends[0]['atMs']
            details.append({'phase': phase, 'complete': complete,
                            'startedAtMs': starts[0]['atMs'] if starts else None,
                            'finishedAtMs': ends[0]['atMs'] if ends else None,
                            'elapsedMs': ends[0]['elapsedMs'] if complete else None,
                            'startCount': len(starts), 'finishCount': len(ends)})
        parent, children = details[0], details[1:]
        nested = parent['complete'] and all(c['complete'] and parent['startedAtMs'] <= c['startedAtMs'] <= c['finishedAtMs'] <= parent['finishedAtMs'] for c in children)
        ordered = nested and all(a['finishedAtMs'] <= b['startedAtMs'] for a, b in zip(children, children[1:]))
        accounted = sum(c['elapsedMs'] for c in children) if nested else None
        residual = parent['elapsedMs'] - accounted if nested else None
        consistent = ordered and residual >= -1
        overlaps = []
        for sync in sync_attempts:
            db = sync.get('dbReadBreakdown', {})
            waits = [r for r in db.get('records', []) if r['stage'] == 'connection-wait']
            if not parent['complete'] or not db.get('consistent') or len(waits) != 1:
                continue
            wait = waits[0]
            begin, end = wait['atMs'] - wait['elapsedMs'], wait['atMs']
            overlap = max(0, min(parent['finishedAtMs'], end) - max(parent['startedAtMs'], begin))
            overlaps.append({'syncAttemptId': sync['attemptId'], 'connectionWaitMs': wait['elapsedMs'],
                             'waitStartedAtMsApprox': begin, 'waitFinishedAtMs': end,
                             'overlapMsApprox': overlap,
                             'uncoveredWaitMsApprox': max(0, wait['elapsedMs'] - overlap)})
        attempts.append({'attemptId': attempt_id, 'phases': details, 'allQueriesCompleteAndOrdered': bool(consistent),
                         'queryTimeMs': accounted, 'otherConnectionMs': residual, 'syncWaitOverlaps': overlaps})
    complete_parents = [a['phases'][0] for a in attempts if a['phases'][0]['complete']]
    conflicting = sum(min(a['finishedAtMs'], b['finishedAtMs']) - max(a['startedAtMs'], b['startedAtMs']) > 1
                      for i, a in enumerate(complete_parents) for b in complete_parents[i+1:])
    return {'pendingWarningCounts': pending_counts, 'recordCount': len(records), 'malformedRecords': malformed, 'attempts': attempts,
            'overlappingGraduationConnectionPairs': conflicting,
            'scope': 'One shared-provider callback per attempt; child query lifetimes are inside the connection interval. Completion includes failure/cancellation. UTC overlap is approximate to millisecond recording precision and does not identify other holders or establish database success.'}


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
    events, phases, malformed = [], [], 0
    db_events, db_malformed = [], 0
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
            malformed += 'top_combos_build_gate' in text or 'top_combos_sync_phase' in text
            db_malformed += 'top_combos_db_read' in text
            continue
        if isinstance(event, dict) and event.get('event') == 'top_combos_db_read':
            valid = set(event) == {'event','attemptId','atMs','stage','queryAttempt','elapsedMs'} and type(event['atMs']) is int and isinstance(event['attemptId'],str) and re.fullmatch(r'ThreadId \d+:\d+',event['attemptId'])
            valid = valid and event['stage'] in ['connection-wait','query'] and type(event['queryAttempt']) is int and event['queryAttempt'] >= 1
            valid = valid and type(event['elapsedMs']) in [int,float] and math.isfinite(event['elapsedMs']) and event['elapsedMs'] >= 0
            if valid:
                db_events.append(event)
            else:
                db_malformed += 1
            continue
        if isinstance(event, dict) and event.get('event') == 'top_combos_sync_phase':
            valid = set(event) == {'event', 'attemptId', 'atMs', 'phase', 'state', 'elapsedMs'} and type(event['atMs']) is int and isinstance(event['attemptId'], str) and re.fullmatch(r'ThreadId \d+:\d+', event['attemptId'])
            valid = valid and event['phase'] in ['read-db', 'read-s3', 'read-state-sync', 'plan', 'write-db', 'write-remote'] and event['state'] in ['started', 'finished']
            elapsed = event.get('elapsedMs')
            valid = valid and ((event['state'] == 'started' and elapsed is None) or (event['state'] == 'finished' and type(elapsed) in [int, float] and math.isfinite(elapsed) and elapsed >= 0))
            if valid:
                phases.append(event)
            else:
                malformed += 1
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
        observed = [event for event in phases if event['attemptId'] == attempt_id]
        details = []
        for name in dict.fromkeys(event['phase'] for event in observed):
            began = [event for event in observed if event['phase'] == name and event['state'] == 'started']
            finished = [event for event in observed if event['phase'] == name and event['state'] == 'finished']
            complete = len(began) == len(finished) == 1
            details.append({'phase': name, 'complete': complete, 'startedAtMs': began[0]['atMs'] if began else None, 'finishedAtMs': finished[0]['atMs'] if finished else None, 'elapsedMs': finished[0]['elapsedMs'] if complete else None})
        observed_complete = bool(details) and all(row['complete'] for row in details)
        missing_mandatory = sorted({'read-db', 'read-s3', 'read-state-sync', 'plan'} - {row['phase'] for row in details})
        complete_phases = observed_complete and not missing_mandatory
        accounted = sum(row['elapsedMs'] for row in details) if observed_complete else None
        held = end['heldMs'] if end else None
        residual = held - accounted if held is not None and accounted is not None else None
        largest = max(details, key=lambda row: row['elapsedMs']) if observed_complete else None
        sync.append({'attemptId': attempt_id, 'acquiredAtMs': start['atMs'] if start else None, 'finishedAtMs': end['atMs'] if end else None, 'waitMs': end['waitMs'] if end else None, 'heldMs': held, 'complete': end is not None, 'phaseRecords': observed, 'phases': details, 'phasesComplete': complete_phases, 'observedPhasesComplete': observed_complete, 'missingMandatoryPhases': missing_mandatory, 'accountedPhaseMs': accounted, 'unattributedGateMs': residual, 'largestObservedPhase': largest})
    for row in sync:
        records = [event for event in db_events if event['attemptId']==row['attemptId']]
        waits = [event for event in records if event['stage']=='connection-wait']
        queries = [event for event in records if event['stage']=='query']
        parent = next((phase for phase in row['phases'] if phase['phase']=='read-db'),None)
        complete = len(waits)==1 and waits[0]['queryAttempt']==1 and 1 <= len(queries) <= 2 and [event['queryAttempt'] for event in queries]==list(range(1,len(queries)+1)) and parent is not None and parent['complete']
        within = complete and all(parent['startedAtMs']-1 <= event['atMs'] <= parent['finishedAtMs']+1 for event in records)
        query_total = sum(event['elapsedMs'] for event in queries)
        residual = parent['elapsedMs']-waits[0]['elapsedMs']-query_total if complete else None
        row['dbReadBreakdown'] = {'records':records,'complete':bool(complete),'withinParentWindow':bool(within),'initialConnectionWaitMs':waits[0]['elapsedMs'] if len(waits)==1 else None,'queryAttempts':len(queries),'queryTotalMs':query_total,'parentReadDbMs':parent['elapsedMs'] if parent else None,'otherReadDbMs':residual,'consistent':bool(complete and within and residual>=-1),'scope':'Child timings are inside read-db and excluded from gate-phase sums. Query includes transfer and row conversion performed by IO; residual can include deferred work, reconnection and logging, without success certification.'}
    if run(['docker', 'inspect', '--format', '{{.State.StartedAt}}', 'trader-api-1']).strip() != started:
        raise RuntimeError('Container changed')
    out = {'commit': expected, 'collectedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'containerStartedAt': started, 'ready': health['ready'], 'gateEvents': len(events), 'phaseEvents': len(phases), 'dbReadEvents':len(db_events),'malformedDbReadRecords':db_malformed,'orphanDbReadRecords':sum(event['attemptId'] not in attempts or attempts[event['attemptId']][0]['origin']!='sync' for event in db_events), 'orphanPhaseRecords': sum(event['attemptId'] not in attempts for event in phases), 'malformedGateRecords': malformed, 'syncAttempts': sync, 'settledSkipLogCount': settled, 'reconcileLogCount': reconciled, 'replicaWarningCounts': warnings, 'scope': 'Gate and phase events since container startup; includes ordinary reads and read-only probe effects. Completion is lifetime, not success. Phase timings follow existing IO/evaluation boundaries; deferred work may run in a later phase.'}
    out['graduation'] = graduation_summary(logs.stdout + '\n' + logs.stderr, sync)
    print(json.dumps(out, separators=(',', ':')))
except Exception:
    raise SystemExit('Sync gate probe unavailable')
