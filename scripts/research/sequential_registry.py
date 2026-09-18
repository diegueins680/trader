"""Reconcile v1 archive records before report publication; no data or trading IO.

This checks internal consistency against the supplied roster. It does not prove
that the roster matches an external preregistration, recompute return paths, or
establish economic significance.
"""
import math
from statistics import median

RL_FAMILIES = frozenset(('ppo', 'double_dqn', 'cql', 'cql_no_inventory_penalty'))


def require(condition, reason):
    if not condition:
        raise ValueError('inconsistent evidence: ' + reason)


def unique(rows, label):
    require(isinstance(rows, list), label + ' must be a list')
    out = {}
    for row in rows:
        require(isinstance(row, dict), label + ' row must be an object')
        key = row.get('id')
        require(isinstance(key, str) and bool(key) and key not in out, label + ' ID missing or duplicated')
        out[key] = row
    return out


def integer(value):
    return type(value) is int and value >= 0


def segment(value):
    return isinstance(value, str) and bool(value) and '/' not in value


def trial_id(row):
    require(segment(row['algorithm']) and all(integer(row[k]) for k in ('horizon', 'fold', 'seed')),
            'invalid trial metadata')
    require(row['horizon'] in (1, 3, 6), 'unsupported horizon')
    return f"{row['algorithm']}/h{row['horizon']}/f{row['fold']}/s{row['seed']}"


def group_key(row):
    require(all(segment(row[k]) for k in ('algorithm', 'stress')) and
            all(integer(row[k]) for k in ('horizon', 'seed')), 'invalid group metadata')
    return tuple(row[k] for k in ('algorithm', 'horizon', 'seed', 'stress'))


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'non-finite or non-numeric metric')
    return value


def equal_metric(actual, expected):
    if expected is None:
        return actual is None
    return math.isclose(number(actual), expected, rel_tol=1e-12, abs_tol=1e-12)


def reconcile_disposition(manifest, summary):
    """This v1 development screen cannot publish promotion inference."""
    require(number(summary['processPeakRssPlatformUnits']) >= 0, 'invalid process peak RSS')
    for key in ('promotionAllowed', 'holdoutOpened', 'liveAuthorization'):
        require(manifest[key] is False, 'authorizing or invalid manifest ' + key)
    for key in ('promotionAllowed', 'holdoutOpened'):
        require(summary[key] is False, 'authorizing or invalid summary ' + key)
    require(summary.get('liveAuthorization', False) is False, 'authorizing summary')
    require(manifest['evidenceClass'] == summary['evidenceClass'] == 'contaminated_development_only',
            'unsupported evidence class')
    require(summary['decision'] == 'no_candidate_passed', 'unsupported summary decision')
    require(set(summary['statistics']) <= {'DSR', 'PBO', 'SPA', 'pairedConfidence', 'reason'},
            'unsupported statistical fields')
    for key in ('DSR', 'PBO', 'SPA', 'pairedConfidence'):
        require(summary['statistics'][key] is None, 'unsupported statistical inference ' + key)


def require_outcome_reason(status, reason):
    require((isinstance(reason, str) and bool(reason.strip()) and reason.strip() != 'complete')
            if status == 'failed' else reason is None, 'status/reason semantics')


def reconcile_ope_payload(result):
    """Check v1 producer shapes, not the truth or reliability of estimates."""
    require(isinstance(result, dict) and bool(result), 'missing OPE result')
    require_outcome_reason('failed', result.get('reason'))
    if 'status' in result:
        status = result['status']
        require(status in ('failed', 'invalid'), 'unsupported OPE status')
        expected = {'status', 'reason'} | ({'failedEpisodes'} if status == 'invalid' else set())
        require(set(result) == expected, 'invalid OPE failure shape')
        if status == 'invalid':
            require(integer(result['failedEpisodes']) and result['failedEpisodes'] > 0, 'invalid OPE failure count')
        return
    fields = {'ordinaryIS', 'perDecisionIS', 'weightedIS', 'doublyRobust', 'effectiveSampleSize',
              'maxTrajectoryWeight', 'nonzeroTrajectories', 'episodes', 'horizonDecisions',
              'weightClipping', 'reliable', 'reason', 'conditionalBootstrap95', 'directSimulatorValue',
              'valueModel', 'uncertaintyScope', 'liveStateActionSupport', 'simulatedActionSupport'}
    require(set(result) == fields, 'invalid OPE estimate shape')
    require(result['reliable'] is False and result['weightClipping'] == 'none' and
            result['liveStateActionSupport'] == 'unavailable', 'unsupported OPE reliability/support')
    require(all(isinstance(result[k], str) and bool(result[k].strip())
                for k in ('valueModel', 'uncertaintyScope')), 'missing OPE limitations')
    require(result['simulatedActionSupport'] == [1/3, 1/3, 1/3], 'invalid simulated OPE support')
    require(all(integer(result[k]) for k in ('episodes', 'nonzeroTrajectories', 'horizonDecisions')) and
            result['episodes'] > 0 and result['horizonDecisions'] == 6 and
            result['nonzeroTrajectories'] <= result['episodes'], 'invalid OPE episode counts')
    ess, weight = number(result['effectiveSampleSize']), number(result['maxTrajectoryWeight'])
    require(0 <= ess <= result['nonzeroTrajectories'] + 1e-12 and weight >= 0, 'invalid OPE support metrics')
    if result['nonzeroTrajectories'] == 0:
        require(ess == 0 and weight == 0 and result['weightedIS'] is None, 'zero-support OPE metrics')
    else:
        require(ess > 0 and weight > 0, 'positive-support OPE metrics')
        number(result['weightedIS'])
    for key in ('ordinaryIS', 'perDecisionIS', 'doublyRobust', 'directSimulatorValue'):
        number(result[key])
    intervals = result['conditionalBootstrap95']
    require(isinstance(intervals, dict) and set(intervals) == {'IS', 'PDIS', 'DR'}, 'invalid OPE intervals')
    for bounds in intervals.values():
        require(isinstance(bounds, list) and len(bounds) == 2 and
                number(bounds[0]) <= number(bounds[1]), 'invalid OPE interval bounds')


def reconcile_groups(summary, records, training_count, planned_count):
    for key, expected in [('trainingFits', training_count), ('replayPaths', len(records)),
                          ('plannedEntries', planned_count)]:
        require(integer(summary[key]) and summary[key] == expected, 'summary ' + key)
    grouped = {}
    for row in records:
        grouped.setdefault(group_key(row), []).append(row['result'])
    require(isinstance(summary['groups'], list), 'summary groups must be a list')
    seen = set()
    for group in summary['groups']:
        key = group_key(group)
        require(key in grouped and key not in seen, 'unknown or duplicate summary group')
        seen.add(key)
        rows = grouped[key]
        complete = sum(r['status'] == 'complete' for r in rows)
        counts = {'paths': len(rows), 'completePaths': complete, 'failedPaths': len(rows)-complete}
        for name, expected in counts.items():
            require(integer(group[name]) and group[name] == expected, 'group ' + name)
        require(equal_metric(group['failureRate'], counts['failedPaths']/len(rows)), 'group failure rate')
        values = [r for r in rows if 'netReturn' in r]
        def mean(xs):
            return math.fsum(xs)/len(xs) if xs else None
        returns = [number(r['netReturn']) for r in values]
        sharpes = [number(r['sharpe']) for r in values if r['sharpe'] is not None]
        expected = {
            'meanTerminalOrStoppedReturn': mean(returns),
            'worstTerminalOrStoppedReturn': min(returns, default=None),
            'worstDrawdown': max((number(r['maxDrawdown']) for r in values), default=None),
            'worstES95': max((number(r['expectedShortfall95']) for r in values), default=None),
            'medianPathSharpe': median(sharpes) if sharpes else None,
            'meanFees': mean([number(r['costsOverInitialEquity']['fee']) for r in values]),
            'meanFunding': mean([number(r['fundingPnlOverInitialEquity']) for r in values]),
        }
        for name, value in expected.items():
            require(equal_metric(group[name], value), 'group ' + name)
    require(seen == set(grouped), 'missing summary groups')


def reconcile(planned, events, training, records, ope, summary, index):
    """Return terminal events only after all cross-file checks pass."""
    roster = unique(planned, 'planned')
    require(bool(roster), 'empty roster')
    for row in planned:
        require(row['kind'] in ('training', 'replay') and row['status'] == 'planned', 'invalid roster kind/status')
    terminal, started = {}, set()
    for event in events:
        key = event['id']
        require(key in roster and key not in terminal, 'unknown event or event after terminal')
        if event['status'] == 'started':
            require(key not in started, 'duplicate start event')
            started.add(key)
        else:
            require(event['status'] in ('complete', 'failed'), 'invalid event status')
            require_outcome_reason(event['status'], event.get('reason'))
            terminal[key] = event
    require(set(terminal) == set(roster), 'incomplete experiment registry')
    fits, replays = unique(training, 'training'), unique(records, 'replay')
    require(bool(replays), 'no replay rows to report')
    for kind, rows in [('training', fits), ('replay', replays)]:
        require(set(rows) == {key for key, row in roster.items() if row['kind'] == kind},
                kind + ' rows differ from roster')
    successful = set()
    for key, fit in fits.items():
        require(key in started, 'training start event missing')
        require(trial_id(fit) == key, 'training identity differs from metadata')
        # Original v1 successful training records omit status; the terminal
        # ledger and verified artifact digest must still witness completion.
        status = fit.get('status', 'complete')
        require(status in ('complete', 'failed') and status == terminal[key]['status'], 'training status')
        require_outcome_reason(status, fit.get('reason'))
        require(fit.get('reason') == terminal[key].get('reason'), 'training reason')
        if status == 'complete':
            artifact = 'policies/' + key.replace('/', '_') + '.json'
            require(isinstance(fit['artifactSha256'], str) and
                    fit['artifactSha256'] == terminal[key].get('artifactSha256') == index.get(artifact),
                    'training artifact identity')
            successful.add(key)
    for key, row in replays.items():
        trial = trial_id(row)
        require(all(segment(row[k]) for k in ('stress', 'symbol')) and
                key == f"{trial}/{row['stress']}/{row['symbol']}", 'replay identity differs from metadata')
        result, event = row['result'], terminal[key]
        require(not any(field in result for field in ('id', 'algorithm', 'horizon', 'fold', 'seed', 'stress', 'symbol')),
                'result shadows trial identity')
        require(result['status'] in ('complete', 'failed') and result['status'] == event['status'], 'replay status')
        require(integer(result['observations']) and integer(event['observations']) and
                result['observations'] == event['observations'], 'replay observations')
        require((result.get('reason') is None or isinstance(result['reason'], str)) and
                result.get('reason') == event.get('reason'), 'replay reason')
        require_outcome_reason(result['status'], result.get('reason'))
        require(('netReturn' in result) == (result['observations'] > 0), 'replay metric coverage')
        require(result['status'] != 'complete' or result['observations'] > 0, 'empty completed replay')
        if row['algorithm'] in RL_FAMILIES and result['observations'] > 0:
            require(number(result['latencyP99Ms']) >= 0, 'invalid policy latency')
            require(0 <= number(result['oodObservationRate']) <= 1, 'invalid policy OOD rate')
        if trial in fits and trial not in successful:
            require(result['status'] == 'failed' and result.get('reason') == 'training_failed' and
                    result['observations'] == 0, 'failed training has evaluated replay')
        else:
            require(key in started, 'evaluated replay start event missing')
    ope_rows = unique(ope, 'OPE')
    require(set(ope_rows) == successful, 'OPE rows differ from successful fits')
    for row in ope_rows.values():
        reconcile_ope_payload(row.get('result'))
    reconcile_groups(summary, records, len(fits), len(roster))
    return terminal
