"""Registered synthetic inference timing diagnostic; no financial policy selection.

The unchanged executable is the control. A temporary copy adds timestamps and
post-measurement stderr telemetry. Neither is an order-capable component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/formal'))
import ppo_process_bridge as bridge
from ppo_successor_v2 import train_ppo_v2
from ppo_inference_v3 import encode_request_v3

REGISTRATION = ROOT / 'research-notes/registrations/rl-bridge-timing-2026-10-13.json'


def original_source() -> str:
    raw = (ROOT / 'formal/research/fixtures/inference-process-decimal-v3.hs').read_bytes()
    if digest(raw) != 'd5a01cded9ab8b110e0e14ad7348323102e1452e5c2a4fb9397284d722bb2ad1':
        raise ValueError('frozen decimal-transport fixture drift')
    return raw.decode('utf-8')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def phase_source(source: str) -> str:
    replacements = (
        ('hSetBuffering, stdin, stdout)', 'hSetBuffering, stdin, stdout, stderr, hPutStrLn)'),
        ('supervise decode = mask $ \\restore -> do\n',
         'supervise decode = mask $ \\restore -> do\n    origin <- clockNS\n'),
        ('    clean <- cleanup handles\n',
         '    exchanged <- clockNS\n    clean <- cleanup handles\n'),
        ('    ended <- clockNS\n',
         '    ended <- clockNS\n'
         '    hPutStrLn stderr ("RLTIMING " ++ show [started - origin, exchanged - started, ended - exchanged, ended - started])\n'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('diagnostic source anchor changed')
        source = source.replace(old, new)
    return source


def classify(stdout: str, stderr: str, expected: str, instrumented: bool) -> dict:
    result = {'stdout': stdout.strip(), 'stderr': stderr.strip()}
    if stdout.strip() == expected:
        result['status'] = 'accepted_matching'
    elif stdout.strip() == '(Absent,True)':
        result['status'] = 'safe_absence'
    else:
        result['status'] = 'unexpected_output'
    if instrumented:
        try:
            if not stderr.startswith('RLTIMING '):
                raise ValueError('missing phase marker')
            ns = json.loads(stderr[len('RLTIMING '):])
            if (type(ns) is not list or len(ns) != 4 or
                    any(type(n) is not int or n < 0 for n in ns) or ns[1] + ns[2] != ns[3]):
                raise ValueError('invalid phase timing')
            result['phaseMilliseconds'] = dict(zip(
                ('startup', 'exchange', 'cleanup', 'admissionInterval'), (n / 1e6 for n in ns)))
            if result['status'] == 'accepted_matching' and ns[3] >= 20_000_000:
                result['status'] = 'late_accepted_violation'
        except (ValueError, TypeError):
            result['status'] = 'invalid_telemetry'
    elif stderr:
        result['status'] = 'unexpected_stderr'
    return result


def invoke(exe: Path, request: str, expected: str, instrumented: bool) -> dict:
    started = time.perf_counter()
    try:
        completed = subprocess.run([str(exe), '--offline-snapshot-v3'], input=request,
                                   capture_output=True, text=True, timeout=3, check=False)
        result = classify(completed.stdout, completed.stderr, expected, instrumented)
        result['exitCode'] = completed.returncode
        if completed.returncode != 0:
            result['status'] = 'process_failed'
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps the invoked parent. Its supervisor's
        # cleanup completion cannot be inferred from an outer timeout.
        result = {'status': 'outer_timeout', 'exitCode': None,
                  'cleanup': 'not established; no successful inference claimed'}
    result['wallMilliseconds'] = 1000 * (time.perf_counter() - started)
    return result


def run(output: Path) -> dict:
    bridge.extract()  # Authenticate the current original source and helpers.
    registration = json.loads(REGISTRATION.read_bytes())
    source = original_source()
    instrumented = phase_source(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'instrumented.hs').write_text(instrumented)
    manifest = {'schema': 'rl-bridge-timing-v1', 'registrationSha256': digest(REGISTRATION.read_bytes()),
                'sourceSha256': digest(source.encode()), 'instrumentedSha256': digest(instrumented.encode()),
                'runnerSha256': digest(Path(__file__).read_bytes()),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'python': platform.python_version(), 'numpy': np.__version__,
                'platform': platform.platform(), 'logicalCpus': os.cpu_count(),
                'loadBefore': os.getloadavg(), 'financialTrials': 0, 'promotionAllowed': False,
                'holdoutOpened': False, 'registration': registration}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    exes = {name: bridge.process.compile_source(text, output / name, optimization='-O2')
            for name, text in (('original', source), ('instrumented', instrumented))}
    x = np.arange(180, dtype=float)
    prices = {'ALPHA': 100 * np.exp(.0002*x + .002*np.sin(x/9)),
              'BETA': 200 * np.exp(-.0001*x + .003*np.sin(x/11))}
    funding = {s: np.zeros(len(p)) for s, p in prices.items()}
    rng = np.random.default_rng(20261005)
    rows, fits = [], []
    with (output / 'events.jsonl').open('x') as events:
        for seed in (11, 23, 47):
            for horizon in (1, 3, 6):
                started = time.perf_counter()
                trained = train_ppo_v2(prices, funding, horizon, seed, 17, enabled=True)
                if trained is None:
                    raise ValueError('synthetic training failed')
                fits.append({'seed': seed, 'horizon': horizon, 'steps': 17,
                             'milliseconds': 1000 * (time.perf_counter() - started)})
                w1, b1, w2, b2 = [np.frombuffer(raw, dtype='<f8').reshape(shape) for raw, shape in
                                 zip(trained.actor.p, ((12, 16), (16,), (16, 3), (3,)))]
                for case in range(3):
                    observation = rng.normal(0, .5, 12)
                    frame = encode_request_v3(trained, observation, enabled=True)
                    if frame is None:
                        raise ValueError('request rejected')
                    scores = np.tanh(observation @ w1 + b1) @ w2 + b2
                    margin = float(np.diff(np.sort(scores))[-1])
                    if margin <= 1e-9:
                        raise ValueError('near-tie fixture')
                    action = int(scores.argmax()) - 1
                    expected = f'(QuarterTarget {action if action >= 0 else "(-1)"},True)'
                    identity = f'{seed}-{horizon}-{case}'
                    (output / (identity + '.request')).write_bytes(frame)
                    # Alternate order to expose, rather than hide, paired scheduling effects.
                    order = ('original', 'instrumented') if (len(rows) // 2) % 2 == 0 else ('instrumented', 'original')
                    for mode in order:
                        row = {'seed': seed, 'horizon': horizon, 'case': case, 'mode': mode,
                               'requestSha256': digest(frame), 'margin': margin, 'expected': expected,
                               **invoke(exes[mode], frame.decode('ascii'), expected, mode == 'instrumented')}
                        rows.append(row)
                        events.write(json.dumps(row, sort_keys=True) + '\n'); events.flush()
    report = {'schema': 'rl-bridge-timing-v1', 'fits': fits, 'requests': rows,
              'loadAfter': os.getloadavg(), 'financialTrials': 0, 'promotionAllowed': False,
              'scope': 'Paired single-host diagnostic, not a hard timing guarantee or substitute conformance gate.',
              'counts': {mode: {status: sum(r['mode'] == mode and r['status'] == status for r in rows)
                        for status in sorted({r['status'] for r in rows})} for mode in exes}}
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report['counts']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered:
        parser.error('disabled by default; --run-registered is required')
    print(json.dumps(run(args.output.resolve()), sort_keys=True))


if __name__ == '__main__':
    main()
