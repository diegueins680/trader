"""Compare a temporary exact-bit child transport with the unchanged supervisor."""
import argparse
import json
from pathlib import Path
import subprocess

import diagnose_rl_bridge as d


def candidate_source(source: str) -> str:
    old = source[source.index('worker :: IO ()'):source.index('waitExit ::')]
    new = old.replace('worker ::', 'workerBits ::').replace('worker = do', 'workerBits = do')
    new = new.replace('raw >>= readMaybe >>= workerCompute', 'raw >>= decodeSnapshot >>= workerCompute')
    replacements = (
        ('import GHC.Clock (getMonotonicTimeNSec)',
         'import GHC.Clock (getMonotonicTimeNSec)\nimport GHC.Float (castDoubleToWord64)'),
        ('waitExit ::', new + 'waitExit ::'),
        ('BS.pack (show (request :: Request))',
         'BS.pack (show ("PPO-SNAPSHOT-V3" :: String, 4 :: Int, map castDoubleToWord64 (fst request), map castDoubleToWord64 (snd request)))'),
        ('createProcess (proc executable ["--offline-inference-v2", "--worker"])',
         'createProcess (proc executable ["--offline-inference-v2", "--worker-bits-v4"])'),
        ('        ["--offline-inference-v2", "--worker"] -> worker',
         '        ["--offline-inference-v2", "--worker"] -> worker\n'
         '        ["--offline-inference-v2", "--worker-bits-v4"] -> workerBits'),
    )
    for a, b in replacements:
        if source.count(a) != 1:
            raise ValueError('candidate source anchor changed')
        source = source.replace(a, b)
    return source


def run(parent: Path, expected_hash: str, output: Path) -> dict:
    if output == parent or parent in output.parents:
        raise ValueError('output must be outside parent archive')
    d.bridge.extract()
    raw = (parent / 'report.json').read_bytes()
    if d.digest(raw) != expected_hash:
        raise ValueError('parent report hash mismatch')
    report = json.loads(raw)
    fixtures = [r for r in report['requests'] if r['mode'] == 'original']
    expected = {(s, h, c) for s in (11, 23, 47) for h in (1, 3, 6) for c in range(3)}
    if len(fixtures) != 27 or {(r['seed'], r['horizon'], r['case']) for r in fixtures} != expected:
        raise ValueError('parent fixture roster mismatch')
    admitted = []
    for row in fixtures:
        frame = (parent / f"{row['seed']}-{row['horizon']}-{row['case']}.request").read_bytes()
        if d.digest(frame) != row['requestSha256']:
            raise ValueError('request hash mismatch')
        admitted.append((row, frame.decode('ascii')))
    source = d.original_source()
    candidate = candidate_source(source)
    timed = d.phase_source(candidate)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidate.hs').write_text(candidate)
    (output / 'instrumented-candidate.hs').write_text(timed)
    registration = d.ROOT / 'research-notes/registrations/rl-bridge-bit-transport-2026-10-13.json'
    manifest = {'parentReportSha256': expected_hash, 'registrationSha256': d.digest(registration.read_bytes()),
                'originalSha256': d.digest(source.encode()), 'candidateSha256': d.digest(candidate.encode()),
                'instrumentedSha256': d.digest(timed.encode()),
                'runnerSha256': d.digest(Path(__file__).read_bytes()),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=d.ROOT, text=True).strip(),
                'financialTrials': 0, 'newFits': 0, 'promotionAllowed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    exes = {mode: d.bridge.process.compile_source(text, output / mode, optimization='-O2')
            for mode, text in (('original', source), ('bit_transport', timed))}
    rows = []
    with (output / 'events.jsonl').open('x') as events:
        for i, (fixture, frame) in enumerate(admitted):
            order = ('original', 'bit_transport') if i % 2 == 0 else ('bit_transport', 'original')
            for mode in order:
                row = {k: fixture[k] for k in ('seed', 'horizon', 'case', 'requestSha256', 'expected')}
                row.update(mode=mode, **d.invoke(exes[mode], frame, fixture['expected'], mode == 'bit_transport'))
                rows.append(row)
                events.write(json.dumps(row, sort_keys=True) + '\n'); events.flush()
    counts = {mode: {status: sum(r['mode'] == mode and r['status'] == status for r in rows)
                     for status in sorted({r['status'] for r in rows})} for mode in exes}
    per_policy = {f'{s}/{h}': sum(r['mode'] == 'bit_transport' and r['seed'] == s and r['horizon'] == h
                               and r['status'] == 'accepted_matching' for r in rows)
                  for s in (11, 23, 47) for h in (1, 3, 6)}
    gate = min(per_policy.values()) > 0 and all(r['status'] in ('accepted_matching', 'safe_absence')
                                              for r in rows if r['mode'] == 'bit_transport')
    result = {'counts': counts, 'candidateSuccessesByPolicy': per_policy, 'engineeringScreenPassed': gate,
              'requests': rows, 'financialTrials': 0, 'promotionAllowed': False,
              'scope': 'Temporary candidate only. Further proof/conformance and canonical verification required.'}
    (output / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return {k: v for k, v in result.items() if k != 'requests'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-registered', action='store_true')
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--parent-report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.run_registered:
        parser.error('disabled by default; --run-registered required')
    print(json.dumps(run(args.parent.resolve(), args.parent_report_sha256, args.output.resolve()), sort_keys=True))


if __name__ == '__main__':
    main()
