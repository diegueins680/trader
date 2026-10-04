"""Cross-language bounded and generated conformance; explicitly not a proof."""
import itertools
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def bits(x):
    return struct.unpack('>Q', struct.pack('>d', x))[0]


def value(b):
    return struct.unpack('>d', struct.pack('>Q', b))[0]


def cases():
    edge = [-math.inf, -1.0, -0.25, -0.0, 0.0, 0.25, 20.0, 21.0, math.inf, math.nan,
            math.nextafter(0.25, 0), math.nextafter(0.25, 1),
            math.nextafter(20.0, 0), math.nextafter(20.0, 21),
            math.nextafter(0.0, 1), math.nextafter(0.0, -1)]
    for guards in itertools.product((False, True), repeat=6):
        for elapsed, target in itertools.product(edge, repeat=2):
            yield (*guards, bits(elapsed), bits(target))
    rng = random.Random(20260920)
    for n in range(4096):
        guards = (True,) * 6 if n % 2 else tuple(bool(rng.getrandbits(1)) for _ in range(6))
        elapsed = rng.getrandbits(64) if n % 3 else bits(rng.uniform(0, 20))
        target = rng.getrandbits(64) if n % 5 else bits(rng.choice((-0.25, 0, 0.25)))
        yield (*guards, elapsed, target)


def expected(case):
    *guards, elapsed_bits, target_bits = case
    elapsed, target = value(elapsed_bits), value(target_bits)
    admitted = (all(guards) and math.isfinite(elapsed) and 0 <= elapsed <= 20
                and math.isfinite(target) and target in (-0.25, 0, 0.25))
    return f'({target_bits},False)' if admitted else 'absent'


def check_haskell(root):
    inputs = list(cases())
    lines = ['(' + ','.join(str(x) for x in case) + ')' for case in inputs]
    with tempfile.TemporaryDirectory(prefix='trader-formal-') as build:
        exe = str(Path(build) / 'conformance')
        subprocess.run(['ghc', '-v0', '-O0', '-ihaskell/app', '-outputdir', build,
                        'formal/research/Conformance.hs', '-o', exe], cwd=root, check=True, timeout=60)
        result = subprocess.run([exe], input='default\n' + '\n'.join(lines) + '\n',
                                text=True, capture_output=True, check=True, timeout=60)
        negative = Path(build) / 'Forgery.hs'
        negative.write_text('module Forgery where\nimport Trader.Research.PolicyProposalV1\nforged :: ResearchProposal\nforged = ResearchProposal 0.25\n')
        rejection = subprocess.run(['ghc', '-v0', '-fno-code', '-ihaskell/app', '-outputdir', build, str(negative)],
                                   cwd=root, text=True, capture_output=True, timeout=60)
        if rejection.returncode == 0 or not any(message in rejection.stderr for message in ('Data constructor not in scope', 'Illegal term-level use of the type constructor')):
            raise RuntimeError('private constructor boundary not established: ' + rejection.stderr)
    actual = result.stdout.splitlines()
    wanted = ['Disabled'] + [expected(c) for c in inputs]
    if actual != wanted:
        mismatch = next((i for i, pair in enumerate(zip(actual, wanted)) if pair[0] != pair[1]), min(len(actual), len(wanted)))
        raise RuntimeError(f'Haskell/model mismatch at fixture {mismatch}')
    return {'boundedCases': 64 * 16 * 16, 'generatedCases': 4096,
            'seed': 20260920, 'defaultDisabled': True, 'privateConstructorChecked': True,
            'accepted': sum(x != 'absent' for x in wanted[1:])}
