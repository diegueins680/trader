"""Telemetry must not turn an absence or late answer into an inference success."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research'))
import diagnose_rl_bridge as d


class BridgeDiagnosticTests(unittest.TestCase):
    def test_absence_is_not_success(self):
        expected = '(QuarterTarget 1,True)'
        self.assertEqual(d.classify('(Absent,True)', '', expected, False)['status'], 'safe_absence')
        self.assertEqual(d.classify(expected, '', expected, False)['status'], 'accepted_matching')
        self.assertEqual(d.classify('(QuarterTarget 0,True)', '', expected, False)['status'], 'unexpected_output')
        self.assertEqual(d.classify('(Absent,False)', '', expected, False)['status'], 'unexpected_output')

    def test_timing_identity_and_strict_deadline(self):
        expected = '(QuarterTarget 1,True)'
        for line in ('', 'RLTIMING [0,1,2,4]', 'RLTIMING [0,true,0,1]',
                     'RLTIMING [0,-1,1,0]', 'RLTIMING [0,0,0,0]\nextra'):
            self.assertEqual(d.classify(expected, line, expected, True)['status'], 'invalid_telemetry')
        self.assertEqual(d.classify(expected, 'RLTIMING [50,19000000,1000000,20000000]', expected, True)['status'],
                         'late_accepted_violation')
        self.assertEqual(d.classify(expected, 'RLTIMING [50,19000000,999999,19999999]', expected, True)['status'],
                         'accepted_matching')

    def test_instrumentation_preserves_original_control(self):
        source = (d.ROOT / d.bridge.process.SOURCE).read_text()
        variant = d.phase_source(source)
        self.assertEqual(variant.count('RLTIMING'), 1)
        self.assertIn('budgetNS = 20000000', variant)
        self.assertIn('    clean <- cleanup handles\n    ended <- clockNS\n    hPutStrLn stderr', variant)
        with self.assertRaises(ValueError):
            d.phase_source(source.replace('    ended <- clockNS\n', ''))


if __name__ == '__main__':
    unittest.main()
