import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_position_history as aph


class AnalyzePositionHistoryTest(unittest.TestCase):
    def write_ndjson(self, rows: list[dict]) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "trades.ndjson"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot json\n")
        return path

    def test_report_ranks_methods_symbols_and_exit_reasons(self) -> None:
        path = self.write_ndjson(
            [
                {
                    "trade_id": "t1",
                    "timestamp": "2026-06-01T00:00:00Z",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "method": "ta_best",
                    "volConfGate": "disabled",
                    "exitReason": "TAKE_PROFIT",
                    "pnlPercent": 0.05,
                    "pnl": 12.5,
                },
                {
                    "trade_id": "t1",
                    "timestamp": "2026-06-01T00:00:00Z",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "method": "ta_best",
                    "volConfGate": "disabled",
                    "exitReason": "TAKE_PROFIT",
                    "pnlPercent": 0.05,
                    "pnl": 12.5,
                },
                {
                    "trade_id": "t2",
                    "timestamp": "2026-06-01T01:00:00Z",
                    "symbol": "SOLUSDT",
                    "side": "LONG",
                    "signal_method": "ta_regime_switch",
                    "vol_conf_gate": "disabled",
                    "exitReason": "TRAILING_STOP",
                    "pnl_pct": "0.02",
                    "pnl_quote": "4.0",
                },
                {
                    "trade_id": "t3",
                    "timestamp": "2026-06-01T02:00:00Z",
                    "symbol": "ETHUSDT",
                    "side": "LONG",
                    "method": "ta_trend",
                    "volConfGate": "vol_conf_v1_default",
                    "exitReason": "SIGNAL",
                    "pnlPercent": -0.03,
                    "pnl": -7.0,
                },
                {
                    "trade_id": "t4",
                    "timestamp": "2026-06-01T03:00:00Z",
                    "symbol": "ETHUSDT",
                    "side": "LONG",
                    "method": "ta_trend",
                    "volConfGate": "vol_conf_v1_default",
                    "exitReason": "SIGNAL",
                    "pnlPercent": -0.04,
                    "pnl": -8.0,
                },
            ]
        )

        records, skipped, duplicates = aph.load_trades(path)
        report = aph.build_report(records, skipped, duplicates, min_group_trades=1, top=3)

        self.assertEqual(report["summary"]["trades"], 4)
        self.assertEqual(report["summary"]["duplicateTrades"], 1)
        self.assertEqual(report["summary"]["skippedLines"], 1)
        self.assertEqual(report["summary"]["sourceCounts"], {"paper_or_backtest": 4})
        self.assertEqual(report["recommendations"]["favorMethods"][:2], ["ta_best", "ta_regime_switch"])
        self.assertEqual(report["recommendations"]["avoidMethods"], ["ta_trend"])
        self.assertEqual(report["recommendations"]["favorSymbols"], ["SOLUSDT"])
        self.assertEqual(report["recommendations"]["avoidSymbols"], ["ETHUSDT"])
        self.assertEqual(report["recommendations"]["reviewExitReasons"], ["SIGNAL"])

    def test_markdown_contains_gain_and_loss_tables(self) -> None:
        record = aph.TradeRecord(
            line_no=1,
            timestamp="2026-06-01T00:00:00Z",
            symbol="SOLUSDT",
            side="LONG",
            method="ta_best",
            vol_conf_gate="disabled",
            exit_reason="TAKE_PROFIT",
            pnl_pct=0.05,
            pnl=10.0,
            quantity=1.0,
            trade_id="t1",
            source="paper_or_backtest",
        )
        report = aph.build_report([record], skipped=0, duplicates=0, min_group_trades=1, top=1)
        markdown = aph.render_markdown(report)

        self.assertIn("Favor methods: ta_best", markdown)
        self.assertIn('Source counts: {"paper_or_backtest": 1}', markdown)
        self.assertIn("Biggest Gains By method", markdown)
        self.assertIn("| ta_best | 1 | 5.00% | 5.00% | 100.00% | 10.0000 |", markdown)

    def test_source_filter_separates_live_fills_from_zero_events(self) -> None:
        path = self.write_ndjson(
            [
                {
                    "trade_id": "live-zero",
                    "timestamp": "2026-06-30T00:00:00Z",
                    "symbol": "SOLUSDT",
                    "side": "OPEN",
                    "method": "live",
                    "pnlPercent": 0,
                    "pnl": 0,
                    "quantity": 0,
                },
                {
                    "trade_id": "live-fill",
                    "timestamp": "2026-06-30T00:01:00Z",
                    "symbol": "SOLUSDT",
                    "side": "OPEN",
                    "method": "live",
                    "pnlPercent": 0.01,
                    "pnl": 1.0,
                    "quantity": 2.0,
                },
                {
                    "trade_id": "backtest",
                    "timestamp": "2020-06-30T00:00:00Z",
                    "symbol": "BNBUSDT",
                    "side": "LONG",
                    "method": "01",
                    "pnlPercent": -0.01,
                    "pnl": -1.0,
                    "quantity": 1.0,
                },
            ]
        )

        records, skipped, duplicates = aph.load_trades(path, source_filter="live-fill")
        report = aph.build_report(records, skipped, duplicates, min_group_trades=1, top=3, source_filter="live-fill")

        self.assertEqual(report["summary"]["trades"], 1)
        self.assertEqual(report["summary"]["sourceCounts"], {"live_fill": 1})
        self.assertEqual(report["summary"]["totalPnl"], 1.0)


if __name__ == "__main__":
    unittest.main()
