import json
import math
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_bot_day


class ReviewBotDayTest(unittest.TestCase):
    def write_snapshot(self, tenant_dir: Path, symbol: str, status: dict) -> None:
        payload = {"savedAtMs": status["updatedAtMs"], "status": status}
        (tenant_dir / f"bot-state-{symbol}.json").write_text(json.dumps(payload))

    def at_ms(self, iso_text: str) -> int:
        return int(datetime.fromisoformat(iso_text).timestamp() * 1000)

    def prices_from_returns(self, returns: list[float], start: float = 100.0) -> list[float]:
        prices = [start]
        for ret in returns:
            prices.append(prices[-1] * (1.0 + ret))
        return prices

    def test_current_day_default_end_local_uses_now(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "NOWUSDT",
                {
                    "symbol": "NOWUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-20T01:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-20T01:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (LSTM neutral)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "FUTUREUSDT",
                {
                    "symbol": "FUTUREUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-20T23:00:00-05:00"),
                    "prices": [200.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-20T23:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (EDGE_SPIKE)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )

            report = review_bot_day.build_report(
                "2026-04-20",
                "America/Guayaquil",
                tenant_dir,
                now_local=datetime.fromisoformat("2026-04-20T01:30:00-05:00"),
            )

        self.assertEqual(report["windowLocal"]["endExclusive"], "2026-04-20T01:30:00-05:00")
        self.assertEqual(report["summary"]["snapshotsUpdatedAfterWindow"], 1)
        self.assertEqual(report["latestActionCensus"]["eligibleSymbols"], 1)
        self.assertEqual(report["latestActionCensus"]["updatedAfterCutoffSymbols"], ["FUTUREUSDT"])
        self.assertEqual(report["summary"]["staleSnapshotsAtCutoff"], 0)

    def test_historical_day_default_end_local_stays_midnight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "HISTUSDT",
                {
                    "symbol": "HISTUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-19T23:30:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-19T23:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (LSTM neutral)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "AFTERUSDT",
                {
                    "symbol": "AFTERUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-20T00:30:00-05:00"),
                    "prices": [200.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-20T00:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (EDGE_SPIKE)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )

            report = review_bot_day.build_report(
                "2026-04-19",
                "America/Guayaquil",
                tenant_dir,
                now_local=datetime.fromisoformat("2026-04-20T01:30:00-05:00"),
            )

        self.assertEqual(report["windowLocal"]["endExclusive"], "2026-04-20T00:00:00-05:00")
        self.assertEqual(report["summary"]["snapshotsUpdatedAfterWindow"], 1)
        self.assertEqual(report["latestActionCensus"]["eligibleSymbols"], 1)
        self.assertEqual(report["latestActionCensus"]["updatedAfterCutoffSymbols"], ["AFTERUSDT"])

    def test_excludes_adopted_carry_with_prior_order_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "BTCUSDT",
                {
                    "symbol": "BTCUSDT",
                    "interval": "1d",
                    "updatedAtMs": 1775113285353,
                    "prices": [67755.7, 68203.0, 68046.8],
                    "positions": [0, 0, -1],
                    "openTimes": [1774915200000, 1775001600000, 1775088000000],
                    "equityCurve": [1.0, 1.0, 0.999455840452014],
                    "latestSignal": {"volatility": 0.4914676811785376, "regimes": {}},
                    "trades": [],
                    "openTrade": {
                        "entryIndex": 2,
                        "entryEquity": 0.999455840452014,
                        "entryPrice": 68046.8,
                        "holdingPeriods": 0,
                        "side": "short",
                        "size": 0.8640402067633041,
                        "trail": 68046.8,
                    },
                    "orders": [
                        {
                            "atMs": 1775001636514,
                            "index": 1,
                            "opSide": "SELL",
                            "openTime": 1775001600000,
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent. (min size applied).",
                                "quantity": 0.002,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "BTCUSDT",
                            },
                            "price": 68203.0,
                        },
                        {
                            "atMs": 1775088047094,
                            "index": 2,
                            "opSide": "SELL",
                            "openTime": 1775088000000,
                            "order": {
                                "message": "No order: already short.",
                                "sent": False,
                                "symbol": "BTCUSDT",
                            },
                            "price": 68046.8,
                        },
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-01", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["openPositionsEnteredToday"], 0)
        self.assertEqual(report["summary"]["openPositionsCarriedIn"], 1)
        self.assertEqual(report["summary"]["ambiguousOpenPositionOrigins"], 0)
        self.assertEqual(report["summary"]["sameDayOrderEvents"], 1)
        self.assertEqual(report["openPositionsEnteredToday"], [])
        self.assertEqual(report["anomalies"]["ambiguousOpenPositionOrigins"], [])
        carried = report["openPositionsCarriedIn"][0]
        self.assertEqual(carried["symbol"], "BTCUSDT")
        self.assertEqual(carried["provenance"], "prior_day_order_evidence")
        self.assertEqual(carried["supportingOrderAtLocal"], "2026-03-31T19:00:36.514000-05:00")
        self.assertEqual(carried["adoptionEventAtLocal"], "2026-04-01T19:00:47.094000-05:00")
        self.assertEqual(report["orderEvents"][0]["message"], "No order: already short.")

    def test_flags_ambiguous_adoption_without_prior_order_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "ETHUSDT",
                {
                    "symbol": "ETHUSDT",
                    "interval": "1d",
                    "updatedAtMs": 1775113285353,
                    "prices": [2000.0, 2010.0],
                    "positions": [0, 1],
                    "openTimes": [1775001600000, 1775088000000],
                    "equityCurve": [1.0, 1.01],
                    "latestSignal": {"volatility": 0.2, "regimes": {}},
                    "trades": [],
                    "openTrade": {
                        "entryIndex": 1,
                        "entryEquity": 1.0,
                        "entryPrice": 2010.0,
                        "holdingPeriods": 0,
                        "side": "long",
                        "size": 1.0,
                        "trail": 2010.0,
                    },
                    "orders": [
                        {
                            "atMs": 1775088047094,
                            "index": 1,
                            "opSide": "BUY",
                            "openTime": 1775088000000,
                            "order": {
                                "message": "No order: already long.",
                                "sent": False,
                                "symbol": "ETHUSDT",
                            },
                            "price": 2010.0,
                        }
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-01", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["openPositionsEnteredToday"], 0)
        self.assertEqual(report["summary"]["openPositionsCarriedIn"], 0)
        self.assertEqual(report["summary"]["ambiguousOpenPositionOrigins"], 1)
        self.assertEqual(report["openPositionsEnteredToday"], [])
        ambiguous = report["anomalies"]["ambiguousOpenPositionOrigins"][0]
        self.assertEqual(ambiguous["symbol"], "ETHUSDT")
        self.assertEqual(ambiguous["provenance"], "adoption_without_saved_entry_order")
        self.assertEqual(ambiguous["adoptionEventAtLocal"], "2026-04-01T19:00:47.094000-05:00")

    def test_cutoff_reconstructs_open_position_before_later_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "BNBUSDT",
                {
                    "symbol": "BNBUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775199000000,
                    "prices": [100.0, 101.0, 102.0, 103.0],
                    "positions": [0, -1, -1, 0],
                    "openTimes": [1775188800000, 1775192400000, 1775196000000, 1775199600000],
                    "equityCurve": [1.0, 1.0, 0.999, 0.998],
                    "latestSignal": {"volatility": 0.4, "regimes": {"trend": 0.2, "mr": 0.8}},
                    "trades": [
                        {
                            "entryEquity": 1.0,
                            "entryHighVolProb": 0.1,
                            "entryIndex": 1,
                            "exitEquity": 0.998,
                            "exitIndex": 3,
                            "exitReason": "SIGNAL",
                            "holdingPeriods": 2,
                            "return": -0.002,
                        }
                    ],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775192460000,
                            "index": 1,
                            "opSide": "SELL",
                            "openTime": 1775192400000,
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent.",
                                "quantity": 1.0,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "BNBUSDT",
                            },
                            "price": 101.0,
                        },
                        {
                            "atMs": 1775199660000,
                            "index": 3,
                            "opSide": "BUY",
                            "openTime": 1775199600000,
                            "order": {
                                "message": "No order: already flat.",
                                "sent": False,
                                "symbol": "BNBUSDT",
                            },
                            "price": 103.0,
                        },
                    ],
                },
            )

            report = review_bot_day.build_report(
                "2026-04-03",
                "America/Guayaquil",
                tenant_dir,
                end_local_text="2026-04-03T01:30:00-05:00",
            )

        self.assertEqual(report["summary"]["completedTrades"], 0)
        self.assertEqual(report["summary"]["openPositionsEnteredToday"], 1)
        self.assertEqual(report["summary"]["sameDayOrderEvents"], 1)
        self.assertEqual(report["summary"]["ackOnlyOrderEvents"], 1)
        self.assertEqual(report["summary"]["snapshotsUpdatedAfterWindow"], 1)
        open_position = report["openPositionsEnteredToday"][0]
        self.assertEqual(open_position["symbol"], "BNBUSDT")
        self.assertEqual(open_position["entryTimeLocal"], "2026-04-03T00:00:00-05:00")
        self.assertEqual(open_position["currentPrice"], 102.0)
        self.assertAlmostEqual(open_position["markToMarketPct"], -0.1)
        self.assertIsNone(open_position["latestRegimes"])
        self.assertEqual(report["anomalies"]["snapshotsUpdatedAfterWindow"][0]["symbol"], "BNBUSDT")

    def test_directionality_snapshot_flags_mr_range_drift(self) -> None:
        snapshot = review_bot_day.build_directionality_snapshot(
            [100.0, 102.0, 101.0, 103.0, 102.0],
            4,
            {"trend": 0.08102, "mr": 0.82061, "highVol": 0.09837},
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["reason"], "NON_DIRECTIONAL_MR")
        self.assertTrue(snapshot["nonDirectional"])
        self.assertEqual(snapshot["regimeLeader"], "mr")

    def test_directionality_snapshot_flags_chop_veto(self) -> None:
        prices = self.prices_from_returns([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
        snapshot = review_bot_day.build_directionality_snapshot(
            prices,
            len(prices) - 1,
            {"trend": 0.2, "mr": 0.6, "highVol": 0.2},
            requested_side="long",
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["reason"], "NON_DIRECTIONAL_CHOP")
        self.assertTrue(snapshot["nonDirectional"])

    def test_directionality_snapshot_flags_malformed_hysteresis_fail_closed(self) -> None:
        prices = self.prices_from_returns([0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01])
        snapshot = review_bot_day.build_directionality_snapshot(
            prices,
            len(prices) - 1,
            {"trend": 0.6, "mr": 0.2, "highVol": 0.2},
            regime_hysteresis=math.nan,
            requested_side="long",
        )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["reason"], "NON_DIRECTIONAL_MALFORMED")
        self.assertTrue(snapshot["nonDirectional"])

    def test_directionality_snapshot_keeps_additive_monotonic_trend_directional(self) -> None:
        prices = self.prices_from_returns([0.01] * 24)
        snapshot = review_bot_day.build_directionality_snapshot(
            prices,
            len(prices) - 1,
            {"trend": 0.6, "mr": 0.2, "highVol": 0.2},
            requested_side="long",
        )
        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot["reason"])
        self.assertFalse(snapshot["nonDirectional"])

    def test_directionality_snapshot_uses_weak_band_confirmation_when_regimes_are_unavailable(self) -> None:
        weak_band_prices = self.prices_from_returns([0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01])
        blocked_short = review_bot_day.build_directionality_snapshot(
            weak_band_prices,
            len(weak_band_prices) - 1,
            None,
            requested_side="short",
        )
        borderline_trend_prices = self.prices_from_returns([0.01, -0.006, 0.01, -0.006, 0.01, -0.006, 0.01, -0.006])
        admitted_long = review_bot_day.build_directionality_snapshot(
            borderline_trend_prices,
            len(borderline_trend_prices) - 1,
            None,
            requested_side="long",
        )

        self.assertIsNotNone(blocked_short)
        self.assertEqual(blocked_short["reason"], "NON_DIRECTIONAL_WEAK_BAND")
        self.assertTrue(blocked_short["nonDirectional"])
        self.assertIsNotNone(admitted_long)
        self.assertIsNone(admitted_long["reason"])
        self.assertFalse(admitted_long["nonDirectional"])

    def test_report_counts_non_directional_order_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "BNBUSDT",
                {
                    "symbol": "BNBUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775347200000,
                    "prices": [100.0, 102.0, 101.0, 103.0, 102.0],
                    "positions": [0, 0, 0, 0, 0],
                    "openTimes": [
                        1775332800000,
                        1775336400000,
                        1775340000000,
                        1775343600000,
                        1775347200000,
                    ],
                    "equityCurve": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "latestSignal": {"volatility": 0.2, "regimes": {"trend": 0.08102, "mr": 0.82061, "highVol": 0.09837}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775340030000,
                            "index": 4,
                            "opSide": "BUY",
                            "openTime": 1775340000000,
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent.",
                                "quantity": 1.0,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "BNBUSDT",
                            },
                            "price": 102.0,
                        }
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["sameDayOrderEvents"], 1)
        self.assertEqual(report["summary"]["nonDirectionalOrderAttempts"], 1)
        self.assertEqual(report["summary"]["nonDirectionalExitOrFlattenEvents"], 0)
        self.assertEqual(report["summary"]["nonDirectionalUnknownRoleEvents"], 0)
        self.assertEqual(report["orderEvents"][0]["flowRole"], "entry_or_add")
        self.assertEqual(report["orderEvents"][0]["flowRoleEvidence"], "default_without_close_context")
        self.assertEqual(report["orderEvents"][0]["nonDirectionalReason"], "NON_DIRECTIONAL_MR")
        self.assertEqual(report["anomalies"]["nonDirectionalOrderAttempts"][0]["symbol"], "BNBUSDT")

    def test_report_flags_weak_band_short_blocked_by_positive_zscore_on_entry_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            prices = self.prices_from_returns([0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01])
            open_times = [self.at_ms(f"2026-04-04T{hour:02d}:00:00-05:00") for hour in range(len(prices))]
            self.write_snapshot(
                tenant_dir,
                "ADAUSDT",
                {
                    "symbol": "ADAUSDT",
                    "interval": "1h",
                    "updatedAtMs": open_times[-1],
                    "prices": prices,
                    "positions": [0] * len(prices),
                    "openTimes": open_times,
                    "equityCurve": [1.0] * len(prices),
                    "latestSignal": {"volatility": 0.2, "regimes": {"trend": 0.6, "mr": 0.2, "highVol": 0.2}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": open_times[-1] + 30_000,
                            "index": len(prices) - 1,
                            "opSide": "SELL",
                            "openTime": open_times[-1],
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent.",
                                "quantity": 10.0,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "ADAUSDT",
                            },
                            "price": prices[-1],
                        }
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["sameDayOrderEvents"], 1)
        self.assertEqual(report["summary"]["nonDirectionalOrderAttempts"], 1)
        self.assertEqual(report["orderEvents"][0]["flowRole"], "entry_or_add")
        self.assertEqual(report["orderEvents"][0]["nonDirectionalReason"], "NON_DIRECTIONAL_WEAK_BAND")
        self.assertEqual(report["anomalies"]["nonDirectionalOrderAttempts"][0]["symbol"], "ADAUSDT")

    def test_excludes_adopted_close_and_flatten_events_from_non_directional_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "UNIUSDT",
                {
                    "symbol": "UNIUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775505600000,
                    "prices": [3.178, 3.188, 3.157, 3.148, 3.16, 3.162, 3.149],
                    "positions": [0, 0, 0, 1, 1, 0, 0],
                    "openTimes": [
                        1775484000000,
                        1775487600000,
                        1775491200000,
                        1775494800000,
                        1775498400000,
                        1775502000000,
                        1775505600000,
                    ],
                    "equityCurve": [1.0, 1.0, 1.0, 1.0, 1.0001, 1.0002, 1.0002],
                    "latestSignal": {"volatility": 0.2, "regimes": {"trend": 0.06771, "mr": 0.8977, "highVol": 0.03458}},
                    "trades": [
                        {
                            "entryEquity": 1.0,
                            "entryHighVolProb": 0.03458,
                            "entryIndex": 3,
                            "entrySource": "adopted",
                            "exitEquity": 1.0002,
                            "exitIndex": 5,
                            "exitReason": "SIGNAL",
                            "holdingPeriods": 2,
                            "return": 0.0002,
                        }
                    ],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775495817372,
                            "index": 3,
                            "opSide": "SELL",
                            "openTime": 1775494800000,
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent.",
                                "quantity": 25.0,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "UNIUSDT",
                            },
                            "price": 3.148,
                            "directionality": {
                                "efficiency": 0.21973,
                                "highVolProb": 0.03458,
                                "label": "chop",
                                "lookbackBars": 24,
                                "mrProb": 0.8977,
                                "netReturnPct": 2.00907,
                                "nonDirectional": True,
                                "realizedVolPct": 0.51158,
                                "reason": "NON_DIRECTIONAL_CHOP",
                                "regimeGap": 0.82999,
                                "regimeLeader": "mr",
                                "trendProb": 0.06771,
                                "zScore": 0.81887,
                            },
                        },
                        {
                            "atMs": 1775502051136,
                            "index": 5,
                            "opSide": "SELL",
                            "openTime": 1775502000000,
                            "order": {
                                "message": "No order: already flat.",
                                "sent": False,
                                "symbol": "UNIUSDT",
                            },
                            "price": 3.162,
                            "directionality": {
                                "efficiency": 0.24610,
                                "highVolProb": 0.02908,
                                "label": "chop",
                                "lookbackBars": 24,
                                "mrProb": 0.91494,
                                "netReturnPct": 2.26390,
                                "nonDirectional": True,
                                "realizedVolPct": 0.51195,
                                "reason": "NON_DIRECTIONAL_CHOP",
                                "regimeGap": 0.85896,
                                "regimeLeader": "mr",
                                "trendProb": 0.05597,
                                "zScore": 0.92207,
                            },
                        },
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-06", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["sameDayOrderEvents"], 2)
        self.assertEqual(report["summary"]["nonDirectionalOrderAttempts"], 0)
        self.assertEqual(report["summary"]["nonDirectionalExitOrFlattenEvents"], 2)
        self.assertEqual(report["summary"]["nonDirectionalUnknownRoleEvents"], 0)
        self.assertEqual([event["flowRole"] for event in report["orderEvents"]], ["exit_or_flatten", "exit_or_flatten"])
        self.assertEqual(report["orderEvents"][0]["flowRoleEvidence"], "completed_trade_entry_side")
        self.assertEqual(report["orderEvents"][1]["flowRoleEvidence"], "message_already_flat")
        self.assertEqual(report["anomalies"]["nonDirectionalOrderAttempts"], [])
        self.assertEqual(len(report["anomalies"]["nonDirectionalExitOrFlattenEvents"]), 2)

    def test_classifies_binance_auth_failures_by_order_flow_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "BNBUSDT",
                {
                    "symbol": "BNBUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775347200000,
                    "prices": [600.0, 605.0, 610.0],
                    "positions": [0, 0, 0],
                    "openTimes": [1775336400000, 1775340000000, 1775343600000],
                    "equityCurve": [1.0, 1.0, 1.0],
                    "latestSignal": {"volatility": 0.2, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775340030000,
                            "index": 1,
                            "opSide": "BUY",
                            "openTime": 1775340000000,
                            "order": {
                                "message": "Order failed: futures/positionRisk HTTP 401 / Binance code -2015: Invalid API-key, IP, or permissions for action.",
                                "sent": False,
                                "status": None,
                                "symbol": "BNBUSDT",
                            },
                            "price": 605.0,
                        }
                    ],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "BTCUSDT",
                {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775347200000,
                    "prices": [68000.0, 68100.0, 67950.0],
                    "positions": [0, -1, 0],
                    "openTimes": [1775336400000, 1775340000000, 1775343600000],
                    "equityCurve": [1.0, 1.0, 0.999],
                    "latestSignal": {"volatility": 0.2, "regimes": {}},
                    "trades": [
                        {
                            "entryEquity": 1.0,
                            "entryIndex": 1,
                            "exitEquity": 0.999,
                            "exitIndex": 2,
                            "exitReason": "SIGNAL",
                            "holdingPeriods": 1,
                            "return": -0.001,
                        }
                    ],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775343630000,
                            "index": 2,
                            "opSide": "BUY",
                            "openTime": 1775343600000,
                            "order": {
                                "message": "Order failed: futures/positionRisk HTTP 401 / Binance code -2015: Invalid API-key, IP, or permissions for action.",
                                "sent": False,
                                "status": None,
                                "symbol": "BTCUSDT",
                            },
                            "price": 67950.0,
                        }
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["authFailureOrderEvents"], 2)
        self.assertEqual(report["summary"]["authFailureEntryOrAddEvents"], 1)
        self.assertEqual(report["summary"]["authFailureExitOrFlattenEvents"], 1)
        self.assertEqual(report["summary"]["authFailureUnknownRoleEvents"], 0)
        self.assertEqual([event["authFailure"] for event in report["orderEvents"]], [True, True])
        self.assertEqual(
            [event["authFailureSummary"] for event in report["orderEvents"]],
            [
                "Invalid API-key, IP, or permissions for action.",
                "Invalid API-key, IP, or permissions for action.",
            ],
        )
        self.assertEqual([event["flowRole"] for event in report["orderEvents"]], ["entry_or_add", "exit_or_flatten"])
        self.assertEqual(report["anomalies"]["authFailureEntryOrAddEvents"][0]["symbol"], "BNBUSDT")
        self.assertEqual(report["anomalies"]["authFailureExitOrFlattenEvents"][0]["symbol"], "BTCUSDT")

    def test_marks_adopted_trade_closure_as_carry_in_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "LINKUSDT",
                {
                    "symbol": "LINKUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775347200000,
                    "prices": [8.60, 8.65, 8.70],
                    "positions": [0, -1, 0],
                    "openTimes": [1775336400000, 1775340000000, 1775343600000],
                    "equityCurve": [1.0, 1.0, 0.999],
                    "latestSignal": {"volatility": 0.2, "regimes": {"mr": 0.8}},
                    "trades": [
                        {
                            "entryEquity": 1.0,
                            "entryHighVolProb": 0.05,
                            "entryIndex": 1,
                            "entrySource": "adopted",
                            "exitEquity": 0.999,
                            "exitIndex": 2,
                            "exitReason": "SIGNAL",
                            "holdingPeriods": 1,
                            "return": -0.001,
                        }
                    ],
                    "openTrade": None,
                    "orders": [
                        {
                            "atMs": 1775340030000,
                            "index": 1,
                            "opSide": "BUY",
                            "openTime": 1775340000000,
                            "order": {
                                "executedQty": 0,
                                "message": "Order sent.",
                                "quantity": 5.0,
                                "sent": True,
                                "status": "NEW",
                                "symbol": "LINKUSDT",
                            },
                            "price": 8.65,
                        }
                    ],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["completedTrades"], 1)
        self.assertEqual(report["completedTrades"][0]["entrySource"], "adopted")
        self.assertEqual(report["completedTrades"][0]["provenance"], "startup_adopted_position")
        self.assertEqual(report["anomalies"]["adoptedTradeClosures"][0]["symbol"], "LINKUSDT")

    def test_classifies_adopted_open_position_as_carried_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "SOLUSDT",
                {
                    "symbol": "SOLUSDT",
                    "interval": "1h",
                    "updatedAtMs": 1775347200000,
                    "prices": [120.0, 121.0],
                    "positions": [0, 1],
                    "openTimes": [1775340000000, 1775343600000],
                    "equityCurve": [1.0, 1.002],
                    "latestSignal": {"volatility": 0.15, "regimes": {"trend": 0.7}},
                    "trades": [],
                    "openTrade": {
                        "entryIndex": 1,
                        "entryEquity": 1.0,
                        "entryHighVolProb": 0.02,
                        "entryPrice": 121.0,
                        "entrySource": "adopted",
                        "holdingPeriods": 0,
                        "partialTaken": False,
                        "side": "long",
                        "size": 1.0,
                        "trail": 121.0,
                    },
                    "orders": [],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["openPositionsEnteredToday"], 0)
        self.assertEqual(report["summary"]["openPositionsCarriedIn"], 1)
        carried = report["openPositionsCarriedIn"][0]
        self.assertEqual(carried["symbol"], "SOLUSDT")
        self.assertEqual(carried["entrySource"], "adopted")
        self.assertEqual(carried["provenance"], "startup_adopted_position")

    def test_cutoff_freshness_flags_stale_snapshot_by_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            updated_at_ms = self.at_ms("2026-04-04T12:00:00-05:00")
            self.write_snapshot(
                tenant_dir,
                "FASTUSDT",
                {
                    "symbol": "FASTUSDT",
                    "interval": "5m",
                    "updatedAtMs": updated_at_ms,
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T11:55:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (LSTM neutral)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "SLOWUSDT",
                {
                    "symbol": "SLOWUSDT",
                    "interval": "1d",
                    "updatedAtMs": updated_at_ms,
                    "prices": [200.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T00:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (TREND_FILTER)", "volatility": 0.1, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["staleSnapshotsAtCutoff"], 1)
        self.assertEqual(report["cutoffFreshness"]["availableSymbols"], 2)
        self.assertEqual(report["cutoffFreshness"]["staleSymbols"], 1)
        stale = report["anomalies"]["staleSnapshotsAtCutoff"][0]
        self.assertEqual(stale["symbol"], "FASTUSDT")
        self.assertTrue(stale["staleAtCutoff"])
        self.assertEqual(stale["freshnessBudgetMs"], 5 * 60 * 1000)
        slow = next(row for row in report["cutoffFreshness"]["symbols"] if row["symbol"] == "SLOWUSDT")
        self.assertFalse(slow["staleAtCutoff"])
        markdown = review_bot_day.render_markdown(report)
        self.assertIn("## Cutoff Freshness", markdown)
        self.assertIn("`FASTUSDT` `5m` snapshot updated", markdown)

    def test_action_census_uses_latest_pre_cutoff_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "BTCUSDT",
                {
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T10:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T10:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (LSTM neutral)", "volatility": 0.2, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "ETHUSDT",
                {
                    "symbol": "ETHUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T11:00:00-05:00"),
                    "prices": [200.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T11:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (EDGE_SPIKE)", "volatility": 0.2, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "SOLUSDT",
                {
                    "symbol": "SOLUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-05T00:30:00-05:00"),
                    "prices": [300.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-05T00:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {"action": "HOLD (LSTM neutral)", "volatility": 0.2, "regimes": {}},
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(
            report["latestActionCensus"]["counts"],
            [
                {"action": "HOLD (EDGE_SPIKE)", "count": 1},
                {"action": "HOLD (LSTM neutral)", "count": 1},
            ],
        )
        self.assertEqual(report["latestActionCensus"]["eligibleSymbols"], 2)
        self.assertEqual(report["latestActionCensus"]["updatedAfterCutoffSymbols"], ["SOLUSDT"])
        markdown = review_bot_day.render_markdown(report)
        self.assertIn("## Latest Action Census", markdown)
        self.assertIn("`HOLD (EDGE_SPIKE)=1`", markdown)
        self.assertIn("`HOLD (LSTM neutral)=1`", markdown)
        self.assertIn("Omitted post-cutoff snapshots from the action census for: `SOLUSDT`", markdown)

    def test_cutoff_latest_signal_audit_fails_closed_on_negative_threshold(self) -> None:
        snapshot = review_bot_day.BotSnapshot(
            path=Path("bot-state-NEGUSDT.json"),
            status={
                "symbol": "NEGUSDT",
                "interval": "1h",
                "updatedAtMs": self.at_ms("2026-04-04T10:00:00-05:00"),
                "latestSignal": {
                    "action": "HOLD (TREND)",
                    "kalmanReturn": 0.04,
                    "openThreshold": -0.01,
                },
            },
        )

        row = review_bot_day.build_cutoff_latest_signal_audit_row(
            snapshot,
            snapshot.updated_at_ms,
            ZoneInfo("America/Guayaquil"),
        )

        self.assertIsNotNone(row)
        self.assertTrue(row["usableEdgeSample"])
        self.assertFalse(row["usableOpenThreshold"])
        self.assertFalse(row["clearsOpenThreshold"])
        self.assertFalse(row["clearsHeadroomFloor"])
        self.assertIsNone(row["thresholdRatio"])
        self.assertIsNone(row["headroomRatio"])

    def test_cutoff_latest_signal_audit_recovers_legacy_missing_regime_malformed_directionality(self) -> None:
        prices = self.prices_from_returns([0.018, 0.018, 0.018, -0.01, -0.01, -0.01, 0.018, 0.018, -0.01, -0.01])
        snapshot = review_bot_day.BotSnapshot(
            path=Path("bot-state-LTCUSDT.json"),
            status={
                "symbol": "LTCUSDT",
                "interval": "4h",
                "updatedAtMs": self.at_ms("2026-04-25T00:00:55-05:00"),
                "prices": prices,
                "latestSignal": {
                    "action": "HOLD (TREND)",
                    "currentPrice": prices[-1],
                    "methodNext": prices[-1] * 0.95,
                    "openThreshold": 0.01,
                    "lstmDirection": "DOWN",
                    "directionality": {"nonDirectional": True, "reason": "NON_DIRECTIONAL_MALFORMED"},
                    "regimes": None,
                },
            },
        )

        row = review_bot_day.build_cutoff_latest_signal_audit_row(
            snapshot,
            snapshot.updated_at_ms,
            ZoneInfo("America/Guayaquil"),
        )

        self.assertIsNotNone(row)
        self.assertEqual(row["directionalityReason"], "NON_DIRECTIONAL_WEAK_BAND")
        self.assertFalse(row["malformedDirectionality"])

    def test_cutoff_latest_signal_audit_counts_measurable_edges_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tenant_dir = Path(tmpdir)
            self.write_snapshot(
                tenant_dir,
                "KALMUSDT",
                {
                    "symbol": "KALMUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T10:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T10:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (Kalman neutral)",
                        "currentPrice": 100.0,
                        "kalmanReturn": 0.03,
                        "methodNext": 104.0,
                        "openThreshold": 0.02,
                        "regimes": {},
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "MISSUSDT",
                {
                    "symbol": "MISSUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T11:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T11:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (LSTM neutral)",
                        "currentPrice": 100.0,
                        "kalmanReturn": None,
                        "methodNext": 101.0,
                        "openThreshold": 0.02,
                        "regimes": {},
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "MALFUSDT",
                {
                    "symbol": "MALFUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T12:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T12:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (NON_DIRECTIONAL_MALFORMED)",
                        "currentPrice": 100.0,
                        "kalmanReturn": None,
                        "methodNext": 110.0,
                        "openThreshold": 0.05,
                        "directionality": {"nonDirectional": True, "reason": "NON_DIRECTIONAL_MALFORMED"},
                        "regimes": None,
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "BADREGUSDT",
                {
                    "symbol": "BADREGUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T13:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T13:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (TREND)",
                        "currentPrice": 100.0,
                        "kalmanReturn": None,
                        "methodNext": 112.0,
                        "openThreshold": 0.05,
                        "regimes": {"trend": 0.8, "mr": 0.3, "highVol": -0.1},
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "SPIKEUSDT",
                {
                    "symbol": "SPIKEUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-04T14:00:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-04T14:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (LSTM neutral)",
                        "currentPrice": 100.0,
                        "kalmanReturn": None,
                        "methodNext": 200.0,
                        "openThreshold": 0.02,
                        "regimes": {},
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )
            self.write_snapshot(
                tenant_dir,
                "AFTERUSDT",
                {
                    "symbol": "AFTERUSDT",
                    "interval": "1h",
                    "updatedAtMs": self.at_ms("2026-04-05T00:30:00-05:00"),
                    "prices": [100.0],
                    "positions": [0],
                    "openTimes": [self.at_ms("2026-04-05T00:00:00-05:00")],
                    "equityCurve": [1.0],
                    "latestSignal": {
                        "action": "HOLD (TREND)",
                        "currentPrice": 100.0,
                        "kalmanReturn": 0.5,
                        "methodNext": 150.0,
                        "openThreshold": 0.01,
                        "regimes": {},
                    },
                    "trades": [],
                    "openTrade": None,
                    "orders": [],
                },
            )

            report = review_bot_day.build_report("2026-04-04", "America/Guayaquil", tenant_dir)

        self.assertEqual(report["summary"]["cutoffSignalsWithMeasurableEdge"], 4)
        self.assertEqual(report["summary"]["cutoffSignalsAboveOpenThreshold"], 3)
        self.assertEqual(report["summary"]["cutoffSignalsAboveHeadroomFloor"], 3)
        self.assertEqual(report["summary"]["cutoffSignalsWithMalformedDirectionality"], 1)
        self.assertEqual(report["summary"]["cutoffSignalsWithMalformedLatestSignalRegimes"], 1)

        audit = report["cutoffLatestSignalAudit"]
        self.assertEqual(audit["eligibleSymbols"], 5)
        self.assertEqual(
            audit["counts"],
            {
                "withMeasurableEdge": 4,
                "withoutMeasurableEdge": 1,
                "aboveOpenThreshold": 3,
                "aboveHeadroomFloor": 3,
                "withMalformedDirectionality": 1,
                "withMalformedLatestSignalRegimes": 1,
            },
        )

        rows = {row["symbol"]: row for row in audit["symbols"]}
        self.assertEqual(rows["KALMUSDT"]["edgeSource"], "latestSignal.kalmanReturn")
        self.assertAlmostEqual(rows["KALMUSDT"]["thresholdRatio"], 1.5)
        self.assertAlmostEqual(rows["KALMUSDT"]["headroomRatio"], 1.0)
        self.assertTrue(rows["KALMUSDT"]["clearsHeadroomFloor"])
        self.assertEqual(rows["MISSUSDT"]["edgeSource"], "inferred:abs(methodNext/currentPrice-1)")
        self.assertFalse(rows["MISSUSDT"]["clearsOpenThreshold"])
        self.assertFalse(rows["MISSUSDT"]["clearsHeadroomFloor"])
        self.assertTrue(rows["MALFUSDT"]["malformedDirectionality"])
        self.assertEqual(rows["MALFUSDT"]["actionReason"], "NON_DIRECTIONAL_MALFORMED")
        self.assertTrue(rows["BADREGUSDT"]["malformedLatestSignalRegimes"])
        self.assertEqual(rows["BADREGUSDT"]["actionReason"], "TREND")
        self.assertFalse(rows["SPIKEUSDT"]["usableEdgeSample"])
        self.assertIsNone(rows["SPIKEUSDT"]["edgeSource"])
        self.assertEqual([row["symbol"] for row in audit["strongestCandidates"][:3]], ["BADREGUSDT", "MALFUSDT", "KALMUSDT"])
        self.assertEqual(audit["blockedSignals"][0]["symbol"], "MALFUSDT")
        self.assertEqual(audit["blockedSignals"][1]["symbol"], "BADREGUSDT")

        markdown = review_bot_day.render_markdown(report)
        self.assertIn("## Cutoff Latest-Signal Audit", markdown)
        self.assertIn("`usable_edge=4`", markdown)
        self.assertIn("`above_headroom=3`", markdown)
        self.assertIn("`malformed_directionality=1`", markdown)
        self.assertIn("`BADREGUSDT` `1h` action `HOLD (TREND)`", markdown)
        self.assertIn("`MALFUSDT` `1h` action `HOLD (NON_DIRECTIONAL_MALFORMED)`", markdown)


if __name__ == "__main__":
    unittest.main()
