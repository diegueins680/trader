import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_bot_day


class ReviewBotDayTest(unittest.TestCase):
    def write_snapshot(self, tenant_dir: Path, symbol: str, status: dict) -> None:
        payload = {"savedAtMs": status["updatedAtMs"], "status": status}
        (tenant_dir / f"bot-state-{symbol}.json").write_text(json.dumps(payload))

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


if __name__ == "__main__":
    unittest.main()
