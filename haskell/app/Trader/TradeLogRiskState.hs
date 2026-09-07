{-# LANGUAGE OverloadedStrings #-}

module Trader.TradeLogRiskState (
    LiveTradeEvent (..),
    TradeLogRiskState,
    TradeLogRiskStateInput (..),
    encodeLiveTradeEvent,
    liveTradeEventForTransition,
    mkTradeLogRiskState,
    tradeLogRiskStateSchemaVersion,
) where

import Data.Aeson (ToJSON (..), object, (.=))
import Data.Aeson.Encoding (encodingToLazyByteString, pair, pairs)
import qualified Data.ByteString.Lazy as BL
import Data.Int (Int64)
import Data.Maybe (isJust, isNothing)
import Data.Text (Text)
import qualified Data.Text as T

{- | Version of live trade-event records that carry a native risk snapshot.
Backtest closed-trade records retain their existing v1.1 semantics.
-}
tradeLogRiskStateSchemaVersion :: Text
tradeLogRiskStateSchemaVersion = "1.2"

data LiveTradeEvent = LiveTradeEvent
    { lteTimestamp :: !Text
    , lteSymbol :: !Text
    , lteEventType :: !Text
    , ltePrice :: !Double
    , lteTradeId :: !Text
    , lteCloseReason :: !(Maybe Text)
    , lteRiskState :: !TradeLogRiskState
    }
    deriving (Eq, Show)

{- | Preserve the historical event-row boundary while carrying the canonical
close cause on a completed transition to flat. Reversals do not currently
emit a separate event row and remain outside this observability change.
-}
liveTradeEventForTransition :: Int -> Int -> Maybe Text -> Maybe (Text, Maybe Text)
liveTradeEventForTransition previousPosition nextPosition closeReason
    | previousPosition == 0 && nextPosition /= 0 = Just ("OPEN", Nothing)
    | previousPosition /= 0 && nextPosition == 0 = Just ("CLOSE", closeReason)
    | otherwise = Nothing

{- | Raw inputs captured at the same pre-execution decision boundary used by
the live risk halt. The smart constructor prevents non-finite JSON numbers
from escaping into the durable NDJSON record.
-}
data TradeLogRiskStateInput = TradeLogRiskStateInput
    { tlrsiAsOfMs :: !Int64
    , tlrsiMarketEventTimeMs :: !Int64
    , tlrsiEquity :: !Double
    , tlrsiPeakEquity :: !Double
    , tlrsiDayKey :: !Int64
    , tlrsiDayStartEquity :: !Double
    , tlrsiWeekKey :: !Int64
    , tlrsiWeekStartEquity :: !Double
    , tlrsiDrawdown :: !Double
    , tlrsiDailyLoss :: !Double
    , tlrsiWeeklyLoss :: !Double
    , tlrsiExpectancy :: !(Maybe Double)
    , tlrsiExpectancyLookback :: !Int
    , tlrsiExpectancyObservations :: !Int
    , tlrsiExpectancyRequired :: !Bool
    , tlrsiHaltReason :: !(Maybe Text)
    }
    deriving (Eq, Show)

data TradeLogRiskState = TradeLogRiskState
    { tlrsAsOfMs :: !Int64
    , tlrsMarketEventTimeMs :: !Int64
    , tlrsEquity :: !(Maybe Double)
    , tlrsPeakEquity :: !(Maybe Double)
    , tlrsDayKey :: !Int64
    , tlrsDayStartEquity :: !(Maybe Double)
    , tlrsWeekKey :: !Int64
    , tlrsWeekStartEquity :: !(Maybe Double)
    , tlrsDrawdown :: !(Maybe Double)
    , tlrsDailyLoss :: !(Maybe Double)
    , tlrsWeeklyLoss :: !(Maybe Double)
    , tlrsExpectancy :: !(Maybe Double)
    , tlrsExpectancyLookback :: !Int
    , tlrsExpectancyObservations :: !Int
    , tlrsExpectancyRequired :: !Bool
    , tlrsHaltReason :: !(Maybe Text)
    , tlrsFinite :: !Bool
    , tlrsValid :: !Bool
    }
    deriving (Eq, Show)

mkTradeLogRiskState :: TradeLogRiskStateInput -> TradeLogRiskState
mkTradeLogRiskState input =
    let finite value = not (isNaN value || isInfinite value)
        nonNegative value = finite value && value >= 0
        positive value = finite value && value > 0
        safeWhen predicate value = if predicate value then Just value else Nothing
        expectancyFinite = maybe True finite (tlrsiExpectancy input)
        allFinite = all finite requiredValues && expectancyFinite
        lookback = max 0 (tlrsiExpectancyLookback input)
        observations = max 0 (min lookback (tlrsiExpectancyObservations input))
        requiredValues =
            [ tlrsiEquity input
            , tlrsiPeakEquity input
            , tlrsiDayStartEquity input
            , tlrsiWeekStartEquity input
            , tlrsiDrawdown input
            , tlrsiDailyLoss input
            , tlrsiWeeklyLoss input
            ]
        domainsValid =
            and
                [ tlrsiAsOfMs input >= 0
                , tlrsiMarketEventTimeMs input >= 0
                , nonNegative (tlrsiEquity input)
                , positive (tlrsiPeakEquity input)
                , tlrsiDayKey input >= 0
                , positive (tlrsiDayStartEquity input)
                , tlrsiWeekKey input >= 0
                , positive (tlrsiWeekStartEquity input)
                , nonNegative (tlrsiDrawdown input)
                , nonNegative (tlrsiDailyLoss input)
                , nonNegative (tlrsiWeeklyLoss input)
                , tlrsiExpectancyLookback input >= 0
                , tlrsiExpectancyObservations input >= 0
                , tlrsiExpectancyObservations input <= lookback
                , isNothing (tlrsiExpectancy input) || observations > 0
                , not (tlrsiExpectancyRequired input) || (lookback > 0 && isJust (tlrsiExpectancy input))
                ]
     in TradeLogRiskState
            { tlrsAsOfMs = tlrsiAsOfMs input
            , tlrsMarketEventTimeMs = tlrsiMarketEventTimeMs input
            , tlrsEquity = safeWhen nonNegative (tlrsiEquity input)
            , tlrsPeakEquity = safeWhen positive (tlrsiPeakEquity input)
            , tlrsDayKey = tlrsiDayKey input
            , tlrsDayStartEquity = safeWhen positive (tlrsiDayStartEquity input)
            , tlrsWeekKey = tlrsiWeekKey input
            , tlrsWeekStartEquity = safeWhen positive (tlrsiWeekStartEquity input)
            , tlrsDrawdown = safeWhen nonNegative (tlrsiDrawdown input)
            , tlrsDailyLoss = safeWhen nonNegative (tlrsiDailyLoss input)
            , tlrsWeeklyLoss = safeWhen nonNegative (tlrsiWeeklyLoss input)
            , tlrsExpectancy = tlrsiExpectancy input >>= safeWhen finite
            , tlrsExpectancyLookback = lookback
            , tlrsExpectancyObservations = observations
            , tlrsExpectancyRequired = tlrsiExpectancyRequired input
            , tlrsHaltReason = tlrsiHaltReason input
            , tlrsFinite = allFinite
            , tlrsValid = allFinite && domainsValid
            }

instance ToJSON TradeLogRiskState where
    toJSON snapshot =
        object
            [ "phase" .= ("pre_execution_decision" :: Text)
            , "asOfMs" .= tlrsAsOfMs snapshot
            , "marketEventTimeMs" .= tlrsMarketEventTimeMs snapshot
            , "equity" .= tlrsEquity snapshot
            , "peakEquity" .= tlrsPeakEquity snapshot
            , "dayKey" .= tlrsDayKey snapshot
            , "dayStartEquity" .= tlrsDayStartEquity snapshot
            , "weekKey" .= tlrsWeekKey snapshot
            , "weekStartEquity" .= tlrsWeekStartEquity snapshot
            , "drawdown" .= tlrsDrawdown snapshot
            , "dailyLoss" .= tlrsDailyLoss snapshot
            , "weeklyLoss" .= tlrsWeeklyLoss snapshot
            , "expectancy" .= tlrsExpectancy snapshot
            , "expectancyLookback" .= tlrsExpectancyLookback snapshot
            , "expectancyObservations" .= tlrsExpectancyObservations snapshot
            , "expectancyRequired" .= tlrsExpectancyRequired snapshot
            , "expectancyAvailable" .= isJust (tlrsExpectancy snapshot)
            , "haltReason" .= tlrsHaltReason snapshot
            , "finite" .= tlrsFinite snapshot
            , "valid" .= tlrsValid snapshot
            ]

{- | Encode the live OPEN/CLOSE event without changing any legacy v1.1 field.
The v1.2 extension adds close-cause and risk-state aliases plus version markers.
-}
encodeLiveTradeEvent :: LiveTradeEvent -> BL.ByteString
encodeLiveTradeEvent event =
    encodingToLazyByteString $
        pairs $
            pair "timestamp" (toEncoding (lteTimestamp event))
                <> pair "symbol" (toEncoding (lteSymbol event))
                <> pair "side" (toEncoding (lteEventType event))
                <> pair "entryPrice" (toEncoding (ltePrice event))
                <> pair "exitPrice" (toEncoding (ltePrice event))
                <> pair "quantity" (toEncoding (0.0 :: Double))
                <> pair "pnl" (toEncoding (0.0 :: Double))
                <> pair "pnlPercent" (toEncoding (0.0 :: Double))
                <> pair "fees" (toEncoding (0.0 :: Double))
                <> pair "method" (toEncoding ("live" :: Text))
                <> pair "volConfGate" (toEncoding ("live" :: Text))
                <> pair "exitReason" (toEncoding (lteEventType event))
                <> pair "entry_price" (toEncoding (showText (ltePrice event)))
                <> pair "exit_price" (toEncoding (showText (ltePrice event)))
                <> pair "quantity_text" (toEncoding ("0.0" :: Text))
                <> pair "pnl_quote" (toEncoding ("0.0" :: Text))
                <> pair "pnl_pct" (toEncoding ("0.0" :: Text))
                <> pair "fee_quote" (toEncoding ("0.0" :: Text))
                <> pair "signal_method" (toEncoding ("live" :: Text))
                <> pair "vol_conf_gate" (toEncoding ("live" :: Text))
                <> pair "regime_filter" (toEncoding (Nothing :: Maybe Text))
                <> pair "slippage_estimate" (toEncoding (Nothing :: Maybe Text))
                <> pair "latency_ms" (toEncoding (Nothing :: Maybe Text))
                <> pair "trade_id" (toEncoding (lteTradeId event))
                <> pair "closeReason" (toEncoding (lteCloseReason event))
                <> pair "close_reason" (toEncoding (lteCloseReason event))
                <> pair "riskState" (toEncoding (lteRiskState event))
                <> pair "risk_state" (toEncoding (lteRiskState event))
                <> pair "schemaVersion" (toEncoding tradeLogRiskStateSchemaVersion)
                <> pair "schema_version" (toEncoding tradeLogRiskStateSchemaVersion)
  where
    showText = T.pack . show
