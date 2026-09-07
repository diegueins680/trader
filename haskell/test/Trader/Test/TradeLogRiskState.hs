{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.TradeLogRiskState (
    tradeLogRiskStateSuite,
) where

import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.List (isInfixOf)
import Data.String (fromString)

import Trader.TradeLogRiskState (
    LiveTradeEvent (..),
    TradeLogRiskStateInput (..),
    encodeLiveTradeEvent,
    liveTradeEventForTransition,
    mkTradeLogRiskState,
    tradeLogRiskStateSchemaVersion,
 )

tradeLogRiskStateSuite :: [(String, IO ())]
tradeLogRiskStateSuite =
    [ ("schema v1.2 risk snapshot preserves exact finite decision inputs", testFiniteSnapshot)
    , ("non-finite risk evidence serializes absent and marks the snapshot invalid", testNonFiniteSnapshot)
    , ("required unavailable expectancy remains absent and invalid", testUnavailableExpectancy)
    , ("live transition classification preserves the canonical close cause", testLiveTransitionClassification)
    , ("live v1.2 records preserve every legacy v1.1 field and both risk aliases", testLiveEventCompatibility)
    ]

testFiniteSnapshot :: IO ()
testFiniteSnapshot = do
    expectEq "live risk-state schema version" "1.2" tradeLogRiskStateSchemaVersion
    let value = Aeson.toJSON (mkTradeLogRiskState finiteInput)
    expectField "phase" (Aeson.String "pre_execution_decision") value
    expectField "asOfMs" (Aeson.toJSON (2000 :: Integer)) value
    expectField "marketEventTimeMs" (Aeson.toJSON (1500 :: Integer)) value
    expectField "equity" (Aeson.toJSON (0.94 :: Double)) value
    expectField "peakEquity" (Aeson.toJSON (1.0 :: Double)) value
    expectField "drawdown" (Aeson.toJSON (0.06 :: Double)) value
    expectField "dailyLoss" (Aeson.toJSON (0.04 :: Double)) value
    expectField "weeklyLoss" (Aeson.toJSON (0.03 :: Double)) value
    expectField "expectancy" (Aeson.toJSON ((-0.002) :: Double)) value
    expectField "expectancyAvailable" (Aeson.Bool True) value
    expectField "haltReason" (Aeson.String "MAX_DRAWDOWN") value
    expectField "finite" (Aeson.Bool True) value
    expectField "valid" (Aeson.Bool True) value

testNonFiniteSnapshot :: IO ()
testNonFiniteSnapshot = do
    let value = Aeson.toJSON (mkTradeLogRiskState finiteInput{tlrsiDrawdown = 0 / 0, tlrsiExpectancy = Just (1 / 0)})
        encoded = BL8.unpack (Aeson.encode value)
    expectField "drawdown" Aeson.Null value
    expectField "expectancy" Aeson.Null value
    expectField "expectancyAvailable" (Aeson.Bool False) value
    expectField "finite" (Aeson.Bool False) value
    expectField "valid" (Aeson.Bool False) value
    expectEq "serialized snapshot contains no NaN token" False ("NaN" `isInfixOf` encoded)
    expectEq "serialized snapshot contains no Infinity token" False ("Infinity" `isInfixOf` encoded)

testUnavailableExpectancy :: IO ()
testUnavailableExpectancy = do
    let value =
            Aeson.toJSON
                ( mkTradeLogRiskState
                    finiteInput
                        { tlrsiExpectancy = Nothing
                        , tlrsiExpectancyObservations = 3
                        }
                )
    expectField "expectancy" Aeson.Null value
    expectField "expectancyAvailable" (Aeson.Bool False) value
    expectField "finite" (Aeson.Bool True) value
    expectField "valid" (Aeson.Bool False) value

testLiveTransitionClassification :: IO ()
testLiveTransitionClassification = do
    expectEq "flat to long is an open without close cause" (Just ("OPEN", Nothing)) (liveTradeEventForTransition 0 1 (Just "SIGNAL"))
    expectEq "short to flat carries its close cause" (Just ("CLOSE", Just "TRAILING_STOP")) (liveTradeEventForTransition (-1) 0 (Just "TRAILING_STOP"))
    expectEq "unchanged exposure emits no event" Nothing (liveTradeEventForTransition 1 1 (Just "SIGNAL"))
    expectEq "reversal retains legacy no-event behavior" Nothing (liveTradeEventForTransition 1 (-1) (Just "SIGNAL"))

testLiveEventCompatibility :: IO ()
testLiveEventCompatibility = do
    let riskState = mkTradeLogRiskState finiteInput
        encoded =
            encodeLiveTradeEvent
                LiveTradeEvent
                    { lteTimestamp = "2026-09-07T04:30:00Z"
                    , lteSymbol = "BTCUSDT"
                    , lteEventType = "CLOSE"
                    , ltePrice = 100.25
                    , lteTradeId = "trade-1"
                    , lteCloseReason = Just "TRAILING_STOP"
                    , lteRiskState = riskState
                    }
    value <-
        case Aeson.eitherDecode encoded of
            Left err -> error ("live event did not decode: " ++ err)
            Right decoded -> pure decoded
    let legacyFields =
            [ "timestamp"
            , "symbol"
            , "side"
            , "entryPrice"
            , "exitPrice"
            , "quantity"
            , "pnl"
            , "pnlPercent"
            , "fees"
            , "method"
            , "volConfGate"
            , "exitReason"
            , "entry_price"
            , "exit_price"
            , "quantity_text"
            , "pnl_quote"
            , "pnl_pct"
            , "fee_quote"
            , "signal_method"
            , "vol_conf_gate"
            , "regime_filter"
            , "slippage_estimate"
            , "latency_ms"
            , "trade_id"
            ]
    mapM_ (`expectPresent` value) legacyFields
    expectField "schemaVersion" (Aeson.String "1.2") value
    expectField "schema_version" (Aeson.String "1.2") value
    expectField "closeReason" (Aeson.String "TRAILING_STOP") value
    expectField "close_reason" (Aeson.String "TRAILING_STOP") value
    expectField "riskState" (Aeson.toJSON riskState) value
    expectField "risk_state" (Aeson.toJSON riskState) value
    let openEncoded =
            encodeLiveTradeEvent
                LiveTradeEvent
                    { lteTimestamp = "2026-09-07T04:30:00Z"
                    , lteSymbol = "BTCUSDT"
                    , lteEventType = "OPEN"
                    , ltePrice = 100.25
                    , lteTradeId = "trade-2"
                    , lteCloseReason = Nothing
                    , lteRiskState = riskState
                    }
    openValue <-
        case Aeson.eitherDecode openEncoded of
            Left err -> error ("open event did not decode: " ++ err)
            Right decoded -> pure decoded
    expectField "closeReason" Aeson.Null openValue
    expectField "close_reason" Aeson.Null openValue

finiteInput :: TradeLogRiskStateInput
finiteInput =
    TradeLogRiskStateInput
        { tlrsiAsOfMs = 2000
        , tlrsiMarketEventTimeMs = 1500
        , tlrsiEquity = 0.94
        , tlrsiPeakEquity = 1.0
        , tlrsiDayKey = 10
        , tlrsiDayStartEquity = 0.98
        , tlrsiWeekKey = 2
        , tlrsiWeekStartEquity = 0.97
        , tlrsiDrawdown = 0.06
        , tlrsiDailyLoss = 0.04
        , tlrsiWeeklyLoss = 0.03
        , tlrsiExpectancy = Just (-0.002)
        , tlrsiExpectancyLookback = 20
        , tlrsiExpectancyObservations = 20
        , tlrsiExpectancyRequired = True
        , tlrsiHaltReason = Just "MAX_DRAWDOWN"
        }

expectField :: String -> Aeson.Value -> Aeson.Value -> IO ()
expectField name expected value =
    case value of
        Aeson.Object obj ->
            case KM.lookup (fromString name) obj of
                Just actual -> expectEq name expected actual
                Nothing -> error (name ++ ": field missing")
        _ -> error "risk snapshot did not encode as an object"

expectPresent :: String -> Aeson.Value -> IO ()
expectPresent name value =
    case value of
        Aeson.Object obj ->
            case KM.lookup (fromString name) obj of
                Just _ -> pure ()
                Nothing -> error (name ++ ": legacy field missing")
        _ -> error "live event did not encode as an object"

expectEq :: (Eq a, Show a) => String -> a -> a -> IO ()
expectEq label expected actual =
    if expected == actual
        then pure ()
        else error (label ++ ": expected " ++ show expected ++ ", got " ++ show actual)
