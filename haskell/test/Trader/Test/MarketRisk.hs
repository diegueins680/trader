{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.MarketRisk (marketRiskSuite) where

import Control.Monad (unless)
import Data.Aeson (eitherDecode)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.KeyMap as KM
import qualified Data.ByteString.Lazy.Char8 as BL8
import Data.Int (Int64)
import Data.List (isInfixOf)
import Trader.Binance (
    DepthLevel (..),
    FuturesAdlRiskSnapshot (..),
    FuturesMarketSnapshot (..),
    FuturesOpenInterestSnapshot (..),
    FuturesPremiumSnapshot (..),
    Kline (..),
    OrderBookSnapshot (..),
 )
import Trader.MarketRisk (
    MarketRiskAdmissionOutcome (..),
    MarketRiskConfig (..),
    MarketRiskDecision (..),
    MarketRiskInput (..),
    defaultMarketRiskConfig,
    marketRiskDecision,
    marketRiskSummary,
 )

marketRiskSuite :: [(String, IO ())]
marketRiskSuite =
    [ ("allows liquid entry with edge headroom", testAllowsLiquidEntry)
    , ("blocks insufficient depth", testBlocksInsufficientDepth)
    , ("blocks excessive spread", testBlocksExcessiveSpread)
    , ("blocks requested-size impact", testBlocksExcessiveImpact)
    , ("blocks adverse funding that consumes edge", testBlocksAdverseFunding)
    , ("blocks missing or counter-directional forecast edge", testBlocksMissingDirectionalEdge)
    , ("blocks forecast edge below the configured minimum", testBlocksEdgeBelowMinimum)
    , ("measures remaining edge from the live order book", testUsesLiveBookForRemainingEdge)
    , ("blocks high ADL risk", testBlocksHighAdl)
    , ("blocks stale critical feeds", testBlocksStaleFeeds)
    , ("allows bounded venue timestamp skew", testFutureTimestampSkew)
    , ("fails closed when critical feeds are missing", testFailsClosedOnMissingFeeds)
    , ("keeps missing shadow feeds non-blocking and visible", testShadowFeedWarnings)
    , ("classifies market-risk vetoes as non-error admission denials", testClassifiesPolicyDenyOutcome)
    , ("classifies malformed admission requests as invalid", testClassifiesInvalidAdmissionOutcome)
    , ("emits structured market-risk evidence", testStructuredEvidence)
    , ("decodes full Binance kline flow fields", testDecodesFullKline)
    ]

testAllowsLiquidEntry :: IO ()
testAllowsLiquidEntry = do
    let decision = marketRiskDecision defaultMarketRiskConfig baseInput baseSnapshot
    assert "expected liquid entry to pass" (mrdAllowed decision)
    assert "expected spread evidence" (maybe False (< 11) (mrdSpreadBps decision))
    assert "expected impact evidence" (maybe False (< 6) (mrdImpactBps decision))
    assert "expected basis evidence in bps" ("basis_bps=2.00" `isInfixOf` marketRiskSummary decision)

testBlocksInsufficientDepth :: IO ()
testBlocksInsufficientDepth = do
    let decision = marketRiskDecision defaultMarketRiskConfig baseInput{mriQuantity = 1000} baseSnapshot
    assert "expected insufficient depth to block" (not (mrdAllowed decision))
    assertReason "cannot fill requested quantity" decision

testBlocksExcessiveSpread :: IO ()
testBlocksExcessiveSpread = do
    let book = (requiredBook baseSnapshot){obsBids = [DepthLevel 99 100], obsAsks = [DepthLevel 101 100]}
        decision = marketRiskDecision defaultMarketRiskConfig baseInput baseSnapshot{fmsOrderBook = Just book}
    assert "expected excessive spread to block" (not (mrdAllowed decision))
    assertReason "spread" decision

testBlocksExcessiveImpact :: IO ()
testBlocksExcessiveImpact = do
    let book =
            (requiredBook baseSnapshot)
                { obsAsks = [DepthLevel 100.05 0.1, DepthLevel 101 100]
                }
        decision = marketRiskDecision defaultMarketRiskConfig baseInput baseSnapshot{fmsOrderBook = Just book}
    assert "expected excessive impact to block" (not (mrdAllowed decision))
    assertReason "expected impact" decision

testBlocksAdverseFunding :: IO ()
testBlocksAdverseFunding = do
    let premium = (requiredPremium baseSnapshot){fpsLastFundingRate = 0.01}
        snapshot = baseSnapshot{fmsPremium = Just premium}
        input = baseInput{mriPredictedPrice = Just 100.2, mriMinimumEdge = 0.001}
        decision = marketRiskDecision defaultMarketRiskConfig input snapshot
    assert "expected adverse funding to block" (not (mrdAllowed decision))
    assertReason "consumes edge" decision

testBlocksMissingDirectionalEdge :: IO ()
testBlocksMissingDirectionalEdge = do
    let missingDecision = marketRiskDecision defaultMarketRiskConfig baseInput{mriPredictedPrice = Nothing} baseSnapshot
        counterDecision = marketRiskDecision defaultMarketRiskConfig baseInput{mriPredictedPrice = Just 99} baseSnapshot
    assert "expected missing forecast edge to block" (not (mrdAllowed missingDecision))
    assert "expected counter-directional forecast edge to block" (not (mrdAllowed counterDecision))
    assertReason "directional forecast edge unavailable or non-positive" missingDecision
    assertReason "directional forecast edge unavailable or non-positive" counterDecision

testBlocksEdgeBelowMinimum :: IO ()
testBlocksEdgeBelowMinimum = do
    let input = baseInput{mriPredictedPrice = Just 100.05, mriMinimumEdge = 0.001}
        decision = marketRiskDecision defaultMarketRiskConfig input baseSnapshot
    assert "expected sub-threshold forecast edge to block" (not (mrdAllowed decision))
    assertReason "directional forecast edge" decision
    assertReason "below minimum" decision

testUsesLiveBookForRemainingEdge :: IO ()
testUsesLiveBookForRemainingEdge = do
    let movedBook =
            (requiredBook baseSnapshot)
                { obsBids = [DepthLevel 100.9 100]
                , obsAsks = [DepthLevel 101 100]
                }
        snapshot = baseSnapshot{fmsOrderBook = Just movedBook}
        decision = marketRiskDecision defaultMarketRiskConfig baseInput snapshot
    assert "expected depleted live edge to block" (not (mrdAllowed decision))
    assertReason "directional forecast edge" decision
    assertReason "below minimum" decision

testBlocksHighAdl :: IO ()
testBlocksHighAdl = do
    let snapshot = baseSnapshot{fmsAdlRisk = Just (FuturesAdlRiskSnapshot "high" nowMs)}
        decision = marketRiskDecision defaultMarketRiskConfig baseInput snapshot
    assert "expected high ADL risk to block" (not (mrdAllowed decision))
    assertReason "ADL risk is high" decision

testBlocksStaleFeeds :: IO ()
testBlocksStaleFeeds = do
    let staleAt = nowMs - mrcMaxSnapshotAgeMs defaultMarketRiskConfig - 1
        book =
            (requiredBook baseSnapshot)
                { obsEventTime = Just staleAt
                , obsTransactionTime = Just staleAt
                }
        premium = (requiredPremium baseSnapshot){fpsTime = staleAt}
        snapshot = baseSnapshot{fmsOrderBook = Just book, fmsPremium = Just premium}
        decision = marketRiskDecision defaultMarketRiskConfig baseInput snapshot
    assert "expected stale critical feeds to block" (not (mrdAllowed decision))
    assertReason "source timestamp stale" decision

testFutureTimestampSkew :: IO ()
testFutureTimestampSkew = do
    let withinDecision = marketRiskDecision defaultMarketRiskConfig baseInput (shiftCriticalTimestamps 5000 baseSnapshot)
        beyondDecision = marketRiskDecision defaultMarketRiskConfig baseInput (shiftCriticalTimestamps 5001 baseSnapshot)
    assert "expected bounded venue timestamp skew to pass" (mrdAllowed withinDecision)
    assert "expected excessive future timestamp skew to block" (not (mrdAllowed beyondDecision))
    assertReason "stale" beyondDecision

testFailsClosedOnMissingFeeds :: IO ()
testFailsClosedOnMissingFeeds = do
    let snapshot =
            baseSnapshot
                { fmsOrderBook = Nothing
                , fmsPremium = Nothing
                , fmsAdlRisk = Nothing
                }
        decision = marketRiskDecision defaultMarketRiskConfig baseInput snapshot
    assert "expected missing critical feeds to block" (not (mrdAllowed decision))
    assertReason "order book unavailable" decision
    assertReason "premium/funding unavailable" decision
    assertReason "ADL risk unavailable" decision

testShadowFeedWarnings :: IO ()
testShadowFeedWarnings = do
    let snapshot =
            baseSnapshot
                { fmsOpenInterest = Nothing
                , fmsOpenInterestChangePct = Nothing
                , fmsTakerBuySellRatio = Nothing
                , fmsBasisRate = Nothing
                }
        decision = marketRiskDecision defaultMarketRiskConfig baseInput snapshot
    assert "expected shadow-only gaps not to block a liquid entry" (mrdAllowed decision)
    assertShadowWarning "open interest unavailable" decision
    assertShadowWarning "open interest change unavailable" decision
    assertShadowWarning "taker ratio unavailable" decision
    assertShadowWarning "historical basis unavailable" decision

testClassifiesPolicyDenyOutcome :: IO ()
testClassifiesPolicyDenyOutcome = do
    let spreadBook = (requiredBook baseSnapshot){obsBids = [DepthLevel 99 100], obsAsks = [DepthLevel 101 100]}
        spreadDecision = marketRiskDecision defaultMarketRiskConfig baseInput baseSnapshot{fmsOrderBook = Just spreadBook}
        depthDecision = marketRiskDecision defaultMarketRiskConfig baseInput{mriQuantity = 1000} baseSnapshot
        staleAdlAt = nowMs - mrcMaxAdlAgeMs defaultMarketRiskConfig - 1
        staleAdlDecision =
            marketRiskDecision
                defaultMarketRiskConfig
                baseInput
                baseSnapshot{fmsAdlRisk = Just (FuturesAdlRiskSnapshot "low" staleAdlAt)}
    mapM_
        ( \(label, expectedReason, decision) -> do
            assert ("expected " ++ label ++ " to block") (not (mrdAllowed decision))
            assert ("expected " ++ label ++ " to be a policy denial") (mrdAdmissionOutcome decision == MarketRiskAdmissionDenied)
            assertReason expectedReason decision
        )
        [ ("insufficient depth", "cannot fill requested quantity", depthDecision)
        , ("excessive spread", "spread", spreadDecision)
        , ("stale ADL", "ADL risk stale", staleAdlDecision)
        ]
    assert "expected policy-denied summary" ("market-risk policy-denied" `isInfixOf` marketRiskSummary spreadDecision)

testClassifiesInvalidAdmissionOutcome :: IO ()
testClassifiesInvalidAdmissionOutcome = do
    let invalidDirection = marketRiskDecision defaultMarketRiskConfig baseInput{mriDirection = 0} baseSnapshot
        invalidQuantity = marketRiskDecision defaultMarketRiskConfig baseInput{mriQuantity = 0} baseSnapshot
    assert "expected invalid direction to block" (not (mrdAllowed invalidDirection))
    assert "expected invalid direction classification" (mrdAdmissionOutcome invalidDirection == MarketRiskAdmissionInvalid)
    assertReason "invalid direction" invalidDirection
    assert "expected invalid quantity to block" (not (mrdAllowed invalidQuantity))
    assert "expected invalid quantity classification" (mrdAdmissionOutcome invalidQuantity == MarketRiskAdmissionInvalid)
    assertReason "invalid quantity" invalidQuantity
    assert "expected invalid-request summary" ("market-risk invalid-request" `isInfixOf` marketRiskSummary invalidDirection)

testStructuredEvidence :: IO ()
testStructuredEvidence = do
    let decision = marketRiskDecision defaultMarketRiskConfig baseInput baseSnapshot
    case Aeson.toJSON decision of
        Aeson.Object evidence -> do
            assert "structured evidence includes the allow/deny flag" (KM.lookup "allowed" evidence == Just (Aeson.Bool True))
            assert "structured evidence includes the explicit admission outcome" (KM.lookup "admissionOutcome" evidence == Just (Aeson.String "allowed"))
            assert "structured evidence includes order-book imbalance" (KM.member "bookImbalance" evidence)
            assert "structured evidence includes open-interest change" (KM.member "openInterestChangePct" evidence)
            assert "structured evidence includes shadow data-health warnings" (KM.member "shadowWarnings" evidence)
        _ -> fail "expected structured market-risk JSON object"

testDecodesFullKline :: IO ()
testDecodesFullKline = do
    let payload = "[1700000000000,\"100\",\"102\",\"99\",\"101\",\"12.5\",1700000059999,\"1260.5\",42,\"7.5\",\"756.2\",\"0\"]"
    case eitherDecode (BL8.pack payload) of
        Left err -> fail ("failed to decode kline fixture: " ++ err)
        Right kline -> do
            assert "quote volume retained" (kQuoteVolume kline == Just 1260.5)
            assert "trade count retained" (kTradeCount kline == Just 42)
            assert "taker base volume retained" (kTakerBuyBaseVolume kline == Just 7.5)
            assert "taker quote volume retained" (kTakerBuyQuoteVolume kline == Just 756.2)

baseInput :: MarketRiskInput
baseInput =
    MarketRiskInput
        { mriNowMs = nowMs
        , mriDirection = 1
        , mriQuantity = 1
        , mriPredictedPrice = Just 101
        , mriMinimumEdge = 0.001
        }

baseSnapshot :: FuturesMarketSnapshot
baseSnapshot =
    FuturesMarketSnapshot
        { fmsObservedAt = nowMs
        , fmsOrderBook =
            Just
                OrderBookSnapshot
                    { obsLastUpdateId = 1
                    , obsEventTime = Just nowMs
                    , obsTransactionTime = Just nowMs
                    , obsBids = [DepthLevel 99.95 100]
                    , obsAsks = [DepthLevel 100.05 100]
                    }
        , fmsPremium =
            Just
                FuturesPremiumSnapshot
                    { fpsMarkPrice = 100
                    , fpsIndexPrice = 100
                    , fpsLastFundingRate = 0
                    , fpsNextFundingTime = nowMs + 1000
                    , fpsTime = nowMs
                    }
        , fmsOpenInterest = Just (FuturesOpenInterestSnapshot 12345 nowMs)
        , fmsOpenInterestChangePct = Just (nowMs, 2.5)
        , fmsAdlRisk = Just (FuturesAdlRiskSnapshot "low" nowMs)
        , fmsTakerBuySellRatio = Just (nowMs, 1.1)
        , fmsBasisRate = Just (nowMs, 0.0002)
        }

requiredPremium :: FuturesMarketSnapshot -> FuturesPremiumSnapshot
requiredPremium snapshot =
    case fmsPremium snapshot of
        Just premium -> premium
        Nothing -> error "test fixture missing premium"

requiredBook :: FuturesMarketSnapshot -> OrderBookSnapshot
requiredBook snapshot =
    case fmsOrderBook snapshot of
        Just book -> book
        Nothing -> error "test fixture missing order book"

shiftCriticalTimestamps :: Int64 -> FuturesMarketSnapshot -> FuturesMarketSnapshot
shiftCriticalTimestamps offset snapshot =
    let shiftedAt = nowMs + offset
        book =
            (requiredBook snapshot)
                { obsEventTime = Just shiftedAt
                , obsTransactionTime = Just shiftedAt
                }
        premium = (requiredPremium snapshot){fpsTime = shiftedAt}
        adl = FuturesAdlRiskSnapshot "low" shiftedAt
     in snapshot
            { fmsObservedAt = shiftedAt
            , fmsOrderBook = Just book
            , fmsPremium = Just premium
            , fmsAdlRisk = Just adl
            }

nowMs :: Int64
nowMs = 1700000000000

assertReason :: String -> MarketRiskDecision -> IO ()
assertReason expected decision =
    assert
        ("expected reason containing " ++ show expected ++ ", got " ++ show (mrdReasons decision))
        (any (isInfixOf expected) (mrdReasons decision))

assertShadowWarning :: String -> MarketRiskDecision -> IO ()
assertShadowWarning expected decision =
    assert
        ("expected shadow warning containing " ++ show expected ++ ", got " ++ show (mrdShadowWarnings decision))
        (any (isInfixOf expected) (mrdShadowWarnings decision))

assert :: String -> Bool -> IO ()
assert message condition = unless condition (fail message)
