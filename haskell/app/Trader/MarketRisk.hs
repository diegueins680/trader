{-# LANGUAGE OverloadedStrings #-}

module Trader.MarketRisk (
    MarketRiskConfig (..),
    MarketRiskInput (..),
    MarketRiskAdmissionOutcome (..),
    MarketRiskDecision (..),
    defaultMarketRiskConfig,
    loadMarketRiskConfig,
    marketRiskDecision,
    marketRiskSummary,
) where

import Data.Aeson (ToJSON (..), object, (.=))
import Data.Char (toLower)
import Data.Int (Int64)
import Data.List (intercalate)
import Data.Maybe (catMaybes, isNothing)
import System.Environment (lookupEnv)
import Text.Printf (printf)
import Text.Read (readMaybe)
import Trader.Binance (
    DepthLevel (..),
    FuturesAdlRiskSnapshot (..),
    FuturesMarketSnapshot (..),
    FuturesOpenInterestSnapshot (..),
    FuturesPremiumSnapshot (..),
    OrderBookSnapshot (..),
 )

data MarketRiskConfig = MarketRiskConfig
    { mrcEnabled :: !Bool
    , mrcFailClosed :: !Bool
    , mrcMaxSnapshotAgeMs :: !Int64
    , mrcMaxAdlAgeMs :: !Int64
    , mrcMaxShadowAgeMs :: !Int64
    , mrcMaxSpreadBps :: !Double
    , mrcMaxImpactBps :: !Double
    , mrcMaxAbsMarkBasisBps :: !Double
    }
    deriving (Eq, Show)

data MarketRiskInput = MarketRiskInput
    { mriNowMs :: !Int64
    , mriDirection :: !Int
    , mriQuantity :: !Double
    , mriPredictedPrice :: !(Maybe Double)
    , mriMinimumEdge :: !Double
    }
    deriving (Eq, Show)

data MarketRiskAdmissionOutcome
    = MarketRiskAdmissionAllowed
    | MarketRiskAdmissionDenied
    | MarketRiskAdmissionInvalid
    deriving (Eq, Show)

data MarketRiskDecision = MarketRiskDecision
    { mrdAllowed :: !Bool
    , mrdAdmissionOutcome :: !MarketRiskAdmissionOutcome
    , mrdReasons :: ![String]
    , mrdEdgeBps :: !(Maybe Double)
    , mrdSpreadBps :: !(Maybe Double)
    , mrdImpactBps :: !(Maybe Double)
    , mrdAdverseFundingBps :: !(Maybe Double)
    , mrdMarkBasisBps :: !(Maybe Double)
    , mrdBookImbalance :: !(Maybe Double)
    , mrdOpenInterest :: !(Maybe Double)
    , mrdOpenInterestChangePct :: !(Maybe Double)
    , mrdTakerBuySellRatio :: !(Maybe Double)
    , mrdHistoricalBasisRate :: !(Maybe Double)
    , mrdAdlRisk :: !(Maybe String)
    , mrdShadowWarnings :: ![String]
    }
    deriving (Eq, Show)

marketRiskAdmissionOutcomeCode :: MarketRiskAdmissionOutcome -> String
marketRiskAdmissionOutcomeCode outcome =
    case outcome of
        MarketRiskAdmissionAllowed -> "allowed"
        MarketRiskAdmissionDenied -> "policy_denied"
        MarketRiskAdmissionInvalid -> "invalid_request"

instance ToJSON MarketRiskAdmissionOutcome where
    toJSON = toJSON . marketRiskAdmissionOutcomeCode

instance ToJSON MarketRiskDecision where
    toJSON decision =
        object
            [ "allowed" .= mrdAllowed decision
            , "admissionOutcome" .= mrdAdmissionOutcome decision
            , "reasons" .= mrdReasons decision
            , "edgeBps" .= mrdEdgeBps decision
            , "spreadBps" .= mrdSpreadBps decision
            , "impactBps" .= mrdImpactBps decision
            , "adverseFundingBps" .= mrdAdverseFundingBps decision
            , "markBasisBps" .= mrdMarkBasisBps decision
            , "bookImbalance" .= mrdBookImbalance decision
            , "openInterest" .= mrdOpenInterest decision
            , "openInterestChangePct" .= mrdOpenInterestChangePct decision
            , "takerBuySellRatio" .= mrdTakerBuySellRatio decision
            , "historicalBasisRate" .= mrdHistoricalBasisRate decision
            , "adlRisk" .= mrdAdlRisk decision
            , "shadowWarnings" .= mrdShadowWarnings decision
            ]

defaultMarketRiskConfig :: MarketRiskConfig
defaultMarketRiskConfig =
    MarketRiskConfig
        { mrcEnabled = True
        , mrcFailClosed = True
        , mrcMaxSnapshotAgeMs = 30000
        , mrcMaxAdlAgeMs = 60 * 60 * 1000
        , mrcMaxShadowAgeMs = 15 * 60 * 1000
        , mrcMaxSpreadBps = 20
        , mrcMaxImpactBps = 35
        , mrcMaxAbsMarkBasisBps = 100
        }

loadMarketRiskConfig :: IO MarketRiskConfig
loadMarketRiskConfig = do
    enabled <- envBool "TRADER_MARKET_RISK_ENABLED" (mrcEnabled defaultMarketRiskConfig)
    failClosed <- envBool "TRADER_MARKET_RISK_FAIL_CLOSED" (mrcFailClosed defaultMarketRiskConfig)
    snapshotAgeSec <- envDouble "TRADER_MARKET_RISK_MAX_AGE_SEC" 30
    adlAgeSec <- envDouble "TRADER_MARKET_RISK_MAX_ADL_AGE_SEC" 3600
    shadowAgeSec <- envDouble "TRADER_MARKET_RISK_MAX_SHADOW_AGE_SEC" 900
    maxSpread <- envDouble "TRADER_MARKET_RISK_MAX_SPREAD_BPS" (mrcMaxSpreadBps defaultMarketRiskConfig)
    maxImpact <- envDouble "TRADER_MARKET_RISK_MAX_IMPACT_BPS" (mrcMaxImpactBps defaultMarketRiskConfig)
    maxBasis <- envDouble "TRADER_MARKET_RISK_MAX_MARK_BASIS_BPS" (mrcMaxAbsMarkBasisBps defaultMarketRiskConfig)
    pure
        MarketRiskConfig
            { mrcEnabled = enabled
            , mrcFailClosed = failClosed
            , mrcMaxSnapshotAgeMs = secondsToMs snapshotAgeSec
            , mrcMaxAdlAgeMs = secondsToMs adlAgeSec
            , mrcMaxShadowAgeMs = secondsToMs shadowAgeSec
            , mrcMaxSpreadBps = nonNegative maxSpread (mrcMaxSpreadBps defaultMarketRiskConfig)
            , mrcMaxImpactBps = nonNegative maxImpact (mrcMaxImpactBps defaultMarketRiskConfig)
            , mrcMaxAbsMarkBasisBps = nonNegative maxBasis (mrcMaxAbsMarkBasisBps defaultMarketRiskConfig)
            }
  where
    secondsToMs seconds = floor (1000 * nonNegative seconds 0)

marketRiskDecision :: MarketRiskConfig -> MarketRiskInput -> FuturesMarketSnapshot -> MarketRiskDecision
marketRiskDecision cfg input snapshot
    | not (mrcEnabled cfg) = emptyDecision{mrdAllowed = True, mrdAdmissionOutcome = MarketRiskAdmissionAllowed}
    | otherwise =
        let invalidInputReasons =
                [ "invalid direction" | mriDirection input /= 1 && mriDirection input /= -1
                ]
                    ++ ["invalid quantity" | not (finitePositive (mriQuantity input))]
                    ++ ["invalid minimum edge" | not (finiteNonNegative (mriMinimumEdge input))]
            snapshotStale = timestampStale (mrcMaxSnapshotAgeMs cfg) (mriNowMs input) (fmsObservedAt snapshot)
            missingReasons =
                [ "market snapshot stale" | snapshotStale
                ]
                    ++ ["order book unavailable" | isNothing (fmsOrderBook snapshot)]
                    ++ ["premium/funding unavailable" | isNothing (fmsPremium snapshot)]
                    ++ ["ADL risk unavailable" | isNothing (fmsAdlRisk snapshot)]
            criticalReasons = if mrcFailClosed cfg then missingReasons else []
            bookMetrics = fmsOrderBook snapshot >>= orderBookMetrics (mriDirection input) (mriQuantity input)
            edgeBps = expectedEdgeBps input (obmReferencePrice <$> bookMetrics)
            minimumEdgeBps =
                if finiteNonNegative (mriMinimumEdge input)
                    then mriMinimumEdge input * 10000
                    else 0
            edgeReasons =
                case edgeBps of
                    Nothing -> ["directional forecast edge unavailable or non-positive"]
                    Just edge ->
                        [ printf "directional forecast edge %.2f bps is below minimum %.2f bps" edge minimumEdgeBps
                        | edge < minimumEdgeBps
                        ]
            spreadBps = obmSpreadBps <$> bookMetrics
            impactBps = obmImpactBps <$> bookMetrics
            imbalance = obmImbalance <$> bookMetrics
            bookReasons =
                case fmsOrderBook snapshot of
                    Nothing -> []
                    Just book ->
                        let sourceTime =
                                case obsTransactionTime book of
                                    Just ts -> Just ts
                                    Nothing -> obsEventTime book
                            sourceStale = maybe True (timestampStale (mrcMaxSnapshotAgeMs cfg) (mriNowMs input)) sourceTime
                            freshnessReasons = ["order book source timestamp stale or unavailable" | sourceStale && mrcFailClosed cfg]
                         in freshnessReasons
                                ++ case bookMetrics of
                                    Nothing -> ["order book cannot fill requested quantity"]
                                    Just metrics ->
                                        [ printf "spread %.2f bps exceeds %.2f bps" (obmSpreadBps metrics) (mrcMaxSpreadBps cfg)
                                        | obmSpreadBps metrics > mrcMaxSpreadBps cfg
                                        ]
                                            ++ [ printf "expected impact %.2f bps exceeds %.2f bps" (obmImpactBps metrics) (mrcMaxImpactBps cfg)
                                               | obmImpactBps metrics > mrcMaxImpactBps cfg
                                               ]
            fundingBps = adverseFundingBps (mriDirection input) <$> fmsPremium snapshot
            markBasisBps = markBasis <$> fmsPremium snapshot
            premiumReasons =
                case fmsPremium snapshot of
                    Nothing -> []
                    Just premium ->
                        let fundingCost = adverseFundingBps (mriDirection input) premium
                            basis = markBasis premium
                            allInCost = (+ fundingCost) <$> impactBps
                            sourceStale = timestampStale (mrcMaxSnapshotAgeMs cfg) (mriNowMs input) (fpsTime premium)
                         in ["premium/funding source timestamp stale" | sourceStale && mrcFailClosed cfg]
                                ++ [ printf "mark/index basis %.2f bps exceeds %.2f bps" basis (mrcMaxAbsMarkBasisBps cfg)
                                   | abs basis > mrcMaxAbsMarkBasisBps cfg
                                   ]
                                ++ [ printf "marketability plus adverse funding %.2f bps consumes edge %.2f bps" cost edge
                                   | Just cost <- [allInCost]
                                   , Just edge <- [edgeBps]
                                   , cost >= edge
                                   ]
            (adlRisk, adlReasons) =
                case fmsAdlRisk snapshot of
                    Nothing -> (Nothing, [])
                    Just adl ->
                        let risk = map toLower (farsRisk adl)
                            stale = timestampStale (mrcMaxAdlAgeMs cfg) (mriNowMs input) (farsUpdateTime adl)
                            reasons =
                                ["ADL risk stale" | stale && mrcFailClosed cfg]
                                    ++ ["ADL risk is high" | risk == "high"]
                                    ++ ["invalid ADL risk rating" | risk `notElem` ["low", "medium", "high"] && mrcFailClosed cfg]
                         in (Just risk, reasons)
            shadowWarnings =
                openInterestWarnings cfg input (fmsOpenInterest snapshot)
                    ++ shadowSeriesWarnings cfg input "open interest change" finite (fmsOpenInterestChangePct snapshot)
                    ++ shadowSeriesWarnings cfg input "taker ratio" finitePositive (fmsTakerBuySellRatio snapshot)
                    ++ shadowSeriesWarnings cfg input "historical basis" finite (fmsBasisRate snapshot)
            policyReasons = edgeReasons ++ criticalReasons ++ bookReasons ++ premiumReasons ++ adlReasons
            reasons = invalidInputReasons ++ policyReasons
            admissionOutcome
                | not (null invalidInputReasons) = MarketRiskAdmissionInvalid
                | null policyReasons = MarketRiskAdmissionAllowed
                | otherwise = MarketRiskAdmissionDenied
         in MarketRiskDecision
                { mrdAllowed = admissionOutcome == MarketRiskAdmissionAllowed
                , mrdAdmissionOutcome = admissionOutcome
                , mrdReasons = reasons
                , mrdEdgeBps = edgeBps
                , mrdSpreadBps = spreadBps
                , mrdImpactBps = impactBps
                , mrdAdverseFundingBps = fundingBps
                , mrdMarkBasisBps = markBasisBps
                , mrdBookImbalance = imbalance
                , mrdOpenInterest = foisOpenInterest <$> fmsOpenInterest snapshot
                , mrdOpenInterestChangePct = snd <$> fmsOpenInterestChangePct snapshot
                , mrdTakerBuySellRatio = snd <$> fmsTakerBuySellRatio snapshot
                , mrdHistoricalBasisRate = snd <$> fmsBasisRate snapshot
                , mrdAdlRisk = adlRisk
                , mrdShadowWarnings = shadowWarnings
                }
  where
    emptyDecision =
        MarketRiskDecision
            { mrdAllowed = False
            , mrdAdmissionOutcome = MarketRiskAdmissionDenied
            , mrdReasons = []
            , mrdEdgeBps = Nothing
            , mrdSpreadBps = Nothing
            , mrdImpactBps = Nothing
            , mrdAdverseFundingBps = Nothing
            , mrdMarkBasisBps = Nothing
            , mrdBookImbalance = Nothing
            , mrdOpenInterest = Nothing
            , mrdOpenInterestChangePct = Nothing
            , mrdTakerBuySellRatio = Nothing
            , mrdHistoricalBasisRate = Nothing
            , mrdAdlRisk = Nothing
            , mrdShadowWarnings = []
            }

marketRiskSummary :: MarketRiskDecision -> String
marketRiskSummary decision =
    let outcome =
            case mrdAdmissionOutcome decision of
                MarketRiskAdmissionAllowed -> "allowed"
                MarketRiskAdmissionDenied -> "policy-denied"
                MarketRiskAdmissionInvalid -> "invalid-request"
        metric label = fmap (printf (label ++ "=%.2f"))
        metrics =
            catMaybes
                [ metric "edge_bps" (mrdEdgeBps decision)
                , metric "spread_bps" (mrdSpreadBps decision)
                , metric "impact_bps" (mrdImpactBps decision)
                , metric "funding_bps" (mrdAdverseFundingBps decision)
                , metric "mark_basis_bps" (mrdMarkBasisBps decision)
                , metric "book_imbalance" (mrdBookImbalance decision)
                , metric "open_interest" (mrdOpenInterest decision)
                , metric "open_interest_change_pct" (mrdOpenInterestChangePct decision)
                , metric "taker_ratio" (mrdTakerBuySellRatio decision)
                , metric "basis_bps" ((* 10000) <$> mrdHistoricalBasisRate decision)
                , fmap ("adl=" ++) (mrdAdlRisk decision)
                ]
        reasons =
            case mrdReasons decision of
                [] -> []
                xs -> ["reasons=" ++ intercalate "; " xs]
        shadowWarnings =
            case mrdShadowWarnings decision of
                [] -> []
                xs -> ["shadow_warnings=" ++ intercalate "; " xs]
     in intercalate ", " (("market-risk " ++ outcome) : metrics ++ reasons ++ shadowWarnings)

openInterestWarnings :: MarketRiskConfig -> MarketRiskInput -> Maybe FuturesOpenInterestSnapshot -> [String]
openInterestWarnings cfg input snapshot =
    case snapshot of
        Nothing -> ["open interest unavailable"]
        Just openInterest ->
            ["open interest invalid" | not (finitePositive (foisOpenInterest openInterest))]
                ++ [ "open interest stale"
                   | timestampStale (mrcMaxShadowAgeMs cfg) (mriNowMs input) (foisTime openInterest)
                   ]

shadowSeriesWarnings :: MarketRiskConfig -> MarketRiskInput -> String -> (Double -> Bool) -> Maybe (Int64, Double) -> [String]
shadowSeriesWarnings cfg input label validValue point =
    case point of
        Nothing -> [label ++ " unavailable"]
        Just (observedAt, value) ->
            [label ++ " invalid" | not (validValue value)]
                ++ [label ++ " stale" | timestampStale (mrcMaxShadowAgeMs cfg) (mriNowMs input) observedAt]

data OrderBookMetrics = OrderBookMetrics
    { obmReferencePrice :: !Double
    , obmSpreadBps :: !Double
    , obmImpactBps :: !Double
    , obmImbalance :: !Double
    }

orderBookMetrics :: Int -> Double -> OrderBookSnapshot -> Maybe OrderBookMetrics
orderBookMetrics direction quantity book = do
    bestBid <- firstPositive (obsBids book)
    bestAsk <- firstPositive (obsAsks book)
    let bid = dlPrice bestBid
        ask = dlPrice bestAsk
        midpoint = (bid + ask) / 2
    if not (finitePositive midpoint) || ask < bid
        then Nothing
        else do
            vwap <- walkBook quantity (if direction == 1 then obsAsks book else obsBids book)
            let spread = (ask - bid) / midpoint * 10000
                impact =
                    if direction == 1
                        then (vwap / midpoint - 1) * 10000
                        else (1 - vwap / midpoint) * 10000
                bidQuote = sum [dlPrice level * dlQuantity level | level <- obsBids book]
                askQuote = sum [dlPrice level * dlQuantity level | level <- obsAsks book]
                totalQuote = bidQuote + askQuote
                imbalance = if totalQuote <= 0 then 0 else (bidQuote - askQuote) / totalQuote
            if all finite [spread, impact, imbalance]
                then
                    Just
                        OrderBookMetrics
                            { obmReferencePrice = midpoint
                            , obmSpreadBps = max 0 spread
                            , obmImpactBps = max 0 impact
                            , obmImbalance = max (-1) (min 1 imbalance)
                            }
                else Nothing

walkBook :: Double -> [DepthLevel] -> Maybe Double
walkBook wanted levels
    | not (finitePositive wanted) = Nothing
    | otherwise = go wanted 0 0 levels
  where
    go remaining filled notional rest
        | remaining <= 1e-12 =
            if filled > 0 then Just (notional / filled) else Nothing
        | otherwise =
            case rest of
                [] -> Nothing
                level : more ->
                    let available = max 0 (dlQuantity level)
                        takeQty = min remaining available
                     in if not (finitePositive (dlPrice level)) || not (finiteNonNegative available)
                            then Nothing
                            else go (remaining - takeQty) (filled + takeQty) (notional + takeQty * dlPrice level) more

firstPositive :: [DepthLevel] -> Maybe DepthLevel
firstPositive [] = Nothing
firstPositive (level : rest)
    | finitePositive (dlPrice level) && finitePositive (dlQuantity level) = Just level
    | otherwise = firstPositive rest

expectedEdgeBps :: MarketRiskInput -> Maybe Double -> Maybe Double
expectedEdgeBps input referencePrice = do
    predicted <- mriPredictedPrice input
    livePrice <- referencePrice
    if not (finitePositive predicted) || not (finitePositive livePrice)
        then Nothing
        else
            let directional = fromIntegral (mriDirection input) * (predicted / livePrice - 1) * 10000
             in if finite directional && directional > 0 then Just directional else Nothing

adverseFundingBps :: Int -> FuturesPremiumSnapshot -> Double
adverseFundingBps direction premium =
    max 0 (fromIntegral direction * fpsLastFundingRate premium * 10000)

markBasis :: FuturesPremiumSnapshot -> Double
markBasis premium =
    let indexPrice = fpsIndexPrice premium
     in if finitePositive indexPrice
            then (fpsMarkPrice premium / indexPrice - 1) * 10000
            else 1 / 0

envBool :: String -> Bool -> IO Bool
envBool name fallback = do
    raw <- lookupEnv name
    pure $
        case fmap (map toLower . trim) raw of
            Just "1" -> True
            Just "true" -> True
            Just "yes" -> True
            Just "on" -> True
            Just "0" -> False
            Just "false" -> False
            Just "no" -> False
            Just "off" -> False
            _ -> fallback

envDouble :: String -> Double -> IO Double
envDouble name fallback = do
    raw <- lookupEnv name
    pure $
        case raw >>= readMaybe . trim of
            Just value | finite value -> value
            _ -> fallback

nonNegative :: Double -> Double -> Double
nonNegative value fallback
    | finite value && value >= 0 = value
    | otherwise = fallback

finitePositive :: Double -> Bool
finitePositive value = finite value && value > 0

finiteNonNegative :: Double -> Bool
finiteNonNegative value = finite value && value >= 0

finite :: Double -> Bool
finite value = not (isNaN value || isInfinite value)

timestampStale :: Int64 -> Int64 -> Int64 -> Bool
timestampStale maxAge now observed =
    let age = now - observed
     in age < negate maxFutureTimestampSkewMs || age > max 0 maxAge

maxFutureTimestampSkewMs :: Int64
maxFutureTimestampSkewMs = 5000

trim :: String -> String
trim = reverse . dropWhile (== ' ') . reverse . dropWhile (== ' ')
