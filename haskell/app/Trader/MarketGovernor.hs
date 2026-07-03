{-# LANGUAGE OverloadedStrings #-}

module Trader.MarketGovernor (
    MarketGovernorConfig (..),
    MarketGovernorDecision (..),
    MarketGovernorInputs (..),
    MarketGovernorProfile (..),
    defaultMarketGovernorConfig,
    marketGovernorDecision,
    marketGovernorDecisionJson,
    marketGovernorFreshEntryBlockReason,
    marketGovernorIsEntryOnlyReason,
    marketGovernorProfileCode,
) where

import Data.Aeson (Value, object, (.=))
import Data.Maybe (isJust)

data MarketGovernorProfile
    = MarketGovernorOff
    | MarketGovernorDataUnsafe
    | MarketGovernorStress
    | MarketGovernorDeRisk
    | MarketGovernorHighVol
    | MarketGovernorRange
    | MarketGovernorRiskOnTrend
    | MarketGovernorNeutral
    deriving (Eq, Show)

data MarketGovernorConfig = MarketGovernorConfig
    { mgcEnabled :: !Bool
    , mgcStressDrawdown :: !Double
    , mgcDeRiskDrawdown :: !Double
    , mgcStressRollingLoss :: !Double
    , mgcStressLossStreak :: !Int
    , mgcHighVolProbability :: !Double
    , mgcHighVolatility :: !Double
    , mgcStrongConfidence :: !Double
    , mgcTrendProbability :: !Double
    , mgcRangeProbability :: !Double
    , mgcHighVolSizeMultiplier :: !Double
    , mgcDeRiskSizeMultiplier :: !Double
    , mgcRangeSizeMultiplier :: !Double
    , mgcNeutralSizeMultiplier :: !Double
    }
    deriving (Eq, Show)

data MarketGovernorInputs = MarketGovernorInputs
    { mgiMarketDataStale :: !Bool
    , mgiVolatility :: !(Maybe Double)
    , mgiConfidence :: !(Maybe Double)
    , mgiTrendProbability :: !(Maybe Double)
    , mgiMeanReversionProbability :: !(Maybe Double)
    , mgiHighVolProbability :: !(Maybe Double)
    , mgiDrawdown :: !Double
    , mgiLossStreak :: !Int
    , mgiRollingLoss :: !(Maybe Double)
    , mgiCapitalPreservationReason :: !(Maybe String)
    }
    deriving (Eq, Show)

data MarketGovernorDecision = MarketGovernorDecision
    { mgdEnabled :: !Bool
    , mgdProfile :: !MarketGovernorProfile
    , mgdEntrySizeMultiplier :: !Double
    , mgdBlockFreshEntries :: !Bool
    , mgdReduceOnly :: !Bool
    , mgdReason :: !(Maybe String)
    , mgdRecommendedVolConfGate :: !(Maybe String)
    , mgdMethodBias :: ![String]
    , mgdInputs :: !MarketGovernorInputs
    }
    deriving (Eq, Show)

defaultMarketGovernorConfig :: MarketGovernorConfig
defaultMarketGovernorConfig =
    MarketGovernorConfig
        { mgcEnabled = True
        , mgcStressDrawdown = 0.10
        , mgcDeRiskDrawdown = 0.06
        , mgcStressRollingLoss = 0.05
        , mgcStressLossStreak = 3
        , mgcHighVolProbability = 0.65
        , mgcHighVolatility = 1.20
        , mgcStrongConfidence = 0.80
        , mgcTrendProbability = 0.55
        , mgcRangeProbability = 0.55
        , mgcHighVolSizeMultiplier = 0.35
        , mgcDeRiskSizeMultiplier = 0.50
        , mgcRangeSizeMultiplier = 0.70
        , mgcNeutralSizeMultiplier = 0.85
        }

marketGovernorDecision :: MarketGovernorConfig -> MarketGovernorInputs -> MarketGovernorDecision
marketGovernorDecision cfg inputs
    | not (mgcEnabled cfg) =
        mkDecision MarketGovernorOff 1 False False Nothing Nothing ["manual"]
    | dataUnsafe =
        mkDecision MarketGovernorDataUnsafe 0 True True (Just "MARKET_GOVERNOR_DATA_UNSAFE") (Just "vol_conf_v1_high_vol_tighter") ["flat", "reduce_only"]
    | stress =
        mkDecision MarketGovernorStress 0 True True (Just "MARKET_GOVERNOR_STRESS") (Just "vol_conf_v1_high_vol_tighter") ["flat", "reduce_only"]
    | highVol && not strongConfidence =
        mkDecision MarketGovernorHighVol 0 True True (Just "MARKET_GOVERNOR_HIGH_VOL_LOW_CONFIDENCE") (Just "vol_conf_v1_high_vol_tighter") ["flat", "reduce_only"]
    | highVol =
        mkDecision MarketGovernorHighVol highVolSize False False Nothing (Just "vol_conf_v1_high_vol_tighter") ["trend", "breakout", "meta_hedge_blend"]
    | deRisk =
        mkDecision MarketGovernorDeRisk deRiskSize False False (Just "MARKET_GOVERNOR_DERISK") (Just "vol_conf_v1_default") ["conservative"]
    | rangeLike =
        mkDecision MarketGovernorRange rangeSize False False Nothing (Just "vol_conf_v1_conf_stricter") ["mean_reversion", "ta_reversion", "ta_regime_switch"]
    | trendLike =
        mkDecision MarketGovernorRiskOnTrend 1 False False Nothing Nothing ["trend", "breakout", "ta_regime_switch"]
    | otherwise =
        mkDecision MarketGovernorNeutral neutralSize False False Nothing Nothing ["ta_regime_switch", "meta_hedge_blend"]
  where
    drawdown = mgiDrawdown inputs
    dataUnsafe = mgiMarketDataStale inputs || not (finiteDouble drawdown)
    stress =
        isJust (mgiCapitalPreservationReason inputs)
            || finiteAtLeast (mgcStressDrawdown cfg) drawdown
            || mgiLossStreak inputs >= max 1 (mgcStressLossStreak cfg)
            || maybeAtLeast (mgcStressRollingLoss cfg) (mgiRollingLoss inputs)
    deRisk = finiteAtLeast (mgcDeRiskDrawdown cfg) drawdown
    highVol =
        maybeAtLeast (mgcHighVolProbability cfg) (mgiHighVolProbability inputs)
            || maybeAtLeast (mgcHighVolatility cfg) (mgiVolatility inputs)
    strongConfidence = maybeAtLeast (mgcStrongConfidence cfg) (mgiConfidence inputs)
    trendProb = finiteMaybe (mgiTrendProbability inputs)
    rangeProb = finiteMaybe (mgiMeanReversionProbability inputs)
    trendLike =
        maybe False (>= nonNegative (mgcTrendProbability cfg)) trendProb
            && maybe True (\mr -> maybe False (>= mr) trendProb) rangeProb
    rangeLike =
        maybe False (>= nonNegative (mgcRangeProbability cfg)) rangeProb
            && maybe True (\tr -> maybe False (>= tr) rangeProb) trendProb
    highVolSize = unitOr 0.35 (mgcHighVolSizeMultiplier cfg)
    deRiskSize = unitOr 0.50 (mgcDeRiskSizeMultiplier cfg)
    rangeSize = unitOr 0.70 (mgcRangeSizeMultiplier cfg)
    neutralSize = unitOr 0.85 (mgcNeutralSizeMultiplier cfg)
    mkDecision profile size block reduce reason volConf methods =
        MarketGovernorDecision
            { mgdEnabled = mgcEnabled cfg
            , mgdProfile = profile
            , mgdEntrySizeMultiplier = unitOr 0 size
            , mgdBlockFreshEntries = block
            , mgdReduceOnly = reduce
            , mgdReason = reason
            , mgdRecommendedVolConfGate = volConf
            , mgdMethodBias = methods
            , mgdInputs = inputs
            }

marketGovernorFreshEntryBlockReason :: MarketGovernorDecision -> Maybe String
marketGovernorFreshEntryBlockReason decision =
    if mgdBlockFreshEntries decision
        then case mgdReason decision of
            Just reason -> Just reason
            Nothing -> Just "MARKET_GOVERNOR_BLOCK"
        else Nothing

marketGovernorIsEntryOnlyReason :: String -> Bool
marketGovernorIsEntryOnlyReason reason =
    take (length prefix) reason == prefix
  where
    prefix = "MARKET_GOVERNOR_"

marketGovernorProfileCode :: MarketGovernorProfile -> String
marketGovernorProfileCode profile =
    case profile of
        MarketGovernorOff -> "off"
        MarketGovernorDataUnsafe -> "data_unsafe"
        MarketGovernorStress -> "stress"
        MarketGovernorDeRisk -> "derisk"
        MarketGovernorHighVol -> "high_vol"
        MarketGovernorRange -> "range"
        MarketGovernorRiskOnTrend -> "risk_on_trend"
        MarketGovernorNeutral -> "neutral"

marketGovernorDecisionJson :: MarketGovernorDecision -> Value
marketGovernorDecisionJson decision =
    object
        [ "enabled" .= mgdEnabled decision
        , "profile" .= marketGovernorProfileCode (mgdProfile decision)
        , "entrySizeMultiplier" .= mgdEntrySizeMultiplier decision
        , "blockFreshEntries" .= mgdBlockFreshEntries decision
        , "reduceOnly" .= mgdReduceOnly decision
        , "reason" .= mgdReason decision
        , "recommendedVolConfGate" .= mgdRecommendedVolConfGate decision
        , "methodBias" .= mgdMethodBias decision
        , "inputs" .= marketGovernorInputsJson (mgdInputs decision)
        ]

marketGovernorInputsJson :: MarketGovernorInputs -> Value
marketGovernorInputsJson inputs =
    object
        [ "marketDataStale" .= mgiMarketDataStale inputs
        , "volatility" .= finiteMaybe (mgiVolatility inputs)
        , "confidence" .= finiteMaybe (mgiConfidence inputs)
        , "trendProbability" .= finiteMaybe (mgiTrendProbability inputs)
        , "meanReversionProbability" .= finiteMaybe (mgiMeanReversionProbability inputs)
        , "highVolProbability" .= finiteMaybe (mgiHighVolProbability inputs)
        , "drawdown" .= if finiteDouble (mgiDrawdown inputs) then Just (max 0 (mgiDrawdown inputs)) else Nothing
        , "lossStreak" .= max 0 (mgiLossStreak inputs)
        , "rollingLoss" .= finiteMaybe (mgiRollingLoss inputs)
        , "capitalPreservationReason" .= mgiCapitalPreservationReason inputs
        ]

maybeAtLeast :: Double -> Maybe Double -> Bool
maybeAtLeast threshold =
    maybe False (finiteAtLeast threshold) . finiteMaybe

finiteAtLeast :: Double -> Double -> Bool
finiteAtLeast threshold value =
    finiteDouble value && value >= nonNegative threshold

finiteMaybe :: Maybe Double -> Maybe Double
finiteMaybe (Just value)
    | finiteDouble value = Just value
finiteMaybe _ = Nothing

finiteDouble :: Double -> Bool
finiteDouble value = not (isNaN value || isInfinite value)

nonNegative :: Double -> Double
nonNegative value
    | finiteDouble value = max 0 value
    | otherwise = 0

unitOr :: Double -> Double -> Double
unitOr fallback value
    | finiteDouble value = max 0 (min 1 value)
    | otherwise = fallback
