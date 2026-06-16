{-# LANGUAGE OverloadedStrings #-}

module Trader.Optimizer.Optimize (
    OptimizerArgs (..),
    OptimizerRecordsSummary (..),
    TechnicalOptimizerRanges (..),
    TechnicalTrialParams (..),
    appliedCloseTimingMaxHoldBars,
    applyCloseTimingMetrics,
    applyWalkForwardSummaryMetrics,
    closeTimingReportFromBacktest,
    applyQualityPreset,
    emptyOptimizerRecordsSummary,
    kellyLiteExposureContractReason,
    normalizeObjectiveCode,
    normalizeOptionalPositiveFraction,
    objectiveScore,
    optimizerOptionPresent,
    optimizerRecordsShouldRetryDiscovery,
    optimizerRecordsSummaryJson,
    qualityPresetIntervalFields,
    qualityPresetBudget,
    qualityPresetCeiling,
    qualityPresetWeightFloor,
    readOptimizerRecordsSummary,
    runOptimizer,
    sampleTakeProfitPartial,
) where

import Control.Applicative ((<|>))
import Control.Concurrent (forkIO, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar, tryTakeMVar)
import Control.Exception (SomeException, evaluate, try)
import Control.Monad (foldM, forM_, unless, when)
import Crypto.Hash (Digest, hash)
import Crypto.Hash.Algorithms (SHA256)
import Data.Aeson (Value (..), object, toJSON, (.=))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import Data.ByteArray (convert)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy as BL
import Data.Char (isAlphaNum, isDigit, isSpace, toLower, toUpper)
import Data.Foldable (for_)
import Data.IORef (modifyIORef', newIORef, readIORef)
import Data.List (foldl', group, intercalate, isPrefixOf, sort, sortOn, stripPrefix)
import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe, isJust, isNothing, listToMaybe, mapMaybe, maybeToList)
import qualified Data.Ord
import Data.Scientific (Scientific, toBoundedInteger, toRealFloat)
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Text.Encoding.Error as TEE
import Data.Time.Clock.POSIX (POSIXTime, getPOSIXTime)
import qualified Data.Vector as V
import System.Directory (
    canonicalizePath,
    createDirectoryIfMissing,
    doesFileExist,
    findExecutable,
    getCurrentDirectory,
    getHomeDirectory,
 )
import System.Environment (getEnvironment, lookupEnv)
import System.Exit (ExitCode (..))
import System.FilePath (takeBaseName, takeDirectory, (</>))
import System.IO (
    IOMode (..),
    hClose,
    hFlush,
    hGetContents,
    hPutStrLn,
    hSetBinaryMode,
    hSetEncoding,
    openFile,
    stderr,
    stdout,
    utf8,
 )
import System.Process (
    CreateProcess (..),
    StdStream (..),
    createProcess,
    proc,
    terminateProcess,
    waitForProcess,
 )
import Text.Printf (printf)
import Text.Read (readMaybe)

import Trader.BinanceIntervals (binanceIntervalsCsv)
import Trader.CostCalibration (venueMinEdgeFloor, venueSlippageFloor, venueSpreadFloor, venueTakerFeeFloor)
import Trader.Duration (inferPeriodsPerYear, lookbackBarsFrom)
import Trader.Formal.CloseTiming (
    ComboCloseTimingReport (..),
    ComboTrade (..),
    analyzeComboCloseTiming,
    minimumCloseTimingSamples,
    minimumPositiveLiftSupportSamples,
 )
import Trader.LstmDefaults (defaultLstmAdamBeta1, defaultLstmAdamBeta2, defaultLstmAdamEps)
import Trader.Method (methodCode, parseMethod)
import Trader.Optimization (TuneObjective (..), parseTuneObjective, tuneObjectiveCode)
import Trader.Optimizer.Json (encodePretty)
import Trader.Optimizer.Random (
    Rng,
    nextChoice,
    nextDouble,
    nextIntRange,
    nextLogUniform,
    nextMaybe,
    nextUniform,
    seedRng,
 )
import Trader.Platform (Platform (..), platformIntervals)
import Trader.RoiScore (
    RoiScoreConfig (..),
    defaultRoiScoreConfig,
    paybackBonusForWithConfig,
    roiEvidencePenaltyWithConfig,
    sanitizeRoiScoreConfig,
 )
import Trader.SignalGates (signalEntryOpenThresholdFeasibilityCap, signalEntryOpenThresholdFeasible)
import Trader.Symbol (sanitizeComboSymbolForPlatform)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

clamp :: Double -> Double -> Double -> Double
clamp x lo hi = max lo (min hi x)

clampInt :: Int -> Int -> Int -> Int
clampInt x lo hi = max lo (min hi x)

orderedPair :: (Ord a) => (a, a) -> (a, a)
orderedPair (a, b) = if a <= b then (a, b) else (b, a)

sampleTakeProfitPartial :: (Double, Double) -> Double -> Rng -> (Maybe Double, Rng)
sampleTakeProfitPartial takeProfitPartialRange pDisableTakeProfitPartial rng0 =
    let (lo, hi) = orderedPair takeProfitPartialRange
        lo' = clamp lo 0 0.999999
        hi' = clamp hi 0 0.999999
     in if hi' <= 0
            then (Nothing, rng0)
            else
                nextMaybe
                    pDisableTakeProfitPartial
                    (nextUniform lo' (max lo' hi'))
                    rng0

normalizeOptionalPositiveFraction :: Maybe Double -> Maybe Double
normalizeOptionalPositiveFraction raw =
    case raw of
        Nothing -> Nothing
        Just x
            | isNaN x || isInfinite x -> Nothing
            | x <= 0 -> Nothing
            | otherwise -> Just (min x 0.999999)

normalizeOptionalUnitInterval :: Maybe Double -> Maybe Double
normalizeOptionalUnitInterval raw =
    case raw of
        Nothing -> Nothing
        Just x
            | isNaN x || isInfinite x -> Nothing
            | otherwise -> Just (clamp x 0 1)

normalizeLstmAdamBeta :: Double -> Double -> Double
normalizeLstmAdamBeta fallback x
    | isNaN x || isInfinite x = fallback
    | otherwise = clamp x 0 0.999999

normalizeLstmAdamEps :: Double -> Double -> Double
normalizeLstmAdamEps fallback x
    | isNaN x || isInfinite x = fallback
    | x <= 0 = fallback
    | otherwise = max 1e-12 x

lstmAdamBetaRange :: Double -> Maybe Double -> Maybe Double -> (Double, Double)
lstmAdamBetaRange fallback mLo mHi =
    orderedPair
        ( normalizeLstmAdamBeta fallback (fromMaybe fallback mLo)
        , normalizeLstmAdamBeta fallback (fromMaybe fallback mHi)
        )

lstmAdamEpsRange :: Double -> Maybe Double -> Maybe Double -> (Double, Double)
lstmAdamEpsRange fallback mLo mHi =
    orderedPair
        ( normalizeLstmAdamEps fallback (fromMaybe fallback mLo)
        , normalizeLstmAdamEps fallback (fromMaybe fallback mHi)
        )

minPositionSizeFloor :: Double
minPositionSizeFloor = 1e-12

data TechnicalOptimizerRanges = TechnicalOptimizerRanges
    { torTaEntryOpenThreshold :: !(Double, Double)
    , torTaTrendAdxThreshold :: !(Double, Double)
    , torTaTrendStopAtrMult :: !(Double, Double)
    , torTaTrendTakeProfitAtrMult :: !(Double, Double)
    , torTaReversionRsiLower :: !(Double, Double)
    , torTaReversionRsiUpper :: !(Double, Double)
    , torTaReversionStochLower :: !(Double, Double)
    , torTaReversionStochUpper :: !(Double, Double)
    , torTaReversionEnvelopeLowerMult :: !(Double, Double)
    , torTaReversionEnvelopeUpperMult :: !(Double, Double)
    , torTaReversionStopAtrMult :: !(Double, Double)
    , torTaBreakoutMfiThreshold :: !(Double, Double)
    , torTaBreakoutStopAtrMult :: !(Double, Double)
    , torTaBreakoutTakeProfitAtrMult :: !(Double, Double)
    , torTaSmaCrossStopAtrMult :: !(Double, Double)
    , torTaSmaCrossTakeProfitAtrMult :: !(Double, Double)
    , torTaRegimeSwitchAdxThreshold :: !(Double, Double)
    , torTaRegimeSwitchBollingerBandwidthThreshold :: !(Double, Double)
    , torTaBestCandidateMinConfidence :: !(Double, Double)
    , torTaTrendConfidenceAdxOffset :: !(Double, Double)
    , torTaTrendConfidenceAdxScale :: !(Double, Double)
    , torTaTrendConfidenceMaSpreadMult :: !(Double, Double)
    , torTaTrendConfidenceAroonGapMult :: !(Double, Double)
    , torTaReversionConfidenceRocMult :: !(Double, Double)
    , torTaBreakoutFlowDisagreementConfidence :: !(Double, Double)
    , torTaBreakoutConfidenceDistanceMult :: !(Double, Double)
    , torTaSmaCrossConfidenceSpreadMult :: !(Double, Double)
    , torBlendSoftmaxScale :: !(Double, Double)
    , torBlendNetSoftmaxScale :: !(Double, Double)
    , torBlendEdgePower :: !(Double, Double)
    , torBlendSmoothAlpha :: !(Double, Double)
    , torBlendHedgeEta :: !(Double, Double)
    , torBlendHedgeMaxError :: !(Double, Double)
    , torBlendDivergenceK :: !(Double, Double)
    , torBlendRegimeHighVolCutoff :: !(Double, Double)
    , torBlendRegimeKalmanZCutoff :: !(Double, Double)
    , torBlendBanditExploreScale :: !(Double, Double)
    , torBlendFractalReturnClamp :: !(Double, Double)
    , torBlendFractalAlignedGain :: !(Double, Double)
    , torBlendFractalConflictGain :: !(Double, Double)
    , torBlendCoherenceConflictFloor :: !(Double, Double)
    , torBlendCoherenceConflictScale :: !(Double, Double)
    , torBlendCoherenceBoostThreshold :: !(Double, Double)
    , torBlendCoherenceBoostGain :: !(Double, Double)
    , torBlendCoherenceBoostSpan :: !(Double, Double)
    , torBlendAnchorConflictBase :: !(Double, Double)
    , torBlendAnchorConflictScale :: !(Double, Double)
    , torBlendAnchorAlignedScale :: !(Double, Double)
    , torBlendTensionConflictShrink :: !(Double, Double)
    , torBlendTensionNeutralShrink :: !(Double, Double)
    , torBlendEntropyConflictFloor :: !(Double, Double)
    , torBlendEntropyConflictScale :: !(Double, Double)
    , torBlendEntropyAlignedBase :: !(Double, Double)
    , torBlendEntropyAlignedEntropyScale :: !(Double, Double)
    , torBlendPhaseCancelReturnClamp :: !(Double, Double)
    , torBlendPhaseCancelConflictFloor :: !(Double, Double)
    , torBlendPhaseCancelConflictScale :: !(Double, Double)
    , torBlendPhaseCancelAlignmentScale :: !(Double, Double)
    , torSignalEntryEdgeHeadroomMult :: !(Double, Double)
    , torSignalEntryEdgeSpikeMult :: !(Double, Double)
    , torSignalEntryEdgeSpikeCap :: !(Double, Double)
    , torSignalEntryEdgeSpikeConsecutiveLimit :: !(Int, Int)
    , torSignalTrendSlackMult :: !(Double, Double)
    , torSignalTrendSlackCap :: !(Double, Double)
    , torSignalDirectionalityLookbackBars :: !(Int, Int)
    , torSignalDirectionalityChopEfficiencyMax :: !(Double, Double)
    , torSignalDirectionalityMrEfficiencyMax :: !(Double, Double)
    , torSignalDirectionalityWeakZMin :: !(Double, Double)
    , torSignalPredictorTrackingFloor :: !(Double, Double)
    , torPredictorGbdtTrees :: !(Int, Int)
    , torPredictorGbdtLearningRate :: !(Double, Double)
    , torPredictorCalibrationRatio :: !(Double, Double)
    , torPredictorConformalAlpha :: !(Double, Double)
    , torPredictorTcnRidgeLambda :: !(Double, Double)
    , torPredictorQuantileEpochs :: !(Int, Int)
    , torPredictorQuantileLearningRate :: !(Double, Double)
    , torPredictorQuantileL2 :: !(Double, Double)
    , torPredictorKnnMaxExamples :: !(Int, Int)
    , torPredictorKnnK :: !(Int, Int)
    , torPredictorDecisionTreeMaxDepth :: !(Int, Int)
    , torPredictorDecisionTreeMinLeafSize :: !(Int, Int)
    , torPredictorTransformerTemperature :: !(Double, Double)
    , torPredictorTransformerMaxExamples :: !(Int, Int)
    , torPredictorHmmIterations :: !(Int, Int)
    , torVolConfVolatilityEvidenceMax :: !(Double, Double)
    , torVolConfLowVolThreshold :: !(Double, Double)
    , torVolConfHighVolThreshold :: !(Double, Double)
    , torVolConfWeakConfidenceThreshold :: !(Double, Double)
    , torVolConfStrongConfidenceThreshold :: !(Double, Double)
    , torVolConfLowMediumSizeMult :: !(Double, Double)
    , torVolConfLowStrongSizeMult :: !(Double, Double)
    , torVolConfMediumMediumSizeMult :: !(Double, Double)
    , torVolConfMediumStrongSizeMult :: !(Double, Double)
    , torVolConfHighStrongSizeMult :: !(Double, Double)
    , torRegimeAdxWeight :: !(Double, Double)
    , torRegimeTrendThreshold :: !(Double, Double)
    , torRegimeRangeThreshold :: !(Double, Double)
    , torRegimeTrendAdxFloor :: !(Double, Double)
    , torRegimeTrendAdxSpan :: !(Double, Double)
    , torRegimeTrendAroonFloor :: !(Double, Double)
    , torRegimeTrendAroonSpan :: !(Double, Double)
    , torRegimeTrendSlopeFloor :: !(Double, Double)
    , torRegimeTrendSlopeSpan :: !(Double, Double)
    , torRegimeTrendAroonWeight :: !(Double, Double)
    , torRegimeRangeAdxCeiling :: !(Double, Double)
    , torRegimeRangeAdxSpan :: !(Double, Double)
    , torRegimeRangeWidthCeiling :: !(Double, Double)
    , torRegimeRangeWidthSpan :: !(Double, Double)
    , torRegimeRangeAdxWeight :: !(Double, Double)
    }
    deriving (Eq, Show)

data TechnicalTrialParams = TechnicalTrialParams
    { ttpTaEntryOpenThreshold :: !Double
    , ttpTaTrendAdxThreshold :: !Double
    , ttpTaTrendStopAtrMult :: !Double
    , ttpTaTrendTakeProfitAtrMult :: !Double
    , ttpTaReversionRsiLower :: !Double
    , ttpTaReversionRsiUpper :: !Double
    , ttpTaReversionStochLower :: !Double
    , ttpTaReversionStochUpper :: !Double
    , ttpTaReversionEnvelopeLowerMult :: !Double
    , ttpTaReversionEnvelopeUpperMult :: !Double
    , ttpTaReversionStopAtrMult :: !Double
    , ttpTaBreakoutMfiThreshold :: !Double
    , ttpTaBreakoutStopAtrMult :: !Double
    , ttpTaBreakoutTakeProfitAtrMult :: !Double
    , ttpTaSmaCrossStopAtrMult :: !Double
    , ttpTaSmaCrossTakeProfitAtrMult :: !Double
    , ttpTaRegimeSwitchAdxThreshold :: !Double
    , ttpTaRegimeSwitchBollingerBandwidthThreshold :: !Double
    , ttpTaBestCandidateMinConfidence :: !Double
    , ttpTaTrendConfidenceAdxOffset :: !Double
    , ttpTaTrendConfidenceAdxScale :: !Double
    , ttpTaTrendConfidenceMaSpreadMult :: !Double
    , ttpTaTrendConfidenceAroonGapMult :: !Double
    , ttpTaReversionConfidenceRocMult :: !Double
    , ttpTaBreakoutFlowDisagreementConfidence :: !Double
    , ttpTaBreakoutConfidenceDistanceMult :: !Double
    , ttpTaSmaCrossConfidenceSpreadMult :: !Double
    , ttpBlendSoftmaxScale :: !Double
    , ttpBlendNetSoftmaxScale :: !Double
    , ttpBlendEdgePower :: !Double
    , ttpBlendSmoothAlpha :: !Double
    , ttpBlendHedgeEta :: !Double
    , ttpBlendHedgeMaxError :: !Double
    , ttpBlendDivergenceK :: !Double
    , ttpBlendRegimeHighVolCutoff :: !Double
    , ttpBlendRegimeKalmanZCutoff :: !Double
    , ttpBlendBanditExploreScale :: !Double
    , ttpBlendFractalReturnClamp :: !Double
    , ttpBlendFractalAlignedGain :: !Double
    , ttpBlendFractalConflictGain :: !Double
    , ttpBlendCoherenceConflictFloor :: !Double
    , ttpBlendCoherenceConflictScale :: !Double
    , ttpBlendCoherenceBoostThreshold :: !Double
    , ttpBlendCoherenceBoostGain :: !Double
    , ttpBlendCoherenceBoostSpan :: !Double
    , ttpBlendAnchorConflictBase :: !Double
    , ttpBlendAnchorConflictScale :: !Double
    , ttpBlendAnchorAlignedScale :: !Double
    , ttpBlendTensionConflictShrink :: !Double
    , ttpBlendTensionNeutralShrink :: !Double
    , ttpBlendEntropyConflictFloor :: !Double
    , ttpBlendEntropyConflictScale :: !Double
    , ttpBlendEntropyAlignedBase :: !Double
    , ttpBlendEntropyAlignedEntropyScale :: !Double
    , ttpBlendPhaseCancelReturnClamp :: !Double
    , ttpBlendPhaseCancelConflictFloor :: !Double
    , ttpBlendPhaseCancelConflictScale :: !Double
    , ttpBlendPhaseCancelAlignmentScale :: !Double
    , ttpSignalEntryEdgeHeadroomMult :: !Double
    , ttpSignalEntryEdgeSpikeMult :: !Double
    , ttpSignalEntryEdgeSpikeCap :: !Double
    , ttpSignalEntryEdgeSpikeConsecutiveLimit :: !Int
    , ttpSignalTrendSlackMult :: !Double
    , ttpSignalTrendSlackCap :: !Double
    , ttpSignalDirectionalityLookbackBars :: !Int
    , ttpSignalDirectionalityChopEfficiencyMax :: !Double
    , ttpSignalDirectionalityMrEfficiencyMax :: !Double
    , ttpSignalDirectionalityWeakZMin :: !Double
    , ttpSignalPredictorTrackingFloor :: !Double
    , ttpPredictorGbdtTrees :: !Int
    , ttpPredictorGbdtLearningRate :: !Double
    , ttpPredictorCalibrationRatio :: !Double
    , ttpPredictorConformalAlpha :: !Double
    , ttpPredictorTcnRidgeLambda :: !Double
    , ttpPredictorQuantileEpochs :: !Int
    , ttpPredictorQuantileLearningRate :: !Double
    , ttpPredictorQuantileL2 :: !Double
    , ttpPredictorKnnMaxExamples :: !Int
    , ttpPredictorKnnK :: !Int
    , ttpPredictorDecisionTreeMaxDepth :: !Int
    , ttpPredictorDecisionTreeMinLeafSize :: !Int
    , ttpPredictorTransformerTemperature :: !Double
    , ttpPredictorTransformerMaxExamples :: !Int
    , ttpPredictorHmmIterations :: !Int
    , ttpVolConfVolatilityEvidenceMax :: !Double
    , ttpVolConfLowVolThreshold :: !Double
    , ttpVolConfHighVolThreshold :: !Double
    , ttpVolConfWeakConfidenceThreshold :: !Double
    , ttpVolConfStrongConfidenceThreshold :: !Double
    , ttpVolConfLowMediumSizeMult :: !Double
    , ttpVolConfLowStrongSizeMult :: !Double
    , ttpVolConfMediumMediumSizeMult :: !Double
    , ttpVolConfMediumStrongSizeMult :: !Double
    , ttpVolConfHighStrongSizeMult :: !Double
    , ttpRegimeAdxWeight :: !Double
    , ttpRegimeTrendThreshold :: !Double
    , ttpRegimeRangeThreshold :: !Double
    , ttpRegimeTrendAdxFloor :: !Double
    , ttpRegimeTrendAdxSpan :: !Double
    , ttpRegimeTrendAroonFloor :: !Double
    , ttpRegimeTrendAroonSpan :: !Double
    , ttpRegimeTrendSlopeFloor :: !Double
    , ttpRegimeTrendSlopeSpan :: !Double
    , ttpRegimeTrendAroonWeight :: !Double
    , ttpRegimeRangeAdxCeiling :: !Double
    , ttpRegimeRangeAdxSpan :: !Double
    , ttpRegimeRangeWidthCeiling :: !Double
    , ttpRegimeRangeWidthSpan :: !Double
    , ttpRegimeRangeAdxWeight :: !Double
    }
    deriving (Eq, Show)

normalizeTrialParams :: TrialParams -> TrialParams
normalizeTrialParams p =
    let maxPositionSize' = max minPositionSizeFloor (tpMaxPositionSize p)
        minPositionSize' = clamp (tpMinPositionSize p) minPositionSizeFloor maxPositionSize'
        kellyLiteFloor' = max 0 (tpKellyLiteFloor p)
        kellyLiteCap' = max kellyLiteFloor' (tpKellyLiteCap p)
        thresholdFactorMin' = max 0 (tpThresholdFactorMin p)
        thresholdFactorMax' = max thresholdFactorMin' (tpThresholdFactorMax p)
        kalmanZMin' = max 0 (tpKalmanZMin p)
        kalmanZMax' = max kalmanZMin' (tpKalmanZMax p)
        lstmConfidenceHard' = clamp (tpLstmConfidenceHard p) 0 1
        lstmConfidenceSoftMax =
            if lstmConfidenceHard' == 0
                then 1
                else lstmConfidenceHard'
        lstmConfidenceSoft' = clamp (tpLstmConfidenceSoft p) 0 lstmConfidenceSoftMax
        lstmAdamBeta1' = normalizeLstmAdamBeta defaultLstmAdamBeta1 (tpLstmAdamBeta1 p)
        lstmAdamBeta2' = normalizeLstmAdamBeta defaultLstmAdamBeta2 (tpLstmAdamBeta2 p)
        lstmAdamEps' = normalizeLstmAdamEps defaultLstmAdamEps (tpLstmAdamEps p)
     in p
            { tpBlendWeight = clamp (tpBlendWeight p) 0 1
            , tpRouterScorePnlWeight = clamp (tpRouterScorePnlWeight p) 0 1
            , tpSnrSizeWeight = clamp (tpSnrSizeWeight p) 0 1
            , tpThresholdFactorAlpha = clamp (tpThresholdFactorAlpha p) 0 1
            , tpThresholdFactorMin = thresholdFactorMin'
            , tpThresholdFactorMax = thresholdFactorMax'
            , tpMaxPositionSize = maxPositionSize'
            , tpVolEwmaAlpha = normalizeOptionalPositiveFraction (tpVolEwmaAlpha p)
            , tpSensorVarianceEwmaAlpha = clamp (tpSensorVarianceEwmaAlpha p) 0 1
            , tpLstmAdamBeta1 = lstmAdamBeta1'
            , tpLstmAdamBeta2 = lstmAdamBeta2'
            , tpLstmAdamEps = lstmAdamEps'
            , tpLstmConfidenceSoft = lstmConfidenceSoft'
            , tpLstmConfidenceHard = lstmConfidenceHard'
            , tpStopLoss = normalizeOptionalPositiveFraction (tpStopLoss p)
            , tpTakeProfit = normalizeOptionalPositiveFraction (tpTakeProfit p)
            , tpTrailingStop = normalizeOptionalPositiveFraction (tpTrailingStop p)
            , tpRiskPerTrade = normalizeOptionalPositiveFraction (tpRiskPerTrade p)
            , tpMaxDrawdown = normalizeOptionalPositiveFraction (tpMaxDrawdown p)
            , tpMaxDailyLoss = normalizeOptionalPositiveFraction (tpMaxDailyLoss p)
            , tpMaxWeeklyLoss = normalizeOptionalPositiveFraction (tpMaxWeeklyLoss p)
            , tpKalmanPhysicsBars = max 4 (tpKalmanPhysicsBars p)
            , tpKalmanPhysicsBacktestRatio = clamp (tpKalmanPhysicsBacktestRatio p) 0.000001 0.999999
            , tpKalmanPhysicsVolumeEwmaAlpha = clamp (tpKalmanPhysicsVolumeEwmaAlpha p) 0 1
            , tpKalmanPhysicsVolumeSignalClamp = max 0 (tpKalmanPhysicsVolumeSignalClamp p)
            , tpKalmanZMin = kalmanZMin'
            , tpKalmanZMax = kalmanZMax'
            , tpKalmanSensorCorrelationInflation = clamp (tpKalmanSensorCorrelationInflation p) 0 1
            , tpKalmanInnovationInflationThreshold = max 0 (tpKalmanInnovationInflationThreshold p)
            , tpKalmanInnovationInflationMax = max 1 (tpKalmanInnovationInflationMax p)
            , tpMaxHighVolProb = normalizeOptionalUnitInterval (tpMaxHighVolProb p)
            , tpProtectionMinConfidence = clamp (tpProtectionMinConfidence p) 0 1
            , tpMinPositionSize = minPositionSize'
            , tpKellyLiteFraction = max 0 (tpKellyLiteFraction p)
            , tpKellyLiteFloor = kellyLiteFloor'
            , tpKellyLiteCap = kellyLiteCap'
            , tpRouterRegimeMinBars = max 0 (tpRouterRegimeMinBars p)
            , tpRouterRegimeMinFraction = clamp (tpRouterRegimeMinFraction p) 0 1
            , tpRouterMinScore = clamp (tpRouterMinScore p) 0 1
            , tpTakeProfitPartial = normalizeOptionalPositiveFraction (tpTakeProfitPartial p)
            , tpAdaptiveWinRateSlack = max 0 (tpAdaptiveWinRateSlack p)
            , tpAdaptiveProfitFactorSlack = max 0 (tpAdaptiveProfitFactorSlack p)
            , tpTriLayerPriceActionBody = max 0 (tpTriLayerPriceActionBody p)
            , tpTriLayerPriceActionWickRatio = max 0 (tpTriLayerPriceActionWickRatio p)
            , tpTriLayerPriceActionOppositeWickMax = max 0 (tpTriLayerPriceActionOppositeWickMax p)
            , tpTriLayerPriceActionBodyTolerance = max 0 (tpTriLayerPriceActionBodyTolerance p)
            , tpPerfMinWinRate = normalizeOptionalUnitInterval (tpPerfMinWinRate p)
            , tpMetaLabelMinConfidence = clamp (tpMetaLabelMinConfidence p) 0 1
            , tpRegimeBankHysteresis = clamp (tpRegimeBankHysteresis p) 0 1
            , tpPairsStatArbSizeMult = clamp (tpPairsStatArbSizeMult p) 0 1
            , tpFundingOiFundingCap = normalizeOptionalPositiveFraction (tpFundingOiFundingCap p)
            , tpFundingOiVolLookback = max 2 (tpFundingOiVolLookback p)
            , tpFundingOiVolCap = normalizeOptionalPositiveFraction (tpFundingOiVolCap p)
            , tpFundingOiSizeMult = clamp (tpFundingOiSizeMult p) 0 1
            , tpTechnicalParams = normalizeTechnicalTrialParams (tpTechnicalParams p)
            }

normalizeTechnicalTrialParams :: TechnicalTrialParams -> TechnicalTrialParams
normalizeTechnicalTrialParams p =
    let rsiLower = clamp (ttpTaReversionRsiLower p) 0 100
        rsiUpper = max rsiLower (clamp (ttpTaReversionRsiUpper p) 0 100)
        stochLower = clamp (ttpTaReversionStochLower p) 0 100
        stochUpper = max stochLower (clamp (ttpTaReversionStochUpper p) 0 100)
        volConfEvidenceMax = max 1e-6 (ttpVolConfVolatilityEvidenceMax p)
        volConfLowVolThreshold = clamp (ttpVolConfLowVolThreshold p) 0 volConfEvidenceMax
        volConfHighVolThreshold = clamp (max volConfLowVolThreshold (ttpVolConfHighVolThreshold p)) volConfLowVolThreshold volConfEvidenceMax
        volConfWeakConfidenceThreshold = clamp (ttpVolConfWeakConfidenceThreshold p) 0 1
        volConfStrongConfidenceThreshold = clamp (max volConfWeakConfidenceThreshold (ttpVolConfStrongConfidenceThreshold p)) volConfWeakConfidenceThreshold 1
     in p
            { ttpTaEntryOpenThreshold = max 0 (ttpTaEntryOpenThreshold p)
            , ttpTaTrendAdxThreshold = clamp (ttpTaTrendAdxThreshold p) 0 100
            , ttpTaTrendStopAtrMult = max 1e-6 (ttpTaTrendStopAtrMult p)
            , ttpTaTrendTakeProfitAtrMult = max 1e-6 (ttpTaTrendTakeProfitAtrMult p)
            , ttpTaReversionRsiLower = rsiLower
            , ttpTaReversionRsiUpper = rsiUpper
            , ttpTaReversionStochLower = stochLower
            , ttpTaReversionStochUpper = stochUpper
            , ttpTaReversionEnvelopeLowerMult = max 1e-6 (ttpTaReversionEnvelopeLowerMult p)
            , ttpTaReversionEnvelopeUpperMult = max 1e-6 (ttpTaReversionEnvelopeUpperMult p)
            , ttpTaReversionStopAtrMult = max 1e-6 (ttpTaReversionStopAtrMult p)
            , ttpTaBreakoutMfiThreshold = clamp (ttpTaBreakoutMfiThreshold p) 0 100
            , ttpTaBreakoutStopAtrMult = max 1e-6 (ttpTaBreakoutStopAtrMult p)
            , ttpTaBreakoutTakeProfitAtrMult = max 1e-6 (ttpTaBreakoutTakeProfitAtrMult p)
            , ttpTaSmaCrossStopAtrMult = max 1e-6 (ttpTaSmaCrossStopAtrMult p)
            , ttpTaSmaCrossTakeProfitAtrMult = max 1e-6 (ttpTaSmaCrossTakeProfitAtrMult p)
            , ttpTaRegimeSwitchAdxThreshold = clamp (ttpTaRegimeSwitchAdxThreshold p) 0 100
            , ttpTaRegimeSwitchBollingerBandwidthThreshold = max 0 (ttpTaRegimeSwitchBollingerBandwidthThreshold p)
            , ttpTaBestCandidateMinConfidence = clamp (ttpTaBestCandidateMinConfidence p) 0 1
            , ttpTaTrendConfidenceAdxOffset = clamp (ttpTaTrendConfidenceAdxOffset p) 0 100
            , ttpTaTrendConfidenceAdxScale = max 1e-6 (ttpTaTrendConfidenceAdxScale p)
            , ttpTaTrendConfidenceMaSpreadMult = max 0 (ttpTaTrendConfidenceMaSpreadMult p)
            , ttpTaTrendConfidenceAroonGapMult = max 0 (ttpTaTrendConfidenceAroonGapMult p)
            , ttpTaReversionConfidenceRocMult = max 0 (ttpTaReversionConfidenceRocMult p)
            , ttpTaBreakoutFlowDisagreementConfidence = clamp (ttpTaBreakoutFlowDisagreementConfidence p) 0 1
            , ttpTaBreakoutConfidenceDistanceMult = max 0 (ttpTaBreakoutConfidenceDistanceMult p)
            , ttpTaSmaCrossConfidenceSpreadMult = max 0 (ttpTaSmaCrossConfidenceSpreadMult p)
            , ttpBlendSoftmaxScale = max 1e-6 (ttpBlendSoftmaxScale p)
            , ttpBlendNetSoftmaxScale = max 1e-6 (ttpBlendNetSoftmaxScale p)
            , ttpBlendEdgePower = max 1e-6 (ttpBlendEdgePower p)
            , ttpBlendSmoothAlpha = clamp (ttpBlendSmoothAlpha p) 0 1
            , ttpBlendHedgeEta = max 0 (ttpBlendHedgeEta p)
            , ttpBlendHedgeMaxError = max 1e-6 (ttpBlendHedgeMaxError p)
            , ttpBlendDivergenceK = max 1e-6 (ttpBlendDivergenceK p)
            , ttpBlendRegimeHighVolCutoff = clamp (ttpBlendRegimeHighVolCutoff p) 0 1
            , ttpBlendRegimeKalmanZCutoff = max 0 (ttpBlendRegimeKalmanZCutoff p)
            , ttpBlendBanditExploreScale = max 0 (ttpBlendBanditExploreScale p)
            , ttpBlendFractalReturnClamp = max 1e-6 (ttpBlendFractalReturnClamp p)
            , ttpBlendFractalAlignedGain = max 0 (ttpBlendFractalAlignedGain p)
            , ttpBlendFractalConflictGain = max 0 (ttpBlendFractalConflictGain p)
            , ttpBlendCoherenceConflictFloor = max 0 (ttpBlendCoherenceConflictFloor p)
            , ttpBlendCoherenceConflictScale = max 0 (ttpBlendCoherenceConflictScale p)
            , ttpBlendCoherenceBoostThreshold = clamp (ttpBlendCoherenceBoostThreshold p) 0 1
            , ttpBlendCoherenceBoostGain = max 0 (ttpBlendCoherenceBoostGain p)
            , ttpBlendCoherenceBoostSpan = max 1e-6 (ttpBlendCoherenceBoostSpan p)
            , ttpBlendAnchorConflictBase = clamp (ttpBlendAnchorConflictBase p) 0 1
            , ttpBlendAnchorConflictScale = clamp (ttpBlendAnchorConflictScale p) 0 1
            , ttpBlendAnchorAlignedScale = clamp (ttpBlendAnchorAlignedScale p) 0 1
            , ttpBlendTensionConflictShrink = clamp (ttpBlendTensionConflictShrink p) 0 1
            , ttpBlendTensionNeutralShrink = clamp (ttpBlendTensionNeutralShrink p) 0 1
            , ttpBlendEntropyConflictFloor = clamp (ttpBlendEntropyConflictFloor p) 0 1
            , ttpBlendEntropyConflictScale = clamp (ttpBlendEntropyConflictScale p) 0 1
            , ttpBlendEntropyAlignedBase = clamp (ttpBlendEntropyAlignedBase p) 0 1
            , ttpBlendEntropyAlignedEntropyScale = clamp (ttpBlendEntropyAlignedEntropyScale p) 0 1
            , ttpBlendPhaseCancelReturnClamp = max 1e-6 (ttpBlendPhaseCancelReturnClamp p)
            , ttpBlendPhaseCancelConflictFloor = max 0 (ttpBlendPhaseCancelConflictFloor p)
            , ttpBlendPhaseCancelConflictScale = max 0 (ttpBlendPhaseCancelConflictScale p)
            , ttpBlendPhaseCancelAlignmentScale = max 0 (ttpBlendPhaseCancelAlignmentScale p)
            , ttpSignalEntryEdgeHeadroomMult = max 1e-6 (ttpSignalEntryEdgeHeadroomMult p)
            , ttpSignalEntryEdgeSpikeMult = max 0 (ttpSignalEntryEdgeSpikeMult p)
            , ttpSignalEntryEdgeSpikeCap = max 1e-6 (ttpSignalEntryEdgeSpikeCap p)
            , ttpSignalEntryEdgeSpikeConsecutiveLimit = max 0 (ttpSignalEntryEdgeSpikeConsecutiveLimit p)
            , ttpSignalTrendSlackMult = max 0 (ttpSignalTrendSlackMult p)
            , ttpSignalTrendSlackCap = max 0 (ttpSignalTrendSlackCap p)
            , ttpSignalDirectionalityLookbackBars = max 1 (ttpSignalDirectionalityLookbackBars p)
            , ttpSignalDirectionalityChopEfficiencyMax = clamp (ttpSignalDirectionalityChopEfficiencyMax p) 0 1
            , ttpSignalDirectionalityMrEfficiencyMax = max (clamp (ttpSignalDirectionalityChopEfficiencyMax p) 0 1) (clamp (ttpSignalDirectionalityMrEfficiencyMax p) 0 1)
            , ttpSignalDirectionalityWeakZMin = max 0 (ttpSignalDirectionalityWeakZMin p)
            , ttpSignalPredictorTrackingFloor = max 0 (ttpSignalPredictorTrackingFloor p)
            , ttpPredictorGbdtTrees = max 1 (ttpPredictorGbdtTrees p)
            , ttpPredictorGbdtLearningRate = max 1e-6 (ttpPredictorGbdtLearningRate p)
            , ttpPredictorCalibrationRatio = clamp (ttpPredictorCalibrationRatio p) 0 0.95
            , ttpPredictorConformalAlpha = clamp (ttpPredictorConformalAlpha p) 1e-6 0.999999
            , ttpPredictorTcnRidgeLambda = max 0 (ttpPredictorTcnRidgeLambda p)
            , ttpPredictorQuantileEpochs = max 0 (ttpPredictorQuantileEpochs p)
            , ttpPredictorQuantileLearningRate = max 1e-6 (ttpPredictorQuantileLearningRate p)
            , ttpPredictorQuantileL2 = max 0 (ttpPredictorQuantileL2 p)
            , ttpPredictorKnnMaxExamples = max 1 (ttpPredictorKnnMaxExamples p)
            , ttpPredictorKnnK = max 1 (ttpPredictorKnnK p)
            , ttpPredictorDecisionTreeMaxDepth = max 1 (ttpPredictorDecisionTreeMaxDepth p)
            , ttpPredictorDecisionTreeMinLeafSize = max 1 (ttpPredictorDecisionTreeMinLeafSize p)
            , ttpPredictorTransformerTemperature = max 1e-6 (ttpPredictorTransformerTemperature p)
            , ttpPredictorTransformerMaxExamples = max 1 (ttpPredictorTransformerMaxExamples p)
            , ttpPredictorHmmIterations = max 0 (ttpPredictorHmmIterations p)
            , ttpVolConfVolatilityEvidenceMax = volConfEvidenceMax
            , ttpVolConfLowVolThreshold = volConfLowVolThreshold
            , ttpVolConfHighVolThreshold = volConfHighVolThreshold
            , ttpVolConfWeakConfidenceThreshold = volConfWeakConfidenceThreshold
            , ttpVolConfStrongConfidenceThreshold = volConfStrongConfidenceThreshold
            , ttpVolConfLowMediumSizeMult = clamp (ttpVolConfLowMediumSizeMult p) 0 1
            , ttpVolConfLowStrongSizeMult = clamp (ttpVolConfLowStrongSizeMult p) 0 1
            , ttpVolConfMediumMediumSizeMult = clamp (ttpVolConfMediumMediumSizeMult p) 0 1
            , ttpVolConfMediumStrongSizeMult = clamp (ttpVolConfMediumStrongSizeMult p) 0 1
            , ttpVolConfHighStrongSizeMult = clamp (ttpVolConfHighStrongSizeMult p) 0 1
            , ttpRegimeAdxWeight = clamp (ttpRegimeAdxWeight p) 0 1
            , ttpRegimeTrendThreshold = clamp (ttpRegimeTrendThreshold p) 0 1
            , ttpRegimeRangeThreshold = clamp (ttpRegimeRangeThreshold p) 0 1
            , ttpRegimeTrendAdxFloor = clamp (ttpRegimeTrendAdxFloor p) 0 100
            , ttpRegimeTrendAdxSpan = max 1e-6 (ttpRegimeTrendAdxSpan p)
            , ttpRegimeTrendAroonFloor = clamp (ttpRegimeTrendAroonFloor p) 0 100
            , ttpRegimeTrendAroonSpan = max 1e-6 (ttpRegimeTrendAroonSpan p)
            , ttpRegimeTrendSlopeFloor = max 0 (ttpRegimeTrendSlopeFloor p)
            , ttpRegimeTrendSlopeSpan = max 1e-6 (ttpRegimeTrendSlopeSpan p)
            , ttpRegimeTrendAroonWeight = clamp (ttpRegimeTrendAroonWeight p) 0 1
            , ttpRegimeRangeAdxCeiling = clamp (ttpRegimeRangeAdxCeiling p) 0 100
            , ttpRegimeRangeAdxSpan = max 1e-6 (ttpRegimeRangeAdxSpan p)
            , ttpRegimeRangeWidthCeiling = max 0 (ttpRegimeRangeWidthCeiling p)
            , ttpRegimeRangeWidthSpan = max 1e-6 (ttpRegimeRangeWidthSpan p)
            , ttpRegimeRangeAdxWeight = clamp (ttpRegimeRangeAdxWeight p) 0 1
            }

sampleTechnicalTrialParams :: TechnicalOptimizerRanges -> Rng -> (TechnicalTrialParams, Rng)
sampleTechnicalTrialParams ranges rng0 =
    let (taEntryOpenThreshold, rng1) = sampleNonNegative (torTaEntryOpenThreshold ranges) rng0
        (taTrendAdxThreshold, rng2) = sampleClamped 0 100 (torTaTrendAdxThreshold ranges) rng1
        (taTrendStopAtrMult, rng3) = samplePositive (torTaTrendStopAtrMult ranges) rng2
        (taTrendTakeProfitAtrMult, rng4) = samplePositive (torTaTrendTakeProfitAtrMult ranges) rng3
        (taReversionRsiLower, rng5) = sampleClamped 0 100 (torTaReversionRsiLower ranges) rng4
        (taReversionRsiUpper0, rng6) = sampleClamped 0 100 (torTaReversionRsiUpper ranges) rng5
        taReversionRsiUpper = max taReversionRsiLower taReversionRsiUpper0
        (taReversionStochLower, rng7) = sampleClamped 0 100 (torTaReversionStochLower ranges) rng6
        (taReversionStochUpper0, rng8) = sampleClamped 0 100 (torTaReversionStochUpper ranges) rng7
        taReversionStochUpper = max taReversionStochLower taReversionStochUpper0
        (taReversionEnvelopeLowerMult, rng9) = samplePositive (torTaReversionEnvelopeLowerMult ranges) rng8
        (taReversionEnvelopeUpperMult, rng10) = samplePositive (torTaReversionEnvelopeUpperMult ranges) rng9
        (taReversionStopAtrMult, rng11) = samplePositive (torTaReversionStopAtrMult ranges) rng10
        (taBreakoutMfiThreshold, rng12) = sampleClamped 0 100 (torTaBreakoutMfiThreshold ranges) rng11
        (taBreakoutStopAtrMult, rng13) = samplePositive (torTaBreakoutStopAtrMult ranges) rng12
        (taBreakoutTakeProfitAtrMult, rng14) = samplePositive (torTaBreakoutTakeProfitAtrMult ranges) rng13
        (taSmaCrossStopAtrMult, rng15) = samplePositive (torTaSmaCrossStopAtrMult ranges) rng14
        (taSmaCrossTakeProfitAtrMult, rng16) = samplePositive (torTaSmaCrossTakeProfitAtrMult ranges) rng15
        (taRegimeSwitchAdxThreshold, rng17) = sampleClamped 0 100 (torTaRegimeSwitchAdxThreshold ranges) rng16
        (taRegimeSwitchBollingerBandwidthThreshold, rng18) = sampleNonNegative (torTaRegimeSwitchBollingerBandwidthThreshold ranges) rng17
        (taBestCandidateMinConfidence, rng19) = sampleClamped 0 1 (torTaBestCandidateMinConfidence ranges) rng18
        (taTrendConfidenceAdxOffset, rng20) = sampleClamped 0 100 (torTaTrendConfidenceAdxOffset ranges) rng19
        (taTrendConfidenceAdxScale, rng21) = samplePositive (torTaTrendConfidenceAdxScale ranges) rng20
        (taTrendConfidenceMaSpreadMult, rng22) = sampleNonNegative (torTaTrendConfidenceMaSpreadMult ranges) rng21
        (taTrendConfidenceAroonGapMult, rng23) = sampleNonNegative (torTaTrendConfidenceAroonGapMult ranges) rng22
        (taReversionConfidenceRocMult, rng24) = sampleNonNegative (torTaReversionConfidenceRocMult ranges) rng23
        (taBreakoutFlowDisagreementConfidence, rng25) = sampleClamped 0 1 (torTaBreakoutFlowDisagreementConfidence ranges) rng24
        (taBreakoutConfidenceDistanceMult, rng26) = sampleNonNegative (torTaBreakoutConfidenceDistanceMult ranges) rng25
        (taSmaCrossConfidenceSpreadMult, rng27) = sampleNonNegative (torTaSmaCrossConfidenceSpreadMult ranges) rng26
        (blendSoftmaxScale, rng28) = samplePositive (torBlendSoftmaxScale ranges) rng27
        (blendNetSoftmaxScale, rng29) = samplePositive (torBlendNetSoftmaxScale ranges) rng28
        (blendEdgePower, rng30) = samplePositive (torBlendEdgePower ranges) rng29
        (blendSmoothAlpha, rng31) = sampleClamped 0 1 (torBlendSmoothAlpha ranges) rng30
        (blendHedgeEta, rng32) = sampleNonNegative (torBlendHedgeEta ranges) rng31
        (blendHedgeMaxError, rng33) = samplePositive (torBlendHedgeMaxError ranges) rng32
        (blendDivergenceK, rng34) = samplePositive (torBlendDivergenceK ranges) rng33
        (blendRegimeHighVolCutoff, rng35) = sampleClamped 0 1 (torBlendRegimeHighVolCutoff ranges) rng34
        (blendRegimeKalmanZCutoff, rng36) = sampleNonNegative (torBlendRegimeKalmanZCutoff ranges) rng35
        (blendBanditExploreScale, rng37) = sampleNonNegative (torBlendBanditExploreScale ranges) rng36
        (blendFractalReturnClamp, rng38) = samplePositive (torBlendFractalReturnClamp ranges) rng37
        (blendFractalAlignedGain, rng39) = sampleNonNegative (torBlendFractalAlignedGain ranges) rng38
        (blendFractalConflictGain, rng40) = sampleNonNegative (torBlendFractalConflictGain ranges) rng39
        (blendCoherenceConflictFloor, rng41) = sampleNonNegative (torBlendCoherenceConflictFloor ranges) rng40
        (blendCoherenceConflictScale, rng42) = sampleNonNegative (torBlendCoherenceConflictScale ranges) rng41
        (blendCoherenceBoostThreshold, rng43) = sampleClamped 0 1 (torBlendCoherenceBoostThreshold ranges) rng42
        (blendCoherenceBoostGain, rng44) = sampleNonNegative (torBlendCoherenceBoostGain ranges) rng43
        (blendCoherenceBoostSpan, rng45) = samplePositive (torBlendCoherenceBoostSpan ranges) rng44
        (blendAnchorConflictBase, rng46) = sampleClamped 0 1 (torBlendAnchorConflictBase ranges) rng45
        (blendAnchorConflictScale, rng47) = sampleClamped 0 1 (torBlendAnchorConflictScale ranges) rng46
        (blendAnchorAlignedScale, rng48) = sampleClamped 0 1 (torBlendAnchorAlignedScale ranges) rng47
        (blendTensionConflictShrink, rng49) = sampleClamped 0 1 (torBlendTensionConflictShrink ranges) rng48
        (blendTensionNeutralShrink, rng50) = sampleClamped 0 1 (torBlendTensionNeutralShrink ranges) rng49
        (blendEntropyConflictFloor, rng51) = sampleClamped 0 1 (torBlendEntropyConflictFloor ranges) rng50
        (blendEntropyConflictScale, rng52) = sampleClamped 0 1 (torBlendEntropyConflictScale ranges) rng51
        (blendEntropyAlignedBase, rng53) = sampleClamped 0 1 (torBlendEntropyAlignedBase ranges) rng52
        (blendEntropyAlignedEntropyScale, rng54) = sampleClamped 0 1 (torBlendEntropyAlignedEntropyScale ranges) rng53
        (blendPhaseCancelReturnClamp, rng55) = samplePositive (torBlendPhaseCancelReturnClamp ranges) rng54
        (blendPhaseCancelConflictFloor, rng56) = sampleNonNegative (torBlendPhaseCancelConflictFloor ranges) rng55
        (blendPhaseCancelConflictScale, rng57) = sampleNonNegative (torBlendPhaseCancelConflictScale ranges) rng56
        (blendPhaseCancelAlignmentScale, rng58) = sampleNonNegative (torBlendPhaseCancelAlignmentScale ranges) rng57
        (signalEntryEdgeHeadroomMult, rng59) = samplePositive (torSignalEntryEdgeHeadroomMult ranges) rng58
        (signalEntryEdgeSpikeMult, rng60) = sampleNonNegative (torSignalEntryEdgeSpikeMult ranges) rng59
        (signalEntryEdgeSpikeCap, rng61) = samplePositive (torSignalEntryEdgeSpikeCap ranges) rng60
        (signalEntryEdgeSpikeConsecutiveLimit, rng62) =
            let (lo, hi) = orderedPair (torSignalEntryEdgeSpikeConsecutiveLimit ranges)
             in nextIntRange (max 0 lo) (max (max 0 lo) hi) rng61
        (signalTrendSlackMult, rng63) = sampleNonNegative (torSignalTrendSlackMult ranges) rng62
        (signalTrendSlackCap, rng64) = sampleNonNegative (torSignalTrendSlackCap ranges) rng63
        (signalDirectionalityLookbackBars, rng65) =
            let (lo, hi) = orderedPair (torSignalDirectionalityLookbackBars ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng64
        (signalDirectionalityChopEfficiencyMax, rng66) = sampleClamped 0 1 (torSignalDirectionalityChopEfficiencyMax ranges) rng65
        (signalDirectionalityMrEfficiencyMax0, rng67) = sampleClamped 0 1 (torSignalDirectionalityMrEfficiencyMax ranges) rng66
        signalDirectionalityMrEfficiencyMax = max signalDirectionalityChopEfficiencyMax signalDirectionalityMrEfficiencyMax0
        (signalDirectionalityWeakZMin, rng68) = sampleNonNegative (torSignalDirectionalityWeakZMin ranges) rng67
        (signalPredictorTrackingFloor, rng69) = sampleNonNegative (torSignalPredictorTrackingFloor ranges) rng68
        (predictorGbdtTrees, rng70) =
            let (lo, hi) = orderedPair (torPredictorGbdtTrees ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng69
        (predictorGbdtLearningRate, rng71) = samplePositive (torPredictorGbdtLearningRate ranges) rng70
        (predictorCalibrationRatio, rng72) = sampleClamped 0 0.95 (torPredictorCalibrationRatio ranges) rng71
        (predictorConformalAlpha, rng73) = sampleClamped 1e-6 0.999999 (torPredictorConformalAlpha ranges) rng72
        (predictorTcnRidgeLambda, rng74) = sampleNonNegative (torPredictorTcnRidgeLambda ranges) rng73
        (predictorQuantileEpochs, rng75) =
            let (lo, hi) = orderedPair (torPredictorQuantileEpochs ranges)
             in nextIntRange (max 0 lo) (max (max 0 lo) hi) rng74
        (predictorQuantileLearningRate, rng76) = samplePositive (torPredictorQuantileLearningRate ranges) rng75
        (predictorQuantileL2, rng77) = sampleNonNegative (torPredictorQuantileL2 ranges) rng76
        (predictorKnnMaxExamples, rng78) =
            let (lo, hi) = orderedPair (torPredictorKnnMaxExamples ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng77
        (predictorKnnK, rng79) =
            let (lo, hi) = orderedPair (torPredictorKnnK ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng78
        (predictorDecisionTreeMaxDepth, rng80) =
            let (lo, hi) = orderedPair (torPredictorDecisionTreeMaxDepth ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng79
        (predictorDecisionTreeMinLeafSize, rng81) =
            let (lo, hi) = orderedPair (torPredictorDecisionTreeMinLeafSize ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng80
        (predictorTransformerTemperature, rng82) = samplePositive (torPredictorTransformerTemperature ranges) rng81
        (predictorTransformerMaxExamples, rng83) =
            let (lo, hi) = orderedPair (torPredictorTransformerMaxExamples ranges)
             in nextIntRange (max 1 lo) (max (max 1 lo) hi) rng82
        (predictorHmmIterations, rng84) =
            let (lo, hi) = orderedPair (torPredictorHmmIterations ranges)
             in nextIntRange (max 0 lo) (max (max 0 lo) hi) rng83
        (volConfVolatilityEvidenceMax, rng85) = samplePositive (torVolConfVolatilityEvidenceMax ranges) rng84
        (volConfLowVolThreshold0, rng86) = sampleClamped 0 volConfVolatilityEvidenceMax (torVolConfLowVolThreshold ranges) rng85
        (volConfHighVolThreshold0, rng87) = sampleClamped 0 volConfVolatilityEvidenceMax (torVolConfHighVolThreshold ranges) rng86
        volConfLowVolThreshold = min volConfLowVolThreshold0 volConfHighVolThreshold0
        volConfHighVolThreshold = max volConfLowVolThreshold0 volConfHighVolThreshold0
        (volConfWeakConfidenceThreshold0, rng88) = sampleClamped 0 1 (torVolConfWeakConfidenceThreshold ranges) rng87
        (volConfStrongConfidenceThreshold0, rng89) = sampleClamped 0 1 (torVolConfStrongConfidenceThreshold ranges) rng88
        volConfWeakConfidenceThreshold = min volConfWeakConfidenceThreshold0 volConfStrongConfidenceThreshold0
        volConfStrongConfidenceThreshold = max volConfWeakConfidenceThreshold0 volConfStrongConfidenceThreshold0
        (volConfLowMediumSizeMult, rng90) = sampleClamped 0 1 (torVolConfLowMediumSizeMult ranges) rng89
        (volConfLowStrongSizeMult, rng91) = sampleClamped 0 1 (torVolConfLowStrongSizeMult ranges) rng90
        (volConfMediumMediumSizeMult, rng92) = sampleClamped 0 1 (torVolConfMediumMediumSizeMult ranges) rng91
        (volConfMediumStrongSizeMult, rng93) = sampleClamped 0 1 (torVolConfMediumStrongSizeMult ranges) rng92
        (volConfHighStrongSizeMult, rng94) = sampleClamped 0 1 (torVolConfHighStrongSizeMult ranges) rng93
        (regimeAdxWeight, rng95) = sampleClamped 0 1 (torRegimeAdxWeight ranges) rng94
        (regimeTrendThreshold, rng96) = sampleClamped 0 1 (torRegimeTrendThreshold ranges) rng95
        (regimeRangeThreshold, rng97) = sampleClamped 0 1 (torRegimeRangeThreshold ranges) rng96
        (regimeTrendAdxFloor, rng98) = sampleClamped 0 100 (torRegimeTrendAdxFloor ranges) rng97
        (regimeTrendAdxSpan, rng99) = samplePositive (torRegimeTrendAdxSpan ranges) rng98
        (regimeTrendAroonFloor, rng100) = sampleClamped 0 100 (torRegimeTrendAroonFloor ranges) rng99
        (regimeTrendAroonSpan, rng101) = samplePositive (torRegimeTrendAroonSpan ranges) rng100
        (regimeTrendSlopeFloor, rng102) = sampleNonNegative (torRegimeTrendSlopeFloor ranges) rng101
        (regimeTrendSlopeSpan, rng103) = samplePositive (torRegimeTrendSlopeSpan ranges) rng102
        (regimeTrendAroonWeight, rng104) = sampleClamped 0 1 (torRegimeTrendAroonWeight ranges) rng103
        (regimeRangeAdxCeiling, rng105) = sampleClamped 0 100 (torRegimeRangeAdxCeiling ranges) rng104
        (regimeRangeAdxSpan, rng106) = samplePositive (torRegimeRangeAdxSpan ranges) rng105
        (regimeRangeWidthCeiling, rng107) = sampleNonNegative (torRegimeRangeWidthCeiling ranges) rng106
        (regimeRangeWidthSpan, rng108) = samplePositive (torRegimeRangeWidthSpan ranges) rng107
        (regimeRangeAdxWeight, rng109) = sampleClamped 0 1 (torRegimeRangeAdxWeight ranges) rng108
     in ( normalizeTechnicalTrialParams
            TechnicalTrialParams
                { ttpTaEntryOpenThreshold = taEntryOpenThreshold
                , ttpTaTrendAdxThreshold = taTrendAdxThreshold
                , ttpTaTrendStopAtrMult = taTrendStopAtrMult
                , ttpTaTrendTakeProfitAtrMult = taTrendTakeProfitAtrMult
                , ttpTaReversionRsiLower = taReversionRsiLower
                , ttpTaReversionRsiUpper = taReversionRsiUpper
                , ttpTaReversionStochLower = taReversionStochLower
                , ttpTaReversionStochUpper = taReversionStochUpper
                , ttpTaReversionEnvelopeLowerMult = taReversionEnvelopeLowerMult
                , ttpTaReversionEnvelopeUpperMult = taReversionEnvelopeUpperMult
                , ttpTaReversionStopAtrMult = taReversionStopAtrMult
                , ttpTaBreakoutMfiThreshold = taBreakoutMfiThreshold
                , ttpTaBreakoutStopAtrMult = taBreakoutStopAtrMult
                , ttpTaBreakoutTakeProfitAtrMult = taBreakoutTakeProfitAtrMult
                , ttpTaSmaCrossStopAtrMult = taSmaCrossStopAtrMult
                , ttpTaSmaCrossTakeProfitAtrMult = taSmaCrossTakeProfitAtrMult
                , ttpTaRegimeSwitchAdxThreshold = taRegimeSwitchAdxThreshold
                , ttpTaRegimeSwitchBollingerBandwidthThreshold = taRegimeSwitchBollingerBandwidthThreshold
                , ttpTaBestCandidateMinConfidence = taBestCandidateMinConfidence
                , ttpTaTrendConfidenceAdxOffset = taTrendConfidenceAdxOffset
                , ttpTaTrendConfidenceAdxScale = taTrendConfidenceAdxScale
                , ttpTaTrendConfidenceMaSpreadMult = taTrendConfidenceMaSpreadMult
                , ttpTaTrendConfidenceAroonGapMult = taTrendConfidenceAroonGapMult
                , ttpTaReversionConfidenceRocMult = taReversionConfidenceRocMult
                , ttpTaBreakoutFlowDisagreementConfidence = taBreakoutFlowDisagreementConfidence
                , ttpTaBreakoutConfidenceDistanceMult = taBreakoutConfidenceDistanceMult
                , ttpTaSmaCrossConfidenceSpreadMult = taSmaCrossConfidenceSpreadMult
                , ttpBlendSoftmaxScale = blendSoftmaxScale
                , ttpBlendNetSoftmaxScale = blendNetSoftmaxScale
                , ttpBlendEdgePower = blendEdgePower
                , ttpBlendSmoothAlpha = blendSmoothAlpha
                , ttpBlendHedgeEta = blendHedgeEta
                , ttpBlendHedgeMaxError = blendHedgeMaxError
                , ttpBlendDivergenceK = blendDivergenceK
                , ttpBlendRegimeHighVolCutoff = blendRegimeHighVolCutoff
                , ttpBlendRegimeKalmanZCutoff = blendRegimeKalmanZCutoff
                , ttpBlendBanditExploreScale = blendBanditExploreScale
                , ttpBlendFractalReturnClamp = blendFractalReturnClamp
                , ttpBlendFractalAlignedGain = blendFractalAlignedGain
                , ttpBlendFractalConflictGain = blendFractalConflictGain
                , ttpBlendCoherenceConflictFloor = blendCoherenceConflictFloor
                , ttpBlendCoherenceConflictScale = blendCoherenceConflictScale
                , ttpBlendCoherenceBoostThreshold = blendCoherenceBoostThreshold
                , ttpBlendCoherenceBoostGain = blendCoherenceBoostGain
                , ttpBlendCoherenceBoostSpan = blendCoherenceBoostSpan
                , ttpBlendAnchorConflictBase = blendAnchorConflictBase
                , ttpBlendAnchorConflictScale = blendAnchorConflictScale
                , ttpBlendAnchorAlignedScale = blendAnchorAlignedScale
                , ttpBlendTensionConflictShrink = blendTensionConflictShrink
                , ttpBlendTensionNeutralShrink = blendTensionNeutralShrink
                , ttpBlendEntropyConflictFloor = blendEntropyConflictFloor
                , ttpBlendEntropyConflictScale = blendEntropyConflictScale
                , ttpBlendEntropyAlignedBase = blendEntropyAlignedBase
                , ttpBlendEntropyAlignedEntropyScale = blendEntropyAlignedEntropyScale
                , ttpBlendPhaseCancelReturnClamp = blendPhaseCancelReturnClamp
                , ttpBlendPhaseCancelConflictFloor = blendPhaseCancelConflictFloor
                , ttpBlendPhaseCancelConflictScale = blendPhaseCancelConflictScale
                , ttpBlendPhaseCancelAlignmentScale = blendPhaseCancelAlignmentScale
                , ttpSignalEntryEdgeHeadroomMult = signalEntryEdgeHeadroomMult
                , ttpSignalEntryEdgeSpikeMult = signalEntryEdgeSpikeMult
                , ttpSignalEntryEdgeSpikeCap = signalEntryEdgeSpikeCap
                , ttpSignalEntryEdgeSpikeConsecutiveLimit = signalEntryEdgeSpikeConsecutiveLimit
                , ttpSignalTrendSlackMult = signalTrendSlackMult
                , ttpSignalTrendSlackCap = signalTrendSlackCap
                , ttpSignalDirectionalityLookbackBars = signalDirectionalityLookbackBars
                , ttpSignalDirectionalityChopEfficiencyMax = signalDirectionalityChopEfficiencyMax
                , ttpSignalDirectionalityMrEfficiencyMax = signalDirectionalityMrEfficiencyMax
                , ttpSignalDirectionalityWeakZMin = signalDirectionalityWeakZMin
                , ttpSignalPredictorTrackingFloor = signalPredictorTrackingFloor
                , ttpPredictorGbdtTrees = predictorGbdtTrees
                , ttpPredictorGbdtLearningRate = predictorGbdtLearningRate
                , ttpPredictorCalibrationRatio = predictorCalibrationRatio
                , ttpPredictorConformalAlpha = predictorConformalAlpha
                , ttpPredictorTcnRidgeLambda = predictorTcnRidgeLambda
                , ttpPredictorQuantileEpochs = predictorQuantileEpochs
                , ttpPredictorQuantileLearningRate = predictorQuantileLearningRate
                , ttpPredictorQuantileL2 = predictorQuantileL2
                , ttpPredictorKnnMaxExamples = predictorKnnMaxExamples
                , ttpPredictorKnnK = predictorKnnK
                , ttpPredictorDecisionTreeMaxDepth = predictorDecisionTreeMaxDepth
                , ttpPredictorDecisionTreeMinLeafSize = predictorDecisionTreeMinLeafSize
                , ttpPredictorTransformerTemperature = predictorTransformerTemperature
                , ttpPredictorTransformerMaxExamples = predictorTransformerMaxExamples
                , ttpPredictorHmmIterations = predictorHmmIterations
                , ttpVolConfVolatilityEvidenceMax = volConfVolatilityEvidenceMax
                , ttpVolConfLowVolThreshold = volConfLowVolThreshold
                , ttpVolConfHighVolThreshold = volConfHighVolThreshold
                , ttpVolConfWeakConfidenceThreshold = volConfWeakConfidenceThreshold
                , ttpVolConfStrongConfidenceThreshold = volConfStrongConfidenceThreshold
                , ttpVolConfLowMediumSizeMult = volConfLowMediumSizeMult
                , ttpVolConfLowStrongSizeMult = volConfLowStrongSizeMult
                , ttpVolConfMediumMediumSizeMult = volConfMediumMediumSizeMult
                , ttpVolConfMediumStrongSizeMult = volConfMediumStrongSizeMult
                , ttpVolConfHighStrongSizeMult = volConfHighStrongSizeMult
                , ttpRegimeAdxWeight = regimeAdxWeight
                , ttpRegimeTrendThreshold = regimeTrendThreshold
                , ttpRegimeRangeThreshold = regimeRangeThreshold
                , ttpRegimeTrendAdxFloor = regimeTrendAdxFloor
                , ttpRegimeTrendAdxSpan = regimeTrendAdxSpan
                , ttpRegimeTrendAroonFloor = regimeTrendAroonFloor
                , ttpRegimeTrendAroonSpan = regimeTrendAroonSpan
                , ttpRegimeTrendSlopeFloor = regimeTrendSlopeFloor
                , ttpRegimeTrendSlopeSpan = regimeTrendSlopeSpan
                , ttpRegimeTrendAroonWeight = regimeTrendAroonWeight
                , ttpRegimeRangeAdxCeiling = regimeRangeAdxCeiling
                , ttpRegimeRangeAdxSpan = regimeRangeAdxSpan
                , ttpRegimeRangeWidthCeiling = regimeRangeWidthCeiling
                , ttpRegimeRangeWidthSpan = regimeRangeWidthSpan
                , ttpRegimeRangeAdxWeight = regimeRangeAdxWeight
                }
        , rng109
        )
  where
    samplePositive range rng =
        let (lo, hi) = orderedPair range
            lo' = max 1e-6 lo
            hi' = max lo' hi
         in nextUniform lo' hi' rng
    sampleNonNegative range rng =
        let (lo, hi) = orderedPair range
            lo' = max 0 lo
            hi' = max lo' hi
         in nextUniform lo' hi' rng
    sampleClamped loBound hiBound range rng =
        let (lo, hi) = orderedPair range
            lo' = clamp lo loBound hiBound
            hi' = max lo' (clamp hi loBound hiBound)
         in nextUniform lo' hi' rng

technicalTrialParamsArgs :: TechnicalTrialParams -> [String]
technicalTrialParamsArgs p =
    [ "--ta-entry-open-threshold"
    , printf "%.12g" (ttpTaEntryOpenThreshold p)
    , "--ta-trend-adx-threshold"
    , printf "%.12g" (ttpTaTrendAdxThreshold p)
    , "--ta-trend-stop-atr-mult"
    , printf "%.12g" (ttpTaTrendStopAtrMult p)
    , "--ta-trend-tp-atr-mult"
    , printf "%.12g" (ttpTaTrendTakeProfitAtrMult p)
    , "--ta-reversion-rsi-lower"
    , printf "%.12g" (ttpTaReversionRsiLower p)
    , "--ta-reversion-rsi-upper"
    , printf "%.12g" (ttpTaReversionRsiUpper p)
    , "--ta-reversion-stoch-lower"
    , printf "%.12g" (ttpTaReversionStochLower p)
    , "--ta-reversion-stoch-upper"
    , printf "%.12g" (ttpTaReversionStochUpper p)
    , "--ta-reversion-envelope-lower-mult"
    , printf "%.12g" (ttpTaReversionEnvelopeLowerMult p)
    , "--ta-reversion-envelope-upper-mult"
    , printf "%.12g" (ttpTaReversionEnvelopeUpperMult p)
    , "--ta-reversion-stop-atr-mult"
    , printf "%.12g" (ttpTaReversionStopAtrMult p)
    , "--ta-breakout-mfi-threshold"
    , printf "%.12g" (ttpTaBreakoutMfiThreshold p)
    , "--ta-breakout-stop-atr-mult"
    , printf "%.12g" (ttpTaBreakoutStopAtrMult p)
    , "--ta-breakout-tp-atr-mult"
    , printf "%.12g" (ttpTaBreakoutTakeProfitAtrMult p)
    , "--ta-sma-cross-stop-atr-mult"
    , printf "%.12g" (ttpTaSmaCrossStopAtrMult p)
    , "--ta-sma-cross-tp-atr-mult"
    , printf "%.12g" (ttpTaSmaCrossTakeProfitAtrMult p)
    , "--ta-regime-switch-adx-threshold"
    , printf "%.12g" (ttpTaRegimeSwitchAdxThreshold p)
    , "--ta-regime-switch-bb-width-threshold"
    , printf "%.12g" (ttpTaRegimeSwitchBollingerBandwidthThreshold p)
    , "--ta-best-min-confidence"
    , printf "%.12g" (ttpTaBestCandidateMinConfidence p)
    , "--ta-trend-confidence-adx-offset"
    , printf "%.12g" (ttpTaTrendConfidenceAdxOffset p)
    , "--ta-trend-confidence-adx-scale"
    , printf "%.12g" (ttpTaTrendConfidenceAdxScale p)
    , "--ta-trend-confidence-ma-spread-mult"
    , printf "%.12g" (ttpTaTrendConfidenceMaSpreadMult p)
    , "--ta-trend-confidence-aroon-gap-mult"
    , printf "%.12g" (ttpTaTrendConfidenceAroonGapMult p)
    , "--ta-reversion-confidence-roc-mult"
    , printf "%.12g" (ttpTaReversionConfidenceRocMult p)
    , "--ta-breakout-flow-disagreement-confidence"
    , printf "%.12g" (ttpTaBreakoutFlowDisagreementConfidence p)
    , "--ta-breakout-confidence-distance-mult"
    , printf "%.12g" (ttpTaBreakoutConfidenceDistanceMult p)
    , "--ta-sma-cross-confidence-spread-mult"
    , printf "%.12g" (ttpTaSmaCrossConfidenceSpreadMult p)
    , "--blend-softmax-scale"
    , printf "%.12g" (ttpBlendSoftmaxScale p)
    , "--blend-net-softmax-scale"
    , printf "%.12g" (ttpBlendNetSoftmaxScale p)
    , "--blend-edge-power"
    , printf "%.12g" (ttpBlendEdgePower p)
    , "--blend-smooth-alpha"
    , printf "%.12g" (ttpBlendSmoothAlpha p)
    , "--blend-hedge-eta"
    , printf "%.12g" (ttpBlendHedgeEta p)
    , "--blend-hedge-max-error"
    , printf "%.12g" (ttpBlendHedgeMaxError p)
    , "--blend-divergence-k"
    , printf "%.12g" (ttpBlendDivergenceK p)
    , "--blend-regime-high-vol-cutoff"
    , printf "%.12g" (ttpBlendRegimeHighVolCutoff p)
    , "--blend-regime-kalman-z-cutoff"
    , printf "%.12g" (ttpBlendRegimeKalmanZCutoff p)
    , "--blend-bandit-explore-scale"
    , printf "%.12g" (ttpBlendBanditExploreScale p)
    , "--blend-fractal-return-clamp"
    , printf "%.12g" (ttpBlendFractalReturnClamp p)
    , "--blend-fractal-aligned-gain"
    , printf "%.12g" (ttpBlendFractalAlignedGain p)
    , "--blend-fractal-conflict-gain"
    , printf "%.12g" (ttpBlendFractalConflictGain p)
    , "--blend-coherence-conflict-floor"
    , printf "%.12g" (ttpBlendCoherenceConflictFloor p)
    , "--blend-coherence-conflict-scale"
    , printf "%.12g" (ttpBlendCoherenceConflictScale p)
    , "--blend-coherence-boost-threshold"
    , printf "%.12g" (ttpBlendCoherenceBoostThreshold p)
    , "--blend-coherence-boost-gain"
    , printf "%.12g" (ttpBlendCoherenceBoostGain p)
    , "--blend-coherence-boost-span"
    , printf "%.12g" (ttpBlendCoherenceBoostSpan p)
    , "--blend-anchor-conflict-base"
    , printf "%.12g" (ttpBlendAnchorConflictBase p)
    , "--blend-anchor-conflict-scale"
    , printf "%.12g" (ttpBlendAnchorConflictScale p)
    , "--blend-anchor-aligned-scale"
    , printf "%.12g" (ttpBlendAnchorAlignedScale p)
    , "--blend-tension-conflict-shrink"
    , printf "%.12g" (ttpBlendTensionConflictShrink p)
    , "--blend-tension-neutral-shrink"
    , printf "%.12g" (ttpBlendTensionNeutralShrink p)
    , "--blend-entropy-conflict-floor"
    , printf "%.12g" (ttpBlendEntropyConflictFloor p)
    , "--blend-entropy-conflict-scale"
    , printf "%.12g" (ttpBlendEntropyConflictScale p)
    , "--blend-entropy-aligned-base"
    , printf "%.12g" (ttpBlendEntropyAlignedBase p)
    , "--blend-entropy-aligned-entropy-scale"
    , printf "%.12g" (ttpBlendEntropyAlignedEntropyScale p)
    , "--blend-phase-cancel-return-clamp"
    , printf "%.12g" (ttpBlendPhaseCancelReturnClamp p)
    , "--blend-phase-cancel-conflict-floor"
    , printf "%.12g" (ttpBlendPhaseCancelConflictFloor p)
    , "--blend-phase-cancel-conflict-scale"
    , printf "%.12g" (ttpBlendPhaseCancelConflictScale p)
    , "--blend-phase-cancel-alignment-scale"
    , printf "%.12g" (ttpBlendPhaseCancelAlignmentScale p)
    , "--signal-entry-edge-headroom-mult"
    , printf "%.12g" (ttpSignalEntryEdgeHeadroomMult p)
    , "--signal-entry-edge-spike-mult"
    , printf "%.12g" (ttpSignalEntryEdgeSpikeMult p)
    , "--signal-entry-edge-spike-cap"
    , printf "%.12g" (ttpSignalEntryEdgeSpikeCap p)
    , "--signal-entry-edge-spike-consecutive-limit"
    , show (ttpSignalEntryEdgeSpikeConsecutiveLimit p)
    , "--signal-trend-slack-mult"
    , printf "%.12g" (ttpSignalTrendSlackMult p)
    , "--signal-trend-slack-cap"
    , printf "%.12g" (ttpSignalTrendSlackCap p)
    , "--signal-directionality-lookback-bars"
    , show (ttpSignalDirectionalityLookbackBars p)
    , "--signal-directionality-chop-efficiency-max"
    , printf "%.12g" (ttpSignalDirectionalityChopEfficiencyMax p)
    , "--signal-directionality-mr-efficiency-max"
    , printf "%.12g" (ttpSignalDirectionalityMrEfficiencyMax p)
    , "--signal-directionality-weak-z-min"
    , printf "%.12g" (ttpSignalDirectionalityWeakZMin p)
    , "--signal-predictor-tracking-floor"
    , printf "%.12g" (ttpSignalPredictorTrackingFloor p)
    , "--predictor-gbdt-trees"
    , show (ttpPredictorGbdtTrees p)
    , "--predictor-gbdt-learning-rate"
    , printf "%.12g" (ttpPredictorGbdtLearningRate p)
    , "--predictor-calibration-ratio"
    , printf "%.12g" (ttpPredictorCalibrationRatio p)
    , "--predictor-conformal-alpha"
    , printf "%.12g" (ttpPredictorConformalAlpha p)
    , "--predictor-tcn-ridge-lambda"
    , printf "%.12g" (ttpPredictorTcnRidgeLambda p)
    , "--predictor-quantile-epochs"
    , show (ttpPredictorQuantileEpochs p)
    , "--predictor-quantile-learning-rate"
    , printf "%.12g" (ttpPredictorQuantileLearningRate p)
    , "--predictor-quantile-l2"
    , printf "%.12g" (ttpPredictorQuantileL2 p)
    , "--predictor-knn-max-examples"
    , show (ttpPredictorKnnMaxExamples p)
    , "--predictor-knn-k"
    , show (ttpPredictorKnnK p)
    , "--predictor-decision-tree-max-depth"
    , show (ttpPredictorDecisionTreeMaxDepth p)
    , "--predictor-decision-tree-min-leaf-size"
    , show (ttpPredictorDecisionTreeMinLeafSize p)
    , "--predictor-transformer-temperature"
    , printf "%.12g" (ttpPredictorTransformerTemperature p)
    , "--predictor-transformer-max-examples"
    , show (ttpPredictorTransformerMaxExamples p)
    , "--predictor-hmm-iterations"
    , show (ttpPredictorHmmIterations p)
    , "--vol-conf-volatility-evidence-max"
    , printf "%.12g" (ttpVolConfVolatilityEvidenceMax p)
    , "--vol-conf-low-vol-threshold"
    , printf "%.12g" (ttpVolConfLowVolThreshold p)
    , "--vol-conf-high-vol-threshold"
    , printf "%.12g" (ttpVolConfHighVolThreshold p)
    , "--vol-conf-weak-confidence-threshold"
    , printf "%.12g" (ttpVolConfWeakConfidenceThreshold p)
    , "--vol-conf-strong-confidence-threshold"
    , printf "%.12g" (ttpVolConfStrongConfidenceThreshold p)
    , "--vol-conf-low-medium-size"
    , printf "%.12g" (ttpVolConfLowMediumSizeMult p)
    , "--vol-conf-low-strong-size"
    , printf "%.12g" (ttpVolConfLowStrongSizeMult p)
    , "--vol-conf-medium-medium-size"
    , printf "%.12g" (ttpVolConfMediumMediumSizeMult p)
    , "--vol-conf-medium-strong-size"
    , printf "%.12g" (ttpVolConfMediumStrongSizeMult p)
    , "--vol-conf-high-strong-size"
    , printf "%.12g" (ttpVolConfHighStrongSizeMult p)
    , "--regime-adx-weight"
    , printf "%.12g" (ttpRegimeAdxWeight p)
    , "--regime-trend-threshold"
    , printf "%.12g" (ttpRegimeTrendThreshold p)
    , "--regime-range-threshold"
    , printf "%.12g" (ttpRegimeRangeThreshold p)
    , "--regime-trend-adx-floor"
    , printf "%.12g" (ttpRegimeTrendAdxFloor p)
    , "--regime-trend-adx-span"
    , printf "%.12g" (ttpRegimeTrendAdxSpan p)
    , "--regime-trend-aroon-floor"
    , printf "%.12g" (ttpRegimeTrendAroonFloor p)
    , "--regime-trend-aroon-span"
    , printf "%.12g" (ttpRegimeTrendAroonSpan p)
    , "--regime-trend-slope-floor"
    , printf "%.12g" (ttpRegimeTrendSlopeFloor p)
    , "--regime-trend-slope-span"
    , printf "%.12g" (ttpRegimeTrendSlopeSpan p)
    , "--regime-trend-aroon-weight"
    , printf "%.12g" (ttpRegimeTrendAroonWeight p)
    , "--regime-range-adx-ceiling"
    , printf "%.12g" (ttpRegimeRangeAdxCeiling p)
    , "--regime-range-adx-span"
    , printf "%.12g" (ttpRegimeRangeAdxSpan p)
    , "--regime-range-width-ceiling"
    , printf "%.12g" (ttpRegimeRangeWidthCeiling p)
    , "--regime-range-width-span"
    , printf "%.12g" (ttpRegimeRangeWidthSpan p)
    , "--regime-range-adx-weight"
    , printf "%.12g" (ttpRegimeRangeAdxWeight p)
    ]

technicalTrialParamsPairs :: TechnicalTrialParams -> [AT.Pair]
technicalTrialParamsPairs p =
    [ "taEntryOpenThreshold" .= ttpTaEntryOpenThreshold p
    , "taTrendAdxThreshold" .= ttpTaTrendAdxThreshold p
    , "taTrendStopAtrMult" .= ttpTaTrendStopAtrMult p
    , "taTrendTakeProfitAtrMult" .= ttpTaTrendTakeProfitAtrMult p
    , "taReversionRsiLower" .= ttpTaReversionRsiLower p
    , "taReversionRsiUpper" .= ttpTaReversionRsiUpper p
    , "taReversionStochLower" .= ttpTaReversionStochLower p
    , "taReversionStochUpper" .= ttpTaReversionStochUpper p
    , "taReversionEnvelopeLowerMult" .= ttpTaReversionEnvelopeLowerMult p
    , "taReversionEnvelopeUpperMult" .= ttpTaReversionEnvelopeUpperMult p
    , "taReversionStopAtrMult" .= ttpTaReversionStopAtrMult p
    , "taBreakoutMfiThreshold" .= ttpTaBreakoutMfiThreshold p
    , "taBreakoutStopAtrMult" .= ttpTaBreakoutStopAtrMult p
    , "taBreakoutTakeProfitAtrMult" .= ttpTaBreakoutTakeProfitAtrMult p
    , "taSmaCrossStopAtrMult" .= ttpTaSmaCrossStopAtrMult p
    , "taSmaCrossTakeProfitAtrMult" .= ttpTaSmaCrossTakeProfitAtrMult p
    , "taRegimeSwitchAdxThreshold" .= ttpTaRegimeSwitchAdxThreshold p
    , "taRegimeSwitchBollingerBandwidthThreshold" .= ttpTaRegimeSwitchBollingerBandwidthThreshold p
    , "taBestCandidateMinConfidence" .= ttpTaBestCandidateMinConfidence p
    , "taTrendConfidenceAdxOffset" .= ttpTaTrendConfidenceAdxOffset p
    , "taTrendConfidenceAdxScale" .= ttpTaTrendConfidenceAdxScale p
    , "taTrendConfidenceMaSpreadMult" .= ttpTaTrendConfidenceMaSpreadMult p
    , "taTrendConfidenceAroonGapMult" .= ttpTaTrendConfidenceAroonGapMult p
    , "taReversionConfidenceRocMult" .= ttpTaReversionConfidenceRocMult p
    , "taBreakoutFlowDisagreementConfidence" .= ttpTaBreakoutFlowDisagreementConfidence p
    , "taBreakoutConfidenceDistanceMult" .= ttpTaBreakoutConfidenceDistanceMult p
    , "taSmaCrossConfidenceSpreadMult" .= ttpTaSmaCrossConfidenceSpreadMult p
    , "blendSoftmaxScale" .= ttpBlendSoftmaxScale p
    , "blendNetSoftmaxScale" .= ttpBlendNetSoftmaxScale p
    , "blendEdgePower" .= ttpBlendEdgePower p
    , "blendSmoothAlpha" .= ttpBlendSmoothAlpha p
    , "blendHedgeEta" .= ttpBlendHedgeEta p
    , "blendHedgeMaxError" .= ttpBlendHedgeMaxError p
    , "blendDivergenceK" .= ttpBlendDivergenceK p
    , "blendRegimeHighVolCutoff" .= ttpBlendRegimeHighVolCutoff p
    , "blendRegimeKalmanZCutoff" .= ttpBlendRegimeKalmanZCutoff p
    , "blendBanditExploreScale" .= ttpBlendBanditExploreScale p
    , "blendFractalReturnClamp" .= ttpBlendFractalReturnClamp p
    , "blendFractalAlignedGain" .= ttpBlendFractalAlignedGain p
    , "blendFractalConflictGain" .= ttpBlendFractalConflictGain p
    , "blendCoherenceConflictFloor" .= ttpBlendCoherenceConflictFloor p
    , "blendCoherenceConflictScale" .= ttpBlendCoherenceConflictScale p
    , "blendCoherenceBoostThreshold" .= ttpBlendCoherenceBoostThreshold p
    , "blendCoherenceBoostGain" .= ttpBlendCoherenceBoostGain p
    , "blendCoherenceBoostSpan" .= ttpBlendCoherenceBoostSpan p
    , "blendAnchorConflictBase" .= ttpBlendAnchorConflictBase p
    , "blendAnchorConflictScale" .= ttpBlendAnchorConflictScale p
    , "blendAnchorAlignedScale" .= ttpBlendAnchorAlignedScale p
    , "blendTensionConflictShrink" .= ttpBlendTensionConflictShrink p
    , "blendTensionNeutralShrink" .= ttpBlendTensionNeutralShrink p
    , "blendEntropyConflictFloor" .= ttpBlendEntropyConflictFloor p
    , "blendEntropyConflictScale" .= ttpBlendEntropyConflictScale p
    , "blendEntropyAlignedBase" .= ttpBlendEntropyAlignedBase p
    , "blendEntropyAlignedEntropyScale" .= ttpBlendEntropyAlignedEntropyScale p
    , "blendPhaseCancelReturnClamp" .= ttpBlendPhaseCancelReturnClamp p
    , "blendPhaseCancelConflictFloor" .= ttpBlendPhaseCancelConflictFloor p
    , "blendPhaseCancelConflictScale" .= ttpBlendPhaseCancelConflictScale p
    , "blendPhaseCancelAlignmentScale" .= ttpBlendPhaseCancelAlignmentScale p
    , "signalEntryEdgeHeadroomMult" .= ttpSignalEntryEdgeHeadroomMult p
    , "signalEntryEdgeSpikeMult" .= ttpSignalEntryEdgeSpikeMult p
    , "signalEntryEdgeSpikeCap" .= ttpSignalEntryEdgeSpikeCap p
    , "signalEntryEdgeSpikeConsecutiveLimit" .= ttpSignalEntryEdgeSpikeConsecutiveLimit p
    , "signalTrendSlackMult" .= ttpSignalTrendSlackMult p
    , "signalTrendSlackCap" .= ttpSignalTrendSlackCap p
    , "signalDirectionalityLookbackBars" .= ttpSignalDirectionalityLookbackBars p
    , "signalDirectionalityChopEfficiencyMax" .= ttpSignalDirectionalityChopEfficiencyMax p
    , "signalDirectionalityMrEfficiencyMax" .= ttpSignalDirectionalityMrEfficiencyMax p
    , "signalDirectionalityWeakZMin" .= ttpSignalDirectionalityWeakZMin p
    , "signalPredictorTrackingFloor" .= ttpSignalPredictorTrackingFloor p
    , "predictorGbdtTrees" .= ttpPredictorGbdtTrees p
    , "predictorGbdtLearningRate" .= ttpPredictorGbdtLearningRate p
    , "predictorCalibrationRatio" .= ttpPredictorCalibrationRatio p
    , "predictorConformalAlpha" .= ttpPredictorConformalAlpha p
    , "predictorTcnRidgeLambda" .= ttpPredictorTcnRidgeLambda p
    , "predictorQuantileEpochs" .= ttpPredictorQuantileEpochs p
    , "predictorQuantileLearningRate" .= ttpPredictorQuantileLearningRate p
    , "predictorQuantileL2" .= ttpPredictorQuantileL2 p
    , "predictorKnnMaxExamples" .= ttpPredictorKnnMaxExamples p
    , "predictorKnnK" .= ttpPredictorKnnK p
    , "predictorDecisionTreeMaxDepth" .= ttpPredictorDecisionTreeMaxDepth p
    , "predictorDecisionTreeMinLeafSize" .= ttpPredictorDecisionTreeMinLeafSize p
    , "predictorTransformerTemperature" .= ttpPredictorTransformerTemperature p
    , "predictorTransformerMaxExamples" .= ttpPredictorTransformerMaxExamples p
    , "predictorHmmIterations" .= ttpPredictorHmmIterations p
    , "volConfVolatilityEvidenceMax" .= ttpVolConfVolatilityEvidenceMax p
    , "volConfLowVolThreshold" .= ttpVolConfLowVolThreshold p
    , "volConfHighVolThreshold" .= ttpVolConfHighVolThreshold p
    , "volConfWeakConfidenceThreshold" .= ttpVolConfWeakConfidenceThreshold p
    , "volConfStrongConfidenceThreshold" .= ttpVolConfStrongConfidenceThreshold p
    , "volConfLowMediumSizeMult" .= ttpVolConfLowMediumSizeMult p
    , "volConfLowStrongSizeMult" .= ttpVolConfLowStrongSizeMult p
    , "volConfMediumMediumSizeMult" .= ttpVolConfMediumMediumSizeMult p
    , "volConfMediumStrongSizeMult" .= ttpVolConfMediumStrongSizeMult p
    , "volConfHighStrongSizeMult" .= ttpVolConfHighStrongSizeMult p
    , "regimeAdxWeight" .= ttpRegimeAdxWeight p
    , "regimeTrendThreshold" .= ttpRegimeTrendThreshold p
    , "regimeRangeThreshold" .= ttpRegimeRangeThreshold p
    , "regimeTrendAdxFloor" .= ttpRegimeTrendAdxFloor p
    , "regimeTrendAdxSpan" .= ttpRegimeTrendAdxSpan p
    , "regimeTrendAroonFloor" .= ttpRegimeTrendAroonFloor p
    , "regimeTrendAroonSpan" .= ttpRegimeTrendAroonSpan p
    , "regimeTrendSlopeFloor" .= ttpRegimeTrendSlopeFloor p
    , "regimeTrendSlopeSpan" .= ttpRegimeTrendSlopeSpan p
    , "regimeTrendAroonWeight" .= ttpRegimeTrendAroonWeight p
    , "regimeRangeAdxCeiling" .= ttpRegimeRangeAdxCeiling p
    , "regimeRangeAdxSpan" .= ttpRegimeRangeAdxSpan p
    , "regimeRangeWidthCeiling" .= ttpRegimeRangeWidthCeiling p
    , "regimeRangeWidthSpan" .= ttpRegimeRangeWidthSpan p
    , "regimeRangeAdxWeight" .= ttpRegimeRangeAdxWeight p
    ]

normalizeSymbol :: Maybe String -> Maybe String
normalizeSymbol raw =
    case raw of
        Nothing -> Nothing
        Just v ->
            let s = map toUpper (trim v)
             in if null s then Nothing else Just s

normalizeHeaderName :: String -> String
normalizeHeaderName raw =
    let s = map toLower (trim raw)
     in [c | c <- s, isAlphaNum c]

radicalInverse :: Int -> Int -> Double
radicalInverse base = go 0 1
  where
    go acc f i
        | i <= 0 = acc
        | otherwise =
            let f' = f / fromIntegral base
                acc' = acc + f' * fromIntegral (i `mod` base)
             in go acc' f' (i `div` base)

sobolValue :: Int -> Double
sobolValue i =
    let v2 = radicalInverse 2 i
        v3 = radicalInverse 3 i
     in fmod1 (v2 + v3 * 0.5)
  where
    fmod1 x = x - fromIntegral (floor x :: Int)

sobolSeeds :: Int -> Int -> [Int]
sobolSeeds seed count =
    [ seed + round (sobolValue i * 1000000) + i * 997
    | i <- [1 .. count]
    ]

expandUser :: FilePath -> IO FilePath
expandUser path =
    case path of
        ('~' : '/' : rest) -> do
            home <- getHomeDirectory
            pure (home </> rest)
        "~" -> getHomeDirectory
        _ -> pure path

parseCsvLine :: String -> [String]
parseCsvLine input =
    let go acc field inQuotes chars =
            case chars of
                [] -> reverse (reverse field : acc)
                c : cs
                    | c == '"' ->
                        if inQuotes
                            then case cs of
                                '"' : rest -> go acc ('"' : field) True rest
                                _ -> go acc field False cs
                            else go acc field True cs
                    | c == ',' && not inQuotes -> go (reverse field : acc) [] False cs
                    | (c == '\n' || c == '\r') && not inQuotes -> reverse (reverse field : acc)
                    | otherwise -> go acc (c : field) inQuotes cs
     in case input of
            [] -> []
            _ -> go [] [] False input

detectHighLowColumns :: FilePath -> IO (Maybe String, Maybe String)
detectHighLowColumns path = do
    res <- try (BS.readFile path) :: IO (Either SomeException BS.ByteString)
    case res of
        Left _ -> pure (Nothing, Nothing)
        Right bs ->
            let (line, _) = BS.break (== 10) bs
             in if BS.null line
                    then pure (Nothing, Nothing)
                    else do
                        let headerText =
                                TE.decodeUtf8With TEE.lenientDecode line
                            header = parseCsvLine (T.unpack headerText)
                        if null header
                            then pure (Nothing, Nothing)
                            else do
                                let normalized = foldl' insertHeaderKey M.empty header
                                    high = firstMatch normalized ["high", "highprice", "highpx"]
                                    low = firstMatch normalized ["low", "lowprice", "lowpx"]
                                pure (high, low)
  where
    insertHeaderKey acc col =
        let key = normalizeHeaderName col
         in if null key || M.member key acc
                then acc
                else M.insert key col acc
    firstMatch acc keys =
        case keys of
            [] -> Nothing
            k : ks ->
                case M.lookup k acc of
                    Just v -> Just v
                    Nothing -> firstMatch acc ks

countCsvRows :: FilePath -> IO Int
countCsvRows path = do
    bs <- BS.readFile path
    case BS.unsnoc bs of
        Nothing -> pure 0
        Just (_, lastByte) -> do
            let newlines = BS.count 10 bs
                extra = if lastByte == 10 then 0 else 1
            pure (fromIntegral newlines + extra)

metricFloat :: Maybe (KM.KeyMap Value) -> String -> Double -> Double
metricFloat m key def =
    case m of
        Nothing -> def
        Just metrics ->
            case KM.lookup (Key.fromString key) metrics of
                Just (Number n) -> fromMaybe def (scientificToDouble n)
                _ -> def

metricInt :: Maybe (KM.KeyMap Value) -> String -> Int -> Int
metricInt m key def =
    case m of
        Nothing -> def
        Just metrics ->
            case KM.lookup (Key.fromString key) metrics of
                Just (Bool v) -> if v then 1 else 0
                Just (Number n) ->
                    fromMaybe def (scientificToBoundedInt n)
                _ -> def

metricProfitFactor :: Maybe (KM.KeyMap Value) -> Maybe Double
metricProfitFactor m =
    case m of
        Nothing -> Nothing
        Just metrics ->
            case KM.lookup (Key.fromString "profitFactor") metrics of
                Just (Number n) -> scientificToDouble n
                Just Null -> derived
                _ -> derived
  where
    derived =
        let grossProfit = metricFloat m "grossProfit" 0
            grossLoss = metricFloat m "grossLoss" 0
         in if grossLoss > 0
                then Just (grossProfit / grossLoss)
                else
                    if grossProfit > 0
                        then Nothing
                        else Just 0

scientificToDouble :: Scientific -> Maybe Double
scientificToDouble n =
    let d = toRealFloat n
     in if isInfinite d || isNaN d then Nothing else Just d

extractWalkForwardSummary :: Maybe Value -> Maybe (KM.KeyMap Value)
extractWalkForwardSummary raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    wf <- valueObjectAt (Object bt) "walkForward"
    valueObjectAt (Object wf) "summary"

valueObjectAt :: Value -> String -> Maybe (KM.KeyMap Value)
valueObjectAt val key =
    case val of
        Object obj ->
            case KM.lookup (Key.fromString key) obj of
                Just (Object v) -> Just v
                _ -> Nothing
        _ -> Nothing

data PriorTrial = PriorTrial
    { ptParams :: !(KM.KeyMap Value)
    , ptMetrics :: !(KM.KeyMap Value)
    , ptScore :: !Double
    , ptCreatedAtMs :: !(Maybe Int)
    , ptEligible :: !Bool
    , ptSymbol :: !(Maybe String)
    , ptPlatform :: !(Maybe String)
    , ptInterval :: !(Maybe String)
    , ptMethod :: !(Maybe String)
    }
    deriving (Eq, Show)

readOptimizerPriorTrials :: FilePath -> IO [PriorTrial]
readOptimizerPriorTrials rawPath = do
    let raw = trim rawPath
    if null raw
        then pure []
        else do
            path <- expandUser raw
            exists <- doesFileExist path
            if not exists
                then pure []
                else do
                    contentsOrErr <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
                    case contentsOrErr of
                        Left _ -> pure []
                        Right contents ->
                            case Aeson.eitherDecode contents :: Either String Value of
                                Right value -> pure (priorTrialsFromValue value)
                                Left _ ->
                                    pure $
                                        concatMap
                                            priorTrialsFromJsonLine
                                            (filter (not . BS.null) (BS8.lines (BL.toStrict contents)))

priorTrialsFromJsonLine :: BS.ByteString -> [PriorTrial]
priorTrialsFromJsonLine line =
    case Aeson.eitherDecodeStrict' line :: Either String Value of
        Right value -> priorTrialsFromValue value
        Left _ -> []

priorTrialsFromValue :: Value -> [PriorTrial]
priorTrialsFromValue value =
    case value of
        Object obj ->
            case KM.lookup (Key.fromString "combos") obj of
                Just (Array combos) -> mapMaybe priorTrialFromValue (V.toList combos)
                _ -> maybeToList (priorTrialFromObject obj)
        Array xs -> mapMaybe priorTrialFromValue (V.toList xs)
        _ -> []

priorTrialFromValue :: Value -> Maybe PriorTrial
priorTrialFromValue value =
    case value of
        Object obj -> priorTrialFromObject obj
        _ -> Nothing

priorTrialFromObject :: KM.KeyMap Value -> Maybe PriorTrial
priorTrialFromObject obj = do
    params <- objectField "params" obj
    let metrics = fromMaybe KM.empty (objectField "metrics" obj)
        eligible = fromMaybe True (boolField "eligible" obj)
        score =
            fromMaybe
                (metricFloat (Just metrics) "annualizedReturn" 0)
                (doubleField ["score", "finalEquity"] obj)
        createdAtMs = intField ["createdAtMs"] obj
        symbol =
            normalizeSymbol
                ( stringField ["binanceSymbol", "symbol", "binance_symbol"] params
                    <|> stringField ["binanceSymbol", "symbol", "binance_symbol"] obj
                )
        platform = fmap (map toLower) (stringField ["platform"] params)
        interval = stringField ["interval"] params
        method = fmap (map toLower) (stringField ["method"] params)
     in Just
            PriorTrial
                { ptParams = params
                , ptMetrics = metrics
                , ptScore = if isNaN score || isInfinite score then -(1 / 0) else score
                , ptCreatedAtMs = createdAtMs
                , ptEligible = eligible
                , ptSymbol = symbol
                , ptPlatform = platform
                , ptInterval = interval
                , ptMethod = method
                }

selectOptimizerPriorTrials :: OptimizerArgs -> Int -> Maybe String -> [String] -> [PriorTrial] -> [PriorTrial]
selectOptimizerPriorTrials args nowMs symbol intervals rawTrials =
    let eligible =
            filter
                (priorTrialMeetsEvidence args symbol intervals)
                rawTrials
        sorted = sortOn (Data.Ord.Down . priorTrialRankScore args nowMs) eligible
        fraction = clamp (oaPriorTopFraction args) 0 1
        total = length sorted
        diversityCap = max 0 (oaPriorDiversityMaxPerBucket args)
        requested =
            if fraction <= 0 || total <= 0
                then 0
                else
                    let byFraction = ceiling (fromIntegral total * fraction :: Double)
                     in max (max 0 (oaPriorMinSamples args)) byFraction
     in selectDiversePriorTrials diversityCap (min total requested) sorted

selectDiversePriorTrials :: Int -> Int -> [PriorTrial] -> [PriorTrial]
selectDiversePriorTrials diversityCap requested trials
    | requested <= 0 = []
    | diversityCap <= 0 = take requested trials
    | otherwise =
        let indexed = zip [0 :: Int ..] trials
            capped = takePriorTrialsWithBucketCap diversityCap requested indexed
            cappedIndexes = Set.fromList (map fst capped)
            backfillNeeded = requested - length capped
            backfill =
                if backfillNeeded <= 0
                    then []
                    else
                        take
                            backfillNeeded
                            [ trial
                            | (idx, trial) <- indexed
                            , Set.notMember idx cappedIndexes
                            ]
         in map snd capped ++ backfill

takePriorTrialsWithBucketCap :: Int -> Int -> [(Int, PriorTrial)] -> [(Int, PriorTrial)]
takePriorTrialsWithBucketCap diversityCap requested =
    reverse
        . (\(picked, _, _) -> picked)
        . foldl' pick ([], M.empty, 0)
  where
    pick acc@(_, _, pickedCount) _
        | pickedCount >= requested = acc
    pick (picked, bucketCounts, pickedCount) indexed@(_, trial) =
        let bucket = priorTrialDiversityBucket trial
            bucketCount = M.findWithDefault 0 bucket bucketCounts
         in if bucketCount >= diversityCap
                then (picked, bucketCounts, pickedCount)
                else
                    ( indexed : picked
                    , M.insert bucket (bucketCount + 1) bucketCounts
                    , pickedCount + 1
                    )

priorTrialDiversityBucket :: PriorTrial -> (Maybe String, Maybe String, Maybe String)
priorTrialDiversityBucket trial =
    (ptPlatform trial, ptInterval trial, ptMethod trial)

priorTrialRankScore :: OptimizerArgs -> Int -> PriorTrial -> Double
priorTrialRankScore args nowMs trial =
    let score =
            if KM.null (ptMetrics trial)
                then ptScore trial
                else
                    let objectiveScoreValue =
                            objectiveScoreWithConfig
                                (roiScoreConfigFromOptimizerArgs args)
                                (ptMetrics trial)
                                (oaObjective args)
                                (oaPenaltyMaxDrawdown args)
                                (oaPenaltyTurnover args)
                     in if isNaN objectiveScoreValue || isInfinite objectiveScoreValue
                            then ptScore trial
                            else objectiveScoreValue
     in ageAdjustedPriorScore (oaPriorAgeHalfLifeDays args) nowMs (ptCreatedAtMs trial) score

ageAdjustedPriorScore :: Double -> Int -> Maybe Int -> Double -> Double
ageAdjustedPriorScore halfLifeDays nowMs mCreatedAtMs score
    | halfLifeDays <= 0 = score
    | isNaN halfLifeDays || isInfinite halfLifeDays = score
    | otherwise =
        let decay = priorAgeDecayMultiplier halfLifeDays nowMs mCreatedAtMs
         in if score >= 0
                then score * decay
                else score / max 1e-9 decay

priorAgeDecayMultiplier :: Double -> Int -> Maybe Int -> Double
priorAgeDecayMultiplier halfLifeDays nowMs mCreatedAtMs =
    case mCreatedAtMs of
        Nothing -> 1
        Just createdAtMs ->
            let ageDays =
                    max 0 $
                        fromIntegral (max 0 (nowMs - createdAtMs))
                            / (1000 * 60 * 60 * 24 :: Double)
             in 0.5 ** (ageDays / halfLifeDays)

priorTrialDistributionSummary :: [PriorTrial] -> String
priorTrialDistributionSummary trials =
    let methodSummary = countSummary (fromMaybe "unknown" . ptMethod) trials
        intervalSummary = countSummary (fromMaybe "unknown" . ptInterval) trials
        wfCount = length (filter priorTrialHasWalkForwardSummary trials)
        createdAtCount = length (filter (isJust . ptCreatedAtMs) trials)
     in " methods="
            ++ methodSummary
            ++ " intervals="
            ++ intervalSummary
            ++ " walkForward="
            ++ show wfCount
            ++ "/"
            ++ show (length trials)
            ++ " createdAt="
            ++ show createdAtCount
            ++ "/"
            ++ show (length trials)

countSummary :: (PriorTrial -> String) -> [PriorTrial] -> String
countSummary pick trials =
    let add acc trial = M.insertWith (+) (pick trial) (1 :: Int) acc
        counts = sortOn (Data.Ord.Down . snd) (M.toList (foldl' add M.empty trials))
        render (key, n) = key ++ ":" ++ show n
     in intercalate "," (map render (take 8 counts))

priorTrialHasWalkForwardSummary :: PriorTrial -> Bool
priorTrialHasWalkForwardSummary trial =
    isJust (valueObjectAt (Object (ptMetrics trial)) "walkForwardSummary")

priorTrialMeetsEvidence :: OptimizerArgs -> Maybe String -> [String] -> PriorTrial -> Bool
priorTrialMeetsEvidence args symbol intervals trial =
    let metrics = Just (ptMetrics trial)
        roundTrips = metricInt metrics "roundTrips" 0
        tradeCount = metricInt metrics "tradeCount" 0
        activityCount = max roundTrips tradeCount
        exposure = metricFloat metrics "exposure" 0
        sharpe = metricFloat metrics "sharpe" 0
        annRet = metricFloat metrics "annualizedReturn" 0
        maxDd = max 0 (metricFloat metrics "maxDrawdown" 0)
        calmar = if maxDd <= 0 then annRet else annRet / max 1e-12 maxDd
        turnover = metricFloat metrics "turnover" 0
        winRate = metricFloat metrics "winRate" 0
        profitFactorOk =
            case metricProfitFactor metrics of
                Nothing -> True
                Just pf -> pf >= max 0 (oaMinProfitFactor args)
        symbolOk =
            case symbol of
                Nothing -> True
                Just sym -> maybe True (== sym) (ptSymbol trial)
        intervalOk =
            case ptInterval trial of
                Nothing -> True
                Just interval -> interval `elem` intervals
        minWfSharpeMean = max 0 (oaMinWfSharpeMean args)
        maxWfSharpeStd = max 0 (oaMaxWfSharpeStd args)
        walkForwardOk =
            case valueObjectAt (Object (ptMetrics trial)) "walkForwardSummary" of
                -- Prior rows are search hints. Legacy top-combo payloads often
                -- predate walk-forward metrics, but fresh trials still have to
                -- clear the active walk-forward gates before ranking/deployment.
                Nothing -> True
                Just wf ->
                    let wfSharpeMean = metricFloat (Just wf) "sharpeMean" 0
                        wfSharpeStd = metricFloat (Just wf) "sharpeStd" 0
                     in (minWfSharpeMean <= 0 || wfSharpeMean >= minWfSharpeMean)
                            && (maxWfSharpeStd <= 0 || wfSharpeStd <= maxWfSharpeStd)
     in ptEligible trial
            && symbolOk
            && intervalOk
            && (oaMinRoundTrips args <= 0 || activityCount >= oaMinRoundTrips args)
            && (oaMinWinRate args <= 0 || winRate >= oaMinWinRate args)
            && profitFactorOk
            && (oaMinExposure args <= 0 || exposure >= oaMinExposure args)
            && (oaMinSharpe args <= 0 || sharpe >= oaMinSharpe args)
            && (oaMinAnnualizedReturn args <= 0 || annRet >= oaMinAnnualizedReturn args)
            && (oaMinCalmar args <= 0 || calmar >= oaMinCalmar args)
            && (oaMaxTurnover args <= 0 || turnover <= oaMaxTurnover args)
            && walkForwardOk

objectField :: String -> KM.KeyMap Value -> Maybe (KM.KeyMap Value)
objectField key obj =
    case KM.lookup (Key.fromString key) obj of
        Just (Object value) -> Just value
        _ -> Nothing

stringField :: [String] -> KM.KeyMap Value -> Maybe String
stringField keys obj =
    listToMaybe (mapMaybe (`stringFieldOne` obj) keys)

stringFieldOne :: String -> KM.KeyMap Value -> Maybe String
stringFieldOne key obj =
    case KM.lookup (Key.fromString key) obj of
        Just (String value) ->
            let raw = trim (T.unpack value)
             in if null raw then Nothing else Just raw
        _ -> Nothing

doubleField :: [String] -> KM.KeyMap Value -> Maybe Double
doubleField keys obj =
    listToMaybe (mapMaybe (`doubleFieldOne` obj) keys)

doubleFieldOne :: String -> KM.KeyMap Value -> Maybe Double
doubleFieldOne key obj =
    KM.lookup (Key.fromString key) obj >>= coerceFloatValue

intField :: [String] -> KM.KeyMap Value -> Maybe Int
intField keys obj =
    listToMaybe (mapMaybe (`intFieldOne` obj) keys)

intFieldOne :: String -> KM.KeyMap Value -> Maybe Int
intFieldOne key obj =
    KM.lookup (Key.fromString key) obj >>= coerceIntValue

boolField :: String -> KM.KeyMap Value -> Maybe Bool
boolField key obj =
    case KM.lookup (Key.fromString key) obj of
        Just (Bool value) -> Just value
        _ -> Nothing

extractKellyLiteSummary :: Maybe Value -> Maybe (KM.KeyMap Value)
extractKellyLiteSummary raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    valueObjectAt (Object bt) "kellyLite"

latestSignalAction :: Maybe Value -> Maybe String
latestSignalAction raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    latest <- valueObjectAt (Object bt) "latestSignal"
    case KM.lookup (Key.fromString "action") latest of
        Just (String action) -> Just (T.unpack action)
        _ -> Nothing

diagnosticToken :: String -> String
diagnosticToken raw =
    let trimmed = trim raw
        body =
            case stripPrefix "HOLD (" trimmed of
                Just rest | not (null rest) && last rest == ')' -> init rest
                _ -> trimmed
        tokenChar c
            | isAlphaNum c = toUpper c
            | otherwise = '_'
        compacted = map head (group (map tokenChar body))
     in reverse (dropWhile (== '_') (reverse (dropWhile (== '_') compacted)))

activityFilterDiagnostic :: TrialResult -> Maybe String
activityFilterDiagnostic tr = do
    reason <- trFilterReason tr
    required <- stripPrefix "activityCount<" reason
    if null required || not (all isDigit required)
        then Nothing
        else
            let roundTrips = metricInt (trMetrics tr) "roundTrips" 0
                tradeCount = metricInt (trMetrics tr) "tradeCount" 0
                activityCount = max roundTrips tradeCount
                base =
                    [ "activity=" ++ show activityCount ++ "/" ++ required
                    , "roundTrips=" ++ show roundTrips
                    , "tradeCount=" ++ show tradeCount
                    ]
                latest =
                    case latestSignalAction (trStdoutJson tr) of
                        Just action ->
                            let token = diagnosticToken action
                             in ["latest=" ++ token | not (null token)]
                        Nothing -> []
             in Just (unwords (base ++ latest))

data OptimizerRecordsSummary = OptimizerRecordsSummary
    { orsRecords :: !Int
    , orsEligible :: !Int
    , orsActivityFiltered :: !Int
    , orsExposureFiltered :: !Int
    , orsOpenThresholdFiltered :: !Int
    , orsMinEdgeFiltered :: !Int
    , orsSharpeFiltered :: !Int
    , orsCalmarFiltered :: !Int
    , orsWalkForwardMissing :: !Int
    , orsWalkForwardSharpeMeanFiltered :: !Int
    , orsWalkForwardSharpeStdFiltered :: !Int
    , orsRecoveryFiltered :: !Int
    , orsTimeouts :: !Int
    }
    deriving (Eq, Show)

emptyOptimizerRecordsSummary :: OptimizerRecordsSummary
emptyOptimizerRecordsSummary =
    OptimizerRecordsSummary
        { orsRecords = 0
        , orsEligible = 0
        , orsActivityFiltered = 0
        , orsExposureFiltered = 0
        , orsOpenThresholdFiltered = 0
        , orsMinEdgeFiltered = 0
        , orsSharpeFiltered = 0
        , orsCalmarFiltered = 0
        , orsWalkForwardMissing = 0
        , orsWalkForwardSharpeMeanFiltered = 0
        , orsWalkForwardSharpeStdFiltered = 0
        , orsRecoveryFiltered = 0
        , orsTimeouts = 0
        }

optimizerRecordsSummaryJson :: OptimizerRecordsSummary -> Aeson.Value
optimizerRecordsSummaryJson s =
    object
        [ "records" .= orsRecords s
        , "eligible" .= orsEligible s
        , "activityFiltered" .= orsActivityFiltered s
        , "exposureFiltered" .= orsExposureFiltered s
        , "openThresholdFiltered" .= orsOpenThresholdFiltered s
        , "minEdgeFiltered" .= orsMinEdgeFiltered s
        , "sharpeFiltered" .= orsSharpeFiltered s
        , "calmarFiltered" .= orsCalmarFiltered s
        , "walkForwardMissing" .= orsWalkForwardMissing s
        , "walkForwardSharpeMeanFiltered" .= orsWalkForwardSharpeMeanFiltered s
        , "walkForwardSharpeStdFiltered" .= orsWalkForwardSharpeStdFiltered s
        , "recoveryFiltered" .= orsRecoveryFiltered s
        , "timeouts" .= orsTimeouts s
        ]

optimizerRecordsShouldRetryDiscovery :: OptimizerRecordsSummary -> Bool
optimizerRecordsShouldRetryDiscovery s =
    orsRecords s > 0
        && orsEligible s == 0
        && (orsRecoveryFiltered s > 0 || orsTimeouts s > 0)

readOptimizerRecordsSummary :: FilePath -> IO OptimizerRecordsSummary
readOptimizerRecordsSummary path = do
    exists <- doesFileExist path
    if not exists
        then pure emptyOptimizerRecordsSummary
        else do
            contentsOrErr <- try (BL.readFile path) :: IO (Either SomeException BL.ByteString)
            case contentsOrErr of
                Left _ -> pure emptyOptimizerRecordsSummary
                Right contents ->
                    pure $
                        foldl'
                            addLine
                            emptyOptimizerRecordsSummary
                            (filter (not . BS.null) (BS8.lines (BL.toStrict contents)))
  where
    addLine acc line =
        case Aeson.eitherDecodeStrict' line of
            Right (Aeson.Object obj) -> addRecord acc obj
            _ -> acc
    addRecord acc obj =
        let eligible = fromMaybe False (KM.lookup (Key.fromString "eligible") obj >>= AT.parseMaybe Aeson.parseJSON)
            reason =
                fromMaybe
                    ""
                    ( (KM.lookup (Key.fromString "filterReason") obj >>= AT.parseMaybe Aeson.parseJSON)
                        <|> (KM.lookup (Key.fromString "reason") obj >>= AT.parseMaybe Aeson.parseJSON)
                    )
            hasPrefix prefix = prefix `isPrefixOf` reason
            recoveryFiltered =
                any
                    hasPrefix
                    [ "activityCount<"
                    , "exposure<"
                    , "openThreshold>"
                    , "minEdge<"
                    , "winRate<"
                    , "profitFactor<"
                    , "kellyLiteExposure"
                    , "sharpe<"
                    , "annualizedReturn<"
                    , "calmar<"
                    , "turnover>"
                    , "walkForwardMissing"
                    , "wfSharpeMean<"
                    , "wfSharpeStd>"
                    ]
            records' = orsRecords acc + 1
            eligible' = orsEligible acc + if eligible then 1 else 0
            activity' = orsActivityFiltered acc + if hasPrefix "activityCount<" then 1 else 0
            exposure' = orsExposureFiltered acc + if hasPrefix "exposure<" then 1 else 0
            openThreshold' = orsOpenThresholdFiltered acc + if hasPrefix "openThreshold>" then 1 else 0
            minEdge' = orsMinEdgeFiltered acc + if hasPrefix "minEdge<" then 1 else 0
            sharpe' = orsSharpeFiltered acc + if hasPrefix "sharpe<" then 1 else 0
            calmar' = orsCalmarFiltered acc + if hasPrefix "calmar<" then 1 else 0
            walkForwardMissing' = orsWalkForwardMissing acc + if hasPrefix "walkForwardMissing" then 1 else 0
            walkForwardSharpeMean' = orsWalkForwardSharpeMeanFiltered acc + if hasPrefix "wfSharpeMean<" then 1 else 0
            walkForwardSharpeStd' = orsWalkForwardSharpeStdFiltered acc + if hasPrefix "wfSharpeStd>" then 1 else 0
            recoveryFiltered' = orsRecoveryFiltered acc + if recoveryFiltered then 1 else 0
            timeouts' = orsTimeouts acc + if hasPrefix "timeout>" then 1 else 0
         in acc
                { orsRecords = records'
                , orsEligible = eligible'
                , orsActivityFiltered = activity'
                , orsExposureFiltered = exposure'
                , orsOpenThresholdFiltered = openThreshold'
                , orsMinEdgeFiltered = minEdge'
                , orsSharpeFiltered = sharpe'
                , orsCalmarFiltered = calmar'
                , orsWalkForwardMissing = walkForwardMissing'
                , orsWalkForwardSharpeMeanFiltered = walkForwardSharpeMean'
                , orsWalkForwardSharpeStdFiltered = walkForwardSharpeStd'
                , orsRecoveryFiltered = recoveryFiltered'
                , orsTimeouts = timeouts'
                }

coerceFloatValue :: Value -> Maybe Double
coerceFloatValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToDouble n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case reads trimmed of
                        [(v, "")] -> if isInfinite v || isNaN v then Nothing else Just v
                        _ -> Nothing
        _ -> Nothing

coerceFloatValueLenient :: Value -> Maybe Double
coerceFloatValueLenient value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> Just (toRealFloat n)
        String s ->
            let trimmed = trim (T.unpack s)
                lowered = map toLower trimmed
             in if null trimmed
                    then Nothing
                    else case reads trimmed of
                        [(v, "")] -> Just v
                        _ ->
                            case lowered of
                                "nan" -> Just (0 / 0)
                                "+nan" -> Just (0 / 0)
                                "-nan" -> Just (0 / 0)
                                "inf" -> Just (1 / 0)
                                "+inf" -> Just (1 / 0)
                                "-inf" -> Just (-(1 / 0))
                                "infinity" -> Just (1 / 0)
                                "+infinity" -> Just (1 / 0)
                                "-infinity" -> Just (-(1 / 0))
                                _ -> Nothing
        _ -> Nothing

coerceIntValue :: Value -> Maybe Int
coerceIntValue value =
    case value of
        Null -> Nothing
        Bool v -> Just (if v then 1 else 0)
        Number n -> scientificToBoundedInt n
        String s ->
            let trimmed = trim (T.unpack s)
             in if null trimmed
                    then Nothing
                    else case readMaybe trimmed :: Maybe Scientific of
                        Just n -> scientificToBoundedInt n
                        _ -> Nothing
        _ -> Nothing

scientificToBoundedInt :: Scientific -> Maybe Int
scientificToBoundedInt = toBoundedInteger

sanitizeFinalEquity :: Double -> Double
sanitizeFinalEquity eq
    | isNaN eq || isInfinite eq || eq < 0 = 0
    | otherwise = eq

calcAnnualizedReturn :: Double -> Double -> Int -> Double
calcAnnualizedReturn finalEq periodsPerYear periods
    | periodsPerYear <= 0 = 0
    | periods <= 0 = 0
    | otherwise = finalEq ** (periodsPerYear / fromIntegral periods) - 1

metricsHasAnnualizedReturn :: KM.KeyMap Value -> Bool
metricsHasAnnualizedReturn metrics =
    case KM.lookup (Key.fromString "annualizedReturn") metrics >>= coerceFloatValue of
        Just v -> not (isNaN v || isInfinite v)
        Nothing -> False

ensureAnnualizedReturnMetrics :: Maybe (KM.KeyMap Value) -> Double -> Double -> Int -> Maybe (KM.KeyMap Value)
ensureAnnualizedReturnMetrics metrics finalEq periodsPerYear periods =
    let eq = sanitizeFinalEquity finalEq
        annRet = calcAnnualizedReturn eq periodsPerYear periods
        annVal = Aeson.toJSON annRet
        addMetric = KM.insert (Key.fromString "annualizedReturn") annVal
     in case metrics of
            Just m ->
                if metricsHasAnnualizedReturn m
                    then Just m
                    else Just (addMetric m)
            Nothing -> Just (KM.fromList [(Key.fromString "annualizedReturn", annVal)])

objectiveScore :: KM.KeyMap Value -> String -> Double -> Double -> Double
objectiveScore = objectiveScoreWithConfig defaultRoiScoreConfig

objectiveScoreWithConfig :: RoiScoreConfig -> KM.KeyMap Value -> String -> Double -> Double -> Double
objectiveScoreWithConfig roiCfg0 metrics objective penaltyMaxDd penaltyTurnover =
    let finalEq = metricFloat (Just metrics) "finalEquity" 0
        maxDd = metricFloat (Just metrics) "maxDrawdown" 0
        cvar95 = metricFloat (Just metrics) "cvar95" 0
        sharpe = metricFloat (Just metrics) "sharpe" 0
        annRet = metricFloat (Just metrics) "annualizedReturn" 0
        turnover = metricFloat (Just metrics) "turnover" 0
        maxDdN = max 0 maxDd
        cvar95N = max 0 cvar95
        turnoverN = max 0 turnover
        pDd = max 0 penaltyMaxDd
        pTurn = max 0 penaltyTurnover
        avgTradeReturn = metricFloat (Just metrics) "avgTradeReturn" 0
        avgHoldingPeriods = metricFloat (Just metrics) "avgHoldingPeriods" 0
        exposure = metricFloat (Just metrics) "exposure" 0
        roundTrips = metricInt (Just metrics) "roundTrips" 0
        tradeCount = metricInt (Just metrics) "tradeCount" 0
        activityCount = max roundTrips tradeCount
        roiCfg = sanitizeRoiScoreConfig roiCfg0
        sparseEvidencePenalty = roiEvidencePenaltyWithConfig roiCfg roundTrips activityCount exposure
        paybackBonus
            | avgHoldingPeriods <= 0 = 0
            | otherwise = paybackBonusForWithConfig roiCfg avgHoldingPeriods
        baseScore =
            case parseTuneObjective objective of
                Right TuneFinalEquity -> finalEq
                Right TuneAnnualizedEquity -> annRet
                Right TuneRoi ->
                    annRet - pDd * (maxDdN + cvar95N) - pTurn * turnoverN + rscExpectancyRewardWeight roiCfg * avgTradeReturn + paybackBonus
                Right TuneSharpe -> sharpe
                Right TuneCalmar ->
                    if maxDdN <= 0
                        then annRet
                        else annRet / max 1e-12 maxDdN
                Right TuneEquityDd -> finalEq - pDd * maxDdN
                Right TuneEquityDdTurnover -> finalEq - pDd * maxDdN - pTurn * turnoverN
                Left _ -> finalEq
     in baseScore - sparseEvidencePenalty

kellyLiteMetricFloat :: Maybe Value -> String -> Maybe Double
kellyLiteMetricFloat raw key = do
    report <- extractKellyLiteSummary raw
    value <- KM.lookup (Key.fromString key) report
    coerceFloatValue value

kellyLiteExposureContractReason :: Bool -> Maybe Value -> Double -> Double -> Maybe String
kellyLiteExposureContractReason enabled raw minReduction maxRatio
    | not enabled = Nothing
    | minReduction <= 0 && (maxRatio <= 0 || maxRatio >= 1) = Nothing
    | otherwise =
        case extractKellyLiteSummary raw of
            Nothing -> Just "kellyLiteExposureMissing"
            Just _ ->
                let reduction = fromMaybe 0 (kellyLiteMetricFloat raw "exposureReduction")
                    ratio = fromMaybe 1 (kellyLiteMetricFloat raw "exposureRatio")
                    uncappedExposure = fromMaybe 0 (kellyLiteMetricFloat raw "uncappedExposure")
                    ratioFilterEnabled = maxRatio > 0 && maxRatio < 1
                 in if (minReduction > 0 || ratioFilterEnabled) && uncappedExposure <= 0
                        then Just "kellyLiteUncappedExposure<=0"
                        else
                            if minReduction > 0 && reduction < minReduction
                                then Just (printf "kellyLiteExposureReduction<%.3f" minReduction)
                                else
                                    if ratioFilterEnabled && ratio > maxRatio
                                        then Just (printf "kellyLiteExposureRatio>%.3f" maxRatio)
                                        else Nothing

kellyLiteExposureFilterReason :: TrialParams -> Maybe Value -> Double -> Double -> Maybe String
kellyLiteExposureFilterReason params =
    kellyLiteExposureContractReason (tpKellyLiteSizing params)

normalizeObjectiveCode :: String -> Either String String
normalizeObjectiveCode raw = tuneObjectiveCode <$> parseTuneObjective raw

extractOperations :: Maybe Value -> Maybe [Value]
extractOperations raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    case KM.lookup (Key.fromString "trades") bt of
        Just (Array trades) ->
            let ops = mapMaybe tradeToOp (V.toList trades)
             in if null ops then Nothing else Just ops
        _ -> Nothing
  where
    tradeToOp value =
        case value of
            Object trade -> do
                entryIdx <- KM.lookup (Key.fromString "entryIndex") trade >>= coerceIntValue
                exitIdx <- KM.lookup (Key.fromString "exitIndex") trade >>= coerceIntValue
                let entryEquity = KM.lookup (Key.fromString "entryEquity") trade >>= coerceFloatValue
                    exitEquity = KM.lookup (Key.fromString "exitEquity") trade >>= coerceFloatValue
                    retVal = KM.lookup (Key.fromString "return") trade >>= coerceFloatValue
                    holding = KM.lookup (Key.fromString "holdingPeriods") trade >>= coerceIntValue
                    exitReason =
                        case KM.lookup (Key.fromString "exitReason") trade of
                            Just (String s) -> Just (T.unpack s)
                            _ -> Nothing
                Just
                    ( object
                        [ "entryIndex" .= entryIdx
                        , "exitIndex" .= exitIdx
                        , "entryEquity" .= entryEquity
                        , "exitEquity" .= exitEquity
                        , "return" .= retVal
                        , "holdingPeriods" .= holding
                        , "exitReason" .= exitReason
                        ]
                    )
            _ -> Nothing

coerceFloatArray :: Value -> Maybe [Double]
coerceFloatArray value =
    case value of
        Array xs -> traverse coerceFloatValue (V.toList xs)
        _ -> Nothing

trialCloseTimingReport :: TrialParams -> Maybe String -> Maybe Value -> ComboCloseTimingReport
trialCloseTimingReport params symbolLabel =
    closeTimingReportFromBacktest (trialCloseTimingComboId params symbolLabel)

trialCloseTimingComboId :: TrialParams -> Maybe String -> String
trialCloseTimingComboId params symbolLabel =
    intercalate
        ":"
        ( filter
            (not . null)
            [ fromMaybe "" (tpPlatform params)
            , fromMaybe "" symbolLabel
            , tpInterval params
            , tpMethod params
            ]
        )

extractCloseTimingInputs :: String -> Maybe Value -> Maybe ([Double], [ComboTrade])
extractCloseTimingInputs comboId raw = do
    v <- raw
    bt <- valueObjectAt v "backtest"
    pricesValue <- KM.lookup (Key.fromString "prices") bt
    tradesValue <- KM.lookup (Key.fromString "trades") bt
    prices <- coerceFloatArray pricesValue
    let positions =
            case KM.lookup (Key.fromString "positions") bt of
                Just value -> fromMaybe [] (coerceFloatArray value)
                Nothing -> []
    trades <- coerceCloseTimingTrades comboId prices positions tradesValue
    pure (prices, trades)

coerceCloseTimingTrades :: String -> [Double] -> [Double] -> Value -> Maybe [ComboTrade]
coerceCloseTimingTrades comboId prices positions value =
    case value of
        Array trades -> Just (mapMaybe (coerceCloseTimingTrade comboId prices positions) (V.toList trades))
        _ -> Nothing

coerceCloseTimingTrade :: String -> [Double] -> [Double] -> Value -> Maybe ComboTrade
coerceCloseTimingTrade comboId prices positions value =
    case value of
        Object trade -> do
            entryIdx <- KM.lookup (Key.fromString "entryIndex") trade >>= coerceIntValue
            exitIdx <- KM.lookup (Key.fromString "exitIndex") trade >>= coerceIntValue
            entryPrice <- safeAtList prices entryIdx
            let side = resolveCloseTimingTradeSide trade entryIdx exitIdx prices positions
            if entryPrice > 0 && side /= 0
                then
                    Just
                        ComboTrade
                            { ctComboId = comboId
                            , ctEntryIndex = entryIdx
                            , ctExitIndex = exitIdx
                            , ctEntryPrice = entryPrice
                            , ctSide = side
                            }
                else Nothing
        _ -> Nothing

resolveCloseTimingTradeSide :: KM.KeyMap Value -> Int -> Int -> [Double] -> [Double] -> Double
resolveCloseTimingTradeSide trade entryIdx exitIdx prices positions =
    fromMaybe 1 (sideFromPositions entryIdx positions <|> sideFromReturn trade entryIdx exitIdx prices)

sideFromPositions :: Int -> [Double] -> Maybe Double
sideFromPositions entryIdx positions = do
    pos <- safeAtList positions entryIdx
    let side = signum pos
    if isNaN side || isInfinite side || side == 0
        then Nothing
        else Just side

sideFromReturn :: KM.KeyMap Value -> Int -> Int -> [Double] -> Maybe Double
sideFromReturn trade entryIdx exitIdx prices = do
    retVal <- KM.lookup (Key.fromString "return") trade >>= coerceFloatValue
    entryPrice <- safeAtList prices entryIdx
    exitPrice <- safeAtList prices exitIdx
    inferTradeSideFromReturn entryPrice exitPrice retVal

inferTradeSideFromReturn :: Double -> Double -> Double -> Maybe Double
inferTradeSideFromReturn entryPrice exitPrice retVal
    | entryPrice <= 0 || exitPrice <= 0 = Nothing
    | isNaN retVal || isInfinite retVal = Nothing
    | otherwise =
        let priceMove = signum (exitPrice - entryPrice)
            retMove = signum retVal
            side = retMove * priceMove
         in if priceMove == 0 || retMove == 0 || side == 0 then Nothing else Just side

safeAtList :: [a] -> Int -> Maybe a
safeAtList xs idx
    | idx < 0 = Nothing
    | otherwise = go xs idx
  where
    go [] _ = Nothing
    go (y : ys) n
        | n == 0 = Just y
        | otherwise = go ys (n - 1)

closeTimingReportFromBacktest :: String -> Maybe Value -> ComboCloseTimingReport
closeTimingReportFromBacktest comboId raw =
    case extractCloseTimingInputs comboId raw of
        Just (prices, trades) -> analyzeComboCloseTiming comboId prices trades
        Nothing -> analyzeComboCloseTiming comboId [] []

appliedCloseTimingMaxHoldBars :: Maybe Int -> ComboCloseTimingReport -> Maybe Int
appliedCloseTimingMaxHoldBars currentMaxHoldBars report =
    cctrRecommendedMaxHoldBars report <|> currentMaxHoldBars

applyCloseTimingMetrics ::
    Maybe (KM.KeyMap Value) ->
    Maybe Int ->
    Maybe Int ->
    ComboCloseTimingReport ->
    Maybe (KM.KeyMap Value)
applyCloseTimingMetrics metrics currentMaxHoldBars appliedMaxHoldBars report =
    let base = fromMaybe KM.empty metrics
        reportValue = closeTimingReportToValue currentMaxHoldBars appliedMaxHoldBars report
     in Just (KM.insert (Key.fromString "closeTiming") reportValue base)

applyWalkForwardSummaryMetrics ::
    Maybe (KM.KeyMap Value) ->
    Maybe Value ->
    Maybe (KM.KeyMap Value)
applyWalkForwardSummaryMetrics metrics raw =
    case extractWalkForwardSummary raw of
        Nothing -> metrics
        Just summary ->
            Just (KM.insert (Key.fromString "walkForwardSummary") (Object summary) (fromMaybe KM.empty metrics))

closeTimingReportToValue :: Maybe Int -> Maybe Int -> ComboCloseTimingReport -> Value
closeTimingReportToValue currentMaxHoldBars appliedMaxHoldBars report =
    object
        [ "comboId" .= cctrComboId report
        , "sampleCount" .= cctrSampleCount report
        , "minimumSampleCount" .= minimumCloseTimingSamples
        , "positiveLiftSampleCount" .= cctrPositiveLiftSampleCount report
        , "minimumPositiveLiftSampleCount" .= minimumPositiveLiftSupportSamples
        , "medianRatio" .= cctrMedianRatio report
        , "q25Ratio" .= cctrQ25Ratio report
        , "q75Ratio" .= cctrQ75Ratio report
        , "madRatio" .= cctrMadRatio report
        , "meanLift" .= cctrMeanLift report
        , "medianLift" .= cctrMedianLift report
        , "medianObservedDuration" .= cctrMedianObservedDuration report
        , "medianOptimalDuration" .= cctrMedianOptimalDuration report
        , "q75OptimalDuration" .= cctrQ75OptimalDuration report
        , "recommendedMaxHoldBars" .= cctrRecommendedMaxHoldBars report
        , "originalMaxHoldBars" .= currentMaxHoldBars
        , "appliedMaxHoldBars" .= appliedMaxHoldBars
        , "positiveLift" .= maybe False (> 0) (cctrMedianLift report)
        ]

parsePlatforms :: String -> Either String [String]
parsePlatforms raw =
    let cleaned = map (\c -> if c == ',' then ' ' else c) raw
        parts = filter (not . null) (words (map toLower (trim cleaned)))
        allowed = ["binance", "coinbase", "kraken", "poloniex"]
     in if null parts
            then Left "no platforms provided"
            else collect allowed [] parts
  where
    collect _ acc [] = Right (reverse acc)
    collect allowed acc (p : ps)
        | p `elem` allowed =
            if p `elem` acc
                then collect allowed acc ps
                else collect allowed (p : acc) ps
        | otherwise =
            Left
                ( "invalid platform: "
                    ++ p
                    ++ " (expected "
                    ++ intercalate ", " (sort allowed)
                    ++ ")"
                )

resolveSourceLabel :: Maybe String -> String -> String -> String
resolveSourceLabel platform dataSource override
    | not (null override) = override
    | otherwise = fromMaybe dataSource platform

pickIntervals :: [String] -> String -> Int -> [String]
pickIntervals intervals lookbackWindow maxBarsCap =
    let keep itv =
            case lookbackBarsFrom itv lookbackWindow of
                Left _ -> False
                Right lb -> lb >= 2 && lb + 3 <= maxBarsCap
     in filter keep intervals

resolvePlatformIntervals :: [String] -> [String] -> String -> Int -> Either String [(String, [String])]
resolvePlatformIntervals platforms intervalsRaw lookbackWindow maxBarsCap =
    traverse resolveOne platforms
  where
    resolveOne platform =
        let supported = platformIntervals (platformFromString platform)
            filtered =
                if null intervalsRaw
                    then supported
                    else filter (`elem` supported) intervalsRaw
            intervals = pickIntervals filtered lookbackWindow maxBarsCap
         in if null intervals
                then
                    let supportedStr = if null supported then "none" else intercalate ", " supported
                     in Left ("No valid intervals for platform=" ++ platform ++ " (supported: " ++ supportedStr ++ ").")
                else Right (platform, intervals)

platformFromString :: String -> Platform
platformFromString platform =
    case platform of
        "coinbase" -> PlatformCoinbase
        "kraken" -> PlatformKraken
        "poloniex" -> PlatformPoloniex
        _ -> PlatformBinance

data OptimizerArgs = OptimizerArgs
    { oaData :: !(Maybe FilePath)
    , oaBinanceSymbol :: !(Maybe String)
    , oaSymbolLabel :: !String
    , oaSourceLabel :: !String
    , oaPriceColumn :: !String
    , oaHighColumn :: !String
    , oaLowColumn :: !String
    , oaLookbackWindow :: !String
    , oaBacktestRatio :: !Double
    , oaTuneRatio :: !Double
    , oaTrials :: !Int
    , oaSeed :: !Int
    , oaSeedTrials :: !(Maybe Int)
    , oaSeedRatio :: !(Maybe Double)
    , oaSurvivorFraction :: !Double
    , oaSurvivorParentActivityFloor :: !Int
    , oaSurvivorParentAnnualizedReturnFloor :: !Double
    , oaPerturbScaleDouble :: !Double
    , oaPerturbScaleInt :: !Int
    , oaEarlyStopNoImprove :: !Int
    , oaTimeoutSec :: !Double
    , oaOutput :: !String
    , oaAppend :: !Bool
    , oaBinary :: !String
    , oaNoSweepThreshold :: !Bool
    , oaDisableLstmPersistence :: !Bool
    , oaTopJson :: !String
    , oaPriorJson :: !String
    , oaPriorSampleProb :: !Double
    , oaPriorMethodSampleProb :: !Double
    , oaPriorRankBias :: !Double
    , oaPriorTopFraction :: !Double
    , oaPriorMinSamples :: !Int
    , oaPriorPerturbScaleDouble :: !Double
    , oaPriorPerturbScaleInt :: !Int
    , oaPriorAgeHalfLifeDays :: !Double
    , oaPriorDiversityMaxPerBucket :: !Int
    , oaPriorSeedCount :: !Int
    , oaQuality :: !Bool
    , oaQualityMinTrials :: !Int
    , oaQualityMaxEpochs :: !Int
    , oaQualityMaxHiddenSize :: !Int
    , oaAutoHighLow :: !Bool
    , oaObjective :: !String
    , oaPenaltyMaxDrawdown :: !Double
    , oaPenaltyTurnover :: !Double
    , oaRoiScoreExpectancyWeight :: !Double
    , oaRoiScorePaybackCap :: !Double
    , oaRoiScoreMinActivity :: !Int
    , oaRoiScoreMinExposure :: !Double
    , oaRoiScoreZeroRoundTripPenalty :: !Double
    , oaRoiScoreLowRoundTripPenalty :: !Double
    , oaRoiScoreZeroActivityPenalty :: !Double
    , oaRoiScoreLowActivityPenalty :: !Double
    , oaRoiScoreZeroExposurePenalty :: !Double
    , oaRoiScoreLowExposurePenalty :: !Double
    , oaRoiScoreLowExposureGapPenalty :: !Double
    , oaMinAnnualizedReturn :: !Double
    , oaMinCalmar :: !Double
    , oaMaxTurnover :: !Double
    , oaMinRoundTrips :: !Int
    , oaMinWinRate :: !Double
    , oaMinProfitFactor :: !Double
    , oaMinExposure :: !Double
    , oaMinKellyLiteExposureReduction :: !Double
    , oaMaxKellyLiteExposureRatio :: !Double
    , oaMinSharpe :: !Double
    , oaMinWfSharpeMean :: !Double
    , oaMaxWfSharpeStd :: !Double
    , oaTuneObjective :: !String
    , oaTunePenaltyMaxDrawdown :: !Double
    , oaTunePenaltyTurnover :: !Double
    , oaTuneMaxThresholdCandidates :: !Int
    , oaSensorVarianceEwmaAlpha :: !Double
    , oaKalmanSensorCorrelationInflation :: !Double
    , oaKalmanInnovationInflationThreshold :: !Double
    , oaKalmanInnovationInflationMax :: !Double
    , oaLstmAdamBeta1 :: !Double
    , oaLstmAdamBeta2 :: !Double
    , oaLstmAdamEps :: !Double
    , oaTuneStressVolMult :: !Double
    , oaTuneStressShock :: !Double
    , oaTuneStressWeight :: !Double
    , oaTuneStressVolMultMin :: !(Maybe Double)
    , oaTuneStressVolMultMax :: !(Maybe Double)
    , oaTuneStressShockMin :: !(Maybe Double)
    , oaTuneStressShockMax :: !(Maybe Double)
    , oaTuneStressWeightMin :: !(Maybe Double)
    , oaTuneStressWeightMax :: !(Maybe Double)
    , oaWalkForwardFoldsMin :: !Int
    , oaWalkForwardFoldsMax :: !Int
    , oaWalkForwardEmbargoBarsMin :: !Int
    , oaWalkForwardEmbargoBarsMax :: !Int
    , oaInterval :: !(Maybe String)
    , oaIntervals :: !(Maybe String)
    , oaPlatform :: !(Maybe String)
    , oaPlatforms :: !(Maybe String)
    , oaBinanceFutures :: !Bool
    , oaBarsMin :: !Int
    , oaBarsMax :: !Int
    , oaBarsAutoProb :: !Double
    , oaBarsDistribution :: !String
    , oaEpochsMin :: !Int
    , oaEpochsMax :: !Int
    , oaSlippageMax :: !Double
    , oaSpreadMax :: !Double
    , oaFeeMin :: !Double
    , oaFeeMax :: !Double
    , oaFundingRateMin :: !Double
    , oaFundingRateMax :: !Double
    , oaPFundingBySide :: !Double
    , oaPFundingOnOpen :: !Double
    , oaRebalanceBarsMin :: !Int
    , oaRebalanceBarsMax :: !Int
    , oaRebalanceThresholdMin :: !Double
    , oaRebalanceThresholdMax :: !Double
    , oaRebalanceCostMultMin :: !Double
    , oaRebalanceCostMultMax :: !Double
    , oaPRebalanceGlobal :: !Double
    , oaPRebalanceResetOnSignal :: !Double
    , oaOpenThresholdMin :: !Double
    , oaOpenThresholdMax :: !Double
    , oaOpenThresholdMaxExplicit :: !Bool
    , oaCloseThresholdMin :: !Double
    , oaCloseThresholdMax :: !Double
    , oaCloseThresholdMaxExplicit :: !Bool
    , oaMinHoldBarsMin :: !Int
    , oaMinHoldBarsMax :: !Int
    , oaCooldownBarsMin :: !Int
    , oaCooldownBarsMax :: !Int
    , oaMaxHoldBarsMin :: !Int
    , oaMaxHoldBarsMax :: !Int
    , oaMinEdgeMin :: !Double
    , oaMinEdgeMax :: !Double
    , oaMinSignalToNoiseMin :: !Double
    , oaMinSignalToNoiseMax :: !Double
    , oaSnrSizeWeightMin :: !Double
    , oaSnrSizeWeightMax :: !Double
    , oaPThresholdFactor :: !Double
    , oaThresholdFactorAlphaMin :: !Double
    , oaThresholdFactorAlphaMax :: !Double
    , oaThresholdFactorMinMin :: !Double
    , oaThresholdFactorMinMax :: !Double
    , oaThresholdFactorMaxMin :: !Double
    , oaThresholdFactorMaxMax :: !Double
    , oaThresholdFactorFloorMin :: !Double
    , oaThresholdFactorFloorMax :: !Double
    , oaThresholdFactorWeightMin :: !Double
    , oaThresholdFactorWeightMax :: !Double
    , oaEdgeBufferMin :: !Double
    , oaEdgeBufferMax :: !Double
    , oaPCostAwareEdge :: !Double
    , oaTrendLookbackMin :: !Int
    , oaTrendLookbackMax :: !Int
    , oaPLongShort :: !Double
    , oaPIntrabarTakeProfitFirst :: !Double
    , oaPTriLayer :: !Double
    , oaTriLayerFastMultMin :: !Double
    , oaTriLayerFastMultMax :: !Double
    , oaTriLayerSlowMultMin :: !Double
    , oaTriLayerSlowMultMax :: !Double
    , oaPTriLayerPriceAction :: !Double
    , oaTriLayerCloudPaddingMin :: !Double
    , oaTriLayerCloudPaddingMax :: !Double
    , oaTriLayerCloudSlopeMin :: !Double
    , oaTriLayerCloudSlopeMax :: !Double
    , oaTriLayerCloudWidthMin :: !Double
    , oaTriLayerCloudWidthMax :: !Double
    , oaTriLayerTouchLookbackMin :: !Int
    , oaTriLayerTouchLookbackMax :: !Int
    , oaTriLayerPriceActionBodyMin :: !Double
    , oaTriLayerPriceActionBodyMax :: !Double
    , oaTriLayerPriceActionWickRatioMin :: !Double
    , oaTriLayerPriceActionWickRatioMax :: !Double
    , oaTriLayerPriceActionOppositeWickMaxMin :: !Double
    , oaTriLayerPriceActionOppositeWickMaxMax :: !Double
    , oaTriLayerPriceActionBodyToleranceMin :: !Double
    , oaTriLayerPriceActionBodyToleranceMax :: !Double
    , oaTriLayerExitOnSlow :: !Bool
    , oaKalmanBandLookbackMin :: !Int
    , oaKalmanBandLookbackMax :: !Int
    , oaKalmanBandStdMultMin :: !Double
    , oaKalmanBandStdMultMax :: !Double
    , oaLstmExitFlipBarsMin :: !Int
    , oaLstmExitFlipBarsMax :: !Int
    , oaLstmExitFlipGraceBarsMin :: !Int
    , oaLstmExitFlipGraceBarsMax :: !Int
    , oaLstmExitFlipStrong :: !Bool
    , oaLstmConfidenceSoftMin :: !Double
    , oaLstmConfidenceSoftMax :: !Double
    , oaLstmConfidenceHardMin :: !Double
    , oaLstmConfidenceHardMax :: !Double
    , oaProtectionMinConfidenceMin :: !Double
    , oaProtectionMinConfidenceMax :: !Double
    , oaKalmanDtMin :: !Double
    , oaKalmanDtMax :: !Double
    , oaKalmanProcessVarMin :: !Double
    , oaKalmanProcessVarMax :: !Double
    , oaKalmanMeasurementVarMin :: !Double
    , oaKalmanMeasurementVarMax :: !Double
    , oaKalmanPhysicsBarsMin :: !Int
    , oaKalmanPhysicsBarsMax :: !Int
    , oaKalmanPhysicsBacktestRatioMin :: !Double
    , oaKalmanPhysicsBacktestRatioMax :: !Double
    , oaKalmanPhysicsVolumeEwmaAlphaMin :: !Double
    , oaKalmanPhysicsVolumeEwmaAlphaMax :: !Double
    , oaKalmanPhysicsVolumeSignalClampMin :: !Double
    , oaKalmanPhysicsVolumeSignalClampMax :: !Double
    , oaKalmanPhysicsCloseBiasScaleMin :: !Double
    , oaKalmanPhysicsCloseBiasScaleMax :: !Double
    , oaKalmanZMinMin :: !Double
    , oaKalmanZMinMax :: !Double
    , oaKalmanZMaxMin :: !Double
    , oaKalmanZMaxMax :: !Double
    , oaKalmanMarketTopNMin :: !Int
    , oaKalmanMarketTopNMax :: !Int
    , oaPDisableMaxHighVolProb :: !Double
    , oaMaxHighVolProbMin :: !Double
    , oaMaxHighVolProbMax :: !Double
    , oaPDisableMaxConformalWidth :: !Double
    , oaMaxConformalWidthMin :: !Double
    , oaMaxConformalWidthMax :: !Double
    , oaPDisableMaxQuantileWidth :: !Double
    , oaMaxQuantileWidthMin :: !Double
    , oaMaxQuantileWidthMax :: !Double
    , oaPConfirmConformal :: !Double
    , oaPConfirmQuantiles :: !Double
    , oaPConfidenceSizing :: !Double
    , oaMinPositionSizeMin :: !Double
    , oaMinPositionSizeMax :: !Double
    , oaMaxPositionSizeMin :: !Double
    , oaMaxPositionSizeMax :: !Double
    , oaVolTargetMin :: !Double
    , oaVolTargetMax :: !Double
    , oaPDisableVolTarget :: !Double
    , oaVolLookbackMin :: !Int
    , oaVolLookbackMax :: !Int
    , oaVolEwmaAlphaMin :: !Double
    , oaVolEwmaAlphaMax :: !Double
    , oaPDisableVolEwmaAlpha :: !Double
    , oaVolFloorMin :: !Double
    , oaVolFloorMax :: !Double
    , oaVolScaleMaxMin :: !Double
    , oaVolScaleMaxMax :: !Double
    , oaMaxVolatilityMin :: !Double
    , oaMaxVolatilityMax :: !Double
    , oaPDisableMaxVolatility :: !Double
    , oaPeriodsPerYearMin :: !Double
    , oaPeriodsPerYearMax :: !Double
    , oaStopMin :: !Double
    , oaStopMax :: !Double
    , oaTpMin :: !Double
    , oaTpMax :: !Double
    , oaTrailMin :: !Double
    , oaTrailMax :: !Double
    , oaPDisableStop :: !Double
    , oaPDisableTp :: !Double
    , oaPDisableTrail :: !Double
    , oaStopVolMultMin :: !Double
    , oaStopVolMultMax :: !Double
    , oaTpVolMultMin :: !Double
    , oaTpVolMultMax :: !Double
    , oaTrailVolMultMin :: !Double
    , oaTrailVolMultMax :: !Double
    , oaPDisableStopVolMult :: !Double
    , oaPDisableTpVolMult :: !Double
    , oaPDisableTrailVolMult :: !Double
    , oaRiskPerTradeMin :: !Double
    , oaRiskPerTradeMax :: !Double
    , oaPDisableRiskPerTrade :: !Double
    , oaPDisableMaxDd :: !Double
    , oaPDisableMaxDl :: !Double
    , oaPDisableMaxWl :: !Double
    , oaPDisableMaxOe :: !Double
    , oaMaxDdMin :: !Double
    , oaMaxDdMax :: !Double
    , oaMaxDlMin :: !Double
    , oaMaxDlMax :: !Double
    , oaMaxWlMin :: !Double
    , oaMaxWlMax :: !Double
    , oaMaxOeMin :: !Int
    , oaMaxOeMax :: !Int
    , oaMethodWeight11 :: !Double
    , oaMethodWeight10 :: !Double
    , oaMethodWeight01 :: !Double
    , oaMethodWeightBlend :: !Double
    , oaMethodWeightConfBlend :: !Double
    , oaMethodWeightConfPick :: !Double
    , oaMethodWeightConformalClip :: !Double
    , oaMethodWeightCostPick :: !Double
    , oaMethodWeightHarmonicBlend :: !Double
    , oaMethodWeightDisagreementGuard :: !Double
    , oaMethodWeightMedianBlend :: !Double
    , oaMethodWeightNeutralGuard :: !Double
    , oaMethodWeightRiskParityBlend :: !Double
    , oaMethodWeightConsensusBoost :: !Double
    , oaMethodWeightAnchorBlend :: !Double
    , oaMethodWeightTensionGate :: !Double
    , oaMethodWeightEntropyBlend :: !Double
    , oaMethodWeightCoherenceGate :: !Double
    , oaMethodWeightDivergenceGate :: !Double
    , oaMethodWeightFractalBlend :: !Double
    , oaMethodWeightPhaseCancel :: !Double
    , oaMethodWeightSoftmaxBlend :: !Double
    , oaMethodWeightSmoothSoftmaxBlend :: !Double
    , oaMethodWeightHedgeBlend :: !Double
    , oaMethodWeightNetSoftmaxBlend :: !Double
    , oaMethodWeightEdgeBlend :: !Double
    , oaMethodWeightEdgePick :: !Double
    , oaMethodWeightGeoBlend :: !Double
    , oaMethodWeightRegimeSwitch :: !Double
    , oaMethodWeightBanditRouter :: !Double
    , oaBlendWeightMin :: !Double
    , oaBlendWeightMax :: !Double
    , oaRouterScorePnlWeightMin :: !Double
    , oaRouterScorePnlWeightMax :: !Double
    , oaNormalizations :: !String
    , oaHiddenSizeMin :: !Int
    , oaHiddenSizeMax :: !Int
    , oaLrMin :: !Double
    , oaLrMax :: !Double
    , oaLstmAdamBeta1Min :: !(Maybe Double)
    , oaLstmAdamBeta1Max :: !(Maybe Double)
    , oaLstmAdamBeta2Min :: !(Maybe Double)
    , oaLstmAdamBeta2Max :: !(Maybe Double)
    , oaLstmAdamEpsMin :: !(Maybe Double)
    , oaLstmAdamEpsMax :: !(Maybe Double)
    , oaValRatioMin :: !Double
    , oaValRatioMax :: !Double
    , oaPatienceMax :: !Int
    , oaGradClipMin :: !Double
    , oaGradClipMax :: !Double
    , oaPDisableGradClip :: !Double
    , oaPredictors :: !String
    , oaRouterLookbackMin :: !Int
    , oaRouterLookbackMax :: !Int
    , oaRouterRegimeMinBarsMin :: !Int
    , oaRouterRegimeMinBarsMax :: !Int
    , oaRouterRegimeMinFractionMin :: !Double
    , oaRouterRegimeMinFractionMax :: !Double
    , oaRouterMinScoreMin :: !Double
    , oaRouterMinScoreMax :: !Double
    , oaFeeFixedMin :: !Double
    , oaFeeFixedMax :: !Double
    , oaSlippageImpactMin :: !Double
    , oaSlippageImpactMax :: !Double
    , oaSlippageImpactPowerMin :: !Double
    , oaSlippageImpactPowerMax :: !Double
    , oaSlippageVolMultMin :: !Double
    , oaSlippageVolMultMax :: !Double
    , oaSpreadVolMultMin :: !Double
    , oaSpreadVolMultMax :: !Double
    , oaTakeProfitPartialMin :: !Double
    , oaTakeProfitPartialMax :: !Double
    , oaPDisableTakeProfitPartial :: !Double
    , oaMaxTradesPerDayMin :: !Int
    , oaMaxTradesPerDayMax :: !Int
    , oaPDisableMaxTradesPerDay :: !Double
    , oaExpectancyLookbackMin :: !Int
    , oaExpectancyLookbackMax :: !Int
    , oaMinExpectancyMin :: !Double
    , oaMinExpectancyMax :: !Double
    , oaPDisableMinExpectancy :: !Double
    , oaLossStreakMaxMin :: !Int
    , oaLossStreakMaxMax :: !Int
    , oaPDisableLossStreakMax :: !Double
    , oaLossStreakCooldownBarsMin :: !Int
    , oaLossStreakCooldownBarsMax :: !Int
    , oaPDisableLossStreakCooldownBars :: !Double
    , oaMaxOpenPositionsMin :: !Int
    , oaMaxOpenPositionsMax :: !Int
    , oaPDisableMaxOpenPositions :: !Double
    , oaMaxGrossExposureMin :: !Double
    , oaMaxGrossExposureMax :: !Double
    , oaPDisableMaxGrossExposure :: !Double
    , oaMaxNetExposureMin :: !Double
    , oaMaxNetExposureMax :: !Double
    , oaPDisableMaxNetExposure :: !Double
    , oaMaxExposurePerBaseMin :: !Double
    , oaMaxExposurePerBaseMax :: !Double
    , oaPDisableMaxExposurePerBase :: !Double
    , oaMaxOpenPerBaseMin :: !Int
    , oaMaxOpenPerBaseMax :: !Int
    , oaPDisableMaxOpenPerBase :: !Double
    , oaAdaptiveEdgeBufferMaxMin :: !Double
    , oaAdaptiveEdgeBufferMaxMax :: !Double
    , oaAdaptiveMinSignalToNoiseMaxMin :: !Double
    , oaAdaptiveMinSignalToNoiseMaxMax :: !Double
    , oaAdaptiveTrendLookbackMaxMin :: !Int
    , oaAdaptiveTrendLookbackMaxMax :: !Int
    , oaAdaptiveKalmanZMinMaxMin :: !Double
    , oaAdaptiveKalmanZMinMaxMax :: !Double
    , oaAdaptiveWinRateSlackMin :: !Double
    , oaAdaptiveWinRateSlackMax :: !Double
    , oaAdaptiveProfitFactorSlackMin :: !Double
    , oaAdaptiveProfitFactorSlackMax :: !Double
    , oaPAdaptiveFilters :: !Double
    , oaPerfLookbackMin :: !Int
    , oaPerfLookbackMax :: !Int
    , oaPerfMinWinRateMin :: !Double
    , oaPerfMinWinRateMax :: !Double
    , oaPDisablePerfMinWinRate :: !Double
    , oaPerfMinProfitFactorMin :: !Double
    , oaPerfMinProfitFactorMax :: !Double
    , oaPDisablePerfMinProfitFactor :: !Double
    , oaPMetaLabelFilter :: !Double
    , oaMetaLabelMinEdgeMin :: !Double
    , oaMetaLabelMinEdgeMax :: !Double
    , oaMetaLabelMinConfidenceMin :: !Double
    , oaMetaLabelMinConfidenceMax :: !Double
    , oaPMetaLabelRequireBand :: !Double
    , oaPRegimeParameterBank :: !Double
    , oaRegimeBankHysteresisMin :: !Double
    , oaRegimeBankHysteresisMax :: !Double
    , oaRegimeTrendOpenMultMin :: !Double
    , oaRegimeTrendOpenMultMax :: !Double
    , oaRegimeMrOpenMultMin :: !Double
    , oaRegimeMrOpenMultMax :: !Double
    , oaRegimeHighVolOpenMultMin :: !Double
    , oaRegimeHighVolOpenMultMax :: !Double
    , oaRegimeTrendSizeMultMin :: !Double
    , oaRegimeTrendSizeMultMax :: !Double
    , oaRegimeMrSizeMultMin :: !Double
    , oaRegimeMrSizeMultMax :: !Double
    , oaRegimeHighVolSizeMultMin :: !Double
    , oaRegimeHighVolSizeMultMax :: !Double
    , oaTechnicalRanges :: !TechnicalOptimizerRanges
    , oaPMultiTimeframeConsensus :: !Double
    , oaMtfFastBarsMin :: !Int
    , oaMtfFastBarsMax :: !Int
    , oaMtfMidBarsMin :: !Int
    , oaMtfMidBarsMax :: !Int
    , oaMtfSlowBarsMin :: !Int
    , oaMtfSlowBarsMax :: !Int
    , oaMtfMinAgreeMin :: !Int
    , oaMtfMinAgreeMax :: !Int
    , oaPCrossAssetConfirmation :: !Double
    , oaCrossAssetMinBetaMin :: !Double
    , oaCrossAssetMinBetaMax :: !Double
    , oaCrossAssetMinEdgeMin :: !Double
    , oaCrossAssetMinEdgeMax :: !Double
    , oaPPairsStatArb :: !Double
    , oaPairsStatArbLookbackMin :: !Int
    , oaPairsStatArbLookbackMax :: !Int
    , oaPairsStatArbZEntryMin :: !Double
    , oaPairsStatArbZEntryMax :: !Double
    , oaPairsStatArbSizeMultMin :: !Double
    , oaPairsStatArbSizeMultMax :: !Double
    , oaPFundingOiAware :: !Double
    , oaFundingOiFundingCapMin :: !Double
    , oaFundingOiFundingCapMax :: !Double
    , oaPDisableFundingOiFundingCap :: !Double
    , oaFundingOiVolLookbackMin :: !Int
    , oaFundingOiVolLookbackMax :: !Int
    , oaFundingOiVolCapMin :: !Double
    , oaFundingOiVolCapMax :: !Double
    , oaPDisableFundingOiVolCap :: !Double
    , oaFundingOiSizeMultMin :: !Double
    , oaFundingOiSizeMultMax :: !Double
    , oaPKellyLiteSizing :: !Double
    , oaKellyLiteFractionMin :: !Double
    , oaKellyLiteFractionMax :: !Double
    , oaKellyLiteFloorMin :: !Double
    , oaKellyLiteFloorMax :: !Double
    , oaKellyLiteCapMin :: !Double
    , oaKellyLiteCapMax :: !Double
    , oaCrossExchangeCoinbase :: !Bool
    }
    deriving (Eq, Show)

roiScoreConfigFromOptimizerArgs :: OptimizerArgs -> RoiScoreConfig
roiScoreConfigFromOptimizerArgs args =
    sanitizeRoiScoreConfig
        defaultRoiScoreConfig
            { rscExpectancyRewardWeight = oaRoiScoreExpectancyWeight args
            , rscPaybackRewardCap = oaRoiScorePaybackCap args
            , rscMinimumActivityFloor = oaRoiScoreMinActivity args
            , rscMinimumExposureFloor = oaRoiScoreMinExposure args
            , rscZeroRoundTripPenalty = oaRoiScoreZeroRoundTripPenalty args
            , rscLowRoundTripPenalty = oaRoiScoreLowRoundTripPenalty args
            , rscZeroActivityPenalty = oaRoiScoreZeroActivityPenalty args
            , rscLowActivityPenalty = oaRoiScoreLowActivityPenalty args
            , rscZeroExposurePenalty = oaRoiScoreZeroExposurePenalty args
            , rscLowExposurePenaltyBase = oaRoiScoreLowExposurePenalty args
            , rscLowExposurePenaltyGapScale = oaRoiScoreLowExposureGapPenalty args
            }

applyQualityPreset :: OptimizerArgs -> OptimizerArgs
applyQualityPreset args =
    let objective = map toLower (trim (oaObjective args))
        objective' =
            if objective `elem` ["final-equity", "final_equity", "finalequity"]
                then "roi"
                else oaObjective args
        (interval', intervals') = qualityPresetIntervalFields (oaInterval args) (oaIntervals args)
        maxIf :: (Ord a) => a -> a -> a
        maxIf = max
        minIf :: (Ord a) => a -> a -> a
        minIf = min
     in args
            { oaTrials = maxIf (oaTrials args) (max 1 (oaQualityMinTrials args))
            , oaMinRoundTrips = maxIf (oaMinRoundTrips args) 20
            , oaOpenThresholdMax = qualityPresetCeiling 2e-2 5e-2 (oaOpenThresholdMaxExplicit args) (oaOpenThresholdMax args)
            , oaCloseThresholdMax = qualityPresetCeiling 2e-2 5e-2 (oaCloseThresholdMaxExplicit args) (oaCloseThresholdMax args)
            , oaMinWinRate = maxIf (oaMinWinRate args) 0.45
            , oaMinProfitFactor = maxIf (oaMinProfitFactor args) 1.1
            , oaMinExposure = maxIf (oaMinExposure args) 0.10
            , oaMaxKellyLiteExposureRatio = minIf (oaMaxKellyLiteExposureRatio args) 0.95
            , oaMinSharpe = maxIf (oaMinSharpe args) 1.0
            , oaMinCalmar = maxIf (oaMinCalmar args) 0.8
            , oaMinWfSharpeMean = maxIf (oaMinWfSharpeMean args) 0.8
            , oaMaxWfSharpeStd =
                if oaMaxWfSharpeStd args <= 0
                    then 1.0
                    else min 1.0 (oaMaxWfSharpeStd args)
            , oaMinSignalToNoiseMin = maxIf (oaMinSignalToNoiseMin args) 0.2
            , oaMinSignalToNoiseMax = maxIf (oaMinSignalToNoiseMax args) 1.0
            , oaEpochsMax = qualityPresetBudget 50 (oaEpochsMax args) (oaQualityMaxEpochs args)
            , oaHiddenSizeMax = qualityPresetBudget 128 (oaHiddenSizeMax args) (oaQualityMaxHiddenSize args)
            , oaLrMax = maxIf (oaLrMax args) 5e-2
            , oaBacktestRatio = minIf (oaBacktestRatio args) 0.10
            , oaTuneRatio = minIf (oaTuneRatio args) 0.15
            , oaTuneStressVolMult = maxIf (oaTuneStressVolMult args) 1.25
            , oaTuneStressWeight = maxIf (oaTuneStressWeight args) 0.2
            , oaObjective = objective'
            , oaPenaltyTurnover = maxIf (oaPenaltyTurnover args) 0.1
            , oaAutoHighLow = True
            , oaWalkForwardFoldsMin = maxIf (oaWalkForwardFoldsMin args) 3
            , oaWalkForwardFoldsMax = maxIf (oaWalkForwardFoldsMax args) (oaWalkForwardFoldsMin args)
            , oaWalkForwardEmbargoBarsMin = maxIf (oaWalkForwardEmbargoBarsMin args) 1
            , oaWalkForwardEmbargoBarsMax = maxIf (oaWalkForwardEmbargoBarsMax args) 3
            , oaMethodWeightRegimeSwitch = qualityPresetWeightFloor 1.0 (oaMethodWeightRegimeSwitch args)
            , oaMethodWeightBanditRouter = maxIf (oaMethodWeightBanditRouter args) 0.0
            , oaPConfidenceSizing = maxIf (oaPConfidenceSizing args) 0.85
            , oaPDisableRiskPerTrade = minIf (oaPDisableRiskPerTrade args) 0.3
            , oaStopVolMultMin = maxIf (oaStopVolMultMin args) 0.8
            , oaStopVolMultMax = maxIf (oaStopVolMultMax args) 3.0
            , oaTpVolMultMin = maxIf (oaTpVolMultMin args) 1.2
            , oaTpVolMultMax = maxIf (oaTpVolMultMax args) 4.0
            , oaTrailVolMultMin = maxIf (oaTrailVolMultMin args) 0.8
            , oaTrailVolMultMax = maxIf (oaTrailVolMultMax args) 3.0
            , oaPDisableStopVolMult = minIf (oaPDisableStopVolMult args) 0.35
            , oaPDisableTpVolMult = minIf (oaPDisableTpVolMult args) 0.35
            , oaPDisableTrailVolMult = minIf (oaPDisableTrailVolMult args) 0.5
            , oaPAdaptiveFilters = maxIf (oaPAdaptiveFilters args) 0.35
            , oaPerfLookbackMin = maxIf (oaPerfLookbackMin args) 10
            , oaPerfLookbackMax = maxIf (oaPerfLookbackMax args) 40
            , oaPerfMinWinRateMin = maxIf (oaPerfMinWinRateMin args) 0.48
            , oaPerfMinWinRateMax = maxIf (oaPerfMinWinRateMax args) 0.62
            , oaPDisablePerfMinWinRate = minIf (oaPDisablePerfMinWinRate args) 0.35
            , oaPerfMinProfitFactorMin = maxIf (oaPerfMinProfitFactorMin args) 1.05
            , oaPerfMinProfitFactorMax = maxIf (oaPerfMinProfitFactorMax args) 1.8
            , oaPDisablePerfMinProfitFactor = minIf (oaPDisablePerfMinProfitFactor args) 0.35
            , oaPMetaLabelFilter = maxIf (oaPMetaLabelFilter args) 0.3
            , oaMetaLabelMinEdgeMax = maxIf (oaMetaLabelMinEdgeMax args) 0.001
            , oaMetaLabelMinConfidenceMin = maxIf (oaMetaLabelMinConfidenceMin args) 0.45
            , oaMetaLabelMinConfidenceMax = maxIf (oaMetaLabelMinConfidenceMax args) 0.8
            , oaPMetaLabelRequireBand = maxIf (oaPMetaLabelRequireBand args) 0.75
            , oaPRegimeParameterBank = maxIf (oaPRegimeParameterBank args) 0.35
            , oaRegimeBankHysteresisMin = minIf (oaRegimeBankHysteresisMin args) 0.0
            , oaRegimeBankHysteresisMax = maxIf (oaRegimeBankHysteresisMax args) 0.12
            , oaRegimeTrendOpenMultMin = minIf (oaRegimeTrendOpenMultMin args) 0.75
            , oaRegimeTrendOpenMultMax = maxIf (oaRegimeTrendOpenMultMax args) 1.05
            , oaRegimeMrOpenMultMin = minIf (oaRegimeMrOpenMultMin args) 1.0
            , oaRegimeMrOpenMultMax = maxIf (oaRegimeMrOpenMultMax args) 1.35
            , oaRegimeHighVolOpenMultMin = minIf (oaRegimeHighVolOpenMultMin args) 1.05
            , oaRegimeHighVolOpenMultMax = maxIf (oaRegimeHighVolOpenMultMax args) 1.6
            , oaRegimeTrendSizeMultMin = minIf (oaRegimeTrendSizeMultMin args) 0.9
            , oaRegimeTrendSizeMultMax = maxIf (oaRegimeTrendSizeMultMax args) 1.35
            , oaRegimeMrSizeMultMin = minIf (oaRegimeMrSizeMultMin args) 0.6
            , oaRegimeMrSizeMultMax = maxIf (oaRegimeMrSizeMultMax args) 1.0
            , oaRegimeHighVolSizeMultMin = minIf (oaRegimeHighVolSizeMultMin args) 0.4
            , oaRegimeHighVolSizeMultMax = maxIf (oaRegimeHighVolSizeMultMax args) 0.9
            , oaPMultiTimeframeConsensus = maxIf (oaPMultiTimeframeConsensus args) 0.3
            , oaMtfFastBarsMin = minIf (oaMtfFastBarsMin args) 3
            , oaMtfFastBarsMax = maxIf (oaMtfFastBarsMax args) 12
            , oaMtfMidBarsMin = minIf (oaMtfMidBarsMin args) 12
            , oaMtfMidBarsMax = maxIf (oaMtfMidBarsMax args) 36
            , oaMtfSlowBarsMin = minIf (oaMtfSlowBarsMin args) 36
            , oaMtfSlowBarsMax = maxIf (oaMtfSlowBarsMax args) 120
            , oaMtfMinAgreeMin = maxIf (oaMtfMinAgreeMin args) 2
            , oaMtfMinAgreeMax = maxIf (oaMtfMinAgreeMax args) 3
            , oaPCrossAssetConfirmation = maxIf (oaPCrossAssetConfirmation args) 0.25
            , oaCrossAssetMinBetaMin = minIf (oaCrossAssetMinBetaMin args) 0.0
            , oaCrossAssetMinBetaMax = maxIf (oaCrossAssetMinBetaMax args) 0.2
            , oaCrossAssetMinEdgeMin = minIf (oaCrossAssetMinEdgeMin args) 0.0
            , oaCrossAssetMinEdgeMax = maxIf (oaCrossAssetMinEdgeMax args) 0.001
            , oaPPairsStatArb = maxIf (oaPPairsStatArb args) 0.15
            , oaPairsStatArbLookbackMin = minIf (oaPairsStatArbLookbackMin args) 60
            , oaPairsStatArbLookbackMax = maxIf (oaPairsStatArbLookbackMax args) 180
            , oaPairsStatArbZEntryMin = minIf (oaPairsStatArbZEntryMin args) 1.5
            , oaPairsStatArbZEntryMax = maxIf (oaPairsStatArbZEntryMax args) 3.0
            , oaPairsStatArbSizeMultMin = minIf (oaPairsStatArbSizeMultMin args) 0.4
            , oaPairsStatArbSizeMultMax = maxIf (oaPairsStatArbSizeMultMax args) 0.9
            , oaPFundingOiAware = maxIf (oaPFundingOiAware args) 0.2
            , oaFundingOiFundingCapMin = minIf (oaFundingOiFundingCapMin args) 0.02
            , oaFundingOiFundingCapMax = maxIf (oaFundingOiFundingCapMax args) 0.2
            , oaPDisableFundingOiFundingCap = minIf (oaPDisableFundingOiFundingCap args) 0.6
            , oaFundingOiVolLookbackMin = minIf (oaFundingOiVolLookbackMin args) 24
            , oaFundingOiVolLookbackMax = maxIf (oaFundingOiVolLookbackMax args) 96
            , oaFundingOiVolCapMin = minIf (oaFundingOiVolCapMin args) 0.3
            , oaFundingOiVolCapMax = maxIf (oaFundingOiVolCapMax args) 2.0
            , oaPDisableFundingOiVolCap = minIf (oaPDisableFundingOiVolCap args) 0.6
            , oaFundingOiSizeMultMin = minIf (oaFundingOiSizeMultMin args) 0.4
            , oaFundingOiSizeMultMax = maxIf (oaFundingOiSizeMultMax args) 0.9
            , oaPKellyLiteSizing = maxIf (oaPKellyLiteSizing args) 0.25
            , oaKellyLiteFractionMin = minIf (oaKellyLiteFractionMin args) 0.1
            , oaKellyLiteFractionMax = maxIf (oaKellyLiteFractionMax args) 0.8
            , oaKellyLiteFloorMin = minIf (oaKellyLiteFloorMin args) 0.0
            , oaKellyLiteFloorMax = maxIf (oaKellyLiteFloorMax args) 0.35
            , oaKellyLiteCapMin = minIf (oaKellyLiteCapMin args) 0.6
            , oaKellyLiteCapMax = maxIf (oaKellyLiteCapMax args) 1.25
            , oaInterval = interval'
            , oaIntervals = intervals'
            }

optimizerOptionPresent :: String -> [String] -> Bool
optimizerOptionPresent name argv =
    let flag = "--" ++ name
        flagWithValue = flag ++ "="
     in any (\arg -> arg == flag || isJust (stripPrefix flagWithValue arg)) argv

qualityPresetBudget :: Int -> Int -> Int -> Int
qualityPresetBudget productionDefault requested rawBudget =
    let budget = max 1 rawBudget
     in if budget >= productionDefault
            then max requested productionDefault
            else min requested budget

qualityPresetCeiling :: (Eq a) => a -> a -> Bool -> a -> a
qualityPresetCeiling productionDefault qualityDefault explicitlyRequested requested =
    if not explicitlyRequested && requested == productionDefault
        then qualityDefault
        else requested

qualityPresetWeightFloor :: (Eq a, Num a, Ord a) => a -> a -> a
qualityPresetWeightFloor qualityDefault requested =
    if requested == 0
        then 0
        else max requested qualityDefault

qualityPresetIntervalFields :: Maybe String -> Maybe String -> (Maybe String, Maybe String)
qualityPresetIntervalFields rawInterval rawIntervals =
    let normalizeMaybe value =
            case value of
                Nothing -> Nothing
                Just v ->
                    let trimmed = trim v
                     in if null trimmed then Nothing else Just trimmed
        interval = normalizeMaybe rawInterval
        intervals = normalizeMaybe rawIntervals
     in if isJust interval || isJust intervals
            then (interval, intervals)
            else (Nothing, Just binanceIntervalsCsv)

data TrialParams = TrialParams
    { tpPlatform :: !(Maybe String)
    , tpInterval :: !String
    , tpBars :: !Int
    , tpMethod :: !String
    , tpBlendWeight :: !Double
    , tpRouterScorePnlWeight :: !Double
    , tpPositioning :: !String
    , tpNormalization :: !String
    , tpBaseOpenThreshold :: !Double
    , tpBaseCloseThreshold :: !Double
    , tpMinHoldBars :: !Int
    , tpCooldownBars :: !Int
    , tpMaxHoldBars :: !(Maybe Int)
    , tpMinEdge :: !Double
    , tpMinSignalToNoise :: !Double
    , tpSnrSizeWeight :: !Double
    , tpThresholdFactorEnabled :: !Bool
    , tpThresholdFactorAlpha :: !Double
    , tpThresholdFactorMin :: !Double
    , tpThresholdFactorMax :: !Double
    , tpThresholdFactorFloor :: !Double
    , tpThresholdFactorEdgeKalWeight :: !Double
    , tpThresholdFactorEdgeLstmWeight :: !Double
    , tpThresholdFactorKalmanZWeight :: !Double
    , tpThresholdFactorHighVolWeight :: !Double
    , tpThresholdFactorConformalWeight :: !Double
    , tpThresholdFactorQuantileWeight :: !Double
    , tpThresholdFactorLstmConfWeight :: !Double
    , tpThresholdFactorLstmHealthWeight :: !Double
    , tpEdgeBuffer :: !Double
    , tpCostAwareEdge :: !Bool
    , tpTrendLookback :: !Int
    , tpMaxPositionSize :: !Double
    , tpVolTarget :: !(Maybe Double)
    , tpVolLookback :: !Int
    , tpVolEwmaAlpha :: !(Maybe Double)
    , tpVolFloor :: !Double
    , tpVolScaleMax :: !Double
    , tpMaxVolatility :: !(Maybe Double)
    , tpPeriodsPerYear :: !(Maybe Double)
    , tpKalmanMarketTopN :: !Int
    , tpSensorVarianceEwmaAlpha :: !Double
    , tpFee :: !Double
    , tpFundingRate :: !Double
    , tpFundingBySide :: !Bool
    , tpFundingOnOpen :: !Bool
    , tpRebalanceBars :: !Int
    , tpRebalanceThreshold :: !Double
    , tpRebalanceCostMult :: !Double
    , tpRebalanceGlobal :: !Bool
    , tpRebalanceResetOnSignal :: !Bool
    , tpEpochs :: !Int
    , tpHiddenSize :: !Int
    , tpLearningRate :: !Double
    , tpLstmAdamBeta1 :: !Double
    , tpLstmAdamBeta2 :: !Double
    , tpLstmAdamEps :: !Double
    , tpValRatio :: !Double
    , tpPatience :: !Int
    , tpWalkForwardFolds :: !Int
    , tpWalkForwardEmbargoBars :: !Int
    , tpTuneStressVolMult :: !Double
    , tpTuneStressShock :: !Double
    , tpTuneStressWeight :: !Double
    , tpGradClip :: !(Maybe Double)
    , tpSlippage :: !Double
    , tpSpread :: !Double
    , tpIntrabarFill :: !String
    , tpTriLayer :: !Bool
    , tpTriLayerFastMult :: !Double
    , tpTriLayerSlowMult :: !Double
    , tpTriLayerCloudPadding :: !Double
    , tpTriLayerCloudSlope :: !Double
    , tpTriLayerCloudWidth :: !Double
    , tpTriLayerTouchLookback :: !Int
    , tpTriLayerPriceAction :: !Bool
    , tpTriLayerPriceActionBody :: !Double
    , tpTriLayerPriceActionWickRatio :: !Double
    , tpTriLayerPriceActionOppositeWickMax :: !Double
    , tpTriLayerPriceActionBodyTolerance :: !Double
    , tpTriLayerExitOnSlow :: !Bool
    , tpKalmanBandLookback :: !Int
    , tpKalmanBandStdMult :: !Double
    , tpLstmExitFlipBars :: !Int
    , tpLstmExitFlipGraceBars :: !Int
    , tpLstmExitFlipStrong :: !Bool
    , tpLstmConfidenceSoft :: !Double
    , tpLstmConfidenceHard :: !Double
    , tpStopLoss :: !(Maybe Double)
    , tpTakeProfit :: !(Maybe Double)
    , tpTrailingStop :: !(Maybe Double)
    , tpStopLossVolMult :: !(Maybe Double)
    , tpTakeProfitVolMult :: !(Maybe Double)
    , tpTrailingStopVolMult :: !(Maybe Double)
    , tpRiskPerTrade :: !(Maybe Double)
    , tpMaxDrawdown :: !(Maybe Double)
    , tpMaxDailyLoss :: !(Maybe Double)
    , tpMaxWeeklyLoss :: !(Maybe Double)
    , tpMaxOrderErrors :: !(Maybe Int)
    , tpKalmanDt :: !Double
    , tpKalmanProcessVar :: !Double
    , tpKalmanMeasurementVar :: !Double
    , tpKalmanPhysicsBars :: !Int
    , tpKalmanPhysicsBacktestRatio :: !Double
    , tpKalmanPhysicsVolumeEwmaAlpha :: !Double
    , tpKalmanPhysicsVolumeSignalClamp :: !Double
    , tpKalmanPhysicsCloseBiasScale :: !Double
    , tpKalmanSensorCorrelationInflation :: !Double
    , tpKalmanInnovationInflationThreshold :: !Double
    , tpKalmanInnovationInflationMax :: !Double
    , tpKalmanZMin :: !Double
    , tpKalmanZMax :: !Double
    , tpMaxHighVolProb :: !(Maybe Double)
    , tpMaxConformalWidth :: !(Maybe Double)
    , tpMaxQuantileWidth :: !(Maybe Double)
    , tpConfirmConformal :: !Bool
    , tpConfirmQuantiles :: !Bool
    , tpConfidenceSizing :: !Bool
    , tpProtectionMinConfidence :: !Double
    , tpMinPositionSize :: !Double
    , tpKellyLiteSizing :: !Bool
    , tpKellyLiteFraction :: !Double
    , tpKellyLiteFloor :: !Double
    , tpKellyLiteCap :: !Double
    , tpPredictors :: !String
    , tpRouterLookback :: !Int
    , tpRouterRegimeMinBars :: !Int
    , tpRouterRegimeMinFraction :: !Double
    , tpRouterMinScore :: !Double
    , tpFeeFixed :: !Double
    , tpSlippageImpact :: !Double
    , tpSlippageImpactPower :: !Double
    , tpSlippageVolMult :: !Double
    , tpSpreadVolMult :: !Double
    , tpTakeProfitPartial :: !(Maybe Double)
    , tpMaxTradesPerDay :: !(Maybe Int)
    , tpExpectancyLookback :: !Int
    , tpMinExpectancy :: !(Maybe Double)
    , tpLossStreakMax :: !(Maybe Int)
    , tpLossStreakCooldownBars :: !(Maybe Int)
    , tpMaxOpenPositions :: !(Maybe Int)
    , tpMaxGrossExposure :: !(Maybe Double)
    , tpMaxNetExposure :: !(Maybe Double)
    , tpMaxExposurePerBase :: !(Maybe Double)
    , tpMaxOpenPerBase :: !(Maybe Int)
    , tpAdaptiveEdgeBufferMax :: !Double
    , tpAdaptiveMinSignalToNoiseMax :: !Double
    , tpAdaptiveTrendLookbackMax :: !Int
    , tpAdaptiveKalmanZMinMax :: !Double
    , tpAdaptiveWinRateSlack :: !Double
    , tpAdaptiveProfitFactorSlack :: !Double
    , tpAdaptiveFilters :: !Bool
    , tpPerfLookback :: !Int
    , tpPerfMinWinRate :: !(Maybe Double)
    , tpPerfMinProfitFactor :: !(Maybe Double)
    , tpMetaLabelFilter :: !Bool
    , tpMetaLabelMinEdge :: !Double
    , tpMetaLabelMinConfidence :: !Double
    , tpMetaLabelRequireBand :: !Bool
    , tpRegimeParameterBank :: !Bool
    , tpRegimeBankHysteresis :: !Double
    , tpRegimeTrendOpenMult :: !Double
    , tpRegimeMrOpenMult :: !Double
    , tpRegimeHighVolOpenMult :: !Double
    , tpRegimeTrendSizeMult :: !Double
    , tpRegimeMrSizeMult :: !Double
    , tpRegimeHighVolSizeMult :: !Double
    , tpTechnicalParams :: !TechnicalTrialParams
    , tpMultiTimeframeConsensus :: !Bool
    , tpMtfFastBars :: !Int
    , tpMtfMidBars :: !Int
    , tpMtfSlowBars :: !Int
    , tpMtfMinAgree :: !Int
    , tpCrossAssetConfirmation :: !Bool
    , tpCrossAssetMinBeta :: !Double
    , tpCrossAssetMinEdge :: !Double
    , tpPairsStatArb :: !Bool
    , tpPairsStatArbLookback :: !Int
    , tpPairsStatArbZEntry :: !Double
    , tpPairsStatArbSizeMult :: !Double
    , tpFundingOiAware :: !Bool
    , tpFundingOiFundingCap :: !(Maybe Double)
    , tpFundingOiVolLookback :: !Int
    , tpFundingOiVolCap :: !(Maybe Double)
    , tpFundingOiSizeMult :: !Double
    }
    deriving (Eq, Show)

data TrialResult = TrialResult
    { trOk :: !Bool
    , trReason :: !(Maybe String)
    , trElapsedSec :: !Double
    , trOrigin :: !String
    , trParams :: !TrialParams
    , trFinalEquity :: !(Maybe Double)
    , trMetrics :: !(Maybe (KM.KeyMap Value))
    , trOpenThreshold :: !(Maybe Double)
    , trCloseThreshold :: !(Maybe Double)
    , trStdoutJson :: !(Maybe Value)
    , trEligible :: !Bool
    , trFilterReason :: !(Maybe String)
    , trObjective :: !String
    , trScore :: !(Maybe Double)
    }
    deriving (Eq, Show)

fmtOptFloat :: Maybe Double -> String
fmtOptFloat v =
    case v of
        Nothing -> "null"
        Just x -> printf "%.8f" x

fmtOptInt :: Maybe Int -> String
fmtOptInt = maybe "null" show

buildCommand :: FilePath -> [String] -> TrialParams -> Double -> Bool -> [String]
buildCommand traderBin baseArgs params0 tuneRatio useSweepThreshold =
    let params = normalizeTrialParams params0
        cmd0 = traderBin : baseArgs
        perfLookbackArg =
            if tpAdaptiveFilters params || isJust (tpPerfMinWinRate params) || isJust (tpPerfMinProfitFactor params)
                then max 1 (tpPerfLookback params)
                else 0
        cmd1 =
            case tpPlatform params of
                Just platform -> cmd0 ++ ["--platform", platform]
                Nothing -> cmd0
        cmd2 = cmd1 ++ ["--interval", tpInterval params]
        isBinance = tpPlatform params == Just "binance" && "--binance-symbol" `elem` baseArgs
        cmd3 =
            if tpBars params <= 0
                then cmd2 ++ ["--bars", if isBinance then "auto" else "0"]
                else cmd2 ++ ["--bars", show (tpBars params)]
        cmd4 =
            cmd3
                ++ [ "--positioning"
                   , tpPositioning params
                   , "--method"
                   , tpMethod params
                   , "--blend-weight"
                   , printf "%.6f" (clamp (tpBlendWeight params) 0 1)
                   , "--router-score-pnl-weight"
                   , printf "%.6f" (clamp (tpRouterScorePnlWeight params) 0 1)
                   , "--normalization"
                   , tpNormalization params
                   , "--open-threshold"
                   , printf "%.12g" (max 0 (tpBaseOpenThreshold params))
                   , "--close-threshold"
                   , printf "%.12g" (max 0 (tpBaseCloseThreshold params))
                   , "--min-hold-bars"
                   , show (max 0 (tpMinHoldBars params))
                   , "--cooldown-bars"
                   , show (max 0 (tpCooldownBars params))
                   ]
        cmd5 =
            case tpMaxHoldBars params of
                Just v -> cmd4 ++ ["--max-hold-bars", show (max 0 v)]
                Nothing -> cmd4
        cmd6 =
            cmd5
                ++ [ "--min-edge"
                   , printf "%.12g" (max 0 (tpMinEdge params))
                   , "--min-signal-to-noise"
                   , printf "%.12g" (max 0 (tpMinSignalToNoise params))
                   , "--snr-size-weight"
                   , printf "%.6f" (clamp (tpSnrSizeWeight params) 0 1)
                   , "--edge-buffer"
                   , printf "%.12g" (max 0 (tpEdgeBuffer params))
                   ]
        cmd6a =
            cmd6
                ++ (if tpThresholdFactorEnabled params then ["--threshold-factor"] else ["--no-threshold-factor"])
                ++ [ "--threshold-factor-alpha"
                   , printf "%.6f" (clamp (tpThresholdFactorAlpha params) 0 1)
                   , "--threshold-factor-min"
                   , printf "%.6f" (max 0 (tpThresholdFactorMin params))
                   , "--threshold-factor-max"
                   , printf "%.6f" (max 0 (tpThresholdFactorMax params))
                   , "--threshold-factor-floor"
                   , printf "%.12g" (max 0 (tpThresholdFactorFloor params))
                   , "--threshold-factor-edge-kal-weight"
                   , printf "%.6f" (tpThresholdFactorEdgeKalWeight params)
                   , "--threshold-factor-edge-lstm-weight"
                   , printf "%.6f" (tpThresholdFactorEdgeLstmWeight params)
                   , "--threshold-factor-kalman-z-weight"
                   , printf "%.6f" (tpThresholdFactorKalmanZWeight params)
                   , "--threshold-factor-high-vol-weight"
                   , printf "%.6f" (tpThresholdFactorHighVolWeight params)
                   , "--threshold-factor-conformal-weight"
                   , printf "%.6f" (tpThresholdFactorConformalWeight params)
                   , "--threshold-factor-quantile-weight"
                   , printf "%.6f" (tpThresholdFactorQuantileWeight params)
                   , "--threshold-factor-lstm-conf-weight"
                   , printf "%.6f" (tpThresholdFactorLstmConfWeight params)
                   , "--threshold-factor-lstm-health-weight"
                   , printf "%.6f" (tpThresholdFactorLstmHealthWeight params)
                   ]
        cmd7 =
            if tpCostAwareEdge params
                then cmd6a ++ ["--cost-aware-edge"]
                else cmd6a ++ ["--no-cost-aware-edge"]
        cmd8 =
            cmd7
                ++ [ "--trend-lookback"
                   , show (max 0 (tpTrendLookback params))
                   , "--max-position-size"
                   , printf "%.12g" (max 0 (tpMaxPositionSize params))
                   ]
        cmd9 =
            case tpVolTarget params of
                Just v -> cmd8 ++ ["--vol-target", printf "%.12g" (max 1e-12 v)]
                Nothing -> cmd8
        cmd10 =
            cmd9
                ++ [ "--vol-lookback"
                   , show (max 0 (tpVolLookback params))
                   ]
        cmd11 =
            case tpVolEwmaAlpha params of
                Just v -> cmd10 ++ ["--vol-ewma-alpha", printf "%.6f" (clamp v 0 1)]
                Nothing -> cmd10
        cmd12 =
            cmd11
                ++ [ "--vol-floor"
                   , printf "%.12g" (max 0 (tpVolFloor params))
                   , "--vol-scale-max"
                   , printf "%.12g" (max 0 (tpVolScaleMax params))
                   ]
        cmd13 =
            case tpMaxVolatility params of
                Just v -> cmd12 ++ ["--max-volatility", printf "%.12g" (max 1e-12 v)]
                Nothing -> cmd12
        cmd14 =
            case tpPeriodsPerYear params of
                Just v -> cmd13 ++ ["--periods-per-year", printf "%.6f" (max 1e-12 v)]
                Nothing -> cmd13
        cmd15 =
            cmd14
                ++ [ "--fee"
                   , printf "%.12g" (max 0 (tpFee params))
                   , "--fee-fixed"
                   , printf "%.12g" (max 0 (tpFeeFixed params))
                   , "--funding-rate"
                   , printf "%.12g" (tpFundingRate params)
                   , "--rebalance-bars"
                   , show (max 0 (tpRebalanceBars params))
                   , "--rebalance-threshold"
                   , printf "%.12g" (max 0 (tpRebalanceThreshold params))
                   , "--rebalance-cost-mult"
                   , printf "%.12g" (max 0 (tpRebalanceCostMult params))
                   , "--epochs"
                   , show (tpEpochs params)
                   , "--hidden-size"
                   , show (tpHiddenSize params)
                   , "--lr"
                   , printf "%.8f" (tpLearningRate params)
                   , "--lstm-adam-beta1"
                   , printf "%.6f" (tpLstmAdamBeta1 params)
                   , "--lstm-adam-beta2"
                   , printf "%.6f" (tpLstmAdamBeta2 params)
                   , "--lstm-adam-eps"
                   , printf "%.12g" (tpLstmAdamEps params)
                   , "--val-ratio"
                   , printf "%.6f" (tpValRatio params)
                   , "--patience"
                   , show (tpPatience params)
                   , "--walk-forward-folds"
                   , show (max 1 (tpWalkForwardFolds params))
                   , "--walk-forward-embargo-bars"
                   , show (max 0 (tpWalkForwardEmbargoBars params))
                   , "--tune-stress-vol-mult"
                   , printf "%.6f" (max 1e-12 (tpTuneStressVolMult params))
                   , "--tune-stress-shock"
                   , printf "%.6f" (tpTuneStressShock params)
                   , "--tune-stress-weight"
                   , printf "%.6f" (max 0 (tpTuneStressWeight params))
                   ]
                ++ (["--funding-by-side" | tpFundingBySide params])
                ++ (["--funding-on-open" | tpFundingOnOpen params])
                ++ (["--rebalance-global" | tpRebalanceGlobal params])
                ++ (["--rebalance-reset-on-signal" | tpRebalanceResetOnSignal params])
        cmd16 =
            case tpGradClip params of
                Just v -> cmd15 ++ ["--grad-clip", printf "%.8f" v]
                Nothing -> cmd15
        cmd17 =
            cmd16
                ++ [ "--slippage"
                   , printf "%.8f" (tpSlippage params)
                   , "--spread"
                   , printf "%.8f" (tpSpread params)
                   , "--slippage-impact"
                   , printf "%.12g" (max 0 (tpSlippageImpact params))
                   , "--slippage-impact-power"
                   , printf "%.12g" (max 0 (tpSlippageImpactPower params))
                   , "--slippage-vol-mult"
                   , printf "%.12g" (max 0 (tpSlippageVolMult params))
                   , "--spread-vol-mult"
                   , printf "%.12g" (max 0 (tpSpreadVolMult params))
                   , "--intrabar-fill"
                   , tpIntrabarFill params
                   , "--predictors"
                   , tpPredictors params
                   , "--router-lookback"
                   , show (max 2 (tpRouterLookback params))
                   , "--router-regime-min-bars"
                   , show (max 0 (tpRouterRegimeMinBars params))
                   , "--router-regime-min-fraction"
                   , printf "%.12g" (clamp (tpRouterRegimeMinFraction params) 0 1)
                   , "--router-min-score"
                   , printf "%.12g" (tpRouterMinScore params)
                   , "--adaptive-edge-buffer-max"
                   , printf "%.12g" (max 0 (tpAdaptiveEdgeBufferMax params))
                   , "--adaptive-min-signal-to-noise-max"
                   , printf "%.12g" (max 0 (tpAdaptiveMinSignalToNoiseMax params))
                   , "--adaptive-trend-lookback-max"
                   , show (max 1 (tpAdaptiveTrendLookbackMax params))
                   , "--adaptive-kalman-z-min-max"
                   , printf "%.12g" (max 0 (tpAdaptiveKalmanZMinMax params))
                   , "--adaptive-win-rate-slack"
                   , printf "%.12g" (max 0 (tpAdaptiveWinRateSlack params))
                   , "--adaptive-profit-factor-slack"
                   , printf "%.12g" (max 0 (tpAdaptiveProfitFactorSlack params))
                   , "--perf-lookback"
                   , show perfLookbackArg
                   ]
                ++ (if tpAdaptiveFilters params then ["--adaptive-filters"] else ["--no-adaptive-filters"])
                ++ maybe [] (\v -> ["--perf-min-win-rate", printf "%.12g" (clamp v 0 1)]) (tpPerfMinWinRate params)
                ++ maybe [] (\v -> ["--perf-min-profit-factor", printf "%.12g" (max 0 v)]) (tpPerfMinProfitFactor params)
                ++ (if tpMetaLabelFilter params then ["--meta-label-filter"] else ["--no-meta-label-filter"])
                ++ [ "--meta-label-min-edge"
                   , printf "%.12g" (max 0 (tpMetaLabelMinEdge params))
                   , "--meta-label-min-confidence"
                   , printf "%.12g" (clamp (tpMetaLabelMinConfidence params) 0 1)
                   , "--regime-bank-hysteresis"
                   , printf "%.12g" (clamp (tpRegimeBankHysteresis params) 0 1)
                   , "--regime-trend-open-mult"
                   , printf "%.12g" (max 0 (tpRegimeTrendOpenMult params))
                   , "--regime-mr-open-mult"
                   , printf "%.12g" (max 0 (tpRegimeMrOpenMult params))
                   , "--regime-high-vol-open-mult"
                   , printf "%.12g" (max 0 (tpRegimeHighVolOpenMult params))
                   , "--regime-trend-size-mult"
                   , printf "%.12g" (max 0 (tpRegimeTrendSizeMult params))
                   , "--regime-mr-size-mult"
                   , printf "%.12g" (max 0 (tpRegimeMrSizeMult params))
                   , "--regime-high-vol-size-mult"
                   , printf "%.12g" (max 0 (tpRegimeHighVolSizeMult params))
                   , "--mtf-fast-bars"
                   , show (max 1 (tpMtfFastBars params))
                   , "--mtf-mid-bars"
                   , show (max 1 (tpMtfMidBars params))
                   , "--mtf-slow-bars"
                   , show (max 1 (tpMtfSlowBars params))
                   , "--mtf-min-agree"
                   , show (clampInt (tpMtfMinAgree params) 1 3)
                   , "--cross-asset-min-beta"
                   , printf "%.12g" (max 0 (tpCrossAssetMinBeta params))
                   , "--cross-asset-min-edge"
                   , printf "%.12g" (max 0 (tpCrossAssetMinEdge params))
                   , "--pairs-stat-arb-lookback"
                   , show (max 2 (tpPairsStatArbLookback params))
                   , "--pairs-stat-arb-z-entry"
                   , printf "%.12g" (max 1e-12 (tpPairsStatArbZEntry params))
                   , "--pairs-stat-arb-size-mult"
                   , printf "%.12g" (clamp (tpPairsStatArbSizeMult params) 0 1)
                   , "--funding-oi-vol-lookback"
                   , show (max 2 (tpFundingOiVolLookback params))
                   , "--funding-oi-size-mult"
                   , printf "%.12g" (clamp (tpFundingOiSizeMult params) 0 1)
                   ]
                ++ technicalTrialParamsArgs (tpTechnicalParams params)
                ++ (if tpMetaLabelRequireBand params then ["--meta-label-require-band"] else ["--no-meta-label-require-band"])
                ++ (if tpRegimeParameterBank params then ["--regime-parameter-bank"] else ["--no-regime-parameter-bank"])
                ++ (if tpMultiTimeframeConsensus params then ["--multi-timeframe-consensus"] else ["--no-multi-timeframe-consensus"])
                ++ (if tpCrossAssetConfirmation params then ["--cross-asset-confirmation"] else ["--no-cross-asset-confirmation"])
                ++ (if tpPairsStatArb params then ["--pairs-stat-arb"] else ["--no-pairs-stat-arb"])
                ++ (if tpFundingOiAware params then ["--funding-oi-aware"] else ["--no-funding-oi-aware"])
                ++ maybe [] (\v -> ["--funding-oi-funding-cap", printf "%.12g" (max 0 v)]) (tpFundingOiFundingCap params)
                ++ maybe [] (\v -> ["--funding-oi-vol-cap", printf "%.12g" (max 0 v)]) (tpFundingOiVolCap params)
                ++ (["--tri-layer" | tpTriLayer params])
                ++ [ "--tri-layer-fast-mult"
                   , printf "%.12g" (max 1e-6 (tpTriLayerFastMult params))
                   , "--tri-layer-slow-mult"
                   , printf "%.12g" (max 1e-6 (tpTriLayerSlowMult params))
                   , "--tri-layer-cloud-padding"
                   , printf "%.8f" (max 0 (tpTriLayerCloudPadding params))
                   , "--tri-layer-cloud-slope"
                   , printf "%.8f" (max 0 (tpTriLayerCloudSlope params))
                   , "--tri-layer-cloud-width"
                   , printf "%.8f" (max 0 (tpTriLayerCloudWidth params))
                   , "--tri-layer-touch-lookback"
                   , show (max 1 (tpTriLayerTouchLookback params))
                   , "--tri-layer-price-action-body"
                   , printf "%.8f" (max 0 (tpTriLayerPriceActionBody params))
                   , "--tri-layer-price-action-wick-ratio"
                   , printf "%.8f" (max 0 (tpTriLayerPriceActionWickRatio params))
                   , "--tri-layer-price-action-opposite-wick-max"
                   , printf "%.8f" (max 0 (tpTriLayerPriceActionOppositeWickMax params))
                   , "--tri-layer-price-action-body-tolerance"
                   , printf "%.8f" (max 0 (tpTriLayerPriceActionBodyTolerance params))
                   ]
                ++ (["--no-tri-layer-price-action" | tpTriLayer params && not (tpTriLayerPriceAction params)])
                ++ (["--tri-layer-exit-on-slow" | tpTriLayerExitOnSlow params])
                ++ [ "--kalman-band-lookback"
                   , show (max 0 (tpKalmanBandLookback params))
                   , "--kalman-band-std-mult"
                   , printf "%.8f" (max 0 (tpKalmanBandStdMult params))
                   , "--lstm-exit-flip-bars"
                   , show (max 0 (tpLstmExitFlipBars params))
                   , "--lstm-exit-flip-grace-bars"
                   , show (max 0 (tpLstmExitFlipGraceBars params))
                   ]
                ++ (["--lstm-exit-flip-strong" | tpLstmExitFlipStrong params])
                ++ [ "--lstm-confidence-soft"
                   , printf "%.4f" (clamp (tpLstmConfidenceSoft params) 0 1)
                   , "--lstm-confidence-hard"
                   , printf "%.4f" (clamp (tpLstmConfidenceHard params) 0 1)
                   ]
        cmd18 =
            case tpStopLoss params of
                Just v -> cmd17 ++ ["--stop-loss", printf "%.8f" v]
                Nothing -> cmd17
        cmd19 =
            case tpTakeProfit params of
                Just v -> cmd18 ++ ["--take-profit", printf "%.8f" v]
                Nothing -> cmd18
        cmd20 =
            case tpTrailingStop params of
                Just v -> cmd19 ++ ["--trailing-stop", printf "%.8f" v]
                Nothing -> cmd19
        cmd21 =
            case tpStopLossVolMult params of
                Just v -> cmd20 ++ ["--stop-loss-vol-mult", printf "%.8f" v]
                Nothing -> cmd20
        cmd22 =
            case tpTakeProfitVolMult params of
                Just v -> cmd21 ++ ["--take-profit-vol-mult", printf "%.8f" v]
                Nothing -> cmd21
        cmd23 =
            case tpTrailingStopVolMult params of
                Just v -> cmd22 ++ ["--trailing-stop-vol-mult", printf "%.8f" v]
                Nothing -> cmd22
        cmd23b =
            case tpRiskPerTrade params of
                Just v -> cmd23 ++ ["--risk-per-trade", printf "%.8f" (clamp v 1e-6 0.999999)]
                Nothing -> cmd23
        cmd24 =
            case tpMaxDrawdown params of
                Just v -> cmd23b ++ ["--max-drawdown", printf "%.8f" v]
                Nothing -> cmd23b
        cmd25 =
            case tpMaxDailyLoss params of
                Just v -> cmd24 ++ ["--max-daily-loss", printf "%.8f" v]
                Nothing -> cmd24
        cmd26 =
            case tpMaxWeeklyLoss params of
                Just v -> cmd25 ++ ["--max-weekly-loss", printf "%.8f" v]
                Nothing -> cmd25
        cmd27 =
            case tpMaxOrderErrors params of
                Just v -> cmd26 ++ ["--max-order-errors", show v]
                Nothing -> cmd26
        cmd28 =
            cmd27
                ++ [ "--kalman-market-top-n"
                   , show (max 0 (tpKalmanMarketTopN params))
                   , "--kalman-dt"
                   , printf "%.12g" (max 1e-12 (tpKalmanDt params))
                   , "--kalman-process-var"
                   , printf "%.12g" (max 1e-12 (tpKalmanProcessVar params))
                   , "--kalman-measurement-var"
                   , printf "%.12g" (max 1e-12 (tpKalmanMeasurementVar params))
                   , "--kalman-physics-bars"
                   , show (max 4 (tpKalmanPhysicsBars params))
                   , "--kalman-physics-backtest-ratio"
                   , printf "%.6f" (clamp (tpKalmanPhysicsBacktestRatio params) 0.000001 0.999999)
                   , "--kalman-physics-volume-ewma-alpha"
                   , printf "%.6f" (clamp (tpKalmanPhysicsVolumeEwmaAlpha params) 0 1)
                   , "--kalman-physics-volume-signal-clamp"
                   , printf "%.6f" (max 0 (tpKalmanPhysicsVolumeSignalClamp params))
                   , "--kalman-physics-close-bias-scale"
                   , printf "%.6f" (tpKalmanPhysicsCloseBiasScale params)
                   , "--sensor-variance-ewma-alpha"
                   , printf "%.6f" (clamp (tpSensorVarianceEwmaAlpha params) 0 1)
                   , "--kalman-sensor-correlation-inflation"
                   , printf "%.6f" (clamp (tpKalmanSensorCorrelationInflation params) 0 1)
                   , "--kalman-innovation-inflation-threshold"
                   , printf "%.6f" (max 0 (tpKalmanInnovationInflationThreshold params))
                   , "--kalman-innovation-inflation-max"
                   , printf "%.6f" (max 1 (tpKalmanInnovationInflationMax params))
                   , "--kalman-z-min"
                   , printf "%.12g" (max 0 (tpKalmanZMin params))
                   , "--kalman-z-max"
                   , printf "%.12g" (max (max 0 (tpKalmanZMin params)) (tpKalmanZMax params))
                   ]
        cmd29 =
            case tpMaxHighVolProb params of
                Just v -> cmd28 ++ ["--max-high-vol-prob", printf "%.12g" (clamp v 0 1)]
                Nothing -> cmd28
        cmd30 =
            case tpMaxConformalWidth params of
                Just v -> cmd29 ++ ["--max-conformal-width", printf "%.12g" (max 0 v)]
                Nothing -> cmd29
        cmd31 =
            case tpMaxQuantileWidth params of
                Just v -> cmd30 ++ ["--max-quantile-width", printf "%.12g" (max 0 v)]
                Nothing -> cmd30
        cmd32 =
            if tpConfirmConformal params
                then cmd31 ++ ["--confirm-conformal"]
                else cmd31 ++ ["--no-confirm-conformal"]
        cmd33 =
            if tpConfirmQuantiles params
                then cmd32 ++ ["--confirm-quantiles"]
                else cmd32 ++ ["--no-confirm-quantiles"]
        cmd34 =
            if tpConfidenceSizing params
                then cmd33 ++ ["--confidence-sizing"]
                else cmd33 ++ ["--no-confidence-sizing"]
        cmd35 =
            cmd34
                ++ [ "--protection-min-confidence"
                   , printf "%.4f" (clamp (tpProtectionMinConfidence params) 0 1)
                   ]
        cmd36 =
            cmd35 ++ ["--min-position-size", printf "%.12g" (clamp (tpMinPositionSize params) 0 1)]
        cmd36a0 =
            cmd36
                ++ [ if tpKellyLiteSizing params then "--kelly-lite-sizing" else "--no-kelly-lite-sizing"
                   , "--kelly-lite-fraction"
                   , printf "%.12g" (max 0 (tpKellyLiteFraction params))
                   , "--kelly-lite-floor"
                   , printf "%.12g" (max 0 (tpKellyLiteFloor params))
                   , "--kelly-lite-cap"
                   , printf "%.12g" (max (max 0 (tpKellyLiteFloor params)) (tpKellyLiteCap params))
                   ]
        cmd36a =
            case tpTakeProfitPartial params of
                Just v -> cmd36a0 ++ ["--take-profit-partial", printf "%.12g" (clamp v 0 0.999999)]
                Nothing -> cmd36a0
        cmd36b =
            case tpMaxTradesPerDay params of
                Just v -> cmd36a ++ ["--max-trades-per-day", show (max 0 v)]
                Nothing -> cmd36a
        cmd36c =
            case tpMinExpectancy params of
                Just v -> cmd36b ++ ["--min-expectancy", printf "%.12g" v, "--expectancy-lookback", show (max 1 (tpExpectancyLookback params))]
                Nothing -> cmd36b ++ ["--expectancy-lookback", show (max 1 (tpExpectancyLookback params))]
        cmd36d =
            case tpLossStreakMax params of
                Just v -> cmd36c ++ ["--loss-streak-max", show (max 0 v)]
                Nothing -> cmd36c
        cmd36e =
            case tpLossStreakCooldownBars params of
                Just v -> cmd36d ++ ["--loss-streak-cooldown-bars", show (max 0 v)]
                Nothing -> cmd36d
        cmd36f =
            case tpMaxOpenPositions params of
                Just v -> cmd36e ++ ["--max-open-positions", show (max 0 v)]
                Nothing -> cmd36e
        cmd36g =
            case tpMaxGrossExposure params of
                Just v -> cmd36f ++ ["--max-gross-exposure", printf "%.12g" (max 0 v)]
                Nothing -> cmd36f
        cmd36h =
            case tpMaxNetExposure params of
                Just v -> cmd36g ++ ["--max-net-exposure", printf "%.12g" (max 0 v)]
                Nothing -> cmd36g
        cmd36i =
            case tpMaxExposurePerBase params of
                Just v -> cmd36h ++ ["--max-exposure-per-base", printf "%.12g" (max 0 v)]
                Nothing -> cmd36h
        cmd36j =
            case tpMaxOpenPerBase params of
                Just v -> cmd36i ++ ["--max-open-per-base", show (max 0 v)]
                Nothing -> cmd36i
        cmd37 =
            if useSweepThreshold
                then cmd36j ++ ["--sweep-threshold", "--tune-ratio", printf "%.6f" tuneRatio]
                else cmd36j
     in cmd37 ++ ["--json"]

{- | Heap cap passed to each spawned trial process (GHC RTS @-M@; the trader
binaries are built with @-rtsopts@). A runaway trial then dies with a clean
heap-overflow error recorded in its trial record, instead of growing until
the host OOM killer reaps it (and stalls every other trial on the box).
Override with TRADER_OPTIMIZER_TRIAL_HEAP_CAP; \"0\", \"off\" or \"none\"
disables the cap.
-}
trialHeapCapRtsArgs :: IO [String]
trialHeapCapRtsArgs = do
    raw <- lookupEnv "TRADER_OPTIMIZER_TRIAL_HEAP_CAP"
    let val =
            case fmap trim raw of
                Just s | not (null s) -> s
                _ -> "12g"
    pure $
        if map toLower val `elem` ["0", "off", "none", "disabled"]
            then []
            else ["+RTS", "-M" ++ val, "-RTS"]

runTrial :: FilePath -> [String] -> TrialParams -> Double -> Bool -> Double -> Bool -> IO TrialResult
runTrial traderBin baseArgs params0 tuneRatio useSweepThreshold timeoutSec disableLstm = do
    let params = normalizeTrialParams params0
    let cmd = buildCommand traderBin baseArgs params tuneRatio useSweepThreshold
    rtsCapArgs <- trialHeapCapRtsArgs
    let cmdArgs = drop 1 cmd ++ rtsCapArgs
    env0 <- getEnvironment
    let env = if disableLstm then setEnv "TRADER_LSTM_WEIGHTS_DIR" "" env0 else env0
        procSpec =
            (proc traderBin cmdArgs)
                { cwd = Just (takeDirectory traderBin)
                , env = Just env
                , std_out = CreatePipe
                , std_err = CreatePipe
                }
    t0 <- getPOSIXTime
    res <- runWithTimeout procSpec timeoutSec
    t1 <- getPOSIXTime
    let elapsed = realToFrac (t1 - t0)
    case res of
        Nothing ->
            pure
                TrialResult
                    { trOk = False
                    , trReason = Just ("timeout>" ++ show timeoutSec ++ "s")
                    , trElapsedSec = elapsed
                    , trOrigin = "direct"
                    , trParams = params
                    , trFinalEquity = Nothing
                    , trMetrics = Nothing
                    , trOpenThreshold = Nothing
                    , trCloseThreshold = Nothing
                    , trStdoutJson = Nothing
                    , trEligible = False
                    , trFilterReason = Nothing
                    , trObjective = "final-equity"
                    , trScore = Nothing
                    }
        Just (exitCode, out, err) ->
            case exitCode of
                ExitFailure code -> do
                    let chosen = if null err then out else err
                        trimmed = trim chosen
                        short =
                            case splitAt 300 trimmed of
                                (prefix, []) -> prefix
                                (prefix, _) -> prefix ++ "…"
                        reason = if null short then "exit=" ++ show code else short
                    pure
                        TrialResult
                            { trOk = False
                            , trReason = Just reason
                            , trElapsedSec = elapsed
                            , trOrigin = "direct"
                            , trParams = params
                            , trFinalEquity = Nothing
                            , trMetrics = Nothing
                            , trOpenThreshold = Nothing
                            , trCloseThreshold = Nothing
                            , trStdoutJson = Nothing
                            , trEligible = False
                            , trFilterReason = Nothing
                            , trObjective = "final-equity"
                            , trScore = Nothing
                            }
                ExitSuccess ->
                    case Aeson.eitherDecode (BL.fromStrict (TE.encodeUtf8 (T.pack out))) of
                        Left e ->
                            pure
                                TrialResult
                                    { trOk = False
                                    , trReason = Just ("json parse error: " ++ e)
                                    , trElapsedSec = elapsed
                                    , trOrigin = "direct"
                                    , trParams = params
                                    , trFinalEquity = Nothing
                                    , trMetrics = Nothing
                                    , trOpenThreshold = Nothing
                                    , trCloseThreshold = Nothing
                                    , trStdoutJson = Nothing
                                    , trEligible = False
                                    , trFilterReason = Nothing
                                    , trObjective = "final-equity"
                                    , trScore = Nothing
                                    }
                        Right outVal ->
                            case extractBacktest outVal of
                                Left e ->
                                    pure
                                        TrialResult
                                            { trOk = False
                                            , trReason = Just ("unexpected json schema: " ++ e)
                                            , trElapsedSec = elapsed
                                            , trOrigin = "direct"
                                            , trParams = params
                                            , trFinalEquity = Nothing
                                            , trMetrics = Nothing
                                            , trOpenThreshold = Nothing
                                            , trCloseThreshold = Nothing
                                            , trStdoutJson = Just outVal
                                            , trEligible = False
                                            , trFilterReason = Nothing
                                            , trObjective = "final-equity"
                                            , trScore = Nothing
                                            }
                                Right (metrics, finalEq, openThr, closeThr) ->
                                    pure
                                        TrialResult
                                            { trOk = True
                                            , trReason = Nothing
                                            , trElapsedSec = elapsed
                                            , trOrigin = "direct"
                                            , trParams = params
                                            , trFinalEquity = Just finalEq
                                            , trMetrics = Just metrics
                                            , trOpenThreshold = openThr
                                            , trCloseThreshold = closeThr
                                            , trStdoutJson = Just outVal
                                            , trEligible = False
                                            , trFilterReason = Nothing
                                            , trObjective = "final-equity"
                                            , trScore = Nothing
                                            }

extractBacktest :: Value -> Either String (KM.KeyMap Value, Double, Maybe Double, Maybe Double)
extractBacktest val =
    case val of
        Object obj ->
            case KM.lookup (Key.fromString "backtest") obj of
                Just (Object bt) ->
                    case KM.lookup (Key.fromString "metrics") bt of
                        Just (Object metrics) -> do
                            finalEq <- lookupRequiredFloat metrics "finalEquity"
                            openThr <- lookupOptionalFloat bt "openThreshold"
                            closeThr <- lookupOptionalFloat bt "closeThreshold"
                            Right (metrics, finalEq, openThr, closeThr)
                        _ -> Left "metrics missing"
                _ -> Left "backtest missing"
        _ -> Left "backtest missing"
  where
    lookupRequiredFloat obj key =
        case KM.lookup (Key.fromString key) obj of
            Nothing -> Left (key ++ " missing")
            Just v ->
                case coerceFloatValueLenient v of
                    Just d -> Right d
                    Nothing -> Left (key ++ " invalid")
    lookupOptionalFloat obj key =
        case KM.lookup (Key.fromString key) obj of
            Nothing -> Right Nothing
            Just Null -> Right Nothing
            Just v ->
                case coerceFloatValueLenient v of
                    Just d -> Right (Just d)
                    Nothing -> Left (key ++ " invalid")

runWithTimeout :: CreateProcess -> Double -> IO (Maybe (ExitCode, String, String))
runWithTimeout procSpec timeoutSec = do
    let timeoutMicros = max 0 (floor (timeoutSec * 1000000))
        terminateWaitMicros = 2 * 1000000
        captureWaitMicros = 2 * 1000000
        pollMicros = 50 * 1000
    (_, Just hout, Just herr, ph) <- createProcess procSpec
    hSetBinaryMode hout True
    hSetBinaryMode herr True
    (outRef, outDone, outHandle) <- startCapture hout
    (errRef, errDone, errHandle) <- startCapture herr
    exitVar <- newEmptyMVar
    _ <- forkIO $ do
        exitCode <- waitForProcess ph
        putMVar exitVar exitCode
    mExit <- waitForExit timeoutMicros pollMicros exitVar
    case mExit of
        Nothing -> do
            _ <- try (terminateProcess ph) :: IO (Either SomeException ())
            _ <- waitForExit terminateWaitMicros pollMicros exitVar
            closeHandle outHandle
            closeHandle errHandle
            _ <- waitForExit captureWaitMicros pollMicros outDone
            _ <- waitForExit captureWaitMicros pollMicros errDone
            pure Nothing
        Just exitCode -> do
            outReady <- waitForExit captureWaitMicros pollMicros outDone
            errReady <- waitForExit captureWaitMicros pollMicros errDone
            when (isNothing outReady) $ do
                closeHandle outHandle
                _ <- waitForExit captureWaitMicros pollMicros outDone
                pure ()
            when (isNothing errReady) $ do
                closeHandle errHandle
                _ <- waitForExit captureWaitMicros pollMicros errDone
                pure ()
            out <- decodeBytes <$> readIORef outRef
            err <- decodeBytes <$> readIORef errRef
            pure (Just (exitCode, out, err))
  where
    waitForExit timeoutMicros pollMicros var
        | timeoutMicros <= 0 = tryTakeMVar var
        | otherwise = go 0
      where
        pollDelay = max 1000 pollMicros
        go waited = do
            mVal <- tryTakeMVar var
            case mVal of
                Just v -> pure (Just v)
                Nothing ->
                    if waited >= timeoutMicros
                        then pure Nothing
                        else do
                            let remaining = timeoutMicros - waited
                                delay = min pollDelay remaining
                            threadDelay delay
                            go (waited + delay)
    startCapture h = do
        ref <- newIORef BS.empty
        done <- newEmptyMVar
        _ <- forkIO $ do
            let loop = do
                    chunk <- BS.hGetSome h 8192
                    if BS.null chunk
                        then pure ()
                        else do
                            modifyIORef' ref (<> chunk)
                            loop
            _ <- try loop :: IO (Either SomeException ())
            _ <- try (hClose h) :: IO (Either SomeException ())
            putMVar done ()
        pure (ref, done, h)
    closeHandle h = do
        _ <- try (hClose h) :: IO (Either SomeException ())
        pure ()
    decodeBytes bs = T.unpack (TE.decodeUtf8With TEE.lenientDecode bs)

setEnv :: String -> String -> [(String, String)] -> [(String, String)]
setEnv key val env =
    let filtered = filter (\(k, _) -> k /= key) env
     in (key, val) : filtered

trialToRecord :: TrialResult -> Maybe String -> Value
trialToRecord tr symbolLabel =
    let symbol = symbolLabel >>= sanitizeComboSymbolForPlatform (tpPlatform (trParams tr))
        currentMaxHoldBars = tpMaxHoldBars (trParams tr)
        closeTimingReport = trialCloseTimingReport (trParams tr) symbol (trStdoutJson tr)
        paramsPairs =
            [ "platform" .= tpPlatform (trParams tr)
            , "interval" .= tpInterval (trParams tr)
            , "bars" .= tpBars (trParams tr)
            , "method" .= tpMethod (trParams tr)
            , "blendWeight" .= tpBlendWeight (trParams tr)
            , "routerScorePnlWeight" .= tpRouterScorePnlWeight (trParams tr)
            , "positioning" .= tpPositioning (trParams tr)
            , "normalization" .= tpNormalization (trParams tr)
            , "baseOpenThreshold" .= tpBaseOpenThreshold (trParams tr)
            , "baseCloseThreshold" .= tpBaseCloseThreshold (trParams tr)
            , "minHoldBars" .= tpMinHoldBars (trParams tr)
            , "cooldownBars" .= tpCooldownBars (trParams tr)
            , "maxHoldBars" .= currentMaxHoldBars
            , "minEdge" .= tpMinEdge (trParams tr)
            , "minSignalToNoise" .= tpMinSignalToNoise (trParams tr)
            , "snrSizeWeight" .= tpSnrSizeWeight (trParams tr)
            , "thresholdFactorEnabled" .= tpThresholdFactorEnabled (trParams tr)
            , "thresholdFactorAlpha" .= tpThresholdFactorAlpha (trParams tr)
            , "thresholdFactorMin" .= tpThresholdFactorMin (trParams tr)
            , "thresholdFactorMax" .= tpThresholdFactorMax (trParams tr)
            , "thresholdFactorFloor" .= tpThresholdFactorFloor (trParams tr)
            , "thresholdFactorEdgeKalWeight" .= tpThresholdFactorEdgeKalWeight (trParams tr)
            , "thresholdFactorEdgeLstmWeight" .= tpThresholdFactorEdgeLstmWeight (trParams tr)
            , "thresholdFactorKalmanZWeight" .= tpThresholdFactorKalmanZWeight (trParams tr)
            , "thresholdFactorHighVolWeight" .= tpThresholdFactorHighVolWeight (trParams tr)
            , "thresholdFactorConformalWeight" .= tpThresholdFactorConformalWeight (trParams tr)
            , "thresholdFactorQuantileWeight" .= tpThresholdFactorQuantileWeight (trParams tr)
            , "thresholdFactorLstmConfWeight" .= tpThresholdFactorLstmConfWeight (trParams tr)
            , "thresholdFactorLstmHealthWeight" .= tpThresholdFactorLstmHealthWeight (trParams tr)
            , "edgeBuffer" .= tpEdgeBuffer (trParams tr)
            , "costAwareEdge" .= tpCostAwareEdge (trParams tr)
            , "trendLookback" .= tpTrendLookback (trParams tr)
            , "maxPositionSize" .= tpMaxPositionSize (trParams tr)
            , "volTarget" .= tpVolTarget (trParams tr)
            , "volLookback" .= tpVolLookback (trParams tr)
            , "volEwmaAlpha" .= tpVolEwmaAlpha (trParams tr)
            , "volFloor" .= tpVolFloor (trParams tr)
            , "volScaleMax" .= tpVolScaleMax (trParams tr)
            , "maxVolatility" .= tpMaxVolatility (trParams tr)
            , "periodsPerYear" .= tpPeriodsPerYear (trParams tr)
            , "kalmanMarketTopN" .= tpKalmanMarketTopN (trParams tr)
            , "sensorVarianceEwmaAlpha" .= tpSensorVarianceEwmaAlpha (trParams tr)
            , "fee" .= tpFee (trParams tr)
            , "fundingRate" .= tpFundingRate (trParams tr)
            , "fundingBySide" .= tpFundingBySide (trParams tr)
            , "fundingOnOpen" .= tpFundingOnOpen (trParams tr)
            , "rebalanceBars" .= tpRebalanceBars (trParams tr)
            , "rebalanceThreshold" .= tpRebalanceThreshold (trParams tr)
            , "rebalanceCostMult" .= tpRebalanceCostMult (trParams tr)
            , "rebalanceGlobal" .= tpRebalanceGlobal (trParams tr)
            , "rebalanceResetOnSignal" .= tpRebalanceResetOnSignal (trParams tr)
            , "epochs" .= tpEpochs (trParams tr)
            , "hiddenSize" .= tpHiddenSize (trParams tr)
            , "learningRate" .= tpLearningRate (trParams tr)
            , "lstmAdamBeta1" .= tpLstmAdamBeta1 (trParams tr)
            , "lstmAdamBeta2" .= tpLstmAdamBeta2 (trParams tr)
            , "lstmAdamEps" .= tpLstmAdamEps (trParams tr)
            , "valRatio" .= tpValRatio (trParams tr)
            , "patience" .= tpPatience (trParams tr)
            , "walkForwardFolds" .= tpWalkForwardFolds (trParams tr)
            , "walkForwardEmbargoBars" .= tpWalkForwardEmbargoBars (trParams tr)
            , "tuneStressVolMult" .= tpTuneStressVolMult (trParams tr)
            , "tuneStressShock" .= tpTuneStressShock (trParams tr)
            , "tuneStressWeight" .= tpTuneStressWeight (trParams tr)
            , "gradClip" .= tpGradClip (trParams tr)
            , "slippage" .= tpSlippage (trParams tr)
            , "spread" .= tpSpread (trParams tr)
            , "intrabarFill" .= tpIntrabarFill (trParams tr)
            , "triLayer" .= tpTriLayer (trParams tr)
            , "triLayerFastMult" .= tpTriLayerFastMult (trParams tr)
            , "triLayerSlowMult" .= tpTriLayerSlowMult (trParams tr)
            , "triLayerCloudPadding" .= tpTriLayerCloudPadding (trParams tr)
            , "triLayerCloudSlope" .= tpTriLayerCloudSlope (trParams tr)
            , "triLayerCloudWidth" .= tpTriLayerCloudWidth (trParams tr)
            , "triLayerTouchLookback" .= tpTriLayerTouchLookback (trParams tr)
            , "triLayerPriceAction" .= tpTriLayerPriceAction (trParams tr)
            , "triLayerPriceActionBody" .= tpTriLayerPriceActionBody (trParams tr)
            , "triLayerPriceActionWickRatio" .= tpTriLayerPriceActionWickRatio (trParams tr)
            , "triLayerPriceActionOppositeWickMax" .= tpTriLayerPriceActionOppositeWickMax (trParams tr)
            , "triLayerPriceActionBodyTolerance" .= tpTriLayerPriceActionBodyTolerance (trParams tr)
            , "triLayerExitOnSlow" .= tpTriLayerExitOnSlow (trParams tr)
            , "kalmanBandLookback" .= tpKalmanBandLookback (trParams tr)
            , "kalmanBandStdMult" .= tpKalmanBandStdMult (trParams tr)
            , "lstmExitFlipBars" .= tpLstmExitFlipBars (trParams tr)
            , "lstmExitFlipGraceBars" .= tpLstmExitFlipGraceBars (trParams tr)
            , "lstmExitFlipStrong" .= tpLstmExitFlipStrong (trParams tr)
            , "lstmConfidenceSoft" .= tpLstmConfidenceSoft (trParams tr)
            , "lstmConfidenceHard" .= tpLstmConfidenceHard (trParams tr)
            , "stopLoss" .= tpStopLoss (trParams tr)
            , "takeProfit" .= tpTakeProfit (trParams tr)
            , "trailingStop" .= tpTrailingStop (trParams tr)
            , "stopLossVolMult" .= tpStopLossVolMult (trParams tr)
            , "takeProfitVolMult" .= tpTakeProfitVolMult (trParams tr)
            , "trailingStopVolMult" .= tpTrailingStopVolMult (trParams tr)
            , "riskPerTrade" .= tpRiskPerTrade (trParams tr)
            , "maxDrawdown" .= tpMaxDrawdown (trParams tr)
            , "maxDailyLoss" .= tpMaxDailyLoss (trParams tr)
            , "maxWeeklyLoss" .= tpMaxWeeklyLoss (trParams tr)
            , "maxOrderErrors" .= tpMaxOrderErrors (trParams tr)
            , "kalmanDt" .= tpKalmanDt (trParams tr)
            , "kalmanProcessVar" .= tpKalmanProcessVar (trParams tr)
            , "kalmanMeasurementVar" .= tpKalmanMeasurementVar (trParams tr)
            , "kalmanPhysicsBars" .= tpKalmanPhysicsBars (trParams tr)
            , "kalmanPhysicsBacktestRatio" .= tpKalmanPhysicsBacktestRatio (trParams tr)
            , "kalmanPhysicsVolumeEwmaAlpha" .= tpKalmanPhysicsVolumeEwmaAlpha (trParams tr)
            , "kalmanPhysicsVolumeSignalClamp" .= tpKalmanPhysicsVolumeSignalClamp (trParams tr)
            , "kalmanPhysicsCloseBiasScale" .= tpKalmanPhysicsCloseBiasScale (trParams tr)
            , "kalmanSensorCorrelationInflation" .= tpKalmanSensorCorrelationInflation (trParams tr)
            , "kalmanInnovationInflationThreshold" .= tpKalmanInnovationInflationThreshold (trParams tr)
            , "kalmanInnovationInflationMax" .= tpKalmanInnovationInflationMax (trParams tr)
            , "kalmanZMin" .= tpKalmanZMin (trParams tr)
            , "kalmanZMax" .= tpKalmanZMax (trParams tr)
            , "maxHighVolProb" .= tpMaxHighVolProb (trParams tr)
            , "maxConformalWidth" .= tpMaxConformalWidth (trParams tr)
            , "maxQuantileWidth" .= tpMaxQuantileWidth (trParams tr)
            , "confirmConformal" .= tpConfirmConformal (trParams tr)
            , "confirmQuantiles" .= tpConfirmQuantiles (trParams tr)
            , "confidenceSizing" .= tpConfidenceSizing (trParams tr)
            , "protectionMinConfidence" .= tpProtectionMinConfidence (trParams tr)
            , "minPositionSize" .= tpMinPositionSize (trParams tr)
            , "kellyLiteSizing" .= tpKellyLiteSizing (trParams tr)
            , "kellyLiteFraction" .= tpKellyLiteFraction (trParams tr)
            , "kellyLiteFloor" .= tpKellyLiteFloor (trParams tr)
            , "kellyLiteCap" .= tpKellyLiteCap (trParams tr)
            , "predictors" .= tpPredictors (trParams tr)
            , "routerLookback" .= tpRouterLookback (trParams tr)
            , "routerRegimeMinBars" .= tpRouterRegimeMinBars (trParams tr)
            , "routerRegimeMinFraction" .= tpRouterRegimeMinFraction (trParams tr)
            , "routerMinScore" .= tpRouterMinScore (trParams tr)
            , "feeFixed" .= tpFeeFixed (trParams tr)
            , "slippageImpact" .= tpSlippageImpact (trParams tr)
            , "slippageImpactPower" .= tpSlippageImpactPower (trParams tr)
            , "slippageVolMult" .= tpSlippageVolMult (trParams tr)
            , "spreadVolMult" .= tpSpreadVolMult (trParams tr)
            , "takeProfitPartial" .= tpTakeProfitPartial (trParams tr)
            , "maxTradesPerDay" .= tpMaxTradesPerDay (trParams tr)
            , "expectancyLookback" .= tpExpectancyLookback (trParams tr)
            , "minExpectancy" .= tpMinExpectancy (trParams tr)
            , "lossStreakMax" .= tpLossStreakMax (trParams tr)
            , "lossStreakCooldownBars" .= tpLossStreakCooldownBars (trParams tr)
            , "maxOpenPositions" .= tpMaxOpenPositions (trParams tr)
            , "maxGrossExposure" .= tpMaxGrossExposure (trParams tr)
            , "maxNetExposure" .= tpMaxNetExposure (trParams tr)
            , "maxExposurePerBase" .= tpMaxExposurePerBase (trParams tr)
            , "maxOpenPerBase" .= tpMaxOpenPerBase (trParams tr)
            , "adaptiveEdgeBufferMax" .= tpAdaptiveEdgeBufferMax (trParams tr)
            , "adaptiveMinSignalToNoiseMax" .= tpAdaptiveMinSignalToNoiseMax (trParams tr)
            , "adaptiveTrendLookbackMax" .= tpAdaptiveTrendLookbackMax (trParams tr)
            , "adaptiveKalmanZMinMax" .= tpAdaptiveKalmanZMinMax (trParams tr)
            , "adaptiveWinRateSlack" .= tpAdaptiveWinRateSlack (trParams tr)
            , "adaptiveProfitFactorSlack" .= tpAdaptiveProfitFactorSlack (trParams tr)
            , "adaptiveFilters" .= tpAdaptiveFilters (trParams tr)
            , "perfLookback" .= tpPerfLookback (trParams tr)
            , "perfMinWinRate" .= tpPerfMinWinRate (trParams tr)
            , "perfMinProfitFactor" .= tpPerfMinProfitFactor (trParams tr)
            , "metaLabelFilter" .= tpMetaLabelFilter (trParams tr)
            , "metaLabelMinEdge" .= tpMetaLabelMinEdge (trParams tr)
            , "metaLabelMinConfidence" .= tpMetaLabelMinConfidence (trParams tr)
            , "metaLabelRequireBand" .= tpMetaLabelRequireBand (trParams tr)
            , "regimeParameterBank" .= tpRegimeParameterBank (trParams tr)
            , "regimeBankHysteresis" .= tpRegimeBankHysteresis (trParams tr)
            , "regimeTrendOpenMult" .= tpRegimeTrendOpenMult (trParams tr)
            , "regimeMrOpenMult" .= tpRegimeMrOpenMult (trParams tr)
            , "regimeHighVolOpenMult" .= tpRegimeHighVolOpenMult (trParams tr)
            , "regimeTrendSizeMult" .= tpRegimeTrendSizeMult (trParams tr)
            , "regimeMrSizeMult" .= tpRegimeMrSizeMult (trParams tr)
            , "regimeHighVolSizeMult" .= tpRegimeHighVolSizeMult (trParams tr)
            , "multiTimeframeConsensus" .= tpMultiTimeframeConsensus (trParams tr)
            , "mtfFastBars" .= tpMtfFastBars (trParams tr)
            , "mtfMidBars" .= tpMtfMidBars (trParams tr)
            , "mtfSlowBars" .= tpMtfSlowBars (trParams tr)
            , "mtfMinAgree" .= tpMtfMinAgree (trParams tr)
            , "crossAssetConfirmation" .= tpCrossAssetConfirmation (trParams tr)
            , "crossAssetMinBeta" .= tpCrossAssetMinBeta (trParams tr)
            , "crossAssetMinEdge" .= tpCrossAssetMinEdge (trParams tr)
            , "pairsStatArb" .= tpPairsStatArb (trParams tr)
            , "pairsStatArbLookback" .= tpPairsStatArbLookback (trParams tr)
            , "pairsStatArbZEntry" .= tpPairsStatArbZEntry (trParams tr)
            , "pairsStatArbSizeMult" .= tpPairsStatArbSizeMult (trParams tr)
            , "fundingOiAware" .= tpFundingOiAware (trParams tr)
            , "fundingOiFundingCap" .= tpFundingOiFundingCap (trParams tr)
            , "fundingOiVolLookback" .= tpFundingOiVolLookback (trParams tr)
            , "fundingOiVolCap" .= tpFundingOiVolCap (trParams tr)
            , "fundingOiSizeMult" .= tpFundingOiSizeMult (trParams tr)
            ]
        paramsPairsWithTechnical =
            paramsPairs ++ technicalTrialParamsPairs (tpTechnicalParams (trParams tr))
        paramsPairs' =
            case symbol of
                Just sym -> paramsPairsWithTechnical ++ ["binanceSymbol" .= sym]
                Nothing -> paramsPairsWithTechnical
        metricsWithWalkForward =
            applyWalkForwardSummaryMetrics
                (trMetrics tr)
                (trStdoutJson tr)
        metricsWithCloseTiming =
            applyCloseTimingMetrics
                metricsWithWalkForward
                currentMaxHoldBars
                currentMaxHoldBars
                closeTimingReport
        baseFields =
            [ "ok" .= trOk tr
            , "eligible" .= trEligible tr
            , "objective" .= trObjective tr
            , "score" .= trScore tr
            , "filterReason" .= trFilterReason tr
            , "activityDiagnostic" .= activityFilterDiagnostic tr
            , "reason" .= trReason tr
            , "elapsedSec" .= trElapsedSec tr
            , "optimizerTrialOrigin" .= trOrigin tr
            , "finalEquity" .= trFinalEquity tr
            , "openThreshold" .= trOpenThreshold tr
            , "closeThreshold" .= trCloseThreshold tr
            , "params" .= object paramsPairs'
            ]
        metricsField =
            case metricsWithCloseTiming of
                Just m -> ["metrics" .= Object m]
                Nothing -> []
        kellyLiteField =
            case extractKellyLiteSummary (trStdoutJson tr) of
                Just report -> ["kellyLite" .= Object report]
                Nothing -> []
        opsField =
            case extractOperations (trStdoutJson tr) of
                Just ops -> ["operations" .= ops]
                Nothing -> []
     in object (baseFields ++ metricsField ++ kellyLiteField ++ opsField)

sampleParams
    rng0
    platforms
    platformIntervals
    intervals
    pAutoBars
    barsMin
    barsMax
    barsDistribution
    openThresholdMin
    openThresholdMax
    closeThresholdMin
    closeThresholdMax
    minHoldBarsRange
    cooldownBarsRange
    maxHoldBarsRange
    minEdgeRange
    minSignalToNoiseRange
    snrSizeWeightRange
    pThresholdFactor
    thresholdFactorAlphaRange
    thresholdFactorMinRange
    thresholdFactorMaxRange
    thresholdFactorFloorRange
    thresholdFactorWeightRange
    edgeBufferRange
    trendLookbackRange
    maxPositionSizeRange
    volTargetRange
    volLookbackRange
    volEwmaAlphaRange
    pDisableVolEwmaAlpha
    volFloorRange
    volScaleMaxRange
    maxVolatilityRange
    periodsPerYearRange
    kalmanMarketTopNRange
    pCostAwareEdge
    feeMin
    feeMax
    fundingRateRange
    pFundingBySide
    pFundingOnOpen
    rebalanceBarsRange
    rebalanceThresholdRange
    rebalanceCostMultRange
    pRebalanceGlobal
    pRebalanceResetOnSignal
    pLongShort
    pIntrabarTakeProfitFirst
    pTriLayer
    triLayerFastRange
    triLayerSlowRange
    pTriLayerPriceAction
    triLayerCloudPaddingRange
    triLayerCloudSlopeRange
    triLayerCloudWidthRange
    triLayerTouchLookbackRange
    triLayerPriceActionBodyRange
    triLayerPriceActionWickRatioRange
    triLayerPriceActionOppositeWickMaxRange
    triLayerPriceActionBodyToleranceRange
    triLayerExitOnSlowInput
    kalmanBandLookbackRange
    kalmanBandStdMultRange
    lstmExitFlipBarsRange
    lstmExitFlipGraceBarsRange
    lstmExitFlipStrongInput
    lstmConfidenceSoftRange
    lstmConfidenceHardRange
    epochsMin
    epochsMax
    hiddenMin
    hiddenMax
    lrMin
    lrMax
    lstmAdamBeta1Range
    lstmAdamBeta2Range
    lstmAdamEpsRange
    valMin
    valMax
    patienceMax
    walkForwardFoldsRange
    walkForwardEmbargoBarsRange
    tuneStressVolMultRange
    tuneStressShockRange
    tuneStressWeightRange
    gradClipMin
    gradClipMax
    slippageMax
    spreadMax
    kalmanDtMin
    kalmanDtMax
    kalmanProcessVarMin
    kalmanProcessVarMax
    kalmanMeasurementVarMin
    kalmanMeasurementVarMax
    kalmanPhysicsBarsMin
    kalmanPhysicsBarsMax
    kalmanPhysicsBacktestRatioMin
    kalmanPhysicsBacktestRatioMax
    kalmanPhysicsVolumeEwmaAlphaMin
    kalmanPhysicsVolumeEwmaAlphaMax
    kalmanPhysicsVolumeSignalClampMin
    kalmanPhysicsVolumeSignalClampMax
    kalmanPhysicsCloseBiasScaleMin
    kalmanPhysicsCloseBiasScaleMax
    sensorVarianceEwmaAlpha
    kalmanSensorCorrelationInflation
    kalmanInnovationInflationThreshold
    kalmanInnovationInflationMax
    kalmanZMinMin
    kalmanZMinMax
    kalmanZMaxMin
    kalmanZMaxMax
    pDisableMaxHighVolProb
    maxHighVolProbRange
    pDisableMaxConformalWidth
    maxConformalWidthRange
    pDisableMaxQuantileWidth
    maxQuantileWidthRange
    pConfirmConformal
    pConfirmQuantiles
    pConfidenceSizing
    protectionMinConfidenceRange
    minPositionSizeRange
    stopRange
    takeRange
    trailRange
    stopVolMultRange
    takeVolMultRange
    trailVolMultRange
    (_methodW11, methodW10, methodW01, methodWBlend, methodWConfBlend, methodWConfPick, methodWConformalClip, methodWCostPick, methodWHarmonicBlend, methodWDisagreementGuard, methodWMedianBlend, methodWNeutralGuard, methodWRiskParityBlend, methodWConsensusBoost, methodWAnchorBlend, methodWTensionGate, methodWEntropyBlend, methodWCoherenceGate, methodWDivergenceGate, methodWFractalBlend, methodWPhaseCancel, methodWSoftmaxBlend, methodWSmoothSoftmaxBlend, methodWHedgeBlend, methodWNetSoftmaxBlend, methodWEdgeBlend, methodWEdgePick, methodWGeoBlend, methodWRegimeSwitch, methodWBanditRouter)
    normalizationChoices
    blendWeightRange
    routerScorePnlWeightRange
    pDisableStop
    pDisableTp
    pDisableTrail
    pDisableStopVolMult
    pDisableTpVolMult
    pDisableTrailVolMult
    riskPerTradeRange
    pDisableRiskPerTrade
    pDisableMaxDd
    pDisableMaxDl
    pDisableMaxWl
    pDisableMaxOe
    pDisableGradClip
    pDisableVolTarget
    pDisableMaxVolatility
    maxDdRange
    maxDlRange
    maxWlRange
    maxOeRange
    predictorChoices
    routerLookbackRange
    routerRegimeMinBarsRange
    routerRegimeMinFractionRange
    routerMinScoreRange
    feeFixedRange
    slippageImpactRange
    slippageImpactPowerRange
    slippageVolMultRange
    spreadVolMultRange
    takeProfitPartialRange
    pDisableTakeProfitPartial
    maxTradesPerDayRange
    pDisableMaxTradesPerDay
    expectancyLookbackRange
    minExpectancyRange
    pDisableMinExpectancy
    lossStreakMaxRange
    pDisableLossStreakMax
    lossStreakCooldownBarsRange
    pDisableLossStreakCooldownBars
    maxOpenPositionsRange
    pDisableMaxOpenPositions
    maxGrossExposureRange
    pDisableMaxGrossExposure
    maxNetExposureRange
    pDisableMaxNetExposure
    maxExposurePerBaseRange
    pDisableMaxExposurePerBase
    maxOpenPerBaseRange
    pDisableMaxOpenPerBase
    adaptiveEdgeBufferMaxRange
    adaptiveMinSignalToNoiseMaxRange
    adaptiveTrendLookbackMaxRange
    adaptiveKalmanZMinMaxRange
    adaptiveWinRateSlackRange
    adaptiveProfitFactorSlackRange
    pAdaptiveFilters
    perfLookbackRange
    perfMinWinRateRange
    pDisablePerfMinWinRate
    perfMinProfitFactorRange
    pDisablePerfMinProfitFactor
    pMetaLabelFilter
    metaLabelMinEdgeRange
    metaLabelMinConfidenceRange
    pMetaLabelRequireBand
    pRegimeParameterBank
    regimeBankHysteresisRange
    regimeTrendOpenMultRange
    regimeMrOpenMultRange
    regimeHighVolOpenMultRange
    regimeTrendSizeMultRange
    regimeMrSizeMultRange
    regimeHighVolSizeMultRange
    technicalRanges
    pMultiTimeframeConsensus
    mtfFastBarsRange
    mtfMidBarsRange
    mtfSlowBarsRange
    mtfMinAgreeRange
    pCrossAssetConfirmation
    crossAssetMinBetaRange
    crossAssetMinEdgeRange
    pPairsStatArb
    pairsStatArbLookbackRange
    pairsStatArbZEntryRange
    pairsStatArbSizeMultRange
    pFundingOiAware
    fundingOiFundingCapRange
    pDisableFundingOiFundingCap
    fundingOiVolLookbackRange
    fundingOiVolCapRange
    pDisableFundingOiVolCap
    fundingOiSizeMultRange
    pKellyLiteSizing
    kellyLiteFractionRange
    kellyLiteFloorRange
    kellyLiteCapRange =
        let (platform, rng1) =
                case platforms of
                    [] -> (Nothing, rng0)
                    _ ->
                        let (p, rng') = nextChoice platforms rng0
                         in (p, rng')
            intervalPool =
                case platform of
                    Nothing -> intervals
                    Just p -> fromMaybe intervals (lookup p platformIntervals)
            (intervalChoice, rng2) = nextChoice intervalPool rng1
            interval = fromMaybe (fromMaybe "1h" (listToMaybe intervals)) intervalChoice
            (bars, rng3) = sampleBars rng2
            methods =
                [ ("10", methodW10)
                , ("01", methodW01)
                , ("blend", methodWBlend)
                , ("conf_blend", methodWConfBlend)
                , ("conf_pick", methodWConfPick)
                , ("conformal_clip", methodWConformalClip)
                , ("cost_pick", methodWCostPick)
                , ("harmonic_blend", methodWHarmonicBlend)
                , ("disagreement_guard", methodWDisagreementGuard)
                , ("median_blend", methodWMedianBlend)
                , ("neutral_guard", methodWNeutralGuard)
                , ("risk_parity_blend", methodWRiskParityBlend)
                , ("consensus_boost", methodWConsensusBoost)
                , ("anchor_blend", methodWAnchorBlend)
                , ("tension_gate", methodWTensionGate)
                , ("entropy_blend", methodWEntropyBlend)
                , ("coherence_gate", methodWCoherenceGate)
                , ("divergence_gate", methodWDivergenceGate)
                , ("fractal_blend", methodWFractalBlend)
                , ("phase_cancel", methodWPhaseCancel)
                , ("softmax_blend", methodWSoftmaxBlend)
                , ("smooth_softmax_blend", methodWSmoothSoftmaxBlend)
                , ("hedge_blend", methodWHedgeBlend)
                , ("net_softmax_blend", methodWNetSoftmaxBlend)
                , ("edge_blend", methodWEdgeBlend)
                , ("edge_pick", methodWEdgePick)
                , ("geo_blend", methodWGeoBlend)
                , ("regime_switch", methodWRegimeSwitch)
                , ("bandit_router", methodWBanditRouter)
                ]
            (method, rng4) = chooseWeighted methods rng3
            (blendWeight, rng5) =
                let (bwLo, bwHi) = ordered blendWeightRange
                    (val, rng') = nextUniform bwLo bwHi rng4
                 in (clamp val 0 1, rng')
            (routerScorePnlWeight, rng5a) =
                let (rwLo, rwHi) = ordered routerScorePnlWeightRange
                    (val, rng') = nextUniform rwLo rwHi rng5
                 in (clamp val 0 1, rng')
            (normalizationChoice, rng6) = nextChoice normalizationChoices rng5a
            normalization = fromMaybe "none" normalizationChoice
            (baseOpenThreshold, rng7) =
                nextLogUniform (max 1e-12 openThresholdMin) (max 1e-12 openThresholdMax) rng6
            (baseCloseThreshold, rng8) =
                nextLogUniform (max 1e-12 closeThresholdMin) (max 1e-12 closeThresholdMax) rng7
            (minHoldBars, rng9) = uncurry nextIntRange minHoldBarsRange rng8
            (cooldownBars, rng10) = uncurry nextIntRange cooldownBarsRange rng9
            (maxHoldBars, rng11) =
                if snd maxHoldBarsRange > 0
                    then
                        let (sample, rng') = uncurry nextIntRange maxHoldBarsRange rng10
                         in if sample > 0 then (Just sample, rng') else (Nothing, rng')
                    else (Nothing, rng10)
            -- Clamp the sampled minEdge at the venue cost floor: a trial whose
            -- entry threshold sits below 1.5x round-trip cost is
            -- guaranteed-negative-expectancy on the real venue, so promoting
            -- it to the topboard only pollutes the live fleet.
            (minEdgeRaw, rng12) = uncurry nextUniform minEdgeRange rng11
            minEdge = max venueMinEdgeFloor minEdgeRaw
            (minSignalToNoise, rng13) =
                let (lo, hi) = ordered minSignalToNoiseRange
                 in nextUniform lo hi rng12
            (snrSizeWeight, rng13a) =
                let (lo, hi) = ordered snrSizeWeightRange
                    (val, rng') = nextUniform lo hi rng13
                 in (clamp val 0 1, rng')
            (edgeBuffer, rng14) = uncurry nextUniform edgeBufferRange rng13a
            (costAwareEdge, edgeBuffer', rng15) =
                if pCostAwareEdge < 0
                    then (edgeBuffer > 0, edgeBuffer, rng14)
                    else
                        let (r, rng') = nextDouble rng14
                            enabled = r < clamp pCostAwareEdge 0 1
                         in if enabled
                                then (True, edgeBuffer, rng')
                                else (False, 0, rng')
            (trendLookback, rng16) = uncurry nextIntRange trendLookbackRange rng15
            (maxPositionSize, rng17) =
                let (val, rng') = uncurry nextUniform maxPositionSizeRange rng16
                 in (max 0 val, rng')
            (volTarget, rng18) =
                if uncurry max volTargetRange > 0
                    then
                        let (r, rng') = nextDouble rng17
                         in if r >= clamp pDisableVolTarget 0 1
                                then
                                    let vtLo = max 1e-12 (uncurry min volTargetRange)
                                        vtHi = max vtLo (uncurry max volTargetRange)
                                        (val, rng'') = nextUniform vtLo vtHi rng'
                                     in (Just val, rng'')
                                else (Nothing, rng')
                    else (Nothing, rng17)
            (volEwmaAlpha, rng19) =
                if uncurry max volEwmaAlphaRange > 0
                    then
                        let (r, rng') = nextDouble rng18
                         in if r >= clamp pDisableVolEwmaAlpha 0 1
                                then
                                    let vaLo = max 1e-6 (uncurry min volEwmaAlphaRange)
                                        vaHi = min 0.999 (uncurry max volEwmaAlphaRange)
                                     in if vaHi >= vaLo
                                            then
                                                let (val, rng'') = nextUniform vaLo vaHi rng'
                                                 in (Just val, rng'')
                                            else (Nothing, rng')
                                else (Nothing, rng')
                    else (Nothing, rng18)
            (volLookback0, rng20) = uncurry nextIntRange volLookbackRange rng19
            volLookback = if isJust volTarget && isNothing volEwmaAlpha then max 2 volLookback0 else volLookback0
            (volFloor, rng21) =
                let (val, rng') = uncurry nextUniform volFloorRange rng20
                 in (max 0 val, rng')
            (volScaleMax, rng22) =
                let (val, rng') = uncurry nextUniform volScaleMaxRange rng21
                 in (max 0 val, rng')
            (periodsPerYear, rng23) =
                if uncurry max periodsPerYearRange > 0
                    then
                        let ppyMin = max 1e-12 (uncurry min periodsPerYearRange)
                            ppyMax = max ppyMin (uncurry max periodsPerYearRange)
                            (val, rng') = nextUniform ppyMin ppyMax rng22
                         in (Just val, rng')
                    else (Nothing, rng22)
            (kalmanMarketTopN, rng24) = uncurry nextIntRange kalmanMarketTopNRange rng23
            (maxVolatility, rng25) =
                if uncurry max maxVolatilityRange > 0
                    then
                        let (r, rng') = nextDouble rng24
                         in if r >= clamp pDisableMaxVolatility 0 1
                                then
                                    let mvLo = max 1e-12 (uncurry min maxVolatilityRange)
                                        mvHi = max mvLo (uncurry max maxVolatilityRange)
                                        (val, rng'') = nextUniform mvLo mvHi rng'
                                     in (Just val, rng'')
                                else (Nothing, rng')
                    else (Nothing, rng24)
            (fee, rng26) = nextUniform (max venueTakerFeeFloor feeMin) (max venueTakerFeeFloor feeMax) rng25
            (fundingRate, rng26a) =
                let (lo, hi) = ordered fundingRateRange
                 in nextUniform lo hi rng26
            (fundingBySide, rng26b) =
                let (r, rng') = nextDouble rng26a
                 in (r < clamp pFundingBySide 0 1, rng')
            (fundingOnOpen, rng26c) =
                let (r, rng') = nextDouble rng26b
                 in (r < clamp pFundingOnOpen 0 1, rng')
            (rebalanceBarsRaw, rng26d) =
                uncurry nextIntRange rebalanceBarsRange rng26c
            rebalanceBars = max 0 rebalanceBarsRaw
            (rebalanceThreshold, rng26e) =
                let (lo, hi) = ordered rebalanceThresholdRange
                 in nextUniform lo hi rng26d
            (rebalanceCostMult, rng26e') =
                let (lo, hi) = ordered rebalanceCostMultRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng26e
            (rebalanceGlobal, rng26f) =
                let (r, rng') = nextDouble rng26e'
                 in (r < clamp pRebalanceGlobal 0 1, rng')
            (rebalanceResetOnSignal, rng26g) =
                let (r, rng') = nextDouble rng26f
                 in (r < clamp pRebalanceResetOnSignal 0 1, rng')
            (epochs, rng27) = nextIntRange epochsMin epochsMax rng26g
            (hiddenSize, rng28) = nextIntRange hiddenMin hiddenMax rng27
            (learningRate, rng29) = nextLogUniform lrMin lrMax rng28
            (lstmAdamBeta1, rng29a) =
                let (lo, hi) = ordered lstmAdamBeta1Range
                 in nextUniform lo hi rng29
            (lstmAdamBeta2, rng29b) =
                let (lo, hi) = ordered lstmAdamBeta2Range
                 in nextUniform lo hi rng29a
            (lstmAdamEps, rng29c) =
                let (lo, hi) = ordered lstmAdamEpsRange
                 in nextLogUniform lo hi rng29b
            (valRatio, rng30) = nextUniform valMin valMax rng29c
            (patience, rng31) = nextIntRange 0 patienceMax rng30
            (walkForwardFolds, rng32) =
                let (lo, hi) = ordered walkForwardFoldsRange
                 in nextIntRange (max 1 lo) (max 1 hi) rng31
            (walkForwardEmbargoBarsRaw, rng32a) =
                uncurry nextIntRange walkForwardEmbargoBarsRange rng32
            walkForwardEmbargoBars = max 0 walkForwardEmbargoBarsRaw
            (tuneStressVolMult, rng33) =
                let (lo, hi) = ordered tuneStressVolMultRange
                    lo' = max 1e-12 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng32a
            (tuneStressShock, rng34) =
                let (lo, hi) = ordered tuneStressShockRange
                 in nextUniform lo hi rng33
            (tuneStressWeight, rng35) =
                let (lo, hi) = ordered tuneStressWeightRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng34
            (gradClip, rng36) = nextMaybe pDisableGradClip (nextLogUniform gradClipMin gradClipMax) rng35
            (slippage, rng37) = nextUniform venueSlippageFloor (max venueSlippageFloor slippageMax) rng36
            (spread, rng38) = nextUniform venueSpreadFloor (max venueSpreadFloor spreadMax) rng37
            (positioning, rng39) =
                let (r, rng') = nextDouble rng38
                 in (if r < clamp pLongShort 0 1 then "long-short" else "long-flat", rng')
            (intrabarFill, rng40) =
                let (r, rng') = nextDouble rng39
                 in (if r < clamp pIntrabarTakeProfitFirst 0 1 then "take-profit-first" else "stop-first", rng')
            (triLayerEnabled, rng40a) =
                let (r, rng') = nextDouble rng40
                 in (r < clamp pTriLayer 0 1, rng')
            (triLayerFastRaw, rng40b) =
                let (lo, hi) = ordered triLayerFastRange
                    lo' = max 1e-6 lo
                    hi' = max lo' hi
                 in nextLogUniform lo' hi' rng40a
            (triLayerSlowRaw, rng40c) =
                let (lo, hi) = ordered triLayerSlowRange
                    lo' = max 1e-6 lo
                    hi' = max lo' hi
                 in nextLogUniform lo' hi' rng40b
            triLayerFastMult = max 1e-6 triLayerFastRaw
            triLayerSlowMult = max triLayerFastMult (max 1e-6 triLayerSlowRaw)
            (triLayerPriceAction, rng40d) =
                if triLayerEnabled
                    then
                        let (r, rng') = nextDouble rng40c
                         in (r < clamp pTriLayerPriceAction 0 1, rng')
                    else (False, rng40c)
            (triLayerCloudPadding, rng40e) =
                let (lo, hi) = ordered triLayerCloudPaddingRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40d
            (triLayerCloudSlope, rng40f) =
                let (lo, hi) = ordered triLayerCloudSlopeRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40e
            (triLayerCloudWidth, rng40g) =
                let (lo, hi) = ordered triLayerCloudWidthRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40f
            (triLayerTouchLookback, rng40h) =
                let (lo, hi) = ordered triLayerTouchLookbackRange
                    lo' = max 1 lo
                    hi' = max lo' hi
                 in nextIntRange lo' hi' rng40g
            (triLayerPriceActionBody, rng40i) =
                let (lo, hi) = ordered triLayerPriceActionBodyRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40h
            (triLayerPriceActionWickRatio, rng40i1) =
                let (lo, hi) = ordered triLayerPriceActionWickRatioRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40i
            (triLayerPriceActionOppositeWickMax, rng40i2) =
                let (lo, hi) = ordered triLayerPriceActionOppositeWickMaxRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40i1
            (triLayerPriceActionBodyTolerance, rng40i3) =
                let (lo, hi) = ordered triLayerPriceActionBodyToleranceRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40i2
            triLayerExitOnSlow = triLayerExitOnSlowInput
            (kalmanBandLookback0, rng40j) =
                let (lo, hi) = ordered kalmanBandLookbackRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextIntRange lo' hi' rng40i3
            (kalmanBandStdMult0, rng40k) =
                let (lo, hi) = ordered kalmanBandStdMultRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40j
            kalmanBandLookback = kalmanBandLookback0
            kalmanBandStdMult =
                if kalmanBandStdMult0 > 0 && kalmanBandLookback0 < 2
                    then 0
                    else kalmanBandStdMult0
            (lstmExitFlipBars, rng40l) =
                let (lo, hi) = ordered lstmExitFlipBarsRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextIntRange lo' hi' rng40k
            (lstmExitFlipGraceBars, rng40m) =
                let (lo, hi) = ordered lstmExitFlipGraceBarsRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextIntRange lo' hi' rng40l
            (lstmConfidenceSoftRaw, rng40n) =
                let (lo, hi) = ordered lstmConfidenceSoftRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40m
            (lstmConfidenceHardRaw, rng40o) =
                let (lo, hi) = ordered lstmConfidenceHardRange
                    lo' = max 0 lo
                    hi' = max lo' hi
                 in nextUniform lo' hi' rng40n
            lstmConfidenceSoft = clamp lstmConfidenceSoftRaw 0 1
            lstmConfidenceHard = clamp (max lstmConfidenceHardRaw lstmConfidenceSoft) 0 1
            (kalmanDt, rng41) = nextUniform (max 1e-12 kalmanDtMin) (max 1e-12 kalmanDtMax) rng40o
            (kalmanProcessVar, rng42) =
                nextLogUniform (max 1e-12 kalmanProcessVarMin) (max 1e-12 kalmanProcessVarMax) rng41
            (kalmanMeasurementVar, rng43) =
                nextLogUniform (max 1e-12 kalmanMeasurementVarMin) (max 1e-12 kalmanMeasurementVarMax) rng42
            (kalmanPhysicsBars, rng43a) = nextIntRange kalmanPhysicsBarsMin kalmanPhysicsBarsMax rng43
            (kalmanPhysicsBacktestRatio, rng43b) =
                nextUniform kalmanPhysicsBacktestRatioMin kalmanPhysicsBacktestRatioMax rng43a
            (kalmanPhysicsVolumeEwmaAlpha, rng43c) =
                nextUniform kalmanPhysicsVolumeEwmaAlphaMin kalmanPhysicsVolumeEwmaAlphaMax rng43b
            (kalmanPhysicsVolumeSignalClamp, rng43d) =
                nextUniform kalmanPhysicsVolumeSignalClampMin kalmanPhysicsVolumeSignalClampMax rng43c
            (kalmanPhysicsCloseBiasScale, rng43e) =
                nextUniform kalmanPhysicsCloseBiasScaleMin kalmanPhysicsCloseBiasScaleMax rng43d
            (kalmanZMin, rng44) = nextUniform (max 0 kalmanZMinMin) (max 0 kalmanZMinMax) rng43e
            (kalmanZMax, rng45) =
                let zMaxLo = max kalmanZMin (max 0 kalmanZMaxMin)
                    zMaxHi = max zMaxLo kalmanZMaxMax
                 in nextUniform zMaxLo zMaxHi rng44
            (maxHighVolProb, rng46) =
                let (lo, hi) = ordered maxHighVolProbRange
                    (val, rng') =
                        nextMaybe pDisableMaxHighVolProb (nextUniform (min lo hi) (max lo hi)) rng45
                 in (fmap (\v -> clamp v 0 1) val, rng')
            (maxConformalWidth, rng47) =
                let (lo, hi) = ordered maxConformalWidthRange
                    lo' = max 1e-12 (min lo hi)
                    hi' = max lo' (max lo hi)
                 in nextMaybe pDisableMaxConformalWidth (nextLogUniform lo' hi') rng46
            (maxQuantileWidth, rng48) =
                let (lo, hi) = ordered maxQuantileWidthRange
                    lo' = max 1e-12 (min lo hi)
                    hi' = max lo' (max lo hi)
                 in nextMaybe pDisableMaxQuantileWidth (nextLogUniform lo' hi') rng47
            (confirmConformal, rng49) =
                let (r, rng') = nextDouble rng48
                 in (r < clamp pConfirmConformal 0 1, rng')
            (confirmQuantiles, rng50) =
                let (r, rng') = nextDouble rng49
                 in (r < clamp pConfirmQuantiles 0 1, rng')
            (confidenceSizing, rng51) =
                let (r, rng') = nextDouble rng50
                 in (r < clamp pConfidenceSizing 0 1, rng')
            (protectionMinConfidence, rng51b) =
                let (lo, hi) = ordered protectionMinConfidenceRange
                    lo' = clamp lo 0 1
                    hi' = max lo' (clamp hi 0 1)
                 in nextUniform lo' hi' rng51
            (minPositionSize, rng52) =
                if confidenceSizing
                    then
                        let (lo, hi) = ordered minPositionSizeRange
                            (val, rng') = nextUniform lo hi rng51b
                         in (clamp val 0 1, rng')
                    else (0, rng51b)
            (stopLoss, rng53) = nextMaybe pDisableStop (uncurry nextLogUniform stopRange) rng52
            (takeProfit, rng54) = nextMaybe pDisableTp (uncurry nextLogUniform takeRange) rng53
            (trailingStop, rng55) = nextMaybe pDisableTrail (uncurry nextLogUniform trailRange) rng54
            (stopLossVolMult, rng56) =
                if uncurry max stopVolMultRange > 0
                    then
                        let lo = max 1e-6 (uncurry min stopVolMultRange)
                            hi = max lo (uncurry max stopVolMultRange)
                         in nextMaybe pDisableStopVolMult (nextLogUniform lo hi) rng55
                    else (Nothing, rng55)
            (takeProfitVolMult, rng57) =
                if uncurry max takeVolMultRange > 0
                    then
                        let lo = max 1e-6 (uncurry min takeVolMultRange)
                            hi = max lo (uncurry max takeVolMultRange)
                         in nextMaybe pDisableTpVolMult (nextLogUniform lo hi) rng56
                    else (Nothing, rng56)
            (trailingStopVolMult, rng58) =
                if uncurry max trailVolMultRange > 0
                    then
                        let lo = max 1e-6 (uncurry min trailVolMultRange)
                            hi = max lo (uncurry max trailVolMultRange)
                         in nextMaybe pDisableTrailVolMult (nextLogUniform lo hi) rng57
                    else (Nothing, rng57)
            (riskPerTrade, rng58b) =
                let hasStop = isJust stopLoss || isJust stopLossVolMult
                    (lo, hi) = ordered riskPerTradeRange
                    hi' = min 0.999999 (max 0 hi)
                    lo' = min hi' (max 1e-6 lo)
                 in if not hasStop || hi' <= 0
                        then (Nothing, rng58)
                        else
                            let (val, rng') = nextMaybe pDisableRiskPerTrade (nextUniform lo' hi') rng58
                             in (fmap (\v -> clamp v 1e-6 0.999999) val, rng')
            (maxDrawdown, rng59) = nextMaybe pDisableMaxDd (uncurry nextUniform maxDdRange) rng58b
            (maxDailyLoss, rng60) = nextMaybe pDisableMaxDl (uncurry nextUniform maxDlRange) rng59
            (maxWeeklyLoss, rng60a) = nextMaybe pDisableMaxWl (uncurry nextUniform maxWlRange) rng60
            (maxOrderErrors, rng61) =
                nextMaybe
                    pDisableMaxOe
                    ( \r ->
                        let (val, r') = uncurry nextIntRange maxOeRange r
                         in (val, r')
                    )
                    rng60a
            (thresholdFactorEnabled, rng62) =
                let (r, rng') = nextDouble rng61
                 in (r < clamp pThresholdFactor 0 1, rng')
            (thresholdFactorAlpha0, rng63) =
                let (lo, hi) = ordered thresholdFactorAlphaRange
                    (val, rng') = nextUniform lo hi rng62
                 in (clamp val 0 1, rng')
            (thresholdFactorMin0, rng64) =
                let (lo, hi) = ordered thresholdFactorMinRange
                    (val, rng') = nextUniform lo hi rng63
                 in (max 0 val, rng')
            (thresholdFactorMax0, rng65) =
                let (lo, hi) = ordered thresholdFactorMaxRange
                    (val, rng') = nextUniform lo hi rng64
                 in (max 0 val, rng')
            (thresholdFactorMin, thresholdFactorMax) =
                let (lo, hi) = ordered (thresholdFactorMin0, thresholdFactorMax0)
                 in (lo, max lo hi)
            (thresholdFactorFloor, rng66) =
                let (lo, hi) = ordered thresholdFactorFloorRange
                    (val, rng') = nextUniform lo hi rng65
                 in (max 0 val, rng')
            (thresholdFactorEdgeKalWeight, rng67) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng66
            (thresholdFactorEdgeLstmWeight, rng68) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng67
            (thresholdFactorKalmanZWeight, rng69) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng68
            (thresholdFactorHighVolWeight, rng70) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng69
            (thresholdFactorConformalWeight, rng71) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng70
            (thresholdFactorQuantileWeight, rng72) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng71
            (thresholdFactorLstmConfWeight, rng73) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng72
            (thresholdFactorLstmHealthWeight, rng74) =
                let (lo, hi) = ordered thresholdFactorWeightRange
                 in nextUniform lo hi rng73
            (predictorsChoice, rng75) = nextChoice predictorChoices rng74
            predictors = fromMaybe "all" predictorsChoice
            (routerLookback, rng76) = uncurry nextIntRange routerLookbackRange rng75
            (routerRegimeMinBars, rng76a) = uncurry nextIntRange routerRegimeMinBarsRange rng76
            (routerRegimeMinFraction, rng76b) =
                let (lo, hi) = ordered routerRegimeMinFractionRange
                 in nextUniform (clamp lo 0 1) (clamp hi 0 1) rng76a
            (routerMinScore, rng77) = uncurry nextUniform routerMinScoreRange rng76b
            (feeFixed, rng78) =
                let (lo, hi) = ordered feeFixedRange
                 in nextUniform (max 0 lo) (max 0 hi) rng77
            (slippageImpact, rng79) =
                let (lo, hi) = ordered slippageImpactRange
                 in nextUniform (max 0 lo) (max 0 hi) rng78
            (slippageImpactPower, rng80) =
                let (lo, hi) = ordered slippageImpactPowerRange
                 in nextUniform (max 0 lo) (max 0 hi) rng79
            (slippageVolMult, rng81) =
                let (lo, hi) = ordered slippageVolMultRange
                 in nextUniform (max 0 lo) (max 0 hi) rng80
            (spreadVolMult, rng82) =
                let (lo, hi) = ordered spreadVolMultRange
                 in nextUniform (max 0 lo) (max 0 hi) rng81
            (takeProfitPartial, rng83) =
                sampleTakeProfitPartial
                    takeProfitPartialRange
                    pDisableTakeProfitPartial
                    rng82
            (maxTradesPerDay, rng84) =
                nextMaybe pDisableMaxTradesPerDay (uncurry nextIntRange maxTradesPerDayRange) rng83
            (expectancyLookback, rng85) =
                let (v, r) = uncurry nextIntRange expectancyLookbackRange rng84
                 in (max 1 v, r)
            (minExpectancy, rng86) =
                nextMaybe pDisableMinExpectancy (uncurry nextUniform minExpectancyRange) rng85
            (lossStreakMax, rng87) =
                nextMaybe pDisableLossStreakMax (uncurry nextIntRange lossStreakMaxRange) rng86
            (lossStreakCooldownBars, rng88) =
                nextMaybe pDisableLossStreakCooldownBars (uncurry nextIntRange lossStreakCooldownBarsRange) rng87
            (maxOpenPositions, rng89) =
                nextMaybe pDisableMaxOpenPositions (uncurry nextIntRange maxOpenPositionsRange) rng88
            (maxGrossExposure, rng90) =
                nextMaybe pDisableMaxGrossExposure (uncurry nextUniform maxGrossExposureRange) rng89
            (maxNetExposure, rng91) =
                nextMaybe pDisableMaxNetExposure (uncurry nextUniform maxNetExposureRange) rng90
            (maxExposurePerBase, rng92) =
                nextMaybe pDisableMaxExposurePerBase (uncurry nextUniform maxExposurePerBaseRange) rng91
            (maxOpenPerBase, rng93) =
                nextMaybe pDisableMaxOpenPerBase (uncurry nextIntRange maxOpenPerBaseRange) rng92
            (adaptiveEdgeBufferMax, rng94) = uncurry nextUniform adaptiveEdgeBufferMaxRange rng93
            (adaptiveMinSignalToNoiseMax, rng95) = uncurry nextUniform adaptiveMinSignalToNoiseMaxRange rng94
            (adaptiveTrendLookbackMax, rng96) = uncurry nextIntRange adaptiveTrendLookbackMaxRange rng95
            (adaptiveKalmanZMinMax, rng97) = uncurry nextUniform adaptiveKalmanZMinMaxRange rng96
            (adaptiveWinRateSlack, rng97a) =
                let (lo, hi) = ordered adaptiveWinRateSlackRange
                 in nextUniform (max 0 lo) (max 0 hi) rng97
            (adaptiveProfitFactorSlack, rng97b) =
                let (lo, hi) = ordered adaptiveProfitFactorSlackRange
                 in nextUniform (max 0 lo) (max 0 hi) rng97a
            (adaptiveFiltersEnabled, rng98) =
                let (r, rng') = nextDouble rng97b
                 in (r < pAdaptiveFilters, rng')
            (perfLookbackRaw, rng99) = uncurry nextIntRange perfLookbackRange rng98
            (perfMinWinRateRaw, rng100) =
                let (lo, hi) = ordered perfMinWinRateRange
                 in nextUniform lo hi rng99
            (perfMinWinRateEnabled, rng101) =
                let (r, rng') = nextDouble rng100
                 in (r >= pDisablePerfMinWinRate, rng')
            (perfMinProfitFactorRaw, rng102) =
                let (lo, hi) = ordered perfMinProfitFactorRange
                 in nextUniform lo hi rng101
            (perfMinProfitFactorEnabled, rng103) =
                let (r, rng') = nextDouble rng102
                 in (r >= pDisablePerfMinProfitFactor, rng')
            perfMinWinRate0 =
                if perfMinWinRateEnabled
                    then Just (clamp perfMinWinRateRaw 0 1)
                    else Nothing
            perfMinProfitFactor0 =
                if perfMinProfitFactorEnabled
                    then Just (max 0 perfMinProfitFactorRaw)
                    else Nothing
            (perfMinWinRate, perfMinProfitFactor) =
                if adaptiveFiltersEnabled && isNothing perfMinWinRate0 && isNothing perfMinProfitFactor0
                    then (Just (clamp perfMinWinRateRaw 0 1), Nothing)
                    else (perfMinWinRate0, perfMinProfitFactor0)
            perfLookback =
                if adaptiveFiltersEnabled || isJust perfMinWinRate || isJust perfMinProfitFactor
                    then max 1 perfLookbackRaw
                    else 0
            (metaLabelFilterEnabled, rng104) =
                let (r, rng') = nextDouble rng103
                 in (r < pMetaLabelFilter, rng')
            (metaLabelMinEdge, rng105) =
                let (lo, hi) = ordered metaLabelMinEdgeRange
                 in nextUniform lo hi rng104
            (metaLabelMinConfidence, rng106) =
                let (lo, hi) = ordered metaLabelMinConfidenceRange
                 in nextUniform lo hi rng105
            (metaLabelRequireBand, rng107) =
                let (r, rng') = nextDouble rng106
                 in (r < pMetaLabelRequireBand, rng')
            (regimeParameterBankEnabled, rng108) =
                let (r, rng') = nextDouble rng107
                 in (r < pRegimeParameterBank, rng')
            (regimeBankHysteresis, rng109) =
                let (lo, hi) = ordered regimeBankHysteresisRange
                 in nextUniform lo hi rng108
            (regimeTrendOpenMult, rng110) =
                let (lo, hi) = ordered regimeTrendOpenMultRange
                 in nextUniform lo hi rng109
            (regimeMrOpenMult, rng111) =
                let (lo, hi) = ordered regimeMrOpenMultRange
                 in nextUniform lo hi rng110
            (regimeHighVolOpenMult, rng112) =
                let (lo, hi) = ordered regimeHighVolOpenMultRange
                 in nextUniform lo hi rng111
            (regimeTrendSizeMult, rng113) =
                let (lo, hi) = ordered regimeTrendSizeMultRange
                 in nextUniform lo hi rng112
            (regimeMrSizeMult, rng114) =
                let (lo, hi) = ordered regimeMrSizeMultRange
                 in nextUniform lo hi rng113
            (regimeHighVolSizeMult, rng115) =
                let (lo, hi) = ordered regimeHighVolSizeMultRange
                 in nextUniform lo hi rng114
            (technicalParams, rng115a) =
                sampleTechnicalTrialParams technicalRanges rng115
            (multiTimeframeConsensusEnabled, rng116) =
                let (r, rng') = nextDouble rng115a
                 in (r < pMultiTimeframeConsensus, rng')
            (mtfFastBars, rng117) = uncurry nextIntRange mtfFastBarsRange rng116
            (mtfMidBars, rng118) = uncurry nextIntRange mtfMidBarsRange rng117
            (mtfSlowBars, rng119) = uncurry nextIntRange mtfSlowBarsRange rng118
            (mtfMinAgree, rng120) = uncurry nextIntRange mtfMinAgreeRange rng119
            (crossAssetConfirmationEnabled, rng121) =
                let (r, rng') = nextDouble rng120
                 in (r < pCrossAssetConfirmation, rng')
            (crossAssetMinBeta, rng122) =
                let (lo, hi) = ordered crossAssetMinBetaRange
                 in nextUniform lo hi rng121
            (crossAssetMinEdge, rng123) =
                let (lo, hi) = ordered crossAssetMinEdgeRange
                 in nextUniform lo hi rng122
            (pairsStatArbEnabled, rng124) =
                let (r, rng') = nextDouble rng123
                 in (r < pPairsStatArb, rng')
            (pairsStatArbLookback, rng125) = uncurry nextIntRange pairsStatArbLookbackRange rng124
            (pairsStatArbZEntry, rng126) =
                let (lo, hi) = ordered pairsStatArbZEntryRange
                 in nextUniform lo hi rng125
            (pairsStatArbSizeMult, rng127) =
                let (lo, hi) = ordered pairsStatArbSizeMultRange
                 in nextUniform lo hi rng126
            (fundingOiAwareEnabled, rng128) =
                let (r, rng') = nextDouble rng127
                 in (r < pFundingOiAware, rng')
            (fundingOiFundingCapRaw, rng129) =
                let (lo, hi) = ordered fundingOiFundingCapRange
                 in nextUniform lo hi rng128
            (fundingOiFundingCapEnabled, rng130) =
                let (r, rng') = nextDouble rng129
                 in (r >= pDisableFundingOiFundingCap, rng')
            (fundingOiVolLookback, rng131) = uncurry nextIntRange fundingOiVolLookbackRange rng130
            (fundingOiVolCapRaw, rng132) =
                let (lo, hi) = ordered fundingOiVolCapRange
                 in nextUniform lo hi rng131
            (fundingOiVolCapEnabled, rng133) =
                let (r, rng') = nextDouble rng132
                 in (r >= pDisableFundingOiVolCap, rng')
            (fundingOiSizeMult, rng134) =
                let (lo, hi) = ordered fundingOiSizeMultRange
                 in nextUniform lo hi rng133
            (kellyLiteSizingEnabled, rng135) =
                let (r, rng') = nextDouble rng134
                 in (r < pKellyLiteSizing, rng')
            (kellyLiteFraction, rng136) =
                let (lo, hi) = ordered kellyLiteFractionRange
                 in nextUniform lo hi rng135
            (kellyLiteFloor, rng137) =
                let (lo, hi) = ordered kellyLiteFloorRange
                 in nextUniform lo hi rng136
            (kellyLiteCap, rng138) =
                let (lo, hi) = ordered kellyLiteCapRange
                 in nextUniform lo hi rng137
            fundingOiFundingCap =
                if fundingOiFundingCapEnabled
                    then Just (max 0 fundingOiFundingCapRaw)
                    else Nothing
            fundingOiVolCap =
                if fundingOiVolCapEnabled
                    then Just (max 0 fundingOiVolCapRaw)
                    else Nothing
         in ( TrialParams
                { tpPlatform = platform
                , tpInterval = interval
                , tpBars = bars
                , tpMethod = method
                , tpBlendWeight = blendWeight
                , tpRouterScorePnlWeight = routerScorePnlWeight
                , tpPositioning = positioning
                , tpNormalization = normalization
                , tpBaseOpenThreshold = baseOpenThreshold
                , tpBaseCloseThreshold = baseCloseThreshold
                , tpMinHoldBars = minHoldBars
                , tpCooldownBars = cooldownBars
                , tpMaxHoldBars = maxHoldBars
                , tpMinEdge = minEdge
                , tpMinSignalToNoise = minSignalToNoise
                , tpSnrSizeWeight = snrSizeWeight
                , tpThresholdFactorEnabled = thresholdFactorEnabled
                , tpThresholdFactorAlpha = thresholdFactorAlpha0
                , tpThresholdFactorMin = thresholdFactorMin
                , tpThresholdFactorMax = thresholdFactorMax
                , tpThresholdFactorFloor = thresholdFactorFloor
                , tpThresholdFactorEdgeKalWeight = thresholdFactorEdgeKalWeight
                , tpThresholdFactorEdgeLstmWeight = thresholdFactorEdgeLstmWeight
                , tpThresholdFactorKalmanZWeight = thresholdFactorKalmanZWeight
                , tpThresholdFactorHighVolWeight = thresholdFactorHighVolWeight
                , tpThresholdFactorConformalWeight = thresholdFactorConformalWeight
                , tpThresholdFactorQuantileWeight = thresholdFactorQuantileWeight
                , tpThresholdFactorLstmConfWeight = thresholdFactorLstmConfWeight
                , tpThresholdFactorLstmHealthWeight = thresholdFactorLstmHealthWeight
                , tpEdgeBuffer = edgeBuffer'
                , tpCostAwareEdge = costAwareEdge
                , tpTrendLookback = trendLookback
                , tpMaxPositionSize = maxPositionSize
                , tpVolTarget = volTarget
                , tpVolLookback = volLookback
                , tpVolEwmaAlpha = volEwmaAlpha
                , tpVolFloor = volFloor
                , tpVolScaleMax = volScaleMax
                , tpMaxVolatility = maxVolatility
                , tpPeriodsPerYear = periodsPerYear
                , tpKalmanMarketTopN = kalmanMarketTopN
                , tpSensorVarianceEwmaAlpha = clamp sensorVarianceEwmaAlpha 0 1
                , tpFee = fee
                , tpFundingRate = fundingRate
                , tpFundingBySide = fundingBySide
                , tpFundingOnOpen = fundingOnOpen
                , tpRebalanceBars = rebalanceBars
                , tpRebalanceThreshold = rebalanceThreshold
                , tpRebalanceCostMult = rebalanceCostMult
                , tpRebalanceGlobal = rebalanceGlobal
                , tpRebalanceResetOnSignal = rebalanceResetOnSignal
                , tpEpochs = epochs
                , tpHiddenSize = hiddenSize
                , tpLearningRate = learningRate
                , tpLstmAdamBeta1 = lstmAdamBeta1
                , tpLstmAdamBeta2 = lstmAdamBeta2
                , tpLstmAdamEps = lstmAdamEps
                , tpValRatio = valRatio
                , tpPatience = patience
                , tpWalkForwardFolds = walkForwardFolds
                , tpWalkForwardEmbargoBars = walkForwardEmbargoBars
                , tpTuneStressVolMult = tuneStressVolMult
                , tpTuneStressShock = tuneStressShock
                , tpTuneStressWeight = tuneStressWeight
                , tpGradClip = gradClip
                , tpSlippage = slippage
                , tpSpread = spread
                , tpIntrabarFill = intrabarFill
                , tpTriLayer = triLayerEnabled
                , tpTriLayerFastMult = triLayerFastMult
                , tpTriLayerSlowMult = triLayerSlowMult
                , tpTriLayerCloudPadding = triLayerCloudPadding
                , tpTriLayerCloudSlope = triLayerCloudSlope
                , tpTriLayerCloudWidth = triLayerCloudWidth
                , tpTriLayerTouchLookback = triLayerTouchLookback
                , tpTriLayerPriceAction = triLayerPriceAction
                , tpTriLayerPriceActionBody = triLayerPriceActionBody
                , tpTriLayerPriceActionWickRatio = triLayerPriceActionWickRatio
                , tpTriLayerPriceActionOppositeWickMax = triLayerPriceActionOppositeWickMax
                , tpTriLayerPriceActionBodyTolerance = triLayerPriceActionBodyTolerance
                , tpTriLayerExitOnSlow = triLayerExitOnSlow
                , tpKalmanBandLookback = kalmanBandLookback
                , tpKalmanBandStdMult = kalmanBandStdMult
                , tpLstmExitFlipBars = lstmExitFlipBars
                , tpLstmExitFlipGraceBars = lstmExitFlipGraceBars
                , tpLstmExitFlipStrong = lstmExitFlipStrongInput
                , tpLstmConfidenceSoft = lstmConfidenceSoft
                , tpLstmConfidenceHard = lstmConfidenceHard
                , tpStopLoss = stopLoss
                , tpTakeProfit = takeProfit
                , tpTrailingStop = trailingStop
                , tpStopLossVolMult = stopLossVolMult
                , tpTakeProfitVolMult = takeProfitVolMult
                , tpTrailingStopVolMult = trailingStopVolMult
                , tpRiskPerTrade = riskPerTrade
                , tpMaxDrawdown = maxDrawdown
                , tpMaxDailyLoss = maxDailyLoss
                , tpMaxWeeklyLoss = maxWeeklyLoss
                , tpMaxOrderErrors = maxOrderErrors
                , tpKalmanDt = kalmanDt
                , tpKalmanProcessVar = kalmanProcessVar
                , tpKalmanMeasurementVar = kalmanMeasurementVar
                , tpKalmanPhysicsBars = max 4 kalmanPhysicsBars
                , tpKalmanPhysicsBacktestRatio = clamp kalmanPhysicsBacktestRatio 0.000001 0.999999
                , tpKalmanPhysicsVolumeEwmaAlpha = clamp kalmanPhysicsVolumeEwmaAlpha 0 1
                , tpKalmanPhysicsVolumeSignalClamp = max 0 kalmanPhysicsVolumeSignalClamp
                , tpKalmanPhysicsCloseBiasScale = kalmanPhysicsCloseBiasScale
                , tpKalmanSensorCorrelationInflation = clamp kalmanSensorCorrelationInflation 0 1
                , tpKalmanInnovationInflationThreshold = max 0 kalmanInnovationInflationThreshold
                , tpKalmanInnovationInflationMax = max 1 kalmanInnovationInflationMax
                , tpKalmanZMin = kalmanZMin
                , tpKalmanZMax = kalmanZMax
                , tpMaxHighVolProb = maxHighVolProb
                , tpMaxConformalWidth = maxConformalWidth
                , tpMaxQuantileWidth = maxQuantileWidth
                , tpConfirmConformal = confirmConformal
                , tpConfirmQuantiles = confirmQuantiles
                , tpConfidenceSizing = confidenceSizing
                , tpProtectionMinConfidence = protectionMinConfidence
                , tpMinPositionSize = minPositionSize
                , tpKellyLiteSizing = kellyLiteSizingEnabled
                , tpKellyLiteFraction = max 0 kellyLiteFraction
                , tpKellyLiteFloor = max 0 kellyLiteFloor
                , tpKellyLiteCap = max (max 0 kellyLiteFloor) kellyLiteCap
                , tpPredictors = predictors
                , tpRouterLookback = routerLookback
                , tpRouterRegimeMinBars = routerRegimeMinBars
                , tpRouterRegimeMinFraction = routerRegimeMinFraction
                , tpRouterMinScore = routerMinScore
                , tpFeeFixed = feeFixed
                , tpSlippageImpact = slippageImpact
                , tpSlippageImpactPower = slippageImpactPower
                , tpSlippageVolMult = slippageVolMult
                , tpSpreadVolMult = spreadVolMult
                , tpTakeProfitPartial = takeProfitPartial
                , tpMaxTradesPerDay = maxTradesPerDay
                , tpExpectancyLookback = expectancyLookback
                , tpMinExpectancy = minExpectancy
                , tpLossStreakMax = lossStreakMax
                , tpLossStreakCooldownBars = lossStreakCooldownBars
                , tpMaxOpenPositions = maxOpenPositions
                , tpMaxGrossExposure = maxGrossExposure
                , tpMaxNetExposure = maxNetExposure
                , tpMaxExposurePerBase = maxExposurePerBase
                , tpMaxOpenPerBase = maxOpenPerBase
                , tpAdaptiveEdgeBufferMax = adaptiveEdgeBufferMax
                , tpAdaptiveMinSignalToNoiseMax = adaptiveMinSignalToNoiseMax
                , tpAdaptiveTrendLookbackMax = adaptiveTrendLookbackMax
                , tpAdaptiveKalmanZMinMax = adaptiveKalmanZMinMax
                , tpAdaptiveWinRateSlack = adaptiveWinRateSlack
                , tpAdaptiveProfitFactorSlack = adaptiveProfitFactorSlack
                , tpAdaptiveFilters = adaptiveFiltersEnabled
                , tpPerfLookback = perfLookback
                , tpPerfMinWinRate = perfMinWinRate
                , tpPerfMinProfitFactor = perfMinProfitFactor
                , tpMetaLabelFilter = metaLabelFilterEnabled
                , tpMetaLabelMinEdge = max 0 metaLabelMinEdge
                , tpMetaLabelMinConfidence = clamp metaLabelMinConfidence 0 1
                , tpMetaLabelRequireBand = metaLabelRequireBand
                , tpRegimeParameterBank = regimeParameterBankEnabled
                , tpRegimeBankHysteresis = clamp regimeBankHysteresis 0 1
                , tpRegimeTrendOpenMult = max 0 regimeTrendOpenMult
                , tpRegimeMrOpenMult = max 0 regimeMrOpenMult
                , tpRegimeHighVolOpenMult = max 0 regimeHighVolOpenMult
                , tpRegimeTrendSizeMult = max 0 regimeTrendSizeMult
                , tpRegimeMrSizeMult = max 0 regimeMrSizeMult
                , tpRegimeHighVolSizeMult = max 0 regimeHighVolSizeMult
                , tpTechnicalParams = technicalParams
                , tpMultiTimeframeConsensus = multiTimeframeConsensusEnabled
                , tpMtfFastBars = max 1 mtfFastBars
                , tpMtfMidBars = max 1 mtfMidBars
                , tpMtfSlowBars = max 1 mtfSlowBars
                , tpMtfMinAgree = clampInt mtfMinAgree 1 3
                , tpCrossAssetConfirmation = crossAssetConfirmationEnabled
                , tpCrossAssetMinBeta = max 0 crossAssetMinBeta
                , tpCrossAssetMinEdge = max 0 crossAssetMinEdge
                , tpPairsStatArb = pairsStatArbEnabled
                , tpPairsStatArbLookback = max 2 pairsStatArbLookback
                , tpPairsStatArbZEntry = max 1e-12 pairsStatArbZEntry
                , tpPairsStatArbSizeMult = clamp pairsStatArbSizeMult 0 1
                , tpFundingOiAware = fundingOiAwareEnabled
                , tpFundingOiFundingCap = fundingOiFundingCap
                , tpFundingOiVolLookback = max 2 fundingOiVolLookback
                , tpFundingOiVolCap = fundingOiVolCap
                , tpFundingOiSizeMult = clamp fundingOiSizeMult 0 1
                }
            , rng138
            )
      where
        ordered (a, b) = if a <= b then (a, b) else (b, a)
        sampleBars rng =
            let (r, rng') = nextDouble rng
             in if r < clamp pAutoBars 0 1
                    then (0, rng')
                    else
                        if map toLower (trim barsDistribution) == "log"
                            then
                                let lo = max 2 barsMin
                                    hi = max lo barsMax
                                    (val, rng'') = nextLogUniform (fromIntegral lo) (fromIntegral hi) rng'
                                 in (round val, rng'')
                            else nextIntRange barsMin barsMax rng'
        chooseWeighted options rng =
            case options of
                [] -> ("", rng)
                (firstOpt : _) ->
                    let weights = map (max 0 . snd) options
                        total = sum weights
                     in if total <= 0
                            then
                                let (choice, rng') = nextChoice (map fst options) rng
                                 in (fromMaybe (fst firstOpt) choice, rng')
                            else
                                let (r, rng') = nextDouble rng
                                    target = r * total
                                    pick acc ((name, w) : rest) =
                                        let acc' = acc + max 0 w
                                         in if target <= acc' then name else pick acc' rest
                                    pick _ [] = fst (fromMaybe firstOpt (listToMaybe (reverse options)))
                                 in (pick 0 options, rng')

runOptimizer :: OptimizerArgs -> IO Int
runOptimizer args0 = do
    let args = if oaQuality args0 then applyQualityPreset args0 else args0
        roiScoreConfig = roiScoreConfigFromOptimizerArgs args
    traderBin <- resolveTraderBin args
    case traderBin of
        Left err -> do
            hPutStrLn stderr err
            pure 2
        Right traderBinPath -> do
            intervalsIn <- resolveIntervals args
            case intervalsIn of
                Left err -> do
                    hPutStrLn stderr err
                    pure 2
                Right intervalsRaw -> do
                    maxBarsCap <- resolveOptimizerMaxPoints
                    csvInfo <-
                        case oaData args of
                            Nothing -> pure (Nothing, maxBarsCap, Nothing)
                            Just raw -> do
                                p0 <- expandUser raw
                                exists <- doesFileExist p0
                                if not exists
                                    then do
                                        hPutStrLn stderr ("--data not found: " ++ p0)
                                        pure (Just (Left "missing"), 0, Nothing)
                                    else do
                                        p <- canonicalizePath p0
                                        let autoDetect = oaAutoHighLow args && null (oaHighColumn args) && null (oaLowColumn args)
                                        (highCol, lowCol) <-
                                            if autoDetect
                                                then do
                                                    (h, l) <- detectHighLowColumns p
                                                    case (h, l) of
                                                        (Just hc, Just lc) -> do
                                                            hPutStrLn stderr ("Auto-detected high/low columns: " ++ hc ++ "/" ++ lc)
                                                            pure (Just hc, Just lc)
                                                        _ -> pure (Nothing, Nothing)
                                                else pure (Nothing, Nothing)
                                        rows <- countCsvRows p
                                        let barsCap = max 2 (max 0 (rows - 1))
                                            cols =
                                                if autoDetect
                                                    then (highCol, lowCol)
                                                    else
                                                        let hCol = trim (oaHighColumn args)
                                                            lCol = trim (oaLowColumn args)
                                                         in (if null hCol then Nothing else Just hCol, if null lCol then Nothing else Just lCol)
                                        pure (Nothing, barsCap, Just (p, cols))
                    case csvInfo of
                        (Just (Left _), _, _) -> pure 2
                        (_, maxBarsCap, csvCols) -> do
                            let (platforms, platformIntervalsMap, intervalsResolved) =
                                    if isNothing (oaBinanceSymbol args)
                                        then ([], [], Right (pickIntervals intervalsRaw (oaLookbackWindow args) maxBarsCap))
                                        else
                                            let rawPlatforms =
                                                    case oaPlatform args of
                                                        Just v | not (null (trim v)) -> v
                                                        _ ->
                                                            case oaPlatforms args of
                                                                Just v | not (null (trim v)) -> v
                                                                _ -> "binance"
                                             in case parsePlatforms rawPlatforms of
                                                    Left e -> ([], [], Left e)
                                                    Right ps ->
                                                        case resolvePlatformIntervals ps intervalsRaw (oaLookbackWindow args) maxBarsCap of
                                                            Left e -> ([], [], Left e)
                                                            Right platformIntervalsMap' ->
                                                                let allIntervals = sort (unique (concatMap snd platformIntervalsMap'))
                                                                 in (ps, platformIntervalsMap', Right allIntervals)
                            case intervalsResolved of
                                Left err -> do
                                    hPutStrLn stderr err
                                    pure 2
                                Right intervals ->
                                    if null intervals
                                        then do
                                            hPutStrLn
                                                stderr
                                                ( "No feasible intervals for lookback-window="
                                                    ++ show (oaLookbackWindow args)
                                                    ++ " and max bars="
                                                    ++ show maxBarsCap
                                                    ++ "."
                                                )
                                            pure 2
                                        else do
                                            let useSweepThreshold = not (oaNoSweepThreshold args)
                                            barsResult <- resolveBars args intervals maxBarsCap useSweepThreshold
                                            case barsResult of
                                                Left err -> do
                                                    hPutStrLn stderr err
                                                    pure 2
                                                Right (barsMin, barsMax) -> do
                                                    let barsAutoProb = clamp (oaBarsAutoProb args) 0 1
                                                        barsDistribution = map toLower (trim (oaBarsDistribution args))
                                                        epochsMin = max 0 (oaEpochsMin args)
                                                        epochsMax = max epochsMin (oaEpochsMax args)
                                                        hiddenMin = max 1 (oaHiddenSizeMin args)
                                                        hiddenMax = max hiddenMin (oaHiddenSizeMax args)
                                                        lrMin = max 1e-9 (oaLrMin args)
                                                        lrMax = max lrMin (oaLrMax args)
                                                        lstmAdamBeta1Range =
                                                            lstmAdamBetaRange
                                                                (oaLstmAdamBeta1 args)
                                                                (oaLstmAdamBeta1Min args)
                                                                (oaLstmAdamBeta1Max args)
                                                        lstmAdamBeta2Range =
                                                            lstmAdamBetaRange
                                                                (oaLstmAdamBeta2 args)
                                                                (oaLstmAdamBeta2Min args)
                                                                (oaLstmAdamBeta2Max args)
                                                        lstmAdamEpsRange' =
                                                            lstmAdamEpsRange
                                                                (oaLstmAdamEps args)
                                                                (oaLstmAdamEpsMin args)
                                                                (oaLstmAdamEpsMax args)
                                                        valMin = clamp (oaValRatioMin args) 0 0.9
                                                        valMax = max valMin (oaValRatioMax args)
                                                        patienceMax = max 0 (oaPatienceMax args)
                                                        gradClipMin = max 1e-9 (oaGradClipMin args)
                                                        gradClipMax = max gradClipMin (oaGradClipMax args)
                                                        walkForwardFoldsMin = max 1 (oaWalkForwardFoldsMin args)
                                                        walkForwardFoldsMax = max walkForwardFoldsMin (oaWalkForwardFoldsMax args)
                                                        walkForwardEmbargoBarsMin = max 0 (oaWalkForwardEmbargoBarsMin args)
                                                        walkForwardEmbargoBarsMax = max walkForwardEmbargoBarsMin (oaWalkForwardEmbargoBarsMax args)
                                                        walkForwardEmbargoBarsRange = (walkForwardEmbargoBarsMin, walkForwardEmbargoBarsMax)
                                                        tuneStressVolMultBase = max 1e-12 (oaTuneStressVolMult args)
                                                        tuneStressShockBase = oaTuneStressShock args
                                                        tuneStressWeightBase = max 0 (oaTuneStressWeight args)
                                                        (tuneStressVolMultRange, tuneStressShockRange, tuneStressWeightRange) =
                                                            resolveStressRanges
                                                                tuneStressVolMultBase
                                                                tuneStressShockBase
                                                                tuneStressWeightBase
                                                                (oaTuneStressVolMultMin args, oaTuneStressVolMultMax args)
                                                                (oaTuneStressShockMin args, oaTuneStressShockMax args)
                                                                (oaTuneStressWeightMin args, oaTuneStressWeightMax args)
                                                        stopRange = (oaStopMin args, oaStopMax args)
                                                        takeRange = (oaTpMin args, oaTpMax args)
                                                        trailRange = (oaTrailMin args, oaTrailMax args)
                                                        stopVolMultRange = (max 0 (oaStopVolMultMin args), max 0 (oaStopVolMultMax args))
                                                        takeVolMultRange = (max 0 (oaTpVolMultMin args), max 0 (oaTpVolMultMax args))
                                                        trailVolMultRange = (max 0 (oaTrailVolMultMin args), max 0 (oaTrailVolMultMax args))
                                                        riskPerTradeMin = max 0 (oaRiskPerTradeMin args)
                                                        riskPerTradeMax = max riskPerTradeMin (oaRiskPerTradeMax args)
                                                        riskPerTradeRange = (riskPerTradeMin, riskPerTradeMax)
                                                        feeMin = max 0 (oaFeeMin args)
                                                        feeMax = max feeMin (oaFeeMax args)
                                                        fundingRateMin = oaFundingRateMin args
                                                        fundingRateMax = oaFundingRateMax args
                                                        fundingRateRange = (fundingRateMin, fundingRateMax)
                                                        pFundingBySide = clamp (oaPFundingBySide args) 0 1
                                                        pFundingOnOpen = clamp (oaPFundingOnOpen args) 0 1
                                                        rebalanceBarsMin = max 0 (oaRebalanceBarsMin args)
                                                        rebalanceBarsMax = max rebalanceBarsMin (oaRebalanceBarsMax args)
                                                        rebalanceBarsRange = (rebalanceBarsMin, rebalanceBarsMax)
                                                        rebalanceThresholdMin = max 0 (oaRebalanceThresholdMin args)
                                                        rebalanceThresholdMax = max rebalanceThresholdMin (oaRebalanceThresholdMax args)
                                                        rebalanceThresholdRange = (rebalanceThresholdMin, rebalanceThresholdMax)
                                                        rebalanceCostMultMin = max 0 (oaRebalanceCostMultMin args)
                                                        rebalanceCostMultMax = max rebalanceCostMultMin (oaRebalanceCostMultMax args)
                                                        rebalanceCostMultRange = (rebalanceCostMultMin, rebalanceCostMultMax)
                                                        pRebalanceGlobal = clamp (oaPRebalanceGlobal args) 0 1
                                                        pRebalanceResetOnSignal = clamp (oaPRebalanceResetOnSignal args) 0 1
                                                        openThresholdMin = max 1e-12 (oaOpenThresholdMin args)
                                                        openThresholdMaxRaw = max openThresholdMin (oaOpenThresholdMax args)
                                                        openThresholdMax = max openThresholdMin (min signalEntryOpenThresholdFeasibilityCap openThresholdMaxRaw)
                                                        closeThresholdMin = max 1e-12 (oaCloseThresholdMin args)
                                                        closeThresholdMax = max closeThresholdMin (oaCloseThresholdMax args)
                                                        minHoldMin = max 0 (oaMinHoldBarsMin args)
                                                        minHoldMax = max minHoldMin (oaMinHoldBarsMax args)
                                                        cooldownMin = max 0 (oaCooldownBarsMin args)
                                                        cooldownMax = max cooldownMin (oaCooldownBarsMax args)
                                                        maxHoldMin = max 0 (oaMaxHoldBarsMin args)
                                                        maxHoldMax = max maxHoldMin (oaMaxHoldBarsMax args)
                                                        minEdgeMin = max 0 (oaMinEdgeMin args)
                                                        minEdgeMax = max minEdgeMin (oaMinEdgeMax args)
                                                        minSnMin = max 0 (oaMinSignalToNoiseMin args)
                                                        minSnMax = max minSnMin (oaMinSignalToNoiseMax args)
                                                        snrSizeWeightMin = clamp (oaSnrSizeWeightMin args) 0 1
                                                        snrSizeWeightMax = clamp (oaSnrSizeWeightMax args) 0 1
                                                        snrSizeWeightRange = (snrSizeWeightMin, snrSizeWeightMax)
                                                        pThresholdFactor = clamp (oaPThresholdFactor args) 0 1
                                                        thresholdFactorAlphaMin = clamp (oaThresholdFactorAlphaMin args) 0 1
                                                        thresholdFactorAlphaMax = clamp (oaThresholdFactorAlphaMax args) 0 1
                                                        thresholdFactorAlphaRange = (thresholdFactorAlphaMin, thresholdFactorAlphaMax)
                                                        thresholdFactorMinMin = max 0 (oaThresholdFactorMinMin args)
                                                        thresholdFactorMinMax = max thresholdFactorMinMin (oaThresholdFactorMinMax args)
                                                        thresholdFactorMinRange = (thresholdFactorMinMin, thresholdFactorMinMax)
                                                        thresholdFactorMaxMin = max 0 (oaThresholdFactorMaxMin args)
                                                        thresholdFactorMaxMax = max thresholdFactorMaxMin (oaThresholdFactorMaxMax args)
                                                        thresholdFactorMaxRange = (thresholdFactorMaxMin, thresholdFactorMaxMax)
                                                        thresholdFactorFloorMin = max 0 (oaThresholdFactorFloorMin args)
                                                        thresholdFactorFloorMax = max thresholdFactorFloorMin (oaThresholdFactorFloorMax args)
                                                        thresholdFactorFloorRange = (thresholdFactorFloorMin, thresholdFactorFloorMax)
                                                        thresholdFactorWeightRange = (oaThresholdFactorWeightMin args, oaThresholdFactorWeightMax args)
                                                        edgeBufferMin = max 0 (oaEdgeBufferMin args)
                                                        edgeBufferMax = max edgeBufferMin (oaEdgeBufferMax args)
                                                        trendLookbackMin = max 0 (oaTrendLookbackMin args)
                                                        trendLookbackMax = max trendLookbackMin (oaTrendLookbackMax args)
                                                        pLongShort = clamp (oaPLongShort args) 0 1
                                                        pIntrabarTakeProfitFirst = clamp (oaPIntrabarTakeProfitFirst args) 0 1
                                                        pTriLayer = clamp (oaPTriLayer args) 0 1
                                                        triLayerFastMin = max 1e-6 (oaTriLayerFastMultMin args)
                                                        triLayerFastMax = max triLayerFastMin (oaTriLayerFastMultMax args)
                                                        triLayerFastRange = (triLayerFastMin, triLayerFastMax)
                                                        triLayerSlowMin = max 1e-6 (oaTriLayerSlowMultMin args)
                                                        triLayerSlowMax = max triLayerSlowMin (oaTriLayerSlowMultMax args)
                                                        triLayerSlowRange = (triLayerSlowMin, triLayerSlowMax)
                                                        pTriLayerPriceAction = clamp (oaPTriLayerPriceAction args) 0 1
                                                        triLayerCloudPaddingMin = max 0 (oaTriLayerCloudPaddingMin args)
                                                        triLayerCloudPaddingMax = max triLayerCloudPaddingMin (oaTriLayerCloudPaddingMax args)
                                                        triLayerCloudPaddingRange = (triLayerCloudPaddingMin, triLayerCloudPaddingMax)
                                                        triLayerCloudSlopeMin = max 0 (oaTriLayerCloudSlopeMin args)
                                                        triLayerCloudSlopeMax = max triLayerCloudSlopeMin (oaTriLayerCloudSlopeMax args)
                                                        triLayerCloudSlopeRange = (triLayerCloudSlopeMin, triLayerCloudSlopeMax)
                                                        triLayerCloudWidthMin = max 0 (oaTriLayerCloudWidthMin args)
                                                        triLayerCloudWidthMax = max triLayerCloudWidthMin (oaTriLayerCloudWidthMax args)
                                                        triLayerCloudWidthRange = (triLayerCloudWidthMin, triLayerCloudWidthMax)
                                                        triLayerTouchLookbackMin = max 1 (oaTriLayerTouchLookbackMin args)
                                                        triLayerTouchLookbackMax = max triLayerTouchLookbackMin (oaTriLayerTouchLookbackMax args)
                                                        triLayerTouchLookbackRange = (triLayerTouchLookbackMin, triLayerTouchLookbackMax)
                                                        triLayerPriceActionBodyMin = max 0 (oaTriLayerPriceActionBodyMin args)
                                                        triLayerPriceActionBodyMax = max triLayerPriceActionBodyMin (oaTriLayerPriceActionBodyMax args)
                                                        triLayerPriceActionBodyRange = (triLayerPriceActionBodyMin, triLayerPriceActionBodyMax)
                                                        triLayerPriceActionWickRatioMin = max 0 (oaTriLayerPriceActionWickRatioMin args)
                                                        triLayerPriceActionWickRatioMax = max triLayerPriceActionWickRatioMin (oaTriLayerPriceActionWickRatioMax args)
                                                        triLayerPriceActionWickRatioRange = (triLayerPriceActionWickRatioMin, triLayerPriceActionWickRatioMax)
                                                        triLayerPriceActionOppositeWickMaxMin = max 0 (oaTriLayerPriceActionOppositeWickMaxMin args)
                                                        triLayerPriceActionOppositeWickMaxMax = max triLayerPriceActionOppositeWickMaxMin (oaTriLayerPriceActionOppositeWickMaxMax args)
                                                        triLayerPriceActionOppositeWickMaxRange = (triLayerPriceActionOppositeWickMaxMin, triLayerPriceActionOppositeWickMaxMax)
                                                        triLayerPriceActionBodyToleranceMin = max 0 (oaTriLayerPriceActionBodyToleranceMin args)
                                                        triLayerPriceActionBodyToleranceMax = max triLayerPriceActionBodyToleranceMin (oaTriLayerPriceActionBodyToleranceMax args)
                                                        triLayerPriceActionBodyToleranceRange = (triLayerPriceActionBodyToleranceMin, triLayerPriceActionBodyToleranceMax)
                                                        kalmanBandLookbackMin = max 0 (oaKalmanBandLookbackMin args)
                                                        kalmanBandLookbackMax = max kalmanBandLookbackMin (oaKalmanBandLookbackMax args)
                                                        kalmanBandLookbackRange = (kalmanBandLookbackMin, kalmanBandLookbackMax)
                                                        kalmanBandStdMultMin = max 0 (oaKalmanBandStdMultMin args)
                                                        kalmanBandStdMultMax = max kalmanBandStdMultMin (oaKalmanBandStdMultMax args)
                                                        kalmanBandStdMultRange = (kalmanBandStdMultMin, kalmanBandStdMultMax)
                                                        lstmExitFlipBarsMin = max 0 (oaLstmExitFlipBarsMin args)
                                                        lstmExitFlipBarsMax = max lstmExitFlipBarsMin (oaLstmExitFlipBarsMax args)
                                                        lstmExitFlipBarsRange = (lstmExitFlipBarsMin, lstmExitFlipBarsMax)
                                                        lstmExitFlipGraceBarsMin = max 0 (oaLstmExitFlipGraceBarsMin args)
                                                        lstmExitFlipGraceBarsMax = max lstmExitFlipGraceBarsMin (oaLstmExitFlipGraceBarsMax args)
                                                        lstmExitFlipGraceBarsRange = (lstmExitFlipGraceBarsMin, lstmExitFlipGraceBarsMax)
                                                        lstmConfidenceSoftMin = max 0 (oaLstmConfidenceSoftMin args)
                                                        lstmConfidenceSoftMax = max lstmConfidenceSoftMin (oaLstmConfidenceSoftMax args)
                                                        lstmConfidenceSoftRange = (lstmConfidenceSoftMin, lstmConfidenceSoftMax)
                                                        lstmConfidenceHardMin = max 0 (oaLstmConfidenceHardMin args)
                                                        lstmConfidenceHardMax = max lstmConfidenceHardMin (oaLstmConfidenceHardMax args)
                                                        lstmConfidenceHardRange = (lstmConfidenceHardMin, lstmConfidenceHardMax)
                                                        kalmanDtMin = max 1e-12 (oaKalmanDtMin args)
                                                        kalmanDtMax = max kalmanDtMin (oaKalmanDtMax args)
                                                        kalmanProcessVarMin = max 1e-12 (oaKalmanProcessVarMin args)
                                                        kalmanProcessVarMax = max kalmanProcessVarMin (oaKalmanProcessVarMax args)
                                                        kalmanMeasurementVarMin = max 1e-12 (oaKalmanMeasurementVarMin args)
                                                        kalmanMeasurementVarMax = max kalmanMeasurementVarMin (oaKalmanMeasurementVarMax args)
                                                        kalmanPhysicsBarsMin = max 4 (oaKalmanPhysicsBarsMin args)
                                                        kalmanPhysicsBarsMax = max kalmanPhysicsBarsMin (oaKalmanPhysicsBarsMax args)
                                                        kalmanPhysicsBacktestRatioMin =
                                                            clamp (oaKalmanPhysicsBacktestRatioMin args) 0.000001 0.999999
                                                        kalmanPhysicsBacktestRatioMax =
                                                            max
                                                                kalmanPhysicsBacktestRatioMin
                                                                (clamp (oaKalmanPhysicsBacktestRatioMax args) 0.000001 0.999999)
                                                        kalmanPhysicsVolumeEwmaAlphaMin =
                                                            clamp (oaKalmanPhysicsVolumeEwmaAlphaMin args) 0 1
                                                        kalmanPhysicsVolumeEwmaAlphaMax =
                                                            max
                                                                kalmanPhysicsVolumeEwmaAlphaMin
                                                                (clamp (oaKalmanPhysicsVolumeEwmaAlphaMax args) 0 1)
                                                        kalmanPhysicsVolumeSignalClampMin = max 0 (oaKalmanPhysicsVolumeSignalClampMin args)
                                                        kalmanPhysicsVolumeSignalClampMax =
                                                            max kalmanPhysicsVolumeSignalClampMin (oaKalmanPhysicsVolumeSignalClampMax args)
                                                        kalmanPhysicsCloseBiasScaleMin = oaKalmanPhysicsCloseBiasScaleMin args
                                                        kalmanPhysicsCloseBiasScaleMax =
                                                            max kalmanPhysicsCloseBiasScaleMin (oaKalmanPhysicsCloseBiasScaleMax args)
                                                        kalmanZMinMin = max 0 (oaKalmanZMinMin args)
                                                        kalmanZMinMax = max kalmanZMinMin (oaKalmanZMinMax args)
                                                        kalmanZMaxMin = max 0 (oaKalmanZMaxMin args)
                                                        kalmanZMaxMax = max kalmanZMaxMin (oaKalmanZMaxMax args)
                                                        kalmanMarketTopNMin = max 0 (oaKalmanMarketTopNMin args)
                                                        kalmanMarketTopNMax = max kalmanMarketTopNMin (oaKalmanMarketTopNMax args)
                                                        pDisableMaxHighVolProb = clamp (oaPDisableMaxHighVolProb args) 0 1
                                                        maxHighVolProbRange = (oaMaxHighVolProbMin args, oaMaxHighVolProbMax args)
                                                        pDisableMaxConformalWidth = clamp (oaPDisableMaxConformalWidth args) 0 1
                                                        maxConformalWidthRange = (oaMaxConformalWidthMin args, oaMaxConformalWidthMax args)
                                                        pDisableMaxQuantileWidth = clamp (oaPDisableMaxQuantileWidth args) 0 1
                                                        maxQuantileWidthRange = (oaMaxQuantileWidthMin args, oaMaxQuantileWidthMax args)
                                                        pConfirmConformal = clamp (oaPConfirmConformal args) 0 1
                                                        pConfirmQuantiles = clamp (oaPConfirmQuantiles args) 0 1
                                                        pConfidenceSizing = clamp (oaPConfidenceSizing args) 0 1
                                                        protectionMinConfidenceMin = max 0 (oaProtectionMinConfidenceMin args)
                                                        protectionMinConfidenceMax = max protectionMinConfidenceMin (oaProtectionMinConfidenceMax args)
                                                        protectionMinConfidenceRange = (protectionMinConfidenceMin, protectionMinConfidenceMax)
                                                        minPositionSizeRange = (oaMinPositionSizeMin args, oaMinPositionSizeMax args)
                                                        maxPositionSizeMin = max 0 (oaMaxPositionSizeMin args)
                                                        maxPositionSizeMax = max maxPositionSizeMin (oaMaxPositionSizeMax args)
                                                        maxPositionSizeRange = (maxPositionSizeMin, maxPositionSizeMax)
                                                        volTargetMin = max 0 (oaVolTargetMin args)
                                                        volTargetMax = max volTargetMin (oaVolTargetMax args)
                                                        volTargetRange = (volTargetMin, volTargetMax)
                                                        pDisableVolTarget = clamp (oaPDisableVolTarget args) 0 1
                                                        volLookbackMin = max 0 (oaVolLookbackMin args)
                                                        volLookbackMax = max volLookbackMin (oaVolLookbackMax args)
                                                        volLookbackRange = (volLookbackMin, volLookbackMax)
                                                        volEwmaAlphaMin = max 0 (oaVolEwmaAlphaMin args)
                                                        volEwmaAlphaMax = max volEwmaAlphaMin (oaVolEwmaAlphaMax args)
                                                        volEwmaAlphaRange = (volEwmaAlphaMin, volEwmaAlphaMax)
                                                        pDisableVolEwmaAlpha = clamp (oaPDisableVolEwmaAlpha args) 0 1
                                                        volFloorMin = max 0 (oaVolFloorMin args)
                                                        volFloorMax = max volFloorMin (oaVolFloorMax args)
                                                        volFloorRange = (volFloorMin, volFloorMax)
                                                        volScaleMaxMin = max 0 (oaVolScaleMaxMin args)
                                                        volScaleMaxMax = max volScaleMaxMin (oaVolScaleMaxMax args)
                                                        volScaleMaxRange = (volScaleMaxMin, volScaleMaxMax)
                                                        maxVolatilityMin = max 0 (oaMaxVolatilityMin args)
                                                        maxVolatilityMax = max maxVolatilityMin (oaMaxVolatilityMax args)
                                                        maxVolatilityRange = (maxVolatilityMin, maxVolatilityMax)
                                                        pDisableMaxVolatility = clamp (oaPDisableMaxVolatility args) 0 1
                                                        periodsPerYearMin = max 0 (oaPeriodsPerYearMin args)
                                                        periodsPerYearMax = max periodsPerYearMin (oaPeriodsPerYearMax args)
                                                        periodsPerYearRange = (periodsPerYearMin, periodsPerYearMax)
                                                        methodWeights =
                                                            ( oaMethodWeight11 args
                                                            , oaMethodWeight10 args
                                                            , oaMethodWeight01 args
                                                            , oaMethodWeightBlend args
                                                            , oaMethodWeightConfBlend args
                                                            , oaMethodWeightConfPick args
                                                            , oaMethodWeightConformalClip args
                                                            , oaMethodWeightCostPick args
                                                            , oaMethodWeightHarmonicBlend args
                                                            , oaMethodWeightDisagreementGuard args
                                                            , oaMethodWeightMedianBlend args
                                                            , oaMethodWeightNeutralGuard args
                                                            , oaMethodWeightRiskParityBlend args
                                                            , oaMethodWeightConsensusBoost args
                                                            , oaMethodWeightAnchorBlend args
                                                            , oaMethodWeightTensionGate args
                                                            , oaMethodWeightEntropyBlend args
                                                            , oaMethodWeightCoherenceGate args
                                                            , oaMethodWeightDivergenceGate args
                                                            , oaMethodWeightFractalBlend args
                                                            , oaMethodWeightPhaseCancel args
                                                            , oaMethodWeightSoftmaxBlend args
                                                            , oaMethodWeightSmoothSoftmaxBlend args
                                                            , oaMethodWeightHedgeBlend args
                                                            , oaMethodWeightNetSoftmaxBlend args
                                                            , oaMethodWeightEdgeBlend args
                                                            , oaMethodWeightEdgePick args
                                                            , oaMethodWeightGeoBlend args
                                                            , oaMethodWeightRegimeSwitch args
                                                            , oaMethodWeightBanditRouter args
                                                            )
                                                        blendWeightRange =
                                                            let lo = clamp (oaBlendWeightMin args) 0 1
                                                                hi = clamp (oaBlendWeightMax args) 0 1
                                                             in (lo, hi)
                                                        routerScorePnlWeightRange =
                                                            let lo = clamp (oaRouterScorePnlWeightMin args) 0 1
                                                                hi = clamp (oaRouterScorePnlWeightMax args) 0 1
                                                             in (lo, hi)
                                                        normalizationChoices =
                                                            [trim s | s <- splitCsv (oaNormalizations args), not (null (trim s))]
                                                        predictorChoices =
                                                            [trim s | s <- splitCsv (oaPredictors args), not (null (trim s))]
                                                        routerLookbackRange =
                                                            (max 2 (oaRouterLookbackMin args), max 2 (oaRouterLookbackMax args))
                                                        routerRegimeMinBarsRange =
                                                            (max 0 (oaRouterRegimeMinBarsMin args), max 0 (oaRouterRegimeMinBarsMax args))
                                                        routerRegimeMinFractionRange =
                                                            (clamp (oaRouterRegimeMinFractionMin args) 0 1, clamp (oaRouterRegimeMinFractionMax args) 0 1)
                                                        routerMinScoreRange = (oaRouterMinScoreMin args, oaRouterMinScoreMax args)
                                                        feeFixedRange = (max 0 (oaFeeFixedMin args), max 0 (oaFeeFixedMax args))
                                                        slippageImpactRange = (max 0 (oaSlippageImpactMin args), max 0 (oaSlippageImpactMax args))
                                                        slippageImpactPowerRange = (max 0 (oaSlippageImpactPowerMin args), max 0 (oaSlippageImpactPowerMax args))
                                                        slippageVolMultRange = (max 0 (oaSlippageVolMultMin args), max 0 (oaSlippageVolMultMax args))
                                                        spreadVolMultRange = (max 0 (oaSpreadVolMultMin args), max 0 (oaSpreadVolMultMax args))
                                                        takeProfitPartialRange = (oaTakeProfitPartialMin args, oaTakeProfitPartialMax args)
                                                        pDisableTakeProfitPartial = clamp (oaPDisableTakeProfitPartial args) 0 1
                                                        maxTradesPerDayRange = (max 0 (oaMaxTradesPerDayMin args), max 0 (oaMaxTradesPerDayMax args))
                                                        pDisableMaxTradesPerDay = clamp (oaPDisableMaxTradesPerDay args) 0 1
                                                        expectancyLookbackRange = (max 1 (oaExpectancyLookbackMin args), max 1 (oaExpectancyLookbackMax args))
                                                        minExpectancyRange = (oaMinExpectancyMin args, oaMinExpectancyMax args)
                                                        pDisableMinExpectancy = clamp (oaPDisableMinExpectancy args) 0 1
                                                        lossStreakMaxRange = (max 0 (oaLossStreakMaxMin args), max 0 (oaLossStreakMaxMax args))
                                                        pDisableLossStreakMax = clamp (oaPDisableLossStreakMax args) 0 1
                                                        lossStreakCooldownBarsRange = (max 0 (oaLossStreakCooldownBarsMin args), max 0 (oaLossStreakCooldownBarsMax args))
                                                        pDisableLossStreakCooldownBars = clamp (oaPDisableLossStreakCooldownBars args) 0 1
                                                        maxOpenPositionsRange = (max 0 (oaMaxOpenPositionsMin args), max 0 (oaMaxOpenPositionsMax args))
                                                        pDisableMaxOpenPositions = clamp (oaPDisableMaxOpenPositions args) 0 1
                                                        maxGrossExposureRange = (max 0 (oaMaxGrossExposureMin args), max 0 (oaMaxGrossExposureMax args))
                                                        pDisableMaxGrossExposure = clamp (oaPDisableMaxGrossExposure args) 0 1
                                                        maxNetExposureRange = (max 0 (oaMaxNetExposureMin args), max 0 (oaMaxNetExposureMax args))
                                                        pDisableMaxNetExposure = clamp (oaPDisableMaxNetExposure args) 0 1
                                                        maxExposurePerBaseRange = (max 0 (oaMaxExposurePerBaseMin args), max 0 (oaMaxExposurePerBaseMax args))
                                                        pDisableMaxExposurePerBase = clamp (oaPDisableMaxExposurePerBase args) 0 1
                                                        maxOpenPerBaseRange = (max 0 (oaMaxOpenPerBaseMin args), max 0 (oaMaxOpenPerBaseMax args))
                                                        pDisableMaxOpenPerBase = clamp (oaPDisableMaxOpenPerBase args) 0 1
                                                        adaptiveEdgeBufferMaxRange = (max 0 (oaAdaptiveEdgeBufferMaxMin args), max 0 (oaAdaptiveEdgeBufferMaxMax args))
                                                        adaptiveMinSignalToNoiseMaxRange = (max 0 (oaAdaptiveMinSignalToNoiseMaxMin args), max 0 (oaAdaptiveMinSignalToNoiseMaxMax args))
                                                        adaptiveTrendLookbackMaxRange = (max 1 (oaAdaptiveTrendLookbackMaxMin args), max 1 (oaAdaptiveTrendLookbackMaxMax args))
                                                        adaptiveKalmanZMinMaxRange = (max 0 (oaAdaptiveKalmanZMinMaxMin args), max 0 (oaAdaptiveKalmanZMinMaxMax args))
                                                        adaptiveWinRateSlackRange = (max 0 (oaAdaptiveWinRateSlackMin args), max 0 (oaAdaptiveWinRateSlackMax args))
                                                        adaptiveProfitFactorSlackRange = (max 0 (oaAdaptiveProfitFactorSlackMin args), max 0 (oaAdaptiveProfitFactorSlackMax args))
                                                        pAdaptiveFilters = clamp (oaPAdaptiveFilters args) 0 1
                                                        perfLookbackRange = (max 1 (oaPerfLookbackMin args), max 1 (oaPerfLookbackMax args))
                                                        perfMinWinRateRange = (clamp (oaPerfMinWinRateMin args) 0 1, clamp (oaPerfMinWinRateMax args) 0 1)
                                                        pDisablePerfMinWinRate = clamp (oaPDisablePerfMinWinRate args) 0 1
                                                        perfMinProfitFactorRange = (max 0 (oaPerfMinProfitFactorMin args), max 0 (oaPerfMinProfitFactorMax args))
                                                        pDisablePerfMinProfitFactor = clamp (oaPDisablePerfMinProfitFactor args) 0 1
                                                        pMetaLabelFilter = clamp (oaPMetaLabelFilter args) 0 1
                                                        metaLabelMinEdgeRange = (max 0 (oaMetaLabelMinEdgeMin args), max 0 (oaMetaLabelMinEdgeMax args))
                                                        metaLabelMinConfidenceRange = (clamp (oaMetaLabelMinConfidenceMin args) 0 1, clamp (oaMetaLabelMinConfidenceMax args) 0 1)
                                                        pMetaLabelRequireBand = clamp (oaPMetaLabelRequireBand args) 0 1
                                                        pRegimeParameterBank = clamp (oaPRegimeParameterBank args) 0 1
                                                        regimeBankHysteresisRange = (clamp (oaRegimeBankHysteresisMin args) 0 1, clamp (oaRegimeBankHysteresisMax args) 0 1)
                                                        regimeTrendOpenMultRange = (max 0 (oaRegimeTrendOpenMultMin args), max 0 (oaRegimeTrendOpenMultMax args))
                                                        regimeMrOpenMultRange = (max 0 (oaRegimeMrOpenMultMin args), max 0 (oaRegimeMrOpenMultMax args))
                                                        regimeHighVolOpenMultRange = (max 0 (oaRegimeHighVolOpenMultMin args), max 0 (oaRegimeHighVolOpenMultMax args))
                                                        regimeTrendSizeMultRange = (max 0 (oaRegimeTrendSizeMultMin args), max 0 (oaRegimeTrendSizeMultMax args))
                                                        regimeMrSizeMultRange = (max 0 (oaRegimeMrSizeMultMin args), max 0 (oaRegimeMrSizeMultMax args))
                                                        regimeHighVolSizeMultRange = (max 0 (oaRegimeHighVolSizeMultMin args), max 0 (oaRegimeHighVolSizeMultMax args))
                                                        technicalRanges = oaTechnicalRanges args
                                                        pMultiTimeframeConsensus = clamp (oaPMultiTimeframeConsensus args) 0 1
                                                        mtfFastBarsRange = (max 1 (oaMtfFastBarsMin args), max 1 (oaMtfFastBarsMax args))
                                                        mtfMidBarsRange = (max 1 (oaMtfMidBarsMin args), max 1 (oaMtfMidBarsMax args))
                                                        mtfSlowBarsRange = (max 1 (oaMtfSlowBarsMin args), max 1 (oaMtfSlowBarsMax args))
                                                        mtfMinAgreeRange = (clampInt (oaMtfMinAgreeMin args) 1 3, clampInt (oaMtfMinAgreeMax args) 1 3)
                                                        pCrossAssetConfirmation = clamp (oaPCrossAssetConfirmation args) 0 1
                                                        crossAssetMinBetaRange = (max 0 (oaCrossAssetMinBetaMin args), max 0 (oaCrossAssetMinBetaMax args))
                                                        crossAssetMinEdgeRange = (max 0 (oaCrossAssetMinEdgeMin args), max 0 (oaCrossAssetMinEdgeMax args))
                                                        pPairsStatArb = clamp (oaPPairsStatArb args) 0 1
                                                        pairsStatArbLookbackRange = (max 2 (oaPairsStatArbLookbackMin args), max 2 (oaPairsStatArbLookbackMax args))
                                                        pairsStatArbZEntryRange = (max 1e-12 (oaPairsStatArbZEntryMin args), max 1e-12 (oaPairsStatArbZEntryMax args))
                                                        pairsStatArbSizeMultRange = (clamp (oaPairsStatArbSizeMultMin args) 0 1, clamp (oaPairsStatArbSizeMultMax args) 0 1)
                                                        pFundingOiAware = clamp (oaPFundingOiAware args) 0 1
                                                        fundingOiFundingCapRange = (max 0 (oaFundingOiFundingCapMin args), max 0 (oaFundingOiFundingCapMax args))
                                                        pDisableFundingOiFundingCap = clamp (oaPDisableFundingOiFundingCap args) 0 1
                                                        fundingOiVolLookbackRange = (max 2 (oaFundingOiVolLookbackMin args), max 2 (oaFundingOiVolLookbackMax args))
                                                        fundingOiVolCapRange = (max 0 (oaFundingOiVolCapMin args), max 0 (oaFundingOiVolCapMax args))
                                                        pDisableFundingOiVolCap = clamp (oaPDisableFundingOiVolCap args) 0 1
                                                        fundingOiSizeMultRange = (clamp (oaFundingOiSizeMultMin args) 0 1, clamp (oaFundingOiSizeMultMax args) 0 1)
                                                        pKellyLiteSizing = clamp (oaPKellyLiteSizing args) 0 1
                                                        kellyLiteFractionRange = (max 0 (oaKellyLiteFractionMin args), max 0 (oaKellyLiteFractionMax args))
                                                        kellyLiteFloorRange = (max 0 (oaKellyLiteFloorMin args), max 0 (oaKellyLiteFloorMax args))
                                                        kellyLiteCapRange = (max 0 (oaKellyLiteCapMin args), max 0 (oaKellyLiteCapMax args))
                                                    if null normalizationChoices
                                                        then do
                                                            hPutStrLn stderr "No normalizations provided."
                                                            pure 2
                                                        else do
                                                            baseArgsResult <- buildBaseArgs args csvCols
                                                            case baseArgsResult of
                                                                Left err -> do
                                                                    hPutStrLn stderr err
                                                                    pure 2
                                                                Right baseArgs -> do
                                                                    priorTrialsRaw <-
                                                                        if oaPriorSampleProb args <= 0
                                                                            then pure []
                                                                            else readOptimizerPriorTrials (oaPriorJson args)
                                                                    priorNowMs <- fmap (floor . (* 1000) :: POSIXTime -> Int) getPOSIXTime
                                                                    let rngStart = seedRng (oaSeed args)
                                                                        dataSource = if isNothing (oaData args) then "binance" else "csv"
                                                                        sourceOverride = map toLower (trim (oaSourceLabel args))
                                                                        symbolLabel = normalizeSymbol (Just (oaSymbolLabel args))
                                                                        symbolFallback = normalizeSymbol (oaBinanceSymbol args)
                                                                        symbolFromData =
                                                                            case oaData args of
                                                                                Just path -> normalizeSymbol (Just (takeBaseName path))
                                                                                Nothing -> Nothing
                                                                        symbolFinal =
                                                                            case symbolLabel of
                                                                                Just _ -> symbolLabel
                                                                                Nothing ->
                                                                                    case symbolFallback of
                                                                                        Just _ -> symbolFallback
                                                                                        Nothing -> symbolFromData
                                                                        trials = max 1 (oaTrials args)
                                                                        seedTrialsOverride = oaSeedTrials args
                                                                        seedRatioOverride = oaSeedRatio args
                                                                        survivorFraction = clamp (oaSurvivorFraction args) 0 1
                                                                        survivorParentActivityFloor = max 0 (oaSurvivorParentActivityFloor args)
                                                                        survivorParentAnnualizedReturnFloor =
                                                                            let raw = oaSurvivorParentAnnualizedReturnFloor args
                                                                             in if isNaN raw || isInfinite raw then 1 else raw
                                                                        perturbScaleDouble = max 0 (oaPerturbScaleDouble args)
                                                                        perturbScaleInt = max 0 (oaPerturbScaleInt args)
                                                                        earlyStopNoImprove = max 0 (oaEarlyStopNoImprove args)
                                                                        priorSampleProb = clamp (oaPriorSampleProb args) 0 1
                                                                        priorMethodSampleProb = clamp (oaPriorMethodSampleProb args) 0 1
                                                                        priorRankBias = max 1 (oaPriorRankBias args)
                                                                        priorPerturbScaleDouble = max 0 (oaPriorPerturbScaleDouble args)
                                                                        priorPerturbScaleInt = max 0 (oaPriorPerturbScaleInt args)
                                                                        priorAgeHalfLifeDays = max 0 (oaPriorAgeHalfLifeDays args)
                                                                        priorDiversityMaxPerBucket = max 0 (oaPriorDiversityMaxPerBucket args)
                                                                        priorSeedCount = max 0 (oaPriorSeedCount args)
                                                                        minRoundTrips = max 0 (oaMinRoundTrips args)
                                                                        minWinRate = max 0 (oaMinWinRate args)
                                                                        minProfitFactor = max 0 (oaMinProfitFactor args)
                                                                        minExposure = max 0 (oaMinExposure args)
                                                                        minKellyLiteExposureReduction = max 0 (oaMinKellyLiteExposureReduction args)
                                                                        maxKellyLiteExposureRatio = max 0 (oaMaxKellyLiteExposureRatio args)
                                                                        minSharpe = max 0 (oaMinSharpe args)
                                                                        minAnnualizedReturn = max 0 (oaMinAnnualizedReturn args)
                                                                        minCalmar = max 0 (oaMinCalmar args)
                                                                        maxTurnover = max 0 (oaMaxTurnover args)
                                                                        minWfSharpeMean = max 0 (oaMinWfSharpeMean args)
                                                                        maxWfSharpeStd = max 0 (oaMaxWfSharpeStd args)
                                                                        priorTrials = selectOptimizerPriorTrials args{oaPriorAgeHalfLifeDays = priorAgeHalfLifeDays, oaPriorDiversityMaxPerBucket = priorDiversityMaxPerBucket} priorNowMs symbolFinal intervals priorTrialsRaw
                                                                    when (priorSampleProb > 0 && not (null (trim (oaPriorJson args)))) $
                                                                        hPutStrLn
                                                                            stderr
                                                                            ( "Empirical optimizer priors: using "
                                                                                ++ show (length priorTrials)
                                                                                ++ "/"
                                                                                ++ show (length priorTrialsRaw)
                                                                                ++ " rows from "
                                                                                ++ oaPriorJson args
                                                                                ++ priorTrialDistributionSummary priorTrials
                                                                            )
                                                                    outHandle <-
                                                                        if null (trim (oaOutput args))
                                                                            then pure Nothing
                                                                            else do
                                                                                p <- expandUser (oaOutput args)
                                                                                createDirectoryIfMissing True (takeDirectory p)
                                                                                h <- openFile p (if oaAppend args then AppendMode else WriteMode)
                                                                                hSetEncoding h utf8
                                                                                pure (Just h)
                                                                    let techniqueSummaryBase =
                                                                            emptyTechniqueSummary
                                                                                { otsAppliedWalkForward = walkForwardFoldsMax > 1 || walkForwardFoldsMin > 1
                                                                                , otsAppliedEmpiricalPriors = priorSampleProb > 0 && not (null priorTrials)
                                                                                }
                                                                        seedTrialsDefault = max 1 (min trials (max 3 (trials `div` 2)))
                                                                        seedTrials =
                                                                            case seedTrialsOverride of
                                                                                Just n -> max 0 (min trials n)
                                                                                Nothing ->
                                                                                    case seedRatioOverride of
                                                                                        Just r ->
                                                                                            let ratio = clamp r 0 1
                                                                                                count = floor (fromIntegral trials * ratio)
                                                                                             in max 0 (min trials count)
                                                                                        Nothing -> seedTrialsDefault
                                                                        priorSeedTrials = take (min seedTrials priorSeedCount) priorTrials
                                                                        survivorCount =
                                                                            if seedTrials <= 0
                                                                                then 0
                                                                                else max 1 (floor (fromIntegral seedTrials * survivorFraction))
                                                                        sobolRngs = map seedRng (sobolSeeds (oaSeed args) seedTrials)
                                                                        remainingTrials = max 0 (trials - seedTrials)
                                                                        exploitationRngs = map (\i -> seedRng (oaSeed args + 10000 + i)) [1 .. remainingTrials]
                                                                        sampleParamsWithRng rng =
                                                                            sampleParams
                                                                                rng
                                                                                platforms
                                                                                platformIntervalsMap
                                                                                intervals
                                                                                barsAutoProb
                                                                                barsMin
                                                                                barsMax
                                                                                barsDistribution
                                                                                openThresholdMin
                                                                                openThresholdMax
                                                                                closeThresholdMin
                                                                                closeThresholdMax
                                                                                (minHoldMin, minHoldMax)
                                                                                (cooldownMin, cooldownMax)
                                                                                (maxHoldMin, maxHoldMax)
                                                                                (minEdgeMin, minEdgeMax)
                                                                                (minSnMin, minSnMax)
                                                                                snrSizeWeightRange
                                                                                pThresholdFactor
                                                                                thresholdFactorAlphaRange
                                                                                thresholdFactorMinRange
                                                                                thresholdFactorMaxRange
                                                                                thresholdFactorFloorRange
                                                                                thresholdFactorWeightRange
                                                                                (edgeBufferMin, edgeBufferMax)
                                                                                (trendLookbackMin, trendLookbackMax)
                                                                                maxPositionSizeRange
                                                                                volTargetRange
                                                                                volLookbackRange
                                                                                volEwmaAlphaRange
                                                                                pDisableVolEwmaAlpha
                                                                                volFloorRange
                                                                                volScaleMaxRange
                                                                                maxVolatilityRange
                                                                                periodsPerYearRange
                                                                                (kalmanMarketTopNMin, kalmanMarketTopNMax)
                                                                                (oaPCostAwareEdge args)
                                                                                feeMin
                                                                                feeMax
                                                                                fundingRateRange
                                                                                pFundingBySide
                                                                                pFundingOnOpen
                                                                                rebalanceBarsRange
                                                                                rebalanceThresholdRange
                                                                                rebalanceCostMultRange
                                                                                pRebalanceGlobal
                                                                                pRebalanceResetOnSignal
                                                                                pLongShort
                                                                                pIntrabarTakeProfitFirst
                                                                                pTriLayer
                                                                                triLayerFastRange
                                                                                triLayerSlowRange
                                                                                pTriLayerPriceAction
                                                                                triLayerCloudPaddingRange
                                                                                triLayerCloudSlopeRange
                                                                                triLayerCloudWidthRange
                                                                                triLayerTouchLookbackRange
                                                                                triLayerPriceActionBodyRange
                                                                                triLayerPriceActionWickRatioRange
                                                                                triLayerPriceActionOppositeWickMaxRange
                                                                                triLayerPriceActionBodyToleranceRange
                                                                                (oaTriLayerExitOnSlow args)
                                                                                kalmanBandLookbackRange
                                                                                kalmanBandStdMultRange
                                                                                lstmExitFlipBarsRange
                                                                                lstmExitFlipGraceBarsRange
                                                                                (oaLstmExitFlipStrong args)
                                                                                lstmConfidenceSoftRange
                                                                                lstmConfidenceHardRange
                                                                                epochsMin
                                                                                epochsMax
                                                                                hiddenMin
                                                                                hiddenMax
                                                                                lrMin
                                                                                lrMax
                                                                                lstmAdamBeta1Range
                                                                                lstmAdamBeta2Range
                                                                                lstmAdamEpsRange'
                                                                                valMin
                                                                                valMax
                                                                                patienceMax
                                                                                (walkForwardFoldsMin, walkForwardFoldsMax)
                                                                                walkForwardEmbargoBarsRange
                                                                                tuneStressVolMultRange
                                                                                tuneStressShockRange
                                                                                tuneStressWeightRange
                                                                                gradClipMin
                                                                                gradClipMax
                                                                                (oaSlippageMax args)
                                                                                (oaSpreadMax args)
                                                                                kalmanDtMin
                                                                                kalmanDtMax
                                                                                kalmanProcessVarMin
                                                                                kalmanProcessVarMax
                                                                                kalmanMeasurementVarMin
                                                                                kalmanMeasurementVarMax
                                                                                kalmanPhysicsBarsMin
                                                                                kalmanPhysicsBarsMax
                                                                                kalmanPhysicsBacktestRatioMin
                                                                                kalmanPhysicsBacktestRatioMax
                                                                                kalmanPhysicsVolumeEwmaAlphaMin
                                                                                kalmanPhysicsVolumeEwmaAlphaMax
                                                                                kalmanPhysicsVolumeSignalClampMin
                                                                                kalmanPhysicsVolumeSignalClampMax
                                                                                kalmanPhysicsCloseBiasScaleMin
                                                                                kalmanPhysicsCloseBiasScaleMax
                                                                                (clamp (oaSensorVarianceEwmaAlpha args) 0 1)
                                                                                (clamp (oaKalmanSensorCorrelationInflation args) 0 1)
                                                                                (max 0 (oaKalmanInnovationInflationThreshold args))
                                                                                (max 1 (oaKalmanInnovationInflationMax args))
                                                                                kalmanZMinMin
                                                                                kalmanZMinMax
                                                                                kalmanZMaxMin
                                                                                kalmanZMaxMax
                                                                                pDisableMaxHighVolProb
                                                                                maxHighVolProbRange
                                                                                pDisableMaxConformalWidth
                                                                                maxConformalWidthRange
                                                                                pDisableMaxQuantileWidth
                                                                                maxQuantileWidthRange
                                                                                pConfirmConformal
                                                                                pConfirmQuantiles
                                                                                pConfidenceSizing
                                                                                protectionMinConfidenceRange
                                                                                minPositionSizeRange
                                                                                stopRange
                                                                                takeRange
                                                                                trailRange
                                                                                stopVolMultRange
                                                                                takeVolMultRange
                                                                                trailVolMultRange
                                                                                methodWeights
                                                                                normalizationChoices
                                                                                blendWeightRange
                                                                                routerScorePnlWeightRange
                                                                                (clamp (oaPDisableStop args) 0 1)
                                                                                (clamp (oaPDisableTp args) 0 1)
                                                                                (clamp (oaPDisableTrail args) 0 1)
                                                                                (clamp (oaPDisableStopVolMult args) 0 1)
                                                                                (clamp (oaPDisableTpVolMult args) 0 1)
                                                                                (clamp (oaPDisableTrailVolMult args) 0 1)
                                                                                riskPerTradeRange
                                                                                (clamp (oaPDisableRiskPerTrade args) 0 1)
                                                                                (clamp (oaPDisableMaxDd args) 0 1)
                                                                                (clamp (oaPDisableMaxDl args) 0 1)
                                                                                (clamp (oaPDisableMaxWl args) 0 1)
                                                                                (clamp (oaPDisableMaxOe args) 0 1)
                                                                                (clamp (oaPDisableGradClip args) 0 1)
                                                                                pDisableVolTarget
                                                                                pDisableMaxVolatility
                                                                                (oaMaxDdMin args, oaMaxDdMax args)
                                                                                (oaMaxDlMin args, oaMaxDlMax args)
                                                                                (oaMaxWlMin args, oaMaxWlMax args)
                                                                                (oaMaxOeMin args, oaMaxOeMax args)
                                                                                predictorChoices
                                                                                routerLookbackRange
                                                                                routerRegimeMinBarsRange
                                                                                routerRegimeMinFractionRange
                                                                                routerMinScoreRange
                                                                                feeFixedRange
                                                                                slippageImpactRange
                                                                                slippageImpactPowerRange
                                                                                slippageVolMultRange
                                                                                spreadVolMultRange
                                                                                takeProfitPartialRange
                                                                                pDisableTakeProfitPartial
                                                                                maxTradesPerDayRange
                                                                                pDisableMaxTradesPerDay
                                                                                expectancyLookbackRange
                                                                                minExpectancyRange
                                                                                pDisableMinExpectancy
                                                                                lossStreakMaxRange
                                                                                pDisableLossStreakMax
                                                                                lossStreakCooldownBarsRange
                                                                                pDisableLossStreakCooldownBars
                                                                                maxOpenPositionsRange
                                                                                pDisableMaxOpenPositions
                                                                                maxGrossExposureRange
                                                                                pDisableMaxGrossExposure
                                                                                maxNetExposureRange
                                                                                pDisableMaxNetExposure
                                                                                maxExposurePerBaseRange
                                                                                pDisableMaxExposurePerBase
                                                                                maxOpenPerBaseRange
                                                                                pDisableMaxOpenPerBase
                                                                                adaptiveEdgeBufferMaxRange
                                                                                adaptiveMinSignalToNoiseMaxRange
                                                                                adaptiveTrendLookbackMaxRange
                                                                                adaptiveKalmanZMinMaxRange
                                                                                adaptiveWinRateSlackRange
                                                                                adaptiveProfitFactorSlackRange
                                                                                pAdaptiveFilters
                                                                                perfLookbackRange
                                                                                perfMinWinRateRange
                                                                                pDisablePerfMinWinRate
                                                                                perfMinProfitFactorRange
                                                                                pDisablePerfMinProfitFactor
                                                                                pMetaLabelFilter
                                                                                metaLabelMinEdgeRange
                                                                                metaLabelMinConfidenceRange
                                                                                pMetaLabelRequireBand
                                                                                pRegimeParameterBank
                                                                                regimeBankHysteresisRange
                                                                                regimeTrendOpenMultRange
                                                                                regimeMrOpenMultRange
                                                                                regimeHighVolOpenMultRange
                                                                                regimeTrendSizeMultRange
                                                                                regimeMrSizeMultRange
                                                                                regimeHighVolSizeMultRange
                                                                                technicalRanges
                                                                                pMultiTimeframeConsensus
                                                                                mtfFastBarsRange
                                                                                mtfMidBarsRange
                                                                                mtfSlowBarsRange
                                                                                mtfMinAgreeRange
                                                                                pCrossAssetConfirmation
                                                                                crossAssetMinBetaRange
                                                                                crossAssetMinEdgeRange
                                                                                pPairsStatArb
                                                                                pairsStatArbLookbackRange
                                                                                pairsStatArbZEntryRange
                                                                                pairsStatArbSizeMultRange
                                                                                pFundingOiAware
                                                                                fundingOiFundingCapRange
                                                                                pDisableFundingOiFundingCap
                                                                                fundingOiVolLookbackRange
                                                                                fundingOiVolCapRange
                                                                                pDisableFundingOiVolCap
                                                                                fundingOiSizeMultRange
                                                                                pKellyLiteSizing
                                                                                kellyLiteFractionRange
                                                                                kellyLiteFloorRange
                                                                                kellyLiteCapRange
                                                                        sampleParamsMaybePrior rng =
                                                                            let (baseParams, rng') = sampleParamsWithRng rng
                                                                             in samplePriorParams
                                                                                    priorSampleProb
                                                                                    intervals
                                                                                    barsMin
                                                                                    barsMax
                                                                                    priorPerturbScaleDouble
                                                                                    priorPerturbScaleInt
                                                                                    priorMethodSampleProb
                                                                                    priorRankBias
                                                                                    priorTrials
                                                                                    baseParams
                                                                                    rng'
                                                                        samplePriorSeedParams prior rng =
                                                                            let (baseParams, _) = sampleParamsWithRng rng
                                                                                overlaid =
                                                                                    applyPriorMethodIfEnabled True prior $
                                                                                        applyPriorOverlay intervals prior baseParams
                                                                             in normalizeTrialParams
                                                                                    overlaid
                                                                                        { tpBars =
                                                                                            clampBarsForPlatform
                                                                                                (tpPlatform overlaid)
                                                                                                barsMin
                                                                                                barsMax
                                                                                                (tpBars overlaid)
                                                                                        }
                                                                        runTrialWith idx rng mPriorSeed mBase mParents best recordsRev = do
                                                                            let (params, origin) =
                                                                                    case mPriorSeed of
                                                                                        Just prior -> (samplePriorSeedParams prior rng, "prior-seed")
                                                                                        Nothing ->
                                                                                            case mBase of
                                                                                                Nothing ->
                                                                                                    let (params0, _, mOrigin) = sampleParamsMaybePrior rng
                                                                                                     in (params0, fromMaybe "sobol-seed" mOrigin)
                                                                                                Just base ->
                                                                                                    case mParents of
                                                                                                        Just parents@(_ : _ : _) ->
                                                                                                            case pickParentPair parents rng of
                                                                                                                Just ((p1, p2), rng1) ->
                                                                                                                    let crossover = crossoverTrialParams p1 p2 rng1
                                                                                                                        child = fst crossover
                                                                                                                        rng2 = snd crossover
                                                                                                                        params0 = fst (perturbTrialParams barsMin barsMax perturbScaleDouble perturbScaleInt child rng2)
                                                                                                                     in (params0, "survivor-crossover")
                                                                                                                Nothing ->
                                                                                                                    let (params0, _) = perturbTrialParams barsMin barsMax perturbScaleDouble perturbScaleInt base rng
                                                                                                                     in (params0, "survivor-perturb")
                                                                                                        _ ->
                                                                                                            let (params0, _) = perturbTrialParams barsMin barsMax perturbScaleDouble perturbScaleInt base rng
                                                                                                             in (params0, "survivor-perturb")
                                                                            tr0 <-
                                                                                runTrial
                                                                                    traderBinPath
                                                                                    baseArgs
                                                                                    params
                                                                                    (oaTuneRatio args)
                                                                                    useSweepThreshold
                                                                                    (oaTimeoutSec args)
                                                                                    (oaDisableLstmPersistence args)
                                                                            let objective = oaObjective args
                                                                                (eligible, filterReason, score) =
                                                                                    case (trOk tr0, trFinalEquity tr0, trMetrics tr0) of
                                                                                        (True, Just _, Just metrics) ->
                                                                                            let mOpenThreshold = trOpenThreshold tr0 <|> Just (tpBaseOpenThreshold params)
                                                                                                openThresholdDeployable =
                                                                                                    maybe True signalEntryOpenThresholdFeasible mOpenThreshold
                                                                                                roundTrips = metricInt (trMetrics tr0) "roundTrips" 0
                                                                                                tradeCount = metricInt (trMetrics tr0) "tradeCount" 0
                                                                                                activityCount = max roundTrips tradeCount
                                                                                                -- Cost-floor gate: a trial whose minEdge sits below the
                                                                                                -- venue round-trip cost (with 1.5x safety) cannot have
                                                                                                -- positive live expectancy regardless of how good the
                                                                                                -- backtest looks. Reject before scoring so the
                                                                                                -- leaderboard never carries such combos.
                                                                                                minEdgeOk = tpMinEdge params >= venueMinEdgeFloor
                                                                                             in if not openThresholdDeployable
                                                                                                    then (False, Just (printf "openThreshold>%.6f" signalEntryOpenThresholdFeasibilityCap), Nothing)
                                                                                                    else
                                                                                                        if not minEdgeOk
                                                                                                            then (False, Just (printf "minEdge<%.6f(costFloor)" venueMinEdgeFloor), Nothing)
                                                                                                            else
                                                                                                                if minRoundTrips > 0 && activityCount < minRoundTrips
                                                                                                                    then (False, Just ("activityCount<" ++ show minRoundTrips), Nothing)
                                                                                                                    else
                                                                                                                        let winRate = metricFloat (trMetrics tr0) "winRate" 0
                                                                                                                         in if minWinRate > 0 && winRate < minWinRate
                                                                                                                                then (False, Just (printf "winRate<%.3f" minWinRate), Nothing)
                                                                                                                                else
                                                                                                                                    let profitFactor = metricProfitFactor (trMetrics tr0)
                                                                                                                                        pfOk =
                                                                                                                                            case profitFactor of
                                                                                                                                                Nothing -> True
                                                                                                                                                Just pf -> pf >= minProfitFactor
                                                                                                                                     in if minProfitFactor > 0 && not pfOk
                                                                                                                                            then (False, Just (printf "profitFactor<%.3f" minProfitFactor), Nothing)
                                                                                                                                            else
                                                                                                                                                let exposure = metricFloat (trMetrics tr0) "exposure" 0
                                                                                                                                                 in if minExposure > 0 && exposure < minExposure
                                                                                                                                                        then (False, Just (printf "exposure<%.3f" minExposure), Nothing)
                                                                                                                                                        else case kellyLiteExposureFilterReason
                                                                                                                                                            params
                                                                                                                                                            (trStdoutJson tr0)
                                                                                                                                                            minKellyLiteExposureReduction
                                                                                                                                                            maxKellyLiteExposureRatio of
                                                                                                                                                            Just reason -> (False, Just reason, Nothing)
                                                                                                                                                            Nothing ->
                                                                                                                                                                let sharpe = metricFloat (trMetrics tr0) "sharpe" 0
                                                                                                                                                                 in if minSharpe > 0 && sharpe < minSharpe
                                                                                                                                                                        then (False, Just (printf "sharpe<%.3f" minSharpe), Nothing)
                                                                                                                                                                        else
                                                                                                                                                                            let annRet = metricFloat (trMetrics tr0) "annualizedReturn" 0
                                                                                                                                                                                maxDd = metricFloat (trMetrics tr0) "maxDrawdown" 0
                                                                                                                                                                                maxDdN = max 0 maxDd
                                                                                                                                                                                calmar =
                                                                                                                                                                                    if maxDdN <= 0
                                                                                                                                                                                        then annRet
                                                                                                                                                                                        else annRet / max 1e-12 maxDdN
                                                                                                                                                                             in if minAnnualizedReturn > 0 && annRet < minAnnualizedReturn
                                                                                                                                                                                    then (False, Just (printf "annualizedReturn<%.3f" minAnnualizedReturn), Nothing)
                                                                                                                                                                                    else
                                                                                                                                                                                        if minCalmar > 0 && calmar < minCalmar
                                                                                                                                                                                            then (False, Just (printf "calmar<%.3f" minCalmar), Nothing)
                                                                                                                                                                                            else
                                                                                                                                                                                                let turnover = metricFloat (trMetrics tr0) "turnover" 0
                                                                                                                                                                                                 in if maxTurnover > 0 && turnover > maxTurnover
                                                                                                                                                                                                        then (False, Just (printf "turnover>%.3f" maxTurnover), Nothing)
                                                                                                                                                                                                        else
                                                                                                                                                                                                            if minWfSharpeMean > 0 || maxWfSharpeStd > 0
                                                                                                                                                                                                                then case extractWalkForwardSummary (trStdoutJson tr0) of
                                                                                                                                                                                                                    Nothing -> (False, Just "walkForwardMissing", Nothing)
                                                                                                                                                                                                                    Just wfSummary ->
                                                                                                                                                                                                                        let wfSharpeMean = metricFloat (Just wfSummary) "sharpeMean" 0
                                                                                                                                                                                                                            wfSharpeStd = metricFloat (Just wfSummary) "sharpeStd" 0
                                                                                                                                                                                                                         in if minWfSharpeMean > 0 && wfSharpeMean < minWfSharpeMean
                                                                                                                                                                                                                                then (False, Just (printf "wfSharpeMean<%.3f" minWfSharpeMean), Nothing)
                                                                                                                                                                                                                                else
                                                                                                                                                                                                                                    if maxWfSharpeStd > 0 && wfSharpeStd > maxWfSharpeStd
                                                                                                                                                                                                                                        then (False, Just (printf "wfSharpeStd>%.3f" maxWfSharpeStd), Nothing)
                                                                                                                                                                                                                                        else
                                                                                                                                                                                                                                            ( True
                                                                                                                                                                                                                                            , Nothing
                                                                                                                                                                                                                                            , Just
                                                                                                                                                                                                                                                ( objectiveScoreWithConfig
                                                                                                                                                                                                                                                    roiScoreConfig
                                                                                                                                                                                                                                                    metrics
                                                                                                                                                                                                                                                    objective
                                                                                                                                                                                                                                                    (oaPenaltyMaxDrawdown args)
                                                                                                                                                                                                                                                    (oaPenaltyTurnover args)
                                                                                                                                                                                                                                                )
                                                                                                                                                                                                                                            )
                                                                                                                                                                                                                else
                                                                                                                                                                                                                    ( True
                                                                                                                                                                                                                    , Nothing
                                                                                                                                                                                                                    , Just
                                                                                                                                                                                                                        ( objectiveScoreWithConfig
                                                                                                                                                                                                                            roiScoreConfig
                                                                                                                                                                                                                            metrics
                                                                                                                                                                                                                            objective
                                                                                                                                                                                                                            (oaPenaltyMaxDrawdown args)
                                                                                                                                                                                                                            (oaPenaltyTurnover args)
                                                                                                                                                                                                                        )
                                                                                                                                                                                                                    )
                                                                                        _ -> (False, Nothing, Nothing)
                                                                                tr =
                                                                                    tr0
                                                                                        { trEligible = eligible
                                                                                        , trFilterReason = filterReason
                                                                                        , trObjective = objective
                                                                                        , trScore = score
                                                                                        , trOrigin = origin
                                                                                        }
                                                                            case outHandle of
                                                                                Nothing -> pure ()
                                                                                Just h -> do
                                                                                    createdAtMs <- fmap (floor . (* 1000) :: POSIXTime -> Int) getPOSIXTime
                                                                                    let rec0 = trialToRecord tr symbolFinal
                                                                                        source = resolveSourceLabel (tpPlatform params) dataSource sourceOverride
                                                                                        rec =
                                                                                            addField "createdAtMs" (toJSON createdAtMs) $
                                                                                                addField "source" (String (T.pack source)) rec0
                                                                                    BL.hPutStr h (Aeson.encode rec)
                                                                                    hPutStrLn h ""
                                                                                    hFlush h
                                                                            let best' =
                                                                                    case (trEligible tr, trScore tr, best) of
                                                                                        (True, Just sc, Nothing) -> Just tr
                                                                                        (True, Just sc, Just b) ->
                                                                                            let bScore = fromMaybe (-1e18) (trScore b)
                                                                                             in if sc > bScore then Just tr else Just b
                                                                                        _ -> best
                                                                            printTrialStatus idx trials tr
                                                                            pure (best', tr : recordsRev, tr)
                                                                    (bestSeed, seedRecordsRev, seedResultsRev) <-
                                                                        foldM
                                                                            ( \(b, recs, res) (idx, rng) -> do
                                                                                let mPriorSeed = listToMaybe (drop (idx - 1) priorSeedTrials)
                                                                                (b', recs', tr) <- runTrialWith idx rng mPriorSeed Nothing Nothing b recs
                                                                                pure (b', recs', tr : res)
                                                                            )
                                                                            (Nothing, [], [])
                                                                            (zip [1 .. seedTrials] sobolRngs)
                                                                    let seedResults = reverse seedResultsRev
                                                                        scored =
                                                                            sortOn
                                                                                (Data.Ord.Down . fromMaybe (-1e18) . trScore)
                                                                                (filter (isJust . trScore) seedResults)
                                                                        survivorsRaw = take survivorCount (filter trEligible scored ++ scored)
                                                                        techniqueSummarySeed =
                                                                            techniqueSummaryBase
                                                                                { otsAppliedSobolSeeding = seedTrials > 0
                                                                                , otsAppliedSuccessiveHalving = length survivorsRaw < length seedResults
                                                                                }
                                                                        eliteCapacity = max 1 survivorCount
                                                                        eliteScore tr = fromMaybe (-1e18) (trScore tr)
                                                                        eligibleElite tr = trEligible tr && isJust (trScore tr)
                                                                        trimElitePool =
                                                                            take eliteCapacity
                                                                                . sortOn (Data.Ord.Down . eliteScore)
                                                                                . filter eligibleElite
                                                                        initialElitePool = trimElitePool survivorsRaw
                                                                        fallbackSurvivorParams = map trParams survivorsRaw
                                                                        hasExploitationBase = not (null initialElitePool) || not (null fallbackSurvivorParams)
                                                                        eliteBaseParams elitePool =
                                                                            case map trParams elitePool of
                                                                                [] -> fallbackSurvivorParams
                                                                                params -> params
                                                                        updateElitePool tr elitePool = trimElitePool (tr : elitePool)
                                                                        gaParentPoolFrom elitePool =
                                                                            let gaParents =
                                                                                    [ trParams tr
                                                                                    | tr <- elitePool
                                                                                    , let opCount =
                                                                                            max
                                                                                                (metricInt (trMetrics tr) "operationCount" 0)
                                                                                                (metricInt (trMetrics tr) "tradeCount" 0)
                                                                                    , opCount > survivorParentActivityFloor
                                                                                    , metricFloat (trMetrics tr) "annualizedReturn" 0 > survivorParentAnnualizedReturnFloor
                                                                                    ]
                                                                             in case gaParents of
                                                                                    _ : _ : _ -> Just gaParents
                                                                                    _ -> Nothing
                                                                        bestScore b = fromMaybe (-1e18) (trScore =<< b)
                                                                        runTrialsWithEarlyStop best recs survivorsIx stagnant elitePool trialsList =
                                                                            case trialsList of
                                                                                [] -> pure (best, recs, survivorsIx, elitePool)
                                                                                (idx, rng) : rest -> do
                                                                                    let survivorParams = eliteBaseParams elitePool
                                                                                        gaParentPool = gaParentPoolFrom elitePool
                                                                                        baseParam =
                                                                                            case survivorParams of
                                                                                                [] -> Nothing
                                                                                                _ ->
                                                                                                    let ix = survivorsIx `mod` length survivorParams
                                                                                                     in listToMaybe (drop ix survivorParams)
                                                                                        prevScore = bestScore best
                                                                                    (best', recs', tr) <- runTrialWith idx rng Nothing baseParam gaParentPool best recs
                                                                                    let survivorsIx' = survivorsIx + 1
                                                                                        bestScore' = bestScore best'
                                                                                        stagnant' = if bestScore' > prevScore then 0 else stagnant + 1
                                                                                        elitePool' = updateElitePool tr elitePool
                                                                                    if earlyStopNoImprove > 0 && stagnant' >= earlyStopNoImprove
                                                                                        then pure (best', recs', survivorsIx', elitePool')
                                                                                        else runTrialsWithEarlyStop best' recs' survivorsIx' stagnant' elitePool' rest
                                                                    (bestFinal, allRecordsRev, _, _) <-
                                                                        runTrialsWithEarlyStop
                                                                            bestSeed
                                                                            seedRecordsRev
                                                                            0
                                                                            0
                                                                            initialElitePool
                                                                            (zip [seedTrials + 1 .. seedTrials + remainingTrials] exploitationRngs)
                                                                    let records = reverse allRecordsRev
                                                                        best = bestFinal
                                                                        techniqueSummaryFinal =
                                                                            techniqueSummarySeed
                                                                                { otsAppliedBayesianEi = remainingTrials > 0 && hasExploitationBase
                                                                                , otsAppliedEnsemble =
                                                                                    case records of
                                                                                        _ : _ : _ -> True
                                                                                        _ -> False
                                                                                }
                                                                    Data.Foldable.for_ outHandle hClose
                                                                    case best of
                                                                        Nothing -> do
                                                                            printNoEligible
                                                                                minRoundTrips
                                                                                minWinRate
                                                                                minProfitFactor
                                                                                minExposure
                                                                                minKellyLiteExposureReduction
                                                                                maxKellyLiteExposureRatio
                                                                                minSharpe
                                                                                minAnnualizedReturn
                                                                                minCalmar
                                                                                maxTurnover
                                                                                minWfSharpeMean
                                                                                maxWfSharpeStd
                                                                            pure 1
                                                                        Just b -> do
                                                                            printBest b
                                                                            printRepro traderBinPath baseArgs b (oaTuneRatio args) useSweepThreshold (oaDisableLstmPersistence args)
                                                                            unless (null (trim (oaTopJson args))) $ do
                                                                                writeTopJson
                                                                                    (oaTopJson args)
                                                                                    dataSource
                                                                                    sourceOverride
                                                                                    symbolFinal
                                                                                    records
                                                                                    techniqueSummaryFinal
                                                                            pure 0
  where
    resolveStressRanges baseVol baseShock baseWeight volRange shockRange weightRange =
        let (vMin0, vMax0) = fillRange baseVol volRange
            vMin = max 1e-12 vMin0
            vMax = max 1e-12 vMax0
            (sMin, sMax) = fillRange baseShock shockRange
            (wMin0, wMax0) = fillRange baseWeight weightRange
            wMin = max 0 wMin0
            wMax = max 0 wMax0
         in ((min vMin vMax, max vMin vMax), (min sMin sMax, max sMin sMax), (min wMin wMax, max wMin wMax))
    fillRange base (mn, mx) =
        case (mn, mx) of
            (Nothing, Nothing) -> (base, base)
            (Just a, Nothing) -> (a, a)
            (Nothing, Just b) -> (b, b)
            (Just a, Just b) -> (a, b)
    unique = Set.toList . Set.fromList

resolveTraderBin :: OptimizerArgs -> IO (Either String FilePath)
resolveTraderBin args =
    if not (null (trim (oaBinary args)))
        then do
            p <- expandUser (oaBinary args)
            pure (Right p)
        else do
            mPath <- findExecutable "trader-hs"
            case mPath of
                Just p -> pure (Right p)
                Nothing -> do
                    cwd <- getCurrentDirectory
                    let cabalDirCandidates = [cwd, cwd </> "haskell"]
                    cabalDir <- firstExistingCabalDir cabalDirCandidates
                    case cabalDir of
                        Nothing ->
                            pure (Left "failed to discover trader-hs binary via cabal: trader.cabal not found")
                        Just dir -> do
                            let procSpec =
                                    (proc "cabal" ["list-bin", "trader-hs"])
                                        { cwd = Just dir
                                        , std_out = CreatePipe
                                        , std_err = CreatePipe
                                        }
                            r <- try (readProcessOutput procSpec) :: IO (Either SomeException (ExitCode, String, String))
                            case r of
                                Left e -> pure (Left ("failed to discover trader-hs binary via cabal: " ++ show e))
                                Right (ExitFailure _, out, err) ->
                                    pure (Left ("failed to discover trader-hs binary via cabal: " ++ trim (if null err then out else err)))
                                Right (ExitSuccess, out, _) ->
                                    let p = trim out
                                     in if null p
                                            then pure (Left "cabal returned empty binary path")
                                            else do
                                                exists <- doesFileExist p
                                                if exists
                                                    then pure (Right p)
                                                    else pure (Left ("cabal returned non-existent binary path: " ++ p))
  where
    firstExistingCabalDir [] = pure Nothing
    firstExistingCabalDir (dir : rest) = do
        exists <- doesFileExist (dir </> "trader.cabal")
        if exists
            then pure (Just dir)
            else firstExistingCabalDir rest

resolveOptimizerMaxPoints :: IO Int
resolveOptimizerMaxPoints = do
    env <- lookupEnv "TRADER_OPTIMIZER_MAX_POINTS"
    let defaultMax = 1000
        maxCap = 5000
    pure $
        case env >>= readMaybe of
            Just n | n >= 2 -> min maxCap n
            _ -> defaultMax

readProcessOutput :: CreateProcess -> IO (ExitCode, String, String)
readProcessOutput procSpec = do
    (_, Just hout, Just herr, ph) <- createProcess procSpec
    hSetEncoding hout utf8
    hSetEncoding herr utf8
    out <- hGetContents hout
    err <- hGetContents herr
    _ <- evaluate (length out + length err)
    code <- waitForProcess ph
    pure (code, out, err)

resolveIntervals :: OptimizerArgs -> IO (Either String [String])
resolveIntervals args =
    let intervalsIn =
            case oaInterval args of
                Just v | not (null (trim v)) -> [trim v]
                _ ->
                    let raw = fromMaybe binanceIntervalsCsv (oaIntervals args)
                     in [trim s | s <- splitCsv raw, not (null (trim s))]
     in if null intervalsIn then pure (Left "No intervals provided.") else pure (Right intervalsIn)

splitCsv :: String -> [String]
splitCsv raw =
    case raw of
        [] -> []
        _ ->
            let go acc current [] = reverse (reverse current : acc)
                go acc current (c : cs)
                    | c == ',' = go (reverse current : acc) [] cs
                    | otherwise = go acc (c : current) cs
             in map trim (go [] [] raw)

resolveBars :: OptimizerArgs -> [String] -> Int -> Bool -> IO (Either String (Int, Int))
resolveBars args intervals maxBarsCap useSweepThreshold = do
    let barsMax0 = oaBarsMax args
        barsMax = if barsMax0 <= 0 then maxBarsCap else barsMax0
        barsMin0 = oaBarsMin args
    if barsMin0 > 0
        then case deriveWorstLookback (oaLookbackWindow args) intervals of
            Left err -> pure (Left err)
            Right worstLb -> do
                -- Never emit a combo whose bar budget can't cover the lookback
                -- (such combos can't train/run as bots: "Not enough data for lookback").
                let minRequired = worstLb + 3
                if minRequired > barsMax
                    then pure (Left (insufficientBarsErr worstLb minRequired))
                    else do
                        let barsMin = max 2 (max minRequired (min barsMin0 barsMax))
                        pure (Right (min barsMin barsMax, barsMax))
        else case deriveWorstLookback (oaLookbackWindow args) intervals of
            Left err -> pure (Left err)
            Right worstLb -> do
                let minRequired0 = worstLb + 3
                if useSweepThreshold
                    then do
                        let br = oaBacktestRatio args
                            tr = oaTuneRatio args
                        if br <= 0 || br >= 1
                            then pure (Left "--backtest-ratio must be between 0 and 1.")
                            else
                                if tr <= 0 || tr >= 1
                                    then pure (Left "--tune-ratio must be between 0 and 1 when sweep-threshold is enabled.")
                                    else do
                                        let denom = max 1e-12 ((1 - br) * (1 - tr))
                                            minRequired1 = max minRequired0 (ceiling ((fromIntegral worstLb + 1) / denom) + 2)
                                            minTrain = ceiling (2 / tr)
                                            minRequired2 =
                                                max minRequired1 (ceiling (fromIntegral minTrain / max 1e-12 (1 - br)) + 2)
                                            autoBars = if isNothing (oaBinanceSymbol args) then maxBarsCap else 500
                                        if minRequired2 > max barsMax autoBars
                                            then
                                                pure
                                                    ( Left
                                                        ( "Not enough bars for lookback="
                                                            ++ show worstLb
                                                            ++ " with backtest-ratio="
                                                            ++ show br
                                                            ++ " and tune-ratio="
                                                            ++ show tr
                                                            ++ ". Need bars >= "
                                                            ++ show minRequired2
                                                            ++ ". Increase --bars-max, reduce --tune-ratio/--backtest-ratio, reduce --lookback-window, or pass --no-sweep-threshold."
                                                        )
                                                    )
                                            else do
                                                let barsMin = min barsMax (max 10 minRequired2)
                                                pure (Right (max 2 (min barsMin barsMax), barsMax))
                    else
                        if minRequired0 > barsMax
                            then pure (Left (insufficientBarsErr worstLb minRequired0))
                            else do
                                let barsMin = max 10 minRequired0
                                pure (Right (max 2 (min barsMin barsMax), barsMax))
  where
    insufficientBarsErr worstLb need =
        "Not enough bars for lookback="
            ++ show worstLb
            ++ " (need bars >= "
            ++ show need
            ++ "). Increase --bars-max / TRADER_OPTIMIZER_MAX_POINTS or reduce --lookback-window."
    deriveWorstLookback lookbackWindow itvs =
        case traverse (lookbackForInterval lookbackWindow) itvs of
            Left err -> Left err
            Right [] -> Left "No intervals provided."
            Right lbs -> Right (maximum lbs)

    lookbackForInterval lookbackWindow itv =
        case lookbackBarsFrom itv lookbackWindow of
            Left err -> Left ("Invalid interval/lookback combination for " ++ show itv ++ ": " ++ err)
            Right lb -> Right lb

buildBaseArgs :: OptimizerArgs -> Maybe (FilePath, (Maybe String, Maybe String)) -> IO (Either String [String])
buildBaseArgs args csvCols = do
    let base0 = []
    base1 <-
        case oaData args of
            Just _ ->
                case csvCols of
                    Nothing -> pure (Left "CSV path not resolved")
                    Just (csvPath, (mHigh, mLow)) -> do
                        -- When cross-exchange is requested, pass the flag plus a --symbol
                        -- so the CSV trial can map to a Coinbase product and align by the
                        -- CSV's own openTime grid. Fail-open if the symbol/interval is not
                        -- Coinbase-eligible.
                        let cbxArgs =
                                if oaCrossExchangeCoinbase args
                                    then
                                        let sym = fromMaybe (oaSymbolLabel args) (oaBinanceSymbol args)
                                         in if null sym then ["--cross-exchange-coinbase"] else ["--cross-exchange-coinbase", "--cross-exchange-symbol", sym]
                                    else []
                            base = base0 ++ ["--data", csvPath, "--price-column", oaPriceColumn args] ++ cbxArgs
                        case (mHigh, mLow) of
                            (Nothing, Nothing) -> pure (Right base)
                            (Just h, Just l) -> pure (Right (base ++ ["--high-column", h, "--low-column", l]))
                            _ -> pure (Left "When using --high-column/--low-column, you must provide both.")
            Nothing ->
                case oaBinanceSymbol args of
                    Just sym -> pure (Right (base0 ++ ["--binance-symbol", sym]))
                    Nothing -> pure (Left "--symbol/--data is required")
    case base1 of
        Left err -> pure (Left err)
        Right baseArgs -> do
            let baseArgs' =
                    baseArgs
                        ++ [ "--lookback-window"
                           , oaLookbackWindow args
                           , "--backtest-ratio"
                           , printf "%.6f" (oaBacktestRatio args)
                           , "--tune-objective"
                           , oaTuneObjective args
                           , "--tune-penalty-max-drawdown"
                           , printf "%.6f" (max 0 (oaTunePenaltyMaxDrawdown args))
                           , "--tune-penalty-turnover"
                           , printf "%.6f" (max 0 (oaTunePenaltyTurnover args))
                           , "--tune-max-threshold-candidates"
                           , show (max 0 (oaTuneMaxThresholdCandidates args))
                           , "--seed"
                           , show (oaSeed args)
                           ]
            let baseArgs'' =
                    if oaBinanceFutures args
                        then baseArgs' ++ ["--futures"]
                        else baseArgs'
            pure (Right baseArgs'')

printTrialStatus :: Int -> Int -> TrialResult -> IO ()
printTrialStatus i trials tr = do
    let status :: String
        status
            | trEligible tr = "OK"
            | trOk tr = "SKIP"
            | otherwise = "FAIL"
        eq :: String
        eq = maybe "-" (printf "%.6fx") (trFinalEquity tr)
        scoreLabel :: String
        scoreLabel = maybe "-" (printf "%.6f") (trScore tr)
        params = trParams tr
        msg =
            printf
                "[%4d/%d] %s score=%s eq=%s t=%.2fs interval=%s bars=%d method=%s norm=%s epochs=%d slip=%.6f spr=%.6f sl=%s tp=%s trail=%s maxDD=%s maxDL=%s maxWL=%s maxOE=%s"
                i
                trials
                status
                scoreLabel
                eq
                (trElapsedSec tr)
                (tpInterval params)
                (tpBars params)
                (tpMethod params)
                (tpNormalization params)
                (tpEpochs params)
                (tpSlippage params)
                (tpSpread params)
                (fmtOptFloat (tpStopLoss params))
                (fmtOptFloat (tpTakeProfit params))
                (fmtOptFloat (tpTrailingStop params))
                (fmtOptFloat (tpMaxDrawdown params))
                (fmtOptFloat (tpMaxDailyLoss params))
                (fmtOptFloat (tpMaxWeeklyLoss params))
                (fmtOptInt (tpMaxOrderErrors params))
        suffix =
            case (trFilterReason tr, trReason tr) of
                (Just reason, _) -> " (filter: " ++ reason ++ ")"
                (Nothing, Just reason) -> " (" ++ reason ++ ")"
                _ -> ""
        diagnosticSuffix =
            case activityFilterDiagnostic tr of
                Just diagnostic -> " (diagnostic: " ++ diagnostic ++ ")"
                Nothing -> ""
    putStrLn (msg ++ suffix ++ diagnosticSuffix)
    hFlush stdout

printNoEligible :: Int -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> Double -> IO ()
printNoEligible minRoundTrips minWinRate minProfitFactor minExposure minKellyLiteExposureReduction maxKellyLiteExposureRatio minSharpe minAnnualizedReturn minCalmar maxTurnover minWfSharpeMean maxWfSharpeStd = do
    let hints =
            ["--min-round-trips" | minRoundTrips > 0]
                ++ ["--min-win-rate" | minWinRate > 0]
                ++ ["--min-profit-factor" | minProfitFactor > 0]
                ++ ["--min-exposure" | minExposure > 0]
                ++ ["--min-kelly-lite-exposure-reduction" | minKellyLiteExposureReduction > 0]
                ++ ["--max-kelly-lite-exposure-ratio" | maxKellyLiteExposureRatio > 0 && maxKellyLiteExposureRatio < 1]
                ++ ["--min-sharpe" | minSharpe > 0]
                ++ ["--min-annualized-return" | minAnnualizedReturn > 0]
                ++ ["--min-calmar" | minCalmar > 0]
                ++ ["--max-turnover" | maxTurnover > 0]
                ++ ["--min-wf-sharpe-mean" | minWfSharpeMean > 0]
                ++ ["--max-wf-sharpe-std" | maxWfSharpeStd > 0]
        msg =
            if null hints
                then "No eligible trials."
                else "No eligible trials. (Try lowering " ++ intercalate ", " hints ++ ".)"
    hPutStrLn stderr msg

printBest :: TrialResult -> IO ()
printBest tr = do
    let p = trParams tr
    putStrLn "\nBest:"
    case trScore tr of
        Just sc -> putStrLn ("  objective:   " ++ trObjective tr ++ " (score=" ++ printf "%.8f" sc ++ ")")
        Nothing -> pure ()
    putStrLn ("  finalEquity: " ++ printf "%.8f" (fromMaybe 0 (trFinalEquity tr)) ++ "x")
    case tpPlatform p of
        Just platform -> putStrLn ("  platform:    " ++ platform)
        Nothing -> pure ()
    putStrLn ("  interval:    " ++ tpInterval p)
    putStrLn ("  bars:        " ++ show (tpBars p))
    putStrLn ("  method:      " ++ tpMethod p)
    putStrLn ("  positioning: " ++ tpPositioning p)
    putStrLn ("  thresholds:  open=" ++ showMaybe (trOpenThreshold tr) ++ " close=" ++ showMaybe (trCloseThreshold tr) ++ " (from sweep)")
    putStrLn ("  base thresholds: open=" ++ show (tpBaseOpenThreshold p) ++ " close=" ++ show (tpBaseCloseThreshold p))
    putStrLn ("  blendWeight:  " ++ show (tpBlendWeight p))
    putStrLn ("  routerScorePnlWeight: " ++ show (tpRouterScorePnlWeight p))
    putStrLn ("  routerRegimeMinBars: " ++ show (tpRouterRegimeMinBars p))
    putStrLn ("  routerRegimeMinFraction: " ++ show (tpRouterRegimeMinFraction p))
    putStrLn ("  minHoldBars:  " ++ show (tpMinHoldBars p))
    putStrLn ("  cooldownBars: " ++ show (tpCooldownBars p))
    putStrLn ("  maxHoldBars:  " ++ showMaybe (tpMaxHoldBars p))
    putStrLn ("  minEdge:      " ++ show (tpMinEdge p))
    putStrLn ("  minSignalToNoise: " ++ show (tpMinSignalToNoise p))
    putStrLn ("  snrSizeWeight: " ++ show (tpSnrSizeWeight p))
    putStrLn ("  thresholdFactorEnabled: " ++ show (tpThresholdFactorEnabled p))
    putStrLn ("  thresholdFactorAlpha: " ++ show (tpThresholdFactorAlpha p))
    putStrLn ("  thresholdFactorBounds: " ++ show (tpThresholdFactorMin p) ++ ".." ++ show (tpThresholdFactorMax p))
    putStrLn ("  thresholdFactorFloor: " ++ show (tpThresholdFactorFloor p))
    putStrLn
        ( "  thresholdFactorWeights:"
            ++ " edgeKal="
            ++ show (tpThresholdFactorEdgeKalWeight p)
            ++ " edgeLstm="
            ++ show (tpThresholdFactorEdgeLstmWeight p)
            ++ " kalmanZ="
            ++ show (tpThresholdFactorKalmanZWeight p)
            ++ " highVol="
            ++ show (tpThresholdFactorHighVolWeight p)
            ++ " conformal="
            ++ show (tpThresholdFactorConformalWeight p)
            ++ " quantile="
            ++ show (tpThresholdFactorQuantileWeight p)
            ++ " lstmConf="
            ++ show (tpThresholdFactorLstmConfWeight p)
            ++ " lstmHealth="
            ++ show (tpThresholdFactorLstmHealthWeight p)
        )
    putStrLn ("  edgeBuffer:   " ++ show (tpEdgeBuffer p))
    putStrLn ("  costAwareEdge:" ++ show (tpCostAwareEdge p))
    putStrLn ("  trendLookback:" ++ show (tpTrendLookback p))
    putStrLn ("  maxPositionSize:" ++ show (tpMaxPositionSize p))
    putStrLn ("  volTarget:    " ++ showMaybe (tpVolTarget p))
    putStrLn ("  volLookback:  " ++ show (tpVolLookback p))
    putStrLn ("  volEwmaAlpha: " ++ showMaybe (tpVolEwmaAlpha p))
    putStrLn ("  volFloor:     " ++ show (tpVolFloor p))
    putStrLn ("  volScaleMax:  " ++ show (tpVolScaleMax p))
    putStrLn ("  maxVolatility: " ++ showMaybe (tpMaxVolatility p))
    putStrLn ("  periodsPerYear:" ++ showMaybe (tpPeriodsPerYear p))
    putStrLn ("  kalmanMarketTopN:" ++ show (tpKalmanMarketTopN p))
    putStrLn ("  normalization: " ++ show (tpNormalization p))
    putStrLn ("  epochs:        " ++ show (tpEpochs p))
    putStrLn ("  hiddenSize:    " ++ show (tpHiddenSize p))
    putStrLn ("  lr:            " ++ show (tpLearningRate p))
    putStrLn ("  lstmAdamBeta1: " ++ show (tpLstmAdamBeta1 p))
    putStrLn ("  lstmAdamBeta2: " ++ show (tpLstmAdamBeta2 p))
    putStrLn ("  lstmAdamEps:   " ++ show (tpLstmAdamEps p))
    putStrLn ("  valRatio:      " ++ show (tpValRatio p))
    putStrLn ("  patience:      " ++ show (tpPatience p))
    putStrLn ("  walkForwardFolds:" ++ show (tpWalkForwardFolds p))
    putStrLn ("  walkForwardEmbargoBars:" ++ show (tpWalkForwardEmbargoBars p))
    putStrLn ("  tuneStressVolMult:" ++ show (tpTuneStressVolMult p))
    putStrLn ("  tuneStressShock: " ++ show (tpTuneStressShock p))
    putStrLn ("  tuneStressWeight:" ++ show (tpTuneStressWeight p))
    putStrLn ("  gradClip:      " ++ showMaybe (tpGradClip p))
    putStrLn ("  fee:           " ++ show (tpFee p))
    putStrLn ("  fundingRate:   " ++ show (tpFundingRate p))
    putStrLn ("  fundingBySide: " ++ show (tpFundingBySide p))
    putStrLn ("  fundingOnOpen: " ++ show (tpFundingOnOpen p))
    putStrLn ("  rebalanceBars: " ++ show (tpRebalanceBars p))
    putStrLn ("  rebalanceThreshold:" ++ show (tpRebalanceThreshold p))
    putStrLn ("  rebalanceCostMult:" ++ show (tpRebalanceCostMult p))
    putStrLn ("  rebalanceGlobal: " ++ show (tpRebalanceGlobal p))
    putStrLn ("  rebalanceResetOnSignal:" ++ show (tpRebalanceResetOnSignal p))
    putStrLn ("  slippage:      " ++ show (tpSlippage p))
    putStrLn ("  spread:        " ++ show (tpSpread p))
    putStrLn ("  intrabarFill:  " ++ show (tpIntrabarFill p))
    putStrLn ("  triLayer:      " ++ show (tpTriLayer p))
    putStrLn ("  triLayerFastMult: " ++ show (tpTriLayerFastMult p))
    putStrLn ("  triLayerSlowMult: " ++ show (tpTriLayerSlowMult p))
    putStrLn ("  triLayerCloudPadding: " ++ show (tpTriLayerCloudPadding p))
    putStrLn ("  triLayerCloudSlope: " ++ show (tpTriLayerCloudSlope p))
    putStrLn ("  triLayerCloudWidth: " ++ show (tpTriLayerCloudWidth p))
    putStrLn ("  triLayerTouchLookback: " ++ show (tpTriLayerTouchLookback p))
    putStrLn ("  triLayerPriceAction: " ++ show (tpTriLayerPriceAction p))
    putStrLn ("  triLayerPriceActionBody: " ++ show (tpTriLayerPriceActionBody p))
    putStrLn ("  triLayerPriceActionWickRatio: " ++ show (tpTriLayerPriceActionWickRatio p))
    putStrLn ("  triLayerPriceActionOppositeWickMax: " ++ show (tpTriLayerPriceActionOppositeWickMax p))
    putStrLn ("  triLayerPriceActionBodyTolerance: " ++ show (tpTriLayerPriceActionBodyTolerance p))
    putStrLn ("  triLayerExitOnSlow: " ++ show (tpTriLayerExitOnSlow p))
    putStrLn ("  kalmanBandLookback: " ++ show (tpKalmanBandLookback p))
    putStrLn ("  kalmanBandStdMult: " ++ show (tpKalmanBandStdMult p))
    putStrLn ("  lstmExitFlipBars: " ++ show (tpLstmExitFlipBars p))
    putStrLn ("  lstmExitFlipGraceBars: " ++ show (tpLstmExitFlipGraceBars p))
    putStrLn ("  lstmExitFlipStrong: " ++ show (tpLstmExitFlipStrong p))
    putStrLn ("  lstmConfidenceSoft: " ++ show (tpLstmConfidenceSoft p))
    putStrLn ("  lstmConfidenceHard: " ++ show (tpLstmConfidenceHard p))
    putStrLn ("  stopLoss:      " ++ showMaybe (tpStopLoss p))
    putStrLn ("  takeProfit:    " ++ showMaybe (tpTakeProfit p))
    putStrLn ("  trailingStop:  " ++ showMaybe (tpTrailingStop p))
    putStrLn ("  stopLossVolMult:    " ++ showMaybe (tpStopLossVolMult p))
    putStrLn ("  takeProfitVolMult:  " ++ showMaybe (tpTakeProfitVolMult p))
    putStrLn ("  trailingStopVolMult:" ++ showMaybe (tpTrailingStopVolMult p))
    putStrLn ("  riskPerTrade:       " ++ showMaybe (tpRiskPerTrade p))
    putStrLn ("  maxDrawdown:   " ++ showMaybe (tpMaxDrawdown p))
    putStrLn ("  maxDailyLoss:  " ++ showMaybe (tpMaxDailyLoss p))
    putStrLn ("  maxWeeklyLoss:" ++ showMaybe (tpMaxWeeklyLoss p))
    putStrLn ("  maxOrderErrors:" ++ showMaybe (tpMaxOrderErrors p))
    putStrLn ("  kalmanDt:            " ++ show (tpKalmanDt p))
    putStrLn ("  kalmanProcessVar:    " ++ show (tpKalmanProcessVar p))
    putStrLn ("  kalmanMeasurementVar:" ++ show (tpKalmanMeasurementVar p))
    putStrLn ("  kalmanPhysicsBars:   " ++ show (tpKalmanPhysicsBars p))
    putStrLn ("  kalmanPhysicsBacktestRatio:" ++ show (tpKalmanPhysicsBacktestRatio p))
    putStrLn ("  kalmanPhysicsVolumeEwmaAlpha:" ++ show (tpKalmanPhysicsVolumeEwmaAlpha p))
    putStrLn ("  kalmanPhysicsVolumeSignalClamp:" ++ show (tpKalmanPhysicsVolumeSignalClamp p))
    putStrLn ("  kalmanPhysicsCloseBiasScale:" ++ show (tpKalmanPhysicsCloseBiasScale p))
    putStrLn ("  kalmanZMin:          " ++ show (tpKalmanZMin p))
    putStrLn ("  kalmanZMax:          " ++ show (tpKalmanZMax p))
    putStrLn ("  maxHighVolProb:      " ++ showMaybe (tpMaxHighVolProb p))
    putStrLn ("  maxConformalWidth:   " ++ showMaybe (tpMaxConformalWidth p))
    putStrLn ("  maxQuantileWidth:    " ++ showMaybe (tpMaxQuantileWidth p))
    putStrLn ("  confirmConformal:    " ++ show (tpConfirmConformal p))
    putStrLn ("  confirmQuantiles:    " ++ show (tpConfirmQuantiles p))
    putStrLn ("  confidenceSizing:    " ++ show (tpConfidenceSizing p))
    putStrLn ("  protectionMinConf:   " ++ show (tpProtectionMinConfidence p))
    putStrLn ("  minPositionSize:     " ++ show (tpMinPositionSize p))
    putStrLn ("  adaptiveFilters:     " ++ show (tpAdaptiveFilters p))
    putStrLn ("  perfLookback:        " ++ show (tpPerfLookback p))
    putStrLn ("  perfMinWinRate:      " ++ showMaybe (tpPerfMinWinRate p))
    putStrLn ("  perfMinProfitFactor: " ++ showMaybe (tpPerfMinProfitFactor p))
    putStrLn ("  adaptiveWinRateSlack:" ++ show (tpAdaptiveWinRateSlack p))
    putStrLn ("  adaptiveProfitFactorSlack:" ++ show (tpAdaptiveProfitFactorSlack p))
    putStrLn ("  metaLabelFilter:     " ++ show (tpMetaLabelFilter p))
    putStrLn ("  metaLabelMinEdge:    " ++ show (tpMetaLabelMinEdge p))
    putStrLn ("  metaLabelMinConfidence:" ++ show (tpMetaLabelMinConfidence p))
    putStrLn ("  metaLabelRequireBand:" ++ show (tpMetaLabelRequireBand p))
    putStrLn ("  regimeParameterBank:" ++ show (tpRegimeParameterBank p))
    putStrLn ("  regimeBankHysteresis:" ++ show (tpRegimeBankHysteresis p))
    putStrLn ("  regimeTrendOpenMult:" ++ show (tpRegimeTrendOpenMult p))
    putStrLn ("  regimeMrOpenMult:   " ++ show (tpRegimeMrOpenMult p))
    putStrLn ("  regimeHighVolOpenMult:" ++ show (tpRegimeHighVolOpenMult p))
    putStrLn ("  regimeTrendSizeMult:" ++ show (tpRegimeTrendSizeMult p))
    putStrLn ("  regimeMrSizeMult:   " ++ show (tpRegimeMrSizeMult p))
    putStrLn ("  regimeHighVolSizeMult:" ++ show (tpRegimeHighVolSizeMult p))
    putStrLn ("  technicalParams:    " ++ show (tpTechnicalParams p))
    putStrLn ("  multiTimeframeConsensus:" ++ show (tpMultiTimeframeConsensus p))
    putStrLn ("  mtfFastBars:        " ++ show (tpMtfFastBars p))
    putStrLn ("  mtfMidBars:         " ++ show (tpMtfMidBars p))
    putStrLn ("  mtfSlowBars:        " ++ show (tpMtfSlowBars p))
    putStrLn ("  mtfMinAgree:        " ++ show (tpMtfMinAgree p))
    putStrLn ("  crossAssetConfirmation:" ++ show (tpCrossAssetConfirmation p))
    putStrLn ("  crossAssetMinBeta:  " ++ show (tpCrossAssetMinBeta p))
    putStrLn ("  crossAssetMinEdge:  " ++ show (tpCrossAssetMinEdge p))
    putStrLn ("  pairsStatArb:       " ++ show (tpPairsStatArb p))
    putStrLn ("  pairsStatArbLookback:" ++ show (tpPairsStatArbLookback p))
    putStrLn ("  pairsStatArbZEntry: " ++ show (tpPairsStatArbZEntry p))
    putStrLn ("  pairsStatArbSizeMult:" ++ show (tpPairsStatArbSizeMult p))
    putStrLn ("  fundingOiAware:     " ++ show (tpFundingOiAware p))
    putStrLn ("  fundingOiFundingCap:" ++ showMaybe (tpFundingOiFundingCap p))
    putStrLn ("  fundingOiVolLookback:" ++ show (tpFundingOiVolLookback p))
    putStrLn ("  fundingOiVolCap:    " ++ showMaybe (tpFundingOiVolCap p))
    putStrLn ("  fundingOiSizeMult:  " ++ show (tpFundingOiSizeMult p))
    putStrLn ("  kellyLiteSizing:    " ++ show (tpKellyLiteSizing p))
    putStrLn ("  kellyLiteFraction:  " ++ show (tpKellyLiteFraction p))
    putStrLn ("  kellyLiteFloor:     " ++ show (tpKellyLiteFloor p))
    putStrLn ("  kellyLiteCap:       " ++ show (tpKellyLiteCap p))

showMaybe :: (Show a) => Maybe a -> String
showMaybe = maybe "None" show

printRepro :: FilePath -> [String] -> TrialResult -> Double -> Bool -> Bool -> IO ()
printRepro traderBin baseArgs tr tuneRatio useSweepThreshold disableLstm = do
    putStrLn "\nRepro command:"
    let envPrefix = if disableLstm then "TRADER_LSTM_WEIGHTS_DIR='' " else ""
        cmd = buildCommand traderBin baseArgs (trParams tr) tuneRatio useSweepThreshold
    putStrLn ("  " ++ envPrefix ++ unwords cmd)

data OptimizationTechnique = OptimizationTechnique
    { otName :: !String
    , otSummary :: !String
    , otWhyItHelps :: !String
    }
    deriving (Eq, Show)

data OptimizationTechniqueSummary = OptimizationTechniqueSummary
    { otsAppliedSobolSeeding :: !Bool
    , otsAppliedSuccessiveHalving :: !Bool
    , otsAppliedBayesianEi :: !Bool
    , otsAppliedEmpiricalPriors :: !Bool
    , otsAppliedWalkForward :: !Bool
    , otsAppliedEnsemble :: !Bool
    }
    deriving (Eq, Show)

emptyTechniqueSummary :: OptimizationTechniqueSummary
emptyTechniqueSummary =
    OptimizationTechniqueSummary
        { otsAppliedSobolSeeding = False
        , otsAppliedSuccessiveHalving = False
        , otsAppliedBayesianEi = False
        , otsAppliedEmpiricalPriors = False
        , otsAppliedWalkForward = False
        , otsAppliedEnsemble = False
        }

bestOptimizationTechniques :: [OptimizationTechnique]
bestOptimizationTechniques =
    [ OptimizationTechnique
        { otName = "Bayesian optimization (expected improvement)"
        , otSummary = "Model the objective surface and pick trials that maximize expected improvement over the best result."
        , otWhyItHelps = "Improves sample efficiency versus uniform random search when trials are expensive or noisy."
        }
    , OptimizationTechnique
        { otName = "Successive halving / ASHA"
        , otSummary = "Run many cheap partial trainings, pruning low performers early while allocating budget to promising candidates."
        , otWhyItHelps = "Cuts wasted computation on weak trials and converges faster to strong configurations."
        }
    , OptimizationTechnique
        { otName = "Sobol / Latin hypercube seeding"
        , otSummary = "Start the search with low-discrepancy samples that cover the space more uniformly than pure random draws."
        , otWhyItHelps = "Reduces blind spots and gives later exploitation steps better anchor points."
        }
    , OptimizationTechnique
        { otName = "Empirical prior sampling"
        , otSummary = "Bias new trials toward validated historical parameter neighborhoods before broad survivor exploitation."
        , otWhyItHelps = "Spends more budget near symbol, interval, and method contexts that already cleared live-relevant evidence gates."
        }
    , OptimizationTechnique
        { otName = "Walk-forward / blocked cross-validation"
        , otSummary = "Score candidates across rolling, non-overlapping time blocks to avoid leakage and stress-test stability."
        , otWhyItHelps = "Rewards configurations that generalize across regimes instead of overfitting a single window."
        }
    , OptimizationTechnique
        { otName = "Ensemble the top performers"
        , otSummary = "Blend the strongest, diverse configurations into a composite strategy instead of picking a single winner."
        , otWhyItHelps = "Smooths variance and reduces the risk of relying on one brittle optimum."
        }
    ]

optimizationTechniqueToJson :: OptimizationTechnique -> Value
optimizationTechniqueToJson t =
    object
        [ "name" .= otName t
        , "summary" .= otSummary t
        , "whyItHelps" .= otWhyItHelps t
        ]

perturbDouble :: Double -> Double -> Rng -> (Double, Rng)
perturbDouble v scale rng =
    let s = max 0 scale
        (factor, rng1) = nextUniform (1 - s) (1 + s) rng
     in (max 0 (v * factor), rng1)

perturbDoubleSigned :: Double -> Double -> Rng -> (Double, Rng)
perturbDoubleSigned v scale rng =
    let s = max 0 scale
        (factor, rng1) = nextUniform (1 - s) (1 + s) rng
     in (v * factor, rng1)

perturbMaybeDouble :: Maybe Double -> Double -> Rng -> (Maybe Double, Rng)
perturbMaybeDouble v scale rng =
    case v of
        Nothing -> (Nothing, rng)
        Just x ->
            let (x', rng1) = perturbDouble x scale rng
             in (Just x', rng1)

perturbInt :: Int -> Int -> Rng -> (Int, Rng)
perturbInt v maxDelta rng =
    let deltaMax = max 0 maxDelta
        (delta, rng1) = nextIntRange (-deltaMax) deltaMax rng
     in (max 1 (v + delta), rng1)

perturbMaybeInt :: Maybe Int -> Int -> Rng -> (Maybe Int, Rng)
perturbMaybeInt v maxDelta rng =
    case v of
        Nothing -> (Nothing, rng)
        Just x ->
            let (x', rng1) = perturbInt x maxDelta rng
             in (Just x', rng1)

pickValue :: a -> a -> Rng -> (a, Rng)
pickValue a b rng =
    let (r, rng1) = nextDouble rng
     in (if r < 0.5 then a else b, rng1)

pickParentPair :: [TrialParams] -> Rng -> Maybe ((TrialParams, TrialParams), Rng)
pickParentPair parents rng0 =
    case parents of
        [] -> Nothing
        p0 : rest ->
            let n = length parents
                p1 = fromMaybe p0 (listToMaybe rest)
                (i, rng1) = nextIntRange 0 (n - 1) rng0
                (j, rng2) = nextIntRange 0 (n - 1) rng1
                j' = if j == i then (j + 1) `mod` n else j
                pi' = fromMaybe p0 (listToMaybe (drop i parents))
                pj' = fromMaybe p1 (listToMaybe (drop j' parents))
             in Just ((pi', pj'), rng2)

crossoverTrialParams :: TrialParams -> TrialParams -> Rng -> (TrialParams, Rng)
crossoverTrialParams a b rng0 =
    let (tpPlatform', rng1) = pickValue (tpPlatform a) (tpPlatform b) rng0
        (tpInterval', rng2) = pickValue (tpInterval a) (tpInterval b) rng1
        (tpBars', rng3) = pickValue (tpBars a) (tpBars b) rng2
        (tpMethod', rng4) = pickValue (tpMethod a) (tpMethod b) rng3
        (tpBlendWeight', rng5) = pickValue (tpBlendWeight a) (tpBlendWeight b) rng4
        (tpRouterScorePnlWeight', rng5a) = pickValue (tpRouterScorePnlWeight a) (tpRouterScorePnlWeight b) rng5
        (tpPositioning', rng6) = pickValue (tpPositioning a) (tpPositioning b) rng5a
        (tpNormalization', rng7) = pickValue (tpNormalization a) (tpNormalization b) rng6
        (tpBaseOpenThreshold', rng8) = pickValue (tpBaseOpenThreshold a) (tpBaseOpenThreshold b) rng7
        (tpBaseCloseThreshold', rng9) = pickValue (tpBaseCloseThreshold a) (tpBaseCloseThreshold b) rng8
        (tpMinHoldBars', rng10) = pickValue (tpMinHoldBars a) (tpMinHoldBars b) rng9
        (tpCooldownBars', rng11) = pickValue (tpCooldownBars a) (tpCooldownBars b) rng10
        (tpMaxHoldBars', rng12) = pickValue (tpMaxHoldBars a) (tpMaxHoldBars b) rng11
        (tpMinEdge', rng13) = pickValue (tpMinEdge a) (tpMinEdge b) rng12
        (tpMinSignalToNoise', rng14) = pickValue (tpMinSignalToNoise a) (tpMinSignalToNoise b) rng13
        (tpSnrSizeWeight', rng14a) = pickValue (tpSnrSizeWeight a) (tpSnrSizeWeight b) rng14
        (tpThresholdFactorEnabled', rng15) = pickValue (tpThresholdFactorEnabled a) (tpThresholdFactorEnabled b) rng14a
        (tpThresholdFactorAlpha', rng16) = pickValue (tpThresholdFactorAlpha a) (tpThresholdFactorAlpha b) rng15
        (tpThresholdFactorMin', rng17) = pickValue (tpThresholdFactorMin a) (tpThresholdFactorMin b) rng16
        (tpThresholdFactorMax', rng18) = pickValue (tpThresholdFactorMax a) (tpThresholdFactorMax b) rng17
        (tpThresholdFactorFloor', rng19) = pickValue (tpThresholdFactorFloor a) (tpThresholdFactorFloor b) rng18
        (tpThresholdFactorEdgeKalWeight', rng20) = pickValue (tpThresholdFactorEdgeKalWeight a) (tpThresholdFactorEdgeKalWeight b) rng19
        (tpThresholdFactorEdgeLstmWeight', rng21) = pickValue (tpThresholdFactorEdgeLstmWeight a) (tpThresholdFactorEdgeLstmWeight b) rng20
        (tpThresholdFactorKalmanZWeight', rng22) = pickValue (tpThresholdFactorKalmanZWeight a) (tpThresholdFactorKalmanZWeight b) rng21
        (tpThresholdFactorHighVolWeight', rng23) = pickValue (tpThresholdFactorHighVolWeight a) (tpThresholdFactorHighVolWeight b) rng22
        (tpThresholdFactorConformalWeight', rng24) = pickValue (tpThresholdFactorConformalWeight a) (tpThresholdFactorConformalWeight b) rng23
        (tpThresholdFactorQuantileWeight', rng25) = pickValue (tpThresholdFactorQuantileWeight a) (tpThresholdFactorQuantileWeight b) rng24
        (tpThresholdFactorLstmConfWeight', rng26) = pickValue (tpThresholdFactorLstmConfWeight a) (tpThresholdFactorLstmConfWeight b) rng25
        (tpThresholdFactorLstmHealthWeight', rng27) = pickValue (tpThresholdFactorLstmHealthWeight a) (tpThresholdFactorLstmHealthWeight b) rng26
        (tpEdgeBuffer', rng28) = pickValue (tpEdgeBuffer a) (tpEdgeBuffer b) rng27
        (tpCostAwareEdge', rng29) = pickValue (tpCostAwareEdge a) (tpCostAwareEdge b) rng28
        (tpTrendLookback', rng30) = pickValue (tpTrendLookback a) (tpTrendLookback b) rng29
        (tpMaxPositionSize', rng31) = pickValue (tpMaxPositionSize a) (tpMaxPositionSize b) rng30
        (tpVolTarget', rng32) = pickValue (tpVolTarget a) (tpVolTarget b) rng31
        (tpVolLookback', rng33) = pickValue (tpVolLookback a) (tpVolLookback b) rng32
        (tpVolEwmaAlpha', rng34) = pickValue (tpVolEwmaAlpha a) (tpVolEwmaAlpha b) rng33
        (tpVolFloor', rng35) = pickValue (tpVolFloor a) (tpVolFloor b) rng34
        (tpVolScaleMax', rng36) = pickValue (tpVolScaleMax a) (tpVolScaleMax b) rng35
        (tpMaxVolatility', rng37) = pickValue (tpMaxVolatility a) (tpMaxVolatility b) rng36
        (tpPeriodsPerYear', rng38) = pickValue (tpPeriodsPerYear a) (tpPeriodsPerYear b) rng37
        (tpKalmanMarketTopN', rng39) = pickValue (tpKalmanMarketTopN a) (tpKalmanMarketTopN b) rng38
        (tpSensorVarianceEwmaAlpha', rng39a) =
            pickValue (tpSensorVarianceEwmaAlpha a) (tpSensorVarianceEwmaAlpha b) rng39
        (tpFee', rng40) = pickValue (tpFee a) (tpFee b) rng39a
        (tpFundingRate', rng41) = pickValue (tpFundingRate a) (tpFundingRate b) rng40
        (tpFundingBySide', rng42) = pickValue (tpFundingBySide a) (tpFundingBySide b) rng41
        (tpFundingOnOpen', rng43) = pickValue (tpFundingOnOpen a) (tpFundingOnOpen b) rng42
        (tpRebalanceBars', rng44) = pickValue (tpRebalanceBars a) (tpRebalanceBars b) rng43
        (tpRebalanceThreshold', rng45) = pickValue (tpRebalanceThreshold a) (tpRebalanceThreshold b) rng44
        (tpRebalanceCostMult', rng45a) = pickValue (tpRebalanceCostMult a) (tpRebalanceCostMult b) rng45
        (tpRebalanceGlobal', rng46) = pickValue (tpRebalanceGlobal a) (tpRebalanceGlobal b) rng45a
        (tpRebalanceResetOnSignal', rng47) = pickValue (tpRebalanceResetOnSignal a) (tpRebalanceResetOnSignal b) rng46
        (tpEpochs', rng48) = pickValue (tpEpochs a) (tpEpochs b) rng47
        (tpHiddenSize', rng49) = pickValue (tpHiddenSize a) (tpHiddenSize b) rng48
        (tpLearningRate', rng50) = pickValue (tpLearningRate a) (tpLearningRate b) rng49
        (tpLstmAdamBeta1', rng50a) = pickValue (tpLstmAdamBeta1 a) (tpLstmAdamBeta1 b) rng50
        (tpLstmAdamBeta2', rng50b) = pickValue (tpLstmAdamBeta2 a) (tpLstmAdamBeta2 b) rng50a
        (tpLstmAdamEps', rng50c) = pickValue (tpLstmAdamEps a) (tpLstmAdamEps b) rng50b
        (tpValRatio', rng51) = pickValue (tpValRatio a) (tpValRatio b) rng50c
        (tpPatience', rng52) = pickValue (tpPatience a) (tpPatience b) rng51
        (tpWalkForwardFolds', rng53) = pickValue (tpWalkForwardFolds a) (tpWalkForwardFolds b) rng52
        (tpWalkForwardEmbargoBars', rng53a) = pickValue (tpWalkForwardEmbargoBars a) (tpWalkForwardEmbargoBars b) rng53
        (tpTuneStressVolMult', rng54) = pickValue (tpTuneStressVolMult a) (tpTuneStressVolMult b) rng53a
        (tpTuneStressShock', rng55) = pickValue (tpTuneStressShock a) (tpTuneStressShock b) rng54
        (tpTuneStressWeight', rng56) = pickValue (tpTuneStressWeight a) (tpTuneStressWeight b) rng55
        (tpGradClip', rng57) = pickValue (tpGradClip a) (tpGradClip b) rng56
        (tpSlippage', rng58) = pickValue (tpSlippage a) (tpSlippage b) rng57
        (tpSpread', rng59) = pickValue (tpSpread a) (tpSpread b) rng58
        (tpIntrabarFill', rng60) = pickValue (tpIntrabarFill a) (tpIntrabarFill b) rng59
        (tpTriLayer', rng61) = pickValue (tpTriLayer a) (tpTriLayer b) rng60
        (tpTriLayerFastMult', rng62) = pickValue (tpTriLayerFastMult a) (tpTriLayerFastMult b) rng61
        (tpTriLayerSlowMult', rng63) = pickValue (tpTriLayerSlowMult a) (tpTriLayerSlowMult b) rng62
        (tpTriLayerCloudPadding', rng64) = pickValue (tpTriLayerCloudPadding a) (tpTriLayerCloudPadding b) rng63
        (tpTriLayerCloudSlope', rng65) = pickValue (tpTriLayerCloudSlope a) (tpTriLayerCloudSlope b) rng64
        (tpTriLayerCloudWidth', rng66) = pickValue (tpTriLayerCloudWidth a) (tpTriLayerCloudWidth b) rng65
        (tpTriLayerTouchLookback', rng67) = pickValue (tpTriLayerTouchLookback a) (tpTriLayerTouchLookback b) rng66
        (tpTriLayerPriceAction', rng68) = pickValue (tpTriLayerPriceAction a) (tpTriLayerPriceAction b) rng67
        (tpTriLayerPriceActionBody', rng69) = pickValue (tpTriLayerPriceActionBody a) (tpTriLayerPriceActionBody b) rng68
        (tpTriLayerPriceActionWickRatio', rng69a) = pickValue (tpTriLayerPriceActionWickRatio a) (tpTriLayerPriceActionWickRatio b) rng69
        (tpTriLayerPriceActionOppositeWickMax', rng69b) = pickValue (tpTriLayerPriceActionOppositeWickMax a) (tpTriLayerPriceActionOppositeWickMax b) rng69a
        (tpTriLayerPriceActionBodyTolerance', rng69c) = pickValue (tpTriLayerPriceActionBodyTolerance a) (tpTriLayerPriceActionBodyTolerance b) rng69b
        (tpTriLayerExitOnSlow', rng70) = pickValue (tpTriLayerExitOnSlow a) (tpTriLayerExitOnSlow b) rng69c
        (tpKalmanBandLookback', rng71) = pickValue (tpKalmanBandLookback a) (tpKalmanBandLookback b) rng70
        (tpKalmanBandStdMult', rng72) = pickValue (tpKalmanBandStdMult a) (tpKalmanBandStdMult b) rng71
        (tpLstmExitFlipBars', rng73) = pickValue (tpLstmExitFlipBars a) (tpLstmExitFlipBars b) rng72
        (tpLstmExitFlipGraceBars', rng74) = pickValue (tpLstmExitFlipGraceBars a) (tpLstmExitFlipGraceBars b) rng73
        (tpLstmExitFlipStrong', rng75) = pickValue (tpLstmExitFlipStrong a) (tpLstmExitFlipStrong b) rng74
        (tpLstmConfidenceSoft', rng76) = pickValue (tpLstmConfidenceSoft a) (tpLstmConfidenceSoft b) rng75
        (tpLstmConfidenceHard', rng77) = pickValue (tpLstmConfidenceHard a) (tpLstmConfidenceHard b) rng76
        (tpStopLoss', rng78) = pickValue (tpStopLoss a) (tpStopLoss b) rng77
        (tpTakeProfit', rng79) = pickValue (tpTakeProfit a) (tpTakeProfit b) rng78
        (tpTrailingStop', rng80) = pickValue (tpTrailingStop a) (tpTrailingStop b) rng79
        (tpStopLossVolMult', rng81) = pickValue (tpStopLossVolMult a) (tpStopLossVolMult b) rng80
        (tpTakeProfitVolMult', rng82) = pickValue (tpTakeProfitVolMult a) (tpTakeProfitVolMult b) rng81
        (tpTrailingStopVolMult', rng83) = pickValue (tpTrailingStopVolMult a) (tpTrailingStopVolMult b) rng82
        (tpRiskPerTrade', rng83a) = pickValue (tpRiskPerTrade a) (tpRiskPerTrade b) rng83
        (tpMaxDrawdown', rng84) = pickValue (tpMaxDrawdown a) (tpMaxDrawdown b) rng83a
        (tpMaxDailyLoss', rng85) = pickValue (tpMaxDailyLoss a) (tpMaxDailyLoss b) rng84
        (tpMaxWeeklyLoss', rng85a) = pickValue (tpMaxWeeklyLoss a) (tpMaxWeeklyLoss b) rng85
        (tpMaxOrderErrors', rng86) = pickValue (tpMaxOrderErrors a) (tpMaxOrderErrors b) rng85a
        (tpKalmanDt', rng87) = pickValue (tpKalmanDt a) (tpKalmanDt b) rng86
        (tpKalmanProcessVar', rng88) = pickValue (tpKalmanProcessVar a) (tpKalmanProcessVar b) rng87
        (tpKalmanMeasurementVar', rng89) = pickValue (tpKalmanMeasurementVar a) (tpKalmanMeasurementVar b) rng88
        (tpKalmanPhysicsBars', rng89p) = pickValue (tpKalmanPhysicsBars a) (tpKalmanPhysicsBars b) rng89
        (tpKalmanPhysicsBacktestRatio', rng89q) =
            pickValue (tpKalmanPhysicsBacktestRatio a) (tpKalmanPhysicsBacktestRatio b) rng89p
        (tpKalmanPhysicsVolumeEwmaAlpha', rng89r) =
            pickValue (tpKalmanPhysicsVolumeEwmaAlpha a) (tpKalmanPhysicsVolumeEwmaAlpha b) rng89q
        (tpKalmanPhysicsVolumeSignalClamp', rng89s) =
            pickValue (tpKalmanPhysicsVolumeSignalClamp a) (tpKalmanPhysicsVolumeSignalClamp b) rng89r
        (tpKalmanPhysicsCloseBiasScale', rng89t) =
            pickValue (tpKalmanPhysicsCloseBiasScale a) (tpKalmanPhysicsCloseBiasScale b) rng89s
        (tpKalmanSensorCorrelationInflation', rng89a) =
            pickValue (tpKalmanSensorCorrelationInflation a) (tpKalmanSensorCorrelationInflation b) rng89t
        (tpKalmanInnovationInflationThreshold', rng89b) =
            pickValue (tpKalmanInnovationInflationThreshold a) (tpKalmanInnovationInflationThreshold b) rng89a
        (tpKalmanInnovationInflationMax', rng89c) =
            pickValue (tpKalmanInnovationInflationMax a) (tpKalmanInnovationInflationMax b) rng89b
        (tpKalmanZMin', rng90) = pickValue (tpKalmanZMin a) (tpKalmanZMin b) rng89c
        (tpKalmanZMax', rng91) = pickValue (tpKalmanZMax a) (tpKalmanZMax b) rng90
        (tpMaxHighVolProb', rng92) = pickValue (tpMaxHighVolProb a) (tpMaxHighVolProb b) rng91
        (tpMaxConformalWidth', rng93) = pickValue (tpMaxConformalWidth a) (tpMaxConformalWidth b) rng92
        (tpMaxQuantileWidth', rng94) = pickValue (tpMaxQuantileWidth a) (tpMaxQuantileWidth b) rng93
        (tpConfirmConformal', rng95) = pickValue (tpConfirmConformal a) (tpConfirmConformal b) rng94
        (tpConfirmQuantiles', rng96) = pickValue (tpConfirmQuantiles a) (tpConfirmQuantiles b) rng95
        (tpConfidenceSizing', rng97) = pickValue (tpConfidenceSizing a) (tpConfidenceSizing b) rng96
        (tpProtectionMinConfidence', rng97a) =
            pickValue (tpProtectionMinConfidence a) (tpProtectionMinConfidence b) rng97
        (tpMinPositionSize', rng98) = pickValue (tpMinPositionSize a) (tpMinPositionSize b) rng97a
        (tpKellyLiteSizing', rng98a) = pickValue (tpKellyLiteSizing a) (tpKellyLiteSizing b) rng98
        (tpKellyLiteFraction', rng98b) = pickValue (tpKellyLiteFraction a) (tpKellyLiteFraction b) rng98a
        (tpKellyLiteFloor', rng98c) = pickValue (tpKellyLiteFloor a) (tpKellyLiteFloor b) rng98b
        (tpKellyLiteCap', rng98d) = pickValue (tpKellyLiteCap a) (tpKellyLiteCap b) rng98c
        (tpPredictors', rng99) = pickValue (tpPredictors a) (tpPredictors b) rng98d
        (tpRouterLookback', rng100) = pickValue (tpRouterLookback a) (tpRouterLookback b) rng99
        (tpRouterRegimeMinBars', rng100a) = pickValue (tpRouterRegimeMinBars a) (tpRouterRegimeMinBars b) rng100
        (tpRouterRegimeMinFraction', rng100b) =
            pickValue (tpRouterRegimeMinFraction a) (tpRouterRegimeMinFraction b) rng100a
        (tpRouterMinScore', rng101) = pickValue (tpRouterMinScore a) (tpRouterMinScore b) rng100b
        (tpFeeFixed', rng102) = pickValue (tpFeeFixed a) (tpFeeFixed b) rng101
        (tpSlippageImpact', rng103) = pickValue (tpSlippageImpact a) (tpSlippageImpact b) rng102
        (tpSlippageImpactPower', rng104) = pickValue (tpSlippageImpactPower a) (tpSlippageImpactPower b) rng103
        (tpSlippageVolMult', rng105) = pickValue (tpSlippageVolMult a) (tpSlippageVolMult b) rng104
        (tpSpreadVolMult', rng106) = pickValue (tpSpreadVolMult a) (tpSpreadVolMult b) rng105
        (tpTakeProfitPartial', rng107) = pickValue (tpTakeProfitPartial a) (tpTakeProfitPartial b) rng106
        (tpMaxTradesPerDay', rng108) = pickValue (tpMaxTradesPerDay a) (tpMaxTradesPerDay b) rng107
        (tpExpectancyLookback', rng109) = pickValue (tpExpectancyLookback a) (tpExpectancyLookback b) rng108
        (tpMinExpectancy', rng110) = pickValue (tpMinExpectancy a) (tpMinExpectancy b) rng109
        (tpLossStreakMax', rng111) = pickValue (tpLossStreakMax a) (tpLossStreakMax b) rng110
        (tpLossStreakCooldownBars', rng112) = pickValue (tpLossStreakCooldownBars a) (tpLossStreakCooldownBars b) rng111
        (tpMaxOpenPositions', rng113) = pickValue (tpMaxOpenPositions a) (tpMaxOpenPositions b) rng112
        (tpMaxGrossExposure', rng114) = pickValue (tpMaxGrossExposure a) (tpMaxGrossExposure b) rng113
        (tpMaxNetExposure', rng115) = pickValue (tpMaxNetExposure a) (tpMaxNetExposure b) rng114
        (tpMaxExposurePerBase', rng116) = pickValue (tpMaxExposurePerBase a) (tpMaxExposurePerBase b) rng115
        (tpMaxOpenPerBase', rng117) = pickValue (tpMaxOpenPerBase a) (tpMaxOpenPerBase b) rng116
        (tpAdaptiveEdgeBufferMax', rng118) = pickValue (tpAdaptiveEdgeBufferMax a) (tpAdaptiveEdgeBufferMax b) rng117
        (tpAdaptiveMinSignalToNoiseMax', rng119) =
            pickValue (tpAdaptiveMinSignalToNoiseMax a) (tpAdaptiveMinSignalToNoiseMax b) rng118
        (tpAdaptiveTrendLookbackMax', rng120) = pickValue (tpAdaptiveTrendLookbackMax a) (tpAdaptiveTrendLookbackMax b) rng119
        (tpAdaptiveKalmanZMinMax', rng121) = pickValue (tpAdaptiveKalmanZMinMax a) (tpAdaptiveKalmanZMinMax b) rng120
        (tpAdaptiveWinRateSlack', rng121a) = pickValue (tpAdaptiveWinRateSlack a) (tpAdaptiveWinRateSlack b) rng121
        (tpAdaptiveProfitFactorSlack', rng121b) = pickValue (tpAdaptiveProfitFactorSlack a) (tpAdaptiveProfitFactorSlack b) rng121a
        (tpAdaptiveFilters', rng122) = pickValue (tpAdaptiveFilters a) (tpAdaptiveFilters b) rng121b
        (tpPerfLookback', rng123) = pickValue (tpPerfLookback a) (tpPerfLookback b) rng122
        (tpPerfMinWinRate', rng124) = pickValue (tpPerfMinWinRate a) (tpPerfMinWinRate b) rng123
        (tpPerfMinProfitFactor', rng125) = pickValue (tpPerfMinProfitFactor a) (tpPerfMinProfitFactor b) rng124
        (tpMetaLabelFilter', rng126) = pickValue (tpMetaLabelFilter a) (tpMetaLabelFilter b) rng125
        (tpMetaLabelMinEdge', rng127) = pickValue (tpMetaLabelMinEdge a) (tpMetaLabelMinEdge b) rng126
        (tpMetaLabelMinConfidence', rng128) = pickValue (tpMetaLabelMinConfidence a) (tpMetaLabelMinConfidence b) rng127
        (tpMetaLabelRequireBand', rng129) = pickValue (tpMetaLabelRequireBand a) (tpMetaLabelRequireBand b) rng128
        (tpRegimeParameterBank', rng130) = pickValue (tpRegimeParameterBank a) (tpRegimeParameterBank b) rng129
        (tpRegimeBankHysteresis', rng131) = pickValue (tpRegimeBankHysteresis a) (tpRegimeBankHysteresis b) rng130
        (tpRegimeTrendOpenMult', rng132) = pickValue (tpRegimeTrendOpenMult a) (tpRegimeTrendOpenMult b) rng131
        (tpRegimeMrOpenMult', rng133) = pickValue (tpRegimeMrOpenMult a) (tpRegimeMrOpenMult b) rng132
        (tpRegimeHighVolOpenMult', rng134) = pickValue (tpRegimeHighVolOpenMult a) (tpRegimeHighVolOpenMult b) rng133
        (tpRegimeTrendSizeMult', rng135) = pickValue (tpRegimeTrendSizeMult a) (tpRegimeTrendSizeMult b) rng134
        (tpRegimeMrSizeMult', rng136) = pickValue (tpRegimeMrSizeMult a) (tpRegimeMrSizeMult b) rng135
        (tpRegimeHighVolSizeMult', rng137) = pickValue (tpRegimeHighVolSizeMult a) (tpRegimeHighVolSizeMult b) rng136
        (tpMultiTimeframeConsensus', rng138) = pickValue (tpMultiTimeframeConsensus a) (tpMultiTimeframeConsensus b) rng137
        (tpMtfFastBars', rng139) = pickValue (tpMtfFastBars a) (tpMtfFastBars b) rng138
        (tpMtfMidBars', rng140) = pickValue (tpMtfMidBars a) (tpMtfMidBars b) rng139
        (tpMtfSlowBars', rng141) = pickValue (tpMtfSlowBars a) (tpMtfSlowBars b) rng140
        (tpMtfMinAgree', rng142) = pickValue (tpMtfMinAgree a) (tpMtfMinAgree b) rng141
        (tpCrossAssetConfirmation', rng143) = pickValue (tpCrossAssetConfirmation a) (tpCrossAssetConfirmation b) rng142
        (tpCrossAssetMinBeta', rng144) = pickValue (tpCrossAssetMinBeta a) (tpCrossAssetMinBeta b) rng143
        (tpCrossAssetMinEdge', rng145) = pickValue (tpCrossAssetMinEdge a) (tpCrossAssetMinEdge b) rng144
        (tpPairsStatArb', rng146) = pickValue (tpPairsStatArb a) (tpPairsStatArb b) rng145
        (tpPairsStatArbLookback', rng147) = pickValue (tpPairsStatArbLookback a) (tpPairsStatArbLookback b) rng146
        (tpPairsStatArbZEntry', rng148) = pickValue (tpPairsStatArbZEntry a) (tpPairsStatArbZEntry b) rng147
        (tpPairsStatArbSizeMult', rng149) = pickValue (tpPairsStatArbSizeMult a) (tpPairsStatArbSizeMult b) rng148
        (tpFundingOiAware', rng150) = pickValue (tpFundingOiAware a) (tpFundingOiAware b) rng149
        (tpFundingOiFundingCap', rng151) = pickValue (tpFundingOiFundingCap a) (tpFundingOiFundingCap b) rng150
        (tpFundingOiVolLookback', rng152) = pickValue (tpFundingOiVolLookback a) (tpFundingOiVolLookback b) rng151
        (tpFundingOiVolCap', rng153) = pickValue (tpFundingOiVolCap a) (tpFundingOiVolCap b) rng152
        (tpFundingOiSizeMult', rng154) = pickValue (tpFundingOiSizeMult a) (tpFundingOiSizeMult b) rng153
        (tpTechnicalParams', rng155) = pickValue (tpTechnicalParams a) (tpTechnicalParams b) rng154
     in ( normalizeTrialParams
            ( a
                { tpPlatform = tpPlatform'
                , tpInterval = tpInterval'
                , tpBars = tpBars'
                , tpMethod = tpMethod'
                , tpBlendWeight = tpBlendWeight'
                , tpRouterScorePnlWeight = tpRouterScorePnlWeight'
                , tpPositioning = tpPositioning'
                , tpNormalization = tpNormalization'
                , tpBaseOpenThreshold = tpBaseOpenThreshold'
                , tpBaseCloseThreshold = tpBaseCloseThreshold'
                , tpMinHoldBars = tpMinHoldBars'
                , tpCooldownBars = tpCooldownBars'
                , tpMaxHoldBars = tpMaxHoldBars'
                , tpMinEdge = tpMinEdge'
                , tpMinSignalToNoise = tpMinSignalToNoise'
                , tpSnrSizeWeight = tpSnrSizeWeight'
                , tpThresholdFactorEnabled = tpThresholdFactorEnabled'
                , tpThresholdFactorAlpha = tpThresholdFactorAlpha'
                , tpThresholdFactorMin = tpThresholdFactorMin'
                , tpThresholdFactorMax = tpThresholdFactorMax'
                , tpThresholdFactorFloor = tpThresholdFactorFloor'
                , tpThresholdFactorEdgeKalWeight = tpThresholdFactorEdgeKalWeight'
                , tpThresholdFactorEdgeLstmWeight = tpThresholdFactorEdgeLstmWeight'
                , tpThresholdFactorKalmanZWeight = tpThresholdFactorKalmanZWeight'
                , tpThresholdFactorHighVolWeight = tpThresholdFactorHighVolWeight'
                , tpThresholdFactorConformalWeight = tpThresholdFactorConformalWeight'
                , tpThresholdFactorQuantileWeight = tpThresholdFactorQuantileWeight'
                , tpThresholdFactorLstmConfWeight = tpThresholdFactorLstmConfWeight'
                , tpThresholdFactorLstmHealthWeight = tpThresholdFactorLstmHealthWeight'
                , tpEdgeBuffer = tpEdgeBuffer'
                , tpCostAwareEdge = tpCostAwareEdge'
                , tpTrendLookback = tpTrendLookback'
                , tpMaxPositionSize = tpMaxPositionSize'
                , tpVolTarget = tpVolTarget'
                , tpVolLookback = tpVolLookback'
                , tpVolEwmaAlpha = tpVolEwmaAlpha'
                , tpVolFloor = tpVolFloor'
                , tpVolScaleMax = tpVolScaleMax'
                , tpMaxVolatility = tpMaxVolatility'
                , tpPeriodsPerYear = tpPeriodsPerYear'
                , tpKalmanMarketTopN = tpKalmanMarketTopN'
                , tpSensorVarianceEwmaAlpha = tpSensorVarianceEwmaAlpha'
                , tpFee = tpFee'
                , tpFundingRate = tpFundingRate'
                , tpFundingBySide = tpFundingBySide'
                , tpFundingOnOpen = tpFundingOnOpen'
                , tpRebalanceBars = tpRebalanceBars'
                , tpRebalanceThreshold = tpRebalanceThreshold'
                , tpRebalanceCostMult = tpRebalanceCostMult'
                , tpRebalanceGlobal = tpRebalanceGlobal'
                , tpRebalanceResetOnSignal = tpRebalanceResetOnSignal'
                , tpEpochs = tpEpochs'
                , tpHiddenSize = tpHiddenSize'
                , tpLearningRate = tpLearningRate'
                , tpLstmAdamBeta1 = tpLstmAdamBeta1'
                , tpLstmAdamBeta2 = tpLstmAdamBeta2'
                , tpLstmAdamEps = tpLstmAdamEps'
                , tpValRatio = tpValRatio'
                , tpPatience = tpPatience'
                , tpWalkForwardFolds = tpWalkForwardFolds'
                , tpWalkForwardEmbargoBars = tpWalkForwardEmbargoBars'
                , tpTuneStressVolMult = tpTuneStressVolMult'
                , tpTuneStressShock = tpTuneStressShock'
                , tpTuneStressWeight = tpTuneStressWeight'
                , tpGradClip = tpGradClip'
                , tpSlippage = tpSlippage'
                , tpSpread = tpSpread'
                , tpIntrabarFill = tpIntrabarFill'
                , tpTriLayer = tpTriLayer'
                , tpTriLayerFastMult = tpTriLayerFastMult'
                , tpTriLayerSlowMult = tpTriLayerSlowMult'
                , tpTriLayerCloudPadding = tpTriLayerCloudPadding'
                , tpTriLayerCloudSlope = tpTriLayerCloudSlope'
                , tpTriLayerCloudWidth = tpTriLayerCloudWidth'
                , tpTriLayerTouchLookback = tpTriLayerTouchLookback'
                , tpTriLayerPriceAction = tpTriLayerPriceAction'
                , tpTriLayerPriceActionBody = tpTriLayerPriceActionBody'
                , tpTriLayerPriceActionWickRatio = tpTriLayerPriceActionWickRatio'
                , tpTriLayerPriceActionOppositeWickMax = tpTriLayerPriceActionOppositeWickMax'
                , tpTriLayerPriceActionBodyTolerance = tpTriLayerPriceActionBodyTolerance'
                , tpTriLayerExitOnSlow = tpTriLayerExitOnSlow'
                , tpKalmanBandLookback = tpKalmanBandLookback'
                , tpKalmanBandStdMult = tpKalmanBandStdMult'
                , tpLstmExitFlipBars = tpLstmExitFlipBars'
                , tpLstmExitFlipGraceBars = tpLstmExitFlipGraceBars'
                , tpLstmExitFlipStrong = tpLstmExitFlipStrong'
                , tpLstmConfidenceSoft = tpLstmConfidenceSoft'
                , tpLstmConfidenceHard = tpLstmConfidenceHard'
                , tpStopLoss = tpStopLoss'
                , tpTakeProfit = tpTakeProfit'
                , tpTrailingStop = tpTrailingStop'
                , tpStopLossVolMult = tpStopLossVolMult'
                , tpTakeProfitVolMult = tpTakeProfitVolMult'
                , tpTrailingStopVolMult = tpTrailingStopVolMult'
                , tpRiskPerTrade = tpRiskPerTrade'
                , tpMaxDrawdown = tpMaxDrawdown'
                , tpMaxDailyLoss = tpMaxDailyLoss'
                , tpMaxWeeklyLoss = tpMaxWeeklyLoss'
                , tpMaxOrderErrors = tpMaxOrderErrors'
                , tpKalmanDt = tpKalmanDt'
                , tpKalmanProcessVar = tpKalmanProcessVar'
                , tpKalmanMeasurementVar = tpKalmanMeasurementVar'
                , tpKalmanPhysicsBars = tpKalmanPhysicsBars'
                , tpKalmanPhysicsBacktestRatio = tpKalmanPhysicsBacktestRatio'
                , tpKalmanPhysicsVolumeEwmaAlpha = tpKalmanPhysicsVolumeEwmaAlpha'
                , tpKalmanPhysicsVolumeSignalClamp = tpKalmanPhysicsVolumeSignalClamp'
                , tpKalmanPhysicsCloseBiasScale = tpKalmanPhysicsCloseBiasScale'
                , tpKalmanSensorCorrelationInflation = tpKalmanSensorCorrelationInflation'
                , tpKalmanInnovationInflationThreshold = tpKalmanInnovationInflationThreshold'
                , tpKalmanInnovationInflationMax = tpKalmanInnovationInflationMax'
                , tpKalmanZMin = tpKalmanZMin'
                , tpKalmanZMax = tpKalmanZMax'
                , tpMaxHighVolProb = tpMaxHighVolProb'
                , tpMaxConformalWidth = tpMaxConformalWidth'
                , tpMaxQuantileWidth = tpMaxQuantileWidth'
                , tpConfirmConformal = tpConfirmConformal'
                , tpConfirmQuantiles = tpConfirmQuantiles'
                , tpConfidenceSizing = tpConfidenceSizing'
                , tpProtectionMinConfidence = tpProtectionMinConfidence'
                , tpMinPositionSize = tpMinPositionSize'
                , tpKellyLiteSizing = tpKellyLiteSizing'
                , tpKellyLiteFraction = tpKellyLiteFraction'
                , tpKellyLiteFloor = tpKellyLiteFloor'
                , tpKellyLiteCap = tpKellyLiteCap'
                , tpPredictors = tpPredictors'
                , tpRouterLookback = tpRouterLookback'
                , tpRouterRegimeMinBars = tpRouterRegimeMinBars'
                , tpRouterRegimeMinFraction = tpRouterRegimeMinFraction'
                , tpRouterMinScore = tpRouterMinScore'
                , tpFeeFixed = tpFeeFixed'
                , tpSlippageImpact = tpSlippageImpact'
                , tpSlippageImpactPower = tpSlippageImpactPower'
                , tpSlippageVolMult = tpSlippageVolMult'
                , tpSpreadVolMult = tpSpreadVolMult'
                , tpTakeProfitPartial = tpTakeProfitPartial'
                , tpMaxTradesPerDay = tpMaxTradesPerDay'
                , tpExpectancyLookback = tpExpectancyLookback'
                , tpMinExpectancy = tpMinExpectancy'
                , tpLossStreakMax = tpLossStreakMax'
                , tpLossStreakCooldownBars = tpLossStreakCooldownBars'
                , tpMaxOpenPositions = tpMaxOpenPositions'
                , tpMaxGrossExposure = tpMaxGrossExposure'
                , tpMaxNetExposure = tpMaxNetExposure'
                , tpMaxExposurePerBase = tpMaxExposurePerBase'
                , tpMaxOpenPerBase = tpMaxOpenPerBase'
                , tpAdaptiveEdgeBufferMax = tpAdaptiveEdgeBufferMax'
                , tpAdaptiveMinSignalToNoiseMax = tpAdaptiveMinSignalToNoiseMax'
                , tpAdaptiveTrendLookbackMax = tpAdaptiveTrendLookbackMax'
                , tpAdaptiveKalmanZMinMax = tpAdaptiveKalmanZMinMax'
                , tpAdaptiveWinRateSlack = tpAdaptiveWinRateSlack'
                , tpAdaptiveProfitFactorSlack = tpAdaptiveProfitFactorSlack'
                , tpAdaptiveFilters = tpAdaptiveFilters'
                , tpPerfLookback = tpPerfLookback'
                , tpPerfMinWinRate = tpPerfMinWinRate'
                , tpPerfMinProfitFactor = tpPerfMinProfitFactor'
                , tpMetaLabelFilter = tpMetaLabelFilter'
                , tpMetaLabelMinEdge = tpMetaLabelMinEdge'
                , tpMetaLabelMinConfidence = tpMetaLabelMinConfidence'
                , tpMetaLabelRequireBand = tpMetaLabelRequireBand'
                , tpRegimeParameterBank = tpRegimeParameterBank'
                , tpRegimeBankHysteresis = tpRegimeBankHysteresis'
                , tpRegimeTrendOpenMult = tpRegimeTrendOpenMult'
                , tpRegimeMrOpenMult = tpRegimeMrOpenMult'
                , tpRegimeHighVolOpenMult = tpRegimeHighVolOpenMult'
                , tpRegimeTrendSizeMult = tpRegimeTrendSizeMult'
                , tpRegimeMrSizeMult = tpRegimeMrSizeMult'
                , tpRegimeHighVolSizeMult = tpRegimeHighVolSizeMult'
                , tpMultiTimeframeConsensus = tpMultiTimeframeConsensus'
                , tpMtfFastBars = tpMtfFastBars'
                , tpMtfMidBars = tpMtfMidBars'
                , tpMtfSlowBars = tpMtfSlowBars'
                , tpMtfMinAgree = tpMtfMinAgree'
                , tpCrossAssetConfirmation = tpCrossAssetConfirmation'
                , tpCrossAssetMinBeta = tpCrossAssetMinBeta'
                , tpCrossAssetMinEdge = tpCrossAssetMinEdge'
                , tpPairsStatArb = tpPairsStatArb'
                , tpPairsStatArbLookback = tpPairsStatArbLookback'
                , tpPairsStatArbZEntry = tpPairsStatArbZEntry'
                , tpPairsStatArbSizeMult = tpPairsStatArbSizeMult'
                , tpFundingOiAware = tpFundingOiAware'
                , tpFundingOiFundingCap = tpFundingOiFundingCap'
                , tpFundingOiVolLookback = tpFundingOiVolLookback'
                , tpFundingOiVolCap = tpFundingOiVolCap'
                , tpFundingOiSizeMult = tpFundingOiSizeMult'
                , tpTechnicalParams = tpTechnicalParams'
                }
            )
        , rng155
        )

clampBarsForPlatform :: Maybe String -> Int -> Int -> Int -> Int
clampBarsForPlatform platform barsMin barsMax value =
    let (lo0, hi0) = if barsMin <= barsMax then (barsMin, barsMax) else (barsMax, barsMin)
        lo = max 2 lo0
        hiBase = max lo hi0
        hi =
            case platform of
                Just p | map toLower (trim p) == "binance" -> min 1000 hiBase
                _ -> hiBase
        lo' = min lo hi
     in clampInt value lo' hi

perturbTrialParams :: Int -> Int -> Double -> Int -> TrialParams -> Rng -> (TrialParams, Rng)
perturbTrialParams barsMin barsMax scaleDouble scaleInt p rng0 =
    let (barsRaw, rng1) = perturbInt (tpBars p) scaleInt rng0
        bars' = clampBarsForPlatform (tpPlatform p) barsMin barsMax barsRaw
        (blendWeight', rng2) = perturbDouble (tpBlendWeight p) scaleDouble rng1
        (routerScorePnlWeight', rng2a) = perturbDouble (tpRouterScorePnlWeight p) scaleDouble rng2
        (openThreshold', rng3) = perturbDouble (tpBaseOpenThreshold p) scaleDouble rng2a
        (closeThreshold', rng4) = perturbDouble (tpBaseCloseThreshold p) scaleDouble rng3
        (minEdge', rng5) = perturbDouble (tpMinEdge p) scaleDouble rng4
        (minSn', rng6) = perturbDouble (tpMinSignalToNoise p) scaleDouble rng5
        (snrSizeWeight', rng6a) = perturbDouble (tpSnrSizeWeight p) scaleDouble rng6
        (edgeBuffer', rng7) = perturbDouble (tpEdgeBuffer p) scaleDouble rng6a
        (trendLookback', rng8) = perturbInt (tpTrendLookback p) scaleInt rng7
        (maxPositionSize', rng9) = perturbDouble (tpMaxPositionSize p) scaleDouble rng8
        (volTarget', rng10) = perturbMaybeDouble (tpVolTarget p) scaleDouble rng9
        (volLookback', rng11) = perturbInt (tpVolLookback p) scaleInt rng10
        (volEwma', rng12) = perturbMaybeDouble (tpVolEwmaAlpha p) scaleDouble rng11
        (volFloor', rng13) = perturbDouble (tpVolFloor p) scaleDouble rng12
        (volScaleMax', rng14) = perturbDouble (tpVolScaleMax p) scaleDouble rng13
        (maxVolatility', rng15) = perturbMaybeDouble (tpMaxVolatility p) scaleDouble rng14
        (kalmanBandLookback', rng16) = perturbInt (tpKalmanBandLookback p) scaleInt rng15
        (kalmanBandStd', rng17) = perturbDouble (tpKalmanBandStdMult p) scaleDouble rng16
        (stopLoss', rng18) = perturbMaybeDouble (tpStopLoss p) scaleDouble rng17
        (takeProfit', rng19) = perturbMaybeDouble (tpTakeProfit p) scaleDouble rng18
        (trailingStop', rng20) = perturbMaybeDouble (tpTrailingStop p) scaleDouble rng19
        (stopLossVolMult', rng21) = perturbMaybeDouble (tpStopLossVolMult p) scaleDouble rng20
        (takeProfitVolMult', rng22) = perturbMaybeDouble (tpTakeProfitVolMult p) scaleDouble rng21
        (trailingStopVolMult', rng23) = perturbMaybeDouble (tpTrailingStopVolMult p) scaleDouble rng22
        (riskPerTrade', rng23a) = perturbMaybeDouble (tpRiskPerTrade p) scaleDouble rng23
        (maxDrawdown', rng23b) = perturbMaybeDouble (tpMaxDrawdown p) scaleDouble rng23a
        (maxDailyLoss', rng23c) = perturbMaybeDouble (tpMaxDailyLoss p) scaleDouble rng23b
        (maxWeeklyLoss', rng23d) = perturbMaybeDouble (tpMaxWeeklyLoss p) scaleDouble rng23c
        (kalmanDt', rng24) = perturbDouble (tpKalmanDt p) scaleDouble rng23d
        (kalmanProcessVar', rng25) = perturbDouble (tpKalmanProcessVar p) scaleDouble rng24
        (kalmanMeasurementVar', rng26) = perturbDouble (tpKalmanMeasurementVar p) scaleDouble rng25
        (kalmanPhysicsBars', rng26a) = perturbInt (tpKalmanPhysicsBars p) scaleInt rng26
        (kalmanPhysicsBacktestRatio', rng26b) = perturbDouble (tpKalmanPhysicsBacktestRatio p) scaleDouble rng26a
        (kalmanPhysicsVolumeEwmaAlpha', rng26c) = perturbDouble (tpKalmanPhysicsVolumeEwmaAlpha p) scaleDouble rng26b
        (kalmanPhysicsVolumeSignalClamp', rng26d) = perturbDouble (tpKalmanPhysicsVolumeSignalClamp p) scaleDouble rng26c
        (kalmanPhysicsCloseBiasScale', rng26e) = perturbDoubleSigned (tpKalmanPhysicsCloseBiasScale p) scaleDouble rng26d
        (kalmanZMin', rng27) = perturbDouble (tpKalmanZMin p) scaleDouble rng26e
        (kalmanZMax', rng28) = perturbDouble (tpKalmanZMax p) scaleDouble rng27
        (fundingRate', rng29) = perturbDoubleSigned (tpFundingRate p) scaleDouble rng28
        (rebalanceBars', rng30) = perturbInt (tpRebalanceBars p) scaleInt rng29
        (rebalanceThreshold', rng31) = perturbDouble (tpRebalanceThreshold p) scaleDouble rng30
        (rebalanceCostMult', rng31a) = perturbDouble (tpRebalanceCostMult p) scaleDouble rng31
        (learningRate', rng32) = perturbDouble (tpLearningRate p) scaleDouble rng31a
        (lstmAdamBeta1', rng32a) = perturbDouble (tpLstmAdamBeta1 p) scaleDouble rng32
        (lstmAdamBeta2', rng32b) = perturbDouble (tpLstmAdamBeta2 p) scaleDouble rng32a
        (lstmAdamEps', rng32c) = perturbDouble (tpLstmAdamEps p) scaleDouble rng32b
        (thresholdFactorAlpha', rng33) = perturbDouble (tpThresholdFactorAlpha p) scaleDouble rng32c
        (thresholdFactorMin', rng34) = perturbDouble (tpThresholdFactorMin p) scaleDouble rng33
        (thresholdFactorMax', rng35) = perturbDouble (tpThresholdFactorMax p) scaleDouble rng34
        (thresholdFactorFloor', rng36) = perturbDouble (tpThresholdFactorFloor p) scaleDouble rng35
        (thresholdFactorEdgeKalWeight', rng37) = perturbDoubleSigned (tpThresholdFactorEdgeKalWeight p) scaleDouble rng36
        (thresholdFactorEdgeLstmWeight', rng38) = perturbDoubleSigned (tpThresholdFactorEdgeLstmWeight p) scaleDouble rng37
        (thresholdFactorKalmanZWeight', rng39) = perturbDoubleSigned (tpThresholdFactorKalmanZWeight p) scaleDouble rng38
        (thresholdFactorHighVolWeight', rng40) = perturbDoubleSigned (tpThresholdFactorHighVolWeight p) scaleDouble rng39
        (thresholdFactorConformalWeight', rng41) = perturbDoubleSigned (tpThresholdFactorConformalWeight p) scaleDouble rng40
        (thresholdFactorQuantileWeight', rng42) = perturbDoubleSigned (tpThresholdFactorQuantileWeight p) scaleDouble rng41
        (thresholdFactorLstmConfWeight', rng43) = perturbDoubleSigned (tpThresholdFactorLstmConfWeight p) scaleDouble rng42
        (thresholdFactorLstmHealthWeight', rng44) = perturbDoubleSigned (tpThresholdFactorLstmHealthWeight p) scaleDouble rng43
        (routerLookback', rng45) = perturbInt (tpRouterLookback p) scaleInt rng44
        (routerRegimeMinBars', rng45a) = perturbInt (tpRouterRegimeMinBars p) scaleInt rng45
        (routerRegimeMinFraction', rng45b) = perturbDouble (tpRouterRegimeMinFraction p) scaleDouble rng45a
        (routerMinScore', rng46) = perturbDouble (tpRouterMinScore p) scaleDouble rng45b
        (feeFixed', rng47) = perturbDouble (tpFeeFixed p) scaleDouble rng46
        (slippageImpact', rng48) = perturbDouble (tpSlippageImpact p) scaleDouble rng47
        (slippageImpactPower', rng49) = perturbDouble (tpSlippageImpactPower p) scaleDouble rng48
        (slippageVolMult', rng50) = perturbDouble (tpSlippageVolMult p) scaleDouble rng49
        (spreadVolMult', rng51) = perturbDouble (tpSpreadVolMult p) scaleDouble rng50
        (takeProfitPartial', rng52) = perturbMaybeDouble (tpTakeProfitPartial p) scaleDouble rng51
        (maxTradesPerDay', rng53) = perturbMaybeInt (tpMaxTradesPerDay p) scaleInt rng52
        (expectancyLookback', rng54) = perturbInt (tpExpectancyLookback p) scaleInt rng53
        (minExpectancy', rng55) = perturbMaybeDouble (tpMinExpectancy p) scaleDouble rng54
        (lossStreakMax', rng56) = perturbMaybeInt (tpLossStreakMax p) scaleInt rng55
        (lossStreakCooldownBars', rng57) = perturbMaybeInt (tpLossStreakCooldownBars p) scaleInt rng56
        (maxOpenPositions', rng58) = perturbMaybeInt (tpMaxOpenPositions p) scaleInt rng57
        (maxGrossExposure', rng59) = perturbMaybeDouble (tpMaxGrossExposure p) scaleDouble rng58
        (maxNetExposure', rng60) = perturbMaybeDouble (tpMaxNetExposure p) scaleDouble rng59
        (maxExposurePerBase', rng61) = perturbMaybeDouble (tpMaxExposurePerBase p) scaleDouble rng60
        (maxOpenPerBase', rng62) = perturbMaybeInt (tpMaxOpenPerBase p) scaleInt rng61
        (adaptiveEdgeBufferMax', rng63) = perturbDouble (tpAdaptiveEdgeBufferMax p) scaleDouble rng62
        (adaptiveMinSignalToNoiseMax', rng64) = perturbDouble (tpAdaptiveMinSignalToNoiseMax p) scaleDouble rng63
        (adaptiveTrendLookbackMax', rng65) = perturbInt (tpAdaptiveTrendLookbackMax p) scaleInt rng64
        (adaptiveKalmanZMinMax', rng66) = perturbDouble (tpAdaptiveKalmanZMinMax p) scaleDouble rng65
        (adaptiveWinRateSlack', rng66a) = perturbDouble (tpAdaptiveWinRateSlack p) scaleDouble rng66
        (adaptiveProfitFactorSlack', rng66b) = perturbDouble (tpAdaptiveProfitFactorSlack p) scaleDouble rng66a
        (perfLookback', rng67) = perturbInt (tpPerfLookback p) scaleInt rng66b
        (perfMinWinRate', rng68) = perturbMaybeDouble (tpPerfMinWinRate p) scaleDouble rng67
        (perfMinProfitFactor', rng69) = perturbMaybeDouble (tpPerfMinProfitFactor p) scaleDouble rng68
        (metaLabelMinEdge', rng70) = perturbDouble (tpMetaLabelMinEdge p) scaleDouble rng69
        (metaLabelMinConfidence', rng71) = perturbDouble (tpMetaLabelMinConfidence p) scaleDouble rng70
        (regimeBankHysteresis', rng72) = perturbDouble (tpRegimeBankHysteresis p) scaleDouble rng71
        (regimeTrendOpenMult', rng73) = perturbDouble (tpRegimeTrendOpenMult p) scaleDouble rng72
        (regimeMrOpenMult', rng74) = perturbDouble (tpRegimeMrOpenMult p) scaleDouble rng73
        (regimeHighVolOpenMult', rng75) = perturbDouble (tpRegimeHighVolOpenMult p) scaleDouble rng74
        (regimeTrendSizeMult', rng76) = perturbDouble (tpRegimeTrendSizeMult p) scaleDouble rng75
        (regimeMrSizeMult', rng77) = perturbDouble (tpRegimeMrSizeMult p) scaleDouble rng76
        (regimeHighVolSizeMult', rng78) = perturbDouble (tpRegimeHighVolSizeMult p) scaleDouble rng77
        (mtfFastBars', rng79) = perturbInt (tpMtfFastBars p) scaleInt rng78
        (mtfMidBars', rng80) = perturbInt (tpMtfMidBars p) scaleInt rng79
        (mtfSlowBars', rng81) = perturbInt (tpMtfSlowBars p) scaleInt rng80
        (mtfMinAgree', rng82) = perturbInt (tpMtfMinAgree p) scaleInt rng81
        (crossAssetMinBeta', rng83) = perturbDouble (tpCrossAssetMinBeta p) scaleDouble rng82
        (crossAssetMinEdge', rng84) = perturbDouble (tpCrossAssetMinEdge p) scaleDouble rng83
        (pairsStatArbLookback', rng85) = perturbInt (tpPairsStatArbLookback p) scaleInt rng84
        (pairsStatArbZEntry', rng86) = perturbDouble (tpPairsStatArbZEntry p) scaleDouble rng85
        (pairsStatArbSizeMult', rng87) = perturbDouble (tpPairsStatArbSizeMult p) scaleDouble rng86
        (fundingOiFundingCap', rng88) = perturbMaybeDouble (tpFundingOiFundingCap p) scaleDouble rng87
        (fundingOiVolLookback', rng89) = perturbInt (tpFundingOiVolLookback p) scaleInt rng88
        (fundingOiVolCap', rng90) = perturbMaybeDouble (tpFundingOiVolCap p) scaleDouble rng89
        (fundingOiSizeMult', rng91) = perturbDouble (tpFundingOiSizeMult p) scaleDouble rng90
        (kellyLiteFraction', rng91a) = perturbDouble (tpKellyLiteFraction p) scaleDouble rng91
        (kellyLiteFloor', rng91b) = perturbDouble (tpKellyLiteFloor p) scaleDouble rng91a
        (kellyLiteCap', rng91c) = perturbDouble (tpKellyLiteCap p) scaleDouble rng91b
        (triLayerPriceActionWickRatio', rng91d) = perturbDouble (tpTriLayerPriceActionWickRatio p) scaleDouble rng91c
        (triLayerPriceActionOppositeWickMax', rng91e) = perturbDouble (tpTriLayerPriceActionOppositeWickMax p) scaleDouble rng91d
        (triLayerPriceActionBodyTolerance', rng91f) = perturbDouble (tpTriLayerPriceActionBodyTolerance p) scaleDouble rng91e
     in ( normalizeTrialParams
            ( p
                { tpBars = bars'
                , tpBlendWeight = clamp blendWeight' 0 1
                , tpRouterScorePnlWeight = clamp routerScorePnlWeight' 0 1
                , tpBaseOpenThreshold = openThreshold'
                , tpBaseCloseThreshold = closeThreshold'
                , tpMinEdge = minEdge'
                , tpMinSignalToNoise = minSn'
                , tpSnrSizeWeight = clamp snrSizeWeight' 0 1
                , tpThresholdFactorAlpha = thresholdFactorAlpha'
                , tpThresholdFactorMin = thresholdFactorMin'
                , tpThresholdFactorMax = thresholdFactorMax'
                , tpThresholdFactorFloor = thresholdFactorFloor'
                , tpThresholdFactorEdgeKalWeight = thresholdFactorEdgeKalWeight'
                , tpThresholdFactorEdgeLstmWeight = thresholdFactorEdgeLstmWeight'
                , tpThresholdFactorKalmanZWeight = thresholdFactorKalmanZWeight'
                , tpThresholdFactorHighVolWeight = thresholdFactorHighVolWeight'
                , tpThresholdFactorConformalWeight = thresholdFactorConformalWeight'
                , tpThresholdFactorQuantileWeight = thresholdFactorQuantileWeight'
                , tpThresholdFactorLstmConfWeight = thresholdFactorLstmConfWeight'
                , tpThresholdFactorLstmHealthWeight = thresholdFactorLstmHealthWeight'
                , tpEdgeBuffer = edgeBuffer'
                , tpTrendLookback = trendLookback'
                , tpMaxPositionSize = maxPositionSize'
                , tpVolTarget = volTarget'
                , tpVolLookback = volLookback'
                , tpVolEwmaAlpha = volEwma'
                , tpVolFloor = volFloor'
                , tpVolScaleMax = volScaleMax'
                , tpMaxVolatility = maxVolatility'
                , tpKalmanBandLookback = kalmanBandLookback'
                , tpKalmanBandStdMult = kalmanBandStd'
                , tpStopLoss = stopLoss'
                , tpTakeProfit = takeProfit'
                , tpTrailingStop = trailingStop'
                , tpStopLossVolMult = stopLossVolMult'
                , tpTakeProfitVolMult = takeProfitVolMult'
                , tpTrailingStopVolMult = trailingStopVolMult'
                , tpRiskPerTrade = riskPerTrade'
                , tpMaxDrawdown = maxDrawdown'
                , tpMaxDailyLoss = maxDailyLoss'
                , tpMaxWeeklyLoss = maxWeeklyLoss'
                , tpKalmanDt = kalmanDt'
                , tpKalmanProcessVar = kalmanProcessVar'
                , tpKalmanMeasurementVar = kalmanMeasurementVar'
                , tpKalmanPhysicsBars = max 4 kalmanPhysicsBars'
                , tpKalmanPhysicsBacktestRatio = clamp kalmanPhysicsBacktestRatio' 0.000001 0.999999
                , tpKalmanPhysicsVolumeEwmaAlpha = clamp kalmanPhysicsVolumeEwmaAlpha' 0 1
                , tpKalmanPhysicsVolumeSignalClamp = max 0 kalmanPhysicsVolumeSignalClamp'
                , tpKalmanPhysicsCloseBiasScale = kalmanPhysicsCloseBiasScale'
                , tpKalmanZMin = kalmanZMin'
                , tpKalmanZMax = kalmanZMax'
                , tpFundingRate = fundingRate'
                , tpRebalanceBars = rebalanceBars'
                , tpRebalanceThreshold = rebalanceThreshold'
                , tpRebalanceCostMult = rebalanceCostMult'
                , tpLearningRate = learningRate'
                , tpLstmAdamBeta1 = lstmAdamBeta1'
                , tpLstmAdamBeta2 = lstmAdamBeta2'
                , tpLstmAdamEps = lstmAdamEps'
                , tpRouterLookback = routerLookback'
                , tpRouterRegimeMinBars = max 0 routerRegimeMinBars'
                , tpRouterRegimeMinFraction = clamp routerRegimeMinFraction' 0 1
                , tpRouterMinScore = routerMinScore'
                , tpFeeFixed = feeFixed'
                , tpSlippageImpact = slippageImpact'
                , tpSlippageImpactPower = slippageImpactPower'
                , tpSlippageVolMult = slippageVolMult'
                , tpSpreadVolMult = spreadVolMult'
                , tpTakeProfitPartial = takeProfitPartial'
                , tpTriLayerPriceActionWickRatio = max 0 triLayerPriceActionWickRatio'
                , tpTriLayerPriceActionOppositeWickMax = max 0 triLayerPriceActionOppositeWickMax'
                , tpTriLayerPriceActionBodyTolerance = max 0 triLayerPriceActionBodyTolerance'
                , tpKellyLiteFraction = max 0 kellyLiteFraction'
                , tpKellyLiteFloor = max 0 kellyLiteFloor'
                , tpKellyLiteCap = max (max 0 kellyLiteFloor') kellyLiteCap'
                , tpMaxTradesPerDay = maxTradesPerDay'
                , tpExpectancyLookback = expectancyLookback'
                , tpMinExpectancy = minExpectancy'
                , tpLossStreakMax = lossStreakMax'
                , tpLossStreakCooldownBars = lossStreakCooldownBars'
                , tpMaxOpenPositions = maxOpenPositions'
                , tpMaxGrossExposure = maxGrossExposure'
                , tpMaxNetExposure = maxNetExposure'
                , tpMaxExposurePerBase = maxExposurePerBase'
                , tpMaxOpenPerBase = maxOpenPerBase'
                , tpAdaptiveEdgeBufferMax = adaptiveEdgeBufferMax'
                , tpAdaptiveMinSignalToNoiseMax = adaptiveMinSignalToNoiseMax'
                , tpAdaptiveTrendLookbackMax = adaptiveTrendLookbackMax'
                , tpAdaptiveKalmanZMinMax = adaptiveKalmanZMinMax'
                , tpAdaptiveWinRateSlack = max 0 adaptiveWinRateSlack'
                , tpAdaptiveProfitFactorSlack = max 0 adaptiveProfitFactorSlack'
                , tpPerfLookback = max 0 perfLookback'
                , tpPerfMinWinRate = fmap (\v -> clamp v 0 1) perfMinWinRate'
                , tpPerfMinProfitFactor = fmap (max 0) perfMinProfitFactor'
                , tpMetaLabelMinEdge = max 0 metaLabelMinEdge'
                , tpMetaLabelMinConfidence = clamp metaLabelMinConfidence' 0 1
                , tpRegimeBankHysteresis = clamp regimeBankHysteresis' 0 1
                , tpRegimeTrendOpenMult = max 0 regimeTrendOpenMult'
                , tpRegimeMrOpenMult = max 0 regimeMrOpenMult'
                , tpRegimeHighVolOpenMult = max 0 regimeHighVolOpenMult'
                , tpRegimeTrendSizeMult = max 0 regimeTrendSizeMult'
                , tpRegimeMrSizeMult = max 0 regimeMrSizeMult'
                , tpRegimeHighVolSizeMult = max 0 regimeHighVolSizeMult'
                , tpMtfFastBars = max 1 mtfFastBars'
                , tpMtfMidBars = max 1 mtfMidBars'
                , tpMtfSlowBars = max 1 mtfSlowBars'
                , tpMtfMinAgree = clampInt mtfMinAgree' 1 3
                , tpCrossAssetMinBeta = max 0 crossAssetMinBeta'
                , tpCrossAssetMinEdge = max 0 crossAssetMinEdge'
                , tpPairsStatArbLookback = max 2 pairsStatArbLookback'
                , tpPairsStatArbZEntry = max 1e-12 pairsStatArbZEntry'
                , tpPairsStatArbSizeMult = clamp pairsStatArbSizeMult' 0 1
                , tpFundingOiFundingCap = fmap (max 0) fundingOiFundingCap'
                , tpFundingOiVolLookback = max 2 fundingOiVolLookback'
                , tpFundingOiVolCap = fmap (max 0) fundingOiVolCap'
                , tpFundingOiSizeMult = clamp fundingOiSizeMult' 0 1
                }
            )
        , rng91f
        )

samplePriorParams ::
    Double ->
    [String] ->
    Int ->
    Int ->
    Double ->
    Int ->
    Double ->
    Double ->
    [PriorTrial] ->
    TrialParams ->
    Rng ->
    (TrialParams, Rng, Maybe String)
samplePriorParams priorProb allowedIntervals barsMin barsMax scaleDouble scaleInt priorMethodProb priorRankBias priors base rng0
    | null priors = (base, rng0, Nothing)
    | priorProb <= 0 = (base, rng0, Nothing)
    | otherwise =
        let (r, rng1) = nextDouble rng0
         in if r >= clamp priorProb 0 1
                then (base, rng1, Nothing)
                else
                    let (methodR, rng1a) = nextDouble rng1
                        usePriorMethod = methodR < clamp priorMethodProb 0 1
                        pool =
                            if usePriorMethod
                                then priorMethodSteeringPool base priors
                                else priorPoolForBase base priors
                     in case pool of
                            [] -> (base, rng1a, Nothing)
                            pool ->
                                let (picked, rng2) = pickRankBiasedPrior priorRankBias pool rng1a
                                 in case picked of
                                        Nothing -> (base, rng2, Nothing)
                                        Just prior ->
                                            let overlaid =
                                                    applyPriorMethodIfEnabled usePriorMethod prior $
                                                        applyPriorOverlay allowedIntervals prior base
                                                (params, rng3) = perturbTrialParams barsMin barsMax scaleDouble scaleInt overlaid rng2
                                                origin =
                                                    if usePriorMethod
                                                        then "prior-method-sample"
                                                        else "prior-sample"
                                             in (params, rng3, Just origin)

pickRankBiasedPrior :: Double -> [PriorTrial] -> Rng -> (Maybe PriorTrial, Rng)
pickRankBiasedPrior bias pool rng
    | null pool = (Nothing, rng)
    | bias <= 1 = nextChoice pool rng
    | otherwise =
        let (u0, rng1) = nextDouble rng
            n = length pool
            u = clamp u0 0 0.999999999999
            idx = min (n - 1) (floor (fromIntegral n * (u ** bias)))
         in (listToMaybe (drop idx pool), rng1)

priorPoolForBase :: TrialParams -> [PriorTrial] -> [PriorTrial]
priorPoolForBase base priors =
    firstNonEmpty
        [ filter (\p -> samePlatform p && sameInterval p && sameMethod p) priors
        , filter (\p -> samePlatform p && sameMethod p) priors
        , filter sameMethod priors
        ]
  where
    samePlatform prior =
        case ptPlatform prior of
            Nothing -> True
            Just platform -> isNothing (tpPlatform base) || tpPlatform base == Just platform
    sameInterval prior =
        case ptInterval prior of
            Nothing -> True
            Just interval -> interval == tpInterval base
    sameMethod prior =
        case ptMethod prior of
            Nothing -> True
            Just method -> method == map toLower (tpMethod base)
    firstNonEmpty pools =
        case filter (not . null) pools of
            pool : _ -> pool
            [] -> []

priorMethodSteeringPool :: TrialParams -> [PriorTrial] -> [PriorTrial]
priorMethodSteeringPool base priors =
    firstNonEmpty
        [ filter (\p -> samePlatform p && sameInterval p && priorMethodEnabled p) priors
        , filter (\p -> samePlatform p && priorMethodEnabled p) priors
        , filter priorMethodEnabled priors
        ]
  where
    samePlatform prior =
        case ptPlatform prior of
            Nothing -> True
            Just platform -> isNothing (tpPlatform base) || tpPlatform base == Just platform
    sameInterval prior =
        case ptInterval prior of
            Nothing -> True
            Just interval -> interval == tpInterval base
    firstNonEmpty pools =
        case filter (not . null) pools of
            pool : _ -> pool
            [] -> []

priorMethodEnabled :: PriorTrial -> Bool
priorMethodEnabled prior =
    isJust (priorMethodCode prior)

priorMethodCode :: PriorTrial -> Maybe String
priorMethodCode prior = do
    method <- ptMethod prior
    either (const Nothing) (Just . methodCode) (parseMethod method)

applyPriorMethodIfEnabled :: Bool -> PriorTrial -> TrialParams -> TrialParams
applyPriorMethodIfEnabled usePriorMethod prior params =
    if usePriorMethod
        then maybe params (\method -> params{tpMethod = method}) (priorMethodCode prior)
        else params

applyPriorOverlay :: [String] -> PriorTrial -> TrialParams -> TrialParams
applyPriorOverlay allowedIntervals prior base =
    let params = ptParams prior
        interval' =
            case stringField ["interval"] params of
                Just interval | interval `elem` allowedIntervals -> interval
                _ -> tpInterval base
        priorMinEdge =
            case doubleField ["minEdge"] params of
                Just v -> Just (max venueMinEdgeFloor v)
                Nothing -> Nothing
        overlaid =
            base
                { tpInterval = interval'
                , tpBars = fromMaybe (tpBars base) (intField ["bars"] params)
                , tpBlendWeight = fromMaybe (tpBlendWeight base) (doubleField ["blendWeight"] params)
                , tpRouterScorePnlWeight = fromMaybe (tpRouterScorePnlWeight base) (doubleField ["routerScorePnlWeight"] params)
                , tpPositioning = fromMaybe (tpPositioning base) (stringField ["positioning"] params)
                , tpNormalization = fromMaybe (tpNormalization base) (stringField ["normalization"] params)
                , tpBaseOpenThreshold = fromMaybe (tpBaseOpenThreshold base) (doubleField ["baseOpenThreshold", "openThreshold"] params)
                , tpBaseCloseThreshold = fromMaybe (tpBaseCloseThreshold base) (doubleField ["baseCloseThreshold", "closeThreshold"] params)
                , tpMinHoldBars = fromMaybe (tpMinHoldBars base) (intField ["minHoldBars"] params)
                , tpCooldownBars = fromMaybe (tpCooldownBars base) (intField ["cooldownBars"] params)
                , tpMaxHoldBars = fromMaybe (tpMaxHoldBars base) (optionalIntField ["maxHoldBars"] params)
                , tpMinEdge = fromMaybe (tpMinEdge base) priorMinEdge
                , tpMinSignalToNoise = fromMaybe (tpMinSignalToNoise base) (doubleField ["minSignalToNoise"] params)
                , tpSnrSizeWeight = fromMaybe (tpSnrSizeWeight base) (doubleField ["snrSizeWeight"] params)
                , tpThresholdFactorEnabled = fromMaybe (tpThresholdFactorEnabled base) (boolField "thresholdFactorEnabled" params)
                , tpThresholdFactorAlpha = fromMaybe (tpThresholdFactorAlpha base) (doubleField ["thresholdFactorAlpha"] params)
                , tpThresholdFactorMin = fromMaybe (tpThresholdFactorMin base) (doubleField ["thresholdFactorMin"] params)
                , tpThresholdFactorMax = fromMaybe (tpThresholdFactorMax base) (doubleField ["thresholdFactorMax"] params)
                , tpThresholdFactorFloor = fromMaybe (tpThresholdFactorFloor base) (doubleField ["thresholdFactorFloor"] params)
                , tpThresholdFactorEdgeKalWeight = fromMaybe (tpThresholdFactorEdgeKalWeight base) (doubleField ["thresholdFactorEdgeKalWeight"] params)
                , tpThresholdFactorEdgeLstmWeight = fromMaybe (tpThresholdFactorEdgeLstmWeight base) (doubleField ["thresholdFactorEdgeLstmWeight"] params)
                , tpThresholdFactorKalmanZWeight = fromMaybe (tpThresholdFactorKalmanZWeight base) (doubleField ["thresholdFactorKalmanZWeight"] params)
                , tpThresholdFactorHighVolWeight = fromMaybe (tpThresholdFactorHighVolWeight base) (doubleField ["thresholdFactorHighVolWeight"] params)
                , tpThresholdFactorConformalWeight = fromMaybe (tpThresholdFactorConformalWeight base) (doubleField ["thresholdFactorConformalWeight"] params)
                , tpThresholdFactorQuantileWeight = fromMaybe (tpThresholdFactorQuantileWeight base) (doubleField ["thresholdFactorQuantileWeight"] params)
                , tpThresholdFactorLstmConfWeight = fromMaybe (tpThresholdFactorLstmConfWeight base) (doubleField ["thresholdFactorLstmConfWeight"] params)
                , tpThresholdFactorLstmHealthWeight = fromMaybe (tpThresholdFactorLstmHealthWeight base) (doubleField ["thresholdFactorLstmHealthWeight"] params)
                , tpEdgeBuffer = fromMaybe (tpEdgeBuffer base) (doubleField ["edgeBuffer"] params)
                , tpCostAwareEdge = fromMaybe (tpCostAwareEdge base) (boolField "costAwareEdge" params)
                , tpTrendLookback = fromMaybe (tpTrendLookback base) (intField ["trendLookback"] params)
                , tpVolTarget = fromMaybe (tpVolTarget base) (optionalDoubleField ["volTarget"] params)
                , tpVolLookback = fromMaybe (tpVolLookback base) (intField ["volLookback"] params)
                , tpVolEwmaAlpha = fromMaybe (tpVolEwmaAlpha base) (optionalDoubleField ["volEwmaAlpha"] params)
                , tpVolFloor = fromMaybe (tpVolFloor base) (doubleField ["volFloor"] params)
                , tpVolScaleMax = fromMaybe (tpVolScaleMax base) (doubleField ["volScaleMax"] params)
                , tpMaxVolatility = fromMaybe (tpMaxVolatility base) (optionalDoubleField ["maxVolatility"] params)
                , tpPeriodsPerYear = fromMaybe (tpPeriodsPerYear base) (optionalDoubleField ["periodsPerYear"] params)
                , tpKalmanMarketTopN = fromMaybe (tpKalmanMarketTopN base) (intField ["kalmanMarketTopN"] params)
                , tpSensorVarianceEwmaAlpha = fromMaybe (tpSensorVarianceEwmaAlpha base) (doubleField ["sensorVarianceEwmaAlpha"] params)
                , tpEpochs = fromMaybe (tpEpochs base) (intField ["epochs"] params)
                , tpHiddenSize = fromMaybe (tpHiddenSize base) (intField ["hiddenSize"] params)
                , tpLearningRate = fromMaybe (tpLearningRate base) (doubleField ["learningRate"] params)
                , tpLstmAdamBeta1 = fromMaybe (tpLstmAdamBeta1 base) (doubleField ["lstmAdamBeta1"] params)
                , tpLstmAdamBeta2 = fromMaybe (tpLstmAdamBeta2 base) (doubleField ["lstmAdamBeta2"] params)
                , tpLstmAdamEps = fromMaybe (tpLstmAdamEps base) (doubleField ["lstmAdamEps"] params)
                , tpValRatio = fromMaybe (tpValRatio base) (doubleField ["valRatio"] params)
                , tpPatience = fromMaybe (tpPatience base) (intField ["patience"] params)
                , tpWalkForwardFolds = fromMaybe (tpWalkForwardFolds base) (intField ["walkForwardFolds"] params)
                , tpWalkForwardEmbargoBars = fromMaybe (tpWalkForwardEmbargoBars base) (intField ["walkForwardEmbargoBars"] params)
                , tpTuneStressVolMult = fromMaybe (tpTuneStressVolMult base) (doubleField ["tuneStressVolMult"] params)
                , tpTuneStressShock = fromMaybe (tpTuneStressShock base) (doubleField ["tuneStressShock"] params)
                , tpTuneStressWeight = fromMaybe (tpTuneStressWeight base) (doubleField ["tuneStressWeight"] params)
                , tpGradClip = fromMaybe (tpGradClip base) (optionalDoubleField ["gradClip"] params)
                , tpIntrabarFill = fromMaybe (tpIntrabarFill base) (stringField ["intrabarFill"] params)
                , tpKalmanBandLookback = fromMaybe (tpKalmanBandLookback base) (intField ["kalmanBandLookback"] params)
                , tpKalmanBandStdMult = fromMaybe (tpKalmanBandStdMult base) (doubleField ["kalmanBandStdMult"] params)
                , tpLstmExitFlipBars = fromMaybe (tpLstmExitFlipBars base) (intField ["lstmExitFlipBars"] params)
                , tpLstmExitFlipGraceBars = fromMaybe (tpLstmExitFlipGraceBars base) (intField ["lstmExitFlipGraceBars"] params)
                , tpLstmExitFlipStrong = fromMaybe (tpLstmExitFlipStrong base) (boolField "lstmExitFlipStrong" params)
                , tpLstmConfidenceSoft = fromMaybe (tpLstmConfidenceSoft base) (doubleField ["lstmConfidenceSoft"] params)
                , tpLstmConfidenceHard = fromMaybe (tpLstmConfidenceHard base) (doubleField ["lstmConfidenceHard"] params)
                , tpStopLoss = fromMaybe (tpStopLoss base) (optionalDoubleField ["stopLoss"] params)
                , tpTakeProfit = fromMaybe (tpTakeProfit base) (optionalDoubleField ["takeProfit"] params)
                , tpTrailingStop = fromMaybe (tpTrailingStop base) (optionalDoubleField ["trailingStop"] params)
                , tpStopLossVolMult = fromMaybe (tpStopLossVolMult base) (optionalDoubleField ["stopLossVolMult"] params)
                , tpTakeProfitVolMult = fromMaybe (tpTakeProfitVolMult base) (optionalDoubleField ["takeProfitVolMult"] params)
                , tpTrailingStopVolMult = fromMaybe (tpTrailingStopVolMult base) (optionalDoubleField ["trailingStopVolMult"] params)
                , tpKalmanDt = fromMaybe (tpKalmanDt base) (doubleField ["kalmanDt"] params)
                , tpKalmanProcessVar = fromMaybe (tpKalmanProcessVar base) (doubleField ["kalmanProcessVar"] params)
                , tpKalmanMeasurementVar = fromMaybe (tpKalmanMeasurementVar base) (doubleField ["kalmanMeasurementVar"] params)
                , tpKalmanPhysicsBars = fromMaybe (tpKalmanPhysicsBars base) (intField ["kalmanPhysicsBars"] params)
                , tpKalmanPhysicsBacktestRatio = fromMaybe (tpKalmanPhysicsBacktestRatio base) (doubleField ["kalmanPhysicsBacktestRatio"] params)
                , tpKalmanPhysicsVolumeEwmaAlpha = fromMaybe (tpKalmanPhysicsVolumeEwmaAlpha base) (doubleField ["kalmanPhysicsVolumeEwmaAlpha"] params)
                , tpKalmanPhysicsVolumeSignalClamp = fromMaybe (tpKalmanPhysicsVolumeSignalClamp base) (doubleField ["kalmanPhysicsVolumeSignalClamp"] params)
                , tpKalmanPhysicsCloseBiasScale = fromMaybe (tpKalmanPhysicsCloseBiasScale base) (doubleField ["kalmanPhysicsCloseBiasScale"] params)
                , tpKalmanSensorCorrelationInflation = fromMaybe (tpKalmanSensorCorrelationInflation base) (doubleField ["kalmanSensorCorrelationInflation"] params)
                , tpKalmanInnovationInflationThreshold = fromMaybe (tpKalmanInnovationInflationThreshold base) (doubleField ["kalmanInnovationInflationThreshold"] params)
                , tpKalmanInnovationInflationMax = fromMaybe (tpKalmanInnovationInflationMax base) (doubleField ["kalmanInnovationInflationMax"] params)
                , tpKalmanZMin = fromMaybe (tpKalmanZMin base) (doubleField ["kalmanZMin"] params)
                , tpKalmanZMax = fromMaybe (tpKalmanZMax base) (doubleField ["kalmanZMax"] params)
                , tpMaxHighVolProb = fromMaybe (tpMaxHighVolProb base) (optionalDoubleField ["maxHighVolProb"] params)
                , tpMaxConformalWidth = fromMaybe (tpMaxConformalWidth base) (optionalDoubleField ["maxConformalWidth"] params)
                , tpMaxQuantileWidth = fromMaybe (tpMaxQuantileWidth base) (optionalDoubleField ["maxQuantileWidth"] params)
                , tpConfirmConformal = fromMaybe (tpConfirmConformal base) (boolField "confirmConformal" params)
                , tpConfirmQuantiles = fromMaybe (tpConfirmQuantiles base) (boolField "confirmQuantiles" params)
                , tpConfidenceSizing = fromMaybe (tpConfidenceSizing base) (boolField "confidenceSizing" params)
                , tpProtectionMinConfidence = fromMaybe (tpProtectionMinConfidence base) (doubleField ["protectionMinConfidence"] params)
                , tpMinPositionSize = fromMaybe (tpMinPositionSize base) (doubleField ["minPositionSize"] params)
                , tpKellyLiteSizing = fromMaybe (tpKellyLiteSizing base) (boolField "kellyLiteSizing" params)
                , tpKellyLiteFraction = fromMaybe (tpKellyLiteFraction base) (doubleField ["kellyLiteFraction"] params)
                , tpKellyLiteFloor = fromMaybe (tpKellyLiteFloor base) (doubleField ["kellyLiteFloor"] params)
                , tpKellyLiteCap = fromMaybe (tpKellyLiteCap base) (doubleField ["kellyLiteCap"] params)
                , tpPredictors = fromMaybe (tpPredictors base) (stringField ["predictors"] params)
                , tpRouterLookback = fromMaybe (tpRouterLookback base) (intField ["routerLookback"] params)
                , tpRouterRegimeMinBars = fromMaybe (tpRouterRegimeMinBars base) (intField ["routerRegimeMinBars"] params)
                , tpRouterRegimeMinFraction = fromMaybe (tpRouterRegimeMinFraction base) (doubleField ["routerRegimeMinFraction"] params)
                , tpRouterMinScore = fromMaybe (tpRouterMinScore base) (doubleField ["routerMinScore"] params)
                , tpFeeFixed = fromMaybe (tpFeeFixed base) (doubleField ["feeFixed"] params)
                , tpSlippageImpact = fromMaybe (tpSlippageImpact base) (doubleField ["slippageImpact"] params)
                , tpSlippageImpactPower = fromMaybe (tpSlippageImpactPower base) (doubleField ["slippageImpactPower"] params)
                , tpSlippageVolMult = fromMaybe (tpSlippageVolMult base) (doubleField ["slippageVolMult"] params)
                , tpSpreadVolMult = fromMaybe (tpSpreadVolMult base) (doubleField ["spreadVolMult"] params)
                , tpTakeProfitPartial = fromMaybe (tpTakeProfitPartial base) (optionalDoubleField ["takeProfitPartial"] params)
                , tpMaxTradesPerDay = fromMaybe (tpMaxTradesPerDay base) (optionalIntField ["maxTradesPerDay"] params)
                , tpExpectancyLookback = fromMaybe (tpExpectancyLookback base) (intField ["expectancyLookback"] params)
                , tpMinExpectancy = fromMaybe (tpMinExpectancy base) (optionalDoubleField ["minExpectancy"] params)
                , tpAdaptiveEdgeBufferMax = fromMaybe (tpAdaptiveEdgeBufferMax base) (doubleField ["adaptiveEdgeBufferMax"] params)
                , tpAdaptiveMinSignalToNoiseMax = fromMaybe (tpAdaptiveMinSignalToNoiseMax base) (doubleField ["adaptiveMinSignalToNoiseMax"] params)
                , tpAdaptiveTrendLookbackMax = fromMaybe (tpAdaptiveTrendLookbackMax base) (intField ["adaptiveTrendLookbackMax"] params)
                , tpAdaptiveKalmanZMinMax = fromMaybe (tpAdaptiveKalmanZMinMax base) (doubleField ["adaptiveKalmanZMinMax"] params)
                , tpAdaptiveWinRateSlack = fromMaybe (tpAdaptiveWinRateSlack base) (doubleField ["adaptiveWinRateSlack"] params)
                , tpAdaptiveProfitFactorSlack = fromMaybe (tpAdaptiveProfitFactorSlack base) (doubleField ["adaptiveProfitFactorSlack"] params)
                , tpAdaptiveFilters = fromMaybe (tpAdaptiveFilters base) (boolField "adaptiveFilters" params)
                , tpPerfLookback = fromMaybe (tpPerfLookback base) (intField ["perfLookback"] params)
                , tpPerfMinWinRate = fromMaybe (tpPerfMinWinRate base) (optionalDoubleField ["perfMinWinRate"] params)
                , tpPerfMinProfitFactor = fromMaybe (tpPerfMinProfitFactor base) (optionalDoubleField ["perfMinProfitFactor"] params)
                , tpMetaLabelFilter = fromMaybe (tpMetaLabelFilter base) (boolField "metaLabelFilter" params)
                , tpMetaLabelMinEdge = fromMaybe (tpMetaLabelMinEdge base) (doubleField ["metaLabelMinEdge"] params)
                , tpMetaLabelMinConfidence = fromMaybe (tpMetaLabelMinConfidence base) (doubleField ["metaLabelMinConfidence"] params)
                , tpMetaLabelRequireBand = fromMaybe (tpMetaLabelRequireBand base) (boolField "metaLabelRequireBand" params)
                , tpRegimeParameterBank = fromMaybe (tpRegimeParameterBank base) (boolField "regimeParameterBank" params)
                , tpRegimeBankHysteresis = fromMaybe (tpRegimeBankHysteresis base) (doubleField ["regimeBankHysteresis"] params)
                , tpRegimeTrendOpenMult = fromMaybe (tpRegimeTrendOpenMult base) (doubleField ["regimeTrendOpenMult"] params)
                , tpRegimeMrOpenMult = fromMaybe (tpRegimeMrOpenMult base) (doubleField ["regimeMrOpenMult"] params)
                , tpRegimeHighVolOpenMult = fromMaybe (tpRegimeHighVolOpenMult base) (doubleField ["regimeHighVolOpenMult"] params)
                , tpRegimeTrendSizeMult = fromMaybe (tpRegimeTrendSizeMult base) (doubleField ["regimeTrendSizeMult"] params)
                , tpRegimeMrSizeMult = fromMaybe (tpRegimeMrSizeMult base) (doubleField ["regimeMrSizeMult"] params)
                , tpRegimeHighVolSizeMult = fromMaybe (tpRegimeHighVolSizeMult base) (doubleField ["regimeHighVolSizeMult"] params)
                , tpMultiTimeframeConsensus = fromMaybe (tpMultiTimeframeConsensus base) (boolField "multiTimeframeConsensus" params)
                , tpMtfFastBars = fromMaybe (tpMtfFastBars base) (intField ["mtfFastBars"] params)
                , tpMtfMidBars = fromMaybe (tpMtfMidBars base) (intField ["mtfMidBars"] params)
                , tpMtfSlowBars = fromMaybe (tpMtfSlowBars base) (intField ["mtfSlowBars"] params)
                , tpMtfMinAgree = fromMaybe (tpMtfMinAgree base) (intField ["mtfMinAgree"] params)
                , tpTechnicalParams = applyTechnicalPriorOverlay params (tpTechnicalParams base)
                }
     in normalizeTrialParams overlaid

applyTechnicalPriorOverlay :: KM.KeyMap Value -> TechnicalTrialParams -> TechnicalTrialParams
applyTechnicalPriorOverlay params base =
    normalizeTechnicalTrialParams
        base
            { ttpTaEntryOpenThreshold = fromMaybe (ttpTaEntryOpenThreshold base) (doubleField ["taEntryOpenThreshold"] params)
            , ttpTaTrendAdxThreshold = fromMaybe (ttpTaTrendAdxThreshold base) (doubleField ["taTrendAdxThreshold"] params)
            , ttpTaTrendStopAtrMult = fromMaybe (ttpTaTrendStopAtrMult base) (doubleField ["taTrendStopAtrMult"] params)
            , ttpTaTrendTakeProfitAtrMult = fromMaybe (ttpTaTrendTakeProfitAtrMult base) (doubleField ["taTrendTakeProfitAtrMult"] params)
            , ttpTaBestCandidateMinConfidence = fromMaybe (ttpTaBestCandidateMinConfidence base) (doubleField ["taBestCandidateMinConfidence"] params)
            , ttpRegimeAdxWeight = fromMaybe (ttpRegimeAdxWeight base) (doubleField ["regimeAdxWeight"] params)
            , ttpRegimeTrendThreshold = fromMaybe (ttpRegimeTrendThreshold base) (doubleField ["regimeTrendThreshold"] params)
            , ttpRegimeRangeThreshold = fromMaybe (ttpRegimeRangeThreshold base) (doubleField ["regimeRangeThreshold"] params)
            , ttpRegimeTrendAdxFloor = fromMaybe (ttpRegimeTrendAdxFloor base) (doubleField ["regimeTrendAdxFloor"] params)
            , ttpRegimeTrendAdxSpan = fromMaybe (ttpRegimeTrendAdxSpan base) (doubleField ["regimeTrendAdxSpan"] params)
            , ttpRegimeTrendAroonFloor = fromMaybe (ttpRegimeTrendAroonFloor base) (doubleField ["regimeTrendAroonFloor"] params)
            , ttpRegimeTrendAroonSpan = fromMaybe (ttpRegimeTrendAroonSpan base) (doubleField ["regimeTrendAroonSpan"] params)
            , ttpRegimeTrendSlopeFloor = fromMaybe (ttpRegimeTrendSlopeFloor base) (doubleField ["regimeTrendSlopeFloor"] params)
            , ttpRegimeTrendSlopeSpan = fromMaybe (ttpRegimeTrendSlopeSpan base) (doubleField ["regimeTrendSlopeSpan"] params)
            , ttpRegimeTrendAroonWeight = fromMaybe (ttpRegimeTrendAroonWeight base) (doubleField ["regimeTrendAroonWeight"] params)
            , ttpRegimeRangeAdxCeiling = fromMaybe (ttpRegimeRangeAdxCeiling base) (doubleField ["regimeRangeAdxCeiling"] params)
            , ttpRegimeRangeAdxSpan = fromMaybe (ttpRegimeRangeAdxSpan base) (doubleField ["regimeRangeAdxSpan"] params)
            , ttpRegimeRangeWidthCeiling = fromMaybe (ttpRegimeRangeWidthCeiling base) (doubleField ["regimeRangeWidthCeiling"] params)
            , ttpRegimeRangeWidthSpan = fromMaybe (ttpRegimeRangeWidthSpan base) (doubleField ["regimeRangeWidthSpan"] params)
            , ttpRegimeRangeAdxWeight = fromMaybe (ttpRegimeRangeAdxWeight base) (doubleField ["regimeRangeAdxWeight"] params)
            , ttpBlendSoftmaxScale = fromMaybe (ttpBlendSoftmaxScale base) (doubleField ["blendSoftmaxScale"] params)
            , ttpBlendNetSoftmaxScale = fromMaybe (ttpBlendNetSoftmaxScale base) (doubleField ["blendNetSoftmaxScale"] params)
            , ttpBlendEdgePower = fromMaybe (ttpBlendEdgePower base) (doubleField ["blendEdgePower"] params)
            , ttpBlendSmoothAlpha = fromMaybe (ttpBlendSmoothAlpha base) (doubleField ["blendSmoothAlpha"] params)
            , ttpBlendHedgeEta = fromMaybe (ttpBlendHedgeEta base) (doubleField ["blendHedgeEta"] params)
            , ttpBlendHedgeMaxError = fromMaybe (ttpBlendHedgeMaxError base) (doubleField ["blendHedgeMaxError"] params)
            , ttpSignalEntryEdgeHeadroomMult = fromMaybe (ttpSignalEntryEdgeHeadroomMult base) (doubleField ["signalEntryEdgeHeadroomMult"] params)
            , ttpSignalEntryEdgeSpikeMult = fromMaybe (ttpSignalEntryEdgeSpikeMult base) (doubleField ["signalEntryEdgeSpikeMult"] params)
            , ttpSignalEntryEdgeSpikeCap = fromMaybe (ttpSignalEntryEdgeSpikeCap base) (doubleField ["signalEntryEdgeSpikeCap"] params)
            , ttpSignalEntryEdgeSpikeConsecutiveLimit = fromMaybe (ttpSignalEntryEdgeSpikeConsecutiveLimit base) (intField ["signalEntryEdgeSpikeConsecutiveLimit"] params)
            , ttpSignalTrendSlackMult = fromMaybe (ttpSignalTrendSlackMult base) (doubleField ["signalTrendSlackMult"] params)
            , ttpSignalTrendSlackCap = fromMaybe (ttpSignalTrendSlackCap base) (doubleField ["signalTrendSlackCap"] params)
            , ttpSignalDirectionalityLookbackBars = fromMaybe (ttpSignalDirectionalityLookbackBars base) (intField ["signalDirectionalityLookbackBars"] params)
            , ttpSignalDirectionalityChopEfficiencyMax = fromMaybe (ttpSignalDirectionalityChopEfficiencyMax base) (doubleField ["signalDirectionalityChopEfficiencyMax"] params)
            , ttpSignalDirectionalityMrEfficiencyMax = fromMaybe (ttpSignalDirectionalityMrEfficiencyMax base) (doubleField ["signalDirectionalityMrEfficiencyMax"] params)
            , ttpSignalDirectionalityWeakZMin = fromMaybe (ttpSignalDirectionalityWeakZMin base) (doubleField ["signalDirectionalityWeakZMin"] params)
            , ttpSignalPredictorTrackingFloor = fromMaybe (ttpSignalPredictorTrackingFloor base) (doubleField ["signalPredictorTrackingFloor"] params)
            , ttpPredictorGbdtTrees = fromMaybe (ttpPredictorGbdtTrees base) (intField ["predictorGbdtTrees"] params)
            , ttpPredictorGbdtLearningRate = fromMaybe (ttpPredictorGbdtLearningRate base) (doubleField ["predictorGbdtLearningRate"] params)
            , ttpPredictorCalibrationRatio = fromMaybe (ttpPredictorCalibrationRatio base) (doubleField ["predictorCalibrationRatio"] params)
            , ttpPredictorConformalAlpha = fromMaybe (ttpPredictorConformalAlpha base) (doubleField ["predictorConformalAlpha"] params)
            , ttpPredictorTcnRidgeLambda = fromMaybe (ttpPredictorTcnRidgeLambda base) (doubleField ["predictorTcnRidgeLambda"] params)
            , ttpPredictorQuantileEpochs = fromMaybe (ttpPredictorQuantileEpochs base) (intField ["predictorQuantileEpochs"] params)
            , ttpPredictorQuantileLearningRate = fromMaybe (ttpPredictorQuantileLearningRate base) (doubleField ["predictorQuantileLearningRate"] params)
            , ttpPredictorQuantileL2 = fromMaybe (ttpPredictorQuantileL2 base) (doubleField ["predictorQuantileL2"] params)
            , ttpPredictorKnnMaxExamples = fromMaybe (ttpPredictorKnnMaxExamples base) (intField ["predictorKnnMaxExamples"] params)
            , ttpPredictorKnnK = fromMaybe (ttpPredictorKnnK base) (intField ["predictorKnnK"] params)
            , ttpPredictorDecisionTreeMaxDepth = fromMaybe (ttpPredictorDecisionTreeMaxDepth base) (intField ["predictorDecisionTreeMaxDepth"] params)
            , ttpPredictorDecisionTreeMinLeafSize = fromMaybe (ttpPredictorDecisionTreeMinLeafSize base) (intField ["predictorDecisionTreeMinLeafSize"] params)
            , ttpPredictorTransformerTemperature = fromMaybe (ttpPredictorTransformerTemperature base) (doubleField ["predictorTransformerTemperature"] params)
            , ttpPredictorTransformerMaxExamples = fromMaybe (ttpPredictorTransformerMaxExamples base) (intField ["predictorTransformerMaxExamples"] params)
            , ttpPredictorHmmIterations = fromMaybe (ttpPredictorHmmIterations base) (intField ["predictorHmmIterations"] params)
            }

optionalDoubleField :: [String] -> KM.KeyMap Value -> Maybe (Maybe Double)
optionalDoubleField keys obj =
    listToMaybe (mapMaybe (`optionalDoubleFieldOne` obj) keys)

optionalDoubleFieldOne :: String -> KM.KeyMap Value -> Maybe (Maybe Double)
optionalDoubleFieldOne key obj =
    case KM.lookup (Key.fromString key) obj of
        Nothing -> Nothing
        Just Null -> Just Nothing
        Just value -> Just <$> coerceFloatValue value

optionalIntField :: [String] -> KM.KeyMap Value -> Maybe (Maybe Int)
optionalIntField keys obj =
    listToMaybe (mapMaybe (`optionalIntFieldOne` obj) keys)

optionalIntFieldOne :: String -> KM.KeyMap Value -> Maybe (Maybe Int)
optionalIntFieldOne key obj =
    case KM.lookup (Key.fromString key) obj of
        Nothing -> Nothing
        Just Null -> Just Nothing
        Just value -> Just <$> coerceIntValue value

techniqueSummaryToJson :: OptimizationTechniqueSummary -> Value
techniqueSummaryToJson t =
    object
        [ "sobolSeeding" .= otsAppliedSobolSeeding t
        , "successiveHalving" .= otsAppliedSuccessiveHalving t
        , "bayesianExpectedImprovement" .= otsAppliedBayesianEi t
        , "empiricalPriorSampling" .= otsAppliedEmpiricalPriors t
        , "walkForwardCrossValidation" .= otsAppliedWalkForward t
        , "ensembleTopPerformers" .= otsAppliedEnsemble t
        ]

writeTopJson ::
    String ->
    String ->
    String ->
    Maybe String ->
    [TrialResult] ->
    OptimizationTechniqueSummary ->
    IO ()
writeTopJson topPath dataSource sourceOverride symbolLabel records summary = do
    nowMs <- fmap (floor . (* 1000) :: POSIXTime -> Int) getPOSIXTime
    let successful =
            [ tr
            | tr <- records
            , trEligible tr
            , Just eq <- [trFinalEquity tr]
            , eq > 1
            , not (isInfinite eq)
            , isJust (trScore tr)
            ]
        sortKey tr =
            let ann = metricFloat (trMetrics tr) "annualizedReturn" (-(1 / 0))
                ann' = if isNaN ann || isInfinite ann then -(1 / 0) else ann
                score = fromMaybe (-(1 / 0)) (trScore tr)
                score' = if isNaN score || isInfinite score then -(1 / 0) else score
                eq = fromMaybe 0 (trFinalEquity tr)
                eq' = if isNaN eq || isInfinite eq then 0 else eq
             in (ann', score', eq')
        sorted = sortOn (Data.Ord.Down . sortKey) successful
        combos = zipWith (comboFromTrial nowMs dataSource sourceOverride symbolLabel) [1 ..] (take 10 sorted)
        topMetrics =
            let topN = take 5 sorted
                extract f = [f tr | tr <- topN, trEligible tr, isJust (trScore tr)]
                avg xs =
                    let (sumX, countX) =
                            foldl'
                                (\(acc, n) x -> (acc + x, n + 1 :: Int))
                                (0, 0)
                                xs
                     in if countX == 0 then Nothing else Just (sumX / fromIntegral countX)
             in object
                    [ "avgScore" .= avg (map (fromMaybe 0 . trScore) topN)
                    , "avgSharpe" .= avg (map (\tr -> metricFloat (trMetrics tr) "sharpe" 0) topN)
                    , "avgMaxDrawdown" .= avg (map (\tr -> metricFloat (trMetrics tr) "maxDrawdown" 0) topN)
                    ]
        ensemble =
            object
                [ "members" .= map (\tr -> object ["objective" .= trObjective tr, "score" .= trScore tr]) (take 3 sorted)
                , "metrics" .= topMetrics
                ]
    path <- expandUser topPath
    createDirectoryIfMissing True (takeDirectory path)
    let export' =
            object
                [ "generatedAtMs" .= nowMs
                , "source" .= ("optimize_equity.py" :: String)
                , "combos" .= combos
                , "ensemble" .= ensemble
                , "bestOptimizationTechniques" .= map optimizationTechniqueToJson bestOptimizationTechniques
                , "optimizationTechniquesApplied" .= techniqueSummaryToJson summary
                ]
    BL.writeFile path (encodePretty export')
    putStrLn ("Wrote top combos JSON: " ++ path)

comboFromTrial :: Int -> String -> String -> Maybe String -> Int -> TrialResult -> Value
comboFromTrial createdAtMs dataSource sourceOverride symbolLabel rank tr =
    let metricsRaw = trMetrics tr
        params = trParams tr
        finalEq = fromMaybe 0 (trFinalEquity tr)
        periodsPerYear = fromMaybe (inferPeriodsPerYear (tpInterval params)) (tpPeriodsPerYear params)
        symbol = symbolLabel >>= sanitizeComboSymbolForPlatform (tpPlatform params)
        currentMaxHoldBars = tpMaxHoldBars params
        closeTimingReport = trialCloseTimingReport params symbol (trStdoutJson tr)
        metrics0 = ensureAnnualizedReturnMetrics metricsRaw finalEq periodsPerYear (tpBars params)
        metricsWithWalkForward = applyWalkForwardSummaryMetrics metrics0 (trStdoutJson tr)
        metrics = applyCloseTimingMetrics metricsWithWalkForward currentMaxHoldBars currentMaxHoldBars closeTimingReport
        metricsVal = maybe Null Object metrics
        source = resolveSourceLabel (tpPlatform params) dataSource sourceOverride
        paramsValue =
            object $
                [ "platform" .= tpPlatform params
                , "interval" .= tpInterval params
                , "bars" .= tpBars params
                , "method" .= tpMethod params
                , "blendWeight" .= tpBlendWeight params
                , "routerScorePnlWeight" .= tpRouterScorePnlWeight params
                , "positioning" .= tpPositioning params
                , "normalization" .= tpNormalization params
                , "baseOpenThreshold" .= tpBaseOpenThreshold params
                , "baseCloseThreshold" .= tpBaseCloseThreshold params
                , "minHoldBars" .= tpMinHoldBars params
                , "cooldownBars" .= tpCooldownBars params
                , "maxHoldBars" .= currentMaxHoldBars
                , "minEdge" .= tpMinEdge params
                , "minSignalToNoise" .= tpMinSignalToNoise params
                , "snrSizeWeight" .= tpSnrSizeWeight params
                , "edgeBuffer" .= tpEdgeBuffer params
                , "costAwareEdge" .= tpCostAwareEdge params
                , "trendLookback" .= tpTrendLookback params
                , "maxPositionSize" .= tpMaxPositionSize params
                , "volTarget" .= tpVolTarget params
                , "volLookback" .= tpVolLookback params
                , "volEwmaAlpha" .= tpVolEwmaAlpha params
                , "volFloor" .= tpVolFloor params
                , "volScaleMax" .= tpVolScaleMax params
                , "maxVolatility" .= tpMaxVolatility params
                , "periodsPerYear" .= periodsPerYear
                , "kalmanMarketTopN" .= tpKalmanMarketTopN params
                , "fee" .= tpFee params
                , "fundingRate" .= tpFundingRate params
                , "fundingBySide" .= tpFundingBySide params
                , "fundingOnOpen" .= tpFundingOnOpen params
                , "rebalanceBars" .= tpRebalanceBars params
                , "rebalanceThreshold" .= tpRebalanceThreshold params
                , "rebalanceCostMult" .= tpRebalanceCostMult params
                , "rebalanceGlobal" .= tpRebalanceGlobal params
                , "rebalanceResetOnSignal" .= tpRebalanceResetOnSignal params
                , "epochs" .= tpEpochs params
                , "hiddenSize" .= tpHiddenSize params
                , "learningRate" .= tpLearningRate params
                , "lstmAdamBeta1" .= tpLstmAdamBeta1 params
                , "lstmAdamBeta2" .= tpLstmAdamBeta2 params
                , "lstmAdamEps" .= tpLstmAdamEps params
                , "valRatio" .= tpValRatio params
                , "patience" .= tpPatience params
                , "walkForwardFolds" .= tpWalkForwardFolds params
                , "walkForwardEmbargoBars" .= tpWalkForwardEmbargoBars params
                , "tuneStressVolMult" .= tpTuneStressVolMult params
                , "tuneStressShock" .= tpTuneStressShock params
                , "tuneStressWeight" .= tpTuneStressWeight params
                , "gradClip" .= tpGradClip params
                , "slippage" .= tpSlippage params
                , "spread" .= tpSpread params
                , "intrabarFill" .= tpIntrabarFill params
                , "triLayer" .= tpTriLayer params
                , "triLayerFastMult" .= tpTriLayerFastMult params
                , "triLayerSlowMult" .= tpTriLayerSlowMult params
                , "triLayerCloudPadding" .= tpTriLayerCloudPadding params
                , "triLayerCloudSlope" .= tpTriLayerCloudSlope params
                , "triLayerCloudWidth" .= tpTriLayerCloudWidth params
                , "triLayerTouchLookback" .= tpTriLayerTouchLookback params
                , "triLayerPriceAction" .= tpTriLayerPriceAction params
                , "triLayerPriceActionBody" .= tpTriLayerPriceActionBody params
                , "triLayerPriceActionWickRatio" .= tpTriLayerPriceActionWickRatio params
                , "triLayerPriceActionOppositeWickMax" .= tpTriLayerPriceActionOppositeWickMax params
                , "triLayerPriceActionBodyTolerance" .= tpTriLayerPriceActionBodyTolerance params
                , "triLayerExitOnSlow" .= tpTriLayerExitOnSlow params
                , "kalmanBandLookback" .= tpKalmanBandLookback params
                , "kalmanBandStdMult" .= tpKalmanBandStdMult params
                , "lstmExitFlipBars" .= tpLstmExitFlipBars params
                , "lstmExitFlipGraceBars" .= tpLstmExitFlipGraceBars params
                , "lstmExitFlipStrong" .= tpLstmExitFlipStrong params
                , "lstmConfidenceSoft" .= tpLstmConfidenceSoft params
                , "lstmConfidenceHard" .= tpLstmConfidenceHard params
                , "stopLoss" .= tpStopLoss params
                , "takeProfit" .= tpTakeProfit params
                , "trailingStop" .= tpTrailingStop params
                , "stopLossVolMult" .= tpStopLossVolMult params
                , "takeProfitVolMult" .= tpTakeProfitVolMult params
                , "trailingStopVolMult" .= tpTrailingStopVolMult params
                , "riskPerTrade" .= tpRiskPerTrade params
                , "maxDrawdown" .= tpMaxDrawdown params
                , "maxDailyLoss" .= tpMaxDailyLoss params
                , "maxWeeklyLoss" .= tpMaxWeeklyLoss params
                , "maxOrderErrors" .= tpMaxOrderErrors params
                , "kalmanDt" .= tpKalmanDt params
                , "kalmanProcessVar" .= tpKalmanProcessVar params
                , "kalmanMeasurementVar" .= tpKalmanMeasurementVar params
                , "kalmanPhysicsBars" .= tpKalmanPhysicsBars params
                , "kalmanPhysicsBacktestRatio" .= tpKalmanPhysicsBacktestRatio params
                , "kalmanPhysicsVolumeEwmaAlpha" .= tpKalmanPhysicsVolumeEwmaAlpha params
                , "kalmanPhysicsVolumeSignalClamp" .= tpKalmanPhysicsVolumeSignalClamp params
                , "kalmanPhysicsCloseBiasScale" .= tpKalmanPhysicsCloseBiasScale params
                , "sensorVarianceEwmaAlpha" .= tpSensorVarianceEwmaAlpha params
                , "kalmanSensorCorrelationInflation" .= tpKalmanSensorCorrelationInflation params
                , "kalmanInnovationInflationThreshold" .= tpKalmanInnovationInflationThreshold params
                , "kalmanInnovationInflationMax" .= tpKalmanInnovationInflationMax params
                , "kalmanZMin" .= tpKalmanZMin params
                , "kalmanZMax" .= tpKalmanZMax params
                , "maxHighVolProb" .= tpMaxHighVolProb params
                , "maxConformalWidth" .= tpMaxConformalWidth params
                , "maxQuantileWidth" .= tpMaxQuantileWidth params
                , "confirmConformal" .= tpConfirmConformal params
                , "confirmQuantiles" .= tpConfirmQuantiles params
                , "confidenceSizing" .= tpConfidenceSizing params
                , "protectionMinConfidence" .= tpProtectionMinConfidence params
                , "minPositionSize" .= tpMinPositionSize params
                , "kellyLiteSizing" .= tpKellyLiteSizing params
                , "kellyLiteFraction" .= tpKellyLiteFraction params
                , "kellyLiteFloor" .= tpKellyLiteFloor params
                , "kellyLiteCap" .= tpKellyLiteCap params
                , "predictors" .= tpPredictors params
                , "routerLookback" .= tpRouterLookback params
                , "routerRegimeMinBars" .= tpRouterRegimeMinBars params
                , "routerRegimeMinFraction" .= tpRouterRegimeMinFraction params
                , "routerMinScore" .= tpRouterMinScore params
                , "feeFixed" .= tpFeeFixed params
                , "slippageImpact" .= tpSlippageImpact params
                , "slippageImpactPower" .= tpSlippageImpactPower params
                , "slippageVolMult" .= tpSlippageVolMult params
                , "spreadVolMult" .= tpSpreadVolMult params
                , "takeProfitPartial" .= tpTakeProfitPartial params
                , "maxTradesPerDay" .= tpMaxTradesPerDay params
                , "expectancyLookback" .= tpExpectancyLookback params
                , "minExpectancy" .= tpMinExpectancy params
                , "lossStreakMax" .= tpLossStreakMax params
                , "lossStreakCooldownBars" .= tpLossStreakCooldownBars params
                , "maxOpenPositions" .= tpMaxOpenPositions params
                , "maxGrossExposure" .= tpMaxGrossExposure params
                , "maxNetExposure" .= tpMaxNetExposure params
                , "maxExposurePerBase" .= tpMaxExposurePerBase params
                , "maxOpenPerBase" .= tpMaxOpenPerBase params
                , "adaptiveEdgeBufferMax" .= tpAdaptiveEdgeBufferMax params
                , "adaptiveMinSignalToNoiseMax" .= tpAdaptiveMinSignalToNoiseMax params
                , "adaptiveTrendLookbackMax" .= tpAdaptiveTrendLookbackMax params
                , "adaptiveKalmanZMinMax" .= tpAdaptiveKalmanZMinMax params
                , "adaptiveWinRateSlack" .= tpAdaptiveWinRateSlack params
                , "adaptiveProfitFactorSlack" .= tpAdaptiveProfitFactorSlack params
                , "adaptiveFilters" .= tpAdaptiveFilters params
                , "perfLookback" .= tpPerfLookback params
                , "perfMinWinRate" .= tpPerfMinWinRate params
                , "perfMinProfitFactor" .= tpPerfMinProfitFactor params
                , "metaLabelFilter" .= tpMetaLabelFilter params
                , "metaLabelMinEdge" .= tpMetaLabelMinEdge params
                , "metaLabelMinConfidence" .= tpMetaLabelMinConfidence params
                , "metaLabelRequireBand" .= tpMetaLabelRequireBand params
                , "regimeParameterBank" .= tpRegimeParameterBank params
                , "regimeBankHysteresis" .= tpRegimeBankHysteresis params
                , "regimeTrendOpenMult" .= tpRegimeTrendOpenMult params
                , "regimeMrOpenMult" .= tpRegimeMrOpenMult params
                , "regimeHighVolOpenMult" .= tpRegimeHighVolOpenMult params
                , "regimeTrendSizeMult" .= tpRegimeTrendSizeMult params
                , "regimeMrSizeMult" .= tpRegimeMrSizeMult params
                , "regimeHighVolSizeMult" .= tpRegimeHighVolSizeMult params
                , "multiTimeframeConsensus" .= tpMultiTimeframeConsensus params
                , "mtfFastBars" .= tpMtfFastBars params
                , "mtfMidBars" .= tpMtfMidBars params
                , "mtfSlowBars" .= tpMtfSlowBars params
                , "mtfMinAgree" .= tpMtfMinAgree params
                , "crossAssetConfirmation" .= tpCrossAssetConfirmation params
                , "crossAssetMinBeta" .= tpCrossAssetMinBeta params
                , "crossAssetMinEdge" .= tpCrossAssetMinEdge params
                , "pairsStatArb" .= tpPairsStatArb params
                , "pairsStatArbLookback" .= tpPairsStatArbLookback params
                , "pairsStatArbZEntry" .= tpPairsStatArbZEntry params
                , "pairsStatArbSizeMult" .= tpPairsStatArbSizeMult params
                , "fundingOiAware" .= tpFundingOiAware params
                , "fundingOiFundingCap" .= tpFundingOiFundingCap params
                , "fundingOiVolLookback" .= tpFundingOiVolLookback params
                , "fundingOiVolCap" .= tpFundingOiVolCap params
                , "fundingOiSizeMult" .= tpFundingOiSizeMult params
                ]
                    ++ technicalTrialParamsPairs (tpTechnicalParams params)
                    ++ ["binanceSymbol" .= symbol]
        identityBase =
            object
                [ "params" .= paramsValue
                , "openThreshold" .= trOpenThreshold tr
                , "closeThreshold" .= trCloseThreshold tr
                , "objective" .= trObjective tr
                ]
        identity =
            if null source
                then identityBase
                else addField "source" (Aeson.toJSON source) identityBase
        comboUuid = comboUuidFromValue identity
        combo =
            object
                [ "uuid" .= comboUuid
                , "rank" .= rank
                , "createdAtMs" .= createdAtMs
                , "finalEquity" .= trFinalEquity tr
                , "objective" .= trObjective tr
                , "score" .= trScore tr
                , "openThreshold" .= trOpenThreshold tr
                , "closeThreshold" .= trCloseThreshold tr
                , "source" .= source
                , "optimizerTrialOrigin" .= trOrigin tr
                , "metrics" .= metricsVal
                , "params" .= paramsValue
                ]
        comboWithKelly =
            case extractKellyLiteSummary (trStdoutJson tr) of
                Just report -> addField "kellyLite" (Object report) combo
                Nothing -> combo
     in case extractOperations (trStdoutJson tr) of
            Just ops -> addField "operations" (Array (V.fromList ops)) comboWithKelly
            Nothing -> comboWithKelly

addField :: String -> Value -> Value -> Value
addField key value val =
    case val of
        Object obj -> Object (KM.insert (Key.fromString key) value obj)
        _ -> val

comboUuidFromValue :: Value -> String
comboUuidFromValue val =
    formatUuidFromHex (hashBytesHex (encodePretty val))

formatUuidFromHex :: String -> String
formatUuidFromHex hex =
    let h = take 32 hex
        seg1 = take 8 h
        seg2 = take 4 (drop 8 h)
        seg3 = take 4 (drop 12 h)
        seg4 = take 4 (drop 16 h)
        seg5 = take 12 (drop 20 h)
     in intercalate "-" [seg1, seg2, seg3, seg4, seg5]

hashBytesHex :: BL.ByteString -> String
hashBytesHex bs =
    let digest :: Digest SHA256
        digest = hash (BL.toStrict bs)
     in BS8.unpack (B16.encode (convert digest))
