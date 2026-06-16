{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PatternSynonyms #-}

module Trader.Trading (
    BacktestCostAttribution (..),
    BacktestResult (..),
    emptyBacktestCostAttribution,
    EnsembleConfig (..),
    StepMeta (..),
    IntrabarFill (..),
    PositionSide (PositionLong, PositionShort, SideLong, SideShort),
    Positioning (..),
    simulateEnsemble,
    simulateEnsembleWithHLChecked,
    simulateEnsembleVWithHLChecked,
    ExitReason (..),
    HaltInputs (..),
    anyRiskLimitNonFinite,
    drawdownLimitInvalid,
    specRiskHalt,
    TradeEntrySource (..),
    Trade (..),
    TradingEntryGateInputs (..),
    mkTradingEntryGateInputs,
    EntryGateState (..),
    exitReasonCode,
    exitReasonFromCode,
    mkEntryGateState,
    defaultTriLayerPriceActionBodyOpenThresholdMult,
    outcomeWeightCap,
    outcomeWeightLossScale,
    outcomeWeightWinScale,
    tradeEntrySourceCode,
    tradeOutcomeWeightFactor,
    tradeOutcomeWeights,
    positionSizeScaleHardFailMultiplier,
) where

import Control.Applicative ((<|>))
import qualified Data.Aeson as Aeson
import Data.Char (toUpper)
import Data.Int (Int64)
import Data.List (foldl')
import Data.Maybe (isJust, isNothing)
import qualified Data.Maybe
import qualified Data.Text as T
import qualified Data.Vector as V
import Trader.Duration (TimeWindow, minuteOfDayFromMs, timeWindowContains)
import Trader.Kalman3 (KalmanRunV (..), runConstantAcceleration1DVec)
import Trader.SignalGates (
    SignalGateConfig,
    finiteDouble,
    normalizeSignalEntryEdge,
    normalizeSignalThreshold,
    signalEntryEdgeSpikeConsecutiveOkWithConfig,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryFeeBufferOkWithConfig,
    signalEntryHeadroomOk,
    signalTrendSmaConfirmed,
    signalTrendSmaConfirmedWithConfig,
 )
import Trader.VolConfGate (
    VolConfGateCell (..),
    VolConfGateConfig,
    VolConfGatePreset (..),
    applyVolConfGateBehavior,
    volConfGateCellWithConfig,
 )

-- Keep the optimizer/reporting simulation surface anchored in Trader.Trading so
-- the public import seam does not depend on a non-built auxiliary module.
data EnsembleConfig = EnsembleConfig
    { ecPeriodsPerYear :: !Double
    , ecOpenThreshold :: !Double
    , ecCloseThreshold :: !Double
    , ecMinEdge :: !Double
    , ecRouterLookback :: !Int
    , ecRouterRegimeMinBars :: !Int
    , ecRouterRegimeMinFraction :: !Double
    , ecRouterMinScore :: !Double
    , ecRouterScorePnlWeight :: !Double
    , ecFee :: !Double
    , ecFeeFixed :: !Double
    , ecFeeMin :: !Double
    , ecSlippage :: !Double
    , ecSlippageVolMult :: !Double
    , ecSlippageImpactPower :: !Double
    , ecSlippageImpact :: !Double
    , ecSpread :: !Double
    , ecSpreadVolMult :: !Double
    , ecStopLoss :: !(Maybe Double)
    , ecTakeProfit :: !(Maybe Double)
    , ecTrailingStop :: !(Maybe Double)
    , ecStopLossVolMult :: !Double
    , ecTakeProfitVolMult :: !Double
    , ecTrailingStopVolMult :: !Double
    , ecMinHoldBars :: !Int
    , ecCooldownBars :: !Int
    , ecMaxHoldBars :: !(Maybe Int)
    , ecMaxDrawdown :: !(Maybe Double)
    , ecMaxDailyLoss :: !(Maybe Double)
    , ecMaxWeeklyLoss :: !(Maybe Double)
    , ecRiskPerTrade :: !(Maybe Double)
    , ecMaxTradesPerDay :: !(Maybe Int)
    , ecExpectancyLookback :: !Int
    , ecMinExpectancy :: !(Maybe Double)
    , ecPerfLookback :: !Int
    , ecPerfMinWinRate :: !(Maybe Double)
    , ecPerfMinProfitFactor :: !(Maybe Double)
    , ecAdaptiveFilters :: !Bool
    , ecAdaptiveEdgeBufferMax :: !Double
    , ecAdaptiveMinSignalToNoiseMax :: !Double
    , ecAdaptiveKalmanZMinMax :: !Double
    , ecAdaptiveWinRateSlack :: !Double
    , ecAdaptiveProfitFactorSlack :: !Double
    , ecAdaptiveTrendLookbackMax :: !Int
    , ecLossStreakMax :: !Int
    , ecLossStreakCooldownBars :: !Int
    , ecNoTradeWindows :: ![TimeWindow]
    , ecIntervalSeconds :: !(Maybe Int)
    , ecOpenTimes :: !(Maybe (V.Vector Int64))
    , ecOpenPrices :: !(Maybe (V.Vector Double))
    , ecMetaMask :: !(Maybe (V.Vector Bool))
    , ecPositioning :: !Positioning
    , ecIntrabarFill :: !IntrabarFill
    , ecMaxPositionSize :: !Double
    , ecSignalGateConfig :: !SignalGateConfig
    , ecEntryEdgeSpikeAuditOnly :: !Bool
    , ecEntryEdgeSpikeConsecutive :: !Int
    , ecMinSignalToNoise :: !Double
    , ecSnrSizeWeight :: !Double
    , ecThresholdFactorEnabled :: !Bool
    , ecThresholdFactorAlpha :: !Double
    , ecThresholdFactorMin :: !Double
    , ecThresholdFactorMax :: !Double
    , ecThresholdFactorFloor :: !Double
    , ecThresholdFactorEdgeKalWeight :: !Double
    , ecThresholdFactorEdgeLstmWeight :: !Double
    , ecThresholdFactorKalmanZWeight :: !Double
    , ecThresholdFactorHighVolWeight :: !Double
    , ecThresholdFactorConformalWeight :: !Double
    , ecThresholdFactorQuantileWeight :: !Double
    , ecThresholdFactorLstmConfWeight :: !Double
    , ecThresholdFactorLstmHealthWeight :: !Double
    , ecLstmTrainingHealth :: !(Maybe Double)
    , ecTrendLookback :: !Int
    , ecVolTarget :: !(Maybe Double)
    , ecVolLookback :: !Int
    , ecVolEwmaAlpha :: !(Maybe Double)
    , ecVolFloor :: !Double
    , ecVolScaleMax :: !Double
    , ecMaxVolatility :: !(Maybe Double)
    , ecVolConfGate :: !VolConfGatePreset
    , ecVolConfGateConfig :: !VolConfGateConfig
    , ecRebalanceBars :: !Int
    , ecRebalanceThreshold :: !Double
    , ecRebalanceGlobal :: !Bool
    , ecRebalanceResetOnSignal :: !Bool
    , ecFundingRate :: !Double
    , ecFundingBySide :: !Bool
    , ecFundingOnOpen :: !Bool
    , ecBlendWeight :: !Double
    , ecBlendSoftmaxScale :: !Double
    , ecBlendNetSoftmaxScale :: !Double
    , ecBlendEdgePower :: !Double
    , ecBlendSmoothAlpha :: !Double
    , ecBlendHedgeEta :: !Double
    , ecBlendHedgeMaxError :: !Double
    , ecBlendDivergenceK :: !Double
    , ecBlendRegimeHighVolCutoff :: !Double
    , ecBlendRegimeKalmanZCutoff :: !Double
    , ecBlendBanditExploreScale :: !Double
    , ecBlendFractalReturnClamp :: !Double
    , ecBlendFractalAlignedGain :: !Double
    , ecBlendFractalConflictGain :: !Double
    , ecBlendCoherenceConflictFloor :: !Double
    , ecBlendCoherenceConflictScale :: !Double
    , ecBlendCoherenceBoostThreshold :: !Double
    , ecBlendCoherenceBoostGain :: !Double
    , ecBlendCoherenceBoostSpan :: !Double
    , ecBlendAnchorConflictBase :: !Double
    , ecBlendAnchorConflictScale :: !Double
    , ecBlendAnchorAlignedScale :: !Double
    , ecBlendTensionConflictShrink :: !Double
    , ecBlendTensionNeutralShrink :: !Double
    , ecBlendEntropyConflictFloor :: !Double
    , ecBlendEntropyConflictScale :: !Double
    , ecBlendEntropyAlignedBase :: !Double
    , ecBlendEntropyAlignedEntropyScale :: !Double
    , ecBlendPhaseCancelReturnClamp :: !Double
    , ecBlendPhaseCancelConflictFloor :: !Double
    , ecBlendPhaseCancelConflictScale :: !Double
    , ecBlendPhaseCancelAlignmentScale :: !Double
    , ecKalmanDt :: !Double
    , ecKalmanProcessVar :: !Double
    , ecKalmanMeasurementVar :: !Double
    , ecTriLayer :: !Bool
    , ecTriLayerFastMult :: !Double
    , ecTriLayerSlowMult :: !Double
    , ecTriLayerCloudPadding :: !Double
    , ecTriLayerCloudSlope :: !Double
    , ecTriLayerCloudWidth :: !Double
    , ecTriLayerTouchLookback :: !Int
    , ecTriLayerRequirePriceAction :: !Bool
    , ecTriLayerPriceActionBody :: !Double
    , ecTriLayerPriceActionBodyOpenThresholdMult :: !Double
    , ecTriLayerPriceActionWickRatio :: !Double
    , ecTriLayerPriceActionOppositeWickMax :: !Double
    , ecTriLayerPriceActionBodyTolerance :: !Double
    , ecTriLayerExitOnSlow :: !Bool
    , ecKalmanBandLookback :: !Int
    , ecKalmanBandStdMult :: !Double
    , ecKalmanZMin :: !Double
    , ecKalmanZMax :: !Double
    , ecKalmanMinStdFloor :: !Double
    , ecLstmExitFlipBars :: !Int
    , ecLstmExitFlipGraceBars :: !Int
    , ecLstmExitFlipStrong :: !Bool
    , ecLstmConfidenceSoft :: !Double
    , ecLstmConfidenceHard :: !Double
    , ecMaxHighVolProb :: !(Maybe Double)
    , ecMaxConformalWidth :: !(Maybe Double)
    , ecMaxQuantileWidth :: !(Maybe Double)
    , ecConfirmConformal :: !Bool
    , ecConfirmQuantiles :: !Bool
    , ecConfidenceSizing :: !Bool
    , ecMinPositionSize :: !Double
    , ecKellyLiteSizing :: !Bool
    , ecKellyLiteFraction :: !Double
    , ecKellyLiteFloor :: !Double
    , ecKellyLiteCap :: !Double
    }
    deriving (Eq, Show)

data StepMeta = StepMeta
    { smKalmanMean :: !Double
    , smKalmanVar :: !Double
    , smHighVolProb :: !(Maybe Double)
    , smConformalLo :: !(Maybe Double)
    , smConformalHi :: !(Maybe Double)
    , smQuantile10 :: !(Maybe Double)
    , smQuantile90 :: !(Maybe Double)
    }
    deriving (Eq, Show)

data IntrabarFill = StopFirst | TakeProfitFirst
    deriving (Eq, Show)

data PositionSide = PositionLong | PositionShort
    deriving (Eq, Show)

pattern SideLong :: PositionSide
pattern SideLong = PositionLong

pattern SideShort :: PositionSide
pattern SideShort = PositionShort

{-# COMPLETE SideLong, SideShort #-}

sideSign :: PositionSide -> Double
sideSign side =
    case side of
        SideLong -> 1
        SideShort -> -1

data Positioning = LongFlat | LongShort
    deriving (Eq, Show)

class ToDoubleVector a where
    toDoubleVector :: a -> V.Vector Double

instance ToDoubleVector [Double] where
    toDoubleVector = V.fromList

instance ToDoubleVector (V.Vector Double) where
    toDoubleVector = id

class ToStepMetaVector a where
    toStepMetaVector :: a -> Maybe (V.Vector StepMeta)

instance ToStepMetaVector (Maybe [StepMeta]) where
    toStepMetaVector = fmap V.fromList

instance ToStepMetaVector (Maybe (V.Vector StepMeta)) where
    toStepMetaVector = id

simulateEnsemble ::
    ( ToDoubleVector prices
    , ToDoubleVector highs
    , ToDoubleVector lows
    , ToDoubleVector kalPredictions
    , ToDoubleVector lstmPredictions
    , ToStepMetaVector meta
    ) =>
    EnsembleConfig ->
    Int ->
    prices ->
    highs ->
    lows ->
    kalPredictions ->
    lstmPredictions ->
    meta ->
    BacktestResult
simulateEnsemble cfg periods series0 series1 series2 series3 series4 meta =
    case simulateEnsembleWithHLChecked cfg periods series0 series1 series2 series3 series4 meta of
        Left err -> error ("Trader.Trading.simulateEnsemble: " ++ err)
        Right result -> result

simulateEnsembleWithHLChecked ::
    ( ToDoubleVector prices
    , ToDoubleVector highs
    , ToDoubleVector lows
    , ToDoubleVector kalPredictions
    , ToDoubleVector lstmPredictions
    , ToStepMetaVector meta
    ) =>
    EnsembleConfig ->
    Int ->
    prices ->
    highs ->
    lows ->
    kalPredictions ->
    lstmPredictions ->
    meta ->
    Either String BacktestResult
simulateEnsembleWithHLChecked cfg periods series0 series1 series2 series3 series4 meta =
    simulateEnsembleVWithHLChecked
        cfg
        periods
        (toDoubleVector series0)
        (toDoubleVector series1)
        (toDoubleVector series2)
        (toDoubleVector series3)
        (toDoubleVector series4)
        (toStepMetaVector meta)

simulateEnsembleVWithHLChecked ::
    EnsembleConfig ->
    Int ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Either String BacktestResult
simulateEnsembleVWithHLChecked = simulateEnsembleLongFlatVWithHLChecked

data ExitReason
    = ExitSignal
    | ExitStopLoss
    | ExitTrailingStop
    | ExitTakeProfit
    | ExitMaxDrawdown
    | ExitMaxDailyLoss
    | ExitMaxWeeklyLoss
    | ExitLiquidation
    | ExitEod
    | ExitOther !String
    deriving (Eq, Show)

exitReasonCode :: ExitReason -> String
exitReasonCode exitReason =
    case exitReason of
        ExitSignal -> "SIGNAL"
        ExitStopLoss -> "STOP_LOSS"
        ExitTrailingStop -> "TRAILING_STOP"
        ExitTakeProfit -> "TAKE_PROFIT"
        ExitMaxDrawdown -> "MAX_DRAWDOWN"
        ExitMaxDailyLoss -> "MAX_DAILY_LOSS"
        ExitMaxWeeklyLoss -> "MAX_WEEKLY_LOSS"
        ExitLiquidation -> "LIQUIDATION"
        ExitEod -> "EOD"
        ExitOther other -> other

exitReasonFromCode :: String -> Maybe ExitReason
exitReasonFromCode code
    | normalized == "SIGNAL" = Just ExitSignal
    | normalized == "STOP_LOSS" = Just ExitStopLoss
    | normalized == "TRAILING_STOP" = Just ExitTrailingStop
    | normalized == "TAKE_PROFIT" = Just ExitTakeProfit
    | normalized == "MAX_DRAWDOWN" = Just ExitMaxDrawdown
    | normalized == "MAX_DAILY_LOSS" = Just ExitMaxDailyLoss
    | normalized == "MAX_WEEKLY_LOSS" = Just ExitMaxWeeklyLoss
    | normalized == "LIQUIDATION" = Just ExitLiquidation
    | normalized == "EOD" = Just ExitEod
    | otherwise = Nothing
  where
    normalized = map toUpper code

instance Aeson.ToJSON ExitReason where
    toJSON = Aeson.String . T.pack . exitReasonCode

{- | Inputs to the risk halt decision, extracted from the simulation loop.
This record is the canonical interface between the simulation and the
formal risk spec; both call sites (the live simulation and the
'Trader.Formal.Risk' property checks) consume it via 'specRiskHalt'.
-}
data HaltInputs = HaltInputs
    { hiPrevHaltReason :: !(Maybe ExitReason)
    , hiDayChanged :: !Bool
    , hiWeekChanged :: !Bool
    , hiDailyLoss :: !Double
    , hiWeeklyLoss :: !Double
    , hiDrawdown :: !Double
    , hiExpectancy :: !(Maybe Double)
    , hiMaxDailyLossLim :: !(Maybe Double)
    , hiMaxWeeklyLossLim :: !(Maybe Double)
    , hiMaxDrawdownLim :: !(Maybe Double)
    , hiMinExpectancy :: !(Maybe Double)
    , hiPositionSize :: !Double
    , hiMaxPositionSizeLim :: !(Maybe Double)
    , hiConsecutiveLosses :: !Int
    , hiMaxLossStreakLim :: !(Maybe Int)
    , hiVolTarget :: !Double
    , hiLeverage :: !Double
    }
    deriving (Eq, Show)

{- | Canonical risk halt spec, shared between the simulation loop and the
formal verification module.

1. If previously halted for daily loss and the day changed, reset.
2. If previously halted for weekly loss and the week changed, reset.
3. Otherwise preserve the previous halt.
4. If still not halted, check new risk conditions in fixed priority order:
   daily loss, weekly loss, drawdown, negative expectancy.

By calling this function directly from 'simulateEnsembleLongFlatVWithHLChecked',
the simulation implementation cannot drift from the spec — they share code.
Properties of this function are proven exhaustively in
'Trader.Formal.Risk.verifyFormalRisk'.
-}

{- | Sanitize a limit to be non-negative. Negative limits are treated as
zero (most restrictive) to prevent nonsensical risk configurations from
silently disabling halts.
-}
sanitizeRiskLimit :: Double -> Double
sanitizeRiskLimit = max 0

{- | FIRM-CRITICAL position-size parity constant. Both the simulation sizing
path (here in 'Trader.Trading') and the live sizing path in @Main@ must hard-fail
with @POSITION_SIZE_SCALE_EXCEEDED@ when overlayed size exceeds this multiple of
the base size, before the absolute cap can mask runaway multiplicative sizing.
Keeping it a single exported constant prevents the two guardrails from silently
drifting apart.
-}
positionSizeScaleHardFailMultiplier :: Double
positionSizeScaleHardFailMultiplier = 2.0

defaultTriLayerPriceActionBodyOpenThresholdMult :: Double
defaultTriLayerPriceActionBodyOpenThresholdMult = 0.25

{- | Check whether any configured risk limit is non-finite (NaN or Infinity).
This is a FIRM-CRITICAL hygiene gate: non-finite limits silently disable
halt checks and allow unbounded losses.
-}
anyRiskLimitNonFinite :: HaltInputs -> Bool
anyRiskLimitNonFinite hi =
    any
        (maybe False (not . finiteDouble))
        [ hiMaxDailyLossLim hi
        , hiMaxWeeklyLossLim hi
        , hiMaxDrawdownLim hi
        , hiMinExpectancy hi
        , hiMaxPositionSizeLim hi
        ]

{- | Check whether the drawdown limit is outside the valid (0,1) interval.
This is a FIRM-CRITICAL hygiene gate: a drawdown limit of 0 would halt on
any non-negative drawdown, and a limit >=1 would never halt, both of which
indicate corrupted configuration.
-}
drawdownLimitInvalid :: HaltInputs -> Bool
drawdownLimitInvalid hi =
    case hiMaxDrawdownLim hi of
        Just v -> v <= 0 || v >= 1 || not (finiteDouble v)
        Nothing -> False

specRiskHalt :: HaltInputs -> Maybe ExitReason
specRiskHalt hi =
    let haltReasonBase =
            case (hiPrevHaltReason hi, hiDayChanged hi, hiWeekChanged hi) of
                (Just ExitMaxDailyLoss, True, _) -> Nothing
                (Just ExitMaxWeeklyLoss, _, True) -> Nothing
                _ -> hiPrevHaltReason hi
        riskHaltReason =
            case haltReasonBase of
                Just _ -> Nothing
                Nothing
                    | anyRiskLimitNonFinite hi ->
                        Just (ExitOther "RISK_LIMIT_NON_FINITE")
                    | drawdownLimitInvalid hi ->
                        Just (ExitOther "DRAWDOWN_LIMIT_INVALID")
                    | not (finiteDouble (hiPositionSize hi)) || hiPositionSize hi < 0 || hiPositionSize hi > 10 ->
                        Just (ExitOther "POSITION_SIZE_INVALID")
                    | isJust (hiMinExpectancy hi) && (isNothing (hiExpectancy hi) || maybe False (not . finiteDouble) (hiExpectancy hi)) ->
                        Just (ExitOther "EXPECTANCY_INVALID")
                    | not (finiteDouble (hiVolTarget hi)) || hiVolTarget hi < 0 || hiVolTarget hi > 10 ->
                        Just (ExitOther "VOL_TARGET_INVALID")
                    | not (finiteDouble (hiLeverage hi)) || hiLeverage hi < 0 || hiLeverage hi > 150 ->
                        Just (ExitOther "LEVERAGE_INVALID")
                    | otherwise ->
                        let mdl = fmap sanitizeRiskLimit (hiMaxDailyLossLim hi)
                            mwl = fmap sanitizeRiskLimit (hiMaxWeeklyLossLim hi)
                            mdd = fmap sanitizeRiskLimit (hiMaxDrawdownLim hi)
                            me = fmap sanitizeRiskLimit (hiMinExpectancy hi)
                            mps = fmap sanitizeRiskLimit (hiMaxPositionSizeLim hi)
                            ps = max 0 (hiPositionSize hi)
                         in case () of
                                _
                                    | maybe False (hiDailyLoss hi >=) mdl ->
                                        Just ExitMaxDailyLoss
                                    | maybe False (hiWeeklyLoss hi >=) mwl ->
                                        Just ExitMaxWeeklyLoss
                                    | maybe False (hiDrawdown hi >=) mdd ->
                                        Just ExitMaxDrawdown
                                    | maybe False (\lim -> maybe False (< lim) (hiExpectancy hi)) me ->
                                        Just (ExitOther "NEGATIVE_EXPECTANCY")
                                    | maybe False (ps >) mps ->
                                        Just (ExitOther "POSITION_SIZE")
                                    | maybe False (\lim -> lim > 0 && hiConsecutiveLosses hi > lim) (hiMaxLossStreakLim hi) ->
                                        Just (ExitOther "LOSS_STREAK")
                                    | otherwise -> Nothing
     in haltReasonBase <|> riskHaltReason

data TradeEntrySource
    = TradeEntrySignal
    | TradeEntryAdopted
    | TradeEntryPostDirectionGates
    deriving (Eq, Show)

tradeEntrySourceCode :: TradeEntrySource -> String
tradeEntrySourceCode entrySource =
    case entrySource of
        TradeEntrySignal -> "signal"
        TradeEntryAdopted -> "adopted"
        TradeEntryPostDirectionGates -> "post_direction_gates"

data Trade = Trade
    { trEntryIndex :: !Int
    , trExitIndex :: !Int
    , trEntryEquity :: !Double
    , trExitEquity :: !Double
    , trReturn :: !Double
    , trHoldingPeriods :: !Int
    , trEntryHighVolProb :: !(Maybe Double)
    , trEntrySource :: !TradeEntrySource
    , trExitReason :: !(Maybe ExitReason)
    , trEntryIp :: !(Maybe T.Text)
    , trExitIp :: !(Maybe T.Text)
    , trFeeCost :: !Double
    }
    deriving (Eq, Show)

{- | Outcome-driven LSTM learning: extra loss weight per unit of realized
return for bars covered by a winning trade. With a 2% win the bars it spanned
train at weight 1.2 — consolidating the pattern that paid.
-}
outcomeWeightWinScale :: Double
outcomeWeightWinScale = 10

{- | Extra loss weight per unit of realized loss. Deliberately larger than
'outcomeWeightWinScale': a 2% losing trade trains its bars at weight 1.5, so
the model corrects hardest exactly where its predictions cost money.
-}
outcomeWeightLossScale :: Double
outcomeWeightLossScale = 25

{- | Ceiling on any single bar's outcome weight, so one extreme trade cannot
dominate the fine-tune window.
-}
outcomeWeightCap :: Double
outcomeWeightCap = 3

{- | Loss-weight multiplier a closed trade assigns to the bars it spanned.
Nothing for break-even or non-finite returns (no signal to learn from).
-}
tradeOutcomeWeightFactor :: Trade -> Maybe Double
tradeOutcomeWeightFactor tr =
    let r = trReturn tr
     in if isNaN r || isInfinite r || r == 0
            then Nothing
            else
                let scale = if r < 0 then outcomeWeightLossScale else outcomeWeightWinScale
                 in Just (min outcomeWeightCap (1 + scale * abs r))

{- | Per-bar loss weights for a price/observation series of length @n@,
derived from closed trades whose entry/exit indices live in the same index
space as the series. Bars not covered by any trade weigh 1. Bots hold one
position at a time, so spans effectively don't overlap; where they touch
(close + reopen on the same bar) the later trade in the list wins.
-}
tradeOutcomeWeights :: [Trade] -> Int -> [Double]
tradeOutcomeWeights trades n
    | n <= 0 = []
    | otherwise =
        let updatesFor tr =
                case tradeOutcomeWeightFactor tr of
                    Nothing -> []
                    Just f ->
                        let lo = max 0 (trEntryIndex tr)
                            hi = min (n - 1) (trExitIndex tr)
                         in [(t, f) | lo <= hi, t <- [lo .. hi]]
         in V.toList (V.replicate n 1 V.// concatMap updatesFor trades)

data BacktestCostAttribution = BacktestCostAttribution
    { bcaGrossEquityCurve :: [Double]
    , bcaNetEquityCurve :: [Double]
    , bcaRealizedFeeCost :: !Double
    , bcaRealizedSlippageCost :: !Double
    , bcaRealizedSpreadCost :: !Double
    , bcaRealizedFundingCost :: !Double
    , bcaRealizedTotalCost :: !Double
    , bcaConsistencyResidual :: !Double
    }
    deriving (Eq, Show)

data CostTotals = CostTotals
    { ctFeeCost :: !Double
    , ctSlippageCost :: !Double
    , ctSpreadCost :: !Double
    , ctFundingCost :: !Double
    }
    deriving (Eq, Show)

emptyCostTotals :: CostTotals
emptyCostTotals = CostTotals 0 0 0 0

addCostTotals :: CostTotals -> CostTotals -> CostTotals
addCostTotals left right =
    CostTotals
        { ctFeeCost = ctFeeCost left + ctFeeCost right
        , ctSlippageCost = ctSlippageCost left + ctSlippageCost right
        , ctSpreadCost = ctSpreadCost left + ctSpreadCost right
        , ctFundingCost = ctFundingCost left + ctFundingCost right
        }

scaleCostTotals :: Double -> CostTotals -> CostTotals
scaleCostTotals factor totals =
    CostTotals
        { ctFeeCost = factor * ctFeeCost totals
        , ctSlippageCost = factor * ctSlippageCost totals
        , ctSpreadCost = factor * ctSpreadCost totals
        , ctFundingCost = factor * ctFundingCost totals
        }

costTotalsTotal :: CostTotals -> Double
costTotalsTotal totals =
    ctFeeCost totals
        + ctSlippageCost totals
        + ctSpreadCost totals
        + ctFundingCost totals

finiteCost :: Double -> Double
finiteCost x = if isNaN x || isInfinite x then 0 else x

costAttributionFromTotals :: [Double] -> [CostTotals] -> BacktestCostAttribution
costAttributionFromTotals netCurve0 cumulativeCosts0 =
    let netCurve = map finiteCost netCurve0
        fallbackCost = case cumulativeCosts0 of
            [] -> emptyCostTotals
            _ -> last cumulativeCosts0
        costsAligned = take (length netCurve) (cumulativeCosts0 ++ repeat fallbackCost)
        grossCurve = zipWith (\net costs -> net + costTotalsTotal costs) netCurve costsAligned
        finalNet = case netCurve of
            [] -> 1
            _ -> last netCurve
        finalCosts = case costsAligned of
            [] -> emptyCostTotals
            _ -> last costsAligned
        totalCost = costTotalsTotal finalCosts
        finalGross = case grossCurve of
            [] -> finalNet + totalCost
            _ -> last grossCurve
     in BacktestCostAttribution
            { bcaGrossEquityCurve = grossCurve
            , bcaNetEquityCurve = netCurve
            , bcaRealizedFeeCost = ctFeeCost finalCosts
            , bcaRealizedSlippageCost = ctSlippageCost finalCosts
            , bcaRealizedSpreadCost = ctSpreadCost finalCosts
            , bcaRealizedFundingCost = ctFundingCost finalCosts
            , bcaRealizedTotalCost = totalCost
            , bcaConsistencyResidual = finalGross - totalCost - finalNet
            }

emptyBacktestCostAttribution :: [Double] -> BacktestCostAttribution
emptyBacktestCostAttribution netCurve = costAttributionFromTotals netCurve []

data OpenTrade = OpenTrade
    { otEntryIndex :: !Int
    , otRebalanceAnchor :: !Int
    , otEntryEquity :: !Double
    , otHoldingPeriods :: !Int
    , otEntryPrice :: !Double
    , otTrail :: !Double
    , otSide :: !PositionSide
    , otBaseSize :: !Double
    , otLstmFlipCount :: !Int
    }
    deriving (Eq, Show)

data RiskState = RiskState
    { rsPeakEquity :: !Double
    , rsDayKey :: !Int
    , rsDayStartEquity :: !Double
    , rsWeekKey :: !Int
    , rsWeekStartEquity :: !Double
    , rsDayTrades :: !Int
    , rsHaltReason :: !(Maybe ExitReason)
    }
    deriving (Eq, Show)

data BacktestResult = BacktestResult
    { brEquityCurve :: [Double]
    , brTrades :: [Trade]
    , brPositions :: [Double]
    , brAgreementOk :: [Bool]
    , brAgreementValid :: [Bool]
    , brPositionChanges :: Int
    , brCostAttribution :: BacktestCostAttribution
    }
    deriving (Eq, Show)

simulateEnsembleLongFlatVWithHLChecked ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double -> -- closes
    V.Vector Double -> -- highs
    V.Vector Double -> -- lows
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    Either String BacktestResult
simulateEnsembleLongFlatVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
    let n = V.length pricesV
        startT = max 0 (lookback - 1)
        stepCount = n - 1
        kalLen = V.length kalPredNextV
        kalNeed = max 0 (stepCount - startT)
        openTimesV =
            case ecOpenTimes cfg of
                Just ts | V.length ts == n -> Just ts
                _ -> Nothing
        openPricesV =
            case ecOpenPrices cfg of
                Just os | V.length os == n -> Just os
                _ -> Nothing
        openTimesMismatch =
            case ecOpenTimes cfg of
                Just ts
                    | V.length ts /= n ->
                        Just
                            ( "open time vector length ("
                                ++ show (V.length ts)
                                ++ ") must match closes length ("
                                ++ show n
                                ++ ")"
                            )
                _ -> Nothing
        openPricesMismatch =
            case ecOpenPrices cfg of
                Just os
                    | V.length os /= n ->
                        Just
                            ( "open price vector length ("
                                ++ show (V.length os)
                                ++ ") must match closes length ("
                                ++ show n
                                ++ ")"
                            )
                _ -> Nothing
        hasCalendar =
            case openTimesV of
                Just _ -> True
                Nothing -> False
        hasDailyKey = hasCalendar
        hasWeeklyKey = hasCalendar
        dailyLossReq =
            case ecMaxDailyLoss cfg of
                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        weeklyLossReq =
            case ecMaxWeeklyLoss cfg of
                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        maxTradesPerDayReq =
            case ecMaxTradesPerDay cfg of
                Just n | n > 0 -> Just n
                _ -> Nothing
        expectancyLookback = max 0 (ecExpectancyLookback cfg)
        minExpectancy =
            case ecMinExpectancy cfg of
                Just v | not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        perfLookback = max 0 (ecPerfLookback cfg)
        perfMinWinRate =
            case ecPerfMinWinRate cfg of
                Just v
                    | not (isNaN v || isInfinite v) ->
                        Just (max 0 (min 1 v))
                _ -> Nothing
        perfMinProfitFactor =
            case ecPerfMinProfitFactor cfg of
                Just v
                    | not (isNaN v || isInfinite v) && v >= 0 ->
                        Just v
                _ -> Nothing
        adaptiveFilters = ecAdaptiveFilters cfg
        adaptiveEdgeBufferMax = max 0 (ecAdaptiveEdgeBufferMax cfg)
        adaptiveMinSnrMax = max 0 (ecAdaptiveMinSignalToNoiseMax cfg)
        adaptiveKalmanZMinMax = max 0 (ecAdaptiveKalmanZMinMax cfg)
        adaptiveWinRateSlack = max 0 (ecAdaptiveWinRateSlack cfg)
        adaptiveProfitFactorSlack = max 0 (ecAdaptiveProfitFactorSlack cfg)
        adaptiveTrendLookbackMax = max 0 (ecAdaptiveTrendLookbackMax cfg)
        lossStreakMax = max 0 (ecLossStreakMax cfg)
        lossStreakCooldownBars = max 0 (ecLossStreakCooldownBars cfg)
        noTradeWindows = ecNoTradeWindows cfg
        noTradeReq = not (null noTradeWindows)
        metaMaskV =
            case ecMetaMask cfg of
                Just mask | V.length mask == stepCount -> Just mask
                _ -> Nothing
        metaMaskMismatch =
            case ecMetaMask cfg of
                Just mask | V.length mask /= stepCount -> Just "meta mask vector length must match step count for simulateEnsembleLongFlatVWithHL"
                _ -> Nothing
        metaV =
            case mMetaV of
                Just mv | V.length mv == stepCount -> Just mv
                _ -> Nothing
        metaMismatch =
            case mMetaV of
                Just mv | V.length mv /= stepCount -> Just "meta vector length must match step count for simulateEnsembleLongFlatVWithHL"
                _ -> Nothing
        kalPredAtE
            | kalLen >= stepCount =
                let dropCount = kalLen - stepCount
                    v = if dropCount == 0 then kalPredNextV else V.drop dropCount kalPredNextV
                 in Right (v V.!)
            | kalLen >= kalNeed =
                let dropCount = kalLen - kalNeed
                    v = if dropCount == 0 then kalPredNextV else V.drop dropCount kalPredNextV
                 in Right (\t -> v V.! (t - startT))
            | otherwise =
                Left
                    ( "kalPredNext too short: need at least "
                        ++ show kalNeed
                        ++ " (or "
                        ++ show stepCount
                        ++ " for full alignment), got "
                        ++ show kalLen
                    )
        lstmLen = V.length lstmPredNextV
        lstmNeed = max 0 (stepCount - startT)
        lstmPredAtE
            | lstmLen >= stepCount =
                let dropCount = lstmLen - stepCount
                    v = if dropCount == 0 then lstmPredNextV else V.drop dropCount lstmPredNextV
                 in Right (v V.!)
            | lstmLen >= lstmNeed =
                let dropCount = lstmLen - lstmNeed
                    v = if dropCount == 0 then lstmPredNextV else V.drop dropCount lstmPredNextV
                 in Right (\t -> v V.! (t - startT))
            | otherwise =
                Left
                    ( "lstmPredNext too short: need at least "
                        ++ show lstmNeed
                        ++ " (or "
                        ++ show stepCount
                        ++ " for full alignment), got "
                        ++ show lstmLen
                    )
        validationError
            | n < 2 = Just "Need at least 2 prices to simulate"
            | V.length highsV /= n || V.length lowsV /= n = Just "high/low vectors must match closes length"
            | lookback >= n = Just "lookback must be less than number of prices"
            | (isJust dailyLossReq || isJust weeklyLossReq || isJust maxTradesPerDayReq || noTradeReq) && not hasDailyKey = Just "--max-daily-loss/--max-weekly-loss/--max-trades-per-day/--no-trade-window require bar timestamps"
            | isJust minExpectancy && expectancyLookback <= 0 = Just "--min-expectancy requires --expectancy-lookback >= 1"
            | otherwise = case openTimesMismatch of
                Just err
                    | isJust dailyLossReq || isJust weeklyLossReq || isJust maxTradesPerDayReq || noTradeReq -> Just err
                _ -> case openPricesMismatch of
                    Just err -> Just err
                    Nothing ->
                        case metaMaskMismatch of
                            Just err -> Just err
                            Nothing ->
                                case metaMismatch of
                                    Just err -> Just err
                                    Nothing ->
                                        case kalPredAtE of
                                            Left err -> Just err
                                            Right _ ->
                                                case lstmPredAtE of
                                                    Left err -> Just err
                                                    Right _ -> Nothing
     in case validationError of
            Just err -> Left err
            Nothing ->
                let positionSizeBoundsOk =
                        finiteDouble (ecMaxPositionSize cfg)
                            && ecMaxPositionSize cfg > 0
                            && finiteDouble (ecMinPositionSize cfg)
                            && ecMinPositionSize cfg >= 0
                 in if not positionSizeBoundsOk
                        then Left "maxPositionSize must be finite and > 0 (risk guardrail: zero position size would silently disable all trades)"
                        else
                            Right $
                                let startT = max 0 (lookback - 1)
                                    openThrRawBase = max 0 (ecOpenThreshold cfg)
                                    minEdgeBase = max 0 (ecMinEdge cfg)
                                    minSignalToNoiseBase = max 0 (ecMinSignalToNoise cfg)
                                    snrSizeWeight = clamp01 (ecSnrSizeWeight cfg)
                                    factorEnabled = ecThresholdFactorEnabled cfg
                                    factorAlpha = clamp01 (ecThresholdFactorAlpha cfg)
                                    factorMinRaw = ecThresholdFactorMin cfg
                                    factorMaxRaw = ecThresholdFactorMax cfg
                                    factorMin = max 1e-6 (min factorMinRaw factorMaxRaw)
                                    factorMax = max factorMin (max factorMinRaw factorMaxRaw)
                                    factorFloor = max 0 (ecThresholdFactorFloor cfg)
                                    factorWEdgeKal = ecThresholdFactorEdgeKalWeight cfg
                                    factorWEdgeLstm = ecThresholdFactorEdgeLstmWeight cfg
                                    factorWKalmanZ = ecThresholdFactorKalmanZWeight cfg
                                    factorWHighVol = ecThresholdFactorHighVolWeight cfg
                                    factorWConformal = ecThresholdFactorConformalWeight cfg
                                    factorWQuantile = ecThresholdFactorQuantileWeight cfg
                                    factorWLstmConf = ecThresholdFactorLstmConfWeight cfg
                                    factorWLstmHealth = ecThresholdFactorLstmHealthWeight cfg
                                    lstmHealthScore =
                                        case ecLstmTrainingHealth cfg of
                                            Just v | not (isBad v) -> clamp01 v
                                            _ -> 0.5
                                    openThr0 = normalizeSignalThreshold (max openThrRawBase minEdgeBase)
                                    priceActionBodyMin = max 0 (ecTriLayerPriceActionBody cfg)
                                    priceActionBodyOpenThresholdMult = max 0 (ecTriLayerPriceActionBodyOpenThresholdMult cfg)
                                    priceActionWickRatio = max 0 (ecTriLayerPriceActionWickRatio cfg)
                                    priceActionOppositeWickMax = max 0 (ecTriLayerPriceActionOppositeWickMax cfg)
                                    priceActionBodyTolerance = max 0 (ecTriLayerPriceActionBodyTolerance cfg)
                                    bodyMinFracBase = max 1e-6 (priceActionBodyOpenThresholdMult * openThr0)
                                    bodyMinFrac =
                                        if priceActionBodyMin > 0
                                            then priceActionBodyMin
                                            else bodyMinFracBase
                                    closeThrBase = normalizeSignalThreshold (max 0 (ecCloseThreshold cfg))
                                    minHoldBars = max 0 (ecMinHoldBars cfg)
                                    cooldownBars = max 0 (ecCooldownBars cfg)
                                    maxHoldBars =
                                        case ecMaxHoldBars cfg of
                                            Just v | v > 0 -> Just v
                                            _ -> Nothing
                                    maxDrawdownLim =
                                        case ecMaxDrawdown cfg of
                                            Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                                            _ -> Nothing
                                    maxDailyLossLim =
                                        case dailyLossReq of
                                            Just v | hasDailyKey -> Just v
                                            _ -> Nothing
                                    maxWeeklyLossLim =
                                        case weeklyLossReq of
                                            Just v | hasWeeklyKey -> Just v
                                            _ -> Nothing
                                    maxTradesPerDayLim =
                                        case maxTradesPerDayReq of
                                            Just n | hasDailyKey && n > 0 -> Just n
                                            _ -> Nothing
                                    riskPerTrade =
                                        case ecRiskPerTrade cfg of
                                            Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                                            _ -> Nothing
                                    noTradeWindows' = noTradeWindows
                                    positionSizeBoundsOk =
                                        finiteDouble (ecMaxPositionSize cfg)
                                            && ecMaxPositionSize cfg >= 0
                                            && finiteDouble (ecMinPositionSize cfg)
                                            && ecMinPositionSize cfg >= 0
                                    maxPositionSize =
                                        if positionSizeBoundsOk
                                            then ecMaxPositionSize cfg
                                            else 0
                                    minPositionSize =
                                        if positionSizeBoundsOk
                                            then ecMinPositionSize cfg
                                            else 0
                                    rebalanceBars = max 0 (ecRebalanceBars cfg)
                                    rebalanceThreshold = max 0 (ecRebalanceThreshold cfg)
                                    rebalanceGlobal = ecRebalanceGlobal cfg
                                    rebalanceResetOnSignal = ecRebalanceResetOnSignal cfg
                                    rebalanceEnabled = rebalanceBars > 0 && rebalanceThreshold > 0
                                    trendLookbackBase = max 0 (ecTrendLookback cfg)
                                    kalmanZMinBase = max 0 (ecKalmanZMin cfg)
                                    ppy = max 1e-12 (ecPeriodsPerYear cfg)
                                    fundingRate =
                                        let r = ecFundingRate cfg
                                         in if isNaN r || isInfinite r then 0 else r
                                    fundingBySide = ecFundingBySide cfg
                                    fundingOnOpen = ecFundingOnOpen cfg
                                    triLayerEnabled = ecTriLayer cfg
                                    kalDt = max 1e-12 (ecKalmanDt cfg)
                                    kalProcessVar = max 0 (ecKalmanProcessVar cfg)
                                    kalMeasVar = max 1e-12 (ecKalmanMeasurementVar cfg)
                                    fastMult = max 1e-6 (ecTriLayerFastMult cfg)
                                    slowMult = max 1e-6 (ecTriLayerSlowMult cfg)
                                    cloudPadFrac = max 0 (ecTriLayerCloudPadding cfg)
                                    cloudSlopeMin = max 0 (ecTriLayerCloudSlope cfg)
                                    cloudWidthMax = max 0 (ecTriLayerCloudWidth cfg)
                                    touchLookback = max 1 (ecTriLayerTouchLookback cfg)
                                    requirePriceAction = ecTriLayerRequirePriceAction cfg
                                    triLayerExitOnSlow = ecTriLayerExitOnSlow cfg
                                    kalmanBandLookback = max 0 (ecKalmanBandLookback cfg)
                                    kalmanBandStdMult = max 0 (ecKalmanBandStdMult cfg)
                                    kalmanBandEnabled =
                                        n >= 2 && kalmanBandStdMult > 0 && kalmanBandLookback >= 2
                                    kalmanCloudEnabled = (triLayerEnabled || kalmanBandEnabled) && n >= 2
                                    cloudReady = triLayerEnabled && n >= 2
                                    lstmFlipBars = max 0 (ecLstmExitFlipBars cfg)
                                    lstmFlipGraceBars = max 0 (ecLstmExitFlipGraceBars cfg)
                                    lstmFlipStrongOnly = ecLstmExitFlipStrong cfg
                                    (cloudFastV, cloudSlowV) =
                                        if kalmanCloudEnabled
                                            then
                                                let run mult =
                                                        let mv = max 1e-12 (kalMeasVar * mult)
                                                            KalmanRunV{krFilteredV = filts} =
                                                                runConstantAcceleration1DVec kalDt kalProcessVar mv pricesV
                                                         in filts
                                                 in (run fastMult, run slowMult)
                                            else (pricesV, pricesV)
                                    kalmanResidualV =
                                        if kalmanBandEnabled
                                            then V.zipWith (-) pricesV cloudSlowV
                                            else V.empty
                                    kalmanResidualPrefix =
                                        if kalmanBandEnabled
                                            then V.scanl' (+) 0 kalmanResidualV
                                            else V.empty
                                    kalmanResidualSqPrefix =
                                        if kalmanBandEnabled
                                            then V.scanl' (+) 0 (V.map (\x -> x * x) kalmanResidualV)
                                            else V.empty
                                    kalmanResidualStdAt :: Int -> Maybe Double
                                    kalmanResidualStdAt t =
                                        if not kalmanBandEnabled
                                            then Nothing
                                            else
                                                let end = min (n - 1) t
                                                    start = max 0 (end - kalmanBandLookback + 1)
                                                    len = end - start + 1
                                                 in if len < 2
                                                        then Nothing
                                                        else
                                                            let sumR = (kalmanResidualPrefix V.! (end + 1)) - (kalmanResidualPrefix V.! start)
                                                                sumSq = (kalmanResidualSqPrefix V.! (end + 1)) - (kalmanResidualSqPrefix V.! start)
                                                                lenF = fromIntegral len
                                                                mean = sumR / lenF
                                                                var = max 0 ((sumSq - lenF * mean * mean) / fromIntegral (len - 1))
                                                                std = sqrt var
                                                             in if isBad std then Nothing else Just std
                                    kalmanBandAt :: Int -> Maybe (Double, Double)
                                    kalmanBandAt t =
                                        case kalmanResidualStdAt t of
                                            Nothing -> Nothing
                                            Just std ->
                                                let slow = cloudSlowV V.! t
                                                    upper = slow + kalmanBandStdMult * std
                                                    lower = slow - kalmanBandStdMult * std
                                                 in if isBad slow || isBad upper || isBad lower
                                                        then Nothing
                                                        else Just (upper, lower)
                                    fundingPerBar =
                                        if fundingRate == 0
                                            then 0
                                            else fundingRate / ppy
                                    volTarget =
                                        case ecVolTarget cfg of
                                            Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                                            _ -> Nothing
                                    volLookback = max 0 (ecVolLookback cfg)
                                    volFloor = max 0 (ecVolFloor cfg)
                                    volScaleMax = max 0 (ecVolScaleMax cfg)
                                    volConfGatePreset = ecVolConfGate cfg
                                    volConfGateConfig = ecVolConfGateConfig cfg
                                    volConfGateEnabled = volConfGatePreset /= VolConfGateDisabled
                                    confidenceSizingEnabled = ecConfidenceSizing cfg && not volConfGateEnabled
                                    kellyLiteEnabled = ecKellyLiteSizing cfg
                                    kellyLiteFraction =
                                        let v = ecKellyLiteFraction cfg
                                         in if isBad v || v < 0 then 0 else v
                                    kellyLiteFloor =
                                        let v = ecKellyLiteFloor cfg
                                         in if isBad v || v < 0 then 0 else v
                                    kellyLiteCap =
                                        let v = ecKellyLiteCap cfg
                                         in if isBad v || v < kellyLiteFloor then kellyLiteFloor else v
                                    volAlpha =
                                        case ecVolEwmaAlpha cfg of
                                            Just a | a > 0 && not (isNaN a || isInfinite a) -> Just (max 0 (min 1 a))
                                            _ -> Nothing
                                    maxVolatility =
                                        case ecMaxVolatility cfg of
                                            Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                                            _ -> Nothing
                                    dayKeyAt :: Int -> Int
                                    dayKeyAt i =
                                        case openTimesV of
                                            Just tsV
                                                | i >= 0 && i < V.length tsV ->
                                                    let dayMs = 86400000 :: Int64
                                                     in fromIntegral ((tsV V.! i) `div` dayMs)
                                            _ -> 0
                                    weekKeyAt :: Int -> Int
                                    weekKeyAt i =
                                        case openTimesV of
                                            Just tsV
                                                | i >= 0 && i < V.length tsV ->
                                                    let dayMs = 86400000 :: Int64
                                                        dayKey = (tsV V.! i) `div` dayMs
                                                        -- Monday-aligned UTC week buckets (epoch day 0 = Thursday).
                                                        mondayWeekKey = (dayKey + 3) `div` 7
                                                     in fromIntegral mondayWeekKey
                                            _ -> 0
                                    minuteOfDayAt :: Int -> Int
                                    minuteOfDayAt i =
                                        case openTimesV of
                                            Just tsV
                                                | i >= 0 && i < V.length tsV -> minuteOfDayFromMs (tsV V.! i)
                                            _ -> 0
                                    direction thr prev pred =
                                        let upEdge = prev * (1 + thr)
                                            downEdge = prev * (1 - thr)
                                         in if pred > upEdge
                                                then Just SideLong
                                                else if pred < downEdge then Just SideShort else Nothing
                                    desiredSideFromDir d =
                                        case d of
                                            SideLong -> Just SideLong
                                            SideShort ->
                                                case ecPositioning cfg of
                                                    LongFlat -> Nothing
                                                    LongShort -> Just SideShort
                                    stepCount = n - 1

                                    costPerSideBreakdown :: Int -> Double -> CostTotals
                                    costPerSideBreakdown t size =
                                        let s = max 0 (abs size)
                                         in if s <= 0
                                                then emptyCostTotals
                                                else
                                                    let vol = Data.Maybe.fromMaybe 0 (volPerBarAt t)
                                                        feeRate = max 0 (ecFee cfg)
                                                        feeFixed = max 0 (ecFeeFixed cfg)
                                                        feeMin = max 0 (ecFeeMin cfg)
                                                        slipBase = max 0 (ecSlippage cfg)
                                                        slipVol = max 0 (ecSlippageVolMult cfg) * max 0 vol
                                                        impactPower = max 0 (ecSlippageImpactPower cfg)
                                                        impactRate = max 0 (ecSlippageImpact cfg) * (s ** impactPower)
                                                        spreadBase = max 0 (ecSpread cfg)
                                                        spreadVol = max 0 (ecSpreadVolMult cfg) * max 0 vol
                                                        spreadTotal = spreadBase + spreadVol
                                                        feeCost0 = feeRate * s + feeFixed
                                                        slipCost0 = (slipBase + slipVol + impactRate) * s
                                                        spreadCost0 = (spreadTotal / 2) * s
                                                        total0 = feeCost0 + slipCost0 + spreadCost0
                                                        feeCost1 = if feeMin > 0 && total0 < feeMin then feeCost0 + (feeMin - total0) else feeCost0
                                                        total1 = feeCost1 + slipCost0 + spreadCost0
                                                        cap = 0.999999
                                                        appliedTotal
                                                            | isBad total1 = cap
                                                            | total1 <= 0 = 0
                                                            | otherwise = min cap total1
                                                        finiteNonNegative x = not (isBad x) && x >= 0
                                                        componentsFinite =
                                                            finiteNonNegative feeCost1
                                                                && finiteNonNegative slipCost0
                                                                && finiteNonNegative spreadCost0
                                                        proportionalAllocation =
                                                            let scale = appliedTotal / total1
                                                                feeCost = min appliedTotal (feeCost1 * scale)
                                                                remainingAfterFee = appliedTotal - feeCost
                                                                slipCost = min remainingAfterFee (slipCost0 * scale)
                                                                spreadCost = appliedTotal - feeCost - slipCost
                                                             in CostTotals
                                                                    { ctFeeCost = feeCost
                                                                    , ctSlippageCost = slipCost
                                                                    , ctSpreadCost = spreadCost
                                                                    , ctFundingCost = 0
                                                                    }
                                                        cappedBucketAllocation
                                                            | isBad feeCost1 || feeCost1 > 0 =
                                                                CostTotals appliedTotal 0 0 0
                                                            | isBad slipCost0 || slipCost0 > 0 =
                                                                CostTotals 0 appliedTotal 0 0
                                                            | isBad spreadCost0 || spreadCost0 > 0 =
                                                                CostTotals 0 0 appliedTotal 0
                                                            | otherwise =
                                                                CostTotals appliedTotal 0 0 0
                                                     in if appliedTotal <= 0
                                                            then emptyCostTotals
                                                            else
                                                                if componentsFinite && not (isBad total1) && total1 > 0
                                                                    then proportionalAllocation
                                                                    else cappedBucketAllocation

                                    costPerSideTotal :: Int -> Double -> Double
                                    costPerSideTotal t size = costTotalsTotal (costPerSideBreakdown t size)

                                    applyCostWithTotals :: Int -> Double -> Double -> CostTotals -> (Double, CostTotals)
                                    applyCostWithTotals t eq size totals =
                                        let costRates = costPerSideBreakdown t size
                                            costRate = costTotalsTotal costRates
                                            realized = scaleCostTotals eq costRates
                                         in (eq * (1 - costRate), addCostTotals totals realized)

                                    costPerSideRate :: Int -> Double -> Double
                                    costPerSideRate t size =
                                        let s = max 0 (abs size)
                                         in if s <= 0
                                                then 0
                                                else costPerSideTotal t s / s

                                    roundTripCostAt :: Int -> Double -> Double
                                    roundTripCostAt t size =
                                        min 0.999999 (2 * costPerSideRate t size)

                                    clampSignedFrac :: Double -> Double
                                    clampSignedFrac x =
                                        let cap = 0.999999
                                         in max (-cap) (min cap x)

                                    applyFundingWithTotals :: Double -> Double -> PositionSide -> CostTotals -> (Double, CostTotals)
                                    applyFundingWithTotals eq size side totals =
                                        if fundingPerBar == 0
                                            then (eq, totals)
                                            else
                                                let s = max 0 (abs size)
                                                    signedRate =
                                                        if fundingBySide
                                                            then fundingPerBar * s * sideSign side
                                                            else fundingPerBar * s
                                                    rate = clampSignedFrac signedRate
                                                    realized = CostTotals 0 0 0 (eq * rate)
                                                 in (eq * (1 - rate), addCostTotals totals realized)

                                    barHigh t1 =
                                        let h = highsV V.! t1
                                            c = pricesV V.! t1
                                         in if isNaN h || isInfinite h then c else h

                                    barLow t1 =
                                        let l = lowsV V.! t1
                                            c = pricesV V.! t1
                                         in if isNaN l || isInfinite l then c else l

                                    barOpen t1 =
                                        let fallback =
                                                if t1 <= 0
                                                    then pricesV V.! t1
                                                    else pricesV V.! (t1 - 1)
                                            candidate =
                                                case openPricesV of
                                                    Just ovs -> ovs V.! t1
                                                    Nothing -> fallback
                                         in if isBad candidate then fallback else candidate

                                    candleAt t1 =
                                        let o = barOpen t1
                                            c = pricesV V.! t1
                                            h = barHigh t1
                                            l = barLow t1
                                         in (o, h, l, c)

                                    candleOpen (o, _, _, _) = o
                                    candleClose (_, _, _, c) = c
                                    candleBody (o, _, _, c) = abs (c - o)
                                    candleBull (o, _, _, c) = c > o
                                    candleBear (o, _, _, c) = c < o
                                    candleUpperWick (o, h, _, c) = max 0 (h - max o c)
                                    candleLowerWick (o, _, l, c) = max 0 (min o c - l)

                                    bodyOk closePx body =
                                        let denom = max 1e-12 (abs closePx)
                                         in body / denom >= bodyMinFrac

                                    hammer candle =
                                        let body = candleBody candle
                                            upper = candleUpperWick candle
                                            lower = candleLowerWick candle
                                            closePx = candleClose candle
                                         in candleBull candle
                                                && body > 0
                                                && bodyOk closePx body
                                                && lower >= priceActionWickRatio * body
                                                && upper <= priceActionOppositeWickMax * body

                                    shootingStar candle =
                                        let body = candleBody candle
                                            upper = candleUpperWick candle
                                            lower = candleLowerWick candle
                                            closePx = candleClose candle
                                         in candleBear candle
                                                && body > 0
                                                && bodyOk closePx body
                                                && upper >= priceActionWickRatio * body
                                                && lower <= priceActionOppositeWickMax * body

                                    bullishEngulf cur prev =
                                        let bodyCur = candleBody cur
                                            bodyPrev = candleBody prev
                                            closePx = candleClose cur
                                         in candleBull cur
                                                && candleBear prev
                                                && bodyCur >= bodyPrev
                                                && bodyOk closePx bodyCur
                                                && candleClose cur >= candleOpen prev
                                                && candleOpen cur <= candleClose prev

                                    bearishEngulf cur prev =
                                        let bodyCur = candleBody cur
                                            bodyPrev = candleBody prev
                                            closePx = candleClose cur
                                         in candleBear cur
                                                && candleBull prev
                                                && bodyCur >= bodyPrev
                                                && bodyOk closePx bodyCur
                                                && candleClose cur <= candleOpen prev
                                                && candleOpen cur >= candleClose prev

                                    railroadTracksLong cur prev =
                                        let bodyCur = candleBody cur
                                            bodyPrev = candleBody prev
                                            closePx = candleClose cur
                                         in candleBull cur
                                                && candleBear prev
                                                && bodyPrev > 0
                                                && bodyOk closePx bodyCur
                                                && abs (bodyCur - bodyPrev) / bodyPrev <= priceActionBodyTolerance

                                    darkCloudCover cur prev =
                                        let midPrev = (candleOpen prev + candleClose prev) / 2
                                         in candleBear cur && candleBull prev && candleClose cur < midPrev

                                    markToMarket :: PositionSide -> Double -> Double -> Double
                                    markToMarket side basePx px =
                                        if basePx == 0
                                            then 1
                                            else case side of
                                                SideLong -> px / basePx
                                                SideShort -> 2 - px / basePx

                                    isBad :: Double -> Bool
                                    isBad x = isNaN x || isInfinite x

                                    meanList :: [Double] -> Double
                                    meanList xs =
                                        let (sumY, countY) =
                                                foldl'
                                                    ( \(acc, n) x ->
                                                        if isBad x
                                                            then (acc, n)
                                                            else (acc + x, n + 1 :: Int)
                                                    )
                                                    (0, 0)
                                                    xs
                                         in if countY == 0 then 0 else sumY / fromIntegral countY

                                    returnsV :: V.Vector Double
                                    returnsV =
                                        V.generate stepCount $ \i ->
                                            let p0 = pricesV V.! i
                                                p1 = pricesV V.! (i + 1)
                                                r =
                                                    if p0 == 0 || isBad p0 || isBad p1
                                                        then 0
                                                        else p1 / p0 - 1
                                             in if isBad r then 0 else r

                                    meanV :: V.Vector Double -> Double
                                    meanV v =
                                        let n0 = V.length v
                                         in if n0 <= 0 then 0 else V.sum v / fromIntegral n0

                                    stddevV :: V.Vector Double -> Double
                                    stddevV v =
                                        let n0 = V.length v
                                         in if n0 < 2
                                                then 0
                                                else
                                                    let m = meanV v
                                                        var = V.sum (V.map (\x -> (x - m) ** 2) v) / fromIntegral (n0 - 1)
                                                     in sqrt var

                                    ewmaVarV :: V.Vector Double
                                    ewmaVarV =
                                        case volAlpha of
                                            Nothing -> V.empty
                                            Just a ->
                                                let update var r = a * var + (1 - a) * (r * r)
                                                 in V.drop 1 (V.scanl' update 0 returnsV)

                                    ewmaVarAt :: Int -> Maybe Double
                                    ewmaVarAt t =
                                        let idx = t - 1
                                         in if idx < 0 || idx >= V.length ewmaVarV
                                                then Nothing
                                                else Just (ewmaVarV V.! idx)

                                    rollingVolAt :: Int -> Maybe Double
                                    rollingVolAt t =
                                        let end = t - 1
                                         in if volLookback <= 1 || end < 1
                                                then Nothing
                                                else
                                                    let start = max 0 (end - volLookback + 1)
                                                        len = end - start + 1
                                                     in if len < 2
                                                            then Nothing
                                                            else
                                                                let v = V.slice start len returnsV
                                                                    std = stddevV v
                                                                 in if isBad std then Nothing else Just (std * sqrt ppy)

                                    ewmaVolAt :: Int -> Maybe Double
                                    ewmaVolAt t =
                                        case ewmaVarAt t of
                                            Nothing -> Nothing
                                            Just var ->
                                                let std = sqrt (max 0 var)
                                                 in if isBad std then Nothing else Just (std * sqrt ppy)

                                    volEstimateAt :: Int -> Maybe Double
                                    volEstimateAt t =
                                        case volAlpha of
                                            Just _ ->
                                                if t < 2
                                                    then Nothing
                                                    else ewmaVolAt t
                                            Nothing -> rollingVolAt t

                                    volPerBarAt :: Int -> Maybe Double
                                    volPerBarAt t =
                                        case volEstimateAt t of
                                            Nothing -> Nothing
                                            Just vol ->
                                                let perBar = vol / sqrt ppy
                                                 in if isBad perBar then Nothing else Just (max 0 perBar)

                                    volScaleAt :: Int -> Double
                                    volScaleAt t =
                                        case volTarget of
                                            Nothing -> 1
                                            Just target ->
                                                case volEstimateAt t of
                                                    Nothing -> 1
                                                    Just vol ->
                                                        let volAdj = max volFloor vol
                                                            scale0 = if volAdj <= 0 then 1 else target / volAdj
                                                            scale1 = min volScaleMax (max 0 scale0)
                                                         in if isBad scale1 then 1 else scale1

                                    volOkAt :: Int -> Bool
                                    volOkAt t =
                                        case maxVolatility of
                                            Nothing -> True
                                            Just maxVol ->
                                                case volEstimateAt t of
                                                    Just vol -> vol <= maxVol
                                                    Nothing -> False

                                    volTargetReadyAt :: Int -> Bool
                                    volTargetReadyAt t =
                                        case volTarget of
                                            Nothing -> True
                                            Just _ ->
                                                case volEstimateAt t of
                                                    Just _ -> True
                                                    Nothing -> False

                                    signalToNoiseOkAt :: Int -> Double -> Double -> Bool
                                    signalToNoiseOkAt t minSn edge =
                                        ( (minSn <= 0)
                                            || ( case volPerBarAt t of
                                                    Just vol | vol > 0 -> edge / vol >= minSn
                                                    _ -> False
                                               )
                                        )

                                    trendOkAt :: Int -> Int -> Double -> PositionSide -> Bool
                                    trendOkAt t lookback openThreshold side =
                                        (lookback <= 1 || t < lookback - 1)
                                            || ( let start = t - lookback + 1
                                                     v = V.slice start lookback pricesV
                                                     sma = meanV v
                                                     px = pricesV V.! t
                                                  in signalTrendSmaConfirmedWithConfig
                                                        (ecSignalGateConfig cfg)
                                                        openThreshold
                                                        px
                                                        sma
                                                        ( case side of
                                                            SideLong -> 1
                                                            SideShort -> -1
                                                        )
                                               )

                                    touchCloudAt :: Int -> Bool
                                    touchCloudAt idx =
                                        let fastIdx = cloudFastV V.! idx
                                            slowIdx = cloudSlowV V.! idx
                                            cloudTopIdx = max fastIdx slowIdx
                                            cloudBotIdx = min fastIdx slowIdx
                                            pxIdx = pricesV V.! idx
                                            padIdx = cloudPadFrac * (if isBad pxIdx then 0 else abs pxIdx)
                                            cloudTopPadIdx = cloudTopIdx + padIdx
                                            cloudBotPadIdx = cloudBotIdx - padIdx
                                            (_, h, l, _) = candleAt idx
                                         in not (isBad fastIdx || isBad slowIdx)
                                                && l <= cloudTopPadIdx
                                                && h >= cloudBotPadIdx

                                    touchCloudPrefix :: V.Vector Int
                                    touchCloudPrefix =
                                        if cloudReady
                                            then
                                                V.scanl'
                                                    (\acc touched -> acc + if touched then 1 else 0)
                                                    0
                                                    (V.generate n touchCloudAt)
                                            else V.replicate (n + 1) 0

                                    touchCloudInWindow :: Int -> Bool
                                    touchCloudInWindow t =
                                        ( not cloudReady
                                            || ( let start = max 0 (t - touchLookback + 1)
                                                     end = min n (t + 1)
                                                  in (touchCloudPrefix V.! end) > (touchCloudPrefix V.! start)
                                               )
                                        )

                                    cloudOkAt :: Int -> PositionSide -> Bool
                                    cloudOkAt t side =
                                        ( (not cloudReady || t <= 0)
                                            || ( let fast = cloudFastV V.! t
                                                     slow = cloudSlowV V.! t
                                                     slope = slow - cloudSlowV V.! (t - 1)
                                                     cloudTop = max fast slow
                                                     cloudBot = min fast slow
                                                     px = pricesV V.! t
                                                     slopeFrac =
                                                        if isBad px || px == 0
                                                            then 0
                                                            else slope / abs px
                                                     widthFrac =
                                                        if isBad px || px == 0
                                                            then 0
                                                            else (cloudTop - cloudBot) / abs px
                                                     widthOk = cloudWidthMax <= 0 || isBad widthFrac || widthFrac <= cloudWidthMax
                                                     touchCloud = touchCloudInWindow t
                                                     trendOk =
                                                        case side of
                                                            SideLong -> fast > slow && slopeFrac >= cloudSlopeMin
                                                            SideShort -> fast < slow && slopeFrac <= -cloudSlopeMin
                                                  in (isBad fast || isBad slow || isBad slope)
                                                        || (touchCloud && trendOk && widthOk)
                                               )
                                        )

                                    priceActionOkAt :: Int -> PositionSide -> Bool
                                    priceActionOkAt t side =
                                        ( (not triLayerEnabled || not requirePriceAction || t < 2)
                                            || ( let cur = candleAt t
                                                     prev = candleAt (t - 1)
                                                     bullish = hammer cur || bullishEngulf cur prev || railroadTracksLong cur prev
                                                     bearish = shootingStar cur || bearishEngulf cur prev || darkCloudCover cur prev
                                                  in case side of
                                                        SideLong -> bullish
                                                        SideShort -> bearish
                                               )
                                        )

                                    clamp01 :: Double -> Double
                                    clamp01 x = max 0 (min 1 x)

                                    clampRange :: Double -> Double -> Double -> Double
                                    clampRange lo hi x =
                                        let lo' = min lo hi
                                            hi' = max lo hi
                                         in max lo' (min hi' x)

                                    signedScore :: Double -> Double
                                    signedScore s = 2 * clamp01 s - 1

                                    scoreOrNeutral :: Maybe Double -> Double
                                    scoreOrNeutral mv =
                                        case mv of
                                            Just v | not (isBad v) -> clamp01 v
                                            _ -> 0.5

                                    edgeScore :: Double -> Double -> Double
                                    edgeScore thr edge =
                                        let denom = max 1e-12 (abs thr)
                                            raw = edge / (2 * denom)
                                         in if isBad raw then 0.5 else clamp01 raw

                                    lstmConfidenceScore :: Double -> Double -> Double -> Maybe Double
                                    lstmConfidenceScore thr0 prev next =
                                        if prev <= 0 || isBad prev || isBad next
                                            then Nothing
                                            else
                                                let edge = abs (next / prev - 1)
                                                    thr = max 1e-12 (abs thr0)
                                                    raw = edge / (2 * thr)
                                                 in if isBad edge || isBad raw
                                                        then Nothing
                                                        else Just (clamp01 raw)

                                    lstmConfidenceSizing :: Double -> Double -> Double -> Double
                                    lstmConfidenceSizing thr0 prev next =
                                        if not (ecConfidenceSizing cfg)
                                            then 1
                                            else
                                                let hard0 = clamp01 (ecLstmConfidenceHard cfg)
                                                    soft0 = clamp01 (ecLstmConfidenceSoft cfg)
                                                    hard = hard0
                                                    soft = min soft0 hard
                                                    denom = max 1e-12 (hard - soft)
                                                 in if hard <= 0
                                                        then 1
                                                        else case lstmConfidenceScore thr0 prev next of
                                                            Nothing -> 1
                                                            Just score ->
                                                                let scaleRaw
                                                                        | score <= soft = 0
                                                                        | score >= hard = 1
                                                                        | otherwise = (score - soft) / denom
                                                                 in clamp01 scaleRaw

                                    clampFrac :: Double -> Double
                                    clampFrac x = min 0.999999 (max 0 x)

                                    stopFracFromVol :: Int -> Double -> Maybe Double
                                    stopFracFromVol t mult =
                                        if mult <= 0
                                            then Nothing
                                            else case volPerBarAt t of
                                                Just vol
                                                    | vol > 0 ->
                                                        let f = mult * vol
                                                         in if isBad f || f <= 0 then Nothing else Just (clampFrac f)
                                                _ -> Nothing

                                    stopLossFracAt :: Int -> Maybe Double
                                    stopLossFracAt t =
                                        case stopFracFromVol t (ecStopLossVolMult cfg) of
                                            Just f -> Just f
                                            Nothing ->
                                                case ecStopLoss cfg of
                                                    Just sl | sl > 0 -> Just (clampFrac sl)
                                                    _ -> Nothing

                                    takeProfitFracAt :: Int -> Maybe Double
                                    takeProfitFracAt t =
                                        case stopFracFromVol t (ecTakeProfitVolMult cfg) of
                                            Just f -> Just f
                                            Nothing ->
                                                case ecTakeProfit cfg of
                                                    Just tp | tp > 0 -> Just (clampFrac tp)
                                                    _ -> Nothing

                                    trailingStopFracAt :: Int -> Maybe Double
                                    trailingStopFracAt t =
                                        case stopFracFromVol t (ecTrailingStopVolMult cfg) of
                                            Just f -> Just f
                                            Nothing ->
                                                case ecTrailingStop cfg of
                                                    Just ts | ts > 0 -> Just (clampFrac ts)
                                                    _ -> Nothing

                                    riskScaleAt :: Int -> Double
                                    riskScaleAt t =
                                        case riskPerTrade of
                                            Nothing -> 1
                                            Just risk ->
                                                case stopLossFracAt t of
                                                    Just sl
                                                        | sl > 0 ->
                                                            let s = risk / sl
                                                             in if isBad s || s <= 0 then 1 else s
                                                    _ -> 1

                                    kellyLiteScaleAt :: Int -> Double -> Double
                                    kellyLiteScaleAt t edge =
                                        if not kellyLiteEnabled
                                            then 1
                                            else
                                                let raw =
                                                        case volPerBarAt t of
                                                            Just vol
                                                                | vol > 0 ->
                                                                    let sig2 = vol * vol
                                                                        mu = max 0 edge
                                                                     in if sig2 <= 1e-12
                                                                            then 0
                                                                            else kellyLiteFraction * (mu / sig2)
                                                            _ -> 0
                                                    scale =
                                                        if isBad raw
                                                            then 0
                                                            else raw
                                                 in max kellyLiteFloor (min kellyLiteCap scale)

                                    scale01 :: Double -> Double -> Double -> Double
                                    scale01 lo hi x =
                                        let lo' = min lo hi
                                            hi' = max lo hi
                                         in if hi' <= lo' + 1e-12
                                                then if x >= hi' then 1 else 0
                                                else clamp01 ((x - lo') / (hi' - lo'))

                                    metaAt :: Int -> Maybe StepMeta
                                    metaAt t =
                                        case metaV of
                                            Nothing -> Nothing
                                            Just mv ->
                                                case metaMaskV of
                                                    Just mask
                                                        | t >= 0 && t < V.length mask && not (mask V.! t) -> Nothing
                                                    _ ->
                                                        if t >= 0 && t < V.length mv
                                                            then Just (mv V.! t)
                                                            else Nothing

                                    kalPredAt :: Int -> Double
                                    kalPredAt =
                                        case kalPredAtE of
                                            Right f -> f
                                            Left _ -> const 0

                                    lstmPredAt :: Int -> Double
                                    lstmPredAt =
                                        case lstmPredAtE of
                                            Right f -> f
                                            Left _ -> const 0

                                    intervalWidth :: StepMeta -> Maybe Double
                                    intervalWidth m =
                                        case (smConformalLo m, smConformalHi m) of
                                            (Just lo', Just hi') ->
                                                let w = hi' - lo'
                                                 in if isNaN w || isInfinite w then Nothing else Just w
                                            _ -> Nothing

                                    quantileWidth :: StepMeta -> Maybe Double
                                    quantileWidth m =
                                        case (smQuantile10 m, smQuantile90 m) of
                                            (Just q10', Just q90') ->
                                                let w = q90' - q10'
                                                 in if isNaN w || isInfinite w then Nothing else Just w
                                            _ -> Nothing

                                    kalmanZ :: Double -> StepMeta -> Double
                                    kalmanZ minStdFloor m =
                                        let var = max 0 (smKalmanVar m)
                                            std = max minStdFloor (sqrt var)
                                            mu = smKalmanMean m
                                         in if std <= 0 || isNaN std || isInfinite std then 0 else abs mu / std

                                    confirmConformal :: Double -> StepMeta -> PositionSide -> Bool
                                    confirmConformal thr m side =
                                        ( not (ecConfirmConformal cfg)
                                            || ( case side of
                                                    SideLong -> maybe True (> thr) (smConformalLo m)
                                                    SideShort -> maybe True (< negate thr) (smConformalHi m)
                                               )
                                        )

                                    confirmQuantiles :: Double -> StepMeta -> PositionSide -> Bool
                                    confirmQuantiles thr m side =
                                        ( not (ecConfirmQuantiles cfg)
                                            || ( case side of
                                                    SideLong -> maybe True (> thr) (smQuantile10 m)
                                                    SideShort -> maybe True (< negate thr) (smQuantile90 m)
                                               )
                                        )

                                    confidenceScoreKalman :: Double -> StepMeta -> Double
                                    confidenceScoreKalman zMin0 m =
                                        let zMin = max 0 zMin0
                                            zMax = max zMin (ecKalmanZMax cfg)
                                            zScore = scale01 zMin zMax (kalmanZ (ecKalmanMinStdFloor cfg) m)
                                            hvScore =
                                                case (ecMaxHighVolProb cfg, smHighVolProb m) of
                                                    (Just maxHv, Just hv) -> clamp01 ((maxHv - hv) / max 1e-12 maxHv)
                                                    _ -> 1
                                            confScore =
                                                case (ecMaxConformalWidth cfg, intervalWidth m) of
                                                    (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                                                    _ -> 1
                                            qScore =
                                                case (ecMaxQuantileWidth cfg, quantileWidth m) of
                                                    (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                                                    _ -> 1
                                         in zScore * hvScore * confScore * qScore

                                    gateKalmanDir :: Double -> Bool -> StepMeta -> Double -> Double -> Maybe PositionSide -> Maybe PositionSide -> (Maybe PositionSide, Double)
                                    gateKalmanDir zMin0 useSizing m thr confScore dirRaw fallbackDir =
                                        let hvOk =
                                                case (ecMaxHighVolProb cfg, smHighVolProb m) of
                                                    (Just maxHv, Just hv) -> hv <= maxHv
                                                    (Just _, Nothing) -> True
                                                    _ -> True
                                            confWidthOk =
                                                case (ecMaxConformalWidth cfg, intervalWidth m) of
                                                    (Just maxW, Just w) -> w <= maxW
                                                    (Just _, Nothing) -> True
                                                    _ -> True
                                            qWidthOk =
                                                case (ecMaxQuantileWidth cfg, quantileWidth m) of
                                                    (Just maxW, Just w) -> w <= maxW
                                                    (Just _, Nothing) -> True
                                                    _ -> True
                                         in case dirRaw of
                                                Nothing ->
                                                    -- Kalman-neutral fallback: if high-vol is low and a fallback
                                                    -- direction (e.g. LSTM) is provided, use it. This prevents
                                                    -- total paralysis when Kalman stays neutral for extended bars.
                                                    case fallbackDir of
                                                        Just side
                                                            | hvOk -> (Just side, if useSizing then confScore else 1)
                                                        _ -> (Nothing, 0)
                                                Just side ->
                                                    let kalZ = kalmanZ (ecKalmanMinStdFloor cfg) m
                                                        zMin = max 0 zMin0
                                                        size0 = if useSizing then confScore else 1
                                                     in if (kalZ < zMin)
                                                            || not hvOk
                                                            || not confWidthOk
                                                            || not qWidthOk
                                                            || not (confirmConformal thr m side)
                                                            || not (confirmQuantiles thr m side)
                                                            || (useSizing && size0 <= 0)
                                                            then (Nothing, 0)
                                                            else (Just side, size0)

                                    stepFn (posSide, posSize, equity, eqAcc, posAcc, agreeAcc, agreeValidAcc, changes, openTrade, tradesAcc, costTotals, costAcc, dead, cooldownLeft, lossStreak, riskState0, factorOpenPrev, factorClosePrev) t =
                                        if dead
                                            then
                                                ( Nothing
                                                , 0
                                                , equity
                                                , equity : eqAcc
                                                , 0 : posAcc
                                                , False : agreeAcc
                                                , False : agreeValidAcc
                                                , changes
                                                , Nothing
                                                , tradesAcc
                                                , costTotals
                                                , costTotals : costAcc
                                                , True
                                                , 0
                                                , lossStreak
                                                , riskState0
                                                , factorOpenPrev
                                                , factorClosePrev
                                                )
                                            else
                                                let RiskState
                                                        { rsPeakEquity = peakEq0
                                                        , rsDayKey = dayKey0
                                                        , rsDayStartEquity = dayStartEq0
                                                        , rsWeekKey = weekKey0
                                                        , rsWeekStartEquity = weekStartEq0
                                                        , rsDayTrades = dayTrades0
                                                        , rsHaltReason = haltReason0
                                                        } = riskState0
                                                    (dayKey1, dayStartEq1, dayChanged) =
                                                        if hasDailyKey
                                                            then
                                                                let dk = dayKeyAt t
                                                                    changed = dk /= dayKey0
                                                                    dayStart = if changed then equity else dayStartEq0
                                                                 in (dk, dayStart, changed)
                                                            else (dayKey0, dayStartEq0, False)
                                                    (weekKey1, weekStartEq1, weekChanged) =
                                                        if hasWeeklyKey
                                                            then
                                                                let wk = weekKeyAt t
                                                                    changed = wk /= weekKey0
                                                                    weekStart = if changed then equity else weekStartEq0
                                                                 in (wk, weekStart, changed)
                                                            else (weekKey0, weekStartEq0, False)
                                                    dayTrades1 = if dayChanged then 0 else dayTrades0
                                                    peakEq1 = max peakEq0 equity
                                                    drawdown =
                                                        if peakEq1 > 0
                                                            then max 0 (1 - equity / peakEq1)
                                                            else 0
                                                    dailyLoss =
                                                        if dayStartEq1 > 0
                                                            then max 0 (1 - equity / dayStartEq1)
                                                            else 0
                                                    weeklyLoss =
                                                        if weekStartEq1 > 0
                                                            then max 0 (1 - equity / weekStartEq1)
                                                            else 0
                                                    expectancy =
                                                        if expectancyLookback <= 0
                                                            then Nothing
                                                            else
                                                                let recent = take expectancyLookback tradesAcc
                                                                 in if length recent < expectancyLookback
                                                                        then Nothing
                                                                        else Just (meanList (map trReturn recent))
                                                    -- The simulation calls the canonical formal spec directly
                                                    -- (Trader.Formal.Risk re-exports specRiskHalt and proves
                                                    -- properties about it). This is spec≡impl by construction:
                                                    -- there is no separate implementation that could drift.
                                                    volTargetVal =
                                                        case ecVolTarget cfg of
                                                            Just v | not (isNaN v || isInfinite v) -> v
                                                            _ -> 0
                                                    haltReason1 =
                                                        specRiskHalt
                                                            HaltInputs
                                                                { hiPrevHaltReason = haltReason0
                                                                , hiDayChanged = dayChanged
                                                                , hiWeekChanged = weekChanged
                                                                , hiDailyLoss = dailyLoss
                                                                , hiWeeklyLoss = weeklyLoss
                                                                , hiDrawdown = drawdown
                                                                , hiExpectancy = expectancy
                                                                , hiMaxDailyLossLim = maxDailyLossLim
                                                                , hiMaxWeeklyLossLim = maxWeeklyLossLim
                                                                , hiMaxDrawdownLim = maxDrawdownLim
                                                                , hiMinExpectancy = minExpectancy
                                                                , hiPositionSize = max 0 posSize
                                                                , hiMaxPositionSizeLim =
                                                                    if positionSizeBoundsOk
                                                                        then Just maxPositionSize
                                                                        else Nothing
                                                                , hiConsecutiveLosses = lossStreak
                                                                , hiMaxLossStreakLim =
                                                                    if lossStreakMax > 0
                                                                        then Just lossStreakMax
                                                                        else Nothing
                                                                , hiVolTarget = volTargetVal
                                                                , hiLeverage = 0
                                                                }
                                                    halted = Data.Maybe.isJust haltReason1

                                                    prev = pricesV V.! t
                                                    nextClose = pricesV V.! (t + 1)
                                                    hi = barHigh (t + 1)
                                                    lo = barLow (t + 1)
                                                    highVolCutoff = max 0 (min 1 (ecBlendRegimeHighVolCutoff cfg))
                                                    tradeHighVolFlag tr =
                                                        case trEntryHighVolProb tr of
                                                            Just hv -> Just (hv >= highVolCutoff)
                                                            Nothing -> Nothing
                                                    mHighVolNow =
                                                        case metaNow >>= smHighVolProb of
                                                            Just hv -> Just (hv >= highVolCutoff)
                                                            Nothing -> Nothing

                                                    perfRecentAll =
                                                        if perfLookback <= 0
                                                            then []
                                                            else take perfLookback tradesAcc
                                                    perfReadyAll =
                                                        perfLookback > 0 && length perfRecentAll >= perfLookback
                                                    perfReturnsAll =
                                                        [ trReturn tr
                                                        | tr <- perfRecentAll
                                                        , not (isBad (trReturn tr))
                                                        ]
                                                    perfWinsAll = length [r | r <- perfReturnsAll, r > 0]
                                                    perfLossesAll = length [r | r <- perfReturnsAll, r < 0]
                                                    perfWinRateAll =
                                                        if perfWinsAll + perfLossesAll == 0
                                                            then 0
                                                            else fromIntegral perfWinsAll / fromIntegral (perfWinsAll + perfLossesAll)
                                                    perfGrossWinAll = sum [r | r <- perfReturnsAll, r > 0]
                                                    perfGrossLossAll = abs (sum [r | r <- perfReturnsAll, r < 0])
                                                    perfProfitFactorAll
                                                        | perfGrossLossAll > 0 = Just (perfGrossWinAll / perfGrossLossAll)
                                                        | perfGrossWinAll > 0 = Nothing
                                                        | otherwise = Just 0

                                                    perfRecentRegime =
                                                        case mHighVolNow of
                                                            Nothing -> []
                                                            Just want ->
                                                                if perfLookback <= 0
                                                                    then []
                                                                    else take perfLookback [tr | tr <- tradesAcc, tradeHighVolFlag tr == Just want]

                                                    perfRecentGate =
                                                        if perfLookback > 0 && length perfRecentRegime >= perfLookback
                                                            then perfRecentRegime
                                                            else perfRecentAll
                                                    perfReadyGate =
                                                        perfLookback > 0 && length perfRecentGate >= perfLookback
                                                    perfReturnsGate =
                                                        [ trReturn tr
                                                        | tr <- perfRecentGate
                                                        , not (isBad (trReturn tr))
                                                        ]
                                                    perfWinsGate = length [r | r <- perfReturnsGate, r > 0]
                                                    perfLossesGate = length [r | r <- perfReturnsGate, r < 0]
                                                    perfWinRateGate =
                                                        if perfWinsGate + perfLossesGate == 0
                                                            then 0
                                                            else fromIntegral perfWinsGate / fromIntegral (perfWinsGate + perfLossesGate)
                                                    perfGrossWinGate = sum [r | r <- perfReturnsGate, r > 0]
                                                    perfGrossLossGate = abs (sum [r | r <- perfReturnsGate, r < 0])
                                                    perfProfitFactorGate
                                                        | perfGrossLossGate > 0 = Just (perfGrossWinGate / perfGrossLossGate)
                                                        | perfGrossWinGate > 0 = Nothing
                                                        | otherwise = Just 0
                                                    perfGateReasonNow =
                                                        if not perfReadyGate
                                                            then Nothing
                                                            else
                                                                let winOk =
                                                                        case perfMinWinRate of
                                                                            Just v | v > 0 -> perfWinRateGate >= v
                                                                            _ -> True
                                                                    pfOk =
                                                                        case perfMinProfitFactor of
                                                                            Just v | v > 0 ->
                                                                                case perfProfitFactorGate of
                                                                                    Nothing -> True
                                                                                    Just pf -> pf >= v
                                                                            _ -> True
                                                                 in if not winOk
                                                                        then Just "PERF_WIN_RATE"
                                                                        else
                                                                            if not pfOk
                                                                                then Just "PERF_PROFIT_FACTOR"
                                                                                else Nothing
                                                    winScore =
                                                        case perfMinWinRate of
                                                            Just v
                                                                | v > 0 ->
                                                                    let start = min 1 (v + adaptiveWinRateSlack)
                                                                        denom = max 1e-12 (start - v)
                                                                        raw = (start - perfWinRateGate) / denom
                                                                     in clamp01 raw
                                                            _ -> 0
                                                    pfScore =
                                                        case perfMinProfitFactor of
                                                            Just v
                                                                | v > 0 ->
                                                                    let start = v * (1 + adaptiveProfitFactorSlack)
                                                                        denom = max 1e-12 (start - v)
                                                                        pfVal = Data.Maybe.fromMaybe start perfProfitFactorGate
                                                                        raw = (start - pfVal) / denom
                                                                     in clamp01 raw
                                                            _ -> 0
                                                    strictness =
                                                        if adaptiveFilters && perfReadyGate
                                                            then max winScore pfScore
                                                            else 0
                                                    edgeAdd = strictness * adaptiveEdgeBufferMax
                                                    minSnrAdd = strictness * adaptiveMinSnrMax
                                                    kalmanZMinAdd = strictness * adaptiveKalmanZMinMax
                                                    trendLookbackAdd = round (strictness * fromIntegral adaptiveTrendLookbackMax)
                                                    minEdgeStep = minEdgeBase + edgeAdd
                                                    openThrBase = normalizeSignalThreshold (max openThrRawBase minEdgeStep)
                                                    minSignalToNoiseStep = minSignalToNoiseBase + minSnrAdd
                                                    kalmanZMinStep = kalmanZMinBase + kalmanZMinAdd
                                                    trendLookbackStep = trendLookbackBase + trendLookbackAdd
                                                    factorOpenBase0 =
                                                        if factorEnabled && not (isBad factorOpenPrev)
                                                            then factorOpenPrev
                                                            else 1
                                                    factorCloseBase0 =
                                                        if factorEnabled && not (isBad factorClosePrev)
                                                            then factorClosePrev
                                                            else 1
                                                    factorOpenBase =
                                                        if factorEnabled
                                                            then clampRange factorMin factorMax factorOpenBase0
                                                            else 1
                                                    factorCloseBase =
                                                        if factorEnabled
                                                            then clampRange factorMin factorMax factorCloseBase0
                                                            else 1
                                                    minEdgeAdj = max factorFloor (minEdgeStep * factorOpenBase)
                                                    openThrAdj =
                                                        normalizeSignalThreshold
                                                            (max minEdgeAdj (max factorFloor (openThrBase * factorOpenBase)))
                                                    closeThrAdj =
                                                        normalizeSignalThreshold
                                                            (max factorFloor (closeThrBase * factorCloseBase))
                                                    minSignalToNoiseAdj = max factorFloor (minSignalToNoiseStep * factorOpenBase)
                                                    openTrade0 =
                                                        case posSide of
                                                            Nothing -> Nothing
                                                            Just _ -> openTrade
                                                    posSideStart = posSide
                                                    posSizeStart = posSize
                                                    holdBars =
                                                        maybe 0 otHoldingPeriods openTrade0
                                                    cooldownActive = isNothing posSide && cooldownLeft > 0
                                                    cooldownNext0 = if isNothing posSide then max 0 (cooldownLeft - 1) else 0
                                                    (agreeOk, agreeValid, desiredSideRaw, desiredSizeRaw, edgeRaw, edgeKal, edgeLstm, mOpenSignal, lstmCloseDir, lstmEntryScaleRaw, lstmConfScoreMaybe, metaNow) =
                                                        if t < startT
                                                            then (False, False, posSide, posSize, 0, 0, 0, Nothing, Nothing, 1, Nothing, Nothing)
                                                            else
                                                                let kp = kalPredAt t
                                                                    lp = lstmPredAt t
                                                                    edgePred p =
                                                                        if prev == 0 || isBad p
                                                                            then 0
                                                                            else
                                                                                let edge = abs (p / prev - 1)
                                                                                 in if isBad edge then 0 else edge
                                                                    edgeKal = edgePred kp
                                                                    edgeLstm = edgePred lp
                                                                    edgeRaw = min edgeKal edgeLstm
                                                                    kalOpenDirRaw = direction openThrAdj prev kp
                                                                    kalCloseDirRaw = direction closeThrAdj prev kp
                                                                    metaNow = metaAt t
                                                                    (kalOpenDir, kalCloseDir, kalSize) =
                                                                        case metaNow of
                                                                            Nothing ->
                                                                                ( kalOpenDirRaw
                                                                                , kalCloseDirRaw
                                                                                , if isNothing kalOpenDirRaw then 0 else 1
                                                                                )
                                                                            Just m ->
                                                                                let confScore = confidenceScoreKalman kalmanZMinStep m
                                                                                    (openDir, openSize) = gateKalmanDir kalmanZMinStep confidenceSizingEnabled m openThrAdj confScore kalOpenDirRaw lstmOpenDir
                                                                                    (closeDir, _) = gateKalmanDir kalmanZMinStep False m closeThrAdj confScore kalCloseDirRaw lstmCloseDir
                                                                                 in (openDir, closeDir, openSize)
                                                                    lstmOpenDir = direction openThrAdj prev lp
                                                                    lstmCloseDir = direction closeThrAdj prev lp
                                                                    agreeValid =
                                                                        case (kalOpenDir, lstmOpenDir) of
                                                                            (Just _, Just _) -> True
                                                                            _ -> False
                                                                    agreeOk =
                                                                        case (kalOpenDir, lstmOpenDir) of
                                                                            (Just a, Just b) -> a == b
                                                                            _ -> False
                                                                    closeAgreeDir =
                                                                        case (kalCloseDir, lstmCloseDir) of
                                                                            (Just a, Just b) | a == b -> Just a
                                                                            _ -> Nothing
                                                                    openAgreeDir =
                                                                        if agreeOk then kalOpenDir else Nothing

                                                                    openSignal =
                                                                        case openAgreeDir of
                                                                            Just dir ->
                                                                                case desiredSideFromDir dir of
                                                                                    Nothing -> Nothing
                                                                                    Just s -> Just (s, kalSize)
                                                                            Nothing -> Nothing

                                                                    openBlockedByPositioning =
                                                                        case openAgreeDir of
                                                                            Just _ -> isNothing openSignal
                                                                            Nothing -> False

                                                                    (desiredSide', desiredSize') =
                                                                        case openSignal of
                                                                            Just (s, sz) -> (Just s, sz)
                                                                            Nothing ->
                                                                                if openBlockedByPositioning
                                                                                    then (Nothing, 0)
                                                                                    else case (posSide, closeAgreeDir) of
                                                                                        (Just side, Just dir) | side == dir -> (Just side, posSize)
                                                                                        _ -> (Nothing, 0)
                                                                    lstmEntryScale = lstmConfidenceSizing openThrBase prev lp
                                                                    lstmScore = lstmConfidenceScore openThrBase prev lp
                                                                 in (agreeOk, agreeValid, desiredSide', desiredSize', edgeRaw, edgeKal, edgeLstm, openSignal, lstmCloseDir, lstmEntryScale, lstmScore, metaNow)

                                                    pendingFactorUpdate =
                                                        factorEnabled && (Data.Maybe.isJust posSide || Data.Maybe.isJust mOpenSignal)

                                                    factorTarget thr =
                                                        let edgeKalScore = edgeScore thr edgeKal
                                                            edgeLstmScore = edgeScore thr edgeLstm
                                                            zScore =
                                                                scoreOrNeutral $
                                                                    case metaNow of
                                                                        Nothing -> Nothing
                                                                        Just m ->
                                                                            let zMin = max 0 kalmanZMinStep
                                                                                zMax = max zMin (ecKalmanZMax cfg)
                                                                             in Just (scale01 zMin zMax (kalmanZ (ecKalmanMinStdFloor cfg) m))
                                                            hvScore =
                                                                scoreOrNeutral $
                                                                    case metaNow >>= smHighVolProb of
                                                                        Nothing -> Nothing
                                                                        Just hv ->
                                                                            let score =
                                                                                    case ecMaxHighVolProb cfg of
                                                                                        Just maxHv | maxHv > 0 -> clamp01 ((maxHv - hv) / max 1e-12 maxHv)
                                                                                        _ -> clamp01 (1 - hv)
                                                                             in Just score
                                                            confScore =
                                                                scoreOrNeutral $
                                                                    case metaNow >>= intervalWidth of
                                                                        Nothing -> Nothing
                                                                        Just w ->
                                                                            case ecMaxConformalWidth cfg of
                                                                                Just maxW | maxW > 0 -> Just (clamp01 ((maxW - w) / max 1e-12 maxW))
                                                                                _ -> Nothing
                                                            qScore =
                                                                scoreOrNeutral $
                                                                    case metaNow >>= quantileWidth of
                                                                        Nothing -> Nothing
                                                                        Just w ->
                                                                            case ecMaxQuantileWidth cfg of
                                                                                Just maxW | maxW > 0 -> Just (clamp01 ((maxW - w) / max 1e-12 maxW))
                                                                                _ -> Nothing
                                                            lstmScore = scoreOrNeutral lstmConfScoreMaybe
                                                            healthScore = lstmHealthScore
                                                            raw =
                                                                1
                                                                    + factorWEdgeKal * signedScore edgeKalScore
                                                                    + factorWEdgeLstm * signedScore edgeLstmScore
                                                                    + factorWKalmanZ * signedScore zScore
                                                                    + factorWHighVol * signedScore hvScore
                                                                    + factorWConformal * signedScore confScore
                                                                    + factorWQuantile * signedScore qScore
                                                                    + factorWLstmConf * signedScore lstmScore
                                                                    + factorWLstmHealth * signedScore healthScore
                                                         in if isBad raw then 1 else raw

                                                    updateFactor prev target =
                                                        if factorAlpha <= 0
                                                            then prev
                                                            else
                                                                let next = (1 - factorAlpha) * prev + factorAlpha * target
                                                                 in if isBad next then prev else clampRange factorMin factorMax next

                                                    factorOpenNext =
                                                        if pendingFactorUpdate
                                                            then updateFactor factorOpenBase (factorTarget openThrBase)
                                                            else factorOpenBase

                                                    factorCloseNext =
                                                        if pendingFactorUpdate
                                                            then updateFactor factorCloseBase (factorTarget closeThrBase)
                                                            else factorCloseBase

                                                    lstmFlipStrongOk =
                                                        ( not lstmFlipStrongOnly
                                                            || ( case lstmConfScoreMaybe of
                                                                    Just score -> score >= max 0 (min 1 (ecLstmConfidenceHard cfg))
                                                                    Nothing -> False
                                                               )
                                                        )

                                                    (openTradeFlip, lstmFlipExit) =
                                                        if lstmFlipBars <= 0
                                                            then (openTrade0, False)
                                                            else case (openTrade0, posSide, lstmCloseDir) of
                                                                (Just ot, Just side, Just lstmDir) ->
                                                                    let inGrace = otHoldingPeriods ot < lstmFlipGraceBars
                                                                     in if inGrace
                                                                            then (Just ot{otLstmFlipCount = 0}, False)
                                                                            else
                                                                                let opposite =
                                                                                        case (side, lstmDir) of
                                                                                            (SideLong, SideShort) -> True
                                                                                            (SideShort, SideLong) -> True
                                                                                            _ -> False
                                                                                    nextCount =
                                                                                        if opposite && lstmFlipStrongOk
                                                                                            then otLstmFlipCount ot + 1
                                                                                            else 0
                                                                                    ot' = ot{otLstmFlipCount = nextCount}
                                                                                 in (Just ot', nextCount >= lstmFlipBars)
                                                                (Just ot, Just _, _) -> (Just ot{otLstmFlipCount = 0}, False)
                                                                _ -> (openTrade0, False)

                                                    desiredSize0 =
                                                        if isNothing desiredSideRaw
                                                            then 0
                                                            else max 0 desiredSizeRaw

                                                    desiredSide0 =
                                                        if desiredSize0 <= 0 then Nothing else desiredSideRaw

                                                    needsEntry = Data.Maybe.isJust desiredSide0 && desiredSide0 /= posSide
                                                    lstmEntryScale =
                                                        if needsEntry && not volConfGateEnabled
                                                            then lstmEntryScaleRaw
                                                            else 1

                                                    trendOk =
                                                        case desiredSide0 of
                                                            Just side | needsEntry -> trendOkAt t trendLookbackStep openThrAdj side
                                                            _ -> True

                                                    volOk = (not needsEntry || volOkAt t)

                                                    snrScale =
                                                        if minSignalToNoiseAdj <= 0
                                                            then 1
                                                            else case volPerBarAt t of
                                                                Just vol | vol > 0 -> clamp01 (max 0 edgeRaw / vol / minSignalToNoiseAdj)
                                                                _ -> 0

                                                    snrScaleWeighted =
                                                        if snrSizeWeight <= 0
                                                            then 1
                                                            else (1 - snrSizeWeight) + snrSizeWeight * snrScale

                                                    snrOk =
                                                        ( not needsEntry
                                                            || ( if snrSizeWeight <= 0
                                                                    then signalToNoiseOkAt t minSignalToNoiseAdj (max 0 edgeRaw)
                                                                    else snrScale > 0
                                                               )
                                                        )

                                                    volTargetReady =
                                                        (not needsEntry || volTargetReadyAt t)

                                                    triLayerOk =
                                                        case (needsEntry, desiredSide0) of
                                                            (True, Just side) -> cloudOkAt t side && priceActionOkAt t side
                                                            _ -> True

                                                    entryEdgeSample = Just (max 0 edgeRaw)

                                                    entryEdgeSpikeOk =
                                                        not needsEntry || signalEntryEdgeSpikeConsecutiveOkWithConfig (ecSignalGateConfig cfg) (ecEntryEdgeSpikeConsecutive cfg) (ecEntryEdgeSpikeAuditOnly cfg) openThrAdj entryEdgeSample

                                                    entryEdgeHeadroomOk =
                                                        not needsEntry || signalEntryHeadroomOk openThrAdj entryEdgeSample

                                                    slowCrossExit =
                                                        case posSide of
                                                            Just side ->
                                                                triLayerExitOnSlow
                                                                    && cloudReady
                                                                    && t > 0
                                                                    && let slow = cloudSlowV V.! t
                                                                           slowPrev = cloudSlowV V.! (t - 1)
                                                                           priorClose = pricesV V.! (t - 1)
                                                                        in ( not (any isBad [slow, slowPrev, prev, priorClose])
                                                                                && ( case side of
                                                                                        SideLong -> priorClose >= slowPrev && prev < slow
                                                                                        SideShort -> priorClose <= slowPrev && prev > slow
                                                                                   )
                                                                           )
                                                            Nothing -> False

                                                    kalmanBandExit =
                                                        case posSide of
                                                            Just side ->
                                                                case kalmanBandAt t of
                                                                    Nothing -> False
                                                                    Just (upper, lower) ->
                                                                        let reversal =
                                                                                case (side, lstmCloseDir) of
                                                                                    (SideLong, Just SideShort) -> True
                                                                                    (SideShort, Just SideLong) -> True
                                                                                    _ -> False
                                                                            softThr = max 0 (min 1 (ecLstmConfidenceSoft cfg))
                                                                            lowConfidence =
                                                                                case lstmConfScoreMaybe of
                                                                                    Just score -> score < softThr
                                                                                    Nothing -> False
                                                                            hiNow = barHigh t
                                                                            loNow = barLow t
                                                                         in case side of
                                                                                SideLong ->
                                                                                    not (isBad hiNow) && hiNow >= upper && (reversal || lowConfidence)
                                                                                SideShort ->
                                                                                    not (isBad loNow) && loNow <= lower && (reversal || lowConfidence)
                                                            Nothing -> False

                                                    kalmanExit = slowCrossExit || kalmanBandExit

                                                    desiredSide1 =
                                                        if not trendOk
                                                            || not volOk
                                                            || not snrOk
                                                            || not volTargetReady
                                                            || not triLayerOk
                                                            || not entryEdgeSpikeOk
                                                            || not entryEdgeHeadroomOk
                                                            then Nothing
                                                            else desiredSide0

                                                    desiredSide2 =
                                                        if lstmFlipExit || kalmanExit
                                                            then Nothing
                                                            else desiredSide1

                                                    volConfCell =
                                                        volConfGateCellWithConfig
                                                            volConfGateConfig
                                                            volConfGatePreset
                                                            (volEstimateAt t)
                                                            ( case metaNow of
                                                                Just m -> Just (confidenceScoreKalman kalmanZMinStep m)
                                                                Nothing -> lstmConfScoreMaybe
                                                            )
                                                    volConfBehavior = vcgBehavior volConfCell
                                                    volConfSizeMult =
                                                        if Data.Maybe.isJust desiredSide2
                                                            then vcgSizeMult volConfCell
                                                            else 1

                                                    baseSizeTarget =
                                                        case desiredSide2 of
                                                            Nothing -> 0
                                                            Just side ->
                                                                case mOpenSignal of
                                                                    Just (sigSide, sigSize) | sigSide == side -> max 0 sigSize
                                                                    _ ->
                                                                        case openTradeFlip of
                                                                            Just ot | otSide ot == side -> max 0 (otBaseSize ot)
                                                                            _ -> max 0 desiredSizeRaw

                                                    entryScale = if Data.Maybe.isJust desiredSide2 && desiredSide2 /= posSide then lstmEntryScale else 1
                                                    sizeScale =
                                                        if isNothing desiredSide2
                                                            then 1
                                                            else volScaleAt t * riskScaleAt t * snrScaleWeighted * kellyLiteScaleAt t edgeRaw
                                                    sizeScaled = baseSizeTarget * entryScale * volConfSizeMult * sizeScale
                                                    -- FIRM-CRITICAL: catch runaway compounding before absolute cap
                                                    sizeScaledChecked =
                                                        if baseSizeTarget > 0 && sizeScaled > positionSizeScaleHardFailMultiplier * baseSizeTarget
                                                            then error ("POSITION_SIZE_SCALE_EXCEEDED: sizeScaled=" ++ show sizeScaled ++ " baseSizeTarget=" ++ show baseSizeTarget)
                                                            else sizeScaled
                                                    sizeCapped = min maxPositionSize (max 0 sizeScaledChecked)
                                                    sizeFloorOk = not (isNaN minPositionSize || isInfinite minPositionSize) && minPositionSize >= 0
                                                    sizeFinal0
                                                        | not sizeFloorOk = 0
                                                        | sizeCapped < minPositionSize && desiredSide2 /= posSide = 0
                                                        | otherwise = sizeCapped

                                                    entryFeeBufferOk =
                                                        not (Data.Maybe.isJust desiredSide2 && desiredSide2 /= posSide && sizeFinal0 > 0)
                                                            || signalEntryFeeBufferOkWithConfig
                                                                (ecSignalGateConfig cfg)
                                                                openThrAdj
                                                                (roundTripCostAt t sizeFinal0)
                                                                entryEdgeSample

                                                    desiredSide =
                                                        if sizeFinal0 <= 0 || not entryFeeBufferOk then Nothing else desiredSide2

                                                    desiredSize =
                                                        if Data.Maybe.isJust desiredSide
                                                            then sizeFinal0
                                                            else 0

                                                    (desiredSideVolConf, desiredSizeVolConf) =
                                                        applyVolConfGateBehavior
                                                            volConfBehavior
                                                            posSide
                                                            posSize
                                                            desiredSide
                                                            desiredSize

                                                    holdForced = Data.Maybe.isJust posSide && desiredSide /= posSide && holdBars < minHoldBars

                                                    desiredSideHoldAdjusted =
                                                        if holdForced
                                                            then posSide
                                                            else desiredSideVolConf

                                                    desiredSizeHoldAdjusted =
                                                        if holdForced
                                                            then posSize
                                                            else desiredSizeVolConf

                                                    noTradeActive =
                                                        case noTradeWindows' of
                                                            [] -> False
                                                            windows -> any (\w -> timeWindowContains w (minuteOfDayAt t)) windows

                                                    tradeLimitReached =
                                                        case maxTradesPerDayLim of
                                                            Just lim -> dayTrades1 >= lim
                                                            Nothing -> False

                                                    entryAttempt =
                                                        case desiredSideHoldAdjusted of
                                                            Just side -> posSide /= Just side
                                                            Nothing -> False

                                                    entryBlockReason
                                                        | entryAttempt && noTradeActive = Just "NO_TRADE_WINDOW"
                                                        | entryAttempt && tradeLimitReached = Just "MAX_TRADES_PER_DAY"
                                                        | entryAttempt = perfGateReasonNow
                                                        | otherwise = Nothing

                                                    (desiredSideFinal0, desiredSizeFinal0) =
                                                        if cooldownActive
                                                            then (Nothing, 0)
                                                            else case entryBlockReason of
                                                                Nothing -> (desiredSideHoldAdjusted, desiredSizeHoldAdjusted)
                                                                Just reason ->
                                                                    case posSide of
                                                                        Just side ->
                                                                            let isPerfGate = reason == "PERF_WIN_RATE" || reason == "PERF_PROFIT_FACTOR"
                                                                                flipAttempt =
                                                                                    case desiredSideHoldAdjusted of
                                                                                        Just s -> Just s /= posSide
                                                                                        Nothing -> False
                                                                             in if isPerfGate && flipAttempt
                                                                                    then (Nothing, 0)
                                                                                    else (Just side, posSize)
                                                                        Nothing -> (Nothing, 0)

                                                    holdTooLong =
                                                        case maxHoldBars of
                                                            Nothing -> False
                                                            Just lim -> Data.Maybe.isJust posSide && holdBars >= lim

                                                    (desiredSideFinal1, desiredSizeFinal1) =
                                                        if holdTooLong
                                                            then (Nothing, 0)
                                                            else (desiredSideFinal0, desiredSizeFinal0)

                                                    (desiredSideFinal, desiredSizeFinal) =
                                                        if halted
                                                            then (Nothing, 0)
                                                            else (desiredSideFinal1, desiredSizeFinal1)

                                                    exitReasonOverride =
                                                        case () of
                                                            _
                                                                | halted -> haltReason1
                                                                | holdTooLong -> Just (ExitOther "MAX_HOLD")
                                                                | lstmFlipExit -> Just (ExitOther "LSTM_FLIP")
                                                                | slowCrossExit -> Just (ExitOther "KALMAN_SLOW")
                                                                | kalmanBandExit -> Just (ExitOther "KALMAN_BAND")
                                                                | otherwise -> Nothing

                                                    switchExitReason =
                                                        case (exitReasonOverride, posSide) of
                                                            (Just r, Just _) -> r
                                                            _ -> ExitSignal

                                                    tradeFeeCost entryIdx exitIdx entryEq exitEq size =
                                                        let s = max 0 (abs size)
                                                            entryRates = costPerSideBreakdown entryIdx s
                                                            exitRates = costPerSideBreakdown exitIdx s
                                                         in entryEq * ctFeeCost entryRates + exitEq * ctFeeCost exitRates

                                                    closeTradeAt exitIndex why eqExit ot =
                                                        Trade
                                                            { trEntryIndex = otEntryIndex ot
                                                            , trExitIndex = exitIndex
                                                            , trEntryEquity = otEntryEquity ot
                                                            , trExitEquity = eqExit
                                                            , trReturn = eqExit / otEntryEquity ot - 1
                                                            , trHoldingPeriods = otHoldingPeriods ot
                                                            , trEntryHighVolProb = metaAt (otEntryIndex ot) >>= smHighVolProb
                                                            , trEntrySource = TradeEntrySignal
                                                            , trExitReason = Just why
                                                            , trEntryIp = Nothing
                                                            , trExitIp = Nothing
                                                            , trFeeCost = tradeFeeCost (otEntryIndex ot) exitIndex (otEntryEquity ot) eqExit (otBaseSize ot)
                                                            }

                                                    openTradeFor side eqEntry baseSize =
                                                        OpenTrade
                                                            { otEntryIndex = t
                                                            , otRebalanceAnchor = t
                                                            , otEntryEquity = eqEntry
                                                            , otHoldingPeriods = 0
                                                            , otEntryPrice = prev
                                                            , otTrail = prev
                                                            , otSide = side
                                                            , otBaseSize = max 0 baseSize
                                                            , otLstmFlipCount = 0
                                                            }

                                                    openTradeUpdated =
                                                        case (openTradeFlip, mOpenSignal) of
                                                            (Just ot, Just (sigSide, sigSize))
                                                                | otSide ot == sigSide && sigSize > 0 ->
                                                                    let ot' = ot{otBaseSize = max 0 sigSize}
                                                                     in Just
                                                                            ( if rebalanceResetOnSignal
                                                                                then ot'{otRebalanceAnchor = t}
                                                                                else ot'
                                                                            )
                                                            _ -> openTradeFlip

                                                    rebalanceDelta = abs (desiredSizeFinal - posSize)
                                                    rebalanceDue
                                                        | not rebalanceEnabled = False
                                                        | rebalanceGlobal = t `mod` rebalanceBars == 0
                                                        | otherwise = case openTradeUpdated of
                                                            Just ot ->
                                                                let age = max 0 (t - otRebalanceAnchor ot)
                                                                 in age `mod` rebalanceBars == 0
                                                            Nothing -> t `mod` rebalanceBars == 0
                                                    rebalanceOk =
                                                        rebalanceEnabled
                                                            && Data.Maybe.isJust posSide
                                                            && desiredSideFinal == posSide
                                                            && rebalanceDue
                                                            && rebalanceDelta > 0
                                                            && rebalanceDelta >= rebalanceThreshold

                                                    (posAfterSwitch, posSizeAfterSwitch, equityAfterSwitch, costTotalsAfterSwitch, changes', openTrade', tradesAcc') =
                                                        if desiredSideFinal == posSide
                                                            then
                                                                if rebalanceOk
                                                                    then
                                                                        let (eqRebalance, costTotalsRebalance) = applyCostWithTotals t equity rebalanceDelta costTotals
                                                                         in (posSide, desiredSizeFinal, eqRebalance, costTotalsRebalance, changes + 1, openTradeUpdated, tradesAcc)
                                                                    else (posSide, posSize, equity, costTotals, changes, openTradeUpdated, tradesAcc)
                                                            else case desiredSideFinal of
                                                                Nothing ->
                                                                    let (eqExit, costTotalsExit) = applyCostWithTotals t equity posSize costTotals
                                                                        tradesAcc1 =
                                                                            case openTradeFlip of
                                                                                Nothing -> tradesAcc
                                                                                Just ot -> closeTradeAt t switchExitReason eqExit ot : tradesAcc
                                                                     in (Nothing, 0, eqExit, costTotalsExit, changes + 1, Nothing, tradesAcc1)
                                                                Just desiredSideFinal' ->
                                                                    case posSide of
                                                                        Nothing ->
                                                                            let (eqEntry, costTotalsEntry) = applyCostWithTotals t equity desiredSizeFinal costTotals
                                                                             in ( Just desiredSideFinal'
                                                                                , desiredSizeFinal
                                                                                , eqEntry
                                                                                , costTotalsEntry
                                                                                , changes + 1
                                                                                , Just (openTradeFor desiredSideFinal' eqEntry baseSizeTarget)
                                                                                , tradesAcc
                                                                                )
                                                                        Just _ ->
                                                                            let (eqExit, costTotalsExit) = applyCostWithTotals t equity posSize costTotals
                                                                                tradesAcc1 =
                                                                                    case openTradeFlip of
                                                                                        Nothing -> tradesAcc
                                                                                        Just ot -> closeTradeAt t ExitSignal eqExit ot : tradesAcc
                                                                                (eqEntry, costTotalsEntry) = applyCostWithTotals t eqExit desiredSizeFinal costTotalsExit
                                                                             in ( Just desiredSideFinal'
                                                                                , desiredSizeFinal
                                                                                , eqEntry
                                                                                , costTotalsEntry
                                                                                , changes + 1
                                                                                , Just (openTradeFor desiredSideFinal' eqEntry baseSizeTarget)
                                                                                , tradesAcc1
                                                                                )

                                                    equityAtClose =
                                                        case posAfterSwitch of
                                                            Just side
                                                                | posSizeAfterSwitch > 0 ->
                                                                    let factor = markToMarket side prev nextClose
                                                                        eq1 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                                     in if isBad eq1 then equityAfterSwitch else eq1
                                                            _ -> equityAfterSwitch

                                                    (posFinal, posSizeFinal, equityFinal0, costTotalsFinal0, changesFinal, openTradeFinal, tradesFinal) =
                                                        case (posAfterSwitch, openTrade') of
                                                            (Just SideLong, Just ot0) ->
                                                                let otHeld = ot0{otHoldingPeriods = otHoldingPeriods ot0 + 1}
                                                                    entryPx = otEntryPrice ot0
                                                                    trail0 = otTrail ot0
                                                                    mTp =
                                                                        case takeProfitFracAt t of
                                                                            Just tp -> Just (entryPx * (1 + tp))
                                                                            _ -> Nothing
                                                                    mSl =
                                                                        case stopLossFracAt t of
                                                                            Just sl -> Just (entryPx * (1 - sl))
                                                                            _ -> Nothing
                                                                    stopPx trail =
                                                                        let mTs =
                                                                                case trailingStopFracAt t of
                                                                                    Just ts -> Just (trail * (1 - ts))
                                                                                    _ -> Nothing
                                                                         in case (mSl, mTs) of
                                                                                (Nothing, Nothing) -> (Nothing, Nothing)
                                                                                (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                                                                (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                                                                (Just slPx, Just tsPx) ->
                                                                                    if tsPx > slPx
                                                                                        then (Just tsPx, Just ExitTrailingStop)
                                                                                        else (Just slPx, Just ExitStopLoss)
                                                                    tpHit = maybe False (hi >=) mTp
                                                                    exitOrTrail =
                                                                        case ecIntrabarFill cfg of
                                                                            StopFirst ->
                                                                                let (mStop, stopWhy) = stopPx trail0
                                                                                    stopHit = maybe False (lo <=) mStop
                                                                                 in if stopHit
                                                                                        then (Just (Data.Maybe.fromMaybe nextClose mStop, stopWhy), trail0)
                                                                                        else
                                                                                            if tpHit
                                                                                                then (Just (Data.Maybe.fromMaybe nextClose mTp, Just ExitTakeProfit), trail0)
                                                                                                else (Nothing, max trail0 hi)
                                                                            TakeProfitFirst ->
                                                                                if tpHit
                                                                                    then (Just (Data.Maybe.fromMaybe nextClose mTp, Just ExitTakeProfit), trail0)
                                                                                    else
                                                                                        let trail1 = max trail0 hi
                                                                                            (mStop, stopWhy) = stopPx trail1
                                                                                            stopHit = maybe False (lo <=) mStop
                                                                                         in if stopHit
                                                                                                then (Just (Data.Maybe.fromMaybe nextClose mStop, stopWhy), trail1)
                                                                                                else (Nothing, trail1)
                                                                 in case exitOrTrail of
                                                                        (Just (exitPx, reason), _trailUsed) ->
                                                                            let factor = markToMarket SideLong prev exitPx
                                                                                exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                                                (exitEq, costTotalsExit) = applyCostWithTotals t exitEq0 posSizeAfterSwitch costTotalsAfterSwitch
                                                                                tr =
                                                                                    Trade
                                                                                        { trEntryIndex = otEntryIndex otHeld
                                                                                        , trExitIndex = t + 1
                                                                                        , trEntryEquity = otEntryEquity otHeld
                                                                                        , trExitEquity = exitEq
                                                                                        , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                                        , trHoldingPeriods = otHoldingPeriods otHeld
                                                                                        , trEntryHighVolProb = metaAt (otEntryIndex otHeld) >>= smHighVolProb
                                                                                        , trEntrySource = TradeEntrySignal
                                                                                        , trExitReason = reason
                                                                                        , trEntryIp = Nothing
                                                                                        , trExitIp = Nothing
                                                                                        , trFeeCost = tradeFeeCost (otEntryIndex otHeld) (t + 1) (otEntryEquity otHeld) exitEq (otBaseSize otHeld)
                                                                                        }
                                                                             in (Nothing, 0, exitEq, costTotalsExit, changes' + 1, Nothing, tr : tradesAcc')
                                                                        (Nothing, trail1) ->
                                                                            let otCont = otHeld{otTrail = trail1}
                                                                             in (Just SideLong, posSizeAfterSwitch, equityAtClose, costTotalsAfterSwitch, changes', Just otCont, tradesAcc')
                                                            (Just SideShort, Just ot0) ->
                                                                let otHeld = ot0{otHoldingPeriods = otHoldingPeriods ot0 + 1}
                                                                    entryPx = otEntryPrice ot0
                                                                    trail0 = otTrail ot0
                                                                    mTp =
                                                                        case takeProfitFracAt t of
                                                                            Just tp -> Just (entryPx * (1 - tp))
                                                                            _ -> Nothing
                                                                    mSl =
                                                                        case stopLossFracAt t of
                                                                            Just sl -> Just (entryPx * (1 + sl))
                                                                            _ -> Nothing
                                                                    stopPx trail =
                                                                        let mTs =
                                                                                case trailingStopFracAt t of
                                                                                    Just ts -> Just (trail * (1 + ts))
                                                                                    _ -> Nothing
                                                                         in case (mSl, mTs) of
                                                                                (Nothing, Nothing) -> (Nothing, Nothing)
                                                                                (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                                                                (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                                                                (Just slPx, Just tsPx) ->
                                                                                    if tsPx < slPx
                                                                                        then (Just tsPx, Just ExitTrailingStop)
                                                                                        else (Just slPx, Just ExitStopLoss)
                                                                    tpHit = maybe False (lo <=) mTp
                                                                    exitOrTrail =
                                                                        case ecIntrabarFill cfg of
                                                                            StopFirst ->
                                                                                let (mStop, stopWhy) = stopPx trail0
                                                                                    stopHit = maybe False (hi >=) mStop
                                                                                 in if stopHit
                                                                                        then (Just (Data.Maybe.fromMaybe nextClose mStop, stopWhy), trail0)
                                                                                        else
                                                                                            if tpHit
                                                                                                then (Just (Data.Maybe.fromMaybe nextClose mTp, Just ExitTakeProfit), trail0)
                                                                                                else (Nothing, min trail0 lo)
                                                                            TakeProfitFirst ->
                                                                                if tpHit
                                                                                    then (Just (Data.Maybe.fromMaybe nextClose mTp, Just ExitTakeProfit), trail0)
                                                                                    else
                                                                                        let trail1 = min trail0 lo
                                                                                            (mStop, stopWhy) = stopPx trail1
                                                                                            stopHit = maybe False (hi >=) mStop
                                                                                         in if stopHit
                                                                                                then (Just (Data.Maybe.fromMaybe nextClose mStop, stopWhy), trail1)
                                                                                                else (Nothing, trail1)
                                                                 in case exitOrTrail of
                                                                        (Just (exitPx, reason), _trailUsed) ->
                                                                            let factor = markToMarket SideShort prev exitPx
                                                                                exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                                                (exitEq, costTotalsExit) = applyCostWithTotals t exitEq0 posSizeAfterSwitch costTotalsAfterSwitch
                                                                                tr =
                                                                                    Trade
                                                                                        { trEntryIndex = otEntryIndex otHeld
                                                                                        , trExitIndex = t + 1
                                                                                        , trEntryEquity = otEntryEquity otHeld
                                                                                        , trExitEquity = exitEq
                                                                                        , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                                        , trHoldingPeriods = otHoldingPeriods otHeld
                                                                                        , trEntryHighVolProb = metaAt (otEntryIndex otHeld) >>= smHighVolProb
                                                                                        , trEntrySource = TradeEntrySignal
                                                                                        , trExitReason = reason
                                                                                        , trEntryIp = Nothing
                                                                                        , trExitIp = Nothing
                                                                                        , trFeeCost = tradeFeeCost (otEntryIndex otHeld) (t + 1) (otEntryEquity otHeld) exitEq (otBaseSize otHeld)
                                                                                        }
                                                                             in (Nothing, 0, exitEq, costTotalsExit, changes' + 1, Nothing, tr : tradesAcc')
                                                                        (Nothing, trail1) ->
                                                                            let otCont = otHeld{otTrail = trail1}
                                                                             in (Just SideShort, posSizeAfterSwitch, equityAtClose, costTotalsAfterSwitch, changes', Just otCont, tradesAcc')
                                                            _ -> (posAfterSwitch, posSizeAfterSwitch, equityAtClose, costTotalsAfterSwitch, changes', openTrade', tradesAcc')

                                                    (fundingSide, fundingSize) =
                                                        if fundingOnOpen
                                                            then (posSideStart, posSizeStart)
                                                            else (posFinal, posSizeFinal)

                                                    (equityFinal, costTotalsFinalFunding) =
                                                        case fundingSide of
                                                            Just side -> applyFundingWithTotals equityFinal0 fundingSize side costTotalsFinal0
                                                            _ -> (equityFinal0, costTotalsFinal0)

                                                    (posFinal2, posSizeFinal2, equityFinal2, costTotalsFinal2, changesFinal2, openTradeFinal2, tradesFinal2, dead2) =
                                                        if isBad equityFinal || equityFinal <= 0
                                                            then
                                                                let exitEq = if isBad equityFinal then 0 else max 0 equityFinal
                                                                    (tradesOut, changesOut) =
                                                                        case openTradeFinal of
                                                                            Nothing -> (tradesFinal, changesFinal)
                                                                            Just otHeld ->
                                                                                let tr =
                                                                                        Trade
                                                                                            { trEntryIndex = otEntryIndex otHeld
                                                                                            , trExitIndex = t + 1
                                                                                            , trEntryEquity = otEntryEquity otHeld
                                                                                            , trExitEquity = exitEq
                                                                                            , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                                            , trHoldingPeriods = otHoldingPeriods otHeld
                                                                                            , trEntryHighVolProb = metaAt (otEntryIndex otHeld) >>= smHighVolProb
                                                                                            , trEntrySource = TradeEntrySignal
                                                                                            , trExitReason = Just ExitLiquidation
                                                                                            , trEntryIp = Nothing
                                                                                            , trExitIp = Nothing
                                                                                            , trFeeCost = tradeFeeCost (otEntryIndex otHeld) (t + 1) (otEntryEquity otHeld) exitEq (otBaseSize otHeld)
                                                                                            }
                                                                                 in (tr : tradesFinal, changesFinal + 1)
                                                                 in (Nothing, 0, exitEq, costTotalsFinalFunding, changesOut, Nothing, tradesOut, True)
                                                            else (posFinal, posSizeFinal, equityFinal, costTotalsFinalFunding, changesFinal, openTradeFinal, tradesFinal, False)

                                                    -- Re-check risk after the bar and flatten at close if limits are breached.
                                                    (posFinal3, posSizeFinal3, equityFinal3, costTotalsFinal3, changesFinal3, openTradeFinal3, tradesFinal3, haltReason2) =
                                                        if dead2 || Data.Maybe.isJust haltReason1
                                                            then (posFinal2, posSizeFinal2, equityFinal2, costTotalsFinal2, changesFinal2, openTradeFinal2, tradesFinal2, haltReason1)
                                                            else
                                                                let peakEqAfter = max peakEq1 equityFinal2
                                                                    drawdownAfter =
                                                                        if peakEqAfter > 0
                                                                            then max 0 (1 - equityFinal2 / peakEqAfter)
                                                                            else 0
                                                                    dailyLossAfter =
                                                                        if dayStartEq1 > 0
                                                                            then max 0 (1 - equityFinal2 / dayStartEq1)
                                                                            else 0
                                                                    weeklyLossAfter =
                                                                        if weekStartEq1 > 0
                                                                            then max 0 (1 - equityFinal2 / weekStartEq1)
                                                                            else 0
                                                                    expectancyAfter =
                                                                        if expectancyLookback <= 0
                                                                            then Nothing
                                                                            else
                                                                                let recent = take expectancyLookback tradesFinal2
                                                                                 in if length recent < expectancyLookback
                                                                                        then Nothing
                                                                                        else Just (meanList (map trReturn recent))
                                                                    riskHaltReason2 =
                                                                        specRiskHalt
                                                                            HaltInputs
                                                                                { hiPrevHaltReason = Nothing
                                                                                , hiDayChanged = False
                                                                                , hiWeekChanged = False
                                                                                , hiDailyLoss = dailyLossAfter
                                                                                , hiWeeklyLoss = weeklyLossAfter
                                                                                , hiDrawdown = drawdownAfter
                                                                                , hiExpectancy = expectancyAfter
                                                                                , hiMaxDailyLossLim = maxDailyLossLim
                                                                                , hiMaxWeeklyLossLim = maxWeeklyLossLim
                                                                                , hiMaxDrawdownLim = maxDrawdownLim
                                                                                , hiMinExpectancy = minExpectancy
                                                                                , hiPositionSize = max 0 posSizeFinal2
                                                                                , hiMaxPositionSizeLim =
                                                                                    if positionSizeBoundsOk
                                                                                        then Just maxPositionSize
                                                                                        else Nothing
                                                                                , hiConsecutiveLosses = lossStreak
                                                                                , hiMaxLossStreakLim =
                                                                                    if lossStreakMax > 0
                                                                                        then Just lossStreakMax
                                                                                        else Nothing
                                                                                , hiVolTarget = volTargetVal
                                                                                , hiLeverage = 0
                                                                                }
                                                                 in case riskHaltReason2 of
                                                                        Nothing -> (posFinal2, posSizeFinal2, equityFinal2, costTotalsFinal2, changesFinal2, openTradeFinal2, tradesFinal2, Nothing)
                                                                        Just r ->
                                                                            if isNothing posFinal2
                                                                                then (posFinal2, posSizeFinal2, equityFinal2, costTotalsFinal2, changesFinal2, openTradeFinal2, tradesFinal2, Just r)
                                                                                else
                                                                                    let (eqExit, costTotalsExit) = applyCostWithTotals t equityFinal2 posSizeFinal2 costTotalsFinal2
                                                                                        tradesAcc1 =
                                                                                            case openTradeFinal2 of
                                                                                                Nothing -> tradesFinal2
                                                                                                Just ot -> closeTradeAt (t + 1) r eqExit ot : tradesFinal2
                                                                                        changesOut = changesFinal2 + 1
                                                                                     in (Nothing, 0, eqExit, costTotalsExit, changesOut, Nothing, tradesAcc1, Just r)

                                                    exitedToFlat =
                                                        isNothing posFinal3 && (Data.Maybe.isJust posAfterSwitch || Data.Maybe.isJust posSide)

                                                    maxHoldCooldown =
                                                        if holdTooLong
                                                            then 1
                                                            else 0

                                                    cooldownAfterExit = max cooldownBars maxHoldCooldown

                                                    cooldownNext =
                                                        if isNothing posFinal3
                                                            then if exitedToFlat then cooldownAfterExit else cooldownNext0
                                                            else 0
                                                    mNewTrade =
                                                        case tradesFinal3 of
                                                            [] -> Nothing
                                                            (tr : _) ->
                                                                case tradesAcc of
                                                                    (tr0 : _) | tr == tr0 -> Nothing
                                                                    _ -> Just tr
                                                    lossStreakNext =
                                                        if lossStreakMax <= 0
                                                            then 0
                                                            else case mNewTrade of
                                                                Nothing -> lossStreak
                                                                Just tr ->
                                                                    let r = trReturn tr
                                                                     in if isBad r || r <= 0
                                                                            then lossStreak + 1
                                                                            else 0
                                                    lossStreakTriggered =
                                                        lossStreakMax > 0
                                                            && case mNewTrade of
                                                                Nothing -> False
                                                                Just tr ->
                                                                    let r = trReturn tr
                                                                     in (isBad r || r <= 0) && lossStreakNext >= lossStreakMax
                                                    cooldownNext2 =
                                                        if lossStreakTriggered && isNothing posFinal3
                                                            then max cooldownNext lossStreakCooldownBars
                                                            else cooldownNext
                                                    entryOccurred =
                                                        case (posSide, posFinal3) of
                                                            (Nothing, Just _) -> True
                                                            (Just side0, Just side1) -> side0 /= side1
                                                            _ -> False
                                                    dayTradesNext =
                                                        if entryOccurred
                                                            then dayTrades1 + 1
                                                            else dayTrades1
                                                    riskStateNext =
                                                        RiskState
                                                            { rsPeakEquity = max peakEq1 equityFinal3
                                                            , rsDayKey = dayKey1
                                                            , rsDayStartEquity = dayStartEq1
                                                            , rsWeekKey = weekKey1
                                                            , rsWeekStartEquity = weekStartEq1
                                                            , rsDayTrades = dayTradesNext
                                                            , rsHaltReason = haltReason2
                                                            }
                                                    posForBar =
                                                        case posFinal3 of
                                                            Nothing -> 0
                                                            Just side ->
                                                                let s = posSizeFinal3
                                                                 in if isBad s then 0 else sideSign side * s
                                                 in ( posFinal3
                                                    , posSizeFinal3
                                                    , equityFinal3
                                                    , equityFinal3 : eqAcc
                                                    , posForBar : posAcc
                                                    , agreeOk : agreeAcc
                                                    , agreeValid : agreeValidAcc
                                                    , changesFinal3
                                                    , openTradeFinal3
                                                    , tradesFinal3
                                                    , costTotalsFinal3
                                                    , costTotalsFinal3 : costAcc
                                                    , dead2
                                                    , cooldownNext2
                                                    , lossStreakNext
                                                    , riskStateNext
                                                    , factorOpenNext
                                                    , factorCloseNext
                                                    )

                                    (_finalPos, finalPosSize, finalEq, eqRev, posRev, agreeRev, agreeValidRev, changes, openTrade, tradesRev, finalCostTotals, costRev, _deadFinal, _cooldownFinal, _lossStreakFinal, _riskFinal, _factorOpenFinal, _factorCloseFinal) =
                                        foldl'
                                            stepFn
                                            ( Nothing :: Maybe PositionSide
                                            , 0 :: Double
                                            , 1.0
                                            , [1.0]
                                            , []
                                            , []
                                            , []
                                            , 0 :: Int
                                            , Nothing :: Maybe OpenTrade
                                            , []
                                            , emptyCostTotals
                                            , [emptyCostTotals]
                                            , False
                                            , 0 :: Int
                                            , 0 :: Int
                                            , RiskState
                                                { rsPeakEquity = 1.0
                                                , rsDayKey = dayKeyAt 0
                                                , rsDayStartEquity = 1.0
                                                , rsWeekKey = weekKeyAt 0
                                                , rsWeekStartEquity = 1.0
                                                , rsDayTrades = 0
                                                , rsHaltReason = Nothing
                                                }
                                            , 1.0
                                            , 1.0
                                            )
                                            [0 .. stepCount - 1]

                                    (eqRev', tradesRev', costRev') =
                                        case openTrade of
                                            Nothing -> (eqRev, tradesRev, costRev)
                                            Just ot ->
                                                let (exitEq, finalCostTotals') = applyCostWithTotals (max 0 (stepCount - 1)) finalEq finalPosSize finalCostTotals
                                                    s = max 0 (abs (otBaseSize ot))
                                                    entryRates = costPerSideBreakdown (otEntryIndex ot) s
                                                    exitRates = costPerSideBreakdown stepCount s
                                                    feeCost = otEntryEquity ot * ctFeeCost entryRates + exitEq * ctFeeCost exitRates
                                                    tr =
                                                        Trade
                                                            { trEntryIndex = otEntryIndex ot
                                                            , trExitIndex = stepCount
                                                            , trEntryEquity = otEntryEquity ot
                                                            , trExitEquity = exitEq
                                                            , trReturn = exitEq / otEntryEquity ot - 1
                                                            , trHoldingPeriods = otHoldingPeriods ot
                                                            , trEntryHighVolProb = metaAt (otEntryIndex ot) >>= smHighVolProb
                                                            , trEntrySource = TradeEntrySignal
                                                            , trExitReason = Just ExitEod
                                                            , trEntryIp = Nothing
                                                            , trExitIp = Nothing
                                                            , trFeeCost = feeCost
                                                            }
                                                    eqRev1 =
                                                        case eqRev of
                                                            [] -> [exitEq]
                                                            (_ : rest) -> exitEq : rest
                                                    costRev1 =
                                                        case costRev of
                                                            [] -> [finalCostTotals']
                                                            (_ : rest) -> finalCostTotals' : rest
                                                 in (eqRev1, tr : tradesRev, costRev1)
                                    eqCurve = reverse eqRev'
                                    costCurve = reverse costRev'
                                 in BacktestResult
                                        { brEquityCurve = eqCurve
                                        , brPositions = reverse posRev
                                        , brAgreementOk = reverse agreeRev
                                        , brAgreementValid = reverse agreeValidRev
                                        , brPositionChanges = changes
                                        , brTrades = reverse tradesRev'
                                        , brCostAttribution = costAttributionFromTotals eqCurve costCurve
                                        }

defaultOpenThreshold :: Double
defaultOpenThreshold = 0.01

data TradingEntryGateInputs = TradingEntryGateInputs
    { entryFeePerSide :: Double
    , rawEntryEdge :: Double
    , currentSide1 :: Maybe Bool
    }
    deriving (Show)

mkTradingEntryGateInputs :: Double -> Double -> Maybe Bool -> TradingEntryGateInputs
mkTradingEntryGateInputs = TradingEntryGateInputs

data EntryGateState = EntryGateState
    { needsEntry :: Bool
    , roundTripFeeFloor :: Double
    , entryEdge :: Maybe Double
    , edgeSpikeOk :: Bool
    , edgeHeadroomOk :: Bool
    , feeBufferOk :: Bool
    , entryGatesOk :: Bool
    , desiredSide1 :: Maybe Bool
    }
    deriving (Show)

mkEntryGateState :: TradingEntryGateInputs -> EntryGateState
mkEntryGateState cfg =
    let needsEntry' = isNothing (currentSide1 cfg)
        roundTripFeeFloor' =
            let feePerSide = entryFeePerSide cfg
             in if isBad feePerSide
                    then 0 / 0
                    else 2 * feePerSide
        entryEdge' = normalizeSignalEntryEdge (rawEntryEdge cfg)
        edgeSpikeOk' =
            not needsEntry' || signalEntryEdgeSpikeOk defaultOpenThreshold entryEdge'
        edgeHeadroomOk' =
            not needsEntry' || signalEntryHeadroomOk defaultOpenThreshold entryEdge'
        feeBufferOk' =
            not needsEntry' || signalEntryFeeBufferOk defaultOpenThreshold roundTripFeeFloor' entryEdge'
        entryGatesOk' =
            not needsEntry' || (edgeSpikeOk' && edgeHeadroomOk' && feeBufferOk')
        desiredSide1'
            | not needsEntry' = currentSide1 cfg
            | entryGatesOk' = Just True
            | otherwise = Nothing
     in EntryGateState
            { needsEntry = needsEntry'
            , roundTripFeeFloor = roundTripFeeFloor'
            , entryEdge = entryEdge'
            , edgeSpikeOk = edgeSpikeOk'
            , edgeHeadroomOk = edgeHeadroomOk'
            , feeBufferOk = feeBufferOk'
            , entryGatesOk = entryGatesOk'
            , desiredSide1 = desiredSide1'
            }

isBad :: Double -> Bool
isBad value = not (finiteDouble value)
