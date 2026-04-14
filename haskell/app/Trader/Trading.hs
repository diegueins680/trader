{-# LANGUAGE PatternSynonyms #-}

module Trader.Trading (
    BacktestResult (..),
    EnsembleConfig (..),
    StepMeta (..),
    IntrabarFill (..),
    PositionSide (PositionLong, PositionShort, SideLong, SideShort),
    Positioning (..),
    simulateEnsemble,
    simulateEnsembleWithHLChecked,
    simulateEnsembleVWithHLChecked,
    ExitReason (..),
    TradeEntrySource (..),
    Trade (..),
    TradingEntryGateInputs (..),
    mkTradingEntryGateInputs,
    EntryGateState (..),
    exitReasonFromCode,
    mkEntryGateState,
    tradeEntrySourceCode,
) where

import Data.Char (toUpper)
import Data.Int (Int64)
import Data.Maybe (isNothing)
import qualified Data.Aeson as Aeson
import qualified Data.Text as T
import qualified Data.Vector as V
import Trader.Duration (TimeWindow)
import Trader.SignalGates (
    finiteDouble,
    normalizeSignalEntryEdge,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )
import Trader.VolConfGate (VolConfGatePreset)

-- Keep the optimizer/reporting simulation surface anchored in Trader.Trading so
-- the public import seam does not depend on a non-built auxiliary module.
data EnsembleConfig = EnsembleConfig
    { ecPeriodsPerYear :: !Double
    , ecOpenThreshold :: !Double
    , ecCloseThreshold :: !Double
    , ecMinEdge :: !Double
    , ecRouterLookback :: !Int
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
    , ecRebalanceBars :: !Int
    , ecRebalanceThreshold :: !Double
    , ecRebalanceGlobal :: !Bool
    , ecRebalanceResetOnSignal :: !Bool
    , ecFundingRate :: !Double
    , ecFundingBySide :: !Bool
    , ecFundingOnOpen :: !Bool
    , ecBlendWeight :: !Double
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
    , ecTriLayerExitOnSlow :: !Bool
    , ecKalmanBandLookback :: !Int
    , ecKalmanBandStdMult :: !Double
    , ecKalmanZMin :: !Double
    , ecKalmanZMax :: !Double
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

data Positioning = LongFlat | LongShort
    deriving (Eq, Show)

simulateEnsemble ::
    EnsembleConfig ->
    Int ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    BacktestResult
simulateEnsemble cfg periods series0 series1 series2 series3 series4 meta =
    case simulateEnsembleWithHLChecked cfg periods series0 series1 series2 series3 series4 meta of
        Left err -> error ("Trader.Trading.simulateEnsemble: " ++ err)
        Right result -> result

simulateEnsembleWithHLChecked ::
    EnsembleConfig ->
    Int ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Either String BacktestResult
simulateEnsembleWithHLChecked = simulateEnsembleVWithHLChecked

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
simulateEnsembleVWithHLChecked _ _ _ _ _ _ _ _ =
    error "Trader.Trading.simulateEnsembleVWithHLChecked: public surface shim"

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

data TradeEntrySource
    = TradeEntrySignal
    | TradeEntryPostDirectionGates
    deriving (Eq, Show)

tradeEntrySourceCode :: TradeEntrySource -> String
tradeEntrySourceCode entrySource =
    case entrySource of
        TradeEntrySignal -> "signal"
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
    }
    deriving (Eq, Show)

data BacktestResult = BacktestResult
    { brEquityCurve :: [Double]
    , brTrades :: [Trade]
    , brPositions :: [Double]
    , brAgreementOk :: [Bool]
    , brAgreementValid :: [Bool]
    , brPositionChanges :: Int
    }
    deriving (Eq, Show)

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
