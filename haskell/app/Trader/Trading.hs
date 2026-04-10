{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE RecordWildCards #-}

module Trader.Trading (
    BacktestResult (..),
    ExitReason (..),
    Trade (..),
    EnsembleConfig (..),
    StepMeta (..),
    simulateEnsembleVWithHLChecked,
    EntryGateInputs (..),
    EntryGateState (..),
    mkEntryGateState,
) where

import qualified Data.Maybe
import qualified Data.Vector as V
import Trader.SignalGates (
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )

-- Keep the optimizer-facing config/meta surface in Trader.Trading so callers
-- no longer depend on the removed Trader.Simulator shim.
data EnsembleConfig = EnsembleConfig
    { ecPeriodsPerYear :: !Double
    , ecOpenThreshold :: !Double
    , ecCloseThreshold :: !Double
    , ecMinEdge :: !Double
    , ecRouterLookback :: !Int
    , ecFee :: !Double
    }
    deriving (Eq, Show)

data StepMeta = StepMeta
    { smKalmanVar :: !Double
    , smKalmanMean :: !Double
    , smHighVolProb :: !(Maybe Double)
    , smConformalLo :: !(Maybe Double)
    , smConformalHi :: !(Maybe Double)
    , smQuantile10 :: !(Maybe Double)
    , smQuantile90 :: !(Maybe Double)
    }
    deriving (Eq, Show)

-- Keep the checked-simulator facade visible to optimizer-facing callers while
-- the canonical implementation is re-homed away from the removed shim.
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
    Left
        "Trader.Trading.simulateEnsembleVWithHLChecked: canonical simulator not linked in this module"

data ExitReason
    = ExitSignal
    | ExitStopLoss
    | ExitTakeProfit
    | ExitTime
    | ExitEod
    deriving (Eq, Show)

-- Keep the canonical backtest result surface exported so metrics and downstream
-- callers fail closed at compile time if this API drifts.
data Trade = Trade
    { trEntryEquity :: !Double
    , trExitEquity :: !Double
    , trReturn :: !Double
    , trHoldingPeriods :: !Int
    , trExitReason :: !(Maybe ExitReason)
    }
    deriving (Eq, Show)

data BacktestResult = BacktestResult
    { brEquityCurve :: ![Double]
    , brTrades :: ![Trade]
    , brPositions :: ![Double]
    , brAgreementOk :: ![Bool]
    , brAgreementValid :: ![Bool]
    , brPositionChanges :: !Int
    }
    deriving (Eq, Show)

data EntryGateInputs side t lookback cfg = EntryGateInputs
    { desiredSideRaw :: Maybe side
    , desiredSizeRaw :: Double
    , posSide :: Maybe side
    , volConfGateEnabled :: Bool
    , lstmEntryScaleRaw :: Double
    , trendOkAt :: t -> lookback -> side -> Bool
    , t :: t
    , trendLookbackStep :: lookback
    , volOkAt :: t -> Bool
    , ecFee :: cfg -> Double
    , cfg :: cfg
    , isBad :: Double -> Bool
    , minSignalToNoiseAdj :: Double
    , volPerBarAt :: t -> Maybe Double
    , clamp01 :: Double -> Double
    , edgeRaw :: Double
    , openThrAdj :: Double
    , snrOk :: Bool
    , volTargetReady :: Bool
    , triLayerOk :: Bool
    }

data EntryGateState side = EntryGateState
    { desiredSize0 :: Double
    , desiredSide0 :: Maybe side
    , needsEntry :: Bool
    , lstmEntryScale :: Double
    , trendOk :: Bool
    , volOk :: Bool
    , roundTripFeeFloor :: Double
    , snrScale :: Double
    , entryEdge :: Maybe Double
    , edgeSpikeOk :: Bool
    , edgeHeadroomOk :: Bool
    , feeBufferOk :: Bool
    , entryGatesOk :: Bool
    , desiredSide1 :: Maybe side
    }
    deriving (Eq, Show)

normalizedEntryEdge :: (Double -> Bool) -> Double -> Maybe Double
normalizedEntryEdge isBad edge =
    Just
        ( if isBad edge
              then 0
              else max 0 edge
        )

-- The live trading loop uses the same binding block to keep entry-only vetoes
-- fail-closed over one shared edge observation after the shim repair.
mkEntryGateState :: (Eq side) => EntryGateInputs side t lookback cfg -> EntryGateState side
mkEntryGateState EntryGateInputs{..} =
    let desiredSize0 =
            if Data.Maybe.isNothing desiredSideRaw
                then 0
                else max 0 desiredSizeRaw

        desiredSide0 =
            if desiredSize0 <= 0
                then Nothing
                else desiredSideRaw

        -- Only fresh entries should consult the entry-only veto gates below.
        needsEntry =
            Data.Maybe.isJust desiredSide0
                && desiredSide0 /= posSide

        lstmEntryScale =
            if needsEntry && not volConfGateEnabled
                then lstmEntryScaleRaw
                else 1

        trendOk =
            case desiredSide0 of
                Just side
                    | needsEntry ->
                        trendOkAt t trendLookbackStep side
                _ ->
                    True

        volOk =
            not needsEntry || volOkAt t

        roundTripFeeFloor =
            let feePerSide = ecFee cfg
             in if isBad feePerSide
                    then 0 / 0
                    else 2 * max 0 feePerSide

        snrScale =
            if minSignalToNoiseAdj <= 0
                then 1
                else case volPerBarAt t of
                    Just vol
                        | vol > 0 ->
                            clamp01 (max 0 edgeRaw / vol / minSignalToNoiseAdj)
                    _ ->
                        0

        -- Reuse the same fail-closed, non-negative edge sample across every
        -- fresh-entry veto gate.
        entryEdge =
            normalizedEntryEdge isBad edgeRaw

        edgeSpikeOk =
            not needsEntry
                || signalEntryEdgeSpikeOk openThrAdj entryEdge

        edgeHeadroomOk =
            not needsEntry
                || signalEntryHeadroomOk openThrAdj entryEdge

        feeBufferOk =
            not needsEntry
                || signalEntryFeeBufferOk openThrAdj roundTripFeeFloor entryEdge

        entryGatesOk =
            edgeSpikeOk
                && edgeHeadroomOk
                && feeBufferOk

        desiredSide1 =
            if not trendOk
                || not volOk
                || not snrOk
                || not volTargetReady
                || not triLayerOk
                || not entryGatesOk
                then Nothing
                else desiredSide0
     in EntryGateState{..}