module Trader.Trading (
    BacktestResult (..),
    EnsembleConfig (..),
    StepMeta (..),
    IntrabarFill (..),
    PositionSide (..),
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

import Data.Maybe (isNothing)
import qualified Data.Vector as V
import Trader.SignalGates (
    finiteDouble,
    normalizeSignalEntryEdge,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )

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
    , ecMaxPositionSize :: !Double
    , ecBlendWeight :: !Double
    , ecKalmanZMin :: !Double
    , ecKalmanZMax :: !Double
    , ecLstmExitFlipBars :: !Int
    , ecLstmExitFlipGraceBars :: !Int
    , ecMetaMask :: !(Maybe (V.Vector Bool))
    , ecOpenTimes :: !(Maybe (V.Vector Int))
    , ecOpenPrices :: !(Maybe (V.Vector Double))
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

-- Keep CLI fill/position configuration types visible from Trader.Trading so
-- app/optimizer consumers share one stable public import seam.
data IntrabarFill = StopFirst | TakeProfitFirst
    deriving (Eq, Show)

data PositionSide = PositionLong | PositionShort
    deriving (Eq, Show)

data Positioning = LongFlat | LongShort
    deriving (Eq, Show)

-- Keep the legacy Main-facing simulation names wired to the checked vector
-- simulator surface.
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

data ExitReason = ExitEod
    deriving (Eq, Show)

exitReasonFromCode :: String -> Maybe ExitReason
exitReasonFromCode code
    | code == "eod" || code == "EOD" = Just ExitEod
    | otherwise = Nothing

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
    { trEntryEquity :: Double
    , trExitEquity :: Double
    , trReturn :: Double
    , trHoldingPeriods :: Int
    , trExitReason :: Maybe ExitReason
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