module Trader.Trading (
    BacktestResult (..),
    EnsembleConfig (..),
    StepMeta (..),
    simulateEnsembleVWithHLChecked,
    ExitReason (..),
    Trade (..),
    TradingEntryGateInputs (..),
    mkTradingEntryGateInputs,
    EntryGateState (..),
    mkEntryGateState,
) where

import Data.Maybe (isNothing)
import Trader.SignalGates (
    finiteDouble,
    normalizeSignalEntryEdge,
    signalEntryEdgeSpikeOk,
    signalEntryFeeBufferOk,
    signalEntryHeadroomOk,
 )

data ExitReason = ExitEod
    deriving (Eq, Show)

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