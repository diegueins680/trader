module Trader.Trading (
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
        desiredSide1' =
            if not needsEntry'
                then currentSide1 cfg
                else
                    if entryGatesOk'
                        then Just True
                        else Nothing
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
