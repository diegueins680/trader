module Trader.LiveOrderIntent (
    LiveRiskHaltAction (..),
    desiredPositionForSignal,
    desiredPositionForSignalWithVolConf,
    liveRiskHaltAction,
    orderDirectionForTransition,
) where

import Data.Maybe (isJust)
import Trader.Trading (ExitReason, HaltInputs, Positioning (..), specRiskHalt)
import Trader.VolConfGate (VolConfGateBehavior (..))

{- | Pure live-bot view of the canonical simulator risk decision. A halt always
targets flat exposure, and its order direction is derived by the same
transition helper used by live execution.
-}
data LiveRiskHaltAction = LiveRiskHaltAction
    { lrhaExitReason :: !(Maybe ExitReason)
    , lrhaDesiredPosition :: !Int
    , lrhaOrderDirection :: !(Maybe Int)
    }
    deriving (Eq, Show)

normalizeDir :: Maybe Int -> Maybe Int
normalizeDir mDir =
    case mDir of
        Just d
            | d > 0 -> Just 1
            | d < 0 -> Just (-1)
        _ -> Nothing

desiredPositionForSignal :: Positioning -> Int -> Maybe Int -> Maybe Int -> Int
desiredPositionForSignal positioning previousPosition chosenDirection closeDirection =
    case normalizedPrevious of
        1 ->
            case chosen of
                Just 1 -> 1
                Just (-1) | allowShort -> -1
                _ -> if close == Just 1 then 1 else 0
        -1 ->
            case chosen of
                Just (-1) -> -1
                Just 1 | allowShort -> 1
                _ -> if close == Just (-1) then -1 else 0
        _ ->
            case chosen of
                Just 1 -> 1
                Just (-1) | allowShort -> -1
                _ -> 0
  where
    allowShort = positioning == LongShort
    normalizedPrevious
        | previousPosition > 0 = 1
        | previousPosition < 0 = -1
        | otherwise = 0
    chosen = normalizeDir chosenDirection
    close = normalizeDir closeDirection

{- | Apply the stateful volatility/confidence HOLD contract around the ordinary
signal transition. Latest-signal is intentionally stateless, so the live
surface supplies its held position here; the simulator supplies the same
state to 'applyVolConfGateBehavior'.
-}
desiredPositionForSignalWithVolConf :: VolConfGateBehavior -> Positioning -> Int -> Maybe Int -> Maybe Int -> Int
desiredPositionForSignalWithVolConf volConfBehavior positioning previousPosition chosenDirection closeDirection =
    case volConfBehavior of
        VolConfGateHold
            | normalizePosition previousPosition /= 0 -> normalizePosition previousPosition
        _ -> desiredPositionForSignal positioning previousPosition chosenDirection closeDirection

liveRiskHaltAction :: Int -> HaltInputs -> LiveRiskHaltAction
liveRiskHaltAction previousPosition inputs =
    let haltReason = specRiskHalt inputs
        desiredPosition =
            if isJust haltReason
                then 0
                else normalizePosition previousPosition
        orderDirection =
            if isJust haltReason
                then orderDirectionForTransition previousPosition desiredPosition
                else Nothing
     in LiveRiskHaltAction
            { lrhaExitReason = haltReason
            , lrhaDesiredPosition = desiredPosition
            , lrhaOrderDirection = orderDirection
            }

orderDirectionForTransition :: Int -> Int -> Maybe Int
orderDirectionForTransition previousPosition desiredPosition =
    case normalizedDesired of
        1 -> Just 1
        -1 -> Just (-1)
        0 ->
            if normalizedPrevious == 0
                then Nothing
                else Just (negate normalizedPrevious)
        _ ->
            if normalizedDesired > 0
                then Just 1
                else Just (-1)
  where
    normalizedPrevious = normalizePosition previousPosition
    normalizedDesired
        | desiredPosition > 0 = 1
        | desiredPosition < 0 = -1
        | otherwise = 0

normalizePosition :: Int -> Int
normalizePosition position
    | position > 0 = 1
    | position < 0 = -1
    | otherwise = 0
