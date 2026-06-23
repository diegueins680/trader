module Trader.LiveOrderIntent (
    desiredPositionForSignal,
    orderDirectionForTransition,
) where

import Trader.Trading (Positioning (..))

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
    normalizedPrevious
        | previousPosition > 0 = 1
        | previousPosition < 0 = -1
        | otherwise = 0
    normalizedDesired
        | desiredPosition > 0 = 1
        | desiredPosition < 0 = -1
        | otherwise = 0
