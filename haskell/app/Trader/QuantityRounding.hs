module Trader.QuantityRounding (quantizeDownExact, quantizeUpExact, validOrderPrice, validateQuantityInput) where

import Data.Ratio (denominator, numerator, (%))
import Trader.OrderNumeric (validateOrderNumber)

{- | Floor the exact value of a binary64 input to an exact rational grid.
Reject invalid inputs before partial conversions; validate the final Double
independently of the runtime's rational-to-floating conversion.
-}
quantizeDownExact :: Integer -> Integer -> Double -> Double
quantizeDownExact scale increment x
    | isNaN x || isInfinite x || x <= 0 || scale <= 0 || increment <= 0 = 0
    | otherwise =
        let value = toRational x
            units = (numerator value * scale) `div` (denominator value * increment)
            rounded = fromRational ((units * increment) % scale)
         in if isNaN rounded || isInfinite rounded || rounded < 0 || rounded > x
                then 0
                else rounded

-- | Errors deliberately do not match the minimum-size retry classifier.
validateQuantityInput :: Maybe (Integer, Integer) -> Double -> Either String ()
validateQuantityInput grid x
    | isNaN x || isInfinite x = Left "Invalid quantity input."
    | Just (scale, increment) <- grid, scale <= 0 || increment <= 0 = Left "Invalid quantity step."
    | otherwise = Right ()

-- | Ceiling on the exact rational grid, with a finite non-decrease guard.
quantizeUpExact :: Integer -> Integer -> Double -> Double
quantizeUpExact scale increment x
    | isNaN x || isInfinite x || x <= 0 || scale <= 0 || increment <= 0 = 0
    | otherwise =
        let value = toRational x
            divisor = denominator value * increment
            units = (numerator value * scale + divisor - 1) `div` divisor
            rounded = fromRational ((units * increment) % scale)
         in if isNaN rounded || isInfinite rounded || rounded < x
                then 0
                else rounded

-- | Invalid maker prices reject before any order or market fallback.
validOrderPrice :: Double -> Bool
validOrderPrice price = either (const False) (const True) (validateOrderNumber "Invalid maker price." price)
