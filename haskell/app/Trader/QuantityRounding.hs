module Trader.QuantityRounding (quantizeDownExact, validateQuantityInput) where

import Data.Ratio (denominator, numerator, (%))

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
