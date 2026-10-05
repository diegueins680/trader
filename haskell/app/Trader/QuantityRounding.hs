module Trader.QuantityRounding (quantizeDownExact) where

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
