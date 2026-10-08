module Trader.OrderNumeric (validateOrderNumber, validateMarketNumbers, renderOrderNumber, orderWireUnits, gridUnits) where

import Data.Fixed (E12, Fixed (MkFixed))
import Numeric (floatToDigits)
import Text.Read (readMaybe)

-- | Validate a selected order quantity or price before effectful request work.
validateOrderNumber :: String -> Double -> Either String ()
validateOrderNumber message value
    | isNaN value || isInfinite value || value <= 0 = Left message
    | otherwise =
        case readMaybe (renderOrderNumber value) :: Maybe (Fixed E12) of
            Just (MkFixed units) | units > 0 -> Right ()
            _ -> Left (message ++ " after eight-decimal wire formatting")

-- | Match the existing wire selection: base wins; futures cannot use quote.
validateMarketNumbers :: Bool -> Maybe Double -> Maybe Double -> Either String ()
validateMarketNumbers futures quantity quoteOrderQty =
    case quantity of
        Just q -> validateOrderNumber "MARKET quantity must be finite and > 0" q
        Nothing
            | futures -> Left "Futures MARKET orders require --order-quantity (or compute it from --order-quote in the caller)"
            | otherwise ->
                case quoteOrderQty of
                    Just qq -> validateOrderNumber "MARKET quoteOrderQty must be finite and > 0" qq
                    Nothing -> Left "Provide quantity or quoteOrderQty for MARKET orders"

-- | Preserve the existing Binance wire representation. Validate before use.

{- | Decimal wire text for an order number (no scientific notation).

A value that is exactly the binary64 of its nearest 8-decimal grid value (every
exchange-quantized quantity or price) renders as that grid value, as before.
Any other value is truncated toward zero at 8 decimals instead of rounded to
nearest, so the wire value's binary64 never exceeds the checked value in
magnitude: rendering cannot raise an order above the cap it was checked
against (obligation 9). Digits are produced from exact integer units.
-}
renderOrderNumber :: Double -> String
renderOrderNumber x
    | isNaN x || isInfinite x = show x
    | otherwise = trimTrailingZeros (renderWireUnits (orderWireUnits x))

-- | The wire value of a finite order number in units of 1e-8 (see 'renderOrderNumber').
orderWireUnits :: Double -> Integer
orderWireUnits = gridUnits 8

{- | A finite value in units of @10^-decimals@. If the shortest decimal that
reads back as the input (its round-trip digits) has at most @decimals@
fractional digits, that decimal is used exactly: it is the value the binary64
stands for (an exchange grid value such as 0.29 or 0.1). Otherwise the exact
binary value is truncated toward zero. Either way the result reads back as a
binary64 no larger in magnitude than the input.
-}
gridUnits :: Int -> Double -> Integer
gridUnits decimals x
    | x == 0 = 0
    | otherwise =
        let (digits, exponent) = floatToDigits 10 (abs x)
            mantissa = foldl (\acc d -> acc * 10 + toInteger d) 0 digits
            fractionDigits = length digits - exponent
            scaleExp = max 0 decimals
            magnitude =
                if fractionDigits <= scaleExp
                    then mantissa * 10 ^ (scaleExp - fractionDigits)
                    else truncate (toRational (abs x) * 10 ^ scaleExp)
         in if x < 0 then negate magnitude else magnitude

renderWireUnits :: Integer -> String
renderWireUnits units =
    let (whole, frac) = abs units `quotRem` 100000000
        digits = show frac
     in (if units < 0 then "-" else "") ++ show whole ++ "." ++ replicate (8 - length digits) '0' ++ digits

trimTrailingZeros :: String -> String
trimTrailingZeros s =
    case break (== '.') s of
        (a, "") -> a
        (a, '.' : b) ->
            let b' = reverse (dropWhile (== '0') (reverse b))
             in if null b' then a else a ++ "." ++ b'
        _ -> s
