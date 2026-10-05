module Trader.OrderNumeric (validateOrderNumber, validateMarketNumbers, renderOrderNumber) where

import Data.Fixed (E12, Fixed (MkFixed))
import Numeric (showFFloat)
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
renderOrderNumber :: Double -> String
renderOrderNumber x =
    -- Avoid scientific notation; Binance expects decimal strings.
    trimTrailingZeros (showFFloat (Just 8) x "")

trimTrailingZeros :: String -> String
trimTrailingZeros s =
    case break (== '.') s of
        (a, "") -> a
        (a, '.' : b) ->
            let b' = reverse (dropWhile (== '0') (reverse b))
             in if null b' then a else a ++ "." ++ b'
        _ -> s
