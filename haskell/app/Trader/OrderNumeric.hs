module Trader.OrderNumeric (validateOrderNumber, validateMarketNumbers) where

-- | Validate a selected order quantity or price before effectful request work.
validateOrderNumber :: String -> Double -> Either String ()
validateOrderNumber message value
    | isNaN value || isInfinite value || value <= 0 = Left message
    | otherwise = Right ()

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
