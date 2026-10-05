{-# LANGUAGE OverloadedStrings #-}

module Trader.Test.QuantityRounding (quantityRoundingSuite) where

import Control.Monad (forM_, unless)
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble)
import Trader.Binance (Step (..), quantizeDown)
import Trader.QuantityRounding (quantizeDownExact, quantizeUpExact, validOrderPrice, validateQuantityInput)

quantityRoundingSuite :: [(String, IO ())]
quantityRoundingSuite =
    [("downward rounding bounds and Binance delegation", testRounding)]

testRounding :: IO ()
testRounding = do
    let belowOne = castWord64ToDouble 0x3fefffffffffffff
    check "CE-ROUND-003" (quantizeUpExact 1 1 (castWord64ToDouble 0x3ff0000000000001) == 2)
    check "CE-ROUND-004" (not (validOrderPrice (0 / 0)))
    check "upward overflow rejection" (quantizeUpExact 1 (10 ^ (400 :: Int)) 1 == 0)
    check "CE-ROUND-002" ((validateQuantityInput (Just (1, 1)) (1 / 0) >> Right (1 :: Double)) == Left "Invalid quantity input.")
    check "CE-ROUND-001" (quantizeDown (Step 1 1 "1") belowOne == 0)
    check "conservative decimal boundary" (quantizeDown (Step 10 1 "0.1") 0.3 == 0.2)
    check "unchanged exact grid" (quantizeDown (Step 100 25 "0.25") 1.5 == 1.5)
    let words64 = take 1000 (iterate (\w -> w * 6364136223846793005 + 1442695040888963407) (20261005 :: Word64))
        values = map castWord64ToDouble (0 : 1 : 0x7ff0000000000000 : 0xfff0000000000000 : 0x7ff8000000000000 : words64)
    forM_ values $ \x ->
        forM_ [(1, 1), (100, 3), (10 ^ (400 :: Int), 1), (1, 10 ^ (400 :: Int)), (0, 1), (1, 0), (-1, 1), (1, -1)] $ \(scale, increment) -> do
            let y = quantizeDown (Step scale increment "fixture") x
                invalid = isNaN x || isInfinite x || x <= 0 || scale <= 0 || increment <= 0
            let checked = validateQuantityInput (Just (scale, increment)) x
                invalidMetadata = isNaN x || isInfinite x || scale <= 0 || increment <= 0
            check "preflight classification" (either (const True) (const False) checked == invalidMetadata)
            check "preflight blocks minimum fallback" (not invalidMetadata || either (const True) (const False) (checked >> Right (1 :: Double)))
            let up = quantizeUpExact scale increment x
            check "upward finite bound" (not (isNaN up || isInfinite up) && (up == 0 || up >= x))
            check "upward invalid fallback" (not invalid || up == 0)
            check "maker price validity" (validOrderPrice x == (not (isNaN x || isInfinite x) && x > 0))
            check "adapter parity" (y == quantizeDownExact scale increment x)
            check "finite nonnegative" (not (isNaN y || isInfinite y) && y >= 0)
            check "invalid fallback / non-increase" (if invalid then y == 0 else y <= x)
  where
    check label ok = unless ok (ioError (userError label))
