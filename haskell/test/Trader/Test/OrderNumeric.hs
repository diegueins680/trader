module Trader.Test.OrderNumeric (orderNumericSuite) where

import Control.Monad (forM_, unless)
import Data.Word (Word64)
import GHC.Float (castWord64ToDouble)
import Trader.OrderNumeric (renderOrderNumber, validateMarketNumbers, validateOrderNumber)

orderNumericSuite :: [(String, IO ())]
orderNumericSuite = [("finite order-number admission and selected amount", testNumbers)]

testNumbers :: IO ()
testNumbers = do
    let accepted = either (const False) (const True)
        words64 = take 1000 (iterate (\w -> w * 6364136223846793005 + 1442695040888963407) (20261005 :: Word64))
        values = map castWord64ToDouble (0x3e112e0be826d694 : 0x3e35798ee2308c39 : 0x3e35798ee2308c3a : 0x3e35798ee2308c3b : 0 : 1 : 0x8000000000000000 : 0x7ff0000000000000 : 0xfff0000000000000 : 0x7ff8000000000000 : words64)
    check "CE-ORDER-NUM-001" (not (accepted (validateOrderNumber "invalid" (0 / 0))))
    check "CE-ORDER-NUM-002" (not (accepted (validateMarketNumbers False (Just (1 / 0)) (Just 1))))
    check "CE-ORDER-WIRE-001 formatter" (renderOrderNumber 1e-9 == "0")
    check "CE-ORDER-WIRE-001 guard" (not (accepted (validateOrderNumber "invalid" 1e-9)))
    check "wire tie" (renderOrderNumber 5e-9 == "0" && not (accepted (validateOrderNumber "invalid" 5e-9)))
    check "wire quantum" (renderOrderNumber 1e-8 == "0.00000001" && accepted (validateOrderNumber "invalid" 1e-8))
    forM_ values $ \x -> do
        -- The wire renderer never rounds up, so a positive wire value needs at least one 1e-8 unit.
        let valid = not (isNaN x || isInfinite x) && x >= 1e-8
        check "single number" (accepted (validateOrderNumber "invalid" x) == valid)
        forM_ [False, True] $ \futures -> do
            check "base priority" (accepted (validateMarketNumbers futures (Just x) (Just 1)) == valid)
            check "ignored quote" (accepted (validateMarketNumbers futures (Just 1) (Just x)))
            check "quote selection" (accepted (validateMarketNumbers futures Nothing (Just x)) == (not futures && valid))
            check "missing values" (not (accepted (validateMarketNumbers futures Nothing Nothing)))
  where
    check label ok = unless ok (ioError (userError label))
