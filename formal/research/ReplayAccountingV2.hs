module Main (main) where

import Data.Ratio ((%), denominator, numerator)
import Text.Read (readMaybe)

-- Independent exact wealth oracle for the Python replay's serialized receipts.
-- This is differential testing, not a proof of the Python interpreter.
reconcile :: [Rational] -> Maybe Rational
reconcile [equity, units, before, after, fundingPerUnit, fee, spread, slippage, impact] =
    Just (equity + units * (after - before) - units * fundingPerUnit - fee - spread - slippage - impact)
reconcile _ = Nothing

decode :: (Integer, Integer) -> Maybe Rational
decode (n, d)
    | d > 0 = Just (n % d)
    | otherwise = Nothing

check :: String -> String
check line = case readMaybe line >>= traverse decode >>= reconcile of
    Nothing -> "invalid"
    Just value -> show (numerator value, denominator value)

main :: IO ()
main = interact (unlines . map check . lines)
