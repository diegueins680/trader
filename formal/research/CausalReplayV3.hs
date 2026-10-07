module Main (main) where

import Data.Ratio ((%), denominator, numerator)
import Text.Read (readMaybe)

-- Independent exact oracle; differential evidence, not interpreter refinement.
mean :: [Rational] -> Maybe Rational
mean [] = Nothing
mean xs = Just (sum xs / fromIntegral (length xs))

lastValue :: [a] -> Maybe a
lastValue [] = Nothing
lastValue [x] = Just x
lastValue (_ : xs) = lastValue xs

lagReturn :: [Rational] -> Int -> Maybe Rational
lagReturn prices lag = do
    current <- lastValue prices
    old <- lastValue (take (length prices - lag) prices)
    if old > 0 then Just (current / old - 1) else Nothing

oracle :: [Rational] -> Maybe [Rational]
oracle values = do
    let (prices, rest) = splitAt 25 values
        (means, rest2) = splitAt 6 rest
        (widths, account) = splitAt 6 rest2
    if length prices /= 25 || length means /= 6 || length widths /= 6 || any (<= 0) (prices ++ widths)
        then Nothing
        else case account of
            [units, price, equity, peak] | equity > 0 && peak >= equity -> do
                trends <- traverse (lagReturn prices) [1, 3, 6, 24]
                let returns = zipWith (\x y -> abs (y / x - 1)) prices (drop 1 prices)
                short <- mean (drop 18 returns)
                long <- mean returns
                let raw = trends ++ [short, long]
                Just (zipWith (/) (zipWith (-) raw means) widths ++ [units * price / equity, 1 - equity / peak, equity])
            _ -> Nothing

decode :: (Integer, Integer) -> Maybe Rational
decode (n, d)
    | d > 0 = Just (n % d)
    | otherwise = Nothing

check :: String -> String
check line = case readMaybe line >>= traverse decode >>= oracle of
    Nothing -> "invalid"
    Just xs -> show [(numerator x, denominator x) | x <- xs]

main :: IO ()
main = interact (unlines . map check . lines)
