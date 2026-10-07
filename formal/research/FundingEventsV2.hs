module Main (main) where

import Data.Char (isDigit)
import Data.Ratio (denominator, numerator, (%))
import Text.Read (readMaybe)

-- Independent exact funding oracle for the Python successor's published buckets.
-- It parses by characters and buckets by prefix counting rather than a sweep.
-- This is differential testing, not a proof of the Python interpreter.
digits :: Int -> String -> Bool
digits limit s = not (null s) && length s <= limit && all isDigit s

decimal :: String -> Maybe Rational
decimal ('-' : rest) = negate <$> unsigned rest
decimal s = unsigned s

unsigned :: String -> Maybe Rational
unsigned s = case break (== '.') s of
    (whole, "") | wholeOk whole -> Just (fromInteger (read whole))
    (whole, '.' : frac)
        | wholeOk whole && digits 20 frac ->
            Just (read (whole ++ frac) % (10 ^ length frac))
    _ -> Nothing
  where
    wholeOk w = digits 20 w && (w == "0" || head w /= '0')

time :: String -> Maybe Integer
time s
    | digits 19 s && (s == "0" || head s /= '0') && read s < 2 ^ (63 :: Int) = Just (read s)
    | otherwise = Nothing

record :: (String, String, String) -> Maybe (Integer, Rational, Rational)
record (t, r, m) = do
    t' <- time t
    r' <- decimal r
    m' <- decimal m
    if m' > 0 then Just (t', r', m') else Nothing

load :: ([Integer], [(String, String, String)]) -> Maybe [(Int, Rational)]
load (closes, rows)
    | null closes || length closes > 8192 || length rows > 65536 = Nothing
    | any (\c -> c < 0 || c >= 2 ^ (63 :: Int)) closes = Nothing
    | not (and (zipWith (<) closes (drop 1 closes))) = Nothing
    | otherwise = do
        parsed <- traverse record rows
        let times = [t | (t, _, _) <- parsed]
        if and (zipWith (<) times (drop 1 times)) then Just () else Nothing
        let index t = length (takeWhile (< t) closes)
            located = [(index t, r * m) | (t, r, m) <- parsed]
        if all ((< length closes) . fst) located then Just () else Nothing
        let bucket j = [v | (k, v) <- located, k == j]
            out = [(length (bucket j), sum (bucket j)) | j <- [0 .. length closes - 1]]
        if all ((<= 128) . fst) out then Just out else Nothing

check :: String -> String
check line = case readMaybe line >>= load of
    Nothing -> "invalid"
    Just out -> show [(n, numerator v, denominator v) | (n, v) <- out]

main :: IO ()
main = interact (unlines . map check . lines)
