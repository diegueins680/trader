-- Transient request decoder; no persisted artifact or authorization capability.
module SnapshotRequestV3 (decodeSnapshot, snapshotBits) where

import Data.Char (isDigit)
import Data.List (foldl', stripPrefix)
import Data.Word (Word64)
import GHC.Float (castDoubleToWord64, castWord64ToDouble)
import Text.Read (readMaybe)

decodeWord :: Integer -> Maybe Double
decodeWord word
    | word < 0 || word >= 18446744073709551616 = Nothing
    | otherwise =
        let value = castWord64ToDouble (fromInteger word :: Word64)
         in if isNaN value || isInfinite value || abs value > 1000 then Nothing else Just value

-- Only ASCII whitespace is accepted outside tokens. Counts and decimal length
-- are bounded before conversion; no generic list Read instance is needed.
spaces :: String -> String
spaces = dropWhile (`elem` " \t\r\n")

literal :: String -> String -> Maybe String
literal token = stripPrefix token . spaces

natural :: String -> Maybe (Integer, String)
natural raw =
    let (digits, rest) = span isDigit (spaces raw)
     in if null digits || length digits > 20
            then Nothing
            else Just (foldl' (\n c -> n * 10 + toInteger (fromEnum c - fromEnum '0')) 0 digits, rest)

wordsOf :: Int -> String -> Maybe ([Integer], String)
wordsOf count raw = literal "[" raw >>= go count
  where
    go remaining rest
        | remaining <= 0 = do
            end <- literal "]" rest
            pure ([], end)
        | otherwise = do
            (word, tailWords) <- natural rest
            next <- if remaining == 1 then Just tailWords else literal "," tailWords
            (following, end) <- go (remaining - 1) next
            pure (word : following, end)

decodeSnapshot :: String -> Maybe ([Double], [Double])
decodeSnapshot raw = do
    a <- literal "(" raw
    b <- literal "\"PPO-SNAPSHOT-V3\"" a
    c <- literal "," b
    (step, d) <- natural c
    e <- literal "," d
    (observation, f) <- wordsOf 12 e
    g <- literal "," f
    (parameters, h) <- wordsOf 259 g
    end <- literal ")" h
    if not (null (spaces end)) || step < 4 || step > 64 || step `mod` 4 /= 0 || length observation /= 12 || length parameters /= 259
        then Nothing
        else do
            obs <- traverse decodeWord observation
            params <- traverse decodeWord parameters
            pure (obs, params)

-- Exercise the internal Show/Read transport as well as the external bit codec.
snapshotBits :: String -> Maybe ([Word64], [Word64])
snapshotBits raw = do
    request <- decodeSnapshot raw
    (observation, parameters) <- readMaybe (show request) :: Maybe ([Double], [Double])
    pure (map castDoubleToWord64 observation, map castDoubleToWord64 parameters)
