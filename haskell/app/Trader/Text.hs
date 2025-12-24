module Trader.Text
  ( trim
  , normalizeKey
  , dedupeStable
  ) where

import Data.Char (isAlphaNum, isSpace, toLower)
import Data.List (dropWhileEnd)
import qualified Data.Set as Set

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

normalizeKey :: String -> String
normalizeKey = map toLower . filter isAlphaNum

dedupeStable :: Ord a => [a] -> [a]
dedupeStable xs = go Set.empty xs
  where
    go _ [] = []
    go seen (y:ys)
      | Set.member y seen = go seen ys
      | otherwise = y : go (Set.insert y seen) ys
