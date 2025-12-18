module Trader.Text
  ( trim
  , normalizeKey
  ) where

import Data.Char (isAlphaNum, isSpace, toLower)
import Data.List (dropWhileEnd)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

normalizeKey :: String -> String
normalizeKey = map toLower . filter isAlphaNum
