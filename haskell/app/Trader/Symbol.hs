module Trader.Symbol
  ( splitSymbol
  , commonQuotes
  ) where

import Data.List (isSuffixOf)

commonQuotes :: [String]
commonQuotes =
  [ "USDT"
  , "USDC"
  , "FDUSD"
  , "TUSD"
  , "BUSD"
  , "BTC"
  , "ETH"
  , "BNB"
  ]

splitSymbol :: String -> (String, String)
splitSymbol symbol =
  let sym = map toUpperAscii symbol
   in case filter (`isSuffixOf` sym) commonQuotes of
        (q:_) -> (take (length sym - length q) sym, q)
        [] ->
          let n = length sym
           in (take (max 0 (n - 3)) sym, drop (max 0 (n - 3)) sym)

toUpperAscii :: Char -> Char
toUpperAscii c =
  if 'a' <= c && c <= 'z'
    then toEnum (fromEnum c - 32)
    else c
