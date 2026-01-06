module Trader.Symbol
  ( splitSymbol
  , commonQuotes
  , isValidSymbolForPlatform
  , sanitizeSymbolForPlatform
  ) where

import Data.Char (isSpace, toLower)
import Data.List (dropWhileEnd, isSuffixOf, maximumBy)
import Data.Maybe (listToMaybe)
import Data.Ord (comparing)

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

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

normalizePlatform :: Maybe String -> Maybe String
normalizePlatform raw =
  case raw of
    Nothing -> Nothing
    Just v ->
      let s = map toLower (trim v)
       in if null s then Nothing else Just s

normalizeSymbolText :: String -> String
normalizeSymbolText = map toUpperAscii . trim

isAsciiAlphaNum :: Char -> Bool
isAsciiAlphaNum c =
  ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9')

isValidSymbolForPlatform :: Maybe String -> String -> Bool
isValidSymbolForPlatform platform raw =
  case normalizePlatform platform of
    Just "coinbase" -> isValidDelimitedSymbol '-' s
    Just "poloniex" -> isValidDelimitedSymbol '_' s
    _ -> isValidBinanceSymbol s
  where
    s = normalizeSymbolText raw

sanitizeSymbolForPlatform :: Maybe String -> String -> Maybe String
sanitizeSymbolForPlatform platform raw =
  let s = normalizeSymbolText raw
   in if null s
        then Nothing
        else case normalizePlatform platform of
          Just "coinbase" -> sanitizeDelimitedSymbol '-' '_' s
          Just "poloniex" -> sanitizeDelimitedSymbol '_' '-' s
          _ ->
            if isValidBinanceSymbol s
              then Just s
              else salvageBinanceSymbol s

isValidBinanceSymbol :: String -> Bool
isValidBinanceSymbol s =
  let n = length s
   in n >= 3 && n <= 30 && all isAsciiAlphaNum s

isValidDelimitedSymbol :: Char -> String -> Bool
isValidDelimitedSymbol delim s =
  case break (== delim) s of
    (a, _ : b) ->
      not (null a)
        && not (null b)
        && all isAsciiAlphaNum (a ++ b)
        && notElem delim b
    _ -> False

sanitizeDelimitedSymbol :: Char -> Char -> String -> Maybe String
sanitizeDelimitedSymbol delim alt s =
  if isValidDelimitedSymbol delim s
    then Just s
    else
      let s' = map (\c -> if c == alt then delim else c) s
       in if s' /= s && isValidDelimitedSymbol delim s' then Just s' else Nothing

salvageBinanceSymbol :: String -> Maybe String
salvageBinanceSymbol raw =
  let tokens = splitAlphaNumTokens raw
      quoteCandidates = filter endsWithQuote tokens
      pickFromQuotes = listToMaybe (filter isValidBinanceSymbol quoteCandidates)
      pickLongest =
        case filter isValidBinanceSymbol tokens of
          [] -> Nothing
          xs -> Just (maximumBy (comparing length) xs)
   in case pickFromQuotes of
        Just sym -> Just sym
        Nothing -> pickLongest

splitAlphaNumTokens :: String -> [String]
splitAlphaNumTokens =
  filter (not . null) . foldr step [""]
  where
    step c acc@(w:ws)
      | isAsciiAlphaNum c = (c:w) : ws
      | otherwise = "" : acc
    step _ [] = []

endsWithQuote :: String -> Bool
endsWithQuote token = any (`isSuffixOf` token) commonQuotes
