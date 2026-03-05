module Trader.Symbol (
    splitSymbol,
    commonQuotes,
    isValidSymbolForPlatform,
    sanitizeComboSymbolForPlatform,
    sanitizeSymbolForPlatform,
) where

import Control.Applicative ((<|>))
import Data.Char (isAlphaNum, isAsciiLower, isAsciiUpper, isDigit, isSpace, toLower)
import Data.Bool (bool)
import Data.List (dropWhileEnd, find, foldl', isPrefixOf, isSuffixOf, maximumBy)
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
            (q : _) -> (take (length sym - length q) sym, q)
            [] ->
                let n = length sym
                 in splitAt (max 0 (n - 3)) sym

toUpperAscii :: Char -> Char
toUpperAscii c =
    if isAsciiLower c
        then toEnum (fromEnum c - 32)
        else c

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

nonEmptyString :: String -> Maybe String
nonEmptyString s =
    case s of
        "" -> Nothing
        _ -> Just s

normalizePlatform :: Maybe String -> Maybe String
normalizePlatform raw = raw >>= nonEmptyString . canonicalPlatformKey . normalizePlatformKey

normalizePlatformKey :: String -> String
normalizePlatformKey = map toLower . filter isAlphaNum . trim

canonicalPlatformKey :: String -> String
canonicalPlatformKey key
    | "coinbase" `isPrefixOf` key = "coinbase"
    | "poloniex" `isPrefixOf` key = "poloniex"
    | "binance" `isPrefixOf` key = "binance"
    | "uniswap" `isPrefixOf` key = "uniswap"
    | "curve" `isPrefixOf` key = "curve"
    | "sushiswap" `isPrefixOf` key = "sushiswap"
    | "balancer" `isPrefixOf` key = "balancer"
    | "pancakeswap" `isPrefixOf` key = "pancakeswap"
    | "1inch" `isPrefixOf` key = "oneinch"
    | "oneinch" `isPrefixOf` key = "oneinch"
    | otherwise = key

isDexPlatformKey :: String -> Bool
isDexPlatformKey key =
    key == "uniswap"
        || key == "curve"
        || key == "sushiswap"
        || key == "balancer"
        || key == "pancakeswap"
        || key == "1inch"
        || key == "oneinch"

normalizeSymbolForPlatform :: Maybe String -> String -> String
normalizeSymbolForPlatform platform raw =
    case normalizePlatform platform of
        Just key | isDexPlatformKey key -> trim raw
        _ -> normalizeSymbolText raw

normalizeSymbolText :: String -> String
normalizeSymbolText = map toUpperAscii . trim

isAsciiAlphaNum :: Char -> Bool
isAsciiAlphaNum c =
    isAsciiUpper c || isDigit c

isValidSymbolForPlatform :: Maybe String -> String -> Bool
isValidSymbolForPlatform platform raw =
    case normalizePlatform platform of
        Just key | isDexPlatformKey key -> not (null s)
        Just "coinbase" -> isValidDelimitedSymbol '-' s
        Just "poloniex" -> isValidDelimitedSymbol '_' s
        _ -> isValidBinanceSymbol s
  where
    s = normalizeSymbolForPlatform platform raw

sanitizeSymbolForPlatform :: Maybe String -> String -> Maybe String
sanitizeSymbolForPlatform platform raw =
    let s = normalizeSymbolForPlatform platform raw
     in if null s
            then Nothing
            else case normalizePlatform platform of
                Just key | isDexPlatformKey key -> Just s
                Just "coinbase" -> sanitizeDelimitedSymbol '-' '_' s
                Just "poloniex" -> sanitizeDelimitedSymbol '_' '-' s
                _ ->
                    if isValidBinanceSymbol s
                        then Just s
                        else salvageBinanceSymbol s

sanitizeComboSymbolForPlatform :: Maybe String -> String -> Maybe String
sanitizeComboSymbolForPlatform platform raw =
    case normalizePlatform platform of
        Just key | isDexPlatformKey key -> sanitizeSymbolForPlatform (Just key) raw
        Just "coinbase" -> sanitizeSymbolForPlatform (Just "coinbase") raw
        Just "poloniex" -> sanitizeSymbolForPlatform (Just "poloniex") raw
        _ -> sanitizeBinanceComboSymbol raw <|> sanitizeSymbolForPlatform platform raw

isValidBinanceSymbol :: String -> Bool
isValidBinanceSymbol s =
    let n = length s
     in n >= 3 && n <= 30 && all isAsciiAlphaNum s && any isAsciiUpper s

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
            let s' =
                    map
                        ( \c ->
                            if c == alt || c == '/'
                                then delim
                                else c
                        )
                        s
             in bool Nothing (Just s') (s' /= s && isValidDelimitedSymbol delim s')

salvageBinanceSymbol :: String -> Maybe String
salvageBinanceSymbol raw =
    let tokens = splitAlphaNumTokens raw
        joinedQuoteCandidates =
            [ joined
            | (a, b) <- zip tokens (drop 1 tokens)
            , b `elem` commonQuotes
            , let joined = a ++ b
            ]
        joinedPairCandidates =
            [ joined
            | (a, b) <- zip tokens (drop 1 tokens)
            , let joined = a ++ b
            ]
        quoteCandidates = filter endsWithKnownQuotePair tokens
        pickFromJoinedQuotes = find isValidBinanceSymbol joinedQuoteCandidates
        pickFromJoinedPairs = find isValidBinanceSymbol joinedPairCandidates
        pickFromQuotes = find isValidBinanceSymbol quoteCandidates
        pickLongest =
            case filter isValidBinanceSymbol tokens of
                [] -> Nothing
                xs -> Just (maximumBy (comparing length) xs)
     in pickFromJoinedQuotes <|> pickFromJoinedPairs <|> pickFromQuotes <|> pickLongest

splitAlphaNumTokens :: String -> [String]
splitAlphaNumTokens =
    filter (not . null) . foldr step [""]
  where
    step c acc@(w : ws)
        | isAsciiAlphaNum c = (c : w) : ws
        | otherwise = "" : acc
    step _ [] = []

endsWithKnownQuotePair :: String -> Bool
endsWithKnownQuotePair token = any (matchesQuote token) commonQuotes
  where
    matchesQuote sym quote = length sym > length quote && quote `isSuffixOf` sym

sanitizeBinanceComboSymbol :: String -> Maybe String
sanitizeBinanceComboSymbol raw =
    let s = normalizeSymbolText raw
        tokens = splitAlphaNumTokens s
        isValid sym =
            sym `notElem` commonQuotes && isValidBinanceSymbol sym
        pickTokenCandidate =
            case tokens of
                [] -> Nothing
                [a] -> bool Nothing (Just a) (isValid a)
                a : b : _rest ->
                    let joined = a ++ b
                     in if b `elem` commonQuotes && isValid joined
                            then Just joined
                            else
                                if isValid a && endsWithKnownQuotePair a
                                    then Just a
                                    else Nothing
        pickQuoteSuffix = trimBinanceComboSuffix s
     in pickQuoteSuffix <|> pickTokenCandidate <|> bool Nothing (Just s) (isValidBinanceSymbol s)

trimBinanceComboSuffix :: String -> Maybe String
trimBinanceComboSuffix raw =
    let compact = filter isAsciiAlphaNum (normalizeSymbolText raw)
        best = foldl' pickLongest Nothing (concatMap (trimQuoteCandidates compact) commonQuotes)
     in best
  where
    pickLongest acc candidate =
        case acc of
            Nothing -> Just candidate
            Just prev -> if length candidate > length prev then Just candidate else acc

trimQuoteCandidates :: String -> String -> [String]
trimQuoteCandidates compact quote =
    let positions = findSubstrPositions quote compact
        total = length compact
        quoteLen = length quote
     in [ candidate
        | idx <- positions
        , let end = idx + quoteLen
        , end < total
        , let suffix = drop end compact
        , any isDigit suffix
        , let candidate = take end compact
        , isValidBinanceSymbol candidate
        , candidate `notElem` commonQuotes
        ]

findSubstrPositions :: String -> String -> [Int]
findSubstrPositions needle hay =
    let go _ [] = []
        go i xs@(x : rest) =
            if needle `isPrefixOf` xs
                then i : go (i + 1) rest
                else go (i + 1) rest
     in case needle of
            "" -> []
            _ -> go 0 hay
