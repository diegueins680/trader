{-# LANGUAGE OverloadedStrings #-}

module Trader.App.BinanceProbe (
    BinanceErrorInfo (..),
    binanceTradeTestConfirmsAuth,
    parseBinanceError,
) where

import Control.Applicative ((<|>))
import Data.Aeson (FromJSON (..), eitherDecode)
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isDigit, isSpace, toLower)
import Data.List (isInfixOf, stripPrefix)
import Data.Maybe (fromMaybe)
import Text.Read (readMaybe)

data BinanceErrorInfo = BinanceErrorInfo
    { beiHttpCode :: !(Maybe Int)
    , beiCode :: !(Maybe Int)
    , beiMsg :: !(Maybe String)
    , beiSummary :: !String
    }
    deriving (Eq, Show)

data BinanceErrorBody = BinanceErrorBody
    { bebCode :: !(Maybe Int)
    , bebMsg :: !(Maybe String)
    }
    deriving (Eq, Show)

instance FromJSON BinanceErrorBody where
    parseJSON = Aeson.withObject "BinanceErrorBody" $ \o -> do
        code <- o Aeson..:? "code"
        msg <- o Aeson..:? "msg"
        pure BinanceErrorBody{bebCode = code, bebMsg = msg}

parseBinanceError :: String -> BinanceErrorInfo
parseBinanceError raw =
    let raw' = truncateString 240 raw
        httpCode = extractHttpStatusCode raw'
        decoded =
            case extractJsonObject raw' of
                Nothing -> Nothing
                Just json ->
                    case eitherDecode (BL.fromStrict (BS.pack json)) of
                        Right body -> Just (body :: BinanceErrorBody)
                        Left _ -> Nothing
        labeled = extractLabeledBinanceError raw'
        outCode = (bebCode =<< decoded) <|> (fst =<< labeled) <|> httpCode
        outMsg = (bebMsg =<< decoded) <|> (snd =<< labeled)
        summary = fromMaybe raw' outMsg
     in BinanceErrorInfo
            { beiHttpCode = httpCode
            , beiCode = outCode
            , beiMsg = outMsg
            , beiSummary = summary
            }

binanceTradeTestConfirmsAuth :: Maybe Int -> String -> Bool
binanceTradeTestConfirmsAuth mCode summary =
    not (looksLikeAuthFailure mCode summary)
        && not (looksLikeTransientFailure mCode summary)
        && (maybe False isOrderValidationCode mCode || isOrderValidationSummary summary)

extractHttpStatusCode :: String -> Maybe Int
extractHttpStatusCode msg =
    let go [] = Nothing
        go ('H' : 'T' : 'T' : 'P' : ' ' : rest) =
            let digits = takeWhile isDigit rest
             in readMaybe digits
        go (_ : xs) = go xs
     in go msg

extractJsonObject :: String -> Maybe String
extractJsonObject msg =
    case dropWhile (/= '{') msg of
        [] -> Nothing
        s0 ->
            case break (== '}') s0 of
                (obj, '}' : _) -> Just (obj ++ "}")
                _ -> Nothing

extractLabeledBinanceError :: String -> Maybe (Maybe Int, Maybe String)
extractLabeledBinanceError = go
  where
    prefix = "Binance code "

    go [] = Nothing
    go s@(_ : xs) =
        case stripPrefix prefix s of
            Just rest ->
                let (digits, rest') = span (\c -> c == '-' || isDigit c) rest
                    msg = parseLabeledMsg rest'
                 in case readMaybe digits of
                        Just code -> Just (Just code, msg)
                        Nothing -> go xs
            Nothing -> go xs

    parseLabeledMsg rest =
        case dropWhile isSpace rest of
            ':' : more -> nonEmptyTrim more
            _ -> Nothing

looksLikeAuthFailure :: Maybe Int -> String -> Bool
looksLikeAuthFailure mCode summary =
    maybe False (`elem` authFailureCodes) mCode || any (`isInfixOf` lowerSummary) authFailurePhrases
  where
    lowerSummary = map toLower summary
    authFailureCodes = [(-2015), (-2014), (-1022), (-1021), 401, 403]
    authFailurePhrases =
        [ "invalid api-key"
        , "invalid api key"
        , "permissions for action"
        , "signature for this request"
        , "signature is not valid"
        , "timestamp for this request is outside of the recvwindow"
        , "recvwindow"
        , "api-key format invalid"
        ]

looksLikeTransientFailure :: Maybe Int -> String -> Bool
looksLikeTransientFailure mCode summary =
    maybe False (`elem` transientCodes) mCode || any (`isInfixOf` lowerSummary) transientPhrases
  where
    lowerSummary = map toLower summary
    transientCodes = [408, 429, 500, 502, 503, 504]
    transientPhrases =
        [ "timed out"
        , "timeout"
        , "too many requests"
        , "temporarily unavailable"
        , "service unavailable"
        , "internal error"
        , "bad gateway"
        ]

isOrderValidationCode :: Int -> Bool
isOrderValidationCode code =
    code == (-1013)
        || code == (-2010)
        || code == (-2011)
        || code == (-2021)
        || code == (-2022)
        || code == (-4164)
        || (code <= (-1100) && code >= (-1199))
        || (code <= (-4000) && code >= (-4999))

isOrderValidationSummary :: String -> Bool
isOrderValidationSummary summary = any (`isInfixOf` lowerSummary) validationPhrases
  where
    lowerSummary = map toLower summary
    validationPhrases =
        [ "insufficient balance"
        , "filter failure"
        , "precision is over the maximum"
        , "mandatory parameter"
        , "illegal characters found in parameter"
        , "invalid symbol"
        , "new order rejected"
        , "order would trigger immediately"
        , "price * qty is zero or less"
        , "notional"
        , "lot_size"
        , "lot size"
        , "min_qty"
        , "min qty"
        ]

nonEmptyTrim :: String -> Maybe String
nonEmptyTrim raw =
    let trimmed = trim raw
     in if null trimmed then Nothing else Just trimmed

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace
  where
    dropWhileEnd p = reverse . dropWhile p . reverse

truncateString :: Int -> String -> String
truncateString n s =
    if length s <= n then s else take n s ++ "..."
