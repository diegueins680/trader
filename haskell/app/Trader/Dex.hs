{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TupleSections #-}

module Trader.Dex (
    DexEnv (..),
    DexToken (..),
    DexSwapTx (..),
    DexSwapResult (..),
    dexNativeAddress,
    resolveDexEnv,
    resolveDexTokens,
    swapDexExactIn,
    tokenAmountToInteger,
    integerToTokenAmount,
) where

import Control.Applicative ((<|>))
import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, try)
import Control.Monad (when)
import Data.Aeson (Value (..))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Types as AT
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (isSpace, toLower)
import qualified Data.HashMap.Strict as HM
import Data.List (dropWhileEnd, find)
import Data.Maybe (fromMaybe)
import Data.Scientific (FPFormat (Fixed), Scientific, floatingOrInteger, formatScientific)
import Data.Text (Text)
import qualified Data.Text as T
import Network.HTTP.Client (Request (..), Response, parseRequest, responseBody)
import Network.HTTP.Types.URI (renderQuery)
import System.Environment (lookupEnv)
import System.Exit (ExitCode (..))
import System.Process (proc, readCreateProcessWithExitCode)
import Text.Read (readMaybe)

import Trader.Http (defaultRetryConfig, getSharedManager, httpLbsWithRetry)

-- 1inch native token placeholder (used for ETH/BNB/etc.)
dexNativeAddress :: String
dexNativeAddress = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"

nativeAliases :: [String]
nativeAliases = ["native", "eth", "bnb", "matic", "avax", "op", "arb"]

oneInchDefaultBaseUrl :: String
oneInchDefaultBaseUrl = "https://api.1inch.com/swap/v6.1"

data DexEnv = DexEnv
    { deChainId :: !Int
    , deRpcUrl :: !String
    , dePrivateKey :: !String
    , deAddress :: !String
    , deBaseUrl :: !String
    , deApiKey :: !(Maybe String)
    , deProtocols :: !(Maybe String)
    , deAutoApprove :: !Bool
    , deApproveWaitSec :: !Int
    }
    deriving (Eq, Show)

data DexToken = DexToken
    { dtAddress :: !String
    , dtSymbol :: !String
    , dtDecimals :: !Int
    }
    deriving (Eq, Show)

data DexSwapTx = DexSwapTx
    { dtxTo :: !String
    , dtxData :: !String
    , dtxValue :: !String
    , dtxGas :: !(Maybe String)
    , dtxGasPrice :: !(Maybe String)
    , dtxMaxFeePerGas :: !(Maybe String)
    , dtxMaxPriorityFeePerGas :: !(Maybe String)
    }
    deriving (Eq, Show)

data DexSwapResult = DexSwapResult
    { dsTxHash :: !String
    , dsTx :: !DexSwapTx
    , dsResponseBody :: !String
    , dsFromAmount :: !(Maybe Integer)
    , dsToAmount :: !(Maybe Integer)
    }
    deriving (Eq, Show)

resolveDexEnv :: Maybe Int -> Maybe Bool -> IO (Either String DexEnv)
resolveDexEnv chainOverride autoApproveOverride = do
    baseUrl <- resolveUrlEnv "TRADER_DEX_1INCH_BASE_URL" oneInchDefaultBaseUrl
    apiKey <- lookupEnv "TRADER_DEX_1INCH_API_KEY"
    apiKeyAlt <- lookupEnv "ONEINCH_API_KEY"
    rpcUrl <- lookupEnv "TRADER_DEX_RPC_URL"
    privKey <- lookupEnv "TRADER_DEX_PRIVATE_KEY"
    address <- lookupEnv "TRADER_DEX_ADDRESS"
    chainEnv <- lookupEnv "TRADER_DEX_CHAIN_ID"
    protocolsEnv <- lookupEnv "TRADER_DEX_PROTOCOLS"
    autoApproveEnv <- lookupEnv "TRADER_DEX_AUTO_APPROVE"
    waitEnv <- lookupEnv "TRADER_DEX_APPROVE_WAIT_SEC"
    let chainId = chainOverride <|> (chainEnv >>= readMaybe)
        autoApprove = fromMaybe True (autoApproveOverride <|> (autoApproveEnv >>= readBool))
        approveWait = fromMaybe 30 (waitEnv >>= readMaybe)
        apiKeyFinal = apiKey <|> apiKeyAlt
    case (chainId, rpcUrl, privKey, address) of
        (Nothing, _, _, _) -> pure (Left "Missing DEX chain id (set TRADER_DEX_CHAIN_ID or dexChainId).")
        (_, Nothing, _, _) -> pure (Left "Missing TRADER_DEX_RPC_URL for DEX trading.")
        (_, _, Nothing, _) -> pure (Left "Missing TRADER_DEX_PRIVATE_KEY for DEX trading.")
        (_, _, _, Nothing) -> pure (Left "Missing TRADER_DEX_ADDRESS for DEX trading.")
        (Just cid, Just rpc, Just pk, Just addr) ->
            pure
                ( Right
                    DexEnv
                        { deChainId = cid
                        , deRpcUrl = rpc
                        , dePrivateKey = normalizeHex pk
                        , deAddress = normalizeHex addr
                        , deBaseUrl = sanitizeBaseUrl baseUrl
                        , deApiKey = apiKeyFinal
                        , deProtocols = protocolsEnv >>= nonEmptyTrim
                        , deAutoApprove = autoApprove
                        , deApproveWaitSec = max 0 approveWait
                        }
                )

resolveDexTokens :: DexEnv -> String -> String -> Maybe Int -> Maybe Int -> IO (Either String (DexToken, DexToken))
resolveDexTokens env baseRaw quoteRaw baseDecimalsOverride quoteDecimalsOverride = do
    tokensOrErr <- fetchDexTokens env
    case tokensOrErr of
        Left err ->
            let baseToken = fallbackToken baseRaw baseDecimalsOverride
                quoteToken = fallbackToken quoteRaw quoteDecimalsOverride
             in case (baseToken, quoteToken) of
                    (Right b, Right q) -> pure (Right (b, q))
                    _ ->
                        pure
                            ( Left
                                ( "Failed to fetch 1inch tokens list: "
                                    ++ err
                                    ++ ". Provide dexBaseDecimals/dexQuoteDecimals and token addresses to bypass metadata lookup."
                                )
                            )
        Right tokenMap -> do
            baseTokenOrErr <- resolveDexToken tokenMap baseRaw baseDecimalsOverride
            quoteTokenOrErr <- resolveDexToken tokenMap quoteRaw quoteDecimalsOverride
            pure $ (,) <$> baseTokenOrErr <*> quoteTokenOrErr

swapDexExactIn :: DexEnv -> DexToken -> DexToken -> Integer -> Double -> Maybe String -> IO (Either String DexSwapResult)
swapDexExactIn env tokenIn tokenOut amountIn slippageFrac mProtocols = do
    if amountIn <= 0
        then pure (Left "DEX swap amount must be > 0.")
        else do
            allowResult <- ensureAllowance env tokenIn amountIn
            case allowResult of
                Left err -> pure (Left err)
                Right _ -> do
                    swapOrErr <- fetchDexSwap env tokenIn tokenOut amountIn slippageFrac mProtocols
                    case swapOrErr of
                        Left err -> pure (Left err)
                        Right (swapTx, body, mFromAmount, mToAmount) -> do
                            txHashOrErr <- sendDexTx env swapTx
                            case txHashOrErr of
                                Left err -> pure (Left err)
                                Right txHash ->
                                    pure
                                        ( Right
                                            DexSwapResult
                                                { dsTxHash = txHash
                                                , dsTx = swapTx
                                                , dsResponseBody = body
                                                , dsFromAmount = mFromAmount
                                                , dsToAmount = mToAmount
                                                }
                                        )

ensureAllowance :: DexEnv -> DexToken -> Integer -> IO (Either String ())
ensureAllowance env token amountIn =
    if isNativeToken token
        then pure (Right ())
        else do
            allowanceOrErr <- fetchDexAllowance env token
            case allowanceOrErr of
                Left err -> pure (Left err)
                Right allowance ->
                    if allowance >= amountIn
                        then pure (Right ())
                        else
                            if not (deAutoApprove env)
                                then pure (Left "DEX allowance too low; enable dexAutoApprove or pre-approve the token.")
                                else do
                                    approveOrErr <- fetchDexApproveTx env token
                                    case approveOrErr of
                                        Left err -> pure (Left err)
                                        Right approveTx -> do
                                            txHashOrErr <- sendDexTx env approveTx
                                            case txHashOrErr of
                                                Left err -> pure (Left err)
                                                Right txHash -> do
                                                    waitRes <- waitForReceipt env txHash
                                                    case waitRes of
                                                        Left err -> pure (Left err)
                                                        Right () -> pure (Right ())

fetchDexTokens :: DexEnv -> IO (Either String (HM.HashMap Text DexToken))
fetchDexTokens env = do
    let url = oneInchEndpoint env "tokens"
    req0 <- parseRequest url
    let req = applyApiKey env req0
    mgr <- getSharedManager
    respOrErr <- try (httpLbsWithRetry defaultRetryConfig (Just "1inch.tokens") mgr req) :: IO (Either SomeException (Response BL.ByteString))
    case respOrErr of
        Left ex -> pure (Left (show ex))
        Right resp ->
            case Aeson.eitherDecode (responseBody resp) of
                Left err -> pure (Left err)
                Right v -> pure (parseTokens v)

fetchDexAllowance :: DexEnv -> DexToken -> IO (Either String Integer)
fetchDexAllowance env token = do
    let url =
            oneInchEndpoint env "approve/allowance"
                ++ "?"
                ++ BS.unpack (renderQuery False [("tokenAddress", Just (BS.pack (dtAddress token))), ("walletAddress", Just (BS.pack (deAddress env)))])
    req0 <- parseRequest url
    let req = applyApiKey env req0
    mgr <- getSharedManager
    respOrErr <- try (httpLbsWithRetry defaultRetryConfig (Just "1inch.allowance") mgr req) :: IO (Either SomeException (Response BL.ByteString))
    case respOrErr of
        Left ex -> pure (Left (show ex))
        Right resp ->
            case Aeson.eitherDecode (responseBody resp) of
                Left err -> pure (Left err)
                Right (Aeson.Object o) ->
                    case KM.lookup (Key.fromString "allowance") o of
                        Just v ->
                            case parseIntegerValue v of
                                Left e -> pure (Left e)
                                Right n -> pure (Right n)
                        _ -> pure (Left "1inch allowance response missing allowance field.")
                Right _ -> pure (Left "Invalid 1inch allowance response.")

fetchDexApproveTx :: DexEnv -> DexToken -> IO (Either String DexSwapTx)
fetchDexApproveTx env token = do
    let url =
            oneInchEndpoint env "approve/transaction"
                ++ "?"
                ++ BS.unpack (renderQuery False [("tokenAddress", Just (BS.pack (dtAddress token)))])
    req0 <- parseRequest url
    let req = applyApiKey env req0
    mgr <- getSharedManager
    respOrErr <- try (httpLbsWithRetry defaultRetryConfig (Just "1inch.approve") mgr req) :: IO (Either SomeException (Response BL.ByteString))
    case respOrErr of
        Left ex -> pure (Left (show ex))
        Right resp ->
            case parseSwapTx (responseBody resp) of
                Left err -> pure (Left err)
                Right tx -> pure (Right tx)

fetchDexSwap :: DexEnv -> DexToken -> DexToken -> Integer -> Double -> Maybe String -> IO (Either String (DexSwapTx, String, Maybe Integer, Maybe Integer))
fetchDexSwap env tokenIn tokenOut amountIn slippageFrac mProtocols = do
    let slippagePct = max 0 (min 50 (slippageFrac * 100))
        baseParams =
            [ ("src", dtAddress tokenIn)
            , ("dst", dtAddress tokenOut)
            , ("amount", show amountIn)
            , ("from", deAddress env)
            , ("slippage", show slippagePct)
            , ("disableEstimate", "true")
            , ("allowPartialFill", "false")
            ]
        params =
            case mProtocols >>= nonEmptyTrim of
                Nothing -> baseParams
                Just protocols -> baseParams ++ [("protocols", protocols)]
        url = oneInchEndpoint env "swap" ++ "?" ++ BS.unpack (renderQuery False (map (\(k, v) -> (BS.pack k, Just (BS.pack v))) params))
    req0 <- parseRequest url
    let req = applyApiKey env req0
    mgr <- getSharedManager
    respOrErr <- try (httpLbsWithRetry defaultRetryConfig (Just "1inch.swap") mgr req) :: IO (Either SomeException (Response BL.ByteString))
    case respOrErr of
        Left ex -> pure (Left (show ex))
        Right resp -> do
            let body = responseBody resp
            case parseSwapTx body of
                Left err -> pure (Left err)
                Right tx ->
                    let (mFrom, mTo) = parseSwapAmounts body
                     in pure (Right (tx, shortResp body, mFrom, mTo))

parseSwapTx :: BL.ByteString -> Either String DexSwapTx
parseSwapTx body =
    case Aeson.eitherDecode body of
        Left err -> Left err
        Right (Aeson.Object o) ->
            case KM.lookup (Key.fromString "tx") o of
                Just v -> parseTxObject v
                Nothing -> parseTxObject (Aeson.Object o)
        Right _ -> Left "Invalid 1inch swap response."

parseTxObject :: Value -> Either String DexSwapTx
parseTxObject (Aeson.Object o) = do
    toAddr <- parseTextField o "to"
    dataField <- parseTextField o "data"
    valueField <- parseTextFieldDefault o "value" "0"
    gasField <- parseTextFieldMaybe o "gas"
    gasPriceField <- parseTextFieldMaybe o "gasPrice"
    maxFee <- parseTextFieldMaybe o "maxFeePerGas"
    maxPriority <- parseTextFieldMaybe o "maxPriorityFeePerGas"
    pure
        DexSwapTx
            { dtxTo = toAddr
            , dtxData = dataField
            , dtxValue = valueField
            , dtxGas = gasField
            , dtxGasPrice = gasPriceField
            , dtxMaxFeePerGas = maxFee
            , dtxMaxPriorityFeePerGas = maxPriority
            }
parseTxObject _ = Left "Invalid 1inch tx object."

parseSwapAmounts :: BL.ByteString -> (Maybe Integer, Maybe Integer)
parseSwapAmounts body =
    case Aeson.eitherDecode body of
        Left _ -> (Nothing, Nothing)
        Right (Aeson.Object o) ->
            let mFrom = KM.lookup (Key.fromString "fromAmount") o >>= parseIntegerValueMaybe
                mTo = KM.lookup (Key.fromString "toAmount") o >>= parseIntegerValueMaybe
             in (mFrom, mTo)
        Right _ -> (Nothing, Nothing)

applyApiKey :: DexEnv -> Request -> Request
applyApiKey env req =
    case deApiKey env of
        Nothing -> req
        Just key ->
            req
                { requestHeaders =
                    ("Authorization", BS.pack ("Bearer " ++ key))
                        : ("X-API-Key", BS.pack key)
                        : requestHeaders req
                }

sendDexTx :: DexEnv -> DexSwapTx -> IO (Either String String)
sendDexTx env tx = do
    let args =
            [ "send"
            , "--rpc-url"
            , deRpcUrl env
            , "--private-key"
            , dePrivateKey env
            , "--chain-id"
            , show (deChainId env)
            , "--data"
            , dtxData tx
            , "--value"
            , dtxValue tx
            ]
                ++ gasArgs tx
                ++ [dtxTo tx]
    result <- try (readCreateProcessWithExitCode (proc "cast" args) "") :: IO (Either SomeException (ExitCode, String, String))
    case result of
        Left ex -> pure (Left ("cast send failed: " ++ show ex))
        Right (exitCode, stdoutText, stderrText) ->
            case exitCode of
                ExitSuccess ->
                    case extractTxHash stdoutText of
                        Just h -> pure (Right h)
                        Nothing ->
                            if null (trimString stdoutText)
                                then pure (Left "cast send succeeded but no tx hash was returned.")
                                else pure (Right (trimString stdoutText))
                ExitFailure _ ->
                    pure (Left ("cast send failed: " ++ trimString (stdoutText ++ " " ++ stderrText)))

waitForReceipt :: DexEnv -> String -> IO (Either String ())
waitForReceipt env txHash = do
    let maxWait = max 0 (deApproveWaitSec env)
    if maxWait <= 0
        then pure (Right ())
        else go maxWait
  where
    go 0 = pure (Left "Timed out waiting for approval transaction receipt.")
    go n = do
        result <- try (readCreateProcessWithExitCode (proc "cast" ["receipt", "--json", "--rpc-url", deRpcUrl env, txHash]) "") :: IO (Either SomeException (ExitCode, String, String))
        case result of
            Left ex -> pure (Left ("cast receipt failed: " ++ show ex))
            Right (exitCode, out, _err) ->
                case exitCode of
                    ExitSuccess ->
                        if "\"status\":\"0x1\"" `isInfixOf` out || "\"status\": \"0x1\"" `isInfixOf` out
                            then pure (Right ())
                            else
                                if "\"status\":\"0x0\"" `isInfixOf` out || "\"status\": \"0x0\"" `isInfixOf` out
                                    then pure (Left "Approval transaction failed.")
                                    else do
                                        sleepSec 1
                                        go (n - 1)
                    ExitFailure _ -> do
                        sleepSec 1
                        go (n - 1)

sleepSec :: Int -> IO ()
sleepSec n =
    when (n > 0) $ do
        let micros = n * 1000000
        _ <- try (threadDelay micros) :: IO (Either SomeException ())
        pure ()

-- Utils

trimString :: String -> String
trimString = dropWhileEnd isSpace . dropWhile isSpace

nonEmptyTrim :: String -> Maybe String
nonEmptyTrim raw =
    let t = trimString raw
     in if null t then Nothing else Just t

sanitizeBaseUrl :: String -> String
sanitizeBaseUrl raw =
    let trimmed = trimString raw
        stripped = dropWhileEnd (== '/') trimmed
     in if null stripped then trimmed else stripped

normalizeHex :: String -> String
normalizeHex raw =
    let t = map toLower (trimString raw)
     in if "0x" `isPrefixOf` t then t else "0x" ++ t

resolveUrlEnv :: String -> String -> IO String
resolveUrlEnv key fallback = do
    raw <- lookupEnv key
    pure $ maybe fallback sanitizeBaseUrl (raw >>= nonEmptyTrim)

readBool :: String -> Maybe Bool
readBool raw =
    case map toLower (trimString raw) of
        "1" -> Just True
        "true" -> Just True
        "yes" -> Just True
        "on" -> Just True
        "0" -> Just False
        "false" -> Just False
        "no" -> Just False
        "off" -> Just False
        _ -> Nothing

oneInchEndpoint :: DexEnv -> String -> String
oneInchEndpoint env path = deBaseUrl env ++ "/" ++ show (deChainId env) ++ "/" ++ path

shortResp :: BL.ByteString -> String
shortResp body =
    let s = take 8000 (BS.unpack (BL.toStrict body))
     in if length s < 8000 then s else s ++ "..."

parseTokens :: Value -> Either String (HM.HashMap Text DexToken)
parseTokens =
    AT.parseEither $ \v ->
        AT.withObject "Tokens" (\o -> o AT..: "tokens" >>= parseTokenMap) v

parseTokenMap :: AT.Object -> AT.Parser (HM.HashMap Text DexToken)
parseTokenMap o =
    HM.fromList
        <$> traverse
            (\(k, v) -> (Key.toText k,) <$> parseTokenEntry (Key.toText k) v)
            (KM.toList o)

parseTokenEntry :: Text -> Value -> AT.Parser DexToken
parseTokenEntry addr =
    AT.withObject "Token" (\t -> DexToken (T.unpack addr) <$> t AT..: "symbol" <*> t AT..: "decimals")

fallbackToken :: String -> Maybe Int -> Either String DexToken
fallbackToken raw decOverride =
    case tokenInputKind raw of
        Left err -> Left err
        Right TokenInputNative ->
            case decOverride of
                Nothing -> Right (DexToken dexNativeAddress "NATIVE" 18)
                Just d -> Right (DexToken dexNativeAddress "NATIVE" d)
        Right (TokenInputAddress addr) ->
            case decOverride of
                Nothing -> Left "Token decimals are required when metadata lookup fails."
                Just d -> Right (DexToken addr addr d)
        Right (TokenInputSymbol _) -> Left "Token metadata lookup failed; provide a token address and decimals."

resolveDexToken :: HM.HashMap Text DexToken -> String -> Maybe Int -> IO (Either String DexToken)
resolveDexToken tokenMap raw decOverride =
    case tokenInputKind raw of
        Left err -> pure (Left err)
        Right TokenInputNative ->
            pure (Right (DexToken dexNativeAddress "NATIVE" 18))
        Right (TokenInputAddress addr) ->
            case lookupTokenByAddress tokenMap addr of
                Just tok ->
                    case decOverride of
                        Nothing -> pure (Right tok)
                        Just d -> pure (Right tok{dtDecimals = d})
                Nothing ->
                    case decOverride of
                        Nothing -> pure (Left ("Token not found in 1inch list: " ++ raw))
                        Just d -> pure (Right (DexToken addr raw d))
        Right (TokenInputSymbol _) ->
            case lookupTokenBySymbol tokenMap raw of
                Just tok ->
                    case decOverride of
                        Nothing -> pure (Right tok)
                        Just d -> pure (Right tok{dtDecimals = d})
                Nothing ->
                    case decOverride of
                        Nothing -> pure (Left ("Token not found in 1inch list: " ++ raw))
                        Just d -> pure (Left ("Token symbol lookup failed; provide an address for: " ++ raw ++ " (decimals=" ++ show d ++ ")."))

lookupTokenByAddress :: HM.HashMap Text DexToken -> String -> Maybe DexToken
lookupTokenByAddress tokenMap addr =
    HM.lookup (T.pack (map toLower addr)) tokenMap
        <|> HM.lookup (T.pack addr) tokenMap

lookupTokenBySymbol :: HM.HashMap Text DexToken -> String -> Maybe DexToken
lookupTokenBySymbol tokenMap raw =
    let sym = map toLower (trimString raw)
     in find (\tok -> map toLower (dtSymbol tok) == sym) (HM.elems tokenMap)

data TokenInput
    = TokenInputNative
    | TokenInputAddress String
    | TokenInputSymbol String
    deriving (Eq, Show)

tokenInputKind :: String -> Either String TokenInput
tokenInputKind raw =
    let t = trimString raw
        tLower = map toLower t
     in if null t
            then Left "Token value cannot be empty."
            else
                if tLower `elem` nativeAliases
                    then Right TokenInputNative
                    else
                        if isHexAddress tLower
                            then Right (TokenInputAddress tLower)
                            else Right (TokenInputSymbol t)

isHexAddress :: String -> Bool
isHexAddress t =
    "0x" `isPrefixOf` t && length t >= 42

isNativeToken :: DexToken -> Bool
isNativeToken tok = map toLower (dtAddress tok) == map toLower dexNativeAddress

tokenAmountToInteger :: Double -> Int -> Either String Integer
tokenAmountToInteger amt decimals
    | isNaN amt || isInfinite amt = Left "Amount is not a valid number."
    | amt <= 0 = Left "Amount must be > 0."
    | decimals < 0 = Left "Token decimals must be >= 0."
    | otherwise =
        let scale = (10 :: Integer) ^ decimals
            scaled = floor (amt * fromIntegral scale + 1e-9)
         in if scaled <= 0 then Left "Amount rounds to 0." else Right scaled

integerToTokenAmount :: Integer -> Int -> Double
integerToTokenAmount amt decimals =
    if decimals <= 0
        then fromIntegral amt
        else fromIntegral amt / (10 ^ decimals)

parseTextField :: AT.Object -> Text -> Either String String
parseTextField o k =
    case KM.lookup (Key.fromText k) o of
        Nothing -> Left ("Missing field: " ++ T.unpack k)
        Just v -> parseTextValue v

parseTextFieldDefault :: AT.Object -> Text -> String -> Either String String
parseTextFieldDefault o k def =
    case KM.lookup (Key.fromText k) o of
        Nothing -> Right def
        Just v -> parseTextValue v

parseTextFieldMaybe :: AT.Object -> Text -> Either String (Maybe String)
parseTextFieldMaybe o k =
    case KM.lookup (Key.fromText k) o of
        Nothing -> Right Nothing
        Just v -> Just <$> parseTextValue v

parseTextValue :: Value -> Either String String
parseTextValue v =
    case v of
        String t -> Right (T.unpack t)
        Number n -> Right (scientificToString n)
        _ -> Left "Unexpected JSON type for numeric field."

parseIntegerValue :: Value -> Either String Integer
parseIntegerValue v =
    case v of
        String t ->
            case readMaybe (T.unpack t) of
                Just n -> Right n
                Nothing -> Left "Invalid integer string."
        Number n ->
            case (floatingOrInteger n :: Either Double Integer) of
                Right i -> Right i
                Left _ -> Left "Invalid integer number."
        _ -> Left "Invalid integer value."

parseIntegerValueMaybe :: Value -> Maybe Integer
parseIntegerValueMaybe v =
    case parseIntegerValue v of
        Right n -> Just n
        Left _ -> Nothing

scientificToString :: Scientific -> String
scientificToString n =
    case (floatingOrInteger n :: Either Double Integer) of
        Right i -> show i
        Left _ -> formatScientific Fixed Nothing n

extractTxHash :: String -> Maybe String
extractTxHash raw =
    let tokens = words raw
        isHash t =
            let t' = map toLower t
             in "0x" `isPrefixOf` t' && length t' == 66
     in find isHash tokens

isInfixOf :: String -> String -> Bool
isInfixOf needle hay =
    case dropWhile (not . isPrefixOf needle) (tails hay) of
        [] -> False
        _ -> True

isPrefixOf :: String -> String -> Bool
isPrefixOf [] _ = True
isPrefixOf _ [] = False
isPrefixOf (x : xs) (y : ys) = x == y && isPrefixOf xs ys

tails :: [a] -> [[a]]
tails [] = [[]]
tails xs@(_ : ys) = xs : tails ys

gasArgs :: DexSwapTx -> [String]
gasArgs tx =
    let base = case dtxGas tx of
            Nothing -> []
            Just g -> ["--gas", g]
        feeArgs =
            case (dtxMaxFeePerGas tx, dtxMaxPriorityFeePerGas tx) of
                (Just maxFee, Just maxPrio) -> ["--max-fee-per-gas", maxFee, "--max-priority-fee-per-gas", maxPrio]
                _ -> case dtxGasPrice tx of
                    Nothing -> []
                    Just gp -> ["--gas-price", gp]
     in base ++ feeArgs
