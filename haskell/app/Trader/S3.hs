{-# LANGUAGE OverloadedStrings #-}
module Trader.S3
  ( AwsCredentials(..)
  , S3State(..)
  , resolveS3State
  , s3KeyFor
  , s3GetObject
  , s3PutObject
  ) where

import Control.Applicative ((<|>))
import Control.Exception (SomeException, try)
import Crypto.Hash (Digest, SHA256, hash)
import Crypto.MAC.HMAC (HMAC, hmac, hmacGetDigest)
import Data.Aeson (FromJSON(..), eitherDecodeStrict', withObject, (.:), (.:?))
import Data.ByteArray (convert)
import Data.Char (isSpace)
import Data.List (intercalate)
import Data.Maybe (fromMaybe)
import Data.Time (UTCTime, defaultTimeLocale, formatTime, getCurrentTime)
import Network.HTTP.Client (Response, method, parseRequest, requestBody, requestHeaders, responseBody, responseStatus)
import Network.HTTP.Types (statusCode)
import Network.HTTP.Types.URI (urlEncode)
import System.Environment (lookupEnv)

import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import qualified Network.HTTP.Client as HTTP
import Trader.Http (RetryConfig(..), defaultRetryConfig, getSharedManager, httpLbsWithRetry)

data AwsCredentials = AwsCredentials
  { awsAccessKeyId :: !BS.ByteString
  , awsSecretAccessKey :: !BS.ByteString
  , awsSessionToken :: !(Maybe BS.ByteString)
  } deriving (Eq, Show)

data S3State = S3State
  { s3Region :: !String
  , s3Bucket :: !String
  , s3Prefix :: !String
  , s3Creds :: !AwsCredentials
  } deriving (Eq, Show)

data ContainerCredentials = ContainerCredentials
  { ccAccessKeyId :: !String
  , ccSecretAccessKey :: !String
  , ccToken :: !(Maybe String)
  } deriving (Eq, Show)

s3RetryConfig :: RetryConfig
s3RetryConfig = defaultRetryConfig { rcRetryWrites = True }

instance FromJSON ContainerCredentials where
  parseJSON =
    withObject "ContainerCredentials" $ \o -> do
      key <- o .: "AccessKeyId"
      secret <- o .: "SecretAccessKey"
      token <- (o .:? "Token") <|> (o .:? "SessionToken")
      pure ContainerCredentials { ccAccessKeyId = key, ccSecretAccessKey = secret, ccToken = token }

resolveS3State :: IO (Maybe S3State)
resolveS3State = do
  mBucket <- lookupEnv "TRADER_STATE_S3_BUCKET"
  case trim <$> mBucket of
    Nothing -> pure Nothing
    Just "" -> pure Nothing
    Just bucket -> do
      regionEnv <- lookupEnv "TRADER_STATE_S3_REGION"
      awsRegion <- lookupEnv "AWS_REGION"
      awsDefaultRegion <- lookupEnv "AWS_DEFAULT_REGION"
      let region = fromMaybe "us-east-1" (firstNonEmpty [regionEnv, awsRegion, awsDefaultRegion])
      prefixEnv <- lookupEnv "TRADER_STATE_S3_PREFIX"
      credsOrErr <- resolveAwsCredentials
      case credsOrErr of
        Left _ -> pure Nothing
        Right creds ->
          pure
            ( Just
                S3State
                  { s3Region = region
                  , s3Bucket = bucket
                  , s3Prefix = trimSlashes (fromMaybe "" prefixEnv)
                  , s3Creds = creds
                  }
            )

s3KeyFor :: S3State -> [String] -> String
s3KeyFor st parts =
  let prefix = trimSlashes (s3Prefix st)
      rest = trimSlashes (intercalate "/" (filter (not . null) parts))
   in case (prefix, rest) of
        ("", "") -> ""
        ("", _) -> rest
        (_, "") -> prefix
        _ -> prefix ++ "/" ++ rest

s3GetObject :: S3State -> String -> IO (Either String (Maybe BL.ByteString))
s3GetObject st key = do
  respOrErr <- s3Request st "GET" key BL.empty
  case respOrErr of
    Left err -> pure (Left err)
    Right resp -> do
      let code = statusCode (responseStatus resp)
      if code == 404
        then pure (Right Nothing)
        else
          if code >= 200 && code < 300
            then pure (Right (Just (responseBody resp)))
            else pure (Left ("S3 GET failed (status " ++ show code ++ ")"))

s3PutObject :: S3State -> String -> BL.ByteString -> IO (Either String ())
s3PutObject st key body = do
  respOrErr <- s3Request st "PUT" key body
  case respOrErr of
    Left err -> pure (Left err)
    Right resp -> do
      let code = statusCode (responseStatus resp)
      if code >= 200 && code < 300
        then pure (Right ())
        else pure (Left ("S3 PUT failed (status " ++ show code ++ ")"))

-- Internal helpers

resolveAwsCredentials :: IO (Either String AwsCredentials)
resolveAwsCredentials = do
  mAccess <- lookupEnv "AWS_ACCESS_KEY_ID"
  mSecret <- lookupEnv "AWS_SECRET_ACCESS_KEY"
  mToken <- lookupEnv "AWS_SESSION_TOKEN"
  case (trim <$> mAccess, trim <$> mSecret) of
    (Just access, Just secret) | not (null access) && not (null secret) ->
      pure
        ( Right
            AwsCredentials
              { awsAccessKeyId = BS.pack access
              , awsSecretAccessKey = BS.pack secret
              , awsSessionToken = BS.pack <$> (mToken >>= nonEmpty)
              }
        )
    _ -> resolveContainerCredentials

resolveContainerCredentials :: IO (Either String AwsCredentials)
resolveContainerCredentials = do
  mFull <- lookupEnv "AWS_CONTAINER_CREDENTIALS_FULL_URI"
  mRel <- lookupEnv "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
  let uri =
        case trim <$> mFull of
          Just v | not (null v) -> Just v
          _ ->
            case trim <$> mRel of
              Just v | not (null v) -> Just ("http://169.254.170.2" ++ v)
              _ -> Nothing
  case uri of
    Nothing -> pure (Left "AWS credentials not found (env or container credentials).")
    Just credUri -> do
      mAuth <- lookupEnv "AWS_CONTAINER_AUTHORIZATION_TOKEN"
      mAuthFile <- lookupEnv "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"
      tokenFromFile <-
        case mAuthFile >>= nonEmpty of
          Nothing -> pure Nothing
          Just path -> do
            contents <- try (readFile path) :: IO (Either SomeException String)
            pure (either (const Nothing) (nonEmpty . trim) contents)
      let auth = tokenFromFile <|> (mAuth >>= nonEmpty)
      manager <- getSharedManager
      req0 <- parseRequest credUri
      let headers =
            case auth of
              Nothing -> requestHeaders req0
              Just tok -> ("Authorization", BS.pack tok) : requestHeaders req0
          req = req0 { requestHeaders = headers }
      respOrErr <- try (httpLbsWithRetry defaultRetryConfig (Just "aws.credentials") manager req) :: IO (Either SomeException (Response BL.ByteString))
      case respOrErr of
        Left e -> pure (Left ("Failed to fetch container credentials: " ++ show e))
        Right resp ->
          if statusCode (responseStatus resp) < 200 || statusCode (responseStatus resp) >= 300
            then pure (Left "Container credentials request failed.")
            else
              case eitherDecodeStrict' (BL.toStrict (responseBody resp)) of
                Left err -> pure (Left ("Failed to decode container credentials: " ++ err))
                Right cc ->
                  pure
                    ( Right
                        AwsCredentials
                          { awsAccessKeyId = BS.pack (ccAccessKeyId cc)
                          , awsSecretAccessKey = BS.pack (ccSecretAccessKey cc)
                          , awsSessionToken = BS.pack <$> ccToken cc
                          }
                    )

s3Request :: S3State -> BS.ByteString -> String -> BL.ByteString -> IO (Either String (Response BL.ByteString))
s3Request st reqMethod key body = do
  now <- getCurrentTime
  manager <- getSharedManager
  let host = s3Host st
      payloadHash = sha256Hex body
      amzDate = formatAmzDate now
      shortDate = formatShortDate now
      encodedPath = s3CanonicalPath key
      signedHeaders =
        if awsSessionToken (s3Creds st) == Nothing
          then "host;x-amz-content-sha256;x-amz-date"
          else "host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
      canonicalHeaders =
        concat
          [ "host:" ++ host ++ "\n"
          , "x-amz-content-sha256:" ++ BS.unpack payloadHash ++ "\n"
          , "x-amz-date:" ++ amzDate ++ "\n"
          , maybe "" (\tok -> "x-amz-security-token:" ++ BS.unpack tok ++ "\n") (awsSessionToken (s3Creds st))
          ]
      canonicalRequest =
        intercalate
          "\n"
          [ BS.unpack reqMethod
          , encodedPath
          , ""
          , canonicalHeaders
          , signedHeaders
          , BS.unpack payloadHash
          ]
      scope = shortDate ++ "/" ++ s3Region st ++ "/s3/aws4_request"
      stringToSign =
        intercalate
          "\n"
          [ "AWS4-HMAC-SHA256"
          , amzDate
          , scope
          , BS.unpack (sha256Hex (BL.fromStrict (BS.pack canonicalRequest)))
          ]
      signature = signatureHex (s3Creds st) (BS.pack shortDate) (BS.pack (s3Region st)) (BS.pack stringToSign)
      authHeader =
        "AWS4-HMAC-SHA256 Credential="
          ++ BS.unpack (awsAccessKeyId (s3Creds st))
          ++ "/"
          ++ scope
          ++ ", SignedHeaders="
          ++ signedHeaders
          ++ ", Signature="
          ++ BS.unpack signature
      url = "https://" ++ host ++ encodedPath
  req0 <- parseRequest url
  let headers =
        [ ("Host", BS.pack host)
        , ("x-amz-content-sha256", payloadHash)
        , ("x-amz-date", BS.pack amzDate)
        , ("Authorization", BS.pack authHeader)
        ]
          ++ maybe [] (\tok -> [("x-amz-security-token", tok)]) (awsSessionToken (s3Creds st))
      req =
        req0
          { method = reqMethod
          , requestHeaders = headers
          , requestBody = HTTP.RequestBodyLBS body
          }
  respOrErr <- try (httpLbsWithRetry s3RetryConfig (Just "s3.request") manager req) :: IO (Either SomeException (Response BL.ByteString))
  case respOrErr of
    Left e -> pure (Left ("S3 request failed: " ++ show e))
    Right resp -> pure (Right resp)

s3Host :: S3State -> String
s3Host st = s3Bucket st ++ ".s3." ++ s3Region st ++ ".amazonaws.com"

s3CanonicalPath :: String -> String
s3CanonicalPath raw =
  let trimmed = trimSlashes raw
   in if null trimmed
        then "/"
        else "/" ++ intercalate "/" (map encodeSegment (splitOnSlash trimmed))
  where
    encodeSegment = BS.unpack . urlEncode False . BS.pack

splitOnSlash :: String -> [String]
splitOnSlash s =
  case break (== '/') s of
    (seg, []) -> [seg]
    (seg, _:rest) -> seg : splitOnSlash rest

signatureHex :: AwsCredentials -> BS.ByteString -> BS.ByteString -> BS.ByteString -> BS.ByteString
signatureHex creds date region stringToSign =
  let kSecret = "AWS4" <> awsSecretAccessKey creds
      kDate = hmacSha256 kSecret date
      kRegion = hmacSha256 kDate region
      kService = hmacSha256 kRegion "s3"
      kSigning = hmacSha256 kService "aws4_request"
      sig = hmacSha256 kSigning stringToSign
   in B16.encode sig

hmacSha256 :: BS.ByteString -> BS.ByteString -> BS.ByteString
hmacSha256 key msg =
  let mac :: HMAC SHA256
      mac = hmac key msg
   in convert (hmacGetDigest mac)

sha256Hex :: BL.ByteString -> BS.ByteString
sha256Hex payload =
  let digest :: Digest SHA256
      digest = hash (BL.toStrict payload)
   in B16.encode (convert digest)

formatAmzDate :: UTCTime -> String
formatAmzDate = formatTime defaultTimeLocale "%Y%m%dT%H%M%SZ"

formatShortDate :: UTCTime -> String
formatShortDate = formatTime defaultTimeLocale "%Y%m%d"

trim :: String -> String
trim = dropWhile isSpace . dropWhileEnd isSpace
  where
    dropWhileEnd f = reverse . dropWhile f . reverse

trimSlashes :: String -> String
trimSlashes =
  let dropSlash = dropWhile (== '/')
      dropEndSlash = reverse . dropSlash . reverse
   in dropEndSlash . dropSlash

nonEmpty :: String -> Maybe String
nonEmpty s =
  let t = trim s
   in if null t then Nothing else Just t

firstNonEmpty :: [Maybe String] -> Maybe String
firstNonEmpty =
  foldr
    (\v acc -> case v >>= nonEmpty of
        Just x -> Just x
        Nothing -> acc
    )
    Nothing
