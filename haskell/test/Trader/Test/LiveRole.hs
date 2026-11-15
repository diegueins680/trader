module Trader.Test.LiveRole (liveRoleSuite) where

import Control.Exception (SomeException, bracket, fromException, try)
import qualified Data.ByteString.Char8 as BS
import Data.Maybe (isJust)
import System.Environment (lookupEnv, setEnv, unsetEnv)
import Trader.App.LiveRole
import Trader.Binance (BinanceMarket (..), BinanceOrderMode (..), OrderSide (..), cancelFuturesOrderByClientId, newBinanceEnv, placeMarketOrder)

liveRoleSuite :: [(String, IO ())]
liveRoleSuite =
    [ ("non-trading roles are refused, trading roles are not", roleDecisions)
    , ("role resolution follows the server identity fallback", roleResolution)
    , ("research server refuses a live order before any network call", gatedBeforeNetwork)
    , ("test-mode orders are not gated", testModeNotGated)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

roleDecisions :: IO ()
roleDecisions = do
    expect "refused" [True, True, True, True, True] (map (refused . liveOrderRoleRefusal) ["research", "read-only", "readonly", "fly", " Research "])
    expect "allowed" [False, False, False, False] (map (refused . liveOrderRoleRefusal) ["trading", "standalone", "local", "anything-else"])
  where
    refused = isJust

-- Run an action with environment variables set or unset, restoring them afterwards.
withEnv :: [(String, Maybe String)] -> IO a -> IO a
withEnv vars action = bracket save restore (const (mapM_ apply vars >> action))
  where
    save = mapM (\(k, _) -> (,) k <$> lookupEnv k) vars
    restore = mapM_ apply
    apply (k, v) = maybe (unsetEnv k) (setEnv k) v

roleResolution :: IO ()
roleResolution = do
    explicit <- withEnv [("TRADER_SERVER_ROLE", Just "Trading"), ("FLY_APP_NAME", Just "x")] resolveServerRole
    onFly <- withEnv [("TRADER_SERVER_ROLE", Nothing), ("FLY_MACHINE_ID", Nothing), ("FLY_APP_NAME", Just "trader-hs")] resolveServerRole
    offFly <- withEnv [("TRADER_SERVER_ROLE", Nothing), ("FLY_MACHINE_ID", Nothing), ("FLY_APP_NAME", Nothing)] resolveServerRole
    expect "roles" ("trading", "fly", "local") (explicit, onFly, offFly)

-- The base URL is unroutable: reaching the network would yield a connection error, not the role refusal.
gatedBeforeNetwork :: IO ()
gatedBeforeNetwork = do
    env <- newBinanceEnv MarketFutures "http://127.0.0.1:9" (Just (BS.pack "key")) (Just (BS.pack "secret"))
    withEnv [("TRADER_SERVER_ROLE", Just "research")] $ do
        order <- try (placeMarketOrder env OrderLive "BTCUSDT" Buy (Just 0.001) Nothing Nothing Nothing)
        cancel <- try (cancelFuturesOrderByClientId env "BTCUSDT" "cid")
        expect "live order refused by role" True (isRoleRefusal order)
        expect "cancel refused by role" True (isRoleRefusal cancel)

testModeNotGated :: IO ()
testModeNotGated = do
    env <- newBinanceEnv MarketSpot "http://127.0.0.1:9" (Just (BS.pack "key")) (Just (BS.pack "secret"))
    withEnv [("TRADER_SERVER_ROLE", Just "research")] $ do
        r <- try (placeMarketOrder env OrderTest "BTCUSDT" Buy (Just 0.001) Nothing Nothing Nothing)
        expect "test order reaches the venue path" False (isRoleRefusal r)

isRoleRefusal :: Either SomeException a -> Bool
isRoleRefusal r =
    case r of
        Left ex -> isJust (fromException ex :: Maybe LiveOrderRoleRefused)
        Right _ -> False
