{- | Server-role gate for live exchange actions.

Only a trading process may place, cancel or swap for real. Research,
read-only and Fly servers can hold exchange credentials for market data and
optimizer work, so without this gate any of them could act on the trading
process's positions (obligation 15, cross-process ownership). The gate sits
in the venue adapters, in front of every live order, cancel and transaction
send, so no caller can bypass it.

It is a deny-list on purpose: roles that are not explicitly non-trading
(@trading@, @standalone@, @local@ and unset off Fly) keep working, so a
trading box whose environment predates the role label is never stopped. The
role is resolved exactly as 'Trader.App.Observability.serverIdentityFromEnv'
does: @TRADER_SERVER_ROLE@, else @fly@ on a Fly machine, else @local@.
-}
module Trader.App.LiveRole (
    LiveOrderRoleRefused (..),
    nonTradingRoles,
    liveOrderRoleRefusal,
    resolveServerRole,
    requireLiveOrderRole,
) where

import Control.Exception (Exception, throwIO)
import Data.Char (isSpace, toLower)
import Data.Maybe (isJust)
import System.Environment (lookupEnv)

newtype LiveOrderRoleRefused = LiveOrderRoleRefused String
    deriving (Show)

instance Exception LiveOrderRoleRefused

-- | Server roles that must never send a live order, cancel or transaction.
nonTradingRoles :: [String]
nonTradingRoles = ["research", "read-only", "readonly", "fly"]

normalizeRole :: String -> String
normalizeRole = map toLower . dropWhile isSpace . reverse . dropWhile isSpace . reverse

-- | The refusal for a resolved server role, if it may not act live.
liveOrderRoleRefusal :: String -> Maybe String
liveOrderRoleRefusal role
    | normalizeRole role `elem` nonTradingRoles =
        Just ("Live exchange actions are disabled on a " ++ normalizeRole role ++ " server (TRADER_SERVER_ROLE); only the trading server may place or cancel orders.")
    | otherwise = Nothing

-- | The server role, resolved like 'Trader.App.Observability.serverIdentityFromEnv'.
resolveServerRole :: IO String
resolveServerRole = do
    explicit <- nonEmpty <$> lookupEnv "TRADER_SERVER_ROLE"
    flyMachine <- nonEmpty <$> lookupEnv "FLY_MACHINE_ID"
    flyApp <- nonEmpty <$> lookupEnv "FLY_APP_NAME"
    pure $ case explicit of
        Just role -> role
        Nothing
            | isJust flyMachine || isJust flyApp -> "fly"
            | otherwise -> "local"
  where
    nonEmpty raw =
        case normalizeRole <$> raw of
            Just v | not (null v) -> Just v
            _ -> Nothing

-- | Throws 'LiveOrderRoleRefused' unless this server may act live. Call before any live network action.
requireLiveOrderRole :: IO ()
requireLiveOrderRole = do
    role <- resolveServerRole
    maybe (pure ()) (throwIO . LiveOrderRoleRefused) (liveOrderRoleRefusal role)
