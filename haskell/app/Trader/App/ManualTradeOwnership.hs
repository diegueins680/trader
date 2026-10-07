{- | Mutual exclusion between live actors on one position identity.

A position identity is the credential account an order is signed with (the
tenant key derived from the API keys actually used, request keys or else the
server's) together with the normalized symbol. A live-capable bot that is
starting or running is the only live owner of its identity. A manual @/trade@
that may send a live order (CE-LIVE-002) would otherwise act on the same
position concurrently with the bot, so it holds a claim on its identity for the
whole trade, admitted only while no bot owns it, and bot start publication
refuses an identity with an outstanding claim or another bot. Both checks run
while holding the bot runtime lock, so neither side can slip in between the
other's check and publication.

Registry tenant keys are not used: without request keys a tenant key comes from
a request header, so two tenants can share one account (CE-LIVE-003).
Close-position is deliberately not claimed: it is a reduce-only operator
override.
-}
module Trader.App.ManualTradeOwnership (
    OwnershipKey,
    ManualTradeClaims,
    ManualTradeOwnershipConflict (..),
    newManualTradeClaims,
    manualTradeConflict,
    manualTradeClaimed,
    withManualTradeClaim,
) where

import Control.Concurrent.MVar (MVar, modifyMVarMasked, modifyMVar_)
import Control.Exception (Exception, bracket, throwIO, uninterruptibleMask_)
import Data.IORef (IORef, modifyIORef', newIORef, readIORef)
import qualified Data.Map.Strict as Map
import Data.Text (Text)

-- | Credential account and normalized symbol.
type OwnershipKey = (Text, String)

-- | Outstanding manual-trade claims per position identity. Read and written only while holding the bot runtime lock.
newtype ManualTradeClaims = ManualTradeClaims (IORef (Map.Map OwnershipKey Int))

newtype ManualTradeOwnershipConflict = ManualTradeOwnershipConflict String
    deriving (Show)

instance Exception ManualTradeOwnershipConflict

newManualTradeClaims :: IO ManualTradeClaims
newManualTradeClaims = ManualTradeClaims <$> newIORef Map.empty

-- | Admission decision for a manual live trade, given every identity a live bot owns.
manualTradeConflict :: [OwnershipKey] -> OwnershipKey -> Maybe String
manualTradeConflict owners key@(_, sym)
    | key `elem` owners =
        Just ("Manual trade refused: a bot owns " ++ sym ++ " on this account. Stop the bot first (close-position remains available).")
    | otherwise = Nothing

-- | Whether a manual trade currently holds a claim on the identity. The caller must hold the bot runtime lock.
manualTradeClaimed :: ManualTradeClaims -> OwnershipKey -> IO Bool
manualTradeClaimed (ManualTradeClaims ref) key = maybe False (> 0) . Map.lookup key <$> readIORef ref

{- | Run a manual trade while holding a claim on its identity. Admission and
release both happen under the runtime lock; release cannot be interrupted, so
a cancelled or timed-out trade never leaks its claim.
-}
withManualTradeClaim :: MVar runtime -> (runtime -> [OwnershipKey]) -> ManualTradeClaims -> OwnershipKey -> IO a -> IO a
withManualTradeClaim lock owned (ManualTradeClaims ref) key action =
    bracket acquire release (either (throwIO . ManualTradeOwnershipConflict) (const action))
  where
    -- Masked even outside 'bracket': once the claim is recorded nothing can interrupt before acquisition returns it.
    acquire =
        modifyMVarMasked lock $ \runtime ->
            case manualTradeConflict (owned runtime) key of
                Just msg -> pure (runtime, Left msg)
                Nothing -> do
                    modifyIORef' ref (Map.insertWith (+) key 1)
                    pure (runtime, Right ())
    release =
        either
            (const (pure ()))
            ( const $
                uninterruptibleMask_ $
                    modifyMVar_ lock $ \runtime -> do
                        modifyIORef' ref (Map.update (\n -> if n <= 1 then Nothing else Just (n - 1)) key)
                        pure runtime
            )
