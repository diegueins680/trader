{- | Exception capture that never swallows asynchronous exceptions.

'Control.Exception.try' at type 'SomeException' also catches asynchronous
exceptions such as 'ThreadKilled' (from 'killThread') and the exception used by
'System.Timeout.timeout'. In a bot worker that turned a stop into an ordinary
"order error" value, and the order routine kept placing or cancelling orders
after 'botStop' had reported success. 'trySync' captures synchronous exceptions
exactly like @try \@SomeException@ and rethrows every asynchronous one, so a
stopped worker unwinds through its @finally@ cleanup instead.
-}
module Trader.App.AsyncSafe (
    trySync,
    isAsyncException,
) where

import Control.Exception (SomeAsyncException, SomeException, fromException, throwIO, try)
import Data.Maybe (isJust)

-- | True for exceptions delivered asynchronously (killThread, timeout, user interrupt).
isAsyncException :: SomeException -> Bool
isAsyncException ex = isJust (fromException ex :: Maybe SomeAsyncException)

-- | Like @try \@SomeException@, but asynchronous exceptions are rethrown, never returned.
trySync :: IO a -> IO (Either SomeException a)
trySync action = do
    result <- try action
    case result of
        Left ex | isAsyncException ex -> throwIO ex
        _ -> pure result
