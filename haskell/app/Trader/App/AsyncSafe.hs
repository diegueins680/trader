{- | Exception capture that never swallows asynchronous exceptions.

'Control.Exception.try' and 'Control.Exception.catch' at type 'SomeException'
also catch asynchronous exceptions such as 'ThreadKilled' (from 'killThread')
and the exception used by 'System.Timeout.timeout'. In a bot worker that turned
a stop into an ordinary "order error" value, and the order routine kept placing
or cancelling orders after 'botStop' had reported success. 'trySync' and
'catchSync' capture synchronous exceptions exactly like 'try' and 'catch' and
rethrow every asynchronous one, so a stopped worker unwinds through its
@finally@ cleanup instead.

This module is the only place in @haskell/app@ allowed to import 'try' or
'catch' from "Control.Exception" (checked by @scripts/formal/async_stop.py@).
-}
module Trader.App.AsyncSafe (
    trySync,
    catchSync,
    tryForwardingAll,
    catchForwardingAll,
    isAsyncException,
) where

import Control.Exception (Exception, SomeAsyncException, SomeException, catch, fromException, throwIO, toException, try)
import Data.Maybe (isJust)

-- | True for exceptions delivered asynchronously (killThread, timeout, user interrupt).
isAsyncException :: SomeException -> Bool
isAsyncException ex = isJust (fromException ex :: Maybe SomeAsyncException)

-- | Like 'try', but asynchronous exceptions are rethrown, never returned.
trySync :: (Exception e) => IO a -> IO (Either e a)
trySync action = do
    result <- try action
    case result of
        Left ex | isAsyncException (toException ex) -> throwIO ex
        _ -> pure result

-- | Like 'catch', but the handler never runs for an asynchronous exception.
catchSync :: (Exception e) => IO a -> (e -> IO a) -> IO a
catchSync action handler =
    action `catch` \ex ->
        if isAsyncException (toException ex) then throwIO ex else handler ex

{- | 'try' that also captures asynchronous exceptions. Only for a thread whose
last act is to hand its outcome to a waiter: nothing may run after the capture
except delivering the result.
-}
tryForwardingAll :: IO a -> IO (Either SomeException a)
tryForwardingAll = try

-- | 'catch' counterpart of 'tryForwardingAll', under the same restriction.
catchForwardingAll :: IO a -> (SomeException -> IO a) -> IO a
catchForwardingAll = catch
