{-# LANGUAGE OverloadedStrings #-}

module Trader.Api.Routes (
    apiEndpointDocs,
    apiRouteLabel,
) where

import Data.Aeson (Value, object, (.=))
import Data.List (intercalate)
import Data.Text (Text)
import qualified Data.Text as T

apiEndpointDocs :: [Value]
apiEndpointDocs =
    [ endpoint "GET" "/health"
    , endpoint "GET" "/version"
    , endpoint "GET" "/metrics"
    , endpoint "GET" "/ops"
    , endpoint "GET" "/ops/performance"
    , endpoint "GET" "/outbox"
    , endpoint "GET" "/cache"
    , endpoint "POST" "/cache/clear"
    , endpoint "POST" "/signal"
    , endpoint "POST" "/signal/async"
    , endpoint "GET" "/signal/async/:jobId"
    , endpoint "POST" "/signal/async/:jobId/cancel"
    , endpoint "POST" "/trade"
    , endpoint "POST" "/trade/async"
    , endpoint "GET" "/trade/async/:jobId"
    , endpoint "POST" "/trade/async/:jobId/cancel"
    , endpoint "POST" "/backtest"
    , endpoint "POST" "/backtest/async"
    , endpoint "GET" "/backtest/async/:jobId"
    , endpoint "POST" "/backtest/async/:jobId/cancel"
    , endpoint "POST" "/binance/keys"
    , endpoint "POST" "/binance/trades"
    , endpoint "POST" "/binance/positions"
    , endpoint "GET" "/binance/proxy/health"
    , endpoint "POST" "/coinbase/keys"
    , endpoint "POST" "/binance/listenKey"
    , endpoint "GET" "/binance/listenKey/stream"
    , endpoint "POST" "/binance/listenKey/keepAlive"
    , endpoint "POST" "/binance/listenKey/close"
    , endpoint "POST" "/bot/start"
    , endpoint "POST" "/bot/stop"
    , endpoint "GET" "/bot/status"
    , endpoint "POST" "/optimizer/run"
    , endpoint "GET" "/optimizer/combos"
    , endpoint "GET" "/state/sync"
    , endpoint "POST" "/state/sync"
    ]
  where
    endpoint method path = object ["method" .= (method :: String), "path" .= (path :: String)]

apiRouteLabel :: [Text] -> String
apiRouteLabel path =
    case path of
        ["signal", "async", _, "cancel"] -> "signal/async/:jobId/cancel"
        ["backtest", "async", _, "cancel"] -> "backtest/async/:jobId/cancel"
        ["trade", "async", _, "cancel"] -> "trade/async/:jobId/cancel"
        ["signal", "async", _] -> "signal/async/:jobId"
        ["backtest", "async", _] -> "backtest/async/:jobId"
        ["trade", "async", _] -> "trade/async/:jobId"
        ["optimizer", "run"] -> "optimizer/run"
        ["optimizer", "combos"] -> "optimizer/combos"
        ["state", "sync"] -> "state/sync"
        _ ->
            case path of
                [] -> "root"
                _ -> intercalate "/" (map T.unpack path)
