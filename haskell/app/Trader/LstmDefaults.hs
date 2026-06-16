module Trader.LstmDefaults (
    defaultLstmAdamBeta1,
    defaultLstmAdamBeta2,
    defaultLstmAdamEps,
) where

defaultLstmAdamBeta1 :: Double
defaultLstmAdamBeta1 = 0.9

defaultLstmAdamBeta2 :: Double
defaultLstmAdamBeta2 = 0.999

defaultLstmAdamEps :: Double
defaultLstmAdamEps = 1e-8
