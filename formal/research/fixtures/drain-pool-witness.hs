{-# LANGUAGE CPP #-}
module Main (main) where
import Control.Monad (void)
import Trader.App.AsyncJobAdmission
import Trader.App.BacktestGate
#ifdef LEGACY
import LegacyDrain
#else
import Trader.App.GracefulShutdown (newDrainController, beginDrain, isDraining)
#endif
main :: IO ()
main = do
    drain <- newDrainController
#ifdef LEGACY
    slots <- newJobSlots 1
    gate <- newBacktestGate 1 2
#else
    slots <- newJobSlotsWithDrain drain 1
    gate <- newBacktestGateWithDrain drain 1 2
#endif
    snapshot <- isDraining drain
    void (beginDrain drain)
    async <- startBoundedJob slots (pure ()) pure (\_ _ -> pure ())
    immediate <- runBacktestWithGate gate (pure ())
    waiting <- runBacktestWithGateWait gate (pure ())
    print ("ingress snapshot", snapshot)
    print ("async post-drain", either (const False) (const True) async)
    print ("backtest post-drain", either (const False) (const True) immediate)
    print ("waiting post-drain", either (const False) (const True) waiting)
    closeJobSlots slots
    waitJobSlots slots
