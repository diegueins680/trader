module LegacyDrain where
import Data.IORef
newtype DrainController = DrainController (IORef Bool)
newDrainController :: IO DrainController
newDrainController = DrainController <$> newIORef False

beginDrain :: DrainController -> IO Bool
beginDrain (DrainController ref) =
    atomicModifyIORef' ref $ \draining ->
        if draining
            then (True, False)
            else (True, True)

isDraining :: DrainController -> IO Bool
isDraining (DrainController ref) = readIORef ref
