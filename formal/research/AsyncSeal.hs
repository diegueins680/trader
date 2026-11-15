module Main (main) where
import Trader.App.AsyncJobAdmission (JobAdmissionFailure (..), reserveOpenSlot)
main :: IO ()
main = interact (unlines . map (show . evaluate . read) . lines)
  where
    evaluate :: (Int, Int, Bool) -> (Int, Bool, Int)
    evaluate (limit, count, closed) =
        let ((next, sealed), result) = reserveOpenSlot limit (count, closed)
            code = case result of
                Right () -> 0
                Left (JobQueueFull _) -> 1
                Left JobQueueClosed -> 2
         in (next, sealed, code)
