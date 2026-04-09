Keep the existing imports, helpers, and tests unchanged.

In the `main` test list, immediately after:
    run "formal ROI penalties stay ordered" testFormalRoiPenaltyOrdering
add:
    run "formal ROI malformed inputs fail closed" testFormalRoiMalformedInputsFailClosed

Add near the other formal optimization tests:

testFormalRoiMalformedInputsFailClosed :: IO ()
testFormalRoiMalformedInputsFailClosed =
    assert "malformed ROI inputs fail closed" (fvrMalformedRoiInputsFailClosed formalVerificationReport)