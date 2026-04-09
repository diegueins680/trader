Add one new test and register it next to the other formal ROI checks.

1. In the `main` test list, immediately after `run "formal ROI penalties stay ordered" testFormalRoiPenaltyOrdering`, add:
- `run "formal ROI malformed inputs fail closed" testFormalRoiMalformedInputsFailClosed`

2. Add the test definition near the other formal optimization tests:
- `testFormalRoiMalformedInputsFailClosed :: IO ()`
- body: `assert "malformed ROI inputs fail closed" (fvrMalformedRoiInputsFailClosed formalVerificationReport)`

3. No other test wiring changes are required; `formalVerificationReport = verifyFormalOptimization` already supplies the updated report.