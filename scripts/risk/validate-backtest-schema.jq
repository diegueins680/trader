# Trader Firm — Backtest JSON Schema Validation Invariant
# Author: trader-firm-risk
# Date: 2026-05-20
# Reference: artifacts/risk/schema-validation-jq-invariant-2026-05-20.md
#
# Usage: jq -f scripts/risk/validate-backtest-schema.jq backtest-output.json
# Failures only: jq -f scripts/risk/validate-backtest-schema.jq backtest-output.json | jq 'map(select(.status == "FAIL"))'

def validate_field(path; type_name; nullable):
  . as $doc |
  (getpath(path) // null) as $val |
  if $val == null then
    if nullable then
      { field: path | join("."), status: "PASS", reason: "null allowed" }
    else
      { field: path | join("."), status: "FAIL", reason: "null not allowed" }
    end
  elif type_name == "string" and ($val | type) == "string" then
    { field: path | join("."), status: "PASS", type: "string" }
  elif type_name == "number" and ($val | type) == "number" then
    if ($val | isnan) or ($val | isinfinite) then
      { field: path | join("."), status: "FAIL", reason: "NaN or Inf" }
    else
      { field: path | join("."), status: "PASS", type: "number" }
    end
  elif type_name == "integer" and ($val | type) == "number" and ($val == ($val | floor)) then
    { field: path | join("."), status: "PASS", type: "integer" }
  elif type_name == "boolean" and ($val | type) == "boolean" then
    { field: path | join("."), status: "PASS", type: "boolean" }
  else
    { field: path | join("."), status: "FAIL", expected: type_name, got: ($val | type) }
  end;

[
  validate_field(["mode"]; "string"; false),
  validate_field(["backtest","method"]; "string"; false),
  validate_field(["backtest","sharpe"]; "number"; false),
  validate_field(["backtest","max_drawdown"]; "number"; false),
  validate_field(["backtest","closed_trades"]; "integer"; false),
  validate_field(["backtest","avg_trade"]; "number"; false),
  validate_field(["backtest","metrics","totalReturn"]; "number"; false),
  validate_field(["backtest","metrics","annualizedReturn"]; "number"; false),
  validate_field(["backtest","metrics","annualizedVolatility"]; "number"; false),
  validate_field(["backtest","metrics","sortino"]; "number"; false),
  validate_field(["backtest","metrics","calmar"]; "number"; false),
  validate_field(["backtest","metrics","winRate"]; "number"; false),
  validate_field(["backtest","metrics","profitFactor"]; "number"; true),
  validate_field(["backtest","metrics","turnover"]; "number"; false),
  validate_field(["backtest","metrics","exposure"]; "number"; false),
  validate_field(["backtest","metrics","tradeCount"]; "integer"; false),
  validate_field(["backtest","metrics","positionChanges"]; "integer"; false),
  validate_field(["backtest","metrics","agreementRate"]; "number"; false),
  validate_field(["backtest","blendWeight"]; "number"; false),
  validate_field(["backtest","openThreshold"]; "number"; false),
  validate_field(["backtest","closeThreshold"]; "number"; false),
  validate_field(["backtest","vol_conf_gate"]; "string"; false),
  validate_field(["backtest","split","train"]; "integer"; false),
  validate_field(["backtest","split","backtest"]; "integer"; false)
]
