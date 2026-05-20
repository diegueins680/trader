## Scorecard

| Row | Sharpe | Max drawdown | Expectancy/trade | Trade-count retention | Closed-trade count | Kelly-lite exposure | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| disabled | 14.135 | 0.90% | 1.31% | 100.0% | 2 | off | baseline |
| vol_conf_v1_default | 14.135 | 0.90% | 1.31% | 100.0% | 2 | off | fail (sharpe<14.235; closedTrades<50) |
| vol_conf_v1_high_vol_tighter | 14.135 | 0.90% | 1.31% | 100.0% | 2 | off | fail (sharpe<14.235; closedTrades<50) |
| vol_conf_v1_high_vol_looser | 14.135 | 0.90% | 1.31% | 100.0% | 2 | off | fail (sharpe<14.235; closedTrades<50) |
| vol_conf_v1_conf_stricter | 14.135 | 0.90% | 1.31% | 100.0% | 2 | off | fail (sharpe<14.235; closedTrades<50) |

### Source artifacts

- `disabled` → `.tmp/research-scorecard-synthetic/synthetic-disabled.json`
- `vol_conf_v1_default` → `.tmp/research-scorecard-synthetic/synthetic-vol_conf_v1_default.json`
- `vol_conf_v1_high_vol_tighter` → `.tmp/research-scorecard-synthetic/synthetic-vol_conf_v1_high_vol_tighter.json`
- `vol_conf_v1_high_vol_looser` → `.tmp/research-scorecard-synthetic/synthetic-vol_conf_v1_high_vol_looser.json`
- `vol_conf_v1_conf_stricter` → `.tmp/research-scorecard-synthetic/synthetic-vol_conf_v1_conf_stricter.json`

### Firm contract applied

- Sharpe improvement vs baseline: `>= +0.10`
- Max drawdown regression vs baseline: `<= +2.00%`
- Trade-count retention: `>= 60%` of baseline unless expectancy/trade improves by `>= 10%`
- Closed-trade minimum: `>= 50`
- Kelly-lite exposure ratio when enabled: `<= 0.95` unless disabled with `>= 1`
- Kelly-lite exposure reduction when enabled: `>= 0.000`
