## Scorecard

| Row | Sharpe | Max drawdown | Expectancy/trade | Trade-count retention | Closed-trade count | Kelly-lite exposure | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| disabled | -0.070 | 6.62% | 0.01% | 100.0% | 7 | off | baseline |
| vol_conf_v1_default | -0.100 | 6.69% | 0.00% | 100.0% | 7 | off | fail (sharpe<0.030; closedTrades<50) |
| vol_conf_v1_high_vol_tighter | -0.100 | 6.69% | 0.00% | 100.0% | 7 | off | fail (sharpe<0.030; closedTrades<50) |
| vol_conf_v1_high_vol_looser | -0.100 | 6.69% | 0.00% | 100.0% | 7 | off | fail (sharpe<0.030; closedTrades<50) |
| vol_conf_v1_conf_stricter | -0.100 | 6.69% | 0.00% | 100.0% | 7 | off | fail (sharpe<0.030; closedTrades<50) |

### Source artifacts

- `disabled` → `.tmp/research-scorecard-spike/spike-disabled.json`
- `vol_conf_v1_default` → `.tmp/research-scorecard-spike/spike-vol_conf_v1_default.json`
- `vol_conf_v1_high_vol_tighter` → `.tmp/research-scorecard-spike/spike-vol_conf_v1_high_vol_tighter.json`
- `vol_conf_v1_high_vol_looser` → `.tmp/research-scorecard-spike/spike-vol_conf_v1_high_vol_looser.json`
- `vol_conf_v1_conf_stricter` → `.tmp/research-scorecard-spike/spike-vol_conf_v1_conf_stricter.json`

### Firm contract applied

- Sharpe improvement vs baseline: `>= +0.10`
- Max drawdown regression vs baseline: `<= +2.00%`
- Trade-count retention: `>= 60%` of baseline unless expectancy/trade improves by `>= 10%`
- Closed-trade minimum: `>= 50`
- Kelly-lite exposure ratio when enabled: `<= 0.95` unless disabled with `>= 1`
- Kelly-lite exposure reduction when enabled: `>= 0.000`
