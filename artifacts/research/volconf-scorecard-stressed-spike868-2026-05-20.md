## Scorecard

| Row | Sharpe | Max drawdown | Expectancy/trade | Trade-count retention | Closed-trade count | Kelly-lite exposure | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| disabled | -1.196 | 6.13% | -0.60% | 100.0% | 5 | off | baseline |
| vol_conf_v1_default | -1.196 | 6.13% | -0.60% | 100.0% | 5 | off | fail (sharpe<-1.096; closedTrades<50) |
| vol_conf_v1_high_vol_tighter | -1.196 | 6.13% | -0.60% | 100.0% | 5 | off | fail (sharpe<-1.096; closedTrades<50) |
| vol_conf_v1_high_vol_looser | -1.196 | 6.13% | -0.60% | 100.0% | 5 | off | fail (sharpe<-1.096; closedTrades<50) |
| vol_conf_v1_conf_stricter | -1.196 | 6.13% | -0.60% | 100.0% | 5 | off | fail (sharpe<-1.096; closedTrades<50) |

### Source artifacts

- `disabled` → `.tmp/research-scorecard-spike-868/spike868-disabled.json`
- `vol_conf_v1_default` → `.tmp/research-scorecard-spike-868/spike868-vol_conf_v1_default.json`
- `vol_conf_v1_high_vol_tighter` → `.tmp/research-scorecard-spike-868/spike868-vol_conf_v1_high_vol_tighter.json`
- `vol_conf_v1_high_vol_looser` → `.tmp/research-scorecard-spike-868/spike868-vol_conf_v1_high_vol_looser.json`
- `vol_conf_v1_conf_stricter` → `.tmp/research-scorecard-spike-868/spike868-vol_conf_v1_conf_stricter.json`

### Firm contract applied

- Sharpe improvement vs baseline: `>= +0.10`
- Max drawdown regression vs baseline: `<= +2.00%`
- Trade-count retention: `>= 60%` of baseline unless expectancy/trade improves by `>= 10%`
- Closed-trade minimum: `>= 50`
- Kelly-lite exposure ratio when enabled: `<= 0.95` unless disabled with `>= 1`
- Kelly-lite exposure reduction when enabled: `>= 0.000`
