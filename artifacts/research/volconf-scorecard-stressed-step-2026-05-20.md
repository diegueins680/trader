## Scorecard

| Row | Sharpe | Max drawdown | Expectancy/trade | Trade-count retention | Closed-trade count | Kelly-lite exposure | Contract |
|---|---:|---:|---:|---:|---:|---:|---|
| disabled | -6.093 | 1.30% | -1.22% | 100.0% | 0 | off | baseline |
| vol_conf_v1_default | -6.093 | 0.98% | -0.92% | 100.0% | 0 | off | fail (sharpe<-5.993; closedTrades<50) |
| vol_conf_v1_high_vol_tighter | -6.093 | 0.98% | -0.92% | 100.0% | 0 | off | fail (sharpe<-5.993; closedTrades<50) |
| vol_conf_v1_high_vol_looser | -6.093 | 0.98% | -0.92% | 100.0% | 0 | off | fail (sharpe<-5.993; closedTrades<50) |
| vol_conf_v1_conf_stricter | -6.093 | 0.98% | -0.92% | 100.0% | 0 | off | fail (sharpe<-5.993; closedTrades<50) |

### Source artifacts

- `disabled` → `.tmp/research-scorecard-step/step-disabled.json`
- `vol_conf_v1_default` → `.tmp/research-scorecard-step/step-vol_conf_v1_default.json`
- `vol_conf_v1_high_vol_tighter` → `.tmp/research-scorecard-step/step-vol_conf_v1_high_vol_tighter.json`
- `vol_conf_v1_high_vol_looser` → `.tmp/research-scorecard-step/step-vol_conf_v1_high_vol_looser.json`
- `vol_conf_v1_conf_stricter` → `.tmp/research-scorecard-step/step-vol_conf_v1_conf_stricter.json`

### Firm contract applied

- Sharpe improvement vs baseline: `>= +0.10`
- Max drawdown regression vs baseline: `<= +2.00%`
- Trade-count retention: `>= 60%` of baseline unless expectancy/trade improves by `>= 10%`
- Closed-trade minimum: `>= 50`
- Kelly-lite exposure ratio when enabled: `<= 0.95` unless disabled with `>= 1`
- Kelly-lite exposure reduction when enabled: `>= 0.000`
