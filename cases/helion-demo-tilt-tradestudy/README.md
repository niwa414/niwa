# Helion Demo: Single-Knob Tilt Trade Study

This demo replays three validated source runs into a unified KPI gate, then generates
an engineering recommendation report.

## Cases

- `baseline`: `shift=0.000` (source `m17-b2-p33-off352-drive100-repeat128-drift0330-shiftp0000`)
- `knob_minus`: `shift=-0.002` (source `m17-b2-p33-off352-drive100-repeat128-drift0330-shiftn0020`)
- `knob_plus`: `shift=+0.002` (source `m17-b2-p33-off352-drive100-repeat128-drift0330-shiftp0020`)

## Run

```bash
python tools/run_case.py --case cases/helion-demo-tilt-tradestudy/baseline/case.json
python tools/run_case.py --case cases/helion-demo-tilt-tradestudy/knob_minus/case.json
python tools/run_case.py --case cases/helion-demo-tilt-tradestudy/knob_plus/case.json
python tools/report_helion_demo.py \
  outputs/helion-demo-tilt-tradestudy-baseline \
  outputs/helion-demo-tilt-tradestudy-knob-minus \
  outputs/helion-demo-tilt-tradestudy-knob-plus
```

## Required KPI fields per case

- `ran_to_completion`
- `no_nan_in_metrics`
- `merge_time_exists`
- `compression_ratio`
- `tilt_amp_max`
- `tilt_growth_rate`
- `energy_accounting_ok`
