# m26-d2-magnetic-load-interface

Purpose: export engineering-facing magnetic load proxies from an existing circuit-coupled run.

Inputs:
- `outputs/m17-b2-circuit-mvp-mainline-Rload2000-coil4/raw/run/diag/reducedfiles/COIL.txt`
- `outputs/m17-b2-circuit-mvp-mainline-Rload2000-coil4/analysis/metrics.json`

Outputs:
- `analysis/metrics.json`
- `analysis/magnetic_load_series.csv`
- `analysis/magnetic_load_summary.md`

Run:
- `python tools/run_case.py --case m26-d2-magnetic-load-interface --stage all`
