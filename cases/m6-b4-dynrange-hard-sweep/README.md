# B4 Dynamic Range Hard Sweep

This sweep stresses dynamic range handling (low background, higher drift, applied B field)
across control/floor/substeps/both variants.

## Run

`python tools/run_case.py --case m6-b4-dynrange-hard-sweep --stage all --update-evidence`

## Outputs

- `analysis/metrics.json`: sweep PASS and outcome.
- `analysis/variant_summary.json`: per-variant metrics + flags.
- Plots: min-density proxy, drop-breach flags, and energy/B_rms comparison.
