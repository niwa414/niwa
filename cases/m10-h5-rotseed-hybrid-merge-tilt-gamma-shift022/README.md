# m10-h5-rotseed-hybrid-merge-tilt-gamma-shift022

Rotation-seeded H5 gate: use the H4.1 rotate_axisym handoff as the seed for the
3D Hybrid merge -> tilt -> gamma chain.

Shift tweak: `opmd_double_seed_shift=[0.22, 0.0, 0.0]`.

## Run

```bash
python tools/run_case.py --case m10-h5-rotseed-hybrid-merge-tilt-gamma-shift022 --stage all --update-evidence
```

## Outputs

- Raw: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-shift022/raw/run/`
- Metrics: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-shift022/analysis/metrics.json`
- Plots: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-shift022/plots/`
