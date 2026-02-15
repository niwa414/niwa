# m10-h5-rotseed-hybrid-merge-tilt-gamma-drift036

Rotation-seeded H5 gate: use the H4.1 rotate_axisym handoff as the seed for the
3D Hybrid merge -> tilt -> gamma chain.

Drift tweak: `opmd_double_seed_drift=[0.36, 0.0, 0.0]` (beta units).

## Run

```bash
python tools/run_case.py --case m10-h5-rotseed-hybrid-merge-tilt-gamma-drift036 --stage all --update-evidence
```

## Outputs

- Raw: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-drift036/raw/run/`
- Metrics: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-drift036/analysis/metrics.json`
- Plots: `outputs/m10-h5-rotseed-hybrid-merge-tilt-gamma-drift036/plots/`
