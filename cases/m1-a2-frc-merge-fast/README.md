# M1-A2 FRC Merge (Fast)

Fast gating variant for M1 evidence (reduced resolution, shorter tlim).

Run:
```
python tools/run_case.py --case m1-a2-frc-merge-fast --stage all --update-evidence
```

Merge indicator:
- Extract 1D density profile at y≈0 (midplane) and detect the two highest peaks.
- Peak separation is the absolute distance between those peaks.
- `merged_time` is the first time the separation falls below the configured threshold.
