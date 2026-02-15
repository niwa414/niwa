# M1-A2 FRC Merge (Full)

Athena++ Hall-MHD FRC merging case using `frc_merge` problem generator.

Run:
```
python tools/run_case.py --case m1-a2-frc-merge-full --stage all --update-evidence
```

Merge indicator:
- Extract 1D density profile at y≈0 (midplane) and detect the two highest peaks.
- Peak separation is the absolute distance between those peaks.
- `merged_time` is the first time the separation falls below the configured threshold.
