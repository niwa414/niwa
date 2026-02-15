# m20-b2-3d-driver-stage2-gate80-dense

Purpose: land an executable B2 3D Stage2 gate:
- GS->3D init generation (`gs_frc_athena_init.py`)
- 3D milestone `stage=3d2` using Gate80 dense input
- machine-readable strict and transition outcomes:
  - strict: `stage_strict_pass` / `stage_strict_suite_pass`
  - transition: `stage_transition_pass` (used as case threshold)

Run:
```bash
python tools/run_case.py --case m20-b2-3d-driver-stage2-gate80-dense --stage all --update-evidence
```

Key outputs:
- Raw summary: `outputs/m20-b2-3d-driver-stage2-gate80-dense/raw/run/stage_summary.json`
- Metrics: `outputs/m20-b2-3d-driver-stage2-gate80-dense/analysis/metrics.json`
- Details: `outputs/m20-b2-3d-driver-stage2-gate80-dense/analysis/stage2_details.json`
