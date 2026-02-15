# m19-b2-3d-driver-stage1-gate60

Purpose: land an executable B2 3D Stage1 gate:
- GS->3D init generation (`gs_frc_athena_init.py`)
- 3D milestone `stage=3d1` using Gate60 input
- machine-readable `stage_pass` / `stage_suite_pass`

Run:
```bash
python tools/run_case.py --case m19-b2-3d-driver-stage1-gate60 --stage all --update-evidence
```

Key outputs:
- Raw summary: `outputs/m19-b2-3d-driver-stage1-gate60/raw/run/stage_summary.json`
- Metrics: `outputs/m19-b2-3d-driver-stage1-gate60/analysis/metrics.json`
- Details: `outputs/m19-b2-3d-driver-stage1-gate60/analysis/stage1_details.json`
