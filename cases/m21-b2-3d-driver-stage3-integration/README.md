# m21-b2-3d-driver-stage3-integration

Purpose: stage3 integration gate for B2 delivery:
- `stage1` gate pass (`m19`)
- `stage2` transition pass (`m20`)
- tilt 2x2 matrix pass + all `gamma_m1v_fit_best24 < 0`

Run:
```bash
python tools/run_case.py --case m21-b2-3d-driver-stage3-integration --stage all --update-evidence
```

Key outputs:
- Raw summary: `outputs/m21-b2-3d-driver-stage3-integration/raw/run/stage3_summary.json`
- Metrics: `outputs/m21-b2-3d-driver-stage3-integration/analysis/metrics.json`
- Details: `outputs/m21-b2-3d-driver-stage3-integration/analysis/stage3_details.json`

Notes:
- `stage2_strict_pass` is recorded as a metric but is non-blocking for stage3 integration.
