# m18-b2-3d-driver-stage0

Purpose: land an executable B2 3D driver gate at Stage0 level:
- 3d0a: GS->3D mapping smoke
- 3d0b: EMF injection smoke
- 3d0c: moving-wall smoke

This case uses `tools/run_3d_milestone.py` and aggregates the three stage results
into one `stage0_suite_pass` metric.

## Run

```bash
python tools/run_case.py --case m18-b2-3d-driver-stage0 --stage all --update-evidence
```

## Outputs

- Raw summary: `outputs/m18-b2-3d-driver-stage0/raw/run/stage0_summary.json`
- Metrics: `outputs/m18-b2-3d-driver-stage0/analysis/metrics.json`
- Details: `outputs/m18-b2-3d-driver-stage0/analysis/stage0_details.json`

## Notes

- This is a stage0 gate only. Stage1/Stage2 threshold fitting is tracked as known gaps.
- The run generates GS init HDF5 at runtime under the case output directory.
