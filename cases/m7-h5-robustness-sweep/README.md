# H5 robustness sweep (+/-10% seed_amp)

Runs two H5 variants with only `tilt_seed_y_offset_amp` adjusted to verify the
merge -> tilt -> gamma gate remains stable under small perturbations.

## Run

`python tools/run_case.py --case m7-h5-robustness-sweep --stage all --update-evidence`

## Outputs

- `analysis/metrics.json`: sweep PASS summary and gamma guard.
- `analysis/variant_summary.json`: per-variant metrics and PASS status.
- `plots/gamma_r2_compare.png`: gamma and r2 comparison.
- `plots/tilt_amp_post_merge_compare.png`: post-merge tilt amplitude overlay.
