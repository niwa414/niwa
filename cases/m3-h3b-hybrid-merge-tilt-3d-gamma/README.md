# M3-H3b Hybrid Merge→Tilt Gamma Fit (3D)

Post-merge tilt growth-rate fit using H3a post-merge centroid series.

## Goal
- Fit ln(A) = gamma * t + b on post-merge window.
- Report best gamma with R^2 and fit window.

## Key metrics
- `fit_found`, `fit_points`, `gamma_best`, `r2_best`, `amp_ratio_fit_best`

## Notes
- Uses `outputs/m3-h3a-hybrid-merge-tilt-3d/analysis/tilt_post_merge_series.csv` as input.
