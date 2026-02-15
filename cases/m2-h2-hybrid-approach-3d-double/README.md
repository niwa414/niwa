# M2-H2 Hybrid Approach (3D Double)

Hybrid 3D double-blob approach gate (pre-merge).

## Goal
- Maintain two structures in Hybrid 3D and demonstrate decreasing separation.
- Track separation and centroid paths robustly.

## Key metrics
- `sep_ratio` (must decrease below threshold)
- `ran_to_completion`, `num_outputs`, `no_nan_in_metrics`, `drop_breach`

## Notes
- Two Gaussian ion blobs with opposite drifts.
- Background particles included to avoid hard vacuum edges.
