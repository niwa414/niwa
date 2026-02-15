# M3-H3a Hybrid Merge→Tilt (3D)

Hybrid 3D merge + post-merge tilt extraction gate.

## Goal
- Demonstrate merge completion (separation drops below threshold).
- Extract post-merge tilt centroid series (no NaN, adequate samples).

## Key metrics
- `merge_time_exists`, `merge_time_frac`, `sep_min`, `sep_ratio`
- `tilt_post_merge_samples`, `tilt_post_merge_amp_max`, `tilt_post_merge_no_nan`

## Notes
- Two Gaussian ion blobs with opposite drift.
- Default merge detector: k-means inertia ratio `I(t)/I0` with persistence window.
- x-split centroids (x<0 / x>0) retained as fallback and for merge-time comparison.
- Tilt series uses global centroid after merge.
- x-split merge threshold uses `max(merge_frac * sep0, merge_floor_mult * sigma)` to account for the split-centroid floor.
