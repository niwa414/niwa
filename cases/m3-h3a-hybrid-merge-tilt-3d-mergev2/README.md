# M3-H3a Merge Detection v2 (k-means)

Merge detection upgrade using k-means clustering on high-density cells.

## Goal
- Provide a merge-time estimate that is less dependent on x-split geometry.
- Use k-means inertia-ratio (I/I0) to determine merge time without x-split floors.
- Compare k-means merge time to x-split baseline (no removal of baseline).

## Key metrics
- `merge_time_exists`, `merge_time_frac` (k-means)
- `kmeans_cluster_ratio_min` (cluster balance)
- `kmeans_inertia_ratio_min` (merge indicator)
