# M2-H1 Hybrid Tilt (3D Single)

Hybrid 3D tilt gate on a single Gaussian ion blob with a small offset.

## Goal
- Demonstrate a stable 3D Hybrid-PIC run with a usable centroid time series.
- No growth requirement; this is a Hybrid tilt indicator gate.

## Key metrics
- `centroid_amp_max` (must be finite and > 0)
- `ran_to_completion`, `num_outputs`, `no_nan_in_metrics`, `drop_breach`

## Notes
- Ions are kinetic; electrons are fluid (Hybrid-PIC).
- A low-density background is included to avoid hard vacuum edges.
