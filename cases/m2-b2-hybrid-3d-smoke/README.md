# M2-B2 Hybrid Smoke (3D)

Minimal 3D **Hybrid-PIC** smoke gate on Cartesian grid (CPU).

## Goal
- Validate 3D Hybrid solver runs to completion and produces stable diagnostics.
- Track a physical centroid amplitude time series (no growth requirement).

## Key metrics
- `centroid_amp_max` (must be finite and > 0)
- `ran_to_completion`, `num_outputs`, `no_nan_in_metrics`, `drop_breach`

## Notes
- Ions are kinetic; electrons are fluid (Hybrid-PIC).
- Background particles are optional and used here to avoid a hard vacuum edge.
