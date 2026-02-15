# A4 EB Wall Step Case (MVP)

Purpose: demonstrate an embedded-boundary (stair-step mask) wall via `cyl_wall`
with a discontinuous step profile and provide auditable leak metrics.
This case is intentionally a short-window stability gate (`tlim=0.25`) to avoid
late-time CFL collapse around the discontinuous step wall.

Geometry
- Wall profile: step at `cyl_wall_x1_step` with radius `cyl_wall_r0` for x1 < step
  and `cyl_wall_r1` for x1 >= step.
- This is a stair-step EB (mask + flux blocking), not a full cut-cell reconstruction.

Key metrics (analysis/metrics.json)
- `leak_mass_frac_max`: max((mass_out - mass_out_initial) / mass_in_initial).
- Leak definition (A4): `leak_mass_frac_max = max_t((M_out(t) - M_out(t0)) / M_in(t0))`, where `M_out` is mass outside the EB wall mask (delta form avoids baseline bias).
- `mass_rel_drift_in`: max(|mass_in - mass_in_initial| / mass_in_initial).
- `wall_cells_count`: number of masked-out cells.
- `blocked_flux_faces_count`: estimated number of wall-crossing faces from mask adjacency.
- `eb_mask_applied`: true when wall is enabled and mask/blocked faces are nonzero.

Evidence
- `plots/eb_mask_snapshot.png`: mask visualization at x3 mid-plane.
- `plots/eb_mask_snapshot_step.png`: step-wall mask visualization (x3 mid-plane).
- `plots/leak_mass_out_vs_time.png`: mass in/out time series (HST).

Notes
- This gate targets A4 MVP (stair-step EB). It does not claim full cut-cell area/volume
  reconstruction or higher-order embedded boundary accuracy.
- Gate thresholds are set for MVP screening (`leak_mass_frac_max <= 1e-2`,
  `mass_rel_drift_in <= 1e-2`) rather than full cut-cell closure.
