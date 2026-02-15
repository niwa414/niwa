# A4.2 EB + Moving-Wall Short (MVP)

Purpose: demonstrate an embedded-boundary (stair-step mask) wall via `cyl_wall`
with a tapered radius profile `r_wall(x1)` plus a short moving-wall drive.

Geometry
- Wall profile: linear taper from `cyl_wall_r0` at `cyl_wall_x1_min` to
  `cyl_wall_r1` at `cyl_wall_x1_max` (constant outside that interval).
- This is a stair-step EB (mask + flux blocking), not a full cut-cell reconstruction.

Key metrics (analysis/metrics.json)
- `leak_mass_frac_max`: max((mass_out - mass_out_initial) / mass_in_initial).
- Leak definition (A4): `leak_mass_frac_max = max_t((M_out(t) - M_out(t0)) / M_in(t0))`, where `M_out` is mass outside the EB wall mask (delta form avoids baseline bias).
- `mass_rel_drift_in`: max(|mass_in - mass_in_initial| / mass_in_initial).
- `wall_cells_count`: number of masked-out cells.
- `blocked_flux_faces_count`: estimated number of wall-crossing faces from mask adjacency.
- `eb_mask_applied`: true when wall is enabled and mask/blocked faces are nonzero.
- `piston_metric`: estimated moving-wall amplitude from waveform (input-based proxy).

Evidence
- `plots/eb_mask_snapshot.png`: mask visualization at x3 mid-plane.
- `plots/leak_mass_out_vs_time.png`: mass in/out time series (HST).

Notes
- This gate targets A4 MVP (stair-step EB) with a short moving-wall drive. It does not
  claim full cut-cell area/volume reconstruction or higher-order embedded boundary accuracy.
- Current tuned profile uses `cyl_wall_r1=0.77` at `tlim=1.4` with tightened closure gate:
  `mass_budget_residual_geom_rel <= 2e-3` plus `leak_mass_frac_max <= 1e-2`.
