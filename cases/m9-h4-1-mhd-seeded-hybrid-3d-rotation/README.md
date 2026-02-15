# H4.1: MHD-seeded 3D Hybrid continuation (rotation mapping)

## Purpose
- Bridge a Hall-MHD snapshot to a 3D Hybrid continuation run.
- Prove end-to-end handoff and short-window stability with auditable diffs.

## Snapshot selection
- Source: `m5-a2-formation-translation-gate` VTK outputs.
- Target time: `formation_time_frac * tlim` (from metrics + athinput).
- Select VTK closest to target time; fallback to latest when time metadata is missing.

## Mapping
- Mode: `rotate_axisym` (axisym -> 3D cartesian).
- Scalar fields are bilinear-interpolated on (r,z).
- Vector map uses `phi = atan2(y,x)`:
  - `Bx = Br*cos(phi) - Bphi*sin(phi)`
  - `By = Br*sin(phi) + Bphi*cos(phi)`
  - `Bz = Bz`
  - `vx = vr*cos(phi) - vphi*sin(phi)`
  - `vy = vr*sin(phi) + vphi*cos(phi)`
  - `vz = vz`

## Key metrics
- `mass_rel_diff_init`, `b_rms_rel_diff_init`, `mag_energy_rel_diff_init`
- `mass_rel_drift_over_run`
- Stability: `warpx_ran_to_completion`, `warpx_no_nan_in_metrics`, `warpx_drop_breach`
- Rotation check: `rotation_mass_integral_rel_diff` (axisym 2D vs 3D cylindrical integral)

## Notes
- Axis singularity handling and r-clip policy are recorded in `handoff_meta.json`.
- openPMD paths and selection metadata are recorded in `analysis/mapping_summary.json`.
