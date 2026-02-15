# H4: MHD-seeded 3D Hybrid continuation (gate)

## Purpose
- Bridge a Hall-MHD snapshot to a 3D Hybrid continuation run.
- Prove end-to-end handoff and short-window stability with auditable diffs.

## Snapshot selection
- Source: `m5-a2-formation-translation-gate` VTK outputs.
- Target time: `formation_time_frac * tlim` (from metrics + athinput).
- Select VTK closest to target time; fallback to latest when time metadata is missing.

## Mapping
- Mode: `extrude_x3` (slab extrude).
- Axis map: `x <- r`, `y <- extrude`, `z <- z`.
- Vector map: `vx<-vr`, `vy<-vphi`, `vz<-vz`; `Bx<-Br`, `By<-Bt`, `Bz<-Bz`.

## Key metrics
- `mass_rel_diff_init`, `b_rms_rel_diff_init`, `mag_energy_rel_diff_init`
- `mass_rel_drift_over_run`
- Stability: `warpx_ran_to_completion`, `warpx_no_nan_in_metrics`, `warpx_drop_breach`

## Notes
- Extrusion is used for a minimal, auditable 3D seed; it is not a full axisym->cartesian rotation.
- openPMD paths and selection metadata are recorded in `analysis/mapping_summary.json`.
