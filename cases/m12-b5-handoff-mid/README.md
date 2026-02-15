# B5 Handoff (mid frame)

## Purpose
- Bridge a Hall-MHD snapshot to a 3D Hybrid continuation run.
- Validate handoff consistency at a mid-sequence snapshot time.

## Snapshot selection
- Source: `m5-a2-formation-translation-gate` VTK outputs.
- Explicit VTK: `frc_merge.block0.out1.00018.vtk` (~60% index of the sequence).
- Selection metadata is recorded in `analysis/mapping_summary.json`.

## Mapping
- Mode: `extrude_x3` (slab extrude).
- Axis map: `x <- r`, `y <- extrude`, `z <- z`.
- Vector map: `vx<-vr`, `vy<-vphi`, `vz<-vz`; `Bx<-Br`, `By<-Bt`, `Bz<-Bz`.

## Key metrics
- `mass_rel_diff_init`, `b_rms_rel_diff_init`, `mag_energy_rel_diff_init`
- Stability: `warpx_ran_to_completion`, `warpx_no_nan_in_metrics`, `warpx_drop_breach`

## Notes
- Extrusion is used for a minimal, auditable 3D seed; it is not a full axisym->cartesian rotation.
