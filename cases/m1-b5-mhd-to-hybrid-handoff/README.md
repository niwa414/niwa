# M1-B5 MHD -> Hybrid Handoff

Gate case for Athena++ VTK → openPMD → WarpX Hybrid-PIC initialization.

## What it runs
1. Select latest VTK from `m1-a2-frc-merge-fast`, export openPMD fluid + B files.
2. WarpX `fluid-init` Hybrid-PIC (m=0) short run.
3. Mapping + WarpX stability diagnostics.

## Key metrics
- Mapping completeness: `opmd_exists`, `opmd_fields_present`, `opmd_no_nan`
- Mapping consistency: `mass_rel_diff`, `mag_energy_rel_diff`
- WarpX stability: `warpx_ran_to_completion`, `warpx_num_outputs`, `warpx_no_nan_in_metrics`, `warpx_drop_breach`

## Notes
- Axis mapping follows `x1 -> z`, `x2 -> r` with `fold_r=true`.
- The VTK source is the latest `frc_merge` snapshot from the fast-merge gate.
