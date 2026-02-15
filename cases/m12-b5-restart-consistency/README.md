# B5 Restart Consistency (pre/post)

## Purpose
- Verify Hybrid continuation stability across a checkpoint/restart boundary.
- Ensure key global metrics remain continuous after restart.

## Workflow
1. Prepare handoff from `m5-a2-formation-translation-gate` (extrude_x3).
2. Run pre-restart segment to step 100 and write a checkpoint.
3. Restart from the checkpoint and continue to step 200.

## Key metrics
- `mass_rel_diff_init`, `b_rms_rel_diff_init`, `mag_energy_rel_diff_init`
- `restart_jump_mass`, `restart_jump_b_rms`, `restart_jump_mag_energy`
- Stability: `warpx_ran_to_completion_pre/post`, `warpx_no_nan_in_metrics`

## Notes
- Checkpoints are written to `raw/run/checkpoints/` with prefix `chk`.
- Diags are separated into `diag_pre/` and `diag_post/` for continuity checks.
