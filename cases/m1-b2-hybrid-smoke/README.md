# M1-B2 Hybrid Smoke

Hybrid-PIC chain gate (m=0 only) to verify WarpX Hybrid runs and diagnostics.

## What it runs
- `warpx-driver/warpx_driver.py` in `fluid-init` mode.
- Hybrid-PIC enabled, `n_azimuthal_modes=1` (RZ m=0).
- Uses `warpx-driver/fluid_init_hall_frc_tilt.h5` and `warpx-driver/B_ext_hall_frc_tilt.h5`.

## Key metrics
- `ran_to_completion`, `num_outputs`, `drop_breach`
- `energy_budget_vs_time.png`, `density_snapshot.png`

## Notes
Hybrid-PIC in RZ does not support m>0; tilt is verified in a separate gate.
