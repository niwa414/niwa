# M1-B2 Hybrid Tilt Smoke

Gate case for Hybrid-PIC tilt in WarpX RZ (m=1) using the Python driver.

## What it runs
- `warpx-driver/warpx_driver.py` in `const-b-plasma` mode.
- `n_azimuthal_modes=2` with a small `tilt_eps` perturbation to seed m=1.
- Hybrid-PIC enabled (`--hybrid`) with synthetic uniform fluid loading.

## Key metrics (analysis)
- `tilt_amp_ratio`: `max(rho_m1_rms) / rho_m1_rms(t0)` from thetaMode diags.
- `num_outputs`: number of diag directories under `raw/run/diag`.
- `ran_to_completion`: inferred from monitor records vs `max_steps`.

## Expected artifacts
- `plots/tilt_m1_amplitude_vs_time.png`
- `plots/energy_budget_vs_time.png` (field energy from diag outputs)
- `plots/density_snapshot.png`
