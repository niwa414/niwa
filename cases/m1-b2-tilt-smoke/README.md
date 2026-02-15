# M1-B2 Tilt Smoke (Non-Hybrid)

Tilt diagnostic gate (m=1) for WarpX RZ without Hybrid-PIC.

## What it runs
- `warpx-driver/warpx_driver.py` in `const-b-plasma` mode.
- `n_azimuthal_modes=2`, `solver=yee` (PSATD recommended), `tilt_eps=0.05`.

## Key metrics
- `tilt_amp_ratio`: `max(particle_m1_amp) / particle_m1_amp(t0)`
  where `particle_m1_amp = |sum(w * exp(i*theta))| / sum(w)` for the chosen species.
- `ran_to_completion`, `num_outputs`, `drop_breach`

## Notes
Hybrid-PIC in RZ is m=0 only; Hybrid tilt requires 3D. In this smoke case, RZ particle theta
defaults to a degenerate value, so the m=1 amplitude is a pipeline sanity check only.
