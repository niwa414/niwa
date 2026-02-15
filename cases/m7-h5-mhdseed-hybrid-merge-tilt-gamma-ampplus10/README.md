# H5: MHD-seeded Hybrid merge → tilt → gamma (gate, seed_amp +10%)

This gate seeds a 3D Hybrid-PIC run from the H4 openPMD output and creates a
second, shifted copy of the seed distribution to induce merge and post-merge tilt.
It reuses the H3a/H3b merge + gamma analysis with identical k-means settings.

## Seed construction (opmd_double_seed)

- **Source**: `outputs/m6-h4-mhd-seeded-hybrid-3d-continuation/raw/run/{fluid_init.h5,B_ext.h5}`
- **Mapping**: extrude_x3 (from H4)
- **Double seed**: the openPMD particle set is duplicated, then shifted by
  `opmd_double_seed_shift` with symmetric offsets (±0.5*shift) and opposite
  drift `opmd_double_seed_drift` applied in momentum space.
- **Tilt seed**: position tilt (`tilt_seed_mode=pos_tilt`) applies an antisymmetric
  `y` offset, `y += +/- Dy(z)`, on the two blobs (profile controlled by
  `tilt_seed_y_offset_*`).
- **Seed amp**: `tilt_seed_y_offset_amp=0.0088` (+10% vs baseline 0.008).

> Note: drift is applied by adding a constant `u = gamma*beta` offset to the
> sampled particle momenta; this is an MVP approximation for gate purposes.

## Gate metrics (PASS/FAIL)

- Merge detection: k-means separation threshold with persist window.
- Post-merge tilt series: non-NaN z-dependent tilt slope with sufficient samples.
- Gamma fit: `gamma_best > 0`, `r2_best >= 0.85`, `fit_points >= 30`.

## Outputs

- `analysis/metrics.json`: combined H5 metrics (merge + tilt + gamma).
- `analysis/metrics_h3a.json`: merge/tilt series metrics.
- `analysis/metrics_h3b.json`: gamma fit metrics.
- Plots: merge indicators, post-merge tilt amplitude, and gamma log fit.
