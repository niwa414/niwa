# B2 Regression Gate (EMF + Moving-Wall + Mass-Core)

This gate freezes the 2D B2 moving-wall regression and defines the pass/fail
criteria for future code changes and 3D comparisons.

## Case Definition
- Case: B2 EMF + moving-wall, smooth waveform.
- Run to waveform progress >= 0.80 (using `Bext_frac_waveform`).
- Outputs: dense cadence (dt ~ 1e-8 in current 2D baseline).

## Gate Parameters
- Gate: `mass_core`, primary `f_mass=0.30`, control `f_mass=0.20`.
- Progress window: 0.10 to 0.80.
- Pre-fit smoothing: 3-point median on `V_core(t)`.
- Segment selection: longest approx-monotonic decreasing segment within window:
  - allow_increase_frac = 0.10
  - rebound_frac = 0.05 (max rebound / total drop)
  - min_points = 25

## Pass Thresholds (Primary: f_mass=0.30)
- n_fit >= 25
- Vdrop_fit <= -0.20
- R2_n >= 0.98, R2_B >= 0.95, R2_T >= 0.95
- alpha_n in [-1.20, -1.00]
- alpha_B in [-0.90, -0.65]
- alpha_T in [-3.40, -2.70]

## Consistency (Control vs Primary)
- rel_diff(alpha_n) <= 0.05
- rel_diff(alpha_B) <= 0.15
- rel_diff(alpha_T) <= 0.12

## Full-Window Sanity (Non-Blocking)
- Vdrop_full < 0
- rho_avg_core_full > 0
- mask_empty_total = 0
- Known behavior: full-window breathing is significant (rebound ~45-51% of net
  drop in the current dense baseline); this is recorded but not a blocker as
  long as the monotonic-segment fit passes.

## Baseline Reference
- Dense run: `outputs/mhd/b2-ramp-mid-emf-mwall80-smooth-dense-20251231-132054`
- Fit outputs:
  - `outputs/analysis/kirtley_scaling_emf_mwall80_smooth_dense_masscore_mass_core0p3.fit.json`
  - `outputs/analysis/kirtley_scaling_emf_mwall80_smooth_dense_masscore_mass_core0p2.fit.json`
