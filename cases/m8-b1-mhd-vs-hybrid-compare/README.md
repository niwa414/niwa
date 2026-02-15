# B1: MHD vs Hybrid comparison (gate)

## Purpose
- Compare an MHD snapshot against a short-window Hybrid continuation with the same handoff.
- Produce auditable t0 agreement and short-run drift metrics.

## Source and mapping
- Source: `m5-a2-formation-translation-gate` VTK outputs.
- Mapping: `extrude_x3` (same as H4), `x <- r`, `y <- extrude`, `z <- z`.

## Key metrics
- `mass_rel_diff_init`, `b_rms_rel_diff_init`, `mag_energy_rel_diff_init`
- `mass_rel_drift_over_run`
- Stability: `warpx_ran_to_completion`, `warpx_no_nan_in_metrics`, `warpx_drop_breach`

## Outputs
- `analysis/metrics.json` (B1 compare metrics)
- `analysis/compare_summary.json` (config + mapping summary)
- Plots reuse H4 continuation diagnostics
