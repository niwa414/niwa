# M4-A3 Closed-Loop Circuit Gate

Minimal gate for A3: LCR circuit driven in-simulation with plasma feedback.

## Goal
- Demonstrate online circuit feedback (not a precomputed waveform).
- Show feedback signal and circuit state evolve together.

## Feedback Definition
- Signal: `radius_rms` from charge density (RZ) in `warpx-driver/warpx_driver.py`.
- Feedback path: `radius_rms -> LCR R_plasma` (smoothed) -> B(t) scaling.

## Expected Outputs
- `plots/lcr_current_vs_time.png`
- `plots/feedback_vs_rplasma.png`
- `analysis/lcr_history.csv`
- `analysis/feedback_series.csv`

## PASS/FAIL
- `closed_loop_enabled == True`
- `feedback_signal_present == True`
- `feedback_update_count >= 10`
- `feedback_signal_std > 0`
