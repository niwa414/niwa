# A3.1 Strong Closed-Loop Base Variant (Per-Step Coupling)

This case upgrades the A3 closed-loop to per-step coupling: feedback is applied
every step (or small stride) and the driver writeback is updated immediately.

Key expectations (analysis/metrics.json)
- `strong_coupling_enabled = true`
- `coupling_stride = 1`
- `circuit_update_fraction >= 0.9`
- `driver_writeback_fraction >= 0.9` and `driver_writeback_match = true`
- `driver_amp_std > 0`
- `feedback_used_fraction >= 0.9`

Artifacts
- `radius_rms_vs_time.png`
- `driver_amp_vs_time.png`
- `radius_rms_and_driver_amp_overlay.png`

Notes
- This is a strong-coupling MVP; it proves per-step circuit updates and writeback,
  not a fully implicit circuit-plasma solve.
