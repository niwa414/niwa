# B3.1 Electron Energy Coupled Sweep

This gate verifies that the electron-energy tracer is coupled back into the
Hybrid-PIC solver at the Python layer (density-floor writeback), and that the
coupling produces a measurable difference versus a control run.

Control vs treatment
- Control: tracer enabled, no writeback (`feedback_target=none`).
- Treatment: tracer enabled with density-floor writeback (`feedback_target=density_floor`).

Key metrics
- `treatment_writeback_ratio` and `treatment_floor_std` prove the writeback is active.
- `observable_rel_diff` confirms a measurable effect on a solver observable.

Limitations
- Python-level coupling only; no C++ Ohm-term electron energy equation.
- A small nonuniform applied Bz profile is used to keep field observables non-degenerate:
  `applied_Bz_expr=1.0e-4*(1.0+0.5*cos(3.141592653589793*x/0.1))`.
