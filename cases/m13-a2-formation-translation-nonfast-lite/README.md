# A2 Non-fast Formation + Translation (Lite)

This case exercises Hall-MHD formation + translation in a non-fast window with a
weaker axial bulk drift and reduced output cadence.
The axial direction is `x1` (axial z); translation is measured via the mass-weighted
centroid shift in `x1`.

Formation proxy
- Uses a poloidal flux proxy based on B1 (axial field) integrated across
  r=|x2|: `psi(r,z) = integral_0^r r' * B1(r',z) dr'`.
- Formation is declared when the normalized psi peak-to-edge indicator exceeds
  the threshold for a persistent window, with closed-flux persistence tracked
  via consecutive outputs.
- Density contrast is retained as a supplemental diagnostic.

Translation proxy
- `centroid_shift_frac = |x1_c(t_end) - x1_c(t0)| / (x1max - x1min)`.
- Monotonicity is tracked via the longest run of non-decreasing centroid motion.

Notes
- This is a gate-only proxy; it does not perform full cut-cell geometry or
  flux-surface topology reconstruction beyond the psi indicator.
- Gate thresholds tightened on 2026-01-02 (psi peak-to-edge >= 0.8, persistence >= 3, formation_time_frac <= 0.10).
