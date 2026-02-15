# M1-B2 Tilt Smoke (3D Non-Hybrid)

Tiny 3D cartesian tilt smoke to produce a **physical** m=1 centroid signal.

## What it runs
- WarpX 3D EM PIC (non-hybrid), single ion species.
- Gaussian density blob with a small x-offset and a uniform drift in +x.

## Key metrics
- `tilt_amp_ratio`: `A(t_end) / A(t_start)` where `A = sqrt(x_c^2 + y_c^2)`
- `ran_to_completion`, `num_outputs`, `drop_breach`

## Notes
- This case is **physical** (3D data, centroid from density) and replaces the RZ pipeline sanity gate.
- Single-species charge cloud is intentional for a cheap centroid signal; it is not a quasi-neutral plasma.
