# 3D Driver Spec (B2 EMF + Moving-Wall + Mass-Core Gate)

## Goal
Establish a 3D driver that reproduces the frozen B2 gate using the same
EMF-driven mirror ramp, moving-wall compression, and mass_core diagnostics.
This spec is MVP-first: get stable 3D runs with consistent gates before adding
extra physics or geometry complexity.

## Coordinate System and Geometry
- Use 3D Cartesian.
- Axis mapping:
  - x1 = z (axial)
  - x2 = x, x3 = y
  - r = sqrt(x2^2 + x3^2)

This keeps the 2D axisymmetric assumptions while avoiding cylindrical CT
complexity.

## GS -> 3D Initialization (MVP)
Input is 2D GS HDF5 in (z, r). At runtime map to 3D by radius.
Enable with `problem/init_axisym_2d = true` when `init_from_hdf5` points to a
2D axisymmetric file.

2D fields (current convention):
- B1 = Bz, B2 = Br, B3 = Bphi
- V1 = Vz, V2 = Vr, V3 = Vphi

3D mapping:
- Bx = Br * (x2 / r) - Bphi * (x3 / r)
- By = Br * (x3 / r) + Bphi * (x2 / r)
- Bz = Bz

- Vx = Vr * (x2 / r) - Vphi * (x3 / r)
- Vy = Vr * (x3 / r) + Vphi * (x2 / r)
- Vz = Vz

Axis handling:
- For r < r_eps (e.g., 1e-12), set x2/r = 1, x3/r = 0 or set Bx=By=0 and
  Vx=Vy=0 to avoid NaN and preserve symmetry.

## EMF External Drive (CT-Consistent)
Define a poloidal flux and use Faraday-consistent EMF injection.

Flux:
  psi_ext(z, r, t) = C(t) * psi_shape(z, r)

Time derivative:
  psi_dot = dC/dt * psi_shape(z, r)

Axisymmetric EMF:
  Ephi = -psi_dot / max(r, r_eps)

Cartesian projection:
  Ex = Ephi * (-x3 / r)
  Ey = Ephi * ( x2 / r)
  Ez = 0

Implementation:
- Add (Ex, Ey) to CT EMF arrays in UserEMF.
- Keep r_eps safeguards for r -> 0.
- Use the same waveform as 2D (CSV) to drive C(t).

## Moving-Wall (Mechanical Compression)
Use moving-wall mask with reflecting interface (no ALE).

Wall position:
- z_w(t) = depth_max * progress(t)
- Solid region: x1 < x1min + z_w(t) or x1 > x1max - z_w(t)

Interface update:
- Reflect conserved vars across the moving wall with wall speed.
- Apply to all (j,k) at a given x1 plane (full cross-section).

Diagnostics:
- Continue HST outputs: mass_mv, Etot_mv, vol_mv (fluid-only).

## Diagnostics and Gate (Reuse 2D)
Reuse `tools/analyze_kirtley_scaling.py` with `mass_core`.

3D requirements:
- Cell volumes must be computed in 3D (dx*dy*dz).
- mass_core sorting and accumulation should handle larger arrays (consider
  chunking if needed).

## 3D Run Stages

### 3D-0 Smoke
- Short run (few outputs) to validate:
  - GS -> 3D mapping (no NaN, reasonable fields).
  - EMF injection in 3D (Bext_progress matches waveform).
  - Moving-wall metrics: vol_mv decreases, mass_mv stable.

### 3D-1 Gate60
- Run to progress ~0.60 (not dense).
- Pass: mass_core(0.30) shows Vdrop < 0, mask_empty=0, stable mass fraction.

### 3D-2 Gate80 Dense
- Run to progress ~0.80 with dense output cadence.
- Use frozen B2 regression gate (see `b2_gate_regression.md`).
- Compare 3D exponents to 2D within tolerance or explain offsets.

## Known Risk Points (Plan Ahead)
- EMF projection correctness at small r (avoid spurious divergence).
- CT EMF injection location for Ex/Ey edges (confirm sign conventions).
- Moving-wall mask in 3D (full cross-section consistency).

## Deliverables
- 3D input templates for smoke, gate60, gate80_dense.
- Documentation of EMF injection and GS->3D mapping in code comments.
- 3D regression outputs using the same gate rules as 2D.
