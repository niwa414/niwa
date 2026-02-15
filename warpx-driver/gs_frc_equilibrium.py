#!/usr/bin/env python3
"""
Generate a simple Grad-Shafranov (GS) FRC equilibrium on an RZ grid and export
openPMD fluid + B files for WarpX.

Notes:
  - Uses a linear pressure profile p(psi) with constant dp/dpsi.
  - Toroidal field is set to zero (Bt=0).
  - Optional pair mode uses linear superposition of two shifted equilibria.
  - Intended as a lightweight initializer for B1/B3/C1/C2 style scans.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from export_b_opmd_from_vtk import write_openpmd_B
from export_fluid_to_opmd import write_openpmd_fluid

MU0 = 4.0e-7 * np.pi
K_B = 1.380649e-23
M_U = 1.66053906660e-27


def solve_gs_unit(
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    omega: float,
    max_iter: int,
    tol: float,
):
    """Solve GS with dp/dpsi = -1 (unit) on a rectangular domain; return psi, r, z, dr, dz."""
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    r = r_min + (np.arange(nr) + 0.5) * dr
    z = z_min + (np.arange(nz) + 0.5) * dz
    psi = np.zeros((nr, nz), dtype=np.float64)

    coeff = 2.0 / (dr * dr) + 2.0 / (dz * dz)
    for it in range(max_iter):
        psi[0, :] = psi[1, :]  # axis symmetry
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0
        max_delta = 0.0
        for i in range(1, nr - 1):
            ri = r[i]
            for j in range(1, nz - 1):
                lap = (psi[i + 1, j] + psi[i - 1, j]) / (dr * dr)
                lap += (psi[i, j + 1] + psi[i, j - 1]) / (dz * dz)
                term_r = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dr * ri)
                rhs = MU0 * ri * ri  # dp/dpsi = -1 -> RHS positive
                psi_new = (lap - term_r + rhs) / coeff
                delta = psi_new - psi[i, j]
                psi[i, j] += omega * delta
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
        if max_delta < tol:
            break
    return psi, r, z, dr, dz


def shift_z(arr: np.ndarray, z: np.ndarray, shift: float) -> np.ndarray:
    """Shift a (nr,nz) array along z by linear interpolation; fill with zeros outside."""
    nr, nz = arr.shape
    z_shifted = z - shift
    out = np.empty_like(arr)
    for i in range(nr):
        out[i, :] = np.interp(z_shifted, z, arr[i, :], left=0.0, right=0.0)
    return out


def compute_b_from_psi(psi: np.ndarray, r: np.ndarray, dr: float, dz: float):
    """Return (Br, Bz) at cell centers from psi on an RZ grid."""
    nr, nz = psi.shape
    Br = np.zeros_like(psi)
    Bz = np.zeros_like(psi)
    for i in range(nr):
        ri = r[i]
        for j in range(nz):
            if i == 0:
                dpsi_dr = (psi[i + 1, j] - psi[i, j]) / dr
            elif i == nr - 1:
                dpsi_dr = (psi[i, j] - psi[i - 1, j]) / dr
            else:
                dpsi_dr = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dr)

            if j == 0:
                dpsi_dz = (psi[i, j + 1] - psi[i, j]) / dz
            elif j == nz - 1:
                dpsi_dz = (psi[i, j] - psi[i, j - 1]) / dz
            else:
                dpsi_dz = (psi[i, j + 1] - psi[i, j - 1]) / (2.0 * dz)

            Bz[i, j] = dpsi_dr / ri
            Br[i, j] = -dpsi_dz / ri
    return Br, Bz


def main():
    ap = argparse.ArgumentParser(description="Generate GS FRC equilibrium (openPMD outputs).")
    ap.add_argument("--nr", type=int, default=128, help="Radial cells.")
    ap.add_argument("--nz", type=int, default=256, help="Axial cells.")
    ap.add_argument("--r-min", type=float, default=0.0, help="Minimum r (m).")
    ap.add_argument("--r-max", type=float, default=0.25, help="Maximum r (m).")
    ap.add_argument("--z-min", type=float, default=-0.25, help="Minimum z (m).")
    ap.add_argument("--z-max", type=float, default=0.25, help="Maximum z (m).")
    ap.add_argument("--b-bias", type=float, default=0.05, help="Target Bz at separatrix (T).")
    ap.add_argument("--beta-s", type=float, default=0.15, help="Separatrix beta.")
    ap.add_argument("--Ti-eV", type=float, default=100.0, help="Ion temperature (eV).")
    ap.add_argument("--Te-Ti", type=float, default=1.0, help="Te/Ti ratio.")
    ap.add_argument("--amu", type=float, default=2.0, help="Ion mass number A.")
    ap.add_argument("--rho-sep", type=float, default=None, help="Override separatrix mass density (kg/m^3).")
    ap.add_argument("--rho-axis", type=float, default=None, help="Override axis mass density (kg/m^3).")
    ap.add_argument("--pair-separation", type=float, default=0.0, help="If >0, superpose two FRCs separated by this distance (m).")
    ap.add_argument("--omega", type=float, default=1.6, help="SOR relaxation parameter.")
    ap.add_argument("--max-iter", type=int, default=5000, help="Max SOR iterations.")
    ap.add_argument("--tol", type=float, default=1.0e-6, help="SOR tolerance.")
    ap.add_argument("--output-b", type=Path, default=Path("warpx-driver/B_ext_gs_frc.h5"), help="Output openPMD B file.")
    ap.add_argument("--output-fluid", type=Path, default=Path("warpx-driver/fluid_init_gs_frc.h5"), help="Output openPMD fluid file.")
    args = ap.parse_args()

    if args.r_min != 0.0:
        raise SystemExit("r_min must be 0.0 for axisymmetry in this GS solver.")

    psi_unit, r, z, dr, dz = solve_gs_unit(
        args.nr, args.nz, args.r_min, args.r_max, args.z_min, args.z_max,
        args.omega, args.max_iter, args.tol
    )

    if args.pair_separation > 0.0:
        psi_left = shift_z(psi_unit, z, -0.5 * args.pair_separation)
        psi_right = shift_z(psi_unit, z, 0.5 * args.pair_separation)
        psi_unit = psi_left + psi_right

    Br_unit, Bz_unit = compute_b_from_psi(psi_unit, r, dr, dz)
    mid_z = args.nz // 2
    b_unit = float(np.abs(Bz_unit[-2, mid_z]))
    if b_unit <= 0.0:
        raise SystemExit("Failed to compute non-zero boundary B; check geometry.")

    scale = args.b_bias / b_unit
    psi = scale * psi_unit
    Br = scale * Br_unit
    Bz = scale * Bz_unit
    Bt = np.zeros_like(Bz)

    p_sep = args.beta_s * args.b_bias * args.b_bias / (2.0 * MU0)
    p_prime = -scale  # dp/dpsi
    p = p_sep + p_prime * psi

    psi_axis = float(np.min(psi))
    if psi_axis == 0.0:
        raise SystemExit("Degenerate psi field; adjust parameters.")
    psi_norm = psi / psi_axis  # 0 at boundary, 1 at axis

    Ti_K = args.Ti_eV * 11604.518
    Te_K = Ti_K * args.Te_Ti
    if args.rho_sep is not None and args.rho_axis is not None:
        rho_sep = float(args.rho_sep)
        rho_axis = float(args.rho_axis)
        rho = rho_sep + (rho_axis - rho_sep) * psi_norm
        n = rho / (args.amu * M_U)
    else:
        n = p / (K_B * (Ti_K + Te_K))
        rho = n * args.amu * M_U

    vr = np.zeros_like(rho)
    vz = np.zeros_like(rho)
    vphi = np.zeros_like(rho)
    Ti = np.full_like(rho, Ti_K)
    Te = np.full_like(rho, Te_K)

    write_openpmd_B(args.output_b, Br, Bz, Bt, args.r_min, args.r_max, args.z_min, args.z_max)
    write_openpmd_fluid(
        args.output_fluid, rho, vr, vz, vphi, Ti, Te, args.r_min, args.r_max, args.z_min, args.z_max
    )

    print(
        f"[gs-frc] wrote B -> {args.output_b}, fluid -> {args.output_fluid} "
        f"(B_bias={args.b_bias} T, beta_s={args.beta_s})"
    )


if __name__ == "__main__":
    main()
