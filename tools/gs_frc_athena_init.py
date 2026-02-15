#!/usr/bin/env python3
"""
Generate a Grad-Shafranov FRC equilibrium and export Athena++ HDF5 init arrays.

The output HDF5 file contains datasets compatible with frc_merge.cpp when
problem/init_from_hdf5 is set:
  - init_dataset_cons (default: cons)
  - init_dataset_b1/b2/b3 (default: b1/b2/b3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

MU0 = 4.0e-7 * np.pi
K_B = 1.380649e-23
M_U = 1.66053906660e-27


def _parse_number(text: str):
    try:
        return float(eval(text, {"__builtins__": {}}, {}))
    except Exception:
        return None


def parse_athinput(path: Path):
    """Minimal Athena++ input parser for mesh/meshblock/hydro values."""
    data: dict[str, dict[str, float]] = {"mesh": {}, "meshblock": {}, "hydro": {}}
    block = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("<") and line.endswith(">"):
            name = line.strip("<>").strip("/").split()[0]
            block = name
            continue
        if block not in data or "=" not in line:
            continue
        key, val = [s.strip() for s in line.split("=", 1)]
        num = _parse_number(val)
        if num is None:
            continue
        data[block][key] = num
    return data


def solve_gs_unit(nr, nz, r_min, r_max, z_min, z_max, omega, max_iter, tol):
    """Solve GS with dp/dpsi = -1 (unit) on a rectangular domain."""
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    r = r_min + (np.arange(nr) + 0.5) * dr
    z = z_min + (np.arange(nz) + 0.5) * dz
    psi = np.zeros((nr, nz), dtype=np.float64)

    coeff = 2.0 / (dr * dr) + 2.0 / (dz * dz)
    for _ in range(max_iter):
        psi[0, :] = psi[1, :]
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
                rhs = MU0 * ri * ri
                psi_new = (lap - term_r + rhs) / coeff
                delta = psi_new - psi[i, j]
                psi[i, j] += omega * delta
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
        if max_delta < tol:
            break
    return psi, r, z, dr, dz


def shift_z(arr, z, shift):
    """Shift a (nr,nz) array along z by linear interpolation; fill with zeros outside."""
    nr, _ = arr.shape
    z_shifted = z - shift
    out = np.empty_like(arr)
    for i in range(nr):
        out[i, :] = np.interp(z_shifted, z, arr[i, :], left=0.0, right=0.0)
    return out


def compute_b_faces(psi, r_centers, z_centers, r_faces, z_faces):
    """Compute face-centered Bz (x1 faces) and Br (x2 faces) from psi."""
    nr, nz = psi.shape
    dr = r_centers[1] - r_centers[0]
    dz = z_centers[1] - z_centers[0]

    psi_zface = np.zeros((nr, nz + 1), dtype=np.float64)
    psi_zface[:, 1:nz] = 0.5 * (psi[:, : nz - 1] + psi[:, 1:nz])

    dpsi_dr = np.zeros_like(psi_zface)
    dpsi_dr[0, :] = (psi_zface[1, :] - psi_zface[0, :]) / dr
    dpsi_dr[-1, :] = (psi_zface[-1, :] - psi_zface[-2, :]) / dr
    dpsi_dr[1:-1, :] = (psi_zface[2:, :] - psi_zface[:-2, :]) / (2.0 * dr)
    B1 = dpsi_dr / r_centers[:, None]

    psi_rface = np.zeros((nr + 1, nz), dtype=np.float64)
    psi_rface[1:nr, :] = 0.5 * (psi[: nr - 1, :] + psi[1:nr, :])

    dpsi_dz = np.zeros_like(psi_rface)
    dpsi_dz[:, 0] = (psi_rface[:, 1] - psi_rface[:, 0]) / dz
    dpsi_dz[:, -1] = (psi_rface[:, -1] - psi_rface[:, -2]) / dz
    dpsi_dz[:, 1:-1] = (psi_rface[:, 2:] - psi_rface[:, :-2]) / (2.0 * dz)

    B2 = np.zeros_like(psi_rface)
    for i, rf in enumerate(r_faces):
        if rf == 0.0:
            B2[i, :] = 0.0
        else:
            B2[i, :] = -dpsi_dz[i, :] / rf

    return B1, B2


def main():
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="GS FRC -> Athena++ HDF5 initializer")
    ap.add_argument("--athinput", type=Path, default=None, help="Optional Athena++ input to infer mesh.")
    ap.add_argument("--nx1", type=int, default=None)
    ap.add_argument("--nx2", type=int, default=None)
    ap.add_argument("--nx3", type=int, default=1)
    ap.add_argument("--x1min", type=float, default=None)
    ap.add_argument("--x1max", type=float, default=None)
    ap.add_argument("--x2min", type=float, default=None)
    ap.add_argument("--x2max", type=float, default=None)
    ap.add_argument("--x3min", type=float, default=0.0)
    ap.add_argument("--x3max", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=None, help="Adiabatic gamma.")
    ap.add_argument("--b-bias", type=float, default=0.05, help="Target Bz at boundary (T).")
    ap.add_argument("--beta-s", type=float, default=0.15, help="Separatrix beta.")
    ap.add_argument("--Ti-eV", type=float, default=100.0, help="Ion temperature (eV).")
    ap.add_argument("--Te-Ti", type=float, default=1.0, help="Te/Ti ratio.")
    ap.add_argument("--amu", type=float, default=2.0, help="Ion mass number A.")
    ap.add_argument("--rho-sep", type=float, default=None, help="Override separatrix mass density (kg/m^3).")
    ap.add_argument("--rho-axis", type=float, default=None, help="Override axis mass density (kg/m^3).")
    ap.add_argument("--rho-floor", type=float, default=0.0, help="Density floor (kg/m^3).")
    ap.add_argument("--p-floor", type=float, default=0.0, help="Pressure floor (Pa).")
    ap.add_argument(
        "--b-toroidal",
        type=float,
        default=None,
        help="Uniform toroidal field Bphi (T). Defaults to b_bias.",
    )
    ap.add_argument("--pair-separation", type=float, default=0.0, help="Superpose two FRCs separated by this distance (m).")
    ap.add_argument("--v-inflow", type=float, default=0.0, help="Axial inflow speed toward midplane (m/s).")
    ap.add_argument("--v-profile-power", type=float, default=1.0, help="Exponent for velocity weighting vs psi_norm.")
    ap.add_argument("--omega", type=float, default=1.6, help="SOR relaxation parameter.")
    ap.add_argument("--max-iter", type=int, default=5000, help="Max SOR iterations.")
    ap.add_argument("--tol", type=float, default=1.0e-6, help="SOR tolerance.")
    ap.add_argument(
        "--output-h5",
        type=Path,
        default=repo_root / "outputs" / "mhd" / "gs_init" / "frc_gs_init.h5",
    )
    args = ap.parse_args()

    if args.athinput is not None:
        parsed = parse_athinput(args.athinput)
        mesh = parsed.get("mesh", {})
        meshblock = parsed.get("meshblock", {})
        hydro = parsed.get("hydro", {})
        args.nx1 = args.nx1 or int(mesh.get("nx1", 0))
        args.nx2 = args.nx2 or int(mesh.get("nx2", 0))
        args.nx3 = args.nx3 or int(mesh.get("nx3", args.nx3))
        args.x1min = args.x1min if args.x1min is not None else mesh.get("x1min")
        args.x1max = args.x1max if args.x1max is not None else mesh.get("x1max")
        args.x2min = args.x2min if args.x2min is not None else mesh.get("x2min")
        args.x2max = args.x2max if args.x2max is not None else mesh.get("x2max")
        args.x3min = args.x3min if args.x3min is not None else mesh.get("x3min", args.x3min)
        args.x3max = args.x3max if args.x3max is not None else mesh.get("x3max", args.x3max)
        args.gamma = args.gamma if args.gamma is not None else hydro.get("gamma")
        if meshblock:
            mb_nx1 = int(meshblock.get("nx1", args.nx1))
            mb_nx2 = int(meshblock.get("nx2", args.nx2))
            mb_nx3 = int(meshblock.get("nx3", args.nx3))
            if mb_nx1 != args.nx1 or mb_nx2 != args.nx2 or mb_nx3 != args.nx3:
                raise SystemExit("This tool assumes a single MeshBlock; set meshblock sizes = mesh sizes.")

    for name in ("nx1", "nx2", "x1min", "x1max", "x2min", "x2max"):
        if getattr(args, name) is None:
            raise SystemExit(f"Missing required mesh parameter: {name}")

    if args.x2min != 0.0:
        raise SystemExit("x2min must be 0.0 for axisymmetric GS initialization.")

    gamma = args.gamma if args.gamma is not None else 5.0 / 3.0
    nx1 = int(args.nx1)
    nx2 = int(args.nx2)
    nx3 = int(args.nx3)
    x1_faces = np.linspace(args.x1min, args.x1max, nx1 + 1)
    x2_faces = np.linspace(args.x2min, args.x2max, nx2 + 1)
    x1_centers = 0.5 * (x1_faces[:-1] + x1_faces[1:])
    x2_centers = 0.5 * (x2_faces[:-1] + x2_faces[1:])

    psi_unit, r, z, _, _ = solve_gs_unit(
        nx2, nx1, args.x2min, args.x2max, args.x1min, args.x1max,
        args.omega, args.max_iter, args.tol
    )

    if args.pair_separation > 0.0:
        psi_left = shift_z(psi_unit, z, -0.5 * args.pair_separation)
        psi_right = shift_z(psi_unit, z, 0.5 * args.pair_separation)
        psi_unit = psi_left + psi_right

    if np.min(psi_unit) >= 0.0:
        psi_unit = -psi_unit

    B1_face, B2_face = compute_b_faces(psi_unit, x2_centers, x1_centers, x2_faces, x1_faces)
    B1_face = np.asarray(B1_face, dtype=np.float64)
    B2_face = np.asarray(B2_face, dtype=np.float64)

    mid_z = nx1 // 2
    b_unit = float(np.abs(B1_face[-1, mid_z]))
    if b_unit <= 0.0:
        raise SystemExit("Failed to compute non-zero boundary B; check geometry.")

    scale = args.b_bias / b_unit
    psi = scale * psi_unit
    B1_face *= scale
    B2_face *= scale

    B1_cc = 0.5 * (B1_face[:, :-1] + B1_face[:, 1:])
    B2_cc = 0.5 * (B2_face[:-1, :] + B2_face[1:, :])
    b_toroidal = args.b_toroidal if args.b_toroidal is not None else args.b_bias
    B3_cc = np.full_like(B1_cc, b_toroidal)

    p_sep = args.beta_s * args.b_bias * args.b_bias / (2.0 * MU0)
    p_prime = -scale
    p = p_sep + p_prime * psi
    if args.p_floor > 0.0:
        p = np.maximum(p, args.p_floor)

    psi_axis = float(np.min(psi))
    if psi_axis == 0.0:
        raise SystemExit("Degenerate psi field; adjust parameters.")
    psi_norm = np.clip(psi / psi_axis, 0.0, 1.0)

    Ti_K = args.Ti_eV * 11604.518
    Te_K = Ti_K * args.Te_Ti
    if args.rho_sep is not None and args.rho_axis is not None:
        rho = args.rho_sep + (args.rho_axis - args.rho_sep) * psi_norm
    else:
        n = p / (K_B * (Ti_K + Te_K))
        rho = n * args.amu * M_U
    if args.rho_floor > 0.0:
        rho = np.maximum(rho, args.rho_floor)

    z_sign = np.sign(x1_centers)
    v1_profile = (-args.v_inflow * z_sign)[None, :] * (psi_norm ** args.v_profile_power)
    v2_profile = np.zeros_like(v1_profile)
    v3_profile = np.zeros_like(v1_profile)

    ke = 0.5 * rho * (v1_profile * v1_profile + v2_profile * v2_profile + v3_profile * v3_profile)
    mag = 0.5 * (B1_cc * B1_cc + B2_cc * B2_cc + B3_cc * B3_cc)
    etot = p / (gamma - 1.0) + ke + mag

    rho_3d = np.repeat(rho[None, :, :], nx3, axis=0)
    v1_3d = np.repeat(v1_profile[None, :, :], nx3, axis=0)
    v2_3d = np.repeat(v2_profile[None, :, :], nx3, axis=0)
    v3_3d = np.repeat(v3_profile[None, :, :], nx3, axis=0)
    etot_3d = np.repeat(etot[None, :, :], nx3, axis=0)

    cons = np.zeros((5, 1, nx3, nx2, nx1), dtype=np.float64)
    cons[0, 0] = rho_3d
    cons[1, 0] = rho_3d * v1_3d
    cons[2, 0] = rho_3d * v2_3d
    cons[3, 0] = rho_3d * v3_3d
    cons[4, 0] = etot_3d

    b1 = np.repeat(B1_face[None, :, :], nx3, axis=0)[None, ...]
    b2 = np.repeat(B2_face[None, :, :], nx3, axis=0)[None, ...]
    b3 = np.full((1, nx3 + 1, nx2, nx1), b_toroidal, dtype=np.float64)

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_h5, "w") as h5f:
        h5f.create_dataset("cons", data=cons)
        h5f.create_dataset("b1", data=b1)
        h5f.create_dataset("b2", data=b2)
        h5f.create_dataset("b3", data=b3)

    print(
        f"[gs-athena] wrote {args.output_h5} with cons/b1/b2/b3 "
        f"(b_bias={args.b_bias} T, b_toroidal={b_toroidal} T, beta_s={args.beta_s})"
    )


if __name__ == "__main__":
    main()
