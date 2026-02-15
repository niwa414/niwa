#!/usr/bin/env python3
"""
Export a 2D Athena++ VTK snapshot to an openPMD fluid file (thetaMode).

Outputs rho, vr, vz, vphi, Ti, Te on a (theta, r, z) mesh with explicit
gridSpacing and gridGlobalOffset so the WarpX driver can sanity-check geometry.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import openpmd_api as io


M_U = 1.66053906660e-27  # atomic mass unit (kg)
K_B = 1.380649e-23  # Boltzmann constant (J/K)


def load_athena_read(path_override: str | None):
    if path_override:
        sys.path.append(path_override)
    else:
        sys.path.append("/Users/ni/Desktop/fusion/athena-24.0/vis/python/")
    try:
        import athena_read  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Failed to import athena_read: {exc}")
    return athena_read


def compute_temperatures(rho: np.ndarray, press: np.ndarray, amu: float, te_ratio: float, te_const: float | None):
    """Return (n, Ti, Te)."""
    n = np.zeros_like(rho)
    mask = rho > 0.0
    n[mask] = rho[mask] / (amu * M_U)

    Ti = np.zeros_like(rho)
    valid = mask & (n > 0.0)
    Ti[valid] = press[valid] / (n[valid] * K_B)

    if te_const is not None:
        Te = np.full_like(Ti, te_const)
    else:
        Te = te_ratio * Ti
    return n, Ti, Te


def write_openpmd_fluid(
    out_path: Path,
    rho: np.ndarray,
    vr: np.ndarray,
    vz: np.ndarray,
    vphi: np.ndarray,
    Ti: np.ndarray,
    Te: np.ndarray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
):
    """Write thetaMode openPMD fluid fields with shape (1, nr, nz); return summary."""
    nr, nz = rho.shape
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz

    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = io.Series(str(out_path), io.Access.create)
    iteration = series.iterations[0]
    mesh = iteration.meshes["fluid"]
    mesh.set_geometry(io.Geometry.thetaMode)
    mesh.set_attribute("dataOrder", "C")
    mesh.set_axis_labels(["r", "z"])
    mesh.set_grid_spacing([dr, dz])
    mesh.set_grid_global_offset([r_min, z_min])
    mesh.unit_dimension = {io.Unit_Dimension.L: 0, io.Unit_Dimension.T: 0, io.Unit_Dimension.M: 0}

    datasets = {
        "rho": rho.reshape((1, nr, nz)),
        "vr": vr.reshape((1, nr, nz)),
        "vz": vz.reshape((1, nr, nz)),
        "vphi": vphi.reshape((1, nr, nz)),
        "Ti": Ti.reshape((1, nr, nz)),
        "Te": Te.reshape((1, nr, nz)),
    }

    for name, arr in datasets.items():
        rc = mesh[name]
        rc.reset_dataset(io.Dataset(arr.dtype, arr.shape))
        rc.set_unit_SI(1.0)
        rc.store_chunk(np.ascontiguousarray(arr))
        rc.set_attribute("gridSpacing", np.array([dr, dz], dtype=np.float64))
        rc.set_attribute("gridGlobalOffset", np.array([r_min, z_min], dtype=np.float64))

    series.flush()
    series.close()
    return {
        "path": out_path,
        "nr": nr,
        "nz": nz,
        "dr": dr,
        "dz": dz,
        "r_min": r_min,
        "r_max": r_max,
        "z_min": z_min,
        "z_max": z_max,
        "rho_min": float(np.nanmin(rho)),
        "rho_max": float(np.nanmax(rho)),
        "vr_min": float(np.nanmin(vr)),
        "vr_max": float(np.nanmax(vr)),
        "vz_min": float(np.nanmin(vz)),
        "vz_max": float(np.nanmax(vz)),
        "vphi_min": float(np.nanmin(vphi)),
        "vphi_max": float(np.nanmax(vphi)),
        "Ti_min": float(np.nanmin(Ti)),
        "Ti_max": float(np.nanmax(Ti)),
        "Te_min": float(np.nanmin(Te)),
        "Te_max": float(np.nanmax(Te)),
    }


def _fold_half_grid(values: np.ndarray, xc: np.ndarray, flip_sign: bool):
    """
    Fold x<0, x>0 halves into r>=0 by averaging symmetric pairs.
    values shape: (nz, nx) or (nx,) broadcastable.
    xc: cell centers along x (length nx).
    flip_sign: whether to flip sign for the negative side before averaging (radial components).
    """
    pos_mask = xc >= 0.0
    neg_mask = xc < 0.0
    pos_vals = values[:, pos_mask]
    neg_vals = values[:, neg_mask]

    # Align neg side in increasing |x| order
    neg_vals = neg_vals[:, ::-1]
    if flip_sign:
        neg_vals = -neg_vals

    # If odd nx, pos may have one extra column at x=0; keep it as is
    if pos_vals.shape[1] == neg_vals.shape[1]:
        folded = 0.5 * (pos_vals + neg_vals)
    else:
        # x=0 column is the first in pos_vals when present
        folded = np.empty((values.shape[0], pos_vals.shape[1]), dtype=values.dtype)
        folded[:, 0] = pos_vals[:, 0]
        folded[:, 1:] = 0.5 * (pos_vals[:, 1:] + neg_vals)
    r_centers = np.abs(xc[pos_mask])
    return folded, r_centers


def _fold_fields(fields, r_coords, flip_signs):
    """Fold a list of (nr, nz) arrays across r=0 using the same coordinates."""
    if len(fields) != len(flip_signs):
        raise ValueError("fields and flip_signs must match lengths")
    if r_coords.min() >= 0.0:
        return fields, r_coords
    folded_fields = []
    r_new = None
    for arr, flip in zip(fields, flip_signs):
        folded, r_vals = _fold_half_grid(arr.T, r_coords, flip_sign=flip)
        folded_fields.append(folded.T)
        r_new = r_vals
    return folded_fields, r_new if r_new is not None else r_coords


def _resample_to_grid(arr: np.ndarray, r_old: np.ndarray, z_old: np.ndarray, r_new: np.ndarray, z_new: np.ndarray):
    """Bilinear resample arr(r_old,z_old) -> arr(r_new,z_new)."""
    nr_new = len(r_new)
    nz_new = len(z_new)
    temp = np.empty((len(r_old), nz_new), dtype=np.float64)
    for i, r in enumerate(r_old):
        temp[i, :] = np.interp(z_new, z_old, arr[i, :], left=arr[i, 0], right=arr[i, -1])
    out = np.empty((nr_new, nz_new), dtype=np.float64)
    for j, z in enumerate(z_new):
        out[:, j] = np.interp(r_new, r_old, temp[:, j], left=temp[0, j], right=temp[-1, j])
    return out


def export_from_vtk(
    vtk_path: Path,
    out_path: Path,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    amu: float,
    te_ratio: float,
    te_const: float | None,
    athena_vis_path: str | None,
    axis_mode: str,
    fold_r: bool,
    resample: bool,
    rho_scale: float,
    vel_scale: float,
    press_scale: float,
    b_scale: float,
    b_only_b: bool = False,
):
    athena_read = load_athena_read(athena_vis_path)
    x_faces, y_faces, z_faces, data = athena_read.vtk(str(vtk_path))

    axis_mode = axis_mode.lower()
    if axis_mode not in {"x_z_y_r", "x_r_y_z"}:
        raise SystemExit(f"Unsupported axis-mode '{axis_mode}'. Use x_z_y_r or x_r_y_z.")

    if b_only_b:
        bcc_in = data.get("bcc") or data.get("Bcc")
        if bcc_in is None:
            raise SystemExit("VTK missing field: bcc or Bcc")
        bcc_raw = bcc_in[0]  # (ny, nx, 3)

        if axis_mode == "x_z_y_r":
            Br_raw = bcc_raw[:, :, 1]
            Bz_raw = bcc_raw[:, :, 0]
            Bphi_raw = bcc_raw[:, :, 2]
            r_faces = y_faces
            z_faces_in = x_faces
        else:
            Br_raw = bcc_raw[:, :, 0].T
            Bz_raw = bcc_raw[:, :, 1].T
            Bphi_raw = bcc_raw[:, :, 2].T
            r_faces = x_faces
            z_faces_in = y_faces

        r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
        z_centers = 0.5 * (z_faces_in[:-1] + z_faces_in[1:])

        if fold_r and np.any(r_centers < 0.0):
            (Br_raw, Bz_raw, Bphi_raw), r_centers = _fold_fields(
                [Br_raw, Bz_raw, Bphi_raw],
                r_centers,
                [True, False, False],
            )

        r_old = np.array(r_centers, dtype=np.float64)
        z_old = np.array(z_centers, dtype=np.float64)
        r_new = np.linspace(r_min, r_max, nr, endpoint=False) + 0.5 * (r_max - r_min) / nr
        z_new = np.linspace(z_min, z_max, nz, endpoint=False) + 0.5 * (z_max - z_min) / nz

        if resample or Br_raw.shape != (nr, nz):
            Br = _resample_to_grid(Br_raw, r_old, z_old, r_new, z_new)
            Bz = _resample_to_grid(Bz_raw, r_old, z_old, r_new, z_new)
            Bphi = _resample_to_grid(Bphi_raw, r_old, z_old, r_new, z_new)
        else:
            if Br_raw.shape != (nr, nz):
                raise SystemExit(
                    f"Grid mismatch: VTK (nr={Br_raw.shape[0]}, nz={Br_raw.shape[1]}) vs requested (nr={nr}, nz={nz}). "
                    "Enable --resample or adjust --nr/--nz."
                )
            Br, Bz, Bphi = Br_raw, Bz_raw, Bphi_raw

        if b_scale != 1.0:
            Br = b_scale * Br
            Bz = b_scale * Bz
            Bphi = b_scale * Bphi

        import openpmd_api as io  # local import to keep base path intact
        series = io.Series(str(out_path), io.Access.create)
        it = series.iterations[0]
        it.time = 0.0
        it.time_unit_SI = 1.0

        mesh_node = it.meshes["B"]
        mesh_node.grid_spacing = [(r_max - r_min) / nr, (z_max - z_min) / nz]
        mesh_node.grid_global_offset = [r_min, z_min]
        mesh_node.axis_labels = ["r", "z"]
        mesh_node.geometry = io.Geometry.thetaMode
        mesh_node.unit_dimension = {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.L: 0,
            io.Unit_Dimension.T: -2,
            io.Unit_Dimension.I: -1,
        }

        fields = {"x": Br, "y": Bphi, "z": Bz}
        for comp, arr in fields.items():
            rec = mesh_node[comp]
            rec.unit_SI = 1.0
            rec.position = [0.0, 0.0]
            d = rec.reset_dataset(io.Dataset(arr.dtype, arr.shape))
            d.store_chunk(np.ascontiguousarray(arr))

        series.flush()
        del series

        return {
            "path": out_path,
            "nr": nr,
            "nz": nz,
            "dr": (r_max - r_min) / nr,
            "dz": (z_max - z_min) / nz,
            "r_min": r_min,
            "r_max": r_max,
            "z_min": z_min,
            "z_max": z_max,
            "rho_min": 0,
            "rho_max": 0,
            "vr_min": 0,
            "vr_max": 0,
            "vz_min": 0,
            "vz_max": 0,
            "vphi_min": 0,
            "vphi_max": 0,
            "Ti_min": 0,
            "Ti_max": 0,
            "Te_min": 0,
            "Te_max": 0,
        }

    rho_in = data.get("rho")
    vel_in = data.get("vel") if "vel" in data else data.get("v")
    press_in = data.get("press") if "press" in data else data.get("prs")
    if rho_in is None or vel_in is None or press_in is None:
        missing = [name for name, arr in (("rho", rho_in), ("vel", vel_in), ("press", press_in)) if arr is None]
        raise SystemExit(f"VTK missing fields: {', '.join(missing)}")

    # (nz, ny, nx, 3) -> take [0] for 2D
    vel_raw = vel_in[0]
    if axis_mode == "x_z_y_r":
        rho_raw = rho_in[0]
        vr_raw = vel_raw[:, :, 1]
        vz_raw = vel_raw[:, :, 0]
        vphi_raw = vel_raw[:, :, 2]
        press_raw = press_in[0]
        r_faces = y_faces
        z_faces_in = x_faces
    else:
        rho_raw = rho_in[0].T
        vr_raw = vel_raw[:, :, 0].T
        vz_raw = vel_raw[:, :, 1].T
        vphi_raw = vel_raw[:, :, 2].T
        press_raw = press_in[0].T
        r_faces = x_faces
        z_faces_in = y_faces

    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces_in[:-1] + z_faces_in[1:])

    if fold_r and np.any(r_centers < 0.0):
        (rho_raw, vr_raw, vz_raw, vphi_raw, press_raw), r_centers = _fold_fields(
            [rho_raw, vr_raw, vz_raw, vphi_raw, press_raw],
            r_centers,
            [False, True, False, False, False],
        )

    r_old = np.array(r_centers, dtype=np.float64)
    z_old = np.array(z_centers, dtype=np.float64)
    r_new = np.linspace(r_min, r_max, nr, endpoint=False) + 0.5 * (r_max - r_min) / nr
    z_new = np.linspace(z_min, z_max, nz, endpoint=False) + 0.5 * (z_max - z_min) / nz

    if resample or rho_raw.shape != (nr, nz):
        rho = _resample_to_grid(rho_raw, r_old, z_old, r_new, z_new)
        vr = _resample_to_grid(vr_raw, r_old, z_old, r_new, z_new)
        vz = _resample_to_grid(vz_raw, r_old, z_old, r_new, z_new)
        vphi = _resample_to_grid(vphi_raw, r_old, z_old, r_new, z_new)
        press = _resample_to_grid(press_raw, r_old, z_old, r_new, z_new)
    else:
        if rho_raw.shape != (nr, nz):
            raise SystemExit(
                f"Grid mismatch: VTK (nr={rho_raw.shape[0]}, nz={rho_raw.shape[1]}) vs requested (nr={nr}, nz={nz}). "
                "Enable --resample or adjust --nr/--nz."
            )
        rho, vr, vz, vphi, press = rho_raw, vr_raw, vz_raw, vphi_raw, press_raw

    if rho_scale != 1.0:
        rho = rho_scale * rho
    if vel_scale != 1.0:
        vr = vel_scale * vr
        vz = vel_scale * vz
        vphi = vel_scale * vphi
    if press_scale != 1.0:
        press = press_scale * press

    _, Ti, Te = compute_temperatures(rho, press, amu, te_ratio, te_const)
    summary = write_openpmd_fluid(out_path, rho, vr, vz, vphi, Ti, Te, r_min, r_max, z_min, z_max)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Convert Athena VTK to openPMD fluid file (thetaMode) with explicit geometry.")
    p.add_argument("--input-vtk", required=True, help="Input Athena++ VTK file.")
    p.add_argument("--output-fluid", default="fluid_init.h5", help="Output openPMD fluid file.")
    p.add_argument("--athena-vis-path", help="Optional path to athena_read module.")
    p.add_argument("--nr", type=int, default=32, help="Number of r cells.")
    p.add_argument("--nz", type=int, default=64, help="Number of z cells.")
    p.add_argument("--r-min", type=float, default=0.0, help="Minimum r.")
    p.add_argument("--r-max", type=float, default=0.1, help="Maximum r.")
    p.add_argument("--z-min", type=float, default=-0.1, help="Minimum z.")
    p.add_argument("--z-max", type=float, default=0.1, help="Maximum z.")
    p.add_argument("--amu", type=float, default=1.0, help="Ion atomic mass number A for density->n conversion.")
    p.add_argument("--Te-ratio", type=float, default=1.0, dest="te_ratio", help="Te = Te_ratio * Ti if Te-const not set.")
    p.add_argument("--Te-const", type=float, dest="te_const", help="Optional constant Te (K) to override Te_ratio.")
    p.add_argument(
        "--axis-mode",
        choices=["x_z_y_r", "x_r_y_z"],
        default="x_z_y_r",
        help="Map Athena axes to (z,r). Default x1->z, x2->r (ipa/belova). Use x_r_y_z for legacy x1->r mapping.",
    )
    p.add_argument("--fold-r", action="store_true", dest="fold_r", default=False, help="Fold negative r to r>=0 (averages symmetric halves).")
    p.add_argument("--no-fold-r", action="store_false", dest="fold_r")
    p.add_argument("--resample", action="store_true", default=True, help="Resample data onto requested grid if dimensions differ.")
    p.add_argument("--no-resample", action="store_false", dest="resample")
    p.add_argument("--rho-scale", type=float, default=1.0, help="Multiply rho by this factor before writing (e.g., kg/m^3 per code unit).")
    p.add_argument("--vel-scale", type=float, default=1.0, help="Multiply velocities by this factor before writing (e.g., m/s per code unit).")
    p.add_argument("--press-scale", type=float, default=1.0, help="Multiply pressure by this factor before temperature conversion.")
    p.add_argument("--B-scale", type=float, default=1.0, help="Scale factor applied to B components when --b-only-b is used.")
    # Back-compat aliases
    p.add_argument("--fold-x", action="store_true", dest="fold_r", help=argparse.SUPPRESS)
    p.add_argument("--no-fold-x", action="store_false", dest="fold_r", help=argparse.SUPPRESS)
    p.add_argument("--b-only-b", action="store_true", help="Only write B field components (for external B file).")
    return p.parse_args()


def main():
    args = parse_args()
    vtk_path = Path(args.input_vtk)
    if not vtk_path.exists():
        raise SystemExit(f"VTK file not found: {vtk_path}")

    out_path = Path(args.output_fluid)
    print(
        f"[fluid-from-vtk] Writing {out_path} with geometry nr={args.nr}, nz={args.nz}, "
        f"r=[{args.r_min},{args.r_max}], z=[{args.z_min},{args.z_max}]"
    )
    summary = export_from_vtk(
        vtk_path=vtk_path,
        out_path=out_path,
        nr=args.nr,
        nz=args.nz,
        r_min=args.r_min,
        r_max=args.r_max,
        z_min=args.z_min,
        z_max=args.z_max,
        amu=args.amu,
        te_ratio=args.te_ratio,
        te_const=args.te_const,
        athena_vis_path=args.athena_vis_path,
        axis_mode=args.axis_mode,
        fold_r=args.fold_r,
        resample=args.resample,
        rho_scale=args.rho_scale,
        vel_scale=args.vel_scale,
        press_scale=args.press_scale,
        b_scale=args.B_scale,
        b_only_b=args.b_only_b,
    )
    if args.b_only_b:
        print(f"[summary] wrote {summary['path']} (B-field only)")
        return

    print(
        f"[summary] wrote {summary['path']} nr={summary['nr']} nz={summary['nz']} "
        f"dr={summary['dr']:.4g} dz={summary['dz']:.4g} "
        f"r=[{summary['r_min']},{summary['r_max']}] z=[{summary['z_min']},{summary['z_max']}] "
        f"rho[min,max]=[{summary['rho_min']:.4g},{summary['rho_max']:.4g}] "
        f"vr[min,max]=[{summary['vr_min']:.4g},{summary['vr_max']:.4g}] "
        f"vz[min,max]=[{summary['vz_min']:.4g},{summary['vz_max']:.4g}] "
        f"vphi[min,max]=[{summary['vphi_min']:.4g},{summary['vphi_max']:.4g}] "
        f"Ti[min,max]=[{summary['Ti_min']:.4g},{summary['Ti_max']:.4g}] "
        f"Te[min,max]=[{summary['Te_min']:.4g},{summary['Te_max']:.4g}]"
    )
    print("Done.")


if __name__ == "__main__":
    main()
