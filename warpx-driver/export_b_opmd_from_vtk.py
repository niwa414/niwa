#!/usr/bin/env python3
"""
Export a 2D Athena++ VTK snapshot to an openPMD B-field file (thetaMode) with explicit geometry.

New features:
- Geometry explicitly controlled via CLI: --nr/--nz --r-min/--r-max --z-min/--z-max
- Modes:
    * from-vtk (default): read Athena++ VTK, map Athena x1->z, x2->r (suitable for ipa/belova inputs)
    * uniform: generate a uniform Bz test field with the same geometry
    * mirror: generate a divergence-free mirror field Bz(z) with consistent Br from A_theta
    * mirror-delta: mirror field minus uniform (Bz(center)=0), useful for ramping mirror ratio
- Metadata (gridSpacing, gridGlobalOffset) is written so the file passes validate_b_field_file().
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import openpmd_api as io


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
    neg_vals = values[:, neg_mask][:, ::-1]
    if flip_sign:
        neg_vals = -neg_vals

    if pos_vals.shape[1] == neg_vals.shape[1]:
        folded = 0.5 * (pos_vals + neg_vals)
    else:
        folded = np.empty((values.shape[0], pos_vals.shape[1]), dtype=values.dtype)
        folded[:, 0] = pos_vals[:, 0]
        folded[:, 1:] = 0.5 * (pos_vals[:, 1:] + neg_vals)

    r_centers = np.abs(xc[pos_mask])
    return folded, r_centers


def _fold_fields(fields, r_coords, flip_signs):
    """Fold a list of (nr, nz) arrays across r=0 using the same coordinates."""
    if len(fields) != len(flip_signs):
        raise ValueError("fields and flip_signs must have the same length")
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


def write_openpmd_B(out_path: Path, Br: np.ndarray, Bz: np.ndarray, Bt: np.ndarray, r_min: float, r_max: float, z_min: float, z_max: float):
    """Write thetaMode openPMD B field with shape (1, nr, nz) and proper metadata; return summary."""
    nr, nz = Br.shape  # Br and Bz expected as (nr, nz)
    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz

    # openPMD expects (theta, r, z)
    shape = (1, nr, nz)
    Br_3d = Br.reshape(shape)
    Bt_3d = Bt.reshape(shape)
    Bz_3d = Bz.reshape(shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = io.Series(str(out_path), io.Access.create)
    iteration = series.iterations[0]
    mesh_B = iteration.meshes["B"]
    mesh_B.set_geometry(io.Geometry.thetaMode)
    mesh_B.set_attribute("dataOrder", "C")
    mesh_B.set_axis_labels(["r", "z"])
    mesh_B.set_grid_spacing([dr, dz])
    mesh_B.set_grid_global_offset([r_min, z_min])
    mesh_B.unit_dimension = {io.Unit_Dimension.L: -1, io.Unit_Dimension.T: 0, io.Unit_Dimension.M: 0}

    # Store components; also write spacing/offset on each component for robustness
    for comp, arr in (("r", Br_3d), ("t", Bt_3d), ("z", Bz_3d)):
        rc = mesh_B[comp]
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
        "Br_min": float(np.min(Br)),
        "Br_max": float(np.max(Br)),
        "Bt_min": float(np.min(Bt)),
        "Bt_max": float(np.max(Bt)),
        "Bz_min": float(np.min(Bz)),
        "Bz_max": float(np.max(Bz)),
    }


def export_from_vtk(
    vtk_path: Path,
    out_path: Path,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    athena_vis_path: str | None,
    axis_mode: str,
    fold_r: bool,
    resample: bool,
    B_scale: float,
):
    athena_read = load_athena_read(athena_vis_path)
    x_faces, y_faces, z_faces, data = athena_read.vtk(str(vtk_path))

    B = data.get("Bcc") if "Bcc" in data else data.get("b")
    if B is None:
        raise SystemExit("VTK missing Bcc/b field.")

    axis_mode = axis_mode.lower()
    if axis_mode not in {"x_z_y_r", "x_r_y_z"}:
        raise SystemExit(f"Unsupported axis-mode '{axis_mode}'. Use x_z_y_r or x_r_y_z.")

    # Athena++ VTK: (nz, ny, nx, 3) -> here nz=1 for 2D, so take [0]
    if axis_mode == "x_z_y_r":
        # x1 -> z (axial), x2 -> r (radial)
        Br_raw = B[0, :, :, 1]  # By -> Br
        Bz_raw = B[0, :, :, 0]  # Bx -> Bz
        Bt_raw = B[0, :, :, 2]  # Bz -> Bt (toroidal)
        r_faces = y_faces
        z_faces_in = x_faces
        Br_fold = Br_raw
        Bz_fold = Bz_raw
        Bt_fold = Bt_raw
    else:
        # legacy: x1 -> r, x2 -> z
        Br_raw = B[0, :, :, 0]
        Bz_raw = B[0, :, :, 1]
        Bt_raw = B[0, :, :, 2]
        r_faces = x_faces
        z_faces_in = y_faces
        Br_fold = Br_raw.T  # (nr_old, nz_old)
        Bz_fold = Bz_raw.T
        Bt_fold = Bt_raw.T

    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces_in[:-1] + z_faces_in[1:])

    if fold_r and np.any(r_centers < 0.0):
        (Br_fold, Bz_fold, Bt_fold), r_centers = _fold_fields(
            [Br_fold, Bz_fold, Bt_fold],
            r_centers,
            [True, False, False],
        )

    r_old = np.array(r_centers, dtype=np.float64)
    z_old = np.array(z_centers, dtype=np.float64)
    r_new = np.linspace(r_min, r_max, nr, endpoint=False) + 0.5 * (r_max - r_min) / nr
    z_new = np.linspace(z_min, z_max, nz, endpoint=False) + 0.5 * (z_max - z_min) / nz

    if resample or Br_fold.shape != (nr, nz):
        Br = _resample_to_grid(Br_fold, r_old, z_old, r_new, z_new)
        Bz = _resample_to_grid(Bz_fold, r_old, z_old, r_new, z_new)
        Bt = _resample_to_grid(Bt_fold, r_old, z_old, r_new, z_new)
    else:
        if Br_fold.shape != (nr, nz):
            raise SystemExit(
                f"Grid mismatch: VTK (nr={Br_fold.shape[0]}, nz={Br_fold.shape[1]}) vs requested (nr={nr}, nz={nz}). "
                "Enable --resample or adjust --nr/--nz."
            )
        Br, Bz, Bt = Br_fold, Bz_fold, Bt_fold

    if B_scale != 1.0:
        Br = B_scale * Br
        Bz = B_scale * Bz
        Bt = B_scale * Bt

    summary = write_openpmd_B(out_path, Br, Bz, Bt, r_min, r_max, z_min, z_max)
    return summary


def export_uniform(out_path: Path, nr: int, nz: int, r_min: float, r_max: float, z_min: float, z_max: float, Bz_const: float):
    Br = np.zeros((nr, nz), dtype=np.float64)
    Bz = np.full((nr, nz), Bz_const, dtype=np.float64)
    Bt = np.zeros_like(Br)
    summary = write_openpmd_B(out_path, Br, Bz, Bt, r_min, r_max, z_min, z_max)
    return summary


def export_mirror(
    out_path: Path,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    Bz_center: float,
    mirror_ratio: float,
    mirror_center_z: float,
    mirror_half_length: float,
    clamp: bool,
):
    """
    Generate an axisymmetric mirror field using an azimuthal vector potential A_theta.

    We define:
      Bz(z) = Bz_center * [ 1 + (mirror_ratio - 1) * ((z - zc)/L)^2 ]
    and choose:
      A_theta(r,z) = 0.5 * r * Bz(z)
    which yields a divergence-free poloidal field:
      Bz = (1/r) d(r A_theta)/dr = Bz(z)
      Br = - dA_theta/dz = -0.5 * r * dBz/dz

    This field is useful as a proxy for end-mirror compression and is compatible with
    induced-E injection via E_theta = -dA_theta/dt in the WarpX driver.
    """
    if mirror_half_length <= 0.0:
        raise SystemExit("--mirror-half-length must be > 0")
    if mirror_ratio < 1.0:
        raise SystemExit("--mirror-ratio must be >= 1")

    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    r = r_min + (np.arange(nr, dtype=np.float64) + 0.5) * dr
    z = z_min + (np.arange(nz, dtype=np.float64) + 0.5) * dz

    xi = (z - mirror_center_z) / mirror_half_length
    if clamp:
        xi = np.clip(xi, -1.0, 1.0)

    bz_z = Bz_center * (1.0 + (mirror_ratio - 1.0) * xi * xi)
    dBz_dz = (
        Bz_center
        * (mirror_ratio - 1.0)
        * (2.0 * (z - mirror_center_z) / (mirror_half_length**2))
    )
    if clamp:
        dBz_dz = np.where(
            np.abs(z - mirror_center_z) <= mirror_half_length, dBz_dz, 0.0
        )

    Bz = np.broadcast_to(bz_z[None, :], (nr, nz)).copy()
    Br = (-0.5 * r[:, None]) * dBz_dz[None, :]
    Bt = np.zeros_like(Bz)

    summary = write_openpmd_B(out_path, Br, Bz, Bt, r_min, r_max, z_min, z_max)
    summary["profile"] = {
        "type": "mirror",
        "Bz_center": float(Bz_center),
        "mirror_ratio": float(mirror_ratio),
        "mirror_center_z": float(mirror_center_z),
        "mirror_half_length": float(mirror_half_length),
        "clamp": bool(clamp),
    }
    return summary


def export_mirror_delta(
    out_path: Path,
    nr: int,
    nz: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    Bz_center: float,
    mirror_ratio: float,
    mirror_center_z: float,
    mirror_half_length: float,
    clamp: bool,
):
    """
    Generate a "mirror-delta" basis field: B_mirror - B_uniform.

    With the same polynomial mirror profile as export_mirror, this produces:
      Bz_delta(center) = 0
      Bz_delta(ends)   = Bz_center * (mirror_ratio - 1)

    This makes it easy to ramp the *mirror ratio* while keeping a constant bias field:
      B_total = B_bias + c(t) * Bz_delta
    so choosing c_final = B_bias yields mirror ratio ~ mirror_ratio.
    """
    if mirror_half_length <= 0.0:
        raise SystemExit("--mirror-half-length must be > 0")
    if mirror_ratio < 1.0:
        raise SystemExit("--mirror-ratio must be >= 1")

    dr = (r_max - r_min) / nr
    dz = (z_max - z_min) / nz
    r = r_min + (np.arange(nr, dtype=np.float64) + 0.5) * dr
    z = z_min + (np.arange(nz, dtype=np.float64) + 0.5) * dz

    xi = (z - mirror_center_z) / mirror_half_length
    if clamp:
        xi = np.clip(xi, -1.0, 1.0)

    # Mirror minus uniform => drop the "+1" term.
    bz_z = Bz_center * (mirror_ratio - 1.0) * (xi * xi)
    dBz_dz = (
        Bz_center
        * (mirror_ratio - 1.0)
        * (2.0 * (z - mirror_center_z) / (mirror_half_length**2))
    )
    if clamp:
        dBz_dz = np.where(
            np.abs(z - mirror_center_z) <= mirror_half_length, dBz_dz, 0.0
        )

    Bz = np.broadcast_to(bz_z[None, :], (nr, nz)).copy()
    Br = (-0.5 * r[:, None]) * dBz_dz[None, :]
    Bt = np.zeros_like(Bz)

    summary = write_openpmd_B(out_path, Br, Bz, Bt, r_min, r_max, z_min, z_max)
    summary["profile"] = {
        "type": "mirror-delta",
        "Bz_center": float(Bz_center),
        "mirror_ratio": float(mirror_ratio),
        "mirror_center_z": float(mirror_center_z),
        "mirror_half_length": float(mirror_half_length),
        "clamp": bool(clamp),
    }
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Convert Athena VTK to openPMD B field (thetaMode) with explicit geometry.")
    p.add_argument("--mode", choices=["from-vtk", "uniform", "mirror", "mirror-delta"], default="from-vtk")
    p.add_argument("--input-vtk", help="Input Athena++ VTK file (required for from-vtk).")
    p.add_argument("--output-bfile", default="B_ext.h5", help="Output openPMD B file.")
    p.add_argument("--athena-vis-path", help="Optional path to athena_read module.")
    p.add_argument("--nr", type=int, default=32, help="Number of r cells.")
    p.add_argument("--nz", type=int, default=64, help="Number of z cells.")
    p.add_argument("--r-min", type=float, default=0.0, help="Minimum r.")
    p.add_argument("--r-max", type=float, default=0.1, help="Maximum r.")
    p.add_argument("--z-min", type=float, default=-0.1, help="Minimum z.")
    p.add_argument("--z-max", type=float, default=0.1, help="Maximum z.")
    p.add_argument("--Bz-const", type=float, default=0.05, help="Uniform Bz for uniform mode.")
    p.add_argument("--Bz-center", type=float, default=0.05, help="On-axis Bz at z=mirror-center-z for mirror mode.")
    p.add_argument("--mirror-ratio", type=float, default=1.5, help="Mirror ratio Bz(|z-zc|=L)/Bz(zc) for mirror mode.")
    p.add_argument("--mirror-center-z", type=float, default=0.0, help="Center z position for mirror mode.")
    p.add_argument(
        "--mirror-half-length",
        type=float,
        default=None,
        help="Half-length L for mirror profile (default: domain half-length).",
    )
    p.add_argument(
        "--mirror-clamp",
        action="store_true",
        default=True,
        help="Clamp mirror profile to |z-zc|<=L (default: on).",
    )
    p.add_argument(
        "--no-mirror-clamp",
        action="store_false",
        dest="mirror_clamp",
        help="Disable clamping for mirror profile (extends parabola beyond L).",
    )
    p.add_argument(
        "--axis-mode",
        choices=["x_z_y_r", "x_r_y_z"],
        default="x_z_y_r",
        help="Map Athena axes to (z,r). Default x1->z, x2->r (ipa/belova). Use x_r_y_z for legacy x1->r mapping.",
    )
    p.add_argument("--fold-r", action="store_true", help="Fold negative r to r>=0 (averages symmetric halves).")
    p.add_argument("--B-scale", type=float, default=1.0, help="Scale factor applied to B components before writing.")
    p.add_argument("--no-fold-r", action="store_false", dest="fold_r")
    p.add_argument("--resample", action="store_true", default=True, help="Resample data onto requested grid if dimensions differ.")
    p.add_argument("--no-resample", action="store_false", dest="resample")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output_bfile)

    if args.mode == "from-vtk":
        if not args.input_vtk:
            raise SystemExit("from-vtk mode requires --input-vtk")
        vtk_path = Path(args.input_vtk)
        if not vtk_path.exists():
            raise SystemExit(f"VTK file not found: {vtk_path}")
        print(f"[from-vtk] Writing {out_path} with geometry nr={args.nr}, nz={args.nz}, "
              f"r=[{args.r_min},{args.r_max}], z=[{args.z_min},{args.z_max}]")
        summary = export_from_vtk(
            vtk_path=vtk_path,
            out_path=out_path,
            nr=args.nr,
            nz=args.nz,
            r_min=args.r_min,
            r_max=args.r_max,
            z_min=args.z_min,
            z_max=args.z_max,
            athena_vis_path=args.athena_vis_path,
            axis_mode=args.axis_mode,
            fold_r=args.fold_r,
            resample=args.resample,
            B_scale=args.B_scale,
        )
    elif args.mode == "mirror":
        half_length = args.mirror_half_length
        if half_length is None:
            half_length = 0.5 * (args.z_max - args.z_min)
        print(
            f"[mirror] Writing {out_path} with geometry nr={args.nr}, nz={args.nz}, "
            f"r=[{args.r_min},{args.r_max}], z=[{args.z_min},{args.z_max}], "
            f"Bz_center={args.Bz_center}, mirror_ratio={args.mirror_ratio}, "
            f"zc={args.mirror_center_z}, L={half_length}"
        )
        summary = export_mirror(
            out_path,
            args.nr,
            args.nz,
            args.r_min,
            args.r_max,
            args.z_min,
            args.z_max,
            Bz_center=args.Bz_center,
            mirror_ratio=args.mirror_ratio,
            mirror_center_z=args.mirror_center_z,
            mirror_half_length=float(half_length),
            clamp=bool(args.mirror_clamp),
        )
    elif args.mode == "mirror-delta":
        half_length = args.mirror_half_length
        if half_length is None:
            half_length = 0.5 * (args.z_max - args.z_min)
        print(
            f"[mirror-delta] Writing {out_path} with geometry nr={args.nr}, nz={args.nz}, "
            f"r=[{args.r_min},{args.r_max}], z=[{args.z_min},{args.z_max}], "
            f"Bz_center={args.Bz_center}, mirror_ratio={args.mirror_ratio}, "
            f"zc={args.mirror_center_z}, L={half_length}"
        )
        summary = export_mirror_delta(
            out_path,
            args.nr,
            args.nz,
            args.r_min,
            args.r_max,
            args.z_min,
            args.z_max,
            Bz_center=args.Bz_center,
            mirror_ratio=args.mirror_ratio,
            mirror_center_z=args.mirror_center_z,
            mirror_half_length=float(half_length),
            clamp=bool(args.mirror_clamp),
        )
    else:
        print(f"[uniform] Writing {out_path} with geometry nr={args.nr}, nz={args.nz}, "
              f"r=[{args.r_min},{args.r_max}], z=[{args.z_min},{args.z_max}], Bz={args.Bz_const}")
        summary = export_uniform(out_path, args.nr, args.nz, args.r_min, args.r_max, args.z_min, args.z_max, args.Bz_const)

    print(
        f"[summary] wrote {summary['path']} nr={summary['nr']} nz={summary['nz']} "
        f"dr={summary['dr']:.4g} dz={summary['dz']:.4g} "
        f"r=[{summary['r_min']},{summary['r_max']}] z=[{summary['z_min']},{summary['z_max']}] "
        f"Br[min,max]=[{summary['Br_min']:.4g},{summary['Br_max']:.4g}] "
        f"Bt[min,max]=[{summary['Bt_min']:.4g},{summary['Bt_max']:.4g}] "
        f"Bz[min,max]=[{summary['Bz_min']:.4g},{summary['Bz_max']:.4g}]"
    )
    print("Done.")


if __name__ == "__main__":
    main()
