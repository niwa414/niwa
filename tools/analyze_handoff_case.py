#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def add_athena_vis_path(path_override: str | None) -> None:
    if path_override:
        sys.path.append(path_override)
        return
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "athena-24.0" / "vis" / "python",
        repo_root / "athena-public-version-21.0" / "vis" / "python",
    ]
    env_path = os.environ.get("ATHENA_VIS_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    for path in candidates:
        if path.exists():
            sys.path.append(str(path))
            return


def load_athena_read(path_override: str | None):
    add_athena_vis_path(path_override)
    try:
        import athena_read  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"athena_read not found. Set ATHENA_VIS_PATH or use athena-24.0/vis/python: {exc}"
        )
    return athena_read


def _fold_half_grid(values: np.ndarray, xc: np.ndarray, flip_sign: bool):
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
    nr_new = len(r_new)
    nz_new = len(z_new)
    temp = np.empty((len(r_old), nz_new), dtype=np.float64)
    for i, r in enumerate(r_old):
        temp[i, :] = np.interp(z_new, z_old, arr[i, :], left=arr[i, 0], right=arr[i, -1])
    out = np.empty((nr_new, nz_new), dtype=np.float64)
    for j, z in enumerate(z_new):
        out[:, j] = np.interp(r_new, r_old, temp[:, j], left=temp[0, j], right=temp[-1, j])
    return out


def read_grid_attrs(dataset, base):
    spacing = dataset.attrs.get("gridSpacing", None)
    if spacing is None:
        spacing = base.attrs.get("gridSpacing", None)
    offset = dataset.attrs.get("gridGlobalOffset", None)
    if offset is None:
        offset = base.attrs.get("gridGlobalOffset", None)
    if spacing is None or offset is None:
        return None, None
    spacing = np.asarray(spacing, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if spacing.size >= 2:
        spacing = spacing[-2:]
    if offset.size >= 2:
        offset = offset[-2:]
    return spacing, offset


def load_openpmd_fluid(path: Path):
    with h5py.File(path, "r") as h5f:
        base = h5f["/data/0/meshes/fluid"]
        if "rho" not in base:
            raise ValueError("fluid/rho missing in openPMD file")
        rho = np.asarray(base["rho"], dtype=float)
        if rho.ndim == 3:
            rho = rho[0]
        spacing, offset = read_grid_attrs(base["rho"], base)
    return rho, spacing, offset


def load_openpmd_B(path: Path):
    with h5py.File(path, "r") as h5f:
        base = h5f["/data/0/meshes/B"]
        for comp in ("r", "t", "z"):
            if comp not in base:
                raise ValueError(f"B/{comp} missing in openPMD file")
        Br = np.asarray(base["r"], dtype=float)
        Bt = np.asarray(base["t"], dtype=float)
        Bz = np.asarray(base["z"], dtype=float)
        if Br.ndim == 3:
            Br = Br[0]
        if Bt.ndim == 3:
            Bt = Bt[0]
        if Bz.ndim == 3:
            Bz = Bz[0]
        spacing, offset = read_grid_attrs(base["r"], base)
    return Br, Bt, Bz, spacing, offset


def map_vtk_to_rz(
    vtk_path: Path,
    axis_mode: str,
    fold_r: bool,
    resample: bool,
    grid,
    athena_vis_path: str | None,
):
    athena_read = load_athena_read(athena_vis_path)
    x_faces, y_faces, z_faces, data = athena_read.vtk(str(vtk_path))
    rho_in = data.get("rho")
    bcc_in = data.get("Bcc") if "Bcc" in data else data.get("b")
    if rho_in is None or bcc_in is None:
        raise SystemExit("VTK missing rho or Bcc field.")

    axis_mode = axis_mode.lower()
    if axis_mode not in {"x_z_y_r", "x_r_y_z"}:
        raise SystemExit(f"Unsupported axis-mode '{axis_mode}'. Use x_z_y_r or x_r_y_z.")

    bcc_raw = bcc_in[0]
    if axis_mode == "x_z_y_r":
        rho_raw = rho_in[0]
        Br_raw = bcc_raw[:, :, 1]
        Bz_raw = bcc_raw[:, :, 0]
        Bt_raw = bcc_raw[:, :, 2]
        r_faces = y_faces
        z_faces_in = x_faces
    else:
        rho_raw = rho_in[0].T
        Br_raw = bcc_raw[:, :, 0].T
        Bz_raw = bcc_raw[:, :, 1].T
        Bt_raw = bcc_raw[:, :, 2].T
        r_faces = x_faces
        z_faces_in = y_faces

    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces_in[:-1] + z_faces_in[1:])

    if fold_r and np.any(r_centers < 0.0):
        (rho_raw, Br_raw, Bz_raw, Bt_raw), r_centers = _fold_fields(
            [rho_raw, Br_raw, Bz_raw, Bt_raw],
            r_centers,
            [False, True, False, False],
        )

    r_old = np.array(r_centers, dtype=np.float64)
    z_old = np.array(z_centers, dtype=np.float64)
    r_new, z_new, nr, nz, r_min, r_max, z_min, z_max = grid
    if resample or rho_raw.shape != (nr, nz):
        rho = _resample_to_grid(rho_raw, r_old, z_old, r_new, z_new)
        Br = _resample_to_grid(Br_raw, r_old, z_old, r_new, z_new)
        Bz = _resample_to_grid(Bz_raw, r_old, z_old, r_new, z_new)
        Bt = _resample_to_grid(Bt_raw, r_old, z_old, r_new, z_new)
    else:
        rho, Br, Bz, Bt = rho_raw, Br_raw, Bz_raw, Bt_raw

    return rho, Br, Bt, Bz


def axisymmetric_mass(rho: np.ndarray, r_centers: np.ndarray, dr: float, dz: float) -> float:
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
    return float(np.sum(rho * volume))


def axisymmetric_mag_energy(Br: np.ndarray, Bt: np.ndarray, Bz: np.ndarray, r_centers: np.ndarray, dr: float, dz: float) -> float:
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
    b2 = Br * Br + Bt * Bt + Bz * Bz
    return float(0.5 * np.sum(b2 * volume))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MHD -> Hybrid handoff consistency.")
    parser.add_argument("--handoff-meta", required=True, help="handoff_meta.json from prepare step.")
    parser.add_argument("--warpx-metrics", help="WarpX metrics JSON to merge.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output mapping summary JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    meta_path = Path(args.handoff_meta)
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    vtk_path = Path(meta["vtk_path"])
    fluid_path = Path(meta["fluid_path"])
    b_path = Path(meta["b_path"])
    axis_mode = meta.get("axis_mode", "x_z_y_r")
    fold_r = bool(meta.get("fold_r", True))
    resample = bool(meta.get("resample", False))
    athena_vis_path = meta.get("athena_vis_path")
    rho_scale = float(meta.get("rho_scale", 1.0))
    b_scale = float(meta.get("B_scale", 1.0))

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    opmd_exists = fluid_path.exists() and b_path.exists()
    opmd_fields_present = False
    opmd_no_nan = False
    mass_rel_diff = None
    mag_energy_rel_diff = None

    metrics = {
        "opmd_exists": opmd_exists,
        "opmd_fields_present": False,
        "opmd_no_nan": False,
        "mass_athena": None,
        "mass_opmd": None,
        "mass_rel_diff": None,
        "mag_energy_athena": None,
        "mag_energy_opmd": None,
        "mag_energy_rel_diff": None,
        "vtk_path": str(vtk_path),
        "fluid_path": str(fluid_path),
        "b_path": str(b_path),
    }

    if opmd_exists:
        try:
            rho_opmd, spacing, offset = load_openpmd_fluid(fluid_path)
            Br_opmd, Bt_opmd, Bz_opmd, spacing_b, offset_b = load_openpmd_B(b_path)
            if spacing is None or offset is None:
                raise ValueError("openPMD grid metadata missing (gridSpacing/gridGlobalOffset).")
            dr, dz = float(spacing[0]), float(spacing[1])
            r0, z0 = float(offset[0]), float(offset[1])
            nr, nz = rho_opmd.shape
            r_centers = r0 + (np.arange(nr) + 0.5) * dr
            z_centers = z0 + (np.arange(nz) + 0.5) * dz
            grid = (r_centers, z_centers, nr, nz, r0, r0 + dr * nr, z0, z0 + dz * nz)

            rho_ath, Br_ath, Bt_ath, Bz_ath = map_vtk_to_rz(
                vtk_path,
                axis_mode=axis_mode,
                fold_r=fold_r,
                resample=resample,
                grid=grid,
                athena_vis_path=athena_vis_path,
            )

            rho_ath = rho_scale * rho_ath
            Br_ath = b_scale * Br_ath
            Bt_ath = b_scale * Bt_ath
            Bz_ath = b_scale * Bz_ath

            mass_ath = axisymmetric_mass(rho_ath, r_centers, dr, dz)
            mass_opmd = axisymmetric_mass(rho_opmd, r_centers, dr, dz)
            mag_energy_ath = axisymmetric_mag_energy(Br_ath, Bt_ath, Bz_ath, r_centers, dr, dz)
            mag_energy_opmd = axisymmetric_mag_energy(Br_opmd, Bt_opmd, Bz_opmd, r_centers, dr, dz)

            denom_mass = max(abs(mass_ath), 1.0e-30)
            denom_mag = max(abs(mag_energy_ath), 1.0e-30)
            mass_rel_diff = float(abs(mass_opmd - mass_ath) / denom_mass)
            mag_energy_rel_diff = float(abs(mag_energy_opmd - mag_energy_ath) / denom_mag)

            opmd_fields_present = True
            opmd_no_nan = not (
                np.isnan(rho_opmd).any()
                or np.isnan(Br_opmd).any()
                or np.isnan(Bt_opmd).any()
                or np.isnan(Bz_opmd).any()
            )

            metrics.update(
                {
                    "opmd_fields_present": opmd_fields_present,
                    "opmd_no_nan": opmd_no_nan,
                    "mass_athena": mass_ath,
                    "mass_opmd": mass_opmd,
                    "mass_rel_diff": mass_rel_diff,
                    "mag_energy_athena": mag_energy_ath,
                    "mag_energy_opmd": mag_energy_opmd,
                    "mag_energy_rel_diff": mag_energy_rel_diff,
                }
            )

            # Plots: mass/energy comparison
            labels = ["mass", "mag_energy"]
            athena_vals = [mass_ath, mag_energy_ath]
            opmd_vals = [mass_opmd, mag_energy_opmd]
            x = np.arange(len(labels))
            width = 0.35
            plt.figure(figsize=(6, 4))
            plt.bar(x - width / 2, athena_vals, width, label="Athena")
            plt.bar(x + width / 2, opmd_vals, width, label="openPMD")
            plt.xticks(x, labels)
            plt.ylabel("Integrated value")
            plt.title("Mapping: Global Mass & Magnetic Energy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "mapping_mass_energy_compare.png")
            plt.close()

            # Field snapshot overlay (rho)
            r_edges = r0 + np.arange(nr + 1) * dr
            z_edges = z0 + np.arange(nz + 1) * dz
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
            im0 = axes[0].pcolormesh(z_edges, r_edges, rho_ath, shading="auto")
            axes[0].set_title("Athena rho (mapped)")
            axes[0].set_xlabel("z")
            axes[0].set_ylabel("r")
            fig.colorbar(im0, ax=axes[0], shrink=0.8)
            im1 = axes[1].pcolormesh(z_edges, r_edges, rho_opmd, shading="auto")
            axes[1].set_title("openPMD rho")
            axes[1].set_xlabel("z")
            fig.colorbar(im1, ax=axes[1], shrink=0.8)
            plt.tight_layout()
            plt.savefig(plots_dir / "field_snapshot_overlay.png")
            plt.close()
        except Exception as exc:
            metrics["opmd_error"] = str(exc)

    if args.warpx_metrics:
        with Path(args.warpx_metrics).open("r", encoding="utf-8") as handle:
            warpx_metrics = json.load(handle)
        for key, value in warpx_metrics.items():
            metrics[f"warpx_{key}"] = value

    summary = {
        "vtk_path": str(vtk_path),
        "fluid_path": str(fluid_path),
        "b_path": str(b_path),
        "axis_mode": axis_mode,
        "fold_r": fold_r,
        "resample": resample,
        "rho_scale": rho_scale,
        "B_scale": b_scale,
        "opmd_exists": opmd_exists,
        "opmd_fields_present": opmd_fields_present,
        "opmd_no_nan": opmd_no_nan,
        "mass_rel_diff": mass_rel_diff,
        "mag_energy_rel_diff": mag_energy_rel_diff,
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
