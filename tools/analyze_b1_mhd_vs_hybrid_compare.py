#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

try:
    import yt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"yt required for WarpX 3D diag analysis: {exc}")

UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "magnetic_unit": (1.0, "T"),
}

Q_E = 1.602176634e-19
M_P = 1.67262192369e-27


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def parse_vtk_time(path: Path) -> float | None:
    try:
        with path.open("rb") as handle:
            for _ in range(3):
                line = handle.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore")
                if "time=" in text:
                    return float(text.split("time=")[1].split()[0])
    except Exception:
        return None
    return None


def list_vtk_files(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in Path().glob(pattern))


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

    return rho, Br, Bt, Bz, r_new, z_new


def axisymmetric_mass(rho: np.ndarray, r_centers: np.ndarray, dr: float, dz: float) -> float:
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
    return float(np.sum(rho * volume))


def axisymmetric_mag_energy(Br: np.ndarray, Bt: np.ndarray, Bz: np.ndarray, r_centers: np.ndarray, dr: float, dz: float) -> float:
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
    b2 = Br * Br + Bt * Bt + Bz * Bz
    return float(0.5 * np.sum(b2 * volume))


def axisymmetric_b_rms(Br: np.ndarray, Bt: np.ndarray, Bz: np.ndarray, r_centers: np.ndarray, dr: float, dz: float) -> float:
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz
    b2 = Br * Br + Bt * Bt + Bz * Bz
    denom = max(float(np.sum(volume)), 1.0e-30)
    return float(np.sqrt(np.sum(b2 * volume) / denom))


def list_diags(diag_root: Path) -> list[Path]:
    if not diag_root.exists():
        return []
    return sorted(
        [
            p
            for p in diag_root.iterdir()
            if p.is_dir() and p.name.startswith("diag") and "old" not in p.name
        ]
    )


def reshape_field(arr: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size == dims[0] * dims[1] * dims[2]:
        return arr.reshape(dims)
    if arr.shape == dims:
        return arr
    return arr


def compute_warpx_series(diag_dir: Path, ion_amu: float) -> dict:
    diags = list_diags(diag_dir)
    times = []
    mass_series = []
    b_rms_series = []
    mag_series = []
    ion_mass = ion_amu * M_P
    for diag in diags:
        ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        t = float(ds.current_time.to_value())
        dims = tuple(int(x) for x in ds.domain_dimensions)
        rho_w = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
        Bx_w = reshape_field(ad["boxlib", "Bx"].to_ndarray(), dims)
        By_w = reshape_field(ad["boxlib", "By"].to_ndarray(), dims)
        Bz_w = reshape_field(ad["boxlib", "Bz"].to_ndarray(), dims)
        if rho_w.ndim != 3:
            continue
        dx = float(ds.domain_width[0].to_value()) / dims[0]
        dy = float(ds.domain_width[1].to_value()) / dims[1]
        dz = float(ds.domain_width[2].to_value()) / dims[2]
        rho_mass = rho_w / Q_E * ion_mass
        mass_val = float(np.sum(rho_mass) * dx * dy * dz)
        b2 = Bx_w * Bx_w + By_w * By_w + Bz_w * Bz_w
        mag_energy = float(0.5 * np.sum(b2) * dx * dy * dz)
        b_rms = float(np.sqrt(np.mean(b2)))
        times.append(t)
        mass_series.append(mass_val)
        b_rms_series.append(b_rms)
        mag_series.append(mag_energy)
    return {
        "times": times,
        "mass": mass_series,
        "b_rms": b_rms_series,
        "mag_energy": mag_series,
    }


def load_openpmd_cartesian(path: Path, mesh: str, components: list[str]) -> dict:
    with h5py.File(path, "r") as h5f:
        base = h5f[f"/data/0/meshes/{mesh}"]
        data = {}
        for comp in components:
            if comp not in base:
                raise SystemExit(f"Missing component '{comp}' in {path}")
            arr = np.asarray(base[comp], dtype=float)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            data[comp] = arr
        data["path"] = str(path)
    return data


def histogram_stats(edges: np.ndarray, counts: np.ndarray) -> dict | None:
    total = float(np.sum(counts))
    if total <= 0.0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = float(np.sum(counts * centers) / total)
    cdf = np.cumsum(counts) / total
    def _quantile(q):
        idx = int(np.searchsorted(cdf, q))
        idx = max(0, min(idx, len(edges) - 1))
        return float(edges[idx])
    return {
        "E_mean": mean,
        "E_p90": _quantile(0.9),
        "E_p99": _quantile(0.99),
        "total_weight": total,
    }


def maxwellian_energy_spectrum(edges: np.ndarray, kT_eV: float) -> np.ndarray | None:
    if kT_eV <= 0.0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    spectrum = np.sqrt(centers) * np.exp(-centers / kT_eV)
    spectrum = spectrum * widths
    total = np.sum(spectrum)
    if total <= 0.0:
        return None
    return spectrum / total


def interpolate_series(times_src, vals_src, times_tgt):
    if not times_src or not vals_src:
        return None
    t_src = np.asarray(times_src, dtype=float)
    v_src = np.asarray(vals_src, dtype=float)
    order = np.argsort(t_src)
    t_src = t_src[order]
    v_src = v_src[order]
    return np.interp(times_tgt, t_src, v_src)


def rel_error_series(ref_vals, test_vals, eps=1.0e-30):
    ref_vals = np.asarray(ref_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    denom = np.maximum(np.abs(ref_vals), eps)
    return np.abs(test_vals - ref_vals) / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MHD vs Hybrid comparison metrics.")
    parser.add_argument("--metrics-in", required=True, help="Metrics JSON from analyze_h4_continuation_case.py")
    parser.add_argument("--summary-in", required=True, help="Mapping summary JSON from analyze_h4_continuation_case.py")
    parser.add_argument("--warpx-meta", default=None, help="WarpX metadata JSON (optional).")
    parser.add_argument("--config", default=None, help="Comparison config JSON (optional).")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output comparison summary JSON.")
    args = parser.parse_args()

    metrics_in_path = Path(args.metrics_in)
    summary_in_path = Path(args.summary_in)
    metrics_in = load_json(metrics_in_path)
    summary_in = load_json(summary_in_path)
    warpx_meta = load_json(Path(args.warpx_meta)) if args.warpx_meta else {}
    config = load_json(Path(args.config)) if args.config else {}
    repo_root = Path(__file__).resolve().parents[1]

    handoff_meta_path = summary_in.get("handoff_meta")
    handoff_meta = load_json(Path(handoff_meta_path)) if handoff_meta_path else {}

    compare_window = {}
    if warpx_meta:
        args_meta = warpx_meta.get("args") or {}
        dt = args_meta.get("dt")
        max_steps = args_meta.get("max_steps")
        diag_period = args_meta.get("diag_period")
        if dt is not None:
            compare_window["dt_s"] = float(dt)
        if max_steps is not None:
            compare_window["max_steps"] = int(max_steps)
        if diag_period is not None:
            compare_window["diag_period"] = int(diag_period)
        if dt is not None and max_steps is not None:
            compare_window["t_end_s"] = float(dt) * float(max_steps)

    metrics = dict(metrics_in)
    if compare_window:
        metrics["compare_window"] = compare_window
    if config:
        metrics["compare_config"] = config
    metrics["source_metrics_path"] = str(metrics_in_path)
    metrics["source_summary_path"] = str(summary_in_path)
    if args.warpx_meta:
        metrics["warpx_meta_path"] = str(Path(args.warpx_meta))

    traj_metrics = {"traj_points_used": 0, "traj_window_status": "not_computed"}
    traj_summary = {"traj_window_status": "not_computed"}
    vtk_pattern = config.get("vtk_pattern")
    traj_cfg = config.get("traj_window") or {}
    traj_points = int(traj_cfg.get("vtk_points", traj_cfg.get("vtk_count", 5)))
    traj_min_points = int(traj_cfg.get("min_points", 3))
    traj_eps = float(traj_cfg.get("rel_eps", 1.0e-30))
    allow_index_fallback = bool(traj_cfg.get("allow_index_fallback", True))

    if vtk_pattern and handoff_meta:
        vtk_files = sorted(Path(p) for p in (repo_root / vtk_pattern).parent.glob(Path(vtk_pattern).name))
        selected_index = None
        selection = handoff_meta.get("selection") or {}
        if "selected_index" in selection:
            selected_index = int(selection.get("selected_index"))
        if selected_index is None:
            selected_index = 0
        idx_end = min(len(vtk_files), selected_index + traj_points)
        vtk_window = vtk_files[selected_index:idx_end]

        axis_mode = handoff_meta.get("axis_mode", "x_z_y_r")
        fold_r = bool(handoff_meta.get("fold_r", True))
        resample = bool(handoff_meta.get("resample", False))
        nr = int(handoff_meta.get("nr", 0))
        nz = int(handoff_meta.get("nz", 0))
        r_min = float(handoff_meta.get("r_min", 0.0))
        r_max = float(handoff_meta.get("r_max", 0.0))
        z_min = float(handoff_meta.get("z_min", 0.0))
        z_max = float(handoff_meta.get("z_max", 0.0))
        athena_vis_path = handoff_meta.get("athena_vis_path")
        rho_scale = float(handoff_meta.get("rho_scale", 1.0))
        b_scale = float(handoff_meta.get("B_scale", 1.0))

        if nr > 0 and nz > 0:
            dr = (r_max - r_min) / nr if r_max > r_min else 1.0
            dz = (z_max - z_min) / nz if z_max > z_min else 1.0
            r_centers = r_min + (np.arange(nr) + 0.5) * dr
            z_centers = z_min + (np.arange(nz) + 0.5) * dz
            grid = (r_centers, z_centers, nr, nz, r_min, r_max, z_min, z_max)
        else:
            grid = None

        mhd_times = []
        mhd_mass = []
        mhd_b_rms = []
        mhd_mag = []
        for vtk_file in vtk_window:
            if not vtk_file.exists():
                continue
            t_val = parse_vtk_time(vtk_file)
            if t_val is None:
                t_val = float(vtk_files.index(vtk_file))
            if grid is None:
                continue
            rho, Br, Bt, Bz, r_centers, _ = map_vtk_to_rz(
                vtk_file,
                axis_mode=axis_mode,
                fold_r=fold_r,
                resample=resample,
                grid=grid,
                athena_vis_path=athena_vis_path,
            )
            rho = rho_scale * rho
            Br = b_scale * Br
            Bt = b_scale * Bt
            Bz = b_scale * Bz
            mass_val = axisymmetric_mass(rho, r_centers, dr, dz)
            mag_val = axisymmetric_mag_energy(Br, Bt, Bz, r_centers, dr, dz)
            b_rms_val = axisymmetric_b_rms(Br, Bt, Bz, r_centers, dr, dz)
            mhd_times.append(float(t_val))
            mhd_mass.append(mass_val)
            mhd_b_rms.append(b_rms_val)
            mhd_mag.append(mag_val)

        warpx_meta_path = Path(args.warpx_meta) if args.warpx_meta else None
        warpx_diag_dir = warpx_meta_path.parent / "diag" if warpx_meta_path else None
        ion_amu = float((warpx_meta.get("args") or {}).get("ion_amu", 1.0))
        warpx_series = (
            compute_warpx_series(warpx_diag_dir, ion_amu) if warpx_diag_dir else {}
        )

        traj_skip_leading = 0
        if mhd_times and warpx_series.get("times"):
            if warpx_series.get("b_rms") and warpx_series.get("mag_energy"):
                for idx, (b_val, mag_val) in enumerate(
                    zip(warpx_series["b_rms"], warpx_series["mag_energy"])
                ):
                    if abs(b_val) > traj_eps and abs(mag_val) > traj_eps:
                        traj_skip_leading = idx
                        break
            if traj_skip_leading > 0:
                max_skip = min(
                    traj_skip_leading,
                    max(0, len(mhd_times) - traj_min_points),
                    max(0, len(warpx_series["times"]) - traj_min_points),
                )
                if max_skip > 0:
                    traj_skip_leading = max_skip
                    mhd_times = mhd_times[traj_skip_leading:]
                    mhd_mass = mhd_mass[traj_skip_leading:]
                    mhd_b_rms = mhd_b_rms[traj_skip_leading:]
                    mhd_mag = mhd_mag[traj_skip_leading:]
                    for key in ("times", "mass", "b_rms", "mag_energy"):
                        warpx_series[key] = warpx_series[key][traj_skip_leading:]
                else:
                    traj_skip_leading = 0

            t0_mhd = mhd_times[0]
            t_rel_mhd = np.asarray(mhd_times, dtype=float) - t0_mhd
            t_rel_w = np.asarray(warpx_series["times"], dtype=float)
            t_rel_w = t_rel_w - t_rel_w[0]

            mask = (t_rel_mhd >= np.min(t_rel_w)) & (t_rel_mhd <= np.max(t_rel_w))
            t_rel_mhd_use = t_rel_mhd[mask]
            if len(t_rel_mhd_use) >= traj_min_points:
                warpx_mass = interpolate_series(t_rel_w, warpx_series["mass"], t_rel_mhd_use)
                warpx_b_rms = interpolate_series(t_rel_w, warpx_series["b_rms"], t_rel_mhd_use)
                warpx_mag = interpolate_series(t_rel_w, warpx_series["mag_energy"], t_rel_mhd_use)

                mhd_mass_use = np.asarray(mhd_mass, dtype=float)[mask]
                mhd_b_rms_use = np.asarray(mhd_b_rms, dtype=float)[mask]
                mhd_mag_use = np.asarray(mhd_mag, dtype=float)[mask]

                def _norm(series):
                    denom = series[0] if abs(series[0]) > traj_eps else 1.0
                    return series / denom

                mhd_mass_norm = _norm(mhd_mass_use)
                warpx_mass_norm = _norm(warpx_mass)
                mhd_b_rms_norm = _norm(mhd_b_rms_use)
                warpx_b_rms_norm = _norm(warpx_b_rms)
                mhd_mag_norm = _norm(mhd_mag_use)
                warpx_mag_norm = _norm(warpx_mag)

                mass_err = rel_error_series(mhd_mass_norm, warpx_mass_norm, eps=traj_eps)
                b_rms_err = rel_error_series(mhd_b_rms_norm, warpx_b_rms_norm, eps=traj_eps)
                mag_err = rel_error_series(mhd_mag_norm, warpx_mag_norm, eps=traj_eps)

                traj_metrics = {
                    "traj_points_mhd": int(len(mhd_times)),
                    "traj_points_warpx": int(len(warpx_series["times"])),
                    "traj_points_used": int(len(t_rel_mhd_use)),
                    "traj_mass_rel_rms": float(np.sqrt(np.mean(mass_err**2))),
                    "traj_b_rms_rel_rms": float(np.sqrt(np.mean(b_rms_err**2))),
                    "traj_mag_energy_rel_rms": float(np.sqrt(np.mean(mag_err**2))),
                    "traj_mass_rel_max": float(np.max(mass_err)),
                    "traj_b_rms_rel_max": float(np.max(b_rms_err)),
                    "traj_mag_energy_rel_max": float(np.max(mag_err)),
                    "traj_window_status": "ok",
                    "traj_window_mode": "time",
                    "traj_norm_mode": "relative_to_t0",
                    "traj_skip_leading": int(traj_skip_leading),
                }
                traj_summary = {
                    "traj_window_t0": float(t0_mhd),
                    "traj_window_t_end": float(mhd_times[len(mhd_times) - 1]),
                    "traj_points_mhd": int(len(mhd_times)),
                    "traj_points_warpx": int(len(warpx_series["times"])),
                    "traj_points_used": int(len(t_rel_mhd_use)),
                    "traj_window_status": "ok",
                    "traj_window_mode": "time",
                    "traj_norm_mode": "relative_to_t0",
                    "traj_skip_leading": int(traj_skip_leading),
                }

                plots_dir = metrics_in_path.parent.parent / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)

                fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)
                axes[0].plot(t_rel_mhd, _norm(np.asarray(mhd_mass, dtype=float)), marker="o", label="MHD")
                axes[0].plot(t_rel_w, _norm(np.asarray(warpx_series["mass"], dtype=float)), marker="o", label="WarpX")
                axes[0].set_title("Mass vs time (normalized)")
                axes[0].set_xlabel("t - t0 [s]")
                axes[0].legend()
                axes[1].plot(t_rel_mhd, _norm(np.asarray(mhd_b_rms, dtype=float)), marker="o", label="MHD")
                axes[1].plot(t_rel_w, _norm(np.asarray(warpx_series["b_rms"], dtype=float)), marker="o", label="WarpX")
                axes[1].set_title("B_rms vs time (normalized)")
                axes[1].set_xlabel("t - t0 [s]")
                axes[1].legend()
                axes[2].plot(t_rel_mhd, _norm(np.asarray(mhd_mag, dtype=float)), marker="o", label="MHD")
                axes[2].plot(t_rel_w, _norm(np.asarray(warpx_series["mag_energy"], dtype=float)), marker="o", label="WarpX")
                axes[2].set_title("Mag energy vs time (normalized)")
                axes[2].set_xlabel("t - t0 [s]")
                axes[2].legend()
                fig.savefig(plots_dir / "global_budget_vs_time_overlay_mhd_vs_warpx.png")
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                ax.plot(t_rel_mhd_use, mass_err, marker="o", label="mass rel err")
                ax.plot(t_rel_mhd_use, b_rms_err, marker="o", label="b_rms rel err")
                ax.plot(t_rel_mhd_use, mag_err, marker="o", label="mag_energy rel err")
                ax.set_xlabel("t - t0 [s]")
                ax.set_ylabel("relative error")
                ax.set_title("Trajectory Relative Error vs Time")
                ax.legend()
                fig.savefig(plots_dir / "traj_rel_error_vs_time.png")
                plt.close(fig)
            else:
                if allow_index_fallback:
                    n_use = min(len(mhd_times), len(warpx_series["times"]), traj_points)
                    if n_use >= traj_min_points:
                        mhd_mass_use = np.asarray(mhd_mass, dtype=float)[:n_use]
                        mhd_b_rms_use = np.asarray(mhd_b_rms, dtype=float)[:n_use]
                        mhd_mag_use = np.asarray(mhd_mag, dtype=float)[:n_use]
                        warpx_mass_use = np.asarray(warpx_series["mass"], dtype=float)[:n_use]
                        warpx_b_rms_use = np.asarray(warpx_series["b_rms"], dtype=float)[:n_use]
                        warpx_mag_use = np.asarray(warpx_series["mag_energy"], dtype=float)[:n_use]

                        def _norm(series):
                            denom = series[0] if abs(series[0]) > traj_eps else 1.0
                            return series / denom

                        mhd_mass_norm = _norm(mhd_mass_use)
                        warpx_mass_norm = _norm(warpx_mass_use)
                        mhd_b_rms_norm = _norm(mhd_b_rms_use)
                        warpx_b_rms_norm = _norm(warpx_b_rms_use)
                        mhd_mag_norm = _norm(mhd_mag_use)
                        warpx_mag_norm = _norm(warpx_mag_use)

                        mass_err = rel_error_series(mhd_mass_norm, warpx_mass_norm, eps=traj_eps)
                        b_rms_err = rel_error_series(mhd_b_rms_norm, warpx_b_rms_norm, eps=traj_eps)
                        mag_err = rel_error_series(mhd_mag_norm, warpx_mag_norm, eps=traj_eps)

                        traj_metrics = {
                            "traj_points_mhd": int(len(mhd_times)),
                            "traj_points_warpx": int(len(warpx_series["times"])),
                            "traj_points_used": int(n_use),
                            "traj_mass_rel_rms": float(np.sqrt(np.mean(mass_err**2))),
                            "traj_b_rms_rel_rms": float(np.sqrt(np.mean(b_rms_err**2))),
                            "traj_mag_energy_rel_rms": float(np.sqrt(np.mean(mag_err**2))),
                            "traj_mass_rel_max": float(np.max(mass_err)),
                            "traj_b_rms_rel_max": float(np.max(b_rms_err)),
                            "traj_mag_energy_rel_max": float(np.max(mag_err)),
                            "traj_window_status": "ok",
                            "traj_window_mode": "index",
                            "traj_norm_mode": "relative_to_t0",
                            "traj_skip_leading": int(traj_skip_leading),
                        }
                        traj_summary = {
                            "traj_window_t0": float(t0_mhd),
                            "traj_window_t_end": float(mhd_times[len(mhd_times) - 1]),
                            "traj_points_mhd": int(len(mhd_times)),
                            "traj_points_warpx": int(len(warpx_series["times"])),
                            "traj_points_used": int(n_use),
                            "traj_window_status": "ok",
                            "traj_window_mode": "index",
                            "traj_norm_mode": "relative_to_t0",
                            "traj_skip_leading": int(traj_skip_leading),
                        }

                        plots_dir = metrics_in_path.parent.parent / "plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        t_index = np.arange(n_use, dtype=float)

                        fig, axes = plt.subplots(3, 1, figsize=(7, 8), constrained_layout=True)
                        axes[0].plot(t_index, mhd_mass_norm, marker="o", label="MHD")
                        axes[0].plot(t_index, warpx_mass_norm, marker="o", label="WarpX")
                        axes[0].set_title("Mass vs output index (normalized)")
                        axes[0].set_xlabel("output index")
                        axes[0].legend()
                        axes[1].plot(t_index, mhd_b_rms_norm, marker="o", label="MHD")
                        axes[1].plot(t_index, warpx_b_rms_norm, marker="o", label="WarpX")
                        axes[1].set_title("B_rms vs output index (normalized)")
                        axes[1].set_xlabel("output index")
                        axes[1].legend()
                        axes[2].plot(t_index, mhd_mag_norm, marker="o", label="MHD")
                        axes[2].plot(t_index, warpx_mag_norm, marker="o", label="WarpX")
                        axes[2].set_title("Mag energy vs output index (normalized)")
                        axes[2].set_xlabel("output index")
                        axes[2].legend()
                        fig.savefig(plots_dir / "global_budget_vs_time_overlay_mhd_vs_warpx.png")
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                        ax.plot(t_index, mass_err, marker="o", label="mass rel err")
                        ax.plot(t_index, b_rms_err, marker="o", label="b_rms rel err")
                        ax.plot(t_index, mag_err, marker="o", label="mag_energy rel err")
                        ax.set_xlabel("output index")
                        ax.set_ylabel("relative error")
                        ax.set_title("Trajectory Relative Error vs Output Index (normalized)")
                        ax.legend()
                        fig.savefig(plots_dir / "traj_rel_error_vs_time.png")
                        plt.close(fig)
                    else:
                        traj_metrics["traj_window_status"] = "insufficient_overlap"
                        traj_summary["traj_window_status"] = "insufficient_overlap"
                else:
                    traj_metrics["traj_window_status"] = "insufficient_overlap"
                    traj_summary["traj_window_status"] = "insufficient_overlap"
        else:
            traj_metrics["traj_window_status"] = "missing_series"
            traj_summary["traj_window_status"] = "missing_series"

    energy_summary = {}
    energy_details = {}
    plots_dir = metrics_in_path.parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    energy_meta = warpx_meta.get("energy_spectrum") or {}
    edges_eV = energy_meta.get("edges_eV")
    records = energy_meta.get("records") or []
    if edges_eV and records:
        edges = np.asarray(edges_eV, dtype=float)
        energy_details = {
            "edges_eV": edges_eV,
            "records": records,
        }
        cfg = energy_meta.get("config") or {}
        species = str(cfg.get("species", "ions"))
        window_min = cfg.get("min_eV")
        window_max = cfg.get("max_eV")
        bins = cfg.get("bins")
        if window_min is None and edges.size > 0:
            window_min = float(edges[0])
        if window_max is None and edges.size > 0:
            window_max = float(edges[-1])
        if bins is None:
            bins = max(0, int(edges.size - 1))
        mhd_kT_eV = None
        opmd = warpx_meta.get("opmd") or {}
        fluid_path = opmd.get("fluid_path")
        if fluid_path and Path(fluid_path).exists():
            comp = "Te" if species == "electrons" else "Ti"
            try:
                fluid = load_openpmd_cartesian(Path(fluid_path), "fluid", ["rho", comp])
                rho = fluid["rho"]
                temp = fluid[comp]
                mask = rho > 0.0
                if np.any(mask):
                    mhd_kT_eV = float(np.sum(temp[mask] * rho[mask]) / np.sum(rho[mask]))
            except Exception:
                mhd_kT_eV = None
        mhd_spectrum = maxwellian_energy_spectrum(edges, mhd_kT_eV) if mhd_kT_eV else None

        stats_records = []
        records_sorted = sorted(records, key=lambda r: r.get("step", 0))
        p90_t0 = None
        for idx, rec in enumerate(records_sorted):
            counts = np.asarray(rec.get("counts") or [], dtype=float)
            if counts.size == 0:
                continue
            stats = histogram_stats(edges, counts)
            if stats is None:
                continue
            if idx == 0:
                p90_t0 = stats.get("E_p90")
            total = stats.get("total_weight", 0.0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if p90_t0 is not None and total > 0.0:
                high_mask = centers >= p90_t0
                stats["high_energy_frac"] = float(np.sum(counts[high_mask]) / total)
            stats.update({"step": rec.get("step"), "time": rec.get("time")})
            stats_records.append(stats)

        energy_summary = {
            "species": species,
            "energy_window_min_eV": window_min,
            "energy_window_max_eV": window_max,
            "bins": bins,
            "Ti_mean_eV" if species != "electrons" else "Te_mean_eV": mhd_kT_eV,
            "records": stats_records,
            "p90_t0": p90_t0,
        }

        plot_records = records_sorted[:3]
        if plot_records:
            fig, axes = plt.subplots(
                1, len(plot_records), figsize=(5 * len(plot_records), 4), constrained_layout=True
            )
            if len(plot_records) == 1:
                axes = [axes]
            for ax, rec in zip(axes, plot_records):
                counts = np.asarray(rec.get("counts") or [], dtype=float)
                if counts.size == 0:
                    continue
                total = float(np.sum(counts))
                if total <= 0.0:
                    continue
                counts_norm = np.asarray(rec.get("counts_norm") or []) if rec.get("counts_norm") else counts / total
                centers = 0.5 * (edges[:-1] + edges[1:])
                y_vals = np.maximum(counts_norm, 1.0e-30)
                ax.step(centers, y_vals, where="mid", label="Hybrid", color="#1f77b4")
                if mhd_spectrum is not None:
                    ax.step(centers, np.maximum(mhd_spectrum, 1.0e-30), where="mid", label="MHD", linestyle="--", color="#444444")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Energy [eV]")
                ax.set_ylabel("Normalized counts")
                t_val = rec.get("time")
                step_val = rec.get("step")
                title = f"step {step_val}" if t_val is None else f"t={t_val:.2e}s"
                ax.set_title(title)
                ax.legend()
            fig.savefig(plots_dir / "particle_energy_spectrum_compare.png")
            plt.close(fig)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.update(traj_metrics)
    if energy_summary:
        metrics["energy_spectrum"] = energy_summary
        metrics["energy_spectrum_records_count"] = len(energy_summary.get("records") or [])
        metrics["energy_spectrum_window_min_eV"] = energy_summary.get("energy_window_min_eV")
        metrics["energy_spectrum_window_max_eV"] = energy_summary.get("energy_window_max_eV")
        metrics["energy_spectrum_bins"] = energy_summary.get("bins")
        metrics["energy_spectrum_species"] = energy_summary.get("species")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if energy_details:
        energy_path = metrics_in_path.parent / "metrics_b1_energy.json"
        energy_path.write_text(json.dumps(energy_details, indent=2, sort_keys=True), encoding="utf-8")

    key_metrics = {
        "mass_rel_diff_init": metrics.get("mass_rel_diff_init"),
        "b_rms_rel_diff_init": metrics.get("b_rms_rel_diff_init"),
        "mag_energy_rel_diff_init": metrics.get("mag_energy_rel_diff_init"),
        "mass_rel_drift_over_run": metrics.get("mass_rel_drift_over_run"),
        "traj_mass_rel_rms": metrics.get("traj_mass_rel_rms"),
        "traj_b_rms_rel_rms": metrics.get("traj_b_rms_rel_rms"),
        "traj_mag_energy_rel_rms": metrics.get("traj_mag_energy_rel_rms"),
        "warpx_ran_to_completion": metrics.get("warpx_ran_to_completion"),
        "warpx_num_outputs": metrics.get("warpx_num_outputs"),
        "warpx_no_nan_in_metrics": metrics.get("warpx_no_nan_in_metrics"),
        "warpx_drop_breach": metrics.get("warpx_drop_breach"),
    }
    summary = {
        "compare_config": config,
        "compare_window": compare_window,
        "mapping_summary": summary_in,
        "key_metrics": key_metrics,
        "traj_window": traj_summary,
        "source_metrics_path": str(metrics_in_path),
        "source_summary_path": str(summary_in_path),
    }
    if args.warpx_meta:
        summary["warpx_meta_path"] = str(Path(args.warpx_meta))

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
