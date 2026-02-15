#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
        spacing = base[components[0]].attrs.get("gridSpacing", base.attrs.get("gridSpacing", None))
        offset = base[components[0]].attrs.get("gridGlobalOffset", base.attrs.get("gridGlobalOffset", None))
        if spacing is None or offset is None:
            raise SystemExit(f"gridSpacing/gridGlobalOffset missing in {path}")

    spacing = np.asarray(spacing, dtype=float)
    offset = np.asarray(offset, dtype=float)
    if spacing.size != 3 or offset.size != 3:
        raise SystemExit(f"Expected 3D spacing/offset in {path}, got {spacing}, {offset}")
    data["spacing"] = spacing
    data["offset"] = offset
    data["path"] = str(path)
    return data


def compute_mass(rho: np.ndarray, spacing: np.ndarray) -> float:
    dx, dy, dz = spacing
    return float(np.sum(rho) * dx * dy * dz)


def compute_b_metrics(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, spacing: np.ndarray) -> dict:
    dx, dy, dz = spacing
    b2 = Bx * Bx + By * By + Bz * Bz
    volume = dx * dy * dz
    mag_energy = float(0.5 * np.sum(b2 * volume))
    b_rms = float(np.sqrt(np.mean(b2)))
    return {"mag_energy": mag_energy, "b_rms": b_rms}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ran_to_completion(meta: dict, time_series: list[float]) -> bool:
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []
    last_step = records[-1].get("step") if records else None
    max_steps = (meta.get("args") or {}).get("max_steps")
    if max_steps is not None and last_step is not None:
        return last_step >= (max_steps - 1)
    return bool(time_series)


def extract_series(diag_root: Path, ion_mass: float) -> dict:
    diags = list_diags(diag_root)
    mass_series = []
    b_rms_series = []
    mag_series = []
    time_series = []
    first_snapshot = None

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
        b_metrics = compute_b_metrics(Bx_w, By_w, Bz_w, np.array([dx, dy, dz], dtype=float))
        time_series.append(t)
        mass_series.append(mass_val)
        b_rms_series.append(b_metrics["b_rms"])
        mag_series.append(b_metrics["mag_energy"])
        if first_snapshot is None:
            first_snapshot = {
                "rho": rho_w,
                "Bx": Bx_w,
                "By": By_w,
                "Bz": Bz_w,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "x0": float(ds.domain_left_edge[0].to_value()),
                "y0": float(ds.domain_left_edge[1].to_value()),
                "z0": float(ds.domain_left_edge[2].to_value()),
                "dims": dims,
            }

    return {
        "diags": diags,
        "time_series": time_series,
        "mass_series": mass_series,
        "b_rms_series": b_rms_series,
        "mag_series": mag_series,
        "first_snapshot": first_snapshot,
    }


def plot_budget(time_series, mass_series, b_rms_series, mag_series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(time_series, mass_series, label="mass")
    ax.plot(time_series, b_rms_series, label="b_rms")
    ax.plot(time_series, mag_series, label="mag_energy")
    ax.set_xlabel("time [s]")
    ax.set_title("WarpX global budget")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze H4 restart consistency (pre/post).")
    parser.add_argument("--handoff-meta", required=True, help="handoff_meta.json from prepare step.")
    parser.add_argument("--warpx-meta-before", required=True, help="WarpX metadata JSON (pre).")
    parser.add_argument("--warpx-meta-after", required=True, help="WarpX metadata JSON (post).")
    parser.add_argument("--diag-dir-before", required=True, help="WarpX diag directory (pre).")
    parser.add_argument("--diag-dir-after", required=True, help="WarpX diag directory (post).")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output mapping summary JSON.")
    parser.add_argument("--plots-dir", required=True, help="Plots directory.")
    args = parser.parse_args()

    meta_path = Path(args.handoff_meta)
    handoff_meta = load_json(meta_path)
    fluid_path = Path(handoff_meta.get("fluid_path", ""))
    b_path = Path(handoff_meta.get("b_path", ""))

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "opmd_exists": fluid_path.exists() and b_path.exists(),
        "opmd_fields_present": False,
        "opmd_no_nan": False,
        "particle_load_ok": False,
        "warpx_ran_to_completion_pre": False,
        "warpx_ran_to_completion_post": False,
        "warpx_num_outputs_pre": 0,
        "warpx_num_outputs_post": 0,
        "warpx_no_nan_in_metrics": False,
        "warpx_drop_breach_pre": None,
        "warpx_drop_breach_post": None,
        "restart_jump_mass": None,
        "restart_jump_b_rms": None,
        "restart_jump_mag_energy": None,
    }

    if metrics["opmd_exists"]:
        fluid = load_openpmd_cartesian(fluid_path, "fluid", ["rho", "vx", "vy", "vz", "Ti", "Te"])
        bfield = load_openpmd_cartesian(b_path, "B", ["x", "y", "z"])
        rho = fluid["rho"]
        Bx = bfield["x"]
        By = bfield["y"]
        Bz = bfield["z"]
        metrics["opmd_fields_present"] = True
        metrics["opmd_no_nan"] = not (
            np.isnan(rho).any()
            or np.isnan(Bx).any()
            or np.isnan(By).any()
            or np.isnan(Bz).any()
        )
        mass_opmd = compute_mass(rho, fluid["spacing"])
        b_metrics = compute_b_metrics(Bx, By, Bz, bfield["spacing"])
        metrics.update(
            {
                "mass_opmd": mass_opmd,
                "mag_energy_opmd": b_metrics["mag_energy"],
                "b_rms_opmd": b_metrics["b_rms"],
            }
        )
    else:
        fluid = None
        bfield = None

    rotation_mass_rel = handoff_meta.get("rotation_mass_integral_rel_diff")
    if rotation_mass_rel is not None:
        metrics["rotation_mass_integral_rel_diff"] = float(rotation_mass_rel)

    meta_pre = load_json(Path(args.warpx_meta_before))
    meta_post = load_json(Path(args.warpx_meta_after))
    ion_amu = float((meta_pre.get("args") or {}).get("ion_amu", 1.0))
    ion_mass = ion_amu * M_P

    series_pre = extract_series(Path(args.diag_dir_before), ion_mass)
    series_post = extract_series(Path(args.diag_dir_after), ion_mass)

    mass_series_pre = series_pre["mass_series"]
    b_rms_series_pre = series_pre["b_rms_series"]
    mag_series_pre = series_pre["mag_series"]
    time_series_pre = series_pre["time_series"]

    mass_series_post = series_post["mass_series"]
    b_rms_series_post = series_post["b_rms_series"]
    mag_series_post = series_post["mag_series"]
    time_series_post = series_post["time_series"]

    metrics["warpx_num_outputs_pre"] = len(series_pre["diags"])
    metrics["warpx_num_outputs_post"] = len(series_post["diags"])

    metrics["warpx_ran_to_completion_pre"] = ran_to_completion(meta_pre, time_series_pre)
    metrics["warpx_ran_to_completion_post"] = ran_to_completion(meta_post, time_series_post)

    monitor_pre = meta_pre.get("monitor") or {}
    monitor_post = meta_post.get("monitor") or {}
    metrics["warpx_drop_breach_pre"] = monitor_pre.get("drop_breach")
    metrics["warpx_drop_breach_post"] = monitor_post.get("drop_breach")

    if mass_series_pre:
        metrics["mass_warpx_initial"] = mass_series_pre[0]
        metrics["mass_warpx_final"] = mass_series_pre[-1]
        metrics["mass_rel_drift_over_run_pre"] = float(
            abs(mass_series_pre[-1] - mass_series_pre[0]) / max(mass_series_pre[0], 1.0e-30)
        )
        metrics["mass_from_rho_mode"] = "charge_density"
        metrics["mass_source"] = "rho_cell_integral"
        metrics["ion_amu"] = ion_amu
    b_init_idx = None
    b_init_floor = 1.0e-12
    if b_rms_series_pre:
        for idx, val in enumerate(b_rms_series_pre):
            if val > b_init_floor:
                b_init_idx = idx
                break
        if b_init_idx is None:
            b_init_idx = 0
        metrics["b_rms_init_index"] = b_init_idx
        if time_series_pre:
            metrics["b_rms_init_time"] = time_series_pre[b_init_idx]
        metrics["b_rms_warpx_initial"] = b_rms_series_pre[b_init_idx]
        metrics["b_rms_warpx_final"] = b_rms_series_pre[-1]
        metrics["b_rms_rel_drift_over_run_pre"] = float(
            abs(b_rms_series_pre[-1] - b_rms_series_pre[b_init_idx])
            / max(b_rms_series_pre[b_init_idx], 1.0e-30)
        )
    if mag_series_pre:
        mag_init_idx = b_init_idx if b_init_idx is not None else 0
        metrics["mag_energy_warpx_initial"] = mag_series_pre[mag_init_idx]
        metrics["mag_energy_warpx_final"] = mag_series_pre[-1]

    if fluid is not None and mass_series_pre:
        denom = max(metrics.get("mass_opmd", 0.0), 1.0e-30)
        metrics["mass_rel_diff_init"] = float(abs(mass_series_pre[0] - metrics["mass_opmd"]) / denom)
    if fluid is not None and b_rms_series_pre:
        denom = max(metrics.get("b_rms_opmd", 0.0), 1.0e-30)
        metrics["b_rms_rel_diff_init"] = float(
            abs(metrics.get("b_rms_warpx_initial", b_rms_series_pre[0]) - metrics["b_rms_opmd"]) / denom
        )
    if fluid is not None and mag_series_pre:
        denom = max(metrics.get("mag_energy_opmd", 0.0), 1.0e-30)
        metrics["mag_energy_rel_diff_init"] = float(
            abs(metrics.get("mag_energy_warpx_initial", mag_series_pre[0]) - metrics["mag_energy_opmd"]) / denom
        )

    if mass_series_post:
        metrics["mass_rel_drift_over_run_post"] = float(
            abs(mass_series_post[-1] - mass_series_post[0]) / max(mass_series_post[0], 1.0e-30)
        )
    if b_rms_series_post:
        metrics["b_rms_rel_drift_over_run_post"] = float(
            abs(b_rms_series_post[-1] - b_rms_series_post[0]) / max(b_rms_series_post[0], 1.0e-30)
        )
    if mag_series_post:
        metrics["mag_energy_rel_drift_over_run_post"] = float(
            abs(mag_series_post[-1] - mag_series_post[0]) / max(mag_series_post[0], 1.0e-30)
        )

    if mass_series_pre and mass_series_post:
        metrics["restart_jump_mass"] = float(
            abs(mass_series_post[0] - mass_series_pre[-1]) / max(mass_series_pre[-1], 1.0e-30)
        )
    if b_rms_series_pre and b_rms_series_post:
        metrics["restart_jump_b_rms"] = float(
            abs(b_rms_series_post[0] - b_rms_series_pre[-1]) / max(b_rms_series_pre[-1], 1.0e-30)
        )
    if mag_series_pre and mag_series_post:
        metrics["restart_jump_mag_energy"] = float(
            abs(mag_series_post[0] - mag_series_pre[-1]) / max(mag_series_pre[-1], 1.0e-30)
        )
    if time_series_pre and time_series_post:
        metrics["restart_time_gap"] = float(time_series_post[0] - time_series_pre[-1])

    species_stats_init = (meta_pre.get("species_stats_init") or {}).get("ions") or {}
    weight_sum = species_stats_init.get("weight_sum")
    if weight_sum is not None:
        metrics["particle_weight_sum_init"] = weight_sum
        metrics["particle_load_ok"] = weight_sum > 0.0
    else:
        weight_hist = meta_pre.get("particle_weight_hist") or {}
        counts = weight_hist.get("counts") or []
        metrics["particle_load_ok"] = bool(counts) and sum(counts) > 0

    metrics["warpx_no_nan_in_metrics"] = all(
        np.isfinite(val)
        for key, val in metrics.items()
        if isinstance(val, (int, float))
    )

    with Path(args.metrics).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    summary = {
        "handoff_meta": str(meta_path),
        "fluid_path": str(fluid_path),
        "b_path": str(b_path),
        "selection": handoff_meta.get("selection"),
        "mapping_mode": handoff_meta.get("mapping_mode"),
        "vector_map": handoff_meta.get("vector_map"),
        "rotation_mass_integral_rel_diff": metrics.get("rotation_mass_integral_rel_diff"),
        "mass_rel_diff_init": metrics.get("mass_rel_diff_init"),
        "b_rms_rel_diff_init": metrics.get("b_rms_rel_diff_init"),
        "mag_energy_rel_diff_init": metrics.get("mag_energy_rel_diff_init"),
        "restart_jump_mass": metrics.get("restart_jump_mass"),
        "restart_jump_b_rms": metrics.get("restart_jump_b_rms"),
        "restart_jump_mag_energy": metrics.get("restart_jump_mag_energy"),
        "b_rms_init_index": metrics.get("b_rms_init_index"),
        "b_rms_init_time": metrics.get("b_rms_init_time"),
        "diag_dir_pre": str(Path(args.diag_dir_before)),
        "diag_dir_post": str(Path(args.diag_dir_after)),
        "warpx_meta_pre": str(Path(args.warpx_meta_before)),
        "warpx_meta_post": str(Path(args.warpx_meta_after)),
    }
    with Path(args.summary).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    first_snapshot = series_pre["first_snapshot"]
    if fluid is not None and bfield is not None and first_snapshot is not None:
        ny = fluid["rho"].shape[1]
        mid_y = ny // 2
        rho_mhd = fluid["rho"][:, mid_y, :]
        bmag_mhd = np.sqrt(
            bfield["x"] * bfield["x"] + bfield["y"] * bfield["y"] + bfield["z"] * bfield["z"]
        )
        bmag_mhd = bmag_mhd[:, mid_y, :]

        dims = first_snapshot["dims"]
        mid_y_w = dims[1] // 2
        rho_w = first_snapshot["rho"][:, mid_y_w, :]
        bmag_w = np.sqrt(
            first_snapshot["Bx"] * first_snapshot["Bx"]
            + first_snapshot["By"] * first_snapshot["By"]
            + first_snapshot["Bz"] * first_snapshot["Bz"]
        )
        bmag_w = bmag_w[:, mid_y_w, :]

        x_edges = fluid["offset"][0] + np.arange(fluid["rho"].shape[0] + 1) * fluid["spacing"][0]
        z_edges = fluid["offset"][2] + np.arange(fluid["rho"].shape[2] + 1) * fluid["spacing"][2]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = axes[0].pcolormesh(x_edges, z_edges, rho_mhd.T, shading="auto")
        axes[0].set_title("MHD rho (t0)")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].pcolormesh(x_edges, z_edges, rho_w.T, shading="auto")
        axes[1].set_title("WarpX rho (t0)")
        fig.colorbar(im1, ax=axes[1])
        fig.savefig(plots_dir / "rho_slice_mhd_vs_warpx_t0.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = axes[0].pcolormesh(x_edges, z_edges, bmag_mhd.T, shading="auto")
        axes[0].set_title("|B| MHD (t0)")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].pcolormesh(x_edges, z_edges, bmag_w.T, shading="auto")
        axes[1].set_title("|B| WarpX (t0)")
        fig.colorbar(im1, ax=axes[1])
        fig.savefig(plots_dir / "B_slice_mhd_vs_warpx_t0.png")
        plt.close(fig)

    if time_series_pre:
        plot_budget(
            time_series_pre,
            mass_series_pre,
            b_rms_series_pre,
            mag_series_pre,
            plots_dir / "global_budget_vs_time_pre.png",
        )
    if time_series_post:
        plot_budget(
            time_series_post,
            mass_series_post,
            b_rms_series_post,
            mag_series_post,
            plots_dir / "global_budget_vs_time_post.png",
        )

    weight_hist = meta_pre.get("particle_weight_hist")
    if weight_hist and "bins" in weight_hist and "counts" in weight_hist:
        bins = np.asarray(weight_hist["bins"])
        counts = np.asarray(weight_hist["counts"])
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.step(bins[:-1], counts, where="post")
        ax.set_xlabel("particle weight")
        ax.set_ylabel("count")
        ax.set_title("Particle weight histogram")
        fig.savefig(plots_dir / "particle_weight_hist.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
