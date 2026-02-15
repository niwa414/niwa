#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import yt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"yt required for WarpX 3D analysis: {exc}")

UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "magnetic_unit": (1.0, "T"),
}


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


def extract_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def total_field_energy(ds, ad) -> float | None:
    try:
        Bx = ad["boxlib", "Bx"].to_ndarray()
        By = ad["boxlib", "By"].to_ndarray()
        Bz = ad["boxlib", "Bz"].to_ndarray()
        Ex = ad["boxlib", "Ex"].to_ndarray()
        Ey = ad["boxlib", "Ey"].to_ndarray()
        Ez = ad["boxlib", "Ez"].to_ndarray()
    except Exception:
        return None

    dims = tuple(int(x) for x in ds.domain_dimensions)
    Bx = reshape_field(Bx, dims)
    By = reshape_field(By, dims)
    Bz = reshape_field(Bz, dims)
    Ex = reshape_field(Ex, dims)
    Ey = reshape_field(Ey, dims)
    Ez = reshape_field(Ez, dims)
    if Bx.ndim != 3 or Ex.ndim != 3:
        return None

    b2 = Bx * Bx + By * By + Bz * Bz
    e2 = Ex * Ex + Ey * Ey + Ez * Ez
    dx = float(ds.domain_width[0].to_value()) / dims[0]
    dy = float(ds.domain_width[1].to_value()) / dims[1]
    dz = float(ds.domain_width[2].to_value()) / dims[2]
    volume = dx * dy * dz
    return float(0.5 * np.sum((b2 + e2) * volume))


def centroid_from_mask(rho_abs, mask, x_coords, y_coords, z_coords):
    mass = float(np.sum(rho_abs * mask))
    if mass <= 0.0:
        return None
    x_c = float(np.sum(rho_abs * mask * x_coords) / mass)
    y_c = float(np.sum(rho_abs * mask * y_coords) / mass)
    z_c = float(np.sum(rho_abs * mask * z_coords) / mass)
    return x_c, y_c, z_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hybrid approach (3D double blob).")
    parser.add_argument("--diag-dir", required=True, help="WarpX diag directory (contains diag*).")
    parser.add_argument("--metadata", required=True, help="WarpX run metadata JSON.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--csv", required=True, help="Output CSV for separation series.")
    parser.add_argument("--start-index", type=int, default=0, help="Index for initial separation ratio.")
    parser.add_argument("--rho-threshold-frac", type=float, default=0.4, help="Threshold fraction above background.")
    parser.add_argument(
        "--rho-bg-quantile", type=float, default=10.0, help="Percentile for background rho."
    )
    parser.add_argument(
        "--sep-mode",
        type=str,
        default="x",
        choices=["x", "xy", "xyz"],
        help="Separation mode (x, xy, xyz).",
    )
    args = parser.parse_args()

    diag_root = Path(args.diag_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    diags = list_diags(diag_root)
    times = []
    sep_series = []
    left_centroids = []
    right_centroids = []
    energy_series = []

    meta = extract_meta(Path(args.metadata))
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []
    use_monitor = False
    if records:
        for rec in records:
            split = rec.get("split_centroids")
            if not split or split.get("error"):
                continue
            t = rec.get("time")
            left = split.get("left")
            right = split.get("right")
            if t is None or not left or not right:
                continue
            sep_key = "sep_x" if args.sep_mode == "x" else ("sep_xy" if args.sep_mode == "xy" else "sep_xyz")
            sep = split.get(sep_key)
            if sep is None:
                continue
            times.append(t)
            sep_series.append(sep)
            left_centroids.append((left.get("x"), left.get("y"), left.get("z")))
            right_centroids.append((right.get("x"), right.get("y"), right.get("z")))
            use_monitor = True

    if not use_monitor:
        for diag in diags:
            ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
            ad = ds.all_data()
            t = float(ds.current_time.to_value())
            dims = tuple(int(x) for x in ds.domain_dimensions)
            rho = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
            if rho.ndim != 3:
                continue
            rho_abs = np.abs(rho)
            rho_peak = float(np.max(rho_abs))
            if rho_peak <= 0.0:
                continue
            rho_bg = float(np.percentile(rho_abs, args.rho_bg_quantile))
            rho_thr = rho_bg + args.rho_threshold_frac * (rho_peak - rho_bg)

            dx = float(ds.domain_width[0].to_value()) / dims[0]
            dy = float(ds.domain_width[1].to_value()) / dims[1]
            dz = float(ds.domain_width[2].to_value()) / dims[2]
            x0 = float(ds.domain_left_edge[0].to_value())
            y0 = float(ds.domain_left_edge[1].to_value())
            z0 = float(ds.domain_left_edge[2].to_value())
            x_centers = x0 + (np.arange(dims[0]) + 0.5) * dx
            y_centers = y0 + (np.arange(dims[1]) + 0.5) * dy
            z_centers = z0 + (np.arange(dims[2]) + 0.5) * dz
            x_coords = x_centers[:, None, None]
            y_coords = y_centers[None, :, None]
            z_coords = z_centers[None, None, :]

            mask = rho_abs >= rho_thr
            left_mask = mask & (x_coords < 0.0)
            right_mask = mask & (x_coords > 0.0)
            left = centroid_from_mask(rho_abs, left_mask, x_coords, y_coords, z_coords)
            right = centroid_from_mask(rho_abs, right_mask, x_coords, y_coords, z_coords)
            if left is None or right is None:
                continue
            if args.sep_mode == "x":
                sep = float(abs(left[0] - right[0]))
            elif args.sep_mode == "xy":
                sep = float(np.hypot(left[0] - right[0], left[1] - right[1]))
            else:
                sep = float(np.linalg.norm(np.array(left) - np.array(right)))
            times.append(t)
            sep_series.append(sep)
            left_centroids.append(left)
            right_centroids.append(right)

            energy = total_field_energy(ds, ad)
            energy_series.append({"time_s": t, "field_energy": energy})

    if use_monitor and diags and not energy_series:
        for diag in diags:
            ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
            ad = ds.all_data()
            t = float(ds.current_time.to_value())
            energy = total_field_energy(ds, ad)
            energy_series.append({"time_s": t, "field_energy": energy})

    sep_ratio = None
    sep_initial = None
    sep_final = None
    sep_min = None
    t_sep_min = None
    no_nan = True
    if sep_series:
        start_idx = min(max(args.start_index, 0), len(sep_series) - 1)
        sep_initial = sep_series[start_idx]
        sep_final = sep_series[-1]
        sep_min = min(sep_series)
        min_idx = sep_series.index(sep_min)
        t_sep_min = times[min_idx]
        if sep_initial and sep_initial > 0.0:
            sep_ratio = float(sep_final / sep_initial)
        for val in (sep_initial, sep_final, sep_ratio, sep_min):
            if val is None or not np.isfinite(val):
                no_nan = False

    run_args = meta.get("args", {})
    records = monitor.get("records") or []
    drop_breach = monitor.get("drop_breach")
    sim_time_reached = records[-1].get("time") if records else (times[-1] if times else None)
    last_step = records[-1].get("step") if records else None
    max_steps = run_args.get("max_steps")
    dt = run_args.get("dt")
    diag_period = run_args.get("diag_period", None)

    ran_to_completion = None
    if max_steps is not None:
        if last_step is not None:
            ran_to_completion = last_step >= (max_steps - 1)
        elif sim_time_reached is not None and dt is not None and diag_period is not None:
            expected = (max_steps - 1) * dt
            slack = dt * diag_period
            ran_to_completion = sim_time_reached >= (expected - slack)
    if ran_to_completion is None:
        ran_to_completion = bool(times)

    field_energy_initial = None
    field_energy_final = None
    field_energy_rel_drift = None
    if energy_series:
        energy_vals = [row.get("field_energy") for row in energy_series if row.get("field_energy") is not None]
        if energy_vals:
            field_energy_initial = energy_vals[0]
            field_energy_final = energy_vals[-1]
            if field_energy_initial not in (None, 0.0):
                field_energy_rel_drift = (field_energy_final - field_energy_initial) / field_energy_initial

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": len(times),
        "no_nan_in_metrics": no_nan,
        "sep_initial": sep_initial,
        "sep_final": sep_final,
        "sep_ratio": sep_ratio,
        "sep_min": sep_min,
        "sep_min_time": t_sep_min,
        "sep_ratio_def": f"sep(t_end)/sep(t_start) ({args.sep_mode})",
        "sep_mode": args.sep_mode,
        "rho_threshold_frac": args.rho_threshold_frac,
        "rho_bg_quantile": args.rho_bg_quantile,
        "drop_breach": drop_breach,
        "sim_time_reached": sim_time_reached,
        "field_energy_initial": field_energy_initial,
        "field_energy_final": field_energy_final,
        "field_energy_rel_drift": field_energy_rel_drift,
        "diag_dir": str(diag_root),
        "metadata_path": str(Path(args.metadata)),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time_s",
                "x_left",
                "y_left",
                "z_left",
                "x_right",
                "y_right",
                "z_right",
                "sep",
            ],
        )
        writer.writeheader()
        for t, left, right, sep in zip(times, left_centroids, right_centroids, sep_series):
            writer.writerow(
                {
                    "time_s": t,
                    "x_left": left[0],
                    "y_left": left[1],
                    "z_left": left[2],
                    "x_right": right[0],
                    "y_right": right[1],
                    "z_right": right[2],
                    "sep": sep,
                }
            )

    if sep_series:
        plt.figure(figsize=(6, 4))
        plt.plot(times, sep_series, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Centroid separation (m)")
        plt.title("Blob Separation vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "separation_vs_time.png")
        plt.close()

    if left_centroids and right_centroids:
        xl = [c[0] for c in left_centroids]
        yl = [c[1] for c in left_centroids]
        xr = [c[0] for c in right_centroids]
        yr = [c[1] for c in right_centroids]
        plt.figure(figsize=(6, 5))
        plt.plot(xl, yl, marker="o", label="left")
        plt.plot(xr, yr, marker="o", label="right")
        plt.xlabel("x centroid (m)")
        plt.ylabel("y centroid (m)")
        plt.title("Blob Centroid Paths")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "centroid_paths_two_blobs.png")
        plt.close()

    if energy_series:
        e_times = [row.get("time_s") for row in energy_series]
        e_vals = [row.get("field_energy") for row in energy_series]
        plt.figure(figsize=(6, 4))
        plt.plot(e_times, e_vals, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Field energy (arb)")
        plt.title("Field Energy vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "energy_budget_vs_time.png")
        plt.close()

    if diags:
        last_diag = diags[-1]
        ds = yt.load(str(last_diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        dims = tuple(int(x) for x in ds.domain_dimensions)
        rho = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
        if rho.ndim == 3:
            mid = dims[2] // 2
            slice_xy = rho[:, :, mid]
            dx = float(ds.domain_width[0].to_value()) / dims[0]
            dy = float(ds.domain_width[1].to_value()) / dims[1]
            x0 = float(ds.domain_left_edge[0].to_value())
            y0 = float(ds.domain_left_edge[1].to_value())
            x_edges = x0 + np.arange(dims[0] + 1) * dx
            y_edges = y0 + np.arange(dims[1] + 1) * dy
            plt.figure(figsize=(6, 4))
            plt.pcolormesh(x_edges, y_edges, slice_xy.T, shading="auto")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("Density Slice (z mid)")
            plt.colorbar(label="rho")
            plt.tight_layout()
            plt.savefig(plots_dir / "density_snapshot.png")
            plt.close()


if __name__ == "__main__":
    main()
