#!/usr/bin/env python3
import argparse
import csv
import json
import math
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


def filter_complete_diags(diags: list[Path]) -> tuple[list[Path], int]:
    complete = []
    skipped = 0
    for diag in diags:
        if (diag / "WarpXHeader").exists():
            complete.append(diag)
        else:
            skipped += 1
    return complete, skipped


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


def compute_centroid(rho: np.ndarray, x_centers, y_centers, z_centers) -> tuple[float, float, float] | None:
    if rho.ndim != 3:
        return None
    rho_abs = np.abs(rho)
    mass = float(np.sum(rho_abs))
    if mass <= 0.0:
        return None
    x_c = float(np.sum(rho_abs * x_centers[:, None, None]) / mass)
    y_c = float(np.sum(rho_abs * y_centers[None, :, None]) / mass)
    z_c = float(np.sum(rho_abs * z_centers[None, None, :]) / mass)
    return x_c, y_c, z_c


def linear_fit(x: np.ndarray, y: np.ndarray):
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    var = float(np.sum(dx * dx))
    if var <= 0.0:
        return None
    slope = float(np.sum(dx * dy) / var)
    intercept = float(y_mean - slope * x_mean)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def compute_tilt_slope(
    rho: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    bins_z: int,
    rho_frac: float,
):
    if rho.ndim != 3 or bins_z < 2:
        return None, None
    rho_abs = np.abs(rho)
    rho_peak = float(np.max(rho_abs))
    if rho_peak <= 0.0:
        return None, None
    rho_bg = float(np.percentile(rho_abs, 10.0))
    rho_thr = rho_bg + rho_frac * (rho_peak - rho_bg)
    mask_core = rho_abs >= rho_thr
    if not np.any(mask_core):
        return None, None
    z_min = float(z_centers[0])
    z_max = float(z_centers[-1])
    edges = np.linspace(z_min, z_max, bins_z + 1)
    x_vals = []
    y_vals = []
    z_vals = []
    for i in range(bins_z):
        z_lo = edges[i]
        z_hi = edges[i + 1]
        if i == bins_z - 1:
            z_mask = (z_centers >= z_lo) & (z_centers <= z_hi)
        else:
            z_mask = (z_centers >= z_lo) & (z_centers < z_hi)
        if not np.any(z_mask):
            continue
        rho_slice = rho_abs[:, :, z_mask]
        mask_slice = mask_core[:, :, z_mask]
        rho_slice = rho_slice * mask_slice
        mass = float(np.sum(rho_slice))
        if mass <= 0.0:
            continue
        x_c = float(np.sum(rho_slice * x_centers[:, None, None]) / mass)
        y_c = float(np.sum(rho_slice * y_centers[None, :, None]) / mass)
        z_c = float(np.mean(z_centers[z_mask]))
        x_vals.append(x_c)
        y_vals.append(y_c)
        z_vals.append(z_c)
    if len(z_vals) < 2:
        return None, len(z_vals)
    z_vals = np.array(z_vals)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    fit_x = linear_fit(z_vals, x_vals)
    fit_y = linear_fit(z_vals, y_vals)
    if fit_x is None or fit_y is None:
        return None, len(z_vals)
    slope_x = float(fit_x[0])
    slope_y = float(fit_y[0])
    tilt_slope = float(np.hypot(slope_x, slope_y))
    return tilt_slope, len(z_vals)


def first_finite(values):
    for val in values:
        if val is None:
            continue
        if np.isfinite(val):
            return val
    return None


def last_finite(values):
    for val in reversed(values):
        if val is None:
            continue
        if np.isfinite(val):
            return val
    return None


def first_finite_after(values, start_index):
    if not values:
        return None
    start_index = max(0, min(start_index, len(values) - 1))
    for val in values[start_index:]:
        if val is None:
            continue
        if np.isfinite(val):
            return val
    return None


def first_finite_with_index(values, start_index):
    if not values:
        return None, None
    start_index = max(0, min(start_index, len(values) - 1))
    for idx in range(start_index, len(values)):
        val = values[idx]
        if val is None:
            continue
        if np.isfinite(val):
            return val, idx
    return None, None


def weighted_mean(points, weights):
    total = float(np.sum(weights))
    if total <= 0.0:
        return None
    return np.sum(points * weights[:, None], axis=0) / total


def weighted_inertia(points, weights, center):
    if center is None:
        return None
    diffs = points - center
    return float(np.sum(weights * np.sum(diffs * diffs, axis=1)))


def compute_merge_time(times, indicator, threshold, persist):
    if not times or not indicator or threshold is None:
        return None, None
    persist = max(1, int(persist))
    streak = 0
    for t, val in zip(times, indicator):
        if val is None or not np.isfinite(val):
            streak = 0
            continue
        if val <= threshold:
            streak += 1
            if streak >= persist:
                return t, val
        else:
            streak = 0
    return None, None


def compute_sep_stats(times, series):
    sep_initial = first_finite(series)
    sep_final = last_finite(series)
    sep_min = None
    sep_min_time = None
    for t, sep in zip(times, series or []):
        if sep is None or not np.isfinite(sep):
            continue
        if sep_min is None or sep < sep_min:
            sep_min = sep
            sep_min_time = t
    sep_ratio = None
    if sep_initial is not None and sep_final is not None and sep_initial > 0.0:
        sep_ratio = float(sep_final / sep_initial)
    return sep_initial, sep_final, sep_min, sep_min_time, sep_ratio


def compute_linear_slope(x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return None
    fit = linear_fit(x[mask], y[mask])
    if fit is None:
        return None
    return float(fit[0])


def compute_sep_trend_metrics(
    times,
    series,
    ignore_samples=50,
    ignore_frac=0.05,
    window=25,
    consecutive=10,
    min_delta=1.0e-4,
):
    if not series:
        return None, None, None
    if times and len(times) == len(series):
        x_vals = times
    else:
        x_vals = list(range(len(series)))
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(series, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None, None, None
    x = x[mask]
    y = y[mask]
    n = len(y)
    ignore_n = max(int(ignore_samples), int(math.ceil(ignore_frac * n)))
    if ignore_n >= n - 1:
        return None, None, None
    start = ignore_n
    remain = n - start
    early_n = max(2, int(math.ceil(0.2 * remain)))
    late_n = max(2, int(math.ceil(0.2 * remain)))
    early_slice = slice(start, min(n, start + early_n))
    late_slice = slice(max(start, n - late_n), n)
    slope_early = compute_linear_slope(x[early_slice], y[early_slice])
    slope_late = compute_linear_slope(x[late_slice], y[late_slice])

    rebound = None
    if remain >= window + consecutive - 1 and window >= 2 and consecutive >= 1:
        slopes = []
        for i in range(start, n - window + 1):
            slopes.append(compute_linear_slope(x[i : i + window], y[i : i + window]))
        rebound = False
        consec = 0
        run_start = None
        for idx, slope in enumerate(slopes):
            if slope is not None and slope > 0:
                if consec == 0:
                    run_start = idx
                consec += 1
                if consec >= consecutive and run_start is not None:
                    first_idx = start + run_start
                    last_idx = start + run_start + window + consecutive - 2
                    if last_idx < n:
                        delta = y[last_idx] - y[first_idx]
                        if delta > min_delta:
                            rebound = True
                            break
            else:
                consec = 0
                run_start = None
    return slope_early, slope_late, rebound


def compute_seedmask_series(diags, sep_mode, plane_point, plane_normal):
    times = []
    series = []
    left_centroids = []
    right_centroids = []
    point = np.array(plane_point, dtype=float)
    normal = np.array(plane_normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        return times, series, left_centroids, right_centroids
    normal = normal / norm
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
        rho_bg = float(np.percentile(rho_abs, 10.0))
        rho_thr = rho_bg + 0.4 * (rho_peak - rho_bg)
        core_mask = rho_abs >= rho_thr
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
        plane = (
            (x_coords - point[0]) * normal[0]
            + (y_coords - point[1]) * normal[1]
            + (z_coords - point[2]) * normal[2]
        )
        left_mask = core_mask & (plane <= 0.0)
        right_mask = core_mask & (plane >= 0.0)
        left_mass = float(np.sum(rho_abs * left_mask))
        right_mass = float(np.sum(rho_abs * right_mask))
        if left_mass <= 0.0 or right_mass <= 0.0:
            continue
        left = (
            float(np.sum(rho_abs * left_mask * x_coords) / left_mass),
            float(np.sum(rho_abs * left_mask * y_coords) / left_mass),
            float(np.sum(rho_abs * left_mask * z_coords) / left_mass),
        )
        right = (
            float(np.sum(rho_abs * right_mask * x_coords) / right_mass),
            float(np.sum(rho_abs * right_mask * y_coords) / right_mass),
            float(np.sum(rho_abs * right_mask * z_coords) / right_mass),
        )
        if sep_mode == "x":
            sep_val = float(abs(left[0] - right[0]))
        elif sep_mode == "xy":
            sep_val = float(np.hypot(left[0] - right[0], left[1] - right[1]))
        else:
            sep_val = float(np.linalg.norm(np.array(left) - np.array(right)))
        times.append(t)
        series.append(sep_val)
        left_centroids.append(left)
        right_centroids.append(right)
    return times, series, left_centroids, right_centroids


def kmeans_two_clusters(points, weights, centers, n_iter):
    centers = np.array(centers, dtype=float)
    for _ in range(n_iter):
        d0 = np.sum((points - centers[0]) ** 2, axis=1)
        d1 = np.sum((points - centers[1]) ** 2, axis=1)
        mask1 = d1 < d0
        mask0 = ~mask1
        if not np.any(mask0) or not np.any(mask1):
            break
        c0 = weighted_mean(points[mask0], weights[mask0])
        c1 = weighted_mean(points[mask1], weights[mask1])
        if c0 is None or c1 is None:
            break
        centers[0] = c0
        centers[1] = c1
        # enforce deterministic ordering by x
        if centers[0][0] > centers[1][0]:
            centers = centers[[1, 0]]
    return centers


def compute_kmeans_series(
    diags,
    sep_mode,
    max_points,
    n_iter,
    threshold_frac,
    bg_quantile,
    min_ratio,
    warm_start,
    kmeans_coords,
):
    times = []
    sep_series = []
    centers_series = []
    ratio_series = []
    inertia_reduction_series = []
    inertia_series = []
    prev_centers = None

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
        rho_bg = float(np.percentile(rho_abs, bg_quantile))
        rho_thr = rho_bg + threshold_frac * (rho_peak - rho_bg)
        mask = rho_abs >= rho_thr
        if not np.any(mask):
            continue

        idx = np.argwhere(mask)
        weights = rho_abs[mask].astype(float)
        if idx.shape[0] > max_points:
            top_idx = np.argpartition(weights, -max_points)[-max_points:]
            idx = idx[top_idx]
            weights = weights[top_idx]

        dx = float(ds.domain_width[0].to_value()) / dims[0]
        dy = float(ds.domain_width[1].to_value()) / dims[1]
        dz = float(ds.domain_width[2].to_value()) / dims[2]
        x0 = float(ds.domain_left_edge[0].to_value())
        y0 = float(ds.domain_left_edge[1].to_value())
        z0 = float(ds.domain_left_edge[2].to_value())
        x_centers = x0 + (idx[:, 0] + 0.5) * dx
        y_centers = y0 + (idx[:, 1] + 0.5) * dy
        z_centers = z0 + (idx[:, 2] + 0.5) * dz
        points_full = np.column_stack([x_centers, y_centers, z_centers])
        if kmeans_coords == "x":
            points = points_full[:, [0]]
        elif kmeans_coords == "xy":
            points = points_full[:, [0, 1]]
        else:
            points = points_full

        if prev_centers is not None and warm_start:
            centers = prev_centers
        else:
            left_mask = points[:, 0] < 0.0
            right_mask = points[:, 0] > 0.0
            if np.any(left_mask) and np.any(right_mask):
                c0 = weighted_mean(points[left_mask], weights[left_mask])
                c1 = weighted_mean(points[right_mask], weights[right_mask])
                centers = np.array([c0, c1])
            else:
                imin = np.argmin(points[:, 0])
                imax = np.argmax(points[:, 0])
                centers = np.array([points[imin], points[imax]])

        centers = kmeans_two_clusters(points, weights, centers, n_iter)
        prev_centers = centers

        # Assign points to clusters to compute weight ratio and inertia reduction.
        d0 = np.sum((points - centers[0]) ** 2, axis=1)
        d1 = np.sum((points - centers[1]) ** 2, axis=1)
        mask1 = d1 < d0
        mask0 = ~mask1
        w0 = float(np.sum(weights[mask0]))
        w1 = float(np.sum(weights[mask1]))
        ratio = min(w0, w1) / max(w0, w1) if max(w0, w1) > 0.0 else 0.0
        inertia_two = float(np.sum(weights * np.minimum(d0, d1)))
        inertia_one = weighted_inertia(points, weights, weighted_mean(points, weights))
        inertia_reduction = None
        if inertia_one is not None and inertia_one > 0.0:
            inertia_reduction = (inertia_one - inertia_two) / inertia_one

        c0_full = weighted_mean(points_full[mask0], weights[mask0])
        c1_full = weighted_mean(points_full[mask1], weights[mask1])
        centers_full = np.array([c0_full, c1_full])

        if centers_full[0][0] > centers_full[1][0]:
            centers_full = centers_full[[1, 0]]
            centers = centers[[1, 0]]
            w0, w1 = w1, w0
            ratio = min(w0, w1) / max(w0, w1) if max(w0, w1) > 0.0 else 0.0

        if sep_mode == "x":
            sep_val = float(abs(centers_full[0][0] - centers_full[1][0]))
        elif sep_mode == "xy":
            sep_val = float(
                np.hypot(
                    centers_full[0][0] - centers_full[1][0],
                    centers_full[0][1] - centers_full[1][1],
                )
            )
        else:
            sep_val = float(np.linalg.norm(centers_full[0] - centers_full[1]))

        times.append(t)
        centers_series.append(centers_full)
        ratio_series.append(ratio)
        sep_series.append(sep_val if ratio >= min_ratio else np.nan)
        if ratio >= min_ratio and inertia_reduction is not None:
            inertia_reduction_series.append(inertia_reduction)
        else:
            inertia_reduction_series.append(np.nan)
        inertia_series.append(inertia_two if ratio >= min_ratio else np.nan)

    return (
        times,
        sep_series,
        centers_series,
        ratio_series,
        inertia_reduction_series,
        inertia_series,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hybrid merge+tilt H3a case.")
    parser.add_argument("--diag-dir", required=True, help="WarpX diag directory (contains diag*).")
    parser.add_argument("--metadata", required=True, help="WarpX run metadata JSON.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--sep-csv", required=True, help="Output CSV for separation series.")
    parser.add_argument("--tilt-csv", required=True, help="Output CSV for post-merge tilt series.")
    parser.add_argument(
        "--tilt-metric",
        choices=["centroid", "slope"],
        default="centroid",
        help="Tilt metric to record after merge.",
    )
    parser.add_argument("--tilt-z-bins", type=int, default=16, help="Z bins for tilt slope fit.")
    parser.add_argument(
        "--tilt-rho-frac",
        type=float,
        default=0.4,
        help="Core rho fraction for tilt slope masking.",
    )
    parser.add_argument("--merge-frac", type=float, default=0.25, help="Merge threshold as fraction of initial sep.")
    parser.add_argument(
        "--merge-floor-mult",
        type=float,
        default=1.6,
        help="Minimum merge threshold as multiple of blob sigma (to avoid split-centroid floor).",
    )
    parser.add_argument("--guard-outputs", type=int, default=2, help="Outputs to skip after merge before tilt window.")
    parser.add_argument("--sep-mode", type=str, default="x", choices=["x", "xy", "xyz"], help="Separation mode.")
    parser.add_argument(
        "--merge-method",
        type=str,
        default="xsplit",
        choices=["xsplit", "kmeans"],
        help="Merge detection method.",
    )
    parser.add_argument("--kmeans-max-points", type=int, default=50000, help="Max points for k-means.")
    parser.add_argument("--kmeans-iter", type=int, default=10, help="k-means iterations.")
    parser.add_argument("--kmeans-threshold-frac", type=float, default=0.6, help="k-means rho threshold fraction.")
    parser.add_argument("--kmeans-bg-quantile", type=float, default=10.0, help="k-means background quantile.")
    parser.add_argument("--kmeans-min-ratio", type=float, default=0.2, help="Min cluster weight ratio.")
    parser.add_argument("--kmeans-warm-start", type=int, default=1, help="Warm-start k-means (1/0).")
    parser.add_argument(
        "--kmeans-coords",
        type=str,
        default="xyz",
        choices=["x", "xy", "xyz"],
        help="Coordinate subset for k-means clustering (default: xyz).",
    )
    parser.add_argument(
        "--kmeans-merge-criterion",
        type=str,
        default="sep",
        choices=["sep", "inertia", "inertia_ratio"],
        help="k-means merge criterion.",
    )
    parser.add_argument(
        "--kmeans-inertia-thr",
        type=float,
        default=0.2,
        help="Merge threshold for inertia reduction (lower means more merged).",
    )
    parser.add_argument(
        "--kmeans-inertia-ratio-thr",
        type=float,
        default=0.9,
        help="Merge threshold for inertia ratio I/I0 (lower means more merged).",
    )
    parser.add_argument(
        "--kmeans-inertia-guard-outputs",
        type=int,
        default=1,
        help="Outputs to skip before setting I0 for inertia ratio.",
    )
    parser.add_argument(
        "--merge-persist",
        type=int,
        default=1,
        help="Consecutive outputs required to declare merge.",
    )
    parser.add_argument(
        "--merge-sep-frac",
        type=float,
        default=0.7,
        help="Seedmask merge threshold as fraction of seedmask start separation.",
    )
    parser.add_argument(
        "--diag-stride",
        type=int,
        default=1,
        help="Process every Nth diag output to reduce runtime (1 = use all).",
    )
    args = parser.parse_args()

    diag_root = Path(args.diag_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    meta = extract_meta(Path(args.metadata))
    run_args = meta.get("args", {})
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []
    opmd_double_seed = meta.get("opmd_double_seed") or {}
    drift_meta = opmd_double_seed.get("drift_meta") or {}
    if not drift_meta:
        drift_meta = extract_meta(Path(args.metadata).parent / "drift_meta.json")
    drift_unit = None
    drift_unit_raw = drift_meta.get("drift_unit") or drift_meta.get("drift_dir")
    if drift_unit_raw is not None:
        try:
            drift_unit = np.array(drift_unit_raw, dtype=float)
            drift_norm = float(np.linalg.norm(drift_unit))
            if drift_norm > 0.0:
                drift_unit = drift_unit / drift_norm
            else:
                drift_unit = None
        except Exception:
            drift_unit = None
    seedmask_plane = None
    seedmask_source = None
    com_a = drift_meta.get("com_a")
    com_b = drift_meta.get("com_b")
    if com_a is not None and com_b is not None:
        try:
            com_a = np.array(com_a, dtype=float)
            com_b = np.array(com_b, dtype=float)
            sep_vec = com_b - com_a
            sep_norm = float(np.linalg.norm(sep_vec))
            if sep_norm > 0.0:
                sep_unit = (sep_vec / sep_norm).tolist()
                mid = (0.5 * (com_a + com_b)).tolist()
                seedmask_plane = {"point": mid, "normal": sep_unit}
                seedmask_source = drift_meta.get("mask_source") or "opmd_double_seed"
        except Exception:
            seedmask_plane = None

    sep_times = []
    sep_series = []
    left_centroids = []
    right_centroids = []
    sep_times_xsplit = []
    sep_series_xsplit = []
    left_centroids_xsplit = []
    right_centroids_xsplit = []
    sep_times_group = []
    sep_series_group = []
    left_centroids_group = []
    right_centroids_group = []
    sep_steps_group = []
    global_times = []
    global_centroids = []

    for rec in records:
        split = rec.get("split_centroids")
        if split and not split.get("error"):
            t = rec.get("time")
            left = split.get("left")
            right = split.get("right")
            if t is not None and left and right:
                sep_key = "sep_x" if args.sep_mode == "x" else ("sep_xy" if args.sep_mode == "xy" else "sep_xyz")
                sep_val = split.get(sep_key)
                if sep_val is not None:
                    sep_times_xsplit.append(t)
                    sep_series_xsplit.append(sep_val)
                    left_centroids_xsplit.append((left.get("x"), left.get("y"), left.get("z")))
                    right_centroids_xsplit.append((right.get("x"), right.get("y"), right.get("z")))
        group = rec.get("group_centroids")
        if group and not group.get("error"):
            t = rec.get("time")
            step = rec.get("step")
            left = group.get("left")
            right = group.get("right")
            if t is not None and left and right:
                sep_key = "sep_x" if args.sep_mode == "x" else ("sep_xy" if args.sep_mode == "xy" else "sep_xyz")
                sep_val = group.get(sep_key)
                if sep_val is not None:
                    sep_times_group.append(t)
                    sep_series_group.append(sep_val)
                    left_centroids_group.append((left.get("x"), left.get("y"), left.get("z")))
                    right_centroids_group.append((right.get("x"), right.get("y"), right.get("z")))
                    sep_steps_group.append(step)
        global_c = rec.get("global_centroid")
        if global_c and not global_c.get("error"):
            t = rec.get("time")
            if t is not None:
                global_times.append(t)
                global_centroids.append((global_c.get("x"), global_c.get("y"), global_c.get("z")))

    diags_all_raw = list_diags(diag_root)
    diags_all, incomplete_skipped = filter_complete_diags(diags_all_raw)
    diag_stride = max(1, int(args.diag_stride))
    diags = diags_all[::diag_stride] if diag_stride > 1 else diags_all
    if diag_stride > 1 and len(diags) < 30 and len(diags_all) >= 30:
        diag_stride = 1
        diags = diags_all
    num_outputs_total = len(diags_all_raw)
    num_outputs_complete = len(diags_all)
    num_outputs_used = len(diags)

    sep_times_seedmask = []
    sep_series_seedmask = []
    left_centroids_seedmask = []
    right_centroids_seedmask = []
    sep_series_seedmask_proj = []
    sep_series_seedmask_norm = []
    sep_series_seedmask_x = []
    sep_series_seedmask_y = []
    sep_series_seedmask_z = []
    sep_steps_seedmask = []
    if sep_times_group:
        seedmask_source = "id_groups"
        seedmask_plane = None
        sep_times_seedmask = sep_times_group
        sep_series_seedmask = sep_series_group
        left_centroids_seedmask = left_centroids_group
        right_centroids_seedmask = right_centroids_group
        sep_steps_seedmask = sep_steps_group
    elif seedmask_plane is not None:
        (
            sep_times_seedmask,
            sep_series_seedmask,
            left_centroids_seedmask,
            right_centroids_seedmask,
        ) = compute_seedmask_series(
            diags, args.sep_mode, seedmask_plane["point"], seedmask_plane["normal"]
        )
    if left_centroids_seedmask and right_centroids_seedmask:
        for left, right in zip(left_centroids_seedmask, right_centroids_seedmask):
            if left is None or right is None:
                continue
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            dz = right[2] - left[2]
            sep_series_seedmask_x.append(dx)
            sep_series_seedmask_y.append(dy)
            sep_series_seedmask_z.append(dz)
            sep_series_seedmask_norm.append(float(np.linalg.norm([dx, dy, dz])))
            if drift_unit is not None:
                sep_series_seedmask_proj.append(float(np.dot([dx, dy, dz], drift_unit)))
            else:
                sep_series_seedmask_proj.append(None)

    # k-means separation series (optional)
    sep_times_kmeans = []
    sep_series_kmeans = []
    centers_kmeans = []
    ratios_kmeans = []
    inertia_reduction_kmeans = []
    inertia_kmeans = []
    if args.merge_method == "kmeans":
        (
            sep_times_kmeans,
            sep_series_kmeans,
            centers_kmeans,
            ratios_kmeans,
            inertia_reduction_kmeans,
            inertia_kmeans,
        ) = compute_kmeans_series(
            diags,
            args.sep_mode,
            args.kmeans_max_points,
            args.kmeans_iter,
            args.kmeans_threshold_frac,
            args.kmeans_bg_quantile,
            args.kmeans_min_ratio,
            bool(args.kmeans_warm_start),
            args.kmeans_coords,
        )

    if args.merge_method == "xsplit":
        sep_times = sep_times_xsplit
        sep_series = sep_series_xsplit
        left_centroids = left_centroids_xsplit
        right_centroids = right_centroids_xsplit
    else:
        sep_times = sep_times_kmeans
        sep_series = sep_series_kmeans
        if centers_kmeans:
            left_centroids = [tuple(c[0]) for c in centers_kmeans]
            right_centroids = [tuple(c[1]) for c in centers_kmeans]

    inertia_ratio_kmeans = []
    inertia0 = None
    inertia0_time = None
    if inertia_kmeans:
        inertia0, inertia0_index = first_finite_with_index(
            inertia_kmeans, args.kmeans_inertia_guard_outputs
        )
        if inertia0_index is not None and inertia0_index < len(sep_times_kmeans):
            inertia0_time = sep_times_kmeans[inertia0_index]
        if inertia0 is not None and inertia0 > 0.0:
            for val in inertia_kmeans:
                if val is None or not np.isfinite(val):
                    inertia_ratio_kmeans.append(np.nan)
                else:
                    inertia_ratio_kmeans.append(float(val / inertia0))
        else:
            inertia_ratio_kmeans = [np.nan for _ in inertia_kmeans]

    if not sep_series:
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
            rho_bg = float(np.percentile(rho_abs, 10.0))
            rho_thr = rho_bg + 0.4 * (rho_peak - rho_bg)
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
            left_mass = float(np.sum(rho_abs * left_mask))
            right_mass = float(np.sum(rho_abs * right_mask))
            if left_mass <= 0.0 or right_mass <= 0.0:
                continue
            left = (
                float(np.sum(rho_abs * left_mask * x_coords) / left_mass),
                float(np.sum(rho_abs * left_mask * y_coords) / left_mass),
                float(np.sum(rho_abs * left_mask * z_coords) / left_mass),
            )
            right = (
                float(np.sum(rho_abs * right_mask * x_coords) / right_mass),
                float(np.sum(rho_abs * right_mask * y_coords) / right_mass),
                float(np.sum(rho_abs * right_mask * z_coords) / right_mass),
            )
            if args.sep_mode == "x":
                sep_val = float(abs(left[0] - right[0]))
            elif args.sep_mode == "xy":
                sep_val = float(np.hypot(left[0] - right[0], left[1] - right[1]))
            else:
                sep_val = float(np.linalg.norm(np.array(left) - np.array(right)))
            sep_times.append(t)
            sep_series.append(sep_val)
            left_centroids.append(left)
            right_centroids.append(right)

    if not global_centroids:
        for diag in diags:
            ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
            ad = ds.all_data()
            t = float(ds.current_time.to_value())
            dims = tuple(int(x) for x in ds.domain_dimensions)
            rho = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
            if rho.ndim != 3:
                continue
            dx = float(ds.domain_width[0].to_value()) / dims[0]
            dy = float(ds.domain_width[1].to_value()) / dims[1]
            dz = float(ds.domain_width[2].to_value()) / dims[2]
            x0 = float(ds.domain_left_edge[0].to_value())
            y0 = float(ds.domain_left_edge[1].to_value())
            z0 = float(ds.domain_left_edge[2].to_value())
            x_centers = x0 + (np.arange(dims[0]) + 0.5) * dx
            y_centers = y0 + (np.arange(dims[1]) + 0.5) * dy
            z_centers = z0 + (np.arange(dims[2]) + 0.5) * dz
            centroid = compute_centroid(rho, x_centers, y_centers, z_centers)
            if centroid is None:
                continue
            global_times.append(t)
            global_centroids.append(centroid)

    # Separation metrics
    sep_initial, sep_final, sep_min, sep_min_time, sep_ratio = compute_sep_stats(sep_times, sep_series)
    sep_initial_xsplit, sep_final_xsplit, sep_min_xsplit, sep_min_time_xsplit, sep_ratio_xsplit = compute_sep_stats(
        sep_times_xsplit, sep_series_xsplit
    )
    sep_initial_kmeans, sep_final_kmeans, sep_min_kmeans, sep_min_time_kmeans, sep_ratio_kmeans = compute_sep_stats(
        sep_times_kmeans, sep_series_kmeans
    )
    sep_initial_seedmask, sep_final_seedmask, sep_min_seedmask, sep_min_time_seedmask, sep_ratio_seedmask = (
        compute_sep_stats(sep_times_seedmask, sep_series_seedmask)
    )
    sep_proj_start, sep_proj_end, sep_proj_min, sep_proj_min_time, sep_proj_ratio = compute_sep_stats(
        sep_times_seedmask, sep_series_seedmask_proj
    )
    sep_norm_start, sep_norm_end, sep_norm_min, sep_norm_min_time, sep_norm_ratio = compute_sep_stats(
        sep_times_seedmask, sep_series_seedmask_norm
    )
    sep_x_start, sep_x_end, sep_x_min, sep_x_min_time, sep_x_ratio = compute_sep_stats(
        sep_times_seedmask, sep_series_seedmask_x
    )
    sep_y_start, sep_y_end, sep_y_min, sep_y_min_time, sep_y_ratio = compute_sep_stats(
        sep_times_seedmask, sep_series_seedmask_y
    )
    sep_z_start, sep_z_end, sep_z_min, sep_z_min_time, sep_z_ratio = compute_sep_stats(
        sep_times_seedmask, sep_series_seedmask_z
    )
    sep_proj_delta = (
        sep_proj_end - sep_proj_start
        if sep_proj_start is not None and sep_proj_end is not None
        else None
    )
    sep_norm_delta = (
        sep_norm_end - sep_norm_start
        if sep_norm_start is not None and sep_norm_end is not None
        else None
    )
    sep_x_delta = (
        sep_x_end - sep_x_start
        if sep_x_start is not None and sep_x_end is not None
        else None
    )
    sep_y_delta = (
        sep_y_end - sep_y_start
        if sep_y_start is not None and sep_y_end is not None
        else None
    )
    sep_z_delta = (
        sep_z_end - sep_z_start
        if sep_z_start is not None and sep_z_end is not None
        else None
    )
    (
        sep_trend_slope_early,
        sep_trend_slope_late,
        sep_rebound_flag_robust,
    ) = compute_sep_trend_metrics(
        sep_steps_seedmask
        if any(step is not None for step in sep_steps_seedmask)
        else sep_times_seedmask,
        sep_series_seedmask_norm,
        ignore_samples=50,
        ignore_frac=0.05,
        window=25,
        consecutive=10,
        min_delta=1.0e-4,
    )
    s_start = sep_initial_seedmask
    s_min = sep_min_seedmask
    t_at_s_min = sep_min_time_seedmask
    merge_sep_frac = float(args.merge_sep_frac)
    merge_sep_thresh = None
    merge_time_seedmask = None
    merge_indicator_at_seedmask = None
    if s_start is not None and s_start > 0.0:
        merge_sep_thresh = merge_sep_frac * s_start
        merge_time_seedmask, merge_indicator_at_seedmask = compute_merge_time(
            sep_times_seedmask, sep_series_seedmask, merge_sep_thresh, 3
        )
    merge_time_exists_seedmask = merge_time_seedmask is not None

    # Compute merge thresholds and times for both detectors.
    merge_floor_xsplit = None
    merge_thr_xsplit = None
    if sep_initial_xsplit is not None and sep_initial_xsplit > 0.0:
        sigma_val = None
        blobs_cfg = run_args.get("blobs") or []
        if blobs_cfg:
            sigmas = [float(blob.get("sigma")) for blob in blobs_cfg if blob.get("sigma") is not None]
            if sigmas:
                sigma_val = float(np.mean(sigmas))
        if sigma_val is None:
            sigma_val = run_args.get("blob_sigma")
        merge_floor_xsplit = args.merge_floor_mult * float(sigma_val) if sigma_val else 0.0
        merge_thr_xsplit = max(args.merge_frac * sep_initial_xsplit, merge_floor_xsplit)

    merge_indicator_kmeans = None
    merge_indicator_def_kmeans = None
    merge_thr_kmeans = None
    if sep_times_kmeans:
        if args.kmeans_merge_criterion == "inertia":
            merge_indicator_kmeans = inertia_reduction_kmeans
            merge_indicator_def_kmeans = "kmeans_inertia_reduction"
            merge_thr_kmeans = args.kmeans_inertia_thr
        elif args.kmeans_merge_criterion == "inertia_ratio":
            merge_indicator_kmeans = inertia_ratio_kmeans
            merge_indicator_def_kmeans = "kmeans_inertia_ratio"
            merge_thr_kmeans = args.kmeans_inertia_ratio_thr
        else:
            merge_indicator_kmeans = sep_series_kmeans
            merge_indicator_def_kmeans = f"separation_{args.sep_mode}"
            if sep_initial_kmeans is not None and sep_initial_kmeans > 0.0:
                merge_thr_kmeans = args.merge_frac * sep_initial_kmeans

    merge_time_kmeans, merge_indicator_at_kmeans = compute_merge_time(
        sep_times_kmeans, merge_indicator_kmeans, merge_thr_kmeans, args.merge_persist
    )
    merge_time_xsplit, merge_indicator_at_xsplit = compute_merge_time(
        sep_times_xsplit, sep_series_xsplit, merge_thr_xsplit, args.merge_persist
    )

    merge_time = None
    merge_time_exists = False
    merge_time_frac = None
    merge_thr = None
    merge_floor = None
    merge_indicator = None
    merge_indicator_at_merge = None
    merge_indicator_def = None
    merge_detector_used = None
    merge_fallback_reason = None

    if args.merge_method == "kmeans":
        if merge_time_kmeans is not None:
            merge_time = merge_time_kmeans
            merge_indicator_at_merge = merge_indicator_at_kmeans
            merge_indicator = merge_indicator_kmeans
            merge_indicator_def = merge_indicator_def_kmeans
            merge_thr = merge_thr_kmeans
            merge_detector_used = merge_indicator_def_kmeans
        elif merge_time_xsplit is not None:
            merge_time = merge_time_xsplit
            merge_indicator_at_merge = merge_indicator_at_xsplit
            merge_indicator = sep_series_xsplit
            merge_indicator_def = f"separation_{args.sep_mode}"
            merge_thr = merge_thr_xsplit
            merge_floor = merge_floor_xsplit
            merge_detector_used = "xsplit_fallback"
            merge_fallback_reason = "kmeans_merge_time_missing"
        else:
            merge_detector_used = merge_indicator_def_kmeans or "kmeans_unavailable"
            merge_fallback_reason = "kmeans_merge_time_missing"
    else:
        if merge_time_xsplit is not None:
            merge_time = merge_time_xsplit
            merge_indicator_at_merge = merge_indicator_at_xsplit
            merge_indicator = sep_series_xsplit
            merge_indicator_def = f"separation_{args.sep_mode}"
            merge_thr = merge_thr_xsplit
            merge_floor = merge_floor_xsplit
            merge_detector_used = "xsplit"
        else:
            merge_detector_used = "xsplit"

    merge_time_exists = merge_time is not None

    t_end = None
    t_end_candidates = []
    if sep_times_xsplit:
        t_end_candidates.append(sep_times_xsplit[-1])
    if sep_times_kmeans:
        t_end_candidates.append(sep_times_kmeans[-1])
    if global_times:
        t_end_candidates.append(global_times[-1])
    if t_end_candidates:
        t_end = max(t_end_candidates)

    if merge_time is not None and t_end is not None and t_end > 0.0:
        merge_time_frac = float(merge_time / t_end)
    merge_time_frac_seedmask = None
    if merge_time_seedmask is not None and t_end is not None and t_end > 0.0:
        merge_time_frac_seedmask = float(merge_time_seedmask / t_end)

    merge_time_delta_frac = None
    merge_detector_disagreement = None
    if merge_time_kmeans is not None and merge_time_xsplit is not None and t_end is not None and t_end > 0.0:
        merge_time_delta_frac = float(abs(merge_time_kmeans - merge_time_xsplit) / t_end)
        merge_detector_disagreement = merge_time_delta_frac > 0.2

    # Tilt post-merge window
    tilt_post_times = []
    tilt_post_amp = []
    tilt_post_merge_samples = 0
    tilt_post_merge_amp_max = None
    tilt_post_merge_no_nan = True
    tilt_metric_used = args.tilt_metric
    tilt_centroid_times = []
    tilt_centroid_amp = []
    tilt_slope_bins_used = []

    dt = run_args.get("dt")
    diag_period = run_args.get("diag_period", 1)
    guard_dt = None
    if dt is not None:
        guard_dt = float(dt) * max(1, int(diag_period)) * max(0, int(args.guard_outputs))

    start_time = None
    if merge_time is not None:
        start_time = merge_time + (guard_dt if guard_dt is not None else 0.0)

    if start_time is not None and global_times:
        for t, centroid in zip(global_times, global_centroids):
            if t < start_time:
                continue
            amp = float(np.hypot(centroid[0], centroid[1]))
            tilt_centroid_times.append(t)
            tilt_centroid_amp.append(amp)

    if tilt_metric_used == "centroid":
        tilt_post_times = tilt_centroid_times
        tilt_post_amp = tilt_centroid_amp
    else:
        for diag in diags:
            ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
            ad = ds.all_data()
            t = float(ds.current_time.to_value())
            if start_time is not None and t < start_time:
                continue
            dims = tuple(int(x) for x in ds.domain_dimensions)
            rho = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
            if rho.ndim != 3:
                continue
            dx = float(ds.domain_width[0].to_value()) / dims[0]
            dy = float(ds.domain_width[1].to_value()) / dims[1]
            dz = float(ds.domain_width[2].to_value()) / dims[2]
            x0 = float(ds.domain_left_edge[0].to_value())
            y0 = float(ds.domain_left_edge[1].to_value())
            z0 = float(ds.domain_left_edge[2].to_value())
            x_centers = x0 + (np.arange(dims[0]) + 0.5) * dx
            y_centers = y0 + (np.arange(dims[1]) + 0.5) * dy
            z_centers = z0 + (np.arange(dims[2]) + 0.5) * dz
            slope, bins_used = compute_tilt_slope(
                rho, x_centers, y_centers, z_centers, int(args.tilt_z_bins), float(args.tilt_rho_frac)
            )
            if slope is None:
                continue
            tilt_post_times.append(t)
            tilt_post_amp.append(float(slope))
            if bins_used is not None:
                tilt_slope_bins_used.append(int(bins_used))

    tilt_post_merge_samples = len(tilt_post_times)
    if tilt_post_amp:
        tilt_post_merge_amp_max = max(tilt_post_amp)
        for val in tilt_post_amp:
            if val is None or not np.isfinite(val):
                tilt_post_merge_no_nan = False
    else:
        tilt_post_merge_no_nan = False
    tilt_centroid_amp_max = max(tilt_centroid_amp) if tilt_centroid_amp else None
    tilt_centroid_samples = len(tilt_centroid_times)
    tilt_slope_bins_min = min(tilt_slope_bins_used) if tilt_slope_bins_used else None
    tilt_slope_bins_max = max(tilt_slope_bins_used) if tilt_slope_bins_used else None
    tilt_slope_bins_mean = (
        float(np.mean(tilt_slope_bins_used)) if tilt_slope_bins_used else None
    )

    # NaN checks
    no_nan = True
    for val in (
        sep_initial,
        sep_final,
        sep_ratio,
        sep_min,
        merge_time_frac,
        merge_time_frac_seedmask,
    ):
        if val is None:
            continue
        if not np.isfinite(val):
            no_nan = False
    if tilt_post_merge_amp_max is not None and not np.isfinite(tilt_post_merge_amp_max):
        no_nan = False
    if not tilt_post_merge_no_nan:
        no_nan = False

    # Energy series
    energy_series = []
    if diags:
        for diag in diags:
            ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
            ad = ds.all_data()
            t = float(ds.current_time.to_value())
            energy = total_field_energy(ds, ad)
            if energy is not None:
                energy_series.append({"time_s": t, "field_energy": energy})

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

    com_sep_source = None
    if sep_series_seedmask:
        com_sep_start = sep_initial_seedmask
        com_sep_end = sep_final_seedmask
        com_sep_source = "seedmask_sep"
    else:
        com_sep_start = sep_initial
        com_sep_end = sep_final
        com_sep_source = args.merge_method
    com_sep_delta = None
    if com_sep_start is not None and com_sep_end is not None:
        com_sep_delta = com_sep_end - com_sep_start

    # Metrics
    # Completion check
    ran_to_completion = None
    sim_time_reached = records[-1].get("time") if records else None
    last_step = records[-1].get("step") if records else None
    max_steps = run_args.get("max_steps")
    dt = run_args.get("dt")
    diag_period = run_args.get("diag_period", 1)
    if max_steps is not None:
        if last_step is not None:
            ran_to_completion = last_step >= (max_steps - 1)
        elif sim_time_reached is not None and dt is not None:
            expected = (max_steps - 1) * dt
            slack = dt * max(1, int(diag_period))
            ran_to_completion = sim_time_reached >= (expected - slack)
    if ran_to_completion is None:
        ran_to_completion = bool(sep_times or global_times)

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": num_outputs_total,
        "num_outputs_complete": num_outputs_complete,
        "num_outputs_used": num_outputs_used,
        "incomplete_diag_skipped": incomplete_skipped,
        "diag_stride": diag_stride,
        "no_nan_in_metrics": no_nan,
        "drop_breach": (monitor.get("drop_breach") if monitor else None),
        "merge_method": args.merge_method,
        "merge_indicator": merge_indicator_def,
        "merge_persist": int(args.merge_persist),
        "merge_guard_outputs": args.kmeans_inertia_guard_outputs,
        "merge_detector_used": merge_detector_used,
        "merge_fallback_reason": merge_fallback_reason,
        "merge_time_seedmask": merge_time_seedmask,
        "merge_time_exists_seedmask": merge_time_exists_seedmask,
        "merge_time_frac_seedmask": merge_time_frac_seedmask,
        "merge_sep_thresh": merge_sep_thresh,
        "merge_sep_frac": merge_sep_frac,
        "merge_by": "seedmask_sep" if sep_series_seedmask else None,
        "s_start": s_start,
        "s_min": s_min,
        "t_at_s_min": t_at_s_min,
        "sep_initial": sep_initial,
        "sep_final": sep_final,
        "sep_min": sep_min,
        "sep_min_time": sep_min_time,
        "sep_initial_seedmask": sep_initial_seedmask,
        "sep_final_seedmask": sep_final_seedmask,
        "sep_min_seedmask": sep_min_seedmask,
        "sep_min_time_seedmask": sep_min_time_seedmask,
        "sep_ratio_seedmask": sep_ratio_seedmask,
        "sep_proj_start": sep_proj_start,
        "sep_proj_end": sep_proj_end,
        "sep_proj_min": sep_proj_min,
        "sep_proj_min_time": sep_proj_min_time,
        "sep_proj_ratio": sep_proj_ratio,
        "sep_proj_delta": sep_proj_delta,
        "sep_proj_unit": drift_unit.tolist() if drift_unit is not None else None,
        "sep_norm_start": sep_norm_start,
        "sep_norm_end": sep_norm_end,
        "sep_norm_min": sep_norm_min,
        "sep_norm_min_time": sep_norm_min_time,
        "sep_norm_ratio": sep_norm_ratio,
        "sep_norm_delta": sep_norm_delta,
        "sep_trend_slope_early": sep_trend_slope_early,
        "sep_trend_slope_late": sep_trend_slope_late,
        "sep_rebound_flag_robust": sep_rebound_flag_robust,
        "sep_x_start": sep_x_start,
        "sep_x_end": sep_x_end,
        "sep_x_min": sep_x_min,
        "sep_x_min_time": sep_x_min_time,
        "sep_x_ratio": sep_x_ratio,
        "sep_x_delta": sep_x_delta,
        "sep_y_start": sep_y_start,
        "sep_y_end": sep_y_end,
        "sep_y_min": sep_y_min,
        "sep_y_min_time": sep_y_min_time,
        "sep_y_ratio": sep_y_ratio,
        "sep_y_delta": sep_y_delta,
        "sep_z_start": sep_z_start,
        "sep_z_end": sep_z_end,
        "sep_z_min": sep_z_min,
        "sep_z_min_time": sep_z_min_time,
        "sep_z_ratio": sep_z_ratio,
        "sep_z_delta": sep_z_delta,
        "com_sep_start": com_sep_start,
        "com_sep_end": com_sep_end,
        "com_sep_delta": com_sep_delta,
        "com_sep_source": com_sep_source,
        "seedmask_source": seedmask_source,
        "seedmask_plane_point": seedmask_plane["point"] if seedmask_plane else None,
        "seedmask_plane_normal": seedmask_plane["normal"] if seedmask_plane else None,
        "sep_ratio": sep_ratio,
        "sep_ratio_def": f"sep(t_end)/sep(t_start) ({args.sep_mode})",
        "merge_frac": args.merge_frac,
        "merge_floor_mult": args.merge_floor_mult,
        "merge_floor": merge_floor,
        "merge_floor_xsplit": merge_floor_xsplit,
        "merge_thr": merge_thr,
        "merge_thr_xsplit": merge_thr_xsplit,
        "merge_thr_kmeans": merge_thr_kmeans,
        "merge_indicator_at_merge": merge_indicator_at_merge,
        "merge_time": merge_time,
        "merge_time_exists": merge_time_exists,
        "merge_time_frac": merge_time_frac,
        "merge_time_kmeans": merge_time_kmeans,
        "merge_time_xsplit": merge_time_xsplit,
        "merge_time_delta_frac": merge_time_delta_frac,
        "merge_detector_disagreement": merge_detector_disagreement,
        "tilt_post_merge_samples": tilt_post_merge_samples,
        "tilt_post_merge_amp_max": tilt_post_merge_amp_max,
        "tilt_post_merge_no_nan": tilt_post_merge_no_nan,
        "tilt_metric_used": tilt_metric_used,
        "tilt_z_bins": args.tilt_z_bins,
        "tilt_rho_frac": args.tilt_rho_frac,
        "tilt_centroid_samples": tilt_centroid_samples,
        "tilt_centroid_amp_max": tilt_centroid_amp_max,
        "tilt_slope_bins_min": tilt_slope_bins_min,
        "tilt_slope_bins_max": tilt_slope_bins_max,
        "tilt_slope_bins_mean": tilt_slope_bins_mean,
        "guard_outputs": args.guard_outputs,
        "sim_time_reached": sim_time_reached,
        "t_end": t_end,
        "field_energy_initial": field_energy_initial,
        "field_energy_final": field_energy_final,
        "field_energy_rel_drift": field_energy_rel_drift,
        "diag_dir": str(diag_root),
        "metadata_path": str(Path(args.metadata)),
    }
    if sep_times_xsplit:
        metrics.update(
            {
                "sep_initial_xsplit": sep_initial_xsplit,
                "sep_final_xsplit": sep_final_xsplit,
                "sep_min_xsplit": sep_min_xsplit,
                "sep_min_time_xsplit": sep_min_time_xsplit,
                "sep_ratio_xsplit": sep_ratio_xsplit,
            }
        )
    if sep_times_kmeans:
        ratio_clean = [r for r in ratios_kmeans if r is not None]
        inertia_clean = [v for v in inertia_reduction_kmeans if v is not None and np.isfinite(v)]
        inertia_ratio_clean = [v for v in inertia_ratio_kmeans if v is not None and np.isfinite(v)]
        metrics.update(
            {
                "sep_initial_kmeans": first_finite(sep_series_kmeans),
                "sep_final_kmeans": last_finite(sep_series_kmeans),
                "sep_min_kmeans": sep_min_kmeans,
                "sep_min_time_kmeans": sep_min_time_kmeans,
                "sep_ratio_kmeans": sep_ratio_kmeans,
                "kmeans_cluster_ratio_min": min(ratio_clean) if ratio_clean else None,
                "kmeans_cluster_ratio_max": max(ratio_clean) if ratio_clean else None,
                "kmeans_valid_fraction": float(np.mean([np.isfinite(s) for s in sep_series_kmeans]))
                if sep_series_kmeans
                else None,
                "kmeans_inertia0": inertia0,
                "kmeans_inertia0_time": inertia0_time,
                "kmeans_inertia_ratio_def": "I(t)/I0",
                "kmeans_inertia_reduction_initial": first_finite(inertia_reduction_kmeans),
                "kmeans_inertia_reduction_final": last_finite(inertia_reduction_kmeans),
                "kmeans_inertia_reduction_min": min(inertia_clean) if inertia_clean else None,
                "kmeans_inertia_reduction_max": max(inertia_clean) if inertia_clean else None,
                "kmeans_inertia_ratio_initial": first_finite(inertia_ratio_kmeans),
                "kmeans_inertia_ratio_final": last_finite(inertia_ratio_kmeans),
                "kmeans_inertia_ratio_min": min(inertia_ratio_clean) if inertia_ratio_clean else None,
                "kmeans_inertia_ratio_max": max(inertia_ratio_clean) if inertia_ratio_clean else None,
                "kmeans_merge_criterion": args.kmeans_merge_criterion,
                "kmeans_inertia_thr": args.kmeans_inertia_thr,
                "kmeans_inertia_ratio_thr": args.kmeans_inertia_ratio_thr,
                "kmeans_coords": args.kmeans_coords,
                "kmeans_inertia_guard_outputs": args.kmeans_inertia_guard_outputs,
            }
        )

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # CSV outputs
    sep_csv = Path(args.sep_csv)
    sep_csv.parent.mkdir(parents=True, exist_ok=True)
    with sep_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "x_left", "y_left", "z_left", "x_right", "y_right", "z_right", "sep"])
        writer.writeheader()
        for t, left, right, sep in zip(sep_times, left_centroids, right_centroids, sep_series):
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

    tilt_csv = Path(args.tilt_csv)
    tilt_csv.parent.mkdir(parents=True, exist_ok=True)
    with tilt_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "amp_xy"])
        writer.writeheader()
        for t, amp in zip(tilt_post_times, tilt_post_amp):
            writer.writerow({"time_s": t, "amp_xy": amp})

    # Plots
    merge_indicator_is_inertia = merge_indicator_def == "kmeans_inertia_reduction"
    merge_indicator_is_ratio = merge_indicator_def == "kmeans_inertia_ratio"
    if sep_series:
        plt.figure(figsize=(6, 4))
        plt.plot(sep_times, sep_series, marker="o")
        if merge_thr is not None and not (merge_indicator_is_inertia or merge_indicator_is_ratio):
            plt.axhline(merge_thr, color="red", linestyle="--", label="merge_thr")
        if merge_time is not None:
            plt.axvline(merge_time, color="gray", linestyle=":", label="merge_time")
        plt.xlabel("Time (s)")
        plt.ylabel("Separation (m)")
        title_suffix = f"{args.merge_method}"
        if merge_indicator_is_inertia:
            title_suffix += ", merge by inertia"
        elif merge_indicator_is_ratio:
            title_suffix += ", merge by inertia_ratio"
        plt.title(f"Separation vs Time ({title_suffix})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "separation_vs_time.png")
        plt.close()

        if args.merge_method == "kmeans":
            plt.figure(figsize=(6, 4))
            plt.plot(sep_times, sep_series, marker="o")
            if merge_thr is not None and not (merge_indicator_is_inertia or merge_indicator_is_ratio):
                plt.axhline(merge_thr, color="red", linestyle="--", label="merge_thr")
            if merge_time is not None:
                plt.axvline(merge_time, color="gray", linestyle=":", label="merge_time")
            plt.xlabel("Time (s)")
            plt.ylabel("Separation (m)")
            plt.title("Separation vs Time (kmeans)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "separation_kmeans_vs_time.png")
            plt.close()

    if merge_indicator_is_inertia and sep_times_kmeans:
        plt.figure(figsize=(6, 4))
        plt.plot(sep_times_kmeans, inertia_reduction_kmeans, marker="o")
        if merge_thr is not None:
            plt.axhline(merge_thr, color="red", linestyle="--", label="merge_thr")
        if merge_time is not None:
            plt.axvline(merge_time, color="gray", linestyle=":", label="merge_time")
        plt.xlabel("Time (s)")
        plt.ylabel("Inertia reduction")
        plt.title("k-means Inertia Reduction vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "kmeans_inertia_reduction_vs_time.png")
        plt.close()

    if merge_indicator_is_ratio and sep_times_kmeans:
        plt.figure(figsize=(6, 4))
        plt.plot(sep_times_kmeans, inertia_ratio_kmeans, marker="o")
        if merge_thr is not None:
            plt.axhline(merge_thr, color="red", linestyle="--", label="merge_thr")
        if merge_time is not None:
            plt.axvline(merge_time, color="gray", linestyle=":", label="merge_time")
        plt.xlabel("Time (s)")
        plt.ylabel("Inertia ratio (I/I0)")
        plt.title("k-means Inertia Ratio vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "kmeans_inertia_ratio_vs_time.png")
        plt.close()

    if sep_times_xsplit and sep_times_kmeans:
        plt.figure(figsize=(6, 4))
        plt.plot(sep_times_xsplit, sep_series_xsplit, label="xsplit", marker="o")
        plt.plot(sep_times_kmeans, sep_series_kmeans, label="kmeans", marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Separation (m)")
        plt.title("Merge Time Comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "merge_time_comparison.png")
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

    if tilt_centroid_amp:
        plt.figure(figsize=(6, 4))
        plt.plot(tilt_centroid_times, tilt_centroid_amp, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Centroid amplitude (m)")
        plt.title("Post-Merge Tilt Centroid Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "tilt_centroid_amp_post_merge.png")
        plt.close()

    if tilt_post_amp:
        plt.figure(figsize=(6, 4))
        plt.plot(tilt_post_times, tilt_post_amp, marker="o")
        plt.xlabel("Time (s)")
        if tilt_metric_used == "slope":
            plt.ylabel("Tilt slope (m/m)")
            plt.title("Post-Merge Tilt Slope")
            plot_name = "tilt_slope_post_merge.png"
        else:
            plt.ylabel("Centroid amplitude (m)")
            plt.title("Post-Merge Tilt Centroid Amplitude")
            plot_name = "tilt_centroid_amp_post_merge.png"
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / plot_name)
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

    if diags_all:
        last_diag = diags_all[-1]
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
