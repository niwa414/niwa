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
    raise SystemExit(f"yt required for WarpX 3D diag analysis: {exc}")

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


def compute_centroid(rho: np.ndarray, x_centers, y_centers) -> tuple[float, float] | None:
    if rho.ndim != 3:
        return None
    rho_xy = np.sum(np.abs(rho), axis=2)
    mass = float(np.sum(rho_xy))
    if mass <= 0.0:
        return None
    x_c = float(np.sum(rho_xy * x_centers[:, None]) / mass)
    y_c = float(np.sum(rho_xy * y_centers[None, :]) / mass)
    return x_c, y_c


def field_metrics(ds, ad) -> dict | None:
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
    field_energy = float(0.5 * np.sum((b2 + e2) * volume))
    mag_energy = float(0.5 * np.sum(b2 * volume))
    b_rms = float(np.sqrt(np.mean(b2)))
    return {
        "field_energy": field_energy,
        "mag_energy": mag_energy,
        "b_rms": b_rms,
    }


def extract_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WarpX 3D tilt smoke case.")
    parser.add_argument("--diag-dir", required=True, help="WarpX diag directory (contains diag*).")
    parser.add_argument("--metadata", required=True, help="WarpX run metadata JSON.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--csv", required=True, help="Output CSV for centroid series.")
    parser.add_argument("--start-index", type=int, default=0, help="Index for initial amplitude ratio.")
    args = parser.parse_args()

    diag_root = Path(args.diag_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    diags = list_diags(diag_root)
    times = []
    centroids = []
    amps = []
    field_series = []

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
        x0 = float(ds.domain_left_edge[0].to_value())
        y0 = float(ds.domain_left_edge[1].to_value())
        x_centers = x0 + (np.arange(dims[0]) + 0.5) * dx
        y_centers = y0 + (np.arange(dims[1]) + 0.5) * dy
        centroid = compute_centroid(rho, x_centers, y_centers)
        if centroid is None:
            continue
        x_c, y_c = centroid
        amp = float(np.sqrt(x_c * x_c + y_c * y_c))
        times.append(t)
        centroids.append((x_c, y_c))
        amps.append(amp)
        fmetrics = field_metrics(ds, ad)
        if fmetrics is None:
            field_series.append({"time_s": t, "field_energy": None, "mag_energy": None, "b_rms": None})
        else:
            row = {"time_s": t}
            row.update(fmetrics)
            field_series.append(row)

    tilt_amp_ratio = None
    tilt_amp_initial = None
    tilt_amp_final = None
    tilt_amp_max = None
    tilt_amp_min = None
    tilt_amp_peak_time = None
    no_nan = True
    if amps:
        start_idx = min(max(args.start_index, 0), len(amps) - 1)
        tilt_amp_initial = amps[start_idx]
        tilt_amp_final = amps[-1]
        tilt_amp_max = max(amps)
        tilt_amp_min = min(amps)
        try:
            peak_idx = amps.index(tilt_amp_max)
            tilt_amp_peak_time = times[peak_idx]
        except Exception:
            tilt_amp_peak_time = None
        if tilt_amp_initial and tilt_amp_initial > 0.0:
            tilt_amp_ratio = float(tilt_amp_final / tilt_amp_initial)
        for val in (tilt_amp_initial, tilt_amp_final, tilt_amp_ratio):
            if val is None or not np.isfinite(val):
                no_nan = False

    meta = extract_meta(Path(args.metadata))
    run_args = meta.get("args", {})
    monitor = meta.get("monitor") or {}
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
    mag_energy_initial = None
    mag_energy_final = None
    mag_energy_rel_drift = None
    b_rms_initial = None
    b_rms_final = None
    if field_series:
        field_vals = [row.get("field_energy") for row in field_series if row.get("field_energy") is not None]
        if field_vals:
            field_energy_initial = field_vals[0]
            field_energy_final = field_vals[-1]
            if field_energy_initial not in (None, 0.0):
                field_energy_rel_drift = (field_energy_final - field_energy_initial) / field_energy_initial
        mag_vals = [row.get("mag_energy") for row in field_series if row.get("mag_energy") is not None]
        if mag_vals:
            mag_energy_initial = mag_vals[0]
            mag_energy_final = mag_vals[-1]
            if mag_energy_initial not in (None, 0.0):
                mag_energy_rel_drift = (mag_energy_final - mag_energy_initial) / mag_energy_initial
        b_vals = [row.get("b_rms") for row in field_series if row.get("b_rms") is not None]
        if b_vals:
            b_rms_initial = b_vals[0]
            b_rms_final = b_vals[-1]

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": len(times),
        "no_nan_in_metrics": no_nan,
        "tilt_amp_initial": tilt_amp_initial,
        "tilt_amp_final": tilt_amp_final,
        "tilt_amp_ratio": tilt_amp_ratio,
        "tilt_amp_max": tilt_amp_max,
        "tilt_amp_min": tilt_amp_min,
        "tilt_amp_peak_time": tilt_amp_peak_time,
        "centroid_amp_initial": tilt_amp_initial,
        "centroid_amp_final": tilt_amp_final,
        "centroid_amp_ratio": tilt_amp_ratio,
        "centroid_amp_max": tilt_amp_max,
        "centroid_amp_min": tilt_amp_min,
        "centroid_amp_peak_time": tilt_amp_peak_time,
        "tilt_ratio_def": "A(t_end)/A(t_start)",
        "centroid_start_index": args.start_index,
        "drop_breach": drop_breach,
        "sim_time_reached": sim_time_reached,
        "field_energy_initial": field_energy_initial,
        "field_energy_final": field_energy_final,
        "field_energy_rel_drift": field_energy_rel_drift,
        "mag_energy_initial": mag_energy_initial,
        "mag_energy_final": mag_energy_final,
        "mag_energy_rel_drift": mag_energy_rel_drift,
        "b_rms_initial": b_rms_initial,
        "b_rms_final": b_rms_final,
        "diag_dir": str(diag_root),
        "metadata_path": str(Path(args.metadata)),
    }
    resistivity = meta.get("resistivity")
    if isinstance(resistivity, dict):
        metrics["plasma_resistivity_expr"] = resistivity.get("plasma_resistivity_expr")
        metrics["plasma_resistivity_scale"] = resistivity.get("plasma_resistivity_scale")
        metrics["plasma_hyper_resistivity_expr"] = resistivity.get("plasma_hyper_resistivity_expr")
        metrics["plasma_hyper_resistivity_scale"] = resistivity.get("plasma_hyper_resistivity_scale")
        metrics["eta_source"] = resistivity.get("eta_source")
    etaJ2_meta = meta.get("etaJ2")
    if isinstance(etaJ2_meta, dict):
        metrics["etaJ2_mean"] = etaJ2_meta.get("etaJ2_mean")
        metrics["J2_mean"] = etaJ2_meta.get("J2_mean")
        metrics["etaJ2_samples"] = etaJ2_meta.get("samples")
        metrics["etaJ2_updates"] = etaJ2_meta.get("updates")
    if run_args:
        metrics["applied_field_enabled"] = run_args.get("applied_field_enabled")
        metrics["applied_Bz_T"] = run_args.get("applied_Bz_T")
        metrics["applied_Bz_expr"] = run_args.get("applied_Bz_expr")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "x_centroid", "y_centroid", "amp"])
        writer.writeheader()
        for t, (x_c, y_c), amp in zip(times, centroids, amps):
            writer.writerow(
                {"time_s": t, "x_centroid": x_c, "y_centroid": y_c, "amp": amp}
            )

    if amps:
        plt.figure(figsize=(6, 4))
        plt.plot(times, amps, marker="o")
        plt.xlabel("Time (s)")
        plt.ylabel("Centroid amplitude (m)")
        plt.title("Tilt Centroid Amplitude vs Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "tilt_centroid_amp_vs_time.png")
        plt.close()

    if centroids:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        plt.figure(figsize=(5, 5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("x centroid (m)")
        plt.ylabel("y centroid (m)")
        plt.title("Tilt Centroid XY Path")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "tilt_centroid_xy_path.png")
        plt.close()

    if field_series:
        e_times = [row.get("time_s") for row in field_series]
        e_vals = [row.get("field_energy") for row in field_series]
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
