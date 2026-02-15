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
    raise SystemExit(f"yt required for WarpX diag analysis: {exc}")

UNITS_OVERRIDE = {
    "length_unit": (1.0, "m"),
    "time_unit": (1.0, "s"),
    "mass_unit": (1.0, "kg"),
    "magnetic_unit": (1.0, "T"),
}


def list_diags(diag_root: Path) -> list[Path]:
    if not diag_root.exists():
        return []
    if (
        diag_root.is_dir()
        and diag_root.name.startswith("diag")
        and (diag_root / "Header").exists()
    ):
        return [diag_root]
    return sorted(
        [p for p in diag_root.iterdir() if p.is_dir() and p.name.startswith("diag")]
    )


def compute_mode_metrics(ad, field: str, max_mode: int) -> dict[str, float]:
    arr = ad["boxlib", field].to_ndarray()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim == 1:
        return {}
    if arr.ndim < 3:
        return {}
    nmodes = min(arr.shape[0], max_mode + 1)
    metrics: dict[str, float] = {}
    for m in range(nmodes):
        amp = np.abs(np.asarray(arr[m]))
        metrics[f"{field}_m{m}_rms"] = float(np.sqrt(np.mean(amp * amp)))
        metrics[f"{field}_m{m}_max"] = float(np.max(amp))
    return metrics


def total_field_energy(ds, ad) -> float:
    Br = ad["boxlib", "Br"].to_ndarray()
    Bt = ad["boxlib", "Bt"].to_ndarray()
    Bz = ad["boxlib", "Bz"].to_ndarray()
    Er = ad["boxlib", "Er"].to_ndarray()
    Et = ad["boxlib", "Et"].to_ndarray()
    Ez = ad["boxlib", "Ez"].to_ndarray()

    nr = int(ds.domain_dimensions[0])
    nz = int(ds.domain_dimensions[1])

    def reshape_rz(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim == 1 and arr.size == nr * nz:
            arr = arr.reshape(nr, nz)
        return arr

    Br = reshape_rz(Br)
    Bt = reshape_rz(Bt)
    Bz = reshape_rz(Bz)
    Er = reshape_rz(Er)
    Et = reshape_rz(Et)
    Ez = reshape_rz(Ez)

    b2 = Br * Br + Bt * Bt + Bz * Bz
    e2 = Er * Er + Et * Et + Ez * Ez

    dr = float(ds.domain_width[0].to_value()) / max(1, nr)
    dz = float(ds.domain_width[1].to_value()) / max(1, nz)
    r0 = float(ds.domain_left_edge[0].to_value())
    r_centers = r0 + (np.arange(nr) + 0.5) * dr
    volume = 2.0 * np.pi * r_centers[:, None] * dr * dz

    energy_density = 0.5 * (b2 + e2)
    return float(np.sum(energy_density * volume))


def summarize_series(diag_root: Path, max_mode: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    diags = list_diags(diag_root)
    if not diags:
        return [], []
    series = []
    energy_series = []
    for diag in diags:
        ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        t = float(ds.current_time.to_value())
        row = {"diag": diag.name, "time_s": t}
        row.update(compute_mode_metrics(ad, "rho", max_mode))
        series.append(row)
        try:
            energy = total_field_energy(ds, ad)
            energy_series.append({"diag": diag.name, "time_s": t, "field_energy": energy})
        except Exception:
            energy_series.append({"diag": diag.name, "time_s": t, "field_energy": None})
    return series, energy_series


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def compute_particle_loss(meta: dict, species: str = "ions") -> float | None:
    init = meta.get("species_stats_init", {}).get(species, {})
    final = meta.get("species_stats", {}).get(species, {})
    init_w = init.get("weight_sum")
    final_w = final.get("weight_sum")
    if init_w is None or final_w is None or init_w <= 0.0:
        return None
    return float(max(0.0, 1.0 - (final_w / init_w)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WarpX tilt smoke case (m=1).")
    parser.add_argument("--diag-dir", required=True, help="WarpX diag directory (contains diag*).")
    parser.add_argument("--metadata", required=True, help="WarpX run metadata JSON.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument("--csv", required=True, help="Output CSV for mode spectrum.")
    parser.add_argument("--max-mode", type=int, default=1, help="Highest theta mode to include.")
    parser.add_argument(
        "--amp-field",
        default="rho_m1_rms",
        help="Field/mode metric to track for amplitude ratios (e.g., rho_m1_rms).",
    )
    parser.add_argument(
        "--amp-label",
        default="m=1 amplitude (RMS)",
        help="Label for amplitude plot.",
    )
    parser.add_argument(
        "--skip-amp",
        action="store_true",
        help="Skip amplitude-series metrics/plots (useful for non-tilt smoke runs).",
    )
    args = parser.parse_args()

    diag_root = Path(args.diag_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    series, energy_series = summarize_series(diag_root, args.max_mode)
    monitor_series = []

    meta = extract_meta(Path(args.metadata))
    run_args = meta.get("args", {})
    monitor = meta.get("monitor") or {}
    records = monitor.get("records") or []
    for rec in records:
        mode_rms = rec.get("mode_rms") or {}
        if args.amp_field in mode_rms:
            monitor_series.append(
                {"time_s": rec.get("time"), args.amp_field: mode_rms.get(args.amp_field)}
            )

    times = [row.get("time_s") for row in series]
    amp_series = []
    amp_initial = None
    amp_final = None
    amp_peak = None
    time_peak = None
    tilt_amp_ratio = None
    tilt_amp_ratio_end = None
    amp_source = "diag"
    diag_has_amp = series and any(row.get(args.amp_field) is not None for row in series)
    if not args.skip_amp and diag_has_amp:
        amp_series = [row.get(args.amp_field) for row in series]
        amp_series = [val if val is not None else np.nan for val in amp_series]
        amp_initial = amp_series[0] if amp_series else None
        amp_final = amp_series[-1] if amp_series else None
        amp_peak = float(np.nanmax(amp_series)) if amp_series else None
        if amp_series and amp_peak is not None and np.isfinite(amp_peak):
            idx = int(np.nanargmax(amp_series))
            time_peak = times[idx]

        if amp_initial is not None and np.isfinite(amp_initial) and amp_initial > 0.0:
            if amp_peak is not None and np.isfinite(amp_peak):
                tilt_amp_ratio = float(amp_peak / amp_initial)
            if amp_final is not None and np.isfinite(amp_final):
                tilt_amp_ratio_end = float(amp_final / amp_initial)
    elif not args.skip_amp and monitor_series:
        amp_source = "monitor"
        amp_series = [row.get(args.amp_field) for row in monitor_series]
        amp_series = [val if val is not None else np.nan for val in amp_series]
        times = [row.get("time_s") for row in monitor_series]
        amp_initial = amp_series[0] if amp_series else None
        amp_final = amp_series[-1] if amp_series else None
        amp_peak = float(np.nanmax(amp_series)) if amp_series else None
        if amp_series and amp_peak is not None and np.isfinite(amp_peak):
            idx = int(np.nanargmax(amp_series))
            time_peak = times[idx]

        if amp_initial is not None and np.isfinite(amp_initial) and amp_initial > 0.0:
            if amp_peak is not None and np.isfinite(amp_peak):
                tilt_amp_ratio = float(amp_peak / amp_initial)
            if amp_final is not None and np.isfinite(amp_final):
                tilt_amp_ratio_end = float(amp_final / amp_initial)

    no_nan = True
    if not args.skip_amp:
        for val in (amp_initial, amp_final, amp_peak, tilt_amp_ratio):
            if val is None:
                continue
            if not np.isfinite(val):
                no_nan = False

    max_steps = run_args.get("max_steps")
    dt = run_args.get("dt")
    diag_period = run_args.get("diag_period", None)
    drop_breach = monitor.get("drop_breach")

    sim_time_reached = None
    if records:
        sim_time_reached = records[-1].get("time")
        last_step = records[-1].get("step")
    else:
        sim_time_reached = times[-1] if times else None
        last_step = None

    ran_to_completion = None
    if max_steps is not None:
        if last_step is not None:
            ran_to_completion = last_step >= (max_steps - 1)
        elif sim_time_reached is not None and dt is not None and diag_period is not None:
            expected = (max_steps - 1) * dt
            slack = dt * diag_period
            ran_to_completion = sim_time_reached >= (expected - slack)
    if ran_to_completion is None:
        ran_to_completion = bool(series)

    particle_loss_frac = compute_particle_loss(meta)

    metrics = {
        "ran_to_completion": ran_to_completion,
        "num_outputs": len(series),
        "no_nan_in_metrics": no_nan,
        "tilt_amp_initial": amp_initial,
        "tilt_amp_final": amp_final,
        "tilt_amp_peak": amp_peak,
        "tilt_time_peak": time_peak,
        "tilt_amp_ratio": tilt_amp_ratio,
        "tilt_amp_ratio_end": tilt_amp_ratio_end,
        "tilt_amp_field": args.amp_field,
        "tilt_amp_source": amp_source,
        "tilt_ratio_def": f"max({args.amp_field})/{args.amp_field}(t0)",
        "sim_time_reached": sim_time_reached,
        "drop_breach": drop_breach,
        "dropped_particles_total": meta.get("dropped_particles_total"),
        "particle_loss_frac": particle_loss_frac,
        "n_azimuthal_modes": run_args.get("n_azimuthal_modes"),
        "solver": run_args.get("solver"),
        "hybrid": run_args.get("hybrid"),
        "diag_dir": str(diag_root),
        "metadata_path": str(Path(args.metadata)),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    series_for_csv = series
    if not diag_has_amp and monitor_series:
        series_for_csv = monitor_series
    if series_for_csv:
        write_csv(Path(args.csv), series_for_csv)

    if series and amp_series:
        plt.figure(figsize=(6, 4))
        plt.plot(times, amp_series, marker="o", label=args.amp_field)
        plt.xlabel("Time (s)")
        plt.ylabel(args.amp_label)
        plt.title("Amplitude vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "tilt_m1_amplitude_vs_time.png")
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

    if series:
        last_diag = list_diags(diag_root)[-1]
        ds = yt.load(str(last_diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        rho = ad["boxlib", "rho"].to_ndarray()
        nr = int(ds.domain_dimensions[0])
        nz = int(ds.domain_dimensions[1])
        if rho.ndim == 3:
            rho = rho[0]
        if rho.ndim == 1 and rho.size == nr * nz:
            rho = rho.reshape(nr, nz)
        dr = float(ds.domain_width[0].to_value()) / max(1, nr)
        dz = float(ds.domain_width[1].to_value()) / max(1, nz)
        r0 = float(ds.domain_left_edge[0].to_value())
        z0 = float(ds.domain_left_edge[1].to_value())
        r_edges = r0 + np.arange(nr + 1) * dr
        z_edges = z0 + np.arange(nz + 1) * dz
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(z_edges, r_edges, rho, shading="auto")
        plt.xlabel("z (m)")
        plt.ylabel("r (m)")
        plt.title("Density Snapshot (m=0)")
        plt.colorbar(label="rho")
        plt.tight_layout()
        plt.savefig(plots_dir / "density_snapshot.png")
        plt.close()


if __name__ == "__main__":
    main()
