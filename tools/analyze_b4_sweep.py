#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import yt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(f"yt required for WarpX 3D diagnostics: {exc}")

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


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_rho_series(diag_root: Path) -> dict:
    diags = list_diags(diag_root)
    times = []
    rho_min = []
    rho_min_pos = []
    rho_p10 = []
    for diag in diags:
        ds = yt.load(str(diag), units_override=UNITS_OVERRIDE)
        ad = ds.all_data()
        dims = tuple(int(x) for x in ds.domain_dimensions)
        rho = reshape_field(ad["boxlib", "rho"].to_ndarray(), dims)
        if rho.ndim != 3:
            continue
        rho_abs = np.abs(rho).ravel()
        if rho_abs.size == 0:
            continue
        times.append(float(ds.current_time.to_value()))
        rho_min.append(float(np.min(rho_abs)))
        pos = rho_abs[rho_abs > 0.0]
        if pos.size:
            rho_min_pos.append(float(np.min(pos)))
            rho_p10.append(float(np.percentile(pos, 10)))
        else:
            rho_min_pos.append(None)
            rho_p10.append(None)
    return {
        "times": times,
        "rho_min": rho_min,
        "rho_min_pos": rho_min_pos,
        "rho_p10": rho_p10,
    }


def select_rho_series(series: dict) -> tuple[list[float], list[float], str]:
    times = series.get("times") or []
    p10 = series.get("rho_p10") or []
    min_pos = series.get("rho_min_pos") or []
    if p10 and any(val is not None for val in p10):
        vals = [v if v is not None else np.nan for v in p10]
        return times, vals, "rho_p10_pos"
    if min_pos and any(val is not None for val in min_pos):
        vals = [v if v is not None else np.nan for v in min_pos]
        return times, vals, "rho_min_pos"
    vals = series.get("rho_min") or []
    return times, vals, "rho_min"


def variant_ok(metrics: dict, min_outputs: int) -> bool:
    if not metrics:
        return False
    if metrics.get("ran_to_completion") is not True:
        return False
    if metrics.get("no_nan_in_metrics") is not True:
        return False
    if metrics.get("drop_breach") is True:
        return False
    outputs = metrics.get("num_outputs") or 0
    return outputs >= min_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B4 dynamic range sweep.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    parser.add_argument(
        "--summary",
        required=True,
        help="Output JSON with per-variant summary.",
    )
    args = parser.parse_args()

    variants_cfg = load_json(Path(args.variants))
    control_id = variants_cfg.get("control")
    treatment_id = variants_cfg.get("treatment")
    floor_id = variants_cfg.get("floor_only")
    substeps_id = variants_cfg.get("substeps_only")
    min_outputs = int(variants_cfg.get("min_outputs", 6))
    drift_margin = float(variants_cfg.get("energy_drift_margin", 0.05))
    drift_cap = float(variants_cfg.get("energy_drift_cap", 0.2))

    variant_order = []
    for cid in (control_id, floor_id, substeps_id, treatment_id):
        if cid and cid not in variant_order:
            variant_order.append(cid)

    variants = {}
    for cid in variant_order:
        passfail_path = Path("outputs") / cid / "analysis" / "PASSFAIL.json"
        metrics_path = Path("outputs") / cid / "analysis" / "metrics.json"
        passfail = load_json(passfail_path)
        metrics = passfail.get("metrics") or load_json(metrics_path)
        diag_root = Path("outputs") / cid / "raw" / "run" / "diag"
        rho_series = compute_rho_series(diag_root)
        variants[cid] = {
            "case_id": cid,
            "result": passfail.get("result") or passfail.get("status"),
            "metrics": metrics,
            "rho_series": rho_series,
        }

    control = variants.get(control_id, {})
    treatment = variants.get(treatment_id, {})
    control_ok = variant_ok(control.get("metrics", {}), min_outputs)
    treatment_ok = variant_ok(treatment.get("metrics", {}), min_outputs)

    control_drift = (control.get("metrics") or {}).get("field_energy_rel_drift")
    treatment_drift = (treatment.get("metrics") or {}).get("field_energy_rel_drift")
    drift_ok = True
    if control_drift is not None and treatment_drift is not None:
        drift_limit = max(control_drift + drift_margin, drift_cap)
        drift_ok = treatment_drift <= drift_limit
    sweep_pass = False
    sweep_outcome = "insufficient_data"
    if not treatment_ok:
        sweep_outcome = "treatment_failed"
    elif not control_ok and treatment_ok:
        sweep_pass = True
        sweep_outcome = "treatment_pass_control_fail"
    elif control_ok and treatment_ok:
        sweep_pass = bool(drift_ok)
        sweep_outcome = "both_pass_no_worse" if sweep_pass else "treatment_regressed"

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot min-density proxy vs time
    plt.figure(figsize=(7, 4))
    for cid in variant_order:
        series = variants[cid]["rho_series"]
        t_vals, rho_vals, label = select_rho_series(series)
        if t_vals:
            plt.plot(t_vals, rho_vals, marker="o", label=f"{cid} ({label})")
    plt.xlabel("Time (s)")
    plt.ylabel("rho min proxy")
    plt.title("Density-Min Proxy vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "min_density_vs_time.png")
    plt.close()

    # Plot energy drift comparison
    plt.figure(figsize=(7, 4))
    labels = []
    drifts = []
    for cid in variant_order:
        labels.append(cid)
        drift_val = (variants[cid]["metrics"] or {}).get("field_energy_rel_drift")
        drifts.append(drift_val if drift_val is not None else np.nan)
    plt.bar(range(len(labels)), drifts)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Field energy rel drift")
    plt.title("Energy Drift Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "energy_drift_compare.png")
    plt.close()

    # Drop breach / NaN flags
    plt.figure(figsize=(7, 4))
    drop_vals = []
    nan_vals = []
    for cid in variant_order:
        metrics = variants[cid]["metrics"] or {}
        drop_vals.append(1 if metrics.get("drop_breach") is True else 0)
        nan_vals.append(1 if metrics.get("no_nan_in_metrics") is True else 0)
    x = np.arange(len(variant_order))
    width = 0.35
    plt.bar(x - width / 2, drop_vals, width, label="drop_breach")
    plt.bar(x + width / 2, nan_vals, width, label="no_nan")
    plt.xticks(x, variant_order, rotation=30, ha="right")
    plt.ylabel("Flag (1=True)")
    plt.title("Drop Breach / NaN Flags")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "drop_breach_flags.png")
    plt.close()

    summary = {
        "control": control_id,
        "treatment": treatment_id,
        "floor_only": floor_id,
        "substeps_only": substeps_id,
        "min_outputs": min_outputs,
        "energy_drift_margin": drift_margin,
        "energy_drift_cap": drift_cap,
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "sweep_pass": sweep_pass,
        "sweep_outcome": sweep_outcome,
        "control_field_energy_rel_drift": control_drift,
        "treatment_field_energy_rel_drift": treatment_drift,
        "variants": variants,
    }

    metrics_out = {
        "sweep_pass": sweep_pass,
        "sweep_outcome": sweep_outcome,
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "control_case": control_id,
        "treatment_case": treatment_id,
        "min_outputs": min_outputs,
        "energy_drift_margin": drift_margin,
        "energy_drift_cap": drift_cap,
        "control_field_energy_rel_drift": control_drift,
        "treatment_field_energy_rel_drift": treatment_drift,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
