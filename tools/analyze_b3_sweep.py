#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def extract_te_series(meta: dict) -> tuple[list[float], list[float], dict]:
    ee = meta.get("electron_energy") or {}
    cfg = ee.get("config") or {}
    records = ee.get("records") or []
    times = []
    te = []
    for rec in records:
        if rec.get("te_eV") is None:
            continue
        times.append(float(rec.get("time", 0.0)))
        te.append(float(rec["te_eV"]))
    if not te:
        te0 = cfg.get("Te0_eV")
        if te0 is None:
            te0 = (meta.get("args", {}).get("hybrid", {}) or {}).get("Te_eV")
        if te0 is not None:
            times = [0.0]
            te = [float(te0)]
    return times, te, cfg


def series_stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "range": None, "std": None}
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B3 electron-energy sweep.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output JSON with per-variant summary.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    variants_cfg = load_json(Path(args.variants))
    control_id = variants_cfg.get("control")
    treatment_id = variants_cfg.get("treatment")
    min_outputs = int(variants_cfg.get("min_outputs", 6))
    te_eps = float(variants_cfg.get("te_variation_eps", 1.0e-6))
    te_off_eps = float(variants_cfg.get("te_off_eps", 1.0e-9))

    variant_order = [cid for cid in (control_id, treatment_id) if cid]
    variants = {}
    for cid in variant_order:
        passfail_path = Path("outputs") / cid / "analysis" / "PASSFAIL.json"
        passfail = load_json(passfail_path)
        metrics = passfail.get("metrics") or {}
        meta_path = metrics.get("metadata_path")
        meta = load_json(Path(meta_path)) if meta_path else {}
        times, te_series, ee_cfg = extract_te_series(meta)
        stats = series_stats(te_series)
        variants[cid] = {
            "case_id": cid,
            "result": passfail.get("result") or passfail.get("status"),
            "metrics": metrics,
            "electron_energy_cfg": ee_cfg,
            "te_times": times,
            "te_series": te_series,
            "te_stats": stats,
        }

    control = variants.get(control_id, {})
    treatment = variants.get(treatment_id, {})
    control_ok = variant_ok(control.get("metrics", {}), min_outputs)
    treatment_ok = variant_ok(treatment.get("metrics", {}), min_outputs)
    control_range = (control.get("te_stats") or {}).get("range")
    treatment_range = (treatment.get("te_stats") or {}).get("range")

    sweep_outcome = "insufficient_data"
    sweep_pass = False
    if not treatment_ok:
        sweep_outcome = "treatment_failed"
    else:
        control_var_ok = (control_range is not None) and (control_range <= te_off_eps)
        treatment_var_ok = (treatment_range is not None) and (treatment_range > te_eps)
        if not control_ok and treatment_ok:
            sweep_pass = bool(treatment_var_ok)
            sweep_outcome = "treatment_pass_control_fail"
        elif control_ok and treatment_ok:
            sweep_pass = bool(control_var_ok and treatment_var_ok)
            sweep_outcome = "both_pass_variation_ok" if sweep_pass else "variation_mismatch"

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for cid in variant_order:
        v = variants[cid]
        if v["te_series"]:
            plt.plot(v["te_times"], v["te_series"], marker="o", label=cid)
    plt.xlabel("Time (s)")
    plt.ylabel("Te (eV)")
    plt.title("Electron Temperature vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "te_mean_vs_time.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    labels = []
    ranges = []
    for cid in variant_order:
        labels.append(cid)
        ranges.append((variants[cid].get("te_stats") or {}).get("range", np.nan))
    plt.bar(range(len(labels)), ranges)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Te range (eV)")
    plt.title("Te Variation Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "te_range_compare.png")
    plt.close()

    summary = {
        "control": control_id,
        "treatment": treatment_id,
        "min_outputs": min_outputs,
        "te_variation_eps": te_eps,
        "te_off_eps": te_off_eps,
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "sweep_pass": sweep_pass,
        "sweep_outcome": sweep_outcome,
        "variants": variants,
    }

    metrics_out = {
        "sweep_pass": sweep_pass,
        "sweep_outcome": sweep_outcome,
        "control_case": control_id,
        "treatment_case": treatment_id,
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "control_te_range": control_range,
        "treatment_te_range": treatment_range,
        "te_variation_eps": te_eps,
        "te_off_eps": te_off_eps,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
