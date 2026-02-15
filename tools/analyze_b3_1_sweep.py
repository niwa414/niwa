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


def choose_observable(control_metrics, treatment_metrics, candidates, norm_eps):
    best_metric = None
    best_diff = None
    for metric in candidates:
        c_val = control_metrics.get(metric)
        t_val = treatment_metrics.get(metric)
        if c_val is None or t_val is None:
            continue
        c_val = float(c_val)
        t_val = float(t_val)
        if abs(c_val) <= norm_eps and abs(t_val) <= norm_eps:
            continue
        denom = max(abs(c_val), norm_eps)
        diff = abs(t_val - c_val) / denom
        if best_diff is None or diff > best_diff:
            best_diff = diff
            best_metric = metric
    return best_metric, best_diff


def extract_electron_series(
    meta: dict,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], dict, dict]:
    ee = meta.get("electron_energy") or {}
    cfg = ee.get("config") or {}
    records = ee.get("records") or []
    times = []
    te_series = []
    eta_series = []
    eta_times = []
    floor_series = []
    floor_times = []
    for rec in records:
        if rec.get("te_eV") is not None:
            times.append(float(rec.get("time", 0.0)))
            te_series.append(float(rec["te_eV"]))
        if rec.get("eta") is not None:
            eta_times.append(float(rec.get("time", 0.0)))
            eta_series.append(float(rec["eta"]))
        if rec.get("n_floor") is not None:
            floor_times.append(float(rec.get("time", 0.0)))
            floor_series.append(float(rec["n_floor"]))
    if not te_series:
        te0 = cfg.get("Te0_eV")
        if te0 is None:
            te0 = (meta.get("args", {}).get("hybrid", {}) or {}).get("Te_eV")
        if te0 is not None:
            times = [0.0]
            te_series = [float(te0)]
    ee_meta = {
        "updates": ee.get("updates"),
        "feedback_updates": ee.get("feedback_updates"),
        "feedback_failures": ee.get("feedback_failures"),
        "feedback_target_used": ee.get("feedback_target_used"),
    }
    return (
        times,
        te_series,
        eta_times,
        eta_series,
        floor_times,
        floor_series,
        cfg,
        ee_meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B3.1 electron-energy coupled sweep.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output JSON with per-variant summary.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    variants_cfg = load_json(Path(args.variants))
    control_id = variants_cfg.get("control")
    treatment_id = variants_cfg.get("treatment")
    min_outputs = int(variants_cfg.get("min_outputs", 6))
    eta_std_min = float(variants_cfg.get("eta_std_min", 1.0e-12))
    writeback_ratio_min = float(variants_cfg.get("writeback_ratio_min", 0.9))
    observable_metric = str(variants_cfg.get("observable_metric", "auto"))
    observable_min_rel_diff = float(variants_cfg.get("observable_min_rel_diff", 0.01))
    observable_norm_eps = float(variants_cfg.get("observable_norm_eps", 1.0e-30))
    observable_abs_floor = float(variants_cfg.get("observable_abs_floor", 0.0))
    observable_candidates = variants_cfg.get(
        "observable_candidates",
        ["mag_energy_final", "field_energy_final", "b_rms_final"],
    )

    variant_order = [cid for cid in (control_id, treatment_id) if cid]
    variants = {}
    for cid in variant_order:
        passfail_path = Path("outputs") / cid / "analysis" / "PASSFAIL.json"
        passfail = load_json(passfail_path)
        metrics = passfail.get("metrics") or {}
        meta_path = metrics.get("metadata_path")
        meta = load_json(Path(meta_path)) if meta_path else {}
        times, te_series, eta_times, eta_series, floor_times, floor_series, ee_cfg, ee_meta = extract_electron_series(meta)
        te_stats = series_stats(te_series)
        eta_stats = series_stats(eta_series)
        floor_stats = series_stats(floor_series)
        variants[cid] = {
            "case_id": cid,
            "result": passfail.get("result") or passfail.get("status"),
            "metrics": metrics,
            "electron_energy_cfg": ee_cfg,
            "electron_energy_meta": ee_meta,
            "te_times": times,
            "te_series": te_series,
            "eta_times": eta_times,
            "eta_series": eta_series,
            "floor_times": floor_times,
            "floor_series": floor_series,
            "te_stats": te_stats,
            "eta_stats": eta_stats,
            "floor_stats": floor_stats,
        }

    control = variants.get(control_id, {})
    treatment = variants.get(treatment_id, {})
    control_ok = variant_ok(control.get("metrics", {}), min_outputs)
    treatment_ok = variant_ok(treatment.get("metrics", {}), min_outputs)

    treatment_meta = treatment.get("electron_energy_meta") or {}
    treatment_updates = treatment_meta.get("updates") or 0
    treatment_writebacks = treatment_meta.get("feedback_updates") or 0
    treatment_failures = treatment_meta.get("feedback_failures") or 0
    treatment_target_used = treatment_meta.get("feedback_target_used")
    writeback_ratio = None
    if treatment_updates:
        writeback_ratio = treatment_writebacks / treatment_updates

    eta_std = (treatment.get("eta_stats") or {}).get("std")
    floor_std = (treatment.get("floor_stats") or {}).get("std")
    writeback_series_name = None
    writeback_std = None
    if treatment.get("floor_series"):
        writeback_series_name = "density_floor"
        writeback_std = floor_std
    elif treatment.get("eta_series"):
        writeback_series_name = "resistivity"
        writeback_std = eta_std
    eta_std_ok = writeback_std is not None and writeback_std > eta_std_min
    writeback_ok = writeback_ratio is not None and writeback_ratio >= writeback_ratio_min

    control_metrics = control.get("metrics") or {}
    treatment_metrics = treatment.get("metrics") or {}
    observable_metric_used = observable_metric
    observable_rel_diff = None
    observable_diffs = {}
    for metric in observable_candidates:
        c_val = control_metrics.get(metric)
        t_val = treatment_metrics.get(metric)
        if c_val is None or t_val is None:
            observable_diffs[metric] = None
            continue
        denom = max(abs(float(c_val)), observable_norm_eps)
        observable_diffs[metric] = float(abs(float(t_val) - float(c_val)) / denom)
    if observable_metric.lower() == "auto":
        chosen_metric, chosen_diff = choose_observable(
            control_metrics,
            treatment_metrics,
            observable_candidates,
            observable_norm_eps,
        )
        if chosen_metric:
            observable_metric_used = chosen_metric
            observable_rel_diff = chosen_diff
        elif observable_candidates:
            observable_metric_used = str(observable_candidates[0])
    control_val = control_metrics.get(observable_metric_used)
    treatment_val = treatment_metrics.get(observable_metric_used)
    if observable_rel_diff is None and control_val is not None and treatment_val is not None:
        denom = max(abs(float(control_val)), observable_norm_eps)
        observable_rel_diff = float(abs(float(treatment_val) - float(control_val)) / denom)
    observable_control_abs = None
    observable_treatment_abs = None
    if control_val is not None:
        observable_control_abs = float(abs(float(control_val)))
    if treatment_val is not None:
        observable_treatment_abs = float(abs(float(treatment_val)))
    observable_degenerate = False
    if observable_abs_floor > 0.0:
        if (
            observable_control_abs is None
            or observable_treatment_abs is None
            or observable_control_abs < observable_abs_floor
            or observable_treatment_abs < observable_abs_floor
        ):
            observable_degenerate = True
    coupling_effect_observed = (
        observable_rel_diff is not None and observable_rel_diff >= observable_min_rel_diff
    )

    sweep_outcome = "insufficient_data"
    sweep_pass = False
    if not treatment_ok:
        sweep_outcome = "treatment_failed"
    elif not control_ok:
        sweep_outcome = "control_failed"
    elif treatment_failures:
        sweep_outcome = "writeback_failed"
    elif not writeback_ok:
        sweep_outcome = "writeback_insufficient"
    elif not eta_std_ok:
        sweep_outcome = "eta_static"
    elif observable_degenerate:
        sweep_outcome = "observable_degenerate"
    elif not coupling_effect_observed:
        sweep_outcome = "effect_not_observed"
    else:
        sweep_outcome = "coupling_effect_observed"
        sweep_pass = True

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

    if treatment.get("eta_series"):
        plt.figure(figsize=(7, 4))
        plt.plot(
            treatment.get("eta_times", []),
            treatment.get("eta_series"),
            marker="o",
            label=treatment_id,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Resistivity eta")
        plt.title("Resistivity vs Time (treatment)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "eta_vs_time.png")
        plt.close()

    if treatment.get("floor_series"):
        plt.figure(figsize=(7, 4))
        plt.plot(
            treatment.get("floor_times", []),
            treatment.get("floor_series"),
            marker="o",
            label=treatment_id,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Density floor")
        plt.title("Density Floor vs Time (treatment)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "density_floor_vs_time.png")
        plt.close()

    plt.figure(figsize=(6, 4))
    labels = []
    vals = []
    for cid in variant_order:
        labels.append(cid)
        vals.append((variants[cid].get("metrics") or {}).get(observable_metric_used, np.nan))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(observable_metric_used)
    plt.title("Observable Comparison")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "observable_compare.png")
    plt.close()

    summary = {
        "control": control_id,
        "treatment": treatment_id,
        "min_outputs": min_outputs,
        "writeback_ratio_min": writeback_ratio_min,
        "eta_std_min": eta_std_min,
        "observable_metric_requested": observable_metric,
        "observable_metric_used": observable_metric_used,
        "observable_min_rel_diff": observable_min_rel_diff,
        "observable_norm_eps": observable_norm_eps,
        "observable_abs_floor": observable_abs_floor,
        "observable_control_abs": observable_control_abs,
        "observable_treatment_abs": observable_treatment_abs,
        "observable_degenerate": observable_degenerate,
        "mag_energy_rel_diff": observable_diffs.get("mag_energy_final"),
        "field_energy_rel_diff": observable_diffs.get("field_energy_final"),
        "b_rms_rel_diff": observable_diffs.get("b_rms_final"),
        "control_ok": control_ok,
        "treatment_ok": treatment_ok,
        "writeback_ratio": writeback_ratio,
        "eta_std": eta_std,
        "floor_std": floor_std,
        "writeback_series_name": writeback_series_name,
        "writeback_std": writeback_std,
        "observable_rel_diff": observable_rel_diff,
        "coupling_effect_observed": coupling_effect_observed,
        "applied_field_enabled": (
            treatment_metrics.get("applied_field_enabled")
            if treatment_metrics.get("applied_field_enabled") is not None
            else control_metrics.get("applied_field_enabled")
        ),
        "applied_Bz_T": (
            treatment_metrics.get("applied_Bz_T")
            if treatment_metrics.get("applied_Bz_T") is not None
            else control_metrics.get("applied_Bz_T")
        ),
        "applied_Bz_expr": (
            treatment_metrics.get("applied_Bz_expr")
            if treatment_metrics.get("applied_Bz_expr") is not None
            else control_metrics.get("applied_Bz_expr")
        ),
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
        "treatment_update_count": treatment_updates,
        "treatment_writeback_count": treatment_writebacks,
        "treatment_writeback_ratio": writeback_ratio,
        "treatment_feedback_failures": treatment_failures,
        "treatment_feedback_target_used": treatment_target_used,
        "treatment_eta_std": eta_std,
        "treatment_floor_std": floor_std,
        "treatment_writeback_series": writeback_series_name,
        "treatment_writeback_std": writeback_std,
        "observable_metric_requested": observable_metric,
        "observable_metric_used": observable_metric_used,
        "observable_norm_eps": observable_norm_eps,
        "observable_abs_floor": observable_abs_floor,
        "observable_rel_diff": observable_rel_diff,
        "observable_control_abs": observable_control_abs,
        "observable_treatment_abs": observable_treatment_abs,
        "mag_energy_rel_diff": observable_diffs.get("mag_energy_final"),
        "field_energy_rel_diff": observable_diffs.get("field_energy_final"),
        "b_rms_rel_diff": observable_diffs.get("b_rms_final"),
        "coupling_effect_observed": coupling_effect_observed,
        "applied_field_enabled": (
            treatment_metrics.get("applied_field_enabled")
            if treatment_metrics.get("applied_field_enabled") is not None
            else control_metrics.get("applied_field_enabled")
        ),
        "applied_Bz_T": (
            treatment_metrics.get("applied_Bz_T")
            if treatment_metrics.get("applied_Bz_T") is not None
            else control_metrics.get("applied_Bz_T")
        ),
        "applied_Bz_expr": (
            treatment_metrics.get("applied_Bz_expr")
            if treatment_metrics.get("applied_Bz_expr") is not None
            else control_metrics.get("applied_Bz_expr")
        ),
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
