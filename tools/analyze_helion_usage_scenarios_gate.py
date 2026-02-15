#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_passfail(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    data = read_json(path)
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    return {
        "case_id": case_id,
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": metrics,
    }


def load_json_rel(repo_root: Path, rel_path: str) -> dict:
    path = repo_root / rel_path
    return read_json(path)


def scenario_rows(metrics: dict) -> list[dict]:
    return [
        {
            "scenario": "design_accelerator",
            "status": "PASS" if metrics.get("design_accelerator_ok") else "FAIL",
            "detail": "pre-hardware design tradeoff loop (control law + inverse solver + timing guidance)",
        },
        {
            "scenario": "shot_planning_param_scan",
            "status": "PASS" if metrics.get("shot_planning_param_scan_ok") else "FAIL",
            "detail": "parameter-space sweeps for merge/compression/stability sensitivity",
        },
        {
            "scenario": "post_shot_reconstruction",
            "status": "PASS" if metrics.get("post_shot_reconstruction_ok") else "FAIL",
            "detail": "measured-boundary replay proxies + synthetic diagnostics backfill",
        },
        {
            "scenario": "vv_calibration_validation",
            "status": "PASS" if metrics.get("vv_calibration_validation_ok") else "FAIL",
            "detail": "benchmark + synthetic diagnostics + regression/restart consistency",
        },
        {
            "scenario": "engineering_interface_outputs",
            "status": "PASS" if metrics.get("engineering_interface_outputs_ok") else "FAIL",
            "detail": "circuit response + magnetic loads + conductor response handoff",
        },
        {
            "scenario": "ml_surrogate_closed_loop",
            "status": "PASS" if metrics.get("ml_surrogate_closed_loop_ok") else "FAIL",
            "detail": "surrogate inference + OOS checks + fallback policy",
        },
    ]


def write_matrix(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "status", "detail"])
        for row in rows:
            writer.writerow([row["scenario"], row["status"], row["detail"]])


def write_summary(path: Path, metrics: dict, rows: list[dict]) -> None:
    lines = []
    lines.append("# Helion Usage Scenarios Gate")
    lines.append("")
    lines.append(f"- all_usage_scenarios_pass: `{metrics.get('all_usage_scenarios_pass')}`")
    lines.append(f"- design_accelerator_ok: `{metrics.get('design_accelerator_ok')}`")
    lines.append(f"- shot_planning_param_scan_ok: `{metrics.get('shot_planning_param_scan_ok')}`")
    lines.append(f"- post_shot_reconstruction_ok: `{metrics.get('post_shot_reconstruction_ok')}`")
    lines.append(f"- vv_calibration_validation_ok: `{metrics.get('vv_calibration_validation_ok')}`")
    lines.append(f"- engineering_interface_outputs_ok: `{metrics.get('engineering_interface_outputs_ok')}`")
    lines.append(f"- ml_surrogate_closed_loop_ok: `{metrics.get('ml_surrogate_closed_loop_ok')}`")
    lines.append("")
    lines.append("| Scenario | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['status']} | {row['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate Helion engineering usage scenarios.")
    parser.add_argument("--full-req-case", default="m27-d1-helion-full-requirements-gate")
    parser.add_argument("--tilt-dual-case", default="m26-b2-tilt-dual-regime-gate")
    parser.add_argument("--compression-case", default="m26-p18-phase-compression-gate")
    parser.add_argument("--circuit-case", default="m26-p2-circuit-load-integration")
    parser.add_argument("--load-case", default="m26-d2-magnetic-load-interface")
    parser.add_argument("--conductor-case", default="m26-a4-conductor-load-gate")
    parser.add_argument("--surrogate-case", default="m26-ml1-transport-surrogate")
    parser.add_argument("--transport-case", default="m26-b3-transport-closure-gate")
    parser.add_argument("--benchmark-case", default="m23-c1-historical-benchmark-belova")
    parser.add_argument("--diagnostics-case", default="m22-c2-synthetic-diagnostics-mvp")
    parser.add_argument("--regression-case", default="m24-c3-regression-pack")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    full_req = load_passfail(repo_root, args.full_req_case)
    tilt = load_passfail(repo_root, args.tilt_dual_case)
    compression = load_passfail(repo_root, args.compression_case)
    circuit = load_passfail(repo_root, args.circuit_case)
    load = load_passfail(repo_root, args.load_case)
    conductor = load_passfail(repo_root, args.conductor_case)
    surrogate = load_passfail(repo_root, args.surrogate_case)
    transport = load_passfail(repo_root, args.transport_case)
    benchmark = load_passfail(repo_root, args.benchmark_case)
    diagnostics = load_passfail(repo_root, args.diagnostics_case)
    regression = load_passfail(repo_root, args.regression_case)

    full_metrics = full_req["metrics"]
    tilt_m = tilt["metrics"]
    compression_m = compression["metrics"]
    circuit_m = circuit["metrics"]
    load_m = load["metrics"]
    conductor_m = conductor["metrics"]
    surrogate_m = surrogate["metrics"]
    transport_m = transport["metrics"]

    p20 = load_json_rel(repo_root, "outputs/p20-control-law/analysis/metrics_p20_control_law.json")
    p21 = load_json_rel(repo_root, "outputs/p21-solver/analysis/metrics_p21_solver.json")
    p22 = load_json_rel(repo_root, "outputs/p22-control-law/analysis/metrics_p22_control_law.json")
    p24 = load_json_rel(repo_root, "outputs/p24-control-law/analysis/metrics_p24_control_law.json")

    p20_r2_circ = as_float(((p20.get("model_circuit") or {}).get("fit") or {}).get("r2"))
    p22_r2_circ = as_float(((p22.get("model_circuit") or {}).get("fit") or {}).get("r2"))
    p24_r2_phase = as_float(((p24.get("model_formation_phase") or {}).get("fit") or {}).get("r2"))
    p24_phase_abs_log_err = as_float((p24.get("model_formation_phase") or {}).get("max_abs_log_err"))
    p21_max_u2_rel_err = as_float(
        (((p21.get("self_check") or {}).get("max_rel_err") or {}).get("u2_pred"))
    )

    design_accelerator_ok = (
        full_req["status"] == "PASS"
        and full_metrics.get("formation_microinstability_timing_ok") is True
        and p20_r2_circ is not None
        and p20_r2_circ >= 0.99
        and p22_r2_circ is not None
        and p22_r2_circ >= 0.99
        and p24_r2_phase is not None
        and p24_r2_phase >= 0.99
        and p24_phase_abs_log_err is not None
        and p24_phase_abs_log_err <= 0.2
        and p21_max_u2_rel_err is not None
        and p21_max_u2_rel_err <= 0.1
    )

    shot_planning_param_scan_ok = (
        tilt["status"] == "PASS"
        and tilt_m.get("dual_regime_pass") is True
        and compression["status"] == "PASS"
        and compression_m.get("p18_full_gate_pass") is True
        and as_float(compression_m.get("rho_delta_rel_phase_gain")) is not None
        and as_float(compression_m.get("rho_delta_rel_phase_gain")) >= 5.0
        and as_float(tilt_m.get("seed_effect_rel")) is not None
        and as_float(tilt_m.get("seed_effect_rel")) >= 5.0e-2
    )

    benchmark_summary = load_json_rel(
        repo_root, "outputs/m23-c1-historical-benchmark-belova/analysis/benchmark_summary.json"
    )
    bdot_points = as_float(read_json(repo_root / "outputs/m22-c2-synthetic-diagnostics-mvp/analysis/metrics.json").get("bdot_points"))
    interf_points = as_float(read_json(repo_root / "outputs/m22-c2-synthetic-diagnostics-mvp/analysis/metrics.json").get("interferometer_points"))
    post_shot_reconstruction_ok = (
        benchmark["status"] == "PASS"
        and diagnostics["status"] == "PASS"
        and as_float(benchmark_summary.get("alignment_corrcoef")) is not None
        and as_float(benchmark_summary.get("alignment_corrcoef")) >= 0.9
        and as_float(benchmark_summary.get("alignment_r2")) is not None
        and as_float(benchmark_summary.get("alignment_r2")) >= 0.85
        and bdot_points is not None
        and bdot_points >= 20.0
        and interf_points is not None
        and interf_points >= 100.0
        and circuit["status"] == "PASS"
        and circuit_m.get("integrated_gate_pass") is True
    )

    vv_calibration_validation_ok = (
        benchmark["status"] == "PASS"
        and diagnostics["status"] == "PASS"
        and regression["status"] == "PASS"
    )

    engineering_interface_outputs_ok = (
        circuit["status"] == "PASS"
        and circuit_m.get("integrated_gate_pass") is True
        and load["status"] == "PASS"
        and load_m.get("interface_ready") is True
        and as_float(load_m.get("force_proxy_peak_N")) is not None
        and as_float(load_m.get("force_proxy_peak_N")) > 1.0
        and as_float(load_m.get("p_mag_peak_Pa")) is not None
        and as_float(load_m.get("p_mag_peak_Pa")) > 0.0
        and conductor["status"] == "PASS"
        and conductor_m.get("a4_full_gate_pass") is True
    )

    ml_surrogate_closed_loop_ok = (
        surrogate["status"] == "PASS"
        and surrogate_m.get("surrogate_ready") is True
        and surrogate_m.get("fallback_policy_defined") is True
        and surrogate_m.get("fallback_simulated_triggered") is True
        and as_float(surrogate_m.get("oos_u2_rel_err_max")) is not None
        and as_float(surrogate_m.get("oos_u2_rel_err_max")) <= 0.1
        and as_float(surrogate_m.get("oos_rho_cls_acc")) is not None
        and as_float(surrogate_m.get("oos_rho_cls_acc")) >= 0.85
        and transport["status"] == "PASS"
        and transport_m.get("b3_full_gate_pass") is True
    )

    all_usage_scenarios_pass = (
        design_accelerator_ok
        and shot_planning_param_scan_ok
        and post_shot_reconstruction_ok
        and vv_calibration_validation_ok
        and engineering_interface_outputs_ok
        and ml_surrogate_closed_loop_ok
    )

    metrics = {
        "full_requirements_case_id": args.full_req_case,
        "design_accelerator_ok": bool(design_accelerator_ok),
        "shot_planning_param_scan_ok": bool(shot_planning_param_scan_ok),
        "post_shot_reconstruction_ok": bool(post_shot_reconstruction_ok),
        "vv_calibration_validation_ok": bool(vv_calibration_validation_ok),
        "engineering_interface_outputs_ok": bool(engineering_interface_outputs_ok),
        "ml_surrogate_closed_loop_ok": bool(ml_surrogate_closed_loop_ok),
        "all_usage_scenarios_pass": bool(all_usage_scenarios_pass),
        "p20_circuit_r2": p20_r2_circ,
        "p22_circuit_r2": p22_r2_circ,
        "p24_phase_r2": p24_r2_phase,
        "p24_phase_max_abs_log_err": p24_phase_abs_log_err,
        "p21_selfcheck_u2_max_rel_err": p21_max_u2_rel_err,
        "benchmark_alignment_corrcoef": as_float(benchmark_summary.get("alignment_corrcoef")),
        "benchmark_alignment_r2": as_float(benchmark_summary.get("alignment_r2")),
        "synthetic_bdot_points": bdot_points,
        "synthetic_interferometer_points": interf_points,
        "load_force_proxy_peak_N": as_float(load_m.get("force_proxy_peak_N")),
        "load_p_mag_peak_Pa": as_float(load_m.get("p_mag_peak_Pa")),
        "ml_oos_u2_rel_err_max": as_float(surrogate_m.get("oos_u2_rel_err_max")),
        "ml_oos_rho_cls_acc": as_float(surrogate_m.get("oos_rho_cls_acc")),
    }

    rows = scenario_rows(metrics)

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    out_matrix = Path(args.matrix)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()
    if not out_matrix.is_absolute():
        out_matrix = (repo_root / out_matrix).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_summary, metrics, rows)
    write_matrix(out_matrix, rows)


if __name__ == "__main__":
    main()
