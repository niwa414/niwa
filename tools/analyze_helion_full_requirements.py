#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEUTERON_MASS_KG = 3.343583719e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19


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
        "path": str(path),
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": metrics,
    }


def load_metrics(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "metrics.json"
    return read_json(path) if path.exists() else {}


def load_pulse_kpi(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "metrics_pulse_kpi.json"
    data = read_json(path)
    pulse_kpi = data.get("pulse_kpi")
    return pulse_kpi if isinstance(pulse_kpi, dict) else {}


def ti_kev_from_u2(u2_m2_s2: float | None, ion_mass_kg: float) -> float | None:
    if u2_m2_s2 is None or u2_m2_s2 <= 0.0:
        return None
    energy_j = 0.5 * ion_mass_kg * u2_m2_s2
    energy_ev = energy_j / ELEMENTARY_CHARGE_C
    return float(energy_ev / 1.0e3)


def find_warpx_args_path(repo_root: Path, case_id: str, override: str | None) -> Path | None:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path
    candidates = [
        repo_root / "outputs" / case_id / "raw" / "inputs" / "inputs" / "warpx_args.json",
        repo_root / "cases" / case_id / "inputs" / "warpx_args.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def pulse_width_from_warpx_args(path: Path | None) -> tuple[float | None, int | None, float | None]:
    if path is None or not path.exists():
        return None, None, None
    data = read_json(path)
    max_steps = as_float(data.get("max_steps"))
    dt = as_float(data.get("dt"))
    if max_steps is None or dt is None:
        return None, int(max_steps) if max_steps is not None else None, dt
    return float(max_steps * dt), int(max_steps), float(dt)


def build_rows(metrics: dict) -> list[dict]:
    return [
        {
            "requirement": "Formation (micro-instability + timing guidance)",
            "status": "PASS" if metrics.get("formation_microinstability_timing_ok") else "FAIL",
            "detail": "3D growth/damping + response sensitivity + coil timing window recommendation",
        },
        {
            "requirement": "Translation & acceleration",
            "status": "PASS" if metrics.get("translation_acceleration_ok") else "FAIL",
            "detail": "formation/translation monotonicity and centroid shift coverage",
        },
        {
            "requirement": "Merging quality & sensitivity",
            "status": "PASS" if metrics.get("merging_quality_ok") else "FAIL",
            "detail": "phase compression sweep + seed/drift sensitivity suite",
        },
        {
            "requirement": "Compression to keV regime",
            "status": "PASS" if metrics.get("compression_kev_ok") else "FAIL",
            "detail": "Ti proxy from u2 tail exceeds 8 keV and >20 keV targets",
        },
        {
            "requirement": "Stability boundaries (tilt/interchange/shearing proxies)",
            "status": "PASS" if metrics.get("stability_boundaries_ok") else "FAIL",
            "detail": "tilt dual-regime PASS + seed/drift proxy separation",
        },
        {
            "requirement": "Expansion + direct electricity recapture",
            "status": "PASS" if metrics.get("expansion_recapture_ok") else "FAIL",
            "detail": "closed-loop circuit recapture and magnetic-load outputs",
        },
        {
            "requirement": "Q_eng style engineering KPI",
            "status": "PASS" if metrics.get("q_eng_modeling_ok") else "FAIL",
            "detail": "pulse-width + energy closure + delivered/recaptured efficiency",
        },
        {
            "requirement": "Multi-fidelity stack (MHD + Hybrid + electron transport + ML)",
            "status": "PASS" if metrics.get("multi_fidelity_stack_ok") else "FAIL",
            "detail": "MHD/Hybrid baselines and B3+ML1 transport closure chain",
        },
        {
            "requirement": "Circuit/EM coupling + conductor/load interface",
            "status": "PASS" if metrics.get("circuit_em_coupling_ok") else "FAIL",
            "detail": "RLC recapture + conductor response + engineering load handoff",
        },
        {
            "requirement": "HPC + engineering workflow",
            "status": "PASS" if metrics.get("hpc_engineering_ok") else "FAIL",
            "detail": "regression pack + benchmark + synthetic diagnostics",
        },
    ]


def write_matrix_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["requirement", "status", "detail"])
        for row in rows:
            writer.writerow([row["requirement"], row["status"], row["detail"]])


def write_summary(path: Path, metrics: dict, rows: list[dict]) -> None:
    lines = []
    lines.append("# Helion Full Requirements Gate")
    lines.append("")
    lines.append(f"- helion_full_requirements_pass: `{metrics.get('helion_full_requirements_pass')}`")
    lines.append(f"- formation_microinstability_timing_ok: `{metrics.get('formation_microinstability_timing_ok')}`")
    lines.append(f"- translation_acceleration_ok: `{metrics.get('translation_acceleration_ok')}`")
    lines.append(f"- merging_quality_ok: `{metrics.get('merging_quality_ok')}`")
    lines.append(f"- compression_kev_ok: `{metrics.get('compression_kev_ok')}`")
    lines.append(f"- stability_boundaries_ok: `{metrics.get('stability_boundaries_ok')}`")
    lines.append(f"- expansion_recapture_ok: `{metrics.get('expansion_recapture_ok')}`")
    lines.append(f"- q_eng_modeling_ok: `{metrics.get('q_eng_modeling_ok')}`")
    lines.append("")
    lines.append(f"- ti_off_keV_proxy: `{metrics.get('ti_off_keV_proxy')}`")
    lines.append(f"- ti_end_keV_proxy: `{metrics.get('ti_end_keV_proxy')}`")
    lines.append(f"- q_eng_proxy: `{metrics.get('q_eng_proxy')}`")
    lines.append(f"- pulse_width_s: `{metrics.get('pulse_width_s')}`")
    lines.append("")
    lines.append("## Requirement Matrix")
    lines.append("")
    lines.append("| Requirement | Status | Detail |")
    lines.append("| --- | --- | --- |")
    for row in rows:
        lines.append(f"| {row['requirement']} | {row['status']} | {row['detail']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full Helion-style requirement coverage using local evidence.")
    parser.add_argument("--formation-case", default="m13-a2-formation-translation-nonfast-lite")
    parser.add_argument("--translation-case", default="m5-a2-formation-translation-gate")
    parser.add_argument("--response-case", default="m16-a3-circuit-radius-response")
    parser.add_argument("--tilt-dual-case", default="m26-b2-tilt-dual-regime-gate")
    parser.add_argument("--tilt-growth-case", default="m17-b2-tilt-growth-baseline")
    parser.add_argument("--compression-case", default="m26-p18-phase-compression-gate")
    parser.add_argument("--circuit-case", default="m26-p2-circuit-load-integration")
    parser.add_argument("--conductor-case", default="m26-a4-conductor-load-gate")
    parser.add_argument("--transport-case", default="m26-b3-transport-closure-gate")
    parser.add_argument("--surrogate-case", default="m26-ml1-transport-surrogate")
    parser.add_argument("--hybrid-rz-case", default="m1-b2-hybrid-smoke")
    parser.add_argument("--hybrid-3d-case", default="m2-b2-hybrid-3d-smoke")
    parser.add_argument("--hybrid-compare-case", default="m8-b1-mhd-vs-hybrid-compare")
    parser.add_argument("--regression-case", default="m24-c3-regression-pack")
    parser.add_argument("--benchmark-case", default="m23-c1-historical-benchmark-belova")
    parser.add_argument("--diagnostics-case", default="m22-c2-synthetic-diagnostics-mvp")
    parser.add_argument("--qeng-case", default="m17-b2-p19-off352-drive200-repeat512")
    parser.add_argument("--qeng-warpx-args", default=None)
    parser.add_argument("--ion-mass-kg", type=float, default=DEUTERON_MASS_KG)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    formation = load_passfail(repo_root, args.formation_case)
    translation = load_passfail(repo_root, args.translation_case)
    response = load_passfail(repo_root, args.response_case)
    tilt_dual = load_passfail(repo_root, args.tilt_dual_case)
    tilt_growth = load_passfail(repo_root, args.tilt_growth_case)
    compression = load_passfail(repo_root, args.compression_case)
    circuit = load_passfail(repo_root, args.circuit_case)
    conductor = load_passfail(repo_root, args.conductor_case)
    transport = load_passfail(repo_root, args.transport_case)
    surrogate = load_passfail(repo_root, args.surrogate_case)
    hybrid_rz = load_passfail(repo_root, args.hybrid_rz_case)
    hybrid_3d = load_passfail(repo_root, args.hybrid_3d_case)
    hybrid_compare = load_passfail(repo_root, args.hybrid_compare_case)
    regression = load_passfail(repo_root, args.regression_case)
    benchmark = load_passfail(repo_root, args.benchmark_case)
    diagnostics = load_passfail(repo_root, args.diagnostics_case)
    qeng_pf = load_passfail(repo_root, args.qeng_case)
    qeng_metrics = load_metrics(repo_root, args.qeng_case)
    qeng_pulse = load_pulse_kpi(repo_root, args.qeng_case)

    fm = formation["metrics"]
    trm = translation["metrics"]
    rm = response["metrics"]
    tm = tilt_dual["metrics"]
    gm = tilt_growth["metrics"]
    cm = compression["metrics"]
    pim = circuit["metrics"]
    a4m = conductor["metrics"]
    b3m = transport["metrics"]
    mlm = surrogate["metrics"]
    qpm = qeng_metrics

    formation_time_frac = as_float(fm.get("formation_time_frac"))
    psi_persist = as_float(fm.get("psi_closed_persist_outputs"))
    translation_monotonic_frac = as_float(trm.get("translation_monotonic_frac"))
    centroid_shift_abs = as_float(trm.get("centroid_shift_abs"))

    formation_ok = (
        formation["status"] == "PASS"
        and fm.get("frc_formed") is True
        and formation_time_frac is not None
        and formation_time_frac <= 0.2
        and psi_persist is not None
        and psi_persist >= 4.0
    )

    translation_acceleration_ok = (
        translation["status"] == "PASS"
        and translation_monotonic_frac is not None
        and translation_monotonic_frac >= 0.8
        and centroid_shift_abs is not None
        and centroid_shift_abs >= 0.02
    )

    merging_quality_ok = (
        compression["status"] == "PASS"
        and cm.get("p18_full_gate_pass") is True
        and cm.get("phase_detected_all") is True
        and cm.get("b_energy_peak_matches_off_step_all") is True
        and as_float(cm.get("rho_delta_rel_phase_gain")) is not None
        and as_float(cm.get("rho_delta_rel_phase_gain")) >= 5.0
        and tm.get("seed_drift_suite_complete") is True
    )

    u2_off = as_float((qeng_pulse.get("tail_proxy") or {}).get("u2_p99_at_stepOff"))
    u2_end = as_float((qeng_pulse.get("tail_proxy") or {}).get("u2_p99_at_stepEnd"))
    ti_off_kev = ti_kev_from_u2(u2_off, ion_mass_kg=args.ion_mass_kg)
    ti_end_kev = ti_kev_from_u2(u2_end, ion_mass_kg=args.ion_mass_kg)

    compression_kev_ok = (
        ti_off_kev is not None
        and ti_off_kev >= 8.0
        and ti_end_kev is not None
        and ti_end_kev >= 20.0
        and compression["status"] == "PASS"
    )

    tilt_boundary_ok = tilt_dual["status"] == "PASS" and tm.get("dual_regime_pass") is True
    interchange_proxy_strength = as_float(tm.get("seed_effect_rel"))
    shearing_proxy_strength = as_float(tm.get("drift_effect_rel"))
    interchange_proxy_ok = interchange_proxy_strength is not None and interchange_proxy_strength >= 5.0e-2
    shearing_proxy_ok = shearing_proxy_strength is not None and shearing_proxy_strength >= 1.0e-8
    stability_boundaries_ok = tilt_boundary_ok and interchange_proxy_ok and shearing_proxy_ok

    growth_gamma = as_float(gm.get("gamma_fit"))
    growth_t0 = as_float(gm.get("fit_window_time0"))
    growth_t1 = as_float(gm.get("fit_window_time1"))
    damping_gamma_negative = tm.get("damping_gamma_negative") is True
    driver_response_ratio = as_float(rm.get("driver_response_ratio"))
    response_ok = (
        response["status"] == "PASS"
        and rm.get("compare_pass") is True
        and rm.get("driver_response_ok") is True
        and driver_response_ratio is not None
        and driver_response_ratio >= 1.1
    )
    growth_window_valid = (
        growth_gamma is not None
        and growth_gamma > 0.0
        and growth_t0 is not None
        and growth_t1 is not None
        and growth_t1 > growth_t0
    )
    recommended_shift_ns = None
    recommended_window_start_ns = None
    recommended_window_end_ns = None
    if growth_window_valid:
        window_ns = (growth_t1 - growth_t0) * 1.0e9
        recommended_shift_ns = 0.5 * window_ns
        recommended_window_start_ns = growth_t0 * 1.0e9 - recommended_shift_ns
        recommended_window_end_ns = growth_t0 * 1.0e9
    timing_guidance_ready = (
        response_ok
        and growth_window_valid
        and damping_gamma_negative
        and recommended_shift_ns is not None
        and math.isfinite(recommended_shift_ns)
        and recommended_shift_ns > 0.0
    )
    formation_microinstability_timing_ok = formation_ok and timing_guidance_ready

    expansion_recapture_ok = (
        circuit["status"] == "PASS"
        and pim.get("integrated_gate_pass") is True
        and as_float(pim.get("eta_recaptured")) is not None
        and as_float(pim.get("eta_recaptured")) >= 0.3
        and as_float(pim.get("e_load_J")) is not None
        and as_float(pim.get("e_load_J")) > 0.0
        and as_float(pim.get("vind_peak_V")) is not None
        and as_float(pim.get("vind_peak_V")) > 1.0e3
        and pim.get("mainline_load_energy_chain_present") is True
    )

    e_in = as_float(qpm.get("e_in_J"))
    e_load = as_float(qpm.get("e_load_J"))
    e_rec = as_float(qpm.get("e_recaptured_J"))
    energy_residual_rel = as_float(qpm.get("energy_residual_rel"))
    q_eng_proxy = None
    if e_in is not None and e_in > 0.0 and e_load is not None and e_rec is not None:
        q_eng_proxy = float((e_load + e_rec) / e_in)

    warpx_args_path = find_warpx_args_path(repo_root, args.qeng_case, args.qeng_warpx_args)
    pulse_width_s, qeng_max_steps, qeng_dt = pulse_width_from_warpx_args(warpx_args_path)
    pulse_short_ok = pulse_width_s is not None and pulse_width_s < 1.0e-3
    q_eng_modeling_ok = (
        qeng_pf["status"] == "PASS"
        and q_eng_proxy is not None
        and q_eng_proxy >= 0.8
        and energy_residual_rel is not None
        and energy_residual_rel <= 1.0e-5
        and pulse_short_ok
    )

    hybrid_stack_ok = (
        hybrid_rz["status"] == "PASS"
        and hybrid_3d["status"] == "PASS"
        and hybrid_compare["status"] == "PASS"
    )
    transport_stack_ok = (
        transport["status"] == "PASS"
        and b3m.get("b3_full_gate_pass") is True
        and surrogate["status"] == "PASS"
        and mlm.get("surrogate_ready") is True
    )
    multi_fidelity_stack_ok = formation["status"] == "PASS" and hybrid_stack_ok and transport_stack_ok

    circuit_em_coupling_ok = (
        circuit["status"] == "PASS"
        and pim.get("integrated_gate_pass") is True
        and conductor["status"] == "PASS"
        and a4m.get("a4_full_gate_pass") is True
        and as_float(a4m.get("load_force_proxy_peak_N")) is not None
        and as_float(a4m.get("load_force_proxy_peak_N")) > 1.0
        and as_float(a4m.get("load_eta_recaptured")) is not None
        and as_float(a4m.get("load_eta_recaptured")) >= 0.3
    )

    hpc_engineering_ok = (
        regression["status"] == "PASS"
        and benchmark["status"] == "PASS"
        and diagnostics["status"] == "PASS"
    )

    helion_full_requirements_pass = (
        formation_microinstability_timing_ok
        and translation_acceleration_ok
        and merging_quality_ok
        and compression_kev_ok
        and stability_boundaries_ok
        and expansion_recapture_ok
        and q_eng_modeling_ok
        and multi_fidelity_stack_ok
        and circuit_em_coupling_ok
        and hpc_engineering_ok
    )

    metrics = {
        "formation_case_id": args.formation_case,
        "response_case_id": args.response_case,
        "translation_case_id": args.translation_case,
        "tilt_dual_case_id": args.tilt_dual_case,
        "tilt_growth_case_id": args.tilt_growth_case,
        "compression_case_id": args.compression_case,
        "circuit_case_id": args.circuit_case,
        "conductor_case_id": args.conductor_case,
        "transport_case_id": args.transport_case,
        "surrogate_case_id": args.surrogate_case,
        "hybrid_rz_case_id": args.hybrid_rz_case,
        "hybrid_3d_case_id": args.hybrid_3d_case,
        "hybrid_compare_case_id": args.hybrid_compare_case,
        "regression_case_id": args.regression_case,
        "benchmark_case_id": args.benchmark_case,
        "diagnostics_case_id": args.diagnostics_case,
        "qeng_case_id": args.qeng_case,
        "formation_ok": bool(formation_ok),
        "translation_acceleration_ok": bool(translation_acceleration_ok),
        "merging_quality_ok": bool(merging_quality_ok),
        "compression_kev_ok": bool(compression_kev_ok),
        "stability_boundaries_ok": bool(stability_boundaries_ok),
        "formation_microinstability_timing_ok": bool(formation_microinstability_timing_ok),
        "timing_guidance_ready": bool(timing_guidance_ready),
        "expansion_recapture_ok": bool(expansion_recapture_ok),
        "q_eng_modeling_ok": bool(q_eng_modeling_ok),
        "multi_fidelity_stack_ok": bool(multi_fidelity_stack_ok),
        "circuit_em_coupling_ok": bool(circuit_em_coupling_ok),
        "hpc_engineering_ok": bool(hpc_engineering_ok),
        "helion_full_requirements_pass": bool(helion_full_requirements_pass),
        "formation_time_frac": formation_time_frac,
        "psi_closed_persist_outputs": psi_persist,
        "translation_monotonic_frac": translation_monotonic_frac,
        "centroid_shift_abs": centroid_shift_abs,
        "seed_effect_rel": interchange_proxy_strength,
        "drift_effect_rel": shearing_proxy_strength,
        "driver_response_ratio": driver_response_ratio,
        "growth_gamma_fit": growth_gamma,
        "growth_fit_window_t0_s": growth_t0,
        "growth_fit_window_t1_s": growth_t1,
        "recommended_timing_shift_ns": recommended_shift_ns,
        "recommended_window_start_ns": recommended_window_start_ns,
        "recommended_window_end_ns": recommended_window_end_ns,
        "ti_off_keV_proxy": ti_off_kev,
        "ti_end_keV_proxy": ti_end_kev,
        "u2_p99_off_m2_s2": u2_off,
        "u2_p99_end_m2_s2": u2_end,
        "q_eng_proxy": q_eng_proxy,
        "energy_residual_rel_qeng_case": energy_residual_rel,
        "pulse_width_s": pulse_width_s,
        "qeng_max_steps": qeng_max_steps,
        "qeng_dt_s": qeng_dt,
        "pulse_short_ok": bool(pulse_short_ok),
        "qeng_warpx_args_path": str(warpx_args_path) if warpx_args_path else None,
    }

    rows = build_rows(metrics)

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
    write_matrix_csv(out_matrix, rows)


if __name__ == "__main__":
    main()
