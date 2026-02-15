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
    return {
        "case_id": case_id,
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {},
    }


def resolve_path(repo_root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# EM Diffusion Non-Proxy Summary")
    lines.append("")
    lines.append(f"- diffusion_activation_ok: `{metrics.get('diffusion_activation_ok')}`")
    lines.append(f"- etaj2_scaling_ok: `{metrics.get('etaj2_scaling_ok')}`")
    lines.append(f"- conductor_interface_ok: `{metrics.get('conductor_interface_ok')}`")
    lines.append(f"- circuit_closed_loop_ok: `{metrics.get('circuit_closed_loop_ok')}`")
    lines.append(f"- etaJ2_gain_2_over_1: `{metrics.get('etaJ2_gain_2_over_1')}`")
    lines.append(f"- em_diffusion_nonproxy_pass: `{metrics.get('em_diffusion_nonproxy_pass')}`")
    lines.append("")
    lines.append("This gate checks explicit diffusion/resistive diagnostics and engineering EM interface closure.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matrix(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["check", "status", "detail"])
        for row in rows:
            writer.writerow([row["check"], row["status"], row["detail"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-proxy EM diffusion + interface closure gate.")
    parser.add_argument("--conductor-case", default="m26-a4-conductor-load-gate")
    parser.add_argument("--load-case", default="m26-d2-magnetic-load-interface")
    parser.add_argument("--circuit-case", default="m26-p2-circuit-load-integration")
    parser.add_argument("--diff0", default="outputs/m17-b2-p10-diff000/analysis/metrics_diffusion.json")
    parser.add_argument("--diff1", default="outputs/m17-b2-p10-diff100/analysis/metrics_diffusion.json")
    parser.add_argument("--diff2", default="outputs/m17-b2-p10-diff400/analysis/metrics_diffusion.json")
    parser.add_argument("--eta0", default="outputs/m11-b3-eta-sweep-0/analysis/metrics.json")
    parser.add_argument("--eta1", default="outputs/m11-b3-eta-sweep-1/analysis/metrics.json")
    parser.add_argument("--eta2", default="outputs/m11-b3-eta-sweep-2/analysis/metrics.json")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    conductor = load_passfail(repo_root, args.conductor_case)
    load = load_passfail(repo_root, args.load_case)
    circuit = load_passfail(repo_root, args.circuit_case)

    diff0 = read_json(resolve_path(repo_root, args.diff0))
    diff1 = read_json(resolve_path(repo_root, args.diff1))
    diff2 = read_json(resolve_path(repo_root, args.diff2))

    eta0 = read_json(resolve_path(repo_root, args.eta0))
    eta1 = read_json(resolve_path(repo_root, args.eta1))
    eta2 = read_json(resolve_path(repo_root, args.eta2))

    calls0 = as_float(diff0.get("diffusion_apply_calls") or diff0.get("num_applied"))
    calls1 = as_float(diff1.get("diffusion_apply_calls") or diff1.get("num_applied"))
    calls2 = as_float(diff2.get("diffusion_apply_calls") or diff2.get("num_applied"))

    coeff0 = as_float(diff0.get("effective_diffusion_coeff"))
    coeff1 = as_float(diff1.get("effective_diffusion_coeff"))
    coeff2 = as_float(diff2.get("effective_diffusion_coeff"))

    delta1 = as_float(diff1.get("diffusion_delta_u2_sum") or diff1.get("delta_u2_sum"))
    delta2 = as_float(diff2.get("diffusion_delta_u2_sum") or diff2.get("delta_u2_sum"))
    touch1 = as_float(diff1.get("diffusion_particles_touched") or diff1.get("num_modified"))
    touch2 = as_float(diff2.get("diffusion_particles_touched") or diff2.get("num_modified"))

    diffusion_activation_ok = bool(
        calls0 is not None
        and calls1 is not None
        and calls2 is not None
        and calls0 == 0.0
        and calls1 > 0.0
        and calls2 > 0.0
        and coeff0 is not None
        and coeff1 is not None
        and coeff2 is not None
        and coeff0 < coeff1 < coeff2
        and delta1 is not None
        and delta2 is not None
        and abs(delta1) > 0.0
        and abs(delta2) > 0.0
        and touch1 is not None
        and touch2 is not None
        and touch1 > 0.0
        and touch2 > 0.0
    )

    eta_scale0 = as_float(eta0.get("plasma_resistivity_scale"))
    eta_scale1 = as_float(eta1.get("plasma_resistivity_scale"))
    eta_scale2 = as_float(eta2.get("plasma_resistivity_scale"))
    etaJ20 = as_float(eta0.get("etaJ2_mean"))
    etaJ21 = as_float(eta1.get("etaJ2_mean"))
    etaJ22 = as_float(eta2.get("etaJ2_mean"))
    eta_samples0 = as_float(eta0.get("etaJ2_samples"))
    eta_samples1 = as_float(eta1.get("etaJ2_samples"))
    eta_samples2 = as_float(eta2.get("etaJ2_samples"))

    etaj2_scaling_ok = bool(
        eta_scale0 is not None
        and eta_scale1 is not None
        and eta_scale2 is not None
        and eta_scale0 < eta_scale1 < eta_scale2
        and etaJ20 is not None
        and etaJ21 is not None
        and etaJ22 is not None
        and etaJ20 <= etaJ21 <= etaJ22
        and eta_samples0 is not None
        and eta_samples1 is not None
        and eta_samples2 is not None
        and eta_samples0 > 0.0
        and eta_samples1 > 0.0
        and eta_samples2 > 0.0
    )

    etaJ2_gain_2_over_1 = None
    if etaJ21 is not None and etaJ22 is not None and etaJ21 > 0.0:
        etaJ2_gain_2_over_1 = float(etaJ22 / etaJ21)

    conductor_metrics = conductor["metrics"]
    load_metrics = load["metrics"]
    circuit_metrics = circuit["metrics"]

    conductor_interface_ok = bool(
        conductor["status"] == "PASS"
        and load["status"] == "PASS"
        and conductor_metrics.get("a4_full_gate_pass") is True
        and load_metrics.get("interface_ready") is True
        and as_float(load_metrics.get("force_proxy_peak_N")) is not None
        and as_float(load_metrics.get("force_proxy_peak_N")) > 1.0
        and as_float(load_metrics.get("dphi_dt_peak_V")) is not None
        and as_float(load_metrics.get("dphi_dt_peak_V")) > 1000.0
    )

    circuit_closed_loop_ok = bool(
        circuit["status"] == "PASS"
        and circuit_metrics.get("integrated_gate_pass") is True
        and circuit_metrics.get("p2_energy_chain_ok") is True
        and as_float(circuit_metrics.get("eta_recaptured")) is not None
        and as_float(circuit_metrics.get("eta_recaptured")) >= 0.3
    )

    em_diffusion_nonproxy_pass = bool(
        diffusion_activation_ok
        and etaj2_scaling_ok
        and conductor_interface_ok
        and circuit_closed_loop_ok
    )

    rows = [
        {
            "check": "diffusion_activation",
            "status": "PASS" if diffusion_activation_ok else "FAIL",
            "detail": "diffusion_apply_calls and effective_diffusion_coeff scale consistently from diff000->diff100->diff400",
        },
        {
            "check": "etaj2_scaling",
            "status": "PASS" if etaj2_scaling_ok else "FAIL",
            "detail": "etaJ2_mean increases monotonically with plasma_resistivity_scale sweep",
        },
        {
            "check": "conductor_interface",
            "status": "PASS" if conductor_interface_ok else "FAIL",
            "detail": "A4 conductor/load gate plus D1 load interface with force and dPhi/dt outputs",
        },
        {
            "check": "circuit_closed_loop",
            "status": "PASS" if circuit_closed_loop_ok else "FAIL",
            "detail": "P2 integrated circuit-plasma gate with energy chain closure",
        },
    ]

    metrics = {
        "conductor_case_id": args.conductor_case,
        "load_case_id": args.load_case,
        "circuit_case_id": args.circuit_case,
        "diffusion_activation_ok": bool(diffusion_activation_ok),
        "etaj2_scaling_ok": bool(etaj2_scaling_ok),
        "conductor_interface_ok": bool(conductor_interface_ok),
        "circuit_closed_loop_ok": bool(circuit_closed_loop_ok),
        "em_diffusion_nonproxy_pass": bool(em_diffusion_nonproxy_pass),
        "diffusion_calls_0": calls0,
        "diffusion_calls_1": calls1,
        "diffusion_calls_2": calls2,
        "diffusion_coeff_0": coeff0,
        "diffusion_coeff_1": coeff1,
        "diffusion_coeff_2": coeff2,
        "diffusion_delta_u2_1": delta1,
        "diffusion_delta_u2_2": delta2,
        "eta_scale_0": eta_scale0,
        "eta_scale_1": eta_scale1,
        "eta_scale_2": eta_scale2,
        "etaJ2_mean_0": etaJ20,
        "etaJ2_mean_1": etaJ21,
        "etaJ2_mean_2": etaJ22,
        "etaJ2_gain_2_over_1": etaJ2_gain_2_over_1,
    }

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
    write_summary(out_summary, metrics)
    write_matrix(out_matrix, rows)


if __name__ == "__main__":
    main()
