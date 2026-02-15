#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_passfail(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    data = read_json(path)
    return {
        "case_id": case_id,
        "path": str(path),
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}),
    }


def value(metrics: dict, *keys):
    for key in keys:
        if key in metrics and metrics.get(key) is not None:
            return metrics.get(key)
    return None


def as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# Circuit + Load Integration Gate")
    lines.append("")
    lines.append(f"- integrated_gate_pass: `{metrics.get('integrated_gate_pass')}`")
    lines.append(f"- a3_closed_loop_ok: `{metrics.get('a3_closed_loop_ok')}`")
    lines.append(f"- p2_energy_chain_ok: `{metrics.get('p2_energy_chain_ok')}`")
    lines.append(f"- p2_drift_invariant_ok: `{metrics.get('p2_drift_invariant_ok')}`")
    lines.append(f"- d1_load_interface_ok: `{metrics.get('d1_load_interface_ok')}`")
    lines.append(f"- mainline_load_embedded: `{metrics.get('mainline_load_embedded')}`")
    lines.append("")
    lines.append("This gate verifies that magnetic-load outputs are part of the regular circuit path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate A3/P2/D1 circuit-load evidence into one gate.")
    parser.add_argument("--a3-case", default="m5-a3-closed-loop-strong")
    parser.add_argument("--p2-mainline-case", default="m17-b2-circuit-mvp-mainline-Rload2000-coil4")
    parser.add_argument("--p2-drift-case", default="m17-b2-circuit-mvp-drift-summary")
    parser.add_argument("--d1-case", default="m26-d2-magnetic-load-interface")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    a3 = load_passfail(repo_root, args.a3_case)
    p2 = load_passfail(repo_root, args.p2_mainline_case)
    drift = load_passfail(repo_root, args.p2_drift_case)
    d1 = load_passfail(repo_root, args.d1_case)

    a3m = a3["metrics"]
    p2m = p2["metrics"]
    dm = drift["metrics"]
    d1m = d1["metrics"]

    a3_closed_loop_ok = (
        a3["status"] == "PASS"
        and bool(a3m.get("strong_coupling_enabled") is True)
        and as_float(a3m.get("circuit_update_fraction")) is not None
        and as_float(a3m.get("circuit_update_fraction")) >= 0.9
        and as_float(a3m.get("driver_writeback_fraction")) is not None
        and as_float(a3m.get("driver_writeback_fraction")) >= 0.9
    )

    eta_recaptured = as_float(value(p2m, "eta_recaptured"))
    energy_residual_rel = as_float(value(p2m, "energy_residual_rel"))
    e_load_j = as_float(value(p2m, "e_load_J"))
    vind_peak_v = as_float(value(p2m, "vind_peak_V"))

    p2_metrics_present = p2["exists"] and isinstance(p2m, dict) and bool(p2m)

    p2_energy_chain_ok = (
        p2_metrics_present
        and eta_recaptured is not None
        and eta_recaptured >= 0.3
        and energy_residual_rel is not None
        and energy_residual_rel <= 1.0e-5
        and e_load_j is not None
        and e_load_j > 0.0
        and vind_peak_v is not None
        and vind_peak_v > 1.0e3
    )

    drift_rel_diff_max = as_float(value(dm, "drift_rel_diff_max"))
    drift_observable_in_b2 = dm.get("drift_observable_in_b2")
    p2_drift_invariant_ok = (
        drift["status"] == "PASS"
        and drift_observable_in_b2 is False
        and drift_rel_diff_max is not None
        and drift_rel_diff_max <= 0.02
    )

    d1_load_interface_ok = (
        d1["status"] == "PASS"
        and d1m.get("interface_ready") is True
        and as_float(d1m.get("force_proxy_peak_N")) is not None
        and as_float(d1m.get("force_proxy_peak_N")) > 1.0
        and as_float(d1m.get("p_mag_peak_Pa")) is not None
        and as_float(d1m.get("p_mag_peak_Pa")) > 0.0
    )

    mainline_load_embedded = p2m.get("load_interface_ready") is True
    mainline_load_force_peak = as_float(p2m.get("load_force_proxy_peak_N"))
    mainline_load_energy_chain_present = p2m.get("load_energy_chain_present") is True

    integrated_gate_pass = (
        a3_closed_loop_ok
        and p2_energy_chain_ok
        and p2_drift_invariant_ok
        and d1_load_interface_ok
        and mainline_load_embedded
        and mainline_load_energy_chain_present
        and mainline_load_force_peak is not None
        and mainline_load_force_peak > 1.0
    )

    metrics = {
        "a3_case_id": args.a3_case,
        "p2_mainline_case_id": args.p2_mainline_case,
        "p2_drift_case_id": args.p2_drift_case,
        "d1_case_id": args.d1_case,
        "a3_case_pass": a3["status"] == "PASS",
        "p2_mainline_case_pass": bool(p2_metrics_present),
        "p2_mainline_case_status": p2["status"],
        "p2_drift_case_pass": drift["status"] == "PASS",
        "d1_case_pass": d1["status"] == "PASS",
        "a3_closed_loop_ok": bool(a3_closed_loop_ok),
        "p2_energy_chain_ok": bool(p2_energy_chain_ok),
        "p2_drift_invariant_ok": bool(p2_drift_invariant_ok),
        "d1_load_interface_ok": bool(d1_load_interface_ok),
        "mainline_load_embedded": bool(mainline_load_embedded),
        "mainline_load_energy_chain_present": bool(mainline_load_energy_chain_present),
        "mainline_load_force_peak_N": mainline_load_force_peak,
        "eta_recaptured": eta_recaptured,
        "energy_residual_rel": energy_residual_rel,
        "e_load_J": e_load_j,
        "vind_peak_V": vind_peak_v,
        "drift_rel_diff_max": drift_rel_diff_max,
        "drift_observable_in_b2": drift_observable_in_b2,
        "integrated_gate_pass": bool(integrated_gate_pass),
    }

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_summary, metrics)


if __name__ == "__main__":
    main()
