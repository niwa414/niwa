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
        "path": str(path),
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {},
    }


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# A4 Conductor/Load Gate Summary")
    lines.append("")
    lines.append(f"- a4_full_gate_pass: `{metrics.get('a4_full_gate_pass')}`")
    lines.append(f"- base_eb_mask_ok: `{metrics.get('base_eb_mask_ok')}`")
    lines.append(f"- moving_wall_ok: `{metrics.get('moving_wall_ok')}`")
    lines.append(f"- load_interface_ok: `{metrics.get('load_interface_ok')}`")
    lines.append(f"- load_handoff_artifacts_present: `{metrics.get('load_handoff_artifacts_present')}`")
    lines.append(f"- load_force_proxy_peak_N: `{metrics.get('load_force_proxy_peak_N')}`")
    lines.append("")
    lines.append("This gate ties wall/EB behavior to the engineering load handoff artifacts.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote A4 wall/EB evidence to full conductor/load interface gate.")
    parser.add_argument("--base-case", default="m4-a4-eb-wall-gate")
    parser.add_argument("--moving-case", default="m22-a4-eb-movingwall-stage2-tight")
    parser.add_argument("--load-case", default="m26-d2-magnetic-load-interface")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base = load_passfail(repo_root, args.base_case)
    moving = load_passfail(repo_root, args.moving_case)
    load = load_passfail(repo_root, args.load_case)

    bm = base["metrics"]
    mm = moving["metrics"]
    lm = load["metrics"]

    base_leak = as_float(bm.get("leak_mass_frac_max"))
    base_blocked_faces = as_float(bm.get("blocked_flux_faces_count"))

    moving_piston_metric = as_float(mm.get("piston_metric"))
    moving_leak = as_float(mm.get("leak_mass_frac_max"))
    moving_leak_improve = as_float(mm.get("leak_improve_factor"))
    moving_residual_improve = as_float(mm.get("residual_improve_factor"))

    load_force_peak = as_float(lm.get("force_proxy_peak_N"))
    load_pressure_peak = as_float(lm.get("p_mag_peak_Pa"))
    load_eta_recaptured = as_float(lm.get("eta_recaptured"))

    load_artifacts = [
        repo_root / "outputs" / args.load_case / "analysis" / "magnetic_load_series.csv",
        repo_root / "outputs" / args.load_case / "analysis" / "magnetic_load_summary.md",
        repo_root / "outputs" / args.load_case / "analysis" / "metrics.json",
    ]
    load_handoff_artifacts_present = all(path.exists() for path in load_artifacts)

    base_eb_mask_ok = (
        base["status"] == "PASS"
        and bm.get("eb_mask_applied") is True
        and bm.get("wall_enabled") is True
        and base_blocked_faces is not None
        and base_blocked_faces > 0.0
        and base_leak is not None
        and base_leak <= 1.0e-2
    )

    moving_wall_ok = (
        moving["status"] == "PASS"
        and mm.get("eb_mask_applied") is True
        and moving_piston_metric is not None
        and moving_piston_metric > 0.0
        and moving_leak is not None
        and moving_leak <= 1.0e-2
        and moving_leak_improve is not None
        and moving_leak_improve >= 1.0
        and moving_residual_improve is not None
        and moving_residual_improve >= 1.0
    )

    load_interface_ok = (
        load["status"] == "PASS"
        and lm.get("interface_ready") is True
        and load_force_peak is not None
        and load_force_peak > 1.0
        and load_pressure_peak is not None
        and load_pressure_peak > 0.0
        and load_eta_recaptured is not None
        and load_eta_recaptured >= 0.3
    )

    a4_full_gate_pass = (
        base["status"] == "PASS"
        and moving["status"] == "PASS"
        and load["status"] == "PASS"
        and base_eb_mask_ok
        and moving_wall_ok
        and load_interface_ok
        and load_handoff_artifacts_present
    )

    metrics = {
        "base_case_id": args.base_case,
        "moving_case_id": args.moving_case,
        "load_case_id": args.load_case,
        "base_case_pass": base["status"] == "PASS",
        "moving_case_pass": moving["status"] == "PASS",
        "load_case_pass": load["status"] == "PASS",
        "base_eb_mask_ok": bool(base_eb_mask_ok),
        "moving_wall_ok": bool(moving_wall_ok),
        "load_interface_ok": bool(load_interface_ok),
        "load_handoff_artifacts_present": bool(load_handoff_artifacts_present),
        "base_leak_mass_frac_max": base_leak,
        "base_blocked_flux_faces_count": base_blocked_faces,
        "moving_piston_metric": moving_piston_metric,
        "moving_leak_mass_frac_max": moving_leak,
        "moving_leak_improve_factor": moving_leak_improve,
        "moving_residual_improve_factor": moving_residual_improve,
        "load_force_proxy_peak_N": load_force_peak,
        "load_p_mag_peak_Pa": load_pressure_peak,
        "load_eta_recaptured": load_eta_recaptured,
        "a4_full_gate_pass": bool(a4_full_gate_pass),
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
