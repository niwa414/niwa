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
    lines.append("# A3 Full Gate Summary")
    lines.append("")
    lines.append(f"- a3_full_gate_pass: `{metrics.get('a3_full_gate_pass')}`")
    lines.append(f"- strong_coupling_ok: `{metrics.get('strong_coupling_ok')}`")
    lines.append(f"- response_driver_ok: `{metrics.get('response_driver_ok')}`")
    lines.append(f"- integrated_energy_chain_ok: `{metrics.get('integrated_energy_chain_ok')}`")
    lines.append(f"- driver_response_ratio: `{metrics.get('driver_response_ratio')}`")
    lines.append(f"- known_gap_count: `{metrics.get('known_gap_count')}`")
    lines.append("")
    lines.append("This gate binds A3 closed-loop behavior to integration-level load and energy chain checks.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote A3 closed-loop evidence to a full integration gate.")
    parser.add_argument("--strong-case", default="m5-a3-closed-loop-strong")
    parser.add_argument("--response-case", default="m16-a3-circuit-radius-response")
    parser.add_argument("--integrated-case", default="m26-p2-circuit-load-integration")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    strong = load_passfail(repo_root, args.strong_case)
    response = load_passfail(repo_root, args.response_case)
    integrated = load_passfail(repo_root, args.integrated_case)

    sm = strong["metrics"]
    rm = response["metrics"]
    im = integrated["metrics"]

    circuit_update_fraction = as_float(sm.get("circuit_update_fraction"))
    driver_writeback_fraction = as_float(sm.get("driver_writeback_fraction"))
    driver_response_ratio = as_float(rm.get("driver_response_ratio"))
    load_force_peak = as_float(im.get("mainline_load_force_peak_N"))

    known_gaps = rm.get("known_gaps")
    if not isinstance(known_gaps, list):
        known_gaps = []

    strong_coupling_ok = (
        strong["status"] == "PASS"
        and sm.get("strong_coupling_enabled") is True
        and circuit_update_fraction is not None
        and circuit_update_fraction >= 0.9
        and driver_writeback_fraction is not None
        and driver_writeback_fraction >= 0.9
    )

    response_driver_ok = (
        response["status"] == "PASS"
        and rm.get("compare_pass") is True
        and rm.get("driver_response_ok") is True
        and driver_response_ratio is not None
        and driver_response_ratio >= 1.1
    )

    integrated_energy_chain_ok = (
        integrated["status"] == "PASS"
        and im.get("integrated_gate_pass") is True
        and im.get("mainline_load_embedded") is True
        and im.get("mainline_load_energy_chain_present") is True
        and load_force_peak is not None
        and load_force_peak > 1.0
    )

    a3_full_gate_pass = (
        strong["status"] == "PASS"
        and response["status"] == "PASS"
        and integrated["status"] == "PASS"
        and strong_coupling_ok
        and response_driver_ok
        and integrated_energy_chain_ok
    )

    metrics = {
        "strong_case_id": args.strong_case,
        "response_case_id": args.response_case,
        "integrated_case_id": args.integrated_case,
        "strong_case_pass": strong["status"] == "PASS",
        "response_case_pass": response["status"] == "PASS",
        "integrated_case_pass": integrated["status"] == "PASS",
        "strong_coupling_ok": bool(strong_coupling_ok),
        "response_driver_ok": bool(response_driver_ok),
        "integrated_energy_chain_ok": bool(integrated_energy_chain_ok),
        "circuit_update_fraction": circuit_update_fraction,
        "driver_writeback_fraction": driver_writeback_fraction,
        "driver_response_ratio": driver_response_ratio,
        "known_gap_count": int(len(known_gaps)),
        "known_gap_radius_unresponsive": "a3_radius_unresponsive_in_window" in known_gaps,
        "integrated_load_force_peak_N": load_force_peak,
        "a3_full_gate_pass": bool(a3_full_gate_pass),
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
