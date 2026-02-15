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
    lines.append("# B3 Transport Closure Gate Summary")
    lines.append("")
    lines.append(f"- b3_full_gate_pass: `{metrics.get('b3_full_gate_pass')}`")
    lines.append(f"- coupling_effect_ok: `{metrics.get('coupling_effect_ok')}`")
    lines.append(f"- electron_energy_balance_ok: `{metrics.get('electron_energy_balance_ok')}`")
    lines.append(f"- te_scaling_ok: `{metrics.get('te_scaling_ok')}`")
    lines.append(f"- electron_records_ok: `{metrics.get('electron_records_ok')}`")
    lines.append(f"- observable_rel_diff: `{metrics.get('observable_rel_diff')}`")
    lines.append("")
    lines.append("This gate combines stage-1 coupling sweep and stage-2 closure checks.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote B3 electron-energy evidence to a full closure gate.")
    parser.add_argument("--sweep-case", default="m5-b3-1-electron-energy-coupled-sweep")
    parser.add_argument("--closure-case", default="m23-b3-electron-energy-stage2-closure")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sweep = load_passfail(repo_root, args.sweep_case)
    closure = load_passfail(repo_root, args.closure_case)

    sm = sweep["metrics"]
    cm = closure["metrics"]

    observable_rel_diff = as_float(sm.get("observable_rel_diff"))
    treatment_writeback_ratio = as_float(sm.get("treatment_writeback_ratio"))

    balance_rel_err = as_float(cm.get("electron_energy_balance_rel_err"))
    te_scaling_corr = as_float(cm.get("te_scaling_corr_te_vs_r_rms"))
    te_scaling_slope = as_float(cm.get("te_scaling_slope_te_vs_r_rms"))
    electron_records = as_float(cm.get("electron_energy_records"))

    coupling_effect_ok = (
        sweep["status"] == "PASS"
        and sm.get("sweep_pass") is True
        and sm.get("coupling_effect_observed") is True
        and sm.get("control_ok") is True
        and sm.get("treatment_ok") is True
        and observable_rel_diff is not None
        and observable_rel_diff >= 0.05
        and treatment_writeback_ratio is not None
        and treatment_writeback_ratio >= 0.9
    )

    electron_energy_balance_ok = (
        closure["status"] == "PASS"
        and cm.get("electron_energy_balance_ok") is True
        and balance_rel_err is not None
        and balance_rel_err <= 1.0e-6
    )

    te_scaling_ok = (
        cm.get("te_scaling_observed") is True
        and te_scaling_corr is not None
        and abs(te_scaling_corr) >= 0.9
        and te_scaling_slope is not None
        and abs(te_scaling_slope) > 1.0e-9
    )

    electron_records_ok = electron_records is not None and electron_records >= 50.0

    b3_full_gate_pass = (
        sweep["status"] == "PASS"
        and closure["status"] == "PASS"
        and coupling_effect_ok
        and electron_energy_balance_ok
        and te_scaling_ok
        and electron_records_ok
    )

    metrics = {
        "sweep_case_id": args.sweep_case,
        "closure_case_id": args.closure_case,
        "sweep_case_pass": sweep["status"] == "PASS",
        "closure_case_pass": closure["status"] == "PASS",
        "coupling_effect_ok": bool(coupling_effect_ok),
        "electron_energy_balance_ok": bool(electron_energy_balance_ok),
        "te_scaling_ok": bool(te_scaling_ok),
        "electron_records_ok": bool(electron_records_ok),
        "observable_rel_diff": observable_rel_diff,
        "treatment_writeback_ratio": treatment_writeback_ratio,
        "electron_energy_balance_rel_err": balance_rel_err,
        "te_scaling_corr_te_vs_r_rms": te_scaling_corr,
        "te_scaling_slope_te_vs_r_rms": te_scaling_slope,
        "electron_energy_records": electron_records,
        "b3_full_gate_pass": bool(b3_full_gate_pass),
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
