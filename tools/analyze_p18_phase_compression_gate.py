#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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


def parse_off_step(case_id: str) -> int | None:
    match = re.search(r"off(\d+)", case_id)
    if not match:
        return None
    try:
        return int(match.group(1))
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


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "status",
                "off_step",
                "step_B_energy_peak",
                "compression_detected_phase",
                "phase_end_step",
                "step_rho_max_peak_phase",
                "rho_delta_rel_phase",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("case_id"),
                    row.get("status"),
                    row.get("off_step"),
                    row.get("step_B_energy_peak"),
                    row.get("compression_detected_phase"),
                    row.get("phase_end_step"),
                    row.get("step_rho_max_peak_phase"),
                    row.get("rho_delta_rel_phase"),
                ]
            )


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# P18 Phase Compression Gate Summary")
    lines.append("")
    lines.append(f"- p18_full_gate_pass: `{metrics.get('p18_full_gate_pass')}`")
    lines.append(f"- cases_all_pass: `{metrics.get('cases_all_pass')}`")
    lines.append(f"- off_steps_strictly_increasing: `{metrics.get('off_steps_strictly_increasing')}`")
    lines.append(f"- b_energy_peak_matches_off_step_all: `{metrics.get('b_energy_peak_matches_off_step_all')}`")
    lines.append(f"- rho_delta_rel_phase_monotonic: `{metrics.get('rho_delta_rel_phase_monotonic')}`")
    lines.append(f"- rho_delta_rel_phase_gain: `{metrics.get('rho_delta_rel_phase_gain')}`")
    lines.append("")
    lines.append("This gate validates phase-window compression response across off_step sweep points.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote P18 off-step phase compression sweep to full gate.")
    parser.add_argument(
        "--cases",
        default="m17-b2-p18-off192,m17-b2-p18-off256,m17-b2-p18-off320",
        help="Comma-separated case IDs.",
    )
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--table", required=True)
    args = parser.parse_args()

    case_ids = [item.strip() for item in str(args.cases).split(",") if item.strip()]
    repo_root = Path(__file__).resolve().parents[1]

    rows = []
    for case_id in case_ids:
        obs = load_passfail(repo_root, case_id)
        m = obs["metrics"]
        phase = m.get("formation_kpi_phase")
        if not isinstance(phase, dict):
            phase = {}
        row = {
            "case_id": case_id,
            "status": obs["status"],
            "off_step": parse_off_step(case_id),
            "step_B_energy_peak": as_float(m.get("step_B_energy_peak")),
            "compression_detected_phase": phase.get("compression_detected_phase"),
            "phase_end_step": as_float(phase.get("phase_end_step")),
            "step_rho_max_peak_phase": as_float(phase.get("step_rho_max_peak")),
            "rho_delta_rel_phase": as_float(phase.get("rho_delta_rel_phase")),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda row: (row.get("off_step") if row.get("off_step") is not None else 10**9))

    off_steps = [int(row["off_step"]) for row in rows if row.get("off_step") is not None]
    b_peaks = [row["step_B_energy_peak"] for row in rows]
    step_rho_peaks = [row["step_rho_max_peak_phase"] for row in rows]
    rho_phase = [row["rho_delta_rel_phase"] for row in rows]

    cases_all_pass = bool(rows) and all(row.get("status") == "PASS" for row in rows)
    off_steps_strictly_increasing = len(off_steps) == len(rows) and all(
        off_steps[idx + 1] > off_steps[idx] for idx in range(len(off_steps) - 1)
    )
    b_energy_peak_matches_off_step_all = len(b_peaks) == len(rows) and all(
        b_peak is not None and row.get("off_step") is not None and int(round(b_peak)) == int(row["off_step"])
        for b_peak, row in zip(b_peaks, rows)
    )
    phase_detected_all = bool(rows) and all(row.get("compression_detected_phase") is True for row in rows)
    rho_delta_rel_phase_monotonic = len(rho_phase) == len(rows) and all(
        value is not None for value in rho_phase
    ) and all(rho_phase[idx + 1] >= rho_phase[idx] for idx in range(len(rho_phase) - 1))
    step_rho_peak_monotonic = len(step_rho_peaks) == len(rows) and all(
        value is not None for value in step_rho_peaks
    ) and all(step_rho_peaks[idx + 1] >= step_rho_peaks[idx] for idx in range(len(step_rho_peaks) - 1))

    rho_delta_rel_phase_gain = None
    if len(rho_phase) >= 2 and rho_phase[0] is not None and rho_phase[-1] is not None and rho_phase[0] > 0.0:
        rho_delta_rel_phase_gain = float(rho_phase[-1] / rho_phase[0])

    phase_kpi_table_present = bool(rows)

    p18_full_gate_pass = (
        cases_all_pass
        and off_steps_strictly_increasing
        and b_energy_peak_matches_off_step_all
        and phase_detected_all
        and rho_delta_rel_phase_monotonic
        and step_rho_peak_monotonic
        and rho_delta_rel_phase_gain is not None
        and rho_delta_rel_phase_gain >= 2.0
        and phase_kpi_table_present
    )

    metrics = {
        "case_ids": case_ids,
        "case_count": int(len(rows)),
        "cases_all_pass": bool(cases_all_pass),
        "off_steps_strictly_increasing": bool(off_steps_strictly_increasing),
        "b_energy_peak_matches_off_step_all": bool(b_energy_peak_matches_off_step_all),
        "phase_detected_all": bool(phase_detected_all),
        "rho_delta_rel_phase_monotonic": bool(rho_delta_rel_phase_monotonic),
        "step_rho_peak_monotonic": bool(step_rho_peak_monotonic),
        "rho_delta_rel_phase_gain": rho_delta_rel_phase_gain,
        "phase_kpi_table_present": bool(phase_kpi_table_present),
        "p18_full_gate_pass": bool(p18_full_gate_pass),
    }

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    out_table = Path(args.table)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()
    if not out_table.is_absolute():
        out_table = (repo_root / out_table).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_table(out_table, rows)
    write_summary(out_summary, metrics)


if __name__ == "__main__":
    main()
