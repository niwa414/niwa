#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def infer_failure_reason(passfail: dict) -> str:
    if not isinstance(passfail, dict):
        return "FAIL_UNKNOWN"
    reason = passfail.get("failure_reason")
    if reason:
        return reason
    for err in passfail.get("errors", []) or []:
        if isinstance(err, str) and err.startswith("metrics_missing"):
            return "METRICS_MISSING"
    threshold_results = passfail.get("threshold_results") or []
    if threshold_results:
        for entry in threshold_results:
            if entry.get("reason") == "missing_metric":
                return "GATE_MISSING_METRIC"
        return "GATE_FAIL"
    return "FAIL_UNKNOWN"


def run_case_with_python(case_path: Path, stage: str, update_evidence: bool) -> None:
    python = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = "python"
    cmd = [python, str(Path(__file__).resolve().parents[1] / "tools" / "run_case.py"), "--case", str(case_path), "--stage", stage]
    if update_evidence:
        cmd.append("--update-evidence")
    subprocess.check_call(cmd)


def run_stage(case_path: Path, stage: str, update_evidence: bool) -> None:
    if stage in ("run", "analyze", "all"):
        run_case_with_python(case_path, stage, update_evidence)
        return
    # allow comma-separated
    for part in [s.strip() for s in stage.split(",") if s.strip()]:
        run_case_with_python(case_path, part, update_evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 1D sweep and summarize results.")
    parser.add_argument("--cases-dir", required=True, help="Directory containing sweep cases.")
    parser.add_argument("--stage", default="all", help="Stage to run (run/analyze/all or comma list).")
    parser.add_argument("--update-evidence", action="store_true", help="Update evidence after each case.")
    parser.add_argument("--skip-run", action="store_true", help="Skip running cases; only summarize.")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        raise SystemExit(f"cases_dir_not_found: {cases_dir}")

    case_paths = sorted(cases_dir.rglob("case.json"))
    rows = []
    root = Path(__file__).resolve().parents[1]

    for case_path in case_paths:
        case = load_json(case_path) or {}
        case_id = case.get("id") or case_path.parent.name
        sweep_value = None
        sweep_shift = None
        meta = case.get("metadata", {})
        if isinstance(meta, dict):
            sweep = meta.get("sweep", {})
            if isinstance(sweep, dict):
                sweep_value = sweep.get("value")
                if sweep.get("knob") == "shift":
                    sweep_shift = sweep_value
        if not args.skip_run:
            run_stage(case_path, args.stage, args.update_evidence)

        out_dir = root / "outputs" / case_id / "analysis"
        passfail = load_json(out_dir / "PASSFAIL.json") or {}
        metrics = load_json(out_dir / "metrics.json") or {}
        status = passfail.get("status") or passfail.get("result") or "UNKNOWN"
        failure_reason = passfail.get("failure_reason") or ""
        if status == "FAIL" and not failure_reason:
            failure_reason = infer_failure_reason(passfail)

        row = {
            "case": case_id,
            "drift": sweep_value if sweep_value is not None else "",
            "shift": sweep_shift if sweep_shift is not None else "",
            "status": status,
            "failure_reason": failure_reason,
            "merge_time_source": metrics.get("merge_time_source", ""),
            "merge_time_ok": metrics.get("merge_time_ok", ""),
            "merge_time_fallback_reason": metrics.get("merge_time_fallback_reason", ""),
            "fit_window_strategy": metrics.get("fit_window_strategy", ""),
            "fit_points": metrics.get("fit_points", ""),
            "drift_rel_diff_max": metrics.get("drift_rel_diff_max", ""),
            "tilt_post_merge_amp_max": metrics.get("tilt_post_merge_amp_max", ""),
            "tilt_amp_window_t0": metrics.get("tilt_amp_window_t0", ""),
            "tilt_amp_window_t1": metrics.get("tilt_amp_window_t1", ""),
        }
        rows.append(row)

    csv_path = cases_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case",
            "drift",
            "shift",
            "status",
            "failure_reason",
            "merge_time_source",
            "merge_time_ok",
            "merge_time_fallback_reason",
            "fit_window_strategy",
            "fit_points",
            "drift_rel_diff_max",
            "tilt_post_merge_amp_max",
            "tilt_amp_window_t0",
            "tilt_amp_window_t1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    best = None
    for row in rows:
        if row.get("status") != "PASS":
            continue
        try:
            drift_val = float(row.get("drift"))
            score = float(row.get("drift_rel_diff_max"))
        except Exception:
            continue
        if best is None or score < best["score"]:
            best = {"case": row.get("case"), "drift": drift_val, "score": score}

    print(f"wrote {csv_path}")
    if best:
        print(
            f"best_pass_case={best['case']} best_drift={best['drift']} "
            f"best_drift_rel_diff_max={best['score']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
