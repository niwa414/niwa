#!/usr/bin/env python3
"""Release gate checker for simulation orchestrator runs.

Checks:
- Job-level completion status from orchestrator state.
- Per-case KPI thresholds from metrics.json.
- Internal parity policy from metrics in m28-d1 gate output.

Exit code:
- 0: PASS
- 1: FAIL
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def as_bool(value: Any) -> bool:
    return bool(value)


def resolve_run_dir(repo_root: Path, run_id: str | None, run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir).resolve()
    if run_id:
        return (repo_root / "outputs" / "orchestrator" / run_id).resolve()
    raise ValueError("Either --run-id or --run-dir is required")


def default_thresholds() -> dict[str, Any]:
    return {
        "require_all_jobs_pass": True,
        "aggregate": {"min_pass_cases": 1},
        "per_case": {
            "require_ran_to_completion": True,
            "require_no_nan_in_metrics": True,
            "require_merge_time_exists": True,
            "require_energy_accounting_ok": True,
            "compression_ratio_min": 1.001,
            "tilt_amp_max_max": 0.05,
            "energy_residual_rel_max": 1.0e-6,
        },
        "internal_parity": {
            "metrics_path": "outputs/m28-d1-helion-internal-parity-gate/analysis/metrics.json",
            "require_internal_parity_claimable": False,
            "require_zero_internal_gaps": False,
            "required_true_fields": [],
            "required_false_fields": [],
        },
    }


def add_check(report: dict[str, Any], name: str, passed: bool, detail: str, context: dict[str, Any] | None = None) -> None:
    item = {
        "name": name,
        "pass": bool(passed),
        "detail": detail,
    }
    if context:
        item["context"] = context
    report["checks"].append(item)
    if not passed:
        report["failures"].append(item)


def check_job_states(state: dict[str, Any], thresholds: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = state.get("jobs", []) or []
    pass_jobs = [j for j in jobs if str(j.get("final_status")) == "PASS"]

    require_all_jobs_pass = as_bool(thresholds.get("require_all_jobs_pass", True))
    if require_all_jobs_pass:
        failed = [j.get("key") for j in jobs if str(j.get("final_status")) != "PASS"]
        add_check(
            report,
            "all_jobs_pass",
            len(failed) == 0,
            "all jobs must end in PASS",
            {"failed_jobs": failed},
        )

    min_pass_cases = int((thresholds.get("aggregate") or {}).get("min_pass_cases", 1))
    add_check(
        report,
        "min_pass_cases",
        len(pass_jobs) >= min_pass_cases,
        f"PASS jobs >= {min_pass_cases}",
        {"pass_jobs": len(pass_jobs)},
    )

    return pass_jobs


def check_case_metrics(repo_root: Path, pass_jobs: list[dict[str, Any]], thresholds: dict[str, Any], report: dict[str, Any]) -> None:
    per_case = thresholds.get("per_case") or {}

    comp_min = to_float(per_case.get("compression_ratio_min"))
    tilt_max = to_float(per_case.get("tilt_amp_max_max"))
    energy_res_max = to_float(per_case.get("energy_residual_rel_max"))

    aggregate_compression: list[float] = []
    aggregate_tilt: list[float] = []
    aggregate_energy: list[float] = []

    for job in pass_jobs:
        case_id = str(job.get("case_id"))
        key = str(job.get("key"))
        metrics_path = repo_root / "outputs" / case_id / "analysis" / "metrics.json"
        metrics = load_json(metrics_path)

        add_check(
            report,
            f"{key}.metrics_exists",
            bool(metrics),
            "metrics.json must exist and be parseable",
            {"metrics_path": str(metrics_path)},
        )
        if not metrics:
            continue

        req_flags = [
            ("require_ran_to_completion", "ran_to_completion"),
            ("require_no_nan_in_metrics", "no_nan_in_metrics"),
            ("require_merge_time_exists", "merge_time_exists"),
            ("require_energy_accounting_ok", "energy_accounting_ok"),
        ]
        for cfg_key, metric_key in req_flags:
            if not as_bool(per_case.get(cfg_key, True)):
                continue
            actual = bool(metrics.get(metric_key))
            add_check(
                report,
                f"{key}.{metric_key}",
                actual,
                f"{metric_key} must be true",
                {"actual": metrics.get(metric_key), "case_id": case_id},
            )

        comp = to_float(metrics.get("compression_ratio"))
        tilt = to_float(metrics.get("tilt_amp_max"))
        energy = to_float(metrics.get("energy_residual_rel"))

        if comp is not None:
            aggregate_compression.append(comp)
        if tilt is not None:
            aggregate_tilt.append(tilt)
        if energy is not None:
            aggregate_energy.append(energy)

        if comp_min is not None:
            add_check(
                report,
                f"{key}.compression_ratio",
                comp is not None and comp >= comp_min,
                f"compression_ratio >= {comp_min}",
                {"actual": comp, "case_id": case_id},
            )
        if tilt_max is not None:
            add_check(
                report,
                f"{key}.tilt_amp_max",
                tilt is not None and tilt <= tilt_max,
                f"tilt_amp_max <= {tilt_max}",
                {"actual": tilt, "case_id": case_id},
            )
        if energy_res_max is not None:
            add_check(
                report,
                f"{key}.energy_residual_rel",
                energy is not None and energy <= energy_res_max,
                f"energy_residual_rel <= {energy_res_max}",
                {"actual": energy, "case_id": case_id},
            )

    if aggregate_compression and comp_min is not None:
        add_check(
            report,
            "aggregate.min_compression_ratio",
            min(aggregate_compression) >= comp_min,
            f"min compression_ratio >= {comp_min}",
            {"actual": min(aggregate_compression)},
        )
    if aggregate_tilt and tilt_max is not None:
        add_check(
            report,
            "aggregate.max_tilt_amp_max",
            max(aggregate_tilt) <= tilt_max,
            f"max tilt_amp_max <= {tilt_max}",
            {"actual": max(aggregate_tilt)},
        )
    if aggregate_energy and energy_res_max is not None:
        add_check(
            report,
            "aggregate.max_energy_residual_rel",
            max(aggregate_energy) <= energy_res_max,
            f"max energy_residual_rel <= {energy_res_max}",
            {"actual": max(aggregate_energy)},
        )


def check_internal_parity(repo_root: Path, thresholds: dict[str, Any], report: dict[str, Any]) -> None:
    cfg = thresholds.get("internal_parity") or {}
    metrics_rel = str(cfg.get("metrics_path") or "outputs/m28-d1-helion-internal-parity-gate/analysis/metrics.json")
    metrics_path = repo_root / metrics_rel
    metrics = load_json(metrics_path)

    add_check(
        report,
        "internal_parity.metrics_exists",
        bool(metrics),
        "internal parity metrics must exist",
        {"metrics_path": str(metrics_path)},
    )
    if not metrics:
        return

    require_claimable = as_bool(cfg.get("require_internal_parity_claimable", False))
    if require_claimable:
        add_check(
            report,
            "internal_parity.claimable",
            bool(metrics.get("internal_parity_claimable")),
            "internal_parity_claimable must be true",
            {"actual": metrics.get("internal_parity_claimable")},
        )

    require_zero_gaps = as_bool(cfg.get("require_zero_internal_gaps", False))
    if require_zero_gaps:
        gap_count = to_float(metrics.get("internal_only_gap_count"))
        add_check(
            report,
            "internal_parity.zero_internal_gaps",
            gap_count is not None and gap_count <= 0,
            "internal_only_gap_count must be 0",
            {"actual": gap_count},
        )

    for field in cfg.get("required_true_fields", []) or []:
        add_check(
            report,
            f"internal_parity.{field}",
            bool(metrics.get(field)),
            f"{field} must be true",
            {"actual": metrics.get(field)},
        )

    for field in cfg.get("required_false_fields", []) or []:
        add_check(
            report,
            f"internal_parity.{field}",
            not bool(metrics.get(field)),
            f"{field} must be false",
            {"actual": metrics.get(field)},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check release gates for a simulation orchestrator run.")
    parser.add_argument("--run-id", default=None, help="Run id under outputs/orchestrator/<run_id>")
    parser.add_argument("--run-dir", default=None, help="Absolute run directory path")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Thresholds JSON path (default: /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json)",
    )
    parser.add_argument("--output", default=None, help="Output JSON path (default: <run_dir>/release_gate.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = resolve_run_dir(repo_root, args.run_id, args.run_dir)

    state_path = run_dir / "state.json"
    state = load_json(state_path)
    if not state:
        raise SystemExit(f"state.json not found or invalid: {state_path}")

    thresholds_path = Path(args.thresholds).resolve() if args.thresholds else (repo_root / "ops" / "release-gate-thresholds.json")
    thresholds = load_json(thresholds_path)
    if not thresholds:
        thresholds = default_thresholds()

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "run_id": state.get("run_id"),
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "thresholds_path": str(thresholds_path),
        "checks": [],
        "failures": [],
    }

    pass_jobs = check_job_states(state, thresholds, report)
    check_case_metrics(repo_root, pass_jobs, thresholds, report)
    check_internal_parity(repo_root, thresholds, report)

    report["summary"] = {
        "total_checks": len(report["checks"]),
        "failed_checks": len(report["failures"]),
        "status": "PASS" if not report["failures"] else "FAIL",
    }

    output_path = Path(args.output).resolve() if args.output else (run_dir / "release_gate.json")
    write_json(output_path, report)

    print(f"[release-gate] status={report['summary']['status']}")
    print(f"[release-gate] checks={report['summary']['total_checks']} failed={report['summary']['failed_checks']}")
    print(f"[release-gate] output={output_path}")

    raise SystemExit(0 if report["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
