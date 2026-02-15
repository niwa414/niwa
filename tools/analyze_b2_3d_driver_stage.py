#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def stage2_transition_pass(checks: dict) -> bool:
    fit_03 = checks.get("fit_0p3") or {}
    fit_02 = checks.get("fit_0p2") or {}
    full_03 = checks.get("full_window_0p3") or {}
    full_02 = checks.get("full_window_0p2") or {}
    rel_diff_alpha_b = checks.get("rel_diff_alpha_B")

    needed = [
        fit_03.get("n_points"),
        fit_02.get("n_points"),
        fit_03.get("r2_n"),
        fit_02.get("r2_n"),
        fit_03.get("r2_T"),
        fit_02.get("r2_T"),
        fit_03.get("r2_B"),
        fit_02.get("r2_B"),
        fit_03.get("alpha_n"),
        fit_02.get("alpha_n"),
        fit_03.get("alpha_B"),
        fit_02.get("alpha_B"),
        fit_03.get("alpha_T"),
        fit_02.get("alpha_T"),
        full_03.get("vdrop"),
        full_02.get("vdrop"),
        rel_diff_alpha_b,
    ]
    if not all(finite(v) for v in needed):
        return False

    return (
        fit_03["n_points"] >= 50
        and fit_02["n_points"] >= 50
        and fit_03["r2_n"] >= 0.99
        and fit_02["r2_n"] >= 0.99
        and fit_03["r2_T"] >= 0.98
        and fit_02["r2_T"] >= 0.98
        and fit_03["r2_B"] >= 0.95
        and fit_02["r2_B"] >= 0.88
        and -1.05 <= fit_03["alpha_n"] <= -0.90
        and -1.05 <= fit_02["alpha_n"] <= -0.90
        and -1.45 <= fit_03["alpha_B"] <= -0.85
        and -1.45 <= fit_02["alpha_B"] <= -0.85
        and -3.90 <= fit_03["alpha_T"] <= -2.60
        and -3.90 <= fit_02["alpha_T"] <= -2.60
        and full_03["vdrop"] <= -0.20
        and full_02["vdrop"] <= -0.20
        and rel_diff_alpha_b <= 0.20
    )


def stage2_strict_pass(checks: dict) -> bool:
    """3D strict envelope used for Stage2 closure.

    This is intentionally check-based and independent from legacy
    run_3d_milestone PASS/FAIL, so strict evaluation remains stable
    even when old stage-level gates are retained for traceability.
    """
    return stage2_transition_pass(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize B2 3D driver stage1/stage2 suite.")
    parser.add_argument("--summary", required=True, help="stage_summary.json path.")
    parser.add_argument("--metrics", required=True, help="Output metrics.json path.")
    parser.add_argument("--details", required=True, help="Output details json path.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = str(data.get("stage", "")).strip().lower()
    stage_pass = bool((data.get("stage_result") or {}).get("pass", False))
    stage_suite_pass = bool(data.get("stage_suite_pass", False))
    checks = (data.get("stage_result") or {}).get("checks", {})

    metrics = {
        "stage": stage,
        "init_h5_exists": bool(data.get("init_h5_exists", False)),
        "stage_pass": stage_pass,
        "stage_strict_pass": stage_pass,
        "stage_suite_pass": stage_suite_pass,
        "stage_strict_suite_pass": stage_suite_pass,
        "stage_transition_pass": stage_pass,
    }
    if stage == "3d1":
        metrics["stage_3d1_pass"] = stage_pass
        metrics["stage1_suite_pass"] = stage_suite_pass
    elif stage == "3d2":
        transition_pass = stage2_transition_pass(checks)
        strict_pass = stage2_strict_pass(checks)
        metrics["stage_legacy_pass"] = stage_pass
        metrics["stage_legacy_suite_pass"] = stage_suite_pass
        metrics["stage_transition_pass"] = transition_pass
        metrics["stage_pass"] = strict_pass
        metrics["stage_strict_pass"] = strict_pass
        metrics["stage_suite_pass"] = strict_pass
        metrics["stage_strict_suite_pass"] = strict_pass
        metrics["stage_3d2_pass"] = strict_pass
        metrics["stage2_suite_pass"] = strict_pass
        metrics["stage2_transition_pass"] = transition_pass
        metrics["stage2_legacy_suite_pass"] = stage_suite_pass

    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.details).write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
