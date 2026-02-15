#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


P0_METRICS = {
    "merge_time_exists",
    "merge_time_proxy",
    "merge_time_ok",
    "fit_window_found",
    "fit_points",
    "no_nan_in_metrics",
}

P0_THRESHOLDS = [
    {"metric": "merge_time_ok", "op": "==", "value": True},
    {"metric": "fit_window_found", "op": "==", "value": True},
    {"metric": "fit_points", "op": ">=", "value": 12},
    {"metric": "no_nan_in_metrics", "op": "==", "value": True},
]


def analyze_uses_h5(case: dict) -> bool:
    for entry in case.get("analyze", []):
        cmd = entry.get("cmd", [])
        cmd_str = " ".join(str(part) for part in cmd)
        if "analyze_h5_merge_tilt_gamma.py" in cmd_str:
            return True
    return False


def metrics_file_is_h5(case: dict) -> bool:
    metrics_file = case.get("metrics_file") or ""
    return "metrics.json" in metrics_file


def patch_thresholds(case: dict) -> tuple[bool, list, list]:
    thresholds = list(case.get("thresholds", []))
    trimmed = [t for t in thresholds if t.get("metric") not in P0_METRICS]
    new_thresholds = trimmed + P0_THRESHOLDS
    changed = new_thresholds != thresholds
    if changed:
        case["thresholds"] = new_thresholds
    return changed, thresholds, new_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch case.json thresholds for P0 merge/fit gating.")
    parser.add_argument("--cases-dir", default="cases", help="Root cases directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        print(f"cases_dir_not_found: {cases_dir}")
        return 2

    patched = []
    skipped = 0
    for case_path in sorted(cases_dir.rglob("case.json")):
        try:
            raw = case_path.read_text()
            case = json.loads(raw)
        except Exception as exc:
            print(f"skip_parse_error: {case_path} ({exc})")
            continue

        if not analyze_uses_h5(case):
            skipped += 1
            continue
        if not metrics_file_is_h5(case):
            skipped += 1
            continue

        changed, old_thresholds, new_thresholds = patch_thresholds(case)
        if not changed:
            continue
        patched.append((case_path, old_thresholds, new_thresholds))
        if not args.dry_run:
            case_path.write_text(json.dumps(case, indent=2, sort_keys=False) + "\n")

    print(f"patched_files={len(patched)} skipped={skipped}")
    for path, old, new in patched:
        old_metrics = [t.get("metric") for t in old]
        new_metrics = [t.get("metric") for t in new]
        print(f"- {path}")
        print(f"  old_metrics={old_metrics}")
        print(f"  new_metrics={new_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
