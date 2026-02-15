#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract metrics from regression pack summary.")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--details", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metrics = {
        "num_cases": int(summary.get("num_cases", 0)),
        "pass_count": int(summary.get("pass_count", 0)),
        "fail_count": int(summary.get("fail_count", 0)),
        "all_run_rc_zero": bool(summary.get("all_run_rc_zero", False)),
        "restart_consistency_pass": bool(summary.get("restart_consistency_pass", False)),
        "regression_pack_pass": bool(summary.get("regression_pack_pass", False)),
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.details).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
