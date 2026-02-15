#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize B2 3D driver stage0 suite.")
    parser.add_argument("--summary", required=True, help="stage0_summary.json path.")
    parser.add_argument("--metrics", required=True, help="Output metrics.json path.")
    parser.add_argument("--details", required=True, help="Output details json path.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    stages = data.get("stage_results", {})

    def stage_pass(name: str) -> bool:
        return bool(stages.get(name, {}).get("pass", False))

    metrics = {
        "init_h5_exists": bool(data.get("init_h5_exists", False)),
        "stage_3d0a_pass": stage_pass("3d0a"),
        "stage_3d0b_pass": stage_pass("3d0b"),
        "stage_3d0c_pass": stage_pass("3d0c"),
        "stage0_suite_pass": bool(data.get("stage0_suite_pass", False)),
    }

    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.details).write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
