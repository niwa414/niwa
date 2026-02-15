#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize B2 stage3 integration summary.")
    parser.add_argument("--summary", required=True, help="stage3_summary.json path.")
    parser.add_argument("--metrics", required=True, help="Output metrics.json path.")
    parser.add_argument("--details", required=True, help="Output details json path.")
    args = parser.parse_args()

    data = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    metrics = {
        "stage": "3d3",
        "stage1_pass": bool(data.get("stage1_pass", False)),
        "stage2_transition_pass": bool(data.get("stage2_transition_pass", False)),
        "stage2_strict_pass": bool(data.get("stage2_strict_pass", False)),
        "tilt_matrix_all_pass": bool(data.get("tilt_matrix_all_pass", False)),
        "tilt_gamma_all_negative": bool(data.get("tilt_gamma_all_negative", False)),
        "stage3_ready": bool(data.get("stage3_ready", False)),
    }

    Path(args.metrics).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.details).write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
