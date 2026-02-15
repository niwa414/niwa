#!/usr/bin/env python3
"""Run live 3-case Helion-style single-knob trade study and emit report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live Helion tilt trade study.")
    parser.add_argument(
        "--cases-root",
        default="cases/helion-live-tilt-tradestudy",
        help="Directory containing baseline/knob_minus/knob_plus case folders",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/helion-live-tilt-tradestudy/report.md",
        help="Report markdown output path",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only regenerate report from existing outputs",
    )
    parser.add_argument(
        "--compression-min",
        type=float,
        default=1.001,
        help="Gate threshold passed to report tool",
    )
    parser.add_argument(
        "--tilt-max",
        type=float,
        default=0.05,
        help="Gate threshold passed to report tool",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cases_root = (repo_root / args.cases_root).resolve()
    output_report = (repo_root / args.output_report).resolve()

    case_paths = [
        cases_root / "baseline" / "case.json",
        cases_root / "knob_minus" / "case.json",
        cases_root / "knob_plus" / "case.json",
    ]

    if not args.skip_run:
        for case_path in case_paths:
            run_cmd(
                [
                    sys.executable,
                    "tools/run_case.py",
                    "--case",
                    str(case_path),
                ],
                cwd=repo_root,
            )

    run_cmd(
        [
            sys.executable,
            "tools/report_helion_demo.py",
            str(repo_root / "outputs/helion-live-tilt-tradestudy-baseline"),
            str(repo_root / "outputs/helion-live-tilt-tradestudy-knob-minus"),
            str(repo_root / "outputs/helion-live-tilt-tradestudy-knob-plus"),
            "--output",
            str(output_report),
            "--compression-min",
            str(args.compression_min),
            "--tilt-max",
            str(args.tilt_max),
        ],
        cwd=repo_root,
    )

    print("[done] report:", output_report)


if __name__ == "__main__":
    main()
