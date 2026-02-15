#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_variants(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ordered = []
    tilt_on = data.get("tilt_on")
    tilt_off = data.get("tilt_off")
    if tilt_on:
        ordered.append(tilt_on)
    if tilt_off:
        ordered.append(tilt_off)
    for item in data.get("variants", []):
        case_id = item if isinstance(item, str) else item.get("id")
        if case_id and case_id not in ordered:
            ordered.append(case_id)
    return ordered


def should_skip(case_id: str) -> bool:
    passfail = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
    return passfail.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B2 tilt on/off variants.")
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "run", "analyze"],
        help="Stage to pass to run_case for each variant.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants that already have PASSFAIL.json.",
    )
    args = parser.parse_args()

    variants_path = Path(args.variants)
    case_ids = load_variants(variants_path)
    if not case_ids:
        raise SystemExit("No variants found in variants.json.")

    for case_id in case_ids:
        if args.skip_existing and should_skip(case_id):
            print(f"[b2] skip existing PASSFAIL: {case_id}")
            continue
        cmd = [
            sys.executable,
            "tools/run_case.py",
            "--case",
            case_id,
            "--stage",
            args.stage,
        ]
        print(f"[b2] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False, env=os.environ.copy())


if __name__ == "__main__":
    main()
