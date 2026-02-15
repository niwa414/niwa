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
    for key in ("amp_minus", "amp_plus", "baseline"):
        case_id = data.get(key)
        if case_id:
            ordered.append(case_id)
    for item in data.get("variants", []):
        if isinstance(item, str):
            case_id = item
        else:
            case_id = item.get("id")
        if case_id and case_id not in ordered:
            ordered.append(case_id)
    return ordered


def should_skip(case_id: str) -> bool:
    passfail = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
    return passfail.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H5 robustness sweep variants.")
    parser.add_argument(
        "--variants",
        required=True,
        help="Path to variants.json describing H5 seed_amp variants.",
    )
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
            print(f"[h5] skip existing PASSFAIL: {case_id}")
            continue
        cmd = [
            sys.executable,
            "tools/run_case.py",
            "--case",
            case_id,
            "--stage",
            args.stage,
        ]
        print(f"[h5] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False, env=os.environ.copy())


if __name__ == "__main__":
    main()
