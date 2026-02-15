#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WarpX with external field files.")
    parser.add_argument("--warpx-args", type=Path, required=True, help="WarpX driver JSON config.")
    parser.add_argument("--opmd-fluid", type=Path, required=True, help="openPMD fluid file.")
    parser.add_argument("--opmd-b", type=Path, required=True, help="openPMD B field file.")
    parser.add_argument("--metadata-dir", type=Path, required=True, help="Output metadata directory.")
    parser.add_argument("--diag-dir", type=Path, required=True, help="Output diagnostics directory.")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag for WarpX driver.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter.")
    args = parser.parse_args()

    warpx_driver = Path("warpx-driver") / "warpx_driver_3d.py"
    if not warpx_driver.exists():
        raise SystemExit(f"WarpX driver not found: {warpx_driver}")

    run_tag = args.run_tag or os.environ.get("FUSION_CASE_ID") or args.metadata_dir.name
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.diag_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(warpx_driver),
        "--from-json",
        str(args.warpx_args),
        "--opmd-fluid",
        str(args.opmd_fluid),
        "--opmd-b",
        str(args.opmd_b),
        "--metadata-dir",
        str(args.metadata_dir),
        "--diag-dir",
        str(args.diag_dir),
        "--run-tag",
        run_tag,
    ]

    env = os.environ.copy()
    if "PYTHONPATH" not in env or not env["PYTHONPATH"]:
        default_pywarpx = str(Path("pic-warpx-25.11") / "build-3d" / "lib" / "site-packages")
        env["PYTHONPATH"] = default_pywarpx
    print("[p1p1] warpX cmd:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
