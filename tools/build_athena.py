#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure and build Athena++.")
    parser.add_argument(
        "--athena-root",
        default="athena-24.0",
        help="Path to Athena++ root directory.",
    )
    parser.add_argument("--prob", required=True, help="Problem generator name (e.g., hall_gem).")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel build jobs.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if binary exists.")
    parser.add_argument("--clean", action="store_true", help="Run make clean before build.")
    parser.add_argument("--mpi", action="store_true", help="Enable MPI build.")
    parser.add_argument("--magnetic", action="store_true", help="Enable magnetic fields (-b).")
    args = parser.parse_args()

    athena_root = Path(args.athena_root)
    athena_bin = athena_root / "bin" / "athena"
    if athena_bin.exists() and not args.force:
        print(f"[athena-build] using existing binary: {athena_bin}")
        return

    configure_cmd = ["./configure.py", f"--prob={args.prob}"]
    if args.mpi:
        configure_cmd.append("--mpi")
    if args.magnetic:
        configure_cmd.append("-b")

    print(f"[athena-build] configure: {' '.join(configure_cmd)}")
    subprocess.run(configure_cmd, cwd=athena_root, check=True)

    if args.clean:
        clean_cmd = ["make", "clean"]
        print(f"[athena-build] clean: {' '.join(clean_cmd)}")
        subprocess.run(clean_cmd, cwd=athena_root, check=True)

    make_cmd = ["make", f"-j{args.jobs}"]
    print(f"[athena-build] build: {' '.join(make_cmd)}")
    subprocess.run(make_cmd, cwd=athena_root, check=True)


if __name__ == "__main__":
    main()
