#!/usr/bin/env python3
"""
One-click pipeline: Athena++ Hall FRC snapshot -> openPMD -> WarpX fluid-init.

Steps:
  1) (optional) run Athena++ with a Hall FRC input to generate VTK/HST.
  2) pick a VTK (default: latest matching pattern) and export fluid/B openPMD.
  3) run WarpX fluid-init with the exported files.
  4) (optional) validate resulting metadata.

Example:
  python hall_frc_pipeline.py \
    --athena-bin athena-24.0/bin/athena \
    --athena-input athena-24.0/inputs/mhd/athinput.hall_frc_init \
    --athena-out outputs/mhd/hall_frc_init \
    --vtk-pattern \"outputs/mhd/hall_frc_init/hall_frc_init.block0.out1.*.vtk\" \
    --nr 64 --nz 64 --r-max 2 --z-max 2 \
    --ppc 2 --max-steps 10 --diag-period 5 \
    --tilt-eps 0.0
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def find_latest(pattern: str) -> Path:
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {pattern}")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description="Athena Hall FRC -> WarpX fluid-init pipeline")
    ap.add_argument("--athena-bin", default="athena-24.0/bin/athena", help="Path to Athena++ binary.")
    ap.add_argument("--athena-input", default="athena-24.0/inputs/mhd/athinput.hall_frc_init", help="Athena++ input file.")
    ap.add_argument(
        "--athena-out",
        default="outputs/mhd/hall_frc_init",
        help="Athena++ output directory (used when running Athena).",
    )
    ap.add_argument("--skip-athena", action="store_true", help="Skip running Athena (reuse existing VTK).")
    ap.add_argument(
        "--vtk-pattern",
        default="outputs/mhd/hall_frc_init/hall_frc_init.block0.out1.*.vtk",
        help="Glob to pick VTK (latest chosen).",
    )
    ap.add_argument("--vtk-path", default=None, help="Explicit VTK path (overrides pattern).")
    ap.add_argument("--nr", type=int, default=64, help="R cells for export/WarpX.")
    ap.add_argument("--nz", type=int, default=64, help="Z cells for export/WarpX.")
    ap.add_argument("--r-max", type=float, default=2.0, dest="r_max", help="r_max for export/WarpX.")
    ap.add_argument("--z-max", type=float, default=2.0, dest="z_max", help="z_max for WarpX (export uses +/-z_max/2 if not set).")
    ap.add_argument("--r-min", type=float, default=0.0, dest="r_min", help="r_min for export.")
    ap.add_argument("--z-min", type=float, default=None, help="z_min for export (default: -z_max/2).")
    ap.add_argument("--z-max-export", type=float, default=None, dest="z_max_export", help="z_max for export (default: +z_max/2).")
    ap.add_argument("--ppc", type=int, default=2, help="Particles per cell for fluid-init.")
    ap.add_argument("--max-steps", type=int, default=10, help="WarpX max steps.")
    ap.add_argument("--diag-period", type=int, default=5, help="WarpX diag period.")
    ap.add_argument("--tilt-eps", type=float, default=0.0, help="Optional m=1 density perturbation in WarpX loader.")
    ap.add_argument("--run-tag", default="hall_frc_pipeline", help="Tag for WarpX metadata.")
    ap.add_argument("--validate", action="store_true", help="Run validate_metadata.py after WarpX.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    z_min = args.z_min if args.z_min is not None else -0.5 * args.z_max
    z_max_export = args.z_max_export if args.z_max_export is not None else 0.5 * args.z_max

    # 1) Athena++ run
    if not args.skip_athena:
        athena_out = Path(args.athena_out)
        if not athena_out.is_absolute():
            athena_out = repo_root / athena_out
        athena_out.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                str(args.athena_bin),
                "-i",
                str(args.athena_input),
                "-d",
                str(athena_out),
            ],
            cwd=repo_root,
        )

    # 2) Locate VTK
    if args.vtk_path:
        vtk = Path(args.vtk_path)
    else:
        vtk = find_latest(str(repo_root / args.vtk_pattern))
    if not vtk.exists():
        raise SystemExit(f"VTK not found: {vtk}")
    print(f"[info] using VTK: {vtk}")

    # 3) Export fluid/B
    output_root = repo_root / "outputs" / "warpx" / args.run_tag
    output_root.mkdir(parents=True, exist_ok=True)
    fluid_out = output_root / "fluid_init_hall_frc.h5"
    b_out = output_root / "B_ext_hall_frc.h5"
    common_args = [
        "--nr",
        str(args.nr),
        "--nz",
        str(args.nz),
        "--r-min",
        str(args.r_min),
        "--r-max",
        str(args.r_max),
        "--z-min",
        str(z_min),
        "--z-max",
        str(z_max_export),
    ]
    run_cmd(
        [
            sys.executable,
            "warpx-driver/export_fluid_to_opmd.py",
            "--input-vtk",
            str(vtk),
            "--output-fluid",
            str(fluid_out),
            "--fold-x",
        ]
        + common_args,
        cwd=repo_root,
    )
    run_cmd(
        [
            sys.executable,
            "warpx-driver/export_b_opmd_from_vtk.py",
            "--mode",
            "from-vtk",
            "--input-vtk",
            str(vtk),
            "--output-bfile",
            str(b_out),
        ]
        + common_args,
        cwd=repo_root,
    )

    # 4) WarpX fluid-init
    warpx_cmd = [
        sys.executable,
        "warpx-driver/warpx_driver.py",
        "--mode",
        "fluid-init",
        "--fluid-file",
        str(fluid_out),
        "--b-file",
        str(b_out),
        "--nr",
        str(args.nr),
        "--nz",
        str(args.nz),
        "--r-max",
        str(args.r_max),
        "--z-max",
        str(args.z_max),
        "--ppc",
        str(args.ppc),
        "--max-steps",
        str(args.max_steps),
        "--diag-period",
        str(args.diag_period),
        "--run-tag",
        args.run_tag,
        "--metadata-dir",
        str(output_root),
        "--diag-dir",
        str(output_root / "diag"),
    ]
    if args.tilt_eps != 0.0:
        # WarpX cannot load external openPMD fields with n_azimuthal_modes>1 (#3829)
        raise SystemExit("tilt-eps requires n_azimuthal_modes>1, but external field loading does not support that; run tilt in const-b mode instead.")
    run_cmd(warpx_cmd, cwd=repo_root)

    # 5) Optional validation
    if args.validate:
        run_cmd(
            [
                sys.executable,
                "warpx-driver/validate_metadata.py",
                "--metadata",
                str(output_root / "warpx_run_*.json"),
                "--diag-root",
                str(output_root / "diag"),
                "--fill-diag",
            ],
            cwd=repo_root,
        )


if __name__ == "__main__":
    main()
