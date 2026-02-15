#!/usr/bin/env python3
"""
Lightweight smoke pipeline that exercises the WarpX driver with tiny step counts.
Steps (all configurable):
  1) field-only (external B load only)
  2) const-b-plasma (uniform cold plasma)
  3) optional: VTK->openPMD export + bfile-plasma + fluid-init
Use --dry-run to print commands without executing.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(name: str, cmd: list[str], env: dict, cwd: Path, dry_run: bool):
    print(f"[{name}] {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def main():
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description="One-click WarpX smoke pipeline.")
    ap.add_argument(
        "--pywarpx-path",
        default=repo_root / "pic-warpx-25.11" / "build-rz" / "lib" / "site-packages",
        help="Path to pywarpx site-packages (prepends PYTHONPATH).",
    )
    ap.add_argument(
        "--output-root",
        default=repo_root / "outputs" / "warpx" / "smoke",
        help="Root directory for WarpX outputs (metadata, diags, exported files).",
    )
    ap.add_argument("--vtk", help="Optional Athena++ VTK to export for bfile/fluid smoke.")
    ap.add_argument("--nr", type=int, default=32, help="R cells for both export and WarpX.")
    ap.add_argument("--nz", type=int, default=64, help="Z cells for both export and WarpX.")
    ap.add_argument("--r-max", type=float, default=0.1, dest="r_max", help="Domain r_max (WarpX) and export r range.")
    ap.add_argument("--z-span", type=float, default=0.2, dest="z_span", help="Total z span; export uses [-z_span/2, z_span/2].")
    ap.add_argument("--const-b", type=float, default=0.05, dest="const_b", help="Constant Bz for const-b-plasma step.")
    ap.add_argument("--max-steps", type=int, default=10, help="Step count for WarpX smoke runs (field-only uses min(5,max_steps)).")
    ap.add_argument("--diag-period", type=int, default=5, help="Diagnostic period for WarpX runs.")
    ap.add_argument("--tilt", action="store_true", help="Include a tilt (m=1, n_azimuthal_modes=2) smoke run.")
    ap.add_argument("--tilt-eps", type=float, default=0.05, help="Tilt perturbation amplitude for tilt smoke.")
    ap.add_argument("--tilt-run-tag", default="tilt_smoke_m1", help="Run tag for tilt smoke metadata.")
    ap.add_argument(
        "--validate-metadata",
        action="store_true",
        help="Run validate_metadata.py on generated warpx_run_*.json after pipeline finishes.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = ap.parse_args()

    env = os.environ.copy()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pywarpx_path = Path(args.pywarpx_path)
    env["PYTHONPATH"] = f"{pywarpx_path}:{env.get('PYTHONPATH','')}"

    python = sys.executable
    z_min = -0.5 * args.z_span
    z_max = 0.5 * args.z_span

    diag_field = output_root / "diag_field_only"
    run_cmd(
        "field-only",
        [
            python,
            "warpx-driver/warpx_driver.py",
            "--mode", "field-only",
            "--max-steps", str(min(5, args.max_steps)),
            "--dt", "1e-9",
            "--diag-period", str(args.diag_period),
            "--nr", str(args.nr),
            "--nz", str(args.nz),
            "--r-max", str(args.r_max),
            "--z-max", str(args.z_span),
            "--metadata-dir", str(output_root),
            "--diag-dir", str(diag_field),
        ],
        env,
        repo_root,
        args.dry_run,
    )

    diag_const = output_root / "diag_const_b_plasma"
    run_cmd(
        "const-b-plasma",
        [
            python,
            "warpx-driver/warpx_driver.py",
            "--mode", "const-b-plasma",
            "--const-B", str(args.const_b),
            "--max-steps", str(args.max_steps),
            "--dt", "1e-9",
            "--diag-period", str(args.diag_period),
            "--nr", str(args.nr),
            "--nz", str(args.nz),
            "--r-max", str(args.r_max),
            "--z-max", str(args.z_span),
            "--metadata-dir", str(output_root),
            "--diag-dir", str(diag_const),
        ],
        env,
        repo_root,
        args.dry_run,
    )

    if args.tilt:
        diag_tilt = output_root / "diag_tilt_m1"
        run_cmd(
            "tilt-m1",
            [
                python,
                "warpx-driver/warpx_driver.py",
                "--mode", "const-b-plasma",
                "--solver", "yee",
                "--n-azimuthal-modes", "2",
                "--const-B", str(args.const_b),
                "--tilt-eps", str(args.tilt_eps),
                "--ppc", "4",
                "--max-steps", str(args.max_steps),
                "--dt", "1e-9",
                "--diag-period", str(args.diag_period),
                "--monitor-interval", str(max(1, args.diag_period)),
                "--drop-threshold", "100",
                "--run-tag", args.tilt_run_tag,
                "--metadata-dir", str(output_root),
                "--diag-dir", str(diag_tilt),
            ],
            env,
            repo_root,
            args.dry_run,
        )

    if args.validate_metadata:
        run_cmd(
            "validate-metadata",
            [
                python,
                "warpx-driver/validate_metadata.py",
                "--metadata",
                str(output_root / "warpx_run_*.json"),
                "--diag-root",
                str(output_root),
                "--max-mode",
                "2",
                "--fill-diag",
            ],
            env,
            repo_root,
            args.dry_run,
        )

    if args.vtk:
        vtk_path = Path(args.vtk)
        if not vtk_path.exists():
            raise SystemExit(f"VTK file not found: {vtk_path}")
        b_file = output_root / "B_ext_from_vtk.h5"
        fluid_file = output_root / "fluid_init_from_vtk.h5"

        run_cmd(
            "export-bfile",
            [
                python,
                "warpx-driver/export_b_opmd_from_vtk.py",
                "--mode", "from-vtk",
                "--input-vtk", str(vtk_path),
                "--output-bfile", str(b_file),
                "--nr", str(args.nr),
                "--nz", str(args.nz),
                "--r-min", "0.0",
                "--r-max", str(args.r_max),
                "--z-min", str(z_min),
                "--z-max", str(z_max),
            ],
            env,
            repo_root,
            args.dry_run,
        )

        run_cmd(
            "export-fluid",
            [
                python,
                "warpx-driver/export_fluid_to_opmd.py",
                "--input-vtk", str(vtk_path),
                "--output-fluid", str(fluid_file),
                "--nr", str(args.nr),
                "--nz", str(args.nz),
                "--r-min", "0.0",
                "--r-max", str(args.r_max),
                "--z-min", str(z_min),
                "--z-max", str(z_max),
            ],
            env,
            repo_root,
            args.dry_run,
        )

        diag_bfile = output_root / "diag_bfile_plasma"
        run_cmd(
            "bfile-plasma",
            [
                python,
                "warpx-driver/warpx_driver.py",
                "--mode", "bfile-plasma",
                "--b-file", str(b_file),
                "--max-steps", str(args.max_steps),
                "--dt", "1e-9",
                "--diag-period", str(args.diag_period),
                "--nr", str(args.nr),
                "--nz", str(args.nz),
                "--r-max", str(args.r_max),
                "--z-max", str(args.z_span),
                "--metadata-dir", str(output_root),
                "--diag-dir", str(diag_bfile),
            ],
            env,
            repo_root,
            args.dry_run,
        )

        diag_fluid = output_root / "diag_fluid_init"
        run_cmd(
            "fluid-init",
            [
                python,
                "warpx-driver/warpx_driver.py",
                "--mode", "fluid-init",
                "--fluid-file", str(fluid_file),
                "--b-file", str(b_file),
                "--max-steps", str(args.max_steps),
                "--dt", "1e-9",
                "--diag-period", str(args.diag_period),
                "--ppc", "1",
                "--nr", str(args.nr),
                "--nz", str(args.nz),
                "--r-max", str(args.r_max),
                "--z-max", str(args.z_span),
                "--metadata-dir", str(output_root),
                "--diag-dir", str(diag_fluid),
            ],
            env,
            repo_root,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
