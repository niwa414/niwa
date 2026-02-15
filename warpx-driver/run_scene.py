#!/usr/bin/env python3
"""
Scene runner helper: load a scene YAML/JSON and emit (or execute) the export + WarpX commands.

Usage:
  PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \\
    python warpx-driver/run_scene.py --scene scenes/ipa_a1_merge.yaml --vtk outputs/mhd/your.vtk --run-tag ipa_a1_smoke

Defaults:
  - Uses export mesh if present, otherwise falls back to warpx mesh.
  - Builds fluid and B export commands, then a WarpX fluid-init command (hybrid disabled by default).
  - Does not execute unless --run is given.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
import json
import time

from scene_loader import load_scene, scene_export_mesh, scene_warpx_mesh


REPO_ROOT = Path(__file__).resolve().parent.parent


def _scene_external_basis(scene: dict) -> list[dict] | None:
    fields = scene.get("fields", {})
    if not isinstance(fields, dict):
        return None
    basis = fields.get("external_basis", None)
    if basis is None:
        return None
    if not isinstance(basis, list):
        raise ValueError("scene.fields.external_basis must be a list")
    out: list[dict] = []
    for i, item in enumerate(basis):
        if not isinstance(item, dict):
            raise ValueError(f"scene.fields.external_basis[{i}] must be a mapping")
        out.append(item)
    return out


def _scene_induced_E(scene: dict) -> bool:
    fields = scene.get("fields", {})
    if not isinstance(fields, dict):
        return False
    return bool(fields.get("induced_E", False))


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _basis_generation_commands(
    external_basis: list[dict],
    export_mesh: dict,
    *,
    regenerate: bool,
) -> list[tuple[str, list[str]]]:
    cmds: list[tuple[str, list[str]]] = []
    for i, cfg in enumerate(external_basis):
        bfile = cfg.get("bfile", None)
        if not bfile:
            raise ValueError(f"scene.fields.external_basis[{i}] missing 'bfile'")
        bfile_repo = _resolve_repo_path(str(bfile))
        gen = cfg.get("generator", None)
        if (not regenerate) and bfile_repo.exists():
            continue
        if gen is None:
            raise ValueError(
                f"External basis file not found and no generator specified: {bfile} "
                f"(scene.fields.external_basis[{i}].generator)"
            )
        if not isinstance(gen, dict):
            raise ValueError(f"scene.fields.external_basis[{i}].generator must be a mapping")
        mode = gen.get("mode", None)
        if not mode:
            raise ValueError(f"scene.fields.external_basis[{i}].generator missing 'mode'")

        cmd = [
            sys.executable,
            "warpx-driver/export_b_opmd_from_vtk.py",
            "--mode",
            str(mode),
            "--output-bfile",
            str(bfile),
            "--nr",
            str(export_mesh["nr"]),
            "--nz",
            str(export_mesh["nz"]),
            "--r-min",
            str(export_mesh["r_min"]),
            "--r-max",
            str(export_mesh["r_max"]),
            "--z-min",
            str(export_mesh["z_min"]),
            "--z-max",
            str(export_mesh["z_max"]),
        ]

        if mode == "uniform":
            cmd += ["--Bz-const", str(float(gen.get("Bz_const", 1.0)))]
        elif mode in {"mirror", "mirror-delta"}:
            cmd += ["--Bz-center", str(float(gen.get("Bz_center", 1.0)))]
            cmd += ["--mirror-ratio", str(float(gen.get("mirror_ratio", 1.5)))]
            if "mirror_center_z" in gen and gen["mirror_center_z"] is not None:
                cmd += ["--mirror-center-z", str(float(gen["mirror_center_z"]))]
            if "mirror_half_length" in gen and gen["mirror_half_length"] is not None:
                cmd += ["--mirror-half-length", str(float(gen["mirror_half_length"]))]
            clamp = gen.get("clamp", True)
            if not bool(clamp):
                cmd.append("--no-mirror-clamp")
        else:
            raise ValueError(
                f"Unsupported external basis generator mode '{mode}' in scene.fields.external_basis[{i}]"
            )

        basis_name = cfg.get("name", f"basis{i}")
        cmds.append((f"export-basis-{basis_name}", cmd))
    return cmds


def build_commands(
    scene_path: Path,
    vtk_path: Path,
    run_tag: str | None,
    hybrid: bool,
    ppc: int,
    axis_mode: str,
    fold_r: bool,
    rho_scale: float,
    press_scale: float,
    vel_scale: float,
    b_scale: float,
    diag_dir: Path,
    max_steps: int | None,
    regenerate_basis: bool,
):
    scene = load_scene(scene_path)
    exp = scene_export_mesh(scene)
    wp = scene_warpx_mesh(scene)
    external_basis = _scene_external_basis(scene)
    induced_E = _scene_induced_E(scene)

    fluid_out = Path("warpx-driver") / f"fluid_init_{scene_path.stem}.h5"
    b_out = Path("warpx-driver") / f"B_ext_{scene_path.stem}.h5"

    export_common = [
        "--nr",
        str(exp["nr"]),
        "--nz",
        str(exp["nz"]),
        "--r-min",
        str(exp["r_min"]),
        "--r-max",
        str(exp["r_max"]),
        "--z-min",
        str(exp["z_min"]),
        "--z-max",
        str(exp["z_max"]),
    ]

    fluid_scaling = [
        "--axis-mode",
        axis_mode,
        "--rho-scale",
        f"{rho_scale:.6g}",
        "--press-scale",
        f"{press_scale:.6g}",
        "--vel-scale",
        f"{vel_scale:.6g}",
    ]
    if fold_r:
        fluid_scaling.append("--fold-r")

    b_scaling = ["--axis-mode", axis_mode, "--B-scale", f"{b_scale:.6g}"]
    if fold_r:
        b_scaling.append("--fold-r")

    cmds: list[tuple[str, list[str]]] = []
    cmds.append(
        (
            "export-fluid",
            [
                sys.executable,
                "warpx-driver/export_fluid_to_opmd.py",
                "--input-vtk",
                str(vtk_path),
                "--output-fluid",
                str(fluid_out),
            ]
            + export_common
            + fluid_scaling,
        )
    )
    if not external_basis:
        cmds.append(
            (
                "export-b",
                [
                    sys.executable,
                    "warpx-driver/export_b_opmd_from_vtk.py",
                    "--mode",
                    "from-vtk",
                    "--input-vtk",
                    str(vtk_path),
                    "--output-bfile",
                    str(b_out),
                ]
                + export_common
                + b_scaling,
            )
        )
    else:
        cmds.extend(_basis_generation_commands(external_basis, exp, regenerate=regenerate_basis))

    warpx_cmd = [
        sys.executable,
        "warpx-driver/warpx_driver.py",
        "--mode",
        "fluid-init",
        "--fluid-file",
        str(fluid_out),
        "--nr",
        str(wp["nr"]),
        "--nz",
        str(wp["nz"]),
        "--r-max",
        str(wp["r_max"]),
        "--z-max",
        str(wp["z_max"] * 2.0),  # warpx_driver expects length, but wp has half-length/coordinate
        "--dt",
        f"{wp['dt']:.6g}",
        "--diag-period",
        str(wp["diag_period"]),
        "--n-azimuthal-modes",
        str(wp["n_azimuthal_modes"]),
        "--ppc",
        str(ppc),
    ]
    if max_steps is not None:
        warpx_cmd += ["--max-steps", str(max_steps)]
    if run_tag:
        warpx_cmd += ["--run-tag", run_tag]
    if diag_dir:
        warpx_cmd += ["--metadata-dir", str(diag_dir)]
    if hybrid:
        warpx_cmd.append("--hybrid")
    if induced_E and external_basis:
        warpx_cmd.append("--induced-E")

    if external_basis:
        # external_basis config:
        #  - const coefficients are consumed sequentially for bases without waveforms
        #  - waveform coefficients are index-based (basis-aligned), so we pass placeholders
        basis_files: list[str] = []
        basis_consts: list[float] = []
        basis_waveforms: list[str] = [""] * len(external_basis)
        basis_scales: list[float] = [1.0] * len(external_basis)

        for i, cfg in enumerate(external_basis):
            bfile = cfg.get("bfile", None)
            if not bfile:
                raise ValueError(f"scene.fields.external_basis[{i}] missing 'bfile'")
            basis_files.append(str(bfile))

            wf_csv = cfg.get("coeff_waveform_csv", None)
            if wf_csv:
                wf_repo = _resolve_repo_path(str(wf_csv))
                if not wf_repo.exists():
                    raise ValueError(f"Waveform CSV not found: {wf_csv}")
                col = cfg.get("coeff_column", None)
                wf_spec = str(wf_csv)
                if col is not None:
                    wf_spec = f"{wf_spec}:{col}"
                basis_waveforms[i] = wf_spec
                basis_scales[i] = float(cfg.get("coeff_scale", 1.0))
            else:
                basis_consts.append(float(cfg.get("coeff_const", 0.0)))

        for bf in basis_files:
            warpx_cmd += ["--external-basis", bf]

        if any(w.strip() for w in basis_waveforms):
            for wf in basis_waveforms:
                warpx_cmd += ["--external-basis-waveform", wf]
            for sc in basis_scales:
                warpx_cmd += ["--external-basis-waveform-scale", f"{sc:.6g}"]

        for cst in basis_consts:
            warpx_cmd += ["--external-basis-const", f"{cst:.6g}"]
    else:
        warpx_cmd += ["--b-file", str(b_out)]

    cmds.append(("warpx", warpx_cmd))
    return scene, exp, wp, cmds


def main():
    ap = argparse.ArgumentParser(description="Run or print export+WarpX commands from a scene file.")
    ap.add_argument("--scene", required=True, help="Scene YAML/JSON path.")
    ap.add_argument("--vtk", required=True, help="Athena VTK snapshot path.")
    ap.add_argument("--run-tag", default=None, help="Optional tag for WarpX metadata.")
    ap.add_argument("--hybrid", action="store_true", help="Use WarpX Hybrid-PIC.")
    ap.add_argument("--ppc", type=int, default=2, help="Particles per cell for fluid-init.")
    ap.add_argument(
        "--axis-mode",
        choices=["x_z_y_r", "x_r_y_z"],
        default="x_z_y_r",
        help="Map Athena axes to (z,r) for exporters (default: x1->z, x2->r).",
    )
    ap.add_argument("--fold-r", action="store_true", help="Fold negative r to r>=0 during export.")
    ap.add_argument("--rho-scale", type=float, default=1.0, help="Scale factor for rho when exporting fluid (kg/m^3 per code unit).")
    ap.add_argument("--press-scale", type=float, default=1.0, help="Scale factor for pressure when exporting fluid.")
    ap.add_argument("--vel-scale", type=float, default=1.0, help="Scale factor for velocities when exporting fluid.")
    ap.add_argument("--B-scale", type=float, default=1.0, help="Scale factor applied to exported B field.")
    ap.add_argument("--max-steps", type=int, default=None, help="Override WarpX --max-steps.")
    ap.add_argument(
        "--regenerate-basis",
        action="store_true",
        help="Regenerate external basis files even if they already exist (only if scene provides generators).",
    )
    ap.add_argument(
        "--diag-dir",
        type=str,
        default="outputs/warpx",
        help="Root directory for diagnostics/metadata.",
    )
    ap.add_argument("--run", action="store_true", help="Actually execute commands (default: print only).")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    vtk_path = Path(args.vtk)
    if not vtk_path.exists():
        raise SystemExit(f"VTK not found: {vtk_path}")

    run_id = args.run_tag or scene_path.stem
    diag_dir = Path(args.diag_dir) / run_id
    diag_dir.mkdir(parents=True, exist_ok=True)

    scene, exp, wp, cmds = build_commands(
        scene_path,
        vtk_path,
        args.run_tag,
        args.hybrid,
        args.ppc,
        args.axis_mode,
        args.fold_r,
        args.rho_scale,
        args.press_scale,
        args.vel_scale,
        args.B_scale,
        diag_dir,
        args.max_steps,
        args.regenerate_basis,
    )
    print(f"[scene] {scene_path} -> using VTK {vtk_path}")
    for name, cmd in cmds:
        print(f"[cmd {name}] {' '.join(shlex.quote(c) for c in cmd)}")
        if args.run:
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent)

    # Save a lightweight manifest for reproducibility
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scene": str(scene_path),
        "vtk": str(vtk_path),
        "run_tag": args.run_tag,
        "hybrid": args.hybrid,
        "axis_mode": args.axis_mode,
        "fold_r": args.fold_r,
        "scales": {
            "rho": args.rho_scale,
            "press": args.press_scale,
            "vel": args.vel_scale,
            "B": args.B_scale,
        },
        "export_mesh": exp,
        "warpx_mesh": wp,
        "commands": [{"name": n, "cmd": c} for n, c in cmds],
        "diag_dir": str(diag_dir),
    }
    try:
        manifest_path = diag_dir / f"run_scene_manifest_{run_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"[manifest] saved to {manifest_path}")
    except Exception as exc:
        print(f"Warning: failed to write manifest: {exc}")


if __name__ == "__main__":
    main()
