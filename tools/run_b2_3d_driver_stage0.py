#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return out


def parse_passfail_path(output_text: str) -> str | None:
    match = re.search(r"\[PASSFAIL\]\s+(.+PASSFAIL\.json)", output_text)
    if not match:
        return None
    return match.group(1).strip()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B2 3D driver stage0 suite.")
    parser.add_argument("--output-root", required=True, help="Case raw run directory.")
    parser.add_argument(
        "--athena-bin",
        default="athena-24.0/bin/athena",
        help="Athena++ executable path.",
    )
    parser.add_argument(
        "--smoke-input",
        default="athena-24.0/inputs/mhd/athinput.belova_b2_gs_emf_mwall3d_smoke",
        help="3D smoke input file.",
    )
    parser.add_argument(
        "--gs-athinput",
        default="athena-24.0/inputs/mhd/athinput.belova_b2_gs_emf_mwall80_smooth",
        help="2D GS input file used to generate init HDF5.",
    )
    parser.add_argument(
        "--waveform",
        default="scenes/belova_b2_mirror_ramp_smooth.csv",
        help="Waveform csv path.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    stage_root = output_root / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)

    init_h5 = stage_root / "gs_init" / "belova_b2_init.h5"
    init_h5.parent.mkdir(parents=True, exist_ok=True)

    gs_cmd = [
        sys.executable,
        str(root / "tools" / "gs_frc_athena_init.py"),
        "--athinput",
        str((root / args.gs_athinput).resolve()),
        "--output-h5",
        str(init_h5),
    ]
    gs_out = run_cmd(gs_cmd, root)
    gs_ok = init_h5.exists()

    stage_runs = [
        {
            "key": "3d0a",
            "stage": "3d0a",
            "overrides": [],
        },
        {
            "key": "3d0b",
            "stage": "3d0b",
            "overrides": [
                "problem/apply_bext_emf=true",
                "problem/piston_bc=false",
            ],
        },
        {
            "key": "3d0c",
            "stage": "3d0c",
            "overrides": [
                "problem/apply_bext_emf=false",
                "problem/piston_bc=true",
                "time/tlim=6.0e-7",
            ],
        },
    ]

    results: dict[str, dict] = {}
    for cfg in stage_runs:
        cmd = [
            sys.executable,
            str(root / "tools" / "run_3d_milestone.py"),
            "--stage",
            cfg["stage"],
            "--athena-bin",
            str((root / args.athena_bin).resolve()),
            "--input",
            str((root / args.smoke_input).resolve()),
            "--run-root",
            str(stage_root),
            "--waveform",
            str((root / args.waveform).resolve()),
            "--set",
            f"problem/init_from_hdf5={init_h5}",
        ]
        for item in cfg["overrides"]:
            cmd.extend(["--set", item])
        text = run_cmd(cmd, root)
        passfail_text_path = parse_passfail_path(text)
        passfail_path = Path(passfail_text_path) if passfail_text_path else None
        passfail = load_json(passfail_path) if passfail_path and passfail_path.exists() else {}
        results[cfg["key"]] = {
            "stage": cfg["stage"],
            "passfail": str(passfail_path) if passfail_path else None,
            "pass": bool(passfail.get("pass", False)),
            "checks": passfail.get("checks", {}),
            "stdout_tail": text.splitlines()[-25:],
        }

    suite_pass = gs_ok and all(item.get("pass", False) for item in results.values())
    summary = {
        "suite": "b2_3d_driver_stage0",
        "init_h5": str(init_h5),
        "init_h5_exists": gs_ok,
        "gs_stdout_tail": gs_out.splitlines()[-25:],
        "stage_results": results,
        "stage0_suite_pass": suite_pass,
    }
    (output_root / "stage0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
