#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_STAGE_INPUT = {
    "3d1": "athena-24.0/inputs/mhd/athinput.belova_b2_gs_emf_mwall3d_gate60",
    "3d2": "athena-24.0/inputs/mhd/athinput.belova_b2_gs_emf_mwall3d_gate80_dense",
}


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


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
    parser = argparse.ArgumentParser(description="Run B2 3D driver stage1/stage2 suite.")
    parser.add_argument("--stage", required=True, choices=["3d1", "3d2"])
    parser.add_argument("--output-root", required=True, help="Case raw run directory.")
    parser.add_argument(
        "--athena-bin",
        default="athena-24.0/bin/athena",
        help="Athena++ executable path.",
    )
    parser.add_argument(
        "--stage-input",
        default="",
        help="Override 3D stage input file (defaults by stage).",
    )
    parser.add_argument(
        "--gs-athinput",
        default="athena-24.0/inputs/mhd/athinput.belova_b2_gs_emf_mwall80_smooth",
        help="2D GS input file used to generate init HDF5.",
    )
    parser.add_argument(
        "--waveform",
        default="scenes/belova_b2_mirror_ramp_smooth.csv",
        help="Waveform csv path used by the analyzer.",
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
    gs_rc, gs_out = run_cmd(gs_cmd, root)
    gs_ok = gs_rc == 0 and init_h5.exists()

    stage_input_rel = args.stage_input.strip() or DEFAULT_STAGE_INPUT[args.stage]
    run_cmdline = [
        sys.executable,
        str(root / "tools" / "run_3d_milestone.py"),
        "--stage",
        args.stage,
        "--athena-bin",
        str((root / args.athena_bin).resolve()),
        "--input",
        str((root / stage_input_rel).resolve()),
        "--run-root",
        str(stage_root),
        "--waveform",
        str((root / args.waveform).resolve()),
        "--set",
        f"problem/init_from_hdf5={init_h5}",
    ]
    run_rc, run_out = run_cmd(run_cmdline, root)
    passfail_text_path = parse_passfail_path(run_out)
    passfail_path = Path(passfail_text_path) if passfail_text_path else None
    passfail = load_json(passfail_path) if passfail_path and passfail_path.exists() else {}
    stage_pass = bool(passfail.get("pass", False))

    suite_pass = gs_ok and run_rc == 0 and stage_pass
    summary = {
        "suite": f"b2_3d_driver_{args.stage}",
        "stage": args.stage,
        "stage_input": stage_input_rel,
        "init_h5": str(init_h5),
        "init_h5_exists": gs_ok,
        "gs_returncode": gs_rc,
        "gs_stdout_tail": gs_out.splitlines()[-25:],
        "run_returncode": run_rc,
        "run_stdout_tail": run_out.splitlines()[-25:],
        "stage_result": {
            "passfail": str(passfail_path) if passfail_path else None,
            "pass": stage_pass,
            "checks": passfail.get("checks", {}),
        },
        "stage_suite_pass": suite_pass,
    }
    (output_root / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
