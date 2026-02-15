#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "tools"))
import lcr_coupling


DEFAULTS = {
    "V0": 20e3,
    "C": 10e-6,
    "R_line": 0.05,
    "L0": 1e-6,
    "L_alpha": 2e-6,
    "R_plasma0": 0.30,
    "R_min": 0.10,
    "v_ramp": 5.0,
    "R_coil": 0.35,
    "turns": 1,
    "dt": 1e-7,
    "tmax": 50e-6,
    "kB": None,
    "vtk_path": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LCR solver from JSON params.")
    parser.add_argument("--params", required=True, help="Path to LCR params JSON.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot output.")
    parser.add_argument("--plot-dir", help="Optional override for plot output directory.")
    args = parser.parse_args()

    params_path = Path(args.params)
    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    merged = DEFAULTS.copy()
    merged.update(params)
    merged["out"] = str(Path(args.out_csv))
    merged["no_plot"] = args.no_plot
    merged["plot_dir"] = args.plot_dir

    lcr_args = SimpleNamespace(**merged)
    lcr_coupling.lcr_circuit_with_compression(lcr_args)


if __name__ == "__main__":
    main()
