#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LCR CSV and emit metrics/plots.")
    parser.add_argument("--csv", required=True, help="Input CSV from lcr_coupling.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON path.")
    parser.add_argument("--plots-dir", required=True, help="Directory for plot PNGs.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    data = load_csv(csv_path)

    t_us = data["t"] * 1e6
    current_ka = data["I"] / 1e3
    b_est = data["B_est"]
    e_cap = data["E_cap"]
    e_ind = data["E_ind"]
    e_total = e_cap + e_ind

    energy_initial = float(e_total[0]) if e_total.size else 0.0
    energy_final = float(e_total[-1]) if e_total.size else 0.0
    energy_rel_drift = None
    if energy_initial != 0.0:
        energy_rel_drift = (energy_final - energy_initial) / energy_initial

    metrics = {
        "energy_initial_J": energy_initial,
        "energy_final_J": energy_final,
        "energy_rel_drift": energy_rel_drift,
        "peak_current_kA": float(np.max(np.abs(current_ka))) if current_ka.size else 0.0,
        "peak_b_est_T": float(np.max(np.abs(b_est))) if b_est.size else 0.0,
        "steps": int(len(t_us)),
    }

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(t_us, current_ka, color="tab:red")
    plt.xlabel("Time (us)")
    plt.ylabel("Current (kA)")
    plt.title("LCR Current")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "lcr_current.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(t_us, b_est, color="tab:blue")
    plt.xlabel("Time (us)")
    plt.ylabel("B_est (T)")
    plt.title("LCR Estimated B")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "lcr_b_est.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(t_us, e_total, color="tab:green", label="E_total")
    plt.plot(t_us, e_cap, color="tab:orange", linestyle="--", label="E_cap")
    plt.plot(t_us, e_ind, color="tab:purple", linestyle="--", label="E_ind")
    plt.xlabel("Time (us)")
    plt.ylabel("Energy (J)")
    plt.title("LCR Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "lcr_energy.png")
    plt.close()


if __name__ == "__main__":
    main()
