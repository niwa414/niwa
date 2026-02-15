#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_variants(path: Path) -> tuple[list[str], dict]:
    data = load_json(path)
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
    return ordered, data


def sign(val: float | None) -> int | None:
    if val is None:
        return None
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def load_tilt_series(path: Path) -> tuple[list[float], list[float]]:
    if not path.exists():
        return [], []
    times = []
    amps = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                times.append(float(row.get("time_s", "nan")))
                amps.append(float(row.get("amp_xy", "nan")))
            except Exception:
                continue
    return times, amps


def label_for(case_id: str, metrics: dict) -> str:
    amp = metrics.get("tilt_seed_y_offset_amp")
    if amp is not None:
        return f"{case_id} (amp={amp:.4f})"
    return case_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze H5 robustness sweep for seed_amp +/-10%."
    )
    parser.add_argument("--variants", required=True, help="Path to variants.json.")
    parser.add_argument("--metrics", required=True, help="Output metrics JSON.")
    parser.add_argument("--summary", required=True, help="Output summary JSON.")
    parser.add_argument("--plots-dir", required=True, help="Output plots directory.")
    args = parser.parse_args()

    variant_ids, cfg = load_variants(Path(args.variants))
    if not variant_ids:
        raise SystemExit("No variants found in variants.json.")

    gamma_ratio_max = float(cfg.get("gamma_ratio_max", 10.0))
    require_same_sign = bool(cfg.get("require_same_sign", True))

    variants = {}
    for case_id in variant_ids:
        passfail_path = Path("outputs") / case_id / "analysis" / "PASSFAIL.json"
        metrics_path = Path("outputs") / case_id / "analysis" / "metrics.json"
        passfail = load_json(passfail_path)
        metrics = passfail.get("metrics") or load_json(metrics_path)
        result = passfail.get("result") or passfail.get("status")
        variants[case_id] = {
            "case_id": case_id,
            "result": result,
            "metrics": metrics,
            "merge_time_frac": metrics.get("merge_time_frac"),
            "amp_ratio_fit_best": metrics.get("amp_ratio_fit_best"),
            "gamma_best": metrics.get("gamma_best"),
            "r2_best": metrics.get("r2_best"),
            "fit_points": metrics.get("fit_points"),
            "tilt_post_merge_samples": metrics.get("tilt_post_merge_samples"),
            "tilt_seed_y_offset_amp": metrics.get("tilt_seed_y_offset_amp"),
        }

    variant_pass = [
        (variants[cid].get("result") == "PASS") for cid in variant_ids
    ]
    all_variants_pass = all(variant_pass)

    gammas = [variants[cid].get("gamma_best") for cid in variant_ids]
    signs = [sign(g) for g in gammas]
    gamma_same_sign = False
    if require_same_sign:
        if None not in signs and len(set(signs)) == 1 and signs[0] != 0:
            gamma_same_sign = True
    else:
        gamma_same_sign = True

    gamma_ratio = None
    gamma_ratio_ok = False
    abs_gammas = [abs(g) if g is not None else None for g in gammas]
    if None not in abs_gammas and all(val > 0 for val in abs_gammas):
        gamma_ratio = max(abs_gammas) / min(abs_gammas)
        gamma_ratio_ok = gamma_ratio <= gamma_ratio_max and gamma_same_sign
    else:
        gamma_ratio_ok = False

    sweep_pass = all_variants_pass and gamma_ratio_ok

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    labels = [label_for(cid, variants[cid].get("metrics") or {}) for cid in variant_ids]
    gamma_vals = [variants[cid].get("gamma_best") for cid in variant_ids]
    r2_vals = [variants[cid].get("r2_best") for cid in variant_ids]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(labels)), gamma_vals)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("gamma_best")
    plt.title("Gamma Comparison")
    plt.grid(True, axis="y")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(labels)), r2_vals)
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("r2_best")
    plt.title("R2 Comparison")
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "gamma_r2_compare.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    for cid, label in zip(variant_ids, labels, strict=False):
        series_path = Path("outputs") / cid / "analysis" / "tilt_post_merge_series.csv"
        times, amps = load_tilt_series(series_path)
        if times:
            plt.plot(times, amps, marker="o", label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Tilt amp (post-merge)")
    plt.title("Post-merge Tilt Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "tilt_amp_post_merge_compare.png")
    plt.close()

    summary = {
        "variant_ids": variant_ids,
        "gamma_ratio_max": gamma_ratio_max,
        "require_same_sign": require_same_sign,
        "gamma_same_sign": gamma_same_sign,
        "gamma_ratio": gamma_ratio,
        "gamma_ratio_ok": gamma_ratio_ok,
        "all_variants_pass": all_variants_pass,
        "sweep_pass": sweep_pass,
        "variants": variants,
    }

    metrics_out = {
        "sweep_pass": sweep_pass,
        "all_variants_pass": all_variants_pass,
        "gamma_ratio_ok": gamma_ratio_ok,
        "gamma_same_sign": gamma_same_sign,
        "gamma_ratio": gamma_ratio,
        "gamma_ratio_max": gamma_ratio_max,
        "require_same_sign": require_same_sign,
        "variant_ids": variant_ids,
    }

    Path(args.metrics).write_text(
        json.dumps(metrics_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
