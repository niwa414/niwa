#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha1_file(path: Path) -> str | None:
    if not path.exists():
        return None
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def parse_histogram(path: Path):
    header = []
    rows = []
    lengths = []
    has_nan = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            header.append(line.rstrip("\n"))
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            step = float(parts[0])
            time = float(parts[1])
            counts = [float(val) for val in parts[2:]]
        except Exception:
            continue
        if any(math.isnan(val) for val in counts) or math.isnan(step) or math.isnan(time):
            has_nan = True
        rows.append((step, time, counts))
        lengths.append(len(counts))
    return header, rows, lengths, has_nan


def parse_particle_number(path: Path):
    if not path.exists():
        return None, None, None, None
    steps = []
    times = []
    n_macro = []
    n_weight = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            steps.append(float(parts[0]))
            times.append(float(parts[1]))
            n_macro.append(float(parts[2]))
            n_weight.append(float(parts[4]))
        except Exception:
            continue
    return steps, times, n_macro, n_weight


def series_has_nan(values):
    return any(math.isnan(val) for val in values)


def first_last_triplets(steps, times, values, count=3):
    head = []
    tail = []
    for idx in range(min(count, len(values))):
        head.append([float(steps[idx]), float(times[idx]), float(values[idx])])
    for idx in range(max(len(values) - count, 0), len(values)):
        tail.append([float(steps[idx]), float(times[idx]), float(values[idx])])
    return head, tail


def histogram_quantile(centers, weights, q):
    total = float(sum(weights))
    if total <= 0.0:
        return None
    cdf = []
    acc = 0.0
    for w in weights:
        acc += float(w)
        cdf.append(acc / total)
    for idx, val in enumerate(cdf):
        if val >= q:
            if idx == 0:
                return float(centers[0])
            prev = cdf[idx - 1]
            denom = val - prev
            frac = 0.0 if denom <= 0.0 else (q - prev) / denom
            return float(centers[idx - 1] + frac * (centers[idx] - centers[idx - 1]))
    return float(centers[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fast-tail histogram (ParticleHistogram).")
    parser.add_argument("--hist", required=True, help="Path to EHist.txt")
    parser.add_argument("--metrics-out", required=True, help="Output metrics JSON")
    parser.add_argument("--warpx-args", default=None, help="Optional warpx_args.json")
    parser.add_argument("--particle-number", default=None, help="Optional PNUM.txt for N_weight_end")
    parser.add_argument("--ethr", type=float, default=1.0, help="Tail threshold for E (gamma-1).")
    parser.add_argument(
        "--ethr-list",
        default=None,
        help="Comma-separated tail thresholds for E (gamma-1), e.g. 0.02,0.05,0.08",
    )
    args = parser.parse_args()

    hist_path = Path(args.hist)
    metrics = {
        "fasttail_obs_kind": "histogram",
        "fasttail_hist_path": str(hist_path),
    }
    particle_number_end = None

    if not hist_path.exists():
        metrics["fasttail_error"] = "hist_path_missing"
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    hist_sha1 = sha1_file(hist_path)
    metrics["fasttail_hist_sha1"] = hist_sha1

    header, rows, lengths, has_nan = parse_histogram(hist_path)
    metrics["fasttail_hist_header"] = header[:3]
    metrics["fasttail_hist_series_len_raw"] = len(rows)
    metrics["fasttail_has_nan"] = bool(has_nan)

    if not rows:
        metrics["fasttail_error"] = "hist_no_rows"
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    length_mode = None
    if lengths:
        length_mode = Counter(lengths).most_common(1)[0][0]
    length_mode = length_mode or len(rows[0][2])
    metrics["fasttail_hist_bin_count_mode"] = int(length_mode)
    metrics["fasttail_hist_bin_count_min"] = int(min(lengths)) if lengths else None
    metrics["fasttail_hist_bin_count_max"] = int(max(lengths)) if lengths else None

    filtered = [row for row in rows if len(row[2]) == length_mode]
    skipped = len(rows) - len(filtered)
    metrics["fasttail_hist_rows_skipped"] = int(skipped)
    metrics["fasttail_hist_series_len"] = len(filtered)
    metrics["fasttail_rows_used"] = len(filtered)

    if not filtered:
        metrics["fasttail_error"] = "hist_all_rows_filtered"
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    if args.particle_number:
        pn_path = Path(args.particle_number)
        metrics["particle_number_path"] = str(pn_path)
        if pn_path.exists():
            metrics["particle_number_sha1"] = sha1_file(pn_path)
            steps_pn, times_pn, n_macro, n_weight = parse_particle_number(pn_path)
            if n_weight:
                particle_number_end = float(n_weight[-1])
                metrics["particle_number_end"] = particle_number_end
                metrics["particle_number_series_len"] = len(n_weight)
            if n_macro:
                metrics["particle_number_macro_end"] = float(n_macro[-1])
        else:
            metrics["particle_number_error"] = "particle_number_path_missing"

    steps = [row[0] for row in filtered]
    times = [row[1] for row in filtered]
    totals = [float(sum(row[2])) for row in filtered]
    first3, last3 = first_last_triplets(steps, times, totals)
    metrics["fasttail_hist_first3"] = first3
    metrics["fasttail_hist_last3"] = last3

    # Load bin configuration from warpx_args.json if provided.
    bin_min = None
    bin_max = None
    bin_number_cfg = None
    species = None
    if args.warpx_args:
        cfg = load_json(Path(args.warpx_args))
        if cfg:
            bin_min = cfg.get("energy_hist_bin_min")
            bin_max = cfg.get("energy_hist_bin_max")
            bin_number_cfg = cfg.get("energy_hist_bin_number")
            species = cfg.get("energy_hist_species")
            metrics["fasttail_hist_species"] = species
            metrics["fasttail_hist_bin_min"] = bin_min
            metrics["fasttail_hist_bin_max"] = bin_max
            metrics["fasttail_hist_bin_number_config"] = bin_number_cfg

    bin_number_used = length_mode
    if bin_number_cfg is not None:
        try:
            bin_number_cfg = int(bin_number_cfg)
            metrics["fasttail_hist_bin_number_config"] = bin_number_cfg
            if bin_number_cfg > 0:
                bin_number_used = bin_number_cfg
        except Exception:
            bin_number_cfg = None

    if bin_min is None:
        bin_min = 0.0
    if bin_max is None:
        bin_max = float(bin_number_used)

    if bin_number_used != length_mode:
        metrics["fasttail_hist_bin_number_used"] = int(length_mode)
        bin_number_used = length_mode
    metrics["fasttail_hist_bin_number_used"] = int(bin_number_used)

    if bin_number_used <= 0:
        metrics["fasttail_error"] = "invalid_bin_number"
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    centers = [
        bin_min + (idx + 0.5) * (bin_max - bin_min) / bin_number_used
        for idx in range(bin_number_used)
    ]

    last_counts = filtered[-1][2]
    if len(last_counts) != len(centers):
        metrics["fasttail_error"] = "bin_count_mismatch"
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        return

    total_weight = float(sum(last_counts))
    total_energy_weight = float(sum(c * w for c, w in zip(centers, last_counts)))
    metrics["fasttail_total_weight_end"] = total_weight
    metrics["fasttail_total_energy_weight_end"] = total_energy_weight
    metrics["fasttail_Wk_end"] = total_energy_weight
    metrics["fasttail_zero_weight_end"] = bool(total_weight <= 0.0)
    if total_weight > 0.0:
        mean_end = total_energy_weight / total_weight
        var_end = float(sum(w * (c - mean_end) ** 2 for c, w in zip(centers, last_counts))) / total_weight
        metrics["fasttail_E_mean_end"] = float(mean_end)
        metrics["fasttail_E_std_end"] = float(math.sqrt(max(var_end, 0.0)))
    else:
        metrics["fasttail_E_mean_end"] = None
        metrics["fasttail_E_std_end"] = None

    p50 = histogram_quantile(centers, last_counts, 0.5)
    p90 = histogram_quantile(centers, last_counts, 0.9)
    p99 = histogram_quantile(centers, last_counts, 0.99)
    metrics["fasttail_E_p50_end"] = p50
    metrics["fasttail_E_p90_end"] = p90
    metrics["fasttail_E_p99_end"] = p99

    p999 = histogram_quantile(centers, last_counts, 0.999)
    metrics["fasttail_E_p999_end"] = p999

    emax = None
    for center, count in zip(reversed(centers), reversed(last_counts)):
        if count > 0.0:
            emax = float(center)
            break
    metrics["fasttail_E_max_end"] = emax

    ethr_list = []
    if args.ethr_list:
        for item in str(args.ethr_list).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                ethr_list.append(float(item))
            except ValueError:
                continue
    if not ethr_list:
        ethr_list = [float(args.ethr)]

    metrics["fasttail_E_thr_list"] = ethr_list
    for idx, ethr in enumerate(ethr_list):
        tail_weights = [w for c, w in zip(centers, last_counts) if c > ethr]
        tail_energy = [c * w for c, w in zip(centers, last_counts) if c > ethr]
        tail_weight_sum = float(sum(tail_weights))
        tail_energy_sum = float(sum(tail_energy))
        # Suffix uses 2-decimal scale: 0.02 -> 002, 0.05 -> 005, 0.08 -> 008
        suffix = f"{int(round(ethr * 100)):03d}"
        metrics[f"fasttail_tail_frac_end_ethr{suffix}"] = (
            tail_weight_sum / total_weight if total_weight > 0.0 else None
        )
        metrics[f"fasttail_tail_energy_frac_end_ethr{suffix}"] = (
            tail_energy_sum / total_energy_weight if total_energy_weight > 0.0 else None
        )
        if idx == 0:
            metrics["fasttail_E_thr"] = ethr
            metrics["fasttail_tail_frac_end"] = (
                tail_weight_sum / total_weight if total_weight > 0.0 else None
            )
            metrics["fasttail_tail_energy_frac_end"] = (
                tail_energy_sum / total_energy_weight if total_energy_weight > 0.0 else None
            )

    metrics["fasttail_has_nan"] = bool(has_nan or series_has_nan(last_counts))

    if total_weight <= 0.0:
        metrics["fasttail_E_p50_end"] = None
        metrics["fasttail_E_p90_end"] = None
        metrics["fasttail_E_p99_end"] = None
        metrics["fasttail_E_p999_end"] = None
        metrics["fasttail_E_max_end"] = None

    gate_ok = True
    if metrics.get("fasttail_zero_weight_end") and particle_number_end is not None:
        gate_ok = not (particle_number_end > 0.0)
    metrics["fasttail_gate_ok"] = bool(gate_ok)

    Path(args.metrics_out).write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
