#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_u2(path: Path):
    rows = []
    has_nan = False
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') or line.startswith('step'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            try:
                step = int(float(parts[0]))
                time = float(parts[1])
                u2_mean = float(parts[2])
                u2_max = float(parts[3])
                sum_w = float(parts[4]) if len(parts) > 4 else None
                count = int(float(parts[5])) if len(parts) > 5 else None
            except Exception:
                continue
            if any(math.isnan(val) for val in (u2_mean, u2_max) if isinstance(val, float)):
                has_nan = True
            rows.append([step, time, u2_mean, u2_max, sum_w, count])
    return rows, has_nan


def dedup_rows_last_write(rows):
    if not rows:
        return [], 0
    step_jumps_negative = 0
    prev_step = None
    by_step = {}
    for row in rows:
        step = row[0]
        if prev_step is not None and step < prev_step:
            step_jumps_negative += 1
        prev_step = step
        by_step[step] = row
    unique_rows = [by_step[k] for k in sorted(by_step.keys())]
    return unique_rows, step_jumps_negative


def main():
    parser = argparse.ArgumentParser(description='Analyze U2 stats reduced diag')
    parser.add_argument('--u2', required=True, help='Path to U2.txt')
    parser.add_argument('--metrics-out', required=True, help='Output metrics json')
    parser.add_argument('--merge-in', help='Optional metrics json to merge into output')
    parser.add_argument('--runtime-guards', help='Optional metrics_runtime_guards.json')
    args = parser.parse_args()

    u2_path = Path(args.u2)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not u2_path.exists():
        metrics_path.write_text(
            json.dumps(
                {
                    "u2_path": str(u2_path),
                    "u2_sha1": None,
                    "u2_series_len": 0,
                    "u2_series_len_raw": 0,
                    "u2_series_len_unique": 0,
                    "u2_has_nan": True,
                    "u2_dedup_policy": "last_write_wins",
                    "u2_dedup_changed": False,
                    "u2_step_jumps_negative_count": 0,
                },
                indent=2,
            )
        )
        return

    rows, has_nan = parse_u2(u2_path)
    raw_len = len(rows)
    unique_rows, step_jumps_negative = dedup_rows_last_write(rows)
    unique_len = len(unique_rows)
    series_len = unique_len
    first3 = unique_rows[:3]
    last3 = unique_rows[-3:] if series_len >= 3 else unique_rows[:]
    u2_mean_end = unique_rows[-1][2] if series_len else None
    u2_max_end = unique_rows[-1][3] if series_len else None
    u2_sum_w_end = unique_rows[-1][4] if series_len else None
    u2_count_end = unique_rows[-1][5] if series_len else None
    u2_step_min = unique_rows[0][0] if series_len else None
    u2_step_max = unique_rows[-1][0] if series_len else None
    dedup_changed = raw_len != unique_len

    max_step = u2_step_max
    # Peak over full unique series
    u2_max_peak = None
    u2_max_peak_step = None
    u2_mean_peak = None
    u2_mean_peak_step = None
    u2_tail_proxy_peak_step = None
    u2_tail_proxy_end = None
    tail_proxy_alpha = 0.01
    if unique_rows:
        u2_max_peak = max(row[3] for row in unique_rows)
        idx_max = [row[3] for row in unique_rows].index(u2_max_peak)
        u2_max_peak_step = unique_rows[idx_max][0]
        u2_mean_peak = max(row[2] for row in unique_rows)
        idx_mean = [row[2] for row in unique_rows].index(u2_mean_peak)
        u2_mean_peak_step = unique_rows[idx_mean][0]
        tail_proxy = [row[2] + tail_proxy_alpha * (row[3] - row[2]) for row in unique_rows]
        if tail_proxy:
            max_tail = max(tail_proxy)
            idx_tail = tail_proxy.index(max_tail)
            u2_tail_proxy_peak_step = unique_rows[idx_tail][0]
            u2_tail_proxy_end = tail_proxy[-1]
    u2_peak_not_tail = None
    if u2_max_peak_step is not None and max_step is not None:
        u2_peak_not_tail = bool(u2_max_peak_step < (int(max_step) - 16))

    metrics = {
        "u2_path": str(u2_path),
        "u2_sha1": sha1_file(u2_path),
        "u2_series_len": series_len,
        "u2_series_len_raw": raw_len,
        "u2_series_len_unique": unique_len,
        "u2_has_nan": bool(has_nan),
        "u2_dedup_policy": "last_write_wins",
        "u2_dedup_changed": bool(dedup_changed),
        "u2_step_jumps_negative_count": int(step_jumps_negative),
        "u2_step_min": u2_step_min,
        "u2_step_max": u2_step_max,
        "u2_first3": first3,
        "u2_last3": last3,
        "u2_mean_end": u2_mean_end,
        "u2_max_end": u2_max_end,
        "u2_mean_peak": u2_mean_peak,
        "u2_mean_peak_step": u2_mean_peak_step,
        "u2_max_peak": u2_max_peak,
        "u2_max_peak_step": u2_max_peak_step,
        "u2_peak_not_tail": u2_peak_not_tail,
        "u2_tail_proxy_alpha": tail_proxy_alpha,
        "u2_tail_proxy_end": u2_tail_proxy_end,
        "u2_tail_proxy_peak_step": u2_tail_proxy_peak_step,
        "u2_sum_w_end": u2_sum_w_end,
        "u2_count_end": u2_count_end,
    }

    guard_path = None
    if args.runtime_guards:
        guard_path = Path(args.runtime_guards)
    else:
        candidate = metrics_path.parent / "metrics_runtime_guards.json"
        if candidate.exists():
            guard_path = candidate
    if guard_path and guard_path.exists():
        try:
            guard = json.loads(guard_path.read_text())
            direct_end = guard.get("u2_mean_end_direct")
            if direct_end is not None:
                metrics["u2_mean_end_old"] = u2_mean_end
                metrics["u2_mean_end"] = float(direct_end)
                metrics["u2_direct_weight_sum_end"] = guard.get("u2_direct_weight_sum_end")
                metrics["u2_direct_count_end"] = guard.get("u2_direct_count_end")
        except Exception:
            pass

    merge_in = args.merge_in
    if merge_in:
        try:
            merge_path = Path(merge_in)
            if merge_path.exists():
                merged = json.loads(merge_path.read_text())
                for key, value in merged.items():
                    if key not in metrics:
                        metrics[key] = value
        except Exception:
            pass

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
