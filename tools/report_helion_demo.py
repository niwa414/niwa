#!/usr/bin/env python3
"""Generate a Helion-style single-knob trade-study report from three case outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CaseRecord:
    label: str
    output_root: Path
    metrics_path: Path
    metrics: dict[str, Any]
    pass_gate: bool
    failed_checks: list[str]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_metrics_path(path: Path) -> tuple[Path, Path]:
    if path.is_file():
        return path.parent.parent if path.parent.name == "analysis" else path.parent, path
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    if path.is_dir():
        analysis_metrics = path / "analysis" / "metrics.json"
        if analysis_metrics.exists():
            return path, analysis_metrics
        direct_metrics = path / "metrics.json"
        if direct_metrics.exists():
            return path.parent, direct_metrics
    raise FileNotFoundError(f"metrics.json not found under: {path}")


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pct_delta(new: float | None, base: float | None) -> str:
    if new is None or base is None or base == 0.0:
        return "n/a"
    return f"{(new - base) / base * 100.0:+.2f}%"


def evaluate_case(
    metrics: dict[str, Any],
    compression_min: float,
    tilt_max: float,
) -> tuple[bool, list[str]]:
    checks: list[tuple[str, bool]] = [
        ("ran_to_completion", bool(metrics.get("ran_to_completion"))),
        ("no_nan_in_metrics", bool(metrics.get("no_nan_in_metrics"))),
        ("merge_time_exists", bool(metrics.get("merge_time_exists"))),
        (
            "compression_ratio",
            (as_float(metrics.get("compression_ratio")) or -1.0) >= compression_min,
        ),
        ("tilt_amp_max", (as_float(metrics.get("tilt_amp_max")) or 1.0e99) <= tilt_max),
        ("energy_accounting_ok", bool(metrics.get("energy_accounting_ok"))),
    ]
    failed = [name for name, ok in checks if not ok]
    return len(failed) == 0, failed


def pick_recommended(records: list[CaseRecord]) -> CaseRecord:
    passing = [rec for rec in records if rec.pass_gate]
    candidates = passing if passing else records

    def score(rec: CaseRecord) -> tuple:
        m = rec.metrics
        tilt = as_float(m.get("tilt_amp_max"))
        comp = as_float(m.get("compression_ratio"))
        merge_time = as_float(m.get("merge_time_s"))
        fail_count = len(rec.failed_checks)
        return (
            fail_count if not passing else 0,
            tilt if tilt is not None else 1.0e99,
            -(comp if comp is not None else -1.0e99),
            merge_time if merge_time is not None else 1.0e99,
        )

    return min(candidates, key=score)


def to_row(rec: CaseRecord) -> str:
    m = rec.metrics
    return (
        f"| {m.get('demo_case_label', rec.label)}"
        f" | {m.get('knob_name', 'shift')}={m.get('knob_value', 'n/a')}"
        f" | {m.get('ran_to_completion')}"
        f" | {m.get('no_nan_in_metrics')}"
        f" | {m.get('merge_time_exists')}"
        f" | {m.get('compression_ratio')}"
        f" | {m.get('tilt_amp_max')}"
        f" | {m.get('tilt_growth_rate')}"
        f" | {m.get('energy_residual_rel')}"
        f" | {m.get('energy_accounting_ok')}"
        f" | {'PASS' if rec.pass_gate else 'FAIL'} |"
    )


def recommendation_text(recommended: CaseRecord, baseline: CaseRecord) -> str:
    rec = recommended.metrics
    base = baseline.metrics
    rec_label = rec.get("demo_case_label", recommended.label)
    base_label = base.get("demo_case_label", baseline.label)
    tilt_delta = pct_delta(as_float(rec.get("tilt_amp_max")), as_float(base.get("tilt_amp_max")))
    comp_delta = pct_delta(
        as_float(rec.get("compression_ratio")), as_float(base.get("compression_ratio"))
    )
    merge_delta = pct_delta(as_float(rec.get("merge_time_s")), as_float(base.get("merge_time_s")))
    if recommended.output_root == baseline.output_root:
        return (
            f"推荐保持 `{base_label}`：在当前阈值下综合最优。"
            f" 相对其他旋钮方向未出现更优的 tilt/compression 组合。"
        )
    return (
        f"推荐 `{rec_label}`（{rec.get('knob_name', 'shift')}={rec.get('knob_value')}）。"
        f" 相对 `{base_label}`：`tilt_amp_max` {tilt_delta}，"
        f"`compression_ratio` {comp_delta}，`merge_time_s` {merge_delta}。"
    )


def engineering_actions_text(recommended: CaseRecord, baseline: CaseRecord) -> list[str]:
    rec = recommended.metrics
    base = baseline.metrics
    knob = str(rec.get("knob_name", "shift"))
    rec_v = as_float(rec.get("knob_value"))
    base_v = as_float(base.get("knob_value"))
    delta = None if rec_v is None or base_v is None else rec_v - base_v
    action = f"保持 `{knob}` 当前设定"
    if delta is not None:
        if delta > 0:
            action = f"`{knob}` 向正方向微调（建议 +{delta:.4f}）"
        elif delta < 0:
            action = f"`{knob}` 向负方向微调（建议 {delta:.4f}）"
    return [
        f"实验控制建议：{action}，并保持其余旋钮锁定，先做 3-5 发重复性确认。",
        "诊断看护建议：重点盯 `tilt_amp_max`、`compression_ratio`、`energy_residual_rel` 三个门禁量。",
        "线下改动顺序建议：先改时序/触发，再决定是否改电路或结构，避免多变量同时变化导致不可归因。",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report for Helion demo trade study.")
    parser.add_argument("baseline", help="Baseline output root or metrics.json path")
    parser.add_argument("knob_minus", help="Knob-minus output root or metrics.json path")
    parser.add_argument("knob_plus", help="Knob-plus output root or metrics.json path")
    parser.add_argument(
        "--output",
        default="outputs/helion-demo-tilt-tradestudy/report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--compression-min",
        type=float,
        default=1.001,
        help="PASS gate threshold for compression_ratio",
    )
    parser.add_argument(
        "--tilt-max",
        type=float,
        default=0.05,
        help="PASS gate threshold for tilt_amp_max",
    )
    args = parser.parse_args()

    inputs = [
        ("baseline", Path(args.baseline)),
        ("knob_minus", Path(args.knob_minus)),
        ("knob_plus", Path(args.knob_plus)),
    ]

    records: list[CaseRecord] = []
    for label, user_path in inputs:
        output_root, metrics_path = resolve_metrics_path(user_path)
        metrics = load_json(metrics_path)
        pass_gate, failed_checks = evaluate_case(metrics, args.compression_min, args.tilt_max)
        records.append(
            CaseRecord(
                label=label,
                output_root=output_root,
                metrics_path=metrics_path,
                metrics=metrics,
                pass_gate=pass_gate,
                failed_checks=failed_checks,
            )
        )

    baseline = records[0]
    recommended = pick_recommended(records)

    lines = [
        "# Helion Demo Tilt Trade Study Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Gate: `compression_ratio >= {args.compression_min}`, `tilt_amp_max <= {args.tilt_max}`",
        "",
        "## KPI 对比",
        "",
        "| case | knob | ran_to_completion | no_nan_in_metrics | merge_time_exists | compression_ratio | tilt_amp_max | tilt_growth_rate (1/s) | energy_residual_rel | energy_accounting_ok | gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(to_row(rec) for rec in records)
    lines.extend(
        [
            "",
            "## 自动推荐",
            "",
            recommendation_text(recommended, baseline),
            "",
            "## 线下调整建议",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in engineering_actions_text(recommended, baseline)])
    lines.extend(
        [
            "",
            "## Trade-off 说明",
            "",
            f"- baseline: `{baseline.output_root.as_posix()}`",
            f"- knob_minus: `{records[1].output_root.as_posix()}`",
            f"- knob_plus: `{records[2].output_root.as_posix()}`",
            "",
        ]
    )

    for rec in records:
        if rec.pass_gate:
            continue
        lines.append(
            f"- `{rec.metrics.get('demo_case_label', rec.label)}` failed checks: {', '.join(rec.failed_checks)}"
        )
    if all(rec.pass_gate for rec in records):
        lines.append("- 三个 case 全部通过门禁，推荐结果按最小 tilt + 保持压缩裕量排序给出。")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
