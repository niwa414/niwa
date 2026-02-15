#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    denom = max(abs(a), 1.0e-30)
    return abs(b - a) / denom


def get_ions(meta: dict[str, Any], key: str) -> dict[str, Any]:
    block = meta.get(key)
    if not isinstance(block, dict):
        return {}
    ions = block.get("ions")
    if not isinstance(ions, dict):
        return {}
    return ions


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze restart consistency from WarpX metadata (pre/post).")
    ap.add_argument("--handoff-meta", required=True)
    ap.add_argument("--warpx-meta-before", required=True)
    ap.add_argument("--warpx-meta-after", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    handoff_path = Path(args.handoff_meta).resolve()
    pre_path = Path(args.warpx_meta_before).resolve()
    post_path = Path(args.warpx_meta_after).resolve()

    handoff = load_json(handoff_path)
    pre = load_json(pre_path)
    post = load_json(post_path)

    pre_init = get_ions(pre, "species_stats_init")
    pre_end = get_ions(pre, "species_stats")
    post_init = get_ions(post, "species_stats_init")
    post_end = get_ions(post, "species_stats")

    pre_weight_end = as_float(pre_end.get("weight_sum"))
    post_weight_init = as_float(post_init.get("weight_sum"))
    pre_charge_end = as_float(pre_end.get("charge_C"))
    post_charge_init = as_float(post_init.get("charge_C"))
    pre_energy_end = as_float(pre_end.get("energy_J"))
    post_energy_init = as_float(post_init.get("energy_J"))
    pre_particles_end = as_float(pre_end.get("num_particles"))
    post_particles_init = as_float(post_init.get("num_particles"))

    restart_weight_rel_jump = rel_diff(pre_weight_end, post_weight_init)
    restart_charge_rel_jump = rel_diff(pre_charge_end, post_charge_init)
    restart_energy_rel_jump = rel_diff(pre_energy_end, post_energy_init)
    restart_particle_rel_jump = rel_diff(pre_particles_end, post_particles_init)

    pre_completed = str(pre.get("run_status", "")).lower() == "completed"
    post_completed = str(post.get("run_status", "")).lower() == "completed"
    pre_last_step = int(pre.get("heartbeat_last_step") or -1)
    post_last_step = int(post.get("heartbeat_last_step") or -1)
    pre_monitor_records = int(pre.get("heartbeat_monitor_records") or 0)
    post_monitor_records = int(post.get("heartbeat_monitor_records") or 0)
    pre_drop_breach = bool(pre.get("heartbeat_monitor_drop_breach", True))
    post_drop_breach = bool(post.get("heartbeat_monitor_drop_breach", True))

    restart_sanity = post.get("restart_sanity") if isinstance(post.get("restart_sanity"), dict) else {}
    restart_sanity_efield_finite = bool(restart_sanity.get("Efield_fp_finite", False))

    handoff_present = bool(handoff)
    pre_meta_exists = bool(pre)
    post_meta_exists = bool(post)

    restart_consistency_pass = all(
        [
            handoff_present,
            pre_meta_exists,
            post_meta_exists,
            pre_completed,
            post_completed,
            pre_last_step >= 99,
            post_last_step >= 199,
            pre_monitor_records >= 100,
            post_monitor_records >= 200,
            not pre_drop_breach,
            not post_drop_breach,
            restart_sanity_efield_finite,
            restart_weight_rel_jump is not None and restart_weight_rel_jump <= 1.0e-12,
            restart_particle_rel_jump is not None and restart_particle_rel_jump <= 1.0e-12,
            restart_charge_rel_jump is not None and restart_charge_rel_jump <= 1.0e-12,
            restart_energy_rel_jump is not None and restart_energy_rel_jump <= 1.0e-10,
        ]
    )

    metrics = {
        "handoff_meta_present": handoff_present,
        "pre_meta_exists": pre_meta_exists,
        "post_meta_exists": post_meta_exists,
        "pre_completed": pre_completed,
        "post_completed": post_completed,
        "pre_last_step": pre_last_step,
        "post_last_step": post_last_step,
        "pre_monitor_records": pre_monitor_records,
        "post_monitor_records": post_monitor_records,
        "pre_drop_breach": pre_drop_breach,
        "post_drop_breach": post_drop_breach,
        "restart_sanity_efield_finite": restart_sanity_efield_finite,
        "restart_weight_rel_jump": restart_weight_rel_jump,
        "restart_particle_rel_jump": restart_particle_rel_jump,
        "restart_charge_rel_jump": restart_charge_rel_jump,
        "restart_energy_rel_jump": restart_energy_rel_jump,
        "restart_consistency_pass": restart_consistency_pass,
    }

    details = {
        "paths": {
            "handoff_meta": str(handoff_path),
            "warpx_meta_before": str(pre_path),
            "warpx_meta_after": str(post_path),
        },
        "pre_species_init": pre_init,
        "pre_species_end": pre_end,
        "post_species_init": post_init,
        "post_species_end": post_end,
        "restart_sanity": restart_sanity,
        "metrics": metrics,
    }

    metrics_path = Path(args.metrics).resolve()
    summary_path = Path(args.summary).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
