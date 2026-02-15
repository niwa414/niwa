#!/usr/bin/env python3
"""Run one BO active-learning cycle: plan -> execute -> update dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one BO cycle: planner -> orchestrator -> dataset update")
    parser.add_argument(
        "--config",
        default="cases/helion-live-tilt-tradestudy/bo-config.json",
        help="Path to bo-config.json",
    )
    parser.add_argument("--mode", choices=["slurm", "local"], default="slurm", help="Orchestrator mode")
    parser.add_argument("--poll-interval-s", type=int, default=30)
    parser.add_argument("--force-stage", choices=["all", "run", "analyze"], default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override BO batch size")
    parser.add_argument("--plan-only", action="store_true", help="Only generate plan/cases; skip execution")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap import in dataset update")
    parser.add_argument("--batch-id", default=None, help="Optional batch id for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    cfg = load_json(config_path)
    if not cfg:
        raise SystemExit(f"Invalid config: {config_path}")

    batch_id = args.batch_id or utc_stamp()
    output_root = repo_root / str(cfg.get("output_root") or "outputs/bo/default")
    plan_path = output_root / "plans" / f"{batch_id}.plan.json"

    update_cmd = [
        sys.executable,
        "tools/bo_update_dataset.py",
        "--config",
        str(config_path),
    ]
    if not args.skip_bootstrap:
        update_cmd.append("--bootstrap")
    run_cmd(update_cmd, cwd=repo_root)

    plan_cmd = [
        sys.executable,
        "tools/bo_plan_next_cases.py",
        "--config",
        str(config_path),
        "--batch-id",
        batch_id,
        "--output-plan",
        str(plan_path),
    ]
    if args.batch_size is not None:
        plan_cmd.extend(["--batch-size", str(args.batch_size)])
    run_cmd(plan_cmd, cwd=repo_root)

    if args.plan_only:
        print(f"[bo-cycle] plan only done: {plan_path}")
        return

    run_id = f"bo-cycle-{batch_id}"
    orch_cmd = [
        sys.executable,
        "tools/sim_ops_orchestrator.py",
        "start",
        "--plan",
        str(plan_path),
        "--mode",
        args.mode,
        "--run-id",
        run_id,
        "--poll-interval-s",
        str(args.poll_interval_s),
    ]
    if args.force_stage:
        orch_cmd.extend(["--force-stage", args.force_stage])
    run_cmd(orch_cmd, cwd=repo_root)

    run_cmd(
        [
            sys.executable,
            "tools/bo_update_dataset.py",
            "--config",
            str(config_path),
            "--run-id",
            run_id,
        ],
        cwd=repo_root,
    )

    print(f"[bo-cycle] completed run_id={run_id}")
    print(f"[bo-cycle] plan={plan_path}")


if __name__ == "__main__":
    main()
