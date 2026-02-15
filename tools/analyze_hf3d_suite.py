#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_passfail(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    data = read_json(path)
    return {
        "exists": path.exists(),
        "case_id": case_id,
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {},
    }


def write_summary(path: Path, metrics: dict) -> None:
    lines = [
        "# HF3D Suite Summary",
        "",
        f"- merge_recapture_pass: `{metrics.get('merge_recapture_pass')}`",
        f"- formation_microinstability_pass: `{metrics.get('formation_microinstability_pass')}`",
        f"- engineering_diffusion_load_pass: `{metrics.get('engineering_diffusion_load_pass')}`",
        f"- all_hf3d_pass: `{metrics.get('all_hf3d_pass')}`",
        "",
        "This gate verifies that all three high-fidelity 3D animation capabilities are green.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate HF3D animation suite status.")
    parser.add_argument("--merge-case", default="m29-hf3d-1-merge-compression-recapture")
    parser.add_argument("--formation-case", default="m29-hf3d-2-formation-microinstability")
    parser.add_argument("--engineering-case", default="m29-hf3d-3-engineering-diffusion-load")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    merge = load_passfail(repo_root, args.merge_case)
    formation = load_passfail(repo_root, args.formation_case)
    engineering = load_passfail(repo_root, args.engineering_case)

    merge_pass = bool(merge["status"] == "PASS" and merge["metrics"].get("render_success") is True)
    formation_pass = bool(
        formation["status"] == "PASS" and formation["metrics"].get("render_success") is True
    )
    engineering_pass = bool(
        engineering["status"] == "PASS" and engineering["metrics"].get("render_success") is True
    )
    all_pass = bool(merge_pass and formation_pass and engineering_pass)

    metrics = {
        "merge_case_id": args.merge_case,
        "formation_case_id": args.formation_case,
        "engineering_case_id": args.engineering_case,
        "merge_recapture_pass": merge_pass,
        "formation_microinstability_pass": formation_pass,
        "engineering_diffusion_load_pass": engineering_pass,
        "all_hf3d_pass": all_pass,
    }

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_summary, metrics)


if __name__ == "__main__":
    main()
