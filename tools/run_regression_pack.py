#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CASES = [
    "m15-a4-eb-movingwall-short",
    "m19-b2-3d-driver-stage1-gate60",
    "m20-b2-3d-driver-stage2-gate80-dense",
    "m21-b2-3d-driver-stage3-integration",
    "m25-b5-restart-metadata-consistency",
]


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=str(cwd), text=True, capture_output=True, check=False
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def load_passfail(root: Path, case_id: str) -> dict:
    path = root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and summarize a 5-case regression pack.")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument(
        "--restart-case",
        default="m25-b5-restart-metadata-consistency",
        help="Case id used as restart-consistency hard gate.",
    )
    ap.add_argument("--stage", choices=["run", "analyze", "all"], default="analyze")
    ap.add_argument("--update-evidence", action="store_true")
    ap.add_argument("--out-dir", default="outputs/stage3-regression-pack/analysis")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in args.cases:
        cmd = [sys.executable, str(root / "tools" / "run_case.py"), "--case", case_id, "--stage", args.stage]
        if args.update_evidence:
            cmd.append("--update-evidence")
        rc, out = run_cmd(cmd, root)
        pf = load_passfail(root, case_id)
        result = str(pf.get("result", "UNKNOWN")).upper()
        run_ok = rc == 0
        strict_pass = run_ok and result == "PASS"
        rows.append(
            {
                "case_id": case_id,
                "run_rc": rc,
                "run_ok": run_ok,
                "result": result,
                "pass": strict_pass,
                "stdout_tail": out.splitlines()[-30:],
                "passfail_path": str((root / "outputs" / case_id / "analysis" / "PASSFAIL.json").resolve()),
            }
        )

    pass_count = sum(1 for r in rows if r["pass"])
    total = len(rows)
    all_run_rc_zero = all(r["run_ok"] for r in rows)
    restart_case_id = args.restart_case
    restart_row = next((r for r in rows if r["case_id"] == restart_case_id), None)
    restart_consistency_pass = bool(restart_row and restart_row["pass"])
    regression_pack_pass = pass_count == total and restart_consistency_pass and all_run_rc_zero

    summary = {
        "cases": rows,
        "num_cases": total,
        "pass_count": pass_count,
        "fail_count": total - pass_count,
        "all_run_rc_zero": all_run_rc_zero,
        "restart_consistency_case": restart_case_id,
        "restart_consistency_pass": restart_consistency_pass,
        "regression_pack_pass": regression_pack_pass,
        "run_stage": args.stage,
    }
    summary_json = out_dir / "regression_pack_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_lines = []
    md_lines.append("# Stage3 Regression Pack")
    md_lines.append("")
    md_lines.append(f"- run_stage: `{args.stage}`")
    md_lines.append(f"- num_cases: `{total}`")
    md_lines.append(f"- pass_count: `{pass_count}`")
    md_lines.append(f"- all_run_rc_zero: `{all_run_rc_zero}`")
    md_lines.append(f"- restart_consistency_pass: `{restart_consistency_pass}`")
    md_lines.append(f"- regression_pack_pass: `{regression_pack_pass}`")
    md_lines.append("")
    md_lines.append("| Case | Result | Run RC | Strict PASS | PASSFAIL |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        md_lines.append(
            f"| {r['case_id']} | {r['result']} | {r['run_rc']} | {r['pass']} | `{Path(r['passfail_path']).as_posix()}` |"
        )
    (out_dir / "regression_pack_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
