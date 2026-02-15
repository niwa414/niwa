#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


try:
    from build_evidence import parse_mapping as parse_mapping_from_evidence
except Exception:  # pragma: no cover - fallback parser below
    parse_mapping_from_evidence = None


DIMENSIONS = [
    {
        "id": "P0.1",
        "tier": "P0",
        "name": "End-to-end pulse chain",
        "requires": ["A2", "P18", "P2"],
        "optional": ["P8", "P9", "P15", "P19"],
        "target": "formation -> merge/compression -> expansion/recapture",
    },
    {
        "id": "P0.2",
        "tier": "P0",
        "name": "3D stability reproducibility",
        "requires": ["B2"],
        "optional": ["C3"],
        "target": "tilt/interchange/shearing sensitivity with parameter scans",
    },
    {
        "id": "P0.3",
        "tier": "P0",
        "name": "Circuit-plasma closed loop + load interface",
        "requires": ["A3", "P2", "D1"],
        "optional": ["A3.1"],
        "target": "closed-loop energy flow and magnetic-load outputs",
    },
    {
        "id": "P0.4",
        "tier": "P0",
        "name": "HPC production workflow",
        "requires": ["C3"],
        "optional": ["B2", "B5"],
        "target": "setup -> run -> postprocess regression reliability",
    },
    {
        "id": "P1.1",
        "tier": "P1",
        "name": "Electron energy/transport closure",
        "requires": ["B3"],
        "optional": ["P26"],
        "target": "solver-level electron transport closure",
    },
    {
        "id": "P1.2",
        "tier": "P1",
        "name": "Conductor response + structural interface",
        "requires": ["A4", "D1"],
        "optional": ["A4.2"],
        "target": "eddy-diffusion response plus load handoff",
    },
    {
        "id": "P1.3",
        "tier": "P1",
        "name": "V&V and diagnostics closure",
        "requires": ["C1", "C2", "C3"],
        "optional": ["B5"],
        "target": "benchmark + synthetic diagnostics + regression pack",
    },
    {
        "id": "P1.4",
        "tier": "P1",
        "name": "ML surrogate transport loop",
        "requires": ["ML1"],
        "optional": [],
        "target": "surrogate-assisted transport inference in production loop",
    },
]


ACTION_TEMPLATES = {
    "P0.1": [
        "Add a dedicated expansion-window case that records recapture chain metrics across drive envelope shutoff windows.",
        "Emit a single lifecycle JSON per shot (formation/merge/compression/expansion) to remove cross-file joins.",
        "Gate lifecycle continuity with explicit stage timestamps and minimum stage durations.",
    ],
    "P0.2": [
        "Promote 3D tilt scans from partial gate to full gate with at least one growth and one damped regime.",
        "Store sensitivity tables (seed/drift/forcing amplitude) as first-class regression artifacts.",
        "Add strict fail conditions for non-physical gamma sign flips in calibration windows.",
    ],
    "P0.3": [
        "Promote magnetic-load interface outputs into regular circuit gate runs, not only standalone postprocess.",
        "Add circuit/plasma co-sim checkpoints that verify energy closure at each major pulse phase.",
        "Export a fixed engineering handoff bundle (loads, dPhi/dt, eta terms) with schema checks.",
    ],
    "P0.4": [
        "Increase regression-pack breadth to include one end-to-end pulse case and one circuit/load case.",
        "Track runtime and resource envelopes per regression case for production readiness drift alerts.",
        "Add deterministic rerun checks (same inputs -> same PASS/FAIL and bounded metric drift).",
    ],
    "P1.1": [
        "Move electron-energy coupling from gate-only proxy to solver-level closure with explicit transport coefficients.",
        "Define calibration targets for transport closure against synthetic/experimental diagnostics.",
        "Add robustness sweeps for closure coefficients and report stability boundaries.",
    ],
    "P1.2": [
        "Upgrade wall/EB model from gate mask behavior to conductor-response metrics tied to eddy diffusion.",
        "Create a stable load handoff format for structural models (force/pressure/impulse over time).",
        "Add acceptance tests for load extraction consistency across geometry variants.",
    ],
    "P1.3": [
        "Expand benchmark package with at least one independent external reference per major subsystem.",
        "Version-lock synthetic diagnostic generation and include checksum-based reproducibility gates.",
        "Tie V&V acceptance to fixed report templates for cross-shot comparability.",
    ],
    "P1.4": [
        "Introduce an explicit surrogate model artifact (training data, model hash, inference config).",
        "Add online/offline error monitoring against baseline transport metrics.",
        "Define fallback behavior when surrogate confidence drops below threshold.",
    ],
}


COMMAND_HINTS = {
    "P0.1": [
        "python tools/run_case.py --case m13-a2-formation-translation-nonfast-lite --stage analyze",
        "python tools/run_case.py --case m17-b2-p18-off320 --stage analyze",
        "python tools/run_case.py --case m17-b2-circuit-mvp-mainline-Rload2000-coil4 --stage analyze",
    ],
    "P0.2": [
        "python tools/run_case.py --case m17-b2-tilt-seedON-driftON-rhocosE002-N008-mainline --stage analyze",
        "python tools/run_case.py --case m17-b2-tilt-seedOFF-driftOFF-rhocosE002-N008-mainline --stage analyze",
    ],
    "P0.3": [
        "python tools/run_case.py --case m5-a3-closed-loop-strong --stage analyze",
        "python tools/run_case.py --case m26-d2-magnetic-load-interface --stage all",
    ],
    "P0.4": [
        "python tools/run_case.py --case m24-c3-regression-pack --stage all",
    ],
    "P1.1": [
        "python tools/run_case.py --case m5-b3-1-electron-energy-coupled-sweep --stage analyze",
    ],
    "P1.2": [
        "python tools/run_case.py --case m15-a4-eb-movingwall-short --stage analyze",
        "python tools/run_case.py --case m26-d2-magnetic-load-interface --stage all",
    ],
    "P1.3": [
        "python tools/run_case.py --case m22-c2-synthetic-diagnostics-mvp --stage analyze",
        "python tools/run_case.py --case m24-c3-regression-pack --stage all",
    ],
    "P1.4": [
        "python tools/run_case.py --case m26-d3-helion-readiness-audit --stage all",
    ],
}


STATUS_SCORE = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}


def parse_mapping_fallback(path: Path) -> tuple[list[str], dict]:
    if not path.exists():
        return [], {}
    mapping = {}
    order = []
    current_key = None
    list_key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key = line.rstrip(":").strip()
            if not key:
                continue
            current_key = key
            order.append(key)
            mapping[current_key] = {"cases": []}
            list_key = None
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.endswith(":"):
            maybe_list = stripped[:-1].strip()
            if maybe_list in {"cases", "supplemental_cases", "pipeline_cases", "primary_cases"}:
                mapping[current_key].setdefault(maybe_list, [])
                list_key = maybe_list
                continue
            list_key = None
        if stripped.startswith("-") and list_key:
            case_id = stripped.lstrip("-").strip()
            if case_id:
                mapping[current_key][list_key].append(case_id)
            continue
        list_key = None
        if ":" in stripped:
            field, value = stripped.split(":", 1)
            field = field.strip()
            value = value.strip()
            if field in {"cases", "supplemental_cases", "pipeline_cases", "primary_cases"}:
                if value in {"[]", ""}:
                    mapping[current_key][field] = []
                    continue
            if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                value = value[1:-1]
            mapping[current_key][field] = value
    return order, mapping


def load_mapping(path: Path) -> tuple[list[str], dict]:
    if parse_mapping_from_evidence is not None:
        return parse_mapping_from_evidence(path)
    return parse_mapping_fallback(path)


def load_case_status(outputs_root: Path, case_id: str) -> dict:
    passfail_path = outputs_root / case_id / "analysis" / "PASSFAIL.json"
    record = {
        "case_id": case_id,
        "exists": passfail_path.exists(),
        "path": str(passfail_path),
        "status": "MISSING",
        "result": "MISSING",
        "case_status": None,
    }
    if not passfail_path.exists():
        return record
    try:
        data = json.loads(passfail_path.read_text(encoding="utf-8"))
    except Exception:
        record["status"] = "FAIL"
        record["result"] = "PARSE_ERROR"
        return record

    status = str(data.get("result") or data.get("status") or "UNKNOWN").upper()
    if status not in {"PASS", "FAIL"}:
        status = "UNKNOWN"
    record.update(
        {
            "status": status,
            "result": status,
            "case_status": data.get("case_status"),
            "gate_class": data.get("gate_class"),
            "physical_validity": data.get("physical_validity"),
        }
    )
    return record


def unique_preserve(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def evaluate_checklist_item(entry: dict, case_db: dict[str, dict]) -> dict:
    coverage = str(entry.get("coverage") or "full").strip().lower()
    title = str(entry.get("title") or "").strip()

    core_cases = unique_preserve(list(entry.get("primary_cases", [])) + list(entry.get("cases", [])))
    supplemental_cases = unique_preserve(list(entry.get("supplemental_cases", [])))
    pipeline_cases = unique_preserve(list(entry.get("pipeline_cases", [])))

    core_obs = [case_db.get(cid) or {"case_id": cid, "status": "MISSING", "exists": False} for cid in core_cases]
    supplemental_obs = [
        case_db.get(cid) or {"case_id": cid, "status": "MISSING", "exists": False}
        for cid in supplemental_cases
    ]
    pipeline_obs = [case_db.get(cid) or {"case_id": cid, "status": "MISSING", "exists": False} for cid in pipeline_cases]

    core_statuses = [obs["status"] for obs in core_obs]
    supplemental_statuses = [obs["status"] for obs in supplemental_obs]
    pipeline_statuses = [obs["status"] for obs in pipeline_obs]

    limited_coverage = coverage in {"gate_only", "partial", "pipeline_sanity", "supplemental"}

    status = "FAIL"
    reason = "no_evidence"

    if core_cases:
        if any(s == "FAIL" for s in core_statuses):
            status = "FAIL"
            reason = "core_case_fail"
        elif any(s in {"MISSING", "UNKNOWN"} for s in core_statuses):
            status = "FAIL"
            reason = "core_case_missing_or_unknown"
        elif limited_coverage:
            status = "PARTIAL"
            reason = "coverage_marked_partial"
        else:
            status = "PASS"
            reason = "core_cases_pass"
    else:
        aux_statuses = supplemental_statuses + pipeline_statuses
        if not aux_statuses:
            status = "FAIL"
            reason = "no_mapped_cases"
        elif any(s == "FAIL" for s in aux_statuses):
            status = "FAIL"
            reason = "supplemental_or_pipeline_fail"
        elif any(s == "PASS" for s in aux_statuses):
            status = "PARTIAL"
            reason = "supplemental_or_pipeline_only"
        else:
            status = "FAIL"
            reason = "supplemental_or_pipeline_missing"

    return {
        "title": title,
        "coverage": coverage,
        "status": status,
        "reason": reason,
        "core_cases": core_obs,
        "supplemental_cases": supplemental_obs,
        "pipeline_cases": pipeline_obs,
        "core_case_ids": core_cases,
        "supplemental_case_ids": supplemental_cases,
        "pipeline_case_ids": pipeline_cases,
    }


def evaluate_dimension(dimension: dict, checklist: dict[str, dict]) -> dict:
    required_keys = list(dimension.get("requires", []))
    optional_keys = list(dimension.get("optional", []))

    required = []
    optional = []

    for key in required_keys:
        item = checklist.get(key)
        if item is None:
            required.append({"key": key, "status": "FAIL", "reason": "checklist_key_missing"})
        else:
            required.append({"key": key, "status": item["status"], "reason": item["reason"]})

    for key in optional_keys:
        item = checklist.get(key)
        if item is None:
            optional.append({"key": key, "status": "FAIL", "reason": "checklist_key_missing"})
        else:
            optional.append({"key": key, "status": item["status"], "reason": item["reason"]})

    required_statuses = [row["status"] for row in required]

    if any(status == "FAIL" for status in required_statuses):
        status = "FAIL"
    elif required_statuses and all(status == "PASS" for status in required_statuses):
        status = "PASS"
    else:
        status = "PARTIAL"

    required_score = 0.0
    for row in required:
        required_score += STATUS_SCORE.get(row["status"], 0.0)
    required_score = required_score / max(len(required), 1)

    return {
        "id": dimension["id"],
        "tier": dimension["tier"],
        "name": dimension["name"],
        "target": dimension["target"],
        "status": status,
        "required": required,
        "optional": optional,
        "required_score": required_score,
    }


def summarize_tier(rows: list[dict]) -> dict:
    total = len(rows)
    pass_count = sum(1 for row in rows if row["status"] == "PASS")
    partial_count = sum(1 for row in rows if row["status"] == "PARTIAL")
    fail_count = sum(1 for row in rows if row["status"] == "FAIL")
    score = 0.0
    if total > 0:
        score = sum(STATUS_SCORE.get(row["status"], 0.0) for row in rows) / total
    return {
        "total": total,
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "score": score,
        "ready": pass_count == total,
    }


def render_readiness_md(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Helion Readiness Audit")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Summary")
    p0 = report["tier_summary"]["P0"]
    p1 = report["tier_summary"]["P1"]
    lines.append(f"- P0 ready: `{p0['ready']}` ({p0['pass']}/{p0['total']} PASS, score={p0['score']:.3f})")
    lines.append(f"- P1 ready: `{p1['ready']}` ({p1['pass']}/{p1['total']} PASS, score={p1['score']:.3f})")
    lines.append(f"- Overall score: `{report['overall_score']:.3f}`")
    lines.append(f"- Checklist items scanned: `{report['checklist_items']}`")
    lines.append(f"- Evidence cases scanned: `{report['evidence_cases_scanned']}`")
    lines.append("")
    lines.append("## Dimension Matrix")
    lines.append("| Dimension | Tier | Status | Required | Target |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["dimensions"]:
        req = ", ".join(f"{r['key']}={r['status']}" for r in row["required"])
        lines.append(
            f"| {row['id']} {row['name']} | {row['tier']} | {row['status']} | {req} | {row['target']} |"
        )
    lines.append("")
    lines.append("## Top Gaps")
    backlog = report.get("backlog", [])
    if not backlog:
        lines.append("- None")
    else:
        for item in backlog:
            lines.append(
                f"- {item['id']} ({item['tier']}): {item['status']} - blocking: {', '.join(item['blocking'])}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_backlog_md(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Helion Gap Backlog")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    backlog = report.get("backlog", [])
    if not backlog:
        lines.append("No open backlog items.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for idx, item in enumerate(backlog, start=1):
        lines.append(f"## {idx}. {item['id']} {item['name']}")
        lines.append(f"- tier: `{item['tier']}`")
        lines.append(f"- status: `{item['status']}`")
        lines.append(f"- blocking checklist: `{', '.join(item['blocking'])}`")
        lines.append("- actions:")
        for action in item.get("actions", []):
            lines.append(f"  - {action}")
        lines.append("- command_hints:")
        for cmd in item.get("command_hints", []):
            lines.append(f"  - `{cmd}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Helion-style P0/P1 readiness and gaps.")
    parser.add_argument("--mapping", default="evidence/pack/helion_mapping.yaml")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-backlog", required=True)
    parser.add_argument("--metrics-out", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mapping_path = Path(args.mapping)
    outputs_root = Path(args.outputs_root)
    if not mapping_path.is_absolute():
        mapping_path = (repo_root / mapping_path).resolve()
    if not outputs_root.is_absolute():
        outputs_root = (repo_root / outputs_root).resolve()

    order, mapping = load_mapping(mapping_path)

    known_cases = set()
    for entry in mapping.values():
        known_cases.update(entry.get("primary_cases", []))
        known_cases.update(entry.get("cases", []))
        known_cases.update(entry.get("supplemental_cases", []))
        known_cases.update(entry.get("pipeline_cases", []))

    case_db = {}
    for case_id in sorted(known_cases):
        case_db[case_id] = load_case_status(outputs_root, case_id)

    checklist = {}
    for key in order:
        checklist[key] = evaluate_checklist_item(mapping[key], case_db)

    dimensions = [evaluate_dimension(dim, checklist) for dim in DIMENSIONS]
    by_tier = {
        "P0": [row for row in dimensions if row["tier"] == "P0"],
        "P1": [row for row in dimensions if row["tier"] == "P1"],
    }
    tier_summary = {tier: summarize_tier(rows) for tier, rows in by_tier.items()}

    backlog = []
    for row in dimensions:
        if row["status"] == "PASS":
            continue
        blocking = [item["key"] for item in row["required"] if item["status"] != "PASS"]
        backlog.append(
            {
                "id": row["id"],
                "tier": row["tier"],
                "name": row["name"],
                "status": row["status"],
                "blocking": blocking,
                "actions": ACTION_TEMPLATES.get(row["id"], []),
                "command_hints": COMMAND_HINTS.get(row["id"], []),
            }
        )

    overall_score = 0.0
    if dimensions:
        overall_score = sum(STATUS_SCORE.get(row["status"], 0.0) for row in dimensions) / len(dimensions)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mapping_path": str(mapping_path),
        "outputs_root": str(outputs_root),
        "checklist_items": len(checklist),
        "evidence_cases_scanned": len(case_db),
        "checklist": checklist,
        "dimensions": dimensions,
        "tier_summary": tier_summary,
        "overall_score": overall_score,
        "backlog": backlog,
    }

    metrics = {
        "report_generated": True,
        "checklist_items": len(checklist),
        "evidence_cases_scanned": len(case_db),
        "dimensions_total": len(dimensions),
        "overall_score": overall_score,
        "p0_total": tier_summary["P0"]["total"],
        "p0_pass": tier_summary["P0"]["pass"],
        "p0_partial": tier_summary["P0"]["partial"],
        "p0_fail": tier_summary["P0"]["fail"],
        "p0_score": tier_summary["P0"]["score"],
        "p0_ready": tier_summary["P0"]["ready"],
        "p1_total": tier_summary["P1"]["total"],
        "p1_pass": tier_summary["P1"]["pass"],
        "p1_partial": tier_summary["P1"]["partial"],
        "p1_fail": tier_summary["P1"]["fail"],
        "p1_score": tier_summary["P1"]["score"],
        "p1_ready": tier_summary["P1"]["ready"],
        "top_priority_gap_count": sum(1 for item in backlog if item["tier"] == "P0"),
        "ml_surrogate_ready": any(
            row["id"] == "P1.4" and row["status"] == "PASS" for row in dimensions
        ),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_backlog = Path(args.out_backlog)
    out_metrics = Path(args.metrics_out)
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()
    if not out_md.is_absolute():
        out_md = (repo_root / out_md).resolve()
    if not out_backlog.is_absolute():
        out_backlog = (repo_root / out_backlog).resolve()
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    render_readiness_md(out_md, report)
    render_backlog_md(out_backlog, report)


if __name__ == "__main__":
    main()
