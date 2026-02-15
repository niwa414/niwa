#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def relpath(target: Path, base: Path, root: Path) -> str:
    if not target.is_absolute():
        target = (root / target).resolve()
    base = base.resolve()
    return os.path.relpath(target, base)


def format_metric(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


PRIORITY_METRICS = [
    "merge_time_frac",
    "merge_time",
    "merge_time_exists",
    "merge_detector_disagreement",
    "tilt_amp_ratio",
    "tilt_post_merge_samples",
    "tilt_post_merge_amp_max",
    "gamma_best",
    "r2_best",
    "compression_ratio",
    "sep_ratio",
    "centroid_amp_ratio",
    "mass_rel_drift",
    "total_energy_rel_drift",
    "divB_rel",
    "mag_energy_rel_diff",
    "mass_rel_diff",
    "particle_loss_frac",
    "num_outputs",
]


def summarize_metrics(metrics: dict) -> str:
    if not metrics:
        return "none"
    pairs = []
    for key in PRIORITY_METRICS:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        pairs.append(f"{key}={format_metric(value)}")
        if len(pairs) >= 5:
            break
    if not pairs:
        for key in sorted(metrics.keys()):
            value = metrics.get(key)
            if value is None or isinstance(value, (dict, list)):
                continue
            pairs.append(f"{key}={format_metric(value)}")
            if len(pairs) >= 5:
                break
    return "; ".join(pairs) if pairs else "none"


def parse_mapping(path: Path) -> tuple[list[str], dict]:
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
            key = stripped[:-1].strip()
            if key in {"cases", "supplemental_cases", "pipeline_cases", "primary_cases"}:
                mapping[current_key].setdefault(key, [])
                list_key = key
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
                    list_key = None
                    continue
            if value.startswith(("'", "\"")) and value.endswith(("'", "\"")) and len(value) >= 2:
                value = value[1:-1]
            mapping[current_key][field] = value
    return order, mapping


def case_limit_tags(case_info: dict) -> list[str]:
    tags = []
    gate_class = case_info.get("gate_class")
    physical = case_info.get("physical_validity")
    notes = case_info.get("case_notes")
    if gate_class:
        tags.append(str(gate_class))
    if physical:
        tags.append(str(physical))
    if notes:
        tags.append("notes")
    return tags


def build_mapping_section(entries, out_path: Path, root: Path) -> list[str]:
    lines = []
    mapping_path = root / "evidence" / "pack" / "helion_mapping.yaml"
    order, mapping = parse_mapping(mapping_path)
    if not mapping:
        return lines
    case_data = {data.get("case_id", path.parent.parent.name): data for path, data in entries}
    lines.append("## Helion Mapping (Checklist A/B/C)")
    def collect_infos(case_ids):
        infos = []
        missing = []
        for case_id in case_ids:
            if case_id == "evidence-pack":
                infos.append(
                    {
                        "case_id": case_id,
                        "status": "PASS",
                        "passfail": relpath(out_path, out_path.parent, root),
                        "gate_class": None,
                        "physical_validity": None,
                        "case_notes": None,
                    }
                )
                continue
            data = case_data.get(case_id)
            if data is None:
                missing.append(case_id)
                continue
            status = data.get("result") or data.get("status", "UNKNOWN")
            infos.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "passfail": relpath(
                        Path("outputs") / case_id / "analysis" / "PASSFAIL.json",
                        out_path.parent,
                        root,
                    ),
                    "gate_class": data.get("gate_class"),
                    "physical_validity": data.get("physical_validity"),
                    "case_notes": data.get("case_notes"),
                }
            )
        return infos, missing

    def format_case_links(infos):
        links = []
        for info in infos:
            links.append(f"[{info['case_id']}]({info['passfail']})")
        return ", ".join(links) if links else "-"

    for key in order:
        entry = mapping.get(key, {})
        title = entry.get("title", "").strip()
        primary_cases = entry.get("primary_cases", [])
        cases = entry.get("cases", [])
        supplemental_cases = entry.get("supplemental_cases", [])
        pipeline_cases = entry.get("pipeline_cases", [])
        coverage = entry.get("coverage", "").strip().lower()
        primary_infos, missing_cases = collect_infos(primary_cases)
        support_infos, missing_support = collect_infos(cases)
        supplemental_infos, missing_supp = collect_infos(supplemental_cases)
        pipeline_infos, missing_pipe = collect_infos(pipeline_cases)
        missing_cases.extend(missing_support)
        missing_cases.extend(missing_supp)
        missing_cases.extend(missing_pipe)
        status = "NOT IMPLEMENTED"
        rationale = "no mapped cases"
        core_infos = primary_infos + support_infos
        eval_infos = core_infos
        if eval_infos or pipeline_infos or supplemental_infos:
            statuses = [str(info["status"]).upper() for info in eval_infos]
            if any(s == "FAIL" for s in statuses):
                status = "FAIL"
                rationale = "at least one mapped case FAIL"
            elif not eval_infos and supplemental_infos:
                status = "PARTIAL"
                rationale = "supplemental evidence only"
            elif not eval_infos and pipeline_infos:
                status = "PARTIAL"
                rationale = "pipeline sanity only"
            else:
                limited_only = True
                for info in eval_infos:
                    gate = str(info.get("gate_class") or "").lower()
                    physical = str(info.get("physical_validity") or "").lower()
                    if gate != "pipeline_sanity" and physical != "limited":
                        limited_only = False
                        break
                if coverage in {"gate_only", "partial", "pipeline_sanity", "supplemental"} or limited_only:
                    status = "PARTIAL"
                    rationale = "mapped evidence is gate-only/limited"
                else:
                    status = "PASS"
                    rationale = "mapped cases PASS"
        rationale_override = entry.get("rationale_override", "").strip()
        if rationale_override:
            rationale = rationale_override
        if missing_cases:
            missing_str = ", ".join(missing_cases)
            rationale = f"{rationale}; missing: {missing_str}"
        cases_str = format_case_links(core_infos)
        primary_str = format_case_links(primary_infos)
        support_str = format_case_links(support_infos)
        supplemental_str = format_case_links(supplemental_infos)
        pipeline_str = format_case_links(pipeline_infos)
        limit_tags = set()
        for info in (core_infos + supplemental_infos + pipeline_infos):
            for tag in case_limit_tags(info):
                limit_tags.add(tag)
        limits_str = ", ".join(sorted(limit_tags)) if limit_tags else "-"
        title_str = f"{key} {title}".strip()
        coverage_str = f"coverage: {coverage}" if coverage else "coverage: full"
        if primary_cases:
            line = (
                f"- {title_str}: {status} ({coverage_str}) — cases (primary): {primary_str}"
                f" — cases (support): {support_str}"
            )
        else:
            line = f"- {title_str}: {status} ({coverage_str}) — cases: {cases_str}"
        if supplemental_cases:
            line += f" — supplemental: {supplemental_str}"
        if pipeline_cases:
            line += f" — pipeline: {pipeline_str}"
        line += f" — rationale: {rationale} — limits: {limits_str}"
        if key == "A4" and status != "NOT IMPLEMENTED":
            line += (
                " — note: Leak definition (A4): "
                "`leak_mass_frac_max = max_t((M_out(t) - M_out(t0)) / M_in(t0))`, "
                "where `M_out` is mass outside the EB wall mask."
            )
        lines.append(line)
    mapped_cases = set()
    for entry in mapping.values():
        for key_name in ("primary_cases", "cases", "supplemental_cases", "pipeline_cases"):
            mapped_cases.update(entry.get(key_name, []))
    unmapped = sorted(set(case_data.keys()) - mapped_cases)
    unmapped = [cid for cid in unmapped if cid != "evidence-pack"]
    filtered_unmapped = []
    for cid in unmapped:
        data = case_data.get(cid, {})
        status = str(data.get("result") or data.get("status", "")).upper()
        if status != "PASS":
            continue
        case_status = str(data.get("case_status") or "").lower()
        if case_status and case_status != "active":
            continue
        gate_class = str(data.get("gate_class") or "").lower()
        if gate_class == "pipeline_sanity":
            continue
        filtered_unmapped.append(cid)
    if filtered_unmapped:
        lines.append("")
        lines.append(
            f"- note: unmapped cases detected (update helion_mapping.yaml): {', '.join(filtered_unmapped)}"
        )
    lines.append("")
    return lines


def build_evidence_pack(entries, out_path: Path, root: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(entries)
    pass_count = 0
    fail_count = 0
    for _, data in entries:
        status = data.get("result") or data.get("status", "UNKNOWN")
        if str(status).upper() == "PASS":
            pass_count += 1
        elif str(status).upper() == "FAIL":
            fail_count += 1
    lines = []
    lines.append("# Evidence Pack: Helion Alignment")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(
        "This pack summarizes the current reproducible evidence against the public Helion checklist."
    )
    lines.append("For full artifact lists and historical failures, see `evidence/index.md`.")
    lines.append("")
    lines.append("## Status Summary")
    lines.append(f"- total_cases: {total}")
    lines.append(f"- pass: {pass_count}")
    lines.append(f"- fail: {fail_count}")
    lines.append("")
    lines.append("## Capability Matrix")
    lines.append("| Case | Status | Gate | Metrics | Evidence |")
    lines.append("| --- | --- | --- | --- | --- |")
    for path, data in sorted(entries, key=lambda item: item[1].get("case_id", "")):
        case_id = data.get("case_id", path.parent.parent.name)
        status = data.get("result") or data.get("status", "UNKNOWN")
        gate_class = data.get("gate_class") or "-"
        metrics = data.get("metrics", {})
        metric_summary = summarize_metrics(metrics)
        passfail_rel = relpath(path, out_path.parent, root)
        lines.append(
            f"| {case_id} | {status} | {gate_class} | {metric_summary} | `{passfail_rel}` |"
        )
    lines.append("")
    lines.append("## Reproduce (standard)")
    lines.append(
        "- `python tools/run_case.py --case <case_id> --stage all --update-evidence`"
    )
    lines.append("")
    mapping_lines = build_mapping_section(entries, out_path, root)
    if mapping_lines:
        lines.extend(mapping_lines)
    lines.append("## Known Limitations")
    lines.append(
        "- WarpX Hybrid-PIC in RZ supports m=0 only; 3D is required for tilt in Hybrid mode."
    )
    lines.append(
        "- Closed-loop circuit coupling (A3), electron-energy proxy coupling (B3), dynamic range sweeps (B4), and EB wall masking (A4) are MVP gate-only."
    )
    lines.append("")
    lines.append("## Traceability")
    lines.append("- Each case references `input_hash`, `run_env`, and artifacts in PASS/FAIL.")
    lines.append(
        "- Metric definitions (units/normalizations) are recorded in each case's "
        "`analysis/metrics.json` and `cases/<case_id>/README.md`."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence index from PASSFAIL.json files.")
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory containing case outputs.",
    )
    parser.add_argument(
        "--out",
        default="evidence/index.md",
        help="Output Markdown path for the evidence index.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    passfails = sorted(outputs_root.glob("*/analysis/PASSFAIL.json"))
    root = Path.cwd()
    entries = []
    for path in passfails:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            entries.append((path, data))
        except Exception:
            continue

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    active_entries = []
    archived_entries = []
    for path, data in entries:
        case_id = data.get("case_id", path.parent.parent.name)
        case_dir = root / "cases" / case_id
        status = data.get("case_status", "active")
        if status == "archived" or not case_dir.exists():
            archived_entries.append((path, data))
        else:
            active_entries.append((path, data))

    lines = []
    lines.append("# Evidence Index")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    if not entries:
        lines.append("No PASSFAIL.json files found.")
    else:
        for label, bucket in (("Active Cases", active_entries), ("Historical Failures", archived_entries)):
            if not bucket:
                continue
            lines.append(f"## {label}")
            lines.append("")
            for path, data in bucket:
                case_id = data.get("case_id", path.parent.parent.name)
                status = data.get("result") or data.get("status", "UNKNOWN")
                case_notes = data.get("case_notes")
                gate_class = data.get("gate_class")
                physical_validity = data.get("physical_validity")
                metrics = data.get("metrics", {})
                metric_pairs = []
                for key, value in metrics.items():
                    metric_pairs.append(f"{key}={value}")
                metrics_str = ", ".join(metric_pairs) if metric_pairs else "none"
                artifacts = data.get("artifacts", [])
                artifacts_rel = []
                for item in artifacts:
                    artifacts_rel.append(relpath(Path(item), out_path.parent, root))
                artifacts_str = ", ".join(artifacts_rel) if artifacts_rel else "none"
                passfail_rel = relpath(path, out_path.parent, root)
                lines.append(f"### {case_id}")
                lines.append(f"- status: {status}")
                lines.append(f"- passfail: {passfail_rel}")
                if case_notes:
                    lines.append(f"- notes: {case_notes}")
                if gate_class:
                    lines.append(f"- gate_class: {gate_class}")
                if physical_validity:
                    lines.append(f"- physical_validity: {physical_validity}")
                lines.append(f"- metrics: {metrics_str}")
                disagreement = metrics.get("merge_detector_disagreement")
                if isinstance(disagreement, str):
                    disagreement = disagreement.strip().lower() == "true"
                if disagreement is True:
                    delta = metrics.get("merge_time_delta_frac")
                    t_kmeans = metrics.get("merge_time_kmeans")
                    t_xsplit = metrics.get("merge_time_xsplit")
                    details = []
                    if delta is not None:
                        details.append(f"delta_frac={format_metric(delta)}")
                    if t_kmeans is not None:
                        details.append(f"t_kmeans={format_metric(t_kmeans)}")
                    if t_xsplit is not None:
                        details.append(f"t_xsplit={format_metric(t_xsplit)}")
                    suffix = ", ".join(details) if details else "delta_frac=unknown"
                    lines.append(
                        f"- note: merge detector disagreement (kmeans vs xsplit) {suffix}"
                    )
                lines.append(f"- artifacts: {artifacts_str}")
                lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[evidence] {out_path}")
    build_evidence_pack(active_entries, Path("evidence/pack/index.md"), root)


if __name__ == "__main__":
    main()
