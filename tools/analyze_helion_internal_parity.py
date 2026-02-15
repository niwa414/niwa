#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_artifact(repo_root: Path, path_value: str | None, hash_value: str | None, label: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    path = resolve_path(repo_root, path_value)
    if path is None:
        reasons.append(f"{label}:path_missing")
        return False, reasons
    if not path.exists():
        reasons.append(f"{label}:path_not_found")
        return False, reasons
    if not hash_value:
        reasons.append(f"{label}:sha256_missing")
        return False, reasons
    try:
        actual = sha256_file(path)
    except Exception:
        reasons.append(f"{label}:sha256_compute_failed")
        return False, reasons
    if actual.lower() != str(hash_value).strip().lower():
        reasons.append(f"{label}:sha256_mismatch")
        return False, reasons
    return True, reasons


def validate_gpu_manifest(repo_root: Path, manifest: dict, schema_ok: bool) -> dict:
    section = manifest.get("gpu_runtime_proof")
    result = {"enabled": False, "bound": False, "reasons": []}
    reasons: list[str] = []
    if not schema_ok:
        reasons.append("manifest_schema_invalid")
        result["reasons"] = reasons
        return result
    if not isinstance(section, dict):
        reasons.append("gpu_runtime_proof:section_missing")
        result["reasons"] = reasons
        return result

    enabled = bool(section.get("enabled", False))
    result["enabled"] = enabled
    if not enabled:
        reasons.append("gpu_runtime_proof:disabled")
        result["reasons"] = reasons
        return result

    token_regex = section.get("runtime_token_regex")
    if not isinstance(token_regex, str) or not token_regex.strip():
        reasons.append("gpu_runtime_proof:runtime_token_regex_missing")
        result["reasons"] = reasons
        return result
    if re.search(r"(cuda|hip|rocm|sycl)", token_regex, re.IGNORECASE) is None:
        reasons.append("gpu_runtime_proof:runtime_token_regex_not_backend_specific")
        result["reasons"] = reasons
        return result

    ok_artifact, artifact_reasons = check_artifact(
        repo_root,
        section.get("source_path"),
        section.get("source_sha256"),
        "gpu_runtime_proof:source",
    )
    reasons.extend(artifact_reasons)
    if not ok_artifact:
        result["reasons"] = reasons
        return result

    source_path = resolve_path(repo_root, section.get("source_path"))
    assert source_path is not None
    min_match = int(section.get("min_match_count", 1))
    min_match = max(min_match, 1)
    count = 0
    try:
        pattern = re.compile(token_regex, re.IGNORECASE)
    except Exception:
        reasons.append("gpu_runtime_proof:runtime_token_regex_invalid")
        result["reasons"] = reasons
        return result

    try:
        with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if pattern.search(line):
                    count += 1
    except Exception:
        reasons.append("gpu_runtime_proof:source_read_failed")
        result["reasons"] = reasons
        return result

    if count < min_match:
        reasons.append("gpu_runtime_proof:match_count_too_low")
        result["reasons"] = reasons
        return result

    result["bound"] = True
    result["reasons"] = reasons
    return result


def validate_private_binding(
    repo_root: Path,
    manifest: dict,
    schema_ok: bool,
    section_key: str,
    artifact_specs: list[tuple[str, str, str]],
) -> dict:
    section = manifest.get(section_key)
    result = {"enabled": False, "bound": False, "reasons": []}
    reasons: list[str] = []
    if not schema_ok:
        reasons.append("manifest_schema_invalid")
        result["reasons"] = reasons
        return result
    if not isinstance(section, dict):
        reasons.append(f"{section_key}:section_missing")
        result["reasons"] = reasons
        return result

    enabled = bool(section.get("enabled", False))
    result["enabled"] = enabled
    if not enabled:
        reasons.append(f"{section_key}:disabled")
        result["reasons"] = reasons
        return result

    binding_id = str(section.get("binding_id") or "").strip()
    if not binding_id:
        reasons.append(f"{section_key}:binding_id_missing")
        result["reasons"] = reasons
        return result

    all_ok = True
    for path_key, hash_key, label in artifact_specs:
        ok_artifact, artifact_reasons = check_artifact(
            repo_root,
            section.get(path_key),
            section.get(hash_key),
            f"{section_key}:{label}",
        )
        reasons.extend(artifact_reasons)
        all_ok = all_ok and ok_artifact

    result["bound"] = bool(all_ok)
    result["reasons"] = reasons
    return result


def load_passfail(repo_root: Path, case_id: str) -> dict:
    path = repo_root / "outputs" / case_id / "analysis" / "PASSFAIL.json"
    data = read_json(path)
    return {
        "case_id": case_id,
        "exists": path.exists(),
        "status": str(data.get("result") or data.get("status") or "MISSING").upper(),
        "metrics": data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {},
    }


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# Helion Internal-Parity Progress Summary")
    lines.append("")
    lines.append(f"- strict_public_stack_pass: `{metrics.get('strict_public_stack_pass')}`")
    lines.append(f"- internal_parity_progress_pass: `{metrics.get('internal_parity_progress_pass')}`")
    lines.append(f"- internal_parity_claimable: `{metrics.get('internal_parity_claimable')}`")
    lines.append(f"- gpu_runtime_proven: `{metrics.get('gpu_runtime_proven')}`")
    lines.append(f"- gpu_runtime_from_runtime_logs: `{metrics.get('gpu_runtime_from_runtime_logs')}`")
    lines.append(f"- gpu_runtime_from_internal_manifest: `{metrics.get('gpu_runtime_from_internal_manifest')}`")
    lines.append(f"- private_shot_dataset_bound: `{metrics.get('private_shot_dataset_bound')}`")
    lines.append(f"- private_hardware_model_bound: `{metrics.get('private_hardware_model_bound')}`")
    lines.append(f"- internal_manifest_exists: `{metrics.get('internal_manifest_exists')}`")
    lines.append(f"- internal_manifest_schema_ok: `{metrics.get('internal_manifest_schema_ok')}`")
    lines.append(f"- internal_only_gap_count: `{metrics.get('internal_only_gap_count')}`")
    lines.append("")
    lines.append("## Runtime Stack Context")
    lines.append(f"- runtime_evidence_level: `{metrics.get('runtime_evidence_level')}`")
    lines.append(f"- runtime_gpu_build_backend_detected: `{metrics.get('runtime_gpu_build_backend_detected')}`")
    lines.append(f"- runtime_warpx_compute_modes: `{metrics.get('runtime_warpx_compute_modes')}`")
    lines.append(f"- runtime_amrex_gpu_backends: `{metrics.get('runtime_amrex_gpu_backends')}`")
    lines.append("")
    lines.append("## Internal-Only Gaps")
    gaps = metrics.get("internal_only_gaps") or []
    if gaps:
        for item in gaps:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Internal Manifest Reasons")
    lines.append(f"- gpu_runtime_manifest_reasons: `{metrics.get('gpu_runtime_manifest_reasons')}`")
    lines.append(f"- private_shot_manifest_reasons: `{metrics.get('private_shot_manifest_reasons')}`")
    lines.append(f"- private_hardware_manifest_reasons: `{metrics.get('private_hardware_manifest_reasons')}`")
    lines.append("")
    lines.append("This gate separates public-evidence strict coverage from internal-only parity blockers.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate strict public coverage and internal-only parity blockers.")
    parser.add_argument("--full-case", default="m27-d1-helion-full-requirements-gate")
    parser.add_argument("--usage-case", default="m27-d2-helion-usage-scenarios-gate")
    parser.add_argument("--runtime-case", default="m28-s1-runtime-stack-evidence")
    parser.add_argument("--stability-case", default="m28-s2-stability-nonproxy-gate")
    parser.add_argument("--em-case", default="m28-s3-em-diffusion-nonproxy-gate")
    parser.add_argument("--readiness-case", default="m26-d3-helion-readiness-audit")
    parser.add_argument("--internal-manifest", default="evidence/internal/helion_internal_parity_manifest.json")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    full = load_passfail(repo_root, args.full_case)
    usage = load_passfail(repo_root, args.usage_case)
    runtime = load_passfail(repo_root, args.runtime_case)
    stability = load_passfail(repo_root, args.stability_case)
    em = load_passfail(repo_root, args.em_case)
    readiness = load_passfail(repo_root, args.readiness_case)

    full_ok = full["status"] == "PASS" and full["metrics"].get("helion_full_requirements_pass") is True
    usage_ok = usage["status"] == "PASS" and usage["metrics"].get("all_usage_scenarios_pass") is True
    runtime_ok = runtime["status"] == "PASS" and runtime["metrics"].get("runtime_stack_evidence_pass") is True
    stability_ok = stability["status"] == "PASS" and stability["metrics"].get("stability_nonproxy_pass") is True
    em_ok = em["status"] == "PASS" and em["metrics"].get("em_diffusion_nonproxy_pass") is True
    readiness_ok = readiness["status"] == "PASS" and readiness["metrics"].get("overall_score") == 1.0

    strict_public_stack_pass = bool(
        full_ok and usage_ok and runtime_ok and stability_ok and em_ok and readiness_ok
    )

    internal_manifest_path = resolve_path(repo_root, args.internal_manifest)
    internal_manifest_exists = bool(internal_manifest_path and internal_manifest_path.exists())
    internal_manifest = read_json(internal_manifest_path) if internal_manifest_path else {}
    internal_manifest_schema_ok = bool(
        isinstance(internal_manifest, dict)
        and isinstance(internal_manifest.get("schema_version"), str)
        and internal_manifest.get("schema_version").strip()
    )

    runtime_metrics = runtime["metrics"]
    runtime_evidence_level = runtime_metrics.get("runtime_evidence_level")
    runtime_gpu_build_backend_detected = bool(runtime_metrics.get("gpu_build_backend_detected") is True)
    runtime_warpx_compute_modes = runtime_metrics.get("warpx_compute_modes") or []
    runtime_amrex_gpu_backends = runtime_metrics.get("amrex_gpu_backends") or []
    gpu_runtime_from_runtime_logs = bool(runtime_metrics.get("explicit_gpu_runtime_detected") is True)

    gpu_manifest = validate_gpu_manifest(repo_root, internal_manifest, internal_manifest_schema_ok)
    private_shot_manifest = validate_private_binding(
        repo_root,
        internal_manifest,
        internal_manifest_schema_ok,
        "private_shot_dataset_binding",
        [
            ("dataset_path", "dataset_sha256", "dataset"),
            ("calibration_report_path", "calibration_report_sha256", "calibration_report"),
        ],
    )
    private_hardware_manifest = validate_private_binding(
        repo_root,
        internal_manifest,
        internal_manifest_schema_ok,
        "private_hardware_model_binding",
        [
            ("model_path", "model_sha256", "model"),
            ("validation_report_path", "validation_report_sha256", "validation_report"),
        ],
    )

    gpu_runtime_from_internal_manifest = bool(gpu_manifest.get("bound") is True)
    gpu_runtime_proven = bool(gpu_runtime_from_runtime_logs or gpu_runtime_from_internal_manifest)
    private_shot_dataset_bound = bool(private_shot_manifest.get("bound") is True)
    private_hardware_model_bound = bool(private_hardware_manifest.get("bound") is True)

    internal_only_gaps = []
    if not gpu_runtime_proven:
        internal_only_gaps.append("gpu_runtime_proven")
    if not private_shot_dataset_bound:
        internal_only_gaps.append("private_shot_dataset_bound")
    if not private_hardware_model_bound:
        internal_only_gaps.append("private_hardware_model_bound")

    internal_only_gap_count = len(internal_only_gaps)
    internal_parity_claimable = bool(strict_public_stack_pass and internal_only_gap_count == 0)
    internal_parity_progress_pass = bool(strict_public_stack_pass)

    metrics = {
        "full_case_id": args.full_case,
        "usage_case_id": args.usage_case,
        "runtime_case_id": args.runtime_case,
        "stability_case_id": args.stability_case,
        "em_case_id": args.em_case,
        "readiness_case_id": args.readiness_case,
        "full_requirements_pass": bool(full_ok),
        "usage_scenarios_pass": bool(usage_ok),
        "runtime_stack_pass": bool(runtime_ok),
        "stability_nonproxy_pass": bool(stability_ok),
        "em_diffusion_nonproxy_pass": bool(em_ok),
        "readiness_pass": bool(readiness_ok),
        "strict_public_stack_pass": bool(strict_public_stack_pass),
        "internal_parity_progress_pass": bool(internal_parity_progress_pass),
        "internal_parity_claimable": bool(internal_parity_claimable),
        "internal_manifest_path": str(internal_manifest_path) if internal_manifest_path else None,
        "internal_manifest_exists": bool(internal_manifest_exists),
        "internal_manifest_schema_ok": bool(internal_manifest_schema_ok),
        "runtime_evidence_level": runtime_evidence_level,
        "runtime_gpu_build_backend_detected": bool(runtime_gpu_build_backend_detected),
        "runtime_warpx_compute_modes": runtime_warpx_compute_modes,
        "runtime_amrex_gpu_backends": runtime_amrex_gpu_backends,
        "gpu_runtime_from_runtime_logs": bool(gpu_runtime_from_runtime_logs),
        "gpu_runtime_from_internal_manifest": bool(gpu_runtime_from_internal_manifest),
        "gpu_runtime_manifest_reasons": gpu_manifest.get("reasons", []),
        "private_shot_manifest_reasons": private_shot_manifest.get("reasons", []),
        "private_hardware_manifest_reasons": private_hardware_manifest.get("reasons", []),
        "gpu_runtime_proven": bool(gpu_runtime_proven),
        "private_shot_dataset_bound": bool(private_shot_dataset_bound),
        "private_hardware_model_bound": bool(private_hardware_model_bound),
        "internal_only_gap_count": int(internal_only_gap_count),
        "internal_only_gaps": internal_only_gaps,
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
