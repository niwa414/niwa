#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


USE_GPU_AWARE_RE = re.compile(r"amrex\.use_gpu_aware_mpi.*\[\s*([01])\s*\]", re.IGNORECASE)
GPU_RUNTIME_HINT_PATTERNS = [
    re.compile(r"\bamrex\.use_gpu\s*=\s*1\b", re.IGNORECASE),
    re.compile(r"\bamrex\.use_gpu\b.*\[\s*1\s*\]", re.IGNORECASE),
    re.compile(r"\bgpu backend\b", re.IGNORECASE),
    re.compile(r"\bcuda runtime\b", re.IGNORECASE),
    re.compile(r"\bhip runtime\b", re.IGNORECASE),
    re.compile(r"\brocm runtime\b", re.IGNORECASE),
    re.compile(r"\bsycl runtime\b", re.IGNORECASE),
]
GPU_BACKEND_TAGS = ("cuda", "hip", "rocm", "sycl", "gpu")
GPU_BUILD_BACKENDS = {"CUDA", "HIP", "SYCL"}
COMPUTE_GPU_BACKENDS = {"CUDA", "HIP", "SYCL"}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_items(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_items(child, next_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            next_prefix = f"{prefix}[{idx}]"
            yield from iter_items(child, next_prefix)
    else:
        yield prefix, value


def scan_metadata(path: Path) -> dict:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "explicit_gpu_runtime": False,
        "backend_tags": [],
    }
    if not path.exists():
        return rec
    data = read_json(path)
    if not data:
        return rec

    backend_tags = []
    explicit = False
    for key, value in iter_items(data):
        key_l = key.lower()
        if isinstance(value, bool):
            if key_l.endswith("gpu_enabled") or key_l.endswith("use_gpu"):
                if value:
                    explicit = True
                    backend_tags.append(f"{key}=true")
            continue
        if not isinstance(value, str):
            continue
        value_l = value.lower().strip()
        if any(tag in value_l for tag in GPU_BACKEND_TAGS):
            if key_l.endswith("backend") or "runtime_backend" in key_l or "compute_backend" in key_l:
                explicit = True
                backend_tags.append(f"{key}={value}")
    rec["explicit_gpu_runtime"] = bool(explicit)
    rec["backend_tags"] = backend_tags
    return rec


def scan_log(path: Path) -> dict:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "line_count": 0,
        "has_amrex_init": False,
        "has_warpx_banner": False,
        "has_omp_init": False,
        "has_step_progress": False,
        "has_evolve_timing": False,
        "has_device_arena": False,
        "has_gpu_oom_guard": False,
        "has_gpu_aware_field": False,
        "use_gpu_aware_mpi_values": [],
        "explicit_gpu_runtime": False,
    }
    if not path.exists():
        return rec

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                rec["line_count"] += 1
                line = raw.strip()
                if not line:
                    continue
                if "Initializing AMReX" in line:
                    rec["has_amrex_init"] = True
                if line.startswith("WarpX (") or line == "WarpX":
                    rec["has_warpx_banner"] = True
                if "OMP initialized" in line:
                    rec["has_omp_init"] = True
                if line.startswith("STEP ") and " starts" in line:
                    rec["has_step_progress"] = True
                if "Evolve time =" in line:
                    rec["has_evolve_timing"] = True
                if "amrex.the_device_arena_init_size" in line:
                    rec["has_device_arena"] = True
                if "amrex.abort_on_out_of_gpu_memory" in line:
                    rec["has_gpu_oom_guard"] = True
                match = USE_GPU_AWARE_RE.search(line)
                if match:
                    rec["has_gpu_aware_field"] = True
                    try:
                        rec["use_gpu_aware_mpi_values"].append(int(match.group(1)))
                    except Exception:
                        pass
                if any(pattern.search(line) for pattern in GPU_RUNTIME_HINT_PATTERNS):
                    rec["explicit_gpu_runtime"] = True
    except Exception:
        pass

    return rec


def scan_build_cache(path: Path) -> dict:
    rec = {
        "path": str(path),
        "exists": path.exists(),
        "warpx_compute": None,
        "amrex_gpu_backend": None,
        "gpu_build_backend_detected": False,
    }
    if not path.exists():
        return rec

    warpx_compute = None
    amrex_gpu_backend = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("WarpX_COMPUTE:") and "=" in line:
                    warpx_compute = line.rsplit("=", 1)[-1].strip().upper()
                elif line.startswith("AMReX_GPU_BACKEND:") and "=" in line:
                    amrex_gpu_backend = line.rsplit("=", 1)[-1].strip().upper()
    except Exception:
        return rec

    gpu_build_backend_detected = bool(
        (warpx_compute in COMPUTE_GPU_BACKENDS) or (amrex_gpu_backend in GPU_BUILD_BACKENDS)
    )
    rec["warpx_compute"] = warpx_compute
    rec["amrex_gpu_backend"] = amrex_gpu_backend
    rec["gpu_build_backend_detected"] = gpu_build_backend_detected
    return rec


def write_log_matrix(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "path",
                "exists",
                "line_count",
                "amrex_init",
                "warpx_banner",
                "omp_init",
                "step_progress",
                "evolve_timing",
                "device_arena",
                "gpu_oom_guard",
                "gpu_aware_values",
                "explicit_gpu_runtime",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.get("path"),
                    row.get("exists"),
                    row.get("line_count"),
                    row.get("has_amrex_init"),
                    row.get("has_warpx_banner"),
                    row.get("has_omp_init"),
                    row.get("has_step_progress"),
                    row.get("has_evolve_timing"),
                    row.get("has_device_arena"),
                    row.get("has_gpu_oom_guard"),
                    ",".join(str(v) for v in row.get("use_gpu_aware_mpi_values", [])),
                    row.get("explicit_gpu_runtime"),
                ]
            )


def write_summary(path: Path, metrics: dict) -> None:
    lines = []
    lines.append("# Runtime Stack Evidence Summary")
    lines.append("")
    lines.append(f"- logs_scanned: `{metrics.get('logs_scanned')}`")
    lines.append(f"- metadata_scanned: `{metrics.get('metadata_scanned')}`")
    lines.append(f"- amrex_stack_present: `{metrics.get('amrex_stack_present')}`")
    lines.append(f"- runtime_progress_present: `{metrics.get('runtime_progress_present')}`")
    lines.append(f"- device_runtime_knobs_present: `{metrics.get('device_runtime_knobs_present')}`")
    lines.append(f"- gpu_aware_field_seen: `{metrics.get('gpu_aware_field_seen')}`")
    lines.append(f"- gpu_aware_nonzero_seen: `{metrics.get('gpu_aware_nonzero_seen')}`")
    lines.append(f"- explicit_gpu_runtime_detected: `{metrics.get('explicit_gpu_runtime_detected')}`")
    lines.append(f"- build_cache_scanned: `{metrics.get('build_cache_scanned')}`")
    lines.append(f"- gpu_build_backend_detected: `{metrics.get('gpu_build_backend_detected')}`")
    lines.append(f"- warpx_compute_modes: `{metrics.get('warpx_compute_modes')}`")
    lines.append(f"- amrex_gpu_backends: `{metrics.get('amrex_gpu_backends')}`")
    lines.append(f"- runtime_evidence_level: `{metrics.get('runtime_evidence_level')}`")
    lines.append(f"- runtime_stack_evidence_pass: `{metrics.get('runtime_stack_evidence_pass')}`")
    lines.append("")
    lines.append("This gate validates runtime-stack evidence from existing WarpX/AMReX logs and metadata.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate runtime-stack evidence (AMReX/WarpX/GPU knobs).")
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--metadata", nargs="*", default=[])
    parser.add_argument("--build-caches", nargs="*", default=[])
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--matrix", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    log_paths = []
    for entry in args.logs:
        path = Path(entry)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        log_paths.append(path)

    metadata_paths = []
    for entry in args.metadata:
        path = Path(entry)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        metadata_paths.append(path)

    build_cache_paths = []
    for entry in args.build_caches:
        path = Path(entry)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        build_cache_paths.append(path)

    log_rows = [scan_log(path) for path in log_paths]
    metadata_rows = [scan_metadata(path) for path in metadata_paths]
    build_cache_rows = [scan_build_cache(path) for path in build_cache_paths]

    amrex_stack_present = any(row.get("has_amrex_init") and row.get("has_warpx_banner") for row in log_rows)
    runtime_progress_present = any(row.get("has_step_progress") and row.get("has_evolve_timing") for row in log_rows)
    device_runtime_knobs_present = any(row.get("has_device_arena") and row.get("has_gpu_oom_guard") for row in log_rows)
    gpu_aware_field_seen = any(row.get("has_gpu_aware_field") for row in log_rows)
    gpu_aware_nonzero_seen = any(
        any(int(v) > 0 for v in row.get("use_gpu_aware_mpi_values", []))
        for row in log_rows
    )

    explicit_gpu_runtime_detected = any(row.get("explicit_gpu_runtime") for row in log_rows) or any(
        row.get("explicit_gpu_runtime") for row in metadata_rows
    )
    gpu_build_backend_detected = any(row.get("gpu_build_backend_detected") for row in build_cache_rows)
    warpx_compute_modes = sorted(
        {
            str(row.get("warpx_compute")).upper()
            for row in build_cache_rows
            if row.get("warpx_compute")
        }
    )
    amrex_gpu_backends = sorted(
        {
            str(row.get("amrex_gpu_backend")).upper()
            for row in build_cache_rows
            if row.get("amrex_gpu_backend")
        }
    )

    runtime_stack_evidence_pass = bool(
        amrex_stack_present
        and runtime_progress_present
        and device_runtime_knobs_present
        and gpu_aware_field_seen
    )

    if explicit_gpu_runtime_detected:
        runtime_evidence_level = "gpu_runtime"
    elif runtime_stack_evidence_pass and gpu_build_backend_detected:
        runtime_evidence_level = "gpu_build_only"
    elif runtime_stack_evidence_pass:
        runtime_evidence_level = "device_config_only"
    else:
        runtime_evidence_level = "insufficient"

    metrics = {
        "logs_scanned": len(log_rows),
        "metadata_scanned": len(metadata_rows),
        "build_cache_scanned": len(build_cache_rows),
        "amrex_stack_present": bool(amrex_stack_present),
        "runtime_progress_present": bool(runtime_progress_present),
        "device_runtime_knobs_present": bool(device_runtime_knobs_present),
        "gpu_aware_field_seen": bool(gpu_aware_field_seen),
        "gpu_aware_nonzero_seen": bool(gpu_aware_nonzero_seen),
        "explicit_gpu_runtime_detected": bool(explicit_gpu_runtime_detected),
        "gpu_build_backend_detected": bool(gpu_build_backend_detected),
        "warpx_compute_modes": warpx_compute_modes,
        "amrex_gpu_backends": amrex_gpu_backends,
        "runtime_evidence_level": runtime_evidence_level,
        "runtime_stack_evidence_pass": bool(runtime_stack_evidence_pass),
    }

    out_metrics = Path(args.metrics)
    out_summary = Path(args.summary)
    out_matrix = Path(args.matrix)
    if not out_metrics.is_absolute():
        out_metrics = (repo_root / out_metrics).resolve()
    if not out_summary.is_absolute():
        out_summary = (repo_root / out_summary).resolve()
    if not out_matrix.is_absolute():
        out_matrix = (repo_root / out_matrix).resolve()

    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(out_summary, metrics)
    write_log_matrix(out_matrix, log_rows)


if __name__ == "__main__":
    main()
