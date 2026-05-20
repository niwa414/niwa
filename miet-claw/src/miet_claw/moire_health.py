from __future__ import annotations

import re
from typing import Any, Dict


def parse_repo_kmc_run_output(run_text: str) -> Dict[str, Any]:
    loop_match = re.search(r"Loop time of ([0-9.eE+-]+) on", run_text)
    row_pattern = re.compile(
        r"^\s*([0-9.eE+-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+(-?[0-9.eE+-]+)",
        re.MULTILINE,
    )
    rows = row_pattern.findall(run_text)
    final_row = rows[-1] if rows else None
    return {
        "loop_time_seconds": float(loop_match.group(1)) if loop_match else None,
        "accepted_events": int(final_row[1]) if final_row else None,
        "rejected_events": int(final_row[2]) if final_row else None,
        "final_time": float(final_row[0]) if final_row else None,
        "final_energy": float(final_row[5]) if final_row else None,
        "has_loop_time": bool(loop_match),
        "has_final_stats_row": final_row is not None,
    }


def detect_repo_kmc_abnormal_markers(run_text: str) -> list[str]:
    markers = []
    patterns = [
        ("ERROR", r"\bERROR\b"),
        ("MPI_ABORT", r"\bMPI_ABORT\b"),
        ("segmentation fault", r"segmentation fault"),
        ("traceback", r"\bTraceback\b"),
        ("exception", r"\bException\b"),
    ]
    for label, pattern in patterns:
        if re.search(pattern, run_text, flags=re.IGNORECASE):
            markers.append(label)
    return markers


def build_repo_kmc_runtime_health(*, returncode: int, parsed: Dict[str, Any], run_text: str) -> tuple[str, Dict[str, Any]]:
    warnings = []
    abnormal_markers = detect_repo_kmc_abnormal_markers(run_text)
    if returncode != 0:
        warnings.append(f"repo misa-kmc returned non-zero code: {returncode}")
    if not parsed["has_loop_time"]:
        warnings.append("repo misa-kmc output did not contain a Loop time line.")
    if not parsed["has_final_stats_row"]:
        warnings.append("repo misa-kmc output did not contain a final stats row.")
    if parsed.get("final_time") is not None and float(parsed["final_time"]) <= 0:
        warnings.append("repo misa-kmc final simulated time was not positive.")
    if abnormal_markers:
        warnings.append(f"repo misa-kmc output contained abnormal marker(s): {', '.join(abnormal_markers)}")

    checks = {
        "returncode_ok": returncode == 0,
        "has_loop_time": parsed["has_loop_time"],
        "has_final_stats_row": parsed["has_final_stats_row"],
        "final_time_positive": parsed.get("final_time") is not None and float(parsed["final_time"]) > 0,
        "abnormal_output_detected": bool(abnormal_markers),
        "abnormal_markers": abnormal_markers,
    }
    status = (
        "completed"
        if checks["returncode_ok"]
        and checks["has_loop_time"]
        and checks["has_final_stats_row"]
        and checks["final_time_positive"]
        and not checks["abnormal_output_detected"]
        else "failed"
    )
    return status, {
        "status": "ok" if status == "completed" else "failed",
        "warnings": warnings,
        "checks": checks,
    }
