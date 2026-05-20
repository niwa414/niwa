import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .local_profile import get_runtime_settings


DEFAULT_BRIDGE_SCRIPT = (
    "/path/to/run_lammps_kmc_bridge.py"
)


class BridgeError(RuntimeError):
    pass


ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _emit_progress(callback: Optional[ProgressCallback], stage: str, **payload: Any) -> None:
    if callback is None:
        return
    callback(stage, payload)


def _bridge_python() -> str:
    return os.environ.get("MIETCLAW_KMC_BRIDGE_PYTHON") or get_runtime_settings()["bridge_python"] or sys.executable


def _bridge_script() -> str:
    return os.environ.get("MIETCLAW_KMC_BRIDGE_SCRIPT") or get_runtime_settings()["bridge_script"] or DEFAULT_BRIDGE_SCRIPT


def run_kmc_lookup_bridge(
    *,
    event_json: str,
    workdir: str,
    neb_txt: Optional[str] = None,
    barrier: Optional[float] = None,
    validate: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if not neb_txt and barrier is None:
        raise BridgeError("Provide either neb_txt or barrier for the KMC bridge.")

    workdir_path = Path(workdir).expanduser().resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, "bridge.start", workdir=str(workdir_path), validate=validate)

    cmd = [
        _bridge_python(),
        _bridge_script(),
        "--event-json",
        str(Path(event_json).expanduser().resolve()),
        "--workdir",
        str(workdir_path),
    ]
    if neb_txt:
        cmd.extend(["--neb-txt", str(Path(neb_txt).expanduser().resolve())])
    else:
        cmd.extend(["--barrier", str(barrier)])
    if validate:
        cmd.append("--validate")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    _emit_progress(progress_callback, "bridge.command.complete", returncode=proc.returncode, workdir=str(workdir_path))
    summary_path = workdir_path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = _augment_bridge_summary(summary)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _emit_progress(
            progress_callback,
            "bridge.complete",
            workdir=str(workdir_path),
            validation_passed=summary.get("validation_passed"),
            safe_validation_passed=summary.get("safe_validation_passed"),
            health=(summary.get("runtime_health") or {}).get("status"),
        )
        return summary

    message = proc.stdout.strip() or f"bridge command failed with return code {proc.returncode}"
    raise BridgeError(message)


def _augment_bridge_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    files = summary.get("files") or {}
    validation = summary.get("validation") or {}
    warnings = []
    checks: Dict[str, Any] = {}

    run_out = files.get("run_out")
    run_out_path = Path(run_out).expanduser().resolve() if run_out else None
    if run_out_path and run_out_path.exists():
        text = run_out_path.read_text(encoding="utf-8", errors="replace")
        checks["run_out_exists"] = True
        checks["has_loop_time"] = "Loop time" in text
        checks["has_mpi_abort"] = "MPI_ABORT" in text
        checks["loaded_lookup_entries"] = "ML lookup entries" in text
        if not checks["has_loop_time"]:
            warnings.append("KMC run.out did not contain a Loop time line.")
        if checks["has_mpi_abort"]:
            warnings.append("KMC run.out contains MPI_ABORT.")
    else:
        checks["run_out_exists"] = False
        warnings.append("KMC run.out was not found.")

    returncode = summary.get("misa_kmc_returncode")
    checks["returncode_ok"] = returncode in {None, 0}
    if returncode not in {None, 0}:
        warnings.append(f"misa-kmc returned non-zero code: {returncode}")

    checks["lookup_hit_any"] = bool(validation.get("hit_any")) or int(validation.get("lookup_hits") or 0) > 0
    checks["lookup_loaded_any"] = bool(validation.get("loaded_any")) or bool(validation.get("loaded_entries_messages"))

    health = "ok"
    if not checks["returncode_ok"]:
        health = "failed"
    elif warnings:
        health = "warning"

    summary["runtime_health"] = {
        "status": health,
        "warnings": warnings,
        "checks": checks,
    }
    summary["safe_validation_passed"] = bool(summary.get("validation_passed")) and health == "ok"
    return summary
