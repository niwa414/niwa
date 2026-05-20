from __future__ import annotations

import csv
import json
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


RUN_KIND_WORKFLOW = "workflow"
RUN_KIND_MOIRE_CHAIN = "moire_lammps_to_kmc"
RUN_KIND_MOIRE_COMPARE = "moire_event_compare"
RUN_KIND_BRIDGE = "bridge_kmc_lookup"
RUN_KIND_SUMMARY = "summary_run"


def _read_summary_json(run_dir: Path) -> Dict[str, Any]:
    return _read_json(run_dir / "summary.json")


def _summary_run_kind(summary: Dict[str, Any]) -> Optional[str]:
    if not summary:
        return None
    if summary.get("mode") == RUN_KIND_MOIRE_COMPARE or "event_runs" in summary:
        return RUN_KIND_MOIRE_COMPARE
    if "kmc" in summary and "source_case_dir" in summary:
        return RUN_KIND_MOIRE_CHAIN
    if "files" in summary and "barrier_eV" in summary:
        return RUN_KIND_BRIDGE
    return RUN_KIND_SUMMARY


def _run_kind(run_dir: Path) -> Optional[str]:
    if (run_dir / "state.json").exists():
        return RUN_KIND_WORKFLOW
    return _summary_run_kind(_read_summary_json(run_dir))


def _run_candidates(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    return [path for path in output_dir.iterdir() if path.is_dir() and _run_kind(path)]


def _summary_run_mode(summary: Dict[str, Any]) -> Optional[str]:
    mode = summary.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    kind = _summary_run_kind(summary)
    if kind == RUN_KIND_MOIRE_COMPARE:
        return RUN_KIND_MOIRE_COMPARE
    if kind == RUN_KIND_MOIRE_CHAIN:
        return RUN_KIND_MOIRE_CHAIN
    if kind == RUN_KIND_BRIDGE:
        return RUN_KIND_BRIDGE
    if kind == RUN_KIND_SUMMARY:
        return RUN_KIND_SUMMARY
    return None


def _status_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"completed", "executed", "ok", "healthy", "passed", "success"}:
        return "completed"
    if text in {"failed", "error", "aborted"}:
        return "failed"
    if text:
        return "running"
    return "unknown"


def _summary_runtime_health(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("runtime_health") or (summary.get("kmc") or {}).get("runtime_health") or {}


def _summary_material_name(summary: Dict[str, Any], run_dir: Path) -> str:
    source_case_dir = summary.get("source_case_dir")
    if isinstance(source_case_dir, str) and source_case_dir.strip():
        case_path = Path(source_case_dir)
        tail = "/".join(case_path.parts[-3:]) if len(case_path.parts) >= 3 else case_path.name
        return f"MoRe case · {tail}"
    case_dir = summary.get("case_dir")
    if _summary_run_kind(summary) == RUN_KIND_MOIRE_COMPARE and isinstance(case_dir, str) and case_dir.strip():
        case_path = Path(case_dir)
        tail = "/".join(case_path.parts[-3:]) if len(case_path.parts) >= 3 else case_path.name
        return f"MoRe compare · {tail}"
    if summary.get("files"):
        return "KMC bridge"
    return run_dir.name


def _summary_barrier_events(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    if _summary_run_kind(summary) == RUN_KIND_MOIRE_COMPARE:
        events: List[Dict[str, Any]] = []
        for item in summary.get("barrier_ranking") or []:
            barrier = _to_float(item.get("barrier_eV"))
            if barrier is None:
                continue
            events.append(
                {
                    "species": str(item.get("label") or Path(str(item.get("event_json") or "event")).stem),
                    "barrier_ev": barrier,
                    "barrier_source": "lammps-neb",
                }
            )
        return events
    kmc_payload = summary.get("kmc") or summary
    assignment = kmc_payload.get("barrier_assignment") or {}
    events: List[Dict[str, Any]] = []
    for species, value in assignment.items():
        if species == "note":
            continue
        barrier = _to_float(value)
        if barrier is None:
            continue
        events.append(
            {
                "species": str(species),
                "barrier_ev": barrier,
                "barrier_source": "lammps-neb",
            }
        )
    if events:
        return events
    barrier = _to_float(kmc_payload.get("barrier_eV"))
    if barrier is None:
        barrier = _to_float(summary.get("barrier_eV"))
    if barrier is not None:
        return [{"species": "shared", "barrier_ev": barrier, "barrier_source": "lammps-neb"}]
    return []


def _summary_step_statuses(summary: Dict[str, Any]) -> Dict[str, Any]:
    kind = _summary_run_kind(summary)
    if kind == RUN_KIND_MOIRE_COMPARE:
        event_runs = summary.get("event_runs") or []
        return {str(item.get("label") or f"event_{index+1}"): item.get("status") for index, item in enumerate(event_runs)}
    if kind == RUN_KIND_MOIRE_CHAIN:
        kmc = summary.get("kmc") or {}
        return {
            "lammps": (summary.get("lammps") or {}).get("status"),
            "postprocess": (summary.get("postprocess") or {}).get("status"),
            "kmc": kmc.get("status"),
        }
    if kind == RUN_KIND_BRIDGE:
        statuses: Dict[str, Any] = {"bridge": summary.get("status") or _summary_runtime_health(summary).get("status")}
        if summary.get("validation_passed") is not None:
            statuses["validation"] = "completed" if summary.get("validation_passed") else "failed"
        return statuses
    status = summary.get("status") or _summary_runtime_health(summary).get("status")
    return {"summary": status} if status else {}


def _summary_run_status(summary: Dict[str, Any]) -> str:
    step_statuses = _summary_step_statuses(summary)
    if any(_status_bucket(value) == "failed" for value in step_statuses.values()):
        return "failed"
    runtime_status = _status_bucket(_summary_runtime_health(summary).get("status"))
    if runtime_status == "failed":
        return "failed"
    status_text = str(summary.get("status") or "").strip().lower()
    if status_text in {"completed", "ok", "success"}:
        return "completed"
    if status_text in {"failed", "error"}:
        return "failed"
    if runtime_status == "completed" and step_statuses:
        return "completed"
    if any(_status_bucket(value) == "completed" for value in step_statuses.values()):
        return "running"
    return status_text or "unknown"


def _summary_completed_steps(summary: Dict[str, Any]) -> Tuple[int, int]:
    step_statuses = _summary_step_statuses(summary)
    total = len(step_statuses)
    completed = sum(1 for value in step_statuses.values() if _status_bucket(value) == "completed")
    if total == 0:
        return 0, 0
    return completed, total


def _summary_log_candidates(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Path]:
    kind = _summary_run_kind(summary)
    if kind == RUN_KIND_MOIRE_COMPARE:
        summary_path = run_dir / "summary.json"
        return {"md": summary_path, "kmc": summary_path, "summary": summary_path}
    if kind == RUN_KIND_MOIRE_CHAIN:
        kmc = summary.get("kmc") or {}
        files = kmc.get("files") or {}
        md_log = (summary.get("lammps") or {}).get("log")
        kmc_log = files.get("run_out")
        return {
            "md": Path(md_log).expanduser() if md_log else run_dir / "lammps_run.out",
            "kmc": Path(kmc_log).expanduser() if kmc_log else run_dir / "kmc_bridge" / "run.out",
            "summary": run_dir / "summary.json",
        }
    if kind == RUN_KIND_BRIDGE:
        files = summary.get("files") or {}
        kmc_log = files.get("run_out")
        md_log = files.get("barriers_tsv")
        return {
            "md": Path(md_log).expanduser() if md_log else run_dir / "summary.json",
            "kmc": Path(kmc_log).expanduser() if kmc_log else run_dir / "run.out",
            "summary": run_dir / "summary.json",
        }
    summary_path = run_dir / "summary.json"
    return {"md": summary_path, "kmc": summary_path, "summary": summary_path}


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_diffusion_from_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    diffusion_path = run_dir / "artifacts" / "kmc" / "diffusion.csv"
    rows = _read_csv_rows(diffusion_path)
    if not rows:
        return None
    latest = rows[-1]
    return {
        "path": str(diffusion_path),
        "jumps": _to_float(latest.get("jumps")),
        "msd": _to_float(latest.get("msd")),
        "simulation_time": _to_float(latest.get("simulation_time")),
        "jump_frequency": _to_float(latest.get("jump frequency")),
        "diffusion_coefficient": _to_float(latest.get("diffusion coefficient")),
    }


def _workflow_execution_provenance(run_dir: Path) -> Dict[str, Any]:
    md_execution = _read_json(run_dir / "artifacts" / "md" / "md_execution.json")
    kmc_execution = _read_json(run_dir / "artifacts" / "kmc" / "kmc_execution.json")
    stages: Dict[str, Dict[str, Any]] = {}
    if md_execution:
        stages["md"] = {
            "mode": md_execution.get("mode"),
            "reason": md_execution.get("reason"),
            "command": md_execution.get("command"),
        }
    if kmc_execution:
        stages["kmc"] = {
            "mode": kmc_execution.get("mode"),
            "reason": kmc_execution.get("reason"),
            "diffusion_mode": kmc_execution.get("diffusion_mode"),
            "command": kmc_execution.get("command"),
        }
    simulated_modes = {"dry-run", "simulated"}
    has_simulated_outputs = any(
        (stage.get("mode") in simulated_modes) or (stage.get("diffusion_mode") == "simulated")
        for stage in stages.values()
    )
    return {
        "stages": stages,
        "has_simulated_outputs": has_simulated_outputs,
        "label": "simulated/dry-run outputs present" if has_simulated_outputs else "real/file-backed execution",
    }


def _shorten(text: str, width: int = 120) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= width:
        return clean
    return clean[: width - 3] + "..."


def _wrap_lines(text: str, width: int) -> List[str]:
    width = max(8, width)
    lines: List[str] = []
    for raw in text.splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        wrapped = textwrap.wrap(raw, width=width, replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def _resolve_run_dir(output_dir: Path, target: Optional[str]) -> Optional[Path]:
    if not target:
        return None
    candidate = Path(target).expanduser()
    if candidate.exists():
        return candidate.resolve()
    nested = (output_dir / target).resolve()
    if nested.exists():
        return nested
    return None


def _read_run_mode(run_dir: Path) -> Optional[str]:
    if (run_dir / "job_spec.resolved.json").exists():
        job_spec = _read_json(run_dir / "job_spec.resolved.json")
        return job_spec.get("mode")
    return _summary_run_mode(_read_summary_json(run_dir))


def list_runs(output_dir: Path, limit: int = 12) -> List[Dict[str, Any]]:
    if not output_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    candidates = _run_candidates(output_dir)
    for run_dir in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        kind = _run_kind(run_dir)
        if kind == RUN_KIND_WORKFLOW:
            state = _read_json(run_dir / "state.json")
            job_spec = _read_json(run_dir / "job_spec.resolved.json")
            steps = state.get("steps", {})
            completed = sum(1 for step in steps.values() if step.get("status") == "completed")
            failed = any(step.get("status") == "failed" for step in steps.values())
            if failed:
                status = "failed"
            elif steps and completed == len(steps):
                status = "completed"
            else:
                status = state.get("status", "running" if steps else "unknown")
            mode = job_spec.get("mode")
            material_name = job_spec.get("material_system", {}).get("name", run_dir.name)
        else:
            summary = _read_summary_json(run_dir)
            completed, total_steps = _summary_completed_steps(summary)
            status = _summary_run_status(summary)
            mode = _summary_run_mode(summary)
            material_name = _summary_material_name(summary, run_dir)
        items.append(
            {
                "job_id": run_dir.name,
                "path": str(run_dir),
                "kind": kind,
                "mode": mode,
                "material_name": material_name,
                "status": status,
                "completed_steps": completed,
                "total_steps": len(steps) if kind == RUN_KIND_WORKFLOW else total_steps,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_dir.stat().st_mtime)),
            }
        )
        if len(items) >= limit:
            break
    return items


def list_artifacts(run_dir: Path, limit: int = 80) -> List[str]:
    artifact_root = run_dir / "artifacts"
    if not artifact_root.exists():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return []
        files = [str(path.relative_to(run_dir)) for path in sorted(run_dir.rglob("*")) if path.is_file()]
        return files[:limit]
    files = [str(path.relative_to(run_dir)) for path in sorted(artifact_root.rglob("*")) if path.is_file()]
    return files[:limit]


def get_log_excerpt(run_dir: Path, target: str = "auto", max_lines: int = 60) -> Dict[str, Any]:
    if _run_kind(run_dir) == RUN_KIND_WORKFLOW:
        candidates = {
            "md": run_dir / "artifacts" / "md" / "md_execution.log",
            "kmc": run_dir / "artifacts" / "kmc" / "log.spparks",
            "summary": run_dir / "explain" / "summary.md",
        }
    else:
        candidates = _summary_log_candidates(run_dir, _read_summary_json(run_dir))
    chosen_key = target
    if chosen_key == "auto":
        for key in ["md", "kmc", "summary"]:
            if candidates[key].exists():
                chosen_key = key
                break
        else:
            chosen_key = "summary"
    path = candidates.get(chosen_key)
    if not path or not path.exists():
        return {"target": chosen_key, "path": str(path) if path else None, "content": "", "available": False}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    excerpt = "\n".join(lines[-max_lines:])
    return {"target": chosen_key, "path": str(path), "content": excerpt, "available": True}


def inspect_run(run_dir: Path) -> Dict[str, Any]:
    kind = _run_kind(run_dir)
    if kind == RUN_KIND_WORKFLOW:
        state = _read_json(run_dir / "state.json")
        job_spec = _read_json(run_dir / "job_spec.resolved.json")
        barriers = _read_json(run_dir / "artifacts" / "md" / "barriers.json")
        summary_path = run_dir / "explain" / "summary.md"
        summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
        steps = state.get("steps", {})
        latest_diffusion = _latest_diffusion_from_run(run_dir)
        failed = any(value.get("status") == "failed" for value in steps.values())
        status = "failed" if failed else state.get("status", "completed" if steps else "unknown")
        job_payload = state.get("job") if isinstance(state.get("job"), dict) else {}
        checkpoints = state.get("checkpoints") if isinstance(state.get("checkpoints"), list) else []
        execution_provenance = _workflow_execution_provenance(run_dir)
        return {
            "job_id": run_dir.name,
            "kind": kind,
            "status": status,
            "path": str(run_dir),
            "mode": job_spec.get("mode"),
            "material_name": job_spec.get("material_system", {}).get("name", run_dir.name),
            "temperature_k": job_spec.get("kmc", {}).get("temperature_k"),
            "step_statuses": {key: value.get("status") for key, value in steps.items()},
            "barrier_source_mode": barriers.get("metadata", {}).get("barrier_source_mode"),
            "workflow_kind": barriers.get("metadata", {}).get("workflow_kind"),
            "neb_images": barriers.get("metadata", {}).get("neb_images"),
            "events": barriers.get("events", []),
            "latest_diffusion": latest_diffusion,
            "summary": summary,
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "artifacts": list_artifacts(run_dir),
            "md_log_path": str(run_dir / "artifacts" / "md" / "md_execution.log"),
            "kmc_log_path": str(run_dir / "artifacts" / "kmc" / "log.spparks"),
            "resume_summary": job_payload.get("resume_summary"),
            "recovery_plan": job_payload.get("recovery_plan"),
            "checkpoint_count": len(checkpoints),
            "execution_provenance": execution_provenance,
            "has_simulated_outputs": execution_provenance["has_simulated_outputs"],
        }

    summary = _read_summary_json(run_dir)
    log_candidates = _summary_log_candidates(run_dir, summary)
    runtime_health = _summary_runtime_health(summary)
    info: Dict[str, Any] = {
        "job_id": run_dir.name,
        "kind": kind,
        "status": _summary_run_status(summary),
        "path": str(run_dir),
        "mode": _summary_run_mode(summary),
        "material_name": _summary_material_name(summary, run_dir),
        "temperature_k": None,
        "step_statuses": _summary_step_statuses(summary),
        "barrier_source_mode": "lammps-neb" if kind == RUN_KIND_MOIRE_CHAIN else "lookup-bridge",
        "workflow_kind": kind,
        "neb_images": None,
        "events": _summary_barrier_events(summary),
        "latest_diffusion": None,
        "summary": "",
        "summary_path": str(run_dir / "summary.json"),
        "artifacts": list_artifacts(run_dir),
        "md_log_path": str(log_candidates.get("md")) if log_candidates.get("md") else None,
        "kmc_log_path": str(log_candidates.get("kmc")) if log_candidates.get("kmc") else None,
        "runtime_health": runtime_health,
        "raw_summary": summary,
    }
    if kind == RUN_KIND_MOIRE_CHAIN:
        kmc = summary.get("kmc") or {}
        parsed_run = kmc.get("parsed_run") or {}
        ensemble = kmc.get("ensemble") or {}
        info.update(
            {
                "source_case_dir": summary.get("source_case_dir"),
                "copied_case_dir": summary.get("copied_case_dir"),
                "generated_lammps_input": summary.get("generated_lammps_input"),
                "neb_txt": summary.get("neb_txt"),
                "barrier_eV": _to_float(kmc.get("barrier_eV", summary.get("barrier_eV"))),
                "accepted_events": parsed_run.get("accepted_events"),
                "final_time": parsed_run.get("final_time"),
                "seed_count": ensemble.get("count"),
                "completed_seed_count": ensemble.get("completed_count"),
                "kmc_seeds": ensemble.get("seeds"),
            }
        )
    elif kind == RUN_KIND_MOIRE_COMPARE:
        info.update(
            {
                "case_dir": summary.get("case_dir"),
                "barrier_eV": _to_float((summary.get("reference_event") or {}).get("barrier_eV")),
                "event_count": summary.get("event_count"),
                "completed_event_count": summary.get("completed_count"),
                "failed_event_count": summary.get("failed_count"),
                "compare_run_kmc": summary.get("run_kmc"),
                "compare_event_jsons": summary.get("event_jsons"),
                "compare_barrier_span_eV": _to_float(summary.get("barrier_span_eV")),
            }
        )
    elif kind == RUN_KIND_BRIDGE:
        validation = summary.get("validation") or {}
        info.update(
            {
                "barrier_eV": _to_float(summary.get("barrier_eV")),
                "validation_passed": summary.get("validation_passed"),
                "lookup_hits": validation.get("lookup_hits"),
                "live_ml_misses": validation.get("live_ml_misses"),
            }
        )
    return info


def _barrier_map(events: List[Dict[str, Any]]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    for event in events or []:
        species = event.get("species")
        barrier = _to_float(event.get("barrier_ev"))
        if species and barrier is not None:
            mapping[str(species)] = barrier
    return mapping


def compare_recent_runs(output_dir: Path, mode: Optional[str] = None) -> Dict[str, Any]:
    candidates = list_runs(output_dir, limit=12)
    if mode:
        filtered = [item for item in candidates if item.get("mode") == mode]
        if len(filtered) < 2:
            raise RuntimeError(f"符合模式 {mode} 的 runs 不足两个。")
        candidates = filtered
    if len(candidates) < 2:
        raise RuntimeError("可比较的 runs 不足两个。")

    left = inspect_run(Path(candidates[0]["path"]))
    right = inspect_run(Path(candidates[1]["path"]))
    left_barriers = _barrier_map(left.get("events") or [])
    right_barriers = _barrier_map(right.get("events") or [])
    species = sorted(set(left_barriers) | set(right_barriers))

    barrier_rows = []
    for item in species:
        left_value = left_barriers.get(item)
        right_value = right_barriers.get(item)
        delta = None if left_value is None or right_value is None else left_value - right_value
        barrier_rows.append(
            {
                "species": item,
                "left": left_value,
                "right": right_value,
                "delta": delta,
            }
        )

    return {
        "mode_filter": mode,
        "left": left,
        "right": right,
        "barriers": barrier_rows,
    }
