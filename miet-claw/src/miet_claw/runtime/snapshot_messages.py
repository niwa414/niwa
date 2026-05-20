from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..planner import build_plan_payload
from ..specs import load_job_spec

_COMPLETED_STATUSES = {"completed", "executed", "ok", "healthy", "passed", "success"}
_FAILED_STATUSES = {"failed", "error", "aborted"}


def _status_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _COMPLETED_STATUSES:
        return "completed"
    if text in _FAILED_STATUSES:
        return "failed"
    if text:
        return "running"
    return "unknown"


def _display_status(value: Any) -> str:
    bucket = _status_bucket(value)
    if bucket == "completed":
        return "Completed"
    if bucket == "failed":
        return "Failed"
    if bucket == "running":
        return str(value or "Running").strip().capitalize() or "Running"
    return str(value or "Unknown").strip().capitalize() or "Unknown"


def _load_current_report_plan(current_report: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(current_report, dict):
        return None
    job_spec = current_report.get("job_spec")
    if isinstance(job_spec, dict) and job_spec.get("mode"):
        try:
            return build_plan_payload(job_spec)
        except Exception:
            pass

    generated_files = current_report.get("generated_files")
    if isinstance(generated_files, dict):
        job_spec_path = generated_files.get("job_spec")
        if isinstance(job_spec_path, str) and job_spec_path.strip():
            try:
                return build_plan_payload(load_job_spec(job_spec_path))
            except Exception:
                return None
    return None


def _load_current_report_notes(current_report: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(current_report, dict):
        return None
    generated_files = current_report.get("generated_files")
    if not isinstance(generated_files, dict):
        return None
    notes_path = generated_files.get("notes")
    if not isinstance(notes_path, str) or not notes_path.strip():
        return None
    path = Path(notes_path).expanduser()
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _normalize_latest_diffusion(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    return {
        "jumps": payload.get("jumps"),
        "msd": payload.get("msd"),
        "simulationTime": payload.get("simulation_time"),
        "jumpFrequency": payload.get("jump_frequency"),
        "diffusionCoefficient": payload.get("diffusion_coefficient"),
    }


def _load_current_run_detail(current_run_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if current_run_dir is None:
        return None
    try:
        run_dir = current_run_dir.expanduser().resolve()
    except Exception:
        return None
    if not run_dir.exists():
        return None

    try:
        from ..chat import inspect_run  # local import to avoid module cycle at import time

        raw = inspect_run(run_dir)
    except Exception:
        return None

    step_statuses = raw.get("step_statuses") or {}
    steps = [
        {
            "id": str(step_id),
            "status": str(status or "unknown").strip().lower() or "unknown",
            "startedAt": None,
            "heartbeatAt": None,
            "completedAt": None,
            "pid": None,
            "detail": None,
        }
        for step_id, status in step_statuses.items()
    ]
    completed_steps = sum(1 for value in step_statuses.values() if _status_bucket(value) == "completed")
    total_steps = len(steps)
    summary_text = str(raw.get("summary") or "")

    try:
        stats = run_dir.stat()
        updated_at = stats.st_mtime
        created_at = stats.st_birthtime if hasattr(stats, "st_birthtime") else stats.st_ctime
    except Exception:
        updated_at = None
        created_at = None

    return {
        "id": raw.get("job_id") or run_dir.name,
        "runDir": str(run_dir),
        "runDirRelative": None,
        "jobId": raw.get("job_id") or run_dir.name,
        "mode": raw.get("mode"),
        "materialName": raw.get("material_name") or run_dir.name,
        "updatedAt": None if updated_at is None else __import__("datetime").datetime.fromtimestamp(updated_at).isoformat(),
        "createdAt": None if created_at is None else __import__("datetime").datetime.fromtimestamp(created_at).isoformat(),
        "status": _display_status(raw.get("status")),
        "completedSteps": completed_steps,
        "totalSteps": total_steps,
        "activeStep": None,
        "summary": summary_text,
        "summaryPreview": summary_text[:800],
        "manifest": {
            "artifacts": [{"path": artifact} for artifact in (raw.get("artifacts") or [])],
        },
        "spec": None,
        "steps": steps,
        "md": {
            "barriers": {
                "metadata": {
                    "workflow_kind": raw.get("workflow_kind"),
                    "barrier_source_mode": raw.get("barrier_source_mode"),
                },
                "events": raw.get("events") or [],
            },
            "execution": None,
            "referenceEnergyEv": None,
        },
        "chain": {
            "eventRows": [],
            "eventTablePath": None,
        },
        "kmc": {
            "execution": None,
            "generatedInputPath": None,
            "generatedInput": None,
            "diffusionRows": [],
            "diffusionPath": None,
            "latestDiffusion": _normalize_latest_diffusion(raw.get("latest_diffusion")),
        },
    }


def build_message_session(session: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "transcriptPath": session["transcript_path"],
        "selectedModel": session["selected_model"],
        "approvalPolicy": session["approval_policy"],
        "historyLength": session["history_length"],
    }
    if "turn_count" in session:
        payload["turnCount"] = session["turn_count"]
    if "active_turn_id" in session:
        payload["activeTurnId"] = session["active_turn_id"]
    if "permission_denial_count" in session:
        payload["permissionDenialCount"] = session["permission_denial_count"]
    if "usage_stats" in session:
        payload["usageStats"] = dict(session["usage_stats"])
    if "latest_turn_id" in session:
        payload["latestTurnId"] = session["latest_turn_id"]
    if "turn_message_count" in session:
        payload["turnMessageCount"] = session["turn_message_count"]
    if "turn_block_count" in session:
        payload["turnBlockCount"] = session["turn_block_count"]
    if "turn_event_count" in session:
        payload["turnEventCount"] = session["turn_event_count"]
    if "turn_followup_count" in session:
        payload["turnFollowupCount"] = session["turn_followup_count"]
    if "turn_denial_count" in session:
        payload["turnDenialCount"] = session["turn_denial_count"]
    if "turn_child_count" in session:
        payload["turnChildCount"] = session["turn_child_count"]
    if "turn_finish_reason" in session:
        payload["turnFinishReason"] = session["turn_finish_reason"]
    if "turn_finish_status" in session:
        payload["turnFinishStatus"] = session["turn_finish_status"]
    if "turn_usage" in session:
        payload["turnUsage"] = dict(session["turn_usage"])
    if "turn_status_detail" in session:
        payload["turnStatusDetail"] = dict(session["turn_status_detail"])
    if "queued_followup_count" in session:
        payload["queuedFollowupCount"] = session["queued_followup_count"]
    if "runnable_followup_count" in session:
        payload["runnableFollowupCount"] = session["runnable_followup_count"]
    if "auto_followup_count" in session:
        payload["autoFollowupCount"] = session["auto_followup_count"]
    if "queued_followups" in session:
        payload["queuedFollowups"] = list(session["queued_followups"])
    if "memory_record_count" in session:
        payload["memoryRecordCount"] = session["memory_record_count"]
    if "fresh_memory_count" in session:
        payload["freshMemoryCount"] = session["fresh_memory_count"]
    if "stale_memory_count" in session:
        payload["staleMemoryCount"] = session["stale_memory_count"]
    if "compact_boundary_turn_id" in session:
        payload["compactBoundaryTurnId"] = session["compact_boundary_turn_id"]
    if "memory_summary" in session:
        payload["memorySummary"] = session["memory_summary"]
    if "aborted_turn_count" in session:
        payload["abortedTurnCount"] = session["aborted_turn_count"]
    if "latest_aborted_turn" in session:
        payload["latestAbortedTurn"] = dict(session["latest_aborted_turn"])
    if "latest_child_turn" in session:
        payload["latestChildTurn"] = dict(session["latest_child_turn"])
    return payload


def build_message_current(current: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "activeKind": current["active_kind"],
        "hasAny": current["has_any"],
        "runDir": current["run_dir"],
        "report": current["report"],
        "bridgeSummary": current["bridge_summary"],
        "moireSummary": current["moire_summary"],
        "moireCompareSummary": current["moire_compare_summary"],
        "moireDiffusionSummary": current["moire_diffusion_summary"],
    }


def build_message_cards(
    *,
    message_session: Dict[str, Any],
    message_current: Dict[str, Any],
    current_plan: Optional[List[Dict[str, Any]]],
    current_report_notes: Optional[str],
    current_run_detail: Optional[Dict[str, Any]],
    tool_trace_summary: Optional[Dict[str, Any]],
    tool_evidence: List[Dict[str, Any]],
    tool_trace_replay: List[Dict[str, Any]],
    tool_timeline: List[Dict[str, Any]],
    turn_state: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    transcript_path = message_session.get("transcriptPath")
    current_report = message_current.get("report")
    job_spec_path = None
    if isinstance(current_report, dict):
        generated_files = current_report.get("generated_files")
        if isinstance(generated_files, dict):
            job_spec_path = generated_files.get("job_spec")
        cards.append(
            {
                "type": "transparency",
                "report": current_report,
                "plan": current_plan,
                "notes": current_report_notes,
                "previewConfirmable": False,
                "previewExecuted": False,
                "jobSpecPath": job_spec_path,
                "transcriptPath": transcript_path,
                "toolTraceSummary": tool_trace_summary,
                "toolEvidence": tool_evidence,
                "toolTraceReplay": tool_trace_replay,
            }
        )

    if message_session or message_current.get("hasAny"):
        if tool_timeline:
            cards.append(
                {
                    "type": "tool_timeline",
                    "timeline": tool_timeline,
                    "transcriptPath": transcript_path,
                }
            )
        cards.append(
            {
                "type": "runtime_snapshot",
                "session": message_session,
                "current": message_current,
                "transcriptPath": transcript_path,
                "toolTraceSummary": tool_trace_summary,
                "toolEvidence": tool_evidence,
                "toolTraceReplay": tool_trace_replay,
                "toolTimeline": tool_timeline,
                "turnState": turn_state,
            }
        )

    if current_run_detail:
        cards.append(
            {
                "type": "run_result",
                "detail": current_run_detail,
            }
        )

    return cards


def build_assistant_message(
    *,
    reply: str,
    kind: str,
    progress_lines: List[str],
    session: Dict[str, Any],
    current: Dict[str, Any],
    tool_trace_summary: Optional[Dict[str, Any]] = None,
    tool_evidence: Optional[List[Dict[str, Any]]] = None,
    tool_trace_replay: Optional[List[Dict[str, Any]]] = None,
    tool_timeline: Optional[List[Dict[str, Any]]] = None,
    tool_trace_id: Optional[str] = None,
    turn_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    message_session = build_message_session(session)
    message_current = build_message_current(current)
    current_report = message_current.get("report")
    current_plan = _load_current_report_plan(current_report)
    current_report_notes = _load_current_report_notes(current_report)
    current_run_detail = _load_current_run_detail(Path(message_current["runDir"])) if message_current.get("runDir") else None
    job_spec_path = None
    resolved_tool_evidence = list(tool_evidence or [])
    resolved_tool_trace_replay = list(tool_trace_replay or [])
    resolved_tool_timeline = list(tool_timeline or [])
    if isinstance(current_report, dict):
        generated_files = current_report.get("generated_files")
        if isinstance(generated_files, dict):
            job_spec_path = generated_files.get("job_spec")

    return {
        "role": "assistant",
        "content": reply,
        "kind": kind,
        "model": session["selected_model"] if kind == "chat" else None,
        "progress": list(progress_lines),
        "session": message_session,
        "current": message_current,
        "currentReport": message_current["report"],
        "currentPlan": current_plan,
        "currentReportNotes": current_report_notes,
        "currentRunDetail": current_run_detail,
        "currentBridgeSummary": message_current["bridgeSummary"],
        "currentMoireSummary": message_current["moireSummary"],
        "currentMoireCompareSummary": message_current["moireCompareSummary"],
        "currentMoireDiffusionSummary": message_current["moireDiffusionSummary"],
        "transcriptPath": message_session["transcriptPath"],
        "previewConfirmable": False,
        "previewExecuted": False,
        "jobSpecPath": job_spec_path,
        "toolTraceSummary": tool_trace_summary,
        "toolEvidence": resolved_tool_evidence,
        "toolTraceReplay": resolved_tool_trace_replay,
        "toolTimeline": resolved_tool_timeline,
        "toolTraceId": tool_trace_id,
        "turnState": turn_state,
        "cards": build_message_cards(
            message_session=message_session,
            message_current=message_current,
            current_plan=current_plan,
            current_report_notes=current_report_notes,
            current_run_detail=current_run_detail,
            tool_trace_summary=tool_trace_summary,
            tool_evidence=resolved_tool_evidence,
            tool_trace_replay=resolved_tool_trace_replay,
            tool_timeline=resolved_tool_timeline,
            turn_state=turn_state,
        ),
    }

