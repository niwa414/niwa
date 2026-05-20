from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .types import ToolTurnState

from .approval import approval_policy_name
from .snapshot_messages import (
    build_assistant_message,
    build_message_cards,
    build_message_current,
    build_message_session,
)

_TRACE_EVIDENCE_LIMIT = 6
_TIMELINE_PREVIEW_LIMIT = 240


def _trace_tool_result_blocks(state: Optional[ToolTurnState]) -> List[Dict[str, Any]]:
    if state is None:
        return []
    blocks: List[Dict[str, Any]] = []
    for event in getattr(getattr(state, "trace", None), "events", []):
        if getattr(event, "kind", "") != "tool_result_block":
            continue
        block = getattr(event, "block", None)
        if block is None:
            continue
        blocks.append(
            {
                "requestId": block.request_id,
                "action": block.intent.action,
                "params": dict(block.intent.params),
                "ok": bool(block.ok),
                "source": block.source,
                "output": str(block.output or ""),
            }
        )
    return blocks


def _trace_assistant_blocks(state: Optional[ToolTurnState]) -> List[Dict[str, Any]]:
    if state is None:
        return []
    blocks: List[Dict[str, Any]] = []
    for event in getattr(getattr(state, "trace", None), "events", []):
        if getattr(event, "kind", "") != "assistant_action_block":
            continue
        block = getattr(event, "block", None)
        if block is None:
            continue
        blocks.append(
            {
                "source": block.source,
                "toolActions": [request.intent.action for request in (block.tool_requests or [])],
                "finalAnswer": block.final_answer.reply if block.final_answer else None,
                "metadata": dict(block.metadata or {}),
            }
        )
    return blocks


def build_tool_trace_summary(state: Optional[ToolTurnState]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    events = list(getattr(getattr(state, "trace", None), "events", []))
    if not events:
        return None
    tool_blocks = _trace_tool_result_blocks(state)
    assistant_blocks = _trace_assistant_blocks(state)
    finish = next((event for event in reversed(events) if getattr(event, "kind", "") == "turn_finish"), None)
    permission_decisions = [
        getattr(event, "decision", None)
        for event in events
        if getattr(event, "kind", "") == "permission_decision" and getattr(event, "decision", None)
    ]
    decision_counts = {decision: permission_decisions.count(decision) for decision in sorted(set(permission_decisions))}
    tool_actions = [block["action"] for block in tool_blocks]
    return {
        "eventCount": len(events),
        "toolStepCount": len(tool_blocks),
        "toolActions": tool_actions,
        "assistantBlockCount": len(assistant_blocks),
        "assistantSources": [block["source"] for block in assistant_blocks],
        "latestAssistantFinalAnswer": next((block["finalAnswer"] for block in reversed(assistant_blocks) if block.get("finalAnswer")), None),
        "permissionDecisions": decision_counts,
        "finishStatus": getattr(finish, "status", None),
        "finishReason": getattr(finish, "reason", None),
        "finishReply": getattr(finish, "reply", None),
    }


def build_tool_trace_replay(state: Optional[ToolTurnState]) -> List[Dict[str, Any]]:
    if state is None:
        return []
    replay: List[Dict[str, Any]] = []
    for index, event in enumerate(getattr(getattr(state, "trace", None), "events", []), start=1):
        kind = getattr(event, "kind", "")
        entry: Dict[str, Any] = {"index": index, "kind": kind}
        if kind == "assistant_turn":
            entry.update({
                "source": getattr(event, "source", None),
                "rawPreview": str(getattr(event, "raw_content", "") or "")[:400],
                "parsed": getattr(event, "parsed", None),
            })
        elif kind == "assistant_action_block":
            block = getattr(event, "block", None)
            if block is not None:
                entry.update({
                    "source": block.source,
                    "toolActions": [request.intent.action for request in (block.tool_requests or [])],
                    "finalAnswer": block.final_answer.reply if block.final_answer else None,
                    "metadata": dict(block.metadata or {}),
                })
        elif kind == "tool_use":
            intent = getattr(event, "intent", None)
            entry.update({
                "source": getattr(event, "source", None),
                "action": getattr(intent, "action", None),
                "params": dict(getattr(intent, "params", {}) or {}),
                "manual": getattr(event, "manual", False),
            })
        elif kind == "permission_decision":
            intent = getattr(event, "intent", None)
            entry.update({
                "source": getattr(event, "source", None),
                "action": getattr(intent, "action", None),
                "decision": getattr(event, "decision", None),
                "reason": getattr(event, "reason", None),
                "manual": getattr(event, "manual", False),
            })
        elif kind == "tool_result_block":
            block = getattr(event, "block", None)
            if block is not None:
                entry.update({
                    "source": block.source,
                    "requestId": block.request_id,
                    "action": block.intent.action,
                    "params": dict(block.intent.params),
                    "ok": bool(block.ok),
                    "outputPreview": str(block.output or "")[:600],
                })
        elif kind == "tool_result":
            intent = getattr(event, "intent", None)
            outcome = getattr(event, "outcome", None)
            entry.update({
                "source": getattr(event, "source", None),
                "action": getattr(intent, "action", None),
                "ok": getattr(outcome, "ok", None),
                "outputPreview": str(getattr(outcome, "output", "") or "")[:600],
            })
        elif kind == "turn_finish":
            entry.update({
                "status": getattr(event, "status", None),
                "reason": getattr(event, "reason", None),
                "reply": getattr(event, "reply", None),
            })
        replay.append(entry)
    return replay


def build_tool_trace_id(state: Optional[ToolTurnState]) -> Optional[str]:
    replay = build_tool_trace_replay(state)
    if not replay:
        return None
    digest = hashlib.sha1(json.dumps(replay, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"trace-{digest[:12]}"


def build_tool_timeline(state: Optional[ToolTurnState]) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    trace_id = build_tool_trace_id(state)
    for item in build_tool_trace_replay(state):
        kind = item.get("kind")
        entry: Dict[str, Any] = {
            "index": item.get("index"),
            "kind": kind,
            "source": item.get("source"),
        }
        if trace_id and item.get("index") is not None:
            entry["transcriptRef"] = {
                "traceId": trace_id,
                "eventIndex": item.get("index"),
                "label": f"{trace_id}#{item.get('index')}",
            }
        if kind == "assistant_turn":
            entry.update(
                {
                    "stage": "assistant_raw",
                    "status": "info",
                    "title": "Assistant raw model turn",
                    "detail": item.get("rawPreview") or "The local model produced raw content before normalization.",
                }
            )
        elif kind == "assistant_action_block":
            tool_actions = [str(action) for action in (item.get("toolActions") or []) if action]
            final_answer = str(item.get("finalAnswer") or "").strip()
            detail = (
                f"Planned tools: {', '.join(tool_actions)}"
                if tool_actions
                else "No tool actions were proposed in this assistant block."
            )
            if final_answer:
                detail = f"{detail} Final answer preview: {final_answer[:_TIMELINE_PREVIEW_LIMIT]}"
            entry.update(
                {
                    "stage": "assistant",
                    "status": "info",
                    "title": "Assistant normalized its next action",
                    "detail": detail,
                    "toolActions": tool_actions,
                    "finalAnswer": final_answer or None,
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
        elif kind == "tool_use":
            action = item.get("action") or "unknown"
            entry.update(
                {
                    "stage": "tool_request",
                    "status": "running",
                    "title": f"Requested tool: {action}",
                    "detail": "This step was issued manually." if item.get("manual") else "This step was issued automatically by the agent.",
                    "action": action,
                    "params": dict(item.get("params") or {}),
                    "manual": bool(item.get("manual")),
                }
            )
        elif kind == "permission_decision":
            action = item.get("action") or "unknown"
            decision = item.get("decision") or "unknown"
            status = "completed" if decision == "allow" else "blocked"
            entry.update(
                {
                    "stage": "permission",
                    "status": status,
                    "title": f"Permission check: {action}",
                    "detail": item.get("reason") or f"Decision: {decision}",
                    "action": action,
                    "decision": decision,
                    "manual": bool(item.get("manual")),
                }
            )
        elif kind == "tool_result":
            action = item.get("action") or "unknown"
            ok = item.get("ok")
            entry.update(
                {
                    "stage": "tool_result_raw",
                    "status": "completed" if ok is not False else "failed",
                    "title": f"Tool returned: {action}",
                    "detail": str(item.get("outputPreview") or "")[:_TIMELINE_PREVIEW_LIMIT] or "The tool returned without a preview payload.",
                    "action": action,
                    "ok": ok,
                }
            )
        elif kind == "tool_result_block":
            action = item.get("action") or "unknown"
            ok = bool(item.get("ok"))
            entry.update(
                {
                    "stage": "tool_result",
                    "status": "completed" if ok else "failed",
                    "title": f"Normalized tool result: {action}",
                    "detail": str(item.get("outputPreview") or "")[:_TIMELINE_PREVIEW_LIMIT] or "No structured tool output preview was recorded.",
                    "action": action,
                    "params": dict(item.get("params") or {}),
                    "requestId": item.get("requestId"),
                    "ok": ok,
                }
            )
        elif kind == "turn_finish":
            finish_status = item.get("status") or "unknown"
            status = "completed" if finish_status == "finish" else ("failed" if finish_status == "error" else "info")
            entry.update(
                {
                    "stage": "finish",
                    "status": status,
                    "title": "Assistant finished this tool-backed turn",
                    "detail": item.get("reason") or item.get("reply") or "The turn finished without an explicit reason.",
                    "finishStatus": finish_status,
                    "reply": item.get("reply"),
                }
            )
        else:
            entry.update(
                {
                    "stage": "event",
                    "status": "info",
                    "title": kind or "event",
                    "detail": "A trace event was recorded.",
                }
            )
        timeline.append(entry)
    return timeline


def build_tool_evidence(state: Optional[ToolTurnState]) -> List[Dict[str, Any]]:
    tool_blocks = _trace_tool_result_blocks(state)
    if not tool_blocks:
        return []
    evidence = []
    for idx, block in enumerate(tool_blocks[-_TRACE_EVIDENCE_LIMIT:], start=1):
        output = block["output"]
        evidence.append(
            {
                "step": idx,
                "requestId": block["requestId"],
                "action": block["action"],
                "params": block["params"],
                "ok": block["ok"],
                "source": block["source"],
                "outputPreview": output[:1200],
            }
        )
    return evidence


def build_current_snapshot(
    *,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = {
        "active_kind": None,
        "has_any": False,
        "run_dir": str(current_run_dir) if current_run_dir else None,
        "report": current_report,
        "bridge_summary": current_bridge_summary,
        "moire_summary": current_moire_summary,
        "moire_compare_summary": current_moire_compare_summary,
        "moire_diffusion_summary": current_moire_diffusion_summary,
    }

    if snapshot["run_dir"]:
        snapshot["active_kind"] = "run"
    elif current_moire_diffusion_summary is not None:
        snapshot["active_kind"] = "moire_diffusion_summary"
    elif current_moire_compare_summary is not None:
        snapshot["active_kind"] = "moire_compare_summary"
    elif current_moire_summary is not None:
        snapshot["active_kind"] = "moire_summary"
    elif current_bridge_summary is not None:
        snapshot["active_kind"] = "bridge_summary"
    elif current_report is not None:
        snapshot["active_kind"] = "report"

    snapshot["has_any"] = snapshot["active_kind"] is not None
    return snapshot


def build_session_snapshot(
    *,
    transcript_path: Path,
    selected_model: Optional[str],
    history_length: int,
    turn_count: Optional[int] = None,
    active_turn_id: Optional[str] = None,
    permission_denial_count: Optional[int] = None,
    usage_stats: Optional[Dict[str, Any]] = None,
    turn_state: Optional[Dict[str, Any]] = None,
    queued_followups: Optional[List[Dict[str, Any]]] = None,
    aborted_turns: Optional[List[Dict[str, Any]]] = None,
    memory_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "transcript_path": str(transcript_path),
        "selected_model": selected_model,
        "approval_policy": approval_policy_name(),
        "history_length": history_length,
    }
    if turn_count is not None:
        payload["turn_count"] = turn_count
    if active_turn_id is not None:
        payload["active_turn_id"] = active_turn_id
    if permission_denial_count is not None:
        payload["permission_denial_count"] = permission_denial_count
    if usage_stats:
        payload["usage_stats"] = dict(usage_stats)
    if queued_followups is not None:
        queued_items = list(queued_followups)
        payload["queued_followup_count"] = len(queued_items)
        payload["queued_followups"] = queued_items[-4:]
        payload["runnable_followup_count"] = len([item for item in queued_items if bool(item.get("runnable"))])
        payload["auto_followup_count"] = len([item for item in queued_items if bool(item.get("auto_continue"))])
    if aborted_turns is not None:
        payload["aborted_turn_count"] = len(list(aborted_turns))
        if aborted_turns:
            payload["latest_aborted_turn"] = dict(list(aborted_turns)[-1])
    if memory_summary:
        payload["memory_record_count"] = int(memory_summary.get("archived_turn_count") or 0)
        payload["fresh_memory_count"] = int(memory_summary.get("fresh_memory_count") or 0)
        payload["stale_memory_count"] = int(memory_summary.get("stale_memory_count") or 0)
        if memory_summary.get("compact_boundary_turn_id"):
            payload["compact_boundary_turn_id"] = memory_summary.get("compact_boundary_turn_id")
        if memory_summary.get("summary"):
            payload["memory_summary"] = str(memory_summary.get("summary"))
    if turn_state:
        payload["latest_turn_id"] = turn_state.get("turnId")
        payload["turn_message_count"] = turn_state.get("messageCount")
        payload["turn_block_count"] = turn_state.get("blockCount")
        payload["turn_event_count"] = turn_state.get("eventCount")
        payload["turn_followup_count"] = turn_state.get("followupCount")
        payload["turn_denial_count"] = turn_state.get("denialCount")
        payload["turn_child_count"] = turn_state.get("childTurnCount")
        if turn_state.get("finishReason"):
            payload["turn_finish_reason"] = turn_state.get("finishReason")
        if turn_state.get("finishStatus"):
            payload["turn_finish_status"] = turn_state.get("finishStatus")
        if turn_state.get("usage"):
            payload["turn_usage"] = dict(turn_state.get("usage") or {})
        if turn_state.get("statusDetail"):
            payload["turn_status_detail"] = dict(turn_state.get("statusDetail") or {})
        if turn_state.get("latestChildTurn"):
            payload["latest_child_turn"] = dict(turn_state.get("latestChildTurn") or {})
    return payload


def build_turn_session_snapshot(
    *,
    session_state: Optional[Any],
    turn_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if session_state is None:
        return None
    resolved_turn_id = turn_id
    if not resolved_turn_id:
        if getattr(session_state, "active_turn_id", None):
            resolved_turn_id = session_state.active_turn_id
        elif getattr(session_state, "turns", None):
            resolved_turn_id = session_state.turns[-1].get("turn_id")
    if not resolved_turn_id:
        return None
    messages = list(getattr(session_state, "turn_messages", {}).get(resolved_turn_id, []))
    blocks = list(getattr(session_state, "turn_blocks", {}).get(resolved_turn_id, []))
    events = list(getattr(session_state, "turn_events", {}).get(resolved_turn_id, []))
    status_detail = dict(getattr(session_state, "turn_status_detail", {}).get(resolved_turn_id, {}))
    followups = list(getattr(session_state, "turn_followups", {}).get(resolved_turn_id, []))
    denials = list(getattr(session_state, "turn_denials", {}).get(resolved_turn_id, []))
    usage = dict(getattr(session_state, "turn_usage", {}).get(resolved_turn_id, {}))
    finish_reason = dict(getattr(session_state, "turn_finish_reason", {}).get(resolved_turn_id, {}))
    resume_source = dict(getattr(session_state, "turn_resume_source", {}).get(resolved_turn_id, {}))
    child_turns = list(getattr(session_state, "turn_children", {}).get(resolved_turn_id, []))
    queued_followups = [
        dict(item)
        for item in list(getattr(session_state, "queued_followups", []) or [])
        if not bool(item.get("consumed"))
    ]
    return {
        "turnId": resolved_turn_id,
        "messageCount": len(messages),
        "blockCount": len(blocks),
        "eventCount": len(events),
        "followupCount": len(followups),
        "denialCount": len(denials),
        "childTurnCount": len(child_turns),
        "finishReason": finish_reason.get("reason"),
        "finishStatus": finish_reason.get("status"),
        "usage": usage,
        "statusDetail": status_detail,
        "messages": messages,
        "blocks": blocks,
        "events": events,
        "followups": followups,
        "denials": denials,
        "resumeSource": resume_source,
        "latestChildTurn": dict(child_turns[-1]) if child_turns else None,
        "queuedFollowupCount": len(queued_followups),
        "queuedRunnableCount": len([item for item in queued_followups if bool(item.get("runnable"))]),
        "queuedAutoFollowupCount": len([item for item in queued_followups if bool(item.get("auto_continue"))]),
    }


def build_snapshot_from_turn_result(
    *,
    turn_result: Any,
    session_state: Optional[Any] = None,
    progress_lines: Optional[List[str]] = None,
    transcript_path: Path,
    selected_model: Optional[str],
    history_length: int,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
    turn_count: Optional[int] = None,
    active_turn_id: Optional[str] = None,
    permission_denial_count: Optional[int] = None,
    usage_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_chat_once_payload(
        reply=getattr(turn_result, "reply", ""),
        progress_lines=list(progress_lines if progress_lines is not None else (getattr(turn_result, "progress_lines", []) or [])),
        transcript_path=transcript_path,
        selected_model=selected_model,
        history_length=history_length,
        current_run_dir=current_run_dir,
        current_report=current_report,
        current_bridge_summary=current_bridge_summary,
        current_moire_summary=current_moire_summary,
        current_moire_compare_summary=current_moire_compare_summary,
        current_moire_diffusion_summary=current_moire_diffusion_summary,
        last_tool_turn_state=getattr(turn_result, "tool_turn_state", None),
        turn_count=turn_count,
        active_turn_id=active_turn_id,
        permission_denial_count=permission_denial_count,
        usage_stats=usage_stats,
        session_state=session_state,
        turn_id=getattr(turn_result, "turn_id", None),
    )


def build_message_from_turn_result(**kwargs: Any) -> Dict[str, Any]:
    return build_snapshot_from_turn_result(**kwargs)["message"]


def build_chat_once_payload(
    *,
    reply: str,
    progress_lines: List[str],
    transcript_path: Path,
    selected_model: Optional[str],
    history_length: int,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
    last_tool_turn_state: Optional[ToolTurnState] = None,
    turn_result: Optional[Any] = None,
    turn_count: Optional[int] = None,
    active_turn_id: Optional[str] = None,
    permission_denial_count: Optional[int] = None,
    usage_stats: Optional[Dict[str, Any]] = None,
    session_state: Optional[Any] = None,
    turn_id: Optional[str] = None,
) -> Dict[str, Any]:
    if turn_result is not None:
        reply = getattr(turn_result, "reply", reply)
        progress_lines = list(getattr(turn_result, "progress_lines", []) or progress_lines)
        last_tool_turn_state = getattr(turn_result, "tool_turn_state", last_tool_turn_state)
        turn_id = getattr(turn_result, "turn_id", turn_id)
    turn_state = build_turn_session_snapshot(session_state=session_state, turn_id=turn_id)
    queued_followups = []
    aborted_turns = []
    memory_summary = {}
    if session_state is not None:
        queued_followups = [
            dict(item)
            for item in list(getattr(session_state, "queued_followups", []) or [])
            if not bool(item.get("consumed"))
        ]
        aborted_turns = list(getattr(session_state, "aborted_turns", []) or [])
        memory_summary = dict(getattr(session_state, "memory_summary", {}) or {})
    session = build_session_snapshot(
        transcript_path=transcript_path,
        selected_model=selected_model,
        history_length=history_length,
        turn_count=turn_count,
        active_turn_id=active_turn_id,
        permission_denial_count=permission_denial_count,
        usage_stats=usage_stats,
        turn_state=turn_state,
        queued_followups=queued_followups,
        aborted_turns=aborted_turns,
        memory_summary=memory_summary,
    )
    current = build_current_snapshot(
        current_run_dir=current_run_dir,
        current_report=current_report,
        current_bridge_summary=current_bridge_summary,
        current_moire_summary=current_moire_summary,
        current_moire_compare_summary=current_moire_compare_summary,
        current_moire_diffusion_summary=current_moire_diffusion_summary,
    )
    tool_trace_summary = build_tool_trace_summary(last_tool_turn_state)
    tool_evidence = build_tool_evidence(last_tool_turn_state)
    tool_trace_replay = build_tool_trace_replay(last_tool_turn_state)
    tool_timeline = build_tool_timeline(last_tool_turn_state)
    tool_trace_id = build_tool_trace_id(last_tool_turn_state)
    used_tools = bool(progress_lines or current["has_any"] or tool_evidence)
    kind = "tool" if used_tools else "chat"
    message = build_assistant_message(
        reply=reply,
        kind=kind,
        progress_lines=progress_lines,
        session=session,
        current=current,
        tool_trace_summary=tool_trace_summary,
        tool_evidence=tool_evidence,
        tool_trace_replay=tool_trace_replay,
        tool_timeline=tool_timeline,
        tool_trace_id=tool_trace_id,
        turn_state=turn_state,
    )

    return {
        "reply": reply,
        "kind": kind,
        "used_tools": used_tools,
        "progress": list(progress_lines),
        "session": session,
        "current": current,
        "message": message,
        "tool_trace_summary": tool_trace_summary,
        "tool_evidence": tool_evidence,
        "tool_trace_replay": tool_trace_replay,
        "tool_timeline": tool_timeline,
        "tool_trace_id": tool_trace_id,
        "turn_state": turn_state,
        # Compatibility fields kept for existing callers.
        "transcript_path": session["transcript_path"],
        "selected_model": session["selected_model"],
        "approval_policy": session["approval_policy"],
        "history_length": session["history_length"],
        "current_context_kind": current["active_kind"],
        "current_run_dir": current["run_dir"],
        "current_report": current["report"],
        "current_bridge_summary": current["bridge_summary"],
        "current_moire_summary": current["moire_summary"],
        "current_moire_compare_summary": current["moire_compare_summary"],
        "current_moire_diffusion_summary": current["moire_diffusion_summary"],
    }
