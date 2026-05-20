from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..tool_router import ToolIntent
from .permissions import read_bounded_int_env


def tool_context_limit() -> int:
    return read_bounded_int_env("MIETCLAW_TOOL_CONTEXT_LIMIT", 6, 0, 12)


def tool_history_payload(
    outputs: List[Tuple[ToolIntent, str]],
    *,
    output_limit: int,
    truncate_output: Callable[[str, int], str],
) -> List[Dict[str, Any]]:
    tool_history = []
    for idx, (intent, result) in enumerate(outputs, start=1):
        tool_history.append(
            {
                "step": idx,
                "action": intent.action,
                "params": intent.params,
                "output": truncate_output(result, output_limit),
            }
        )
    return tool_history






def _normalize_tool_block(block: Any) -> Dict[str, Any]:
    if isinstance(block, dict):
        return {
            "requestId": block.get("requestId") or block.get("request_id"),
            "action": block.get("action"),
            "params": dict(block.get("params") or {}),
            "output": str(block.get("output") or block.get("outputPreview") or ""),
            "ok": block.get("ok"),
            "source": block.get("source"),
        }
    intent = getattr(block, "intent", None)
    return {
        "requestId": getattr(block, "request_id", None) or getattr(block, "requestId", None),
        "action": getattr(intent, "action", None),
        "params": dict(getattr(intent, "params", {}) or {}),
        "output": str(getattr(block, "output", "") or ""),
        "ok": getattr(block, "ok", None),
        "source": getattr(block, "source", None),
    }


def tool_history_payload_from_blocks(
    blocks: List[Dict[str, Any]],
    *,
    output_limit: int,
    truncate_output: Callable[[str, int], str],
) -> List[Dict[str, Any]]:
    tool_history = []
    for idx, raw_block in enumerate(blocks, start=1):
        block = _normalize_tool_block(raw_block)
        tool_history.append(
            {
                "step": idx,
                "action": block.get("action"),
                "params": dict(block.get("params") or {}),
                "output": truncate_output(str(block.get("output") or ""), output_limit),
                "ok": block.get("ok"),
                "source": block.get("source"),
                "requestId": block.get("requestId"),
            }
        )
    return tool_history


def _dedupe_tool_blocks(blocks: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    seen = set()
    deduped_reversed: List[Dict[str, Any]] = []
    for raw_block in reversed(blocks):
        block = _normalize_tool_block(raw_block)
        params_json = json.dumps(block.get("params") or {}, ensure_ascii=False, sort_keys=True)
        key = (
            block.get("requestId") or block.get("action"),
            params_json,
            str(block.get("output") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_reversed.append(block)
        if len(deduped_reversed) >= limit:
            break
    return list(reversed(deduped_reversed))


def remember_tool_context_blocks(
    *,
    existing_blocks: List[Dict[str, Any]],
    new_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not new_blocks:
        return existing_blocks
    combined = list(existing_blocks) + list(new_blocks)
    limit = tool_context_limit()
    if limit <= 0:
        return []
    return _dedupe_tool_blocks(combined, limit=limit)


def collect_tool_context_history(
    *,
    existing_outputs: List[Tuple[ToolIntent, str]],
    current_outputs: Optional[List[Tuple[ToolIntent, str]]],
    existing_blocks: Optional[List[Dict[str, Any]]] = None,
    current_blocks: Optional[List[Dict[str, Any]]] = None,
    output_limit: int,
    intent_signature: Callable[[ToolIntent], str],
    truncate_output: Callable[[str, int], str],
) -> List[Dict[str, Any]]:
    combined_blocks = list(existing_blocks or [])
    if current_blocks:
        combined_blocks.extend(current_blocks)
    limit = tool_context_limit()
    if limit <= 0:
        return []
    if combined_blocks:
        deduped_blocks = _dedupe_tool_blocks(combined_blocks, limit=limit)
        return tool_history_payload_from_blocks(deduped_blocks, output_limit=output_limit, truncate_output=truncate_output)

    combined = list(existing_outputs)
    if current_outputs:
        combined.extend(current_outputs)
    if not combined:
        return []
    seen = set()
    deduped_reversed: List[Tuple[ToolIntent, str]] = []
    for intent, result in reversed(combined):
        key = (intent_signature(intent), result)
        if key in seen:
            continue
        seen.add(key)
        deduped_reversed.append((intent, result))
        if len(deduped_reversed) >= limit:
            break
    deduped = list(reversed(deduped_reversed))
    return tool_history_payload(deduped, output_limit=output_limit, truncate_output=truncate_output)


def remember_tool_turn(
    *,
    existing_outputs: List[Tuple[ToolIntent, str]],
    new_outputs: List[Tuple[ToolIntent, str]],
) -> List[Tuple[ToolIntent, str]]:
    if not new_outputs:
        return existing_outputs
    combined = list(existing_outputs) + list(new_outputs)
    limit = tool_context_limit()
    if limit <= 0:
        return []
    return combined[-limit:]


def build_current_context_for_chat(
    *,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
    api: Dict[str, Any],
) -> Optional[str]:
    if current_run_dir:
        return "Current inspected run:\n" + api["format_inspect_report"](api["inspect_run"](current_run_dir))
    if current_moire_diffusion_summary:
        return "Current MoRe diffusion sweep:\n" + api["format_moire_diffusion_sweep_report"](current_moire_diffusion_summary)
    if current_moire_compare_summary:
        return "Current MoRe compare:\n" + api["format_moire_compare_report"](current_moire_compare_summary)
    if current_moire_summary:
        return "Current MoRe workflow:\n" + api["format_moire_workflow_report"](current_moire_summary)
    if current_bridge_summary:
        if current_bridge_summary.get("parsed_run") is not None:
            return "Current repo KMC result:\n" + api["format_moire_kmc_report"](current_bridge_summary)
        return "Current bridge result:\n" + api["format_bridge_report"](current_bridge_summary)
    if current_report:
        return "Current draft workspace:\n" + api["format_draft_report"](current_report)
    return None


def build_current_work_context(
    *,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    active_kind: Optional[str] = None
    if current_run_dir is not None:
        active_kind = "run"
    elif current_moire_diffusion_summary is not None:
        active_kind = "moire_diffusion"
    elif current_moire_compare_summary is not None:
        active_kind = "moire_compare"
    elif current_moire_summary is not None:
        active_kind = "moire"
    elif current_bridge_summary is not None:
        active_kind = "bridge"
    elif current_report is not None:
        active_kind = "draft"
    return {
        "active_kind": active_kind,
        "current_run_dir": str(current_run_dir) if current_run_dir else None,
        "has_report": current_report is not None,
        "has_bridge_summary": current_bridge_summary is not None,
        "has_moire_summary": current_moire_summary is not None,
        "has_moire_compare_summary": current_moire_compare_summary is not None,
        "has_moire_diffusion_summary": current_moire_diffusion_summary is not None,
    }


def build_turn_context_payload(
    *,
    active_turn_id: Optional[str],
    current_turn: Optional[Dict[str, Any]],
    pending_tool_requests: Optional[List[Dict[str, Any]]],
    permission_denials: Optional[List[Dict[str, Any]]],
    queued_followups: Optional[List[Dict[str, Any]]],
    usage_stats: Optional[Dict[str, Any]],
    current_state: Optional[Any],
    session_state: Optional[Any] = None,
) -> Dict[str, Any]:
    active_queued_followups = [
        dict(item)
        for item in list(queued_followups or [])
        if not bool(item.get("consumed"))
    ]
    current_prompt = current_turn.get("prompt") if isinstance(current_turn, dict) else None
    turn_snapshot = None
    if session_state is not None and active_turn_id:
        try:
            turn_snapshot = session_state.resume_turn_state(active_turn_id)
        except Exception:
            turn_snapshot = None
    turn_messages = list((turn_snapshot or {}).get("messages") or [])
    turn_blocks = list((turn_snapshot or {}).get("blocks") or [])
    turn_events = list((turn_snapshot or {}).get("events") or [])
    turn_denials = list((turn_snapshot or {}).get("denials") or [])
    turn_followups = list((turn_snapshot or {}).get("followups") or [])
    turn_usage = dict((turn_snapshot or {}).get("usage") or {})
    turn_finish = dict((turn_snapshot or {}).get("finish_reason") or {})
    turn_resume_source = dict((turn_snapshot or {}).get("resume_source") or {})
    child_turns = list((turn_snapshot or {}).get("child_turns") or [])
    aborted_turns = list(getattr(session_state, "aborted_turns", []) or [])
    return {
        "active_turn_id": active_turn_id,
        "current_prompt": current_prompt,
        "current_status": current_turn.get("status") if isinstance(current_turn, dict) else None,
        "tool_step_count": len(getattr(current_state, "outputs", []) or []),
        "note_count": len(getattr(current_state, "notes", []) or []),
        "duplicate_steps": getattr(current_state, "duplicate_steps", 0) or 0,
        "message_count": len(turn_messages),
        "block_count": len(turn_blocks),
        "event_count": len(turn_events),
        "pending_tool_requests": list(pending_tool_requests or [])[-4:],
        "recent_permission_denials": list(permission_denials or [])[-4:],
        "queued_followups": active_queued_followups[-4:],
        "queued_followup_count": len(active_queued_followups),
        "queued_runnable_followup_count": len([item for item in active_queued_followups if bool(item.get("runnable"))]),
        "queued_auto_followup_count": len([item for item in active_queued_followups if bool(item.get("auto_continue"))]),
        "usage_stats": dict(usage_stats or {}),
        "turn_denials": turn_denials[-4:],
        "turn_followups": turn_followups[-4:],
        "turn_usage": turn_usage,
        "turn_finish_reason": turn_finish.get("reason"),
        "turn_finish_status": turn_finish.get("status"),
        "turn_resume_source": turn_resume_source,
        "turn_child_count": len(child_turns),
        "latest_child_turn": dict(child_turns[-1]) if child_turns else None,
        "aborted_turn_count": len(aborted_turns),
        "latest_aborted_turn": dict(aborted_turns[-1]) if aborted_turns else None,
    }


def build_followup_context_payload(
    *,
    pending_tool_requests: Optional[List[Dict[str, Any]]],
    queued_followups: Optional[List[Dict[str, Any]]],
    permission_denials: Optional[List[Dict[str, Any]]],
    turn_followups: Optional[List[Dict[str, Any]]] = None,
    turn_denials: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    active_queued_followups = [
        dict(item)
        for item in list(queued_followups or [])
        if not bool(item.get("consumed"))
    ]
    return {
        "pending_tool_requests": list(pending_tool_requests or [])[-4:],
        "queued_followups": active_queued_followups[-4:],
        "recent_permission_denials": list(permission_denials or [])[-4:],
        "turn_followups": list(turn_followups or [])[-4:],
        "turn_denials": list(turn_denials or [])[-4:],
    }


def build_memory_context_payload(
    *,
    session_state: Optional[Any],
) -> Dict[str, Any]:
    if session_state is None:
        return {
            "summary": None,
            "archived_turn_count": 0,
            "fresh_memory_count": 0,
            "stale_memory_count": 0,
            "compact_boundary_turn_id": None,
            "recent_memories": [],
        }
    memory_summary = dict(getattr(session_state, "memory_summary", {}) or {})
    memory_records = list(getattr(session_state, "memory_records", []) or [])
    return {
        "summary": memory_summary.get("summary"),
        "archived_turn_count": int(memory_summary.get("archived_turn_count") or len(memory_records)),
        "fresh_memory_count": int(memory_summary.get("fresh_memory_count") or len([item for item in memory_records if not bool(item.get("stale"))])),
        "stale_memory_count": int(memory_summary.get("stale_memory_count") or len([item for item in memory_records if bool(item.get("stale"))])),
        "compact_boundary_turn_id": memory_summary.get("compact_boundary_turn_id"),
        "recent_memories": list(memory_summary.get("recent_records") or memory_records[-4:]),
    }


def build_engine_context(
    *,
    current_run_dir: Optional[Path],
    current_report: Optional[Dict[str, Any]],
    current_bridge_summary: Optional[Dict[str, Any]],
    current_moire_summary: Optional[Dict[str, Any]],
    current_moire_compare_summary: Optional[Dict[str, Any]],
    current_moire_diffusion_summary: Optional[Dict[str, Any]],
    existing_outputs: List[Tuple[ToolIntent, str]],
    existing_blocks: List[Dict[str, Any]],
    current_outputs: Optional[List[Tuple[ToolIntent, str]]],
    current_blocks: Optional[List[Dict[str, Any]]],
    output_limit: int,
    intent_signature: Callable[[ToolIntent], str],
    truncate_output: Callable[[str, int], str],
    api: Dict[str, Any],
    active_turn_id: Optional[str] = None,
    current_turn: Optional[Dict[str, Any]] = None,
    pending_tool_requests: Optional[List[Dict[str, Any]]] = None,
    permission_denials: Optional[List[Dict[str, Any]]] = None,
    queued_followups: Optional[List[Dict[str, Any]]] = None,
    usage_stats: Optional[Dict[str, Any]] = None,
    current_state: Optional[Any] = None,
    session_state: Optional[Any] = None,
) -> Dict[str, Any]:
    tool_history = collect_tool_context_history(
        existing_outputs=existing_outputs,
        current_outputs=current_outputs,
        existing_blocks=existing_blocks,
        current_blocks=current_blocks,
        output_limit=output_limit,
        intent_signature=intent_signature,
        truncate_output=truncate_output,
    )
    current_context = build_current_context_for_chat(
        current_run_dir=current_run_dir,
        current_report=current_report,
        current_bridge_summary=current_bridge_summary,
        current_moire_summary=current_moire_summary,
        current_moire_compare_summary=current_moire_compare_summary,
        current_moire_diffusion_summary=current_moire_diffusion_summary,
        api=api,
    )
    work_context = build_current_work_context(
        current_run_dir=current_run_dir,
        current_report=current_report,
        current_bridge_summary=current_bridge_summary,
        current_moire_summary=current_moire_summary,
        current_moire_compare_summary=current_moire_compare_summary,
        current_moire_diffusion_summary=current_moire_diffusion_summary,
    )
    turn_context = build_turn_context_payload(
        active_turn_id=active_turn_id,
        current_turn=current_turn,
        pending_tool_requests=pending_tool_requests,
        permission_denials=permission_denials,
        queued_followups=queued_followups,
        usage_stats=usage_stats,
        current_state=current_state,
        session_state=session_state,
    )
    turn_snapshot = None
    if session_state is not None and active_turn_id:
        try:
            turn_snapshot = session_state.resume_turn_state(active_turn_id)
        except Exception:
            turn_snapshot = None
    followup_context = build_followup_context_payload(
        pending_tool_requests=pending_tool_requests,
        queued_followups=queued_followups,
        permission_denials=permission_denials,
        turn_followups=(turn_snapshot or {}).get("followups"),
        turn_denials=(turn_snapshot or {}).get("denials"),
    )
    memory_context = build_memory_context_payload(session_state=session_state)
    tool_evidence = tool_evidence_for_chat(evidence=tool_history)
    return {
        "current_context": current_context,
        "work_context": work_context,
        "turn_context": turn_context,
        "followup_context": followup_context,
        "memory_context": memory_context,
        "tool_history": tool_history,
        "tool_evidence": tool_evidence,
    }


def tool_evidence_for_chat(*, evidence: List[Dict[str, Any]]) -> Optional[str]:
    if not evidence:
        return None
    blocks: List[str] = ["Recent authoritative tool evidence from this session:"]
    for item in evidence:
        output = item.get("output") or item.get("outputPreview") or ""
        blocks.append(
            f"[tool {item['step']}] action={item['action']}\n"
            f"params={json.dumps(item.get('params') or {}, ensure_ascii=False)}\n"
            f"source={item.get('source') or 'unknown'} ok={item.get('ok')}\n"
            f"output=\n{output}"
        )
    return "\n\n".join(blocks)


def tool_backed_response_style() -> str:
    return (
        "When the user is asking about tool-backed work, prefer a short structured answer. "
        "If replying in Chinese, use the headings `结论`, `证据`, and `下一步`. "
        "If replying in English, use `Conclusion`, `Evidence`, and `Next step`. "
        "Keep each section brief and grounded in the available tool evidence. "
        "If the host strategy says some goals are already answered or intentionally deferred, state that briefly and explicitly. "
        "If the host strategy provides followup prompts, surface them as natural next asks."
    )
