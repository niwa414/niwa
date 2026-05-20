from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..tool_router import ToolIntent
from .permissions import permission_scope_for_intent


def intent_from_followup_item(item: Optional[Dict[str, Any]]) -> Optional[ToolIntent]:
    if not item:
        return None
    if str(item.get("kind") or "") != "followup_intent":
        return None
    action = str(item.get("action") or "").strip()
    if not action:
        return None
    params = dict(item.get("params") or {})
    reply = item.get("reply")
    return ToolIntent(action=action, params=params, reply=reply)


def describe_followup_intent(intent: ToolIntent) -> str:
    params = dict(intent.params or {})
    run_label = str(params.get("run") or "当前 run")
    if intent.action == "runs":
        return "继续自动确认最近有哪些 run。"
    if intent.action == "inspect":
        return f"继续自动查看 `{run_label}` 的详细状态。"
    if intent.action == "logs":
        target = str(params.get("target") or "auto")
        return f"继续自动检查 `{run_label}` 的 `{target}` 日志。"
    if intent.action == "artifacts":
        return f"继续自动列出 `{run_label}` 的产物文件。"
    if intent.action == "compare_runs":
        run_a = str(params.get("run_a") or "最近一次 run")
        run_b = str(params.get("run_b") or "上一次 run")
        return f"继续自动比较 `{run_a}` 和 `{run_b}` 的差异。"
    return f"继续自动执行 `{intent.action}`。"


def queued_followup_prompt(item: Dict[str, Any], *, intent: Optional[ToolIntent] = None) -> str:
    text = str(item.get("text") or "").strip()
    if text:
        return text
    resolved_intent = intent or intent_from_followup_item(item)
    if resolved_intent is not None:
        return describe_followup_intent(resolved_intent)
    return "继续执行挂起的 follow-up。"


def resume_source_for_followup(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": "followup",
        "source_turn_id": item.get("source_turn_id"),
        "followup_id": item.get("followup_id"),
        "followup_kind": item.get("kind"),
        "followup_text": item.get("text"),
        "queued_at": item.get("queued_at"),
    }


def format_queued_followups(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "当前没有待处理的 follow-up。"
    lines = ["Queued follow-ups"]
    for item in items:
        followup_id = str(item.get("followup_id") or "—")
        source_turn_id = str(item.get("source_turn_id") or "—")
        kind = str(item.get("kind") or "followup")
        text = str(item.get("text") or "").strip() or "—"
        status = str(item.get("status") or "queued")
        attempt_count = int(item.get("attempt_count") or 0)
        runnable = bool(item.get("runnable"))
        auto_continue = bool(item.get("auto_continue"))
        source_status = str(item.get("source_status") or "").strip()
        source_reason = str(item.get("source_finish_reason") or "").strip()
        lines.append(f"- {followup_id} · {kind} · {status}")
        lines.append(f"  from: {source_turn_id}")
        lines.append(f"  runnable: {runnable} · auto: {auto_continue} · attempts: {attempt_count}")
        if source_status or source_reason:
            summary = " / ".join(part for part in [source_status, source_reason] if part)
            lines.append(f"  source: {summary}")
        lines.append(f"  text: {text}")
    lines.append("可直接输入 follow-up 文本继续，也可以用 `/continue`、`/continue-all`、`/resume <turn-id>` 或 `/retry <turn-id>`。")
    return "\n".join(lines)


def can_auto_continue_followup_item(
    item: Dict[str, Any],
    *,
    intent: Optional[ToolIntent],
    approval_policy: str,
    has_pending_tool_requests: bool,
    max_attempts: int,
) -> bool:
    if not item or bool(item.get("consumed")):
        return False
    if not bool(item.get("runnable")) or not bool(item.get("auto_continue")):
        return False
    if intent is None:
        return False
    scope = permission_scope_for_intent(intent)
    if scope != "read" and approval_policy != "allow_all":
        return False
    if has_pending_tool_requests:
        return False
    attempt_count = int(item.get("attempt_count") or 0)
    return attempt_count < max_attempts
