from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from ..tool_router import ToolIntent, ToolPlan
from .types import AssistantActionBlock, FinalAnswerBlock, ToolRequestBlock

IntentSignatureBuilder = Callable[[ToolIntent], str]


def parse_native_action_blocks(
    raw_content: str,
    *,
    source: str,
    intent_signature: IntentSignatureBuilder,
) -> List[AssistantActionBlock]:
    try:
        payload = json.loads(raw_content)
    except Exception:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("blocks"), list):
        items = payload.get("blocks") or []
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []

    blocks: List[AssistantActionBlock] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tool_requests: List[ToolRequestBlock] = []
        raw_requests = item.get("toolRequests")
        if raw_requests is None:
            raw_requests = item.get("tool_requests")
        if isinstance(raw_requests, dict):
            raw_requests = [raw_requests]
        if not isinstance(raw_requests, list):
            raw_requests = []
        for request_payload in raw_requests:
            if not isinstance(request_payload, dict):
                continue
            intent_payload = request_payload.get("intent") if isinstance(request_payload.get("intent"), dict) else request_payload
            action = intent_payload.get("action")
            if not isinstance(action, str) or not action.strip():
                continue
            params = intent_payload.get("params")
            if not isinstance(params, dict):
                params = {}
            intent = ToolIntent(action=action.strip(), params=dict(params))
            request_id = (
                request_payload.get("requestId")
                or request_payload.get("request_id")
                or intent_signature(intent)
            )
            tool_requests.append(
                ToolRequestBlock(
                    request_id=str(request_id),
                    intent=intent,
                    source=source,
                )
            )

        final_payload = item.get("finalAnswer")
        if final_payload is None:
            final_payload = item.get("final_answer")
        final_answer: Optional[FinalAnswerBlock] = None
        if isinstance(final_payload, dict):
            final_answer = FinalAnswerBlock(reply=str(final_payload.get("reply") or ""), source=source)
        elif isinstance(final_payload, str):
            final_answer = FinalAnswerBlock(reply=final_payload, source=source)

        if not tool_requests and final_answer is None and isinstance(item.get("action"), str):
            action = str(item.get("action") or "").strip()
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            if action:
                intent = ToolIntent(action=action, params=dict(params))
                tool_requests.append(
                    ToolRequestBlock(
                        request_id=intent_signature(intent),
                        intent=intent,
                        source=source,
                    )
                )

        if not tool_requests and final_answer is None:
            continue

        metadata = dict(item.get("metadata") or {})
        metadata.setdefault("native", True)
        blocks.append(
            AssistantActionBlock(
                source=source,
                raw_content=json.dumps(item, ensure_ascii=False),
                tool_requests=tool_requests,
                final_answer=final_answer,
                metadata=metadata,
            )
        )
    return blocks


def synthetic_router_blocks(
    intent: ToolIntent,
    *,
    source: str,
    intent_signature: IntentSignatureBuilder,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[AssistantActionBlock]:
    payload = {
        "synthetic": True,
        "action": intent.action,
        "params": dict(intent.params),
        "reply": intent.reply,
        **dict(metadata or {}),
    }
    if intent.action == "chat":
        return [
            AssistantActionBlock(
                source=source,
                final_answer=FinalAnswerBlock(reply=intent.reply or "", source=source),
                metadata=payload,
            )
        ]
    return [
        AssistantActionBlock(
            source=source,
            tool_requests=[
                ToolRequestBlock(
                    request_id=intent_signature(intent),
                    intent=intent,
                    source=source,
                )
            ],
            metadata=payload,
        )
    ]


def blocks_to_tool_plan(blocks: List[AssistantActionBlock]) -> Optional[ToolPlan]:
    if not blocks:
        return None
    steps: List[ToolIntent] = []
    summarize = False
    reply: Optional[str] = None
    for block in blocks:
        metadata = dict(block.metadata or {})
        summarize = summarize or bool(
            metadata.get("plan_summarize")
            or metadata.get("summarize")
            or metadata.get("planSummarize")
        )
        for request in block.tool_requests or []:
            steps.append(request.intent)
        if reply is None and block.final_answer is not None:
            reply = block.final_answer.reply or None
    if not steps and reply is None:
        return None
    return ToolPlan(steps=steps, summarize=summarize, reply=reply)


def blocks_to_tool_intent(blocks: List[AssistantActionBlock]) -> Optional[ToolIntent]:
    if not blocks:
        return None
    for block in blocks:
        if block.final_answer is not None:
            return ToolIntent(action="chat", params={}, reply=block.final_answer.reply or None)
        if block.tool_requests:
            return block.tool_requests[0].intent
    return None
