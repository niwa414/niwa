from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..tool_router import (
    ToolIntent,
    parse_agent_decision,
    parse_tool_intent,
    parse_tool_plan,
)
from .types import (
    AssistantActionBlock,
    FinalAnswerBlock,
    ToolRequestBlock,
    ToolResponseStrategy,
    ToolResultBlock,
    ToolTurnState,
)


def build_local_model_messages(
    *,
    system_prompt: str,
    compact_history: List[Dict[str, str]],
    current_context: Optional[str] = None,
    memory_context: Optional[str] = None,
    tool_evidence: Optional[str] = None,
    tool_response_style: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if current_context:
        messages.append({"role": "system", "content": current_context[:6000]})
    if memory_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following working-memory summary condenses older conversation state. "
                    "Treat it as helpful archived context, but prefer fresher turn state and current tool evidence when they conflict.\n\n"
                    + memory_context[:5000]
                ),
            }
        )
    if tool_evidence:
        messages.append(
            {
                "role": "system",
                "content": (
                    "The following tool evidence is authoritative session context. "
                    "Use it when it helps answer the user's next question, and do not contradict it.\n\n"
                    + tool_evidence[:7000]
                ),
            }
        )
        if tool_response_style:
            messages.append({"role": "system", "content": tool_response_style})
    messages.extend(compact_history)
    return messages


def build_tool_router_messages(*, system_prompt: str, context: Dict[str, Any], prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": json.dumps(context, ensure_ascii=False)},
        {"role": "user", "content": prompt},
    ]


def build_tool_plan_messages(*, system_prompt: str, context: Dict[str, Any], prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": json.dumps(context, ensure_ascii=False)},
        {"role": "user", "content": prompt},
    ]


def build_agent_loop_messages(
    *,
    system_prompt: str,
    context: Dict[str, Any],
    prompt: str,
    tool_history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": json.dumps(context, ensure_ascii=False)},
        {
            "role": "user",
            "content": json.dumps(
                {"user_request": prompt, "completed_tool_steps": tool_history},
                ensure_ascii=False,
            ),
        },
    ]


def build_tool_summary_messages(
    *,
    original_prompt: str,
    tool_blocks: List[str],
    host_strategy: Dict[str, Any],
    tool_response_style: str,
    note: Optional[str] = None,
) -> List[Dict[str, str]]:
    user_content = f"User request:\n{original_prompt}\n\nTool outputs:\n\n" + "\n\n".join(tool_blocks)
    user_content += "\n\nHost strategy:\n" + json.dumps(host_strategy, ensure_ascii=False)
    if note:
        user_content += f"\n\nAdditional note:\n{note}"
    return [
        {
            "role": "system",
            "content": (
                "You are mietclaw. The tool outputs below are authoritative. "
                "Answer the user's request in clear Chinese unless the user used another language. "
                "Be concise, factual, and do not invent anything that is not in the tool outputs. "
                + tool_response_style
            ),
        },
        {"role": "user", "content": user_content},
    ]


def tool_result_summary_entries(blocks: Sequence[ToolResultBlock], *, truncate_output: Any) -> List[str]:
    entries: List[str] = []
    for idx, block in enumerate(blocks, start=1):
        entries.append(
            f"[step {idx}] action={block.intent.action}\n"
            f"params={json.dumps(block.intent.params, ensure_ascii=False)}\n"
            f"ok={block.ok}\n"
            f"output=\n{truncate_output(block.output, 2800)}"
        )
    return entries


def trace_tool_result_blocks(state: ToolTurnState) -> List[ToolResultBlock]:
    return [
        event.block
        for event in state.trace.events
        if getattr(event, "kind", "") == "tool_result_block"
    ]

def _tool_request_block(session: Any, intent: ToolIntent, *, source: str) -> ToolRequestBlock:
    return ToolRequestBlock(request_id=session._intent_signature(intent), intent=intent, source=source)


def legacy_agent_reply_to_action_block(session: Any, raw_content: str, *, source: str) -> Optional[AssistantActionBlock]:
    decision = parse_agent_decision(raw_content)
    if not decision:
        return None
    tool_requests: List[ToolRequestBlock] = []
    final_answer: Optional[FinalAnswerBlock] = None
    if decision.step is not None and decision.status == "continue":
        tool_requests.append(_tool_request_block(session, decision.step, source=source))
    if decision.status == "finish":
        final_answer = FinalAnswerBlock(reply=decision.reply or "", source=source)
    metadata = {
        "status": decision.status,
        "reply": decision.reply,
        "step_action": decision.step.action if decision.step else None,
    }
    return AssistantActionBlock(
        source=source,
        raw_content=raw_content,
        tool_requests=tool_requests,
        final_answer=final_answer,
        metadata=metadata,
    )


def legacy_router_reply_to_action_block(session: Any, raw_content: str, *, source: str) -> Optional[AssistantActionBlock]:
    intent = parse_tool_intent(raw_content)
    if not intent:
        return None
    tool_requests: List[ToolRequestBlock] = []
    final_answer: Optional[FinalAnswerBlock] = None
    if intent.action == "chat":
        final_answer = FinalAnswerBlock(reply=intent.reply or "", source=source)
    else:
        tool_requests.append(_tool_request_block(session, intent, source=source))
    return AssistantActionBlock(
        source=source,
        raw_content=raw_content,
        tool_requests=tool_requests,
        final_answer=final_answer,
        metadata={"action": intent.action, "params": intent.params, "reply": intent.reply},
    )


def legacy_plan_reply_to_action_blocks(session: Any, raw_content: str, *, source: str) -> List[AssistantActionBlock]:
    plan = parse_tool_plan(raw_content)
    if not plan:
        return []
    blocks: List[AssistantActionBlock] = []
    for step in plan.steps:
        blocks.append(
            AssistantActionBlock(
                source=source,
                raw_content=raw_content,
                tool_requests=[_tool_request_block(session, step, source=source)],
                metadata={"plan_summarize": bool(plan.summarize), "reply": plan.reply, "step_action": step.action},
            )
        )
    if not blocks and plan.reply:
        blocks.append(
            AssistantActionBlock(
                source=source,
                raw_content=raw_content,
                final_answer=FinalAnswerBlock(reply=plan.reply, source=source),
                metadata={"plan_summarize": bool(plan.summarize), "reply": plan.reply},
            )
        )
    return blocks


def normalize_legacy_block_metadata(
    blocks: Sequence[AssistantActionBlock],
    *,
    mode: str,
) -> List[AssistantActionBlock]:
    normalized: List[AssistantActionBlock] = []
    for block in blocks:
        metadata = dict(block.metadata or {})
        metadata.setdefault("legacy", True)
        metadata.setdefault("legacyMode", mode)
        metadata.setdefault("toolRequestCount", len(block.tool_requests or []))
        if block.final_answer is not None:
            metadata.setdefault("hasFinalAnswer", True)
        normalized.append(
            AssistantActionBlock(
                source=block.source,
                raw_content=block.raw_content,
                tool_requests=list(block.tool_requests or []),
                final_answer=block.final_answer,
                metadata=metadata,
            )
        )
    return normalized


def legacy_reply_to_blocks(
    session: Any,
    raw_content: str,
    *,
    source: str,
    mode: str,
) -> List[AssistantActionBlock]:
    if mode == "agent":
        block = legacy_agent_reply_to_action_block(session, raw_content, source=source)
        blocks = [block] if block is not None else []
    elif mode == "router":
        block = legacy_router_reply_to_action_block(session, raw_content, source=source)
        blocks = [block] if block is not None else []
    elif mode == "plan":
        blocks = legacy_plan_reply_to_action_blocks(session, raw_content, source=source)
    else:
        blocks = []
    return normalize_legacy_block_metadata(blocks, mode=mode)


def maybe_extend_tool_turn(session: Any, prompt: str, state: ToolTurnState) -> ToolResponseStrategy:
    strategy = session._response_strategy_for_prompt(prompt, state.outputs)
    while strategy.followup_intent is not None and state.budget.remaining_steps > 0:
        session._add_tool_turn_note(state, strategy.reason)
        _, should_stop = session._apply_tool_intent_to_turn(
            state,
            strategy.followup_intent,
            prompt,
            source="host_followup_strategy",
        )
        if should_stop:
            break
        next_strategy = session._response_strategy_for_prompt(prompt, state.outputs)
        if (
            next_strategy.followup_intent is None
            or session._intent_signature(next_strategy.followup_intent)
            == session._intent_signature(strategy.followup_intent)
        ):
            strategy = next_strategy
            break
        strategy = next_strategy
    return strategy

def run_agent_query_loop(session: Any, prompt: str, *, state: Optional[ToolTurnState] = None) -> Optional[str]:
    return session._run_agent_loop(prompt, state=state)


def run_engine_turn(
    session: Any,
    prompt: str,
    *,
    state: Optional[ToolTurnState] = None,
) -> Tuple[Optional[str], ToolTurnState, bool]:
    return session._run_engine_turn(prompt, state=state)


def run_tool_event_loop(session: Any, prompt: str, *, state: Optional[ToolTurnState] = None) -> Optional[str]:
    reply, _, _ = run_engine_turn(session, prompt, state=state)
    return reply
