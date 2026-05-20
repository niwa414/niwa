from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..tool_router import ToolIntent
from .tool_registry import ToolDefinition, tool_definitions, get_chat_tool_definition
from .types import ToolBudget, ToolExecutionOutcome


@dataclass(frozen=True)
class ToolPermissionProfile:
    tool_name: str
    scope: str = "read"
    read_only: bool = False
    mutating: bool = False
    destructive: bool = False
    manual_only: bool = False
    concurrency_safe: bool = False


def _profile_from_tool(tool: ToolDefinition) -> ToolPermissionProfile:
    payload = tool.permission_profile_payload()
    return ToolPermissionProfile(
        tool_name=str(payload.get("tool_name") or tool.name),
        scope=str(payload.get("scope") or tool.permission_scope),
        read_only=bool(payload.get("read_only", tool.read_only)),
        mutating=bool(payload.get("mutating", tool.mutating)),
        destructive=bool(payload.get("destructive", tool.destructive)),
        manual_only=bool(payload.get("manual_only", tool.manual_only)),
        concurrency_safe=bool(payload.get("concurrency_safe", tool.concurrency_safe)),
    )


def tool_definition_for_intent(intent: ToolIntent) -> Optional[ToolDefinition]:
    return get_chat_tool_definition(intent.action)


def _fallback_permission_scope(intent: ToolIntent) -> str:
    if intent.action in {"draft"}:
        return "plan"
    if intent.action in {"run", "moire_run", "moire_compare", "moire_diffusion_sweep", "bridge_kmc_lookup"}:
        return "run"
    if intent.action in {"open_web"}:
        return "external"
    return "read"


def permission_profile_for_intent(intent: ToolIntent) -> ToolPermissionProfile:
    tool = tool_definition_for_intent(intent)
    if tool is None:
        fallback_scope = _fallback_permission_scope(intent)
        fallback_mutating = fallback_scope in {"plan", "run", "external", "destructive"}
        return ToolPermissionProfile(
            tool_name=intent.action,
            scope=fallback_scope,
            mutating=fallback_mutating,
            read_only=not fallback_mutating,
            destructive=fallback_scope == "destructive",
        )
    return _profile_from_tool(tool)


MUTATING_TOOL_ACTIONS = {
    action
    for tool in tool_definitions(include_internal=True)
    if tool.mutating
    for action in tool.chat_actions
}


def read_bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def build_tool_budget(*, default_steps: int, step_env: Optional[str] = None) -> ToolBudget:
    max_steps = read_bounded_int_env("MIETCLAW_TOOL_MAX_STEPS", default_steps, 1, 8)
    if step_env:
        max_steps = min(max_steps, read_bounded_int_env(step_env, default_steps, 1, 8))
    max_mutating_steps = read_bounded_int_env("MIETCLAW_TOOL_MAX_MUTATIONS", 2, 0, 4)
    max_failures = read_bounded_int_env("MIETCLAW_TOOL_MAX_FAILURES", 2, 1, 4)
    return ToolBudget(
        max_steps=max_steps,
        max_mutating_steps=max_mutating_steps,
        max_failures=max_failures,
    )


def permission_scope_for_intent(intent: ToolIntent) -> str:
    return permission_profile_for_intent(intent).scope


def is_mutating_intent(intent: ToolIntent) -> bool:
    return permission_profile_for_intent(intent).mutating


def tool_budget_context(budget: ToolBudget) -> Dict[str, int]:
    return {
        "max_steps": budget.max_steps,
        "steps_used": budget.steps_used,
        "remaining_steps": budget.remaining_steps,
        "max_mutating_steps": budget.max_mutating_steps,
        "mutating_steps_used": budget.mutating_steps_used,
        "remaining_mutating_steps": budget.remaining_mutating_steps,
        "max_failures": budget.max_failures,
        "failure_count": budget.failures,
    }


def tool_budget_block_note(intent: ToolIntent, budget: ToolBudget) -> Optional[str]:
    profile = permission_profile_for_intent(intent)
    if budget.steps_used >= budget.max_steps:
        return f"为了避免无限工具循环，我把本轮工具预算限制在 {budget.max_steps} 步，已经用完了，所以先基于当前证据给出结论。"
    if profile.mutating and budget.mutating_steps_used >= budget.max_mutating_steps:
        if profile.scope == "plan":
            scope_label = "规划/生成工作区"
        elif profile.scope == "run":
            scope_label = "真实计算执行"
        elif profile.scope in {"external", "destructive"}:
            scope_label = "外部或高影响动作"
        else:
            scope_label = "会改动状态的工具动作"
        return (
            f"为了避免一轮里反复触发{scope_label}，本轮最多允许 {budget.max_mutating_steps} 个会改动状态的工具动作。"
            f"新的 `{intent.action}` 已被拦下，我先根据已有结果继续回答。"
        )
    return None


def tool_failure_budget_note(budget: ToolBudget) -> str:
    return (
        f"本轮工具已经出现 {budget.failures} 次失败或无效结果。"
        f"为了避免继续盲试，我先基于当前拿到的信息总结现状。"
    )


def record_tool_outcome(
    budget: ToolBudget,
    intent: ToolIntent,
    outcome: ToolExecutionOutcome,
    outputs: List[Tuple[ToolIntent, str]],
) -> Optional[str]:
    budget.steps_used += 1
    if is_mutating_intent(intent):
        budget.mutating_steps_used += 1
    outputs.append((intent, outcome.output or ""))
    if outcome.ok:
        return None
    budget.failures += 1
    if budget.failures >= budget.max_failures:
        return tool_failure_budget_note(budget)
    return None
