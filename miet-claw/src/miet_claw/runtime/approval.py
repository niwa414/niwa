from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..tool_router import ToolIntent
from .permissions import permission_profile_for_intent

_SLASH_COMMAND_BY_ACTION = {
    "draft": "/draft",
    "run": "/run",
    "bridge_kmc_lookup": "/bridge",
    "moire_run": "/moire-run",
    "moire_compare": "/moire-compare",
    "moire_diffusion_sweep": "/moire-diffusion-sweep",
    "open_web": "/open web",
}


@dataclass
class ToolApprovalDecision:
    action: str
    reason: str = ""
    policy: str = "allow_all"
    tool_name: str = ""
    suggested_manual_command: Optional[str] = None


@dataclass(frozen=True)
class ToolApprovalContext:
    active_run_dir: Optional[str] = None
    current_context_kind: Optional[str] = None
    preview_before_run: bool = False
    requested_dry_run: bool = False


def approval_policy_name() -> str:
    value = str(os.environ.get("MIETCLAW_TOOL_APPROVAL_POLICY") or "allow_all").strip().lower()
    if value in {"allow_all", "confirm_mutations", "deny_mutations", "read_only"}:
        return value
    return "allow_all"


def _suggested_manual_command(intent: ToolIntent) -> Optional[str]:
    return _SLASH_COMMAND_BY_ACTION.get(intent.action)


def _scope_label(scope: str) -> str:
    if scope == "plan":
        return "规划/生成工作区"
    if scope == "run":
        return "真实计算执行"
    if scope == "external":
        return "外部动作"
    if scope == "destructive":
        return "高影响动作"
    return "只读查看"

def decide_tool_approval(
    intent: ToolIntent,
    *,
    manual: bool = False,
    context: Optional[ToolApprovalContext] = None,
) -> ToolApprovalDecision:
    profile = permission_profile_for_intent(intent)
    policy = approval_policy_name()
    suggested = _suggested_manual_command(intent)
    context = context or ToolApprovalContext()

    if profile.scope == "read" and profile.read_only and not profile.destructive and not profile.manual_only:
        return ToolApprovalDecision(
            action="allow",
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    scope_label = _scope_label(profile.scope)
    manual_hint = (
        f"如果你确认要执行，请直接使用 `{suggested}`。"
        if suggested
        else "如果你确认要执行，请直接使用对应的 slash command。"
    )
    run_overwrite_requested = (
        profile.scope == "run"
        and isinstance(intent.params, dict)
        and bool(intent.params.get("overwrite_existing", False))
    )

    if run_overwrite_requested and not manual and not (context.preview_before_run or context.requested_dry_run):
        return ToolApprovalDecision(
            action="ask",
            reason=(
                f"`{intent.action}` 这次明确请求了 overwrite_existing，这意味着可能直接覆盖已有 run 目录里的结果。"
                f"为了避免误覆盖，我需要你显式确认。{manual_hint}"
            ),
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    if profile.scope == "run" and not manual and context.active_run_dir and not (context.preview_before_run or context.requested_dry_run):
        return ToolApprovalDecision(
            action="ask",
            reason=(
                f"当前已经有一个 run 上下文（{context.active_run_dir}），新的 `{intent.action}` 可能触发另一轮真实计算。"
                f"为了避免在没有明确确认的情况下继续执行，我先停在这里。{manual_hint}"
            ),
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    if profile.scope == "run" and (context.preview_before_run or context.requested_dry_run) and policy == "confirm_mutations":
        return ToolApprovalDecision(
            action="allow",
            reason="当前会先走 dry-run/preview 验证，所以允许继续。",
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    if profile.manual_only and not manual:
        return ToolApprovalDecision(
            action="ask",
            reason=f"`{intent.action}` 属于{scope_label}，需要显式手动命令触发。{manual_hint}",
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    if policy == "allow_all":
        return ToolApprovalDecision(
            action="allow",
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    if policy == "confirm_mutations":
        if manual:
            return ToolApprovalDecision(
                action="allow",
                policy=policy,
                tool_name=profile.tool_name,
                suggested_manual_command=suggested,
            )
        return ToolApprovalDecision(
            action="ask",
            reason=(
                f"当前权限策略要求对{scope_label}做显式确认，所以我先拦下了 `{intent.action}`。"
                f"{manual_hint} 或者把 MIETCLAW_TOOL_APPROVAL_POLICY 设为 allow_all。"
            ),
            policy=policy,
            tool_name=profile.tool_name,
            suggested_manual_command=suggested,
        )

    return ToolApprovalDecision(
        action="deny",
        reason=(
            f"当前权限策略是只读模式，我不会执行{scope_label}，所以 `{intent.action}` 已被拒绝。"
        ),
        policy=policy,
        tool_name=profile.tool_name,
        suggested_manual_command=suggested,
    )
