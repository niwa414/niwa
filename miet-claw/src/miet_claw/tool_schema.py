from __future__ import annotations

from typing import Any, Dict, List

from .runtime.tool_registry import TOOL_SPECS, mcp_tool_definitions, shell_tool_summaries

__all__ = [
    "TOOL_SPECS",
    "mcp_tool_definitions",
    "shell_tool_summaries",
]


def tool_specs() -> List[Dict[str, Any]]:
    return list(TOOL_SPECS)
