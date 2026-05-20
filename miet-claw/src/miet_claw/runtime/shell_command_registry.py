from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ShellCommandSpec:
    command: str
    summary: str
    usage: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    hidden: bool = False


SHELL_COMMAND_SPECS: Tuple[ShellCommandSpec, ...] = (
    ShellCommandSpec("/status", "Show the current shell session, model, provider, and latest run context."),
    ShellCommandSpec(
        "/doctor",
        "Check whether the local model, LAMMPS runtime, MoRe case path, and misa-kmc binary are ready.",
    ),
    ShellCommandSpec("/tools", "List the built-in materials tools that this shell can call directly."),
    ShellCommandSpec("/model", "Show or switch the local model used by the shell.", usage="/model [model-id]"),
    ShellCommandSpec(
        "/provider",
        "Switch the planning provider used for draft/run flows.",
        usage="/provider <local|auto|claude>",
    ),
    ShellCommandSpec("/runs", "List recent workflow runs."),
    ShellCommandSpec("/compare", "Compare recent workflow runs."),
    ShellCommandSpec("/inspect", "Inspect one run and its step status.", usage="/inspect <run>"),
    ShellCommandSpec("/artifacts", "List generated artifacts for a run.", usage="/artifacts [run]"),
    ShellCommandSpec(
        "/logs",
        "Read an excerpt from MD, KMC, or summary logs.",
        usage="/logs [run] [md|kmc|summary]",
    ),
    ShellCommandSpec("/followups", "List queued follow-up prompts that can continue a previous turn."),
    ShellCommandSpec(
        "/continue",
        "Run the next runnable follow-up, or one specific follow-up by id.",
        usage="/continue [followup-id]",
    ),
    ShellCommandSpec(
        "/continue-all",
        "Automatically drain multiple runnable follow-ups in sequence.",
        usage="/continue-all [limit]",
    ),
    ShellCommandSpec("/draft", "Draft a workflow from natural language without launching it.", usage="/draft <prompt>"),
    ShellCommandSpec("/run", "Launch a workflow from natural language.", usage="/run <prompt>"),
    ShellCommandSpec(
        "/resume",
        "Resume a previous turn, optionally with a new prompt override.",
        usage="/resume [turn-id|latest] [prompt]",
    ),
    ShellCommandSpec(
        "/retry",
        "Retry a previous turn from its original prompt or a new override.",
        usage="/retry [turn-id|latest] [prompt]",
    ),
    ShellCommandSpec(
        "/bridge",
        "Turn a LAMMPS/NEB result into a KMC lookup table and validate it.",
        usage="/bridge <event.json> <neb.txt> [workdir]",
    ),
    ShellCommandSpec(
        "/moire-run",
        "Run a real MoRe LAMMPS NEB case on this computer, then bridge it into KMC.",
        usage="/moire-run <event.json> <MoRe-case-dir> [workdir]",
    ),
    ShellCommandSpec(
        "/moire-compare",
        "Compare multiple MoRe events on one case, and optionally continue each event into KMC.",
        usage="/moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]",
    ),
    ShellCommandSpec(
        "/moire-diffusion-sweep",
        "Run one MoRe barrier, sweep KMC over temperature, and summarize diffusion coefficient vs temperature.",
        usage="/moire-diffusion-sweep <event.json> <MoRe-case-dir> [workdir]",
    ),
    ShellCommandSpec("/open", "Open the local web console for this agent.", usage="/open web [port]"),
    ShellCommandSpec("/clear", "Clear the current transcript history."),
    ShellCommandSpec("/help", "Show shell help."),
    ShellCommandSpec("/exit", "Exit the shell.", aliases=("/quit",)),
)


def _alias_map(specs: Iterable[ShellCommandSpec] = SHELL_COMMAND_SPECS) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for spec in specs:
        aliases[spec.command] = spec.command
        for alias in spec.aliases:
            aliases[alias] = spec.command
    return aliases


SHELL_COMMAND_ALIASES = _alias_map()


def canonical_shell_command(command: str) -> Optional[str]:
    return SHELL_COMMAND_ALIASES.get(command)


def shell_command_summaries(*, include_hidden: bool = False) -> List[Dict[str, str]]:
    return [
        {"command": spec.usage or spec.command, "summary": spec.summary}
        for spec in SHELL_COMMAND_SPECS
        if include_hidden or not spec.hidden
    ]


def shell_command_names(*, include_aliases: bool = False) -> List[str]:
    if include_aliases:
        return sorted(SHELL_COMMAND_ALIASES)
    return [spec.command for spec in SHELL_COMMAND_SPECS if not spec.hidden]
