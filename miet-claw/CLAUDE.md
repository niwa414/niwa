# CLAUDE.md

This repository is building **mietclaw** as a Claude Code–style local agent shell for multiscale materials simulation.

## Product goal

mietclaw should feel like a terminal agent shell first:
- you chat in natural language
- the shell decides whether to talk, inspect runs, launch workflows, or use bridge/runtime tools
- the default model backend is your **local 27B model**, not a cloud model
- LAMMPS and repo-local `misa-kmc` are treated as domain tools that the shell can call

## What must keep working

1. Local-model chat through `$REPO_ROOT/config/local-agent.json`
2. Native workflow draft/run through `src/miet_claw/autonomy.py`
3. KMC lookup bridge through `src/miet_claw/bridge.py`
4. MoRe full-chain runtime through `src/miet_claw/moire_runtime.py`
5. MCP exposure through `src/miet_claw/mcp_server.py`

## Claude Code–style shell contract

The shell should always expose these concepts clearly:
- **status**: what model, provider, run context, and workspace are active
- **doctor**: whether local model + runtime tools are actually available
- **tools**: what built-in materials actions the shell can take
- **commands**: explicit slash commands still work even when natural language routing fails

## Local runtime on this machine

- local model endpoint: `http://127.0.0.1:8000`
- preferred model alias: `27b`
- conda exec: `conda`
- conda env: `miet-stack`
- repo KMC binary: `$REPO_ROOT/crystalkmc-fix-diffusion-coef/build/bin/misa-kmc`

## Near-term refactor direction

Phase 1:
- strengthen the terminal shell shape
- add explicit shell introspection (`/status`, `/doctor`, `/tools`)
- keep current workflow code intact

Phase 2:
- split shell layer from domain tool layer more cleanly
- make command/tool registration more declarative
- keep MCP and shell backed by the same tool registry
- expose the full local MoRe runtime through that shared registry, not only the lookup bridge

Phase 3:
- let the shell run real LAMMPS -> KMC chains on the local machine with better progress visibility and safer failure handling
