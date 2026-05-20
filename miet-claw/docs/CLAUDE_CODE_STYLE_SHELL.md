# Claude Code–style shell migration for mietclaw

This document defines how `mietclaw` should evolve from a materials workflow launcher into a **Claude Code–style local agent shell**.

## Target architecture

```text
User
  ↓
Terminal shell (`bin/mietclaw`, `src/miet_claw/chat.py`)
  ↓
Shell reasoning + command layer (`src/miet_claw/tool_router.py`, shell commands)
  ↓
Tool adapters
  ├─ Autonomy draft/run (`src/miet_claw/autonomy.py`)
  ├─ KMC lookup bridge (`src/miet_claw/bridge.py`)
  ├─ MoRe runtime (`src/miet_claw/moire_runtime.py`)
  ├─ Runs / inspect / logs helpers (`src/miet_claw/chat.py`)
  └─ MCP adapter (`src/miet_claw/mcp_server.py`)
  ↓
Local runtime
  ├─ local 27B model server
  ├─ LAMMPS in `miet-stack`
  └─ repo-local `misa-kmc`
```

## Why this refactor

The old framing was “materials workflow app with a chat feature”.
The new framing is:

- `mietclaw` is the **agent shell**
- LAMMPS / KMC / bridge are **tools** behind that shell
- the default model backend is **local**, not Anthropic-hosted

## Phase 1 done in this repo

1. Keep the existing shell entrypoint (`bin/mietclaw`) as the main product surface.
2. Add shell introspection primitives:
   - `/status`
   - `/doctor`
   - `/tools`
3. Add CLI equivalents:
   - `python -m miet_claw.cli doctor`
   - `python -m miet_claw.cli tools`
4. Keep all existing materials workflow codepaths available and compatible.

## Phase 2 now landed

1. Use the shared tool registry in `src/miet_claw/runtime/tool_registry.py`.
2. Drive the shell `/tools` view and the MCP `tools/list` response from the same source.
3. Expose runtime readiness, run inspection, autonomy, bridge, and full MoRe tools through that shared tool catalog.
4. Move slash commands into `src/miet_claw/runtime/shell_command_registry.py` plus handler dispatch in `src/miet_claw/frontends/shell_commands.py`, instead of one long if/else chain.

## What stays where

### Shell layer
- `src/miet_claw/chat.py`
- `src/miet_claw/cli.py`
- `src/miet_claw/tool_router.py`
- `src/miet_claw/shell_runtime.py`

### Model/profile layer
- `src/miet_claw/local_profile.py`
- local model HTTP request helpers in `src/miet_claw/chat.py`
- `$REPO_ROOT/config/local-agent.json`
- the same profile now also carries local runtime paths and defaults for Conda/LAMMPS, MoRe, repo KMC, bridge, diffusion sweeps, and configured KMC retry attempts

### Domain tool layer
- `src/miet_claw/autonomy.py`
- `src/miet_claw/bridge.py`
- `src/miet_claw/moire_runtime.py`
- `src/miet_claw/executor.py`
- `src/miet_claw/transforms.py`

### External interface layer
- `src/miet_claw/mcp_server.py`

## Preservation rules

To avoid wasting the work already done, the shell refactor must not remove:

- the verified local 27B model path
- the verified bridge validation path
- the verified native draft/run path
- the MoRe full-chain runtime path now being added

Instead, the refactor should make those look like **built-in tools of the shell**.

## Phase 2 next

- improve progress reporting for long-running local jobs

## Phase 3 progress now landed

- MoRe local runs now emit stage-by-stage progress in plain shell mode.
- Repo KMC completion checks now require a clean return code, Loop time, final stats row, positive final time, and no fatal output markers.
- Router behavior now has a checked golden eval fixture at `examples/evals/router_golden.json`, runnable with `python -m miet_claw.cli router-golden-eval`; it now covers basic runs, bridge, MoRe run, MoRe compare, diffusion sweep, web console, and latest-run comparison routing.
- Runtime-health behavior now has a checked golden eval fixture at `examples/evals/runtime_health_golden.json`, runnable with `python -m miet_claw.cli runtime-golden-eval`.
- MoRe KMC runtime-health parsing moved into `src/miet_claw/moire_health.py`, MoRe event/lattice state modeling moved into `src/miet_claw/moire_event_model.py`, MoRe dynamic LAMMPS case generation moved into `src/miet_claw/moire_lammps_case_builder.py`, MoRe visualization input/render helpers moved into `src/miet_claw/moire_visualization.py`, MoRe seed/stat/temperature helpers moved into `src/miet_claw/moire_stats.py`, MoRe SVG/GIF plot helpers moved into `src/miet_claw/moire_plots.py`, autonomy NEB campaign drafting moved into `src/miet_claw/autonomy_neb.py`, runtime snapshot message/card assembly moved into `src/miet_claw/runtime/snapshot_messages.py`, run inspection/listing helpers moved into `src/miet_claw/run_inspection.py`, chat report formatting moved into `src/miet_claw/chat_reports.py`, chat evidence/follow-up strategy moved into `src/miet_claw/chat_strategy.py`, local model/web-console client helpers moved into `src/miet_claw/local_model_client.py`, and query-engine block/follow-up helpers moved into `src/miet_claw/runtime/query_blocks.py` and `src/miet_claw/runtime/query_followups.py`, keeping the larger runtime/chat files smaller and more testable.
- Repo KMC can add configured retry seeds when only one seed is requested, so transient seed-specific failures can be diagnosed without silently pretending the first attempt succeeded.
- `run_kmc_lookup_bridge()` now augments bridge summaries with runtime health checks:
  - whether `run.out` exists
  - whether `Loop time` appeared
  - whether `MPI_ABORT` appeared
  - whether lookup hits were actually observed
- direct natural-language prompts with `event.json + MoRe case dir [+ workdir]` can now route into the real local `moire_run` execution path instead of falling back to a generic bridge.
