# mietclaw

mietclaw is a specialized **multiscale materials simulation control plane** for your own MD/KMC software stack.

It now has two layers working together:

- a **deterministic execution layer** for LAMMPS + MISA-KMC orchestration
- an **autonomy layer** that can turn a natural-language task into a runnable draft workspace, validate it, launch the job, or automatically route into inspection / bridge tools

It is still not a generic chatbot. It is a domain agent shell for your own scientific software system.

## Open-source notes

This repository publishes the mietclaw agent/orchestration layer. Local native simulation engines, compiled binaries, private MoRe cases, generated run outputs, and API credentials are intentionally not vendored. Configure those paths in `config/local-agent.json` or with environment variables on your own machine.

The project is released under the MIT License. See `LICENSE`.

For examples below, `$REPO_ROOT` means the directory where you cloned this repository.

## What is in this repo

### 1. Python orchestration core
Location: `$REPO_ROOT/src/miet_claw`

Supports:
- `md_only`
- `kmc_only`
- `md_to_kmc_chain`
- state tracking
- resumable runs
- artifact manifests
- explanation summaries
- optional real command execution for both MD and KMC stages when `md.command` / `kmc.command` are available

### 2. Autonomy layer v1
Location: `$REPO_ROOT/src/miet_claw/autonomy.py`

Supports:
- natural-language task -> generated `job spec`
- generated MD NEB / CI-NEB workflow + KMC preview input
- dry-run validation before real launch
- optional provider selection:
  - `local` / `local-heuristic`
  - `claude` / `auto` when Claude Agent SDK is installed and authenticated

Current reality:
- **local mode is verified and working on this machine today**
- **Claude SDK mode is wired in, but only activates after you install/configure the SDK and authentication**
- MD side now scaffolds and executes a **LAMMPS NEB / CI-NEB workflow campaign** with species-specific input decks, parses the resulting barriers, and falls back to prompt/template seeds only if NEB execution degrades

### 3. OpenClaw plugin toolset
Location: `$REPO_ROOT/packages/openclaw-miet-claw-plugin`

Exposes these tools:
- `miet_runtime_doctor`
- `miet_list_runs`
- `miet_inspect_run`
- `miet_get_logs`
- `miet_list_artifacts`
- `miet_autonomy_draft`
- `miet_autonomy_run`
- `miet_plan_job`
- `miet_run_job`
- `miet_kmc_bridge`
- `miet_moire_run`
- `miet_moire_compare`
- `miet_moire_diffusion_sweep`
- `miet_moire_lammps`
- `miet_moire_kmc`
- `miet_resume_job`

### 4. Brand + frontend shell
Locations:
- Design system: `$REPO_ROOT/DESIGN.md`
- Web app: `$REPO_ROOT/apps/web`

The frontend is positioned as a **scientific operations console** with a branded agent workspace UI instead of a chatbot surface.
The visible product name is now **mietclaw** while internal repo and package paths remain `miet-claw` / `miet_claw` for compatibility.
The Create page now includes a **natural-language autonomy lane** that can draft or launch jobs directly from the UI.

## Quick commands

### Python orchestration
```bash
PYTHONPATH=src python3 -m miet_claw.cli plan examples/jobs/md_to_kmc_chain.json
PYTHONPATH=src python3 -m miet_claw.cli run examples/jobs/md_to_kmc_chain.json --dry-run --output-dir runs
```

### Natural-language autonomy draft
```bash
PYTHONPATH=src python3 -m miet_claw.cli autonomy-draft \
  'Create a native MD to KMC vacancy diffusion job for "Fe-Cu-Ni autonomy demo" at 820 K owned by miet.' \
  --provider local \
  --workspace-root .autonomy-checks
```

This writes a generated workspace containing:
- `job_spec.generated.json`
- `md/generated_md_neb_workflow.py`
- `md/neb/neb_campaign.json`
- `md/neb/<species>/in.neb.ci.lmp`
- `kmc/generated_kmc.preview.in`
- `scripts/plan.sh`
- `scripts/dry_run.sh`
- `scripts/run.sh`
- `autonomy_notes.md`
- `autonomy_report.json`

### Natural-language autonomy run
```bash
PYTHONPATH=src python3 -m miet_claw.cli autonomy-run \
  'Create a native MD to KMC vacancy diffusion job for "Autonomy Native Run Demo" at 799 K owned by miet.' \
  --provider local \
  --workspace-root .autonomy-checks \
  --output-dir runs
```

This does three things in one command:
1. drafts a workspace from the prompt
2. runs a dry-run validation
3. launches the real run unless you pass `--dry-run-only`

### Optional Claude Agent SDK mode
If you want the autonomy layer to use Claude-style agent reasoning instead of the local heuristic mode:

```bash
pip install claude-agent-sdk
export ANTHROPIC_API_KEY=...

PYTHONPATH=src python3 -m miet_claw.cli autonomy-draft \
  'Create a multiscale vacancy diffusion job for Fe-Cu-Ni at 900 K.' \
  --provider claude
```

The CLI will only accept `--provider claude` when both are true:
- `claude-agent-sdk` is installed
- one of these auth env vars is set:
  - `ANTHROPIC_API_KEY`
  - `CLAUDE_CODE_USE_BEDROCK`
  - `CLAUDE_CODE_USE_VERTEX`
  - `CLAUDE_CODE_USE_FOUNDRY`

### Native-stack examples on this machine
These examples use the local Miniforge environment and the compiled MISA-KMC binary:

```bash
PYTHONPATH=src python3 -m miet_claw.cli run examples/jobs/md_only_native.json --output-dir runs
PYTHONPATH=src python3 -m miet_claw.cli run examples/jobs/kmc_only_native.json --output-dir runs
PYTHONPATH=src python3 -m miet_claw.cli run examples/jobs/md_to_kmc_chain_native.json --output-dir runs
```

Native runtime currently expects:
- Conda env: `/path/to/conda/envs/miet-stack`
- Conda entrypoint: `conda`
- KMC binary: `$REPO_ROOT/crystalkmc-fix-diffusion-coef/build/bin/misa-kmc`
- Native KMC examples inject `DYLD_LIBRARY_PATH` / `DYLD_FALLBACK_LIBRARY_PATH` so the compiled binary can find the local LAMMPS/MPI libraries on macOS

### OpenClaw plugin
```bash
openclaw plugins install -l ./packages/openclaw-miet-claw-plugin
openclaw plugins enable miet-claw-sim
openclaw config set plugins.allow '["miet-claw-sim"]' --strict-json
openclaw config set tools.allow '["miet-claw-sim"]' --strict-json
```

If you want OpenClaw to use the local verified 27B model and also see the local mietclaw MCP server, use the helper script:

```bash
./scripts/configure_openclaw_27b_mcp.sh
```

That updates `$HOME/.openclaw/openclaw.json` to:
- prefer `omlx/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit`
- register a compatible `mietclaw` MCP definition under the bundled `acpx` runtime plugin
- enable the bundled `acpx` runtime plugin without turning on OpenClaw’s top-level bundle-mcp auto-probe

### Web frontend
The web app is now a minimal, chat-first mietclaw surface: ordinary messages first go through a tool router, and then either:

- auto-trigger a real MD/KMC tool action when the intent is clear
- or fall back to the local model for ordinary conversation

```bash
npm install
npm run build:web
npm run preview --workspace @miet-claw/web
```

### Terminal-first Claude Code–style shell
If you want a Claude Code-style entry point on macOS / Linux:

```bash
./scripts/install_mietclaw.sh
mietclaw
```

`mietclaw` now carries its own local-model profile in:

- `$REPO_ROOT/config/local-agent.json`

By default that profile points at your local server, prefers the local 27B model, and also records the local runtime profile: Conda/LAMMPS settings, MoRe case path, repo KMC binary, EAM file, bridge script, diffusion sweep defaults, and optional KMC retry attempts.

If you want to force the local 27B model explicitly:

```bash
./scripts/install_mietclaw_27b.sh
mietclaw-27b
```

You can also ask the tool itself what local model profile it is using:

```bash
cd $REPO_ROOT
PYTHONPATH=src python3 -m miet_claw.cli local-status
PYTHONPATH=src python3 -m miet_claw.cli local-self-check
PYTHONPATH=src python3 -m miet_claw.cli router-golden-eval --golden-file examples/evals/router_golden.json
PYTHONPATH=src python3 -m miet_claw.cli runtime-golden-eval --golden-file examples/evals/runtime_health_golden.json
```

The router golden eval covers direct tool routing for runs, bridge, MoRe run/compare/diffusion, web console, and latest-run comparison; the runtime golden eval covers KMC success and failure health checks.

This opens a Claude Code–style local shell where:
- ordinary messages first try to auto-route into the right tool
- ordinary messages can also trigger a multi-step tool plan when the task clearly needs several actions
- if no tool is needed, the message falls back to the local model
- `/status`, `/doctor`, and `/tools` make the shell state and tool inventory explicit
- the shell `/tools` view now comes from the same shared tool catalog that the MCP server exposes
- `/draft`, `/run`, and `/bridge` still remain available as explicit commands
- `/runs`, `/inspect`, `/artifacts`, and `/logs` stay available for inspection
- direct `MoRe` runs now print stage-by-stage progress in plain terminal mode
- direct `MoRe` runs now explicitly show when the shell is calling the local MCP tools for `LAMMPS` and `KMC`
- direct `MoRe` runs now regenerate a fresh LAMMPS NEB input in the workdir, and if you do not provide an event file they auto-generate a KMC seed event from the local `data.lmp`, then regenerate a repo-compatible KMC state/input pair and run the repo `misa-kmc` binary
- direct `MoRe` runs can now repeat the repo KMC stage with multiple random seeds in one call, and they write ensemble statistics into the final summary
- direct `MoRe` runs can now optionally try an OVITO snapshot render for each completed KMC seed run; if OVITO is unavailable, the run still completes and reports that the visualization step was skipped
- multi-seed repo KMC runs now also write a seed comparison chart (`kmc_seed_comparison.svg`) and, when at least two seed snapshots exist, an animated GIF (`kmc_seed_animation.gif`)
- direct `MoRe` runs now also materialize LAMMPS-side visualization inputs in the workdir: `ovito_initial.xyz`, `ovito_final.xyz`, and, when OVITO is available, matching PNG snapshots
- bridge and MoRe summaries now include a stricter runtime health check, not just a raw return code
- plain mode is the default, with TUI still optional when you want panels

Useful terminal commands:

```bash
mietclaw
mietclaw-27b
zsh -lic 'mietclaw --ui tui'
zsh -lic 'mietclaw --ui plain'
PYTHONPATH=src python3 -m miet_claw.cli chat --model 27b --once '你好，简单介绍一下你自己。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once 'Create a native MD to KMC vacancy diffusion job for "Terminal demo" at 805 K owned by miet.'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请帮我搭一个 LAMMPS 计算迁移能垒，然后把结果传给代码仓库里的 KMC 软件继续模拟的工作流。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请直接运行一个 native LAMMPS 迁移能垒计算，并把结果传给代码仓库里的 KMC 软件继续模拟；材料名用 "Barrier Chain Demo"，温度 813 K，owner=miet。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请列出最近的 runs。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '帮我判断最新那个 run 是正常结束还是异常退出，如果有必要再看 KMC 日志。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请把 /abs/path/event.json 和 /abs/path/neb.txt bridge 成 KMC lookup，到 /abs/path/workdir 并验证一下。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC：/abs/path/MoRe_case_dir /abs/path/workdir'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC：/abs/path/event.json /abs/path/MoRe_case_dir /abs/path/workdir'
PYTHONPATH=src python3 -m miet_claw.cli moire-run /abs/path/MoRe_case_dir --workdir /abs/path/workdir --validate
PYTHONPATH=src python3 -m miet_claw.cli moire-run /abs/path/MoRe_case_dir --event-json /abs/path/event.json --workdir /abs/path/workdir --validate
PYTHONPATH=src python3 -m miet_claw.cli moire-run /abs/path/MoRe_case_dir --workdir /abs/path/workdir --kmc-seeds 3401,3402,3403 --validate
PYTHONPATH=src python3 -m miet_claw.cli moire-run /abs/path/MoRe_case_dir --event-json /abs/path/event.json --workdir /abs/path/workdir --kmc-seeds 3401,3402,3403 --ovito --validate
PYTHONPATH=src python3 -m miet_claw.cli doctor
PYTHONPATH=src python3 -m miet_claw.cli tools
PYTHONPATH=src python3 -m miet_claw.cli runs
PYTHONPATH=src python3 -m miet_claw.cli inspect real_neb_archive_verification
```

The terminal app now supports these in-session commands:

```text
/status
/doctor
/tools
/runs
/inspect <run>
/artifacts [run]
/logs [run] [md|kmc|summary]
/open web [port]
/draft <prompt>
/run <prompt>
/bridge <event.json> <neb.txt> [workdir]
/moire-run <MoRe-case-dir> [workdir]
/moire-run <event.json> <MoRe-case-dir> [workdir]
/provider <local|auto|claude>
/exit
```

If you want the shell design target and migration plan in one place, see:

- `$REPO_ROOT/CLAUDE.md`
- `$REPO_ROOT/docs/CLAUDE_CODE_STYLE_SHELL.md`

Natural-language auto routing now covers these core actions:

- list recent runs
- inspect the latest run
- show artifacts or logs
- draft a new MD→KMC workflow
- launch a workflow
- bridge `event.json + neb.txt` into a KMC lookup file and validate it with `misa-kmc`

Natural-language multi-step execution now covers flows like:

- inspect latest run → open KMC log → summarize what happened
- list recent runs → inspect the newest one → explain the status
- inspect a run → show artifacts/logs → summarize next steps

Examples:

```text
请列出最近的 runs。
请检查最新那个 run 的 KMC 日志并总结一下。
先列出最近的 runs，再看最新那个 run 的 KMC 日志，并总结一下。
帮我起草一个 native MD to KMC vacancy diffusion job for "demo" at 800 K owned by miet。
请把 /abs/path/event.json 和 /abs/path/neb.txt bridge 成 KMC lookup，到 /abs/path/workdir 并验证一下。
```

### Local MCP server

If you want your own local model framework to call mietclaw through MCP instead of slash commands, start the stdio MCP server:

```bash
cd $REPO_ROOT
./scripts/install_mietclaw_mcp.sh
mietclaw-mcp
```

To register that server into the local Codex client automatically:

```bash
cd $REPO_ROOT
./scripts/connect_codex_mcp.sh
```

Or without installing a launcher:

```bash
cd $REPO_ROOT
PYTHONPATH=src python3 -m miet_claw.cli mcp-server
```

The local MCP server exposes these tools:

- `miet_runtime_doctor`
- `miet_list_runs`
- `miet_inspect_run`
- `miet_get_logs`
- `miet_list_artifacts`
- `miet_autonomy_draft`
- `miet_autonomy_run`
- `miet_plan_job`
- `miet_run_job`
- `miet_kmc_bridge`
- `miet_moire_run`
- `miet_moire_compare`
- `miet_moire_diffusion_sweep`
- `miet_moire_lammps`
- `miet_moire_kmc`

Example MCP config snippet:

```json
{
  "mcpServers": {
    "mietclaw": {
      "command": "$REPO_ROOT/bin/mietclaw-mcp"
    }
  }
}
```

### Local model backend

The chat shell is designed to work with your own local OpenAI-compatible model server.

Current recommended model on this machine:

```text
Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit
```

Default endpoint:

```text
http://127.0.0.1:8000
```

Override with:

```bash
export MIETCLAW_LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000
export MIETCLAW_LOCAL_MODEL_API_KEY=your-local-key
export MIETCLAW_LOCAL_MODEL=27b
```

The terminal chat layer uses the local model for:

- ordinary conversation
- tool-routing fallback when heuristics are not enough
- stronger multi-step agent reasoning on top of the deterministic MD/KMC tools

You can also override model selection per role:

```bash
export MIETCLAW_CHAT_MODEL=27b
export MIETCLAW_ROUTER_MODEL=27b
export MIETCLAW_PLAN_MODEL=27b
export MIETCLAW_AGENT_MODEL=27b
export MIETCLAW_SUMMARY_MODEL=27b
```

For local MCP + 27B setup details, see:

- `$REPO_ROOT/docs/LOCAL_27B_MCP_SETUP.md`
- `$REPO_ROOT/examples/mcp/mietclaw-local-27b.example.json`

## Verified outputs on this machine

A verified autonomy real run now writes outputs like:

- `$REPO_ROOT/runs/autonomy_native_run_demo/state.json`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/md/barriers.json`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/kmc/generated_kmc.in`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/kmc/diffusion.csv`
- `$REPO_ROOT/runs/autonomy_native_run_demo/explain/summary.md`

A generated autonomy workspace writes files like:

- `$REPO_ROOT/.autonomy-checks/<job-id>-<timestamp>/job_spec.generated.json`
- `$REPO_ROOT/.autonomy-checks/<job-id>-<timestamp>/md/generated_md_neb_workflow.py`
- `$REPO_ROOT/.autonomy-checks/<job-id>-<timestamp>/kmc/generated_kmc.preview.in`
- `$REPO_ROOT/.autonomy-checks/<job-id>-<timestamp>/autonomy_report.json`

## Validation commands used

```bash
python3 -m py_compile src/miet_claw/*.py tests/test_miet_claw.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m miet_claw.cli chat --model 27b --once '你好，简单介绍一下你自己。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请列出最近的 runs。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '帮我判断最新那个 run 是正常结束还是异常退出，如果有必要再看 KMC 日志。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请检查最新那个 run 的 KMC 日志并总结一下。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '先列出最近的 runs，再看最新那个 run 的 KMC 日志，并总结一下。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请帮我起草一个 native MD to KMC vacancy diffusion job for "Auto Tool Draft Demo 2" at 806 K owned by miet。'
PYTHONPATH=src python3 -m miet_claw.cli chat --once '请把 /path/to/kmc-ml/crystalkmc-fix-diffusion-coef/build/validation_lammps_lookup_bridge_20260408/event.json 和 /path/to/kmc-ml/soap-KMC/NEB_new_data/ReMo/Re_0.08/model_4/neb.txt bridge 成 KMC lookup，到 $REPO_ROOT/runs/auto_tool_bridge_demo_20260409 并验证一下。'
PYTHONPATH=src python3 -m miet_claw.cli mcp-server
npm run test:plugin
npm run build:web
openclaw plugins inspect miet-claw-sim --json
```

## Key docs

- Architecture: `$REPO_ROOT/docs/ARCHITECTURE.md`
- Autonomy layer: `$REPO_ROOT/docs/AUTONOMY_LAYER.md`
- OpenClaw integration: `$REPO_ROOT/docs/OPENCLAW_INTEGRATION.md`
- Design system: `$REPO_ROOT/DESIGN.md`
