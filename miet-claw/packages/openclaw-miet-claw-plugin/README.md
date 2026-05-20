# mietclaw OpenClaw Plugin

This plugin now exposes sixteen OpenClaw tools for the mietclaw orchestration layer:

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

## What the autonomy tools do

### `miet_autonomy_draft`
Turns a natural-language materials simulation request into a generated workspace containing:
- `job_spec.generated.json`
- generated MD NEB / CI-NEB workflow with real barrier extraction when executables are available
- KMC preview input
- plan / dry-run / run shell scripts
- autonomy notes and report

### `miet_autonomy_run`
Builds the same workspace, then:
1. runs a dry-run validation
2. launches the real job unless `dry_run: true`

This is the easiest way to make OpenClaw behave like a specialized materials simulation agent instead of a plain tool launcher.

## Inspection + bridge tools

The plugin also exposes read-side tools that make the local OpenClaw agent much more usable in practice:

- `miet_list_runs` — list recent runs
- `miet_inspect_run` — inspect one run snapshot
- `miet_get_logs` — read `md`, `kmc`, or `summary` log excerpts
- `miet_list_artifacts` — list archived files
- `miet_kmc_bridge` — turn `event.json + neb.txt` into `barriers.tsv` and optionally validate the lookup hit with `misa-kmc`
- `miet_runtime_doctor` — check local model, LAMMPS, MoRe, and KMC readiness
- `miet_moire_run` / `miet_moire_compare` / `miet_moire_diffusion_sweep` — run the full local MoRe→KMC workflows
- `miet_moire_lammps` / `miet_moire_kmc` — run each MoRe stage separately when you need a narrower operation

## Local install

```bash
openclaw plugins install -l ./packages/openclaw-miet-claw-plugin
openclaw plugins enable miet-claw-sim
```

## Recommended config

```json5
{
  agents: {
    defaults: {
      model: {
        primary: "omlx/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit"
      }
    }
  },
  mcp: {
    servers: {
      mietclaw: {
        command: "$REPO_ROOT/bin/mietclaw-mcp"
      }
    }
  },
  plugins: {
    allow: ["miet-claw-sim", "acpx"],
    entries: {
      "miet-claw-sim": {
        enabled: true,
        config: {
          projectRoot: "$REPO_ROOT",
          pythonBin: "python3",
          defaultOutputDir: "$REPO_ROOT/runs"
        }
      },
      acpx: {
        enabled: true,
        config: {
          mcpServers: {
            mietclaw: {
              command: "/usr/bin/python3",
              args: ["-m", "miet_claw.cli", "mcp-server", "--project-root", "$REPO_ROOT", "--workspace-root", "$REPO_ROOT/.autonomy-mcp", "--output-dir", "$REPO_ROOT/runs", "--provider", "local"],
              env: {
                PYTHONPATH: "$REPO_ROOT/src"
              }
            }
          }
        }
      }
    }
  },
  tools: {
    allow: ["miet-claw-sim"]
  }
}
```

## Important notes

- `provider: "local"` is verified and works on this machine today.
- `provider: "claude"` is supported only after `claude-agent-sdk` is installed and authenticated.
- The current autonomy layer is best at the fixed vacancy diffusion workflow. It now runs and parses the current NEB / CI-NEB flow, but it is not yet a fully general scientific script generator.
- If you want OpenClaw to keep a compatible MCP definition around, prefer the ACPX-side `plugins.entries.acpx.config.mcpServers.mietclaw` entry.
- The helper script `$REPO_ROOT/scripts/configure_openclaw_27b_mcp.sh` writes a direct Python launch command into that ACPX entry to avoid shell-wrapper startup issues.
