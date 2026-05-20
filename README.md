# Fusion Simulation Integration

Open-source workflow for pulse-driven FRC simulation engineering with Athena++/WarpX style pipelines.

This repository focuses on **engineering operations**:
- case orchestration (local and Slurm)
- KPI extraction and PASS/FAIL gates
- trade-study reporting
- procurement-oriented specification outputs
- BO/active-learning planning for next-case selection

## What Is Included

- `tools/`: run, analysis, orchestration, release-gate, and BO planning scripts
- `cases/`: reproducible case configs for live/demo trade studies and milestone gates
- `warpx-driver/`: WarpX Python driver utilities
- `ops/`: production runbooks, thresholds, and operations playbooks
- `docs/`: example generated deliverables
- `evidence/`: lightweight evidence and templates

## What Is Not Included (By Design)

The following are ignored to keep the repo open-source friendly and lightweight:
- `outputs/` (large generated data/results)
- `pic-warpx-25.11/` (vendored third-party source tree)
- `athena-24.0/` (vendored third-party source tree)
- `warpx_used_inputs` (large binary artifact)

Bring these dependencies/data in your own environment as needed.

## Quick Start (Local, no Slurm)

```bash
cd /Users/ni/Desktop/fusion

python tools/sim_ops_orchestrator.py start \
  --plan cases/helion-live-tilt-tradestudy/orchestrator-plan.json \
  --mode local \
  --force-stage analyze \
  --poll-interval-s 2
```

Then run release gate:

```bash
python tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds ops/release-gate-thresholds.json
```

## Slurm Workflow

```bash
python tools/sim_ops_orchestrator.py start \
  --plan cases/helion-live-tilt-tradestudy/orchestrator-plan.slurm-prod.template.json \
  --mode slurm \
  --poll-interval-s 30
```

If monitoring session is interrupted:

```bash
python tools/sim_ops_orchestrator.py resume --run-id <run_id> --poll-interval-s 30
```

## BO / Active Learning Workflow

```bash
python tools/run_bo_cycle.py \
  --config cases/helion-live-tilt-tradestudy/bo-config.json \
  --plan-only
```

Then execute the generated plan with orchestrator.

## Key Docs

- Beginner + role SOP: `tutorial_beginner_helion_ops.md`
- Orchestrator runbook: `sim_ops_orchestrator_runbook.md`
- Slurm production checklist: `ops/slurm-production-checklist.md`
- BO playbook: `ops/bo-active-learning-playbook.md`

## Publish to GitHub

Create an empty GitHub repository first, then run:

```bash
./tools/publish_github.sh <github_repo_url>
```

Examples:
- `./tools/publish_github.sh git@github.com:<user>/fusion.git`
- `./tools/publish_github.sh https://github.com/<user>/fusion.git`


## License

Apache-2.0. See `LICENSE`.
