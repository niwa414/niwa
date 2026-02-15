# Slurm Production Checklist for Simulation Ops

This checklist hardens the orchestrator path for daily operation in HPC.

## 1) Queue profile templates

Use one profile and commit it in a plan file before starting a campaign.

| Profile | partition | time | cpus_per_task | gpus | mem | Use when |
| --- | --- | --- | --- | --- | --- | --- |
| smoke | gpu | 01:00:00 | 8 | 1 | 32G | quick scheduler and pipeline verification |
| daily | gpu | 12:00:00 | 8 | 1 | 64G | regular daily trade-study batches |
| release | gpu | 24:00:00 | 16 | 1 | 96G | release-candidate campaign with retries |

Plan template: `/Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.slurm-prod.template.json`.

## 2) Retry strategy

- submit failure (`sbatch` error): fail fast and rerun after cluster-side fix.
- runtime non-zero exit: retry up to 2 times for release profile.
- gate fail (`PASSFAIL=FAIL`): do not auto-increase retries beyond plan budget; open debug work order.

Recommended policy:
- smoke: `max_retries=0`
- daily: `max_retries=1`
- release: `max_retries=2`

## 3) Log retention policy

Retain per-run artifacts under `/Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/`.

Required files to keep for audit:
- `state.json`
- `events.jsonl`
- `summary.md`
- `work_orders.json`
- `procurement_spec.json`
- `release_gate.json`

Retention windows:
- raw local/slurm logs: 30 days
- run summaries and JSON artifacts: 180 days
- release-tagged runs: do not delete

Suggested cleanup command (run from cron, adjust date window as needed):

```bash
find /Users/ni/Desktop/fusion/outputs/orchestrator -maxdepth 1 -type d -mtime +30 -name '*' \
  ! -name '*release*' -print
```

## 4) Release gate thresholds

Threshold config is versioned in:
- `/Users/ni/Desktop/fusion/ops/release-gate-thresholds.json`
- `/Users/ni/Desktop/fusion/ops/release-gate-thresholds.strict.json`

Run gate check after orchestrator completes:

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json
```

Go/No-Go criteria:
- all jobs PASS
- per-case gates satisfy thresholds (compression, tilt, energy residual)
- internal parity file exists and required public-stack fields are true

Profile guidance:
- `release-gate-thresholds.json`: day-to-day release candidate gate (public-stack readiness).
- `release-gate-thresholds.strict.json`: procurement-final gate (requires internal parity claimable and zero internal gaps).

## 5) Release sign-off sequence

1. verify run stability in `summary.md`
2. verify work orders in `work_orders.md`
3. verify procurement classes in `procurement_spec.md`
4. run `check_sim_ops_release_gate.py`
5. sign off against:
- `/Users/ni/Desktop/fusion/outputs/analysis/helion-style-design-trade-study.md`
- `/Users/ni/Desktop/fusion/outputs/analysis/procurement-ready-spec.md`
