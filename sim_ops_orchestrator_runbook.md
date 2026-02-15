# Simulation Ops Orchestrator Runbook (Slurm)

## 1) 启动与恢复

启动新批次（Slurm）：

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start \
  --plan /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.json \
  --mode slurm \
  --poll-interval-s 30
```

恢复监控（登录中断后）：

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py resume \
  --run-id <run_id> \
  --poll-interval-s 30
```

本地快速回归（只跑分析，验证流程）：

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start \
  --plan /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.json \
  --mode local \
  --force-stage analyze \
  --poll-interval-s 2
```

## 2) 结果文件怎么看

每次运行输出到：`/Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/`

关键文件：
- `state.json`: 单一事实源（每个 case 的调度 ID、重试、最终状态、产物路径）。
- `events.jsonl`: 事件流水（提交、终态、重试、告警）。
- `summary.md`: 运行总览 + 推荐工况（推荐 knob 方向）。
- `work_orders.md` / `work_orders.json`: 线下动作工单（调时序/排查失败）。
- `procurement_spec.md` / `procurement_spec.json`: 采购草案（含 min/nom/max/safety factor/metrics 绑定）。
- `trade_study_report.md`: baseline/knob-/knob+ 对比报告（当三角色都 PASS 时生成）。

## 3) 如何用结果指导线下采购

按 `procurement_spec.md` 三段执行：

1. A) `可直接采购`
- 条件：该项有有效样本并达 READY。
- 动作：按 `max_with_sf` 下限额定出采购规格，PO 备注绑定对应 `metric_binding` 字段。

2. B) `需先补实验/补模型`
- 条件：`status=LIMITED_SAMPLE` 或 `MISSING`。
- 动作：先补 shot/模型证据，再更新同一 runbook 下的 plan 重跑。

3. C) `Internal-only gap 清单`
- 对应 3 个闭环项：
  - `gpu_runtime_proven`
  - `private_shot_dataset_bound`
  - `private_hardware_model_bound`
- 动作：先完成表内 action，再将原“延后采购”条目转入 A 表。

## 4) 推荐签核顺序

1. 先看 `summary.md`：确认 `status_counts` 全 PASS。  
2. 再看 `work_orders.md`：执行 `WO-OPS-002` 推荐旋钮（先小批量 shot 验证）。  
3. 然后看 `procurement_spec.md`：A 表可发单，B 表延后。  
4. 最后对照：
- `/Users/ni/Desktop/fusion/outputs/analysis/helion-style-design-trade-study.md`
- `/Users/ni/Desktop/fusion/outputs/analysis/procurement-ready-spec.md`

这两份是跨 case 的业务签核文档，用于最终评审。

## 5) 生产发布门禁（新增）

发布前执行：

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json
```

如需“采购最终放行”严格门禁（internal parity 必须闭环）：

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.strict.json
```

查看输出：
- `/Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/release_gate.json`

配套生产清单与计划模板：
- `/Users/ni/Desktop/fusion/ops/slurm-production-checklist.md`
- `/Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.slurm-prod.template.json`
