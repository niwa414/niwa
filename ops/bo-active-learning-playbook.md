# BO / 主动学习实操（Helion式工程迭代）

目标：
- 自动挑下一批“最值钱”的算例，在多目标+约束下推进工程决策。

当前实现位置：
- 配置：`/Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json`
- 数据回灌：`/Users/ni/Desktop/fusion/tools/bo_update_dataset.py`
- 规划器：`/Users/ni/Desktop/fusion/tools/bo_plan_next_cases.py`
- 执行器：`/Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py`
- 一键循环：`/Users/ni/Desktop/fusion/tools/run_bo_cycle.py`

## 1) 优化定义

优化目标（已落地）：

`J = compression_ratio + w1*recapture_efficiency - w2*tilt_amp_max - w3*load_penalty`

约束（已落地）：
- `energy_accounting_ok == true`
- `load_metric <= load_max`（默认候选顺序：`coil_force_peak`, `load_force_proxy_peak_N`, `dphi_dt_peak_V`, `vind_peak_V`）

说明：
- 你的真实 `coil_force_peak` 一旦在 metrics 里可用，会自动优先生效。
- 当前 load constraint 可先用磁/电压代理量驱动 BO，再逐步替换成真实力学载荷指标。

## 2) 多保真策略（已接入）

配置里有两个 fidelity：
- `mhd_proxy`：低成本候选（当前用 demo/replay 模板快速筛点）
- `hybrid_pic`：高成本高保真候选（live 3D hybrid）

每轮配额（默认）：
- `mhd_proxy: 2`
- `hybrid_pic: 1`

你可在 `bo-config.json` 调整配额、cost、模板和阶段。

## 2.1) 一键跑一轮（可选）

```bash
python /Users/ni/Desktop/fusion/tools/run_bo_cycle.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json \
  --mode slurm
```

只生成计划不执行：

```bash
python /Users/ni/Desktop/fusion/tools/run_bo_cycle.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json \
  --plan-only
```

## 3) 一轮 BO 的完整命令链

### Step A: 更新数据集（首次建议加 bootstrap）

```bash
python /Users/ni/Desktop/fusion/tools/bo_update_dataset.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json \
  --bootstrap
```

如果刚跑完 orchestrator，再把 run 回灌进数据集：

```bash
python /Users/ni/Desktop/fusion/tools/bo_update_dataset.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json \
  --run-id <run_id>
```

### Step B: 规划下一批 case（BO 给建议点）

```bash
python /Users/ni/Desktop/fusion/tools/bo_plan_next_cases.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json
```

输出：
- `outputs/bo/.../plans/<batch_id>.plan.json`（可直接给 orchestrator）
- `outputs/bo/.../plans/<batch_id>.suggestions.json`（含 acquisition/prediction）
- `cases/helion-live-tilt-tradestudy/bo_generated/<batch_id>/...`（新 case）

### Step C: 执行（executor）

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start \
  --plan /Users/ni/Desktop/fusion/outputs/bo/helion-live-tilt-tradestudy/plans/<batch_id>.plan.json \
  --mode slurm \
  --poll-interval-s 30
```

### Step D: 分析与门禁（analyzer + gate）

orchestrator 结束后，对 `run_id` 跑发布门禁：

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json
```

### Step E: 回灌（active learning loop 关闭）

```bash
python /Users/ni/Desktop/fusion/tools/bo_update_dataset.py \
  --config /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/bo-config.json \
  --run-id <run_id>
```

然后回到 Step B，进入下一轮。

## 4) 这套为什么像 Helion 日常

你现在的闭环是：
1. planner 自动提议下一批参数点（不是固定网格）
2. executor 跑仿真
3. analyzer + gate 评估是否可行
4. surrogate 吸收新数据再提议

这就是“会思考下一步该算什么”的仿真运营 agent。

## 5) 生产注意事项

- 第一次上集群前，先 `--mode local --force-stage analyze` 验证链路。
- 不要手改 `bo_generated` 中的 case；改参数请改 `bo-config.json`。
- 采购决策仍以：
  - `procurement_spec.md`
  - `release_gate.strict.json`
  - `internal-only gaps`
  为最终签核依据。
