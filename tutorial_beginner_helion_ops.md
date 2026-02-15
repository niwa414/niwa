# Fusion 仿真软件初学者教程 + Helion岗位日常操作SOP

面向对象：
- 第一次接触本仓库的同学（初学者）。
- 需要按 Helion 类岗位节奏做“仿真 -> 门禁 -> 工程结论 -> 线下动作/采购”的同学。

---

## 0) 先理解你在用什么（1分钟）

这套软件的业务主线是：

1. 跑 case（`tools/run_case.py` 或 orchestrator）。
2. 自动生成 `metrics.json` + `PASSFAIL.json`。
3. 汇总成工程交付：
   - `summary.md`
   - `work_orders.md`
   - `procurement_spec.md`
   - `release_gate*.json`

你最终要交付的不是“图”，而是：
- 推荐调整哪个旋钮（时序/波形/参数）。
- 哪些规格可直接采购，哪些必须补实验/补模型。

---

## 1) 环境准备（初学者必做）

在仓库根目录：

```bash
cd /Users/ni/Desktop/fusion
python --version
```

检查关键脚本存在：

```bash
ls /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py
ls /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py
```

说明：
- 如果你本机没有 Slurm 命令（`sbatch/squeue/sacct`），先用 `--mode local` 学流程。
- 上超算后再切到 `--mode slurm`。

---

## 2) 30分钟快速上手（不占用大算力）

目标：先把“运营流程”跑通，不做重计算。

### Step 1: 本地跑一次 orchestrator（仅 analyze）

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start \
  --plan /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.json \
  --mode local \
  --force-stage analyze \
  --poll-interval-s 2
```

完成后会打印一个 `run_id`，例如：
- `helion-live-tilt-tradestudy-YYYYMMDD-HHMMSS`

### Step 2: 查看核心产物

```bash
ls /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/
cat /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/summary.md
cat /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/work_orders.md
cat /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/procurement_spec.md
```

### Step 3: 跑发布门禁（日常版）

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json
```

输出：
- `/Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/release_gate.json`

---

## 3) 类 Helion 岗位“每天怎么干”

下面是可直接执行的班次化流程（仿真运营/集成负责人视角）。

### 早班（目标设定，15-30分钟）

1. 定义今天的“单旋钮”目标（例如 `shift`）。
2. 确认计划文件：
   - `/Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.json`
   - 或生产模板：`/Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.slurm-prod.template.json`
   - 生产计划建议先生成一份本地副本：

```bash
cp /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.slurm-prod.template.json \
   /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.prod.json
```

然后编辑 `orchestrator-plan.prod.json` 里的占位符：
- `__GPU_PARTITION__`
- `__SLURM_ACCOUNT__`
- `__SLURM_QOS__`
3. 明确门禁版本：
   - 日常迭代门禁：`release-gate-thresholds.json`
   - 采购终签门禁：`release-gate-thresholds.strict.json`

### 日间（提交+监控）

超算环境执行：

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py start \
  --plan /Users/ni/Desktop/fusion/cases/helion-live-tilt-tradestudy/orchestrator-plan.prod.json \
  --mode slurm \
  --poll-interval-s 30
```

会话中断后恢复监控：

```bash
python /Users/ni/Desktop/fusion/tools/sim_ops_orchestrator.py resume \
  --run-id <run_id> \
  --poll-interval-s 30
```

### 收班（分析+决策）

1. 看 `summary.md`：确认 PASS 数、推荐工况。
2. 看 `work_orders.md`：把建议转成线下动作工单。
3. 看 `procurement_spec.md`：拆分采购类别。
4. 跑门禁脚本给出 Go/No-Go：

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.json \
  --output /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/release_gate.daily.json
```

如果要“采购最终放行”，再跑严格门禁：

```bash
python /Users/ni/Desktop/fusion/tools/check_sim_ops_release_gate.py \
  --run-id <run_id> \
  --thresholds /Users/ni/Desktop/fusion/ops/release-gate-thresholds.strict.json \
  --output /Users/ni/Desktop/fusion/outputs/orchestrator/<run_id>/release_gate.strict.json
```

---

## 4) 如何把结果转成线下动作（最关键）

看这三个文件就够了：

1. `summary.md`
- 用来选推荐旋钮（例：`shift=0.002`）。
- 作用：指导下一批 shot 的控制参数。

2. `work_orders.md`
- 直接变成执行清单（调时序、复查日志、补采样）。
- 作用：安排电气/诊断/仿真同事谁做什么。

3. `procurement_spec.md`
- A) `可直接采购`：按 `max_with_sf` 下采购规格。
- B) `需先补实验/补模型`：暂缓下单，先补证据。
- C) `Internal-only gap 清单`：闭环 3 项后再放行延后采购。

3个 internal-only gap（必须会背）：
- `gpu_runtime_proven`
- `private_shot_dataset_bound`
- `private_hardware_model_bound`

---

## 5) 你应该固定产出的“日报模板”

建议每天收班发以下 6 行：

1. 今日 run_id：`...`
2. 总体状态：`PASS/FAIL`（来自 `release_gate.daily.json`）
3. 推荐旋钮：`...`
4. 线下动作工单：`WO-OPS-002 / WO-DBG-*`
5. 可直接采购条目数：`N`
6. 严格门禁是否通过：`PASS/FAIL`（若 FAIL，列出 gap）

---

## 6) 初学者最常见错误与处理

### 错误1：`sbatch` 不存在

处理：
- 先用本地模式学习流程：
  - `--mode local --force-stage analyze`
- 上超算节点后再切 `--mode slurm`。

### 错误2：看不懂为什么 FAIL

处理顺序：

1. 打开 `release_gate*.json` 看 `failures` 数组。
2. 打开 `work_orders.md` 看对应调试工单。
3. 打开 `outputs/<case_id>/analysis/PASSFAIL.json` 看具体阈值失败项。

### 错误3：采购和工程建议冲突

处理原则：
- 以 `procurement_spec.md` 的分类为准：
  - A 表可执行采购。
  - B/C 表先补证据，不要硬下单。

---

## 7) 从“初学者”进阶到“岗位可独立值班”

达标标准：

1. 你能独立启动/恢复一次 orchestrator。
2. 你能解释 `summary/work_orders/procurement_spec/release_gate` 各自用途。
3. 你能把推荐旋钮转成线下实验动作（3-5发验证方案）。
4. 你能说清楚为什么严格门禁没过（internal-only gaps）。

做到这4点，就已经是 Helion 类仿真运营岗位的可用水平。

---

## 8) 相关文档（建议一起看）

- `/Users/ni/Desktop/fusion/sim_ops_orchestrator_runbook.md`
- `/Users/ni/Desktop/fusion/ops/slurm-production-checklist.md`
- `/Users/ni/Desktop/fusion/ops/bo-active-learning-playbook.md`
- `/Users/ni/Desktop/fusion/outputs/analysis/helion-style-design-trade-study.md`
- `/Users/ni/Desktop/fusion/outputs/analysis/procurement-ready-spec.md`
