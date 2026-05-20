# mietclaw 自主脚本层 v1

## 它解决什么问题

在原来的 mietclaw 里，执行层已经能把固定的 MD / KMC 任务跑起来，但更像“你先把 job spec 写好，我再替你执行”。

自主脚本层 v1 的目标，是把它往前再推一步：

- 接收自然语言任务描述
- 选一个合适模板
- 自动生成 job spec
- 自动生成并运行 MD NEB / CI-NEB 工作区
- 自动生成 KMC 输入预览
- 先做 dry-run
- 通过后再真跑

所以它不是替换执行层，而是在执行层上面加了一个“起草和自检”的层。

## 当前实现位置

- 代码：`$REPO_ROOT/src/miet_claw/autonomy.py`
- CLI 入口：`$REPO_ROOT/src/miet_claw/cli.py`
- OpenClaw 插件接线：`$REPO_ROOT/packages/openclaw-miet-claw-plugin/runtime.js`

## 现在能做什么

### 1. 自然语言 -> job spec
例如给它一句：

> Create a native MD to KMC vacancy diffusion job for "Autonomy Native Run Demo" at 799 K owned by miet.

它会自动提取：
- 模式：`md_to_kmc_chain`
- 材料名
- 温度
- owner
- 可能的 barrier 提示

然后选一个模板，生成新的 `job_spec.generated.json`。

### 2. 生成 MD / KMC 草稿文件
它会生成：
- `md/generated_md_neb_workflow.py`
- `md/neb/neb_campaign.json`
- `md/neb/<species>/in.relax.initial.lmp`
- `md/neb/<species>/in.relax.final.lmp`
- `md/neb/<species>/coords.final`
- `md/neb/<species>/in.neb.ci.lmp`
- `kmc/generated_kmc.preview.in`
- `scripts/plan.sh`
- `scripts/dry_run.sh`
- `scripts/run.sh`
- `autonomy_notes.md`
- `autonomy_report.json`

### 3. 自动 dry-run
`autonomy-run` 会先自己做 dry-run 校验。
如果 dry-run 失败，就不会直接冒然进入真实运行。

### 4. 真实执行固定链路
如果不是 `--dry-run-only`，它会继续调用现有执行层，跑：
- MD
- barrier -> 事件表 / 速率
- KMC
- 解释输出
- 归档

## Provider 设计

### 本地模式：`local` / `local-heuristic`
这是当前已经实测通过的模式。

它不依赖外部模型服务，而是使用：
- 模板目录
- 简单事实抽取
- 规则和默认值

优点：
- 稳定
- 可预测
- 当前机器上已经验证能用

缺点：
- 更像“聪明模板器”
- 不够灵活
- 还不能做深度科研推理

### Claude 模式：`claude`
这个模式已经留好入口，但只有满足以下条件才会启用：

- 安装 `claude-agent-sdk`
- 配好认证，例如 `ANTHROPIC_API_KEY`

启用后，autonomy 层会让 Claude Agent SDK 帮忙做：
- 模板选择
- 任务结构化
- 假设和警告生成

但真正的运行仍然交给 mietclaw 自己的执行层。

## 当前实测结果

在 **2026-04-04**，我已经在这台机器上实测通过了本地模式的完整链路：

### 1. 自然语言 draft
命令：
```bash
PYTHONPATH=src python3 -m miet_claw.cli autonomy-draft \
  'Create a native MD to KMC vacancy diffusion job for "Autonomy CLI Draft Demo" at 888 K owned by miet.' \
  --provider local \
  --project-root $REPO_ROOT \
  --workspace-root $REPO_ROOT/.autonomy-checks
```

结果：成功生成工作区和脚本。

### 2. 自然语言 real run
命令：
```bash
PYTHONPATH=src python3 -m miet_claw.cli autonomy-run \
  'Create a native MD to KMC vacancy diffusion job for "Autonomy Native Run Demo" at 799 K owned by miet.' \
  --provider local \
  --project-root $REPO_ROOT \
  --workspace-root $REPO_ROOT/.autonomy-checks \
  --output-dir $REPO_ROOT/runs
```

结果：dry-run 成功，真实运行也成功。

最终输出包括：
- `$REPO_ROOT/runs/autonomy_native_run_demo/state.json`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/md/barriers.json`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/kmc/generated_kmc.in`
- `$REPO_ROOT/runs/autonomy_native_run_demo/artifacts/kmc/diffusion.csv`
- `$REPO_ROOT/runs/autonomy_native_run_demo/explain/summary.md`

## 现在还没做到的事

这很重要，必须说清楚。

### 1. 它还不是全自动科研 agent
现在最强的是：
- vacancy diffusion 固定链路
- 从自然语言起草任务骨架
- 自动做一次自检再跑

它还不是：
- 自己为任意材料体系设计完整研究方案
- 自己构建高质量 NEB / CI-NEB 路径
- 多轮读日志、改脚本、反复试到物理上合理

### 2. MD 侧现在已经能真实运行并解析 NEB barrier，但仍不是完整科研自动建模器
现在生成的 MD 工作区会：
- 真正起草 species-specific 的 LAMMPS endpoint relax / CI-NEB 输入文件
- 真正启动一次 LAMMPS reference preflight
- 真正运行每个 species 的 CI-NEB
- 从 LAMMPS NEB 输出里解析 barrier，再写出 barrier JSON 给后续链路用

但它仍然不是严格意义上的“高质量科研自动建模器”，因为当前路径初始化还是规则驱动的。

### 3. 本地模式依然主要靠规则
所以当前 autonomy v1 更适合：
- 把自然语言变成一个靠谱起点
- 帮你少写很多样板
- 提高固定工作流的自动化程度

而不是完全替代一个熟练的科研工程师。

## 正确理解它现在的能力

一句话说：

> 它已经能把“自然语言任务 -> 可跑工作区 -> dry-run -> 真实固定链路运行”这件事自动化起来。

但还没有到：

> 给我任何材料问题，它都能自己像资深研究员一样独立建模、独立调参、独立纠错。

## 下一步最值得做的升级

1. 把当前规则驱动的路径初始化升级成更真实的缺陷/端点构造器
2. 给 autonomy 层加入日志诊断和一次自动重试
3. 让前端可以直接发起自然语言 autonomy draft / run
4. 引入更明确的领域知识库：缺陷类型、跃迁路径、势函数选择规则
5. 在 Claude SDK 模式下加入更强的“先看上下文、再写脚本、再自检”的循环
