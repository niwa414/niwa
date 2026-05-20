# OpenClaw 集成建议

## 为什么以 OpenClaw 做控制平面

OpenClaw 很适合拿来做这个产品的控制平面，因为它天然支持：

- 插件化工具接入
- 权限与配置管理
- 前端工作台和工具注册表
- 让专用能力以 agent 工具的形式被调用，而不是靠一堆散乱提示词硬拼

对 mietclaw 来说，OpenClaw 负责的是“入口、调度和操作台”；真正的材料仿真逻辑仍然在 mietclaw 自己的执行层里。

## 现在已经接上的实际工具

当前插件目录：
`$REPO_ROOT/packages/openclaw-miet-claw-plugin`

已经暴露给 OpenClaw 的工具有 16 个，和 Python MCP 的工具表保持同步；其中 `miet_resume_job` 是 OpenClaw 插件侧额外提供的恢复入口。

### 运行环境与只读检查
- `miet_runtime_doctor`：检查本地模型、LAMMPS、MoRe case、KMC binary 是否可用
- `miet_list_runs`：列出最近 run
- `miet_inspect_run`：检查某个 run 的状态、摘要和执行来源
- `miet_get_logs`：读取 MD / KMC / summary 日志
- `miet_list_artifacts`：列出归档产物

### 自主任务与固定工作流
- `miet_autonomy_draft`：把自然语言任务转成可执行草稿工作区
- `miet_autonomy_run`：草稿、dry-run 校验，并按需启动真实运行
- `miet_plan_job`：对已有 job spec 输出执行计划
- `miet_run_job`：执行已有 job spec
- `miet_resume_job`：从已有 run 目录恢复继续跑

### KMC bridge 与 MoRe
- `miet_kmc_bridge`：把 `event.json + neb.txt` 转成 KMC lookup 并可验证
- `miet_moire_run`：完整运行 MoRe LAMMPS → repo KMC
- `miet_moire_compare`：比较多个 MoRe event
- `miet_moire_diffusion_sweep`：扫温并汇总扩散系数
- `miet_moire_lammps`：只跑 MoRe LAMMPS 阶段
- `miet_moire_kmc`：只跑 repo KMC 阶段

## 推荐的产品分层

### 1. 前端 / 品牌层
负责：
- mietclaw 品牌界面
- 控制台布局
- 任务创建与查看
- 结果浏览与解释

### 2. OpenClaw 控制平面
负责：
- 工具注册
- 任务入口
- 调用 autonomy 工具或执行工具
- 对外统一成一个 agent 操作面

### 3. mietclaw 自主脚本层
负责：
- 自然语言 -> job spec
- 生成 MD NEB / CI-NEB 工作区 + KMC 预览输入
- 先做 dry-run
- 根据 provider 决定使用本地启发式或 Claude Agent SDK

### 4. mietclaw 确定性执行层
负责：
- 真正运行 LAMMPS
- 真正运行 MISA-KMC
- 编译事件表和速率
- 状态跟踪、失败恢复、归档和解释

## 推荐的交互链路

最合适的实际链路是：

1. 用户在 mietclaw 前端里用自然语言描述任务
2. OpenClaw 调 `miet_autonomy_draft`
3. 返回草稿 spec、MD NEB / CI-NEB 工作区、KMC 输入草案和假设说明
4. 用户确认后，或由策略自动决定后，调 `miet_autonomy_run`
5. autonomy 层先做 dry-run
6. dry-run 通过后，调用执行层真跑
7. 前端持续读取 `runs/` 和 explanation 输出展示结果

## 对 Claude Code / Claude Agent SDK 的正确用法

最稳妥的做法不是把整个产品做成“Claude Code 的壳”，而是：

- 保留你自己的 mietclaw 品牌
- 保留你已经跑通的确定性执行层
- 把 Claude Agent SDK 放在上面，承担“理解任务、起草脚本、做一次自检”的角色

也就是：
- **Claude / autonomy 层负责想和写**
- **mietclaw 执行层负责跑和管**

## 当前限制

现在这套 autonomy v1 已经能把自然语言任务变成可跑草稿，并且本地模式已经实测跑通。

但它还不是完全体：

- 目前最强的是“固定 vacancy diffusion 链路”
- MD 侧已经能真实执行并解析当前这套 NEB / CI-NEB 工作流，但还不是完整的科研自动建模器
- Claude SDK 模式入口已经接好，但要先安装 SDK 并配置认证后才会启用

所以它现在更接近：

> 能自己搭出任务骨架并推动固定链路跑起来的专用科研 agent

而不是：

> 对任何材料体系都能完全独立做科研设计的全能 agent
