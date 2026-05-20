# 本地 27B + MCP 接入说明

这份说明面向支持两件事的本地 agent / shell：

1. 能调用 OpenAI-compatible 本地模型接口
2. 能接 stdio MCP server

## 推荐本地模型

当前这台机器上已经验证可用的 27B 模型：

- `Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit`

如果不额外指定，mietclaw 现在会优先选择可用的 27B 模型。

你也可以显式指定：

```bash
export MIETCLAW_LOCAL_MODEL=27b
```

或者直接启动 27B launcher：

```bash
$REPO_ROOT/bin/mietclaw-27b
```

## 本地模型接口

默认本地接口：

```text
http://127.0.0.1:8000
```

可选环境变量：

```bash
export MIETCLAW_LOCAL_MODEL_BASE_URL=http://127.0.0.1:8000
export MIETCLAW_LOCAL_MODEL_API_KEY=omlx-local
export MIETCLAW_LOCAL_MODEL=27b
```

## 启动 MCP server

```bash
cd $REPO_ROOT
PYTHONPATH=src python3 -m miet_claw.cli mcp-server
```

或者：

```bash
$REPO_ROOT/bin/mietclaw-mcp
```

## 已验证的本地客户端接法

### 1）Codex

直接运行：

```bash
$REPO_ROOT/scripts/connect_codex_mcp.sh
```

它会把 `mietclaw-mcp` 注册到本机 Codex 配置里。

### 2）OpenClaw（本地 27B）

直接运行：

```bash
$REPO_ROOT/scripts/configure_openclaw_27b_mcp.sh
```

它会做四件事：

1. 把默认主模型切到本地 27B  
2. 在 OpenClaw 的 ACPX 插件配置里注册一条直接启动 `miet_claw.cli mcp-server` 的兼容命令  
3. 打开 bundled `acpx` 运行时插件  
4. 保留 `miet-claw-sim` 插件可用

## 通用 MCP 配置片段

如果你的本地 agent 框架支持 stdio MCP，可以参考：

```json
{
  "mcpServers": {
    "mietclaw": {
      "command": "$REPO_ROOT/bin/mietclaw-mcp"
    }
  }
}
```

这份脚本只会把 `mietclaw` 写进 ACPX 侧的 `mcpServers`，不会开启 OpenClaw 顶层的 bundle-mcp 自动探测，这样可以避免本地日常聊天时每次都去预热 MCP。

## 通用本地模型配置思路

如果你的框架还需要单独填写模型后端，一般需要这三项：

```json
{
  "model": "Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit",
  "baseURL": "http://127.0.0.1:8000/v1",
  "apiKey": "omlx-local"
}
```

## mietclaw MCP 工具

当前暴露的工具：

- `miet_list_runs`
- `miet_inspect_run`
- `miet_get_logs`
- `miet_list_artifacts`
- `miet_autonomy_draft`
- `miet_autonomy_run`
- `miet_plan_job`
- `miet_run_job`
- `miet_kmc_bridge`
