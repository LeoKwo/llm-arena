# LLM Arena

一个多智能体 LLM 对战模拟：多个由本地 Ollama 或云端 API 驱动的智能体扮演国家，在共享世界状态下通过工具调用行动，具备记忆、反思、规划，并通过 Web UI 实时观战。

## 功能

- 三个国家智能体：Germany / France / United Kingdom
- 每个智能体有独立的人格（persona）与三级目标（primary / secondary / tertiary）
- LangGraph `StateGraph` 回合流程：`observe → (reflect) → plan → act → tools → collect`
- 工具动作真实修改世界状态：`observe / move / interact / attack / wait`
- FAISS + Ollama embedding 的语义记忆检索
- 每 3 回合触发一次反思（reflection），每回合生成计划（plan）
- 胜负规则：资源 ≤ 0 淘汰；达到最大回合后按 `resources + territories×20 + goal_completion×100` 评分，最高者胜
- 本地模型 / 云端 API 模型可切换，每个国家可独立选择不同或相同的提供商
- Web UI 实时 SSE 推送：回合、行动、结果、计划、反思、世界地图、关系、得分
- 支持中断正在进行的模拟

## 环境要求

- macOS / Linux
- Python 3.11
- [Ollama](https://ollama.com/)（本地模型时使用）
- 16GB 内存的机器建议限制 Ollama 同时只加载 1 个模型（见下）

## 依赖安装

```bash
# 创建虚拟环境
python3.11 -m venv myenv

# 安装依赖
./myenv/bin/pip install -r requirements.txt
```

`requirements.txt` 主要包含：`langchain` / `langchain-core` / `langchain-ollama` / `langchain-openai` / `langgraph` / `faiss-cpu` / `fastapi` / `uvicorn` / `ollama`。

## Ollama 准备（本地模型）

```bash
# 拉取对话模型与 embedding 模型
ollama pull qwen3.5:latest
ollama pull qwen3:14b
ollama pull glm4:9b
ollama pull qwen3-embedding:0.6b

# 16GB 内存：限制同时只加载 1 个模型（macOS，重启 Ollama 生效）
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
osascript -e 'quit app "Ollama"'; sleep 3; open -a Ollama
```

> 本地默认模型为 `qwen3.5:latest`（三个国家相同）。embedding 固定使用 `qwen3-embedding:0.6b`。

## API 提供商配置（可选）

在 Web UI 打开 “Use API models” 后，可为每个国家选择提供商。密钥通过环境变量提供：

| 提供商 | 环境变量（任选其一） | 默认模型 |
| --- | --- | --- |
| Zhipu (GLM) | `ZHIPUAI_API_KEY` / `ZHIPU_API_KEY` / `GLM_API_KEY` | `glm-4-plus` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen (DashScope) | `DASHSCOPE_API_KEY` / `QWEN_API_KEY` | `qwen-plus` |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` / `KIMI_API_KEY` | `moonshot-v1-8k` |

```bash
export DEEPSEEK_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."
export ZHIPUAI_API_KEY="..."
export MOONSHOT_API_KEY="..."
```

所有云端提供商通过 OpenAI 兼容接口调用（`langchain-openai` 的 `ChatOpenAI` + `base_url`）。

## 常用命令

```bash
# 启动 Web UI（推荐）
./myenv/bin/python -m web.server
# 浏览器打开 http://127.0.0.1:8000

# 命令行运行完整模拟（默认 20 回合，实时打印）
./myenv/bin/python main.py

# 停止 Web 服务器
pkill -f web.server

# 查看运行状态 / 中断当前模拟（Web 服务运行时）
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8000/api/stop
```

## Web UI 使用

1. 顶部设置 `Max turns`，点击 **Start**。
2. **Use API models** 关闭时使用本地 Ollama；打开后每个国家可分别选择 Zhipu / DeepSeek / Qwen / Kimi，可全选相同或各选不同。
3. 若所选提供商的密钥未配置，页面会提示缺失的环境变量，服务端在启动模拟时也会返回错误。
4. **Interrupt** 按钮可请求中断：当前正在生成的智能体完成后停止，界面显示 `interrupted`。
5. 右侧 Live Feed 实时显示行动、结果、Plan 与 Reflection；左侧显示国家面板、世界地图、关系矩阵与实时得分。
6. 模型输出（planning / reflecting / acting）以可折叠面板实时流式显示：生成中自动展开并逐 token 刷新，结束后自动收起，点击标题可随时展开查看完整内容。

> 注意：SSE 客户端断开不会自动取消后台模拟；请使用 Interrupt 按钮或 `/api/stop`。

## 项目结构

```
main.py                      # 命令行入口
engine/simulation_loop.py    # 回合引擎 + 事件发射（CLI / Web 共用）
world/environment.py         # World 世界状态、动作结算、评分、快照
agent/
  init_agents.py             # 国家名册、默认模型配置、构建智能体
  llm_factory.py             # 本地/云端模型工厂、API Key、代理处理
  base_agent.py              # ChatOllama 基础封装（旧代码，保留）
  agent_state.py             # AgentState (LangGraph TypedDict)
  tools.py                   # 动作工具工厂 make_tools(world, agent)
  persona.py / goals.py / prompt.py
agent_graph/graph.py         # StateGraph: observe/reflect/plan/act/tools/collect
agent_memory/memory_store.py # FAISS 记忆存储
web/
  server.py                  # FastAPI + SSE 服务
  index.html                 # 实时对战前端
requirements.txt
```

## 配置说明

- `MAX_TURNS`：默认 20，可在 Web UI 或 `run(max_turns=...)` 覆盖。
- `REFLECT_EVERY`：默认每 3 回合反思一次（`agent_graph/graph.py`）。
- 评分权重：`TERRITORY_WEIGHT=20`、`GOAL_WEIGHT=100`（`world/environment.py`）。
- 本地代理：若系统开启了全局代理，Python httpx 可能把 `127.0.0.1` 也走代理导致 Ollama 502。`agent/llm_factory.py` 已自动为 `127.0.0.1` / `localhost` / `::1` 设置 `NO_PROXY`。
