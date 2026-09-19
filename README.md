# LLM Arena

一个多智能体 LLM 对战模拟：多个由本地 Ollama 或云端 API 驱动的智能体扮演国家，在六边形欧洲棋盘上通过工具调用指挥军队攻城略地，具备记忆、反思、规划，并通过 Web UI 实时观战。

## 功能

- 四个国家智能体：Germany / France / United Kingdom / Soviet Union
- 每个智能体有独立的人格（persona）与三级目标（primary / secondary / tertiary）
- 六边形棋盘（pointy-top，整体近似矩形），约 26 座按欧洲地理大致布局的城市，城市之间随机生成资源地块
- 单位为 2 行动点：方向为 E / SE / SW / W / NW / NE
- 资源归属于城市与单位；每回合每座**受控城市 +5**（**首都 +10**），城市资源下限 5；分数 = 受控资源 ×（1 + 0.5 × 目标完成度）
- 军队单位由城市划拨资源生成，每回合 2 行动点：`move_unit` 行军 (1) / `attack_city` 攻城 (1) / `loot_tile` 掠夺 (2) / `merge_units` 合并 (0) / `supply_unit` 补给 (1) / `disband_unit` 解散 (0)
- **动态占领**：地块仅在有 ≥1 资源的单位驻留时归其控制，单位离开即回归中立；掠夺一次性取走该格资源
- **战斗有损耗**：获胜方损失败方强度的 30%；城市防御 = 城市资源 + 驻防单位资源之和 + 5（有驻军时）
- 攻城：单位 > 防御 → 掠夺城市资源 30%、城市减半（下限 5）、守军全灭、进攻单位自动入城；否则进攻单位被消灭，守方按进攻强度 30% 受损
- **维护费**：每个单位每回合 1 资源，由城市支付，不足则单位减员；**未消耗行动点的单位每回合恢复 1 资源**
- 解散：单位站在己方城市上可原地解散，资源归还该城市（不再有额外奖励）
- 首都机制：首都每回合 +10（其他城市 +5）；**丢失首都**时己方每座城市 -2、每个单位 -1；**夺回首都**时己方每座城市 +5、每个单位 +2（城市均不低于 5）
- LangGraph `StateGraph` 回合流程：`observe → (reflect) → plan → act → tools → collect`，一回合内可生成并指挥多个单位，最后 `end_turn`
- FAISS + embedding 的语义记忆检索
- 每 3 回合触发一次反思（reflection），每回合生成计划（plan）
- 胜负规则：既无城市也无单位则淘汰；达到最大回合后按 `受控资源 × (1 + GOAL_SCORE_BONUS × 目标完成度)` 评分，最高者胜
- 回合顺序每轮轮换；对局有固定回合上限，模型在每回合都能看到当前回合与剩余回合数
- 本地模型 / 云端 API 模型可切换，每个国家可独立选择不同或相同的提供商
- Web UI 实时 SSE 推送：回合、行动、结果、计划、反思、六边形地图、关系、得分
- 中 / 英界面切换，模型输出语言随界面切换
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

`requirements.txt` 主要包含：`langchain` / `langchain-core` / `langchain-ollama` / `langchain-openai` / `langgraph` / `faiss-cpu` / `fastapi` / `uvicorn` / `ollama` / `python-dotenv`。

## .env 配置

启动时会用 `python-dotenv` 自动读取项目根目录的 `.env`（不存在则忽略；真实环境变量优先）。复制模板即可：

```bash
cp .env.example .env
```

`.env` 内容：

```ini
# ---- API Key ----
DEEPSEEK_API_KEY=
ZHIPUAI_API_KEY=
DASHSCOPE_API_KEY=
MOONSHOT_API_KEY=

# ---- 对话模型：全局默认 + 每国覆盖 ----
# provider: ollama | zhipu | deepseek | qwen | kimi
DEFAULT_PROVIDER=ollama
DEFAULT_MODEL=qwen3.5:latest

GERMANY_PROVIDER=
GERMANY_MODEL=
FRANCE_PROVIDER=
FRANCE_MODEL=
UK_PROVIDER=
UK_MODEL=
USSR_PROVIDER=
USSR_MODEL=

# ---- Embedding（用于智能体记忆 / FAISS）----
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
# 注意：填 base URL（例如 https://open.bigmodel.cn/api/paas/v4），不要带结尾的 /embeddings
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
```

> DeepSeek 目前**没有 embedding 接口**，所以 embedding 默认用智谱 GLM 的 `embedding-3`（需要 `ZHIPUAI_API_KEY`）。想免费本地运行可设 `EMBEDDING_PROVIDER=ollama`（默认模型 `qwen3-embedding:0.6b`）。

解析优先级：

- **模型 provider/model**：Web UI 手动选择 > `.env`（每国覆盖 > 全局默认 > 本地 Ollama）。
- **API Key**：Web UI 输入 > 系统环境变量 > `.env`（`load_dotenv(override=False)`，系统环境变量不会被 `.env` 覆盖）。

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

在 Web UI 打开 “Use API models” 后，可为每个国家选择提供商。

密钥有三种提供方式（优先级从高到低）：

1. **直接在 Web UI 的 “API Keys” 区域输入**：每个云端提供商一个输入框，点 **Start** 时会提交给本地服务端（仅保存在内存中，不落盘）。
2. **系统环境变量**。
3. **`.env` 文件**（见上节）。

提供商的密钥环境变量名：

| 提供商 | 环境变量（任选其一） | 默认模型 |
| --- | --- | --- |
| Zhipu (GLM) | `ZHIPUAI_API_KEY` / `ZHIPU_API_KEY` / `GLM_API_KEY` | `glm-4-plus` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
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
   - 右上角的 **中文 / EN** 按钮可切换界面语言：切到中文后，除界面文案外，模型生成的 plan / 反思 / action reason 也会用简体中文输出（通过提示词指令实现）。语言选择不做持久化，刷新后默认英文。
2. **Use API models** 关闭时使用本地 Ollama；打开后每个国家可分别选择 Zhipu / DeepSeek / Qwen / Kimi，可全选相同或各选不同。页面初次加载时会按 `.env` 的默认/每国配置预选。
3. 在 “API Keys” 区域为所选提供商填入密钥（留空则回退到系统环境变量 / `.env`）；都缺失时页面会提示，服务端在启动模拟时也会返回错误。
4. **Interrupt** 按钮可请求中断：当前正在生成的智能体完成后停止，界面显示 `interrupted`。
5. 右侧 Live Feed 实时显示行动、结果、Plan 与 Reflection，并用高亮广播重要事件（城市被攻占、首都被攻陷/夺回、单位被歼灭、势力被淘汰）；左侧自上而下为世界地图、国家面板与关系矩阵。国家卡片可点击标题折叠/展开。
6. 模型输出（planning / reflecting / acting）以可折叠面板实时流式显示：生成中自动展开并逐 token 刷新，结束后自动收起，点击标题可随时展开查看完整内容。
7. **观战模式**：点击顶部 **观战模式 / Spectator**，隐藏配置面板，让六边形地图与 Live Feed 铺满整屏；再次点击退出。
8. **可交互地图**：拖拽平移、滚轮缩放、**重置视图**按钮复位；点击任意六边形在下方信息框查看城市/地形、归属、资源与驻扎单位。

> 注意：SSE 客户端断开不会自动取消后台模拟；请使用 Interrupt 按钮或 `/api/stop`。

## 项目结构

```
main.py                      # 命令行入口
engine/simulation_loop.py    # 回合引擎 + 事件发射（CLI / Web 共用）
world/hexmap.py              # 六边形坐标/邻接/方向 + 欧洲城市布局 + 地图生成
world/environment.py         # World：城市/地块/单位、行动结算、评分、快照
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
.env.example                 # .env 配置模板（.env 已被 .gitignore 忽略）
requirements.txt
```

## 配置说明

- `.env`：API Key、对话模型（全局默认 + 每国覆盖）、embedding 模型；由 `python-dotenv` 自动加载。
- `DEFAULT_PROVIDER` / `DEFAULT_MODEL`：默认对话模型；`GERMANY_*` / `FRANCE_*` / `UK_*` / `USSR_*` 可逐国覆盖。
- `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`：智能体记忆用的 embedding，默认 `zhipu` / `embedding-3`。
- `MAP_SEED`：可选，固定地图随机种子（地块资源值），便于复现同一张地图；不设置则每次随机。
- `ATTACK_ATTRITION`（0.3）：获胜方损失败方强度的比例。
- `CITY_LOOT_RATE`（0.3）：夺城时掠夺城市资源的比例。
- `UPKEEP_PER_UNIT`（1）：每个单位每回合维护费。
- `SUPPLY_RANGE`（3）：城市向单位补给的最大六边形距离。
- `RELATION_DECAY`（3）：每回合关系向 0 回归的点数。
- `GOAL_SCORE_BONUS`（0.5）：目标完成度对最终分数的加权倍数。
- `MAX_TURNS`：默认 20，可在 Web UI 或 `run(max_turns=...)` 覆盖。
- `REFLECT_EVERY`：默认每 3 回合反思一次（`agent_graph/graph.py`）。
- 评分权重：`TERRITORY_WEIGHT=20`、`GOAL_WEIGHT=100`（`world/environment.py`）。
- 本地代理：若系统开启了全局代理，Python httpx 可能把 `127.0.0.1` 也走代理导致 Ollama 502。`agent/llm_factory.py` 已自动为 `127.0.0.1` / `localhost` / `::1` 设置 `NO_PROXY`。
