# LLM Arena

一个多智能体 LLM 对战模拟：多个由本地 Ollama 或云端 API 驱动的智能体扮演国家，在六边形欧洲棋盘上通过工具调用指挥军队攻城略地，具备记忆、反思、规划，并通过 Web UI 实时观战。

## 功能

- 四个国家智能体：Germany / France / United Kingdom / Soviet Union
- 每个智能体有独立的人格（persona）与三级目标（primary / secondary / tertiary）
- 六边形棋盘（pointy-top，整体近似矩形），约 26 座按欧洲地理大致布局的城市，城市之间随机生成资源地块
- 单位为 2 行动点：方向为 E / SE / SW / W / NW / NE
- 资源归属于城市与单位；每回合每座**受控城市 +5**（**首都 +10**），城市资源下限 5
- **分数 = (受控资源 + 50×城市数 + 5×受控地块数) × (1 + 0.5 × 目标完成度)**，鼓励占地盘；每回合观察里都会给出计分公式、各势力分数与自己的名次
- 军队单位由城市划拨资源生成，每回合 2 行动点：`move_unit` 行军 (1) / `attack_city` 攻城 (1) / `loot_tile` 掠夺 (2) / `forced_march` 强行军 (0，烧资源) / `merge_units` 合并 (0) / `supply_unit` 补给 (1) / `disband_unit` 解散 (0)
- **动态占领**：地块仅在有 **≥5 资源**的单位驻留时归其控制，单位离开即回归中立；掠夺一次性取走该格资源
- **强行军**：额外移动用单位资源支付，第 1 格 4、第 2 格 8、第 3 格 16…（上限 5 格，须保留 ≥1），可拐弯，只增加移动、不增加攻击
- **战斗有损耗**：获胜方损失败方强度的 30%；城市防御 = 城市资源 + 驻防单位资源之和 + 5（有驻军时）
- 攻城：单位 > 防御 → 掠夺城市资源 30%、城市减半（下限 5）、守军全灭、进攻单位自动入城；否则进攻单位被消灭，守方按进攻强度 30% 受损
- **维护费**：每个单位每回合 1 资源，由城市支付，不足则单位减员；**未采取行动的单位每回合恢复 1 资源**
- 解散：单位站在己方城市上可原地解散，资源归还该城市
- **首都吞并**：攻占敌国首都并**保持一整回合**（前两回合为保护期，不可吞并）→ 吞并该国**全部城市**（每座保留 75%、下限 5）；该国**军队不转移、留在原地**成为游击势力，只有在**城市与军队都被消灭**时才判负
- 每个行动的结果都会回显**分数变化**（如 `[score 160 -> 165, +5]`）
- LangGraph `StateGraph` 回合流程：`observe → (reflect) → plan → act → tools → collect`，一回合内可生成并指挥多个单位，最后 `end_turn`
- FAISS + embedding 的语义记忆检索
- 每 3 回合触发一次反思（reflection），每回合生成计划（plan）
- 胜负规则：既无城市也无单位则淘汰；达到最大回合后按上面的公式评分，最高者胜
- 回合顺序每轮轮换；对局有固定回合上限，模型在每回合都能看到当前回合、剩余回合数、计分公式与实时排名
- 本地模型 / 云端 API 模型可切换，每个国家可独立选择不同或相同的提供商
- Web UI 实时 SSE 推送：回合、行动、结果、计划、反思、六边形地图、关系、得分
- 中 / 英界面切换，模型输出语言随界面切换
- 支持中断正在进行的模拟

## 游戏规则

### 棋盘与势力
- 六边形棋盘（pointy-top，轴向坐标，整体近似矩形），约 300+ 格、**26 座城市**按欧洲地理大致布局，全部为陆地（暂无海洋）。
- 四个势力与首都：**Germany (Berlin)、France (Paris)、United Kingdom (London)、Soviet Union (Moscow)**。
- 开局每个势力**只拥有自己的首都**（初始资源 100）；其余城市中立，资源随机 30–80。
- 中立地块随机资源 5–25，**永不增长**。
- 回合顺序**每轮轮换**（起始势力 = 轮次 % 势力数）；对局有固定回合上限（CLI 默认 20，Web 默认 10，可在 UI 或 `run(max_turns=...)` 覆盖）。

### 资源与经济
- 资源归属于**城市**与**单位**，没有国库；只有计分时才把受控资源相加。
- 收入：每回合每座**受控城市 +5**，**本国首都 +10**；城市资源**下限 5**（任何减少都不低于 5）。
- 维护费：每个单位每回合 **1**，由该势力的城市支付（按资源从多到少，保留下限 5）；不足的差额让单位减员（从资源最多的单位开始各 −1，归零消灭）。
- 恢复：单位在本回合**未采取任何行动**则回合结束 **+1**。⇒ 驻守单位净收支为 0，主动行动净 −1。
- 分数公式：`score = (受控资源 + CITY_SCORE×城市数 + TILE_SCORE×受控地块数) × (1 + GOAL_SCORE_BONUS×目标完成度)`。

### 占领（动态）
- 地块**不是永久领土**：只要有一个**资源 ≥ `OCCUPATION_MIN_RESOURCES`(5)** 的己方单位站在上面，该地块即归其控制；单位离开、或资源降到阈值以下，立即**回归中立**。
- 掠夺（`loot_tile`）：一次性取走所在地块的资源并使其清零（消耗 2 AP）。

### 单位
- **生成**：从己方城市划拨资源生成（`spawn_unit`，不耗 AP）。每回合生成上限 = 当前控制的城市数；可在同一城市生成多个；新单位**当回合即可行动**。
- **行动点**：每个单位每回合 **2 AP**。
- **堆叠**：同一格可存在任意多个单位；防守时按该格所有守军资源之和计算。

### 行动与工具
| 工具 | AP | 说明 |
| --- | --- | --- |
| `spawn_unit(city, amount)` | 0 | 从城市划拨资源生成单位 |
| `move_unit(unit_id, direction)` | 1 | 移动到相邻格；进入有敌军的格触发战斗；不可进入非己方城市 |
| `attack_city(unit_id, city)` | 1 | 攻击 1 格内的城市 |
| `loot_tile(unit_id)` | 2 | 掠夺所在地块资源，本回合不能再移动 |
| `forced_march(unit_id, directions)` | 0 | 烧资源强行军，额外移动（见下） |
| `merge_units(unit_id, other_id)` | 0 | 同格两单位合并，资源相加、AP 取最小 |
| `supply_unit(unit_id, city, amount)` | 1 | 3 格内皮，从城市划拨资源给单位 |
| `disband_unit(unit_id)` | 0 | 在己方城市上解散，资源归还城市 |
| `end_turn()` | - | 结束本势力回合 |

方向为 **E / SE / SW / W / NW / NE**。

### 战斗
- **城市防御** = 城市资源 + 该城驻防单位资源之和 +（有驻军时）+5。
- **陆战防御** = 目标格守军资源之和 +（有守军时）+5。
- 进攻方资源 **> 防御** → 胜：
  - 陆战：守军单位全部被消灭；进攻方损失 `floor(防御 × ATTACK_ATTRITION)`（默认 30%）。
  - 攻城：掠夺城市资源的 `CITY_LOOT_RATE`（30%）；城市资源减半（下限 5）；守军全灭；进攻单位**自动进入城市格**；进攻方同样按防御的 30% 受损。
- 进攻方资源 **< 防御** → 败：进攻单位被消灭；守方按进攻方强度的 30% 受损（先扣驻军、再扣城市）。
- **平局判守方胜**。
- 攻击会让双方关系 −30；关系每回合向 0 回归 `RELATION_DECAY`(3)。

### 强行军（Forced March）
- 用单位自身资源购买**额外移动**：第 1 格 4、第 2 格 8、第 3 格 16、第 4 格 32、第 5 格 64（`EXTRA_MOVE_BASE`、`EXTRA_MOVE_GROWTH`），每回合上限 5 格。
- 烧完后单位必须保留 ≥ `MIN_FORCED_MARCH_RESERVE`(1)。
- 可拐弯；路径中间不能有敌方单位、不可进入非己方城市；**只增加移动、不增加攻击**。
- 算作行动（当回合不享受恢复）；烧到 <5 会顺带失去所在地块控制。

### 首都与吞并
- 首都每回合 **+10** 收入（仅本国的首都）。
- **攻占**敌国首都后进入"保持中"状态：需**保持一整个回合**；前 `CAPITAL_ANNEX_PROTECT_TURNS`(2) 回合为**保护期**，不可吞并。期间首都易主则计时取消/转移。
- **吞并**触发：原主**全部城市**归当前控制方，每座保留 `ANNEX_CITY_KEEP`(0.75)、下限 5；原主**军事单位不转移、留在原地**，成为**只有军队的游击势力**。
- 游击势力没有城市 → 维护费无处支付 → 逐回合减员（未行动靠 +1 恢复抵消），需尽快夺回城市。
- **淘汰**：既无城市也无单位才判负（联盟中亦须有生力量全灭）。
- 旧版"丢都 −2/−1、夺回 +5/+2"规则**已移除**。

### 计分与胜负
- 分数：`(受控资源 + 50×城市数 + 5×受控地块数) × (1 + 0.5×目标完成度)`；被淘汰势力 0 分。
- 结束：仅剩一个势力，或达到回合上限按分数排名，最高者胜。
- **分数透明**：每回合观察文本都会给出目标、计分公式、各势力分数、自己的名次与分差、上回合分数变化；每个行动结果会回显分数变化（如 `[score 160 -> 165, +5]`）。
- Web UI 提供**实时排名面板**（分数、城市数、领地数，高亮领先者）。

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

> 本地默认模型为 `qwen3.5:latest`（四个国家相同）。embedding 默认为智谱 `embedding-3`，设 `EMBEDDING_PROVIDER=ollama` 时使用本地 `qwen3-embedding:0.6b`。

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
5. 右侧 Live Feed 实时显示行动、结果、Plan 与 Reflection，并用高亮广播重要事件（城市被攻占、首都被占领/夺回/吞并、单位被歼灭、势力被淘汰）；左侧自上而下为**实时排名面板**、世界地图、国家面板与关系矩阵。国家卡片可点击标题折叠/展开，排名面板按分数显示名次、城市数与领地数并高亮领先者。
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
- `CITY_SCORE`（50）/ `TILE_SCORE`（5）：每座城市 / 每块受控地块计入最终分数的分值。
- `OCCUPATION_MIN_RESOURCES`（5）：控制地块所需的最低单位资源。
- `CAPITAL_ANNEX_PROTECT_TURNS`（2）：前 N 回合首都不可被吞并。
- `CAPITAL_HOLD_ROUNDS`（1）：攻占首都后需保持控制的回合数。
- `ANNEX_CITY_KEEP`（0.75）：吞并移交城市时保留的资源比例。
- `EXTRA_MOVE_BASE`（4）/ `EXTRA_MOVE_GROWTH`（2）：强行军第 1 格费用与逐格增长倍数。
- `MAX_FORCED_MARCH`（5）：每回合强行军上限格数。
- `MIN_FORCED_MARCH_RESERVE`（1）：强行军后单位必须保留的资源。
- `MAX_TURNS`：CLI 默认 20，Web UI 默认 10，均可在 UI 或 `run(max_turns=...)` 覆盖。
- `REFLECT_EVERY`：默认每 3 回合反思一次（`agent_graph/graph.py`）。
- 目标权重：`primary 0.5 / secondary 0.3 / tertiary 0.2`（`world/environment.py`）。
- 本地代理：若系统开启了全局代理，Python httpx 可能把 `127.0.0.1` 也走代理导致 Ollama 502。`agent/llm_factory.py` 已自动为 `127.0.0.1` / `localhost` / `::1` 设置 `NO_PROXY`。
