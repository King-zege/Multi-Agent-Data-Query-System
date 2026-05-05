# 基于多智能体的自然语言数据库查询系统

> 基于 LangGraph 的一主三从多智能体架构，支持自然语言查询数据库、深度分析、联网搜索和数据可视化，内置双层记忆系统和 SQL 结果缓存。

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-1.0+-orange.svg" alt="LangGraph 1.0+">
  <img src="https://img.shields.io/badge/database-SQLite%20%7C%20MySQL%20%7C%20PostgreSQL-lightgrey.svg" alt="Multi-DB">
  <img src="https://img.shields.io/badge/coverage-99_cases-brightgreen.svg" alt="99 Eval Cases">
</p>

## 项目概述

系统采用 **一主三从** 的多智能体架构，将自然语言问题转化为 SQL 查询并进行分析：

- **MasterAgent（主智能体）** — 6 种意图分类、多步骤任务编排、缓存感知调度
- **SQLQueryAgent（SQL 查询智能体）** — Few-shot NL2SQL 生成、Reflection 自动纠错、结果自动缓存
- **DataAnalysisAgent（数据分析智能体）** — 统计洞察、趋势检测、异常发现、ECharts 图表生成
- **WebSearchAgent（联网搜索智能体）** — 基于 Tavily 的互联网搜索，未配置时优雅降级

```
用户问题
      │
      ▼
  MasterAgent（意图识别 + 缓存匹配）
      │
      ├── simple_answer ────────────────► 直接回复
      ├── sql_only ─────────────────────► SQL查询 → 汇总
      ├── analysis_only ────────────────► 数据分析（可直接用缓存）
      ├── sql_and_analysis ─────────────► SQL查询 → 数据分析
      ├── web_search ───────────────────► 联网搜索 → 综合回复
      └── search_and_sql ──────────────► SQL + 联网搜索 → 对比分析
```

## 核心功能

### NL2SQL 自动纠错
- 针对不同数据库方言（SQLite / MySQL / PostgreSQL）的差异化 Few-shot 提示词
- **Reflection 纠错环**：SQL 执行失败时，将错误信息反馈给 LLM 自动修复（最多 N 次重试）
- **双层审查机制**：SQL 结构语义审查 + 结果集合理性校验

### 多数据库支持
- 通过统一 MCP（模型上下文协议）服务器支持 SQLite、MySQL、PostgreSQL
- 提示词自动注入数据库方言规则
- 跨库一致性评测框架，99 道测试用例

### SQL 结果缓存
- 基于 SQLite 的持久化缓存，TTL 为 5 轮对话（命中刷新 TTL）
- 缓存清单仅注入 MasterAgent（不污染 SQL 生成 prompt）
- MasterAgent 智能调度：EXACT 命中跳过 SQL Agent 直接汇总，PARTIAL 命中将缓存数据传给分析/搜索 Agent
- 重复查询提速 2 倍

### 联网搜索与内外对比
- 集成 Tavily API 进行互联网搜索
- **search_and_sql 模式**：将公司内部数据与行业/市场数据进行对比分析
- 未配置搜索密钥时自动降级，不影响其他功能

### 双层记忆系统
- **短期记忆**：对话轮次/Token 超阈值时自动触发 LLM 压缩总结
- **长期记忆**：SQLite 持久化用户偏好与知识，6 轮对话后自动提取

### 流式输出（SSE）
7 种事件实时推送：`status`、`intent`、`sql`、`sources`、`chart`、`chunk`、`done`

### Web 前端
内置 Flask Web 界面，支持 ECharts 图表渲染、Markdown 渲染、代码高亮、流式打字效果。

### 评测框架
- 99 道测试用例，覆盖 7 大类别（意图识别、SQL 生成、端到端、错误处理、执行准确率、精确匹配、结果集重叠率）
- Mock 模式（零 API 消耗快速验证）+ Live 模式（真实 LLM 评估）
- 定量指标：Jaccard 相似度、结果集重叠率、精确匹配率
- 跨库一致性测试

## 快速开始

### 1. 克隆项目并安装依赖

```bash
git clone https://github.com/King-zege/SQL-Search-based-on-Muti-Agents.git
cd SQL-Search-based-on-Muti-Agents
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制环境变量模板文件，然后填入你的 Key
cp .env.example .env          # Linux / macOS
copy .env.example .env        # Windows
```

编辑 `.env` 文件，填入你的 API Key：

```env
LLM_API_KEY="你的API密钥"                # 必填
TAVILY_API_KEY="你的Tavily密钥"          # 可选（联网搜索功能）
```

系统支持任意 OpenAI 兼容接口，在 `config/config.yaml` 中选择你的 LLM 提供商：

```yaml
llm:
  provider: "minimax"          # minimax | dashscope | zhipu
  model: "MiniMax-M2.7"
  api_key: "${LLM_API_KEY}"    # 通过 ${VAR_NAME} 从 .env 读取
  temperature: 0.1
  max_tokens: 2048
```

> **安全提醒**：`.env` 已加入 `.gitignore`，不会被提交到 GitHub。切勿将真实 Key 直接写在 `config.yaml` 中，请使用 `${VAR_NAME}` 环境变量引用。

### 3. 初始化数据库

```bash
python data/init_db.py          # 业务数据库（公司员工/部门/薪资表）
python data/init_memory_db.py   # 长期记忆数据库
```

### 4. 启动 Web 界面

```bash
# Windows
start_web.bat

# Linux / macOS
./start_web.sh
```

浏览器访问 **http://localhost:5000**。

### 5. 命令行模式（可选）

```bash
python agent.py
```

命令行内特殊指令：`new`（开始新会话）、`info`（查看用户信息）、`exit` / `quit`（退出）。

## 配置说明

`config/config.yaml` 完整参考：

```yaml
llm:
  provider: "minimax"             # minimax | dashscope | zhipu
  model: "MiniMax-M2.7"
  api_key: "${LLM_API_KEY}"       # 推荐使用环境变量引用
  temperature: 0.1
  max_tokens: 2048

database:
  type: "sqlite"                  # sqlite | mysql | postgresql
  path: "./data/company.db"       # SQLite 文件路径
  host: "localhost"               # MySQL / PG 主机
  port: 3306                      # MySQL=3306, PostgreSQL=5432
  database: "company"
  username: "root"
  password: ""

nl2sql:
  num_examples: 5                 # Few-shot 示例数量（最多5个）

memory:
  long_term_db: "./data/long_term_memory.db"
  short_term_max_tokens: 1000     # 短期记忆压缩阈值
  compression_threshold: 10       # 超过10条消息开始压缩
  auto_extract_knowledge: true    # 自动提取用户知识

search:
  tavily_api_key: ""              # 留空则从 TAVILY_API_KEY 环境变量读取
  max_results: 5                  # 单次搜索最大返回结果
```

## 项目结构

```
.
├── agents/                          # 智能体模块
│   ├── base.py                      # 智能体基类
│   ├── master_agent.py              # 主智能体（意图路由、缓存调度、结果汇总）
│   ├── sql_agent.py                 # SQL 查询智能体（NL2SQL、纠错、缓存CRUD）
│   ├── analysis_agent.py            # 数据分析智能体（统计、图表）
│   └── search_agent.py              # 联网搜索智能体（Tavily）
├── memory/                          # 记忆系统
│   ├── long_term_memory.py          # 长期记忆（SQLite 持久化）
│   └── memory_extractor.py          # LLM 自动提取用户知识
├── eval/                            # 评测框架
│   ├── eval_cases_v2.json           # 99 道测试用例
│   ├── evaluator.py                 # 评测执行器（mock + live 模式）
│   ├── metrics.py                   # 定量指标计算
│   ├── reporter.py                  # 评测报告生成
│   ├── llm_judge.py                 # LLM-as-judge 打分
│   └── run_eval.py                  # 命令行入口
├── config/
│   └── config.yaml                  # 主配置文件
├── data/
│   ├── init_db.py                   # 业务数据库初始化
│   ├── init_memory_db.py            # 记忆数据库初始化
│   └── company.db                   # 示例公司数据库
├── scripts/
│   ├── generate_eval_cases.py       # 评测用例生成器
│   └── import_falcon.py             # Falcon 数据集导入
├── tests/
│   ├── test_agents/                 # 智能体单元测试
│   ├── test_integration/            # 集成测试
│   ├── test_memory/                 # 记忆系统测试
│   └── test_utils/                  # 测试工具（Fake LLM）
├── static/
│   ├── index.html                   # Web 前端页面
│   ├── style.css                    # 蓝紫渐变主题
│   └── app.js                       # SSE 流式接收、ECharts 渲染
├── agent.py                         # 主入口：MultiAgentSystem 类
├── app.py                           # Flask REST API 服务
├── mcp_sql_server.py                # MCP SQL 执行服务器（多数据库）
├── prompts.py                       # 所有提示词模板
├── requirements.txt
└── start_web.sh / start_web.bat
```

## REST API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/login` | POST | 用户登录，返回已存储的偏好与知识 |
| `/api/query` | POST | 阻塞式查询（返回完整回答） |
| `/api/query_stream` | POST | SSE 流式查询（前端推荐使用） |
| `/api/new_session` | POST | 创建新会话（清空短期记忆） |
| `/api/user_info` | POST | 获取用户画像与提取的知识 |
| `/api/health` | GET | 健康检查，返回各功能可用状态 |

### SSE 事件类型

| 事件 | 数据 | 说明 |
|------|------|------|
| `status` | `{message}` | 当前处理步骤 |
| `intent` | `{intent}` | 识别到的意图标签 |
| `sql` | `{sql, attempt}` | 生成的 SQL 语句（含重试次数） |
| `sources` | `{labels, urls}` | 联网搜索来源 URL 列表 |
| `chart` | `{config}` | ECharts 图表配置 JSON |
| `chunk` | `{content}` | 回答文字流片段（打字机效果） |
| `done` | `{answer}` | 流结束信号（含完整回答） |

## 评测系统

评测框架不依赖字符串精确匹配，而是从语义和结果集层面衡量系统质量：

```bash
# 快速 Mock 模式（不消耗 API，模式匹配验证）
python -m eval.run_eval --mode mock

# 跨库一致性测试
python -m eval.run_eval --mode mock --cross-db

# 真实 LLM 评估（指定数据库）
python -m eval.run_eval --mode live --db-type mysql

# 全数据库评估
python -m eval.run_eval --mode live --db-type all

# 仅运行特定类别
python -m eval.run_eval --mode live --category sql_execution_accuracy
```

**7 大评测类别**：`intent_recognition`、`sql_generation`、`end_to_end`、`error_handling`、`sql_execution_accuracy`、`result_exact_match`、`result_set_overlap`。

## 示例问题

**数据查询**
- 公司总共有多少个部门？每个部门分别在哪个城市？
- 哪些员工的基本工资超过 30000 元？
- 哪个部门的平均薪资最高？

**查询 + 分析**
- 对比公司研发部、产品部和设计部的平均薪资水平
- 找出薪资最高的 10 名员工，分析他们的职位和部门分布特征

**联网搜索**
- 2025 年互联网行业软件工程师的平均薪资是多少？
- 目前 AI 大模型领域的就业趋势如何？

**搜索 + SQL 联合对比**
- 我们公司研发部的薪资水平和行业平均水平相比怎么样？
- 公司的薪资结构在同行业中处于什么水平？

**仅分析**（缓存数据可用时跳过查询）
- 帮我分析一下上一次查询的数据
- 总结一下我们刚才看到的薪资分布

## 技术栈

| 层级 | 技术 |
|------|------|
| 智能体编排 | LangGraph 1.0+ |
| LLM 接口 | LangChain + OpenAI 兼容 API |
| 支持模型 | MiniMax M2.7、Qwen（DashScope）、GLM（Zhipu） |
| 联网搜索 | Tavily API |
| 数据库协议 | MCP（Model Context Protocol）via FastMCP |
| 数据库 | SQLite / MySQL / PostgreSQL（SQLAlchemy） |
| Web 框架 | Flask + Flask-CORS |
| 前端 | ECharts、marked.js、highlight.js |
| 日志 | Python logging + Rich（终端美化） |

## 开发计划

- [ ] 多智能体并行执行（独立子任务）
- [ ] 报表生成智能体（PDF / Excel 导出）
- [ ] 异常检测智能体（主动数据监控）
- [ ] 向量语义记忆（ChromaDB）
- [ ] 大数据库场景下的 Schema 智能检索

## 参与贡献

欢迎提交 Pull Request。重大修改请先开 Issue 讨论你的想法。

提交前请确保通过现有测试：

```bash
pytest tests/ -v
```

## 许可证

MIT — 详见 [LICENSE](LICENSE) 文件。
