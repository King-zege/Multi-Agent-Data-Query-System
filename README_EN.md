# Multi-Agent Data Query System

> A LangGraph-powered multi-agent system that translates natural language into SQL, analyzes data, searches the web, and generates visualizations — with long-term memory and streaming output.

[中文文档](README.md)

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/LangGraph-1.0+-orange.svg" alt="LangGraph 1.0+">
  <img src="https://img.shields.io/badge/database-SQLite%20%7C%20MySQL%20%7C%20PostgreSQL-lightgrey.svg" alt="Multi-DB">
</p>

## Overview

The system uses a **one-master / three-worker** agent architecture to answer natural language questions about your database:

- **MasterAgent** — intent classification (6-way), multi-step orchestration, cache-aware dispatch
- **SQLQueryAgent** — NL2SQL generation with few-shot prompting, reflection-based self-correction, and automatic result caching
- **DataAnalysisAgent** — statistical insights, trend detection, anomaly flagging, and ECharts chart generation
- **WebSearchAgent** — Tavily-powered internet search with graceful degradation

```
User Question
      │
      ▼
  MasterAgent (intent router + cache matcher)
      │
      ├── simple_answer ────────────────► Direct reply
      ├── sql_only ─────────────────────► SQLAgent → Summarize
      ├── analysis_only ────────────────► AnalysisAgent (uses cached data if available)
      ├── sql_and_analysis ─────────────► SQLAgent → AnalysisAgent
      ├── web_search ───────────────────► WebSearchAgent → Synthesize
      └── search_and_sql ──────────────► SQLAgent + WebSearchAgent → Compare
```

## Key Features

### NL2SQL with Self-Correction
- Few-shot prompting with CTE examples tailored to each database dialect (SQLite / MySQL / PostgreSQL)
- **Reflection loop**: on execution error, the error message is fed back to the LLM for automatic SQL repair (up to N retries)
- **Two-layer review**: structural SQL review + result-set sanity check before returning results

### Multi-Database Support
- SQLite, MySQL, and PostgreSQL via a unified MCP (Model Context Protocol) server
- Database-specific SQL dialect rules injected into prompts
- Cross-DB evaluation framework with 99 test cases

### SQL Result Cache
- SQLite-persisted cache with 5-round TTL (hit refreshes TTL to 5)
- Cache inventory injected into MasterAgent only (never into sql_agent prompts — keeps SQL generation clean)
- MasterAgent dispatches on cache hits: EXACT match skips SQL agent entirely, PARTIAL match feeds cached data to analysis/search
- 2x speedup on repeated queries

### Web Search & Cross-Reference
- Tavily API integration for internet search
- **search_and_sql** mode: compares internal database results against industry/ market data
- Graceful degradation when Tavily key is not configured

### Dual-Layer Memory
- **Short-term**: conversation compression when messages exceed threshold (token-based, LLM-powered summarization)
- **Long-term**: SQLite-backed persistent user preferences and knowledge extraction (auto-triggered after 6 rounds)

### Streaming Output (SSE)
7 event types streamed in real-time: `status`, `intent`, `sql`, `sources`, `chart`, `chunk`, `done`

### Web UI
Built-in Flask web interface with ECharts visualization, Markdown rendering, syntax highlighting, and streaming typewriter effect.

### Evaluation Framework
- 99 curated test cases across 7 categories (intent recognition, SQL generation, end-to-end, error handling, execution accuracy, exact match, result overlap)
- Mock mode for fast iteration (no API cost) and live mode for real LLM evaluation
- Quantitative metrics: Jaccard similarity, result overlap ratio, exact match rate
- Cross-DB consistency testing (same question across SQLite / MySQL / PostgreSQL)

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/King-zege/Multi-Agent-Data-Query-System.git
cd Multi-Agent-Data-Query-System
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
# Copy the example environment file and fill in your keys
cp .env.example .env          # Linux / macOS
copy .env.example .env        # Windows
```

Edit `.env` and fill in your API keys:

```env
LLM_API_KEY="your_api_key_here"      # Required
TAVILY_API_KEY="your_tavily_key_here" # Optional (web search)
```

The system supports any OpenAI-compatible API. Choose your provider in `config/config.yaml`:

```yaml
llm:
  provider: "minimax"          # minimax | dashscope | zhipu
  model: "MiniMax-M2.7"
  api_key: "${LLM_API_KEY}"    # Reads from .env via ${VAR_NAME}
  temperature: 0.1
  max_tokens: 2048
```

> **Security**: `.env` is in `.gitignore` and will never be committed. Never paste real keys into `config.yaml` — use `${VAR_NAME}` references instead.

### 3. Initialize databases

```bash
python data/init_db.py          # Business database (company.db)
python data/init_memory_db.py   # Long-term memory database
```

### 4. Start the web interface

```bash
# Windows
start_web.bat

# Linux / macOS
./start_web.sh
```

Open **http://localhost:5000** in your browser.

### 5. CLI mode (alternative)

```bash
python agent.py
```

Special commands inside the CLI: `new` (fresh session), `info` (user profile), `exit` / `quit`.

## Configuration

Full `config/config.yaml` reference:

```yaml
llm:
  provider: "minimax"             # minimax | dashscope | zhipu
  model: "MiniMax-M2.7"
  api_key: "${LLM_API_KEY}"       # Env-var reference (recommended)
  temperature: 0.1
  max_tokens: 2048

database:
  type: "sqlite"                  # sqlite | mysql | postgresql
  path: "./data/company.db"       # SQLite path
  host: "localhost"               # MySQL / PG host
  port: 3306
  database: "company"
  username: "root"
  password: ""

nl2sql:
  num_examples: 5                 # Few-shot examples (max 5)

memory:
  long_term_db: "./data/long_term_memory.db"
  short_term_max_tokens: 1000
  compression_threshold: 10
  auto_extract_knowledge: true

search:
  tavily_api_key: ""              # Leave empty to read from TAVILY_API_KEY env var
  max_results: 5
```

## Project Structure

```
.
├── agents/                          # Agent modules
│   ├── __init__.py
│   ├── base.py                      # Base agent class
│   ├── master_agent.py              # Master orchestrator (intent, cache, dispatch)
│   ├── sql_agent.py                 # NL2SQL agent (generation, self-correction, cache CRUD)
│   ├── analysis_agent.py            # Data analysis agent (stats, charts)
│   └── search_agent.py              # Web search agent (Tavily)
├── memory/                          # Memory system
│   ├── long_term_memory.py          # SQLite-backed persistent memory
│   └── memory_extractor.py          # LLM-based knowledge extraction
├── eval/                            # Evaluation framework
│   ├── eval_cases_v2.json           # 99 test cases
│   ├── evaluator.py                 # Eval runner (mock + live modes)
│   ├── metrics.py                   # Quantitative metrics
│   ├── reporter.py                  # Report generation
│   ├── llm_judge.py                 # LLM-as-judge scoring
│   └── run_eval.py                  # CLI entry point
├── config/
│   └── config.yaml                  # Main configuration
├── data/
│   ├── init_db.py                   # Business DB initializer
│   ├── init_memory_db.py            # Memory DB initializer
│   └── company.db                   # Sample company database
├── scripts/
│   ├── generate_eval_cases.py       # Eval case generator
│   └── import_falcon.py             # Falcon dataset importer
├── tests/
│   ├── test_agents/                 # Agent unit tests
│   ├── test_integration/            # Integration tests
│   ├── test_memory/                 # Memory system tests
│   └── test_utils/                  # Test helpers & fake LLM
├── static/
│   ├── index.html                   # Web UI
│   ├── style.css
│   └── app.js                       # SSE streaming, ECharts rendering
├── agent.py                         # Main entry: MultiAgentSystem class
├── app.py                           # Flask REST API server
├── mcp_sql_server.py                # MCP SQL execution server (multi-DB)
├── prompts.py                       # All prompt templates
├── requirements.txt
└── start_web.sh / start_web.bat
```

## REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/login` | POST | User login, returns stored preferences & knowledge |
| `/api/query` | POST | Blocking query (full response) |
| `/api/query_stream` | POST | SSE streaming query (recommended for UI) |
| `/api/new_session` | POST | Start a new conversation session |
| `/api/user_info` | POST | Get user profile and extracted knowledge |
| `/api/health` | GET | Health check with feature availability |

### SSE Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{message}` | Current processing step |
| `intent` | `{intent}` | Classified intent label |
| `sql` | `{sql, attempt}` | Generated SQL (with retry count) |
| `sources` | `{labels, urls}` | Web search source URLs |
| `chart` | `{config}` | ECharts chart configuration JSON |
| `chunk` | `{content}` | Answer text fragment (typewriter effect) |
| `done` | `{answer}` | Completion signal with full answer |

## Evaluation

The evaluation framework measures system quality without relying on exact string matching:

```bash
# Fast mock mode (no API cost, uses pattern matching)
python -m eval.run_eval --mode mock

# Cross-DB consistency test
python -m eval.run_eval --mode mock --cross-db

# Live LLM evaluation (specific database)
python -m eval.run_eval --mode live --db-type mysql

# All databases
python -m eval.run_eval --mode live --db-type all

# Specific category only
python -m eval.run_eval --mode live --category sql_execution_accuracy
```

**7 evaluation categories**: intent recognition, SQL generation, end-to-end, error handling, execution accuracy, exact match, result set overlap.

## Example Questions

**Data queries**
- "How many employees are in the R&D department? What are their positions?"
- "Which employees have a base salary above 30,000?"
- "What is the highest-paid department?"

**Query + analysis**
- "Compare average salaries across the R&D, Product, and Design departments"
- "Find the top 10 highest-paid employees and analyze their department distribution"

**Web search**
- "What is the average salary for software engineers in the internet industry in 2025?"
- "What are the current trends in AI employment?"

**Cross-reference (internal + external)**
- "How does our R&D department salary compare to the industry average?"
- "Where does our company's compensation structure stand in the industry?"

**Analysis only** (uses cached data when available)
- "Analyze the last query results for me"
- "Summarize the salary distribution we just looked at"

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph 1.0+ |
| LLM Interface | LangChain + OpenAI-compatible API |
| Supported LLMs | MiniMax M2.7, Qwen (DashScope), GLM (Zhipu) |
| Web Search | Tavily API |
| Database Protocol | MCP (Model Context Protocol) via FastMCP |
| Databases | SQLite, MySQL, PostgreSQL (SQLAlchemy) |
| Web Framework | Flask + Flask-CORS |
| Frontend | ECharts, marked.js, highlight.js |
| Logging | Python logging + Rich (terminal) |

## Roadmap

- [ ] Parallel agent execution for independent sub-tasks
- [ ] Report generation agent (PDF / Excel export)
- [ ] Anomaly detection agent (proactive data monitoring)
- [ ] Vector-based semantic memory (ChromaDB)
- [ ] Schema-aware table selection for large databases (>100 tables)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

Make sure to run the existing test suite before submitting:

```bash
pytest tests/ -v
```

## License

MIT — see the [LICENSE](LICENSE) file for details.
