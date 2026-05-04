"""
SQL查询子智能体

负责将自然语言转换为SQL并执行查询，支持自动纠错循环（最多2次重试）。
Reflection 模式：执行失败时将错误信息反馈给 LLM 重新生成。

通过 MCP 子进程执行 SQL 和获取 Schema，支持 SQLite / MySQL / PostgreSQL。
"""

import json
import sys
import asyncio
import hashlib
import sqlite3
import concurrent.futures
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_core.language_models import BaseLLM
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.append(str(Path(__file__).parent.parent))
from agents.base import BaseSubAgent
from prompts import get_few_shot_prompt, get_sql_correction_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class SQLQueryAgent(BaseSubAgent):
    """SQL查询子智能体，支持自动纠错循环（ReAct/Reflection 模式）

    通过 MCP 子进程执行 SQL，支持 SQLite / MySQL / PostgreSQL。
    """

    intent_name = "sql_only"
    node_name = "call_sql"

    def __init__(self, llm: BaseLLM, db_config: Dict[str, Any], num_examples: int = 3,
                 cache_db_path: str = "data/sql_cache.db"):
        """初始化SQL查询智能体

        Args:
            llm: 语言模型实例
            db_config: 数据库配置字典，格式：
                {"type": "sqlite", "path": "./data/company.db"}
                {"type": "mysql", "host": "...", "port": 3306, "database": "...",
                 "username": "...", "password": "..."}
                {"type": "postgresql", ...}
            num_examples: Few-shot示例数量
            cache_db_path: SQL结果缓存数据库路径
        """
        super().__init__(llm)
        self.db_config = db_config
        self.num_examples = num_examples
        self._schema_cache = None
        self._cache_db_path = cache_db_path
        self._init_cache_db()

    # ---- BaseSubAgent 接口 ----

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 SQL 查询，将结果写入 state"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")

        result = self.query(question, thread_id=thread_id)
        state["sql_result"] = result
        state["metadata"]["sql_result"] = result

        # 保存到会话数据
        if "_session_data" not in state["metadata"]:
            state["metadata"]["_session_data"] = {}
        session_data = state["metadata"]["_session_data"]
        if thread_id not in session_data:
            session_data[thread_id] = {}
        session_data[thread_id]["last_sql_result"] = result

        return state

    # ---- 内部方法 ----

    @staticmethod
    def _llm_to_str(result) -> str:
        """安全地从 LLM 返回值中提取文本，处理 Qwen 模型 think 标签"""
        import re
        if isinstance(result, str):
            text = result
        elif hasattr(result, 'content'):
            text = str(result.content)
        elif hasattr(result, 'text'):
            text = str(result.text)
        else:
            text = str(result)
        think_end = text.rfind('</think>')
        if think_end != -1:
            text = text[think_end + len('</think>'):].strip()
        else:
            text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        return text

    def _get_schema(self) -> str:
        """获取数据库 Schema（通过 MCP get_schema 工具，不走直接连接）"""
        if self._schema_cache is not None:
            return self._schema_cache

        schema_json = self._run_async(self._get_schema_via_mcp())
        try:
            tables = json.loads(schema_json)
        except json.JSONDecodeError:
            return "Schema 获取失败"

        if isinstance(tables, dict) and "error" in tables:
            return f"Schema 获取失败: {tables['error']}"

        text = self._format_schema_text(tables)
        self._schema_cache = text
        return text

    def _format_schema_text(self, tables: list) -> str:
        """将 Schema JSON 转为 LLM prompt 友好文本"""
        lines = []
        for table in tables:
            lines.append(f"\n表：{table['name']}")
            for col in table["columns"]:
                pk = " (主键)" if col.get("primary_key") else ""
                notnull = "" if col.get("nullable", True) else " NOT NULL"
                line = f"  - {col['name']}: {col['type']}{notnull}{pk}"
                samples = col.get("sample_values", [])
                if samples:
                    line += f"  [样例: {', '.join(samples[:5])}]"
                lines.append(line)
        return "\n".join(lines).strip()

    def _clean_sql(self, sql: str) -> str:
        """清理SQL语句（移除代码块标记和多余前缀）"""
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]

        prefixes = ["SQL：", "SQL:", "sql:", "sql："]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):]
                break

        if sql.endswith("```"):
            sql = sql[:-3]

        return sql.strip()

    def _generate_sql(self, question: str) -> str:
        """生成SQL语句"""
        schema = self._get_schema()
        db_type = self.db_config.get("type", "sqlite")
        prompt = get_few_shot_prompt(
            question=question,
            schema=schema,
            db_type=db_type,
            num_examples=self.num_examples
        )
        sql = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(sql)

    def _correct_sql(self, question: str, original_sql: str, error_msg: str, attempt: int) -> str:
        """SQL 自动纠错（Reflection 模式）

        Args:
            question: 用户原始问题
            original_sql: 出错的 SQL 语句
            error_msg: 错误信息
            attempt: 当前重试次数（从1开始）

        Returns:
            修正后的 SQL 语句
        """
        schema = self._get_schema()
        db_type = self.db_config.get("type", "sqlite")
        prompt = get_sql_correction_prompt(
            question=question,
            schema=schema,
            original_sql=original_sql,
            error_msg=error_msg,
            attempt=attempt,
            db_type=db_type
        )
        corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(corrected)

    def _build_mcp_env(self) -> dict:
        """构建 MCP 子进程所需的环境变量"""
        import os
        env = {**os.environ}
        db_type = self.db_config.get("type", "sqlite")
        env["DB_TYPE"] = db_type

        if db_type == "sqlite":
            env["DB_PATH"] = self.db_config.get("path", "./data/company.db")
        else:
            env["DB_HOST"] = str(self.db_config.get("host", "localhost"))
            env["DB_PORT"] = str(self.db_config.get("port", 3306))
            env["DB_NAME"] = str(self.db_config.get("database", ""))
            env["DB_USER"] = str(self.db_config.get("username", ""))
            env["DB_PASSWORD"] = str(self.db_config.get("password", ""))

        return env

    async def _execute_sql_via_mcp(self, sql: str) -> str:
        """通过 MCP 工具执行 SQL

        Args:
            sql: SQL语句

        Returns:
            查询结果 JSON 字符串
        """
        mcp_script = Path(__file__).parent.parent / "mcp_sql_server.py"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_script)],
            env=self._build_mcp_env(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "execute_sql",
                    arguments={"sql": sql}
                )

                if result.content:
                    return result.content[0].text
                return json.dumps({"error": "无返回结果"})

    async def _get_schema_via_mcp(self) -> str:
        """通过 MCP 工具获取数据库 Schema

        Returns:
            Schema JSON 字符串
        """
        mcp_script = Path(__file__).parent.parent / "mcp_sql_server.py"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_script)],
            env=self._build_mcp_env(),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "get_schema",
                    arguments={}
                )

                if result.content:
                    return result.content[0].text
                return json.dumps({"error": "无返回结果"})

    def _run_async(self, coro):
        """安全地执行异步代码，兼容已有事件循环"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)

    def _is_empty_result(self, data: str) -> bool:
        """检查查询结果是否为空列表"""
        if not data:
            return True
        try:
            rows = json.loads(data)
            return isinstance(rows, list) and len(rows) == 0
        except json.JSONDecodeError:
            return False

    def _extract_where_string_values(self, sql: str) -> list:
        """从 SQL WHERE 子句中提取字符串比较模式

        Returns:
            [(column_name, string_value), ...]
        """
        import re
        pairs = []
        # 匹配 col = 'value' 或 col = "value" 模式
        for m in re.finditer(r"(\w+)\s*=\s*'([^']+)'", sql):
            col = m.group(1)
            val = m.group(2)
            if col.lower() not in ('limit',):
                pairs.append((col, val))
        # 匹配 col IN ('v1', 'v2') 模式
        for m in re.finditer(r"(\w+)\s+IN\s+\(([^)]+)\)", sql, re.IGNORECASE):
            col = m.group(1)
            vals = re.findall(r"'([^']+)'", m.group(2))
            for val in vals:
                pairs.append((col, val))
        return pairs

    def _probe_column_values(self, table: str, column: str) -> list:
        """查询某列在数据库中的实际 distinct 值（最多10个）"""
        try:
            sql = f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT 10'
            result_json = self._run_async(self._execute_sql_via_mcp(sql))
            rows = json.loads(result_json)
            if isinstance(rows, list):
                return [list(row.values())[0] for row in rows if row]
            return []
        except Exception:
            return []

    def _get_value_correction_prompt(self, question: str, original_sql: str,
                                       value_mismatches: list) -> str:
        """构建值纠正提示词

        Args:
            question: 原始问题
            original_sql: 返回0行的SQL
            value_mismatches: [(col, guessed_value, actual_values_list), ...]
        """
        details = []
        for col, guessed, actuals in value_mismatches:
            details.append(
                f"  列 {col}: 你的SQL用了 '{guessed}'，"
                f"数据库中实际值样例: {actuals}"
            )

        schema = self._get_schema()
        db_type = self.db_config.get("type", "sqlite")

        return f"""你生成的SQL查询返回了0行（空结果），很可能是因为WHERE条件中的字符串值与数据库中的实际值不匹配。

数据库类型：{db_type}

Schema：
{schema}

原始问题：{question}

原始SQL（返回0行）：
{original_sql}

值不匹配详情：
{chr(10).join(details)}

请根据实际值修正SQL。规则：
- 字符串值使用小写英文匹配（因为列名是英文）
- 如果实际值中有与你猜测值含义相近的值（如'active'对应'活跃'），请使用实际值
- 只修改WHERE条件中的字符串值，保持SQL结构不变

直接返回修正后的SQL语句，不要任何解释。"""

    # ---- 双层语义审查 ----

    def _review_sql_structure(self, question: str, schema: str, sql: str) -> str | None:
        """Layer 1: 审查 SQL 结构是否正确实现问题语义

        Returns:
            问题描述字符串，无问题则返回 None
        """
        from prompts import get_sql_review_prompt
        db_type = self.db_config.get("type", "sqlite")
        prompt = get_sql_review_prompt(question, schema, sql, db_type)
        response = self._llm_to_str(self.llm.invoke(prompt)).strip()
        if response.upper().startswith("OK"):
            return None
        return response

    def _review_results(self, question: str, sql: str,
                         result_sample: str, row_count: int) -> str | None:
        """Layer 2: 审查查询结果是否合理

        Returns:
            问题描述字符串，无问题则返回 None
        """
        from prompts import get_result_review_prompt
        db_type = self.db_config.get("type", "sqlite")
        prompt = get_result_review_prompt(question, sql, result_sample, row_count, db_type)
        response = self._llm_to_str(self.llm.invoke(prompt)).strip()
        if response.upper().startswith("OK"):
            return None
        return response

    def _semantic_review(self, question: str, sql: str, query_result: str) -> str | None:
        """并行执行双层审查，合并发现的问题

        Returns:
            合并后的问题描述，无问题则返回 None
        """
        schema = self._get_schema()
        db_type = self.db_config.get("type", "sqlite")

        rows = json.loads(query_result)
        row_count = len(rows) if isinstance(rows, list) else 0
        result_sample = json.dumps(rows[:5], ensure_ascii=False) if row_count > 0 else "空结果"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_struct = executor.submit(self._review_sql_structure, question, schema, sql)
            f_result = executor.submit(self._review_results, question, sql, result_sample, row_count)
            struct_issues = f_struct.result()
            result_issues = f_result.result()

        issues = []
        if struct_issues:
            issues.append(f"[SQL结构] {struct_issues}")
        if result_issues:
            issues.append(f"[结果] {result_issues}")
        return "; ".join(issues) if issues else None

    def _correct_for_semantics(self, question: str, sql: str, issues: str) -> str:
        """根据语义审查发现的问题修正 SQL"""
        schema = self._get_schema()
        db_type = self.db_config.get("type", "sqlite")

        prompt = f"""你是一个SQL专家。你的SQL被审查出以下语义问题，请修正。

数据库类型：{db_type}

Schema：
{schema}

用户问题：{question}

原始SQL：
{sql}

审查发现的问题：
{issues}

请根据问题描述修正SQL。只修改真正有问题的部分，保持其他部分不变。

直接返回修正后的SQL语句，不要任何解释。"""

        corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(corrected)

    # ---- SQL 结果缓存（SQLite 持久化） ----

    def _init_cache_db(self):
        """初始化缓存数据库及表结构"""
        Path(self._cache_db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sql_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    description TEXT NOT NULL,
                    sql_text TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    ttl INTEGER NOT NULL DEFAULT 5,
                    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    last_accessed_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(thread_id, cache_key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_thread ON sql_cache(thread_id)
            """)
            conn.commit()

    @staticmethod
    def _cache_key(sql: str) -> str:
        """归一化SQL → MD5哈希前16位"""
        normalized = " ".join(sql.strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def tick_cache(self, thread_id: str) -> int:
        """该thread_id下所有条目ttl减1，删除ttl≤0的条目"""
        with sqlite3.connect(self._cache_db_path) as conn:
            conn.execute("UPDATE sql_cache SET ttl = ttl - 1 WHERE thread_id = ?",
                        (thread_id,))
            cursor = conn.execute("DELETE FROM sql_cache WHERE ttl <= 0")
            conn.commit()
            return cursor.rowcount

    def get_cache_inventory(self, thread_id: str) -> dict:
        """返回 {cache_key: description, ...}，供MasterAgent调度使用"""
        with sqlite3.connect(self._cache_db_path) as conn:
            rows = conn.execute(
                "SELECT cache_key, description FROM sql_cache WHERE thread_id = ? AND ttl > 0",
                (thread_id,)
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_cached_result(self, thread_id: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果，命中时ttl重置为5"""
        with sqlite3.connect(self._cache_db_path) as conn:
            row = conn.execute(
                "SELECT result_json FROM sql_cache WHERE thread_id = ? AND cache_key = ? AND ttl > 0",
                (thread_id, cache_key)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE sql_cache SET ttl = 5, last_accessed_at = datetime('now','localtime') "
                    "WHERE thread_id = ? AND cache_key = ?",
                    (thread_id, cache_key)
                )
                conn.commit()
                return json.loads(row[0])
        return None

    def _generate_description(self, sql: str, result_sample: str) -> str:
        """用LLM生成一句中文描述，概括该SQL查询返回了什么数据"""
        prompt = f"""用一句简短的中文描述以下SQL查询返回了什么数据（不超过20个字）。

SQL：{sql}

查询结果样例（前3行）：
{result_sample}

只返回一句中文描述，不要任何解释。"""
        return self._llm_to_str(self.llm.invoke(prompt)).strip()

    def _cache_put(self, thread_id: str, sql: str, description: str, result: Dict[str, Any]):
        """存入缓存（INSERT OR REPLACE）"""
        key = self._cache_key(sql)
        result_json = json.dumps(result, ensure_ascii=False)
        import re
        description = re.sub(r'[\r\n]+', ' ', description).strip()
        with sqlite3.connect(self._cache_db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sql_cache
                   (thread_id, cache_key, description, sql_text, result_json, ttl, last_accessed_at)
                   VALUES (?, ?, ?, ?, ?, 5, datetime('now','localtime'))""",
                (thread_id, key, description, sql, result_json)
            )
            conn.commit()

    # ---- 主查询方法 ----

    def query(self, question: str, thread_id: str = "default", max_retries: int = 2) -> Dict[str, Any]:
        """执行查询，失败时自动纠错并重试（Reflection 循环）

        支持三层纠错：
        1. SQL执行报错 → 常规错误修复
        2. SQL执行成功但返回0行 → 探测实际值后修正WHERE条件
        3. SQL执行成功 → 双层语义审查（结构 + 结果），发现问题则修正

        Args:
            question: 用户问题
            thread_id: 会话线程ID，用于缓存隔离
            max_retries: 最大重试次数（默认2次）

        Returns:
            {
                "sql": 最终执行的SQL,
                "data": 查询结果JSON字符串（成功时）,
                "error": 错误信息（成功时为None）,
                "retry_count": 实际重试次数（0表示首次成功）,
                "review_issues": 语义审查发现的问题（None表示无问题或未审查）
            }
        """
        result = {
            "sql": None,
            "data": None,
            "error": None,
            "retry_count": 0,
            "review_issues": None,
        }

        try:
            sql = self._generate_sql(question)
            result["sql"] = sql

            if not sql:
                result["error"] = "未能生成有效的SQL"
                return result

            for attempt in range(max_retries):
                query_result = self._run_async(self._execute_sql_via_mcp(sql))
                result_data = json.loads(query_result)

                if isinstance(result_data, dict) and "error" in result_data:
                    error_msg = result_data["error"]

                    if attempt < max_retries - 1:
                        logger.info("SQL纠错 第%d次执行失败: %s，正在让LLM自动修复...", attempt + 1, error_msg)
                        sql = self._correct_sql(question, sql, error_msg, attempt + 1)
                        result["sql"] = sql
                        result["retry_count"] = attempt + 1
                    else:
                        result["error"] = f"SQL执行失败（已自动重试{attempt}次）: {error_msg}"
                elif self._is_empty_result(query_result):
                    # 查询成功但返回0行 → 值探测纠正
                    if attempt < max_retries - 1:
                        logger.info("SQL返回0行，探测实际值... (attempt %d)", attempt + 1)
                        where_pairs = self._extract_where_string_values(sql)
                        value_mismatches = []
                        schema_tables = json.loads(
                            self._run_async(self._get_schema_via_mcp())
                        )
                        # 构建列名→表名映射
                        col_to_table = {}
                        for t in schema_tables:
                            for c in t.get("columns", []):
                                col_to_table[c["name"].lower()] = t["name"]

                        for col, guessed_val in where_pairs:
                            table = col_to_table.get(col.lower())
                            if table:
                                actuals = self._probe_column_values(table, col)
                                if actuals and guessed_val.lower() not in [str(a).lower() for a in actuals]:
                                    value_mismatches.append((col, guessed_val, actuals))

                        if value_mismatches:
                            prompt = self._get_value_correction_prompt(
                                question, sql, value_mismatches
                            )
                            corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
                            sql = self._clean_sql(corrected)
                            result["sql"] = sql
                            result["retry_count"] = attempt + 1
                        else:
                            # 没有可探测的值→查不出来就是真没数据
                            break
                    else:
                        result["data"] = query_result
                        break
                else:
                    # 查询成功 → 语义审查（仅首次成功时）
                    result["data"] = query_result
                    if attempt == 0 and max_retries > 1:
                        issues = self._semantic_review(question, sql, query_result)
                        result["review_issues"] = issues
                        if issues:
                            logger.info("语义审查发现问题: %s", issues[:120])
                            sql = self._correct_for_semantics(question, sql, issues)
                            result["sql"] = sql
                            result["retry_count"] = 1
                            continue
                    if attempt > 0:
                        logger.info("SQL纠错 第%d次修复后执行成功", attempt)
                    break

        except Exception as e:
            result["error"] = f"查询失败: {str(e)}"

        # 成功后自动缓存（有数据且无错误）
        if result["data"] and not result["error"]:
            try:
                rows = json.loads(result["data"])
                if isinstance(rows, list) and len(rows) > 0:
                    sample = json.dumps(rows[:3], ensure_ascii=False)
                    description = self._generate_description(result["sql"], sample)
                    self._cache_put(thread_id, result["sql"], description, result)
                    logger.info("SQL结果已缓存: key=%s, description=%s",
                               self._cache_key(result["sql"]), description)
            except Exception:
                pass  # 缓存失败不影响主流程

        return result
