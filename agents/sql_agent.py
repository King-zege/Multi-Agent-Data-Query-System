"""
SQL查询子智能体

负责将自然语言转换为SQL并执行查询，支持自动纠错循环（最多3次重试）。
Reflection 模式：执行失败时将错误信息反馈给 LLM 重新生成。
"""

import json
import sqlite3
import sys
import asyncio
import concurrent.futures
from typing import Dict, Any
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
    """SQL查询子智能体，支持自动纠错循环（ReAct/Reflection 模式）"""

    intent_name = "sql_only"
    node_name = "call_sql"

    def __init__(self, llm: BaseLLM, db_path: str, num_examples: int = 3):
        """初始化SQL查询智能体

        Args:
            llm: 语言模型实例
            db_path: 数据库路径
            num_examples: Few-shot示例数量
        """
        super().__init__(llm)
        self.db_path = db_path
        self.num_examples = num_examples

    # ---- BaseSubAgent 接口 ----

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 SQL 查询，将结果写入 state"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")

        result = self.query(question)
        state["sql_result"] = result
        state["metadata"]["sql_result"] = result

        # 保存到会话数据（通过 metadata 传递引用）
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
        """获取数据库Schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        schema_text = ""
        for table in tables:
            table_name = table[0]
            schema_text += f"\n表：{table_name}\n"
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                cid, name, dtype, notnull, default, pk = col
                pk_text = " (主键)" if pk else ""
                notnull_text = " NOT NULL" if notnull else ""
                schema_text += f"  - {name}: {dtype}{notnull_text}{pk_text}\n"
        
        conn.close()
        return schema_text.strip()
    
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
        prompt = get_few_shot_prompt(
            question=question,
            schema=schema,
            num_examples=self.num_examples
        )
        sql = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(sql)
    
    def _correct_sql(self, question: str, original_sql: str, error_msg: str, attempt: int) -> str:
        """SQL 自动纠错（Reflection 模式）
        
        将错误信息和原始 SQL 反馈给 LLM，要求其重新生成正确的 SQL。
        
        Args:
            question: 用户原始问题
            original_sql: 出错的 SQL 语句
            error_msg: 错误信息
            attempt: 当前重试次数（从1开始）
            
        Returns:
            修正后的 SQL 语句
        """
        schema = self._get_schema()
        prompt = get_sql_correction_prompt(
            question=question,
            schema=schema,
            original_sql=original_sql,
            error_msg=error_msg,
            attempt=attempt
        )
        corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(corrected)
    
    async def _execute_sql_via_mcp(self, sql: str) -> str:
        """通过MCP工具执行SQL
        
        Args:
            sql: SQL语句
            
        Returns:
            查询结果JSON字符串
        """
        mcp_script = Path(__file__).parent.parent / "mcp_sql_server.py"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_script)]
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
    
    def _run_async(self, coro):
        """安全地执行异步代码，兼容已有事件循环（如 httpx/openai 遗留的）"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)
    
    def query(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        """执行查询，失败时自动纠错并重试（Reflection 循环）
        
        流程：SQL生成 → 执行 → [失败] → 错误反馈给LLM → 重新生成 → 最多重试 max_retries 次
        
        Args:
            question: 用户问题
            max_retries: 最大重试次数（默认3次）
            
        Returns:
            {
                "sql": 最终执行的SQL,
                "data": 查询结果JSON字符串（成功时）,
                "error": 错误信息（成功时为None）,
                "retry_count": 实际重试次数（0表示首次成功）
            }
        """
        result = {
            "sql": None,
            "data": None,
            "error": None,
            "retry_count": 0
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
                else:
                    result["data"] = query_result
                    if attempt > 0:
                        logger.info("SQL纠错 第%d次修复后执行成功", attempt)
                    break
                    
        except Exception as e:
            result["error"] = f"查询失败: {str(e)}"
        
        return result

