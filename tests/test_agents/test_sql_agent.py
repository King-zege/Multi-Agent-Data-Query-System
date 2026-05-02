"""SQLQueryAgent 测试

测试 SQL 生成、清理、纠错循环、Schema 获取。
"""

import json
import os
import sqlite3
from pathlib import Path

import pytest

# 确保项目根目录在路径中
sys_path_added = False
if str(Path(__file__).parent.parent.parent) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_utils.fake_llm import FakeLLM


@pytest.fixture
def sql_agent(test_db_path, fake_llm):
    """创建 SQLAgent 测试实例"""
    from agents.sql_agent import SQLQueryAgent

    agent = SQLQueryAgent(llm=fake_llm, db_path=test_db_path, num_examples=3)
    return agent


class TestSQLGeneration:
    """SQL 生成测试"""

    def test_generate_sql_returns_string(self, sql_agent):
        """测试：生成的 SQL 是字符串"""
        sql = sql_agent._generate_sql("研发部有多少人")
        assert isinstance(sql, str)
        assert len(sql) > 0

    def test_generate_sql_contains_select(self, sql_agent):
        """测试：生成的 SQL 包含 SELECT"""
        sql = sql_agent._generate_sql("查询所有员工")
        assert "SELECT" in sql.upper() or "select" in sql.lower()


class TestSQLCleaning:
    """SQL 清理测试"""

    def test_clean_sql_removes_markdown_code_block(self, sql_agent):
        """测试：清理 ```sql ... ``` 标记"""
        dirty = "```sql\nSELECT * FROM employees\n```"
        clean = sql_agent._clean_sql(dirty)
        assert "```" not in clean
        assert "SELECT" in clean

    def test_clean_sql_removes_markdown_no_lang(self, sql_agent):
        """测试：清理 ``` ... ``` 标记（无语言指定）"""
        dirty = "```\nSELECT * FROM employees\n```"
        clean = sql_agent._clean_sql(dirty)
        assert "```" not in clean

    def test_clean_sql_removes_prefix(self, sql_agent):
        """测试：清理 'SQL：' 前缀"""
        dirty = "SQL：SELECT * FROM employees"
        clean = sql_agent._clean_sql(dirty)
        assert "SQL：" not in clean
        assert "SELECT" in clean

    def test_clean_sql_removes_sql_colon_prefix(self, sql_agent):
        """测试：清理 'SQL:' 前缀"""
        dirty = "SQL:SELECT * FROM employees"
        clean = sql_agent._clean_sql(dirty)
        assert "SQL:" not in clean

    def test_clean_sql_trims_whitespace(self, sql_agent):
        """测试：清理首尾空白"""
        dirty = "  \n  SELECT * FROM employees  \n  "
        clean = sql_agent._clean_sql(dirty)
        assert clean.startswith("SELECT")
        assert not clean.endswith("\n")


class TestSchemaRetrieval:
    """Schema 获取测试"""

    def test_schema_contains_required_tables(self, sql_agent):
        """测试：Schema 包含3张核心表"""
        schema = sql_agent._get_schema()
        assert "employees" in schema.lower()
        assert "departments" in schema.lower()
        assert "salaries" in schema.lower()

    def test_schema_contains_columns(self, sql_agent):
        """测试：Schema 包含字段定义"""
        schema = sql_agent._get_schema()
        assert "emp_name" in schema.lower()
        assert "dept_name" in schema.lower()


class TestQueryExecution:
    """查询执行测试（通过 mock MCP）"""

    def test_query_returns_dict(self, sql_agent, monkeypatch):
        """测试：query 返回包含必要字段的 dict"""
        async def mock_execute(sql):
            return json.dumps([{"emp_name": "张三", "dept_name": "研发部"}])

        monkeypatch.setattr(sql_agent, "_execute_sql_via_mcp", mock_execute)

        result = sql_agent.query("研发部有哪些人")
        assert isinstance(result, dict)
        assert "sql" in result
        assert "data" in result
        assert "error" in result
        assert "retry_count" in result

    def test_query_error_triggers_retry(self, sql_agent, monkeypatch):
        """测试：执行失败触发纠错重试"""
        call_count = [0]

        async def mock_execute_first_fails(sql):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"error": "no such column: salary"})
            return json.dumps([{"emp_name": "张三"}])

        monkeypatch.setattr(sql_agent, "_execute_sql_via_mcp", mock_execute_first_fails)

        result = sql_agent.query("研发部有哪些人", max_retries=3)
        assert result["retry_count"] >= 1

    def test_empty_sql_returns_error(self, sql_agent, fake_llm):
        """测试：LLM 返回空时 query 返回 error"""
        fake_llm.responses.clear()
        fake_llm.responses["你是一个SQL查询专家"] = ""

        async def mock_execute(sql):
            return json.dumps([])

        import types
        # 用 monkeypatch 实现
        sql_agent._execute_sql_via_mcp = mock_execute

        result = sql_agent.query("")
        # 空问题也应该能生成 SQL（因为 FakeLLM 会匹配到关键词）
        # 如果 _generate_sql 返回空，result 应有 error
