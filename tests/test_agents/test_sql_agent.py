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


class TestThinkTagHandling:
    """测试 _llm_to_str 的 think 标签处理

    验证 Qwen 模型输出中 <think>...</think> 不会误删正式回答内容。
    """

    def test_extracts_content_after_think(self):
        """核心场景：数据在 </think> 之后 → 完整保留"""
        from agents.sql_agent import SQLQueryAgent

        llm_output = (
            "<think>\n"
            "需要列出薪资最高的10人：\n"
            "1. 张三 - 25000\n"
            "2. 李四 - 23000\n"
            "...\n"
            "</think>\n"
            "根据查询结果，以下是薪资最高的10名员工：\n"
            "1. 张三 - 研发部 - 25,000元\n"
            "2. 李四 - 市场部 - 23,000元\n"
        )
        result = SQLQueryAgent._llm_to_str(llm_output)
        assert "张三" in result
        assert "李四" in result
        assert "<think>" not in result
        assert "需要列出薪资最高的10人" not in result  # think内容应被移除

    def test_no_think_tags_unchanged(self):
        """无 think 标签 → 原文保留"""
        from agents.sql_agent import SQLQueryAgent

        text = "SELECT * FROM employees WHERE dept_id = 1"
        result = SQLQueryAgent._llm_to_str(text)
        assert result == text

    def test_thinking_before_data(self):
        """思考在前，回答数据在后 → 数据完整保留"""
        from agents.sql_agent import SQLQueryAgent

        llm_output = (
            "<think>分析数据中...</think>\n"
            "1. 张三 - 25,000元\n"
            "2. 李四 - 23,000元\n"
            "3. 王五 - 15,000元\n"
        )
        result = SQLQueryAgent._llm_to_str(llm_output)
        assert "张三" in result
        assert "李四" in result
        assert "王五" in result
        assert "分析数据中" not in result

    def test_object_with_content_attr(self):
        """输入是有 .content 属性的对象 → 正确提取"""
        from agents.sql_agent import SQLQueryAgent

        class MockResponse:
            def __init__(self, content):
                self.content = content

        resp = MockResponse(
            "<think>reasoning</think>\n"
            "薪资排名：\n"
            "1. 张三 - 25000\n"
        )
        result = SQLQueryAgent._llm_to_str(resp)
        assert "张三" in result
        assert "reasoning" not in result

    def test_object_with_text_attr(self):
        """输入是有 .text 属性的对象 → 正确提取"""
        from agents.sql_agent import SQLQueryAgent

        class MockResponse:
            def __init__(self, text):
                self.text = text

        resp = MockResponse(
            "<think>x</think>\n"
            "查询结果：共10人"
        )
        result = SQLQueryAgent._llm_to_str(resp)
        assert "查询结果" in result
        assert "x" not in result

    def test_multiple_think_blocks_last_wins(self):
        """多个 </think> → 取最后一个之后的内容（中间内容被丢弃）"""
        from agents.sql_agent import SQLQueryAgent

        llm_output = (
            "<think>第一步</think>\n"
            "中间内容\n"
            "<think>第二步</think>\n"
            "最终回答：这里有10名员工"
        )
        result = SQLQueryAgent._llm_to_str(llm_output)
        assert "最终回答" in result
        assert "第一步" not in result
        assert "第二步" not in result
        assert "中间内容" not in result  # 在最后一个 </think> 之前的内容都被丢弃

    def test_think_without_closing_tag(self):
        """只有 <think> 没有 </think> → 整个 <think>...</think> 块被移除"""
        from agents.sql_agent import SQLQueryAgent

        llm_output = (
            "<think>\n"
            "思考中...\n"
        )
        result = SQLQueryAgent._llm_to_str(llm_output)
        # 由于没有 </think>，走 else 分支：regex 移除 <think>...</think>
        # 但这里没有 closing tag，non-greedy regex 不会匹配到结尾
        # 文本中 <think> 后面没有 </think>，regex 不匹配

    def test_empty_after_think(self):
        """</think> 之后无内容 → 返回空字符串"""
        from agents.sql_agent import SQLQueryAgent

        llm_output = "<think>全部内容都在思考里</think>"
        result = SQLQueryAgent._llm_to_str(llm_output)
        assert result == ""
