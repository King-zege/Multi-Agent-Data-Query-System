"""
pytest 全局 fixtures

提供测试数据库、FakeLLM、Flask 测试客户端等共享资源。
每个测试模块独立重建数据库，保证隔离性。
"""

import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import pytest

# 确保项目根目录在路径最前面
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_utils.fake_llm import FakeLLM

# 临时设置 API Key（测试中不需要真实 API）
os.environ.setdefault("DASHSCOPE_API_KEY", "test-api-key-for-testing")
# 清空 TAVILY_API_KEY，确保搜索智能体测试默认不可用
os.environ.pop("TAVILY_API_KEY", None)


# ======================== 数据库 fixtures ========================

@pytest.fixture
def test_db_path(tmp_path):
    """创建测试用业务数据库（真实文件，每次测试独立）"""
    db_path = tmp_path / "test_company.db"

    from data.init_db import create_tables, insert_sample_data

    conn = sqlite3.connect(str(db_path))
    create_tables(conn)
    insert_sample_data(conn)
    conn.commit()
    conn.close()

    return str(db_path)


@pytest.fixture
def memory_db_path(tmp_path):
    """创建测试用长期记忆数据库（真实文件，每次测试独立）"""
    db_path = tmp_path / "test_memory.db"

    from data.init_memory_db import init_memory_database

    init_memory_database(str(db_path))

    return str(db_path)


# ======================== LLM fixtures ========================

@pytest.fixture
def fake_llm():
    """创建预设响应的 FakeLLM，覆盖 6 种意图 + SQL + 分析场景

    注意：关键词按问题内容匹配，而非 prompt 模板文本。
    意图识别 prompt 中包含所有意图的描述文本，所以不能用"简单问候"等作为关键词。
    """
    return FakeLLM(responses={
        # 意图识别 — 按问题内容精确匹配
        "研发部有多少名员工": "sql_only",
        "分析一下刚才的结果": "analysis_only",
        "分析我们公司各部门的薪资水平": "sql_and_analysis",
        "2024年互联网行业软件工程师": "web_search",
        "我们公司研发部薪资和行业平均水平": "search_and_sql",
        # 兜底意图（通用关键词，放在最后避免误匹配）
        "判断以下用户输入是否需要查询数据库": "需要查询",
        # SQL 生成
        "你是一个SQL查询专家": "SELECT e.emp_name, d.dept_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id",
        "修复一段出错的SQL": "SELECT e.emp_name FROM employees e WHERE e.dept_id = 1",
        # 数据分析
        "对以下数据进行深度分析": "## 数据分析报告\n\n### 1. 数据概览\n数据包含多条记录。\n\n### 2. 关键发现\n- 发现1\n- 发现2\n\n### 3. 建议\n建议优化。",
        # 图表配置
        "生成一个适合可视化的 ECharts": json.dumps({
            "title": {"text": "测试图表"},
            "tooltip": {},
            "xAxis": {"data": ["A", "B"]},
            "yAxis": {},
            "series": [{"type": "bar", "data": [1, 2]}]
        }),
        # 搜索结果综合
        "根据以下联网搜索结果": "根据搜索结果，2025年互联网行业软件工程师平均薪资约为25-35万/年。",
        # 内外部对比
        "将行业外部数据与公司内部数据进行对比分析": "## 内外部数据对比分析\n\n公司数据与行业数据对比结论。",
        # 结果汇总
        "为以下问题提供一个完整、清晰的回答": "根据查询结果，研发部共有15名员工。",
        # 对话历史压缩
        "总结以下对话历史": "用户询问了研发部的人员和薪资情况。",
        # 记忆提取 - 偏好
        "提取用户的偏好信息": json.dumps({
            "favorite_department": "研发部",
            "query_focus": "薪资分析"
        }),
        # 记忆提取 - 知识
        "提取值得记住的用户知识点": json.dumps([{
            "category": "常问问题",
            "content": "经常询问研发部的薪资情况",
            "confidence": 0.9
        }]),
        # 意图纠错
        "你之前对一个分类任务返回了无效的答案": "sql_only",
    })


# ======================== MasterAgent fixture ========================

@pytest.fixture
def master_agent(test_db_path, memory_db_path, fake_llm):
    """创建用于测试的 MasterAgent 实例（使用 FakeLLM）"""
    from agents.master_agent import MasterAgent

    agent = MasterAgent(
        llm=fake_llm,
        db_path=test_db_path,
        num_examples=3,
        memory_db_path=memory_db_path,
        short_term_max_tokens=1000,
        tavily_api_key="",
    )
    return agent


# ======================== Flask 客户端 fixture ========================

@pytest.fixture
def app_client():
    """Flask 测试客户端"""
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ======================== SQLite 连接 fixture ========================

@pytest.fixture
def test_db_conn(test_db_path):
    """测试业务数据库连接"""
    conn = sqlite3.connect(test_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def memory_db_conn(memory_db_path):
    """测试记忆数据库连接"""
    conn = sqlite3.connect(memory_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
