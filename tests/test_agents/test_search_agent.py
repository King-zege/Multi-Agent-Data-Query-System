"""WebSearchAgent 测试

测试搜索格式化、不可用时的降级处理。
"""

import json
import os
from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_utils.fake_llm import FakeLLM


@pytest.fixture
def search_agent_unavailable(fake_llm, monkeypatch):
    """创建不可用的搜索智能体（无 API Key）"""
    # 清空环境变量，确保不会从系统环境中读取到 TAVILY_API_KEY
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    from agents.search_agent import WebSearchAgent

    return WebSearchAgent(llm=fake_llm, tavily_api_key="")


class TestFormatResult:
    """搜索结果格式化测试"""

    def test_format_empty_results(self, search_agent_unavailable):
        """测试：空结果列表"""
        text = search_agent_unavailable._format_search_results([])
        assert "未找到" in text

    def test_format_results_with_content(self, search_agent_unavailable):
        """测试：正常结果格式化"""
        results = [
            {
                "title": "测试标题",
                "content": "这是测试内容",
                "url": "https://example.com",
            }
        ]
        text = search_agent_unavailable._format_search_results(results)
        assert "测试标题" in text
        assert "example.com" in text

    def test_format_results_long_content_truncated(self, search_agent_unavailable):
        """测试：超长内容被截断"""
        results = [
            {
                "title": "Test",
                "content": "A" * 1000,
                "url": "https://example.com",
            }
        ]
        text = search_agent_unavailable._format_search_results(results)
        assert len(text) < 1200  # 内容应被截断


class TestSearchUnavailable:
    """不可用状态测试"""

    def test_available_is_false_without_api_key(self, search_agent_unavailable):
        """测试：无 API Key 时 available=False"""
        assert not search_agent_unavailable.available

    def test_search_returns_error_when_unavailable(self, search_agent_unavailable):
        """测试：不可用时 search 返回 error"""
        result = search_agent_unavailable.search("测试搜索")
        assert result["error"] is not None
        assert "未启用" in result["error"]
        assert result["answer"] is None

    def test_search_and_compare_returns_error_when_unavailable(
        self, search_agent_unavailable
    ):
        """测试：不可用时 search_and_compare 返回 error"""
        result = search_agent_unavailable.search_and_compare(
            "测试", '{"data": []}'
        )
        assert result["error"] is not None
