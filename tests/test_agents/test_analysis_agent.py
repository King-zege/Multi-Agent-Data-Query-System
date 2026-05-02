"""DataAnalysisAgent 测试

测试数据分析、图表生成判断逻辑。
"""

import json
from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_utils.fake_llm import FakeLLM


@pytest.fixture
def analysis_agent(fake_llm):
    """创建 DataAnalysisAgent 测试实例"""
    from agents.analysis_agent import DataAnalysisAgent

    return DataAnalysisAgent(llm=fake_llm)


class TestDataParsing:
    """数据解析测试"""

    def test_parse_valid_json(self, analysis_agent):
        """测试：解析合法 JSON"""
        data = analysis_agent._parse_data('[{"name": "张三", "salary": 10000}]')
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "张三"

    def test_parse_invalid_json_returns_none(self, analysis_agent):
        """测试：解析非法 JSON 返回 None"""
        data = analysis_agent._parse_data("not json")
        assert data is None

    def test_parse_empty_string_returns_none(self, analysis_agent):
        """测试：解析空字符串返回 None"""
        data = analysis_agent._parse_data("")
        assert data is None


class TestDataSummary:
    """数据摘要测试"""

    def test_summary_includes_count(self, analysis_agent):
        """测试：摘要包含记录数"""
        data = [{"name": "A", "salary": 100}, {"name": "B", "salary": 200}]
        summary = analysis_agent._prepare_data_summary(data)
        assert "2" in summary

    def test_summary_empty_data(self, analysis_agent):
        """测试：空数据摘要"""
        summary = analysis_agent._prepare_data_summary([])
        assert "空" in summary

    def test_summary_single_record(self, analysis_agent):
        """测试：单条记录摘要"""
        summary = analysis_agent._prepare_data_summary({"name": "A"})
        assert "单条记录" in summary

    def test_summary_numeric_stats(self, analysis_agent):
        """测试：数值字段统计"""
        data = [
            {"name": "A", "salary": 100},
            {"name": "B", "salary": 200},
            {"name": "C", "salary": 300},
        ]
        summary = analysis_agent._prepare_data_summary(data)
        assert "salary" in summary
        assert "最小" in summary
        assert "最大" in summary
        assert "平均" in summary


class TestChartDecision:
    """图表生成判断测试"""

    def test_should_generate_for_list_of_dicts_with_numeric(self, analysis_agent):
        """测试：包含数值字段的列表数据应生成图表"""
        data = [{"name": "A", "value": 10}, {"name": "B", "value": 20}]
        assert analysis_agent._should_generate_chart(data)

    def test_should_not_generate_for_single_record(self, analysis_agent):
        """测试：单条记录不应生成图表"""
        data = [{"name": "A", "value": 10}]
        assert not analysis_agent._should_generate_chart(data)

    def test_should_not_generate_for_non_dict_items(self, analysis_agent):
        """测试：非字典列表不应生成图表"""
        data = ["a", "b", "c"]
        assert not analysis_agent._should_generate_chart(data)

    def test_should_not_generate_without_numeric_fields(self, analysis_agent):
        """测试：无数值字段不应生成图表"""
        data = [{"name": "A", "desc": "x"}, {"name": "B", "desc": "y"}]
        assert not analysis_agent._should_generate_chart(data)


class TestAnalysis:
    """完整分析流程测试"""

    def test_analyze_returns_required_fields(self, analysis_agent):
        """测试：分析结果包含必要字段"""
        data = json.dumps([
            {"name": "研发部", "avg_salary": 15000},
            {"name": "市场部", "avg_salary": 12000},
        ])
        result = analysis_agent.analyze(data, "比较各部门薪资")
        assert "analysis" in result
        assert "chart" in result
        assert "error" in result
        assert result["error"] is None

    def test_analyze_with_invalid_data(self, analysis_agent):
        """测试：非法数据返回 error"""
        result = analysis_agent.analyze("not json", "test")
        assert result["error"] is not None

    def test_analyze_with_error_in_data(self, analysis_agent):
        """测试：数据中含 error 字段时透传"""
        data = json.dumps({"error": "查询失败"})
        result = analysis_agent.analyze(data, "test")
        assert result["error"] is not None
        assert "查询失败" in result["error"]
