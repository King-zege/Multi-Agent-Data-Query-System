"""MasterAgent 意图识别测试

测试意图路由：精确匹配 + 纠错 + 降级。
"""

import pytest

from agents.master_agent import MasterAgent, MasterAgentState


class TestIntentRecognition:
    """意图识别单元测试"""

    def _make_state(self, question: str, user_id: str = "test_user"):
        return {
            "messages": [],
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {"thread_id": "test_thread", "user_id": user_id},
        }

    def test_simple_greeting_is_simple_answer(self, master_agent):
        """测试：问候语识别为 simple_answer"""
        state = self._make_state("你好")
        result = master_agent._intent_node(state)
        assert result["intent"] == "simple_answer"

    def test_sql_query_is_sql_only(self, master_agent):
        """测试：数据库查询识别为 sql_only"""
        state = self._make_state("研发部有多少名员工？")
        result = master_agent._intent_node(state)
        assert result["intent"] == "sql_only"

    def test_analysis_only_intent(self, master_agent):
        """测试：分析已有数据识别为 analysis_only"""
        state = self._make_state("帮我分析一下刚才的结果")
        result = master_agent._intent_node(state)
        assert result["intent"] == "analysis_only"

    def test_query_and_analyze(self, master_agent):
        """测试：查询+分析识别为 sql_and_analysis"""
        state = self._make_state("分析我们公司各部门的薪资水平")
        result = master_agent._intent_node(state)
        assert result["intent"] == "sql_and_analysis"

    def test_web_search_intent(self, master_agent):
        """测试：联网搜索识别为 web_search"""
        state = self._make_state("2024年互联网行业软件工程师平均薪资是多少？")
        result = master_agent._intent_node(state)
        assert result["intent"] == "web_search"

    def test_search_and_compare_intent(self, master_agent):
        """测试：搜索+对比识别为 search_and_sql"""
        state = self._make_state("我们公司研发部薪资和行业平均水平相比怎么样？")
        result = master_agent._intent_node(state)
        assert result["intent"] == "search_and_sql"

    def test_exact_match_no_substring_bug(self, master_agent, fake_llm):
        """回归测试：'sql_and_analysis' 不会被误匹配为 'analysis_only'"""
        # 先清除预设响应，设为一个精确值（用提问关键词匹配）
        fake_llm.responses.clear()
        fake_llm.responses["分析公司各部门薪资"] = "sql_and_analysis"

        state = self._make_state("分析公司各部门薪资")
        result = master_agent._intent_node(state)

        # 必须精确匹配 sql_and_analysis，不能是 analysis_only
        assert result["intent"] == "sql_and_analysis"
        assert result["intent"] != "analysis_only"

    def test_invalid_intent_triggers_correction(self, master_agent, fake_llm):
        """测试：无效意图触发纠错prompt"""
        fake_llm.responses.clear()
        # 纠错标记关键词必须排在前面，因为在纠错 prompt 中采用全文匹配
        fake_llm.responses["你之前对一个分类任务返回了无效的答案"] = "sql_only"
        fake_llm.responses["研发部有多少人"] = "我不知道该怎么分类"

        state = self._make_state("研发部有多少人？")
        result = master_agent._intent_node(state)

        assert result["intent"] == "sql_only"
        # 至少调用了2次（第一次 + 至少1次纠错）
        assert fake_llm.call_count >= 2

    def test_correction_exhausted_falls_back(self, master_agent, fake_llm):
        """测试：纠错耗尽后降级为 simple_answer"""
        fake_llm.responses.clear()
        # 纠错标记关键词排在前面，确保纠错 prompt 中优先匹配
        fake_llm.responses["你之前对一个分类任务返回了无效的答案"] = "还是不知道"
        fake_llm.responses["模糊的问题"] = "不知道"

        state = self._make_state("模糊的问题")
        result = master_agent._intent_node(state)

        # 兜底降级
        assert result["intent"] == "simple_answer"

    def test_intent_with_extraneous_text(self, master_agent, fake_llm):
        """测试：LLM返回带多余文字时通过纠错机制处理"""
        fake_llm.responses.clear()
        # 纠错标记关键词排在前面，确保纠错 prompt 中优先匹配
        fake_llm.responses["你之前对一个分类任务返回了无效的答案"] = "sql_only"
        fake_llm.responses["研发部有多少人"] = "我认为这应该是sql_only查询"

        state = self._make_state("研发部有多少人？")
        result = master_agent._intent_node(state)

        # 首次返回无效，纠错后正确
        assert result["intent"] == "sql_only"


class TestIntentRouting:
    """路由逻辑测试"""

    def test_route_simple_answer(self, master_agent):
        state = {"intent": "simple_answer", "final_answer": None}
        route = master_agent._route_after_intent(state)
        assert route == "simple_answer"

    def test_route_sql_only(self, master_agent):
        state = {"intent": "sql_only", "final_answer": None}
        route = master_agent._route_after_intent(state)
        assert route == "sql_only"

    def test_route_web_search_degraded_when_unavailable(self, master_agent):
        """测试：搜索不可用时，web_search 降级为 simple_answer"""
        master_agent.search_agent.available = False
        state = {"intent": "web_search", "final_answer": None}
        route = master_agent._route_after_intent(state)
        assert route == "simple_answer"
        # 应该预设了降级消息
        assert "未启用" in state.get("final_answer", "")

    def test_route_search_and_sql_degraded_when_unavailable(self, master_agent):
        """测试：搜索不可用时，search_and_sql 降级为 simple_answer"""
        master_agent.search_agent.available = False
        state = {"intent": "search_and_sql", "final_answer": None}
        route = master_agent._route_after_intent(state)
        assert route == "simple_answer"


class TestSimpleAnswer:
    """简单回答节点测试"""

    def _make_state(self, question: str):
        return {
            "messages": [],
            "user_question": question,
            "intent": "simple_answer",
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {"thread_id": "test", "user_id": "test_user"},
        }

    def test_greeting_gets_response(self, master_agent):
        """测试：问候语获得非空回答"""
        state = self._make_state("你好")
        result = master_agent._simple_answer_node(state)
        assert result["final_answer"]
        assert len(result["final_answer"]) > 5

    def test_thanks_gets_response(self, master_agent):
        """测试：感谢获得回复"""
        state = self._make_state("谢谢")
        result = master_agent._simple_answer_node(state)
        assert result["final_answer"]

    def test_unknown_gets_default_response(self, master_agent):
        """测试：未知内容获得默认回复"""
        state = self._make_state("xyzxyz")
        result = master_agent._simple_answer_node(state)
        assert result["final_answer"]
