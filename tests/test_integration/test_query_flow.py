"""集成测试 — 端到端查询流程

使用 FakeLLM 模拟完整查询链路。
"""

import json
import uuid
from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def system(test_db_path, memory_db_path, fake_llm):
    """创建完整的 MultiAgentSystem 测试实例"""
    import agent

    # 直接构建 MasterAgent（绕过 config.yaml）
    from agents.master_agent import MasterAgent

    master = MasterAgent(
        llm=fake_llm,
        db_path=test_db_path,
        num_examples=3,
        memory_db_path=memory_db_path,
        short_term_max_tokens=1000,
        tavily_api_key="",
    )

    # 创建一个简化的系统包装
    class _TestSystem:
        def __init__(self):
            self.master_agent = master
            self.user_id = None
            self.session_id = None

        def login(self, user_id):
            self.user_id = user_id
            self.session_id = str(uuid.uuid4())
            self.master_agent.long_term_memory.update_user_activity(user_id)
            return True

        def query(self, question):
            if not self.user_id:
                return "请先登录"
            thread_id = f"{self.user_id}_{self.session_id}"
            return self.master_agent.query(
                question, thread_id=thread_id, user_id=self.user_id
            )

    return _TestSystem()


class TestFullQueryFlow:
    """端到端查询流程测试"""

    def test_sql_only_query(self, system):
        """测试：完整 sql_only 流程"""
        system.login("test_user")
        answer = system.query("研发部有多少人？")

        assert answer is not None
        assert len(answer) > 0

    def test_multiple_queries_in_session(self, system):
        """测试：同一会话内多次查询"""
        system.login("test_user")

        answer1 = system.query("查询员工信息")
        assert answer1 is not None

        answer2 = system.query("分析刚才的数据")
        assert answer2 is not None

    def test_query_with_long_term_memory(self, system, memory_db_path):
        """测试：带长期记忆的查询"""
        # 预设用户偏好
        from memory.long_term_memory import LongTermMemory

        ltm = LongTermMemory(memory_db_path)
        ltm.save_preference("mem_user", "favorite_department", "研发部")
        ltm.save_knowledge("mem_user", "业务领域", "关注技术部门薪资", 0.9)

        system.login("mem_user")
        answer = system.query("我们公司薪资情况如何？")

        assert answer is not None


class TestSimpleAnswerFlow:
    """简单回答流程测试"""

    def test_greeting(self, system):
        """测试：问候直接返回"""
        system.login("test_user")
        answer = system.query("你好")
        assert len(answer) > 0

    def test_no_user_login(self, system):
        """测试：未登录时提示登录"""
        answer = system.query("研发部有多少人")
        assert "登录" in answer
