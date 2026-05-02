"""MemoryExtractor 测试

测试从对话中自动提取偏好和知识。
"""

import json
from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_utils.fake_llm import FakeLLM
from langchain.messages import HumanMessage, AIMessage


@pytest.fixture
def memory_extractor(fake_llm):
    """创建 MemoryExtractor 测试实例"""
    from memory.memory_extractor import MemoryExtractor

    return MemoryExtractor(llm=fake_llm)


class TestShouldExtract:
    """提取时机判断测试"""

    def test_should_not_extract_with_few_messages(self, memory_extractor):
        """测试：消息数不足时不提取"""
        msgs = [HumanMessage(content="你好"), AIMessage(content="你好")]
        assert not memory_extractor.should_extract(msgs, threshold=6)

    def test_should_extract_with_enough_messages(self, memory_extractor):
        """测试：消息数达标时触发提取"""
        msgs = [HumanMessage(content=f"msg{i}") for i in range(6)]
        assert memory_extractor.should_extract(msgs, threshold=6)

    def test_default_threshold_is_6(self, memory_extractor):
        """测试：默认阈值为6"""
        assert memory_extractor.should_extract([]) is False
        msgs = [HumanMessage(content=f"msg{i}") for i in range(7)]
        assert memory_extractor.should_extract(msgs)


class TestExtractPreferences:
    """偏好提取测试"""

    def test_extract_preferences_returns_dict(self, memory_extractor):
        """测试：返回值为字典"""
        msgs = [
            HumanMessage(content="研发部有多少人"),
            AIMessage(content="研发部共有15名员工"),
            HumanMessage(content="他们的薪资如何"),
            AIMessage(content="研发部平均薪资为20000元"),
        ]
        prefs = memory_extractor.extract_preferences_from_conversation(
            msgs, "test_user"
        )
        assert isinstance(prefs, dict)

    def test_extract_preferences_with_few_messages_returns_empty(
        self, memory_extractor
    ):
        """测试：消息数不足时返回空"""
        msgs = [HumanMessage(content="你好")]
        prefs = memory_extractor.extract_preferences_from_conversation(
            msgs, "test_user"
        )
        assert prefs == {}


class TestExtractKnowledge:
    """知识提取测试"""

    def test_extract_knowledge_returns_list(self, memory_extractor):
        """测试：返回值为列表"""
        msgs = [
            HumanMessage(content="研发部有多少人"),
            AIMessage(content="研发部共有15名员工"),
            HumanMessage(content="他们的薪资如何"),
            AIMessage(content="研发部平均薪资为20000元"),
        ]
        knowledge = memory_extractor.extract_knowledge_from_conversation(
            msgs, "test_user"
        )
        assert isinstance(knowledge, list)

    def test_extract_knowledge_with_few_messages_returns_empty(
        self, memory_extractor
    ):
        """测试：消息数不足时返回空"""
        msgs = [HumanMessage(content="你好")]
        knowledge = memory_extractor.extract_knowledge_from_conversation(
            msgs, "test_user"
        )
        assert knowledge == []


class TestFormatConversation:
    """对话格式化测试"""

    def test_format_conversation(self, memory_extractor):
        """测试：对话格式化包含用户和助手标签"""
        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮你？"),
        ]
        text = memory_extractor._format_conversation(msgs)
        assert "用户:" in text
        assert "助手:" in text
