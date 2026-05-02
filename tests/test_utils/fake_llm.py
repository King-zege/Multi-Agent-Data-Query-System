"""
FakeLLM — 模拟语言模型，用于测试中不消耗 API 调用

根据 prompt 关键词匹配返回预设响应，支持 invoke() 和 stream() 两种调用方式。
所有调用记录在 self.calls 中，方便测试断言。
"""

import re
from typing import Dict, Optional, List, Tuple, Any


class FakeLLM:
    """模拟语言模型，按关键词匹配返回预设内容

    对于意图识别 prompt（包含 "当前问题：" 模式），只针对用户问题部分进行匹配，
    防止 prompt 模板中的意图描述示例污染关键词匹配。
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Args:
            responses: {keyword: response_text} 映射。
                       当 prompt 中包含 keyword 时返回对应 response_text。
        """
        self.responses = responses or {}
        self.calls: List[Tuple[str, str]] = []  # (prompt_preview, response)

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """从 prompt 中提取用户实际提问，避免与模板示例混淆"""
        match = re.search(r'当前问题：(.+?)(?:\n\n|\n请判断)', prompt)
        if match:
            return match.group(1).strip()
        return ""

    def _match(self, prompt: str) -> str:
        """按关键词匹配响应

        对意图识别 prompt（包含 "当前问题："），只针对用户提问进行匹配，
        防止 prompt 模板中的意图描述示例污染关键词匹配。
        对纠错等其他 prompt（含 "用户问题：" 但无 "当前问题："），
        使用全文匹配，以便纠错标记关键词能正确命中。
        """
        # 仅对意图识别 prompt 使用提问精确匹配
        if "当前问题：" in prompt:
            question = self._extract_question(prompt)
            if question:
                for keyword, response in self.responses.items():
                    if keyword in question:
                        return response
                return "simple_answer"

        # 纠错/SQL/分析/图表等 prompt：全文匹配
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return response
        return "simple_answer"  # 默认响应

    def invoke(self, prompt: str) -> Any:
        """模拟 LLM invoke，返回类似 AIMessage 的对象"""
        text = self._match(prompt)
        self.calls.append((self._truncate(prompt), text))

        class _FakeMessage:
            def __init__(self, content):
                self.content = content

        return _FakeMessage(text)

    def stream(self, prompt: str):
        """模拟 LLM stream，逐字 yield"""
        text = self._match(prompt)
        self.calls.append((self._truncate(prompt), text))

        class _FakeChunk:
            def __init__(self, content):
                self.content = content

        for char in text:
            yield _FakeChunk(char)

    @staticmethod
    def _truncate(text: str, max_len: int = 100) -> str:
        return text[:max_len] + "..." if len(text) > max_len else text

    def assert_called_with(self, keyword: str):
        """断言 LLM 被以包含某关键词的 prompt 调用过"""
        for prompt_snippet, _ in self.calls:
            if keyword in prompt_snippet:
                return
        raise AssertionError(
            f"Expected LLM call with keyword '{keyword}', "
            f"but got {len(self.calls)} calls: {self.calls}"
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_response(self) -> str:
        return self.calls[-1][1] if self.calls else ""
