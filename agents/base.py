"""
子智能体基类 + 注册中心

提供 BaseSubAgent 抽象类和 AgentRegistry，实现插件化架构：
- 新增子智能体只需：继承 BaseSubAgent → AgentRegistry.register() → 自动路由
- MasterAgent 不再直接依赖具体子智能体类
"""

from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional, Any


# 前向引用，避免循环导入
# MasterAgentState 在 master_agent.py 中定义为 TypedDict


class BaseSubAgent(ABC):
    """子智能体抽象基类

    每个子智能体对应一种或多种意图处理能力。
    子类必须实现 intent_name、node_name 和 execute()。

    属性：
        intent_name: 该智能体处理的意图名称（如 "sql_only"）
        node_name:   LangGraph 中的节点名称（如 "call_sql"）
    """

    intent_name: str = ""
    node_name: str = ""

    def __init__(self, llm=None):
        self.llm = llm

    @property
    def available(self) -> bool:
        """该智能体是否可用（默认可用，子类可 override）"""
        return True

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行该智能体的核心逻辑

        Args:
            state: 当前 MasterAgentState

        Returns:
            更新后的 MasterAgentState
        """
        ...

    def stream_events(self, state: Dict[str, Any]) -> Generator[str, None, None]:
        """流式事件生成器（默认实现：调用 execute() 后 yield chunk）

        子类可 override 以推送自定义 SSE 事件（如 sources、chart 等）。

        Args:
            state: 当前 MasterAgentState

        Yields:
            SSE 格式字符串: "data: {...}\\n\\n"
        """
        import json
        result_state = self.execute(state)
        answer = result_state.get("final_answer", "")
        if answer:
            yield f"data: {json.dumps({'type': 'chunk', 'content': answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=False)}\n\n"

    @staticmethod
    def _state_to_json(obj: Any) -> str:
        """安全地将对象转为 JSON 字符串"""
        import json
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)


class AgentRegistry:
    """子智能体注册中心

    使用示例::

        registry = AgentRegistry()
        registry.register(SQLQueryAgent(llm, db_path))
        registry.register(DataAnalysisAgent(llm))

        agent = registry.get("sql_only")  # 按意图名获取
        print(registry.get_available_intents())  # 所有可用意图
    """

    def __init__(self):
        self._agents: Dict[str, BaseSubAgent] = {}  # intent_name → agent 实例

    def register(self, agent: BaseSubAgent):
        """注册一个子智能体

        Args:
            agent: BaseSubAgent 实例

        Raises:
            ValueError: 如果 intent_name 已存在或为空
        """
        if not agent.intent_name:
            raise ValueError(f"智能体 {agent.__class__.__name__} 的 intent_name 不能为空")
        if agent.intent_name in self._agents:
            raise ValueError(f"意图 '{agent.intent_name}' 已被 {self._agents[agent.intent_name].__class__.__name__} 注册")
        self._agents[agent.intent_name] = agent

    def get(self, intent_name: str) -> Optional[BaseSubAgent]:
        """按意图名获取智能体

        Args:
            intent_name: 意图名称（如 "sql_only"）

        Returns:
            BaseSubAgent 实例，或 None（未找到时）
        """
        return self._agents.get(intent_name)

    def get_available(self) -> Dict[str, BaseSubAgent]:
        """获取所有可用的智能体"""
        return {name: agent for name, agent in self._agents.items() if agent.available}

    def get_all_agents(self) -> Dict[str, BaseSubAgent]:
        """获取所有注册的智能体（包含不可用的）"""
        return dict(self._agents)

    def get_available_intents(self) -> List[str]:
        """获取所有可用意图名称列表"""
        return list(self.get_available().keys())

    def get_all_intents(self) -> List[str]:
        """获取所有意图名称列表（包含不可用的）"""
        return list(self._agents.keys())

    def __contains__(self, intent_name: str) -> bool:
        return intent_name in self._agents

    def __len__(self) -> int:
        return len(self._agents)
