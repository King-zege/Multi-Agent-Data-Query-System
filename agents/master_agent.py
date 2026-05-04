"""
主智能体

负责意图识别、任务路由、协调子智能体和结果汇总。
支持6种意图：simple_answer / sql_only / analysis_only / sql_and_analysis / web_search / search_and_sql
"""

import json
from typing import TypedDict, Sequence, Dict, Any, Optional, Annotated, Generator
from pathlib import Path

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLLM

import sys
sys.path.append(str(Path(__file__).parent.parent))
from prompts import get_master_intent_prompt, get_summary_prompt, get_intent_correction_prompt, get_cache_match_prompt
from agents.base import AgentRegistry
from agents.sql_agent import SQLQueryAgent
from agents.analysis_agent import DataAnalysisAgent
from agents.search_agent import WebSearchAgent
from memory.long_term_memory import LongTermMemory
from memory.memory_extractor import MemoryExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

# ---- 意图路由配置 ----

# 简单意图：直接映射到子智能体的 intent_name
SIMPLE_INTENT_MAP = {
    "simple_answer": "simple_answer",
    "sql_only": "sql_only",
    "analysis_only": "analysis_only",
    "web_search": "web_search",
}

# 复合意图：步骤定义（每步是一个 intent_name，串行执行）
COMPOSITE_PIPELINES = {
    "sql_and_analysis": {
        "steps": ["sql_only", "analysis_only"],
        "summarize_with": "analysis",  # 用分析结果作为汇总重点
    },
    "search_and_sql": {
        "steps": ["sql_only", "web_search"],
        "summarize_with": "search",  # 用搜索对比结果作为汇总重点
    },
}


class MasterAgentState(TypedDict):
    """主智能体状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question: str
    intent: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    search_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]


class MasterAgent:
    """主智能体 - 协调SQL查询和数据分析子智能体"""
    
    @staticmethod
    def _llm_to_str(result) -> str:
        """安全地从 LLM 返回值中提取文本字符串

        兼容 str / AIMessage / GenerationChunk 等多种返回类型。
        处理 Qwen 模型的 <think>...</think> 标签：
        - 优先提取 </think> 之后的内容（正式回答部分）
        - 防止将回答内容误删（Qwen 可能把数据列表也放入 think 块）
        """
        import re
        if isinstance(result, str):
            text = result
        elif hasattr(result, 'content'):
            text = str(result.content)
        elif hasattr(result, 'text'):
            text = str(result.text)
        else:
            text = str(result)
        # 优先提取 </think> 之后的内容（Qwen 模型的正式回答部分）
        think_end = text.rfind('</think>')
        if think_end != -1:
            text = text[think_end + len('</think>'):].strip()
        else:
            # 没有 </think> 时，移除 <think>...</think> 块（如果有）
            text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        return text
    
    def __init__(self, llm: BaseLLM, db_config: Dict[str, Any], num_examples: int = 3,
                memory_db_path: str = "./data/long_term_memory.db",
                short_term_max_tokens: int = 1000,
                tavily_api_key: str = ""):
        """初始化主智能体

        Args:
            llm: 语言模型实例
            db_config: 数据库配置字典，支持 SQLite/MySQL/PostgreSQL
            num_examples: Few-shot示例数量
            memory_db_path: 长期记忆数据库路径
            short_term_max_tokens: 短期记忆最大token数
            tavily_api_key: Tavily 搜索 API Key
        """
        self.llm = llm
        self.db_config = db_config
        self.short_term_max_tokens = short_term_max_tokens

        # 初始化子智能体
        self.sql_agent = SQLQueryAgent(llm, db_config, num_examples)
        self.analysis_agent = DataAnalysisAgent(llm)
        self.search_agent = WebSearchAgent(llm, tavily_api_key=tavily_api_key)

        # 注册中心：插件化管理子智能体
        self.registry = AgentRegistry()
        self.registry.register(self.sql_agent)
        self.registry.register(self.analysis_agent)
        self.registry.register(self.search_agent)

        # 初始化短期记忆（MemorySaver）
        self.memory = MemorySaver()

        # 初始化长期记忆（LongTermMemory）
        self.long_term_memory = LongTermMemory(memory_db_path)

        # 初始化记忆提取器
        self.memory_extractor = MemoryExtractor(llm)

        # 会话数据存储：保存每个thread_id的最近查询结果
        self.session_data = {}

        # 构建工作流
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图（插件化：从 AgentRegistry 动态注册节点）"""
        workflow = StateGraph(MasterAgentState)

        # 核心节点
        workflow.add_node("intent", self._intent_node)
        workflow.add_node("simple_answer", self._simple_answer_node)

        # 子智能体节点（通过包装方法管理会话数据）
        workflow.add_node("call_sql", self._dispatch_sql)
        workflow.add_node("call_analysis", self._dispatch_analysis)
        workflow.add_node("call_web_search", self._dispatch_web_search)

        # 复合意图节点（管线编排）
        workflow.add_node("pipeline_sql_analysis", self._pipeline_sql_analysis)
        workflow.add_node("pipeline_search_sql", self._pipeline_search_sql)

        workflow.add_node("summarize", self._summarize_node)
        workflow.set_entry_point("intent")

        # 路由映射
        route_map = {
            "simple_answer": "simple_answer",
            "sql_only": "call_sql",
            "analysis_only": "call_analysis",
            "web_search": "call_web_search",
            "sql_and_analysis": "pipeline_sql_analysis",
            "search_and_sql": "pipeline_search_sql",
        }

        workflow.add_conditional_edges("intent", self._route_after_intent, route_map)

        # 所有处理节点都连接到汇总节点
        for node_name in ("call_sql", "call_analysis", "call_web_search",
                          "pipeline_sql_analysis", "pipeline_search_sql"):
            workflow.add_edge(node_name, "summarize")
        workflow.add_edge("simple_answer", END)
        workflow.add_edge("summarize", END)

        return workflow.compile(checkpointer=self.memory)
    
    def _get_conversation_history(self, state: MasterAgentState) -> str:
        """获取对话历史摘要（智能压缩版本）
        
        策略：
        1. 如果消息少于等于10条，直接返回所有
        2. 如果消息较多但token未超限，返回近期消息
        3. 如果消息很多且超过token限制，使用LLM总结压缩
        
        Args:
            state: 当前状态
            
        Returns:
            对话历史摘要
        """
        messages = state.get("messages", [])
        if len(messages) <= 1:
            return ""
        
        # 构建原始历史（排除当前消息）
        history_text = self._format_messages(messages[:-1])
        
        # 如果消息数量少，直接返回
        if len(messages) <= 11:  # 10条历史消息
            return history_text
        
        # 简单token估算（中文按2字符=1token，英文按4字符=1token）
        estimated_tokens = len(history_text) / 2.5
        
        if estimated_tokens <= self.short_term_max_tokens:
            return history_text
        
        # 需要压缩：使用LLM总结
        return self._compress_history_with_llm(history_text)
    
    def _format_messages(self, messages: Sequence[BaseMessage]) -> str:
        """格式化消息列表为文本
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的文本
        """
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"助手: {msg.content}")
        
        return "\n".join(history) if history else ""
    
    def _compress_history_with_llm(self, history_text: str) -> str:
        """使用LLM总结压缩对话历史
        
        Args:
            history_text: 原始对话历史文本
            
        Returns:
            压缩后的摘要文本
        """
        prompt = f"""请总结以下对话历史，保留关键信息、用户偏好和重要上下文：

{history_text}

总结要求：
1. 保留关键事实和数据（如查询的部门、员工、数据结果）
2. 提取用户关注点和偏好
3. 保留重要的上下文信息
4. 简洁但信息完整
5. 不超过300字

总结："""
        
        try:
            summary = self._llm_to_str(self.llm.invoke(prompt)).strip()
            return f"[对话历史总结]\n{summary}"
        except Exception as e:
            logger.warning("压缩对话历史失败: %s", e)
            # 如果压缩失败，返回最近的部分对话
            lines = history_text.split("\n")
            recent_lines = lines[-20:] if len(lines) > 20 else lines
            return "\n".join(recent_lines)
    
    def _format_long_term_context(
        self, 
        knowledge: list, 
        preferences: Dict[str, str]
    ) -> str:
        """格式化长期记忆上下文
        
        Args:
            knowledge: 用户知识列表
            preferences: 用户偏好字典
            
        Returns:
            格式化的上下文文本
        """
        context_parts = []
        
        # 添加用户偏好
        if preferences:
            pref_lines = [f"- {key}: {value}" for key, value in preferences.items()]
            context_parts.append("用户偏好：\n" + "\n".join(pref_lines))
        
        # 添加相关知识
        if knowledge:
            know_lines = [f"- {item['content']}" for item in knowledge[:3]]
            context_parts.append("相关背景：\n" + "\n".join(know_lines))
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _intent_node(self, state: MasterAgentState) -> MasterAgentState:
        """意图识别节点（支持6种意图）"""
        question = state["user_question"]
        user_id = state["metadata"].get("user_id")
        
        # 获取对话历史（短期记忆）
        conversation_history = self._get_conversation_history(state)
        
        # 获取用户知识（长期记忆）
        user_context = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=3)
                preferences = self.long_term_memory.get_all_preferences(user_id)
                user_context = self._format_long_term_context(knowledge, preferences)
            except Exception as e:
                logger.warning("获取长期记忆失败: %s", e)
        
        prompt = get_master_intent_prompt(question, conversation_history, user_context)

        VALID_INTENTS = (
            "simple_answer", "sql_only", "analysis_only",
            "sql_and_analysis", "web_search", "search_and_sql"
        )

        try:
            response = self._llm_to_str(self.llm.invoke(prompt)).strip().lower()
            intent = response

            # 精确匹配
            if intent not in VALID_INTENTS:
                # 第一次纠错：让 LLM 重新输出
                correction_prompt = get_intent_correction_prompt(question, response)
                response2 = self._llm_to_str(self.llm.invoke(correction_prompt)).strip().lower()
                intent = response2

                # 第二次纠错
                if intent not in VALID_INTENTS:
                    correction_prompt2 = get_intent_correction_prompt(question, response2)
                    response3 = self._llm_to_str(self.llm.invoke(correction_prompt2)).strip().lower()
                    intent = response3

                    # 仍无效，安全降级
                    if intent not in VALID_INTENTS:
                        intent = "simple_answer"

            state["intent"] = intent
            state["metadata"]["intent_response"] = response

        except Exception as e:
            state["error"] = f"意图识别失败: {str(e)}"
            state["intent"] = "simple_answer"

        return state
    
    def _route_after_intent(self, state: MasterAgentState) -> str:
        """意图识别后的路由（支持6种意图，搜索不可用时降级）"""
        intent = state.get("intent", "simple_answer")
        # 如果涉及搜索但搜索智能体不可用，降级为 simple_answer
        if intent in ("web_search", "search_and_sql"):
            search_agent = self.registry.get("web_search")
            if search_agent and not search_agent.available:
                logger.info("搜索智能体不可用，意图降级为 simple_answer")
                state["final_answer"] = (
                    "联网搜索功能暂未启用。请配置 TAVILY_API_KEY 环境变量后重启系统。\n"
                    "联网搜索功能未启用，请配置Tavily API Key"
                )
                return "simple_answer"
        return intent
    
    def _simple_answer_node(self, state: MasterAgentState) -> MasterAgentState:
        """简单回答节点"""
        question = state["user_question"]
        
        common_responses = {
            "你好": "你好！我是智能数据查询助手，可以帮你查询员工、部门、薪资等信息，还可以进行数据分析。有什么可以帮你的吗？",
            "谢谢": "不客气！还有什么其他问题吗？",
            "再见": "再见！祝你工作顺利！",
            "帮助": "我可以帮你：\n1. 查询数据库信息（如：有多少员工？）\n2. 分析数据（如：分析各部门薪资水平）\n3. 综合查询和分析（如：找出高薪员工并分析特征）",
        }
        
        # 检查常见问候
        answer = None
        for key, response in common_responses.items():
            if key in question:
                answer = response
                break
        
        # 默认回复
        if not answer:
            answer = "我是智能数据查询助手。请问有什么关于员工、部门或薪资的问题需要我帮忙吗？"
        
        state["final_answer"] = answer
        
        # 将AI回答添加到messages中
        state["messages"] = state["messages"] + [AIMessage(content=answer)]
        
        return state
    
    def _call_sql_node(self, state: MasterAgentState) -> MasterAgentState:
        """调用SQL查询子智能体"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        try:
            result = self.sql_agent.query(question, thread_id=thread_id)
            state["sql_result"] = result
            state["metadata"]["sql_result"] = result
            
            # 保存到会话数据存储
            if thread_id not in self.session_data:
                self.session_data[thread_id] = {}
            self.session_data[thread_id]["last_sql_result"] = result
            
        except Exception as e:
            state["error"] = f"SQL查询失败: {str(e)}"
            state["sql_result"] = {"error": str(e)}
        
        return state
    
    def _call_analysis_node(self, state: MasterAgentState) -> MasterAgentState:
        """调用数据分析子智能体"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        # 从会话数据存储中获取最近的查询结果
        data_to_analyze = None
        
        # 首先检查当前state中是否有查询结果
        if state.get("sql_result") and "data" in state["sql_result"]:
            data_to_analyze = state["sql_result"]["data"]
        # 否则从会话数据存储中获取历史查询结果
        elif thread_id in self.session_data and "last_sql_result" in self.session_data[thread_id]:
            last_sql_result = self.session_data[thread_id]["last_sql_result"]
            if last_sql_result and "data" in last_sql_result:
                data_to_analyze = last_sql_result["data"]
        
        if not data_to_analyze:
            state["error"] = "没有找到可以分析的数据。请先进行数据查询。"
            state["analysis_result"] = {"error": "无可用数据"}
            return state
        
        try:
            result = self.analysis_agent.analyze(data_to_analyze, question)
            state["analysis_result"] = result
            state["metadata"]["analysis_result"] = result
        except Exception as e:
            state["error"] = f"数据分析失败: {str(e)}"
            state["analysis_result"] = {"error": str(e)}
        
        return state
    
    def _pipeline_sql_analysis(self, state: MasterAgentState) -> MasterAgentState:
        """复合意图管线：sql_only → analysis_only（串行）"""
        return self._run_pipeline(state, "sql_and_analysis")

    def _pipeline_search_sql(self, state: MasterAgentState) -> MasterAgentState:
        """复合意图管线：sql_only → web_search（串行，搜索智能体自动识别联合分析）"""
        return self._run_pipeline(state, "search_and_sql")

    def _dispatch_sql(self, state: MasterAgentState) -> MasterAgentState:
        """调度 SQL 智能体并管理会话数据（支持缓存命中跳过查询）"""
        thread_id = state["metadata"].get("thread_id", "default")
        question = state["user_question"]

        # 检查缓存（精确命中 → 跳过 SQL 查询）
        cache_match = self._match_cache(question, thread_id)
        if cache_match and cache_match["mode"] == "exact":
            cached = self.sql_agent.get_cached_result(thread_id, cache_match["key"])
            if cached:
                state["sql_result"] = cached
                state["metadata"]["sql_result"] = cached
                if thread_id not in self.session_data:
                    self.session_data[thread_id] = {}
                self.session_data[thread_id]["last_sql_result"] = cached
                return state

        agent = self.registry.get("sql_only")
        if agent:
            state = agent.execute(state)
            if state.get("sql_result") and not state["sql_result"].get("error"):
                if thread_id not in self.session_data:
                    self.session_data[thread_id] = {}
                self.session_data[thread_id]["last_sql_result"] = state["sql_result"]
        return state

    def _dispatch_analysis(self, state: MasterAgentState) -> MasterAgentState:
        """调度分析智能体，自动补充会话数据（含缓存回退）"""
        thread_id = state["metadata"].get("thread_id", "default")
        question = state["user_question"]

        agent = self.registry.get("analysis_only")
        if agent:
            # 如果 state 和 session_data 都没有 sql_result，尝试从缓存获取
            has_data = (
                (state.get("sql_result") and "data" in state.get("sql_result", {}))
                or (thread_id in self.session_data
                    and "last_sql_result" in self.session_data[thread_id]
                    and self.session_data[thread_id]["last_sql_result"]
                    and "data" in self.session_data[thread_id]["last_sql_result"])
            )
            if not has_data:
                cache_match = self._match_cache(question, thread_id)
                if cache_match:
                    cached = self.sql_agent.get_cached_result(thread_id, cache_match["key"])
                    if cached:
                        state["sql_result"] = cached
                        if thread_id not in self.session_data:
                            self.session_data[thread_id] = {}
                        self.session_data[thread_id]["last_sql_result"] = cached

            # 将 session_data 注入 state 供分析智能体读取
            state["metadata"]["_session_data"] = self.session_data
            state = agent.execute(state)
        return state

    def _dispatch_web_search(self, state: MasterAgentState) -> MasterAgentState:
        """调度搜索智能体"""
        agent = self.registry.get("web_search")
        if agent:
            state = agent.execute(state)
        return state

    def _run_pipeline(self, state: MasterAgentState, pipeline_key: str) -> MasterAgentState:
        """执行复合意图管线，自动管理会话数据（支持缓存跳过SQL步骤）"""
        pipeline = COMPOSITE_PIPELINES[pipeline_key]
        thread_id = state["metadata"].get("thread_id", "default")
        question = state["user_question"]

        for step_intent in pipeline["steps"]:
            # SQL步骤：优先检查缓存
            if step_intent == "sql_only":
                cache_match = self._match_cache(question, thread_id)
                if cache_match:
                    cached = self.sql_agent.get_cached_result(thread_id, cache_match["key"])
                    if cached:
                        state["sql_result"] = cached
                        state["metadata"]["sql_result"] = cached
                        if thread_id not in self.session_data:
                            self.session_data[thread_id] = {}
                        self.session_data[thread_id]["last_sql_result"] = cached
                        continue  # 跳过SQL查询，直接进入下一步

            agent = self.registry.get(step_intent)
            if not agent:
                state["error"] = f"未找到意图 {step_intent} 对应的智能体"
                return state
            if not agent.available:
                state["error"] = f"智能体 {step_intent} 当前不可用"
                return state

            try:
                state = agent.execute(state)
                # 同步会话数据到 MasterAgent.session_data
                if state.get("sql_result") and not state["sql_result"].get("error"):
                    if thread_id not in self.session_data:
                        self.session_data[thread_id] = {}
                    self.session_data[thread_id]["last_sql_result"] = state["sql_result"]
            except Exception as e:
                state["error"] = f"管线步骤 {step_intent} 执行失败: {str(e)}"
                return state

        return state
    
    def _summarize_node(self, state: MasterAgentState) -> MasterAgentState:
        """汇总结果节点（支持搜索结果和图表元数据）"""
        question = state["user_question"]
        intent = state.get("intent", "sql_only")
        
        # 预设回答已经生成（如降级处理）
        if state.get("final_answer"):
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        if state.get("error"):
            state["final_answer"] = f"抱歉，处理过程中出现错误：{state['error']}"
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        search_result = state.get("search_result")
        
        # 联网搜索相关意图：搜索智能体已生成完整回答
        if intent in ("web_search", "search_and_sql") and search_result:
            if search_result.get("error"):
                state["final_answer"] = f"搜索出错：{search_result['error']}"
            else:
                answer = search_result.get("answer", "未能获取搜索结果")
                sources = search_result.get("sources", [])
                if sources:
                    sources_text = "\n\n**参考来源：**\n" + "\n".join(
                        f"- {url}" for url in sources[:5]
                    )
                    answer = answer + sources_text
                state["final_answer"] = answer
                # 将图表元数据附加在 metadata 中供前端使用
                if search_result.get("chart"):
                    state["metadata"]["chart"] = search_result["chart"]
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        # 数据库查询/分析相关意图
        sql_data = None
        analysis_data = None
        
        if sql_result:
            if sql_result.get("error"):
                state["final_answer"] = f"查询出错：{sql_result['error']}"
                state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
                return state
            sql_data = sql_result.get("data")
        
        if analysis_result:
            if analysis_result.get("error"):
                state["final_answer"] = f"分析出错：{analysis_result['error']}"
                state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
                return state
            analysis_data = analysis_result.get("analysis")
            # 将图表配置存入 metadata，流式接口和前端可读取
            if analysis_result.get("chart"):
                state["metadata"]["chart"] = analysis_result["chart"]
        
        try:
            prompt = get_summary_prompt(
                question=question,
                sql_result=sql_data,
                analysis_result=analysis_data
            )
            
            answer = self._llm_to_str(self.llm.invoke(prompt))
            state["final_answer"] = answer
            state["messages"] = list(state["messages"]) + [AIMessage(content=answer)]
            
        except Exception as e:
            state["final_answer"] = f"生成回答时出错：{str(e)}"
        
        return state
    
    def _match_cache(self, question: str, thread_id: str) -> Optional[Dict[str, str]]:
        """判断用户问题是否命中缓存（轻量LLM调用）

        Returns:
            {"key": "abc123", "mode": "exact"} 或 {"key": "abc123", "mode": "partial"} 或 None
        """
        try:
            inventory = self.sql_agent.get_cache_inventory(thread_id)
            if not inventory:
                return None

            lines = [f"- {k}: {v}" for k, v in inventory.items()]
            inventory_text = "\n".join(lines)

            prompt = get_cache_match_prompt(question, inventory_text)
            response = self._llm_to_str(self.llm.invoke(prompt)).strip()

            if response.upper().startswith("EXACT:"):
                key = response.split(":", 1)[1].strip()
                if key in inventory:
                    logger.info("缓存精确命中: key=%s", key)
                    return {"key": key, "mode": "exact"}
            elif response.upper().startswith("PARTIAL:"):
                key = response.split(":", 1)[1].strip()
                if key in inventory:
                    logger.info("缓存部分命中: key=%s", key)
                    return {"key": key, "mode": "partial"}
        except Exception as e:
            logger.warning("缓存匹配失败，跳过缓存: %s", e)

        return None

    def query(self, question: str, thread_id: str = "default", user_id: Optional[str] = None) -> str:
        """执行查询

        Args:
            question: 用户问题
            thread_id: 线程ID，用于区分不同的会话
            user_id: 用户ID，用于长期记忆

        Returns:
            回答结果
        """
        # 每轮递减缓存TTL
        cleared = self.sql_agent.tick_cache(thread_id)
        if cleared > 0:
            logger.debug("清除了 %d 条过期缓存", cleared)

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # 使用checkpointer保存会话状态
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = self.graph.invoke(initial_state, config)
        
        answer = final_state.get("final_answer", "抱歉，无法处理你的问题。")
        
        # 获取完整的对话历史（已经包含了当前的问题和回答）
        all_messages = list(final_state["messages"])
        
        logger.debug("当前会话共有 %d 条消息", len(all_messages))
        
        # 自动提取并保存长期记忆
        if user_id:
            self._extract_and_save_memory(all_messages, user_id)
        
        return answer
    
    def _extract_and_save_memory(self, messages: Sequence[BaseMessage], user_id: str):
        """自动提取并保存长期记忆"""
        try:
            if not self.memory_extractor.should_extract(messages, threshold=6):
                return
            
            preferences = self.memory_extractor.extract_preferences_from_conversation(
                messages, user_id
            )
            for key, value in preferences.items():
                self.long_term_memory.save_preference(user_id, key, str(value))
            
            knowledge_list = self.memory_extractor.extract_knowledge_from_conversation(
                messages, user_id
            )
            for knowledge in knowledge_list:
                self.long_term_memory.save_knowledge(
                    user_id,
                    knowledge.get("category", "其他"),
                    knowledge.get("content", ""),
                    knowledge.get("confidence", 0.8)
                )
        except Exception as e:
            logger.warning("提取记忆失败: %s", e)
    
    def stream_query(
        self,
        question: str,
        thread_id: str = "default",
        user_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式查询，以 SSE 格式生成事件流
        
        使用 LangGraph 的 graph.stream() 在每个节点完成后推送状态更新，
        最终 LLM 汇总回答以流式方式逐字输出。
        
        Yields:
            SSE 格式字符串：data: {...}\\n\\n
        """
        def sse(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs}, ensure_ascii=False)}\n\n"

        # 每轮递减缓存TTL
        cleared = self.sql_agent.tick_cache(thread_id)
        if cleared > 0:
            logger.debug("清除了 %d 条过期缓存", cleared)

        # --- 意图识别（直接调用，以便立即推送状态）---
        yield sse("status", message="正在识别问题意图...")
        
        user_context = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=3)
                preferences = self.long_term_memory.get_all_preferences(user_id)
                user_context = self._format_long_term_context(knowledge, preferences)
            except Exception:
                pass
        
        # 从 checkpointer 获取对话历史
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self.graph.get_state(config)
            existing_msgs = list(snapshot.values.get("messages", []))
        except Exception:
            existing_msgs = []
        
        temp_state: MasterAgentState = {
            "messages": existing_msgs,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {"thread_id": thread_id, "user_id": user_id}
        }
        conversation_history = self._get_conversation_history(temp_state)
        
        intent_prompt = get_master_intent_prompt(question, conversation_history, user_context)

        VALID_INTENTS = (
            "simple_answer", "sql_only", "analysis_only",
            "sql_and_analysis", "web_search", "search_and_sql"
        )

        try:
            raw = self.llm.invoke(intent_prompt)
            intent = self._llm_to_str(raw).strip().lower()

            # 精确匹配
            if intent not in VALID_INTENTS:
                # 第一次纠错
                correction_prompt = get_intent_correction_prompt(question, intent)
                raw2 = self.llm.invoke(correction_prompt)
                intent = self._llm_to_str(raw2).strip().lower()

                if intent not in VALID_INTENTS:
                    # 第二次纠错
                    correction_prompt2 = get_intent_correction_prompt(question, intent)
                    raw3 = self.llm.invoke(correction_prompt2)
                    intent = self._llm_to_str(raw3).strip().lower()

                    if intent not in VALID_INTENTS:
                        intent = "simple_answer"
        except Exception as e:
            intent = "simple_answer"
            err_msg = f"{type(e).__name__}: {e}"
            # 尝试获取更详细的错误信息（如 API 配额不足等）
            if "FreeTierOnly" in str(e) or "Quota" in str(e):
                err_msg = "通义千问 API 免费额度已用完，请在控制台开通付费或更换模型。"
            elif "InvalidApiKey" in str(e) or "Unauthorized" in str(e):
                err_msg = "DASHSCOPE_API_KEY 无效，请检查 API Key 是否正确。"
            yield sse("error", message=f"LLM 调用失败: {err_msg}")
        
        yield sse("intent", intent=intent)
        
        # --- 搜索不可用时降级 ---
        if intent in ("web_search", "search_and_sql") and not self.search_agent.available:
            msg = "联网搜索功能暂未启用，请配置 TAVILY_API_KEY 环境变量后重启。"
            yield sse("chunk", content=msg)
            yield sse("done", answer=msg)
            return
        
        sql_result = None
        analysis_result = None
        search_result = None
        final_answer = ""
        
        # --- 执行各子任务 ---
        if intent == "simple_answer":
            yield sse("status", message="正在生成回答...")
            final_answer = (
                "你好！我是智能数据查询助手，可以帮你查询数据库信息、"
                "进行数据分析，还支持联网搜索。有什么可以帮你的吗？"
            )
            yield sse("chunk", content=final_answer)
        
        else:
            # SQL 查询（适用于 sql_only / sql_and_analysis / search_and_sql）
            if intent in ("sql_only", "sql_and_analysis", "search_and_sql"):
                # 检查缓存
                cache_match = self._match_cache(question, thread_id)
                if cache_match:
                    cached = self.sql_agent.get_cached_result(thread_id, cache_match["key"])
                    if cached:
                        sql_result = cached
                        logger.info("流式路径缓存命中: key=%s, mode=%s",
                                   cache_match["key"], cache_match["mode"])
                        yield sse("status", message="使用缓存数据（跳过数据库查询）...")
                    else:
                        yield sse("status", message="正在查询数据库...")
                        sql_result = self.sql_agent.query(question, thread_id=thread_id)
                else:
                    yield sse("status", message="正在查询数据库...")
                    sql_result = self.sql_agent.query(question, thread_id=thread_id)

                if sql_result.get("sql"):
                    yield sse(
                        "sql",
                        sql=sql_result["sql"],
                        retry_count=sql_result.get("retry_count", 0)
                    )
                if sql_result.get("error"):
                    yield sse("error", message=f"数据库查询出错: {sql_result['error']}")
                
                # 保存会话数据
                if thread_id not in self.session_data:
                    self.session_data[thread_id] = {}
                self.session_data[thread_id]["last_sql_result"] = sql_result
            
            # 数据分析（适用于 analysis_only / sql_and_analysis）
            if intent in ("analysis_only", "sql_and_analysis"):
                yield sse("status", message="正在分析数据...")
                
                data_to_analyze = None
                if sql_result and sql_result.get("data"):
                    data_to_analyze = sql_result["data"]
                elif thread_id in self.session_data:
                    last = self.session_data[thread_id].get("last_sql_result", {})
                    data_to_analyze = last.get("data") if last else None
                
                if data_to_analyze:
                    analysis_result = self.analysis_agent.analyze(data_to_analyze, question)
                    if analysis_result.get("chart"):
                        yield sse("chart", config=analysis_result["chart"])
                else:
                    yield sse("error", message="没有可分析的数据，请先执行数据查询")
            
            # 纯联网搜索
            if intent == "web_search":
                yield sse("status", message="正在联网搜索...")
                search_result = self.search_agent.search(question)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])
            
            # 搜索 + 数据库联合分析
            if intent == "search_and_sql":
                yield sse("status", message="正在联网搜索行业数据...")
                sql_data_str = (sql_result.get("data") or "{}") if sql_result else "{}"
                search_result = self.search_agent.search_and_compare(question, sql_data_str)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])
            
            # --- 生成最终回答（流式输出 LLM 结果）---
            yield sse("status", message="正在生成回答...")
            
            if intent in ("web_search", "search_and_sql") and search_result:
                # 搜索智能体已生成完整回答，直接流式输出
                answer_text = search_result.get("answer", "未能获取搜索结果")
                if search_result.get("error"):
                    answer_text = f"搜索出错：{search_result['error']}"
                else:
                    sources = search_result.get("sources", [])
                    if sources:
                        answer_text += "\n\n**参考来源：**\n" + "\n".join(
                            f"- {url}" for url in sources[:5]
                        )
                final_answer = answer_text
                yield sse("chunk", content=final_answer)
            
            elif sql_result and sql_result.get("error"):
                final_answer = f"数据库查询出错：{sql_result['error']}"
                yield sse("chunk", content=final_answer)
            
            else:
                # 使用 LLM 流式生成汇总回答
                sql_data = sql_result.get("data") if sql_result else None
                analysis_data = analysis_result.get("analysis") if analysis_result else None
                
                summary_prompt = get_summary_prompt(
                    question=question,
                    sql_result=sql_data,
                    analysis_result=analysis_data
                )
                
                try:
                    import re
                    in_think = False
                    think_buffer = ""
                    for chunk in self.llm.stream(summary_prompt):
                        if isinstance(chunk, str):
                            chunk_text = chunk
                        elif hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        elif hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        else:
                            chunk_text = str(chunk)

                        # 过滤 <think>...</think> 思考内容，不发送给前端
                        # 采用 extract-after-</think> 策略：只丢弃 </think> 之前的内容
                        think_buffer += chunk_text
                        if '<think>' in think_buffer and not in_think:
                            in_think = True
                        if in_think:
                            think_end = think_buffer.rfind('</think>')
                            if think_end != -1:
                                # 提取 </think> 之后的内容（正式回答）
                                cleaned = think_buffer[think_end + len('</think>'):].strip()
                                if cleaned:
                                    final_answer += cleaned
                                    yield sse("chunk", content=cleaned)
                                think_buffer = ""
                                in_think = False
                            continue

                        think_buffer = ""
                        final_answer += chunk_text
                        yield sse("chunk", content=chunk_text)
                except Exception as e:
                    raw = self.llm.invoke(summary_prompt)
                    final_answer = self._llm_to_str(raw)
                    yield sse("chunk", content=final_answer)
        
        yield sse("done", answer=final_answer)
        
        # --- 保存对话历史到 LangGraph checkpointer ---
        try:
            new_messages = [HumanMessage(content=question), AIMessage(content=final_answer)]
            self.graph.update_state(
                config,
                {"messages": new_messages},
                as_node="summarize"
            )
            all_msgs = existing_msgs + new_messages
            if user_id:
                self._extract_and_save_memory(all_msgs, user_id)
        except Exception as e:
            logger.warning("保存对话历史失败（不影响本次回答）: %s", e)

