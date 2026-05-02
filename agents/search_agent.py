"""
DeepSearch 联网搜索子智能体

基于 Tavily 搜索引擎实现联网搜索能力，支持：
1. 纯联网搜索（web_search）- 回答与数据库无关的外部信息查询
2. 搜索+数据库联合分析（search_and_sql）- 将行业数据与公司内部数据对比
"""

import json
import os
from typing import Dict, Any, Generator, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseLLM

import sys
sys.path.append(str(Path(__file__).parent.parent))
from agents.base import BaseSubAgent
from prompts import get_search_synthesis_prompt, get_search_and_sql_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchAgent(BaseSubAgent):
    """DeepSearch 联网搜索子智能体

    使用 Tavily 搜索引擎获取实时网络信息，结合 LLM 综合生成回答。
    支持纯搜索和「搜索+数据库」联合分析两种模式。
    """

    intent_name = "web_search"
    node_name = "call_web_search"

    def __init__(self, llm: BaseLLM, tavily_api_key: str = "", max_results: int = 5):
        """初始化搜索智能体

        Args:
            llm: 语言模型实例
            tavily_api_key: Tavily API Key（请先申请获得）
            max_results: 每次搜索返回的最大结果数
        """
        super().__init__(llm)
        self.max_results = max_results
        self._available = False
        self.search_tool = None
        self._init_search_tool(tavily_api_key)
    
    @property
    def available(self) -> bool:
        """搜索智能体是否可用（取决于 TAVILY_API_KEY 是否配置成功）"""
        return self._available

    @available.setter
    def available(self, value: bool):
        self._available = value

    def _init_search_tool(self, api_key: str):
        """初始化 Tavily 搜索工具"""
        effective_key = api_key or os.getenv("TAVILY_API_KEY", "")

        if not effective_key or effective_key.startswith("${"):
            logger.warning("未配置 TAVILY_API_KEY，联网搜索功能不可用")
            return

        try:
            os.environ["TAVILY_API_KEY"] = effective_key
            from langchain_tavily import TavilySearch
            self.search_tool = TavilySearch(max_results=self.max_results)
            self._available = True
            logger.info("Tavily 搜索工具初始化成功")
        except ImportError:
            logger.error("langchain-tavily 未安装，请运行: pip install langchain-tavily")
        except Exception as e:
            logger.error("搜索工具初始化失败: %s", e)

    # ---- BaseSubAgent 接口 ----

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜索，若 state 中有 sql_result 则进行联合分析"""
        question = state["user_question"]

        # 检查是否需要联合分析（search_and_sql 管道场景）
        sql_result = state.get("sql_result")
        if sql_result and sql_result.get("data") and not sql_result.get("error"):
            result = self.search_and_compare(question, sql_result["data"])
        else:
            result = self.search(question)

        state["search_result"] = result
        state["metadata"]["search_result"] = result
        return state

    def stream_events(self, state: Dict[str, Any]) -> Generator[str, None, None]:
        """流式推送搜索结果，包含 sources 事件"""
        import re

        result_state = self.execute(state)
        search_result = result_state.get("search_result", {})

        # 推送来源
        if search_result.get("sources"):
            yield f"data: {json.dumps({'type': 'sources', 'sources': search_result['sources']}, ensure_ascii=False)}\n\n"

        # 推送错误
        if search_result.get("error"):
            yield f"data: {json.dumps({'type': 'error', 'message': search_result['error']}, ensure_ascii=False)}\n\n"

        # 推送回答
        answer = search_result.get("answer", "未能获取搜索结果")
        yield f"data: {json.dumps({'type': 'chunk', 'content': answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'answer': answer}, ensure_ascii=False)}\n\n"

    # ---- 内部方法 ----

    def _format_search_results(self, results: List[Dict]) -> str:
        """格式化搜索结果为可读文本
        
        Args:
            results: Tavily 原始搜索结果列表
            
        Returns:
            格式化的文本
        """
        if not results:
            return "未找到相关搜索结果"
        
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            content = r.get("content", "")
            url = r.get("url", "")
            # 截取内容避免过长
            content_preview = content[:600] if len(content) > 600 else content
            formatted.append(f"[来源{i}] {title}\n{content_preview}\n链接: {url}")
        
        return "\n\n".join(formatted)
    
    def _invoke_search(self, question: str):
        """调用 Tavily 搜索，统一处理多种返回格式
        
        Returns:
            (formatted_text: str, sources: List[str])
        """
        invoke_result = self.search_tool.invoke(question)
        
        sources = []
        formatted_text = ""

        # 新版返回 dict，包含 "results" 列表
        if isinstance(invoke_result, dict):
            results = invoke_result.get("results", [])
            sources = [r.get("url", "") for r in results if isinstance(r, dict) and r.get("url")]
            formatted_text = self._format_search_results(results)

        # tuple (content_str, artifact_list)
        elif isinstance(invoke_result, tuple) and len(invoke_result) == 2:
            content_str, artifact = invoke_result
            formatted_text = content_str if isinstance(content_str, str) else str(content_str)
            if isinstance(artifact, list):
                sources = [r.get("url", "") for r in artifact if isinstance(r, dict) and r.get("url")]

        # list of dicts
        elif isinstance(invoke_result, list):
            sources = [r.get("url", "") for r in invoke_result if isinstance(r, dict) and r.get("url")]
            formatted_text = self._format_search_results(invoke_result)

        # 纯字符串
        elif isinstance(invoke_result, str):
            formatted_text = invoke_result

        else:
            formatted_text = str(invoke_result)
        
        return formatted_text, sources

    def search(self, question: str) -> Dict[str, Any]:
        """纯联网搜索模式
        
        搜索外部信息并用 LLM 综合生成回答，适合与数据库无关的信息查询。
        
        Args:
            question: 用户搜索问题
            
        Returns:
            {
                "answer": LLM综合后的回答,
                "sources": 来源URL列表,
                "error": 错误信息（成功时为None）
            }
        """
        result = {
            "answer": None,
            "sources": [],
            "error": None
        }
        
        if not self.available:
            result["error"] = (
                "联网搜索功能未启用。请配置 TAVILY_API_KEY 环境变量后重启系统。\n"
                "联网搜索功能未启用"
            )
            return result
        
        try:
            logger.info("正在搜索: %s", question)
            formatted_text, sources = self._invoke_search(question)
            result["sources"] = sources
            
            # 用 LLM 综合搜索结果生成回答
            prompt = get_search_synthesis_prompt(question, formatted_text)
            import re
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, 'content') else str(raw)
            text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
            text = re.sub(r'</think>', '', text).strip()
            result["answer"] = text
            
            logger.info("搜索完成，来源 %d 个", len(sources))
            
        except Exception as e:
            result["error"] = f"联网搜索失败: {str(e)}"
            logger.error("搜索出错: %s", e)
        
        return result
    
    def search_and_compare(self, question: str, sql_result_json: str) -> Dict[str, Any]:
        """联网搜索 + 数据库数据联合分析模式
        
        先搜索行业/外部数据，再与数据库查询结果进行对比分析，
        实现「公司内部数据 vs 行业外部数据」的深度对比。
        
        Args:
            question: 用户问题（包含对比分析意图）
            sql_result_json: 数据库查询结果 JSON 字符串
            
        Returns:
            {
                "answer": 联合分析回答,
                "sources": 搜索来源URL列表,
                "error": 错误信息（成功时为None）
            }
        """
        result = {
            "answer": None,
            "sources": [],
            "error": None
        }
        
        if not self.available:
            result["error"] = "联网搜索功能未启用，请配置 TAVILY_API_KEY"
            return result
        
        try:
            logger.info("联合分析搜索: %s", question)
            formatted_text, sources = self._invoke_search(question)
            result["sources"] = sources
            
            # 联合分析：搜索结果 + 数据库结果
            prompt = get_search_and_sql_prompt(question, formatted_text, sql_result_json)
            import re
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, 'content') else str(raw)
            text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
            text = re.sub(r'</think>', '', text).strip()
            result["answer"] = text
            
            logger.info("联合分析完成")
            
        except Exception as e:
            result["error"] = f"联合搜索分析失败: {str(e)}"
            logger.error("联合分析出错: %s", e)
        
        return result
