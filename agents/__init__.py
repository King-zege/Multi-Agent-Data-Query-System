"""
智能体模块

包含主智能体和子智能体的实现。
采用插件化架构：BaseSubAgent 抽象类 + AgentRegistry 注册中心。
"""

from .base import BaseSubAgent, AgentRegistry
from .master_agent import MasterAgent
from .sql_agent import SQLQueryAgent
from .analysis_agent import DataAnalysisAgent
from .search_agent import WebSearchAgent

__all__ = [
    'BaseSubAgent', 'AgentRegistry',
    'MasterAgent', 'SQLQueryAgent', 'DataAnalysisAgent', 'WebSearchAgent',
]

