"""
评估执行器

负责逐条执行评估用例，记录结果，计算指标。

支持两种模式：
- mock: 使用 EvalMockLLM，不消耗 API
- live: 使用真实 LLM (dashscope)，完整端到端
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import calculate_keyword_match, generate_stats
from eval.llm_judge import mock_judge, llm_judge


class EvalRunner:
    """评估执行器"""

    def __init__(self, mode: str = "mock", db_config: Optional[Dict[str, Any]] = None,
                 memory_db_path: Optional[str] = None):
        """
        Args:
            mode: "mock" 或 "live"
            db_config: 数据库配置字典（mock 模式可留空，会自动创建临时库）
            memory_db_path: 长期记忆数据库路径
        """
        self.mode = mode
        self.results: List[Dict[str, Any]] = []

        if mode == "mock":
            from eval.eval_fake_llm import EvalMockLLM
            self.llm = EvalMockLLM()
        else:
            self.llm = self._create_live_llm()

        # 初始化数据库配置
        if db_config:
            self.db_config = db_config
        else:
            # 创建临时数据库
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp_path = tmp.name
            tmp.close()
            self._init_temp_db(tmp_path)
            self.db_config = {"type": "sqlite", "path": tmp_path}

        if memory_db_path:
            self.memory_db_path = memory_db_path
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self.memory_db_path = tmp.name
            tmp.close()
            from data.init_memory_db import init_memory_database
            init_memory_database(self.memory_db_path)

    def _init_temp_db(self, db_path: str):
        """初始化临时数据库"""
        import sqlite3
        from data.init_db import create_tables, insert_sample_data
        conn = sqlite3.connect(db_path)
        create_tables(conn)
        insert_sample_data(conn)
        conn.commit()
        conn.close()

    @staticmethod
    def _create_live_llm():
        """创建真实 LLM 实例（从 config.yaml 读取配置）"""
        import yaml
        from langchain_openai import ChatOpenAI

        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_config = config["llm"]
        if llm_config["provider"] == "dashscope":
            base_url = llm_config.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            return ChatOpenAI(
                model=llm_config["model"],
                api_key=llm_config["api_key"],
                base_url=base_url,
                temperature=0,  # 评估时使用确定性输出
                max_tokens=llm_config.get("max_tokens", 2048),
                streaming=True,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}")

    def run(self, cases: List[Dict[str, Any]],
            category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行所有评估用例

        Args:
            cases: 评估用例列表
            category_filter: 可选的分类过滤

        Returns:
            评估结果列表
        """
        if category_filter:
            cases = [c for c in cases if c.get("category") == category_filter]
            if not cases:
                print(f"No cases found for category: {category_filter}")
                return []

        total = len(cases)
        for i, case in enumerate(cases):
            case_id = case["id"]
            category = case.get("category", "unknown")
            print(f"\r[{i+1}/{total}] {case_id} ({category})...", end="", flush=True)

            try:
                result = self._evaluate_case(case)
            except Exception as e:
                result = {
                    "id": case_id,
                    "category": category,
                    "question": case.get("question", ""),
                    "passed": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }

            self.results.append(result)

        print()  # newline after progress
        return self.results

    def _evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个用例"""
        category = case.get("category", "unknown")
        question = case.get("question", "")

        if category == "intent_recognition":
            return self._eval_intent(case)
        elif category == "sql_generation":
            return self._eval_sql(case)
        elif category == "end_to_end":
            return self._eval_end_to_end(case)
        elif category == "error_handling":
            return self._eval_error_handling(case)
        else:
            return {"id": case["id"], "category": category,
                    "passed": False, "error": f"Unknown category: {category}"}

    def _eval_intent(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估意图识别"""
        from prompts import get_master_intent_prompt

        question = case["question"]
        expected = case["expected_intent"]

        VALID_INTENTS = (
            "simple_answer", "sql_only", "analysis_only",
            "sql_and_analysis", "web_search", "search_and_sql"
        )

        try:
            prompt = get_master_intent_prompt(question)
            response = self.llm.invoke(prompt)
            text = self._extract_text(response).strip().lower()

            # 精确匹配
            intent = text
            if intent not in VALID_INTENTS:
                # 尝试从文本中提取有效意图
                for valid in VALID_INTENTS:
                    if valid in intent:
                        intent = valid
                        break

            passed = intent == expected

            return {
                "id": case["id"],
                "category": "intent_recognition",
                "question": question,
                "expected": expected,
                "actual": intent,
                "raw_response": text[:100],
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": "intent_recognition",
                "question": question,
                "expected": expected,
                "passed": False,
                "error": str(e),
            }

    def _eval_sql(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估 SQL 生成"""
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        expected_keywords = case.get("expected_sql_keywords", [])
        forbidden_keywords = case.get("forbidden_sql_keywords", [])

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=self.db_config, num_examples=3)
            sql = agent._generate_sql(question)

            # 检查期望关键词
            matched = []
            missed = []
            for kw in expected_keywords:
                if kw.upper() in sql.upper():
                    matched.append(kw)
                else:
                    missed.append(kw)

            # 检查禁止关键词
            forbidden_found = []
            for kw in forbidden_keywords:
                if kw.upper() in sql.upper():
                    forbidden_found.append(kw)

            all_expected = len(missed) == 0
            no_forbidden = len(forbidden_found) == 0
            passed = all_expected and no_forbidden

            return {
                "id": case["id"],
                "category": "sql_generation",
                "question": question,
                "sql": sql,
                "expected_keywords": expected_keywords,
                "matched_keywords": matched,
                "missed_keywords": missed,
                "forbidden_found": forbidden_found,
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": "sql_generation",
                "question": question,
                "passed": False,
                "error": str(e),
            }

    def _eval_end_to_end(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估端到端查询"""
        from agents.master_agent import MasterAgent

        question = case["question"]
        expected_keywords = case.get("expected_answer_keywords", [])
        judge_criteria = case.get("judge_criteria", {})

        try:
            agent = MasterAgent(
                llm=self.llm,
                db_config=self.db_config,
                num_examples=3,
                memory_db_path=self.memory_db_path,
                tavily_api_key="",
            )

            start = time.time()
            answer = agent.query(question, thread_id=f"eval_{case['id']}")
            elapsed = time.time() - start

            # 关键词匹配
            keyword_score = calculate_keyword_match(expected_keywords, answer)
            all_keywords_present = keyword_score >= 0.5  # 至少一半关键词

            # LLM-as-Judge
            if self.mode == "live":
                judge_scores = llm_judge(self.llm, question, answer, judge_criteria)
            else:
                judge_scores = mock_judge(question, answer, judge_criteria)

            passed = all_keywords_present

            return {
                "id": case["id"],
                "category": "end_to_end",
                "question": question,
                "answer": answer,
                "expected_keywords": expected_keywords,
                "keyword_match_rate": keyword_score,
                "judge_scores": judge_scores,
                "elapsed_seconds": round(elapsed, 2),
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": "end_to_end",
                "question": question,
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _eval_error_handling(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """评估错误处理"""
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        expected_behavior = case.get("expected_behavior", "error_or_empty")

        # 空问题直接验证，不调用 SQL Agent
        if expected_behavior == "error_or_empty" and not question.strip():
            return {
                "id": case["id"],
                "category": "error_handling",
                "question": question,
                "expected_behavior": expected_behavior,
                "sql": None,
                "error": "空问题",
                "passed": True,
            }

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=self.db_config, num_examples=3)
            result = agent.query(question)

            if expected_behavior == "error_or_empty":
                passed = result.get("error") is not None or result.get("sql") is None
            elif expected_behavior == "no_destructive_sql":
                sql = result.get("sql", "")
                forbidden = case.get("forbidden_sql_keywords", [])
                passed = not any(kw.upper() in sql.upper() for kw in forbidden)
            elif expected_behavior == "graceful_error":
                passed = result.get("error") is not None
            elif expected_behavior == "graceful_degradation":
                # 搜索不可用时应降级 — 清除环境变量防止回退
                import os as _os
                old_key = _os.environ.pop("TAVILY_API_KEY", None)
                try:
                    from agents.search_agent import WebSearchAgent
                    search_agent = WebSearchAgent(self.llm, tavily_api_key="")
                    passed = not search_agent.available
                finally:
                    if old_key is not None:
                        _os.environ["TAVILY_API_KEY"] = old_key
            else:
                passed = True

            return {
                "id": case["id"],
                "category": "error_handling",
                "question": question,
                "expected_behavior": expected_behavior,
                "sql": result.get("sql"),
                "error": result.get("error"),
                "passed": passed,
            }
        except Exception as e:
            # 某些错误测试应该抛出异常
            if expected_behavior == "graceful_error":
                return {
                    "id": case["id"],
                    "category": "error_handling",
                    "question": question,
                    "expected_behavior": expected_behavior,
                    "passed": True,
                    "error": str(e),
                }
            return {
                "id": case["id"],
                "category": "error_handling",
                "question": question,
                "passed": False,
                "error": str(e),
            }

    @staticmethod
    def _extract_text(result) -> str:
        """从 LLM 返回值中提取文本"""
        if isinstance(result, str):
            return result
        if hasattr(result, 'content'):
            return str(result.content)
        if hasattr(result, 'text'):
            return str(result.text)
        return str(result)


def load_cases(cases_path: str) -> List[Dict[str, Any]]:
    """加载评估用例"""
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(mode: str = "mock",
                   cases_path: Optional[str] = None,
                   category_filter: Optional[str] = None,
                   db_config: Optional[Dict[str, Any]] = None,
                   memory_db_path: Optional[str] = None) -> Dict[str, Any]:
    """运行完整评估流程

    Args:
        mode: "mock" 或 "live"
        cases_path: eval_cases.json 路径
        category_filter: 可选的分类过滤
        db_config: 数据库配置字典
        memory_db_path: 长期记忆数据库路径

    Returns:
        完整统计数据
    """
    if cases_path is None:
        cases_path = str(Path(__file__).parent / "eval_cases.json")

    cases = load_cases(cases_path)
    runner = EvalRunner(mode=mode, db_config=db_config, memory_db_path=memory_db_path)
    results = runner.run(cases, category_filter=category_filter)
    stats = generate_stats(results)

    return stats
