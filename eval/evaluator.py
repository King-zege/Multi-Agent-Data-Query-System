"""
评估执行器

负责逐条执行评估用例，记录结果，计算指标。

支持两种模式：
- mock: 使用 EvalMockLLM，不消耗 API
- live: 使用真实 LLM (dashscope)，完整端到端

支持五种新评估分类：
- sql_execution_accuracy: SQL 执行成功/失败 + 重试
- result_exact_match: 结果集精确匹配
- result_set_overlap: 结果集 Jaccard 重叠
- cross_db_consistency: 跨库一致性（多数据库类型）
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

from eval.metrics import (
    calculate_keyword_match,
    compare_result_sets,
    generate_stats,
)
from eval.llm_judge import mock_judge, llm_judge


# ============================================================================
# 数据库连接工具
# ============================================================================

def _create_db_engine(db_config: Dict[str, Any]):
    """从 db_config 创建 SQLAlchemy 引擎"""
    from sqlalchemy import create_engine

    db_type = db_config["type"]
    if db_type == "sqlite":
        return create_engine(f"sqlite:///{db_config['path']}")
    elif db_type == "mysql":
        import pymysql
        return create_engine(
            f"mysql+pymysql://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    elif db_type == "postgresql":
        import psycopg2
        return create_engine(
            f"postgresql+psycopg2://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    else:
        raise ValueError(f"Unsupported db_type: {db_type}")


def _execute_sql(db_config: Dict[str, Any], sql: str) -> tuple:
    """执行 SQL 并返回 (columns, rows, error)

    Returns:
        (columns: list, rows: list of dicts, error: str or None)
    """
    try:
        engine = _create_db_engine(db_config)
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys()) if result.returns_rows else []
            rows = [dict(zip(columns, row)) for row in result.fetchall()] if columns else []
        engine.dispose()
        return columns, rows, None
    except Exception as e:
        return [], [], f"{type(e).__name__}: {e}"


def _resolve_db_config(case: Dict[str, Any], db_type: str) -> Dict[str, Any]:
    """根据 case 的 falcon_db_id 和 db_type 解析数据库配置

    如果 case 有 falcon_db_id，从 db_configs 注册表查找；
    否则回退到默认 company 数据库。
    """
    from eval.db_configs import get_db_config

    falcon_db_id = case.get("falcon_db_id")
    if falcon_db_id:
        return get_db_config(falcon_db_id, db_type)
    else:
        # 原始 company 用例: 使用公司数据库 (SQLite only)
        # 对于 MySQL/PG，这些用例仍然使用 SQLite（向后兼容）
        if db_type == "sqlite":
            return get_db_config("company", "sqlite")
        else:
            # 非 SQLite 模式下，company 用例会失败 — 返回 None 表示跳过
            return None


# ============================================================================
# EvalRunner — 单数据库评估
# ============================================================================

class EvalRunner:
    """评估执行器"""

    def __init__(self, mode: str = "mock",
                 db_type: str = "sqlite",
                 db_config: Optional[Dict[str, Any]] = None,
                 memory_db_path: Optional[str] = None):
        """
        Args:
            mode: "mock" 或 "live"
            db_type: 数据库类型标签 ("sqlite"/"mysql"/"postgresql")
            db_config: 默认数据库配置（用于没有 falcon_db_id 的旧用例）
            memory_db_path: 长期记忆数据库路径
        """
        self.mode = mode
        self.db_type = db_type
        self.default_db_config = db_config
        self.results: List[Dict[str, Any]] = []

        if mode == "mock":
            from eval.eval_fake_llm import EvalMockLLM
            self.llm = EvalMockLLM()
        else:
            self.llm = self._create_live_llm()

        # 初始化默认数据库配置（用于公司用例）
        if db_config is None and db_type == "sqlite":
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp_path = tmp.name
            tmp.close()
            self._init_temp_db(tmp_path)
            self.default_db_config = {"type": "sqlite", "path": tmp_path}

        if memory_db_path:
            self.memory_db_path = memory_db_path
        else:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            self.memory_db_path = tmp.name
            tmp.close()
            from data.init_memory_db import init_memory_database
            init_memory_database(self.memory_db_path)

        # 为 mock 模式加载 Falcon fallback
        if mode == "mock":
            self._setup_mock_fallbacks()

    def _setup_mock_fallbacks(self):
        """为 mock 模式预加载 Falcon ground truth SQL"""
        if hasattr(self.llm, 'load_falcon_fallbacks'):
            # 尝试加载 v2 用例
            cases_path = Path(__file__).parent / "eval_cases_v2.json"
            if cases_path.exists():
                cases = load_cases(str(cases_path))
                self.llm.load_falcon_fallbacks(cases)

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
                temperature=0,
                max_tokens=llm_config.get("max_tokens", 2048),
                streaming=True,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}")

    def _get_db_config_for_case(self, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """为指定用例获取数据库配置"""
        return _resolve_db_config(case, self.db_type)

    def run(self, cases: List[Dict[str, Any]],
            category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行所有评估用例"""
        if category_filter:
            cases = [c for c in cases if c.get("category") == category_filter]
            if not cases:
                print(f"No cases found for category: {category_filter}")
                return []

        total = len(cases)
        crossed_off = 0

        for i, case in enumerate(cases):
            case_id = case["id"]
            category = case.get("category", "unknown")
            print(f"\r[{i+1}/{total}] {case_id} ({category}) [{self.db_type}]...",
                  end="", flush=True)

            # 检查该 case 在当前 db_type 下是否有可用的数据库配置
            db_config = self._get_db_config_for_case(case)
            if db_config is None:
                crossed_off += 1
                self.results.append({
                    "id": case_id,
                    "category": category,
                    "question": case.get("question", ""),
                    "db_type": self.db_type,
                    "passed": False,
                    "error": f"No db_config for db_type={self.db_type}",
                })
                continue

            try:
                result = self._evaluate_case(case, db_config)
            except Exception as e:
                result = {
                    "id": case_id,
                    "category": category,
                    "question": case.get("question", ""),
                    "db_type": self.db_type,
                    "passed": False,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }

            result["db_type"] = self.db_type
            self.results.append(result)

        if crossed_off > 0:
            print(f"  ({crossed_off} cases skipped — no db_config for {self.db_type})")
        print()
        return self.results

    def _evaluate_case(self, case: Dict[str, Any],
                       db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个用例"""
        category = case.get("category", "unknown")

        dispatch = {
            "intent_recognition": self._eval_intent,
            "sql_generation": self._eval_sql_gen,
            "end_to_end": self._eval_end_to_end,
            "error_handling": self._eval_error_handling,
            "sql_execution_accuracy": self._eval_sql_execution,
            "result_exact_match": self._eval_result_match,
            "result_set_overlap": self._eval_result_match,
            "cross_db_consistency": self._eval_sql_execution,
        }

        handler = dispatch.get(category)
        if handler is None:
            return {
                "id": case["id"], "category": category,
                "passed": False,
                "error": f"Unknown category: {category}",
            }

        return handler(case, db_config)

    # ============== 原有分类 ==============

    def _eval_intent(self, case: Dict[str, Any],
                     db_config: Dict[str, Any]) -> Dict[str, Any]:
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

            intent = text
            if intent not in VALID_INTENTS:
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

    def _eval_sql_gen(self, case: Dict[str, Any],
                      db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估 SQL 生成（关键词检查）"""
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        expected_keywords = case.get("expected_sql_keywords", [])
        forbidden_keywords = case.get("forbidden_sql_keywords", [])

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=db_config, num_examples=3)
            sql = agent._generate_sql(question)

            matched = []
            missed = []
            for kw in expected_keywords:
                if kw.upper() in sql.upper():
                    matched.append(kw)
                else:
                    missed.append(kw)

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
                "db_type": self.db_type,
                "expected_keywords": expected_keywords,
                "matched_keywords": matched,
                "missed_keywords": missed,
                "forbidden_found": forbidden_found,
                "required_tables": case.get("required_tables", []),
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": "sql_generation",
                "question": question,
                "db_type": self.db_type,
                "passed": False,
                "error": str(e),
            }

    def _eval_end_to_end(self, case: Dict[str, Any],
                         db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估端到端查询"""
        from agents.master_agent import MasterAgent

        question = case["question"]
        expected_keywords = case.get("expected_answer_keywords", [])
        judge_criteria = case.get("judge_criteria", {})

        try:
            agent = MasterAgent(
                llm=self.llm,
                db_config=db_config,
                num_examples=3,
                memory_db_path=self.memory_db_path,
                tavily_api_key="",
            )

            start = time.time()
            answer = agent.query(question, thread_id=f"eval_{case['id']}")
            elapsed = time.time() - start

            keyword_score = calculate_keyword_match(expected_keywords, answer)
            all_keywords_present = keyword_score >= 0.5

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
                "db_type": self.db_type,
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
                "db_type": self.db_type,
                "passed": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def _eval_error_handling(self, case: Dict[str, Any],
                             db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估错误处理"""
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        expected_behavior = case.get("expected_behavior", "error_or_empty")

        if expected_behavior == "error_or_empty" and not question.strip():
            return {
                "id": case["id"],
                "category": "error_handling",
                "question": question,
                "db_type": self.db_type,
                "expected_behavior": expected_behavior,
                "sql": None,
                "error": "空问题",
                "passed": True,
            }

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=db_config, num_examples=3)
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
                "db_type": self.db_type,
                "expected_behavior": expected_behavior,
                "sql": result.get("sql"),
                "error": result.get("error"),
                "passed": passed,
            }
        except Exception as e:
            if expected_behavior == "graceful_error":
                return {
                    "id": case["id"],
                    "category": "error_handling",
                    "question": question,
                    "db_type": self.db_type,
                    "expected_behavior": expected_behavior,
                    "passed": True,
                    "error": str(e),
                }
            return {
                "id": case["id"],
                "category": "error_handling",
                "question": question,
                "db_type": self.db_type,
                "passed": False,
                "error": str(e),
            }

    # ============== 新分类 ==============

    def _eval_sql_execution(self, case: Dict[str, Any],
                            db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估 SQL 执行准确率

        1. 生成 SQL
        2. 执行 SQL（live 模式）或模拟执行（mock 模式）
        3. 如果失败，尝试一次修复重试
        4. 记录执行状态和重试信息
        """
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        ground_truth_sql = case.get("ground_truth_sql", "")
        forbidden_keywords = case.get("forbidden_sql_keywords", [])

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=db_config, num_examples=3)
            sql = agent._generate_sql(question)

            # 检查禁止关键词
            forbidden_found = [
                kw for kw in forbidden_keywords
                if kw.upper() in sql.upper()
            ]

            # 执行 SQL
            if self.mode == "live":
                sql_executed, exec_error, retry_sql, retry_count = \
                    self._execute_with_retry(db_config, sql, agent)
            else:
                # Mock 模式：检查 SQL 非空且无禁止关键词即为"执行成功"
                sql_executed = bool(sql.strip()) and len(forbidden_found) == 0
                exec_error = None if sql_executed else "Empty or destructive SQL"
                retry_sql = None
                retry_count = 0

            passed = sql_executed

            return {
                "id": case["id"],
                "category": case["category"],
                "question": question,
                "falcon_db_id": case.get("falcon_db_id"),
                "db_type": self.db_type,
                "sql": sql,
                "sql_executed": sql_executed,
                "execution_error": exec_error,
                "retry_sql": retry_sql,
                "retry_count": retry_count,
                "forbidden_found": forbidden_found,
                "ground_truth_sql": ground_truth_sql[:200] if ground_truth_sql else None,
                "required_tables": case.get("required_tables", []),
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": case["category"],
                "question": question,
                "falcon_db_id": case.get("falcon_db_id"),
                "db_type": self.db_type,
                "sql": None,
                "sql_executed": False,
                "execution_error": str(e),
                "retry_count": 0,
                "required_tables": case.get("required_tables", []),
                "passed": False,
            }

    def _eval_result_match(self, case: Dict[str, Any],
                           db_config: Dict[str, Any]) -> Dict[str, Any]:
        """评估结果集匹配

        1. 生成 SQL 或使用 ground truth SQL
        2. 执行 SQL 获取实际结果
        3. 与 ground_truth_answer 比较
        4. 记录 Jaccard、precision、recall
        """
        from agents.sql_agent import SQLQueryAgent

        question = case["question"]
        ground_truth_sql = case.get("ground_truth_sql", "")
        expected_answer = case.get("ground_truth_answer", [])
        is_order_matters = case.get("is_order_matters", False)
        category = case["category"]

        try:
            agent = SQLQueryAgent(llm=self.llm, db_config=db_config, num_examples=3)
            sql = agent._generate_sql(question)

            # 执行生成的 SQL（live）或使用期望答案作为"模拟结果"（mock）
            if self.mode == "live":
                columns, actual_rows, exec_error = _execute_sql(db_config, sql)
                sql_executed = exec_error is None
            else:
                # Mock 模式：模拟结果与 ground truth 完全匹配
                sql_executed = True
                exec_error = None
                actual_rows = expected_answer

            # 结果比较
            if sql_executed and actual_rows is not None:
                comparison = compare_result_sets(
                    actual_rows, expected_answer,
                    order_matters=(category == "result_exact_match")
                )
            else:
                comparison = {
                    "is_match": False, "actual_count": 0, "expected_count": len(expected_answer),
                    "jaccard": 0.0, "precision": 0.0, "recall": 0.0,
                }

            # 判断通过条件
            if category == "result_exact_match":
                passed = sql_executed and comparison.get("is_match", False)
            else:  # result_set_overlap
                passed = sql_executed and comparison.get("jaccard", 0) >= 0.6

            return {
                "id": case["id"],
                "category": category,
                "question": question,
                "falcon_db_id": case.get("falcon_db_id"),
                "db_type": self.db_type,
                "sql": sql,
                "sql_executed": sql_executed,
                "execution_error": exec_error,
                "actual_rows": actual_rows,
                "expected_rows": expected_answer,
                "result_comparison": comparison,
                "ground_truth_sql": ground_truth_sql[:200] if ground_truth_sql else None,
                "required_tables": case.get("required_tables", []),
                "passed": passed,
            }
        except Exception as e:
            return {
                "id": case["id"],
                "category": category,
                "question": question,
                "falcon_db_id": case.get("falcon_db_id"),
                "db_type": self.db_type,
                "sql": None,
                "sql_executed": False,
                "execution_error": str(e),
                "required_tables": case.get("required_tables", []),
                "passed": False,
            }

    def _execute_with_retry(self, db_config: Dict[str, Any], sql: str,
                            agent) -> tuple:
        """执行 SQL，失败时尝试一次修复重试

        Returns:
            (sql_executed: bool, exec_error: str|None, retry_sql: str|None, retry_count: int)
        """
        columns, rows, error = _execute_sql(db_config, sql)

        if error is None:
            return True, None, None, 0

        # 尝试修复
        try:
            question = ""  # 从 context 获取
            fixed_sql = agent._fix_sql(question, sql, error)
            if fixed_sql and fixed_sql != sql:
                _, _, retry_error = _execute_sql(db_config, fixed_sql)
                if retry_error is None:
                    return True, None, fixed_sql, 1
                else:
                    return False, f"Original: {error}; Retry: {retry_error}", fixed_sql, 1
        except Exception:
            pass

        return False, error, None, 1

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


# ============================================================================
# CrossDBEvalRunner — 跨数据库评估
# ============================================================================

class CrossDBEvalRunner:
    """跨数据库评估执行器 — 同一批用例在多种数据库类型上执行"""

    def __init__(self, mode: str = "mock",
                 db_types: Optional[List[str]] = None,
                 memory_db_path: Optional[str] = None):
        """
        Args:
            mode: "mock" 或 "live"
            db_types: 数据库类型列表，默认 ["sqlite", "mysql", "postgresql"]
            memory_db_path: 长期记忆数据库路径
        """
        self.mode = mode
        self.db_types = db_types or ["sqlite", "mysql", "postgresql"]
        self.memory_db_path = memory_db_path

    def run(self, cases: List[Dict[str, Any]],
            category_filter: Optional[str] = None) -> Dict[str, Any]:
        """跨数据库执行评估

        对于 cross_db_consistency 用例：在每个 db_type 上分别执行
        对于其他用例：在第一个 db_type 上执行（避免重复）
        """
        from eval.db_configs import get_db_config

        all_results: List[Dict[str, Any]] = []

        # 分离 cross_db 用例和普通用例
        cross_cases = [c for c in cases if c.get("category") == "cross_db_consistency"]
        normal_cases = [c for c in cases if c.get("category") != "cross_db_consistency"]

        if category_filter:
            if category_filter == "cross_db_consistency":
                normal_cases = []
                cross_cases = [c for c in cross_cases if c.get("category") == category_filter]
            else:
                cross_cases = []
                normal_cases = [c for c in normal_cases if c.get("category") == category_filter]

        # 普通用例：仅在第一个 db_type 上运行
        if normal_cases and self.db_types:
            primary_db = self.db_types[0]
            # 使用公司数据库配置作为默认值
            default_config = get_db_config("company", "sqlite") if primary_db == "sqlite" else None
            runner = EvalRunner(
                mode=self.mode,
                db_type=primary_db,
                db_config=default_config,
                memory_db_path=self.memory_db_path,
            )
            # 覆盖 mock fallback 加载（避免重复加载）
            normal_results = runner.run(normal_cases)
            all_results.extend(normal_results)

        # 跨库一致性用例：在每个 db_type 上运行
        for db_type in self.db_types:
            if not cross_cases:
                break

            default_config = get_db_config("company", "sqlite") if db_type == "sqlite" else None
            runner = EvalRunner(
                mode=self.mode,
                db_type=db_type,
                db_config=default_config,
                memory_db_path=self.memory_db_path,
            )

            db_results = runner.run(cross_cases)
            all_results.extend(db_results)
            print(f"  Cross-DB [{db_type}]: {len(db_results)} results")

        # 生成统计
        cross_db_results = [r for r in all_results
                           if r.get("category") == "cross_db_consistency"]
        stats = generate_stats(all_results, cross_db_results=cross_db_results)

        return stats


# ============================================================================
# 工具函数
# ============================================================================

def load_cases(cases_path: str) -> List[Dict[str, Any]]:
    """加载评估用例"""
    with open(cases_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(mode: str = "mock",
                   cases_path: Optional[str] = None,
                   category_filter: Optional[str] = None,
                   db_type: str = "sqlite",
                   db_config: Optional[Dict[str, Any]] = None,
                   memory_db_path: Optional[str] = None,
                   cross_db: bool = False) -> Dict[str, Any]:
    """运行完整评估流程

    Args:
        mode: "mock" 或 "live"
        cases_path: 用例文件路径
        category_filter: 可选的分类过滤
        db_type: 数据库类型 ("sqlite" / "mysql" / "postgresql" / "all")
        db_config: 默认数据库配置（用于公司用例）
        memory_db_path: 长期记忆数据库路径
        cross_db: 是否启用跨库一致性测试

    Returns:
        完整统计数据
    """
    if cases_path is None:
        v2_path = Path(__file__).parent / "eval_cases_v2.json"
        v1_path = Path(__file__).parent / "eval_cases.json"
        cases_path = str(v2_path) if v2_path.exists() else str(v1_path)

    cases = load_cases(cases_path)

    if cross_db or db_type == "all":
        db_types = ["sqlite", "mysql", "postgresql"]
        runner = CrossDBEvalRunner(
            mode=mode,
            db_types=db_types,
            memory_db_path=memory_db_path,
        )
        return runner.run(cases, category_filter=category_filter)
    else:
        runner = EvalRunner(
            mode=mode,
            db_type=db_type,
            db_config=db_config,
            memory_db_path=memory_db_path,
        )
        results = runner.run(cases, category_filter=category_filter)
        cross_results = [r for r in results if r.get("category") == "cross_db_consistency"]
        stats = generate_stats(results, cross_db_results=cross_results)
        return stats
