"""
评估指标计算

支持的指标：
- accuracy: 准确率（pass / total）
- keyword_match: 关键词匹配率
- execution_accuracy: SQL 执行成功率
- fix_success_rate: SQL 修复成功率
- result_set_comparison: 结果集精确匹配 / Jaccard 相似度
- schema_linking_accuracy: Schema 链接准确率
- cross_db_consistency: 跨库一致性
- llm_judge_score: LLM-as-Judge 平均分
- category_stats: 分类统计
"""

import hashlib
import re
from typing import Dict, List, Any, Optional


# ======================== 基础指标 ========================

def calculate_accuracy(results: List[Dict[str, Any]]) -> float:
    """计算整体准确率"""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed", False))
    return passed / len(results)


def calculate_keyword_match(expected: List[str], actual: str) -> float:
    """计算关键词匹配率"""
    if not expected:
        return 1.0
    actual_lower = actual.lower()
    matched = sum(1 for kw in expected if kw.lower() in actual_lower)
    return matched / len(expected)


def calculate_category_accuracy(results: List[Dict[str, Any]],
                                 category: str) -> float:
    """计算特定分类的准确率"""
    cat_results = [r for r in results if r.get("category") == category]
    return calculate_accuracy(cat_results)


def aggregate_llm_judge_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """聚合 LLM-as-Judge 评分"""
    judged = [r for r in results if "judge_scores" in r]
    if not judged:
        return {}

    total = len(judged)
    accuracy_sum = sum(r["judge_scores"].get("accuracy", 0) for r in judged)
    completeness_sum = sum(r["judge_scores"].get("completeness", 0) for r in judged)
    readability_sum = sum(r["judge_scores"].get("readability", 0) for r in judged)

    return {
        "accuracy_avg": accuracy_sum / total,
        "completeness_avg": completeness_sum / total,
        "readability_avg": readability_sum / total,
        "overall_avg": (accuracy_sum + completeness_sum + readability_sum) / (total * 3),
        "count": total,
    }


# ======================== SQL 执行指标 ========================

def calculate_execution_accuracy(results: List[Dict[str, Any]]) -> float:
    """SQL 执行准确率：生成 SQL 无错误执行的比例"""
    exec_results = [r for r in results if "sql_executed" in r]
    if not exec_results:
        return 0.0
    return sum(1 for r in exec_results if r["sql_executed"]) / len(exec_results)


def calculate_fix_success_rate(results: List[Dict[str, Any]]) -> float:
    """SQL 修复成功率：首次失败后经过纠错成功的比例"""
    retried = [r for r in results if r.get("retry_count", 0) > 0]
    if not retried:
        return 1.0
    return sum(1 for r in retried if r.get("sql_executed", False)) / len(retried)


def calculate_avg_retry_count(results: List[Dict[str, Any]]) -> float:
    """平均重试次数"""
    exec_results = [r for r in results if "retry_count" in r]
    if not exec_results:
        return 0.0
    return sum(r.get("retry_count", 0) for r in exec_results) / len(exec_results)


# ======================== 结果集比较指标 ========================

def _row_hash(row: Any) -> str:
    """将行数据哈希化为可比较的字符串"""
    if isinstance(row, dict):
        items = tuple(sorted(str(v) for v in row.values()))
    elif isinstance(row, (list, tuple)):
        items = tuple(str(v) for v in row)
    else:
        items = (str(row),)
    return hashlib.md5(str(items).encode()).hexdigest()


def _normalize_rows(rows: list) -> list:
    """标准化行数据，统一类型"""
    normalized = []
    for row in rows:
        if isinstance(row, dict):
            norm = tuple(str(row.get(k, "")).strip() for k in sorted(row.keys()))
        elif isinstance(row, (list, tuple)):
            norm = tuple(str(v).strip() for v in row)
        else:
            norm = (str(row).strip(),)
        normalized.append(norm)
    return normalized


def compare_result_sets(actual_rows: list, expected_rows: list,
                         order_matters: bool = False) -> dict:
    """比较两个结果集

    Args:
        actual_rows: 实际查询结果行列表
        expected_rows: 标准答案行列表
        order_matters: 是否考虑顺序

    Returns:
        {
            "is_match": bool,
            "actual_count": int,
            "expected_count": int,
            "jaccard": float,
            "precision": float,
            "recall": float,
        }
    """
    if not actual_rows and not expected_rows:
        return {"is_match": True, "actual_count": 0, "expected_count": 0,
                "jaccard": 1.0, "precision": 1.0, "recall": 1.0}

    actual_norm = _normalize_rows(actual_rows)
    expected_norm = _normalize_rows(expected_rows)

    if order_matters:
        is_match = actual_norm == expected_norm
    else:
        is_match = set(actual_norm) == set(expected_norm)

    set_a = set(actual_norm)
    set_b = set(expected_norm)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0.0

    precision = intersection / len(set_a) if set_a else 0.0
    recall = intersection / len(set_b) if set_b else 0.0

    return {
        "is_match": is_match,
        "actual_count": len(actual_rows),
        "expected_count": len(expected_rows),
        "jaccard": round(jaccard, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def jaccard_similarity(actual_rows: list, expected_rows: list) -> float:
    """计算 Jaccard 相似度"""
    comp = compare_result_sets(actual_rows, expected_rows, order_matters=False)
    return comp["jaccard"]


def calculate_exact_match_rate(results: List[Dict[str, Any]]) -> float:
    """结果精确匹配率"""
    match_results = [r for r in results if "result_comparison" in r]
    if not match_results:
        return 0.0
    return sum(1 for r in match_results
               if r["result_comparison"].get("is_match", False)) / len(match_results)


def calculate_avg_jaccard(results: List[Dict[str, Any]]) -> float:
    """平均 Jaccard 相似度"""
    match_results = [r for r in results if "result_comparison" in r]
    if not match_results:
        return 0.0
    return sum(r["result_comparison"].get("jaccard", 0) for r in match_results) / len(match_results)


# ======================== Schema 链接指标 ========================

def parse_sql_tables(sql: str) -> set:
    """从 SQL 中提取表名"""
    if not sql:
        return set()
    tables = set()
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE):
        tables.add(match.group(1).lower())
    return tables


def calculate_schema_linking_accuracy(results: List[Dict[str, Any]]) -> float:
    """Schema 链接准确率：生成的 SQL 引用了正确表名的比例"""
    linkable = [r for r in results
                if r.get("sql") and r.get("required_tables")]
    if not linkable:
        return 0.0

    correct = 0
    for r in linkable:
        sql_tables = parse_sql_tables(r["sql"])
        required = {t.lower() for t in r["required_tables"]}
        # At minimum all required tables should be present
        # (extra tables are sometimes fine for complex queries)
        if required.issubset(sql_tables):
            correct += 1

    return correct / len(linkable)


# ======================== 跨库一致性指标 ========================

def calculate_cross_db_consistency(results: List[Dict[str, Any]]) -> dict:
    """计算跨库一致性：同一问题在不同数据库上的结果等价性

    将结果按 question_id 分组，比较每组内不同 db_type 的结果一致性。
    """
    cross_results = [r for r in results
                     if r.get("category") == "cross_db_consistency"]

    if not cross_results:
        return {"consistency_rate": 1.0, "total_groups": 0, "consistent_groups": 0,
                "groups": []}

    # Group by question
    groups = {}
    for r in cross_results:
        qid = r.get("id", r.get("question", ""))
        if qid not in groups:
            groups[qid] = []
        groups[qid].append(r)

    total_groups = len(groups)
    consistent_groups = 0
    group_details = []

    for qid, group_results in groups.items():
        db_results = {}
        for r in group_results:
            db_type = r.get("db_type", "unknown")
            db_results[db_type] = {
                "sql_executed": r.get("sql_executed", False),
                "row_count": r.get("result_row_count"),
            }

        # Consistent if all DB types executed successfully and have same characteristics
        executed = [v["sql_executed"] for v in db_results.values()]
        all_executed = all(executed)

        if all_executed:
            counts = [v["row_count"] for v in db_results.values() if v["row_count"] is not None]
            all_same_count = len(set(counts)) <= 1 if counts else True
        else:
            all_same_count = False

        is_consistent = all_executed and all_same_count
        if is_consistent:
            consistent_groups += 1

        group_details.append({
            "question_id": qid,
            "db_results": {k: v for k, v in db_results.items()},
            "all_executed": all_executed,
            "all_same_count": all_same_count,
            "consistent": is_consistent,
        })

    return {
        "consistency_rate": consistent_groups / total_groups if total_groups > 0 else 1.0,
        "total_groups": total_groups,
        "consistent_groups": consistent_groups,
        "groups": group_details,
    }


# ======================== 分库统计 ========================

def calculate_db_type_breakdown(results: List[Dict[str, Any]]) -> Dict[str, dict]:
    """按数据库类型分组统计"""
    breakdown = {}
    for db_type in ["sqlite", "mysql", "postgresql"]:
        db_results = [r for r in results if r.get("db_type") == db_type]
        if not db_results:
            continue

        total = len(db_results)
        passed = sum(1 for r in db_results if r.get("passed", False))

        # Execution accuracy (for cases that have sql_executed)
        exec_results = [r for r in db_results if "sql_executed" in r]
        exec_acc = (sum(1 for r in exec_results if r["sql_executed"]) / len(exec_results)
                     if exec_results else 0.0)

        # Avg Jaccard
        match_results = [r for r in db_results if "result_comparison" in r]
        avg_jaccard = (sum(r["result_comparison"].get("jaccard", 0) for r in match_results) / len(match_results)
                       if match_results else 0.0)

        breakdown[db_type] = {
            "total": total,
            "passed": passed,
            "accuracy": passed / total if total > 0 else 0.0,
            "execution_accuracy": exec_acc,
            "avg_jaccard": round(avg_jaccard, 4),
        }
    return breakdown


# ======================== 主统计函数 ========================

def generate_stats(results: List[Dict[str, Any]],
                   cross_db_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """生成完整的统计报告

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "accuracy": float,
            "by_category": {category: {total, passed, accuracy}},
            # 新增指标
            "execution_accuracy": float,
            "fix_success_rate": float,
            "avg_retry_count": float,
            "exact_match_rate": float,
            "avg_jaccard": float,
            "avg_precision": float,
            "avg_recall": float,
            "schema_linking_accuracy": float,
            "cross_db_consistency": dict,
            "by_db_type": dict,
            # 原有
            "llm_judge": dict,
            "avg_elapsed_seconds": float,
            "failed_cases": list,
        }
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed

    # 按分类统计
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.get("passed", False):
            categories[cat]["passed"] += 1

    for cat in categories:
        categories[cat]["accuracy"] = (
            categories[cat]["passed"] / categories[cat]["total"]
            if categories[cat]["total"] > 0 else 0.0
        )

    # 新增指标
    exec_accuracy = calculate_execution_accuracy(results)
    fix_rate = calculate_fix_success_rate(results)
    avg_retry = calculate_avg_retry_count(results)
    match_rate = calculate_exact_match_rate(results)
    avg_jaccard = calculate_avg_jaccard(results)
    schema_acc = calculate_schema_linking_accuracy(results)

    # 精度和召回率均值
    match_results = [r for r in results if "result_comparison" in r]
    avg_precision = (
        sum(r["result_comparison"].get("precision", 0) for r in match_results) / len(match_results)
        if match_results else 0.0
    )
    avg_recall = (
        sum(r["result_comparison"].get("recall", 0) for r in match_results) / len(match_results)
        if match_results else 0.0
    )

    # 跨库一致性
    if cross_db_results:
        cross_db = calculate_cross_db_consistency(cross_db_results)
    else:
        cross_db = calculate_cross_db_consistency(results)

    # 分库统计
    db_breakdown = calculate_db_type_breakdown(results)

    # LLM Judge
    llm_judge = aggregate_llm_judge_scores(results)

    # 平均耗时
    elapsed_times = [r.get("elapsed_seconds", 0) for r in results if "elapsed_seconds" in r]
    avg_elapsed = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0.0

    # 失败用例
    failed_cases = [r for r in results if not r.get("passed", False)]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": passed / total if total > 0 else 0.0,
        "by_category": categories,
        # 新增
        "execution_accuracy": round(exec_accuracy, 4),
        "fix_success_rate": round(fix_rate, 4),
        "avg_retry_count": round(avg_retry, 2),
        "exact_match_rate": round(match_rate, 4),
        "avg_jaccard": round(avg_jaccard, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "schema_linking_accuracy": round(schema_acc, 4),
        "cross_db_consistency": cross_db,
        "by_db_type": db_breakdown,
        # 原有
        "llm_judge": llm_judge,
        "avg_elapsed_seconds": round(avg_elapsed, 2),
        "failed_cases": failed_cases,
    }
