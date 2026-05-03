"""
评估指标计算

支持的指标：
- accuracy: 准确率（pass / total）
- keyword_match: 关键词匹配率
- llm_judge_score: LLM-as-Judge 平均分
- category_stats: 分类统计
"""

from typing import Dict, List, Any


def calculate_accuracy(results: List[Dict[str, Any]]) -> float:
    """计算整体准确率"""
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.get("passed", False))
    return passed / len(results)


def calculate_keyword_match(expected: List[str], actual: str) -> float:
    """计算关键词匹配率

    Args:
        expected: 期望出现的关键词列表
        actual: 实际文本

    Returns:
        匹配率 (0.0 ~ 1.0)
    """
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


def generate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成完整的统计报告

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "accuracy": float,
            "by_category": {category: {"total": int, "passed": int, "accuracy": float}},
            "llm_judge": Dict (only if applicable),
            "failed_cases": List[Dict],
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

    # LLM Judge 聚合
    llm_judge = aggregate_llm_judge_scores(results)

    # 失败用例
    failed_cases = [r for r in results if not r.get("passed", False)]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": passed / total if total > 0 else 0.0,
        "by_category": categories,
        "llm_judge": llm_judge,
        "failed_cases": failed_cases,
    }
