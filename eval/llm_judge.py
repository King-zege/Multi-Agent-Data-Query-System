"""
LLM-as-Judge — 使用 LLM 对端到端回答进行多维度评分

评分维度：
- accuracy (1-5): 回答是否准确
- completeness (1-5): 是否完整回答了问题
- readability (1-5): 结构是否清晰、易读
"""

import json
import re
from typing import Dict, Any, Optional


JUDGE_PROMPT_TEMPLATE = """你是一个Agent回答质量评估专家。请对以下回答打分：

用户问题：{question}
Agent回答：{answer}
评估标准：{judge_criteria}

请从以下维度打分（1-5分）：
1. accuracy: 回答是否准确、数据是否正确
2. completeness: 是否完整回答了问题，是否遗漏关键信息
3. readability: 结构是否清晰、易于理解

返回纯JSON格式（不要代码块标记）：
{{"accuracy": 4, "completeness": 3, "readability": 5, "summary": "一句话总结优缺点"}}"""


def build_judge_prompt(question: str, answer: str, judge_criteria: Dict[str, str]) -> str:
    """构建 LLM Judge 评估 prompt

    Args:
        question: 用户问题
        answer: Agent 回答
        judge_criteria: 评分标准，如 {"accuracy": "...", "completeness": "...", "readability": "..."}

    Returns:
        完整的 judge prompt
    """
    criteria_text = "\n".join(f"- {k}: {v}" for k, v in judge_criteria.items())
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        judge_criteria=criteria_text
    )


def parse_judge_response(response: str) -> Optional[Dict[str, Any]]:
    """解析 LLM Judge 的返回结果

    Args:
        response: LLM 返回的文本

    Returns:
        {"accuracy": int, "completeness": int, "readability": int, "summary": str}
        解析失败返回 None
    """
    try:
        # 尝试直接解析
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试从代码块中提取
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试从文本中提取 JSON 对象
    match = re.search(r'\{[\s\S]*"accuracy"[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def mock_judge(question: str, answer: str, judge_criteria: Dict[str, str]) -> Dict[str, Any]:
    """Mock Judge — 基于关键词匹配的简单评分（用于 mock 模式）

    不调用真实 LLM，仅基于 expected_answer_keywords 进行关键词匹配评分。
    """
    keywords = []
    for criteria_text in judge_criteria.values():
        # 从评估标准中提取关键概念词
        import re
        words = re.findall(r'[\u4e00-\u9fff\w]+', criteria_text)
        keywords.extend(w for w in words if len(w) >= 2)

    # 去重
    keywords = list(set(keywords))

    if not keywords:
        return {"accuracy": 3, "completeness": 3, "readability": 3,
                "summary": "无关键词可评估"}

    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    match_rate = matched / len(keywords)

    # 映射到 1-5 分
    score = max(1, min(5, round(match_rate * 5)))

    return {
        "accuracy": score,
        "completeness": max(1, score - 1),
        "readability": min(5, score + 1),
        "summary": f"关键词匹配率: {match_rate:.0%} ({matched}/{len(keywords)})"
    }


def llm_judge(llm, question: str, answer: str, judge_criteria: Dict[str, str]) -> Dict[str, Any]:
    """使用真实 LLM 进行评分

    Args:
        llm: 语言模型实例
        question: 用户问题
        answer: Agent 回答
        judge_criteria: 评分标准

    Returns:
        {"accuracy": int, "completeness": int, "readability": int, "summary": str}
    """
    prompt = build_judge_prompt(question, answer, judge_criteria)

    try:
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            text = response.content
        elif hasattr(response, 'text'):
            text = response.text
        else:
            text = str(response)

        result = parse_judge_response(text)
        if result:
            return result
    except Exception:
        pass

    # 降级为 mock judge
    return mock_judge(question, answer, judge_criteria)
