"""
报告生成器

支持两种输出格式：
- 终端彩色表格（Rich）
- JSON 文件
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


import sys
import os

# 强制 UTF-8 输出（Windows 兼容）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True, legacy_windows=False)


def _format_pct(value: float) -> str:
    """格式化百分比"""
    return f"{value * 100:.1f}%"


def _format_score(value: float) -> str:
    """格式化分数"""
    return f"{value:.1f}"


def print_header(mode: str, model: str = "EvalMockLLM / qwen-plus"):
    """打印报告头部"""
    console.print()
    title = Panel(
        f"[bold cyan]Agent Evaluation Report[/bold cyan]\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    "
        f"Model: {model}\n"
        f"Mode: {mode}",
        box=box.DOUBLE,
    )
    console.print(title)


def print_category_table(stats: Dict[str, Any]):
    """打印分类统计表格"""
    table = Table(title="Results by Category", box=box.ROUNDED)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right", style="white")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Details")

    by_category = stats.get("by_category", {})

    for cat_name in ["intent_recognition", "sql_generation", "end_to_end", "error_handling"]:
        cat = by_category.get(cat_name)
        if cat is None:
            continue

        total = cat["total"]
        passed = cat["passed"]
        accuracy = cat["accuracy"]

        # 颜色
        if accuracy >= 0.9:
            acc_style = "green"
        elif accuracy >= 0.7:
            acc_style = "yellow"
        else:
            acc_style = "red"

        # 平均分（仅 end_to_end 有）
        if cat_name == "end_to_end":
            llm_judge = stats.get("llm_judge", {})
            avg_score = f"{llm_judge.get('overall_avg', 0):.1f}"
            details = "LLM Judge avg"
        else:
            avg_score = "-"
            details = f"{passed}/{total} passed"

        table.add_row(
            cat_name,
            str(total),
            f"[{acc_style}]{_format_pct(accuracy)}[/{acc_style}]",
            avg_score,
            details,
        )

    console.print(table)


def print_overall(stats: Dict[str, Any]):
    """打印总体统计"""
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    accuracy = stats["accuracy"]

    if accuracy >= 0.9:
        acc_color = "green"
    elif accuracy >= 0.7:
        acc_color = "yellow"
    else:
        acc_color = "red"

    text = Text()
    text.append("OVERALL  ", style="bold")
    text.append(f"{total} cases    ", style="white")
    text.append(f"[{acc_color}]{_format_pct(accuracy)}[/{acc_color}]    ", style="bold")

    llm_judge = stats.get("llm_judge", {})
    if llm_judge.get("overall_avg"):
        text.append(f"Avg Score: {llm_judge['overall_avg']:.1f}", style="white")

    text.append(f"    {passed}/{total} passed", style="white")
    console.print(text)


def print_failed_cases(stats: Dict[str, Any]):
    """打印失败用例详情"""
    failed = stats.get("failed_cases", [])
    if not failed:
        console.print("\n[green]All cases passed![/green]")
        return

    console.print(f"\n[bold red]Failed cases ({len(failed)}):[/bold red]")
    for case in failed:
        case_id = case["id"]
        category = case.get("category", "?")
        question = case.get("question", "?")
        error = case.get("error", "")

        console.print(f"  [red]X[/red] {case_id} [[{category}]]: \"{question}\"")
        if error:
            console.print(f"    → [dim]{error}[/dim]")

        # SQL generation specifics
        if "missed_keywords" in case and case["missed_keywords"]:
            console.print(f"    → Missing keywords: {', '.join(case['missed_keywords'])}")
        if "forbidden_found" in case and case["forbidden_found"]:
            console.print(f"    → Forbidden keywords found: {', '.join(case['forbidden_found'])}")
        if "sql" in case:
            console.print(f"    → SQL: [dim]{case['sql'][:120]}[/dim]")

        # Intent specifics
        if "expected" in case and "actual" in case:
            console.print(f"    → Expected: {case['expected']}, Got: {case['actual']}")

        if "judge_scores" in case:
            scores = case["judge_scores"]
            console.print(f"    → Judge: acc={scores.get('accuracy')}, "
                         f"comp={scores.get('completeness')}, "
                         f"read={scores.get('readability')}")


def save_json_report(stats: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """保存 JSON 报告

    Args:
        stats: 统计数据
        output_path: 输出路径，None 则自动生成

    Returns:
        保存的文件路径
    """
    if output_path is None:
        reports_dir = Path(__file__).parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(reports_dir / f"eval_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    return output_path


def generate_report(stats: Dict[str, Any],
                    mode: str = "mock",
                    output_path: Optional[str] = None):
    """生成完整报告（终端 + JSON）"""
    print_header(mode)
    print_category_table(stats)
    print_overall(stats)
    print_failed_cases(stats)

    # 保存 JSON
    json_path = save_json_report(stats, output_path)
    console.print(f"\n[dim]JSON report saved to: {json_path}[/dim]")
