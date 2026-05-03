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
import os as _os

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


def _style_for_pct(value: float) -> str:
    """根据百分比返回颜色样式"""
    if value >= 0.9:
        return "green"
    elif value >= 0.7:
        return "yellow"
    else:
        return "red"


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

    # 所有已知分类
    all_categories = [
        "intent_recognition",
        "sql_generation",
        "end_to_end",
        "error_handling",
        "sql_execution_accuracy",
        "result_exact_match",
        "result_set_overlap",
        "cross_db_consistency",
    ]

    for cat_name in all_categories:
        cat = by_category.get(cat_name)
        if cat is None:
            continue

        total = cat["total"]
        passed = cat["passed"]
        accuracy = cat["accuracy"]
        acc_style = _style_for_pct(accuracy)

        # 平均分
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


def print_metrics_table(stats: Dict[str, Any]):
    """打印量化指标表格"""
    table = Table(title="Quantitative Metrics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right")
    table.add_column("Description")

    metrics = [
        ("Accuracy", _format_pct(stats.get("accuracy", 0)),
         "Overall pass rate"),
        ("SQL Execution Accuracy", _format_pct(stats.get("execution_accuracy", 0)),
         "Generated SQL executed without error"),
        ("Fix Success Rate", _format_pct(stats.get("fix_success_rate", 0)),
         "Failed SQL recovered after retry"),
        ("Exact Match Rate", _format_pct(stats.get("exact_match_rate", 0)),
         "Result sets match ground truth exactly"),
        ("Avg Jaccard", f"{stats.get('avg_jaccard', 0):.3f}",
         "Average Jaccard similarity (intersection / union)"),
        ("Avg Precision", f"{stats.get('avg_precision', 0):.3f}",
         "Average precision of result rows"),
        ("Avg Recall", f"{stats.get('avg_recall', 0):.3f}",
         "Average recall of result rows"),
        ("Schema Linking Accuracy", _format_pct(stats.get("schema_linking_accuracy", 0)),
         "Correct tables referenced in generated SQL"),
        ("Avg Retry Count", f"{stats.get('avg_retry_count', 0):.2f}",
         "Average SQL retries per case"),
        ("Avg Elapsed (s)", f"{stats.get('avg_elapsed_seconds', 0):.2f}",
         "Average end-to-end query time"),
    ]

    # LLM Judge (if available)
    llm_judge = stats.get("llm_judge", {})
    if llm_judge.get("count", 0) > 0:
        metrics.append(("LLM Judge — Accuracy", f"{llm_judge.get('accuracy_avg', 0):.1f}/5",
                        "LLM-assessed answer accuracy"))
        metrics.append(("LLM Judge — Completeness", f"{llm_judge.get('completeness_avg', 0):.1f}/5",
                        "LLM-assessed answer completeness"))
        metrics.append(("LLM Judge — Readability", f"{llm_judge.get('readability_avg', 0):.1f}/5",
                        "LLM-assessed answer readability"))
        metrics.append(("LLM Judge — Overall", f"{llm_judge.get('overall_avg', 0):.1f}/5",
                        f"LLM Judge average ({llm_judge['count']} cases)"))

    # Cross-DB consistency
    cross_db = stats.get("cross_db_consistency", {})
    if cross_db.get("total_groups", 0) > 0:
        table.add_section()
        table.add_row(
            "[bold]Cross-DB Consistency[/bold]",
            _format_pct(cross_db.get("consistency_rate", 0)),
            f"{cross_db['consistent_groups']}/{cross_db['total_groups']} groups consistent",
            style="bold",
        )

    for metric_name, value, desc in metrics:
        table.add_row(metric_name, value, desc)

    console.print(table)


def print_db_breakdown(stats: Dict[str, Any]):
    """打印分库统计表格"""
    db_breakdown = stats.get("by_db_type", {})
    if not db_breakdown:
        return

    table = Table(title="Per Database Type Breakdown", box=box.ROUNDED)
    table.add_column("DB Type", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Exec Accuracy", justify="right")
    table.add_column("Avg Jaccard", justify="right")

    for db_type, info in sorted(db_breakdown.items()):
        acc_style = _style_for_pct(info.get("accuracy", 0))
        exec_style = _style_for_pct(info.get("execution_accuracy", 0))
        table.add_row(
            db_type,
            str(info.get("total", 0)),
            f"[{acc_style}]{_format_pct(info.get('accuracy', 0))}[/{acc_style}]",
            f"[{exec_style}]{_format_pct(info.get('execution_accuracy', 0))}[/{exec_style}]",
            f"{info.get('avg_jaccard', 0):.3f}",
        )

    console.print(table)


def print_cross_db_details(stats: Dict[str, Any]):
    """打印跨库一致性组详情"""
    cross_db = stats.get("cross_db_consistency", {})
    groups = cross_db.get("groups", [])
    if not groups:
        return

    inconsistent = [g for g in groups if not g.get("consistent", False)]
    if not inconsistent:
        console.print("\n[green]All cross-DB groups consistent![/green]")
        return

    console.print(f"\n[bold red]Inconsistent groups ({len(inconsistent)}/{len(groups)}):[/bold red]")
    for g in inconsistent:
        qid = g.get("question_id", "?")
        db_results = g.get("db_results", {})
        all_exec = g.get("all_executed", False)
        same_count = g.get("all_same_count", False)
        issues = []
        if not all_exec:
            issues.append("not all executed")
        if not same_count:
            issues.append("row counts differ")
        detail = ", ".join(
            f"{db}: {'OK' if info.get('sql_executed') else 'FAIL'} (rows={info.get('row_count')})"
            for db, info in db_results.items()
        )
        console.print(f"  [red]X[/red] {qid}: {', '.join(issues)} → {detail}")


def print_overall(stats: Dict[str, Any]):
    """打印总体统计"""
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    accuracy = stats["accuracy"]

    acc_color = _style_for_pct(accuracy)

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
        db_type = case.get("db_type", "")
        execution_error = case.get("execution_error", "")

        db_tag = f" [dim]({db_type})[/dim]" if db_type else ""
        console.print(f"  [red]X[/red] {case_id}{db_tag} [[{category}]]: \"{question[:80]}\"")

        if error:
            console.print(f"    → [dim]{error}[/dim]")
        if execution_error and execution_error != error:
            console.print(f"    → SQL Error: [dim]{execution_error[:120]}[/dim]")

        # SQL generation specifics
        if "missed_keywords" in case and case["missed_keywords"]:
            console.print(f"    → Missing keywords: {', '.join(case['missed_keywords'])}")
        if "forbidden_found" in case and case["forbidden_found"]:
            console.print(f"    → Forbidden keywords found: {', '.join(case['forbidden_found'])}")
        if "sql" in case and case["sql"]:
            console.print(f"    → SQL: [dim]{case['sql'][:120]}[/dim]")

        # Intent specifics
        if "expected" in case and "actual" in case:
            console.print(f"    → Expected: {case['expected']}, Got: {case['actual']}")

        # Result comparison specifics
        if "result_comparison" in case:
            comp = case["result_comparison"]
            console.print(f"    → Jaccard: {comp.get('jaccard', 'N/A')}, "
                         f"Precision: {comp.get('precision', 'N/A')}, "
                         f"Recall: {comp.get('recall', 'N/A')}")
            console.print(f"    → Actual rows: {comp.get('actual_count', 0)}, "
                         f"Expected rows: {comp.get('expected_count', 0)}")

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
    print_metrics_table(stats)
    print_db_breakdown(stats)
    print_cross_db_details(stats)
    print_failed_cases(stats)

    # 保存 JSON
    json_path = save_json_report(stats, output_path)
    console.print(f"\n[dim]JSON report saved to: {json_path}[/dim]")
