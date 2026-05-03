"""
评估框架 CLI 入口

用法:
    # Mock 模式（不消耗 API，快速验证）
    python -m eval.run_eval --mode mock

    # 真实模式（调用 LLM，完整评估）
    python -m eval.run_eval --mode live

    # 只跑特定类别
    python -m eval.run_eval --mode live --category intent_recognition

    # 指定输出
    python -m eval.run_eval --mode live --output eval/reports/my_report.json

    # 指定数据库路径（跳过自动创建）
    python -m eval.run_eval --mode live --db-path data/company.db --memory-db-path data/long_term_memory.db
"""

import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Agent 评估框架 — 自动化测试系统回答质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m eval.run_eval --mode mock
  python -m eval.run_eval --mode live --category sql_generation
  python -m eval.run_eval --mode live --output my_report.json
        """,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["mock", "live"],
        default="mock",
        help="评估模式: mock (不消耗API) / live (真实LLM)",
    )
    parser.add_argument(
        "--category", "-c",
        choices=["intent_recognition", "sql_generation", "end_to_end", "error_handling"],
        default=None,
        help="只评估指定分类",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="JSON 报告输出路径",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="eval_cases.json 路径（默认: eval/eval_cases.json）",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="业务数据库路径（不指定则自动创建临时数据库）",
    )
    parser.add_argument(
        "--memory-db-path",
        default=None,
        help="长期记忆数据库路径（不指定则自动创建临时数据库）",
    )

    args = parser.parse_args()

    # 导入评估模块
    from eval.evaluator import run_evaluation
    from eval.reporter import generate_report

    # 确定 cases 路径
    cases_path = args.cases
    if cases_path is None:
        cases_path = str(Path(__file__).parent / "eval_cases.json")

    if not os.path.exists(cases_path):
        print(f"Error: eval_cases.json not found at {cases_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning evaluation in [{args.mode}] mode...")
    if args.category:
        print(f"Category filter: {args.category}")
    print(f"Cases: {cases_path}")

    # 构建 db_config（兼容旧的 --db-path 参数）
    db_config = None
    if args.db_path:
        db_config = {"type": "sqlite", "path": args.db_path}

    # 运行评估
    stats = run_evaluation(
        mode=args.mode,
        cases_path=cases_path,
        category_filter=args.category,
        db_config=db_config,
        memory_db_path=args.memory_db_path,
    )

    # 生成报告
    generate_report(stats, mode=args.mode, output_path=args.output)

    # 返回退出码
    if stats.get("failed", 0) > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
