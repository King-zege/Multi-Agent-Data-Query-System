"""
评估框架 CLI 入口

用法:
    # Mock 模式（不消耗 API，快速验证）
    python -m eval.run_eval --mode mock

    # Mock 模式 + 跨库一致性测试
    python -m eval.run_eval --mode mock --cross-db

    # 真实模式（调用 LLM，完整评估）
    python -m eval.run_eval --mode live

    # 指定数据库类型
    python -m eval.run_eval --mode live --db-type mysql

    # 全库模式（sqlite + mysql + postgresql）
    python -m eval.run_eval --mode live --db-type all

    # 只跑特定类别
    python -m eval.run_eval --mode live --category sql_execution_accuracy

    # 指定用例文件
    python -m eval.run_eval --mode mock --cases eval/eval_cases_v2.json

    # 指定输出
    python -m eval.run_eval --mode live --output eval/reports/my_report.json

    # 旧版兼容（仍支持 --db-path）
    python -m eval.run_eval --mode mock --db-path data/company.db
"""

import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

VALID_DB_TYPES = ["sqlite", "mysql", "postgresql", "all"]
VALID_CATEGORIES = [
    "intent_recognition", "sql_generation", "end_to_end",
    "error_handling", "sql_execution_accuracy",
    "result_exact_match", "result_set_overlap",
    "cross_db_consistency",
]


def main():
    parser = argparse.ArgumentParser(
        description="Agent 评估框架 — 自动化测试系统回答质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m eval.run_eval --mode mock
  python -m eval.run_eval --mode mock --cross-db
  python -m eval.run_eval --mode live --db-type mysql
  python -m eval.run_eval --mode live --db-type all
  python -m eval.run_eval --mode live --category sql_execution_accuracy
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
        choices=VALID_CATEGORIES,
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
        help="用例文件路径（默认: eval/eval_cases_v2.json 或 eval/eval_cases.json）",
    )
    parser.add_argument(
        "--db-type", "-t",
        choices=VALID_DB_TYPES,
        default="sqlite",
        help="数据库类型: sqlite / mysql / postgresql / all（默认: sqlite）",
    )
    parser.add_argument(
        "--cross-db",
        action="store_true",
        default=False,
        help="启用跨库一致性测试（对 cross_db_consistency 用例在所有 db_type 上执行）",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="[已弃用] SQLite 数据库路径，请使用 --db-type sqlite",
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
        v2_path = str(Path(__file__).parent / "eval_cases_v2.json")
        v1_path = str(Path(__file__).parent / "eval_cases.json")
        if os.path.exists(v2_path):
            cases_path = v2_path
        elif os.path.exists(v1_path):
            cases_path = v1_path
        else:
            print(f"Error: No eval cases file found.", file=sys.stderr)
            print(f"  Tried: {v2_path}", file=sys.stderr)
            print(f"  Tried: {v1_path}", file=sys.stderr)
            sys.exit(1)

    print(f"\nRunning evaluation in [{args.mode}] mode...")
    print(f"Cases: {cases_path}")
    if args.category:
        print(f"Category filter: {args.category}")
    print(f"DB type: {args.db_type}")
    if args.cross_db:
        print(f"Cross-DB mode: enabled")

    # 构建 db_config（用于公司用例的默认配置）
    db_config = None
    if args.db_type == "sqlite":
        if args.db_path:
            db_config = {"type": "sqlite", "path": args.db_path}
        else:
            db_config = None  # EvalRunner will create a temp SQLite
    elif args.db_path:
        print("Warning: --db-path is ignored when --db-type is not 'sqlite'", file=sys.stderr)

    # 运行评估
    cross_db = args.cross_db or args.db_type == "all"
    stats = run_evaluation(
        mode=args.mode,
        cases_path=cases_path,
        category_filter=args.category,
        db_type=args.db_type if not cross_db else "sqlite",
        db_config=db_config,
        memory_db_path=args.memory_db_path,
        cross_db=cross_db,
    )

    # 生成报告
    generate_report(stats, mode=args.mode, output_path=args.output)

    # 返回退出码
    if stats.get("failed", 0) > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
