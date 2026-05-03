"""
Generate expanded eval cases from Falcon dataset + original company cases.

Reads Falcon dev.json, extracts questions for DBs 14/16/17/20,
classifies by SQL complexity, assigns categories, and outputs
eval/eval_cases_v2.json with ~100 cases.

Usage:
    python scripts/generate_eval_cases.py
    python scripts/generate_eval_cases.py --output eval/eval_cases_v2.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Falcon DBs to include
TARGET_DB_IDS = {"14", "16", "17", "20"}

# SQL clause keywords to extract
SQL_CLAUSE_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN",
    "RIGHT JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
    "UNION", "WITH", "DISTINCT", "BETWEEN", "LIKE", "IN",
]
AGG_FUNCTIONS = ["COUNT", "SUM", "AVG", "MAX", "MIN"]


def parse_sql_tables(sql):
    """Extract table names from SQL using simple regex."""
    tables = set()
    # Match FROM/JOIN clauses: FROM table, JOIN table, FROM table alias
    patterns = [
        r'\bFROM\s+(\w+)',
        r'\bJOIN\s+(\w+)',
    ]
    for pat in patterns:
        for match in re.finditer(pat, sql, re.IGNORECASE):
            tables.add(match.group(1).lower())
    return sorted(tables)


def extract_sql_keywords(sql):
    """Extract SQL features as keywords for evaluation."""
    sql_upper = sql.upper()
    keywords = []

    # SQL clauses
    for kw in SQL_CLAUSE_KEYWORDS:
        if re.search(r'\b' + kw.replace(' ', r'\s+') + r'\b', sql_upper):
            keywords.append(kw)

    # Aggregate functions
    for fn in AGG_FUNCTIONS:
        if re.search(r'\b' + fn + r'\s*\(', sql_upper):
            keywords.append(fn)

    # Table names
    tables = parse_sql_tables(sql)
    keywords.extend(tables)

    return keywords


def count_joins(sql):
    """Count the number of JOIN clauses in SQL."""
    return len(re.findall(r'\b(?:INNER\s+)?(?:LEFT\s+|RIGHT\s+|FULL\s+|CROSS\s+)?JOIN\b',
                          sql, re.IGNORECASE))


def has_cte(sql):
    """Check if SQL uses CTEs (WITH clause)."""
    return bool(re.search(r'\bWITH\b\s+\w+\s+AS\s*\(', sql, re.IGNORECASE))


def classify_difficulty(sql):
    """Classify SQL difficulty based on structural complexity."""
    joins = count_joins(sql)
    cte = has_cte(sql)
    agg = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql, re.IGNORECASE))
    subquery = bool(re.search(r'\(\s*SELECT\b', sql, re.IGNORECASE))

    score = joins * 2 + (2 if cte else 0) + (1 if agg else 0) + (3 if subquery else 0)
    if score <= 2:
        return "simple"
    elif score <= 6:
        return "medium"
    else:
        return "complex"


def load_original_cases():
    """Load original 26 eval cases and keep select ones."""
    cases_path = Path(__file__).parent.parent / "eval" / "eval_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Keep: 5 intent_recognition + 5 error_handling
    kept = []
    intent_cases = [c for c in cases if c["category"] == "intent_recognition"]
    error_cases = [c for c in cases if c["category"] == "error_handling"]
    kept.extend(intent_cases[:5])
    kept.extend(error_cases[:5])
    return kept


def build_falcon_cases():
    """Generate eval cases from Falcon dev.json."""
    falcon_path = Path(__file__).parent.parent / "falcon_dataset" / "dev_data" / "dev.json"
    with open(falcon_path, "r", encoding="utf-8") as f:
        falcon_data = json.load(f)

    # Filter for target DBs
    entries = [e for e in falcon_data if e["db_id"] in TARGET_DB_IDS]

    print(f"Falcon entries for DBs {TARGET_DB_IDS}: {len(entries)}")
    print(f"  DB 14 (toy_store):  {sum(1 for e in entries if e['db_id'] == '14')}")
    print(f"  DB 16 (school):     {sum(1 for e in entries if e['db_id'] == '16')}")
    print(f"  DB 17 (ecommerce):  {sum(1 for e in entries if e['db_id'] == '17')}")
    print(f"  DB 20 (city_ride):  {sum(1 for e in entries if e['db_id'] == '20')}")

    cases = []
    case_idx = 0

    # Distribution plan:
    # - sql_execution_accuracy: ~30 cases (spread across all 4 DBs)
    # - sql_generation: 20 cases (keyword checks, diverse DBs)
    # - result_exact_match: 20 cases (deterministic answers)
    # - result_overlap: 10 cases (order-insensitive)
    # - cross_db_consistency: 10 cases from DB 14 (imported to all 3 types)

    exec_count = 0
    gen_count = 0
    match_count = 0
    overlap_count = 0
    cross_count = 0

    for entry in entries:
        db_id = entry["db_id"]
        question = entry["question"]
        sql = entry["SQL"][0] if entry["SQL"] else ""
        ground_answer = entry.get("answer", [])
        question_id = entry["question_id"]
        is_order = entry.get("is_order", "0")

        if not sql:
            continue

        difficulty = classify_difficulty(sql)
        keywords = extract_sql_keywords(sql)
        tables = parse_sql_tables(sql)

        base_case = {
            "falcon_db_id": db_id,
            "falcon_question_id": question_id,
            "question": question,
            "ground_truth_sql": sql,
            "ground_truth_answer": ground_answer,
            "is_order_matters": is_order == "1",
            "difficulty": difficulty,
            "expected_sql_keywords": keywords,
            "required_tables": tables,
            "forbidden_sql_keywords": ["DELETE", "DROP", "INSERT", "UPDATE", "ALTER"],
        }

        # sql_execution_accuracy (~30 cases)
        if exec_count < 30:
            case = {
                **base_case,
                "id": f"eval_falcon_{case_idx:03d}",
                "category": "sql_execution_accuracy",
            }
            cases.append(case)
            case_idx += 1
            exec_count += 1

        # sql_generation (20 cases, prefer medium+complex)
        if gen_count < 20 and difficulty != "simple":
            case = {
                **base_case,
                "id": f"eval_falcon_{case_idx:03d}",
                "category": "sql_generation",
            }
            cases.append(case)
            case_idx += 1
            gen_count += 1

        # result_exact_match (20 cases with clear answers)
        if match_count < 20 and ground_answer and len(ground_answer) > 0:
            # Check if the answer has clear, deterministic structure
            first_answer = ground_answer[0] if isinstance(ground_answer, list) else ground_answer
            if isinstance(first_answer, dict) and len(first_answer) > 0:
                case = {
                    **base_case,
                    "id": f"eval_falcon_{case_idx:03d}",
                    "category": "result_exact_match",
                }
                cases.append(case)
                case_idx += 1
                match_count += 1

        # result_overlap (10 cases, order-insensitive)
        if overlap_count < 10 and is_order == "0" and ground_answer:
            case = {
                **base_case,
                "id": f"eval_falcon_{case_idx:03d}",
                "category": "result_set_overlap",
            }
            cases.append(case)
            case_idx += 1
            overlap_count += 1

        # cross_db_consistency (10 cases from DB 14)
        if cross_count < 10 and db_id == "14" and difficulty in ("medium", "complex"):
            case = {
                **base_case,
                "id": f"eval_falcon_{case_idx:03d}",
                "category": "cross_db_consistency",
                "db_types": ["sqlite", "mysql", "postgresql"],
            }
            cases.append(case)
            case_idx += 1
            cross_count += 1

    print(f"\nGenerated {len(cases)} Falcon cases:")
    print(f"  sql_execution_accuracy: {exec_count}")
    print(f"  sql_generation:        {gen_count}")
    print(f"  result_exact_match:    {match_count}")
    print(f"  result_set_overlap:    {overlap_count}")
    print(f"  cross_db_consistency:  {cross_count}")

    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Generate expanded eval cases from Falcon dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: eval/eval_cases_v2.json)",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(Path(__file__).parent.parent / "eval" / "eval_cases_v2.json")

    # Load original cases (keep 10)
    original = load_original_cases()
    print(f"Original cases kept: {len(original)}")
    for c in original:
        print(f"  {c['id']}: {c['category']} — {c.get('question', '')[:50]}")

    # Build Falcon cases
    falcon_cases = build_falcon_cases()

    # Merge
    all_cases = original + falcon_cases
    print(f"\nTotal cases: {len(all_cases)}")

    # Stats
    categories = {}
    for c in all_cases:
        cat = c["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("\nBy category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    difficulties = {}
    for c in falcon_cases:
        d = c.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1
    print("\nBy difficulty (Falcon cases):")
    for d, count in sorted(difficulties.items()):
        print(f"  {d}: {count}")

    # Write output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_cases)} cases to: {output}")


if __name__ == "__main__":
    main()
