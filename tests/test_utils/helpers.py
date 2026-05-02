"""测试辅助函数"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any


def count_table_rows(db_path: str, table: str) -> int:
    """统计数据库表的行数"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def read_json_from_db_result(result: Dict[str, Any], key: str = "data"):
    """从 agent 返回结果中解析 JSON data 字段"""
    data = result.get(key)
    if isinstance(data, str):
        return json.loads(data)
    return data
