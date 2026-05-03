"""
将 Falcon 数据集导入 MySQL / PostgreSQL / SQLite

用法：
    python scripts/import_falcon.py --db 14 --target mysql
    python scripts/import_falcon.py --db 14 --target postgresql
    python scripts/import_falcon.py --db 14 --target sqlite
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_schema(falcon_base, db_id):
    """从 tables.json 加载指定数据库的 schema"""
    with open(f"{falcon_base}/tables.json", "r", encoding="utf-8") as f:
        all_dbs = json.load(f)
    for db in all_dbs:
        if db["db_id"] == db_id:
            return db
    raise ValueError(f"Database {db_id} not found in tables.json")


def detect_column_type(values, max_sample=20):
    """从数据值推断列类型，自动处理超大整数（BIGINT）"""
    sample = [v for v in values[:max_sample] if v.strip() and v.strip() != "NULL"]
    if not sample:
        return "TEXT"

    is_int = True
    is_float = False
    max_abs = 0
    for val in sample:
        cleaned = val.replace(",", "").replace("$", "").replace(" ", "").strip()
        try:
            n = float(cleaned)
            max_abs = max(max_abs, abs(n))
            if n != int(n):
                is_int = False
                is_float = True
        except ValueError:
            return "TEXT"
    if is_int:
        # 超过 32-bit 有符号整数范围时使用 BIGINT
        if max_abs > 2147483647:
            return "BIGINT"
        return "INTEGER"
    if is_float:
        return "DECIMAL(12,2)"
    return "TEXT"


def import_to_sqlite(db_info, falcon_base, output_path):
    """导入到 SQLite"""
    import sqlite3

    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    for table in db_info["tables"]:
        table_name = table["table_name"]
        csv_path = f"{falcon_base}/dev_databases/{db_info['db_id']}/database_description/{table_name}.csv"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [row for row in reader]

        col_defs = []
        for i, header in enumerate(headers):
            col_type = detect_column_type([r[i] for r in rows if i < len(r)])
            col_defs.append(f'"{header}" {col_type}')

        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        create_sql = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
        cur.execute(create_sql)

        if rows:
            placeholders = ", ".join(["?"] * len(headers))
            cols = ", ".join([f'"{h}"' for h in headers])
            insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders})'
            clean_rows = []
            for row in rows:
                clean = []
                for val in row:
                    val = val.strip()
                    if val == "" or val == "NULL":
                        clean.append(None)
                    else:
                        try:
                            clean.append(float(val.replace(",", "").replace("$", "").replace(" ", "")))
                        except ValueError:
                            clean.append(val)
                clean_rows.append(clean)
            cur.executemany(insert_sql, clean_rows)

        print(f"  {table_name}: {len(rows)} rows")

    conn.commit()
    conn.close()
    print(f"SQLite database created: {output_path}")
    return output_path


def import_to_mysql(db_info, falcon_base, host, port, user, password, database):
    """导入到 MySQL"""
    import pymysql

    conn = pymysql.connect(host=host, port=port, user=user, password=password)
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {database}")
    cur.execute(f"CREATE DATABASE {database} CHARACTER SET utf8mb4")
    cur.execute(f"USE {database}")

    for table in db_info["tables"]:
        table_name = table["table_name"]
        csv_path = f"{falcon_base}/dev_databases/{db_info['db_id']}/database_description/{table_name}.csv"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [row for row in reader]

        col_defs = []
        for i, header in enumerate(headers):
            col_type = detect_column_type([r[i] for r in rows if i < len(r)])
            col_defs.append(f"`{header}` {col_type}")

        cur.execute(f"DROP TABLE IF EXISTS `{table_name}`")
        create_sql = f"CREATE TABLE `{table_name}` ({', '.join(col_defs)})"
        cur.execute(create_sql)

        if rows:
            placeholders = ", ".join(["%s"] * len(headers))
            cols = ", ".join([f"`{h}`" for h in headers])
            insert_sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"
            clean_rows = []
            for row in rows:
                clean = []
                for val in row:
                    val = val.strip()
                    if val == "" or val == "NULL":
                        clean.append(None)
                    else:
                        try:
                            clean.append(float(val.replace(",", "").replace("$", "").replace(" ", "")))
                        except ValueError:
                            clean.append(val)
                clean_rows.append(clean)
            cur.executemany(insert_sql, clean_rows)

        print(f"  {table_name}: {len(rows)} rows")

    conn.commit()
    conn.close()
    print(f"MySQL database '{database}' populated on {host}:{port}")


def import_to_postgresql(db_info, falcon_base, host, port, user, password, database):
    """导入到 PostgreSQL"""
    import psycopg2

    # Connect to default database first to create our database
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {database}")
    cur.execute(f"CREATE DATABASE {database}")
    cur.close()
    conn.close()

    # Connect to the new database
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=database)
    cur = conn.cursor()

    for table in db_info["tables"]:
        table_name = table["table_name"]
        csv_path = f"{falcon_base}/dev_databases/{db_info['db_id']}/database_description/{table_name}.csv"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [row for row in reader]

        col_defs = []
        for i, header in enumerate(headers):
            col_type = detect_column_type([r[i] for r in rows if i < len(r)])
            # Map to PostgreSQL types
            if col_type == "INTEGER":
                pg_type = "INTEGER"
            elif col_type.startswith("DECIMAL"):
                pg_type = "NUMERIC(12,2)"
            else:
                pg_type = "TEXT"
            col_defs.append(f'"{header}" {pg_type}')

        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        create_sql = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
        cur.execute(create_sql)

        if rows:
            placeholders = ", ".join(["%s"] * len(headers))
            cols = ", ".join([f'"{h}"' for h in headers])
            insert_sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders})'
            clean_rows = []
            for row in rows:
                clean = []
                for val in row:
                    val = val.strip()
                    if val == "" or val == "NULL":
                        clean.append(None)
                    else:
                        try:
                            clean.append(float(val.replace(",", "").replace("$", "").replace(" ", "")))
                        except ValueError:
                            clean.append(val)
                clean_rows.append(clean)
            cur.executemany(insert_sql, clean_rows)

        print(f"  {table_name}: {len(rows)} rows")

    conn.commit()
    cur.close()
    conn.close()
    print(f"PostgreSQL database '{database}' populated on {host}:{port}")


def main():
    parser = argparse.ArgumentParser(description="Import Falcon dataset into a database")
    parser.add_argument("--db", default="14", help="Falcon database ID (default: 14 = toy store)")
    parser.add_argument("--target", choices=["sqlite", "mysql", "postgresql"], default="sqlite")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="toy_store")
    parser.add_argument("--output", default=None, help="Output path (for SQLite)")
    args = parser.parse_args()

    falcon_base = str(Path(__file__).parent.parent / "falcon_dataset" / "dev_data")
    db_info = load_schema(falcon_base, args.db)

    tables = db_info["tables"]
    print(f"Falcon DB {args.db}: {len(tables)} tables — {[t['table_name'] for t in tables]}")

    if args.target == "sqlite":
        output = args.output or f"./data/falcon_{args.db}.db"
        import_to_sqlite(db_info, falcon_base, output)
    elif args.target == "mysql":
        port = args.port or 3306
        import_to_mysql(db_info, falcon_base, args.host, port, args.user, args.password, args.database)
    elif args.target == "postgresql":
        port = args.port or 5432
        import_to_postgresql(db_info, falcon_base, args.host, port, args.user, args.password, args.database)


if __name__ == "__main__":
    main()
