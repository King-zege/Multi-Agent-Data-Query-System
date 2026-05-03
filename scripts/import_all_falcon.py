"""
Batch import Falcon datasets into SQLite, MySQL, and PostgreSQL.

Imports DBs 14 (toy_store), 16 (school), 17 (ecommerce), 20 (city_ride)
into all 3 database targets.

Usage:
    python scripts/import_all_falcon.py
    python scripts/import_all_falcon.py --db-ids 14,16
"""
import argparse
import os
from pathlib import Path

# Ensure project root in path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.import_falcon import (
    load_schema,
    import_to_sqlite,
    import_to_mysql,
    import_to_postgresql,
)

FALCON_BASE = str(Path(__file__).parent.parent / "falcon_dataset" / "dev_data")
DATA_DIR = str(Path(__file__).parent.parent / "data")
MYSQL_PORT = 3307
PG_PORT = 5433

TARGET_DBS = {
    "14": {"name": "toy_store", "tables": 4, "questions": 32, "domain": "Retail"},
    "16": {"name": "school", "tables": 4, "questions": 16, "domain": "Education"},
    "17": {"name": "ecommerce", "tables": 5, "questions": 14, "domain": "E-commerce"},
    "20": {"name": "city_ride", "tables": 2, "questions": 29, "domain": "Ride-sharing"},
}


def import_all(db_ids=None, skip_sqlite=False, skip_mysql=False, skip_pg=False):
    """Import all target Falcon databases into all 3 database targets."""
    if db_ids is None:
        db_ids = list(TARGET_DBS.keys())

    total = len(db_ids) * 3  # 3 targets per db
    count = 0

    for db_id in db_ids:
        info = TARGET_DBS[db_id]
        db_name = info["name"]
        db_info = load_schema(FALCON_BASE, db_id)
        tables = db_info["tables"]
        print(f"\n{'='*60}")
        print(f"  DB {db_id}: {db_name} ({info['domain']}) — {len(tables)} tables, {info['questions']} questions")
        print(f"{'='*60}")

        # SQLite
        if not skip_sqlite:
            count += 1
            output_path = os.path.join(DATA_DIR, f"falcon_{db_id}.db")
            print(f"\n  [{count}/{total}] SQLite → {output_path}")
            import_to_sqlite(db_info, FALCON_BASE, output_path)

        # MySQL
        if not skip_mysql:
            count += 1
            mysql_db = f"falcon_{db_id}_{db_name}"
            print(f"\n  [{count}/{total}] MySQL → localhost:{MYSQL_PORT}/{mysql_db}")
            import_to_mysql(
                db_info, FALCON_BASE,
                host="localhost", port=MYSQL_PORT,
                user="root", password="",
                database=mysql_db,
            )

        # PostgreSQL
        if not skip_pg:
            count += 1
            pg_db = f"falcon_{db_id}_{db_name}"
            print(f"\n  [{count}/{total}] PostgreSQL → localhost:{PG_PORT}/{pg_db}")
            import_to_postgresql(
                db_info, FALCON_BASE,
                host="localhost", port=PG_PORT,
                user="postgres", password="",
                database=pg_db,
            )

    print(f"\n{'='*60}")
    print(f"  Done. Imported {count} databases.")
    print(f"  SQLite files: {DATA_DIR}/falcon_*.db")
    print(f"  MySQL databases on localhost:{MYSQL_PORT}")
    print(f"  PostgreSQL databases on localhost:{PG_PORT}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch import Falcon datasets into all database targets"
    )
    parser.add_argument(
        "--db-ids", default=None,
        help="Comma-separated Falcon DB IDs (default: 14,16,17,20)",
    )
    parser.add_argument(
        "--skip-sqlite", action="store_true",
        help="Skip SQLite import",
    )
    parser.add_argument(
        "--skip-mysql", action="store_true",
        help="Skip MySQL import",
    )
    parser.add_argument(
        "--skip-pg", action="store_true",
        help="Skip PostgreSQL import",
    )
    args = parser.parse_args()

    db_ids = None
    if args.db_ids:
        db_ids = [x.strip() for x in args.db_ids.split(",")]

    import_all(
        db_ids=db_ids,
        skip_sqlite=args.skip_sqlite,
        skip_mysql=args.skip_mysql,
        skip_pg=args.skip_pg,
    )


if __name__ == "__main__":
    main()
