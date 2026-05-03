"""
基于 FastMCP 的通用 SQL 查询工具服务器

支持 SQLite / MySQL / PostgreSQL，通过 SQLAlchemy 统一抽象。
数据库类型和连接信息通过环境变量注入（由 SQL Agent 启动时设置）。

环境变量：
    DB_TYPE     — sqlite | mysql | postgresql
    DB_PATH     — SQLite 数据库文件路径
    DB_HOST     — MySQL/PG 主机地址
    DB_PORT     — MySQL/PG 端口
    DB_NAME     — MySQL/PG 数据库名
    DB_USER     — MySQL/PG 用户名
    DB_PASSWORD — MySQL/PG 密码

提供的 MCP 工具：
    execute_sql(sql) — 执行 SQL 查询，返回 JSON
    get_schema()     — 获取数据库全部表结构，返回 JSON
"""

import json
import os
from sqlalchemy import create_engine, inspect, text
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sql-query-server")


def _get_engine():
    """根据环境变量创建 SQLAlchemy 引擎"""
    db_type = os.environ.get("DB_TYPE", "sqlite")

    if db_type == "sqlite":
        db_path = os.environ.get("DB_PATH", "./data/company.db")
        return create_engine(f"sqlite:///{db_path}")

    elif db_type == "mysql":
        user = os.environ.get("DB_USER", "root")
        password = os.environ.get("DB_PASSWORD", "")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "3306")
        db_name = os.environ.get("DB_NAME", "")
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
        return create_engine(url)

    elif db_type == "postgresql":
        user = os.environ.get("DB_USER", "postgres")
        password = os.environ.get("DB_PASSWORD", "")
        host = os.environ.get("DB_HOST", "localhost")
        port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "")
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        return create_engine(url)

    else:
        raise ValueError(f"Unsupported database type: {db_type}")


@mcp.tool()
def execute_sql(sql: str) -> str:
    """执行 SQL 查询

    Args:
        sql: SQL 查询语句（只读 SELECT，兼容所有数据库方言）

    Returns:
        JSON 格式的查询结果
    """
    engine = _get_engine()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            return json.dumps(rows, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"SQL执行错误: {e}", "sql": sql},
                          ensure_ascii=False)


@mcp.tool()
def get_schema() -> str:
    """获取数据库全部表结构

    自动适配 SQLite / MySQL / PostgreSQL，
    通过 SQLAlchemy Inspector 读取各数据库的 information_schema。

    Returns:
        JSON 格式的表结构列表:
        [{"name": "employees", "columns": [{"name": "emp_id", "type": "INTEGER",
          "nullable": false, "primary_key": true}, ...]}, ...]
    """
    engine = _get_engine()
    try:
        inspector = inspect(engine)
        tables = []
        for table_name in inspector.get_table_names():
            columns = []
            # 获取主键列名集合
            pk_cols = set()
            try:
                pk = inspector.get_pk_constraint(table_name)
                pk_cols = set(pk.get("constrained_columns", []))
            except Exception:
                pass

            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "primary_key": col["name"] in pk_cols,
                })
            tables.append({"name": table_name, "columns": columns})
        return json.dumps(tables, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Schema获取失败: {e}"}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
