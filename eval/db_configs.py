"""
数据库配置注册表

管理所有可用于评估的数据库连接配置。
Key: (falcon_db_id, db_type) — 其中 falcon_db_id="company" 为原始业务数据库
"""

# Falcon 数据库名映射
FALCON_DB_NAMES = {
    "14": "toy_store",
    "16": "school",
    "17": "ecommerce",
    "20": "city_ride",
}

# MySQL/PostgreSQL 连接参数
MYSQL_BASE = {
    "host": "localhost",
    "port": 3307,
    "username": "root",
    "password": "",
}
PG_BASE = {
    "host": "localhost",
    "port": 5433,
    "username": "postgres",
    "password": "",
}


def _build_configs():
    """构建全部数据库配置"""
    configs = {}

    # 原始 company 数据库 (SQLite only)
    configs[("company", "sqlite")] = {
        "type": "sqlite",
        "path": "./data/company.db",
    }

    # Falcon 数据库 × 3 种数据库类型
    for db_id, db_name in FALCON_DB_NAMES.items():
        configs[(db_id, "sqlite")] = {
            "type": "sqlite",
            "path": f"./data/falcon_{db_id}.db",
        }
        configs[(db_id, "mysql")] = {
            "type": "mysql",
            "database": f"falcon_{db_id}_{db_name}",
            **MYSQL_BASE,
        }
        configs[(db_id, "postgresql")] = {
            "type": "postgresql",
            "database": f"falcon_{db_id}_{db_name}",
            **PG_BASE,
        }

    return configs


ALL_DB_CONFIGS = _build_configs()


def get_db_config(db_id: str, db_type: str) -> dict:
    """获取单个数据库配置

    Args:
        db_id: Falcon DB ID ("14", "16", "17", "20") 或 "company"
        db_type: "sqlite" | "mysql" | "postgresql"

    Returns:
        db_config 字典
    """
    key = (db_id, db_type)
    if key not in ALL_DB_CONFIGS:
        available = [str(k) for k in ALL_DB_CONFIGS]
        raise KeyError(f"No config for {key}. Available: {available}")
    return ALL_DB_CONFIGS[key].copy()


def select_db_configs(db_type="all", db_id="all"):
    """选择一组数据库配置

    Args:
        db_type: "sqlite" | "mysql" | "postgresql" | "all"
        db_id: Falcon DB ID ("14"...) | "company" | "all"

    Returns:
        [(key, config), ...] 列表
    """
    db_types = ["sqlite", "mysql", "postgresql"] if db_type == "all" else [db_type]
    result = []
    for (fid, dtype), cfg in ALL_DB_CONFIGS.items():
        if dtype not in db_types:
            continue
        if db_id != "all" and fid != db_id:
            continue
        result.append(((fid, dtype), cfg))
    return result


def list_databases():
    """列出所有可用数据库"""
    print("Available database configurations:")
    for (db_id, db_type), cfg in sorted(ALL_DB_CONFIGS.items()):
        print(f"  ({db_id:>7}, {db_type:<12}) -> {cfg}")
