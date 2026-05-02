"""
结构化日志模块

基于 Python stdlib logging，提供带时间戳、日志级别、模块名的结构化日志。
支持同时输出到控制台和文件，通过 contextvars 携带 thread_id / user_id 上下文。
"""

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path

# 日志上下文（协程安全）
thread_id_var: ContextVar[str] = ContextVar("thread_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")

# 格式：[2026-05-02 14:30:01] [INFO] [module] [thread:xxx] [user:xxx] message
SIMPLE_FORMAT = (
    "[%(asctime)s] [%(levelname)-5s] [%(name)s] "
    "[tid:%(thread_id)s] [uid:%(user_id)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ContextFilter(logging.Filter):
    """将 contextvars 注入日志记录"""

    def filter(self, record):
        record.thread_id = thread_id_var.get("-")
        record.user_id = user_id_var.get("-")
        return True


def set_log_context(thread_id: str = None, user_id: str = None):
    """设置当前协程的日志上下文

    Args:
        thread_id: 会话线程 ID
        user_id: 用户 ID
    """
    if thread_id is not None:
        thread_id_var.set(thread_id)
    if user_id is not None:
        user_id_var.set(user_id)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger

    Args:
        name: 模块名（通常传 __name__）

    Returns:
        配置好的 Logger 实例
    """
    return logging.getLogger(name)


_root_configured = False


def _configure_root():
    """配置根 logger（仅执行一次）"""
    global _root_configured
    if _root_configured:
        return
    _root_configured = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 控制台 handler（INFO 及以上）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT))
    console.addFilter(ContextFilter())
    root.addHandler(console)

    # 文件 handler（DEBUG 及以上，含轮转，最大 5MB × 3 份）
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT))
    file_handler.addFilter(ContextFilter())
    root.addHandler(file_handler)

    # 安静第三方库的 DEBUG 日志
    for lib in ("httpx", "openai", "urllib3", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# 模块导入时自动配置
_configure_root()
