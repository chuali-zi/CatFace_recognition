"""
日志初始化模块。

作用：统一初始化控制台日志、文件日志和“最近日志”内存缓冲。
Web 页面最近日志来自内存缓冲，不直接读取日志文件。
"""

import logging
import os
from collections import deque

from config import resolve_project_path


class MemoryLogHandler(logging.Handler):
    """把最近 N 条日志保存在内存里，供 /api/status 返回。"""

    def __init__(self, capacity=200):
        logging.Handler.__init__(self)
        self.records = deque(maxlen=capacity)

    def emit(self, record):
        self.records.append(self.format(record))

    def tail(self, limit):
        items = list(self.records)
        return items[-limit:]


def setup_logging(config):
    """创建 logger，并返回 memory_handler 方便 Web 层取最近日志。"""
    log_cfg = config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)

    logger = logging.getLogger("phytium_door_controller")
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = resolve_project_path(config, log_cfg.get("file", "logs/door_controller.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    memory = MemoryLogHandler(capacity=200)
    memory.setFormatter(formatter)
    logger.addHandler(memory)

    return logger, memory
