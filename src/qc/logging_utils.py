"""统一日志：同时输出到控制台和文件，关键步骤带时间戳。"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 第三方库的弃用/软链警告与本项目无关，避免污染日志
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")


def setup_logger(name: str = "qc", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    # Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文日志乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - 某些重定向场景不支持
        pass

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # 避免重复添加 handler（同一进程内多次调用）
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(fh)
    return logger
