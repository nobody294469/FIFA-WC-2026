"""logger.py — Structured colour console + rotating file logger."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from core.config import LOG_DIR

_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE = "%Y-%m-%d %H:%M:%S"

_COLORS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        record = logging.makeLogRecord(record.__dict__)
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_ColorFormatter(_FMT, _DATE))
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(fh)

    logger.propagate = False
    return logger