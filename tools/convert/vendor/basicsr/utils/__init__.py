"""Minimal subset of basicsr.utils used by the vendored arch modules.

The real package eagerly imports training/data dependencies (scipy, lmdb, …)
that are irrelevant for ONNX export; this stub exposes only what the arch
files import.
"""
import logging

from .registry import ARCH_REGISTRY  # noqa: F401

_initialized = set()


def get_root_logger(logger_name: str = "basicsr",
                    log_level: int = logging.INFO,
                    log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger_name in _initialized:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(log_level)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
    _initialized.add(logger_name)
    return logger
