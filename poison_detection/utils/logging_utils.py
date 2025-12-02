"""
Logging utilities for the poison detection toolkit.

Provides a centralized logging configuration for consistent logging across modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handlers if the logger doesn't have any
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def setup_file_logger(
    name: str,
    log_file: Path,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger that writes to both console and file.

    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = get_logger(name, level)

    # Add file handler if not already present
    has_file_handler = any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    )

    if not has_file_handler:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
