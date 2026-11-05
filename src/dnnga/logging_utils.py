"""Logging for dnnga.

Follows the standard-library guidance for *libraries*: the package logger
gets a ``NullHandler`` at import time and nothing else, so importing
``dnnga`` never prints anything or fights a host application's logging
setup. Call ``configure_logging()`` explicitly to see output -- the opt-in
equivalent of the original code's unconditional ``print()`` calls.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

LOGGER_NAME = "dnnga"

_logger = logging.getLogger(LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def get_logger() -> logging.Logger:
    """Return the package logger (no handlers beyond the NullHandler by default)."""
    return _logger


def configure_logging(
    level: int = logging.INFO,
    filepath: Optional[str] = None,
    stream=None,
) -> logging.Logger:
    """Attach a console handler (and optionally a file handler) to ``dnnga``.

    Safe to call repeatedly: handlers added by an earlier call are replaced,
    not stacked.

    ``logging.INFO`` gives one line per generation; ``DEBUG`` adds
    per-candidate detail. ``filepath`` is the structured-logging replacement
    for the notebook's ``AG_NN_Output_<timestamp>.txt`` on Drive.
    """
    logger = get_logger()

    for handler in list(logger.handlers):
        if getattr(handler, "_dnnga_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream or sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler._dnnga_managed = True  # type: ignore[attr-defined]
    logger.addHandler(stream_handler)

    if filepath is not None:
        file_handler = logging.FileHandler(filepath, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._dnnga_managed = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
