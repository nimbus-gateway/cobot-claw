"""Tests for cobot.logging — configuration and logger factory."""
from __future__ import annotations

import structlog

from cobot.logging import configure_logging, get_logger


class TestConfigureLogging:
    def test_configure_runs_without_error(self):
        configure_logging("INFO")

    def test_configure_debug_level(self):
        configure_logging("DEBUG")

    def test_configure_warning_level(self):
        configure_logging("WARNING")


class TestGetLogger:
    def test_returns_bound_logger(self):
        configure_logging()
        logger = get_logger()
        assert logger is not None

    def test_named_logger(self):
        configure_logging()
        logger = get_logger("cobot.test")
        assert logger is not None

    def test_logger_can_log(self):
        configure_logging()
        logger = get_logger("cobot.test")
        # Should not raise
        logger.info("test message", key="value")

    def test_logger_can_log_warning(self):
        configure_logging()
        logger = get_logger()
        logger.warning("something happened", component="test")
