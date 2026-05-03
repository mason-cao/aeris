import logging

import pytest

from app import main


def test_resolve_log_level_accepts_lowercase_name() -> None:
    assert main.resolve_log_level("debug") == logging.DEBUG


def test_resolve_log_level_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        main.resolve_log_level("chatty")
