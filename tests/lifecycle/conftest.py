"""Register the ``lifecycle`` marker without touching the global pytest config."""

from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lifecycle: slow package install/upgrade/uninstall checks; run explicitly "
        "from tests/lifecycle/lifecycle_test.py",
    )
