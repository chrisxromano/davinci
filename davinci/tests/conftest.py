"""Pytest configuration and shared fixtures."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after test session."""
    yield
    # Cleanup after all tests complete
    root = Path(__file__).parent.parent.parent
    for artifact in ["test_model_cache"]:
        path = root / artifact
        if path.exists():
            shutil.rmtree(path)
