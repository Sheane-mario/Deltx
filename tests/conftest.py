"""Shared pytest fixtures for Deltx tests."""

import pytest

from deltx.common.config import Config


@pytest.fixture
def config() -> Config:
    """Provide a default Config instance."""
    return Config()
