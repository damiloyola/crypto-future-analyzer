"""Fixtures compartidas.

Los tests nunca le pegan a APIs reales: usan datos sintéticos o archivos de
``tests/fixtures/`` (respuestas grabadas de Binance, etc.).
"""

import pytest

from app.config import DEFAULT_CONFIG_PATH, AppSettings, load_settings


@pytest.fixture(scope="session")
def settings() -> AppSettings:
    return load_settings(DEFAULT_CONFIG_PATH)
