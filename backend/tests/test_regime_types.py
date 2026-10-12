"""Tipos compartidos del régimen: escalador ajustado solo con entrenamiento."""

import numpy as np
import pytest

from app.regime.types import FEATURE_COLUMNS, FeatureScaler


def test_scaler_roundtrip_and_constant_column():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    X[:, 3] = 5.0  # columna constante: desvío 0 -> escala 1
    sc = FeatureScaler.fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z[:, :3].mean(axis=0), 0, atol=1e-12)
    assert np.allclose(Z[:, :3].std(axis=0), 1, atol=1e-12)
    assert np.allclose(Z[:, 3], 0)
    assert np.allclose(sc.inverse_transform(Z), X)
    assert sc.feature_names == FEATURE_COLUMNS


def test_scaler_rejects_nan_and_tiny_input():
    with pytest.raises(ValueError):
        FeatureScaler.fit(np.array([[1.0, np.nan, 0, 0], [2.0, 1, 0, 0]]))
    with pytest.raises(ValueError):
        FeatureScaler.fit(np.zeros((1, 4)))
