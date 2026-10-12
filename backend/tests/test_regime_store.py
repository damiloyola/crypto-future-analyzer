"""Persistencia del modelo de régimen y regla de reentrenamiento."""

from __future__ import annotations

import dataclasses
import logging
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from app.config import RegimeLabel
from app.regime import store as store_mod
from app.regime.model import config_fingerprint, train_regime_model
from app.regime.store import load_model, model_path, needs_retrain, save_model
from app.regime.types import FEATURE_COLUMNS, MODEL_FORMAT_VERSION, RegimeModel

FIXED_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


def sample_two_state(T: int, seed: int) -> np.ndarray:
    src = GaussianHMM(n_components=2, covariance_type="full")
    src.startprob_ = np.array([0.5, 0.5])
    src.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    src.means_ = np.array([[-0.01, 0.03, 0.04, 1.5], [0.008, 0.01, 0.012, -0.3]])
    src.covars_ = np.array([np.diag([0.002, 0.002, 0.003, 0.25]) ** 2] * 2)
    src.n_features = len(FEATURE_COLUMNS)
    X, _ = src.sample(T, random_state=seed)
    return X


@pytest.fixture(scope="module")
def cfg(settings):
    labels = {
        **settings.regime.labels,
        2: [
            RegimeLabel(name="Bajista", category="bajista"),
            RegimeLabel(name="Alcista", category="alcista"),
        ],
    }
    return settings.regime.model_copy(
        update={
            "n_states_min": 2,
            "n_states_max": 3,
            "n_restarts": 1,
            "max_iter": 30,
            "min_training_samples": 200,
            "labels": labels,
        }
    )


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, pd.DatetimeIndex]:
    X = sample_two_state(600, seed=21)
    idx = pd.date_range("2025-03-01", periods=len(X), freq="h", tz="UTC", name="open_time")
    return X, idx


@pytest.fixture(scope="module")
def model(cfg, data) -> RegimeModel:
    X, idx = data
    return train_regime_model(X, idx, "BTC", "1h", cfg, clock=lambda: FIXED_NOW)


# --- rutas ---------------------------------------------------------------------


def test_model_path(tmp_path):
    assert model_path(tmp_path, "btc", "1h") == tmp_path / "BTC_1h.joblib"
    assert model_path(str(tmp_path), "ETH", "1d") == tmp_path / "ETH_1d.joblib"
    assert model_path(tmp_path, " sol ", "1h") == tmp_path / "SOL_1h.joblib"


@pytest.mark.parametrize(
    ("symbol", "timeframe"),
    [
        ("../../outside/evil", "1h"),
        ("BTC/USDT", "1h"),
        ("..", "1h"),
        ("", "1h"),
        ("BTC", "../1h"),
        ("BTC", ""),
        ("BTC\\..", "1h"),
        ("BTCñ", "1h"),
    ],
)
def test_model_path_rejects_anything_that_is_not_letters_and_numbers(tmp_path, symbol, timeframe):
    # Regresión: "../../outside/evil" armaba una ruta fuera de la carpeta de modelos
    # y ensure_model la cargaba con joblib (pickle) antes de validar nada.
    with pytest.raises(ValueError, match="inválido"):
        model_path(tmp_path, symbol, timeframe)


# --- guardar y cargar ----------------------------------------------------------


def test_roundtrip_identical(tmp_path, model, data):
    path = tmp_path / "anidada" / "carpeta" / "BTC_1h.joblib"
    assert save_model(model, path) == path
    loaded = load_model(path)
    assert isinstance(loaded, RegimeModel)

    for attr in ("startprob_", "transmat_", "means_", "covars_"):
        np.testing.assert_array_equal(getattr(loaded.hmm, attr), getattr(model.hmm, attr))
    np.testing.assert_array_equal(loaded.scaler.mean, model.scaler.mean)
    np.testing.assert_array_equal(loaded.scaler.scale, model.scaler.scale)
    assert loaded.scaler.feature_names == model.scaler.feature_names
    assert loaded.states == model.states
    for field in (
        "symbol",
        "timeframe",
        "n_states",
        "bic_by_n",
        "loglik_by_n",
        "trained_at",
        "train_start",
        "train_end",
        "n_samples",
        "config_fingerprint",
        "synthetic",
        "format_version",
        "notes",
    ):
        assert getattr(loaded, field) == getattr(model, field), field

    # Misma verosimilitud (algoritmo forward) sobre los mismos datos.
    X, _ = data
    Z = model.scaler.transform(X)
    assert loaded.hmm.score(Z) == model.hmm.score(Z)


def test_missing_file_returns_none_without_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="app.regime.store"):
        assert load_model(tmp_path / "no_existe.joblib") is None
    assert not caplog.records


def test_corrupt_file_returns_none_with_warning(tmp_path, caplog):
    path = tmp_path / "BTC_1h.joblib"
    path.write_bytes(b"esto no es un modelo")
    with caplog.at_level(logging.WARNING, logger="app.regime.store"):
        assert load_model(path) is None
    assert any("No se pudo leer" in r.getMessage() for r in caplog.records)


def test_truncated_file_returns_none(tmp_path, model):
    path = tmp_path / "BTC_1h.joblib"
    save_model(model, path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    assert load_model(path) is None


def test_wrong_type_returns_none_with_warning(tmp_path, caplog):
    path = tmp_path / "BTC_1h.joblib"
    joblib.dump({"no": "soy un modelo"}, path)
    with caplog.at_level(logging.WARNING, logger="app.regime.store"):
        assert load_model(path) is None
    assert any("no es un modelo de régimen" in r.getMessage() for r in caplog.records)


def test_wrong_format_version_returns_none_with_warning(tmp_path, model, caplog):
    old = dataclasses.replace(model, format_version=MODEL_FORMAT_VERSION + 1)
    path = tmp_path / "BTC_1h.joblib"
    save_model(old, path)
    with caplog.at_level(logging.WARNING, logger="app.regime.store"):
        assert load_model(path) is None
    assert any("formato" in r.getMessage() for r in caplog.records)


def test_inconsistent_model_returns_none(tmp_path, model):
    broken = dataclasses.replace(model, states=model.states[:1])
    path = tmp_path / "BTC_1h.joblib"
    save_model(broken, path)
    assert load_model(path) is None


def test_save_rejects_non_model(tmp_path):
    with pytest.raises(TypeError):
        save_model({"a": 1}, tmp_path / "x.joblib")  # type: ignore[arg-type]


def test_atomic_write_leaves_only_target(tmp_path, model):
    path = tmp_path / "BTC_1h.joblib"
    save_model(model, path)
    save_model(model, path)  # sobrescribir también es atómico
    assert sorted(p.name for p in tmp_path.iterdir()) == ["BTC_1h.joblib"]


def test_failed_write_keeps_previous_model_and_no_temp_files(tmp_path, model, monkeypatch):
    path = tmp_path / "BTC_1h.joblib"
    save_model(model, path)
    before = path.read_bytes()

    def exploding_dump(obj, fh, *args, **kwargs):
        fh.write(b"a medio escribir")
        raise RuntimeError("se cortó la luz")

    monkeypatch.setattr(store_mod.joblib, "dump", exploding_dump)
    newer = dataclasses.replace(model, n_samples=model.n_samples + 1)
    with pytest.raises(RuntimeError):
        save_model(newer, path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["BTC_1h.joblib"]
    assert path.read_bytes() == before
    monkeypatch.undo()
    loaded = load_model(path)
    assert loaded is not None
    assert loaded.n_samples == model.n_samples


# --- ¿hay que reentrenar? ------------------------------------------------------


def test_needs_retrain_without_model():
    assert needs_retrain(None, FIXED_NOW, 7, "abc") == (True, "no hay modelo guardado")


def test_needs_retrain_fresh_model(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    retrain, reason = needs_retrain(
        model,
        FIXED_NOW + timedelta(days=2, hours=3),
        7,
        fp,
        symbol="BTC",
        timeframe="1h",
        synthetic=False,
    )
    assert retrain is False
    assert reason == "el modelo está al día: tiene 2 días y se reentrena cada 7"
    retrain, reason = needs_retrain(model, FIXED_NOW + timedelta(hours=5), 7, fp)
    assert retrain is False
    assert "menos de 1 día" in reason


def test_needs_retrain_old_model(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    assert needs_retrain(model, FIXED_NOW + timedelta(days=9, hours=2), 7, fp) == (
        True,
        "el modelo tiene 9 días y se reentrena cada 7",
    )
    # Justo 7 días: ya toca.
    assert needs_retrain(model, FIXED_NOW + timedelta(days=7), 7, fp)[0] is True
    assert needs_retrain(model, FIXED_NOW + timedelta(days=6, hours=23), 7, fp)[0] is False


def test_needs_retrain_config_changed(model):
    assert needs_retrain(model, FIXED_NOW, 7, "otra-huella") == (
        True,
        "cambió la configuración del régimen",
    )


def test_needs_retrain_other_symbol_or_timeframe(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    retrain, reason = needs_retrain(model, FIXED_NOW, 7, fp, symbol="ETH")
    assert retrain is True
    assert reason.startswith("el modelo es de otra moneda")
    assert needs_retrain(model, FIXED_NOW, 7, fp, symbol="btc")[0] is False
    retrain, reason = needs_retrain(model, FIXED_NOW, 7, fp, timeframe="1d")
    assert retrain is True
    assert reason.startswith("el modelo es de otro timeframe")


def test_needs_retrain_synthetic_mismatch(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    fake = dataclasses.replace(model, synthetic=True)
    retrain, reason = needs_retrain(fake, FIXED_NOW, 7, fp, synthetic=False)
    assert retrain is True
    assert reason.startswith("el modelo guardado es sintético")
    retrain, reason = needs_retrain(model, FIXED_NOW, 7, fp, synthetic=True)
    assert retrain is True
    assert reason == "el modelo guardado es de datos reales y se pidió uno sintético"
    # Sin pedir nada en particular, no se chequea.
    assert needs_retrain(fake, FIXED_NOW, 7, fp)[0] is False
    assert needs_retrain(fake, FIXED_NOW, 7, fp, synthetic=True)[0] is False


def test_needs_retrain_old_format(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    old = dataclasses.replace(model, format_version=MODEL_FORMAT_VERSION - 1)
    assert needs_retrain(old, FIXED_NOW, 7, fp) == (
        True,
        "el modelo guardado tiene un formato viejo",
    )


def test_needs_retrain_config_checked_before_age(model):
    retrain, reason = needs_retrain(model, FIXED_NOW + timedelta(days=30), 7, "otra-huella")
    assert retrain is True
    assert reason == "cambió la configuración del régimen"


def test_needs_retrain_naive_now_is_utc(model, cfg):
    fp = config_fingerprint(cfg, "1h")
    naive = (FIXED_NOW + timedelta(days=8)).replace(tzinfo=None)
    assert needs_retrain(model, naive, 7, fp) == (
        True,
        "el modelo tiene 8 días y se reentrena cada 7",
    )
