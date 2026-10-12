"""Entrenamiento del HMM: selección por BIC, etiquetas, determinismo y avisos.

Los datos salen de un ``GaussianHMM`` conocido con estados bien separados, así
se sabe cuántos estados "de verdad" hay. Las configs son chicas (pocos estados,
inicializaciones e iteraciones) para que los tests sean rápidos.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from app.config import RegimeConfig, RegimeLabel
from app.regime import model as model_mod
from app.regime.labels import stationary_distribution
from app.regime.model import (
    RegimeTrainingError,
    config_fingerprint,
    n_free_parameters,
    train_regime_model,
)
from app.regime.types import FEATURE_COLUMNS, MODEL_FORMAT_VERSION, RegimeModel

FIXED_NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)

# Medias en unidades reales por estado: retorno log, volatilidad, rango, z volumen.
SOURCE_MEANS = {
    2: [[-0.010, 0.030, 0.040, 1.5], [0.008, 0.010, 0.012, -0.3]],
    3: [[-0.010, 0.030, 0.040, 1.5], [0.000, 0.010, 0.010, -0.5], [0.010, 0.020, 0.025, 0.8]],
    5: [
        [-0.020, 0.040, 0.050, 2.0],
        [-0.008, 0.020, 0.022, 0.6],
        [0.000, 0.008, 0.008, -0.6],
        [0.008, 0.016, 0.018, 0.2],
        [0.020, 0.032, 0.040, 1.4],
    ],
}
SOURCE_SD = np.array([0.002, 0.002, 0.003, 0.25])


def sample_source(k: int, T: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Muestra ``T`` velas de un HMM de ``k`` estados con medias conocidas."""
    src = GaussianHMM(n_components=k, covariance_type="full")
    src.startprob_ = np.full(k, 1.0 / k)
    A = np.full((k, k), 0.05 / (k - 1))
    np.fill_diagonal(A, 0.95)
    src.transmat_ = A
    src.means_ = np.asarray(SOURCE_MEANS[k], dtype=float)
    src.covars_ = np.array([np.diag(SOURCE_SD**2)] * k)
    src.n_features = len(FEATURE_COLUMNS)
    X, states = src.sample(T, random_state=seed)
    return X, states


def hourly_index(T: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=T, freq="h", tz="UTC", name="open_time")


def labels_with_two(cfg: RegimeConfig) -> dict[int, list[RegimeLabel]]:
    return {
        **cfg.labels,
        2: [
            RegimeLabel(name="Bajista", category="bajista"),
            RegimeLabel(name="Alcista", category="alcista"),
        ],
    }


@pytest.fixture(scope="module")
def small_cfg(settings) -> RegimeConfig:
    return settings.regime.model_copy(
        update={
            "n_states_min": 2,
            "n_states_max": 4,
            "n_restarts": 2,
            "max_iter": 50,
            "min_training_samples": 200,
            "labels": labels_with_two(settings.regime),
            # La matriz de estos tests ya son las features (sin suavizar).
            "features": settings.regime.features.model_copy(update={"smoothing_window": 1}),
        }
    )


@pytest.fixture(scope="module")
def data3() -> tuple[np.ndarray, pd.DatetimeIndex]:
    X, _ = sample_source(3, 2500, seed=1)
    return X, hourly_index(len(X))


@pytest.fixture(scope="module")
def data2() -> tuple[np.ndarray, pd.DatetimeIndex]:
    X, _ = sample_source(2, 1500, seed=2)
    return X, hourly_index(len(X))


@pytest.fixture(scope="module")
def model3(small_cfg, data3) -> RegimeModel:
    X, idx = data3
    return train_regime_model(X, idx, "BTC", "1h", small_cfg, clock=lambda: FIXED_NOW)


@pytest.fixture(scope="module")
def model2(small_cfg, data2) -> RegimeModel:
    X, idx = data2
    return train_regime_model(X, idx, "ETH", "1h", small_cfg, clock=lambda: FIXED_NOW)


# --- selección por BIC -------------------------------------------------------


def test_bic_picks_true_number_of_states(model3, small_cfg):
    assert model3.n_states == 3
    assert set(model3.bic_by_n) == {2, 3, 4}
    assert set(model3.loglik_by_n) == {2, 3, 4}
    assert min(model3.bic_by_n, key=model3.bic_by_n.get) == 3
    assert model3.hmm.n_components == 3
    assert len(model3.states) == 3


def test_bic_is_not_hardcoded_two_state_source(model2):
    assert model2.n_states == 2
    assert min(model2.bic_by_n, key=model2.bic_by_n.get) == 2


def test_more_states_never_lower_loglik_by_much(model3):
    # Con más estados la verosimilitud no empeora (salvo ruido de EM), pero el
    # BIC penaliza los parámetros extra: por eso 4 no le gana a 3.
    assert model3.loglik_by_n[3] > model3.loglik_by_n[2]
    assert model3.bic_by_n[4] > model3.bic_by_n[3]


# --- etiquetas ---------------------------------------------------------------


def test_labels_ordered_and_named(model3, small_cfg):
    by_rank = model3.states_by_rank()
    returns = [s.mean_return for s in by_rank]
    assert returns == sorted(returns)
    assert [s.rank for s in by_rank] == [0, 1, 2]
    assert [s.name for s in by_rank] == [lb.name for lb in small_cfg.labels[3]]
    assert [s.category for s in by_rank] == [lb.category for lb in small_cfg.labels[3]]
    # Las estadísticas están en unidades reales y recuperan las de la fuente.
    truth = np.asarray(SOURCE_MEANS[3])
    for s, (ret, vol, rng, _vz) in zip(by_rank, truth, strict=True):
        assert s.mean_return == pytest.approx(ret, abs=1e-3)
        assert s.mean_volatility == pytest.approx(vol, abs=1e-3)
        assert s.mean_range == pytest.approx(rng, abs=1e-3)
    # Indexado por índice interno del HMM.
    assert [s.index for s in model3.states] == [0, 1, 2]


def test_occupancy_from_stationary_distribution(model3):
    A = np.asarray(model3.hmm.transmat_)
    occ = np.array([s.occupancy for s in model3.states])
    assert occ.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(occ, stationary_distribution(A))
    np.testing.assert_allclose(occ @ A, occ, atol=1e-10)
    for s in model3.states:
        assert s.expected_duration_bars == pytest.approx(1.0 / (1.0 - A[s.index, s.index]))


def test_five_states_get_crash_to_euforia(settings):
    cfg = settings.regime.model_copy(
        update={"n_states_min": 5, "n_states_max": 5, "n_restarts": 2, "max_iter": 60}
    )
    X, _ = sample_source(5, 2500, seed=3)
    model = train_regime_model(X, hourly_index(len(X)), "SOL", "1h", cfg)
    names = [s.name for s in model.states_by_rank()]
    assert names == ["Crash", "Bajista", "Neutral", "Alcista", "Euforia"]
    assert [s.category for s in model.states_by_rank()] == [
        "crash",
        "bajista",
        "neutral",
        "alcista",
        "euforia",
    ]
    returns = [s.mean_return for s in model.states_by_rank()]
    assert returns == sorted(returns)


# --- metadatos, escalador y determinismo ---------------------------------------


def test_metadata_and_scaler(model3, small_cfg, data3):
    X, idx = data3
    assert model3.symbol == "BTC"
    assert model3.timeframe == "1h"
    assert model3.trained_at == FIXED_NOW
    assert model3.train_start == idx[0].to_pydatetime()
    assert model3.train_end == idx[-1].to_pydatetime()
    assert model3.train_start.tzinfo is not None
    assert model3.n_samples == len(X)
    assert model3.config_fingerprint == config_fingerprint(small_cfg, "1h")
    assert model3.synthetic is False
    assert model3.format_version == MODEL_FORMAT_VERSION
    # El escalador se ajusta solo con los datos de entrenamiento.
    np.testing.assert_allclose(model3.scaler.mean, X.mean(axis=0))
    np.testing.assert_allclose(model3.scaler.scale, X.std(axis=0))
    assert model3.scaler.feature_names == FEATURE_COLUMNS


def test_deterministic(model2, small_cfg, data2):
    X, idx = data2
    again = train_regime_model(X, idx, "ETH", "1h", small_cfg, clock=lambda: FIXED_NOW)
    for attr in ("startprob_", "transmat_", "means_", "covars_"):
        np.testing.assert_array_equal(getattr(again.hmm, attr), getattr(model2.hmm, attr))
    assert again.states == model2.states
    assert again.bic_by_n == model2.bic_by_n
    assert again.loglik_by_n == model2.loglik_by_n
    assert again.notes == model2.notes
    assert again.config_fingerprint == model2.config_fingerprint


def test_dataframe_input_uses_feature_columns_order(small_cfg):
    cfg = small_cfg.model_copy(update={"n_states_min": 2, "n_states_max": 3, "n_restarts": 1})
    X, _ = sample_source(2, 600, seed=4)
    idx = hourly_index(len(X))
    df = pd.DataFrame(X, columns=list(FEATURE_COLUMNS), index=idx)
    shuffled = df[["volume_z", "range_pct", "log_return", "realized_vol"]]
    from_df = train_regime_model(shuffled, idx, "XRP", "1h", cfg, synthetic=True)
    from_array = train_regime_model(X, idx, "XRP", "1h", cfg, synthetic=True)
    np.testing.assert_array_equal(from_df.hmm.means_, from_array.hmm.means_)
    assert from_df.synthetic is True


def test_training_never_classifies_with_lookahead_methods(monkeypatch, small_cfg):
    """Entrenar no usa predict/decode/predict_proba/score_samples."""

    def forbidden(*_a, **_k):
        raise AssertionError("método con look-ahead llamado durante el entrenamiento")

    for name in ("predict", "decode", "predict_proba", "score_samples"):
        monkeypatch.setattr(GaussianHMM, name, forbidden)
    cfg = small_cfg.model_copy(update={"n_states_min": 2, "n_states_max": 3, "n_restarts": 1})
    X, _ = sample_source(2, 500, seed=5)
    model = train_regime_model(X, hourly_index(len(X)), "AVAX", "1h", cfg)
    assert model.n_states in (2, 3)


# --- validaciones --------------------------------------------------------------


def test_rejects_non_finite_rows(small_cfg):
    X, _ = sample_source(2, 400, seed=6)
    X[10, 1] = np.nan
    X[20, 0] = np.inf
    with pytest.raises(RegimeTrainingError, match="2 filas"):
        train_regime_model(X, hourly_index(len(X)), "BTC", "1h", small_cfg)


def test_rejects_too_few_samples(small_cfg):
    X, _ = sample_source(2, 150, seed=7)
    with pytest.raises(RegimeTrainingError, match="al menos 200"):
        train_regime_model(X, hourly_index(len(X)), "BTC", "1h", small_cfg)


def test_rejects_bad_shape_and_index(small_cfg):
    X, _ = sample_source(2, 300, seed=8)
    with pytest.raises(RegimeTrainingError, match="matriz"):
        train_regime_model(X[:, :3], hourly_index(len(X)), "BTC", "1h", small_cfg)
    with pytest.raises(RegimeTrainingError, match="índice"):
        train_regime_model(X, hourly_index(len(X) - 1), "BTC", "1h", small_cfg)
    with pytest.raises(RegimeTrainingError, match="crecientes"):
        train_regime_model(X, hourly_index(len(X))[::-1], "BTC", "1h", small_cfg)
    with pytest.raises(RegimeTrainingError, match="faltan columnas"):
        df = pd.DataFrame(X[:, :3], columns=["log_return", "realized_vol", "range_pct"])
        train_regime_model(df, hourly_index(len(X)), "BTC", "1h", small_cfg)


def test_rejects_unknown_timeframe(small_cfg):
    X, _ = sample_source(2, 300, seed=9)
    with pytest.raises(ValueError, match="timeframe"):
        train_regime_model(X, hourly_index(len(X)), "BTC", "15m", small_cfg)


# --- avisos ----------------------------------------------------------------------


def test_notes_record_non_convergence(small_cfg):
    cfg = small_cfg.model_copy(
        update={"n_states_min": 2, "n_states_max": 2, "n_restarts": 2, "max_iter": 2, "tol": 1e-12}
    )
    X, _ = sample_source(2, 400, seed=10)
    model = train_regime_model(X, hourly_index(len(X)), "BTC", "1h", cfg)
    assert any(
        "2 de 2 intentos no terminaron de ajustarse en 2 iteraciones" in n for n in model.notes
    )
    assert any("El modelo elegido (2 estados) no terminó" in n for n in model.notes)


def test_converged_model_has_no_convergence_notes(model2):
    assert not any("no terminó" in n for n in model2.notes)


def test_skips_state_counts_with_more_parameters_than_samples(small_cfg):
    cfg = small_cfg.model_copy(
        update={
            "n_states_min": 2,
            "n_states_max": 6,
            "n_restarts": 1,
            "max_iter": 20,
            "min_training_samples": 50,
        }
    )
    X, _ = sample_source(2, 100, seed=11)
    # full, 4 features: 5 estados = 94 parámetros (< 100), 6 estados = 119 (>= 100).
    model = train_regime_model(X, hourly_index(len(X)), "BTC", "1h", cfg)
    assert 6 not in model.bic_by_n
    assert set(model.bic_by_n) <= {2, 3, 4, 5}
    assert any("No se probaron 6 estados" in n for n in model.notes)


def test_all_state_counts_skipped_raises(small_cfg):
    cfg = small_cfg.model_copy(
        update={"n_states_min": 6, "n_states_max": 7, "n_restarts": 1, "min_training_samples": 50}
    )
    X, _ = sample_source(2, 100, seed=12)
    with pytest.raises(RegimeTrainingError, match="ningún entrenamiento"):
        train_regime_model(X, hourly_index(len(X)), "BTC", "1h", cfg)


def test_failed_fits_are_skipped_and_recorded(monkeypatch, small_cfg):
    cfg = small_cfg.model_copy(update={"n_states_min": 2, "n_states_max": 3, "n_restarts": 2})
    real_fit = model_mod._fit_one

    def flaky(Z, n_states, seed, regime_cfg):
        if n_states == 3:
            raise np.linalg.LinAlgError("matriz singular")
        attempt = seed - regime_cfg.random_seed - 1000 * n_states
        if attempt == 1:  # segundo intento de n=2
            raise ValueError("falló")
        return real_fit(Z, n_states, seed, regime_cfg)

    monkeypatch.setattr(model_mod, "_fit_one", flaky)
    X, _ = sample_source(2, 400, seed=13)
    model = train_regime_model(X, hourly_index(len(X)), "BTC", "1h", cfg)
    assert model.n_states == 2
    assert set(model.bic_by_n) == {2}
    assert "2 estados: 1 de 2 intentos de entrenamiento fallaron." in model.notes
    assert "3 estados: 2 de 2 intentos de entrenamiento fallaron." in model.notes


def test_all_fits_failing_raises(monkeypatch, small_cfg):
    def broken(*_a, **_k):
        raise ValueError("todo mal")

    monkeypatch.setattr(model_mod, "_fit_one", broken)
    X, _ = sample_source(2, 400, seed=14)
    with pytest.raises(RegimeTrainingError, match="último error: ValueError: todo mal"):
        train_regime_model(X, hourly_index(len(X)), "BTC", "1h", small_cfg)


def test_max_states_choice_is_noted(small_cfg):
    cfg = small_cfg.model_copy(update={"n_states_min": 2, "n_states_max": 3, "n_restarts": 1})
    X, _ = sample_source(3, 1500, seed=15)
    model = train_regime_model(X, hourly_index(len(X)), "BTC", "1h", cfg)
    assert model.n_states == 3
    assert any("máximo de estados permitido" in n and "no es un problema" in n for n in model.notes)


# --- parámetros libres y huella de config -------------------------------------


@pytest.mark.parametrize("cov", ["full", "diag", "spherical", "tied"])
@pytest.mark.parametrize("n", [2, 3, 7])
def test_n_free_parameters_matches_hmmlearn(cov, n):
    hmm = GaussianHMM(n_components=n, covariance_type=cov)
    hmm.n_features = 4
    expected = sum(hmm._get_n_fit_scalars_per_param().values())
    assert n_free_parameters(n, 4, cov) == expected


def test_n_free_parameters_rejects_unknown_covariance():
    with pytest.raises(ValueError):
        n_free_parameters(3, 4, "weird")


def test_fingerprint_stable_and_sensitive(settings):
    cfg = settings.regime
    fp_1h = config_fingerprint(cfg, "1h")
    fp_1d = config_fingerprint(cfg, "1d")
    assert fp_1h == config_fingerprint(cfg.model_copy(deep=True), "1h")
    assert fp_1h != fp_1d
    assert len(fp_1h) == 16

    def fp(update, tf="1h"):
        return config_fingerprint(cfg.model_copy(update=update), tf)

    # Cambian el modelo.
    assert fp({"n_restarts": cfg.n_restarts + 1}) != fp_1h
    assert fp({"max_iter": cfg.max_iter + 1}) != fp_1h
    assert fp({"tol": cfg.tol * 10}) != fp_1h
    assert fp({"random_seed": cfg.random_seed + 1}) != fp_1h
    assert fp({"covariance_type": "diag"}) != fp_1h
    assert fp({"n_states_max": cfg.n_states_max - 1}) != fp_1h
    new_features = cfg.features.model_copy(update={"realized_vol_window": 48})
    assert fp({"features": new_features}) != fp_1h
    renamed = {**cfg.labels, 3: [lb.model_copy() for lb in cfg.labels[3]]}
    renamed[3][0] = RegimeLabel(name="Muy bajista", category="bajista")
    assert fp({"labels": renamed}) != fp_1h
    # El período de entrenamiento afecta solo a su timeframe.
    assert fp({"training_days": cfg.training_days + 10}) != fp_1h
    assert fp({"training_days": cfg.training_days + 10}, "1d") == fp_1d
    assert fp({"context_training_days": cfg.context_training_days + 10}, "1d") != fp_1d
    assert fp({"context_training_days": cfg.context_training_days + 10}) == fp_1h
    # No cambian el modelo.
    assert fp({"confirmation_bars": cfg.confirmation_bars + 1}) == fp_1h
    assert fp({"instability_max_changes": 9}) == fp_1h
    assert fp({"models_dir": "otra/carpeta"}) == fp_1h
    assert fp({"auto_retrain": not cfg.auto_retrain}) == fp_1h
    assert fp({"retrain_every_days": 30}) == fp_1h


def test_fingerprint_rejects_unknown_timeframe(settings):
    with pytest.raises(ValueError):
        config_fingerprint(settings.regime, "5m")
