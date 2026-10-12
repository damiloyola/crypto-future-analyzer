"""Forward filtering del régimen: cero look-ahead, exactitud y robustez.

El test obligatorio del Hito 3 es ``test_zero_lookahead_regime_of_bar_t_never_changes``:
el régimen de la vela *t* no puede cambiar al agregar (o cambiar) velas
posteriores. Para mostrar por qué ``predict_proba`` está prohibido, otro test
verifica que la probabilidad *suavizada* de una vela vieja sí cambia cuando
llegan velas nuevas.

Acá ``hmm.predict_proba`` y ``hmm.score`` se usan solo como oráculo para
comparar resultados; la app nunca los usa para clasificar velas.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

from app.regime.forward import (
    FilterResult,
    filtered_from_model,
    filtered_probabilities,
    log_emission,
    model_covariances,
)
from app.regime.model import train_regime_model
from app.regime.types import FEATURE_COLUMNS, FeatureScaler, RegimeModel, StateInfo

T_BARS = 400


def _spd(rng: np.random.Generator, f: int, scale: float) -> np.ndarray:
    m = rng.normal(size=(f, f))
    return scale * (m @ m.T / f + 0.5 * np.eye(f))


def _known_hmm(n_states: int = 3, n_features: int = 4, seed: int = 0) -> GaussianHMM:
    """GaussianHMM con parámetros conocidos (sin entrenar), estados que se solapan."""
    rng = np.random.default_rng(seed)
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", init_params="", params="")
    hmm.startprob_ = np.full(n_states, 1.0 / n_states)
    a = rng.uniform(size=(n_states, n_states)) + np.eye(n_states) * 12.0
    hmm.transmat_ = a / a.sum(axis=1, keepdims=True)
    hmm.means_ = rng.normal(scale=0.8, size=(n_states, n_features))
    hmm.covars_ = np.stack([_spd(rng, n_features, 0.5 + 0.7 * k) for k in range(n_states)])
    hmm.n_features = n_features  # hmmlearn lo fija al validar; acá lo necesita covars_
    return hmm


def _filter(hmm: GaussianHMM, Z: np.ndarray) -> FilterResult:
    return filtered_probabilities(Z, hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)


@pytest.fixture(scope="module")
def hmm_and_data() -> tuple[GaussianHMM, np.ndarray]:
    hmm = _known_hmm()
    Z, _ = hmm.sample(T_BARS, random_state=7)
    return hmm, Z


# --- cero look-ahead (obligatorio) -------------------------------------------------


def test_zero_lookahead_regime_of_bar_t_never_changes(hmm_and_data):
    """El régimen de la vela t no cambia al agregar velas posteriores."""
    hmm, Z = hmm_and_data
    full = _filter(hmm, Z)
    ks = sorted({1, 2, 3, 5, 10, 50, 199, 200, 201, T_BARS - 1, *range(7, T_BARS, 13)})
    for k in ks:
        prefix = _filter(hmm, Z[:k])
        assert prefix.probs.shape == (k, 3)
        np.testing.assert_allclose(prefix.probs, full.probs[:k], rtol=0, atol=1e-12)
        assert np.array_equal(prefix.states, full.states[:k]), f"cambió el régimen (k={k})"
        np.testing.assert_allclose(prefix.confidence, full.confidence[:k], rtol=0, atol=1e-12)


def test_zero_lookahead_live_bar_by_bar_equals_history(hmm_and_data):
    """Lo que se ve "en vivo" en cada vela (solo datos hasta ella) es lo mismo que la historia."""
    hmm, Z = hmm_and_data
    full = _filter(hmm, Z)
    live = np.array([_filter(hmm, Z[: t + 1]).probs[-1] for t in range(150)])
    np.testing.assert_allclose(live, full.probs[:150], rtol=0, atol=1e-12)


def test_zero_lookahead_future_bars_can_be_anything(hmm_and_data):
    """Cambiar por completo las velas posteriores (incluso por outliers) no toca el pasado."""
    hmm, Z = hmm_and_data
    full = _filter(hmm, Z)
    rng = np.random.default_rng(3)
    for k in (40, 160, 333):
        other = Z.copy()
        other[k:] = rng.normal(scale=25.0, size=other[k:].shape)
        other[-1] = 1e8
        changed = _filter(hmm, other)
        np.testing.assert_allclose(changed.probs[:k], full.probs[:k], rtol=0, atol=1e-12)
        assert np.array_equal(changed.states[:k], full.states[:k])


def test_contrast_smoothed_posterior_of_old_bar_changes_with_future(hmm_and_data):
    """Por qué está prohibido predict_proba: la probabilidad suavizada de una vela vieja
    cambia cuando se agregan velas nuevas (mira el futuro). La filtrada no."""
    hmm, Z = hmm_and_data
    full_smoothed = hmm.predict_proba(Z)
    diffs = [
        np.abs(hmm.predict_proba(Z[:k]) - full_smoothed[:k]).max() for k in range(20, T_BARS, 40)
    ]
    assert max(diffs) > 0.05
    # Y la filtrada es distinta de la suavizada sobre la serie completa.
    assert np.abs(_filter(hmm, Z).probs - full_smoothed).max() > 0.05


def test_contrast_smoothed_rewrites_the_regime_of_an_ambiguous_bar():
    """Vela 10 dudosa después de 10 velas bajistas: con lo que se sabía en ese momento era
    "bajista". Si después llegan velas alcistas, predict_proba la reescribe como "alcista"
    (look-ahead); el forward filtering la deja como estaba."""
    hmm = GaussianHMM(n_components=2, covariance_type="full", init_params="", params="")
    hmm.startprob_ = np.array([0.5, 0.5])
    hmm.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    hmm.means_ = np.array([[-1.0], [1.0]])  # 0 = bajista, 1 = alcista
    hmm.covars_ = np.array([[[1.0]], [[1.0]]])
    hmm.n_features = 1
    Z = np.array([-1.0] * 10 + [0.3] + [2.0] * 10)[:, None]

    assert hmm.predict_proba(Z[:11])[10].argmax() == 0
    assert hmm.predict_proba(Z)[10].argmax() == 1  # el futuro cambió el pasado

    assert _filter(hmm, Z[:11]).states[10] == 0
    assert _filter(hmm, Z).states[10] == 0  # filtrado: no cambia


# --- oráculos ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 5, 37, 200, T_BARS])
def test_last_filtered_row_equals_last_smoothed_row(hmm_and_data, k):
    """En la última vela, suavizada y filtrada coinciden (no hay futuro que mirar)."""
    hmm, Z = hmm_and_data
    res = _filter(hmm, Z[:k])
    np.testing.assert_allclose(res.probs[-1], hmm.predict_proba(Z[:k])[-1], rtol=0, atol=1e-9)


def test_log_likelihood_matches_hmmlearn_score(hmm_and_data):
    hmm, Z = hmm_and_data
    res = _filter(hmm, Z)
    assert res.log_likelihood == pytest.approx(hmm.score(Z), rel=1e-10)
    assert res.n_degenerate == 0
    assert _filter(hmm, Z[:1]).log_likelihood == pytest.approx(hmm.score(Z[:1]), rel=1e-10)


def test_log_emission_matches_scipy(hmm_and_data):
    hmm, Z = hmm_and_data
    got = log_emission(Z, hmm.means_, hmm.covars_)
    assert got.shape == (T_BARS, 3)
    for k in range(3):
        expected = multivariate_normal(mean=hmm.means_[k], cov=hmm.covars_[k]).logpdf(Z)
        np.testing.assert_allclose(got[:, k], expected, rtol=1e-10, atol=1e-10)


def test_result_properties_and_rows_sum_to_one(hmm_and_data):
    hmm, Z = hmm_and_data
    res = _filter(hmm, Z)
    assert res.n_bars == T_BARS and res.n_states == 3
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert (res.probs >= 0).all()
    assert np.array_equal(res.states, res.probs.argmax(axis=1))
    assert ((res.confidence >= 1 / 3 - 1e-12) & (res.confidence <= 1)).all()
    assert not res.probs.flags.writeable


def test_persistent_states_are_recovered(hmm_and_data):
    """Sanidad: con estados bien separados, el régimen filtrado acierta casi siempre."""
    hmm = _known_hmm(seed=4)
    hmm.means_ = np.array([[-3.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 3.0, 3.0]])
    Z, truth = hmm.sample(300, random_state=11)
    res = _filter(hmm, Z)
    assert (res.states == truth).mean() > 0.95


# --- robustez ----------------------------------------------------------------------


def test_huge_outliers_never_nan_and_go_to_widest_state(hmm_and_data):
    """Un outlier enorme va al estado más "ancho" en esa dirección (menor u' Σ⁻¹ u)."""
    hmm, Z = hmm_and_data
    u = np.ones(4)
    widest = int(np.argmin([u @ np.linalg.solve(c, u) for c in hmm.covars_]))
    for magnitude in (1e3, 1e6, 1e12, 1e100):
        bad = Z[:60].copy()
        bad[30] = magnitude
        res = _filter(hmm, bad)
        assert np.isfinite(res.probs).all()
        np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert res.states[30] == widest
        assert np.isfinite(res.log_likelihood)
        assert res.n_degenerate == 0


def test_observation_beyond_float_range_uses_the_limit():
    """Densidad -inf en todos los estados: gana el de menor distancia de Mahalanobis."""
    hmm = _known_hmm(n_features=2, seed=5)
    hmm.means_ = np.zeros((3, 2))
    hmm.covars_ = np.stack([np.eye(2) * v for v in (1.0, 9.0, 4.0)])
    Z = np.zeros((5, 2))
    Z[2] = [1e200, -1e200]
    res = _filter(hmm, Z)
    assert np.isfinite(res.probs).all()
    assert res.states[2] == 1  # la covarianza más ancha
    assert res.probs[2, 1] == pytest.approx(1.0)
    assert res.n_degenerate == 1
    assert res.log_likelihood == -np.inf
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_observation_only_compatible_with_unreachable_state_trusts_the_data():
    """Si la vela solo encaja en un estado al que no se puede llegar, se confía en la vela."""
    means = np.array([[0.0], [0.0]])
    covars = np.array([[[1.0]], [[100.0]]])
    startprob = np.array([1.0, 0.0])
    transmat = np.eye(2)  # el estado 1 es inalcanzable
    Z = np.array([[0.1], [1e160], [0.2]])
    res = filtered_probabilities(Z, startprob, transmat, means, covars)
    assert np.isfinite(res.probs).all()
    assert res.states.tolist() == [0, 1, 1]
    assert res.n_degenerate == 1
    assert res.log_likelihood == -np.inf


def test_near_singular_and_singular_covariances():
    rng = np.random.default_rng(9)
    f = 4
    v = rng.normal(size=(f, 1))
    near = v @ v.T + 1e-13 * np.eye(f)  # casi de rango 1
    singular = np.zeros((f, f))
    singular[0, 0] = 1.0  # exactamente singular
    means = rng.normal(size=(3, f))
    covars = np.stack([near, singular, np.eye(f)])
    Z = rng.normal(size=(80, f))
    res = filtered_probabilities(Z, np.full(3, 1 / 3), np.full((3, 3), 1 / 3), means, covars)
    assert np.isfinite(res.probs).all()
    np.testing.assert_allclose(res.probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.isfinite(res.log_likelihood)
    le = log_emission(Z, means, covars)
    assert not np.isnan(le).any()


def test_single_state_is_always_certain():
    rng = np.random.default_rng(2)
    means = np.array([[0.5, -0.5]])
    covars = np.array([[[1.0, 0.3], [0.3, 2.0]]])
    Z = rng.normal(size=(50, 2))
    res = filtered_probabilities(Z, [1.0], [[1.0]], means, covars)
    assert np.array_equal(res.probs, np.ones((50, 1)))
    assert res.states.tolist() == [0] * 50
    expected = multivariate_normal(mean=means[0], cov=covars[0]).logpdf(Z).sum()
    assert res.log_likelihood == pytest.approx(expected, rel=1e-12)


def test_single_bar_matches_bayes_rule(hmm_and_data):
    hmm, Z = hmm_and_data
    res = _filter(hmm, Z[:1])
    dens = np.array(
        [multivariate_normal(hmm.means_[k], hmm.covars_[k]).pdf(Z[0]) for k in range(3)]
    )
    expected = hmm.startprob_ * dens / np.sum(hmm.startprob_ * dens)
    np.testing.assert_allclose(res.probs[0], expected, rtol=1e-10)


def test_zero_bars_returns_empty_result(hmm_and_data):
    hmm, _ = hmm_and_data
    res = _filter(hmm, np.empty((0, 4)))
    assert res.probs.shape == (0, 3)
    assert res.log_likelihood == 0.0
    assert res.states.shape == (0,) and res.confidence.shape == (0,)


def test_zero_transition_probabilities_are_supported():
    hmm = GaussianHMM(n_components=3, covariance_type="full", init_params="", params="")
    hmm.startprob_ = np.array([1.0, 0.0, 0.0])
    hmm.transmat_ = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]])
    hmm.means_ = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, 1.0]])
    hmm.covars_ = np.stack([np.eye(2)] * 3)
    Z, _ = hmm.sample(120, random_state=3)
    res = _filter(hmm, Z)
    assert res.log_likelihood == pytest.approx(hmm.score(Z), rel=1e-10)
    np.testing.assert_allclose(res.probs[-1], hmm.predict_proba(Z)[-1], atol=1e-9)
    assert res.probs[0, 1] == 0.0 and res.probs[0, 2] == 0.0  # arranca en el estado 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"Z": np.array([[np.nan, 0.0]])}, "no finitos"),
        ({"Z": np.zeros((3, 3))}, "matriz"),
        ({"startprob": [0.7, 0.7]}, "suman"),
        ({"startprob": [1.2, -0.2]}, "negativos"),
        ({"transmat": [[0.5, 0.4], [0.5, 0.5]]}, "suman"),
        ({"transmat": np.eye(3)}, "forma"),
        ({"covars": np.ones((2, 2))}, "completas"),
        ({"covars": -np.stack([np.eye(2)] * 2)}, "definida positiva"),
        ({"means": np.array([[0.0, np.inf], [0.0, 0.0]])}, "no finitos"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    args = {
        "Z": np.zeros((3, 2)),
        "startprob": [0.5, 0.5],
        "transmat": [[0.9, 0.1], [0.1, 0.9]],
        "means": np.zeros((2, 2)),
        "covars": np.stack([np.eye(2)] * 2),
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        filtered_probabilities(**args)


# --- desde un RegimeModel ------------------------------------------------------------


def _regime_model(hmm: GaussianHMM, scaler: FeatureScaler) -> RegimeModel:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    states = tuple(
        StateInfo(
            index=k,
            rank=k,
            name=f"Estado {k}",
            category="neutral",
            mean_return=0.0,
            mean_volatility=0.01,
            mean_range=0.01,
            expected_duration_bars=10.0,
            occupancy=1 / 3,
        )
        for k in range(hmm.n_components)
    )
    return RegimeModel(
        symbol="BTC",
        timeframe="1h",
        hmm=hmm,
        scaler=scaler,
        states=states,
        n_states=hmm.n_components,
        bic_by_n={3: 0.0},
        loglik_by_n={3: 0.0},
        trained_at=now,
        train_start=now,
        train_end=now,
        n_samples=100,
        config_fingerprint="test",
    )


def test_filtered_from_model_standardizes_with_training_scaler(hmm_and_data):
    hmm, Z = hmm_and_data
    scaler = FeatureScaler(
        mean=np.array([1e-4, 0.01, 0.02, 0.0]), scale=np.array([0.01, 0.005, 0.01, 1.0])
    )
    model = _regime_model(hmm, scaler)
    X_raw = scaler.inverse_transform(Z)  # features en unidades reales
    expected = _filter(hmm, Z)
    got = filtered_from_model(model, X_raw)
    np.testing.assert_allclose(got.probs, expected.probs, rtol=0, atol=1e-9)
    assert got.log_likelihood == pytest.approx(expected.log_likelihood, rel=1e-9)

    # DataFrame con las columnas desordenadas y una extra: se eligen por nombre.
    df = pd.DataFrame(X_raw, columns=list(FEATURE_COLUMNS))
    df["otra"] = 123.0
    df = df[["otra", *reversed(FEATURE_COLUMNS)]]
    got_df = filtered_from_model(model, df)
    np.testing.assert_allclose(got_df.probs, expected.probs, rtol=0, atol=1e-9)


def test_filtered_from_model_rejects_bad_features(hmm_and_data):
    hmm, Z = hmm_and_data
    model = _regime_model(hmm, FeatureScaler(mean=np.zeros(4), scale=np.ones(4)))
    with pytest.raises(ValueError, match="faltan"):
        filtered_from_model(model, pd.DataFrame(Z[:, :3], columns=list(FEATURE_COLUMNS[:3])))
    bad = Z[:10].copy()
    bad[0, 1] = np.nan  # calentamiento de la volatilidad realizada
    with pytest.raises(ValueError, match="calentamiento"):
        filtered_from_model(model, bad)
    with pytest.raises(ValueError, match="matriz"):
        filtered_from_model(model, Z[:, :2])


@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical", "tied"])
def test_every_covariance_type_filters_like_hmmlearn(settings, covariance_type):
    # Regresión: con "spherical" hmmlearn 0.3.3 devuelve K·F matrices en
    # ``covars_`` y el filtro fallaba con todos los modelos entrenados así.
    rng = np.random.default_rng(3)
    n = 600
    X = np.column_stack(
        [
            np.concatenate([rng.normal(-0.004, 0.004, n // 2), rng.normal(0.002, 0.012, n // 2)]),
            np.abs(rng.normal(0.01, 0.003, n)),
            np.abs(rng.normal(0.012, 0.004, n)),
            rng.normal(0.0, 1.0, n),
        ]
    )
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    cfg = settings.regime.model_copy(
        update={
            "covariance_type": covariance_type,
            "n_states_min": 3,
            "n_states_max": 3,
            "n_restarts": 1,
            "max_iter": 20,
            "min_training_samples": 100,
            # X ya son las features (sin suavizar): si no, el retorno medio por
            # vela de cada estado saldría dividido por 24.
            "features": settings.regime.features.model_copy(update={"smoothing_window": 1}),
        }
    )
    model = train_regime_model(X, index, "BTC", "1h", cfg)
    covars = model_covariances(model.hmm)
    assert covars.shape == (3, 4, 4)
    got = filtered_from_model(model, X)
    Z = model.scaler.transform(X)
    oracle = model.hmm.predict_proba(Z)  # solo como oráculo de la última fila
    np.testing.assert_allclose(got.probs[-1], oracle[-1], rtol=0, atol=1e-8)
    assert got.log_likelihood == pytest.approx(model.hmm.score(Z), rel=1e-9)
