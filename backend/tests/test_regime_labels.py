"""Etiquetas de los estados: orden, nombres, desempates, duración y ocupación.

Los HMM se arman a mano (sin entrenar) para controlar exactamente las medias.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from app.config import RegimeLabel
from app.regime.labels import (
    LabelingError,
    label_states,
    rank_order,
    regime_stats_table,
    stationary_distribution,
)
from app.regime.types import FeatureScaler, RegimeModel

# Escalador no trivial: las medias del HMM están en espacio estandarizado.
SCALER = FeatureScaler(
    mean=np.array([0.0002, 0.012, 0.015, 0.1]),
    scale=np.array([0.006, 0.008, 0.010, 1.1]),
)


def make_hmm(raw_means: list[list[float]], transmat: np.ndarray | None = None) -> GaussianHMM:
    """HMM con medias dadas en unidades reales (se pasan a espacio estandarizado)."""
    raw = np.asarray(raw_means, dtype=float)
    k = raw.shape[0]
    hmm = GaussianHMM(n_components=k, covariance_type="diag")
    hmm.n_features = raw.shape[1]
    hmm.startprob_ = np.full(k, 1.0 / k)
    if transmat is None:
        transmat = np.full((k, k), 0.1 / (k - 1))
        np.fill_diagonal(transmat, 0.9)
    hmm.transmat_ = transmat
    hmm.means_ = SCALER.transform(raw)
    hmm.covars_ = np.ones((k, raw.shape[1]))
    return hmm


def labels(settings, n: int) -> list[RegimeLabel]:
    return settings.regime.labels[n]


def test_five_states_shuffled_get_crash_to_euforia(settings):
    # Índices internos desordenados respecto del retorno.
    raw = [
        [0.004, 0.015, 0.018, 0.0],  # índice 0 -> Alcista
        [-0.020, 0.040, 0.050, 2.0],  # índice 1 -> Crash
        [0.020, 0.030, 0.035, 1.0],  # índice 2 -> Euforia
        [0.000, 0.008, 0.009, -0.5],  # índice 3 -> Neutral
        [-0.005, 0.020, 0.022, 0.5],  # índice 4 -> Bajista
    ]
    states = label_states(make_hmm(raw), SCALER, labels(settings, 5))
    assert [s.index for s in states] == [0, 1, 2, 3, 4]
    assert [s.name for s in states] == ["Alcista", "Crash", "Euforia", "Neutral", "Bajista"]
    assert [s.rank for s in states] == [3, 0, 4, 2, 1]
    assert [s.category for s in states] == ["alcista", "crash", "euforia", "neutral", "bajista"]
    by_rank = sorted(states, key=lambda s: s.rank)
    assert [s.name for s in by_rank] == ["Crash", "Bajista", "Neutral", "Alcista", "Euforia"]
    # Estadísticas en unidades reales.
    for s in states:
        assert s.mean_return == pytest.approx(raw[s.index][0], abs=1e-15)
        assert s.mean_volatility == pytest.approx(raw[s.index][1], abs=1e-15)
        assert s.mean_range == pytest.approx(raw[s.index][2], abs=1e-15)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_names_follow_config_for_every_state_count(settings, n):
    rets = np.linspace(-0.01, 0.01, n)[::-1]  # al revés: índice 0 = el más alcista
    raw = [[r, 0.01 + 0.001 * i, 0.012, 0.0] for i, r in enumerate(rets)]
    states = label_states(make_hmm(raw), SCALER, labels(settings, n))
    by_rank = sorted(states, key=lambda s: s.rank)
    assert [s.name for s in by_rank] == [lb.name for lb in labels(settings, n)]
    assert [s.category for s in by_rank] == [lb.category for lb in labels(settings, n)]
    returns = [s.mean_return for s in by_rank]
    assert all(a <= b for a, b in zip(returns, returns[1:], strict=False))


def test_tie_break_negative_side_more_volatile_is_lower(settings):
    raw = [
        [-0.010, 0.020, 0.02, 0.0],  # índice 0, menos volátil
        [-0.010, 0.040, 0.04, 0.0],  # índice 1, más volátil -> más extremo (abajo)
        [0.010, 0.015, 0.02, 0.0],
    ]
    states = label_states(make_hmm(raw), SCALER, labels(settings, 3))
    assert states[1].rank == 0
    assert states[0].rank == 1
    assert states[2].rank == 2


def test_tie_break_positive_side_more_volatile_is_higher(settings):
    raw = [
        [-0.010, 0.020, 0.02, 0.0],
        [0.010, 0.040, 0.04, 0.0],  # índice 1, más volátil -> más extremo (arriba)
        [0.010, 0.015, 0.02, 0.0],  # índice 2, menos volátil
    ]
    states = label_states(make_hmm(raw), SCALER, labels(settings, 3))
    assert states[0].rank == 0
    assert states[2].rank == 1
    assert states[1].rank == 2


def test_tie_break_zero_return_counts_as_non_positive():
    order = rank_order(np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.03, 0.02]))
    assert order == [1, 2, 0]  # el más volátil primero


def test_tie_tolerance_is_relative():
    vol = np.array([0.01, 0.05])
    # Diferencia relativa ~1e-14: empate -> manda la volatilidad (lado alcista).
    assert rank_order(np.array([0.01 * (1 + 1e-14), 0.01]), vol) == [0, 1]
    # Diferencia relativa 1e-6: no es empate -> manda el retorno.
    assert rank_order(np.array([0.01 * (1 + 1e-6), 0.01]), vol) == [1, 0]
    # Mismo retorno y misma volatilidad: desempata el índice.
    assert rank_order(np.array([0.01, 0.01]), np.array([0.02, 0.02])) == [0, 1]


def test_rank_order_rejects_bad_input():
    with pytest.raises(LabelingError):
        rank_order(np.array([0.0, np.nan]), np.array([0.01, 0.02]))
    with pytest.raises(LabelingError):
        rank_order(np.array([0.0, 0.1]), np.array([0.01]))


def test_label_count_must_match(settings):
    hmm = make_hmm([[-0.01, 0.02, 0.02, 0], [0.0, 0.01, 0.01, 0], [0.01, 0.02, 0.02, 0]])
    with pytest.raises(LabelingError, match="3 estados"):
        label_states(hmm, SCALER, labels(settings, 4))


def test_expected_duration_and_occupancy(settings):
    A = np.array([[0.9, 0.1, 0.0], [0.05, 0.9, 0.05], [0.0, 0.2, 0.8]])
    raw = [[-0.01, 0.02, 0.02, 0], [0.0, 0.01, 0.01, 0], [0.01, 0.02, 0.02, 0]]
    states = label_states(make_hmm(raw, A), SCALER, labels(settings, 3))
    assert [s.expected_duration_bars for s in states] == pytest.approx([10.0, 10.0, 5.0])
    occ = np.array([s.occupancy for s in states])
    np.testing.assert_allclose(occ, stationary_distribution(A))
    np.testing.assert_allclose(occ @ A, occ, atol=1e-12)
    assert occ.sum() == pytest.approx(1.0)


def test_absorbing_state_has_infinite_duration(settings):
    A = np.array([[1.0, 0.0, 0.0], [0.1, 0.8, 0.1], [0.0, 0.5, 0.5]])
    raw = [[-0.01, 0.02, 0.02, 0], [0.0, 0.01, 0.01, 0], [0.01, 0.02, 0.02, 0]]
    states = label_states(make_hmm(raw, A), SCALER, labels(settings, 3))
    assert math.isinf(states[0].expected_duration_bars)
    # A la larga todo termina en el estado absorbente.
    assert states[0].occupancy == pytest.approx(1.0, abs=1e-9)


# --- distribución estacionaria -------------------------------------------------


def test_stationary_two_state_closed_form():
    pi = stationary_distribution(np.array([[0.9, 0.1], [0.2, 0.8]]))
    np.testing.assert_allclose(pi, [2 / 3, 1 / 3])


def test_stationary_random_chain():
    rng = np.random.default_rng(0)
    A = rng.dirichlet(np.ones(6), size=6)
    pi = stationary_distribution(A)
    assert (pi >= 0).all()
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ A, pi, atol=1e-12)


@pytest.mark.parametrize(
    "A",
    [
        np.eye(3),  # reducible: cualquier distribución sirve
        np.array([[0.0, 1.0], [1.0, 0.0]]),  # periódica
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_stationary_degenerate_chains_give_valid_distribution(A):
    pi = stationary_distribution(A)
    assert (pi >= 0).all()
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ A, pi, atol=1e-9)


def test_stationary_rejects_invalid_matrices():
    with pytest.raises(LabelingError):
        stationary_distribution(np.ones((2, 3)) / 3)
    with pytest.raises(LabelingError):
        stationary_distribution(np.array([[0.5, 0.4], [0.5, 0.5]]))
    with pytest.raises(LabelingError):
        stationary_distribution(np.array([[1.2, -0.2], [0.5, 0.5]]))
    with pytest.raises(LabelingError):
        stationary_distribution(np.array([[np.nan, 1.0], [0.5, 0.5]]))


# --- tabla por régimen ---------------------------------------------------------


def _model_from(hmm: GaussianHMM, settings, n: int) -> RegimeModel:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    return RegimeModel(
        symbol="BTC",
        timeframe="1h",
        hmm=hmm,
        scaler=SCALER,
        states=label_states(hmm, SCALER, labels(settings, n)),
        n_states=n,
        bic_by_n={n: 0.0},
        loglik_by_n={n: 0.0},
        trained_at=t,
        train_start=t,
        train_end=t,
        n_samples=1000,
        config_fingerprint="x",
    )


def test_regime_stats_table_ordered_by_rank(settings):
    raw = [
        [0.003, 0.012, 0.015, 0.0],
        [-0.004, 0.025, 0.030, 1.0],
        [0.0005, 0.007, 0.008, -0.4],
    ]
    model = _model_from(make_hmm(raw), settings, 3)
    rows = regime_stats_table(model)
    assert [r["name"] for r in rows] == ["Bajista", "Neutral", "Alcista"]
    assert [r["rank"] for r in rows] == [0, 1, 2]
    assert [r["index"] for r in rows] == [1, 2, 0]
    assert set(rows[0]) == {
        "rank",
        "index",
        "name",
        "category",
        "mean_return_pct",
        "mean_volatility_pct",
        "mean_range_pct",
        "expected_duration_bars",
        "occupancy_pct",
    }
    assert rows[0]["mean_return_pct"] == pytest.approx(-0.4)
    assert rows[0]["mean_volatility_pct"] == pytest.approx(2.5)
    assert rows[0]["mean_range_pct"] == pytest.approx(3.0)
    assert rows[0]["expected_duration_bars"] == pytest.approx(10.0)
    assert sum(r["occupancy_pct"] for r in rows) == pytest.approx(100.0)


def test_regime_stats_table_infinite_duration_is_none(settings):
    A = np.array([[1.0, 0.0, 0.0], [0.1, 0.8, 0.1], [0.0, 0.5, 0.5]])
    raw = [[-0.01, 0.02, 0.02, 0], [0.0, 0.01, 0.01, 0], [0.01, 0.02, 0.02, 0]]
    rows = regime_stats_table(_model_from(make_hmm(raw, A), settings, 3))
    assert rows[0]["expected_duration_bars"] is None
    assert rows[1]["expected_duration_bars"] == pytest.approx(5.0)


def test_transmat_by_rank_matches_labels(settings):
    A = np.array([[0.7, 0.2, 0.1], [0.05, 0.9, 0.05], [0.3, 0.1, 0.6]])
    raw = [[0.01, 0.02, 0.02, 0], [-0.01, 0.01, 0.01, 0], [0.0, 0.02, 0.02, 0]]
    model = _model_from(make_hmm(raw, A), settings, 3)
    order = [s.index for s in model.states_by_rank()]
    assert order == [1, 2, 0]
    np.testing.assert_allclose(model.transmat_by_rank(), A[np.ix_(order, order)])
