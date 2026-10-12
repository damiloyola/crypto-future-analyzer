"""Filtro de estabilidad: confirmación tras N velas, inestabilidad, antigüedad y candidato.

Los umbrales se toman de ``settings.regime`` (confirmation_bars=3,
instability_window=20, instability_max_changes=4 en config.yaml) para que los
casos a mano reflejen las reglas reales.
"""

import random

import numpy as np
import pytest

from app.regime.stability import (
    StabilitySummary,
    changes_in_window,
    confirmed_series,
    is_unstable,
    pending,
    regime_age,
    summarize_stability,
    unstable_series,
)

N = None


@pytest.fixture(scope="module")
def rules(settings):
    r = settings.regime
    return r.confirmation_bars, r.instability_window, r.instability_max_changes


def test_config_rules_are_the_ones_these_cases_assume(rules):
    assert rules == (3, 20, 4)


# --- confirmación --------------------------------------------------------------------


def test_startup_nothing_confirmed_until_three_bars(rules):
    bars = rules[0]
    assert confirmed_series([], bars) == []
    assert confirmed_series(["A"], bars) == [N]
    assert confirmed_series(["A", "A"], bars) == [N, N]
    assert confirmed_series(["A", "A", "A"], bars) == [N, N, "A"]
    assert confirmed_series(["A", "A", "A", "A"], bars) == [N, N, "A", "A"]


def test_new_regime_confirmed_exactly_at_third_bar(rules):
    bars = rules[0]
    raw = ["A", "A", "A", "B", "B", "B", "B"]
    assert confirmed_series(raw, bars) == [N, N, "A", "A", "A", "B", "B"]


def test_two_bars_are_not_enough(rules):
    bars = rules[0]
    raw = ["A", "A", "A", "B", "B", "A", "A"]
    assert confirmed_series(raw, bars) == [N, N, "A", "A", "A", "A", "A"]


def test_flicker_never_confirms(rules):
    bars = rules[0]
    assert confirmed_series(["A", "B"] * 5, bars) == [N] * 10
    # Con un régimen ya confirmado, el parpadeo no lo reemplaza.
    raw = ["A", "A", "A", "B", "C", "B", "C", "B", "C"]
    assert confirmed_series(raw, bars) == [N, N] + ["A"] * 7


def test_return_to_previous_regime_after_flicker(rules):
    bars = rules[0]
    raw = ["A", "A", "A", "B", "B", "B", "A", "B", "A", "A", "A"]
    expected = [N, N, "A", "A", "A", "B", "B", "B", "B", "B", "A"]
    assert confirmed_series(raw, bars) == expected


def test_bars_one_is_identity_and_bars_must_be_positive():
    raw = [0, 1, 1, 2, 0]
    assert confirmed_series(raw, 1) == raw
    with pytest.raises(ValueError):
        confirmed_series(raw, 0)


def test_none_breaks_runs_and_is_never_confirmed():
    raw = ["A", "A", N, "A", "A", "A", N, N, N, "B"]
    assert confirmed_series(raw, 3) == [N, N, N, N, N, "A", "A", "A", "A", "A"]


def test_numpy_labels_become_plain_python():
    out = confirmed_series(np.array([2, 2, 2, 3]), 3)
    assert out == [N, N, 2, 2]
    assert type(out[2]) is int


# --- cambios en la ventana / inestabilidad ------------------------------------------------


def _with_changes(n_changes: int, length: int = 20) -> list[str]:
    """Secuencia de ``length`` velas con exactamente ``n_changes`` cambios, al final."""
    labels = ["A", "B"]
    seq = ["A"] * (length - n_changes)
    for i in range(n_changes):
        seq.append(labels[(i + 1) % 2])
    return seq


def test_exactly_four_changes_is_stable_five_is_unstable(rules):
    _, window, max_changes = rules
    four = _with_changes(4)
    five = _with_changes(5)
    assert changes_in_window(four, window) == 4
    assert changes_in_window(five, window) == 5
    assert is_unstable(four, window, max_changes) is False
    assert is_unstable(five, window, max_changes) is True


def test_window_has_window_minus_one_comparisons(rules):
    _, window, _ = rules
    # El cambio entre la vela anterior a la ventana y la primera de la ventana no cuenta.
    raw = ["A"] + ["B"] * window
    assert changes_in_window(raw, window) == 0
    raw = ["A"] * 2 + ["B"] * (window - 1)
    assert changes_in_window(raw, window) == 1
    # Cambios viejos (fuera de la ventana) no cuentan.
    old_noise = ["A", "B"] * 10
    assert changes_in_window(old_noise + ["C"] * window, window) == 0


def test_short_series_uses_all_bars(rules):
    _, window, max_changes = rules
    assert changes_in_window([], window) == 0
    assert changes_in_window(["A"], window) == 0
    assert changes_in_window(["A", "B", "A"], window) == 2
    assert is_unstable([], window, max_changes) is False
    with pytest.raises(ValueError):
        changes_in_window(["A"], 0)


def test_none_does_not_count_as_change():
    assert changes_in_window(["A", N, "B", "B", N, N, "C"], 20) == 0
    assert changes_in_window(["A", "B", N, "B"], 20) == 1


def test_unstable_series_matches_last_bar_rule_on_every_prefix(rules):
    _, window, max_changes = rules
    rng = random.Random(5)
    for _ in range(30):
        raw = [rng.choice("ABC") if rng.random() < 0.4 else "A" for _ in range(70)]
        series = unstable_series(raw, window, max_changes)
        assert len(series) == len(raw)
        for t in range(len(raw)):
            assert series[t] == is_unstable(raw[: t + 1], window, max_changes)


def test_unstable_series_hand_case(rules):
    _, window, max_changes = rules
    raw = ["A"] * 15 + ["B", "A", "B", "A", "B"] + ["A"] * 25
    series = unstable_series(raw, window, max_changes)
    # Cambios en las velas 15, 16, 17, 18 y 19.
    assert series.index(True) == 19  # en la vela 19 la ventana ya junta 5 cambios
    assert series[18] is False  # 4 cambios: todavía no es inestable
    # Cuando el parpadeo sale de la ventana, vuelve a ser estable.
    assert series[-1] is False
    assert unstable_series([], window, max_changes) == []


# --- antigüedad y candidato ---------------------------------------------------------


def test_regime_age():
    assert regime_age([]) == 0
    assert regime_age([N, N]) == 0
    assert regime_age([N, N, "A"]) == 1
    assert regime_age([N, N, "A", "A", "A"]) == 3
    assert regime_age([N, N, "A", "A", "B"]) == 1
    assert regime_age(["A", "B", "B", "B", "B"]) == 4


def test_pending_candidate(rules):
    bars = rules[0]
    raw = ["A", "A", "A", "B", "B"]
    conf = confirmed_series(raw, bars)
    assert pending(raw, conf) == ("B", 2)
    raw2 = raw + ["B"]
    assert pending(raw2, confirmed_series(raw2, bars)) is None  # ya se confirmó
    assert pending(["A", "A"], [N, N]) == ("A", 2)  # arranque
    assert pending([], []) is None
    assert pending(["A", "A", "A", N], [N, N, "A", "A"]) is None
    with pytest.raises(ValueError):
        pending(["A"], [])


def test_age_after_confirmation_and_offset_to_real_start(rules):
    """La antigüedad cuenta desde la confirmación; el arranque real fue bars-1 velas antes."""
    bars = rules[0]
    raw = ["A"] * 5 + ["B"] * 6
    conf = confirmed_series(raw, bars)
    age = regime_age(conf)
    assert age == 4
    assert age + bars - 1 == 6  # velas seguidas en "B"


# --- causalidad ---------------------------------------------------------------------


def test_stability_functions_are_causal(rules):
    """Agregar velas nuevas nunca cambia lo calculado para velas viejas."""
    bars, window, max_changes = rules
    rng = random.Random(11)
    for _ in range(25):
        raw = [rng.choice(["crash", "bajista", "neutral", "alcista"]) for _ in range(60)]
        conf_full = confirmed_series(raw, bars)
        unst_full = unstable_series(raw, window, max_changes)
        for k in range(len(raw) + 1):
            assert confirmed_series(raw[:k], bars) == conf_full[:k]
            assert unstable_series(raw[:k], window, max_changes) == unst_full[:k]


# --- resumen ------------------------------------------------------------------------


def test_summarize_stability(rules):
    bars, window, max_changes = rules
    raw = ["neutral"] * 10 + ["bajista", "bajista"]
    s = summarize_stability(raw, bars=bars, window=window, max_changes=max_changes)
    assert isinstance(s, StabilitySummary)
    assert s.raw == "bajista"
    assert s.confirmed == "neutral"
    assert s.is_confirmed is False
    assert s.pending == ("bajista", 2)
    # "neutral" se confirmó en la vela bars-1 y sigue vigente hasta la última.
    assert s.age_bars == len(raw) - (bars - 1)
    assert s.changes_in_window == 1
    assert s.unstable is False
    assert (s.window, s.max_changes) == (window, max_changes)

    flicker = ["neutral"] * 10 + ["bajista", "neutral"] * 3
    s2 = summarize_stability(flicker, bars=bars, window=window, max_changes=max_changes)
    assert s2.unstable is True
    assert s2.changes_in_window == 6
    assert s2.confirmed == "neutral" and s2.is_confirmed is True and s2.pending is None

    empty = summarize_stability([], bars=bars, window=window, max_changes=max_changes)
    assert empty.raw is None and empty.confirmed is None and empty.is_confirmed is False
    assert empty.age_bars == 0 and empty.pending is None and empty.unstable is False
