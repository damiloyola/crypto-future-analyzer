"""Hito 2: sizing por riesgo en USD (riesgo primero, apalancamiento después)."""

import math

import numpy as np
import pytest

from app.risk.sizing import (
    PositionSizing,
    ensure_rate,
    ensure_side,
    fraction_to_pct,
    leverage_for,
    margin_for_leverage,
    pct_to_fraction,
    position_size_usd,
    quantity_for,
    side_sign,
    size_position,
    stop_distance_pct,
)

FEE = 0.0006  # 0,06 % como fracción


def _random_trade(rng: np.random.Generator) -> tuple[str, float, float, float]:
    """Operación al azar válida: (lado, entrada, stop, riesgo)."""
    side = "long" if rng.random() < 0.5 else "short"
    entry = float(10 ** rng.uniform(-1, 5))  # 0.1 a 100 000
    dist = float(rng.uniform(0.0005, 0.2))
    stop = entry * (1 - dist) if side == "long" else entry * (1 + dist)
    risk = float(rng.uniform(1, 500))
    return side, entry, stop, risk


# --- conversiones y fórmulas básicas -------------------------------------------


def test_pct_fraction_conversions():
    assert pct_to_fraction(0.06) == pytest.approx(0.0006)
    assert pct_to_fraction(0.5) == pytest.approx(0.005)
    assert fraction_to_pct(0.0006) == pytest.approx(0.06)
    assert fraction_to_pct(pct_to_fraction(1.23)) == pytest.approx(1.23)


def test_stop_distance_pct_is_a_fraction():
    assert stop_distance_pct(60000, 59400) == pytest.approx(0.01)
    assert stop_distance_pct(60000, 60600) == pytest.approx(0.01)
    assert stop_distance_pct(2.0, 1.9) == pytest.approx(0.05)


def test_user_formula_example():
    # Entrada 60 000, stop 59 400 (1 %), riesgo 50 USD -> tamaño 5000 USD.
    assert position_size_usd(50, 60000, 59400) == pytest.approx(5000)
    assert position_size_usd(50, 60000, 60600) == pytest.approx(5000)


def test_leverage_is_an_output():
    size = position_size_usd(50, 60000, 59400)
    assert leverage_for(size, 100) == pytest.approx(50)
    # Más margen, mismo tamaño (mismo riesgo), menos apalancamiento.
    assert leverage_for(size, 200) == pytest.approx(25)
    assert margin_for_leverage(size, 50) == pytest.approx(100)
    assert margin_for_leverage(size, 10) == pytest.approx(500)


def test_quantity_for():
    assert quantity_for(5000, 60000) == pytest.approx(5000 / 60000)
    assert quantity_for(100, 2.5) == pytest.approx(40)


def test_leverage_margin_roundtrip_random():
    rng = np.random.default_rng(7)
    for _ in range(300):
        size = float(rng.uniform(1, 1e6))
        lev = float(rng.uniform(0.5, 100))
        margin = margin_for_leverage(size, lev)
        assert leverage_for(size, margin) == pytest.approx(lev, rel=1e-12)


# --- comisiones dentro del riesgo -------------------------------------------------


def test_position_size_with_fees_exact_long():
    # q = 50 / (600 + 0.0006·60000 + 0.0006·59400) = 50 / 671.64
    q = 50 / 671.64
    size = position_size_usd(50, 60000, 59400, FEE, FEE)
    assert size == pytest.approx(q * 60000, rel=1e-12)
    assert size < 5000
    loss = q * 600 + FEE * q * 60000 + FEE * q * 59400
    assert loss == pytest.approx(50, rel=1e-12)


def test_position_size_with_fees_exact_short():
    # Short: salida en 60 600, la comisión de salida pesa un poco más.
    q = 50 / (600 + 36 + 36.36)
    size = position_size_usd(50, 60000, 60600, FEE, FEE)
    assert size == pytest.approx(q * 60000, rel=1e-12)
    assert size < position_size_usd(50, 60000, 59400, FEE, FEE)


def test_zero_fees_reduce_to_user_formula_random():
    rng = np.random.default_rng(1)
    for _ in range(500):
        _side, entry, stop, risk = _random_trade(rng)
        expected = risk / (abs(entry - stop) / entry)
        assert position_size_usd(risk, entry, stop) == pytest.approx(expected, rel=1e-12)
        assert position_size_usd(risk, entry, stop, 0.0, 0.0) == pytest.approx(expected, rel=1e-12)


def test_loss_at_stop_equals_risk_with_fees_random():
    rng = np.random.default_rng(2)
    for _ in range(500):
        side, entry, stop, risk = _random_trade(rng)
        f_in = float(rng.uniform(0, 0.002))
        f_out = float(rng.uniform(0, 0.002))
        size = position_size_usd(risk, entry, stop, f_in, f_out)
        q = size / entry
        loss = q * abs(entry - stop) + f_in * q * entry + f_out * q * stop
        assert loss == pytest.approx(risk, rel=1e-10), side
        assert size <= position_size_usd(risk, entry, stop) * (1 + 1e-12)


def test_common_approximation_is_close_but_not_exact():
    # Aproximación riesgo / (d/E + f_in + f_out): en long da un poco menos de
    # tamaño (sobreestima la comisión de salida), en short un poco más.
    exact_long = position_size_usd(50, 60000, 59400, FEE, FEE)
    approx_long = 50 / (0.01 + 2 * FEE)
    assert approx_long < exact_long
    # Diferencia relativa ≈ f_out·d/E / (d/E + f_in + f_out) ≈ 0,05 %.
    assert approx_long == pytest.approx(exact_long, rel=1e-3)
    exact_short = position_size_usd(50, 60000, 60600, FEE, FEE)
    approx_short = 50 / (0.01 + 2 * FEE)
    assert approx_short > exact_short
    assert approx_short == pytest.approx(exact_short, rel=1e-3)


# --- size_position ------------------------------------------------------------------


def test_size_position_example_fees_included():
    s = size_position(60000, 59400, "long", 50, 100, FEE, FEE, True)
    assert isinstance(s, PositionSizing)
    assert s.side == "long"
    assert s.stop_distance_pct == pytest.approx(0.01)
    assert s.size_usd < 5000
    assert s.size_usd == pytest.approx(50 / 671.64 * 60000, rel=1e-12)
    assert s.quantity == pytest.approx(s.size_usd / 60000)
    assert s.margin_usd == 100
    assert s.leverage == pytest.approx(s.size_usd / 100)
    assert s.loss_at_stop_usd == pytest.approx(50, rel=1e-12)
    assert s.price_loss_at_stop_usd + s.fees_at_stop_usd == pytest.approx(s.loss_at_stop_usd)
    assert s.fees_at_stop_usd == pytest.approx(s.quantity * (60000 + 59400) * FEE)


def test_size_position_example_fees_apart():
    s = size_position(60000, 59400, "long", 50, 100, FEE, FEE, False)
    assert s.size_usd == pytest.approx(5000)
    assert s.leverage == pytest.approx(50)
    assert s.price_loss_at_stop_usd == pytest.approx(50)
    # Entrada 5000 × 0,06 % = 3; salida 0.0833.. × 59 400 × 0,06 % = 2,97.
    assert s.fees_at_stop_usd == pytest.approx(5.97)
    assert s.loss_at_stop_usd == pytest.approx(55.97)


def test_size_position_long_short_symmetry_without_fees():
    long_ = size_position(60000, 59400, "long", 50, 100, 0.0, 0.0, True)
    short = size_position(60000, 60600, "short", 50, 100, 0.0, 0.0, True)
    assert long_.size_usd == pytest.approx(short.size_usd)
    assert long_.leverage == pytest.approx(short.leverage)
    assert long_.loss_at_stop_usd == pytest.approx(short.loss_at_stop_usd) == pytest.approx(50)
    assert long_.fees_at_stop_usd == short.fees_at_stop_usd == 0


def test_size_position_long_short_with_fees():
    long_ = size_position(60000, 59400, "long", 50, 100, FEE, FEE, True)
    short = size_position(60000, 60600, "short", 50, 100, FEE, FEE, True)
    # El short sale a un precio más alto: paga un poco más de comisión de salida.
    assert short.size_usd < long_.size_usd
    assert long_.loss_at_stop_usd == pytest.approx(50)
    assert short.loss_at_stop_usd == pytest.approx(50)


def test_size_position_random_invariants():
    rng = np.random.default_rng(3)
    for _ in range(500):
        side, entry, stop, risk = _random_trade(rng)
        margin = float(rng.uniform(5, 5000))
        f_in = float(rng.uniform(0, 0.002))
        f_out = float(rng.uniform(0, 0.002))
        inc = size_position(entry, stop, side, risk, margin, f_in, f_out, True)
        apart = size_position(entry, stop, side, risk, margin, f_in, f_out, False)
        assert inc.loss_at_stop_usd == pytest.approx(risk, rel=1e-10)
        assert apart.price_loss_at_stop_usd == pytest.approx(risk, rel=1e-10)
        assert apart.loss_at_stop_usd == pytest.approx(risk + apart.fees_at_stop_usd, rel=1e-10)
        assert inc.size_usd <= apart.size_usd * (1 + 1e-12)
        for s in (inc, apart):
            assert s.leverage == pytest.approx(s.size_usd / margin, rel=1e-12)
            assert s.quantity == pytest.approx(s.size_usd / entry, rel=1e-12)
            assert s.price_loss_at_stop_usd == pytest.approx(
                s.quantity * abs(entry - stop), rel=1e-12
            )


def test_fees_do_not_depend_on_margin():
    # Las comisiones se cobran sobre el tamaño: cambiar el margen no las mueve.
    a = size_position(60000, 59400, "long", 50, 100, FEE, FEE, True)
    b = size_position(60000, 59400, "long", 50, 1000, FEE, FEE, True)
    assert a.size_usd == pytest.approx(b.size_usd)
    assert a.fees_at_stop_usd == pytest.approx(b.fees_at_stop_usd)
    assert a.leverage == pytest.approx(10 * b.leverage)


# --- validaciones ----------------------------------------------------------------


def test_side_helpers():
    assert ensure_side("long") == "long"
    assert side_sign("long") == 1
    assert side_sign("short") == -1
    with pytest.raises(ValueError, match="long"):
        ensure_side("buy")


@pytest.mark.parametrize(
    ("entry", "stop"),
    [(0, 100), (-1, 100), (100, 0), (100, -5), (100, 100), (math.nan, 90), (100, math.inf)],
)
def test_stop_distance_invalid(entry, stop):
    with pytest.raises(ValueError):
        stop_distance_pct(entry, stop)


@pytest.mark.parametrize("risk", [0, -10, math.nan])
def test_position_size_invalid_risk(risk):
    with pytest.raises(ValueError):
        position_size_usd(risk, 60000, 59400)


@pytest.mark.parametrize("rate", [-0.0001, 1.0, 6.0, math.nan])
def test_invalid_fee_rates(rate):
    with pytest.raises(ValueError):
        position_size_usd(50, 60000, 59400, rate, 0.0)
    with pytest.raises(ValueError):
        position_size_usd(50, 60000, 59400, 0.0, rate)
    with pytest.raises(ValueError):
        ensure_rate(rate, "La tasa")


def test_leverage_and_margin_invalid():
    for bad in (0, -1, math.nan):
        with pytest.raises(ValueError):
            leverage_for(5000, bad)
        with pytest.raises(ValueError):
            leverage_for(bad, 100)
        with pytest.raises(ValueError):
            margin_for_leverage(5000, bad)
        with pytest.raises(ValueError):
            quantity_for(5000, bad)


def test_size_position_wrong_stop_side():
    with pytest.raises(ValueError, match="debajo"):
        size_position(60000, 60600, "long", 50, 100, FEE, FEE, True)
    with pytest.raises(ValueError, match="arriba"):
        size_position(60000, 59400, "short", 50, 100, FEE, FEE, True)
    with pytest.raises(ValueError, match="igual"):
        size_position(60000, 60000, "long", 50, 100, FEE, FEE, True)


def test_size_position_invalid_inputs():
    with pytest.raises(ValueError):
        size_position(60000, 59400, "long", 0, 100, FEE, FEE, True)
    with pytest.raises(ValueError):
        size_position(60000, 59400, "long", 50, 0, FEE, FEE, True)
    with pytest.raises(ValueError):
        size_position(60000, 59400, "long", 50, -100, FEE, FEE, True)
    with pytest.raises(ValueError):
        size_position(60000, 59400, "sell", 50, 100, FEE, FEE, True)
    with pytest.raises(ValueError):
        size_position(60000, 59400, "long", 50, 100, -FEE, FEE, True)
    with pytest.raises(ValueError):
        size_position(-60000, 59400, "long", 50, 100, FEE, FEE, True)


def test_numpy_scalars_are_accepted():
    s = size_position(
        np.float64(60000), np.float64(59400), "long", np.int64(50), np.float32(100), 0, 0, True
    )
    assert s.size_usd == pytest.approx(5000)
