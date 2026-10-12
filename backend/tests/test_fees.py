"""Hito 2: comisiones sobre el tamaño, R:R con comisiones y funding."""

import math

import numpy as np
import pytest

from app.config import FeesConfig, VenueFees
from app.risk.fees import (
    fee_rates_from_config,
    fees_pct_of,
    funding_cost_usd,
    funding_events,
    gross_pnl_usd,
    gross_rr,
    guaranteed_funding_events,
    net_pnl_usd,
    net_rr,
    trade_fees_usd,
)
from app.risk.sizing import size_position

FEE = 0.0006


# --- config ----------------------------------------------------------------------


def test_fee_rates_from_repo_config(settings):
    entry_rate, exit_rate = fee_rates_from_config(settings.fees)
    assert entry_rate == pytest.approx(0.0006)
    assert exit_rate == pytest.approx(0.0006)


def test_fee_rates_maker_taker_selection():
    cfg = FeesConfig(
        nexo=VenueFees(taker_pct=0.06, maker_pct=0.02),
        entry_order="maker",
        exit_order="taker",
    )
    assert fee_rates_from_config(cfg) == pytest.approx((0.0002, 0.0006))
    cfg2 = cfg.model_copy(update={"entry_order": "taker", "exit_order": "maker"})
    assert fee_rates_from_config(cfg2) == pytest.approx((0.0006, 0.0002))


# --- comisiones de una operación ----------------------------------------------------


def test_trade_fees_hand_example():
    # q = 0.1 BTC: entrada 6000 USD × 0,06 % = 3.6; salida 6120 × 0,06 % = 3.672.
    assert trade_fees_usd(0.1, 60000, 61200, FEE, FEE) == pytest.approx(7.272)
    assert trade_fees_usd(0.1, 60000, 61200, 0.0002, FEE) == pytest.approx(1.2 + 3.672)


def test_trade_fees_are_on_position_size_not_margin():
    # Mismo riesgo y distinto margen (distinto apalancamiento): mismas comisiones.
    for margin in (50, 100, 500, 5000):
        s = size_position(60000, 59400, "long", 50, margin, FEE, FEE, False)
        fees = trade_fees_usd(s.quantity, 60000, 59400, FEE, FEE)
        assert fees == pytest.approx(5.97)
        assert fees == pytest.approx(s.fees_at_stop_usd)


def test_trade_fees_scale_linearly():
    rng = np.random.default_rng(11)
    for _ in range(200):
        q = float(rng.uniform(0.001, 100))
        entry = float(rng.uniform(1, 1e5))
        exit_price = float(entry * rng.uniform(0.8, 1.2))
        f_in, f_out = float(rng.uniform(0, 0.002)), float(rng.uniform(0, 0.002))
        fees = trade_fees_usd(q, entry, exit_price, f_in, f_out)
        assert fees == pytest.approx(q * entry * f_in + q * exit_price * f_out, rel=1e-12)
        assert trade_fees_usd(2 * q, entry, exit_price, f_in, f_out) == pytest.approx(2 * fees)


# --- PnL ------------------------------------------------------------------------------


def test_gross_and_net_pnl_long_short():
    q = 0.1
    assert gross_pnl_usd("long", q, 60000, 61200) == pytest.approx(120)
    assert gross_pnl_usd("long", q, 60000, 59400) == pytest.approx(-60)
    assert gross_pnl_usd("short", q, 60000, 58800) == pytest.approx(120)
    assert gross_pnl_usd("short", q, 60000, 60600) == pytest.approx(-60)
    assert net_pnl_usd("long", q, 60000, 61200, FEE, FEE) == pytest.approx(120 - 7.272)
    assert net_pnl_usd("short", q, 60000, 58800, FEE, FEE) == pytest.approx(120 - 3.6 - 3.528)


def test_net_pnl_is_gross_minus_fees_random():
    rng = np.random.default_rng(12)
    for _ in range(300):
        side = "long" if rng.random() < 0.5 else "short"
        q = float(rng.uniform(0.001, 100))
        entry = float(rng.uniform(1, 1e5))
        exit_price = float(entry * rng.uniform(0.8, 1.2))
        f_in, f_out = float(rng.uniform(0, 0.002)), float(rng.uniform(0, 0.002))
        gross = gross_pnl_usd(side, q, entry, exit_price)
        fees = trade_fees_usd(q, entry, exit_price, f_in, f_out)
        net = net_pnl_usd(side, q, entry, exit_price, f_in, f_out)
        assert net == pytest.approx(gross - fees, rel=1e-9, abs=1e-9)
        # Long y short desde la misma entrada al mismo precio: brutos opuestos.
        other = "short" if side == "long" else "long"
        assert gross_pnl_usd(other, q, entry, exit_price) == pytest.approx(-gross)


# --- R:R con comisiones -------------------------------------------------------------------


def test_rr_hand_example_long():
    assert gross_rr("long", 60000, 59400, 61200) == pytest.approx(2.0)
    # Ganancia por unidad: 1200 - 36 - 36.72 = 1127.28; pérdida: 600 + 36 + 35.64 = 671.64.
    assert net_rr("long", 60000, 59400, 61200, FEE, FEE) == pytest.approx(1127.28 / 671.64)


def test_rr_hand_example_short():
    assert gross_rr("short", 60000, 60600, 58800) == pytest.approx(2.0)
    gain = 1200 - 36 - 0.0006 * 58800
    loss = 600 + 36 + 0.0006 * 60600
    assert net_rr("short", 60000, 60600, 58800, FEE, FEE) == pytest.approx(gain / loss)


def test_net_rr_equals_gross_without_fees():
    assert net_rr("long", 60000, 59400, 61200, 0, 0) == pytest.approx(2.0)
    assert net_rr("short", 60000, 60600, 58800, 0, 0) == pytest.approx(2.0)


def test_net_rr_below_gross_with_fees_random():
    rng = np.random.default_rng(13)
    for _ in range(500):
        side = "long" if rng.random() < 0.5 else "short"
        entry = float(10 ** rng.uniform(-1, 5))
        d = float(rng.uniform(0.001, 0.1))
        t = float(rng.uniform(0.001, 0.3))
        sign = 1 if side == "long" else -1
        stop = entry * (1 - sign * d)
        target = entry * (1 + sign * t)
        f_in = float(rng.uniform(0.00001, 0.002))
        f_out = float(rng.uniform(0.00001, 0.002))
        g = gross_rr(side, entry, stop, target)
        n = net_rr(side, entry, stop, target, f_in, f_out)
        assert n < g
        # Coincide con el cociente de los PnL netos (no depende de la cantidad).
        q = float(rng.uniform(0.01, 10))
        win = net_pnl_usd(side, q, entry, target, f_in, f_out)
        lose = net_pnl_usd(side, q, entry, stop, f_in, f_out)
        assert n == pytest.approx(win / -lose, rel=1e-9, abs=1e-12)


def test_net_rr_can_be_negative_when_target_is_too_close():
    # Objetivo a 0,05 %: las comisiones (≈0,12 %) se comen toda la ganancia.
    assert net_rr("long", 60000, 59400, 60030, FEE, FEE) < 0
    assert gross_rr("long", 60000, 59400, 60030) > 0


def test_rr_invalid_inputs():
    with pytest.raises(ValueError, match="objetivo"):
        net_rr("long", 60000, 59400, 59000, FEE, FEE)
    with pytest.raises(ValueError, match="objetivo"):
        net_rr("short", 60000, 60600, 61000, FEE, FEE)
    with pytest.raises(ValueError, match="objetivo"):
        gross_rr("long", 60000, 59400, 60000)
    with pytest.raises(ValueError, match="stop"):
        net_rr("long", 60000, 60600, 61200, FEE, FEE)
    with pytest.raises(ValueError):
        net_rr("long", 60000, 59400, 61200, -FEE, FEE)
    with pytest.raises(ValueError):
        net_rr("hold", 60000, 59400, 61200, FEE, FEE)


# --- comisiones como % ---------------------------------------------------------------


def test_fees_pct_of():
    assert fees_pct_of(5.97, 50) == pytest.approx(11.94)
    assert fees_pct_of(0, 50) == 0
    assert fees_pct_of(7.272, 120) == pytest.approx(6.06)
    with pytest.raises(ValueError):
        fees_pct_of(5, 0)
    with pytest.raises(ValueError):
        fees_pct_of(5, -10)
    with pytest.raises(ValueError):
        fees_pct_of(-1, 50)


# --- funding --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hours", "interval", "expected"),
    [
        (0, 8, 0),
        (1e-6, 8, 1),
        (1, 8, 1),
        (8, 8, 1),
        (8.01, 8, 2),
        (16, 8, 2),
        (24, 8, 3),
        (24, 4, 6),
        (25, 4, 7),
        (0.1 + 0.2, 0.3, 1),  # 0.30000000000000004 / 0.3 no cuenta un evento extra
        (3 * 0.1, 0.1, 3),
    ],
)
def test_funding_events(hours, interval, expected):
    assert funding_events(hours, interval) == expected


def test_funding_cost_hand_example():
    # 5000 USD, 0,01 % por período, 24 h con intervalo de 8 h -> 3 cobros -> 1.5 USD.
    assert funding_cost_usd("long", 5000, 0.0001, 24, 8) == pytest.approx(1.5)
    assert funding_cost_usd("short", 5000, 0.0001, 24, 8) == pytest.approx(-1.5)


def test_funding_sign_convention():
    # Tasa positiva: el long paga (costo > 0) y el short cobra (costo < 0).
    assert funding_cost_usd("long", 1000, 0.0002, 8, 8) > 0
    assert funding_cost_usd("short", 1000, 0.0002, 8, 8) < 0
    # Tasa negativa: al revés.
    assert funding_cost_usd("long", 1000, -0.0002, 8, 8) < 0
    assert funding_cost_usd("short", 1000, -0.0002, 8, 8) > 0
    assert funding_cost_usd("long", 1000, 0.0, 24, 8) == 0
    assert funding_cost_usd("long", 1000, 0.0002, 0, 8) == 0


def test_funding_receiving_side_gets_no_credit_for_a_partial_interval():
    # Regresión: un short con funding positivo 1 h (intervalo 8 h) probablemente no
    # cruza ningún horario de funding; antes se le acreditaba un cobro entero.
    assert funding_cost_usd("short", 10_000, 0.0001, 1, 8) == 0
    assert funding_cost_usd("long", 10_000, -0.0001, 1, 8) == 0
    # Con 9 h cae seguro un cobro (no dos) para quien cobra...
    assert funding_cost_usd("short", 10_000, 0.0001, 9, 8) == pytest.approx(-1.0)
    # ...y quien paga cuenta los dos que podrían caer.
    assert funding_cost_usd("long", 10_000, 0.0001, 9, 8) == pytest.approx(2.0)
    assert funding_cost_usd("long", 10_000, 0.0001, 1, 8) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("hours", "interval", "expected"),
    [
        (0, 8, 0),
        (1, 8, 0),
        (7.99, 8, 0),
        (8, 8, 1),
        (9, 8, 1),
        (24, 8, 3),
        (25, 4, 6),
        (3 * 0.1, 0.1, 3),  # 0.30000000000000004 / 0.1 no pierde un evento
        (0.3, 0.1, 3),  # 0.3 / 0.1 = 2.9999999999999996 tampoco
    ],
)
def test_guaranteed_funding_events(hours, interval, expected):
    assert guaranteed_funding_events(hours, interval) == expected


def test_funding_conservative_random():
    rng = np.random.default_rng(14)
    for _ in range(200):
        size = float(rng.uniform(10, 1e5))
        rate = float(rng.uniform(-0.003, 0.003))
        hours = float(rng.uniform(0, 72))
        interval = float(rng.choice([1, 4, 8]))
        long_ = funding_cost_usd("long", size, rate, hours, interval)
        short = funding_cost_usd("short", size, rate, hours, interval)
        events = funding_events(hours, interval)
        sure = guaranteed_funding_events(hours, interval)
        # Máximo: cubre todas las horas y nunca suma un intervalo de más.
        assert events * interval >= hours - 1e-9
        assert (events - 1) * interval < hours or events == 0
        # Seguros: nunca más que las horas, y como mucho uno menos que el máximo.
        assert sure * interval <= hours + 1e-9
        assert events - 1 <= sure <= events
        payer, receiver = (long_, short) if rate > 0 else (short, long_)
        assert payer == pytest.approx(abs(size * rate) * events, abs=1e-12)
        assert receiver == pytest.approx(-abs(size * rate) * sure, abs=1e-12)
        # Conservador: entre los dos lados nunca "aparece" plata.
        assert long_ + short >= -1e-12


def test_funding_invalid_inputs():
    with pytest.raises(ValueError):
        funding_events(-1, 8)
    with pytest.raises(ValueError):
        funding_events(8, 0)
    with pytest.raises(ValueError):
        guaranteed_funding_events(-1, 8)
    with pytest.raises(ValueError):
        guaranteed_funding_events(8, 0)
    with pytest.raises(ValueError):
        funding_cost_usd("long", 0, 0.0001, 8, 8)
    with pytest.raises(ValueError):
        funding_cost_usd("long", 1000, math.nan, 8, 8)
    with pytest.raises(ValueError):
        funding_cost_usd("flat", 1000, 0.0001, 8, 8)
    with pytest.raises(ValueError):
        funding_cost_usd("long", 1000, 0.0001, 8, -8)


def test_trade_fees_invalid_inputs():
    with pytest.raises(ValueError):
        trade_fees_usd(0, 60000, 61200, FEE, FEE)
    with pytest.raises(ValueError):
        trade_fees_usd(0.1, 0, 61200, FEE, FEE)
    with pytest.raises(ValueError):
        trade_fees_usd(0.1, 60000, -1, FEE, FEE)
    with pytest.raises(ValueError):
        trade_fees_usd(0.1, 60000, 61200, 0.06 * 100, FEE)
