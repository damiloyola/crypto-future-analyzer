"""Hito 4: margen y tamaño de la posición (decisión del usuario: primero margen, después tamaño).

Casos borde de la especificación, uno por bloque:

- liquidación antes del stop: se arregla subiendo el margen; si ni con el margen
  máximo alcanza, se achica la posición; si el riesgo que queda es menor al
  mínimo operable, "No operar";
- apalancamiento por encima del tope: lo mismo (margen, después tamaño, después
  bloqueo);
- stop muy lejos: posición chica, apalancamiento bajo, todo ok; si la posición
  queda más chica que el margen (menos de 1x), el margen baja solo a 1x
  (decisión del usuario); y el caso en que los multiplicadores del régimen dejan
  el riesgo debajo del mínimo operable.

(Los otros dos casos borde están en test_rules.py, timeframes en conflicto, y en
test_regime_adjustments.py, Crash con long.)
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.risk.fees import fee_rates_from_config
from app.risk.liquidation import (
    check_liquidation_vs_stop,
    liquidation_price,
    mmr_from_config,
    wallet_liquidation_price,
)
from app.risk.position import (
    LIMIT_AVAILABLE_MARGIN,
    LIMIT_LEVERAGE,
    LIMIT_LIQUIDATION,
    MARGIN_LIMITED,
    MARGIN_LOWERED_TO_1X,
    MARGIN_RAISED,
    SIZE_REDUCED,
    PositionFit,
    ceil_cents,
    fit_position,
    floor_cents,
)
from app.risk.profiles import get_profile
from app.risk.regime_adjustments import apply_regime_adjustments
from app.risk.rules import check_liquidation
from app.risk.sizing import position_size_usd

ENTRY = 60000.0


@pytest.fixture(autouse=True)
def _default_costs(settings):
    """Los números fijos de estos tests asumen las comisiones y el mantenimiento de la config."""
    assert fee_rates_from_config(settings.fees) == pytest.approx((0.0006, 0.0006))
    assert mmr_from_config(settings.liquidation) == pytest.approx(0.005)
    assert settings.trade_plan.liquidation.min_distance_ratio == 2.0
    assert settings.trade_plan.liquidation.max_margin_multiplier == 2.0
    assert settings.trade_plan.fees_included_in_risk is True


@pytest.fixture
def agresivo(settings):
    return get_profile("agresivo", settings)  # margen 100, riesgo 50, tope 25x, mínimo 15


def with_plan_liquidation(settings, **update):
    tp = settings.trade_plan
    liq = tp.liquidation.model_copy(update=update)
    return settings.model_copy(update={"trade_plan": tp.model_copy(update={"liquidation": liq})})


def codes(fit: PositionFit) -> list[str]:
    return [a.code for a in fit.adjustments]


def assert_safe(fit: PositionFit) -> None:
    """Invariantes de cualquier posición ajustada."""
    assert fit.leverage <= fit.max_leverage * (1 + 1e-9)
    assert fit.leverage >= 1.0  # el exchange no deja operar a menos de 1x
    assert fit.margin_usd <= fit.size_usd
    assert fit.liquidation_ok is True
    assert fit.liq_before_stop is False
    assert fit.margin_usd <= fit.max_margin_usd * (1 + 1e-9)
    assert fit.risk_usd <= fit.requested_risk_usd * (1 + 1e-9)


# --- redondeo ---------------------------------------------------------------------------


def test_cents_rounding():
    assert ceil_cents(162.4) == 162.4  # 162.4 * 100 = 16240.000000000002 no sube
    assert ceil_cents(162.401) == 162.41
    assert ceil_cents(178.6672) == 178.67
    assert floor_cents(4999.999) == 4999.99
    assert floor_cents(5000.0) == 5000.0


# --- caso borde: liquidación antes del stop -----------------------------------------------


@pytest.fixture
def thin_margin(settings):
    """Perfil con margen chico (40 USD) para 50 USD de riesgo y tope 50x.

    Con el stop a 3 % (58 200), el tamaño es ~1.603 USD (40x, dentro del tope) y
    con 40 USD de margen te liquidan a ~58 833: antes de llegar al stop.
    """
    base = get_profile("agresivo", settings, max_leverage_override=50)
    return dataclasses.replace(base, margin_per_trade_usd=40.0)


def test_edge_case_liquidation_before_stop_at_profile_margin(thin_margin):
    size = position_size_usd(50, ENTRY, 58200, 0.0006, 0.0006)
    liq = liquidation_price("long", ENTRY, size, 40.0, 0.005, 0.0006)
    check = check_liquidation_vs_stop("long", ENTRY, 58200, liq, 2.0)
    assert size / 40.0 < 50  # el apalancamiento no es el problema
    assert check.liq_before_stop is True


def test_edge_case_liquidation_before_stop_fixed_by_raising_margin(settings, thin_margin):
    roomy = with_plan_liquidation(settings, max_margin_multiplier=3.0)  # hasta 120 USD
    fit = fit_position("long", ENTRY, 58200, 50, thin_margin, roomy)
    assert fit.ok
    assert fit.initial_liq_before_stop is True
    assert fit.initial_leverage == pytest.approx(40.09, abs=0.01)
    assert fit.margin_raised and not fit.size_reduced
    assert fit.margin_usd == pytest.approx(104.66)
    assert fit.risk_usd == pytest.approx(50)  # el riesgo no cambia
    assert fit.limited_by == (LIMIT_LIQUIDATION,)
    assert fit.liquidation_ratio >= 2.0
    assert codes(fit) == [MARGIN_RAISED]
    assert fit.adjustments[0].message == (
        "Margen subido de 40 a 104,66 USD para que no te liquiden antes de llegar al stop "
        "(la liquidación tiene que quedar al menos a 2 veces la distancia al stop); el riesgo "
        "sigue en 50 USD."
    )
    assert_safe(fit)


def test_edge_case_liquidation_before_stop_shrinks_when_max_margin_is_not_enough(
    settings, thin_margin
):
    fit = fit_position("long", ENTRY, 58200, 50, thin_margin, settings)  # máximo 80 USD
    assert fit.ok
    assert fit.margin_usd == 80.0
    assert fit.size_reduced and fit.risk_reduced
    assert fit.size_usd == pytest.approx(1225.79)
    assert fit.risk_usd == pytest.approx(38.22, abs=0.01)
    assert fit.limited_by == (LIMIT_LIQUIDATION,)
    assert codes(fit) == [MARGIN_RAISED, SIZE_REDUCED]
    assert fit.adjustments[1].message == (
        "Posición achicada de 1.603,49 a 1.225,79 USD: con el margen máximo de 80 USD, una "
        "posición más grande dejaría la liquidación a menos de 2 veces la distancia al stop; "
        "ahora arriesgás 38,22 USD (en vez de 50 USD)."
    )
    assert fit.adjustments[1].risk_multiplier == pytest.approx(fit.risk_usd / 50)
    assert_safe(fit)


def test_edge_case_liquidation_before_stop_blocked_when_shrunk_risk_is_too_small(
    settings, thin_margin
):
    picky = dataclasses.replace(thin_margin, min_operable_risk_usd=40.0)
    fit = fit_position("long", ENTRY, 58200, 50, picky, settings)
    assert fit.ok is False
    assert fit.block_reason == (
        "No operar: aun con el margen máximo de 80 USD la posición entra solo hasta "
        "1.225,79 USD y arriesgarías 38,22 USD, menos que el mínimo operable de 40 USD."
    )
    assert [c.code for c in fit.checks() if not c.ok] == ["min_operable_risk"]


# --- caso borde: apalancamiento por encima del tope --------------------------------------


def test_edge_case_leverage_exceeded_fixed_by_raising_margin(settings, agresivo):
    # Stop a 1 %: 4.466,68 USD de posición; con 100 USD serían 44,7x.
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, settings)
    assert fit.ok
    assert fit.size_usd == pytest.approx(4466.68, abs=0.01)
    assert fit.initial_leverage == pytest.approx(44.67, abs=0.01)
    assert fit.initial_liq_before_stop is False
    assert fit.margin_usd == 178.67
    assert fit.leverage <= 25 and fit.leverage == pytest.approx(25, abs=0.01)
    assert fit.risk_usd == pytest.approx(50)
    assert fit.limited_by == (LIMIT_LEVERAGE, LIMIT_LIQUIDATION)
    assert fit.adjustments[0].message == (
        "Margen subido de 100 a 178,67 USD para no pasar de 25x y para que la liquidación "
        "quede al menos a 2 veces la distancia al stop; el riesgo sigue en 50 USD."
    )
    assert fit.red_warning is False
    assert_safe(fit)


def test_edge_case_leverage_exceeded_shrinks_position(settings, agresivo):
    # Stop a 0,5 %: harían falta 322,74 USD de margen para 25x; el máximo es 200.
    fit = fit_position("long", ENTRY, 59700, 50, agresivo, settings)
    assert fit.ok
    assert fit.margin_usd == 200.0
    assert fit.size_usd == 5000.0
    assert fit.leverage == 25.0
    assert fit.risk_usd == pytest.approx(30.98, abs=0.01)
    assert fit.limited_by == (LIMIT_LEVERAGE,)
    assert [a.message for a in fit.adjustments] == [
        "Margen subido de 100 a 200 USD, el máximo permitido; no alcanza para mantener el riesgo.",
        "Posición achicada de 8.068,42 a 5.000 USD: con el margen máximo de 200 USD no entra "
        "más sin pasar de 25x; ahora arriesgás 30,98 USD (en vez de 50 USD).",
    ]
    assert_safe(fit)


def test_edge_case_leverage_exceeded_blocked_when_risk_too_small(settings, agresivo):
    # Stop a 0,15 %: con 5.000 USD de posición se arriesgan 13,50 USD (< 15).
    fit = fit_position("long", ENTRY, 59910, 50, agresivo, settings)
    assert fit.ok is False
    assert fit.size_usd == 5000.0
    assert fit.risk_usd == pytest.approx(13.50, abs=0.01)
    assert fit.block_reason == (
        "No operar: aun con el margen máximo de 200 USD la posición entra solo hasta "
        "5.000 USD y arriesgarías 13,50 USD, menos que el mínimo operable de 15 USD."
    )


# --- caso borde: stop muy lejos ---------------------------------------------------------


def test_edge_case_stop_far_small_size_low_leverage_all_ok(settings):
    conservador = get_profile("conservador", settings)  # margen 50, riesgo 15, tope 5x
    fit = fit_position("long", ENTRY, 55200, 15, conservador, settings)  # stop a 8 %
    assert fit.ok
    assert fit.adjustments == ()
    assert fit.margin_usd == 50.0
    assert fit.size_usd == pytest.approx(184.84, abs=0.01)
    assert fit.leverage == pytest.approx(3.70, abs=0.01)
    assert fit.risk_usd == pytest.approx(15)
    assert fit.limited_by == ()
    assert all(c.ok for c in fit.checks())
    assert_safe(fit)


def test_edge_case_stop_very_far_below_1x_lowers_margin_to_1x(settings, agresivo):
    fit = fit_position("long", ENTRY, 30000, 50, agresivo, settings)  # stop a 50 %
    assert fit.ok
    assert fit.initial_leverage < 1  # con 100 USD de margen quedaba en 0,998x
    assert fit.margin_lowered_to_1x is True
    assert fit.margin_raised is False and fit.size_reduced is False
    assert fit.size_usd == pytest.approx(99.82, abs=0.01)
    assert fit.margin_usd == fit.size_usd  # sin redondear: 1x justo
    assert fit.leverage == 1.0
    assert fit.risk_usd == pytest.approx(50)  # el riesgo no cambia
    assert codes(fit) == [MARGIN_LOWERED_TO_1X]
    assert fit.adjustments[0].message == (
        "Margen bajado de 100 a 99,82 USD: la posición es más chica que el margen y no se "
        "puede operar a menos de 1x."
    )
    assert fit.adjustments[0].risk_multiplier == 1.0
    assert fit.warnings == ()
    assert fit.liquidation_price is None  # long a 1x: no se puede liquidar
    assert fit.liquidation_distance_pct is None
    assert fit.limited_by == ()
    assert all(c.ok for c in fit.checks())
    assert_safe(fit)


def test_below_1x_message_with_round_numbers(settings, agresivo):
    # Stop a 50 %: con 40,072 USD de riesgo la posición es de 80 USD justos.
    fit = fit_position("long", ENTRY, 30000, 40.072, agresivo, settings)
    assert fit.size_usd == pytest.approx(80.0)
    assert fit.margin_usd == fit.size_usd and fit.leverage == 1.0
    assert fit.adjustments[0].message == (
        "Margen bajado de 100 a 80 USD: la posición es más chica que el margen y no se puede "
        "operar a menos de 1x."
    )


def test_below_1x_after_available_margin_limit_lowers_from_the_available(settings, agresivo):
    fit = fit_position("long", ENTRY, 30000, 40.072, agresivo, settings, available_margin_usd=90)
    assert codes(fit) == [MARGIN_LIMITED, MARGIN_LOWERED_TO_1X]
    assert fit.adjustments[1].message.startswith("Margen bajado de 90 a 80 USD:")
    assert fit.margin_usd == fit.size_usd and fit.leverage == 1.0
    assert_safe(fit)


def test_below_1x_short_still_checks_liquidation_and_passes(settings):
    moderado = get_profile("moderado", settings)  # margen 75, riesgo 30
    fit = fit_position("short", ENTRY, 87000, 30, moderado, settings)  # stop a 45 %
    assert fit.ok
    assert codes(fit) == [MARGIN_LOWERED_TO_1X]
    assert fit.adjustments[0].message.startswith("Margen bajado de 75 a 66,45 USD:")
    assert fit.leverage == 1.0
    # Un short a 1x sí se liquida, con una suba de casi el 100 %: 2,2 veces la
    # distancia al stop, más que el mínimo de 2.
    assert fit.liquidation_price == pytest.approx(119331.74, abs=0.01)
    assert fit.liquidation_ratio == pytest.approx(2.197, abs=0.001)
    assert_safe(fit)


def test_below_1x_short_with_absurd_stop_is_blocked_by_liquidation(settings, agresivo):
    # Stop a 60 % arriba: la regla de liquidación pedía más margen que la
    # posición, pero menos de 1x no se puede; a 1x la liquidación queda a 1,65
    # veces la distancia al stop (menos de 2): "No operar".
    fit = fit_position("short", ENTRY, 96000, 50, agresivo, settings)
    assert fit.ok is False
    assert codes(fit) == [MARGIN_RAISED, MARGIN_LOWERED_TO_1X]
    assert fit.adjustments[1].message.startswith("Margen bajado de 100,77 a 83,12 USD:")
    assert fit.leverage == 1.0 and fit.margin_usd == fit.size_usd
    assert fit.liquidation_ok is False
    assert fit.block_reason == check_liquidation(fit.liquidation_check).message
    assert fit.block_reason.startswith("No operar: la liquidación")
    assert [c.code for c in fit.checks() if not c.ok] == ["liquidation"]


def test_far_stops_never_end_below_1x(settings):
    """Con stops lejanos (hasta absurdos) el apalancamiento final nunca baja de 1x."""
    rng = np.random.default_rng(11)
    profiles = [get_profile(n, settings) for n in ("conservador", "moderado", "agresivo")]
    for _ in range(300):
        p = profiles[int(rng.integers(len(profiles)))]
        side = "long" if rng.random() < 0.5 else "short"
        entry = float(10 ** rng.uniform(-1, 5))
        dist = float(rng.uniform(0.2, 0.95))
        stop = entry * (1 - dist) if side == "long" else entry * (1 + dist)
        risk = float(rng.uniform(1, p.risk_per_trade_usd))
        avail = None if rng.random() < 0.7 else float(rng.uniform(5, 150))
        fit = fit_position(side, entry, stop, risk, p, settings, available_margin_usd=avail)
        assert fit.leverage >= 1.0
        assert fit.margin_usd <= fit.size_usd
        if fit.margin_lowered_to_1x:
            assert fit.leverage == 1.0
            assert codes(fit)[-1] == MARGIN_LOWERED_TO_1X
        if side == "long" and fit.leverage == 1.0:
            assert fit.liquidation_price is None  # un long a 1x no se puede liquidar


def test_raised_margin_rounding_never_goes_below_1x(settings, agresivo):
    # Long con el stop a 50 % y 60 USD de riesgo: la regla de liquidación pide
    # margen = posición (119,7844 USD). Redondeado hacia arriba serían 119,79
    # (0,9999x): el margen queda en la posición justa.
    fit = fit_position("long", ENTRY, 30000, 60, agresivo, settings)
    assert fit.ok
    assert codes(fit) == [MARGIN_RAISED]
    assert fit.margin_raised is True and fit.margin_lowered_to_1x is False
    assert fit.size_usd == pytest.approx(119.7844, abs=1e-4)
    assert fit.margin_usd == fit.size_usd
    assert fit.leverage == 1.0
    assert fit.adjustments[0].message == (
        "Margen subido de 100 a 119,78 USD para que la liquidación quede al menos a 2 veces "
        "la distancia al stop; el riesgo sigue en 60 USD."
    )
    assert fit.liquidation_price is None
    assert_safe(fit)


def test_edge_case_stop_far_but_regime_multipliers_hit_the_risk_floor(settings, agresivo):
    # Neutral 0,75 × inestable 0,5 × confianza baja 0,5: 50 USD -> 9,38 USD (< 15).
    regime = SimpleNamespace(category="neutral", confidence=0.5, unstable=True)
    adj = apply_regime_adjustments("long", 70, regime, None, agresivo, settings)
    assert adj.blocked and adj.risk_usd == pytest.approx(9.375)
    fit = fit_position("long", ENTRY, 48000, adj.risk_usd, agresivo, settings)  # stop a 20 %
    assert fit.margin_lowered_to_1x and fit.leverage == 1.0
    assert fit.ok is False
    assert fit.block_reason == (
        "No operar: el riesgo (9,38 USD) queda debajo del mínimo operable de 15 USD."
    )


# --- margen disponible cargado a mano ----------------------------------------------------


def test_available_margin_caps_the_raise(settings, agresivo):
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, settings, available_margin_usd=150)
    assert fit.max_margin_usd == 150.0
    assert fit.margin_usd == 150.0
    assert fit.size_usd == 3750.0
    assert fit.leverage == 25.0
    assert fit.limited_by == (LIMIT_LEVERAGE, LIMIT_AVAILABLE_MARGIN)
    assert "(lo que tenés disponible)" in fit.adjustments[0].message
    assert fit.ok
    assert_safe(fit)


def test_available_margin_below_profile_margin(settings, agresivo):
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, settings, available_margin_usd=80)
    assert fit.margin_usd == 80.0
    assert codes(fit) == [MARGIN_LIMITED, SIZE_REDUCED]
    assert fit.adjustments[0].message == (
        "Margen bajado de 100 a 80 USD: es lo que tenés disponible."
    )
    assert fit.size_usd == 2000.0
    assert fit.margin_raised is False
    assert_safe(fit)


def test_plenty_of_available_margin_changes_nothing(settings, agresivo):
    a = fit_position("long", ENTRY, 59400, 50, agresivo, settings)
    b = fit_position("long", ENTRY, 59400, 50, agresivo, settings, available_margin_usd=5000)
    assert b.margin_usd == a.margin_usd and b.size_usd == a.size_usd
    assert b.max_margin_usd == 200.0


@pytest.mark.parametrize("bad", [0, -10, float("nan")])
def test_available_margin_must_be_positive(settings, agresivo, bad):
    with pytest.raises(ValueError, match="margen disponible"):
        fit_position("long", ENTRY, 59400, 50, agresivo, settings, available_margin_usd=bad)


# --- simetría long / short ---------------------------------------------------------------


@pytest.mark.parametrize("dist", [0.003, 0.01, 0.02, 0.08])
def test_long_and_short_are_mirrored(settings, agresivo, dist):
    long_fit = fit_position("long", ENTRY, ENTRY * (1 - dist), 50, agresivo, settings)
    short_fit = fit_position("short", ENTRY, ENTRY * (1 + dist), 50, agresivo, settings)
    for fit in (long_fit, short_fit):
        assert_safe(fit)
    assert long_fit.margin_raised == short_fit.margin_raised
    assert long_fit.size_reduced == short_fit.size_reduced
    assert long_fit.ok == short_fit.ok
    # La salida se cobra al precio del stop: los números difieren muy poco.
    assert short_fit.margin_usd == pytest.approx(long_fit.margin_usd, rel=0.01)
    assert short_fit.size_usd == pytest.approx(long_fit.size_usd, rel=0.01)
    assert short_fit.liquidation_price > ENTRY
    if long_fit.liquidation_price is not None:
        assert long_fit.liquidation_price < ENTRY


# --- comisiones, advertencia roja y modelo billetera ---------------------------------------


def test_fees_outside_risk_add_on_top(settings, agresivo):
    custom = settings.model_copy(
        update={
            "trade_plan": settings.trade_plan.model_copy(update={"fees_included_in_risk": False})
        }
    )
    fit = fit_position("long", ENTRY, 55200, 20, agresivo, custom)
    assert fit.fees_included_in_risk is False
    assert fit.size_usd == pytest.approx(20 / 0.08)
    assert fit.price_loss_at_stop_usd == pytest.approx(20)
    assert fit.risk_usd == pytest.approx(20 + fit.fees_usd)


def test_fees_are_round_trip_at_stop(settings, agresivo):
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, settings)
    q = fit.size_usd / ENTRY
    assert fit.quantity == pytest.approx(q)
    assert fit.fees_usd == pytest.approx(q * ENTRY * 0.0006 + q * 59400 * 0.0006)
    assert fit.risk_usd == pytest.approx(fit.price_loss_at_stop_usd + fit.fees_usd)


def test_red_warning_when_leverage_above_25x(settings):
    p = get_profile("agresivo", settings, max_leverage_override=40)
    fit = fit_position("long", ENTRY, 59400, 50, p, settings)
    # Acá manda la regla de liquidación: margen 113,85 USD -> 39,2x (más de 25x).
    assert fit.limited_by == (LIMIT_LEVERAGE, LIMIT_LIQUIDATION)
    assert fit.leverage == pytest.approx(39.23, abs=0.01)
    assert fit.red_warning is True
    assert fit.warnings[0].startswith("Advertencia: el apalancamiento queda en 39,2x, más de 25x")
    assert_safe(fit)


def test_wallet_model_is_informative(settings, agresivo):
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, settings)
    assert fit.wallet_equity_usd == agresivo.monthly_wallet_usd
    expected = wallet_liquidation_price("long", ENTRY, fit.size_usd, 300, 0.005, 0.0006)
    assert fit.wallet_liquidation_price == pytest.approx(expected)
    assert fit.wallet_liquidation_price < fit.liquidation_price  # más lejos que el aislado
    custom = fit_position("long", ENTRY, 59400, 50, agresivo, settings, wallet_equity_usd=1000)
    assert custom.wallet_equity_usd == 1000
    assert custom.wallet_liquidation_price < fit.wallet_liquidation_price
    # Las reglas usan el aislado: el modelo billetera no cambia margen ni tamaño.
    assert custom.margin_usd == fit.margin_usd and custom.size_usd == fit.size_usd


def test_wallet_model_can_be_turned_off(settings, agresivo):
    custom = settings.model_copy(
        update={"liquidation": settings.liquidation.model_copy(update={"show_wallet_model": False})}
    )
    fit = fit_position("long", ENTRY, 59400, 50, agresivo, custom)
    assert fit.wallet_liquidation_price is None and fit.wallet_equity_usd is None


# --- entradas inválidas, JSON y propiedades ------------------------------------------------


@pytest.mark.parametrize(
    ("side", "stop", "risk"),
    [("long", 60600, 50), ("short", 59400, 50), ("long", 59400, 0), ("flat", 59400, 50)],
)
def test_invalid_inputs(settings, agresivo, side, stop, risk):
    with pytest.raises(ValueError):
        fit_position(side, ENTRY, stop, risk, agresivo, settings)


def test_to_dict_is_json_safe(settings, agresivo):
    for stop in (59400, 59700, 59910, 30000):
        d = fit_position("long", ENTRY, stop, 50, agresivo, settings).to_dict()
        json.dumps(d, allow_nan=False)
        assert {
            "margin_usd",
            "size_usd",
            "leverage",
            "risk_usd",
            "ok",
            "adjustments",
            "margin_lowered_to_1x",
        } <= set(d)
        assert d["profile"] == "agresivo"
        assert all(set(a) == {"code", "message", "risk_multiplier"} for a in d["adjustments"])


def test_random_positions_respect_the_rules(settings):
    rng = np.random.default_rng(7)
    profiles = [get_profile(n, settings) for n in ("conservador", "moderado", "agresivo")]
    profiles.append(get_profile("agresivo", settings, max_leverage_override=50))
    for _ in range(400):
        p = profiles[int(rng.integers(len(profiles)))]
        side = "long" if rng.random() < 0.5 else "short"
        entry = float(10 ** rng.uniform(-1, 5))
        dist = float(rng.uniform(0.001, 0.3))
        stop = entry * (1 - dist) if side == "long" else entry * (1 + dist)
        risk = float(rng.uniform(1, p.risk_per_trade_usd))
        fit = fit_position(side, entry, stop, risk, p, settings)
        assert_safe(fit)
        if fit.margin_lowered_to_1x:
            assert fit.leverage == 1.0 and fit.margin_usd == fit.size_usd
            assert fit.size_usd < p.margin_per_trade_usd
            assert codes(fit)[-1] == MARGIN_LOWERED_TO_1X
        else:
            assert fit.margin_usd >= min(p.margin_per_trade_usd, fit.max_margin_usd) - 1e-9
        assert fit.ok == (fit.risk_usd >= p.min_operable_risk_usd * (1 - 1e-9))
        if not fit.size_reduced:
            assert fit.risk_usd == pytest.approx(risk)
        assert all(c.ok for c in fit.checks()) == fit.ok


# --- revisión final del Hito 4 ------------------------------------------------------


def test_block_reason_names_the_max_margin_not_the_lowered_one(settings, agresivo):
    # Short con el stop a 60 %: la liquidación achica la posición y después el
    # margen baja a 1x. El "margen máximo" del motivo es el máximo (100 con el
    # multiplicador en 1), no el margen final ya bajado a 1x (82,48).
    custom = with_plan_liquidation(settings, max_margin_multiplier=1.0)
    profile = dataclasses.replace(agresivo, min_operable_risk_usd=49.9)
    fit = fit_position("short", 100.0, 160.0, 50.0, profile, custom)
    assert fit.size_reduced and fit.margin_lowered_to_1x and not fit.ok
    assert fit.margin_usd == pytest.approx(fit.size_usd) and fit.margin_usd < 100.0
    assert fit.block_reason.startswith("No operar: aun con el margen máximo de 100 USD")
    assert f"margen máximo de {fit.margin_usd:.2f}".replace(".", ",") not in fit.block_reason


def test_floor_compares_the_price_risk_when_fees_are_outside(settings):
    # Comisiones aparte: el riesgo del perfil es solo el de precio. 12 USD de
    # precio + comisiones pasan los 15 USD, pero el riesgo (12) no llega al mínimo.
    custom = settings.model_copy(
        update={
            "trade_plan": settings.trade_plan.model_copy(update={"fees_included_in_risk": False})
        }
    )
    profile = get_profile("agresivo", custom)
    fit = fit_position("long", ENTRY, 59760.0, 12.0, profile, custom)
    assert fit.price_loss_at_stop_usd == pytest.approx(12.0)
    assert fit.risk_usd > profile.min_operable_risk_usd  # con las comisiones sí pasaría
    assert fit.floor_risk_usd == pytest.approx(12.0)
    assert not fit.ok
    assert fit.block_reason == (
        "No operar: el riesgo (12 USD) queda debajo del mínimo operable de 15 USD."
    )
    assert not fit.checks()[-1].ok
    # Con las comisiones adentro (la decisión del usuario) se compara la pérdida total.
    inside = fit_position("long", ENTRY, 59760.0, 20.0, get_profile("agresivo", settings), settings)
    assert inside.floor_risk_usd == pytest.approx(inside.risk_usd)
