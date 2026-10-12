"""Hito 4: casos borde obligatorios del usuario, con el plan completo.

Un test por caso, con el nombre del caso, que verifica el estado del plan y el
motivo en castellano:

- stop demasiado lejos;
- timeframes en conflicto (4h alcista, 1h bajista);
- liquidación antes del stop (se arregla subiendo el margen; y se bloquea cuando
  no hay ajuste posible);
- apalancamiento excedido (se sube el margen, después se achica la posición, y
  se bloquea si el riesgo queda debajo del mínimo operable);
- régimen Crash con long (bloqueado).

Los ``PlanInputs`` se arman a mano con los ayudantes de ``test_plan.py``
(soportes y resistencias exactos, régimen de mentira). El régimen por omisión es
"Alcista" estable con confianza 0,9 para los longs y "Bajista" para los shorts:
no cambia el riesgo, así cada caso aísla una sola regla.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.risk.plan import (
    STATUS_LONG,
    STATUS_NO_TRADE,
    STATUS_SHORT,
    TP2_DAILY_RANGE,
    build_trade_plan,
)
from app.risk.position import MARGIN_LOWERED_TO_1X, MARGIN_RAISED, SIZE_REDUCED
from app.risk.profiles import get_profile
from app.risk.rules import LIQUIDATION, MIN_OPERABLE_RISK, TIMEFRAME_CONFLICT
from tests.test_plan import (
    ENTRY,
    NOW,
    _default_costs,  # noqa: F401  (fixture autouse: valida los números de la config)
    failing_codes,
    fake_regime,
    make_inputs,
    plan_for,
)


def position_codes(plan) -> list[str]:
    return [a.code for a in plan.adjustments if a.source == "position"]


def position_messages(plan) -> list[str]:
    return [a.message for a in plan.adjustments if a.source == "position"]


def with_max_margin_multiplier(settings, value: float):
    tp = settings.trade_plan
    liq = tp.liquidation.model_copy(update={"max_margin_multiplier": value})
    return settings.model_copy(update={"trade_plan": tp.model_copy(update={"liquidation": liq})})


# --- stop demasiado lejos ---------------------------------------------------------------


def test_edge_stop_too_far_blocks_because_tp2_is_beyond_the_daily_range(settings):
    # Soporte a 10 % (54.000): stop en 53.950. La posición sale chica y con poco
    # apalancamiento, pero el TP2 queda a más del 15 % y el precio se mueve 3 % por día.
    plan = plan_for(settings, levels_15m=(54000.0, 60600.0), levels_1h=(54000.0, 61500.0))
    assert plan.status == STATUS_NO_TRADE
    assert plan.is_reference is True  # los niveles se muestran como referencia
    assert failing_codes(plan) == [TP2_DAILY_RANGE]
    tp2 = plan.take_profit("TP2")
    assert tp2.distance_pct > 15
    rule = plan.blocking_rules[0]
    assert rule.message.startswith(
        f"No operar: el TP2 queda a {tp2.distance_pct:.2f} %".replace(".", ",")
    )
    assert rule.message.endswith(
        "de la entrada y el rango típico del día es 3 % (lo que se mueve el precio en un día "
        "normal): es poco probable llegar en una operación del día."
    )
    assert rule.suggestion == (
        "Esperá una entrada con el stop más cerca (así el TP2 también se acerca) o un día con "
        "más movimiento."
    )
    assert plan.headline.startswith("No operar: el TP2 queda a")
    # Stop lejos: posición chica, apalancamiento bajo y el riesgo sigue en 50 USD.
    assert plan.stop.distance_pct == pytest.approx(6050 / 60000 * 100)
    assert plan.position.leverage < 5
    assert plan.position.risk_usd == pytest.approx(50.0)
    assert plan.take_profit("TP2").unlikely_intraday and plan.take_profit("TP3").unlikely_intraday
    assert not plan.take_profit("TP1").unlikely_intraday  # la resistencia de 60.600 sí


def test_edge_stop_very_far_lowers_the_margin_to_1x_and_still_blocks(settings):
    # Stop a 50 %: la posición (99,82 USD) es más chica que el margen (100): el
    # margen baja a 1x (decisión del usuario). El TP2 igual queda fuera del día.
    plan = plan_for(settings, levels_15m=(30050.0, 60600.0), levels_1h=(30050.0,))
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == [TP2_DAILY_RANGE]
    assert position_codes(plan) == [MARGIN_LOWERED_TO_1X]
    assert position_messages(plan) == [
        "Margen bajado de 100 a 99,82 USD: la posición es más chica que el margen y no se "
        "puede operar a menos de 1x."
    ]
    assert plan.position.leverage == 1.0
    assert plan.position.liquidation_price is None  # un long a 1x no se liquida
    assert plan.to_dict()["liquidation"]["price"] is None


# --- timeframes en conflicto ---------------------------------------------------------------


def test_edge_timeframes_in_conflict_4h_bullish_1h_bearish(settings):
    plan = plan_for(settings, trend_4h="alcista", trend_1h="bajista")
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == [TIMEFRAME_CONFLICT]
    rule = plan.blocking_rules[0]
    assert rule.message == "No operar: 4h es alcista y 1h es bajista: van en direcciones opuestas."
    assert rule.suggestion == "Esperá a que 4h y 1h se pongan de acuerdo."
    assert plan.headline == "No operar: 4h es alcista y 1h es bajista: van en direcciones opuestas."
    assert plan.is_reference is True and len(plan.take_profits) == 3


def test_edge_timeframes_in_conflict_block_shorts_too(settings):
    plan = plan_for(settings, score=-45.0, trend_4h="bajista", trend_1h="alcista")
    assert failing_codes(plan) == [TIMEFRAME_CONFLICT]
    assert plan.blocking_rules[0].message == (
        "No operar: 4h es bajista y 1h es alcista: van en direcciones opuestas."
    )


def test_edge_sideways_timeframe_is_not_a_conflict(settings):
    plan = plan_for(settings, trend_4h="alcista", trend_1h="lateral")
    assert plan.status == STATUS_LONG


# --- liquidación antes del stop ---------------------------------------------------------------


@pytest.fixture
def thin_margin(settings):
    """Margen chico (40 USD) para 50 USD de riesgo y tope 50x.

    Con el stop a 3 % (58.200) la posición es de 1.603,49 USD (40x, dentro del
    tope) y con 40 USD de margen te liquidan en ~58.833: antes del stop.
    """
    base = get_profile("agresivo", settings, max_leverage_override=50)
    return dataclasses.replace(base, margin_per_trade_usd=40.0)


# Soporte en 58.250 → stop 58.200 (3 %). El rango del día (6 %) deja entrar el TP2.
LIQ_CASE = {"levels_15m": (58250.0, 61000.0), "levels_1h": (58250.0,), "daily_pct": 6.0}


def test_edge_liquidation_before_stop_fixed_by_raising_margin(settings, thin_margin):
    roomy = with_max_margin_multiplier(settings, 3.0)  # hasta 120 USD de margen
    plan = build_trade_plan(make_inputs(settings, profile=thin_margin, **LIQ_CASE), roomy, now=NOW)
    assert plan.status == STATUS_LONG
    assert plan.stop.price == 58200.0
    fit = plan.position
    assert fit.initial_liq_before_stop is True  # con 40 USD te liquidaban antes del stop
    assert fit.margin_usd == pytest.approx(104.66)
    assert fit.risk_usd == pytest.approx(50.0)  # el riesgo no cambia
    assert position_codes(plan) == [MARGIN_RAISED]
    assert position_messages(plan) == [
        "Margen subido de 40 a 104,66 USD para que no te liquiden antes de llegar al stop "
        "(la liquidación tiene que quedar al menos a 2 veces la distancia al stop); el riesgo "
        "sigue en 50 USD."
    ]
    liq_rule = next(r for r in plan.rules if r.code == LIQUIDATION)
    assert liq_rule.ok and liq_rule.message.startswith("La liquidación (")
    assert fit.liquidation_ratio >= 2.0


def test_edge_liquidation_before_stop_blocked_when_no_adjustment_is_enough(settings, thin_margin):
    # Margen máximo 80 USD (2 × 40): no alcanza, se achica la posición y el riesgo
    # (38,22 USD) queda debajo del mínimo operable de este perfil (40 USD).
    picky = dataclasses.replace(thin_margin, min_operable_risk_usd=40.0)
    plan = plan_for(settings, profile=picky, **LIQ_CASE)
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == [MIN_OPERABLE_RISK]
    assert position_codes(plan) == [MARGIN_RAISED, SIZE_REDUCED]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: aun con el margen máximo de 80 USD la posición entra solo hasta "
        "1.225,79 USD y arriesgarías 38,22 USD, menos que el mínimo operable de 40 USD."
    )
    assert rule.suggestion == (
        "Esperá otra entrada: con este stop y el margen máximo, la liquidación quedaría "
        "demasiado cerca."
    )
    # La liquidación en sí quedó bien (por eso se achicó): lo que falla es el riesgo.
    assert next(r for r in plan.rules if r.code == LIQUIDATION).ok


def test_edge_liquidation_rule_blocks_when_no_adjustment_is_possible(settings):
    # Short con el stop 60 % arriba: a 1x (no se puede menos) la liquidación queda a
    # 1,65 veces la distancia al stop, menos de 2: no hay ajuste que sirva.
    plan = plan_for(
        settings,
        score=-45.0,
        levels_15m=(95950.0, 59000.0),
        levels_1h=(95950.0,),
        daily_pct=200.0,  # para aislar la regla de liquidación
    )
    assert plan.stop.price == 96000.0
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == [LIQUIDATION]
    assert position_codes(plan) == [MARGIN_RAISED, MARGIN_LOWERED_TO_1X]
    rule = plan.blocking_rules[0]
    assert rule.message.startswith("No operar: la liquidación (")
    assert rule.message.endswith(
        "queda a 1,65 veces la distancia al stop y tiene que estar al menos a 2 veces."
    )
    assert rule.suggestion.startswith("No hay ajuste posible: con el stop tan lejos")
    assert plan.position.leverage == 1.0


# --- apalancamiento excedido ---------------------------------------------------------------

# ATR 40: soportes a más de 1 ATR y stop = soporte − 20.
LEV_ATR = 40.0


def lev_plan(settings, support: float):
    return plan_for(
        settings, atr=LEV_ATR, levels_15m=(support, 60300.0), levels_1h=(support, 61500.0)
    )


def test_edge_leverage_exceeded_fixed_by_raising_margin(settings):
    # Stop a 1 % (59.400): 4.466,68 USD de posición; con 100 USD serían 44,7x.
    plan = lev_plan(settings, 59420.0)
    assert plan.status == STATUS_LONG
    assert plan.stop.price == 59400.0
    fit = plan.position
    assert fit.initial_leverage == pytest.approx(44.67, abs=0.01)
    assert fit.margin_usd == 178.67
    assert fit.leverage == pytest.approx(25, abs=0.01)
    assert fit.risk_usd == pytest.approx(50.0)
    assert position_messages(plan) == [
        "Margen subido de 100 a 178,67 USD para no pasar de 25x y para que la liquidación "
        "quede al menos a 2 veces la distancia al stop; el riesgo sigue en 50 USD."
    ]
    assert next(r for r in plan.rules if r.code == "leverage").ok


def test_edge_leverage_exceeded_shrinks_the_position(settings):
    # Stop a 0,5 % (59.700): harían falta 322,74 USD de margen; el máximo es 200.
    plan = lev_plan(settings, 59720.0)
    assert plan.status == STATUS_LONG
    fit = plan.position
    assert (fit.margin_usd, fit.size_usd, fit.leverage) == (200.0, 5000.0, 25.0)
    assert fit.risk_usd == pytest.approx(30.98, abs=0.01)
    assert position_messages(plan) == [
        "Margen subido de 100 a 200 USD, el máximo permitido; no alcanza para mantener el riesgo.",
        "Posición achicada de 8.068,42 a 5.000 USD: con el margen máximo de 200 USD no entra "
        "más sin pasar de 25x; ahora arriesgás 30,98 USD (en vez de 50 USD).",
    ]
    # Con menos riesgo, la ganancia también baja (el R:R se mantiene).
    assert plan.take_profit("TP2").net_gain_usd == pytest.approx(1.5 * fit.risk_usd)
    assert plan.headline.startswith("LONG a mercado en 60.000")
    assert "arriesgás 30,98 USD" in plan.headline


def test_edge_leverage_exceeded_blocked_below_min_operable_risk(settings):
    # Stop a 0,15 % (59.910): con 5.000 USD de posición se arriesgan 13,50 USD (< 15).
    plan = lev_plan(settings, 59930.0)
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == [MIN_OPERABLE_RISK]
    assert position_codes(plan) == [MARGIN_RAISED, SIZE_REDUCED]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: aun con el margen máximo de 200 USD la posición entra solo hasta "
        "5.000 USD y arriesgarías 13,50 USD, menos que el mínimo operable de 15 USD."
    )
    assert rule.suggestion == (
        "Esperá una entrada con el stop más lejos: con un stop tan corto la posición no "
        "entra en tu tope de apalancamiento."
    )
    assert plan.headline == rule.message
    # El apalancamiento nunca pasa el tope: lo que falla es el riesgo mínimo.
    assert next(r for r in plan.rules if r.code == "leverage").ok


# --- régimen Crash ----------------------------------------------------------------------


def test_edge_crash_regime_blocks_longs(settings):
    plan = plan_for(settings, score=80.0, regime=fake_regime("crash", confidence=0.95))
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == ["regime"]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: con el régimen Crash (caída fuerte) los longs están bloqueados."
    )
    assert rule.suggestion == "Esperá a que cambie el régimen de mercado."
    assert plan.headline == rule.message
    regime_adj = [a.message for a in plan.adjustments if a.source == "regime"]
    assert regime_adj == ["Longs bloqueados: régimen Crash (caída fuerte)."]
    assert plan.is_reference is True and plan.stop is not None  # niveles de referencia
    assert plan.regime_summary["category"] == "crash"
    assert plan.to_dict()["regime"]["label"] == "Crash (caída fuerte)"


def test_edge_crash_regime_allows_shorts_with_half_the_risk(settings):
    plan = plan_for(settings, score=-60.0, regime=fake_regime("crash", confidence=0.95))
    assert plan.status == STATUS_SHORT
    assert plan.adjusted_risk_usd == pytest.approx(25.0)
    assert plan.position.risk_usd == pytest.approx(25.0)
    assert [a.message for a in plan.adjustments if a.source == "regime"] == [
        "Riesgo × 0,5: régimen Crash (caída fuerte)."
    ]
    assert plan.entry_price == ENTRY
