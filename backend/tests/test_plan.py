"""Hito 4: generador del plan de operación (``app.risk.plan``).

Los ``PlanInputs`` se arman a mano (soportes y resistencias elegidos, régimen de
mentira) para que cada número sea exacto. Los casos borde obligatorios del
usuario están en ``test_plan_edge_cases.py``, que reutiliza estos ayudantes.

Valores de la config que asumen los números de abajo (ver ``_default_costs``):
comisiones 0,06 % de entrada y de salida, margen de mantenimiento 0,5 %,
perfil agresivo (margen 100, riesgo 50, tope 25x, R:R mínimo 1,5, mínimo
operable 15), TP2 a 1,5R, TP3 a 2,5R, stop por tiempo de 24 h.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.data.snapshot import build_offline_snapshot
from app.indicators.levels import LevelCluster, SRResult, nearest_levels
from app.indicators.pipeline import analyze_all
from app.risk.fees import fee_rates_from_config, funding_cost_usd, net_rr
from app.risk.liquidation import mmr_from_config
from app.risk.plan import (
    DAILY_LOSS_CAP,
    ENTRY_LIMIT,
    ENTRY_MARKET,
    FLAG_TP1_BEYOND_TP2,
    FLAG_UNLIKELY_INTRADAY,
    PRICE_CHECK,
    REGIME_RULE,
    SOURCE_DAILY_LIMIT,
    SOURCE_POSITION,
    SOURCE_REGIME,
    STATUS_LONG,
    STATUS_NO_TRADE,
    STATUS_SHORT,
    STOP_RULE,
    TP2_DAILY_RANGE,
    TP_AFTER_TP2,
    TP_MIN_NET_RR,
    TP_NEXT_LEVEL,
    TP_R_MULTIPLE,
    PlanInputs,
    TradePlan,
    build_trade_plan,
    min_rr_target,
    plan_inputs_from_analysis,
)
from app.risk.profiles import get_profile
from app.risk.rules import (
    DAILY_LIMITS,
    DIRECTION,
    HIGH_VOLATILITY_EVENT,
    MIN_OPERABLE_RISK,
    MIN_RR,
    fmt_num,
    fmt_price,
)
from app.risk.rules import TIMEFRAME_CONFLICT as TF_CONFLICT
from app.scoring import compute_score, inputs_from_analysis
from app.scoring.components import funding_pct_per_8h
from app.scoring.confluence import ConfluenceScore, direction_from_total

NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)
ENTRY = 60000.0
ATR = 100.0
# Soportes y resistencias por omisión: sirven para long y para short.
LEVELS_15M = (59500.0, 60600.0)
LEVELS_1H = (58800.0, 61500.0)
DAILY_PCT = 3.0  # rango típico del día: 1.800 USD con la entrada en 60.000


# --- ayudantes (también los usa test_plan_edge_cases.py) ----------------------------


def cluster(price: float) -> LevelCluster:
    return LevelCluster(
        price=price,
        touches=2,
        sources=frozenset({"pivot_low"}),
        last_touch=None,
        zone_low=price,
        zone_high=price,
    )


def sr(*prices: float, reference: float = ENTRY) -> SRResult:
    """``SRResult`` con niveles en esos precios (lo que usa el plan son los ``clusters``)."""
    clusters = tuple(sorted((cluster(p) for p in prices), key=lambda c: c.price))
    supports, resistances = nearest_levels(clusters, reference, 3)
    return SRResult(
        supports=tuple(supports),
        resistances=tuple(resistances),
        prev_day_high=None,
        prev_day_low=None,
        pivots_used=(),
        prev_day=None,
        reference_price=reference,
        cluster_tolerance=0.0,
        clusters=clusters,
    )


def make_score(total: float, settings) -> ConfluenceScore:
    th = settings.scoring.direction_thresholds
    return ConfluenceScore(
        total=total,
        components=(),
        missing=(),
        direction_hint=direction_from_total(total, th.long, th.short),
        long_threshold=th.long,
        short_threshold=th.short,
        summary=f"Score {total}",
    )


def fake_regime(category: str = "alcista", confidence: float = 0.9, unstable: bool = False):
    return SimpleNamespace(
        category=category,
        confidence=confidence,
        unstable=unstable,
        name=category.capitalize(),
        timeframe="1h",
    )


def make_inputs(
    settings,
    *,
    score: float = 45.0,
    entry: float = ENTRY,
    levels_15m: tuple[float, ...] = LEVELS_15M,
    levels_1h: tuple[float, ...] = LEVELS_1H,
    atr: float | None = ATR,
    trend_4h: str | None = "alcista",
    trend_1h: str | None = "alcista",
    daily_pct: float | None = DAILY_PCT,
    daily_usd: float | None = None,
    funding_pct_8h: float | None = 0.01,
    regime=None,
    profile=None,
    price_source: str = "manual_nexo",
    **extra,
) -> PlanInputs:
    """Entradas a mano. Régimen por omisión: alcista (long) o bajista (short), estable."""
    if regime is None:
        regime = fake_regime("alcista" if score >= 0 else "bajista")
    return PlanInputs(
        symbol="BTC",
        entry_price=entry,
        price_source=price_source,
        levels={"15m": sr(*levels_15m, reference=entry), "1h": sr(*levels_1h, reference=entry)},
        entry_atr=atr,
        trend_4h=trend_4h,
        trend_1h=trend_1h,
        daily_range_usd=daily_usd,
        daily_range_pct=daily_pct,
        score=make_score(score, settings),
        funding_pct_8h=funding_pct_8h,
        regime=regime,
        profile=profile or get_profile("agresivo", settings),
        **extra,
    )


def plan_for(settings, **kwargs) -> TradePlan:
    return build_trade_plan(make_inputs(settings, **kwargs), settings, now=NOW)


def failing_codes(plan: TradePlan) -> list[str]:
    return [r.code for r in plan.blocking_rules]


def with_take_profit(settings, **update):
    tp = settings.trade_plan
    take = tp.take_profit.model_copy(update=update)
    return settings.model_copy(update={"trade_plan": tp.model_copy(update={"take_profit": take})})


def with_trade_plan(settings, **update):
    return settings.model_copy(update={"trade_plan": settings.trade_plan.model_copy(update=update)})


def with_fees(settings, **update):
    return settings.model_copy(update={"fees": settings.fees.model_copy(update=update)})


@pytest.fixture(autouse=True)
def _default_costs(settings):
    """Los números fijos de estos tests asumen la config de comisiones y del plan."""
    assert fee_rates_from_config(settings.fees) == pytest.approx((0.0006, 0.0006))
    assert mmr_from_config(settings.liquidation) == pytest.approx(0.005)
    tp = settings.trade_plan
    assert (tp.stop_atr_buffer, tp.min_stop_distance_atr) == (0.5, 1.0)
    assert (tp.take_profit.tp2_r, tp.take_profit.tp3_r) == (1.5, 2.5)
    assert tp.take_profit.tp1_min_distance_atr == 1.0
    assert tp.take_profit.tp2_ensure_min_rr and tp.take_profit.block_if_tp2_beyond_daily_range
    assert tp.time_stop_hours == 24 and tp.fees_included_in_risk is True
    assert settings.timeframes.entry == "15m" and tp.stop_timeframes == ["15m", "1h"]


# --- un plan normal ------------------------------------------------------------------


def test_default_long_plan_is_complete(settings):
    plan = plan_for(settings)
    assert plan.status == STATUS_LONG and plan.direction == "long"
    assert plan.ok and not plan.is_reference and plan.entry_kind == ENTRY_MARKET
    assert all(r.ok for r in plan.rules)
    # Stop: soporte de 15m (59.500) menos 0,5 × ATR (100).
    assert plan.stop.price == 59450.0
    assert plan.stop.level_timeframe == "15m" and plan.stop.level_kind == "soporte"
    assert plan.stop.distance_pct == pytest.approx(550 / 60000 * 100)
    assert plan.stop.explanation == "Soporte de 15m en 59.500 menos 0,5 × ATR 15m (100)."
    assert [tp.name for tp in plan.take_profits] == ["TP1", "TP2", "TP3"]
    tp1 = plan.take_profit("TP1")
    assert tp1.price == 60600.0 and tp1.basis == TP_NEXT_LEVEL and tp1.level_timeframe == "15m"
    assert plan.headline.startswith("LONG a mercado en 60.000: stop en 59.450 y objetivo")
    assert "arriesgás 50 USD (comisiones incluidas)" in plan.headline
    assert plan.position.risk_usd == pytest.approx(50.0)
    assert plan.requested_risk_usd == 50.0 and plan.adjusted_risk_usd == 50.0
    # Las reglas van todas, en orden.
    assert [r.code for r in plan.rules] == [
        DIRECTION,
        REGIME_RULE,
        "min_abs_score",
        TF_CONFLICT,
        DAILY_LIMITS,
        HIGH_VOLATILITY_EVENT,
        STOP_RULE,
        MIN_RR,
        TP2_DAILY_RANGE,
        "leverage",
        "liquidation",
        "min_operable_risk",
    ]


def test_default_short_plan_is_mirrored(settings):
    plan = plan_for(settings, score=-45.0)
    assert plan.status == STATUS_SHORT and plan.direction == "short"
    assert plan.stop.price == 60650.0  # resistencia 60.600 + 0,5 × ATR
    assert plan.stop.level_kind == "resistencia"
    tp1, tp2, tp3 = plan.take_profits
    assert tp1.price == 59500.0  # siguiente soporte
    assert tp3.price < tp2.price < plan.entry_price
    assert plan.headline.startswith("SHORT a mercado en 60.000: stop en 60.650")


# --- dirección -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "status"),
    [
        (30.0, STATUS_LONG),
        (29.99, STATUS_NO_TRADE),
        (-29.99, STATUS_NO_TRADE),
        (-30.0, STATUS_SHORT),
    ],
)
def test_direction_thresholds(settings, score, status):
    plan = plan_for(settings, score=score)
    assert plan.status == status
    if status == STATUS_NO_TRADE:
        assert plan.direction is None
        assert failing_codes(plan) == [DIRECTION]
        assert plan.stop is None and plan.take_profits == () and plan.position is None
        assert plan.is_reference is False and plan.alternative is None
        assert plan.time_stop is None and plan.fees is None
        assert plan.headline.startswith(f"No operar: el score ({'+' if score > 0 else ''}")
        assert "no hay una dirección clara" in plan.headline


def test_no_direction_still_checks_the_global_rules(settings):
    plan = plan_for(
        settings,
        score=10.0,
        trend_4h="alcista",
        trend_1h="bajista",
        trades_today=3,
        high_volatility_event=True,
    )
    assert failing_codes(plan) == [DIRECTION, TF_CONFLICT, DAILY_LIMITS, HIGH_VOLATILITY_EVENT]
    assert plan.headline.endswith("(y 3 motivos más).")


# --- stop ---------------------------------------------------------------------------


def test_stop_skips_levels_closer_than_one_atr(settings):
    plan = plan_for(settings, levels_15m=(59950.0, 59920.0, 59700.0, 60600.0))
    assert plan.stop.level_price == 59700.0 and plan.stop.price == 59650.0
    assert [s.price for s in plan.stop.skipped] == [59950.0, 59920.0]


def test_stop_not_found_blocks_without_levels(settings):
    plan = plan_for(settings, levels_15m=(60600.0,), levels_1h=(61500.0,))
    assert plan.status == STATUS_NO_TRADE and plan.is_reference is False
    rule = next(r for r in plan.rules if r.code == STOP_RULE)
    assert rule.ok is False
    assert rule.message == "No operar: no hay soportes debajo del precio en 15m/1h."
    assert rule.suggestion
    assert plan.stop is None and plan.position is None and plan.take_profits == ()
    assert plan.stop_missing_reason == "No hay soportes debajo del precio en 15m/1h."


def test_stop_without_atr_blocks(settings):
    plan = plan_for(settings, atr=None)
    rule = next(r for r in plan.rules if r.code == STOP_RULE)
    assert rule.message == "No operar: no hay ATR de 15m: no se puede calcular el stop."


# --- objetivos ------------------------------------------------------------------------


def test_tp1_skips_levels_closer_than_one_atr(settings):
    plan = plan_for(settings, levels_15m=(59500.0, 60050.0, 60090.0, 60300.0))
    tp1 = plan.take_profit("TP1")
    assert tp1.price == 60300.0
    assert [s.price for s in tp1.skipped] == [60050.0, 60090.0]
    assert [round(s.distance_atr, 2) for s in tp1.skipped] == [0.5, 0.9]
    assert tp1.explanation == (
        "Siguiente resistencia de 15m (se saltearon 2 niveles a menos de 1 ATR)."
    )
    no_skip = build_trade_plan(
        make_inputs(settings, levels_15m=(59500.0, 60050.0, 60090.0, 60300.0)),
        with_take_profit(settings, tp1_min_distance_atr=0.0),
        now=NOW,
    )
    assert no_skip.take_profit("TP1").price == 60050.0


def test_tp1_short_skips_levels_too(settings):
    plan = plan_for(settings, score=-45.0, levels_15m=(60600.0, 59960.0, 59700.0))
    tp1 = plan.take_profit("TP1")
    assert tp1.price == 59700.0 and [s.price for s in tp1.skipped] == [59960.0]


def test_tp1_takes_the_nearest_level_across_timeframes(settings):
    plan = plan_for(settings, levels_15m=(59500.0, 61200.0), levels_1h=(58800.0, 60400.0))
    tp1 = plan.take_profit("TP1")
    assert tp1.price == 60400.0 and tp1.level_timeframe == "1h"


def test_missing_tp1_does_not_block(settings):
    plan = plan_for(settings, levels_15m=(59500.0,), levels_1h=(58800.0,))
    assert plan.status == STATUS_LONG
    assert plan.take_profit("TP1") is None
    assert plan.tp1_missing_reason == "No hay TP1: no hay resistencias arriba del precio en 15m/1h."
    assert plan.time_stop.target_name == "TP2"


def test_tp1_beyond_tp2_is_dropped_and_time_stop_uses_tp2(settings):
    # Decisión del usuario: la resistencia de 61.400 queda más lejos que el TP2
    # (~61.005): no hay TP1, quedan TP2 y TP3 y el stop por tiempo mira el TP2.
    assert settings.trade_plan.take_profit.drop_tp1_beyond_tp2 is True
    plan = plan_for(settings, levels_15m=(59500.0, 61400.0), levels_1h=(58800.0,))
    assert plan.status == STATUS_LONG
    tp2 = plan.take_profit("TP2")
    assert tp2.price < 61400.0
    assert plan.take_profit("TP1") is None
    assert [tp.name for tp in plan.take_profits] == ["TP2", "TP3"]
    assert plan.tp1_missing_reason == (
        "No hay TP1: la siguiente resistencia (15m, en 61.400) queda más lejos que el TP2 "
        f"({fmt_price(tp2.price, ENTRY)}); el primer objetivo es el TP2."
    )
    assert plan.time_stop.target_name == "TP2"
    assert plan.to_dict()["tp1_missing_reason"] == plan.tp1_missing_reason


def test_tp1_beyond_tp2_is_dropped_in_a_short_too(settings):
    plan = plan_for(settings, score=-45.0, levels_15m=(60500.0, 58600.0), levels_1h=(61200.0,))
    assert plan.status == STATUS_SHORT and plan.stop.price == 60550.0
    tp2 = plan.take_profit("TP2")
    assert 58600.0 < tp2.price < ENTRY
    assert plan.take_profit("TP1") is None
    assert plan.tp1_missing_reason.startswith(
        "No hay TP1: el siguiente soporte (15m, en 58.600) queda más lejos que el TP2"
    )
    assert plan.time_stop.target_name == "TP2"


def test_tp1_exactly_at_tp2_is_dropped(settings):
    # TP2 fijo en 1,5R (sin el corrimiento por comisiones): 60.000 + 1,5 × 550 = 60.825.
    s = with_take_profit(settings, tp2_ensure_min_rr=False)
    plan = build_trade_plan(
        make_inputs(s, levels_15m=(59500.0, 60825.0), levels_1h=(58800.0,)), s, now=NOW
    )
    assert plan.take_profit("TP2").price == pytest.approx(60825.0)
    assert plan.take_profit("TP1") is None
    assert "queda en el mismo precio que el TP2" in plan.tp1_missing_reason


def test_tp1_beyond_tp2_is_only_flagged_if_the_config_says_so(settings):
    s = with_take_profit(settings, drop_tp1_beyond_tp2=False)
    plan = build_trade_plan(
        make_inputs(s, levels_15m=(59500.0, 61400.0), levels_1h=(58800.0,)), s, now=NOW
    )
    tp1, tp2 = plan.take_profit("TP1"), plan.take_profit("TP2")
    assert tp1.price > tp2.price
    assert FLAG_TP1_BEYOND_TP2 in tp1.flags
    assert plan.tp1_missing_reason is None
    assert plan.time_stop.target_name == "TP2"


def test_tp1_before_tp2_is_kept(settings):
    plan = plan_for(settings)
    tp1, tp2 = plan.take_profit("TP1"), plan.take_profit("TP2")
    assert tp1.price < tp2.price and FLAG_TP1_BEYOND_TP2 not in tp1.flags
    assert plan.time_stop.target_name == "TP1"


@pytest.mark.parametrize("side_score", [45.0, -45.0])
@pytest.mark.parametrize("min_rr", [1.5, 2.0])
def test_tp2_reaches_the_min_net_rr_exactly(settings, side_score, min_rr):
    profile = dataclasses.replace(get_profile("agresivo", settings), min_rr=min_rr)
    plan = plan_for(settings, score=side_score, profile=profile, daily_pct=10.0)
    assert plan.status in (STATUS_LONG, STATUS_SHORT)
    side = plan.direction
    f_in, f_out = fee_rates_from_config(settings.fees)
    tp2 = plan.take_profit("TP2")
    assert tp2.basis == TP_MIN_NET_RR
    rr = net_rr(side, ENTRY, plan.stop.price, tp2.price, f_in, f_out)
    assert rr >= min_rr  # nunca un pelo corto (sin tolerancia)
    assert rr == pytest.approx(min_rr, rel=1e-12)
    assert tp2.net_rr == rr and plan.net_rr_tp2 == rr
    assert plan.min_rr == min_rr
    # Con comisiones, el TP2 queda algo más lejos que min_rr × R.
    assert tp2.r_multiple > max(min_rr, 1.5)
    # La fórmula despejada a mano: (k·L + E·(1 + f)) / (1 − f) en long.
    stop = plan.stop.price
    loss = abs(ENTRY - stop) + f_in * ENTRY + f_out * stop
    expected = (
        (min_rr * loss + ENTRY * (1 + f_in)) / (1 - f_out)
        if side == "long"
        else (ENTRY * (1 - f_in) - min_rr * loss) / (1 + f_out)
    )
    assert tp2.price == pytest.approx(expected, rel=1e-12)
    assert "lo justo para que el R:R con comisiones llegue a" in tp2.explanation


def test_min_rr_target_matches_net_rr(settings):
    for side, stop in (("long", 59000.0), ("short", 61000.0)):
        for k in (1.0, 1.5, 2.0, 3.3):
            target = min_rr_target(side, ENTRY, stop, k, 0.0006, 0.0006)
            assert net_rr(side, ENTRY, stop, target, 0.0006, 0.0006) >= k
            assert net_rr(side, ENTRY, stop, target, 0.0006, 0.0006) == pytest.approx(k)


def test_tp2_stays_at_1_5r_when_that_is_enough(settings):
    profile = dataclasses.replace(get_profile("agresivo", settings), min_rr=1.2)
    plan = plan_for(settings, profile=profile)
    tp2 = plan.take_profit("TP2")
    assert tp2.basis == TP_R_MULTIPLE
    assert tp2.price == pytest.approx(ENTRY + 1.5 * 550)
    assert tp2.r_multiple == pytest.approx(1.5)
    assert tp2.explanation == "1,5R (1,5 veces el riesgo)."


def test_tp2_fixed_at_1_5r_blocks_by_net_rr_when_not_ensured(settings):
    # La razón de la decisión del usuario: con el TP2 fijo en 1,5R, las comisiones
    # dejan el R:R neto apenas debajo de 1,5 y la regla bloquea.
    s = with_take_profit(settings, tp2_ensure_min_rr=False)
    plan = build_trade_plan(make_inputs(settings), s, now=NOW)
    tp2 = plan.take_profit("TP2")
    assert tp2.basis == TP_R_MULTIPLE and tp2.price == pytest.approx(ENTRY + 1.5 * 550)
    assert tp2.net_rr < 1.5
    assert failing_codes(plan) == [MIN_RR]
    rule = plan.blocking_rules[0]
    assert rule.message.startswith("No operar: el R:R a TP2 con comisiones es 1,21 ")
    assert "y el mínimo es 1,5" in rule.message


def test_tp3_is_tp2_plus_one_r_when_tp2_was_pushed(settings):
    plan = plan_for(settings)
    tp2, tp3 = plan.take_profit("TP2"), plan.take_profit("TP3")
    assert tp2.basis == TP_MIN_NET_RR
    assert tp3.basis == TP_AFTER_TP2
    assert tp3.price - tp2.price == pytest.approx(1.0 * 550)  # (2,5 − 1,5) × R
    assert tp3.explanation.endswith("el TP2 más 1R.")


def test_tp3_is_2_5r_when_tp2_is_1_5r(settings):
    profile = dataclasses.replace(get_profile("agresivo", settings), min_rr=1.2)
    plan = plan_for(settings, profile=profile, score=-45.0)
    tp3 = plan.take_profit("TP3")
    assert tp3.basis == TP_R_MULTIPLE
    assert tp3.price == pytest.approx(ENTRY - 2.5 * 650)
    assert tp3.r_multiple == pytest.approx(2.5)


def test_take_profit_gains_use_the_final_quantity(settings):
    plan = plan_for(settings)
    q = plan.position.quantity
    for tp in plan.take_profits:
        gross = q * (tp.price - ENTRY)
        fees = q * ENTRY * 0.0006 + q * tp.price * 0.0006
        assert tp.gross_gain_usd == pytest.approx(gross)
        assert tp.fees_usd == pytest.approx(fees)
        assert tp.net_gain_usd == pytest.approx(gross - fees)
        assert tp.distance_pct == pytest.approx((tp.price - ENTRY) / ENTRY * 100)
        assert tp.r_multiple == pytest.approx((tp.price - ENTRY) / 550)
    # En el TP2 se gana 1,5 veces lo que se pierde en el stop (con comisiones).
    assert plan.take_profit("TP2").net_gain_usd == pytest.approx(1.5 * plan.position.risk_usd)


# --- rango diario ------------------------------------------------------------------


def test_tp_beyond_daily_range_is_flagged_unlikely_intraday(settings):
    # TP2 a 1,68 % y TP3 a 2,59 %: con un rango diario de 2 % solo el TP3 queda afuera.
    plan = plan_for(settings, daily_pct=2.0)
    assert plan.status == STATUS_LONG
    tp1, tp2, tp3 = plan.take_profits
    assert not tp1.beyond_daily_range and not tp2.beyond_daily_range
    assert tp3.beyond_daily_range and tp3.flags == (FLAG_UNLIKELY_INTRADAY,)
    assert tp3.unlikely_intraday and not tp2.unlikely_intraday
    rule = next(r for r in plan.rules if r.code == TP2_DAILY_RANGE)
    assert rule.ok and rule.message.startswith("El TP2 (a 1,68 %)")
    assert rule.message.endswith("entra en el rango típico del día (2 %).")


def test_tp2_beyond_daily_range_blocks(settings):
    plan = plan_for(settings, daily_pct=1.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing_codes(plan) == [TP2_DAILY_RANGE]
    rule = plan.blocking_rules[0]
    tp2 = plan.take_profit("TP2")
    assert rule.message == (
        f"No operar: el TP2 queda a {tp2.distance_pct:.2f} % de la entrada y el rango típico "
        "del día es 1 % (lo que se mueve el precio en un día normal): es poco probable "
        "llegar en una operación del día.".replace(".", ",", 1)
    )
    assert rule.suggestion.startswith("Esperá una entrada con el stop más cerca")
    assert all(tp.unlikely_intraday for tp in plan.take_profits[1:])


def test_tp2_beyond_daily_range_only_flagged_if_block_is_off(settings):
    s = with_take_profit(settings, block_if_tp2_beyond_daily_range=False)
    plan = build_trade_plan(make_inputs(settings, daily_pct=1.0), s, now=NOW)
    assert plan.status == STATUS_LONG
    assert plan.take_profit("TP2").unlikely_intraday
    rule = next(r for r in plan.rules if r.code == TP2_DAILY_RANGE)
    assert rule.ok and "la config no bloquea por esto" in rule.message


def test_flags_can_be_turned_off_but_the_block_stays(settings):
    s = with_trade_plan(settings, flag_tp_beyond_daily_range=False)
    plan = build_trade_plan(make_inputs(settings, daily_pct=1.0), s, now=NOW)
    assert all(tp.flags == () for tp in plan.take_profits)
    assert plan.take_profit("TP2").beyond_daily_range
    assert failing_codes(plan) == [TP2_DAILY_RANGE]


def test_daily_range_from_usd_or_pct(settings):
    by_usd = plan_for(settings, daily_pct=None, daily_usd=1200.0)
    assert by_usd.daily_range_usd == 1200.0 and by_usd.daily_range_pct == pytest.approx(2.0)
    by_pct = plan_for(settings, daily_pct=2.0)
    assert by_pct.daily_range_usd == pytest.approx(1200.0)


@pytest.mark.parametrize("score", [45.0, -45.0])
def test_missing_daily_range_blocks(settings, score):
    # Decisión del usuario: sin velas diarias no se sabe si el TP2 es alcanzable hoy.
    assert settings.trade_plan.block_if_daily_range_missing is True
    plan = plan_for(settings, score=score, daily_pct=None)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing_codes(plan) == [TP2_DAILY_RANGE]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: no se puede saber si el objetivo (TP2) es alcanzable hoy: faltan velas "
        "diarias para medir lo que se mueve el precio en un día normal."
    )
    assert rule.suggestion == "Volvé a intentar cuando se puedan bajar las velas diarias."
    assert plan.headline.startswith("No operar: no se puede saber si el objetivo (TP2)")
    # La regla ya lo dice: no se repite como aviso.
    assert not any("rango típico del día" in w for w in plan.warnings)
    assert len(plan.take_profits) == 3  # los niveles quedan como referencia


def test_missing_daily_range_only_warns_if_the_config_says_so(settings):
    s = with_trade_plan(settings, block_if_daily_range_missing=False)
    plan = build_trade_plan(make_inputs(s, daily_pct=None), s, now=NOW)
    assert plan.status == STATUS_LONG
    rule = next(r for r in plan.rules if r.code == TP2_DAILY_RANGE)
    assert rule.ok and rule.message.startswith("No hay dato del rango típico del día")
    assert any("rango típico del día" in w for w in plan.warnings)
    assert all(not tp.beyond_daily_range for tp in plan.take_profits)


# --- comisiones, funding y stop por tiempo -------------------------------------------


def test_fees_in_usd_and_as_pct_of_risk_and_target(settings):
    plan = plan_for(settings)
    pos, fees, tp2 = plan.position, plan.fees, plan.take_profit("TP2")
    q = pos.quantity
    assert fees.entry_rate_pct == pytest.approx(0.06) and fees.exit_rate_pct == pytest.approx(0.06)
    assert fees.entry_fee_usd == pytest.approx(q * ENTRY * 0.0006)  # sobre el tamaño
    assert fees.entry_fee_usd == pytest.approx(pos.size_usd * 0.0006)
    assert fees.exit_fee_at_stop_usd == pytest.approx(q * plan.stop.price * 0.0006)
    assert fees.round_trip_at_stop_usd == pytest.approx(pos.fees_usd)
    assert fees.pct_of_risk == pytest.approx(fees.round_trip_at_stop_usd / pos.risk_usd * 100)
    assert fees.target_name == "TP2"
    assert fees.exit_fee_at_target_usd == pytest.approx(q * tp2.price * 0.0006)
    assert fees.round_trip_at_target_usd == pytest.approx(tp2.fees_usd)
    assert fees.pct_of_target == pytest.approx(tp2.fees_usd / tp2.gross_gain_usd * 100)
    assert fees.included_in_risk is True
    # Las comisiones están dentro de los 50 USD de riesgo.
    assert pos.price_loss_at_stop_usd + pos.fees_usd == pytest.approx(50.0)


def test_time_stop_deadline(settings):
    plan = plan_for(settings)
    assert plan.time_stop.hours == 24
    assert plan.time_stop.deadline == NOW + timedelta(hours=24)
    assert plan.time_stop.target_name == "TP1"
    assert plan.time_stop.text.startswith("Si en 24 h no tocó el TP1 ni el stop, cerrá")
    s = with_trade_plan(settings, time_stop_hours=12)
    short = build_trade_plan(make_inputs(settings), s, now=NOW)
    assert short.time_stop.deadline == NOW + timedelta(hours=12)


def test_now_needs_a_timezone(settings):
    with pytest.raises(ValueError, match="zona horaria"):
        build_trade_plan(make_inputs(settings), settings, now=datetime(2026, 10, 12, 14, 0))


@pytest.mark.parametrize(
    ("score", "funding", "pays"),
    [(45.0, 0.01, True), (-45.0, 0.01, False), (45.0, -0.02, False), (-45.0, -0.02, True)],
)
def test_funding_estimate_sign(settings, score, funding, pays):
    plan = plan_for(settings, score=score, funding_pct_8h=funding)
    est = plan.funding
    assert est.available and est.hours == 24 and est.rate_pct_8h == funding
    assert (est.cost_usd > 0) == pays
    size = plan.position.size_usd
    expected = funding_cost_usd(plan.direction, size, funding / 100, 24, 8)
    assert est.cost_usd == pytest.approx(expected)
    # 24 h con cobros cada 8 h: 3 cobros.
    assert abs(est.cost_usd) == pytest.approx(size * abs(funding) / 100 * 3)
    word = "pagarías" if pays else "cobrarías"
    assert word in est.text


def test_funding_estimate_with_a_4h_interval_is_the_same_per_day(settings):
    eight = plan_for(settings, funding_pct_8h=0.01)
    four = plan_for(settings, funding_pct_8h=0.01, funding_interval_hours=4.0)
    assert four.funding.interval_hours == 4.0
    assert four.funding.cost_usd == pytest.approx(eight.funding.cost_usd)


def test_funding_estimate_without_data_or_disabled(settings):
    plan = plan_for(settings, funding_pct_8h=None)
    assert plan.funding.available is False and plan.funding.cost_usd is None
    assert plan.funding.text.startswith("Sin dato de funding")
    off = build_trade_plan(
        make_inputs(settings), with_fees(settings, include_funding_estimate=False), now=NOW
    )
    assert off.funding is None


# --- entrada alternativa ----------------------------------------------------------------


def test_alternative_entry_long_at_nearest_15m_support(settings):
    plan = plan_for(settings, levels_15m=(59800.0, 59500.0, 60600.0))
    assert plan.stop.price == 59750.0
    alt = plan.alternative
    assert alt.available and alt.kind == ENTRY_LIMIT and alt.timeframe == "15m"
    assert alt.level_kind == "soporte" and alt.level_price == 59800.0
    assert alt.distance_pct == pytest.approx(-200 / 60000 * 100)
    assert alt.skipped == ()
    ap = alt.plan
    assert ap.entry_kind == ENTRY_LIMIT and ap.entry_price == 59800.0
    assert ap.alternative is None  # la alternativa no tiene su propia alternativa
    # Su propio stop: el soporte de 59.500 (a 3 ATR de 59.800) menos el colchón.
    assert ap.stop.price == 59450.0
    assert ap.headline.startswith("LONG con orden límite en 59.800")
    d = alt.to_dict()
    assert d["entry"] == 59800.0 and d["stop"] == 59450.0 and d["status"] == STATUS_LONG
    assert d["tp2"] == ap.take_profit("TP2").price
    assert d["max_loss_usd"] == pytest.approx(ap.position.risk_usd)
    assert d["skipped_levels"] == [] and d["min_distance_atr"] == 1.0


def test_alternative_entry_skips_levels_closer_than_one_atr(settings):
    # Decisión del usuario: el soporte de 59.950 está a 0,5 ATR del precio (casi lo
    # mismo que entrar a mercado): se saltea, como para el stop.
    assert settings.trade_plan.alt_entry_min_distance_atr == 1.0
    plan = plan_for(settings, levels_15m=(59950.0, 59500.0, 60600.0))
    assert plan.stop.price == 59450.0
    alt = plan.alternative
    assert alt.level_price == 59500.0
    assert [(sk.price, sk.timeframe) for sk in alt.skipped] == [(59950.0, "15m")]
    assert alt.skipped[0].distance_atr == pytest.approx(0.5)
    assert alt.plan.entry_price == 59500.0
    assert alt.to_dict()["skipped_levels"] == [
        {"price": 59950.0, "timeframe": "15m", "distance_atr": pytest.approx(0.5)}
    ]
    # Con el mínimo en 0 se vuelve a usar el nivel pegado.
    s = with_trade_plan(settings, alt_entry_min_distance_atr=0.0)
    loose = build_trade_plan(make_inputs(s, levels_15m=(59950.0, 59500.0, 60600.0)), s, now=NOW)
    assert loose.alternative.level_price == 59950.0 and loose.alternative.skipped == ()


def test_alternative_entry_short_skips_levels_closer_than_one_atr(settings):
    plan = plan_for(settings, score=-45.0, levels_15m=(59500.0, 60080.0, 60600.0))
    alt = plan.alternative
    assert alt.level_kind == "resistencia" and alt.level_price == 60600.0
    assert [sk.price for sk in alt.skipped] == [60080.0]


def test_alternative_entry_when_every_level_is_too_close(settings):
    plan = plan_for(settings, levels_15m=(59950.0, 59920.0, 60600.0))
    alt = plan.alternative
    assert alt.available is False and alt.plan is None and alt.level_price is None
    assert [sk.price for sk in alt.skipped] == [59950.0, 59920.0]
    assert alt.reason == (
        "No hay orden límite: todos los soportes de 15m debajo del precio están a menos de "
        "1 ATR (tan cerca que sería casi lo mismo que entrar a mercado)."
    )
    short = plan_for(settings, score=-45.0, levels_15m=(59500.0, 60050.0))
    assert short.alternative.reason.startswith(
        "No hay orden límite: todas las resistencias de 15m arriba del precio"
    )


def test_alternative_entry_without_atr(settings):
    plan = plan_for(settings, atr=None)
    alt = plan.alternative
    assert alt.available is False
    assert alt.reason == (
        "No hay ATR de 15m: no se puede saber qué soportes están demasiado cerca del precio "
        "para una orden límite."
    )


def test_alternative_entry_short_at_nearest_15m_resistance(settings):
    plan = plan_for(settings, score=-45.0, levels_15m=(59500.0, 60250.0, 60600.0))
    alt = plan.alternative
    assert alt.level_kind == "resistencia" and alt.level_price == 60250.0
    assert alt.distance_pct > 0
    assert alt.plan.stop.price == 60650.0  # la resistencia de 60.600 está a 3,5 ATR


def test_alternative_entry_without_level(settings):
    plan = plan_for(settings, levels_15m=(60600.0,))
    alt = plan.alternative
    assert alt.available is False and alt.plan is None
    assert alt.reason == "No hay soportes de 15m debajo del precio para una orden límite."
    assert alt.to_dict()["available"] is False


def test_headline_points_to_the_alternative_when_only_it_passes(settings):
    # Entrada a mercado con el stop lejos (debajo del soporte de 59.000): el TP2 no
    # entra en el día. Con la orden límite en ese soporte el stop queda debajo del
    # de 58.800, más cerca, y sí entra.
    kwargs = _alternative_only_kwargs(settings)
    plan = plan_for(settings, **kwargs)
    assert plan.stop.price == 58950.0
    assert plan.status == STATUS_NO_TRADE and failing_codes(plan) == [TP2_DAILY_RANGE]
    assert plan.alternative.level_price == 59000.0
    assert plan.alternative.plan.stop.price == 58750.0
    assert plan.alternative.plan.status == STATUS_LONG
    assert plan.headline.endswith(
        "Con una orden límite en 59.000 el plan sí cumple las reglas (mirá la entrada alternativa)."
    )


# --- reglas, ajustes y avisos --------------------------------------------------------------


def test_all_failing_rules_are_collected(settings):
    plan = plan_for(
        settings,
        regime=fake_regime("crash"),
        trend_4h="bajista",
        trend_1h="alcista",
        trades_today=1,
        pnl_today_usd=-150.0,
        high_volatility_event=True,
        event_description="CPI",
        daily_pct=1.0,
    )
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing_codes(plan) == [
        REGIME_RULE,
        TF_CONFLICT,
        DAILY_LIMITS,
        HIGH_VOLATILITY_EVENT,
        TP2_DAILY_RANGE,
    ]
    assert len(plan.blocking_reasons) == 5
    assert all(r.suggestion for r in plan.blocking_rules)
    assert plan.headline == (
        "No operar: con el régimen Crash (caída fuerte) los longs están bloqueados "
        "(y 4 motivos más)."
    )
    # Los niveles se calculan igual, como referencia.
    assert plan.stop is not None and len(plan.take_profits) == 3


def test_regime_multipliers_reduce_the_risk_with_reasons(settings):
    plan = plan_for(settings, regime=fake_regime("alcista", confidence=0.4))
    assert plan.status == STATUS_LONG
    assert plan.adjusted_risk_usd == pytest.approx(25.0) and plan.risk_multiplier == 0.5
    assert plan.position.risk_usd == pytest.approx(25.0)
    regime_adj = [a for a in plan.adjustments if a.source == "regime"]
    assert [a.code for a in regime_adj] == ["low_confidence"]
    assert regime_adj[0].message == (
        "Riesgo × 0,5: confianza del régimen baja (40 %, menos de 60 %)."
    )


def test_euforia_long_without_funding_halves_the_risk(settings):
    # Decisión del usuario: sin dato de funding se toma como muy alto (× 0,5).
    plan = plan_for(settings, regime=fake_regime("euforia"), funding_pct_8h=None)
    assert plan.risk_multiplier == pytest.approx(0.5)
    assert plan.adjusted_risk_usd == pytest.approx(25.0)
    adj = next(a for a in plan.adjustments if a.source == SOURCE_REGIME and a.risk_multiplier < 1)
    assert "no hay dato de funding, así que se toma como muy alto" in adj.message
    # Con el funding normal no se toca.
    normal = plan_for(settings, regime=fake_regime("euforia"), funding_pct_8h=0.01)
    assert normal.risk_multiplier == 1.0 and normal.adjusted_risk_usd == 50.0


def test_regime_min_score_and_min_rr_are_applied(settings):
    # Neutral: score mínimo 45, R:R mínimo 2 y riesgo × 0,75.
    plan = plan_for(settings, regime=fake_regime("neutral"), score=44.0)
    assert failing_codes(plan) == ["min_abs_score"]
    assert plan.min_rr == 2.0 and plan.min_rr_reason == "el régimen Neutral"
    assert plan.take_profit("TP2").net_rr == pytest.approx(2.0)
    ok = plan_for(settings, regime=fake_regime("neutral"), score=45.0)
    assert ok.status == STATUS_LONG and ok.adjusted_risk_usd == pytest.approx(37.5)


def test_regime_unavailable_is_taken_as_low_confidence(settings):
    inputs = dataclasses.replace(make_inputs(settings), regime=None)
    plan = build_trade_plan(inputs, settings, now=NOW)
    assert plan.adjusted_risk_usd == pytest.approx(25.0)
    assert plan.regime_summary == {"available": False}
    rule = next(r for r in plan.rules if r.code == REGIME_RULE)
    assert rule.ok and rule.message.startswith("Régimen no disponible")


def test_position_adjustments_are_listed_after_regime_ones(settings):
    plan = plan_for(
        settings, regime=fake_regime("alcista", unstable=True), levels_15m=(59700.0, 60600.0)
    )
    sources = [a.source for a in plan.adjustments]
    assert sources == sorted(sources, key=lambda s: s != "regime")
    assert "position" in sources and "regime" in sources


def test_price_source_warning(settings):
    binance = plan_for(settings, price_source="binance")
    assert any("precio de Binance, no con el de Nexo" in w for w in binance.warnings)
    nexo = plan_for(settings, price_source="manual_nexo")
    assert not any("Nexo" in w for w in nexo.warnings)


def test_profile_red_warning_is_carried(settings):
    profile = get_profile("agresivo", settings, max_leverage_override=30)
    plan = plan_for(settings, profile=profile)
    assert plan.warnings[0] == profile.red_warning_text
    assert plan.to_dict()["profile"]["max_leverage"] == 30


def test_invalid_inputs_raise(settings):
    with pytest.raises(ValueError):
        plan_for(settings, entry=0.0)
    with pytest.raises(ValueError, match="margen disponible"):
        plan_for(settings, available_margin_usd=0.0)
    with pytest.raises(ValueError, match="alta volatilidad"):
        plan_for(settings, high_volatility_event="sí")


def test_short_with_absurd_stop_does_not_crash(settings):
    # Stop 80 % arriba: el TP2 del R:R mínimo daría un precio negativo.
    plan = plan_for(
        settings,
        score=-45.0,
        entry=100.0,
        atr=1.0,
        levels_15m=(180.0,),
        levels_1h=(180.0,),
        daily_pct=5.0,
    )
    assert plan.status == STATUS_NO_TRADE
    assert {MIN_RR, TP2_DAILY_RANGE} <= set(failing_codes(plan))
    assert plan.take_profit("TP2") is None and plan.take_profit("TP3") is None
    assert plan.net_rr_tp2 is None
    json.dumps(plan.to_dict(), allow_nan=False)


# --- JSON y determinismo ---------------------------------------------------------------


def test_to_dict_is_json_safe(settings):
    for plan in (
        plan_for(settings),
        plan_for(settings, score=5.0),
        plan_for(settings, daily_pct=None, funding_pct_8h=None, levels_1h=()),
        plan_for(settings, regime=fake_regime("crash")),
    ):
        d = plan.to_dict()
        text = json.dumps(d, allow_nan=False, ensure_ascii=False)
        assert json.loads(text) == d
    d = plan_for(settings).to_dict()
    for key in (
        "status",
        "direction",
        "headline",
        "is_reference",
        "entry",
        "stop",
        "take_profits",
        "position",
        "liquidation",
        "risk",
        "fees",
        "funding",
        "time_stop",
        "adjustments",
        "rules",
        "blocking_reasons",
        "warnings",
        "alternative",
        "score",
        "regime",
        "profile",
    ):
        assert key in d, key
    assert d["created_at"] == "2026-10-12T14:37:20+00:00"
    assert d["time_stop"]["deadline"] == "2026-10-13T14:37:20+00:00"
    assert d["stop"]["distance_pct"] == pytest.approx(550 / 60000 * 100)  # % humano
    assert d["liquidation"]["distance_pct"] > d["liquidation"]["stop_distance_pct"] * 2 - 1e-9
    assert d["risk"]["max_loss_usd"] == pytest.approx(50.0)


def test_plan_is_deterministic(settings):
    a = plan_for(settings).to_dict()
    b = plan_for(settings).to_dict()
    assert a == b


# --- desde el análisis --------------------------------------------------------------


@pytest.fixture(scope="module")
def offline_market(settings):
    snap = asyncio.run(build_offline_snapshot("SOL", settings, None, now=NOW))
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    score = compute_score(inputs_from_analysis(analysis, snap, settings), settings)
    return snap, analysis, score


def test_plan_inputs_from_analysis(settings, offline_market):
    snap, analysis, score = offline_market
    profile = get_profile(None, settings)
    regime = fake_regime("bajista")
    inputs = plan_inputs_from_analysis(
        analysis, snap, score, regime, profile, settings, trades_today=1, pnl_today_usd=-5.0
    )
    assert inputs.symbol == "SOL"
    assert inputs.entry_price == snap.current_price and inputs.price_source == "binance"
    assert set(inputs.levels) == {"15m", "1h"}
    assert inputs.levels["1h"] is analysis.timeframes["1h"].summary.levels
    assert inputs.entry_atr == analysis.timeframes["15m"].summary.atr
    assert inputs.trend_4h == analysis.timeframes["4h"].summary.trend
    assert inputs.trend_1h == analysis.timeframes["1h"].summary.trend
    assert inputs.daily_range_pct == analysis.daily_range_pct
    assert inputs.daily_range_usd == pytest.approx(
        analysis.daily_range_pct / 100 * snap.current_price
    )
    rate, interval = snap.funding.rate, snap.funding.interval_hours
    assert inputs.funding_pct_8h == funding_pct_per_8h(rate, interval)
    assert inputs.funding_interval_hours == interval
    assert inputs.score is score and inputs.regime is regime and inputs.profile is profile
    assert (inputs.trades_today, inputs.pnl_today_usd) == (1, -5.0)
    plan = build_trade_plan(inputs, settings, now=snap.created_at)
    json.dumps(plan.to_dict(), allow_nan=False)
    assert plan.direction == score.direction_hint


def test_plan_inputs_with_manual_nexo_price(settings):
    snap = asyncio.run(build_offline_snapshot("SOL", settings, None, now=NOW))
    manual = round(snap.current_price * 1.002, 4)
    snap = asyncio.run(build_offline_snapshot("SOL", settings, manual, now=NOW))
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    score = compute_score(inputs_from_analysis(analysis, snap, settings), settings)
    inputs = plan_inputs_from_analysis(
        analysis, snap, score, None, get_profile(None, settings), settings
    )
    assert inputs.entry_price == manual and inputs.price_source == "manual_nexo"
    plan = build_trade_plan(inputs, settings, now=NOW)
    assert not any("no con el de Nexo" in w for w in plan.warnings)


# --- revisión final del Hito 4 --------------------------------------------------------


def test_regime_risk_floor_is_listed_even_without_a_stop(settings):
    # Neutral (× 0,75) + confianza baja (× 0,5) + inestable (× 0,5): 9,38 USD < 15.
    regime = fake_regime("neutral", confidence=0.3, unstable=True)
    for kwargs in ({"atr": None}, {"levels_15m": (), "levels_1h": ()}, {}):
        plan = plan_for(settings, score=60.0, regime=regime, **kwargs)
        adj = plan.regime_adjustment
        assert adj.blocked and adj.risk_below_min
        assert "min_operable_risk" in failing_codes(plan)
        rule = next(r for r in plan.blocking_rules if r.code == "min_operable_risk")
        assert rule.message == adj.block_reasons[-1]
        assert rule.message.startswith("No operar: con los ajustes el riesgo queda en 9,38 USD")
        assert rule.suggestion
    assert failing_codes(plan_for(settings, score=60.0, regime=regime, atr=None)) == [
        "stop",
        "min_operable_risk",
    ]


def test_regime_risk_floor_is_not_bypassed_with_fees_outside_the_risk(settings):
    # Comisiones aparte: 12,50 USD de riesgo (50 × 0,5 × 0,5) + comisiones pasaban
    # los 15 USD y el plan daba SHORT aunque el régimen decía "No operar".
    custom = with_trade_plan(settings, fees_included_in_risk=False)
    plan = plan_for(
        custom,
        score=-60.0,
        regime=fake_regime("crash", 0.9, True),
        levels_15m=(59000.0, 60120.0),
        levels_1h=(58000.0, 61500.0),
        daily_pct=5.0,
    )
    assert plan.regime_adjustment.blocked
    assert plan.position.risk_usd > 15.0  # con las comisiones, la pérdida total pasa
    assert plan.status == STATUS_NO_TRADE
    assert failing_codes(plan) == ["min_operable_risk"]
    assert plan.blocking_reasons[0].startswith(
        "No operar: con los ajustes el riesgo queda en 12,50 USD"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score": -60.0, "levels_15m": (0.0, 60600.0), "levels_1h": (61500.0,)},
        {"score": -60.0, "levels_15m": (-5.0, 60600.0), "levels_1h": (61500.0,)},
        {"score": 60.0, "levels_15m": (float("nan"), 59500.0, 60600.0)},
        {"score": 60.0, "levels_15m": (float("inf"), 59500.0, 60600.0)},
        # La única resistencia de 15m "arriba" es infinita: no hay entrada alternativa.
        {"score": -60.0, "levels_15m": (59500.0, float("inf")), "levels_1h": (61500.0, 58000.0)},
    ],
)
def test_levels_with_invalid_prices_are_ignored(settings, kwargs):
    # Armados a mano (backtest): antes levantaban ValueError dentro del plan.
    plan = plan_for(settings, **kwargs)
    json.dumps(plan.to_dict(), allow_nan=False)
    assert plan.stop is not None and math.isfinite(plan.stop.price) and plan.stop.price > 0
    assert all(math.isfinite(tp.price) and tp.price > 0 for tp in plan.take_profits)
    if kwargs["score"] > 0:
        assert plan.stop.level_price == 59500.0
    alt = plan.alternative
    assert alt.level_price is None or (math.isfinite(alt.level_price) and alt.level_price > 0)


def test_impossible_short_target_says_why_and_what_to_do(settings):
    # Short con la única resistencia a casi el doble de la entrada: el TP2 caería
    # en un precio negativo. No hay TP1 (nada debajo), ni TP2, ni stop por tiempo.
    plan = plan_for(
        settings,
        score=-45.0,
        entry=100.0,
        atr=1.0,
        levels_15m=(180.0,),
        levels_1h=(180.0,),
        daily_pct=5.0,
    )
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert plan.take_profit("TP2") is None and plan.take_profit("TP1") is None
    assert "el precio tendría que bajar a cero o menos" in plan.tp2_missing_reason
    assert plan.to_dict()["tp2_missing_reason"] == plan.tp2_missing_reason
    rules = {r.code: r for r in plan.blocking_rules}
    assert rules[MIN_RR].message.startswith("No operar: no hay TP2: el stop está tan lejos")
    assert rules[MIN_RR].suggestion and rules[TP2_DAILY_RANGE].suggestion
    assert "Esperá una entrada con el stop más cerca" in rules[MIN_RR].suggestion
    # Sin objetivo no hay stop por tiempo que mire "el TP2" (antes lo nombraba igual).
    assert plan.time_stop is None


def test_time_stop_uses_tp1_when_tp2_is_missing(settings):
    plan = plan_for(
        settings, score=-60.0, entry=1.0, atr=0.01, levels_15m=(0.9, 2.0), levels_1h=(2.0,)
    )
    assert plan.take_profit("TP2") is None and plan.take_profit("TP1") is not None
    assert plan.time_stop is not None and plan.time_stop.target_name == "TP1"


def _alternative_only_kwargs(settings) -> dict:
    """Plan a mercado bloqueado por el rango del día y alternativa que sí cumple.

    Un rango del día enorme deja pasar a los dos; después se pone justo entre la
    distancia al TP2 de cada uno.
    """
    kwargs = {"levels_15m": (59000.0, 58800.0, 60600.0), "levels_1h": (58500.0, 61500.0)}
    free = plan_for(settings, daily_pct=50.0, **kwargs)
    assert free.status == STATUS_LONG and free.alternative.plan.status == STATUS_LONG
    d_main = free.take_profit("TP2").distance_pct / 100 * ENTRY
    d_alt = free.alternative.plan.take_profit("TP2").distance_pct / 100 * 59000.0
    assert d_alt < d_main
    return {**kwargs, "daily_pct": None, "daily_usd": (d_main + d_alt) / 2}


@pytest.mark.parametrize("score", [45.0, -45.0])
def test_suspect_nexo_price_blocks_until_corrected(settings, score):
    # Decisión del usuario: precio de Nexo mal tipeado (10 veces el real, difiere
    # 900 % del de referencia): "No operar" hasta corregirlo.
    assert settings.trade_plan.block_if_price_suspect is True
    plan = plan_for(settings, score=score, price_diff_pct=900.0, reference_price=6000.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert plan.price_suspect is True and plan.price_diff_pct == 900.0
    assert failing_codes(plan) == [PRICE_CHECK]
    rule = plan.blocking_rules[0]
    assert plan.rules[0] is rule  # es la primera regla: sin el precio bien, nada sirve
    assert rule.message == (
        "No operar: el precio de Nexo que cargaste (60.000,00) difiere 900 % del de "
        "referencia (6.000,00), más que el 2 % que se acepta: puede ser un error de tipeo."
    )
    assert rule.suggestion == (
        "Revisá el precio en Nexo y volvé a cargarlo bien: con un precio mal cargado, todo el "
        "plan (stop, objetivos y tamaño) sale mal. Si de verdad es ese, esperá a que los dos "
        "precios se acerquen."
    )
    assert plan.headline == rule.message  # es el único motivo
    # La regla ya lo dice: no se repite como aviso.
    assert not any(w.startswith("Revisá el precio") for w in plan.warnings)
    entry = plan.to_dict()["entry"]
    assert entry["price_suspect"] is True and entry["diff_vs_reference_pct"] == 900.0
    assert entry["reference_price"] == 6000.0


def test_suspect_price_also_blocks_the_alternative(settings):
    kwargs = _alternative_only_kwargs(settings)
    plan = plan_for(settings, price_diff_pct=900.0, **kwargs)
    alt_rule = next(r for r in plan.alternative.plan.blocking_rules if r.code == PRICE_CHECK)
    # Habla del precio de Nexo (el actual), no del nivel de la orden límite (59.000).
    assert alt_rule.message == plan.blocking_rules[0].message
    assert plan.alternative.plan.reference_price == pytest.approx(6000.0)
    assert "sí cumple las reglas" not in plan.headline


def test_suspect_price_reference_is_deduced_from_the_difference(settings):
    # Sin el precio de referencia (backtest armado a mano) se deduce de la diferencia.
    plan = plan_for(settings, price_diff_pct=-50.0)
    assert plan.reference_price == pytest.approx(120000.0)
    assert "difiere 50 % del de referencia (120.000,00)" in plan.blocking_rules[0].message


def test_nexo_price_from_the_connector_is_not_said_to_be_typed(settings):
    plan = plan_for(settings, price_diff_pct=5.0, reference_price=57142.86, price_source="nexo")
    assert plan.blocking_rules[0].message.startswith(
        "No operar: el precio de Nexo (60.000,00) difiere 5 % del de referencia (57.142,86)"
    )


def test_nexo_price_within_tolerance_passes_the_price_rule(settings):
    kwargs = _alternative_only_kwargs(settings)
    ok = plan_for(settings, price_diff_pct=-1.5, reference_price=60913.71, **kwargs)
    assert ok.price_suspect is False
    rule = next(r for r in ok.rules if r.code == PRICE_CHECK)
    assert rule.ok and rule.message == (
        "El precio de Nexo (60.000,00) difiere 1,5 % del de referencia (60.913,71): dentro "
        "del 2 % que se acepta."
    )
    assert not any(w.startswith("Revisá el precio") for w in ok.warnings)
    assert ok.headline.endswith("sí cumple las reglas (mirá la entrada alternativa).")
    # Sin precio de Nexo comparado no hay regla del precio.
    assert PRICE_CHECK not in [r.code for r in plan_for(settings).rules]


@pytest.mark.parametrize(("reference", "nexo"), [(3.0, 3.06), (3.0, 2.94)])
def test_nexo_price_exactly_at_the_max_diff_is_accepted(settings, reference, nexo):
    # Binance 3,00 y Nexo 3,06: justo 2 %, pero la cuenta da 2,0000000000000018 %
    # por coma flotante. La decisión bloquea solo "más que" el máximo.
    assert settings.exchanges.nexo.manual_price_max_diff_pct == 2.0
    diff = (nexo - reference) / reference * 100  # como registry._diff_pct
    assert abs(diff) > 2.0  # el ruido está
    kwargs = {"entry": nexo, "levels_15m": (2.85, 3.2), "levels_1h": (2.8, 3.3), "atr": 0.005}
    plan = plan_for(settings, price_diff_pct=diff, reference_price=reference, **kwargs)
    assert plan.price_suspect is False
    rule = next(r for r in plan.rules if r.code == PRICE_CHECK)
    assert rule.ok and rule.message == (
        f"El precio de Nexo ({fmt_num(nexo, 4, trim=False)}) difiere 2 % del de referencia "
        "(3,0000): dentro del 2 % que se acepta."
    )
    # Con el bloqueo apagado tampoco se avisa.
    s = with_trade_plan(settings, block_if_price_suspect=False)
    warn_only = build_trade_plan(
        make_inputs(s, price_diff_pct=diff, reference_price=reference, **kwargs), s, now=NOW
    )
    assert not any(w.startswith("Revisá el precio") for w in warn_only.warnings)
    # Apenas más que el máximo sí es sospechoso.
    above = plan_for(settings, price_diff_pct=2.0001, **kwargs)
    assert above.price_suspect is True and PRICE_CHECK in failing_codes(above)


def test_blocking_price_message_never_rounds_the_difference_to_the_limit(settings):
    # 2,004 % redondeado a dos decimales es "2 %": el mensaje decía "difiere 2 %
    # del de referencia, más que el 2 % que se acepta".
    plan = plan_for(settings, price_diff_pct=2.004)
    rule = next(r for r in plan.rules if r.code == PRICE_CHECK)
    assert not rule.ok
    assert rule.message == (
        "No operar: el precio de Nexo que cargaste (60.000,00) difiere 2,004 % del de "
        "referencia (58.821,22), más que el 2 % que se acepta: puede ser un error de tipeo."
    )
    # Lo mismo en el aviso, con el bloqueo apagado.
    s = with_trade_plan(settings, block_if_price_suspect=False)
    warn_only = build_trade_plan(make_inputs(s, price_diff_pct=-2.004), s, now=NOW)
    warning = next(w for w in warn_only.warnings if w.startswith("Revisá el precio"))
    assert "60.000,00 difiere 2,004 % del precio de referencia (61.226,99), más que el 2 %" in (
        warning
    )


def test_price_rule_prices_have_the_same_decimals_as_the_prices_section(settings):
    # La consola muestra "186,60" en la sección de precios: la regla no dice "186,6".
    plan = plan_for(
        settings,
        entry=186.6,
        levels_15m=(180.0, 195.0),
        levels_1h=(175.0, 200.0),
        atr=0.5,
        price_diff_pct=5.0,
        reference_price=177.7,
    )
    rule = next(r for r in plan.rules if r.code == PRICE_CHECK)
    assert "el precio de Nexo que cargaste (186,60) difiere 5 % del de referencia (177,70)" in (
        rule.message
    )


def test_suspect_price_only_warns_if_the_config_says_so(settings):
    s = with_trade_plan(settings, block_if_price_suspect=False)
    kwargs = _alternative_only_kwargs(settings)
    plan = build_trade_plan(make_inputs(s, price_diff_pct=900.0, **kwargs), s, now=NOW)
    assert plan.price_suspect is True
    assert PRICE_CHECK not in [r.code for r in plan.rules]
    assert failing_codes(plan) == [TP2_DAILY_RANGE]
    assert plan.alternative.plan.status == STATUS_LONG  # la alternativa "pasaría"...
    assert "sí cumple las reglas" not in plan.headline  # ...pero no se recomienda
    warning = next(w for w in plan.warnings if w.startswith("Revisá el precio de entrada"))
    assert "difiere 900 % del precio de referencia (6.000,00)" in warning
    assert "no lo uses hasta cargarlo bien" in warning


@pytest.mark.parametrize("score", [45.0, -45.0])
def test_risk_is_capped_to_what_is_left_of_the_daily_loss_limit(settings, score):
    # Decisión del usuario: perdiste 80 de 100 USD, quedan 20: el riesgo baja de 50 a 20.
    assert settings.trade_plan.cap_risk_to_daily_remaining is True
    plan = plan_for(settings, score=score, trades_today=1, pnl_today_usd=-80.0)
    assert plan.status in (STATUS_LONG, STATUS_SHORT)
    assert plan.requested_risk_usd == 50.0 and plan.regime_risk_usd == 50.0
    assert plan.adjusted_risk_usd == pytest.approx(20.0)
    assert plan.risk_capped_by_day is True and plan.daily_remaining_usd == pytest.approx(20.0)
    assert plan.position.risk_usd == pytest.approx(20.0)
    cap = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert cap.source == SOURCE_DAILY_LIMIT
    assert cap.risk_multiplier == pytest.approx(0.4)
    assert cap.message == (
        "Riesgo bajado de 50 a 20 USD: hoy ya perdiste 80 de tus 100 USD de límite del día "
        "(así, si esta operación toca el stop, no pasás el límite)."
    )
    assert "arriesgás 20 USD (comisiones incluidas)" in plan.headline
    assert plan.day_state["loss_if_stopped_usd"] == pytest.approx(100.0)
    risk = plan.to_dict()["risk"]
    assert risk["after_regime_usd"] == 50.0 and risk["after_adjustments_usd"] == pytest.approx(20)
    assert risk["capped_by_daily_limit"] is True
    assert risk["daily_remaining_usd"] == pytest.approx(20.0)
    # El aviso que solo avisaba ya no hace falta.
    assert not any(w.startswith("Ojo con el límite del día") for w in plan.warnings)


def test_daily_cap_comes_after_the_regime_multipliers(settings):
    # Confianza baja: 50 × 0,5 = 25; quedan 20 del día: baja de 25 a 20.
    plan = plan_for(
        settings, regime=fake_regime("alcista", confidence=0.4), pnl_today_usd=-80.0, daily_pct=10
    )
    sources = [a.source for a in plan.adjustments]
    assert sources[:2] == [SOURCE_REGIME, SOURCE_DAILY_LIMIT]
    assert all(src == SOURCE_POSITION for src in sources[2:])
    assert plan.regime_risk_usd == pytest.approx(25.0) and plan.risk_multiplier == 0.5
    assert plan.adjusted_risk_usd == pytest.approx(20.0)
    cap = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert cap.message.startswith("Riesgo bajado de 25 a 20 USD")
    # Si el régimen ya dejó el riesgo debajo de lo que queda, no se toca.
    calm = plan_for(settings, regime=fake_regime("alcista", confidence=0.4), pnl_today_usd=-60.0)
    assert calm.adjusted_risk_usd == pytest.approx(25.0) and not calm.risk_capped_by_day
    assert DAILY_LOSS_CAP not in [a.code for a in calm.adjustments]


def test_daily_cap_is_not_applied_when_there_is_room(settings):
    # 40 + 50 = 90: entra en el límite, el riesgo no cambia.
    plan = plan_for(settings, trades_today=1, pnl_today_usd=-40.0)
    assert plan.adjusted_risk_usd == 50.0 and plan.risk_capped_by_day is False
    assert plan.daily_remaining_usd == pytest.approx(60.0)
    assert DAILY_LOSS_CAP not in [a.code for a in plan.adjustments]
    # Un día ganador no agranda el riesgo.
    win = plan_for(settings, pnl_today_usd=+300.0)
    assert win.adjusted_risk_usd == 50.0 and win.daily_remaining_usd == 100.0


@pytest.mark.parametrize("score", [45.0, -45.0])
def test_daily_cap_below_min_operable_risk_blocks(settings, score):
    # Perdiste 99 de 100: queda 1 USD, menos que el mínimo operable (15).
    plan = plan_for(settings, score=score, trades_today=1, pnl_today_usd=-99.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing_codes(plan) == [MIN_OPERABLE_RISK]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: hoy ya perdiste 99 de tus 100 USD de límite del día; lo que te queda (1 USD) "
        "es menos que el mínimo operable de 15 USD."
    )
    assert rule.suggestion == "Esperá a mañana."
    assert plan.headline == (
        "No operar: hoy ya perdiste 99 de tus 100 USD de límite del día; lo que te queda (1 USD) "
        "es menos que el mínimo operable de 15 USD."
    )
    assert plan.adjusted_risk_usd == pytest.approx(1.0)
    assert plan.position.risk_usd == pytest.approx(1.0)  # los niveles, como referencia


def test_daily_cap_floor_is_listed_even_without_a_stop(settings):
    plan = plan_for(settings, pnl_today_usd=-99.0, levels_15m=(60600.0,), levels_1h=(61500.0,))
    assert plan.stop is None
    assert failing_codes(plan) == [STOP_RULE, MIN_OPERABLE_RISK]
    assert plan.blocking_rules[1].message.startswith("No operar: hoy ya perdiste 99")


def test_daily_limit_already_reached_is_blocked_by_the_daily_rule(settings):
    # Ya en el límite: lo dice la regla de límites del día; el riesgo no se toca.
    hit = plan_for(settings, pnl_today_usd=-100.0)
    assert failing_codes(hit) == [DAILY_LIMITS]
    assert hit.adjusted_risk_usd == 50.0 and hit.risk_capped_by_day is False
    assert hit.daily_remaining_usd == 0.0
    assert not any(w.startswith("Ojo con el límite del día") for w in hit.warnings)


def test_daily_cap_floor_wins_over_the_regime_floor(settings):
    # Neutral (× 0,75) + confianza baja (× 0,5) + inestable (× 0,5): 9,38 USD, menos
    # que el mínimo (15). Y hoy ya perdiste 95: quedan 5. El motivo que manda es el
    # del día (aunque el régimen se aclare, quedan 5 USD): "Esperá a mañana".
    regime = fake_regime("neutral", confidence=0.3, unstable=True)
    plan = plan_for(settings, score=50.0, regime=regime, pnl_today_usd=-95.0)
    assert plan.regime_adjustment.risk_below_min
    assert plan.regime_risk_usd == pytest.approx(9.375)
    assert plan.adjusted_risk_usd == pytest.approx(5.0) and plan.risk_capped_by_day
    assert failing_codes(plan) == [MIN_OPERABLE_RISK]
    rule = plan.blocking_rules[0]
    message = (
        "No operar: hoy ya perdiste 95 de tus 100 USD de límite del día; lo que te queda (5 USD) "
        "es menos que el mínimo operable de 15 USD."
    )
    assert rule.message == message and rule.suggestion == "Esperá a mañana."
    assert plan.headline == message
    # Los multiplicadores del régimen se siguen viendo en los ajustes.
    cap = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert cap.message.startswith("Riesgo bajado de 9,38 a 5 USD")
    assert SOURCE_REGIME in [a.source for a in plan.adjustments]
    # Sin stop, igual.
    no_stop = plan_for(settings, score=50.0, regime=regime, pnl_today_usd=-95.0, atr=None)
    assert failing_codes(no_stop) == [STOP_RULE, MIN_OPERABLE_RISK]
    assert no_stop.blocking_rules[1].message == message


def test_reference_plan_is_capped_when_only_the_trade_count_blocks(settings):
    # Ya hiciste 3 de 3 operaciones y perdiste 95 de 100: lo bloquea la cantidad
    # de operaciones, pero el plan de referencia no arriesga más de los 5 USD que
    # quedan del día (antes mostraba "Pérdida máxima 50 USD").
    assert get_profile("agresivo", settings).max_trades_per_day == 3
    plan = plan_for(settings, trades_today=3, pnl_today_usd=-95.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing_codes(plan) == [DAILY_LIMITS, MIN_OPERABLE_RISK]
    assert plan.headline.startswith("No operar: ya hiciste 3 operaciones hoy y el máximo es 3")
    assert plan.adjusted_risk_usd == pytest.approx(5.0) and plan.risk_capped_by_day is True
    assert plan.position.risk_usd == pytest.approx(5.0)
    assert plan.blocking_rules[1].message.startswith("No operar: hoy ya perdiste 95")
    # Con lugar para el mínimo, solo la cantidad de operaciones bloquea (riesgo 20).
    room = plan_for(settings, trades_today=3, pnl_today_usd=-80.0)
    assert failing_codes(room) == [DAILY_LIMITS]
    assert room.adjusted_risk_usd == pytest.approx(20.0) and room.risk_capped_by_day
    # Con el tope apagado, vuelve el aviso (también si bloquea la cantidad).
    s = with_trade_plan(settings, cap_risk_to_daily_remaining=False)
    off = build_trade_plan(make_inputs(s, trades_today=3, pnl_today_usd=-95.0), s, now=NOW)
    assert off.adjusted_risk_usd == 50.0 and not off.risk_capped_by_day
    assert any(w.startswith("Ojo con el límite del día") for w in off.warnings)
    # Con el límite de pérdida alcanzado, el riesgo no se toca (lo dice la regla).
    hit = plan_for(settings, trades_today=3, pnl_today_usd=-100.0)
    assert failing_codes(hit) == [DAILY_LIMITS]
    assert hit.adjusted_risk_usd == 50.0 and hit.risk_capped_by_day is False


def test_daily_cap_with_fees_outside_the_risk_says_so(settings):
    s = with_trade_plan(settings, fees_included_in_risk=False)
    plan = build_trade_plan(make_inputs(s, pnl_today_usd=-80.0), s, now=NOW)
    cap = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert cap.message.endswith(
        "(así, si esta operación toca el stop, solo las comisiones quedan por encima del límite)."
    )
    assert plan.position.price_loss_at_stop_usd == pytest.approx(20.0)


def test_daily_loss_warning_only_without_the_cap(settings):
    # Con el tope apagado vuelve el aviso: perdiste 99 de 100 y esta operación
    # arriesga 50, así que si toca el stop el día cerraría con 149.
    s = with_trade_plan(settings, cap_risk_to_daily_remaining=False)
    plan = build_trade_plan(make_inputs(s, trades_today=1, pnl_today_usd=-99.0), s, now=NOW)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == 50.0
    assert any(
        w
        == (
            "Ojo con el límite del día: hoy ya perdiste 99 USD; si esta operación toca el "
            "stop, el día cerraría con 149 USD de pérdida, más que tu límite de 100 USD."
        )
        for w in plan.warnings
    )
    assert plan.day_state["loss_if_stopped_usd"] == pytest.approx(149.0)


def test_day_state_is_returned_as_data(settings):
    plan = plan_for(
        settings,
        trades_today=2,
        pnl_today_usd=-40.0,
        high_volatility_event=True,
        event_description="CPI",
        available_margin_usd=150.0,
    )
    d = plan.to_dict()["day_state"]
    assert d == {
        "trades_today": 2,
        "max_trades_per_day": 3,
        "pnl_today_usd": -40.0,
        "loss_today_usd": 40.0,
        "max_daily_loss_usd": 100.0,
        "daily_remaining_usd": 60.0,
        "risk_capped_by_day": False,
        "loss_if_stopped_usd": pytest.approx(40.0 + plan.position.risk_usd),
        "high_volatility_event": True,
        "event_description": "CPI",
        "available_margin_usd": 150.0,
        "wallet_equity_usd": None,
    }
    json.dumps(d, allow_nan=False)


def test_alternative_dict_has_its_adjustments_liquidation_fees_and_time_stop(settings):
    plan = plan_for(settings, levels_15m=(59950.0, 59500.0, 60600.0))
    ap = plan.alternative.plan
    d = plan.alternative.to_dict()
    assert d["adjustments"] == [a.to_dict() for a in ap.adjustments]
    assert d["liquidation"] == {
        "model": "isolated",
        "price": ap.position.liquidation_price,
        "distance_pct": ap.position.liquidation_distance_pct * 100,
        "ratio": ap.position.liquidation_ratio,
        "min_ratio": ap.position.liquidation_min_ratio,
        "ok": ap.position.liquidation_ok,
    }
    assert d["fees_round_trip_at_stop_usd"] == pytest.approx(ap.fees.round_trip_at_stop_usd)
    assert d["fees_round_trip_at_tp2_usd"] == pytest.approx(ap.fees.round_trip_at_target_usd)
    assert d["time_stop"] == ap.time_stop.to_dict()
    json.dumps(d, allow_nan=False)


def test_headline_prices_use_the_entry_decimals(settings):
    # Entrada 10,2 (3 decimales, como la tabla de la consola): el stop 9,9537 va
    # como "9,954" y no "9,9537" (antes cada precio usaba su propio tamaño).
    plan = plan_for(
        settings,
        entry=10.2,
        atr=0.05,
        levels_15m=(9.9787, 10.6),
        levels_1h=(9.5, 11.0),
        daily_pct=5.0,
    )
    assert plan.status == STATUS_LONG
    assert plan.stop.price == pytest.approx(9.9537)
    assert plan.headline.startswith("LONG a mercado en 10,2: stop en 9,954 y objetivo")


def test_json_with_numpy_level_prices_keeps_real_booleans(settings):
    np = pytest.importorskip("numpy")
    inputs = make_inputs(settings)
    levels = {
        tf: dataclasses.replace(
            res,
            clusters=tuple(dataclasses.replace(c, price=np.float64(c.price)) for c in res.clusters),
        )
        for tf, res in inputs.levels.items()
    }
    plan = build_trade_plan(dataclasses.replace(inputs, levels=levels), settings, now=NOW)
    d = plan.to_dict()
    for value in (
        d["liquidation"]["ok"],
        d["position"]["red_warning"],
        d["liquidation"]["at_open"],
    ):
        assert type(value) is bool
    assert all(type(tp["beyond_daily_range"]) is bool for tp in d["take_profits"])


def test_plan_inputs_from_analysis_fallbacks(settings, offline_market):
    snap, analysis, score = offline_market
    profile = get_profile(None, settings)
    # 4h sin las EMAs de la tendencia: no es un "lateral" real, queda None.
    tf4 = analysis.timeframes["4h"]
    no_emas = dataclasses.replace(tf4, summary=dataclasses.replace(tf4.summary, trend_emas=None))
    analysis2 = dataclasses.replace(analysis, timeframes={**analysis.timeframes, "4h": no_emas})
    inputs = plan_inputs_from_analysis(analysis2, snap, score, None, profile, settings)
    assert inputs.trend_4h is None and inputs.trend_1h is not None
    # Sin funding: sin dato y sin intervalo.
    inputs = plan_inputs_from_analysis(
        analysis, dataclasses.replace(snap, funding=None), score, None, profile, settings
    )
    assert inputs.funding_pct_8h is None and inputs.funding_interval_hours is None
    # Intervalo inválido (0): se usa el de la config.
    bad = dataclasses.replace(snap.funding, interval_hours=0)
    inputs = plan_inputs_from_analysis(
        analysis, dataclasses.replace(snap, funding=bad), score, None, profile, settings
    )
    default = float(settings.fees.funding_interval_hours)
    assert inputs.funding_interval_hours == default
    assert inputs.funding_pct_8h == funding_pct_per_8h(snap.funding.rate, default)
    # Sin precio de Nexo no hay diferencia contra la referencia.
    assert inputs.price_diff_pct is None
    assert inputs.reference_price == snap.execution_price.binance_price


def test_plan_inputs_carry_the_manual_price_difference(settings):
    snap = asyncio.run(build_offline_snapshot("SOL", settings, None, now=NOW))
    typo = round(snap.current_price * 10, 4)  # un cero de más
    snap = asyncio.run(build_offline_snapshot("SOL", settings, typo, now=NOW))
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    score = compute_score(inputs_from_analysis(analysis, snap, settings), settings)
    inputs = plan_inputs_from_analysis(
        analysis, snap, score, None, get_profile(None, settings), settings
    )
    assert inputs.price_diff_pct == pytest.approx(snap.execution_price.diff_pct)
    assert inputs.price_diff_pct == pytest.approx(900.0, rel=1e-3)
    assert inputs.reference_price == pytest.approx(snap.execution_price.binance_price)
    plan = build_trade_plan(inputs, settings, now=NOW)
    assert plan.price_suspect and plan.rules[0].code == PRICE_CHECK and not plan.rules[0].ok
    assert plan.status == STATUS_NO_TRADE
    assert fmt_price(snap.execution_price.binance_price) in plan.rules[0].message
