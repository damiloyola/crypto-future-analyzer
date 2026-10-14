"""Hito 5: la capa de seguridad y la correlación dentro del plan (``app.risk.plan``).

El plan sigue siendo puro: recibe el veredicto de los frenos
(``safety.breakers.evaluate_breakers``) y la exposición a posiciones
correlacionadas (``safety.correlation.correlated_exposure``) en ``PlanInputs``.
Los veredictos se arman con los frenos de verdad sobre un estado cargado a mano
(``build_trading_state``), así el texto que se prueba es el que ve el usuario.

Perfil agresivo de ``config.yaml``: riesgo 50 USD, mínimo operable 15, límite
diario 100 (riesgo × 0,5 desde 50), billetera mensual 300 (riesgo × 0,5 con 180
perdidos en la semana), 3 pérdidas seguidas → pausa de 4 h, tope de riesgo
combinado 1,5 × 50 = 75 USD. "Ahora" es el miércoles 14/10/2026 15:00 de
Buenos Aires.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.risk.plan import (
    CORRELATION_CAP,
    CORRELATION_RULE,
    DAILY_LOSS_CAP,
    DAILY_REMAINING_RULE,
    OVERRIDE_PNL_TODAY,
    OVERRIDE_TRADES_TODAY,
    SAFETY_RULE,
    SOURCE_CORRELATION,
    SOURCE_DAILY_LIMIT,
    SOURCE_SAFETY,
    STATUS_LONG,
    STATUS_NO_TRADE,
    STATUS_SHORT,
    CorrelatedPosition,
    CorrelationExposure,
    DayOverride,
    build_trade_plan,
)
from app.risk.profiles import get_profile
from app.risk.rules import DAILY_LIMITS, HIGH_VOLATILITY_EVENT, MIN_OPERABLE_RISK
from app.safety.breakers import evaluate_breakers
from app.safety.correlation import (
    combined_risk_limit,
    correlated_exposure,
    correlation_matrix,
)
from app.safety.lockfile import LockInfo
from app.safety.state import build_trading_state
from tests.test_plan import make_inputs

BA = ZoneInfo("America/Argentina/Buenos_Aires")
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=BA)  # miércoles
NO_LOCK = LockInfo(exists=False, path="LOCKED")

_ids = iter(range(1, 100_000))


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=BA)


def trade(
    closed: datetime, pnl: float, *, symbol: str = "BTC", opened: datetime | None = None
) -> dict:
    return {
        "id": next(_ids),
        "symbol": symbol,
        "side": "long",
        "opened_at": closed - timedelta(minutes=30) if opened is None else opened,
        "closed_at": closed,
        "pnl_usd": pnl,
    }


def wallet_balance(amount: float) -> dict:
    return {"id": next(_ids), "ts": local(1, 9), "kind": "balance", "amount_usd": amount}


@pytest.fixture
def profile(settings):
    return get_profile("agresivo", settings)


def state_and_verdict(settings, profile, *, trades=(), balance=1000.0, vol=(), lock=NO_LOCK):
    state = build_trading_state(
        trades=list(trades),
        open_positions=[],
        wallet_events=[wallet_balance(balance)],
        volatility_events=list(vol),
        settings=settings,
        profile=profile,
        now=NOW,
    )
    return state, evaluate_breakers(state, profile, settings, NOW, lock=lock)


def plan_with(settings, profile, *, score=45.0, state=None, verdict=None, **extra):
    """Plan con el estado del día y el veredicto (como lo arma la consola)."""
    kwargs = dict(extra)
    if state is not None:
        kwargs.setdefault("trades_today", state.trades_today)
        kwargs.setdefault("pnl_today_usd", state.plan_pnl_today_usd)
    inputs = make_inputs(settings, score=score, profile=profile, safety=verdict, **kwargs)
    return build_trade_plan(inputs, settings, now=NOW)


def failing(plan) -> list[str]:
    return [r.code for r in plan.blocking_rules]


def exposure(side: str, *counted, hedges=()) -> dict:
    other = "short" if side == "long" else "long"
    return {side: CorrelationExposure(side, tuple(counted), tuple(hedges)), other: None}


def position(symbol: str, side: str, risk: float, reason: str = "high_correlation"):
    return CorrelatedPosition(symbol, side, risk, 0.9, reason)


# --- sin capa de seguridad no cambia nada ------------------------------------------------


def test_without_safety_the_plan_is_unchanged(settings, profile):
    plan = plan_with(settings, profile)
    assert plan.status == STATUS_LONG
    assert all(not r.code.startswith(SAFETY_RULE) for r in plan.rules)
    assert all(r.code != CORRELATION_RULE for r in plan.rules)
    d = plan.to_dict()
    assert d["safety"] == {"evaluated": False} and d["correlation"] == {"evaluated": False}
    assert d["risk"]["safety_multiplier"] is None and d["day_overrides"] == []


def test_green_verdict_adds_a_passing_rule_at_the_end(settings, profile):
    state, verdict = state_and_verdict(settings, profile)
    assert verdict.status == "verde"
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(50.0)
    last = plan.rules[-1]
    assert last.code == SAFETY_RULE and last.ok
    assert last.message == "Ningún freno activo: el plan usa el riesgo normal de tu perfil."
    assert plan.safety_multiplier == 1.0 and plan.safety_risk_usd == pytest.approx(50.0)
    summary = plan.to_dict()["safety"]
    assert summary["evaluated"] is True and summary["status"] == "verde"
    assert summary["blocked"] is False and summary["risk_multiplier"] == 1.0


# --- multiplicador de riesgo ---------------------------------------------------------------


def test_daily_reduce_halves_the_risk_with_its_reason(settings, profile):
    # Perdiste 60 hoy (50 % de 100): riesgo × 0,5 hasta que termine el día.
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -60.0)])
    assert verdict.status == "ambar" and verdict.risk_multiplier == 0.5
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_LONG
    assert plan.regime_risk_usd == pytest.approx(50.0)
    assert plan.safety_multiplier == 0.5 and plan.safety_risk_usd == pytest.approx(25.0)
    assert plan.adjusted_risk_usd == pytest.approx(25.0)
    assert plan.position.risk_usd == pytest.approx(25.0)
    safety_adj = [a for a in plan.adjustments if a.source == SOURCE_SAFETY]
    assert [a.code for a in safety_adj] == ["daily_loss_reduce"]
    assert safety_adj[0].risk_multiplier == 0.5
    message = safety_adj[0].message
    assert message.startswith("Riesgo bajado de 50 a 25 USD (× 0,5) por un freno de seguridad.")
    assert "Hoy perdiste 60 USD" in message and "hasta que termine el día" in message
    assert plan.rules[-1].code == SAFETY_RULE and plan.rules[-1].ok
    assert plan.rules[-1].message.startswith("Riesgo reducido (× 0,5) por la pérdida del día")


def test_safety_multiplier_goes_before_the_daily_cap(settings, profile):
    # Perdiste 80: × 0,5 (50 → 25) y después el tope del día (te quedan 20).
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -80.0)])
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_LONG
    assert [a.source for a in plan.adjustments if a.source != "position"] == [
        SOURCE_SAFETY,
        SOURCE_DAILY_LIMIT,
    ]
    daily = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert daily.message.startswith("Riesgo bajado de 25 a 20 USD: hoy ya perdiste 80")
    risk = plan.to_dict()["risk"]
    assert risk["after_regime_usd"] == pytest.approx(50.0)
    assert risk["after_safety_usd"] == pytest.approx(25.0)
    assert risk["after_daily_usd"] == pytest.approx(20.0)
    assert risk["after_adjustments_usd"] == pytest.approx(20.0)
    assert plan.position.risk_usd <= 20.0 + 1e-9


def test_day_and_week_reduce_multiply_and_can_fall_below_the_minimum(settings, profile):
    # Lunes -150 y hoy -60: semana -210 (>= 180) y día -60 (>= 50): × 0,25 → 12,5 < 15.
    trades = [trade(local(12, 10), -150.0), trade(local(14, 10), -60.0)]
    state, verdict = state_and_verdict(settings, profile, trades=trades)
    assert verdict.risk_multiplier == pytest.approx(0.25) and not verdict.blocked
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing(plan) == [MIN_OPERABLE_RISK]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: con los frenos de seguridad (riesgo × 0,25) el riesgo queda en 12,50 USD, "
        "menos que el mínimo operable de 15 USD."
    )
    assert rule.suggestion.startswith("Esperá: el freno de la pérdida del día dura hasta que")
    assert "el freno de la pérdida de la semana dura hasta el lunes 19/10" in rule.suggestion
    steps = [a for a in plan.adjustments if a.source == SOURCE_SAFETY]
    assert [a.code for a in steps] == ["daily_loss_reduce", "weekly_loss_reduce"]
    assert steps[1].message.startswith("Riesgo bajado de 25 a 12,50 USD (× 0,5)")


def test_regime_and_safety_floor_message_names_both(settings, profile):
    # Neutral × 0,75 (37,5) y la pérdida del día × 0,5 → 18,75; con la semana × 0,25 → 9,375.
    trades = [trade(local(12, 10), -150.0), trade(local(14, 10), -60.0)]
    state, verdict = state_and_verdict(settings, profile, trades=trades)
    from tests.test_plan import fake_regime

    plan = plan_with(
        settings, profile, score=50.0, state=state, verdict=verdict, regime=fake_regime("neutral")
    )
    rule = next(r for r in plan.blocking_rules if r.code == MIN_OPERABLE_RISK)
    assert rule.message.startswith(
        "No operar: con los ajustes del régimen y los frenos de seguridad (riesgo × 0,25)"
    )


# --- veto ----------------------------------------------------------------------------------


def test_losing_streak_vetoes_first_with_until(settings, profile):
    # Abiertas ayer: no llegan al máximo de operaciones de hoy.
    trades = [trade(local(14, h), -10.0, opened=local(13, 20)) for h in (12, 13, 14)]
    state, verdict = state_and_verdict(settings, profile, trades=trades)
    assert verdict.blocked and verdict.block_codes == ("losing_streak",)
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference  # niveles como referencia
    first = plan.rules[0]
    assert first.code == "safety_losing_streak" and not first.ok
    assert first.message.startswith("No operar: perdiste 3 operaciones seguidas")
    assert "No se generan planes hasta hoy a las 18:00." in first.message
    assert first.suggestion == "Tomate la pausa: vuelve a haber planes hoy a las 18:00."
    assert plan.headline.startswith("No operar: perdiste 3 operaciones seguidas")
    assert all(r.code != SAFETY_RULE for r in plan.rules)  # sin el resumen "verde"
    assert plan.to_dict()["safety"]["block_codes"] == ["losing_streak"]


def test_max_trades_is_not_repeated_when_the_day_rule_already_says_it(settings, profile):
    """Tercera ronda (E4): el freno pone el estado en rojo; el plan lo dice una sola vez."""
    trades = [trade(local(14, h), 5.0) for h in (9, 10, 11)]
    state, verdict = state_and_verdict(settings, profile, trades=trades)
    assert verdict.status == "rojo" and verdict.block_codes == ("max_trades_today",)
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_NO_TRADE
    assert failing(plan) == [DAILY_LIMITS]
    assert plan.to_dict()["safety"]["block_codes"] == ["max_trades_today"]
    # Con las operaciones del día puestas a mano en 0 (para probar), el freno sí aparece.
    override = plan_with(settings, profile, state=state, verdict=verdict, trades_today=0)
    assert failing(override)[0] == "safety_max_trades_today"
    assert override.rules[0].message.startswith(
        "No operar: ya hiciste 3 de 3 operaciones permitidas hoy"
    )


def test_daily_block_is_not_repeated_when_the_day_rule_already_says_it(settings, profile):
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -100.0)])
    assert "daily_loss_block" in verdict.block_codes
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert failing(plan) == [DAILY_LIMITS]
    # Con el resultado del día puesto a mano en 0 (para probar), el freno sí aparece.
    override = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        pnl_today_usd=0.0,
        day_overrides=(DayOverride(OVERRIDE_PNL_TODAY, 0.0, -100.0),),
    )
    assert failing(override)[0] == "safety_daily_loss_block"
    assert override.blocking_rules[0].suggestion == (
        "Esperá: vuelve a haber planes cuando termine el día (a las 00:00)."
    )


def test_lock_file_blocks_every_plan(settings, profile):
    lock = LockInfo(
        exists=True,
        path="/datos/LOCKED",
        reason="La billetera del mes cayó 72 % desde su máximo.",
        triggered_at=NOW - timedelta(hours=2),
        report_path="/datos/incidents/informe.md",
    )
    state, verdict = state_and_verdict(settings, profile, lock=lock)
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert plan.status == STATUS_NO_TRADE
    assert failing(plan) == ["safety_lock_file"]
    rule = plan.blocking_rules[0]
    assert rule.message.startswith("No operar: la app está bloqueada desde el 14/10/2026 13:00")
    assert "/datos/incidents/informe.md" in rule.message
    assert rule.suggestion == (
        "Leé el informe del incidente con calma y, cuando termines, borrá a mano el archivo LOCKED."
    )
    assert plan.to_dict()["safety"]["locked"] is True


def test_monthly_drawdown_blocks_even_before_the_lock_file_exists(settings, profile):
    # 300 → 80 en el mes (73 %): el freno del mes bloquea aunque todavía no haya LOCKED.
    trades = [trade(local(5, 12), -110.0), trade(local(8, 12), -110.0)]
    state, verdict = state_and_verdict(settings, profile, trades=trades, balance=300.0)
    assert verdict.needs_lock and not verdict.locked
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert failing(plan) == ["safety_monthly_drawdown_lock"]
    assert plan.blocking_rules[0].message.startswith("No operar: tu billetera cayó 73,33 %")


def test_stored_volatility_event_uses_the_event_rule_with_its_end(settings, profile):
    event = {
        "id": 7,
        "starts_at": NOW - timedelta(hours=1),
        "until": NOW + timedelta(hours=2),
        "note": "CPI de EE. UU.",
    }
    state, verdict = state_and_verdict(settings, profile, vol=[event])
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert failing(plan) == [HIGH_VOLATILITY_EVENT]
    rule = plan.blocking_rules[0]
    assert rule.message.startswith(
        "No operar: marcaste un evento de alta volatilidad: «CPI de EE. UU.» (hasta el "
        "14/10/2026 17:00)"
    )
    assert rule.suggestion == (
        "Esperá a que pase el evento: se apaga solo hoy a las 17:00. Si lo cargaste por error, "
        "borralo."
    )
    # Una sola regla del evento (no hay "No hay eventos marcados" que la contradiga).
    assert [r.code for r in plan.rules].count(HIGH_VOLATILITY_EVENT) == 1
    assert all(r.code != SAFETY_RULE for r in plan.rules)


def test_manual_event_flag_and_stored_event_both_show(settings, profile):
    event = {"id": 8, "starts_at": NOW, "until": NOW + timedelta(hours=1), "note": "Fed"}
    state, verdict = state_and_verdict(settings, profile, vol=[event])
    plan = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        high_volatility_event=True,
        event_description="CPI",
    )
    assert set(failing(plan)) == {"safety_volatility_event", HIGH_VOLATILITY_EVENT}


def test_safety_blocks_reach_the_alternative_entry(settings, profile):
    trades = [trade(local(14, h), -10.0) for h in (12, 13, 14)]
    state, verdict = state_and_verdict(settings, profile, trades=trades)
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    alt = plan.alternative
    assert alt is not None and alt.plan is not None
    assert alt.plan.status == STATUS_NO_TRADE
    assert alt.plan.rules[0].code == "safety_losing_streak"
    assert "sí cumple las reglas" not in plan.headline


def test_invalid_safety_multiplier_is_rejected(settings, profile):
    state, verdict = state_and_verdict(settings, profile)
    bad = dataclasses.replace(verdict, risk_multiplier=0.0)
    with pytest.raises(ValueError, match="frenos de seguridad"):
        plan_with(settings, profile, state=state, verdict=bad)


# --- correlación -------------------------------------------------------------------------


def test_combined_limit_matches_the_safety_module(settings, profile):
    cfg = settings.correlation
    assert combined_risk_limit(profile.risk_per_trade_usd, cfg.max_combined_risk_multiplier) == 75
    plan = plan_with(settings, profile, correlation=exposure("long"))
    assert plan.to_dict()["correlation"]["limit_usd"] == 75.0


def test_correlated_position_shrinks_the_new_plan(settings, profile):
    plan = plan_with(settings, profile, correlation=exposure("long", position("ETH", "long", 40)))
    assert plan.status == STATUS_LONG
    assert plan.adjusted_risk_usd == pytest.approx(35.0) and plan.risk_capped_by_correlation
    assert plan.position.risk_usd <= 35.0 + 1e-9
    adj = next(a for a in plan.adjustments if a.source == SOURCE_CORRELATION)
    assert adj.code == CORRELATION_CAP
    assert adj.message == (
        "Riesgo bajado de 50 a 35 USD: ya tenés una posición en long (compra) en ETH, que se "
        "mueve junto con BTC, con 40 USD de riesgo. Si el mercado se da vuelta pierden juntas: "
        "sumando el plan nuevo, no pueden arriesgar más de 75 USD (1,5 × tu riesgo por "
        "operación de 50 USD)."
    )
    rule = next(r for r in plan.rules if r.code == CORRELATION_RULE)
    assert rule.ok and rule.message == (
        "Riesgo sumado con tus posiciones correlacionadas (ETH): 40 + 35 = 75 USD de 75 USD "
        "permitidos (el plan se achicó para entrar)."
    )
    summary = plan.to_dict()["correlation"]
    assert summary["existing_risk_usd"] == 40.0 and summary["remaining_usd"] == 35.0
    assert summary["capped"] is True and summary["combined_risk_usd"] == pytest.approx(75.0)
    assert plan.to_dict()["risk"]["capped_by_correlation"] is True


def test_correlation_cap_below_the_minimum_means_no_trade(settings, profile):
    plan = plan_with(settings, profile, correlation=exposure("long", position("ETH", "long", 65)))
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing(plan) == [MIN_OPERABLE_RISK]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: ya tenés una posición en long (compra) en ETH, que se mueve junto con BTC, "
        "con 65 USD de riesgo; con el tope de 75 USD para todas juntas, al plan nuevo le quedan "
        "10 USD, menos que el mínimo operable de 15 USD."
    )
    assert rule.suggestion.startswith("Esperá a cerrar alguna de esas posiciones")


def test_correlation_limit_already_full_blocks_without_touching_the_risk(settings, profile):
    plan = plan_with(
        settings,
        profile,
        correlation=exposure("long", position("ETH", "long", 45), position("SOL", "long", 30)),
    )
    assert failing(plan) == [CORRELATION_RULE]
    assert plan.adjusted_risk_usd == pytest.approx(50.0)  # niveles de referencia
    assert plan.blocking_rules[0].message == (
        "No operar: ya tenés 2 posiciones en long (compra) en ETH y SOL, que se mueven junto con "
        "BTC, con 75 USD de riesgo entre todas: el tope de 75 USD para todas juntas ya está lleno."
    )


def test_without_cap_the_correlation_rule_blocks(settings, profile):
    cfg = settings.correlation.model_copy(update={"cap_new_risk_to_limit": False})
    s = settings.model_copy(update={"correlation": cfg})
    plan = plan_with(s, profile, correlation=exposure("long", position("ETH", "long", 40)))
    assert failing(plan) == [CORRELATION_RULE]
    assert plan.blocking_rules[0].message == (
        "No operar: ya tenés una posición en long (compra) en ETH, que se mueve junto con BTC, "
        "con 40 USD de riesgo; sumando el plan nuevo serían 90 USD, más que el tope de 75 USD "
        "para posiciones muy correlacionadas en la misma dirección."
    )


def test_same_coin_same_side_counts_as_the_same_bet(settings, profile):
    plan = plan_with(
        settings, profile, correlation=exposure("long", position("BTC", "long", 30, "same_symbol"))
    )
    adj = next(a for a in plan.adjustments if a.source == SOURCE_CORRELATION)
    assert "una posición en long (compra) en BTC, la misma moneda, con 30 USD" in adj.message
    assert plan.adjusted_risk_usd == pytest.approx(45.0)


def test_only_the_plan_side_is_used(settings, profile):
    # Un short con la exposición armada para un long: no achica nada.
    plan = plan_with(
        settings, profile, score=-45.0, correlation=exposure("long", position("ETH", "long", 70))
    )
    assert plan.status == STATUS_SHORT and plan.adjusted_risk_usd == pytest.approx(50.0)
    rule = next(r for r in plan.rules if r.code == CORRELATION_RULE)
    assert rule.ok and rule.message.startswith("No tenés posiciones abiertas en short (venta)")


def test_hedge_on_the_same_coin_warns_but_adds_no_risk(settings, profile):
    hedge = position("BTC", "short", 30, "hedge")
    plan = plan_with(settings, profile, correlation=exposure("long", hedges=(hedge,)))
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(50.0)
    assert (
        "Ya tenés una posición en short (venta) en BTC (riesgo 30 USD): este plan va al revés y "
        "funciona como cobertura. No suma al tope de posiciones correlacionadas (1,5 × tu "
        "riesgo), pero su riesgo sí cuenta en lo que te queda del límite del día, porque pueden "
        "saltar los dos stops: si el precio va y viene fuerte, perdés en las dos."
    ) in plan.warnings
    assert all("no suma riesgo" not in w for w in plan.warnings)
    tp = settings.trade_plan.model_copy(update={"daily_cap_includes_open_risk": False})
    off = settings.model_copy(update={"trade_plan": tp})
    plain = plan_with(off, profile, correlation=exposure("long", hedges=(hedge,)))
    assert any(
        w.endswith(
            "No suma al tope de posiciones correlacionadas (1,5 × tu riesgo). Ojo: si el precio "
            "va y viene fuerte, pueden saltar los dos stops y perdés en las dos."
        )
        for w in plain.warnings
    )
    assert plan.to_dict()["correlation"]["hedges"][0]["symbol"] == "BTC"


def test_real_exposure_objects_from_the_safety_module(settings, profile):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-09-01", periods=300, freq="h", tz="UTC")
    base = np.random.default_rng(1).normal(0, 0.01, len(idx))
    returns = {
        "BTC": pd.Series(base, index=idx),
        "ETH": pd.Series(base + np.random.default_rng(2).normal(0, 0.002, len(idx)), index=idx),
    }
    matrix = correlation_matrix(returns, min_overlap=200, high_threshold=0.8)
    positions = [
        {"symbol": "ETH", "side": "long", "risk_usd": 40.0},
        {"symbol": "BTC", "side": "short", "risk_usd": 20.0},
    ]
    exposures = {
        side: correlated_exposure("BTC", side, [_obj(p) for p in positions], matrix, 0.8)
        for side in ("long", "short")
    }
    plan = plan_with(settings, profile, correlation=exposures)
    assert plan.adjusted_risk_usd == pytest.approx(35.0)  # ETH long suma (correlación alta)
    assert any("cobertura" in w for w in plan.warnings)  # BTC short: cobertura


def _obj(data: dict):
    from types import SimpleNamespace

    return SimpleNamespace(**data)


def test_exposure_side_mismatch_is_rejected(settings, profile):
    wrong = {"long": CorrelationExposure("short"), "short": None}
    with pytest.raises(ValueError, match="dirección long"):
        plan_with(settings, profile, correlation=wrong)


def test_safety_and_correlation_caps_chain_in_order(settings, profile):
    # × 0,5 por el día (50 → 25), te quedan 40 del día (no toca) y la correlación deja 20.
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -60.0)])
    plan = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        correlation=exposure("long", position("ETH", "long", 55)),
    )
    assert plan.status == STATUS_LONG
    assert [a.source for a in plan.adjustments if a.source != "position"] == [
        SOURCE_SAFETY,
        SOURCE_CORRELATION,
    ]
    risk = plan.to_dict()["risk"]
    assert risk["after_safety_usd"] == pytest.approx(25.0)
    assert risk["after_daily_usd"] == pytest.approx(25.0)
    assert risk["after_adjustments_usd"] == pytest.approx(20.0)


# --- números del día puestos a mano --------------------------------------------------------


def test_day_overrides_are_used_and_warned(settings, profile):
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -10.0)])
    overrides = (
        DayOverride(OVERRIDE_TRADES_TODAY, 3, state.trades_today),
        DayOverride(OVERRIDE_PNL_TODAY, -30.5, state.plan_pnl_today_usd),
    )
    plan = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        trades_today=3,
        pnl_today_usd=-30.5,
        day_overrides=overrides,
    )
    assert failing(plan) == [DAILY_LIMITS]  # el plan usa los números de prueba
    assert (
        "Números del día puestos a mano para probar: operaciones de hoy = 3 (lo cargado dice 1); "
        "resultado de hoy = -30,50 USD (lo cargado dice -10 USD). El plan usa esos números; los "
        "frenos de seguridad siguen mirando lo que cargaste."
    ) in plan.warnings
    assert plan.to_dict()["day_overrides"] == [
        {"field": "trades_today", "value": 3, "loaded": 1},
        {"field": "pnl_today_usd", "value": -30.5, "loaded": -10.0},
    ]


# --- riesgo de las posiciones abiertas en el límite del día (decisión D6) -----------------
#
# Revisión del Hito 5 (hallazgo 7): "lo que queda del límite del día" también descuenta lo
# que se pierde si saltan los stops de todas las posiciones abiertas
# (trade_plan.daily_cap_includes_open_risk), así el día nunca pasa su límite.


def without_open_risk_in_day(settings):
    tp = settings.trade_plan.model_copy(update={"daily_cap_includes_open_risk": False})
    return settings.model_copy(update={"trade_plan": tp})


def test_open_risk_below_the_minimum_means_no_trade_naming_both_parts(settings, profile):
    # Perdiste 40 y tenés 50 en juego en un SOL short (dirección contraria: la correlación
    # no lo cuenta). Quedan 10 del día < 15: "No operar". Antes daba LONG con 50 de riesgo
    # (40 + 50 + 50 = 140 si saltaban todos los stops, contra un límite de 100).
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -40.0)])
    sol = CorrelatedPosition("SOL", "short", 50.0, None, "opposite_side")
    plan = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        open_risk_usd=50.0,
        correlation={
            "long": CorrelationExposure("long"),
            "short": CorrelationExposure("short", (sol,)),
        },
    )
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing(plan) == [MIN_OPERABLE_RISK]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: hoy ya perdiste 40 USD y tenés 50 USD en juego en posiciones abiertas (lo "
        "que perdés si saltan sus stops): de tu límite de 100 USD del día quedan 10 USD, menos "
        "que el mínimo operable de 15 USD."
    )
    assert rule.suggestion == (
        "Esperá a mañana, o a que alguna posición abierta se cierre sin pérdida o tenga el stop "
        "en ganancia (ahí deja de contar)."
    )
    assert plan.headline.startswith("No operar: hoy ya perdiste 40 USD y tenés 50 USD en juego")
    daily = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert daily.message == (
        "Riesgo bajado de 50 a 10 USD: hoy ya perdiste 40 USD y tenés 50 USD en juego en "
        "posiciones abiertas (lo que perdés si saltan sus stops): quedan 10 USD de tu límite de "
        "100 USD del día. Así, aunque salten todos los stops, no pasás el límite."
    )
    # El JSON trae el riesgo abierto usado.
    d = plan.to_dict()
    assert d["risk"]["open_risk_usd"] == 50.0 and d["risk"]["daily_open_risk_usd"] == 50.0
    assert d["risk"]["daily_remaining_usd"] == pytest.approx(10.0)
    assert d["day_state"]["open_risk_usd"] == 50.0
    assert d["day_state"]["daily_open_risk_usd"] == 50.0
    assert d["day_state"]["daily_cap_includes_open_risk"] is True
    assert d["day_state"]["daily_remaining_usd"] == pytest.approx(10.0)


def test_open_risk_caps_the_plan_so_the_day_never_passes_its_limit(settings, profile):
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -30.0)])
    plan = plan_with(settings, profile, state=state, verdict=verdict, open_risk_usd=40.0)
    assert plan.status == STATUS_LONG
    assert plan.adjusted_risk_usd == pytest.approx(30.0) and plan.risk_capped_by_day
    assert plan.position.risk_usd <= 30.0 + 1e-9
    # Si saltan todos los stops: 30 cerrados + 40 abiertos + el plan nuevo <= 100.
    assert 30.0 + 40.0 + plan.position.risk_usd <= profile.max_daily_loss_usd + 1e-9
    daily = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert daily.message.startswith(
        "Riesgo bajado de 50 a 30 USD: hoy ya perdiste 30 USD y tenés 40 USD en juego en "
        "posiciones abiertas"
    )
    assert plan.to_dict()["day_state"]["loss_if_stopped_usd"] == pytest.approx(
        70.0 + plan.position.risk_usd
    )
    # La entrada alternativa usa el mismo riesgo abierto.
    assert plan.alternative is not None and plan.alternative.plan is not None
    assert plan.alternative.plan.daily_open_risk_usd == 40.0


def test_open_risk_without_losses_today(settings, profile):
    state, verdict = state_and_verdict(settings, profile)
    plan = plan_with(settings, profile, state=state, verdict=verdict, open_risk_usd=80.0)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(20.0)
    daily = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert daily.message.startswith(
        "Riesgo bajado de 50 a 20 USD: hoy no perdiste nada, pero tenés 80 USD en juego en "
        "posiciones abiertas"
    )


def test_open_risk_that_uses_the_whole_day_blocks_without_touching_the_risk(settings, profile):
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -40.0)])
    plan = plan_with(settings, profile, state=state, verdict=verdict, open_risk_usd=60.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing(plan) == [DAILY_REMAINING_RULE]
    assert plan.adjusted_risk_usd == pytest.approx(50.0)  # niveles de referencia
    assert not plan.risk_capped_by_day
    assert all(a.code != DAILY_LOSS_CAP for a in plan.adjustments)
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: hoy ya perdiste 40 USD y tenés 60 USD en juego en posiciones abiertas (lo "
        "que perdés si saltan sus stops): entre las dos cosas ya usás todo tu límite de 100 USD "
        "del día y no queda nada para otra operación."
    )
    assert rule.suggestion.startswith("Esperá a mañana, o a que alguna posición abierta")
    assert plan.headline.startswith("No operar: hoy ya perdiste 40 USD y tenés 60 USD en juego")
    assert plan.daily_remaining_usd == 0.0
    # Más riesgo abierto que el límite: lo mismo (nunca negativo).
    over = plan_with(settings, profile, open_risk_usd=150.0)
    assert failing(over) == [DAILY_REMAINING_RULE] and over.daily_remaining_usd == 0.0
    assert over.blocking_rules[0].message.startswith("No operar: hoy no perdiste nada, pero")


def test_daily_limit_already_reached_does_not_repeat_the_open_risk_rule(settings, profile):
    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -100.0)])
    plan = plan_with(settings, profile, state=state, verdict=verdict, open_risk_usd=30.0)
    assert failing(plan) == [DAILY_LIMITS]


def test_open_risk_flag_off_keeps_the_old_daily_cap(settings, profile):
    s = without_open_risk_in_day(settings)
    state, verdict = state_and_verdict(s, profile, trades=[trade(local(14, 10), -40.0)])
    plan = plan_with(s, profile, state=state, verdict=verdict, open_risk_usd=50.0)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(50.0)
    assert plan.daily_remaining_usd == pytest.approx(60.0) and not plan.risk_capped_by_day
    risk = plan.to_dict()["risk"]
    assert risk["open_risk_usd"] == 50.0 and risk["daily_open_risk_usd"] == 0.0
    # Con 80 perdidos, el motivo es el de antes (sin posiciones abiertas).
    state, verdict = state_and_verdict(s, profile, trades=[trade(local(14, 10), -80.0)])
    plan = plan_with(s, profile, state=state, verdict=verdict, open_risk_usd=50.0)
    daily = next(a for a in plan.adjustments if a.code == DAILY_LOSS_CAP)
    assert daily.message.startswith("Riesgo bajado de 25 a 20 USD: hoy ya perdiste 80 de tus")


def test_without_the_daily_cap_open_risk_only_warns(settings, profile):
    tp = settings.trade_plan.model_copy(update={"cap_risk_to_daily_remaining": False})
    s = settings.model_copy(update={"trade_plan": tp})
    state, verdict = state_and_verdict(s, profile, trades=[trade(local(14, 10), -40.0)])
    plan = plan_with(s, profile, state=state, verdict=verdict, open_risk_usd=60.0)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(50.0)
    warning = next(w for w in plan.warnings if w.startswith("Ojo con el límite del día"))
    assert warning.startswith(
        "Ojo con el límite del día: hoy ya perdiste 40 USD y tenés 60 USD en juego en posiciones "
        "abiertas (lo que perdés si saltan sus stops); si saltan todos los stops, también el de "
        "esta operación, el día cerraría con "
    )


def test_open_risk_is_validated(settings, profile):
    with pytest.raises(ValueError, match="posiciones abiertas"):
        plan_with(settings, profile, open_risk_usd=float("nan"))
    # Negativo (no debería pasar: cada posición con el stop en ganancia cuenta 0) = 0.
    plan = plan_with(settings, profile, open_risk_usd=-20.0)
    assert plan.open_risk_usd == 0.0 and plan.adjusted_risk_usd == pytest.approx(50.0)


def test_steps_keep_their_order_and_the_last_one_below_the_minimum_is_the_reason(settings, profile):
    # Régimen neutral × 0,75 (37,5) → la pérdida de 60 × 0,5 (18,75) → el día con 10 abiertos
    # deja 30 (no toca) → la correlación (ETH long 65) deja 10 < 15: el motivo es la correlación.
    from tests.test_plan import fake_regime

    state, verdict = state_and_verdict(settings, profile, trades=[trade(local(14, 10), -60.0)])
    plan = plan_with(
        settings,
        profile,
        score=50.0,
        regime=fake_regime("neutral"),
        state=state,
        verdict=verdict,
        open_risk_usd=10.0,
        correlation=exposure("long", position("ETH", "long", 65)),
    )
    sources = list(dict.fromkeys(a.source for a in plan.adjustments if a.source != "position"))
    assert sources == ["regime", SOURCE_SAFETY, SOURCE_CORRELATION]
    assert plan.to_dict()["risk"]["after_daily_usd"] == pytest.approx(18.75)  # el día no tocó
    rule = next(r for r in plan.blocking_rules if r.code == MIN_OPERABLE_RISK)
    assert rule.message.startswith("No operar: ya tenés una posición en long (compra) en ETH")
    # Ahora el día (con 30 abiertos) deja 10 después de los frenos y la correlación no toca:
    # el motivo es el del día.
    plan = plan_with(
        settings,
        profile,
        state=state,
        verdict=verdict,
        open_risk_usd=30.0,
        correlation=exposure("long", position("ETH", "long", 30)),
    )
    sources = [a.source for a in plan.adjustments if a.source != "position"]
    assert sources == [SOURCE_SAFETY, SOURCE_DAILY_LIMIT]
    rule = next(r for r in plan.blocking_rules if r.code == MIN_OPERABLE_RISK)
    assert rule.message.startswith("No operar: hoy ya perdiste 60 USD y tenés 30 USD en juego")
    risk = plan.to_dict()["risk"]
    assert risk["after_safety_usd"] == pytest.approx(25.0)
    assert risk["after_daily_usd"] == pytest.approx(10.0)
    assert risk["daily_open_risk_usd"] == 30.0


# --- correlación desconocida (hallazgo 8) -------------------------------------------------


def test_unknown_correlation_is_worded_as_unknown(settings, profile):
    # Sin matriz (falló la descarga): ETH long se cuenta por las dudas, sin afirmar nada.
    eth = [_obj({"symbol": "ETH", "side": "long", "risk_usd": 40.0})]
    exposures = {s: correlated_exposure("BTC", s, eth, None, 0.8) for s in ("long", "short")}
    plan = plan_with(settings, profile, correlation=exposures)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(35.0)
    adj = next(a for a in plan.adjustments if a.source == SOURCE_CORRELATION)
    assert adj.message == (
        "Riesgo bajado de 50 a 35 USD: ya tenés una posición en long (compra) en ETH, con 40 USD "
        "de riesgo (no hay datos para saber si ETH se mueve junto con BTC; por las dudas se "
        "cuenta). Si el mercado se da vuelta pueden perder juntas: sumando el plan nuevo, no "
        "pueden arriesgar más de 75 USD (1,5 × tu riesgo por operación de 50 USD)."
    )
    rule = next(r for r in plan.rules if r.code == CORRELATION_RULE)
    assert rule.ok and rule.message == (
        "Riesgo sumado con tus posiciones en long (compra) en ETH: 40 + 35 = 75 USD de 75 USD "
        "permitidos (el plan se achicó para entrar). No hay datos para saber si ETH se mueve "
        "junto con BTC; por las dudas se cuenta."
    )
    assert "junto con BTC, con" not in adj.message  # no lo afirma como un hecho
    assert plan.to_dict()["correlation"]["unknown_symbols"] == ["ETH"]

    # Debajo del mínimo y sin el tope que achica: los mismos textos sin afirmar.
    big = [_obj({"symbol": "ETH", "side": "long", "risk_usd": 65.0})]
    exposures = {s: correlated_exposure("BTC", s, big, None, 0.8) for s in ("long", "short")}
    floor = plan_with(settings, profile, correlation=exposures)
    assert floor.blocking_rules[0].message == (
        "No operar: ya tenés una posición en long (compra) en ETH, con 65 USD de riesgo (no hay "
        "datos para saber si ETH se mueve junto con BTC; por las dudas se cuenta); con el tope "
        "de 75 USD para todas juntas, al plan nuevo le quedan 10 USD, menos que el mínimo "
        "operable de 15 USD."
    )
    cfg = settings.correlation.model_copy(update={"cap_new_risk_to_limit": False})
    s = settings.model_copy(update={"correlation": cfg})
    exposures = {k: correlated_exposure("BTC", k, eth, None, 0.8) for k in ("long", "short")}
    blocked = plan_with(s, profile, correlation=exposures)
    assert blocked.blocking_rules[0].message.endswith(
        "más que el tope de 75 USD para posiciones en la misma dirección que se mueven juntas "
        "(o que no se sabe si lo hacen)."
    )


def test_mixed_known_and_unknown_correlation(settings, profile):
    plan = plan_with(
        settings,
        profile,
        correlation=exposure(
            "long",
            position("ETH", "long", 20),
            position("SOL", "long", 10, "unknown_correlation"),
        ),
    )
    adj = next(a for a in plan.adjustments if a.source == SOURCE_CORRELATION)
    assert adj.message.startswith(
        "Riesgo bajado de 50 a 45 USD: ya tenés 2 posiciones en long (compra) en ETH y SOL, con "
        "30 USD de riesgo entre todas (ETH se mueve junto con BTC; de SOL no hay datos para saber "
        "si se mueve junto con BTC y por las dudas se cuenta)."
    )
    # Sin motivo pero sin correlación (armado a mano): también "sin dato".
    hand = CorrelatedPosition("ETH", "long", 40.0)
    plan = plan_with(settings, profile, correlation=exposure("long", hand))
    adj = next(a for a in plan.adjustments if a.source == SOURCE_CORRELATION)
    assert "no hay datos para saber si ETH se mueve junto con BTC" in adj.message


# --- otra moneda al revés no es cobertura (hallazgo 15) -----------------------------------


def test_other_coin_in_the_opposite_direction_is_not_a_hedge_in_the_plan(settings, profile):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-09-01", periods=300, freq="h", tz="UTC")
    base = np.random.default_rng(1).normal(0, 0.01, len(idx))
    returns = {
        "BTC": pd.Series(base, index=idx),
        "ETH": pd.Series(base + np.random.default_rng(2).normal(0, 0.002, len(idx)), index=idx),
        "DOGE": pd.Series(np.random.default_rng(3).normal(0, 0.01, len(idx)), index=idx),
    }
    matrix = correlation_matrix(returns, min_overlap=200, high_threshold=0.8)
    positions = [
        _obj({"symbol": "ETH", "side": "short", "risk_usd": 30.0}),
        _obj({"symbol": "DOGE", "side": "short", "risk_usd": 10.0}),
    ]
    exposures = {
        s: correlated_exposure("BTC", s, positions, matrix, 0.8) for s in ("long", "short")
    }
    plan = plan_with(settings, profile, correlation=exposures)
    assert plan.status == STATUS_LONG and plan.adjusted_risk_usd == pytest.approx(50.0)
    assert not any("cobertura" in w for w in plan.warnings)
    assert plan.to_dict()["correlation"]["hedges"] == []
    skipped = {e.symbol: e.explanation for e in exposures["long"].skipped}
    assert "se mueve parecido a BTC" in skipped["ETH"]
    assert "no se mueve junto con BTC" in skipped["DOGE"]


# --- el evento guardado va primero (hallazgo 22) ------------------------------------------


def test_stored_volatility_event_goes_first_in_the_rules_and_headline(settings, profile):
    event = {"id": 9, "starts_at": NOW, "until": NOW + timedelta(hours=1), "note": "Fed"}
    state, verdict = state_and_verdict(settings, profile, vol=[event])
    # Score sin dirección: antes el titular decía "el score ... no hay una dirección clara".
    plan = plan_with(settings, profile, score=10.0, state=state, verdict=verdict)
    assert failing(plan) == [HIGH_VOLATILITY_EVENT, "direction"]
    assert plan.rules[0].code == HIGH_VOLATILITY_EVENT
    assert plan.headline.startswith("No operar: marcaste un evento de alta volatilidad: «Fed»")
    assert plan.headline.endswith("(y 1 motivo más).")
    # Con otro freno que bloquea, van en el orden de los frenos (el evento después de la racha).
    trades = [trade(local(14, h), -10.0) for h in (12, 13, 14)]
    state, verdict = state_and_verdict(settings, profile, trades=trades, vol=[event])
    plan = plan_with(settings, profile, state=state, verdict=verdict)
    assert failing(plan)[:2] == ["safety_losing_streak", HIGH_VOLATILITY_EVENT]
    assert [r.code for r in plan.rules].count(HIGH_VOLATILITY_EVENT) == 1


# --- saldo de la billetera (hallazgo 14) -------------------------------------------------


def test_empty_wallet_means_no_trade_with_the_levels_as_reference(settings, profile):
    from app.risk.plan import WALLET_BALANCE_RULE

    plan = plan_with(settings, profile, wallet_balance_usd=0.0)
    assert plan.status == STATUS_NO_TRADE and plan.is_reference
    assert failing(plan) == [WALLET_BALANCE_RULE]
    rule = plan.blocking_rules[0]
    assert rule.message == (
        "No operar: tu billetera de futuros tiene 0 USD (según el saldo que anotaste y lo que "
        "cargaste después): no hay plata para poner de margen."
    )
    assert 'make trade ARGS="deposit --amount ..."' in rule.suggestion
    assert plan.headline.startswith("No operar: tu billetera de futuros tiene 0 USD")
    # El riesgo no se toca: los niveles quedan como referencia.
    assert plan.adjusted_risk_usd == pytest.approx(50.0)
    day = plan.to_dict()["day_state"]
    assert day["wallet_balance_usd"] == 0.0 and day["wallet_balance_assumed"] is False
    # La entrada alternativa hereda el bloqueo.
    alt = plan.alternative
    assert alt is not None and alt.plan is not None
    assert WALLET_BALANCE_RULE in [r.code for r in alt.plan.blocking_rules]


def test_negative_assumed_wallet_says_it_is_assumed(settings, profile):
    from app.risk.plan import WALLET_BALANCE_RULE

    plan = plan_with(settings, profile, wallet_balance_usd=-12.5, wallet_balance_assumed=True)
    rule = next(r for r in plan.rules if r.code == WALLET_BALANCE_RULE)
    assert not rule.ok
    assert rule.message.startswith(
        "No operar: tu billetera de futuros tiene -12,50 USD (según la cuenta de la app (es un "
        "saldo supuesto: nunca anotaste el saldo real;"
    )


def test_positive_wallet_adds_no_rule(settings, profile):
    from app.risk.plan import WALLET_BALANCE_RULE

    plan = plan_with(settings, profile, wallet_balance_usd=0.01, available_margin_usd=0.01)
    assert WALLET_BALANCE_RULE not in [r.code for r in plan.rules]
    # Con un centavo de margen la posición no llega al riesgo mínimo.
    assert plan.status == STATUS_NO_TRADE and MIN_OPERABLE_RISK in failing(plan)


def test_margin_capped_by_an_assumed_wallet_is_warned(settings, profile):
    capped = plan_with(
        settings,
        profile,
        available_margin_usd=60.0,
        wallet_equity_usd=60.0,
        wallet_balance_usd=60.0,
        wallet_balance_assumed=True,
    )
    assert capped.position.margin_usd <= 60.0
    assert any(
        w.startswith("El margen quedó limitado por el saldo supuesto de tu billetera (60 USD)")
        for w in capped.warnings
    )
    noted = plan_with(
        settings,
        profile,
        available_margin_usd=60.0,
        wallet_equity_usd=60.0,
        wallet_balance_usd=60.0,
    )
    assert not any("saldo supuesto" in w for w in noted.warnings)


def test_a_tiny_wallet_says_the_wallet_is_what_limits_the_position(settings, profile):
    plan = plan_with(settings, profile, available_margin_usd=5.0, wallet_balance_usd=5.0)
    rule = next(r for r in plan.blocking_rules if r.code == MIN_OPERABLE_RISK)
    assert "aun con el margen máximo de 5 USD" in rule.message
    assert rule.suggestion.startswith(
        "El margen está limitado por el saldo de tu billetera (5 USD): con tan poca plata la "
        "posición no llega al riesgo mínimo."
    )
    assumed = plan_with(
        settings,
        profile,
        available_margin_usd=5.0,
        wallet_balance_usd=5.0,
        wallet_balance_assumed=True,
    )
    rule = next(r for r in assumed.blocking_rules if r.code == MIN_OPERABLE_RISK)
    assert rule.suggestion.startswith(
        "El margen está limitado por el saldo supuesto de tu billetera"
    )


def test_correlation_rule_says_open_risk_still_counts_in_the_day(settings, profile):
    """Revisión 7 (D6): sin posiciones que sumen al tope, el riesgo abierto igual va al día."""
    positions = [_obj({"symbol": "DOGE", "side": "short", "risk_usd": 40.0})]
    exposures = {s: correlated_exposure("BTC", s, positions, None, 0.8) for s in ("long", "short")}
    plan = plan_with(settings, profile, correlation=exposures, open_risk_usd=40.0)
    rule = next(r for r in plan.rules if r.code == CORRELATION_RULE)
    assert rule.ok
    assert rule.message == (
        "No tenés posiciones abiertas en long (compra) que se muevan junto con BTC: para el tope "
        "de posiciones correlacionadas, el plan no suma riesgo con otras (lo que tenés en juego "
        "en tus posiciones abiertas sí cuenta en lo que te queda del límite del día)."
    )
    assert "no suma riesgo." not in exposures["long"].skipped[0].explanation
    alone = plan_with(settings, profile, correlation=exposures, open_risk_usd=0.0)
    rule = next(r for r in alone.rules if r.code == CORRELATION_RULE)
    assert rule.message.endswith("el plan no suma riesgo con otras.")
