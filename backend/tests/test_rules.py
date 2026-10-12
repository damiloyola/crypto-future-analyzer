"""Hito 4: reglas que fuerzan "No operar" y formato de números de los mensajes."""

import json
import math

import pytest

from app.risk.fees import net_rr
from app.risk.liquidation import (
    check_liquidation_vs_stop,
    liquidation_price,
    max_size_for_safe_liquidation,
    min_margin_for_safe_liquidation,
)
from app.risk.profiles import get_profile
from app.risk.rules import (
    DAILY_LIMITS,
    DIRECTION,
    HIGH_VOLATILITY_EVENT,
    LEVERAGE,
    LIQUIDATION,
    MIN_ABS_SCORE,
    MIN_OPERABLE_RISK,
    MIN_RR,
    TIMEFRAME_CONFLICT,
    RuleResult,
    check_daily_limits,
    check_direction,
    check_high_volatility_event,
    check_leverage,
    check_liquidation,
    check_min_abs_score,
    check_min_operable_risk,
    check_rr,
    check_timeframe_conflict,
    direction_from_score,
    failed_rules,
    fmt_leverage,
    fmt_multiplier,
    fmt_num,
    fmt_percent,
    fmt_price,
    fmt_ratio,
    fmt_score,
    fmt_score_beside,
    fmt_usd,
    json_safe,
    price_decimals,
)

FEE = 0.0006
MMR = 0.005


@pytest.fixture
def agresivo(settings):
    return get_profile("agresivo", settings)


# --- formato -------------------------------------------------------------------------


def test_number_formatting_is_spanish():
    assert fmt_num(1234.5) == "1.234,5"
    assert fmt_num(1234.5, trim=False) == "1.234,50"
    assert fmt_num(2.0) == "2"
    assert fmt_num(-0.0001) == "0"
    assert fmt_num(float("nan")) == "—"
    assert fmt_num(None) == "—"
    assert fmt_usd(50) == "50 USD"
    assert fmt_usd(162.4) == "162,40 USD"
    assert fmt_usd(5000) == "5.000 USD"
    assert fmt_usd(28.404) == "28,40 USD"
    assert fmt_usd(100, unit=False) == "100"
    assert fmt_leverage(25) == "25x"
    assert fmt_leverage(31.24) == "31,2x"
    assert fmt_leverage(0.43) == "0,43x"
    assert fmt_multiplier(0.5) == "0,5"
    assert fmt_multiplier(0.1875) == "0,1875"
    assert fmt_percent(55, 0) == "55 %"
    assert fmt_percent(0.123) == "0,12 %"
    assert fmt_score(42) == "+42"
    assert fmt_score(-30) == "-30"
    assert fmt_score(42.56) == "+42,6"
    assert fmt_price(60000) == "60.000"
    assert fmt_price(59400.5) == "59.400,5"
    assert fmt_price(2.51234) == "2,5123"  # XRP: 4 decimales
    assert fmt_ratio(1.3214) == "1,32"


def test_json_safe_removes_nan_and_tuples():
    out = json_safe({"a": float("nan"), "b": (1, float("inf")), "c": {"d": 2.5}, "e": True})
    assert out == {"a": None, "b": [1, None], "c": {"d": 2.5}, "e": True}
    json.dumps(out, allow_nan=False)


def test_rule_result_to_dict():
    r = RuleResult("x", False, "No operar: algo.", "Esperá.")
    assert r.to_dict() == {
        "code": "x",
        "ok": False,
        "message": "No operar: algo.",
        "suggestion": "Esperá.",
    }


def test_failed_rules_keeps_order():
    a = RuleResult("a", True, "bien")
    b = RuleResult("b", False, "No operar: b")
    c = RuleResult("c", False, "No operar: c")
    assert failed_rules([a, b, c]) == (b, c)
    assert failed_rules([a]) == ()


# --- dirección -------------------------------------------------------------------------


def test_score_between_thresholds_is_no_trade(settings):
    th = settings.scoring.direction_thresholds
    r = check_direction(12, th)
    assert r.code == DIRECTION and r.ok is False
    assert r.message == (
        "No operar: el score (+12) está entre -30 y +30, no hay una dirección clara."
    )
    assert r.suggestion == "Esperá a que el score llegue a +30 (long) o a -30 (short)."
    assert direction_from_score(12, th) is None
    assert check_direction(29.9, th).ok is False
    assert check_direction(-29.9, th).ok is False


def test_thresholds_are_inclusive(settings):
    th = settings.scoring.direction_thresholds
    assert direction_from_score(30, th) == "long"
    assert direction_from_score(-30, th) == "short"
    r = check_direction(30, th)
    assert r.ok and "long" in r.message
    assert check_direction(-75, th).ok


def test_direction_rejects_non_finite_score(settings):
    with pytest.raises(ValueError):
        check_direction(float("nan"), settings.scoring.direction_thresholds)


def test_min_abs_score_long_and_short():
    assert check_min_abs_score("long", 50, 50).ok
    assert check_min_abs_score("short", -50, 50).ok
    r = check_min_abs_score("short", -44, 45, reason="el régimen Neutral")
    assert r.code == MIN_ABS_SCORE and not r.ok
    assert r.message == (
        "No operar: para un short el score tiene que ser -45 o menos y está en -44 "
        "(lo pide el régimen Neutral)."
    )
    # Un score fuerte en la dirección contraria no cuenta.
    assert check_min_abs_score("long", -80, 30).ok is False


# --- caso borde obligatorio: timeframes en conflicto ------------------------------------


@pytest.mark.parametrize(("t4h", "t1h"), [("alcista", "bajista"), ("bajista", "alcista")])
def test_edge_case_timeframes_in_conflict_block(t4h, t1h):
    r = check_timeframe_conflict(t4h, t1h)
    assert r.code == TIMEFRAME_CONFLICT
    assert r.ok is False
    assert r.message == f"No operar: 4h es {t4h} y 1h es {t1h}: van en direcciones opuestas."
    assert r.suggestion == "Esperá a que 4h y 1h se pongan de acuerdo."


@pytest.mark.parametrize(
    ("t4h", "t1h"),
    [
        ("alcista", "alcista"),
        ("bajista", "bajista"),
        ("alcista", "lateral"),
        ("lateral", "bajista"),
        ("lateral", "lateral"),
    ],
)
def test_lateral_or_aligned_is_not_a_conflict(t4h, t1h):
    assert check_timeframe_conflict(t4h, t1h).ok


def test_missing_trend_is_not_a_conflict_but_is_said():
    r = check_timeframe_conflict(None, "bajista")
    assert r.ok and "No hay dato de tendencia en 4h" in r.message


def test_unknown_trend_label_is_an_error():
    with pytest.raises(ValueError, match="alcista, bajista o lateral"):
        check_timeframe_conflict("up", "bajista")


# --- R:R con comisiones --------------------------------------------------------------------


def test_rr_below_minimum_blocks():
    r = check_rr(1.32, 1.5)
    assert r.code == MIN_RR and r.ok is False
    assert r.message.startswith("No operar: el R:R a TP2 con comisiones es 1,32 y el mínimo es 1,5")
    assert r.suggestion


def test_rr_at_minimum_passes():
    assert check_rr(1.5, 1.5).ok
    assert check_rr(1.5 * (1 - 1e-12), 1.5).ok  # ruido de coma flotante
    assert check_rr(2.4, 2.0).ok


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_rr_unknown_blocks(value):
    r = check_rr(value, 1.5)
    assert not r.ok and "no se pudo calcular" in r.message


def test_rr_uses_net_rr_with_fees():
    # Objetivo a 2R bruto: con comisiones el R:R neto queda por debajo de 2.
    entry, stop = 60000.0, 59400.0
    target = entry + 2 * (entry - stop)
    rr = net_rr("long", entry, stop, target, FEE, FEE)
    assert rr < 2
    assert check_rr(rr, 2.0).ok is False
    assert check_rr(rr, 1.5).ok is True


# --- apalancamiento --------------------------------------------------------------------------


def test_leverage_over_profile_cap_blocks(agresivo):
    r = check_leverage(26, agresivo)
    assert r.code == LEVERAGE and not r.ok
    assert r.message == "No operar: el apalancamiento quedaría en 26x y tu tope es 25x."
    assert check_leverage(25, agresivo).ok
    assert check_leverage(25 * (1 + 1e-12), agresivo).ok


def test_leverage_uses_effective_cap(settings):
    p = get_profile("agresivo", settings, 40)
    assert check_leverage(30, p).ok
    assert not check_leverage(41, p).ok


# --- límites del día -------------------------------------------------------------------------


def test_daily_limits_ok(agresivo):
    r = check_daily_limits(2, -50, agresivo)
    assert r.code == DAILY_LIMITS and r.ok
    assert r.message == "Hoy: 2 de 3 operaciones, pérdida 50 USD de 100 USD permitidos."
    assert check_daily_limits(0, 250.0, agresivo).ok  # ganar no bloquea


def test_daily_trade_limit_blocks(agresivo):
    r = check_daily_limits(3, 10, agresivo)
    assert not r.ok
    assert r.message == "No operar: ya hiciste 3 operaciones hoy y el máximo es 3."
    assert r.suggestion == "Esperá a mañana."


def test_daily_loss_limit_blocks(agresivo):
    r = check_daily_limits(1, -100, agresivo)
    assert not r.ok
    assert r.message == "No operar: hoy perdiste 100 USD y el límite es 100 USD."


def test_both_daily_limits_in_one_message(agresivo):
    r = check_daily_limits(4, -120.5, agresivo)
    assert not r.ok
    assert "4 operaciones" in r.message and "120,50 USD" in r.message


@pytest.mark.parametrize("trades", [-1, 1.5, True, "2"])
def test_daily_limits_invalid_trades(agresivo, trades):
    with pytest.raises(ValueError):
        check_daily_limits(trades, 0, agresivo)


# --- evento de alta volatilidad ---------------------------------------------------------------


def test_high_volatility_event_blocks():
    r = check_high_volatility_event(True, "dato de inflación de EE. UU.")
    assert r.code == HIGH_VOLATILITY_EVENT and not r.ok
    assert r.message.startswith(
        "No operar: marcaste un evento de alta volatilidad (dato de inflación de EE. UU.)"
    )
    assert r.suggestion == "Esperá a que pase el evento y desmarcalo."
    assert check_high_volatility_event(False).ok


def test_high_volatility_flag_must_be_bool():
    with pytest.raises(ValueError):
        check_high_volatility_event("sí")


# --- riesgo mínimo operable ---------------------------------------------------------------


def test_min_operable_risk(agresivo):
    assert check_min_operable_risk(15, agresivo).ok
    r = check_min_operable_risk(9.375, agresivo)
    assert r.code == MIN_OPERABLE_RISK and not r.ok
    assert r.message == (
        "No operar: con los ajustes el riesgo queda en 9,38 USD, menos que el mínimo "
        "operable de 15 USD."
    )


# --- liquidación -----------------------------------------------------------------------------


def test_liquidation_ok_message():
    entry, stop, size, margin = 60000.0, 59400.0, 4466.68, 178.67
    liq = liquidation_price("long", entry, size, margin, MMR, FEE)
    check = check_liquidation_vs_stop("long", entry, stop, liq, 2.0)
    r = check_liquidation(check)
    assert r.code == LIQUIDATION and r.ok
    assert "veces la distancia al stop (mínimo 2)" in r.message


def test_liquidation_ratio_too_small_suggests_margin_and_size():
    entry, stop, size, margin = 60000.0, 59400.0, 4466.68, 100.0
    liq = liquidation_price("long", entry, size, margin, MMR, FEE)
    check = check_liquidation_vs_stop("long", entry, stop, liq, 2.0)
    assert not check.ok and not check.liq_before_stop
    m = min_margin_for_safe_liquidation("long", entry, stop, size, MMR, FEE, 2.0)
    s = max_size_for_safe_liquidation("long", entry, stop, margin, MMR, FEE, 2.0)
    r = check_liquidation(check, suggested_margin_usd=m, suggested_size_usd=s)
    assert not r.ok
    assert "tiene que estar al menos a 2 veces" in r.message
    assert r.suggestion.startswith(f"Subí el margen a {fmt_usd(m)} (el riesgo no cambia)")
    assert f"achicá la posición a {fmt_usd(s)}" in r.suggestion


def test_liquidation_before_stop_message():
    entry, stop = 60000.0, 58200.0  # stop a 3 %
    liq = liquidation_price("long", entry, 1603.5, 40.0, MMR, FEE)
    check = check_liquidation_vs_stop("long", entry, stop, liq, 2.0)
    assert check.liq_before_stop
    r = check_liquidation(check)
    assert not r.ok
    assert r.message.startswith("No operar: te liquidan antes de llegar al stop")
    assert "stop en 58.200" in r.message
    assert r.suggestion == "Subí el margen o achicá la posición."


def test_liquidation_short_side():
    entry, stop = 60000.0, 60600.0
    liq = liquidation_price("short", entry, 4466.68, 100.0, MMR, FEE)
    check = check_liquidation_vs_stop("short", entry, stop, liq, 2.0)
    r = check_liquidation(check)
    assert r.ok is check.ok
    assert math.isfinite(check.ratio)


def test_no_liquidation_passes():
    check = check_liquidation_vs_stop("long", 100.0, 50.0, None, 2.0)
    r = check_liquidation(check)
    assert r.ok and "no se puede liquidar" in r.message


# --- revisión final del Hito 4 ------------------------------------------------------


def test_json_safe_turns_numpy_scalars_into_python_values():
    np = pytest.importorskip("numpy")
    # np.bool_(False) terminaba como el texto "False", que en JS es verdadero.
    assert json_safe(np.bool_(False)) is False
    assert json_safe(np.bool_(True)) is True
    assert json_safe(np.float64(1.5)) == 1.5 and type(json_safe(np.float64(1.5))) is float
    assert json_safe(np.int64(3)) == 3 and type(json_safe(np.int64(3))) is int
    assert json_safe(np.float64("nan")) is None
    assert json_safe({"ok": np.bool_(False), "v": (np.float64(2.0),)}) == {
        "ok": False,
        "v": [2.0],
    }


def test_score_messages_do_not_round_the_score_to_the_threshold(settings):
    r = check_min_abs_score("long", 49.96, 50.0, reason="el régimen Bajista")
    assert r.ok is False
    assert "tiene que ser +50 o más y está en +49,96" in r.message
    assert "está en +50" not in r.message
    short = check_min_abs_score("short", -49.96, 50.0)
    assert "tiene que ser -50 o menos y está en -49,96" in short.message
    th = settings.scoring.direction_thresholds
    assert "el score (+29,97) está entre" in check_direction(29.97, th).message
    assert "el score (-29,97) está entre" in check_direction(-29.97, th).message
    # Lejos del umbral queda como siempre (un decimal).
    assert "el score (+12,3) está entre" in check_direction(12.34, th).message
    assert fmt_score_beside(45.0, 50.0) == "+45"


def test_price_uses_the_reference_decimals():
    # Mismo precio, misma cantidad de decimales que la tabla de la consola.
    assert price_decimals(60000.0) == 2 and price_decimals(2.6543) == 4
    assert fmt_price(2.7207, 26.5) == "2,721"  # con la entrada de 26,5
    assert fmt_price(2.7207) == "2,7207"  # sin referencia, por su tamaño
    assert fmt_price(167.5, 167.5) == "167,5"
