"""Hito 4: ajustes del plan por régimen, confianza baja e inestabilidad."""

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from app.regime.service import RegimeReport, RegimeUnavailable
from app.risk.profiles import get_profile
from app.risk.regime_adjustments import (
    LOW_CONFIDENCE,
    MIN_OPERABLE_RISK,
    REGIME_BLOCKED,
    REGIME_FUNDING_NORMAL,
    REGIME_FUNDING_UNKNOWN,
    REGIME_MIN_RR,
    REGIME_MIN_SCORE,
    REGIME_RISK,
    REGIME_UNAVAILABLE,
    UNSTABLE,
    Adjustment,
    RegimeAdjustmentResult,
    apply_regime_adjustments,
)


def regime(category: str | None, confidence: float | None = 0.9, unstable: bool = False):
    """Régimen falso con los tres atributos que se usan (como ``RegimeReport``)."""
    return SimpleNamespace(category=category, confidence=confidence, unstable=unstable)


@pytest.fixture
def agresivo(settings):
    return get_profile("agresivo", settings)


def codes(result: RegimeAdjustmentResult) -> list[str]:
    return [a.code for a in result.adjustments]


def test_regime_report_has_the_attributes_used():
    names = {f.name for f in dataclasses.fields(RegimeReport)}
    assert {"category", "confidence", "unstable"} <= names


# --- caso borde obligatorio: Crash con long ---------------------------------------------


def test_edge_case_crash_regime_blocks_long(settings, agresivo):
    r = apply_regime_adjustments("long", 80, regime("crash"), 0.01, agresivo, settings)
    assert r.blocked is True
    assert r.ok is False
    assert r.block_code == REGIME_BLOCKED
    assert r.block_reason == (
        "No operar: con el régimen Crash (caída fuerte) los longs están bloqueados."
    )
    assert r.score_ok is True  # el score no lo salva
    assert codes(r) == [REGIME_BLOCKED]
    assert r.no_trade_reasons == (r.block_reason,)


def test_crash_short_allowed_with_half_risk(settings, agresivo):
    r = apply_regime_adjustments("short", -60, regime("crash"), 0.01, agresivo, settings)
    assert r.blocked is False and r.ok is True
    assert r.risk_multiplier == pytest.approx(0.5)
    assert r.risk_usd == pytest.approx(25)
    assert r.adjustments == (
        Adjustment(REGIME_RISK, "Riesgo × 0,5: régimen Crash (caída fuerte).", 0.5),
    )


# --- score mínimo efectivo ---------------------------------------------------------------


def test_bajista_long_needs_score_50(settings, agresivo):
    r = apply_regime_adjustments("long", 45, regime("bajista"), None, agresivo, settings)
    assert r.min_abs_score == 50
    assert r.min_abs_score_reason == "el régimen Bajista"
    assert r.score_ok is False and r.ok is False and r.blocked is False
    assert r.score_check.message == (
        "No operar: para un long el score tiene que ser +50 o más y está en +45 "
        "(lo pide el régimen Bajista)."
    )
    assert codes(r) == [REGIME_MIN_SCORE]
    assert r.adjustments[0].message == "Score mínimo +50 para long: régimen Bajista."
    assert apply_regime_adjustments("long", 50, regime("bajista"), None, agresivo, settings).ok


def test_bajista_short_has_no_extra_rules(settings, agresivo):
    r = apply_regime_adjustments("short", -31, regime("bajista"), None, agresivo, settings)
    assert r.min_abs_score == 30
    assert r.min_abs_score_reason == "el umbral de dirección"
    assert r.ok and r.adjustments == () and r.risk_multiplier == 1.0


def test_alcista_short_needs_score_minus_50(settings, agresivo):
    r = apply_regime_adjustments("short", -45, regime("alcista"), None, agresivo, settings)
    assert r.min_abs_score == 50 and r.score_ok is False
    assert "-50 o menos" in r.score_check.message
    assert apply_regime_adjustments("short", -50, regime("alcista"), None, agresivo, settings).ok
    # El long en Alcista no tiene reglas extra.
    assert apply_regime_adjustments("long", 31, regime("alcista"), None, agresivo, settings).ok


def test_euforia_short_needs_score_minus_40(settings, agresivo):
    r = apply_regime_adjustments("short", -35, regime("euforia"), None, agresivo, settings)
    assert r.min_abs_score == 40  # el régimen (40) pisa el umbral de dirección (30)
    assert r.score_ok is False
    assert r.adjustments[0].message == (
        "Score mínimo -40 para short: régimen Euforia (subida exagerada)."
    )


def test_regime_min_score_below_threshold_does_not_lower_it(settings, agresivo):
    cfg = settings.regime_adjustments
    bajista = cfg.bajista.model_copy(
        update={"long": cfg.bajista.long.model_copy(update={"min_abs_score": 20})}
    )
    custom = settings.model_copy(
        update={"regime_adjustments": cfg.model_copy(update={"bajista": bajista})}
    )
    r = apply_regime_adjustments("long", 25, regime("bajista"), None, agresivo, custom)
    assert r.min_abs_score == 30  # nunca baja del umbral de dirección
    assert r.score_ok is False
    assert REGIME_MIN_SCORE not in codes(r)


# --- Neutral: score, R:R y riesgo -------------------------------------------------------


@pytest.mark.parametrize(("side", "score"), [("long", 45), ("short", -45)])
def test_neutral_is_symmetric(settings, agresivo, side, score):
    r = apply_regime_adjustments(side, score, regime("neutral"), None, agresivo, settings)
    assert r.ok
    assert r.min_abs_score == 45
    assert r.min_rr == 2.0
    assert r.min_rr_reason == "el régimen Neutral"
    assert r.risk_multiplier == pytest.approx(0.75)
    assert r.risk_usd == pytest.approx(37.5)
    assert codes(r) == [REGIME_MIN_SCORE, REGIME_MIN_RR, REGIME_RISK]
    assert "R:R mínimo 2: régimen Neutral (tu perfil pide 1,5)." in [
        a.message for a in r.adjustments
    ]
    assert "Riesgo × 0,75: régimen Neutral." in [a.message for a in r.adjustments]


def test_effective_min_rr_keeps_the_profile_when_higher(settings):
    conservador = get_profile("conservador", settings)  # min_rr 2.0, igual al de Neutral
    r = apply_regime_adjustments("long", 50, regime("neutral"), None, conservador, settings)
    assert r.min_rr == 2.0
    assert r.min_rr_reason == "el perfil conservador"
    assert REGIME_MIN_RR not in codes(r)


# --- Euforia: multiplicador condicionado al funding --------------------------------------


def test_euforia_long_with_very_high_funding_halves_risk(settings, agresivo):
    r = apply_regime_adjustments("long", 60, regime("euforia"), 0.12, agresivo, settings)
    assert r.risk_multiplier == pytest.approx(0.5)
    assert r.adjustments[0].code == REGIME_RISK
    assert r.adjustments[0].message == (
        "Riesgo × 0,5: régimen Euforia (subida exagerada) con funding muy alto "
        "(0,12 %; muy alto es desde 0,1 %)."
    )


def test_euforia_funding_exactly_at_threshold_counts(settings, agresivo):
    limit = settings.scoring.derivatives.funding_very_high_pct
    r = apply_regime_adjustments("long", 60, regime("euforia"), limit, agresivo, settings)
    assert r.risk_multiplier == pytest.approx(0.5)


def test_euforia_long_with_normal_funding_keeps_risk(settings, agresivo):
    r = apply_regime_adjustments("long", 60, regime("euforia"), 0.03, agresivo, settings)
    assert r.risk_multiplier == 1.0
    assert codes(r) == [REGIME_FUNDING_NORMAL]
    assert "no está muy alto" in r.adjustments[0].message


@pytest.mark.parametrize("funding", [None, float("nan")])
def test_euforia_long_without_funding_does_not_apply_and_says_so(settings, agresivo, funding):
    r = apply_regime_adjustments("long", 60, regime("euforia"), funding, agresivo, settings)
    assert r.risk_multiplier == 1.0
    assert codes(r) == [REGIME_FUNDING_UNKNOWN]
    assert r.adjustments[0].risk_multiplier == 1.0
    assert "no hay dato de funding" in r.adjustments[0].message


# --- confianza baja e inestabilidad -------------------------------------------------------


def test_low_confidence_halves_risk(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 40, regime("alcista", confidence=0.59), None, agresivo, settings
    )
    assert r.risk_multiplier == pytest.approx(0.5)
    assert r.adjustments == (
        Adjustment(
            LOW_CONFIDENCE, "Riesgo × 0,5: confianza del régimen baja (59 %, menos de 60 %).", 0.5
        ),
    )


def test_confidence_at_threshold_is_not_low(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 40, regime("alcista", confidence=0.60), None, agresivo, settings
    )
    assert r.risk_multiplier == 1.0


def test_unknown_confidence_is_treated_as_low(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 40, regime("alcista", confidence=float("nan")), None, agresivo, settings
    )
    assert codes(r) == [LOW_CONFIDENCE]
    assert "no se sabe la confianza" in r.adjustments[0].message


def test_confidence_out_of_range_is_an_error(settings, agresivo):
    with pytest.raises(ValueError, match="entre 0 y 1"):
        apply_regime_adjustments("long", 40, regime("alcista", 55), None, agresivo, settings)


def test_unstable_halves_risk(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 40, regime("alcista", unstable=True), None, agresivo, settings
    )
    assert r.risk_multiplier == pytest.approx(0.5)
    assert codes(r) == [UNSTABLE]
    assert r.adjustments[0].message.startswith("Riesgo × 0,5: el régimen está inestable")


# --- los multiplicadores se acumulan y el piso de riesgo ---------------------------------


def test_multipliers_accumulate_and_floor_blocks(settings, agresivo):
    # Neutral 0,75 × inestable 0,5 × confianza baja 0,5 = 0,1875 -> 50 × 0,1875 = 9,38 USD.
    r = apply_regime_adjustments(
        "long", 60, regime("neutral", confidence=0.5, unstable=True), None, agresivo, settings
    )
    assert r.risk_multiplier == pytest.approx(0.75 * 0.5 * 0.5)
    assert r.risk_usd == pytest.approx(9.375)
    assert codes(r) == [REGIME_MIN_SCORE, REGIME_MIN_RR, REGIME_RISK, LOW_CONFIDENCE, UNSTABLE]
    assert r.risk_below_min is True
    assert r.blocked is True and r.ok is False
    assert r.block_code == MIN_OPERABLE_RISK
    assert r.block_reason == (
        "No operar: con los ajustes el riesgo queda en 9,38 USD (50 USD × 0,1875), "
        "menos que el mínimo operable de 15 USD."
    )


def test_accumulated_multipliers_above_floor_are_fine(settings, agresivo):
    # Neutral 0,75 × inestable 0,5 = 0,375 -> 18,75 USD >= 15 USD.
    r = apply_regime_adjustments(
        "short", -50, regime("neutral", unstable=True), None, agresivo, settings
    )
    assert r.risk_usd == pytest.approx(18.75)
    assert r.ok and not r.risk_below_min


def test_base_risk_can_come_already_reduced(settings, agresivo):
    # Por ejemplo, un circuit breaker ya dejó el riesgo en 25 USD.
    r = apply_regime_adjustments(
        "long", 40, regime("alcista", unstable=True), None, agresivo, settings, base_risk_usd=25
    )
    assert r.base_risk_usd == 25
    assert r.risk_usd == pytest.approx(12.5)
    assert r.blocked is True and r.block_code == MIN_OPERABLE_RISK


def test_crash_long_plus_floor_lists_both_reasons(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 60, regime("crash", unstable=True), None, agresivo, settings, base_risk_usd=20
    )
    assert r.block_code == REGIME_BLOCKED
    assert len(r.block_reasons) == 2
    assert "bloqueados" in r.block_reasons[0]
    assert "mínimo operable" in r.block_reasons[1]


# --- régimen no disponible ----------------------------------------------------------------


@pytest.mark.parametrize("value", [None, regime(None), RegimeUnavailable("sin historia")])
def test_unavailable_regime_is_conservative(settings, agresivo, value):
    r = apply_regime_adjustments("long", 35, value, 0.2, agresivo, settings)
    assert r.regime_available is False
    assert r.category is None
    assert r.ok
    assert r.min_abs_score == 30 and r.min_rr == agresivo.min_rr
    assert r.risk_multiplier == pytest.approx(
        settings.regime_adjustments.low_confidence.risk_multiplier
    )
    assert codes(r) == [REGIME_UNAVAILABLE]
    assert "no está disponible" in r.adjustments[0].message


def test_unknown_category_is_an_error(settings, agresivo):
    with pytest.raises(ValueError, match="desconocido"):
        apply_regime_adjustments("long", 40, regime("lunar"), None, agresivo, settings)


@pytest.mark.parametrize(
    ("direction", "score"), [("up", 40), ("long", float("nan")), ("long", float("inf"))]
)
def test_invalid_inputs(settings, agresivo, direction, score):
    with pytest.raises(ValueError):
        apply_regime_adjustments(direction, score, regime("alcista"), None, agresivo, settings)


def test_to_dict_is_json_safe(settings, agresivo):
    r = apply_regime_adjustments(
        "long", 40, regime("neutral", confidence=0.4, unstable=True), None, agresivo, settings
    )
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["ok"] is False
    assert d["score_check"]["ok"] is False
    assert len(d["no_trade_reasons"]) == 2  # piso de riesgo + score corto
    assert [a["code"] for a in d["adjustments"]] == codes(r)
    assert math.isclose(d["risk_multiplier"], 0.1875)
