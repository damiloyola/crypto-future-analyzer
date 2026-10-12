"""Score de confluencia: pesos de la config, total, faltantes y armado desde el análisis."""

import asyncio
import dataclasses
import json
import math

import pytest

from app.config import AppSettings, DirectionThresholds, ScoringWeights
from app.data.snapshot import build_offline_snapshot
from app.indicators.pipeline import analyze_all
from app.risk.rules import check_direction, direction_from_score
from app.scoring import ConfluenceScore, ScoreInputs, compute_score, inputs_from_analysis
from app.scoring.confluence import COMPONENT_LABELS, direction_from_total, reliable_trend
from tests.fakes import FIXED_NOW

KEYS = ("trend_4h", "trend_1h", "trend_15m", "momentum", "derivatives", "sentiment")

ALL_BULLISH = ScoreInputs(
    trends={"4h": "alcista", "1h": "alcista", "15m": "alcista"},
    rsi={"15m": 70.0, "1h": 70.0},
    macd={"15m": 2.0, "1h": 2.0},
    macd_signal={"15m": 1.0, "1h": 1.0},
    funding_rate=-0.0005,  # -0,05 % cada 8 h: shorts sobrecargados
    funding_interval_hours=8,
    oi_change_24h_pct=5.0,
    long_short_ratio=0.5,
    fear_greed=10,  # miedo extremo (contrarian: suma)
)
ALL_BEARISH = ScoreInputs(
    trends={"4h": "bajista", "1h": "bajista", "15m": "bajista"},
    rsi={"15m": 30.0, "1h": 30.0},
    macd={"15m": -2.0, "1h": -2.0},
    macd_signal={"15m": -1.0, "1h": -1.0},
    funding_rate=0.0015,  # 0,15 % cada 8 h: muy alto
    funding_interval_hours=8,
    oi_change_24h_pct=5.0,
    long_short_ratio=3.0,
    fear_greed=90,  # codicia extrema (contrarian: resta)
)


def _with_scoring(settings: AppSettings, **update) -> AppSettings:
    return settings.model_copy(update={"scoring": settings.scoring.model_copy(update=update)})


def _with_sentiment(settings: AppSettings, **update) -> AppSettings:
    sentiment = settings.scoring.sentiment.model_copy(update=update)
    return _with_scoring(settings, sentiment=sentiment)


def _assert_consistent(score: ConfluenceScore) -> None:
    for c in score.components:
        assert c.points == pytest.approx(100.0 * c.weight * c.value)
        assert -1.0 <= c.value <= 1.0
    assert score.total == pytest.approx(sum(c.points for c in score.components))
    assert -100.0 <= score.total <= 100.0


# --- total y pesos ---------------------------------------------------------------


def test_all_bullish_is_plus_100_and_all_bearish_minus_100(settings):
    up = compute_score(ALL_BULLISH, settings)
    down = compute_score(ALL_BEARISH, settings)
    assert up.total == pytest.approx(100.0)
    assert down.total == pytest.approx(-100.0)
    assert up.direction_hint == "long" and down.direction_hint == "short"
    assert up.missing == () and up.warnings == ()
    assert up.available_weight == pytest.approx(1.0)
    _assert_consistent(up)
    _assert_consistent(down)


def test_components_follow_config_order_with_default_points(settings):
    score = compute_score(ALL_BULLISH, settings)
    assert tuple(c.key for c in score.components) == KEYS
    assert [c.label for c in score.components] == [COMPONENT_LABELS[k] for k in KEYS]
    # Pesos por defecto: 20 %, 20 %, 15 %, 20 %, 15 %, 10 %.
    assert score.points_by_key() == pytest.approx(
        {
            "trend_4h": 20.0,
            "trend_1h": 20.0,
            "trend_15m": 15.0,
            "momentum": 20.0,
            "derivatives": 15.0,
            "sentiment": 10.0,
        }
    )
    assert score.component("derivatives").max_points == pytest.approx(15.0)
    with pytest.raises(KeyError):
        score.component("noticias")


def test_mixed_case_by_hand(settings):
    inputs = ScoreInputs(
        trends={"4h": "alcista", "1h": "lateral", "15m": "bajista"},
        rsi={"15m": 60.0, "1h": 40.0},
        macd={"15m": 1.0, "1h": 1.0},
        macd_signal={"15m": 0.5, "1h": 2.0},
        funding_rate=0.0001,  # 0,01 % cada 8 h: normal
        funding_interval_hours=8,
        oi_change_24h_pct=5.0,  # OI subiendo, pero con funding normal no cuenta
        long_short_ratio=1.0,
        fear_greed=37.5,  # contrarian: +0,5
    )
    score = compute_score(inputs, settings)
    pts = score.points_by_key()
    assert pts["trend_4h"] == pytest.approx(20.0)
    assert pts["trend_1h"] == pytest.approx(0.0)
    assert pts["trend_15m"] == pytest.approx(-15.0)
    assert pts["momentum"] == pytest.approx(5.0)  # (0,5 + 1 - 0,5 + 0) / 4 × 20
    assert pts["derivatives"] == pytest.approx(0.0)
    assert pts["sentiment"] == pytest.approx(5.0)
    assert score.total == pytest.approx(15.0)
    assert score.direction_hint is None
    assert score.summary == "Score +15: no llega a long (+30 o más) ni a short (-30 o menos)."
    _assert_consistent(score)


def test_weights_come_from_the_config(settings):
    weights = ScoringWeights(
        trend_4h=0.5, trend_1h=0.1, trend_15m=0.1, momentum=0.1, derivatives=0.1, sentiment=0.1
    )
    custom = _with_scoring(settings, weights=weights)
    inputs = ScoreInputs(trends={"4h": "alcista"})
    default = compute_score(inputs, settings)
    changed = compute_score(inputs, custom)
    assert default.component("trend_4h").points == pytest.approx(20.0)
    assert changed.component("trend_4h").points == pytest.approx(50.0)
    assert changed.component("trend_4h").weight == 0.5
    assert changed.total == pytest.approx(50.0)
    assert changed.direction_hint == "long"
    assert compute_score(ALL_BULLISH, custom).total == pytest.approx(100.0)


def test_direction_thresholds_come_from_the_config(settings):
    custom = _with_scoring(settings, direction_thresholds=DirectionThresholds(long=50, short=-50))
    inputs = ScoreInputs(trends={"4h": "alcista", "1h": "alcista"})  # +40
    assert compute_score(inputs, settings).direction_hint == "long"
    score = compute_score(inputs, custom)
    assert score.direction_hint is None
    assert (score.long_threshold, score.short_threshold) == (50, -50)


# --- pista de dirección ----------------------------------------------------------


def test_direction_hint_at_exactly_plus_and_minus_30(settings):
    # Tendencia 4h (20) + Fear & Greed en miedo extremo (10) = +30 justo.
    long_ = compute_score(ScoreInputs(trends={"4h": "alcista"}, fear_greed=25), settings)
    assert long_.total == pytest.approx(30.0)
    assert long_.direction_hint == "long"
    assert long_.summary == "Score +30: candidato a long (desde +30)."
    short = compute_score(ScoreInputs(trends={"4h": "bajista"}, fear_greed=75), settings)
    assert short.total == pytest.approx(-30.0)
    assert short.direction_hint == "short"
    assert short.summary == "Score -30: candidato a short (desde -30)."
    # Un poco menos no alcanza: 20 + 9,6.
    almost = compute_score(ScoreInputs(trends={"4h": "alcista"}, fear_greed=26), settings)
    assert almost.total == pytest.approx(29.6)
    assert almost.direction_hint is None


def test_direction_from_total_edges():
    assert direction_from_total(30.0, 30, -30) == "long"
    # Comparación exacta (igual que el score mínimo de las reglas): el ruido de
    # float ya lo limpia compute_score, que redondea el total a 9 decimales.
    assert direction_from_total(30.0 - 1e-12, 30, -30) is None
    assert direction_from_total(29.99, 30, -30) is None
    assert direction_from_total(-30.0, 30, -30) == "short"
    assert direction_from_total(-29.99, 30, -30) is None
    assert direction_from_total(0.0, 30, -30) is None


# --- datos faltantes -------------------------------------------------------------


def test_no_data_at_all_gives_zero_and_lists_everything(settings):
    score = compute_score(ScoreInputs(), settings)
    assert score.total == 0.0
    assert score.missing == KEYS
    assert all(not c.available and c.points == 0.0 for c in score.components)
    assert score.direction_hint is None
    assert score.available_weight == 0.0
    assert len(score.warnings) == len(KEYS)


def test_missing_data_does_not_renormalize_the_weights(settings):
    # Solo hay tendencia de 4h: suma sus 20 puntos, no 100.
    score = compute_score(ScoreInputs(trends={"4h": "alcista"}), settings)
    assert score.total == pytest.approx(20.0)
    assert score.missing == KEYS[1:]
    assert score.available_weight == pytest.approx(0.2)


def test_derivatives_down_counts_zero(settings):
    inputs = dataclasses.replace(
        ALL_BULLISH, funding_rate=None, oi_change_24h_pct=None, long_short_ratio=None
    )
    score = compute_score(inputs, settings)
    assert score.total == pytest.approx(85.0)
    assert score.missing == ("derivatives",)
    deriv = score.component("derivatives")
    assert (deriv.available, deriv.value, deriv.points) == (False, 0.0, 0.0)
    assert score.warnings == (
        "Sin datos de Derivados: cuenta 0 y su peso (15 %) no se reparte entre los demás.",
    )


def test_partial_component_is_warned_but_available(settings):
    inputs = dataclasses.replace(ALL_BULLISH, rsi={"15m": 70.0})  # sin RSI 1h
    score = compute_score(inputs, settings)
    mom = score.component("momentum")
    assert mom.available and mom.missing_parts == ("RSI 1h",)
    assert mom.points == pytest.approx(20.0)  # promedio de los 3 que hay (todos +1)
    assert score.missing == ()
    assert score.warnings == ("Momentum: sin RSI 1h; se usa lo que hay.",)


def test_timeframe_without_trend_and_nan_values_are_missing(settings):
    inputs = ScoreInputs(
        trends={"4h": "alcista", "1h": None},
        rsi={"15m": math.nan, "1h": math.inf},
        fear_greed=math.nan,
    )
    score = compute_score(inputs, settings)
    assert "trend_1h" in score.missing and "trend_15m" in score.missing
    assert "momentum" in score.missing and "sentiment" in score.missing
    assert score.total == pytest.approx(20.0)


def test_invalid_trend_label_raises(settings):
    with pytest.raises(ValueError, match="Tendencia desconocida"):
        compute_score(ScoreInputs(trends={"4h": "up"}), settings)


# --- sentimiento: modo y noticias ------------------------------------------------


def test_contrarian_vs_trend_mode(settings):
    inputs = ScoreInputs(fear_greed=10)
    contrarian = compute_score(inputs, settings)
    trend = compute_score(inputs, _with_sentiment(settings, fear_greed_mode="trend"))
    assert contrarian.component("sentiment").points == pytest.approx(10.0)
    assert trend.component("sentiment").points == pytest.approx(-10.0)


def test_news_none_vs_given(settings):
    # Config por defecto (decisión del usuario): las noticias no mueven el score.
    base = compute_score(ScoreInputs(fear_greed=50, news_balance=1.0), settings)
    assert base.component("sentiment").points == 0.0
    with_news = _with_sentiment(settings, news_share=0.5)
    none = compute_score(ScoreInputs(fear_greed=50), with_news)
    given = compute_score(ScoreInputs(fear_greed=50, news_balance=1.0), with_news)
    assert none.component("sentiment").points == 0.0
    assert any("Sin balance de noticias" in d for d in none.component("sentiment").details)
    assert given.component("sentiment").points == pytest.approx(5.0)  # 10 × 0,5 × 1


# --- JSON --------------------------------------------------------------------------


def test_to_dict_is_json_serializable(settings):
    inputs = dataclasses.replace(ALL_BULLISH, rsi={"15m": math.nan, "1h": 70.0})
    score = compute_score(inputs, settings)
    d = score.to_dict()
    text = json.dumps(d, allow_nan=False, ensure_ascii=False)
    assert json.loads(text)["total"] == pytest.approx(score.total)
    assert d["direction_hint"] == "long"
    assert d["thresholds"] == {"long": 30, "short": -30}
    assert [c["key"] for c in d["components"]] == list(KEYS)
    first = d["components"][0]
    assert set(first) == {
        "key",
        "label",
        "weight",
        "weight_pct",
        "max_points",
        "value",
        "points",
        "available",
        "details",
        "missing_parts",
    }
    assert first["weight_pct"] == pytest.approx(20.0)
    inputs_d = inputs.to_dict()
    assert inputs_d["rsi"]["15m"] is None
    json.dumps(inputs_d, allow_nan=False)


# --- desde el análisis y la foto del mercado ----------------------------------------


@pytest.fixture(scope="module")
def offline(settings):
    snap = asyncio.run(build_offline_snapshot("BTC", settings, now=FIXED_NOW))
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    return snap, analysis


def test_inputs_from_analysis_reads_the_real_fields(settings, offline):
    snap, analysis = offline
    inputs = inputs_from_analysis(analysis, snap, settings, news_balance=0.3)
    for tf in ("4h", "1h", "15m"):
        summary = analysis[tf].summary
        assert inputs.trends[tf] == summary.trend
        assert inputs.rsi[tf] == pytest.approx(summary.rsi)
        assert inputs.macd[tf] == pytest.approx(summary.macd)
        assert inputs.macd_signal[tf] == pytest.approx(summary.macd_signal)
    assert snap.funding is not None and snap.long_short is not None
    assert snap.fear_greed is not None
    assert inputs.funding_rate == pytest.approx(snap.funding.rate)
    assert inputs.funding_interval_hours == snap.funding.interval_hours
    assert inputs.oi_change_24h_pct == pytest.approx(snap.oi_change_24h_pct)
    assert inputs.long_short_ratio == pytest.approx(snap.long_short.points[-1].ratio)
    assert inputs.fear_greed == snap.fear_greed.value
    assert inputs.news_balance == 0.3

    score = compute_score(inputs, settings)
    assert score.missing == ()
    _assert_consistent(score)
    json.dumps(score.to_dict(), allow_nan=False)


def test_inputs_from_analysis_with_missing_derivatives_and_sentiment(settings, offline):
    snap, analysis = offline
    bare = dataclasses.replace(
        snap, funding=None, long_short=None, fear_greed=None, oi_change_24h_pct=None
    )
    inputs = inputs_from_analysis(analysis, bare, settings)
    assert inputs.funding_rate is None and inputs.long_short_ratio is None
    assert inputs.fear_greed is None and inputs.oi_change_24h_pct is None
    score = compute_score(inputs, settings)
    assert score.missing == ("derivatives", "sentiment")


def test_inputs_from_analysis_funding_interval(settings, offline):
    snap, analysis = offline
    assert snap.funding is not None
    four = dataclasses.replace(snap, funding=dataclasses.replace(snap.funding, interval_hours=4))
    assert inputs_from_analysis(analysis, four, settings).funding_interval_hours == 4
    broken = dataclasses.replace(snap, funding=dataclasses.replace(snap.funding, interval_hours=0))
    got = inputs_from_analysis(analysis, broken, settings).funding_interval_hours
    assert got == settings.fees.funding_interval_hours


def test_inputs_from_analysis_trend_without_long_ema_is_missing(settings, offline):
    snap, analysis = offline
    ta = analysis["4h"]
    _, long_ = ta.summary.trend_emas
    summary = dataclasses.replace(ta.summary, ema={**ta.summary.ema, long_: None}, trend="lateral")
    timeframes = {**analysis.timeframes, "4h": dataclasses.replace(ta, summary=summary)}
    patched = dataclasses.replace(analysis, timeframes=timeframes)
    inputs = inputs_from_analysis(patched, snap, settings)
    assert inputs.trends["4h"] is None
    assert "trend_4h" in compute_score(inputs, settings).missing
    # Un timeframe que ni se analizó también falta.
    without_1h = dataclasses.replace(
        analysis, timeframes={k: v for k, v in analysis.timeframes.items() if k != "1h"}
    )
    assert (
        "trend_1h"
        in compute_score(inputs_from_analysis(without_1h, snap, settings), settings).missing
    )


def test_to_dict_has_no_negative_zero(settings):
    # RSI en 0 (sobreventa recortada a 0) y Fear & Greed neutral en modo tendencia.
    trend_mode = _with_sentiment(settings, fear_greed_mode="trend")
    score = compute_score(ScoreInputs(rsi={"15m": 0.0}, fear_greed=50), trend_mode)
    text = json.dumps(score.to_dict(), allow_nan=False)
    assert "-0.0" not in text
    assert score.total == 0.0


# --- revisión final del Hito 4 ------------------------------------------------------


def _almost_long(rsi_15m: float) -> ScoreInputs:
    # 4h y 1h alcistas (+40), 15m lateral (0), RSI 15m (rsi - 50 puntos), F&G neutral.
    return ScoreInputs(
        trends={"4h": "alcista", "1h": "alcista", "15m": "lateral"},
        rsi={"15m": rsi_15m},
        fear_greed=50,
    )


def test_summary_does_not_round_the_score_to_the_threshold(settings):
    # RSI 15m 39,97 (dato continuo, pasa con datos reales): total 29,97.
    score = compute_score(_almost_long(39.97), settings)
    assert score.total == pytest.approx(29.97)
    assert score.direction_hint is None
    assert score.summary.startswith("Score +29,97: no llega a long (+30 o más)")
    rule = check_direction(score.total, settings.scoring.direction_thresholds)
    assert "el score (+29,97) está entre -30 y +30" in rule.message


def test_score_hint_matches_the_plan_direction_just_below_the_threshold(settings):
    # 29,999999999 (a 1e-9 del umbral): antes la pista decía long y el plan no.
    score = compute_score(_almost_long(39.999999999), settings)
    assert score.total == pytest.approx(30.0 - 1e-9, abs=1e-12)
    th = settings.scoring.direction_thresholds
    assert score.direction_hint is None
    assert direction_from_score(score.total, th) is None
    assert score.direction_hint == direction_from_score(score.total, th)
    assert "candidato" not in score.summary
    for total in (30.0, -30.0, 29.99, 0.0, -45.0):
        hint = direction_from_total(total, th.long, th.short)
        assert hint == direction_from_score(total, th)


def test_reliable_trend_needs_candles_and_emas():
    def summary(**kw):
        base = {"candles_used": 50, "trend_emas": (50, 200), "ema": {50: 1.0, 200: 2.0}}
        base.update(kw)
        return type("S", (), {"trend": "bajista", **base})()

    assert reliable_trend(summary()) == "bajista"
    assert reliable_trend(None) is None
    assert reliable_trend(summary(candles_used=0)) is None
    assert reliable_trend(summary(trend_emas=None)) is None
    assert reliable_trend(summary(ema={50: 1.0, 200: math.nan})) is None
