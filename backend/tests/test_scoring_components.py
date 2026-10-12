"""Componentes del score de confluencia: cada fórmula con valores a mano y bordes."""

import math

import pytest

from app.config import (
    DerivativesScoringConfig,
    MomentumScoringConfig,
    SentimentScoringConfig,
)
from app.scoring.components import (
    clean_number,
    derivatives_component,
    fear_greed_label,
    fear_greed_part,
    fmt_num,
    funding_part,
    funding_pct_per_8h,
    long_short_part,
    macd_score,
    momentum_component,
    open_interest_part,
    rsi_score,
    sentiment_component,
    trend_component,
    trend_value,
)

# Configs fijas (los números por defecto de config.yaml): las fórmulas no dependen
# de lo que tenga el archivo.
MOM = MomentumScoringConfig(
    timeframes=["15m", "1h"], rsi_neutral=50, rsi_overbought=70, rsi_oversold=30
)
DER = DerivativesScoringConfig(
    funding_high_pct=0.05,
    funding_very_high_pct=0.10,
    funding_low_pct=-0.03,
    oi_lookback_hours=24,
    oi_rising_pct=3.0,
    long_short_crowded_long=2.0,
    long_short_crowded_short=0.7,
)
SEN = SentimentScoringConfig(
    fear_greed_extreme_fear=25,
    fear_greed_extreme_greed=75,
    fear_greed_mode="contrarian",
    news_share=0.0,
)


# --- utilidades ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "decimals", "signed", "expected"),
    [
        (0.5, 2, False, "0,5"),
        (1.0, 2, True, "+1"),
        (-0.5, 2, True, "-0,5"),
        (0.0, 2, True, "0"),
        (-0.0001, 2, True, "0"),
        (62.75, 1, False, "62,8"),
        (0.06, 4, False, "0,06"),
        (1234.5, 1, False, "1234,5"),
        (30.0, 1, True, "+30"),
    ],
)
def test_fmt_num_uses_comma_and_no_trailing_zeros(x, decimals, signed, expected):
    assert fmt_num(x, decimals, signed=signed) == expected


def test_clean_number():
    assert clean_number(1) == 1.0
    assert clean_number("2.5") == 2.5
    for bad in (None, math.nan, math.inf, -math.inf, "x", True):
        assert clean_number(bad) is None


# --- tendencia -------------------------------------------------------------------


def test_trend_values():
    assert trend_value("alcista") == 1.0
    assert trend_value("bajista") == -1.0
    assert trend_value("lateral") == 0.0
    assert trend_value(None) is None
    with pytest.raises(ValueError, match="Tendencia desconocida"):
        trend_value("subiendo")


def test_trend_component_texts_and_missing():
    up = trend_component("alcista", "4h")
    assert (up.value, up.available) == (1.0, True)
    assert up.details == ("Tendencia alcista en 4h: suma (+1).",)
    down = trend_component("bajista", "1h")
    assert down.value == -1.0 and "resta (-1)" in down.details[0]
    flat = trend_component("lateral", "15m")
    assert flat.value == 0.0 and flat.available and "ni suma ni resta" in flat.details[0]
    missing = trend_component(None, "4h")
    assert (missing.value, missing.available) == (0.0, False)
    assert missing.missing_parts == ("tendencia 4h",)
    assert "cuenta 0" in missing.details[0]


# --- momentum --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [
        (50.0, 0.0),  # neutral
        (70.0, 1.0),  # justo en sobrecompra: aporte alcista completo
        (30.0, -1.0),  # justo en sobreventa: aporte bajista completo
        (60.0, 0.5),
        (40.0, -0.5),
        (85.0, 0.5),  # sobrecompra: se recorta (100 - 85) / (100 - 70)
        (100.0, 0.0),
        (15.0, -0.5),  # sobreventa: se recorta -15 / 30
        (0.0, 0.0),
        (120.0, 0.0),  # fuera de escala: se acota a 100
        (-5.0, 0.0),
    ],
)
def test_rsi_score_default_thresholds(rsi, expected):
    assert rsi_score(rsi, 50, 70, 30) == pytest.approx(expected)


def test_rsi_score_is_continuous_at_the_edges():
    assert rsi_score(70.0001, 50, 70, 30) == pytest.approx(1.0, abs=1e-4)
    assert rsi_score(29.9999, 50, 70, 30) == pytest.approx(-1.0, abs=1e-4)


def test_rsi_score_asymmetric_thresholds():
    # n = 50, ob = 80, os = 40
    assert rsi_score(65, 50, 80, 40) == pytest.approx(0.5)
    assert rsi_score(45, 50, 80, 40) == pytest.approx(-0.5)
    assert rsi_score(90, 50, 80, 40) == pytest.approx(0.5)  # (100 - 90) / (100 - 80)
    assert rsi_score(20, 50, 80, 40) == pytest.approx(-0.5)  # -20 / 40


def test_rsi_score_rejects_unordered_thresholds():
    with pytest.raises(ValueError):
        rsi_score(50, 50, 40, 30)


@pytest.mark.parametrize(
    ("macd", "signal", "expected"),
    [
        (1.0, 0.5, 1.0),  # arriba de la señal y arriba de 0
        (1.0, 2.0, 0.0),  # debajo de la señal pero arriba de 0
        (-1.0, -2.0, 0.0),  # arriba de la señal pero debajo de 0
        (-1.0, 0.0, -1.0),
        (1.0, 1.0, 0.5),  # igual a la señal
        (-1.0, -1.0, -0.5),
        (0.0, 0.0, 0.0),  # MACD en cero
        (0.0, -1.0, 0.5),
        (0.0, 1.0, -0.5),
    ],
)
def test_macd_score(macd, signal, expected):
    assert macd_score(macd, signal) == pytest.approx(expected)


def test_momentum_is_the_mean_of_rsi_and_macd_in_both_timeframes():
    r = momentum_component(
        {"15m": 60.0, "1h": 40.0},
        {"15m": 1.0, "1h": 1.0},
        {"15m": 0.5, "1h": 2.0},
        MOM,
    )
    # RSI 15m +0,5, MACD 15m +1, RSI 1h -0,5, MACD 1h 0 → 1 / 4
    assert r.value == pytest.approx(0.25)
    assert r.available and r.missing_parts == ()
    assert r.details[0] == "RSI 15m en 60: arriba de 50, a favor de la suba (+0,5)."
    assert r.details[1] == "MACD 15m arriba de su señal y arriba de 0 (+1)."
    assert "debajo de 50, a favor de la baja (-0,5)" in r.details[2]


def test_momentum_texts_for_overbought_and_oversold():
    r = momentum_component({"15m": 85.0, "1h": 15.0}, {}, {}, MOM)
    assert "sobrecompra, el aporte alcista se recorta (+0,5)" in r.details[0]
    assert "sobreventa, el aporte bajista se recorta (-0,5)" in r.details[1]


def test_momentum_averages_only_the_available_parts():
    r = momentum_component(
        {"15m": 70.0, "1h": None},
        {"15m": 1.0, "1h": 1.0},
        {"15m": 0.5, "1h": math.nan},  # sin señal en 1h: ese MACD no cuenta
        MOM,
    )
    assert r.value == pytest.approx(1.0)  # (1 + 1) / 2
    assert r.available
    assert r.missing_parts == ("RSI 1h", "MACD 1h")
    assert r.details[-1] == "Faltan RSI 1h, MACD 1h: se promedia con lo que hay."


def test_momentum_without_data_is_unavailable():
    r = momentum_component({}, {}, {}, MOM)
    assert (r.value, r.available) == (0.0, False)
    assert r.missing_parts == ("RSI 15m", "MACD 15m", "RSI 1h", "MACD 1h")
    assert "Sin RSI ni MACD en 15m y 1h" in r.details[0]


def test_momentum_uses_the_configured_timeframes():
    cfg = MOM.model_copy(update={"timeframes": ["4h"]})
    r = momentum_component({"4h": 30.0, "15m": 70.0}, {}, {}, cfg)
    assert r.value == pytest.approx(-1.0)  # solo el RSI de 4h
    assert r.missing_parts == ("MACD 4h",)


# --- derivados -------------------------------------------------------------------


def test_funding_normalized_to_8h():
    assert funding_pct_per_8h(0.0001, 8) == pytest.approx(0.01)
    assert funding_pct_per_8h(0.0003, 4) == pytest.approx(0.06)  # 0,03 % cada 4 h
    assert funding_pct_per_8h(0.0001, 1) == pytest.approx(0.08)
    with pytest.raises(ValueError):
        funding_pct_per_8h(0.0001, 0)


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (0.10, -1.0),  # justo en "muy alto"
        (0.25, -1.0),
        (0.0999, -0.5),
        (0.05, -0.5),  # justo en "alto"
        (0.0499, 0.0),
        (0.01, 0.0),
        (0.0, 0.0),
        (-0.0299, 0.0),
        (-0.03, 0.5),  # justo en "negativo"
        (-0.2, 0.5),
    ],
)
def test_funding_part_thresholds(f, expected):
    assert funding_part(f, DER) == expected


@pytest.mark.parametrize(
    ("oi", "f", "expected"),
    [
        (3.0, 0.05, -0.5),  # OI subiendo justo en el umbral + funding justo en "alto"
        (10.0, 0.2, -0.5),  # funding muy alto: también -0,5 (el funding ya restó aparte)
        (3.0, 0.0499, 0.0),  # funding positivo pero normal: la suba del OI no cuenta
        (3.0, 0.01, 0.0),  # el funding normal de siempre (+0,01 %): no cuenta
        (3.0, 0.0, 0.0),
        (3.0, -0.01, 0.0),  # negativo pero sin llegar a "shorts sobrecargados"
        (3.0, -0.0299, 0.0),
        (3.0, -0.03, 0.5),  # justo en funding_low_pct: shorts sobrecargados
        (10.0, -0.2, 0.5),
        (2.99, 0.2, 0.0),  # no llega a "subiendo"
        (2.99, -0.2, 0.0),
        (-5.0, -0.2, 0.0),
    ],
)
def test_open_interest_part(oi, f, expected):
    assert open_interest_part(oi, f, DER) == expected


def test_open_interest_part_thresholds_come_from_the_config():
    custom = DER.model_copy(update={"oi_rising_pct": 10.0, "funding_high_pct": 0.02})
    assert open_interest_part(5.0, 0.03, DER) == 0.0  # 0,03 % es normal con la config
    assert open_interest_part(5.0, 0.03, custom) == 0.0  # +5 % ya no es "OI subiendo"
    assert open_interest_part(10.0, 0.03, custom) == -0.5  # 0,03 % ya es "alto"


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(2.0, -0.5), (3.5, -0.5), (1.99, 0.0), (1.0, 0.0), (0.71, 0.0), (0.7, 0.5), (0.3, 0.5)],
)
def test_long_short_part(ratio, expected):
    assert long_short_part(ratio, DER) == expected


def _der(**kw):
    args = {
        "funding_rate": None,
        "funding_interval_hours": None,
        "oi_change_pct": None,
        "long_short_ratio": None,
    }
    args.update(kw)
    return derivatives_component(**args, cfg=DER, default_interval_hours=8)


def test_derivatives_overloaded_longs_is_clipped_to_minus_one():
    r = _der(funding_rate=0.0012, funding_interval_hours=8, oi_change_pct=5.0, long_short_ratio=2.5)
    # -1 (funding 0,12 %) - 0,5 (OI subiendo) - 0,5 (exceso de longs) = -2 → -1
    assert r.value == -1.0
    assert r.available and r.missing_parts == ()
    assert r.details[0] == (
        "Funding 0,12 % cada 8 h: muy alto, demasiados longs pagando por mantenerse, "
        "riesgo de caída (-1)."
    )
    assert r.details[1] == (
        "Interés abierto +5 % en 24 h con funding alto: longs sobrecargados con el OI "
        "subiendo, riesgo de caída (-0,5)."
    )
    assert "demasiadas cuentas en long, riesgo de caída (-0,5)" in r.details[2]
    assert r.details[-1] == "Las partes suman -2; se recorta a -1."


def test_derivatives_oi_rising_with_negative_funding_adds():
    r = _der(
        funding_rate=-0.0004, funding_interval_hours=8, oi_change_pct=4.0, long_short_ratio=0.6
    )
    # +0,5 (funding -0,04 %) + 0,5 (shorts sobrecargados con OI subiendo) + 0,5 → +1
    assert r.value == 1.0
    assert r.details[1] == (
        "Interés abierto +4 % en 24 h con funding negativo: shorts sobrecargados con el OI "
        "subiendo, riesgo de suba (+0,5)."
    )


def test_derivatives_oi_rising_with_normal_funding_does_not_count():
    # Decisión del usuario: +0,01 % cada 8 h es el funding normal; con el OI
    # subiendo no indica que los longs estén sobrecargados.
    r = _der(funding_rate=0.0001, funding_interval_hours=8, oi_change_pct=5.0)
    assert r.value == 0.0
    assert r.available
    assert r.details[0] == "Funding 0,01 % cada 8 h: normal (0)."
    assert r.details[1] == (
        "Interés abierto +5 % en 24 h con funding normal: ningún lado está sobrecargado, así "
        "que la suba del OI no cuenta (0)."
    )


def test_derivatives_oi_rising_with_high_funding_subtracts_on_top_of_the_funding():
    r = _der(funding_rate=0.0005, funding_interval_hours=8, oi_change_pct=3.0)
    # -0,5 (funding 0,05 %, "alto") - 0,5 (longs sobrecargados con el OI subiendo) = -1
    assert r.value == -1.0
    assert "con funding alto: longs sobrecargados con el OI subiendo" in r.details[1]


def test_derivatives_oi_not_rising_says_it_does_not_count():
    r = _der(funding_rate=0.0012, funding_interval_hours=8, oi_change_pct=2.0)
    assert r.value == -1.0  # solo el funding muy alto
    assert r.details[1] == ("Interés abierto +2 % en 24 h: no sube 3 % o más, no indica nada (0).")


def test_derivatives_normalizes_a_4h_funding_interval():
    # 0,03 % cada 4 h = 0,06 % cada 8 h → "alto"; el mismo número cada 8 h es normal.
    r4 = _der(funding_rate=0.0003, funding_interval_hours=4)
    r8 = _der(funding_rate=0.0003, funding_interval_hours=8)
    assert r4.value == -0.5
    assert r8.value == 0.0
    assert r4.details[0].startswith("Funding 0,03 % cada 4 h (= 0,06 % cada 8 h): alto")


@pytest.mark.parametrize(
    ("rate", "interval", "expected"),
    [
        (0.0005, 8, -0.5),  # 0,05 % cada 8 h: justo "alto"
        (0.00025, 4, -0.5),  # 0,025 % cada 4 h = 0,05 % cada 8 h
        (0.001, 8, -1.0),  # 0,10 %: justo "muy alto"
        (0.0005, 4, -1.0),  # 0,05 % cada 4 h = 0,10 % cada 8 h
        (-0.0003, 8, 0.5),  # -0,03 %: justo "negativo"
        (-0.0000375, 1, 0.5),  # -0,00375 % cada 1 h = -0,03 % cada 8 h
    ],
)
def test_derivatives_funding_exactly_at_thresholds_after_normalizing(rate, interval, expected):
    assert _der(funding_rate=rate, funding_interval_hours=interval).value == expected


def test_derivatives_uses_the_fallback_interval_when_missing():
    r = derivatives_component(
        0.0003, None, None, None, DER, default_interval_hours=4
    )  # 0,06 % cada 8 h
    assert r.value == -0.5
    bad = derivatives_component(0.0003, 0, None, None, DER, default_interval_hours=4)
    assert bad.value == -0.5


def test_derivatives_partial_data_counts_zero_and_is_listed():
    r = _der(long_short_ratio=2.0)
    assert r.available and r.value == -0.5
    assert r.missing_parts == ("funding", "interés abierto")
    assert r.details[-1] == "Sin funding, interés abierto: esas partes cuentan 0."


def test_derivatives_oi_without_funding_cannot_be_read():
    r = _der(oi_change_pct=8.0)
    assert (r.value, r.available) == (0.0, False)
    assert "sin funding no se puede saber quién está sobrecargado" in r.details[0]


def test_derivatives_without_data_is_unavailable():
    r = _der(long_short_ratio=0.0)  # un ratio que no es positivo es un dato roto
    assert (r.value, r.available) == (0.0, False)
    assert r.missing_parts == ("funding", "interés abierto", "ratio long/short")


# --- sentimiento -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("g", "expected"),
    [
        (50, 0.0),
        (25, 1.0),  # miedo extremo: suma a pleno
        (10, 1.0),
        (0, 1.0),
        (37.5, 0.5),
        (75, -1.0),  # codicia extrema: resta a pleno
        (62.5, -0.5),
        (100, -1.0),
    ],
)
def test_fear_greed_contrarian(g, expected):
    assert fear_greed_part(g, SEN) == pytest.approx(expected)


def test_fear_greed_trend_mode_is_the_opposite():
    cfg = SEN.model_copy(update={"fear_greed_mode": "trend"})
    for g in (0, 25, 37.5, 50, 62.5, 75, 100):
        assert fear_greed_part(g, cfg) == pytest.approx(-fear_greed_part(g, SEN))


def test_fear_greed_rejects_extremes_on_the_wrong_side_of_50():
    with pytest.raises(ValueError, match="fear_greed_extreme_fear < 50"):
        fear_greed_part(40, SEN.model_copy(update={"fear_greed_extreme_fear": 50}))


def test_fear_greed_label():
    assert fear_greed_label(25, SEN) == "miedo extremo"
    assert fear_greed_label(40, SEN) == "miedo"
    assert fear_greed_label(50, SEN) == "neutral"
    assert fear_greed_label(60, SEN) == "codicia"
    assert fear_greed_label(75, SEN) == "codicia extrema"


def test_sentiment_with_news_share_zero_ignores_news():
    r = sentiment_component(20, 1.0, SEN)
    assert r.value == 1.0 and r.available
    assert r.details[0] == (
        "Fear & Greed en 20 (miedo extremo), modo contrarian: el miedo suma al long y la "
        "codicia resta (+1)."
    )
    assert r.details[1] == "Las noticias no mueven el score: las comenta la IA."


def test_sentiment_news_none_vs_given():
    cfg = SEN.model_copy(update={"news_share": 0.5})
    without = sentiment_component(50, None, cfg)
    assert without.value == 0.0 and without.available
    assert "Sin balance de noticias: no se cuentan" in without.details[1]
    given = sentiment_component(50, 1.0, cfg)
    assert given.value == pytest.approx(0.5)  # 0,5 × 0 + 0,5 × 1
    assert given.details[1] == "Balance de noticias +1 (pesa 50 % del sentimiento)."
    mixed = sentiment_component(37.5, -0.4, cfg)
    assert mixed.value == pytest.approx(0.5 * 0.5 + 0.5 * -0.4)


def test_sentiment_trend_mode_text():
    cfg = SEN.model_copy(update={"fear_greed_mode": "trend"})
    r = sentiment_component(80, None, cfg)
    assert r.value == 1.0
    assert "modo tendencia: la codicia suma al long y el miedo resta" in r.details[0]


def test_sentiment_without_fear_greed():
    assert sentiment_component(None, None, SEN).available is False
    # Con news_share 0 las noticias no alcanzan para tener el componente.
    assert sentiment_component(None, 0.8, SEN).available is False
    cfg = SEN.model_copy(update={"news_share": 0.5})
    r = sentiment_component(math.nan, 0.8, cfg)
    assert r.available and r.value == pytest.approx(0.4)  # F&G cuenta 0, no se reparte
    assert r.missing_parts == ("Fear & Greed",)


def test_sentiment_rejects_news_out_of_range():
    with pytest.raises(ValueError, match="news_balance"):
        sentiment_component(50, 1.5, SEN)
