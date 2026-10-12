"""Pipeline de indicadores: columnas, valores, velas cerradas, avisos y niveles."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from app.config import AppSettings
from app.data.synthetic import SyntheticMarketData
from app.indicators.levels import support_resistance
from app.indicators.momentum import macd, rsi
from app.indicators.moving_averages import ema, vwap_daily
from app.indicators.pipeline import (
    analyze_all,
    analyze_timeframe,
    compute_indicators,
    frame_columns,
    rsi_zone,
    trend_ema_periods,
)
from app.indicators.trend import above_below, classify_trend_series
from app.indicators.volatility import atr, bollinger, daily_range_pct
from app.indicators.volume import relative_volume
from tests.fakes import FIXED_NOW, make_candles, ohlcv_frame


@pytest.fixture(scope="module")
def market() -> SyntheticMarketData:
    return SyntheticMarketData(now=FIXED_NOW)


@pytest.fixture(scope="module")
def candles_15m(market: SyntheticMarketData):
    return make_candles(market.candles_frame("BTC", "15m").iloc[-500:].copy(), "15m")


def _with_timeframes(settings: AppSettings, **changes) -> AppSettings:
    return settings.model_copy(
        update={"timeframes": settings.timeframes.model_copy(update=changes)}
    )


# --- columnas y valores ---------------------------------------------------------


def test_frame_has_every_indicator_column(settings, candles_15m):
    ta = analyze_timeframe(candles_15m, settings, current_price=100_000.0)
    expected = [
        "ema_20",
        "ema_50",
        "ema_200",
        "vwap",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "rel_volume",
        "trend",
    ]
    for column in expected:
        assert column in ta.frame.columns, column
    assert list(ta.frame.columns[:6]) == ["open", "high", "low", "close", "volume", "closed"]
    assert frame_columns(settings, "15m") == list(ta.frame.columns[6:])
    assert len(ta.frame) == len(candles_15m.df)
    assert ta.frame.index.equals(candles_15m.df.index)


def test_vwap_only_for_configured_timeframes(settings, market):
    assert "4h" not in settings.indicators.vwap_timeframes
    c4h = make_candles(market.candles_frame("BTC", "4h").iloc[-300:].copy(), "4h")
    ta = analyze_timeframe(c4h, settings, 100_000.0)
    assert "vwap" not in ta.frame.columns
    assert ta.summary.vwap is None
    assert ta.summary.price_vs_vwap is None
    assert ta.summary.has_vwap is False
    assert not any("VWAP" in f for f in ta.summary.flags)


def test_columns_match_direct_indicator_calls(settings, candles_15m):
    ind = settings.indicators
    df = candles_15m.df
    frame = compute_indicators(df, "15m", settings)
    close = df["close"]

    def same(a: pd.Series, b: pd.Series) -> None:
        pd.testing.assert_series_equal(a, b, check_names=False)

    for p in ind.ema_periods:
        same(frame[f"ema_{p}"], ema(close, p))
    same(frame["vwap"], vwap_daily(df, settings.app.market_timezone))
    same(frame["rsi"], rsi(close, ind.rsi_period))
    m = macd(close, ind.macd.fast, ind.macd.slow, ind.macd.signal)
    same(frame["macd"], m["macd"])
    same(frame["macd_signal"], m["signal"])
    same(frame["macd_hist"], m["hist"])
    same(frame["atr"], atr(df, ind.atr_period))
    bb = bollinger(close, ind.bollinger.period, ind.bollinger.std)
    for col in ("mid", "upper", "lower", "width"):
        same(frame[f"bb_{col}"], bb[col])
    same(frame["rel_volume"], relative_volume(df["volume"], ind.relative_volume_period))
    same(frame["trend"], classify_trend_series(close, frame["ema_50"], frame["ema_200"]))


def test_compute_indicators_does_not_modify_input(settings, candles_15m):
    before = candles_15m.df.copy()
    compute_indicators(candles_15m.df, "15m", settings)
    pd.testing.assert_frame_equal(candles_15m.df, before)


def test_trend_uses_the_two_longest_emas(settings):
    assert trend_ema_periods(settings) == (50, 200)
    custom = settings.model_copy(
        update={"indicators": settings.indicators.model_copy(update={"ema_periods": [100, 9, 21]})}
    )
    assert trend_ema_periods(custom) == (21, 100)


# --- resumen y velas cerradas -----------------------------------------------------


def test_summary_reads_the_last_closed_candle(settings, candles_15m):
    df = candles_15m.df
    assert not bool(df["closed"].iloc[-1])  # la última se está formando
    price = 101_234.5
    ta = analyze_timeframe(candles_15m, settings, price)
    s = ta.summary
    last_closed = df[df["closed"]].index[-1]
    row = ta.frame.loc[last_closed]
    assert s.candle_time == last_closed.to_pydatetime()
    assert s.candle_closed is True
    assert s.close == pytest.approx(row["close"])
    assert s.ema[200] == pytest.approx(row["ema_200"])
    assert s.rsi == pytest.approx(row["rsi"])
    assert s.macd_hist == pytest.approx(row["macd_hist"])
    assert s.atr == pytest.approx(row["atr"])
    assert s.vwap == pytest.approx(row["vwap"])
    assert s.trend == row["trend"]
    assert s.candles_used == int(df["closed"].sum())
    assert s.current_price == price
    assert s.atr_pct == pytest.approx(row["atr"] / price * 100)
    assert s.bb_width == pytest.approx(row["bb_width"])
    assert s.flags == []


def test_forming_candle_does_not_change_the_summary(settings, candles_15m):
    price = 100_000.0
    base = analyze_timeframe(candles_15m, settings, price).summary
    wild = candles_15m.df.copy()
    last = wild.index[-1]
    wild.loc[last, ["open", "high", "low", "close", "volume"]] = [1.0, 10e6, 0.5, 9e6, 1e12]
    changed = analyze_timeframe(make_candles(wild, "15m"), settings, price).summary
    for field in ("close", "rsi", "macd", "macd_signal", "atr", "bb_upper", "rel_volume", "vwap"):
        assert getattr(changed, field) == getattr(base, field), field
    assert changed.ema == base.ema
    assert changed.trend == base.trend
    assert changed.levels == base.levels  # S/R solo con velas cerradas


def test_closed_candles_only_false_uses_the_forming_candle(settings, candles_15m):
    open_settings = _with_timeframes(settings, closed_candles_only=False)
    ta = analyze_timeframe(candles_15m, open_settings, 100_000.0)
    df = candles_15m.df
    assert ta.summary.candle_time == df.index[-1].to_pydatetime()
    assert ta.summary.candle_closed is False
    assert ta.summary.close == pytest.approx(df["close"].iloc[-1])
    assert ta.summary.candles_used == len(df)
    # Los niveles igual salen de velas cerradas.
    closed_levels = analyze_timeframe(candles_15m, settings, 100_000.0).summary.levels
    assert ta.summary.levels == closed_levels


def test_current_price_defaults_to_last_close(settings, candles_15m):
    ta = analyze_timeframe(candles_15m, settings)
    assert ta.summary.current_price == pytest.approx(candles_15m.df["close"].iloc[-1])


def test_rejects_non_positive_current_price(settings, candles_15m):
    with pytest.raises(ValueError, match="positivo"):
        analyze_timeframe(candles_15m, settings, -5.0)


def test_positions_compare_current_price_with_each_ema_and_vwap(settings, candles_15m):
    s = analyze_timeframe(candles_15m, settings, 100_000.0).summary
    for p, value in s.ema.items():
        assert s.price_vs_ema[p] == above_below(100_000.0, value)
    assert s.price_vs_vwap == above_below(100_000.0, s.vwap)
    very_high = analyze_timeframe(candles_15m, settings, 10_000_000.0).summary
    assert set(very_high.price_vs_ema.values()) == {"arriba"}
    assert very_high.price_vs_vwap == "arriba"
    very_low = analyze_timeframe(candles_15m, settings, 1.0).summary
    assert set(very_low.price_vs_ema.values()) == {"abajo"}


def test_steady_uptrend_is_alcista(settings):
    closes = 100 * np.exp(np.linspace(0, 0.5, 260))
    ta = analyze_timeframe(make_candles(ohlcv_frame(closes, timeframe="1h"), "1h"), settings)
    assert ta.summary.trend == "alcista"
    assert ta.summary.rsi_zone == "sobrecompra"
    assert ta.summary.rsi == pytest.approx(100.0)
    assert set(ta.summary.price_vs_ema.values()) == {"arriba"}


def test_rsi_zone_uses_config_thresholds(settings):
    ind = settings.indicators
    assert rsi_zone(ind.rsi_overbought, settings) == "sobrecompra"
    assert rsi_zone(ind.rsi_oversold, settings) == "sobreventa"
    assert rsi_zone((ind.rsi_overbought + ind.rsi_oversold) / 2, settings) == "neutral"
    assert rsi_zone(None, settings) is None


# --- datos insuficientes ----------------------------------------------------------


def test_flags_when_there_are_fewer_than_200_candles(settings, market):
    df = market.candles_frame("BTC", "1h").iloc[-151:].copy()  # 150 cerradas + la que se forma
    s = analyze_timeframe(make_candles(df, "1h"), settings, 100_000.0).summary
    assert s.candles_used == 150
    assert "menos de 200 velas: EMA 200 no disponible" in s.flags
    assert "sin EMA 200 la tendencia queda como lateral" in s.flags
    assert s.ema[200] is None
    assert s.price_vs_ema[200] is None
    assert s.trend == "lateral"
    assert s.ema[50] is not None and s.rsi is not None and s.macd_signal is not None


def test_flags_with_very_few_candles(settings, market):
    df = market.candles_frame("BTC", "15m").iloc[-11:].copy()
    s = analyze_timeframe(make_candles(df, "15m"), settings, 100_000.0).summary
    assert s.candles_used == 10
    rsi_need = settings.indicators.rsi_period + 1
    macd_need = settings.indicators.macd.slow + settings.indicators.macd.signal - 1
    assert f"menos de {rsi_need} velas: RSI no disponible" in s.flags
    assert f"menos de {macd_need} velas: señal del MACD no disponible" in s.flags
    assert f"menos de {settings.indicators.atr_period} velas: ATR no disponible" in s.flags
    assert s.rsi is None and s.atr is None and s.atr_pct is None and s.bb_upper is None
    assert s.vwap is not None  # el VWAP no necesita historia
    assert all("nan" not in f.lower() for f in s.flags)


def test_no_closed_candles(settings):
    df = ohlcv_frame([100.0], timeframe="15m")  # una sola vela, formándose
    ta = analyze_timeframe(make_candles(df, "15m"), settings, 100.0)
    assert ta.summary.flags == ["sin velas cerradas"]
    assert ta.summary.close is None
    assert ta.summary.levels is None
    assert ta.summary.trend == "lateral"


# --- soportes y resistencias ------------------------------------------------------


@pytest.mark.parametrize("symbol", ["BTC", "XRP", "SOL"])
@pytest.mark.parametrize("tf", ["15m", "1h"])
def test_support_resistance_sanity(settings, market, symbol, tf):
    df = market.candles_frame(symbol, tf).iloc[-500:].copy()
    price = float(df["close"].iloc[-1])
    s = analyze_timeframe(make_candles(df, tf, symbol), settings, price).summary
    lv = s.levels
    assert lv is not None
    n = settings.indicators.levels.max_levels_each_side
    assert 0 < len(lv.supports) <= n
    assert 0 < len(lv.resistances) <= n
    assert all(level.price < price and level.kind == "soporte" for level in lv.supports)
    assert all(level.price > price and level.kind == "resistencia" for level in lv.resistances)
    assert [x.price for x in lv.supports] == sorted((x.price for x in lv.supports), reverse=True)
    assert [x.price for x in lv.resistances] == sorted(x.price for x in lv.resistances)
    assert all(x.distance_pct < 0 for x in lv.supports)
    assert all(x.distance_pct > 0 for x in lv.resistances)
    assert lv.reference_price == price
    assert lv.cluster_tolerance == pytest.approx(
        settings.indicators.levels.cluster_atr_mult * s.atr
    )


def test_levels_match_a_direct_call(settings, candles_15m):
    price = 100_000.0
    s = analyze_timeframe(candles_15m, settings, price, now=FIXED_NOW).summary
    lv = settings.indicators.levels
    direct = support_resistance(
        candles_15m.closed_only(),
        price,
        s.atr,
        lv.pivot_left,
        lv.pivot_right,
        lv.cluster_atr_mult,
        lv.max_levels_each_side,
        lv.include_previous_day,
        settings.app.market_timezone,
        as_of=FIXED_NOW,
    )
    assert s.levels == direct


# --- todos los timeframes -----------------------------------------------------------


def test_analyze_all_orders_timeframes_and_adds_daily_range(settings, market):
    candles = {
        tf: make_candles(market.candles_frame("BTC", tf).iloc[-500:].copy(), tf)
        for tf in ["15m", "1d", "5m", "1h", "4h"]
    }
    result = analyze_all(candles, settings, 100_000.0, now=FIXED_NOW)
    assert list(result.timeframes) == ["1d", "4h", "1h", "15m", "5m"]
    assert result["1h"].summary.timeframe == "1h"
    expected = daily_range_pct(candles["1d"].closed_only(), settings.indicators.atr_period)
    assert result.daily_range_pct == pytest.approx(expected)
    assert 0 < result.daily_range_pct < 20
    assert result.flags == []
    for ta in result.timeframes.values():
        s = ta.summary
        assert 0 <= s.rsi <= 100
        assert s.atr > 0 and s.bb_upper > s.bb_mid > s.bb_lower


def test_analyze_all_without_daily_candles(settings, candles_15m):
    result = analyze_all({"15m": candles_15m}, settings, 100_000.0)
    assert result.daily_range_pct is None
    assert any("rango diario" in f for f in result.flags)


def test_analyze_all_with_few_daily_candles(settings, market):
    daily = make_candles(market.candles_frame("BTC", "1d").iloc[-6:].copy(), "1d")
    result = analyze_all({"1d": daily}, settings, 100_000.0)
    assert result.daily_range_pct is None
    assert any("velas diarias cerradas" in f for f in result.flags)
    assert not math.isnan(result["1d"].summary.close)


# --- "día anterior" sin pasar ``now`` -----------------------------------------------


@pytest.mark.parametrize(
    ("timeframe", "now", "n"),
    [
        ("1d", datetime(2026, 10, 12, 14, 0, tzinfo=UTC), 60),  # la vela del 12/10 se forma
        ("4h", datetime(2026, 10, 12, 2, 30, tzinfo=UTC), 300),  # primeras horas del día
        ("1h", datetime(2026, 10, 12, 0, 20, tzinfo=UTC), 300),
        ("15m", datetime(2026, 10, 12, 0, 5, tzinfo=UTC), 300),
        ("15m", datetime(2026, 10, 12, 14, 37, tzinfo=UTC), 300),
    ],
)
def test_previous_day_without_now_matches_passing_now(settings, timeframe, now, n):
    # Regresión: sin ``now`` "hoy" salía del open_time de la última vela cerrada, así
    # que en 1d el "día anterior" quedaba dos días atrás (y en 4h/1h/15m, al empezar el día).
    df = ohlcv_frame(100 + np.arange(float(n)) * 0.1, timeframe=timeframe, end=now)
    candles = make_candles(df, timeframe)
    with_now = analyze_timeframe(candles, settings, 120.0, now=now).summary.levels
    without = analyze_timeframe(candles, settings, 120.0).summary.levels
    via_all = analyze_all({timeframe: candles}, settings, 120.0)[timeframe].summary.levels
    assert with_now.prev_day == date(2026, 10, 11)
    assert without.prev_day == date(2026, 10, 11)
    assert via_all.prev_day == date(2026, 10, 11)
    assert without.prev_day_high == with_now.prev_day_high
    assert without.prev_day_low == with_now.prev_day_low
