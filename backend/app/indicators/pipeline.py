"""Todos los indicadores de un timeframe, listos para el gráfico y el resumen.

``analyze_timeframe(candles, settings, current_price)`` devuelve un
``TimeframeAnalysis`` con:

- ``frame``: las velas (OHLCV + ``closed``) con una columna por indicador:
  ``ema_<p>`` por cada período de ``indicators.ema_periods``, ``vwap`` (solo en
  ``indicators.vwap_timeframes``, diario en ``app.market_timezone``), ``rsi``,
  ``macd``, ``macd_signal``, ``macd_hist``, ``atr``, ``bb_mid``, ``bb_upper``,
  ``bb_lower``, ``bb_width`` (ancho de banda / media, fracción),
  ``bb_percent_b``, ``rel_volume`` y ``trend``. Todos los indicadores son
  causales: el valor de una vela solo usa esa vela y las anteriores, así que la
  vela que se forma no cambia los valores de las cerradas.
- ``summary`` (``TimeframeSummary``): los valores de la **última vela cerrada**
  si ``timeframes.closed_candles_only`` (si no, de la que se está formando);
  tendencia de esa vela; posición del **precio actual** (que se pasa aparte)
  respecto de cada EMA y del VWAP; ATR en % del precio actual; zona del RSI
  según ``indicators.rsi_overbought`` / ``rsi_oversold``; soportes y
  resistencias con ``levels.support_resistance`` y avisos de datos
  insuficientes ("menos de 200 velas: EMA 200 no disponible").

Decisiones:

- La tendencia usa las dos EMAs más largas de ``ema_periods`` (con
  ``[20, 50, 200]``: EMA 50 y EMA 200), con ``trend.classify_trend``.
- Los soportes y resistencias salen **siempre** de velas cerradas, aunque
  ``closed_candles_only`` sea ``false``: un pivote se confirma recién cuando
  cierran ``pivot_right`` velas (cero look-ahead). Se usan los parámetros de
  ``indicators.levels``, el ATR de la última vela cerrada y el precio actual como
  referencia.
- El "día anterior" (máximo/mínimo) se calcula respecto de ``now``; sin
  ``now``, respecto de la hora de cierre de la última vela cerrada.
- Un valor faltante (``NaN``) se devuelve como ``None`` en el resumen.

``analyze_all(candles_by_tf, settings, current_price)`` corre lo mismo para
cada timeframe y suma el rango diario típico (``volatility.daily_range_pct``
con las velas diarias cerradas y ``indicators.atr_period``).

Este módulo no lee archivos: recibe ``AppSettings`` por parámetro.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd

from app.config import AppSettings
from app.exchanges.base import TIMEFRAME_MINUTES, Candles
from app.indicators.levels import SRResult, support_resistance
from app.indicators.momentum import macd, rsi
from app.indicators.moving_averages import ema, vwap_daily
from app.indicators.trend import Position, TrendLabel, above_below, classify_trend_series
from app.indicators.volatility import atr, bollinger, daily_range_pct
from app.indicators.volume import relative_volume

RsiZone = Literal["sobrecompra", "sobreventa", "neutral"]

DAILY_TIMEFRAME = "1d"
TREND_SIDEWAYS: TrendLabel = "lateral"


# --- modelos -------------------------------------------------------------------


@dataclass(frozen=True)
class TimeframeSummary:
    """Foto de los indicadores en la vela que se usa para decidir."""

    timeframe: str
    candles_used: int  # velas disponibles hasta la vela usada (inclusive)
    candle_time: datetime | None  # open_time de la vela usada
    candle_closed: bool | None
    close: float | None  # cierre de la vela usada
    current_price: float | None
    ema: dict[int, float | None]
    vwap: float | None  # None si el timeframe no lleva VWAP
    rsi: float | None
    rsi_zone: RsiZone | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    atr: float | None
    atr_pct: float | None  # ATR / precio actual × 100
    bb_mid: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_width: float | None  # (superior − inferior) / media, fracción
    rel_volume: float | None
    trend: TrendLabel
    trend_emas: tuple[int, int] | None  # (EMA media, EMA larga) usadas para la tendencia
    price_vs_ema: dict[int, Position | None]
    price_vs_vwap: Position | None
    levels: SRResult | None
    has_vwap: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TimeframeAnalysis:
    timeframe: str
    frame: pd.DataFrame  # velas + columnas de indicadores (para el gráfico)
    summary: TimeframeSummary


@dataclass(frozen=True)
class MarketAnalysis:
    """Resultado de ``analyze_all``."""

    timeframes: dict[str, TimeframeAnalysis]
    current_price: float | None
    daily_range_pct: float | None  # ATR diario / cierre diario × 100
    flags: list[str] = field(default_factory=list)

    def __getitem__(self, timeframe: str) -> TimeframeAnalysis:
        return self.timeframes[timeframe]


# --- utilidades ----------------------------------------------------------------


def _value(x: object) -> float | None:
    """``float`` finito o ``None``."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def trend_ema_periods(settings: AppSettings) -> tuple[int, int] | None:
    """Las dos EMAs más largas de ``indicators.ema_periods`` (media, larga)."""
    periods = sorted(set(settings.indicators.ema_periods))
    if len(periods) < 2:
        return None
    return periods[-2], periods[-1]


def rsi_zone(value: float | None, settings: AppSettings) -> RsiZone | None:
    """Zona del RSI según ``indicators.rsi_overbought`` y ``rsi_oversold``."""
    if value is None:
        return None
    if value >= settings.indicators.rsi_overbought:
        return "sobrecompra"
    if value <= settings.indicators.rsi_oversold:
        return "sobreventa"
    return "neutral"


def compute_indicators(df: pd.DataFrame, timeframe: str, settings: AppSettings) -> pd.DataFrame:
    """Copia de ``df`` con todas las columnas de indicadores (ver el docstring del módulo)."""
    ind = settings.indicators
    out = df.copy()
    close = out["close"]
    for p in sorted(set(ind.ema_periods)):
        out[f"ema_{p}"] = ema(close, p)
    if timeframe in ind.vwap_timeframes:
        out["vwap"] = vwap_daily(out, settings.app.market_timezone)
    out["rsi"] = rsi(close, ind.rsi_period)
    m = macd(close, ind.macd.fast, ind.macd.slow, ind.macd.signal)
    out["macd"] = m["macd"]
    out["macd_signal"] = m["signal"]
    out["macd_hist"] = m["hist"]
    out["atr"] = atr(out, ind.atr_period)
    bb = bollinger(close, ind.bollinger.period, ind.bollinger.std)
    out["bb_mid"] = bb["mid"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["bb_width"] = bb["width"]
    out["bb_percent_b"] = bb["percent_b"]
    out["rel_volume"] = relative_volume(out["volume"], ind.relative_volume_period)
    trend_periods = trend_ema_periods(settings)
    if trend_periods is None:
        out["trend"] = pd.Series(TREND_SIDEWAYS, index=out.index, dtype=object)
    else:
        mid, long_ = trend_periods
        out["trend"] = classify_trend_series(close, out[f"ema_{mid}"], out[f"ema_{long_}"])
    return out


def _required_candles(settings: AppSettings) -> list[tuple[str, str, int]]:
    """(columna, nombre para el aviso, velas mínimas) de cada indicador."""
    ind = settings.indicators
    req = [(f"ema_{p}", f"EMA {p}", p) for p in sorted(set(ind.ema_periods))]
    req += [
        ("rsi", "RSI", ind.rsi_period + 1),
        ("macd", "MACD", ind.macd.slow),
        ("macd_signal", "señal del MACD", ind.macd.slow + ind.macd.signal - 1),
        ("atr", "ATR", ind.atr_period),
        ("bb_mid", "Bollinger", ind.bollinger.period),
        ("rel_volume", "volumen relativo", ind.relative_volume_period + 1),
    ]
    return req


def _insufficient_flags(row: pd.Series, n: int, settings: AppSettings, has_vwap: bool) -> list[str]:
    flags: list[str] = []
    for column, label, need in _required_candles(settings):
        if _value(row.get(column)) is not None:
            continue
        if n < need:
            flags.append(f"menos de {need} velas: {label} no disponible")
        else:
            flags.append(f"{label} no disponible")
    if has_vwap and _value(row.get("vwap")) is None:
        flags.append("VWAP no disponible")
    return flags


# --- análisis ------------------------------------------------------------------


def _empty_summary(
    timeframe: str, current_price: float | None, flags: list[str]
) -> TimeframeSummary:
    return TimeframeSummary(
        timeframe=timeframe,
        candles_used=0,
        candle_time=None,
        candle_closed=None,
        close=None,
        current_price=current_price,
        ema={},
        vwap=None,
        rsi=None,
        rsi_zone=None,
        macd=None,
        macd_signal=None,
        macd_hist=None,
        atr=None,
        atr_pct=None,
        bb_mid=None,
        bb_upper=None,
        bb_lower=None,
        bb_width=None,
        rel_volume=None,
        trend=TREND_SIDEWAYS,
        trend_emas=None,
        price_vs_ema={},
        price_vs_vwap=None,
        levels=None,
        flags=flags,
    )


def analyze_timeframe(
    candles: Candles,
    settings: AppSettings,
    current_price: float | None = None,
    *,
    now: datetime | None = None,
) -> TimeframeAnalysis:
    """Indicadores y resumen de un timeframe.

    ``current_price`` es el precio actual (Binance o el de Nexo cargado a
    mano); si no se pasa, se usa el último cierre disponible. ``now`` (el
    momento actual) se pasa a los niveles como ``as_of`` para decidir cuál es
    "hoy" y, con eso, el "día anterior". Si no se pasa, "hoy" sale de la hora de
    **cierre** de la última vela cerrada (open_time + duración del timeframe):
    así en 1d el día anterior es el de la última vela cerrada y no el previo a
    ese, y en 4h/1h/15m no se corre un día en las primeras horas de cada día.
    """
    tf = candles.timeframe
    frame = compute_indicators(candles.df, tf, settings)
    has_vwap = tf in settings.indicators.vwap_timeframes
    closed_mask = frame["closed"].to_numpy(dtype=bool)
    closed_only = settings.timeframes.closed_candles_only
    usable = frame[closed_mask] if closed_only else frame

    price = _value(current_price)
    if price is not None and price <= 0:
        raise ValueError(f"El precio actual tiene que ser positivo (llegó {current_price!r})")
    if price is None and len(frame):
        price = _value(frame["close"].iloc[-1])

    if usable.empty:
        reason = "sin velas cerradas" if len(frame) else "sin velas"
        return TimeframeAnalysis(tf, frame, _empty_summary(tf, price, [reason]))

    row = usable.iloc[-1]
    n = len(usable)
    close = _value(row["close"])
    ema_values = {p: _value(row[f"ema_{p}"]) for p in sorted(set(settings.indicators.ema_periods))}
    vwap = _value(row["vwap"]) if has_vwap else None
    atr_value = _value(row["atr"])
    rsi_value = _value(row["rsi"])
    flags = _insufficient_flags(row, n, settings, has_vwap)
    trend_periods = trend_ema_periods(settings)
    if trend_periods is None:
        flags.append("hacen falta dos EMAs en indicators.ema_periods: tendencia lateral")
    elif ema_values.get(trend_periods[1]) is None or ema_values.get(trend_periods[0]) is None:
        flags.append(f"sin EMA {trend_periods[1]} la tendencia queda como lateral")

    levels: SRResult | None = None
    if price is not None:
        lv = settings.indicators.levels
        sr_frame = frame[closed_mask]
        if not sr_frame.empty:
            as_of = now
            if as_of is None:
                # Sin "ahora": la última vela cerrada terminó en open_time + duración.
                minutes = TIMEFRAME_MINUTES.get(tf, 0)
                close_time = sr_frame.index[-1] + pd.Timedelta(minutes=minutes)
                as_of = close_time.to_pydatetime()
            # El ATR para agrupar también sale de la última vela cerrada.
            levels = support_resistance(
                sr_frame,
                price,
                _value(sr_frame["atr"].iloc[-1]),
                lv.pivot_left,
                lv.pivot_right,
                lv.cluster_atr_mult,
                lv.max_levels_each_side,
                lv.include_previous_day,
                settings.app.market_timezone,
                as_of=as_of,
            )
            if not levels.pivots_used:
                flags.append("sin pivotes confirmados: no hay soportes ni resistencias")

    summary = TimeframeSummary(
        timeframe=tf,
        candles_used=n,
        candle_time=usable.index[-1].to_pydatetime(),
        candle_closed=bool(row["closed"]),
        close=close,
        current_price=price,
        ema=ema_values,
        vwap=vwap,
        rsi=rsi_value,
        rsi_zone=rsi_zone(rsi_value, settings),
        macd=_value(row["macd"]),
        macd_signal=_value(row["macd_signal"]),
        macd_hist=_value(row["macd_hist"]),
        atr=atr_value,
        atr_pct=None if atr_value is None or not price else atr_value / price * 100.0,
        bb_mid=_value(row["bb_mid"]),
        bb_upper=_value(row["bb_upper"]),
        bb_lower=_value(row["bb_lower"]),
        bb_width=_value(row["bb_width"]),
        rel_volume=_value(row["rel_volume"]),
        trend=row["trend"],
        trend_emas=trend_periods,
        price_vs_ema={p: above_below(price, v) for p, v in ema_values.items()},
        price_vs_vwap=above_below(price, vwap) if has_vwap else None,
        levels=levels,
        has_vwap=has_vwap,
        flags=flags,
    )
    return TimeframeAnalysis(tf, frame, summary)


def _timeframe_order(tf: str) -> int:
    return -TIMEFRAME_MINUTES.get(tf, 0)


def analyze_all(
    candles_by_tf: Mapping[str, Candles],
    settings: AppSettings,
    current_price: float | None,
    *,
    now: datetime | None = None,
) -> MarketAnalysis:
    """``analyze_timeframe`` para cada timeframe (del más largo al más corto) + rango diario."""
    result: dict[str, TimeframeAnalysis] = {}
    for tf in sorted(candles_by_tf, key=_timeframe_order):
        result[tf] = analyze_timeframe(candles_by_tf[tf], settings, current_price, now=now)

    flags: list[str] = []
    daily_pct: float | None = None
    daily = candles_by_tf.get(DAILY_TIMEFRAME)
    if daily is None:
        flags.append("sin velas diarias: no se calcula el rango diario típico")
    else:
        closed = daily.closed_only()
        daily_pct = _value(daily_range_pct(closed, settings.indicators.atr_period))
        if daily_pct is None:
            flags.append(
                f"menos de {settings.indicators.atr_period} velas diarias cerradas: "
                "rango diario típico no disponible"
            )
    return MarketAnalysis(result, _value(current_price), daily_pct, flags)


def frame_columns(settings: AppSettings, timeframe: str) -> list[str]:
    """Columnas de indicadores que ``compute_indicators`` agrega para ``timeframe``."""
    ind = settings.indicators
    cols = [f"ema_{p}" for p in sorted(set(ind.ema_periods))]
    if timeframe in ind.vwap_timeframes:
        cols.append("vwap")
    cols += [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_percent_b",
        "rel_volume",
        "trend",
    ]
    return cols


__all__ = [
    "DAILY_TIMEFRAME",
    "MarketAnalysis",
    "RsiZone",
    "TimeframeAnalysis",
    "TimeframeSummary",
    "analyze_all",
    "analyze_timeframe",
    "compute_indicators",
    "frame_columns",
    "rsi_zone",
    "trend_ema_periods",
]
