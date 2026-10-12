"""Clasificación de tendencia por timeframe y posición del precio.

- ``classify_trend(close, ema50, ema200)``: ``"alcista"`` si
  precio > EMA50 > EMA200; ``"bajista"`` si precio < EMA50 < EMA200;
  ``"lateral"`` en cualquier otro caso (incluidos empates y cualquier dato
  faltante: sin EMA200 no se afirma una tendencia).
- ``classify_trend_series(close, ema50, ema200)``: lo mismo vela por vela;
  devuelve una ``Series`` de esas etiquetas alineada al índice de ``close``.
- ``above_below(price, level)``: ``"arriba"``, ``"abajo"`` o ``"igual"`` según
  dónde está el precio respecto de un nivel (EMA, VWAP, S/R). ``None`` si falta
  alguno de los dos.
- ``distance_pct(price, level)``: distancia del precio al nivel en % del nivel
  (positiva si el precio está arriba). ``NaN`` si falta un dato o el nivel es 0.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

TrendLabel = Literal["alcista", "bajista", "lateral"]
Position = Literal["arriba", "abajo", "igual"]

TREND_UP: TrendLabel = "alcista"
TREND_DOWN: TrendLabel = "bajista"
TREND_SIDEWAYS: TrendLabel = "lateral"


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float | np.floating) and math.isnan(value))


def classify_trend(close: float | None, ema50: float | None, ema200: float | None) -> TrendLabel:
    """Tendencia de una vela según el orden precio / EMA50 / EMA200."""
    if _missing(close) or _missing(ema50) or _missing(ema200):
        return TREND_SIDEWAYS
    if close > ema50 > ema200:
        return TREND_UP
    if close < ema50 < ema200:
        return TREND_DOWN
    return TREND_SIDEWAYS


def _aligned(values: pd.Series | float, index: pd.Index) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.reindex(index).to_numpy(dtype="float64")
    return np.full(len(index), np.nan if values is None else float(values))


def classify_trend_series(
    close: pd.Series, ema50: pd.Series | float, ema200: pd.Series | float
) -> pd.Series:
    """Versión vectorizada de ``classify_trend``; las EMAs se alinean al índice de ``close``."""
    c = close.to_numpy(dtype="float64")
    e50 = _aligned(ema50, close.index)
    e200 = _aligned(ema200, close.index)
    # Las comparaciones con NaN dan False, así que los faltantes caen en "lateral".
    up = (c > e50) & (e50 > e200)
    down = (c < e50) & (e50 < e200)
    labels = np.select([up, down], [TREND_UP, TREND_DOWN], default=TREND_SIDEWAYS)
    return pd.Series(labels, index=close.index, name="trend", dtype=object)


def above_below(price: float | None, level: float | None) -> Position | None:
    """Posición del precio respecto de ``level``; ``None`` si falta alguno."""
    if _missing(price) or _missing(level):
        return None
    if price > level:
        return "arriba"
    if price < level:
        return "abajo"
    return "igual"


def distance_pct(price: float | None, level: float | None) -> float:
    """``(price − level) / level × 100``; ``NaN`` si falta un dato o ``level`` es 0."""
    if _missing(price) or _missing(level) or level == 0:
        return math.nan
    return (float(price) - float(level)) / float(level) * 100.0
