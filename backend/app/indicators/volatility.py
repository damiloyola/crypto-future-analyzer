"""Indicadores de volatilidad (mismas definiciones que TradingView/Pine).

- ``true_range(df)``: rango verdadero ``max(H − L, |H − C[t-1]|, |L − C[t-1]|)``.
  En la primera vela (sin cierre anterior) vale ``H − L``, como ``ta.tr(true)``.
- ``atr(df, period=14)``: ATR de Wilder (``ta.atr``): media de Wilder (RMA) del
  rango verdadero, sembrada con la media simple de los primeros ``period``
  rangos. El primer ATR queda en la posición ``period - 1``.
- ``bollinger(close, period=20, std=2.0)``: ``DataFrame`` con ``mid`` (SMA),
  ``upper``/``lower`` (``mid ± std × desvío``), ``width`` (``(upper − lower) / mid``)
  y ``percent_b`` (``(close − lower) / (upper − lower)``). El desvío es el
  **poblacional** (ddof = 0), como ``ta.stdev``. Si la ventana es constante el
  desvío es exactamente 0 (``width = 0``) y ``percent_b`` queda ``NaN`` porque
  las bandas coinciden.
- ``daily_range_pct(daily_df, period=14)``: último ATR de las velas diarias
  como % del último cierre, el "rango típico del día" que acota los objetivos
  intradiarios. ``NaN`` si no hay velas suficientes. Conviene pasarle solo
  velas diarias cerradas.

Las salidas quedan alineadas al índice de la entrada, con ``NaN`` mientras no
hay datos suficientes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.indicators.moving_averages import (
    as_float_series,
    require_period,
    rma,
    rolling_windows,
)


def true_range(df: pd.DataFrame) -> pd.Series:
    """Rango verdadero por vela; la primera vela usa ``high − low``."""
    high = df["high"].astype("float64").to_numpy()
    low = df["low"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    prev_close = np.concatenate(([np.nan], close[:-1])) if len(close) else close
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    # Sin cierre anterior (primera vela o hueco) se usa H − L, como ta.tr(true).
    tr = np.where(np.isnan(prev_close), hl, np.maximum(hl, np.maximum(hc, lc)))
    return pd.Series(tr, index=df.index, name="true_range")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR de Wilder; el primero queda en la posición ``period - 1``."""
    p = require_period(period)
    return rma(true_range(df), p).rename(f"atr_{p}")


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger con desvío poblacional (ver el docstring del módulo)."""
    p = require_period(period)
    if not math.isfinite(std) or std < 0:
        raise ValueError(f"std tiene que ser un número >= 0 (llegó {std!r})")
    s = as_float_series(close)
    values = s.to_numpy()
    mid = np.full(len(values), np.nan)
    dev = np.full(len(values), np.nan)
    if len(values) >= p:
        windows = rolling_windows(values, p)
        mid[p - 1 :] = windows.mean(axis=1)
        window_dev = windows.std(axis=1, ddof=0)
        # Ventana constante: desvío exactamente 0 (sin restos de redondeo).
        window_dev[windows.max(axis=1) == windows.min(axis=1)] = 0.0
        dev[p - 1 :] = window_dev
    upper = mid + std * dev
    lower = mid - std * dev
    band = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.where(mid != 0, band / mid, np.nan)
        percent_b = np.where(band != 0, (values - lower) / band, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b},
        index=s.index,
    )


def daily_range_pct(daily_df: pd.DataFrame, period: int = 14) -> float:
    """Último ATR(``period``) diario / último cierre × 100. ``NaN`` si no alcanza."""
    p = require_period(period)
    if len(daily_df) < p:
        return math.nan
    last_atr = float(atr(daily_df, p).iloc[-1])
    last_close = float(daily_df["close"].iloc[-1])
    if math.isnan(last_atr) or math.isnan(last_close) or last_close == 0:
        return math.nan
    return last_atr / last_close * 100.0
