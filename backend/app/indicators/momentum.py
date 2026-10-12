"""Indicadores de momentum (mismas definiciones que TradingView/Pine).

- ``rsi(close, period=14)``: RSI de Wilder (``ta.rsi``). Subas y bajas se
  suavizan con la media de Wilder (RMA, ``alpha = 1/period``) sembrada con la
  media simple de los primeros ``period`` cambios, así que el primer RSI queda en
  la posición ``period`` (hace falta ``period + 1`` cierres).
  Casos borde: sin bajas y con subas da 100; sin subas y con bajas da 0; sin
  subas ni bajas (precio plano) da **50**. Esto último es una decisión nuestra
  (Pine devuelve ``NaN`` por 0/0): un mercado que no se mueve no está ni
  sobrecomprado ni sobrevendido, y así el scoring no recibe huecos.
- ``macd(close, fast=12, slow=26, signal=9)``: ``DataFrame`` con columnas
  ``macd`` (EMA rápida − EMA lenta), ``signal`` (EMA de la línea MACD, sembrada
  con la media simple de sus primeros ``signal`` valores válidos) y ``hist``
  (``macd − signal``). ``macd`` arranca en la posición ``slow - 1`` y
  ``signal``/``hist`` en ``slow + signal - 2``.

Las salidas quedan alineadas al índice de la entrada, con ``NaN`` mientras no
hay datos suficientes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.moving_averages import as_float_series, ema, require_period, rma

FLAT_RSI = 50.0  # RSI cuando no hubo ni subas ni bajas en el suavizado


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder entre 0 y 100 (ver casos borde en el docstring del módulo)."""
    p = require_period(period)
    s = as_float_series(close)
    change = s.diff()
    # clip conserva los NaN (la primera vela no tiene cambio).
    gains = change.clip(lower=0.0)
    losses = (-change).clip(lower=0.0)
    avg_gain = rma(gains, p).to_numpy()
    avg_loss = rma(losses, p).to_numpy()

    total = avg_gain + avg_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        # 100 * g / (g + l) es lo mismo que 100 - 100 / (1 + g/l), pero sin
        # dividir por cero cuando l = 0. Dividir primero hace que g / g sea
        # exactamente 1 (RSI 100 exacto) y 0 / l exactamente 0.
        values = np.where(total > 0, 100.0 * (avg_gain / total), FLAT_RSI)
    values = np.where(np.isnan(total), np.nan, values)
    return pd.Series(values, index=s.index, name=f"rsi_{p}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD clásico: línea, señal e histograma, alineados al índice de ``close``."""
    f = require_period(fast, "fast")
    sl = require_period(slow, "slow")
    sg = require_period(signal, "signal")
    if f >= sl:
        raise ValueError(f"fast ({f}) tiene que ser menor que slow ({sl})")
    s = as_float_series(close)
    line = ema(s, f) - ema(s, sl)
    signal_line = ema(line, sg)
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "hist": line - signal_line}, index=s.index
    )
