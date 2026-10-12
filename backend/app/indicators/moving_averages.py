"""Medias móviles y suavizados base (mismas definiciones que TradingView/Pine).

Funciones puras: reciben una ``Series`` (o un DataFrame de velas con las
convenciones de ``exchanges/base.py``) y devuelven una ``Series`` alineada al
mismo índice, con ``NaN`` donde todavía no hay datos suficientes. Nunca
modifican la entrada.

- ``sma(series, period)``: media simple de las últimas ``period`` velas
  (``ta.sma``). La primera vale en la posición ``period - 1``. Se calcula por
  ventana (sin sumas acumuladas), así no arrastra errores de redondeo.
- ``ema(series, period)``: EMA con ``alpha = 2 / (period + 1)`` y semilla SMA,
  como ``ta.ema``: la primera EMA es la media simple de los primeros ``period``
  valores válidos y queda en la posición ``period - 1``.
- ``rma(series, period)``: media de Wilder (``ta.rma``), igual que la EMA pero con
  ``alpha = 1 / period``. La usan el RSI y el ATR.
- ``vwap_daily(df, tz="UTC")``: VWAP que se reinicia al comienzo de cada día
  calendario de ``tz``, con precio típico (H+L+C)/3. Si el volumen acumulado del
  día es 0, da ``NaN``.

``NaN`` en la entrada (EMA y RMA): los ``NaN`` iniciales se saltean y la semilla
se toma de los primeros ``period`` valores válidos. Un ``NaN`` en el medio deja
``NaN`` en esa vela y el suavizado sigue desde el último valor válido, como si
esa vela no existiera.

Helpers que reutilizan los demás módulos de indicadores: ``require_period``
(valida que el período sea un entero >= 1), ``as_float_series`` (copia float64)
y ``rolling_windows`` (vista con las ventanas de ``period`` velas).
"""

from __future__ import annotations

from datetime import tzinfo

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def require_period(period: int, name: str = "period") -> int:
    """Devuelve ``period`` si es un entero >= 1; si no, lanza ``ValueError``."""
    if isinstance(period, bool) or not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"{name} tiene que ser un entero mayor o igual a 1 (llegó {period!r})")
    return int(period)


def as_float_series(values: pd.Series) -> pd.Series:
    """``values`` como ``Series`` float64 nueva (el original no se toca)."""
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    # Con Copy-on-Write (pandas 3) astype devuelve un objeto independiente.
    return values.astype("float64")


def rolling_windows(values: np.ndarray, period: int) -> np.ndarray:
    """Vista ``(len - period + 1, period)`` con las ventanas que terminan en cada vela.

    Requiere ``len(values) >= period``. Es una vista: no copia los datos.
    """
    return sliding_window_view(values, period)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Media móvil simple (``ta.sma``). ``NaN`` si la ventana tiene algún ``NaN``."""
    p = require_period(period)
    s = as_float_series(series)
    out = np.full(len(s), np.nan)
    if len(s) >= p:
        out[p - 1 :] = rolling_windows(s.to_numpy(), p).mean(axis=1)
    return pd.Series(out, index=s.index, name=f"sma_{p}")


def _seeded_ewm(series: pd.Series, period: int, alpha: float) -> np.ndarray:
    """Suavizado exponencial con semilla SMA sobre los valores válidos de ``series``.

    ``y[period-1] = media(x[0:period])`` y después
    ``y[t] = alpha * x[t] + (1 - alpha) * y[t-1]``. Los ``NaN`` se saltean y
    quedan como ``NaN`` en la salida (ver el docstring del módulo).
    """
    values = series.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    valid_mask = ~np.isnan(values)
    valid = values[valid_mask]
    if len(valid) < period:
        return out
    seeded = valid.copy()
    seeded[period - 1] = valid[:period].mean()
    seeded[: period - 1] = np.nan
    # Con adjust=False, ewm arranca en el primer valor no NaN (la semilla) y
    # aplica la recursión exacta de Pine.
    smoothed = pd.Series(seeded).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    out[valid_mask] = smoothed
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    """EMA de TradingView (``ta.ema``): ``alpha = 2/(period+1)``, semilla SMA.

    La primera EMA queda en el ``period``-ésimo valor válido (posición
    ``period - 1`` si la serie no arranca con ``NaN``).
    """
    p = require_period(period)
    s = as_float_series(series)
    return pd.Series(_seeded_ewm(s, p, 2.0 / (p + 1)), index=s.index, name=f"ema_{p}")


def rma(series: pd.Series, period: int) -> pd.Series:
    """Media de Wilder (``ta.rma``): ``alpha = 1/period``, semilla SMA."""
    p = require_period(period)
    s = as_float_series(series)
    return pd.Series(_seeded_ewm(s, p, 1.0 / p), index=s.index, name=f"rma_{p}")


def vwap_daily(df: pd.DataFrame, tz: str | tzinfo = "UTC") -> pd.Series:
    """VWAP diario anclado al comienzo de cada día calendario de ``tz``.

    ``VWAP[t] = Σ(precio típico × volumen) / Σ volumen`` desde la primera vela
    del día de ``t`` hasta ``t`` inclusive, con precio típico (H+L+C)/3. El día de
    cada vela se decide por su ``open_time`` convertido a ``tz`` (por defecto
    UTC, la convención cripto). Un índice sin zona horaria se interpreta como UTC.
    Volumen acumulado 0 da ``NaN``.
    """
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("vwap_daily necesita un DataFrame con DatetimeIndex")
    if index.tz is None:
        index = index.tz_localize("UTC")
    # Fecha local "de pared" sin zona: evita problemas con los cambios de horario.
    day = index.tz_convert(tz).tz_localize(None).normalize()

    high = df["high"].astype("float64").to_numpy()
    low = df["low"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    volume = df["volume"].astype("float64").to_numpy()
    pv = (high + low + close) / 3.0 * volume
    # Si falta el precio, esa vela tampoco suma volumen (numerador y denominador parejos).
    volume = np.where(np.isnan(pv), np.nan, volume)

    frame = pd.DataFrame({"pv": pv, "v": volume, "day": day})
    cum = frame.groupby("day", sort=False)[["pv", "v"]].cumsum()
    cum_pv = cum["pv"].to_numpy()
    cum_v = cum["v"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_v != 0, cum_pv / cum_v, np.nan)
    return pd.Series(vwap, index=df.index, name="vwap")
