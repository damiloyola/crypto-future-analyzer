"""Indicadores de volumen.

- ``relative_volume(volume, period=20)``: volumen de la vela dividido por el
  promedio de las ``period`` velas **anteriores** (la vela actual no entra en el
  promedio, así un pico no se diluye a sí mismo). 1.0 = volumen normal, 2.0 = el
  doble de lo habitual. El primer valor queda en la posición ``period``; da
  ``NaN`` si el promedio es 0 o falta algún dato en la ventana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.indicators.moving_averages import as_float_series, require_period, sma


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """``volume[t] / promedio(volume[t-period .. t-1])``, alineado al índice de entrada."""
    p = require_period(period)
    v = as_float_series(volume)
    previous_avg = sma(v, p).shift(1).to_numpy()
    values = v.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.where(previous_avg != 0, values / previous_avg, np.nan)
    return pd.Series(rv, index=v.index, name=f"relative_volume_{p}")
