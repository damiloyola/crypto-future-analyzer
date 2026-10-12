"""Features para el detector de régimen (HMM).

Todas son **causales**: el valor en la vela *t* usa solo velas hasta *t*
inclusive (nada de ventanas centradas, ``bfill`` ni ``shift(-k)``). Así, calcular
sobre ``df.iloc[:k]`` da exactamente lo mismo que calcular sobre ``df`` entero y
quedarse con las primeras ``k`` filas (los tests lo verifican). Hay que pasarles
solo velas **cerradas**: si el ``DataFrame`` trae la columna ``closed`` con
alguna vela sin cerrar, se lanza ``ValueError``.

Las cuatro features, en el orden de ``FEATURE_COLUMNS`` (con
``smoothing_window = N``; con ``N = 1`` son las de la especificación literal):

- ``log_return``: retorno logarítmico **acumulado de las últimas N velas**,
  ``ln(close[t] / close[t-N])`` (suma de los retornos de cada vela). Con N = 1
  es ``ln(close[t] / close[t-1])``.
- ``realized_vol``: desvío **muestral** (ddof = 1) de ``log_return`` en las
  últimas ``realized_vol_window`` velas, incluida *t*. Si los retornos de la
  ventana son todos iguales vale exactamente 0.
- ``range_pct``: promedio de ``(high - low) / close`` en las últimas N velas.
- ``volume_z``: z-score de ``log1p(volume[t])`` contra la media y el desvío
  muestral (ddof = 1) de ``log1p(volume)`` en las últimas
  ``volume_zscore_window`` velas, incluida *t*. Se usa ``log1p`` porque el
  volumen cripto tiene colas muy pesadas (un pico de 20× aplastaría todo lo
  demás); el logaritmo lo lleva a una escala donde "mucho más que lo normal" y
  "mucho menos" pesan parecido, y ``log1p`` admite volumen 0. Si la ventana es
  constante (desvío 0, por ejemplo volumen 0 en todas las velas) el z-score
  vale 0: el volumen está exactamente en su nivel habitual. Después se promedia
  en las últimas N velas.

Por qué el suavizado (decisión del usuario, ``regime.features.smoothing_window``,
24 por defecto): vela por vela, una sola vela grande ya parece un "Crash" o una
"Euforia" y el régimen cambia cada pocas horas. En el régimen de 1 hora, el
resumen de las últimas 24 velas (cuánto subió o bajó en el día, rango y volumen
promedio) hace que el régimen describa el clima del día y dure del orden de un
día. La volatilidad realizada ya es una medida de ventana y no se vuelve a
suavizar. Ojo: ``regime.features`` es uno solo para los dos timeframes, así que
en el régimen diario de contexto las 24 velas son **24 días** (la tendencia de
las últimas tres semanas y media), cada régimen dura del orden de 2 a 4 semanas
y el calentamiento es de 190 días.

Cada ventana se calcula por separado (media y desvío en dos pasadas sobre las
velas de la ventana, sin sumas acumuladas), así no arrastra errores de redondeo
y el resultado de una vela no depende de cuántas velas haya antes o después.

Calentamiento: las primeras ``warmup_bars(...) = max(realized_vol_window, N,
volume_zscore_window + N - 2)`` filas tienen algún ``NaN`` (con 24, 168 y N = 24:
190 velas; con N = 1: 167). ``feature_matrix`` las descarta y devuelve la matriz
lista para el HMM.

La estandarización (media 0, desvío 1) **no** se hace acá: la hace
``FeatureScaler`` (``app.regime.types``), ajustado solo con la ventana de
entrenamiento y aplicado igual hacia adelante.

Funciones:

- ``compute_features(df, realized_vol_window=24, volume_zscore_window=168, *,
  smoothing_window)``: ``DataFrame`` con índice igual al de ``df`` y columnas
  ``FEATURE_COLUMNS``.
- ``warmup_bars(realized_vol_window=24, volume_zscore_window=168, *,
  smoothing_window)``: cuántas filas iniciales quedan con ``NaN``.
- ``feature_matrix(df, realized_vol_window=24, volume_zscore_window=168, *,
  smoothing_window)``: ``(X, index)`` con ``X`` de forma ``(T, 4)`` en float64,
  sin el calentamiento.

``smoothing_window`` es obligatorio y va por nombre: no tiene valor por omisión
para que nadie arme sin querer features sin suavizar para un modelo entrenado
con suavizado (el modelo las aceptaría sin protestar y el régimen saldría
cualquier cosa). Pasá siempre ``regime.features.smoothing_window``.

Validaciones (``ValueError`` en español): faltan columnas, índice que no es
``DatetimeIndex`` ordenado y sin duplicados, precios no finitos o <= 0, volumen
no finito o negativo, velas sin cerrar, ventanas menores a 2 (el suavizado
admite 1).

``FEATURE_COLUMNS`` se toma de ``app.regime.types``, que es un módulo hoja
(solo depende de numpy), así la capa de indicadores no arrastra el resto del
detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from app.indicators.moving_averages import require_period
from app.regime.types import FEATURE_COLUMNS

_PRICE_COLUMNS: tuple[str, ...] = ("high", "low", "close")
_REQUIRED_COLUMNS: tuple[str, ...] = (*_PRICE_COLUMNS, "volume")


def _require_window(window: int, name: str) -> int:
    """Valida que la ventana sea un entero >= 2 (el desvío muestral necesita 2 datos)."""
    w = require_period(window, name)
    if w < 2:
        raise ValueError(f"{name} tiene que ser un entero mayor o igual a 2 (llegó {window!r})")
    return w


def _require_smoothing(window: int) -> int:
    """Valida el suavizado: entero >= 1 (1 = sin suavizar)."""
    return require_period(window, "smoothing_window")


def _validate_candles(df: pd.DataFrame) -> None:
    """Controla que ``df`` tenga el formato de velas que necesitan las features."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("las features del régimen necesitan un DataFrame de velas")
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "a las velas les faltan columnas para calcular las features del régimen: "
            + ", ".join(missing)
        )
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("las velas tienen que venir con índice de fechas (open_time)")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("las velas tienen que estar ordenadas por fecha y sin repetidas")
    if "closed" in df.columns and not df["closed"].astype(bool).all():
        raise ValueError(
            "hay velas sin cerrar: las features del régimen se calculan solo con velas cerradas"
        )
    for col in _PRICE_COLUMNS:
        values = df[col].to_numpy(dtype="float64")
        if not (np.isfinite(values) & (values > 0)).all():
            raise ValueError(
                f"la columna {col!r} tiene precios vacíos, infinitos, en cero o negativos"
            )
    volume = df["volume"].to_numpy(dtype="float64")
    if not (np.isfinite(volume) & (volume >= 0)).all():
        raise ValueError("la columna 'volume' tiene valores vacíos, infinitos o negativos")


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Media y desvío muestral (ddof = 1) de la ventana que termina en cada fila.

    ``NaN`` hasta tener ``window`` datos o si la ventana tiene algún ``NaN``.
    Ventana constante: desvío exactamente 0 (sin restos de redondeo).
    """
    n = len(values)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    if n < window:
        return mean, std
    # Copia contigua: cada fila se reduce siempre igual, sin importar cuántas
    # filas haya (garantiza que el resultado de una vela no cambie al agregar
    # velas posteriores, ni siquiera en el último bit).
    windows = np.ascontiguousarray(sliding_window_view(values, window))
    window_mean = windows.mean(axis=1)
    deviations = windows - window_mean[:, None]
    window_std = np.sqrt((deviations * deviations).sum(axis=1) / (window - 1))
    window_std[windows.max(axis=1) == windows.min(axis=1)] = 0.0
    mean[window - 1 :] = window_mean
    std[window - 1 :] = window_std
    return mean, std


def _rolling_sum_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Suma y media de la ventana que termina en cada fila (``NaN`` si falta algún dato).

    Igual que ``_rolling_mean_std``: cada ventana se reduce por separado sobre
    una copia contigua, así el valor de una vela no depende de cuántas velas
    haya antes o después. Con ``window == 1`` devuelve los mismos valores.
    """
    n = len(values)
    total = np.full(n, np.nan)
    if window == 1:
        return values.copy(), values.copy()
    if n < window:
        return total, total.copy()
    windows = np.ascontiguousarray(sliding_window_view(values, window))
    total[window - 1 :] = windows.sum(axis=1)
    return total, total / window


def warmup_bars(
    realized_vol_window: int = 24, volume_zscore_window: int = 168, *, smoothing_window: int
) -> int:
    """Cantidad de filas iniciales de ``compute_features`` con algún ``NaN``.

    ``realized_vol`` necesita ``realized_vol_window`` retornos (el primero sale
    de la segunda vela), el retorno acumulado necesita ``smoothing_window``
    retornos, el rango promedio ``smoothing_window`` velas y ``volume_z``
    necesita ``volume_zscore_window`` velas más ``smoothing_window - 1`` para
    el promedio.
    """
    rv = _require_window(realized_vol_window, "realized_vol_window")
    vz = _require_window(volume_zscore_window, "volume_zscore_window")
    sm = _require_smoothing(smoothing_window)
    return max(rv, sm, vz + sm - 2)


def compute_features(
    df: pd.DataFrame,
    realized_vol_window: int = 24,
    volume_zscore_window: int = 168,
    *,
    smoothing_window: int,
) -> pd.DataFrame:
    """Las cuatro features del HMM por vela, causales (ver el docstring del módulo)."""
    rv = _require_window(realized_vol_window, "realized_vol_window")
    vz = _require_window(volume_zscore_window, "volume_zscore_window")
    sm = _require_smoothing(smoothing_window)
    _validate_candles(df)

    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    volume = df["volume"].to_numpy(dtype="float64")
    n = len(close)

    log_return = np.full(n, np.nan)
    if n > 1:
        log_return[1:] = np.log(close[1:] / close[:-1])

    _, realized_vol = _rolling_mean_std(log_return, rv)

    range_pct = (high - low) / close

    log_volume = np.log1p(volume)
    vol_mean, vol_std = _rolling_mean_std(log_volume, vz)
    volume_z = np.full(n, np.nan)
    has_std = ~np.isnan(vol_std)
    positive = has_std & (vol_std > 0)
    volume_z[has_std] = 0.0  # ventana constante: volumen en su nivel habitual
    volume_z[positive] = (log_volume[positive] - vol_mean[positive]) / vol_std[positive]

    # La volatilidad realizada se calcula sobre los retornos de cada vela; recién
    # después se resumen retorno, rango y volumen en las últimas ``sm`` velas.
    return_sum, _ = _rolling_sum_mean(log_return, sm)
    _, range_mean = _rolling_sum_mean(range_pct, sm)
    _, volume_z_mean = _rolling_sum_mean(volume_z, sm)

    columns = {
        "log_return": return_sum,
        "realized_vol": realized_vol,
        "range_pct": range_mean,
        "volume_z": volume_z_mean,
    }
    return pd.DataFrame({name: columns[name] for name in FEATURE_COLUMNS}, index=df.index)


def feature_matrix(
    df: pd.DataFrame,
    realized_vol_window: int = 24,
    volume_zscore_window: int = 168,
    *,
    smoothing_window: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Matriz ``(T, 4)`` float64 para el HMM, sin las filas de calentamiento.

    Devuelve también el índice (``open_time``) de las filas que quedaron. Como
    las velas se validan antes (precios > 0, volumen >= 0, todo finito), las
    únicas filas con ``NaN`` son exactamente las ``warmup_bars(...)`` primeras.
    """
    features = compute_features(
        df, realized_vol_window, volume_zscore_window, smoothing_window=smoothing_window
    )
    values = features.to_numpy(dtype="float64")
    keep = np.isfinite(values).all(axis=1)
    X = np.ascontiguousarray(values[keep])
    return X, pd.DatetimeIndex(features.index[keep])
