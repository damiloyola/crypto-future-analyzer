"""Tests de los indicadores técnicos (sin red).

Tres tipos de chequeo:

1. Valores de referencia de ejemplos conocidos (planillas de StockCharts
   ChartSchool para RSI, EMA y ATR).
2. Una implementación ingenua con bucles, escrita acá a propósito sin pandas ni
   reutilizar nada de ``app.indicators``, comparada con el código vectorizado
   sobre caminatas aleatorias con semilla fija (tolerancia 1e-9).
3. Propiedades: constantes, límites, largo exacto del prefijo ``NaN``, reinicio
   del VWAP, índice preservado, etc.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.indicators.momentum import macd, rsi
from app.indicators.moving_averages import ema, require_period, rma, sma, vwap_daily
from app.indicators.trend import above_below, classify_trend, classify_trend_series, distance_pct
from app.indicators.volatility import atr, bollinger, daily_range_pct, true_range
from app.indicators.volume import relative_volume

TOL = 1e-9
SEEDS = [1, 7, 42]
NAN = float("nan")


# --- datos sintéticos --------------------------------------------------------


def random_walk_candles(seed: int, n: int = 600, freq: str = "15min") -> pd.DataFrame:
    """Velas OHLCV sintéticas (caminata aleatoria) que cruzan varios días UTC."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate(([100.0], close[:-1])) * (1 + rng.normal(0.0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.lognormal(mean=3.0, sigma=0.6, size=n)
    index = pd.date_range("2024-03-01 20:00", periods=n, freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )


def assert_matches(actual: pd.Series | np.ndarray, expected: list[float]) -> None:
    """Compara con tolerancia 1e-9 y exige los ``NaN`` en las mismas posiciones."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=TOL, atol=TOL
    )


def leading_nans(series: pd.Series) -> int:
    values = series.to_numpy()
    valid = np.flatnonzero(~np.isnan(values))
    return len(values) if len(valid) == 0 else int(valid[0])


# --- implementaciones ingenuas (independientes del código bajo prueba) -------


def naive_ema(values: list[float], period: int) -> list[float]:
    """EMA estilo StockCharts: semilla SMA y después (x - prev) * k + prev.

    Saltea los NaN: la semilla son los primeros ``period`` valores válidos.
    """
    k = 2.0 / (period + 1)
    out = [NAN] * len(values)
    seed_sum, seed_count, prev = 0.0, 0, None
    for i, x in enumerate(values):
        if math.isnan(x):
            continue
        if prev is None:
            seed_sum += x
            seed_count += 1
            if seed_count == period:
                prev = seed_sum / period
                out[i] = prev
        else:
            prev = (x - prev) * k + prev
            out[i] = prev
    return out


def naive_wilder(values: list[float], period: int) -> list[float]:
    """Suavizado de Wilder en su forma original: (prev × (n − 1) + x) / n."""
    out = [NAN] * len(values)
    seed_sum, seed_count, prev = 0.0, 0, None
    for i, x in enumerate(values):
        if math.isnan(x):
            continue
        if prev is None:
            seed_sum += x
            seed_count += 1
            if seed_count == period:
                prev = seed_sum / period
                out[i] = prev
        else:
            prev = (prev * (period - 1) + x) / period
            out[i] = prev
    return out


def naive_rsi(closes: list[float], period: int) -> list[float]:
    out = [NAN] * len(closes)
    avg_gain = avg_loss = None
    gain_sum = loss_sum = 0.0
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        if i < period:
            gain_sum += gain
            loss_sum += loss
            continue
        if i == period:
            avg_gain = (gain_sum + gain) / period
            avg_loss = (loss_sum + loss) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0 and avg_gain == 0:
            out[i] = 50.0
        elif avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def naive_true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    out = []
    for i in range(len(highs)):
        if i == 0:
            out.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            out.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    return out


def naive_bollinger(closes: list[float], period: int, mult: float) -> dict[str, list[float]]:
    keys = ("mid", "upper", "lower", "width", "percent_b")
    out = {k: [NAN] * len(closes) for k in keys}
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        mean = sum(window) / period
        sd = math.sqrt(sum((x - mean) ** 2 for x in window) / period)  # poblacional
        upper, lower = mean + mult * sd, mean - mult * sd
        out["mid"][i] = mean
        out["upper"][i] = upper
        out["lower"][i] = lower
        out["width"][i] = (upper - lower) / mean
        out["percent_b"][i] = (closes[i] - lower) / (upper - lower) if upper != lower else NAN
    return out


def naive_vwap(df: pd.DataFrame, tz: str) -> list[float]:
    out = []
    current_day, pv_sum, v_sum = None, 0.0, 0.0
    for ts, row in df.iterrows():
        day = ts.tz_convert(tz).date()
        if day != current_day:
            current_day, pv_sum, v_sum = day, 0.0, 0.0
        typical = (row["high"] + row["low"] + row["close"]) / 3.0
        pv_sum += typical * row["volume"]
        v_sum += row["volume"]
        out.append(pv_sum / v_sum if v_sum != 0 else NAN)
    return out


def naive_relative_volume(volumes: list[float], period: int) -> list[float]:
    out = [NAN] * len(volumes)
    for i in range(period, len(volumes)):
        avg = sum(volumes[i - period : i]) / period
        out[i] = volumes[i] / avg if avg != 0 else NAN
    return out


# --- valores de referencia (StockCharts ChartSchool) -------------------------

# Planilla "cs-rsi" de StockCharts: RSI de Wilder de 14 períodos.
STOCKCHARTS_RSI_CLOSES = [
    44.3389, 44.0902, 44.1497, 43.6124, 44.2778, 44.8264, 45.0955, 45.4245, 45.8433,
    46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028, 46.0328, 46.4116,
    46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288,
    44.1783, 44.2181, 44.5672, 43.4205, 42.6628, 43.1314,
]  # fmt: skip
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]  # fmt: skip

# Planilla "cs-movavg" de StockCharts: EMA de 10 días.
STOCKCHARTS_EMA_CLOSES = [
    22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
    22.15, 22.39, 22.38, 22.61, 23.36, 24.05, 23.75, 23.83, 23.95, 23.63,
    23.82, 23.87, 23.65, 23.19, 23.10, 23.33, 22.68, 23.10, 22.40, 22.17,
]  # fmt: skip
STOCKCHARTS_EMA10 = [
    22.22, 22.21, 22.24, 22.27, 22.33, 22.52, 22.80, 22.97, 23.13, 23.28, 23.34,
    23.43, 23.51, 23.53, 23.47, 23.40, 23.39, 23.26, 23.23, 23.08, 22.92,
]  # fmt: skip

# Planilla "cs-atr" de StockCharts: ATR de 14 períodos.
STOCKCHARTS_ATR_HIGH = [
    48.70, 48.72, 48.90, 48.87, 48.82, 49.05, 49.20, 49.35, 49.92, 50.19,
    50.12, 49.66, 49.88, 50.19, 50.36, 50.57, 50.65, 50.43, 49.63, 50.33,
    50.29, 50.17, 49.32, 48.50, 48.32, 46.80, 47.80, 48.39, 48.66, 48.79,
]  # fmt: skip
STOCKCHARTS_ATR_LOW = [
    47.79, 48.14, 48.39, 48.37, 48.24, 48.64, 48.94, 48.86, 49.50, 49.87,
    49.20, 48.90, 49.43, 49.73, 49.26, 50.09, 50.30, 49.21, 48.98, 49.61,
    49.20, 49.43, 48.08, 47.64, 41.55, 44.28, 47.31, 47.20, 47.90, 47.73,
]  # fmt: skip
STOCKCHARTS_ATR_CLOSE = [
    48.16, 48.61, 48.75, 48.63, 48.74, 49.03, 49.07, 49.32, 49.91, 50.13,
    49.53, 49.50, 49.75, 50.03, 50.31, 50.52, 50.41, 49.34, 49.37, 50.23,
    49.24, 49.93, 48.43, 48.18, 46.57, 45.41, 47.77, 47.72, 48.62, 47.85,
]  # fmt: skip
STOCKCHARTS_TR = [
    0.91, 0.58, 0.51, 0.50, 0.58, 0.41, 0.26, 0.49, 0.60, 0.32,
    0.93, 0.76, 0.45, 0.46, 1.10, 0.48, 0.35, 1.22, 0.65, 0.96,
    1.09, 0.93, 1.85, 0.86, 6.77, 2.52, 2.39, 1.19, 0.94, 1.06,
]  # fmt: skip
STOCKCHARTS_ATR14 = [
    0.56, 0.59, 0.59, 0.57, 0.62, 0.62, 0.64, 0.67, 0.69,
    0.77, 0.78, 1.21, 1.30, 1.38, 1.37, 1.34, 1.32,
]  # fmt: skip


def stockcharts_atr_frame() -> pd.DataFrame:
    index = pd.date_range("2010-04-01", periods=30, freq="1D", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": STOCKCHARTS_ATR_CLOSE,  # no se usa
            "high": STOCKCHARTS_ATR_HIGH,
            "low": STOCKCHARTS_ATR_LOW,
            "close": STOCKCHARTS_ATR_CLOSE,
            "volume": 1.0,
        },
        index=index,
    )


# --- validación de períodos --------------------------------------------------


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "14", None])
def test_require_period_rejects_invalid(bad: object) -> None:
    with pytest.raises(ValueError):
        require_period(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "call",
    [
        lambda s, df: ema(s, 0),
        lambda s, df: rma(s, -1),
        lambda s, df: sma(s, 0),
        lambda s, df: rsi(s, 0),
        lambda s, df: macd(s, fast=0),
        lambda s, df: atr(df, 0),
        lambda s, df: bollinger(s, 0),
        lambda s, df: relative_volume(df["volume"], 0),
        lambda s, df: daily_range_pct(df, 0),
    ],
)
def test_invalid_period_raises(call) -> None:
    df = random_walk_candles(1, n=50)
    with pytest.raises(ValueError):
        call(df["close"], df)


# --- EMA / SMA / RMA ---------------------------------------------------------


def test_ema_matches_stockcharts_example() -> None:
    result = ema(pd.Series(STOCKCHARTS_EMA_CLOSES), 10)
    assert leading_nans(result) == 9
    # La planilla muestra 2 decimales: tolerancia de medio centavo.
    np.testing.assert_allclose(result.dropna().to_numpy(), STOCKCHARTS_EMA10, atol=0.005 + 1e-9)
    # Cuentas a mano: semilla = SMA de los 10 primeros y un paso de la recursión.
    seed = sum(STOCKCHARTS_EMA_CLOSES[:10]) / 10
    assert result.iloc[9] == pytest.approx(seed, abs=1e-12)
    assert result.iloc[10] == pytest.approx(22.15 * 2 / 11 + seed * 9 / 11, abs=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("period", [1, 9, 20, 50, 200])
def test_ema_matches_naive_loop(seed: int, period: int) -> None:
    close = random_walk_candles(seed)["close"]
    assert_matches(ema(close, period), naive_ema(close.tolist(), period))


def test_ema_of_constant_is_constant() -> None:
    result = ema(pd.Series([101.37] * 300), 20)
    assert leading_nans(result) == 19
    np.testing.assert_allclose(result.dropna().to_numpy(), 101.37, rtol=1e-12)


def test_ema_period_one_is_identity() -> None:
    close = random_walk_candles(3)["close"]
    assert_matches(ema(close, 1), close.tolist())


def test_ema_skips_leading_nans_and_seeds_from_first_valid_values() -> None:
    values = [NAN, NAN, NAN, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    result = ema(pd.Series(values), 3)
    assert leading_nans(result) == 3 + 3 - 1
    assert result.iloc[5] == pytest.approx((10 + 11 + 12) / 3)
    assert result.iloc[6] == pytest.approx(13 * 0.5 + 11 * 0.5)
    assert_matches(result, naive_ema(values, 3))


def test_ema_interior_nan_is_skipped() -> None:
    values = [1.0, 2.0, 3.0, NAN, 4.0, 5.0]
    result = ema(pd.Series(values), 2)
    assert math.isnan(result.iloc[3])
    # Sigue como si la vela con NaN no existiera.
    expected = ema(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 2).to_numpy()
    assert_matches(result.drop(index=3), expected.tolist())


def test_ema_short_series_is_all_nan_with_same_index() -> None:
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, tz="UTC"))
    result = ema(s, 5)
    assert result.index.equals(s.index)
    assert result.isna().all()
    assert ema(pd.Series([], dtype=float), 5).empty


def test_ema_accepts_ints_and_does_not_mutate_input() -> None:
    s = pd.Series([1, 2, 3, 4, 5, 6])
    original = s.copy()
    result = ema(s, 3)
    assert result.dtype == np.float64
    pd.testing.assert_series_equal(s, original)
    assert result.iloc[2] == pytest.approx(2.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_rma_matches_wilder_loop(seed: int) -> None:
    close = random_walk_candles(seed)["close"]
    assert_matches(rma(close, 14), naive_wilder(close.tolist(), 14))


@pytest.mark.parametrize("seed", SEEDS)
def test_sma_matches_naive_loop(seed: int) -> None:
    close = random_walk_candles(seed)["close"].tolist()
    expected = [NAN] * 19 + [sum(close[i - 19 : i + 1]) / 20 for i in range(19, len(close))]
    assert_matches(sma(pd.Series(close), 20), expected)


# --- RSI ---------------------------------------------------------------------


def test_rsi_matches_stockcharts_example() -> None:
    result = rsi(pd.Series(STOCKCHARTS_RSI_CLOSES), 14)
    assert leading_nans(result) == 14
    assert result.iloc[14] == pytest.approx(70.53, abs=0.005)
    np.testing.assert_allclose(result.dropna().to_numpy(), STOCKCHARTS_RSI, atol=0.005 + 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("period", [2, 7, 14, 30])
def test_rsi_matches_naive_loop(seed: int, period: int) -> None:
    close = random_walk_candles(seed)["close"]
    result = rsi(close, period)
    assert_matches(result, naive_rsi(close.tolist(), period))
    assert leading_nans(result) == period


@pytest.mark.parametrize("seed", SEEDS)
def test_rsi_is_bounded(seed: int) -> None:
    values = rsi(random_walk_candles(seed)["close"], 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_rsi_edge_cases() -> None:
    rising = rsi(pd.Series(np.arange(1.0, 41.0)), 14)
    assert leading_nans(rising) == 14
    assert (rising.dropna() == 100.0).all()

    falling = rsi(pd.Series(np.arange(40.0, 0.0, -1.0)), 14)
    assert (falling.dropna() == 0.0).all()

    # Decisión deliberada: precio plano (ni subas ni bajas) = 50.
    flat = rsi(pd.Series([50.0] * 40), 14)
    assert leading_nans(flat) == 14
    assert (flat.dropna() == 50.0).all()

    # Si después de subir se aplana, las subas promedio decaen pero nunca hay
    # bajas: sigue en 100.
    rise_then_flat = rsi(pd.Series(list(np.arange(1.0, 21.0)) + [20.0] * 20), 14)
    assert (rise_then_flat.dropna() == 100.0).all()


def test_rsi_with_leading_nans_starts_period_bars_after_first_valid() -> None:
    close = pd.Series([NAN] * 4 + STOCKCHARTS_RSI_CLOSES)
    result = rsi(close, 14)
    assert leading_nans(result) == 4 + 14
    np.testing.assert_allclose(result.dropna().to_numpy(), STOCKCHARTS_RSI, atol=0.005 + 1e-9)


def test_rsi_short_series_is_all_nan() -> None:
    assert rsi(pd.Series(np.arange(1.0, 15.0)), 14).isna().all()  # 14 cierres no alcanzan
    assert rsi(pd.Series(np.arange(1.0, 16.0)), 14).notna().sum() == 1


# --- MACD --------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_macd_matches_naive_loop(seed: int) -> None:
    close = random_walk_candles(seed)["close"]
    result = macd(close, 12, 26, 9)
    assert list(result.columns) == ["macd", "signal", "hist"]
    assert result.index.equals(close.index)

    fast = naive_ema(close.tolist(), 12)
    slow = naive_ema(close.tolist(), 26)
    line = [f - s for f, s in zip(fast, slow, strict=True)]
    signal = naive_ema(line, 9)  # semilla con los primeros 9 valores válidos de la línea
    hist = [m - s for m, s in zip(line, signal, strict=True)]
    assert_matches(result["macd"], line)
    assert_matches(result["signal"], signal)
    assert_matches(result["hist"], hist)


def test_macd_nan_prefix_lengths() -> None:
    result = macd(random_walk_candles(5)["close"], 12, 26, 9)
    assert leading_nans(result["macd"]) == 25
    assert leading_nans(result["signal"]) == 33
    assert leading_nans(result["hist"]) == 33
    np.testing.assert_allclose(
        result["hist"].dropna(), (result["macd"] - result["signal"]).dropna(), atol=1e-15
    )


def test_macd_of_constant_is_zero() -> None:
    result = macd(pd.Series([42.0] * 100))
    np.testing.assert_allclose(result.dropna().to_numpy(), 0.0, atol=1e-12)


def test_macd_rejects_fast_not_below_slow() -> None:
    with pytest.raises(ValueError):
        macd(pd.Series(np.arange(1.0, 100.0)), fast=26, slow=12)


# --- True range / ATR --------------------------------------------------------


def test_true_range_matches_stockcharts_example() -> None:
    tr = true_range(stockcharts_atr_frame())
    np.testing.assert_allclose(tr.to_numpy(), STOCKCHARTS_TR, atol=0.005 + 1e-9)
    assert tr.iloc[0] == pytest.approx(48.70 - 47.79)


def test_true_range_uses_previous_close_on_gaps() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"high": [10.0, 15.0, 9.0], "low": [9.0, 14.0, 8.5], "close": [9.5, 14.5, 8.8]},
        index=index,
    )
    tr = true_range(df)
    assert tr.tolist() == pytest.approx([1.0, 15.0 - 9.5, 14.5 - 8.5])


def test_atr_matches_stockcharts_example() -> None:
    result = atr(stockcharts_atr_frame(), 14)
    assert leading_nans(result) == 13
    # Cuenta a mano: el primer ATR es el promedio simple de los 14 primeros TR.
    first = sum(STOCKCHARTS_TR[:14]) / 14
    assert result.iloc[13] == pytest.approx(first, abs=1e-12)
    assert result.iloc[14] == pytest.approx((first * 13 + STOCKCHARTS_TR[14]) / 14, abs=1e-12)
    # La planilla publicada redondea a 2 decimales y parte de precios con más
    # decimales que los que muestra: tolerancia de un centavo.
    np.testing.assert_allclose(result.dropna().to_numpy(), STOCKCHARTS_ATR14, atol=0.01 + 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("period", [1, 14, 50])
def test_atr_matches_naive_loop(seed: int, period: int) -> None:
    df = random_walk_candles(seed)
    tr = naive_true_range(df["high"].tolist(), df["low"].tolist(), df["close"].tolist())
    assert_matches(true_range(df), tr)
    result = atr(df, period)
    assert_matches(result, naive_wilder(tr, period))
    assert leading_nans(result) == period - 1


def test_atr_of_constant_range_is_the_range() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    df = pd.DataFrame({"high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    result = atr(df, 14)
    np.testing.assert_allclose(result.dropna().to_numpy(), 2.0, rtol=1e-12)


# --- Bollinger ---------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("period", "mult"), [(20, 2.0), (10, 1.5), (5, 3.0)])
def test_bollinger_matches_naive_loop(seed: int, period: int, mult: float) -> None:
    close = random_walk_candles(seed)["close"]
    result = bollinger(close, period, mult)
    expected = naive_bollinger(close.tolist(), period, mult)
    assert list(result.columns) == ["mid", "upper", "lower", "width", "percent_b"]
    for column in result.columns:
        assert_matches(result[column], expected[column])
    assert leading_nans(result["mid"]) == period - 1


def test_bollinger_uses_population_std() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = bollinger(close, 5, 1.0)
    population = float(np.std(close.to_numpy(), ddof=0))
    assert result["upper"].iloc[-1] == pytest.approx(3.0 + population)
    assert result["upper"].iloc[-1] != pytest.approx(3.0 + float(close.std(ddof=1)))
    # El cierre 5 respecto de bandas 3 ± sqrt(2): %B = (5 - lower) / (upper - lower).
    expected_b = (5.0 - (3.0 - population)) / (2 * population)
    assert result["percent_b"].iloc[-1] == pytest.approx(expected_b)


def test_bollinger_of_constant_has_zero_width() -> None:
    result = bollinger(pd.Series([0.1] * 50), 20, 2.0)
    valid = result.dropna(subset=["mid"])
    assert len(valid) == 31
    assert (valid["width"] == 0.0).all()
    assert (valid["upper"] == valid["lower"]).all()
    # Bandas pegadas: %B no está definido.
    assert valid["percent_b"].isna().all()


# --- rango típico del día ----------------------------------------------------


def test_daily_range_pct_on_constant_range() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    df = pd.DataFrame({"high": 105.0, "low": 95.0, "close": 100.0}, index=index)
    assert daily_range_pct(df, 14) == pytest.approx(10.0)


def test_daily_range_pct_matches_atr_over_close() -> None:
    df = random_walk_candles(9, n=60, freq="1D")
    expected = atr(df, 14).iloc[-1] / df["close"].iloc[-1] * 100
    assert daily_range_pct(df, 14) == pytest.approx(expected, rel=1e-12)


def test_daily_range_pct_is_nan_without_enough_data() -> None:
    df = random_walk_candles(9, n=13, freq="1D")
    assert math.isnan(daily_range_pct(df, 14))
    assert not math.isnan(daily_range_pct(random_walk_candles(9, n=14, freq="1D"), 14))
    assert math.isnan(daily_range_pct(df.iloc[:0], 14))


# --- VWAP diario -------------------------------------------------------------


def _first_bar_of_each_day(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    local_days = index.tz_convert(tz).tz_localize(None).normalize()
    return np.r_[True, local_days[1:] != local_days[:-1]]


def _typical(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("tz", ["UTC", "America/Argentina/Buenos_Aires", "America/New_York"])
def test_vwap_matches_naive_loop(seed: int, tz: str) -> None:
    df = random_walk_candles(seed, n=1200)  # cruza el cambio de horario de EE. UU. (10/3/2024)
    result = vwap_daily(df, tz=tz)
    assert result.index.equals(df.index)
    assert_matches(result, naive_vwap(df, tz))


def test_vwap_resets_at_utc_midnight() -> None:
    df = random_walk_candles(11)
    result = vwap_daily(df)
    first = _first_bar_of_each_day(df.index, "UTC")
    assert (df.index[first][1:].hour == 0).all()
    assert (df.index[first][1:].minute == 0).all()
    # En la primera vela de cada día el VWAP es el precio típico de esa vela.
    np.testing.assert_allclose(result[first], _typical(df)[first], rtol=1e-12)
    # A las 23:45 todavía acumula el día (no es el precio típico de esa vela).
    late = df.index.hour == 23
    assert not np.allclose(result[late], _typical(df)[late])


def test_vwap_resets_at_local_midnight_of_another_timezone() -> None:
    df = random_walk_candles(12)
    tz = "America/Argentina/Buenos_Aires"  # UTC-3, el día arranca 03:00 UTC
    result = vwap_daily(df, tz=tz)
    first = _first_bar_of_each_day(df.index, tz)
    assert (df.index[first][1:].hour == 3).all()
    np.testing.assert_allclose(result[first], _typical(df)[first], rtol=1e-12)
    # A las 00:00 UTC no se reinicia en esta zona horaria.
    utc_midnight = (df.index.hour == 0) & (df.index.minute == 0)
    assert not np.allclose(result[utc_midnight], _typical(df)[utc_midnight])


def test_vwap_resets_at_local_midnight_across_dst() -> None:
    df = random_walk_candles(13, n=1200)
    tz = "America/New_York"
    result = vwap_daily(df, tz=tz)
    first = _first_bar_of_each_day(df.index, tz)
    starts = df.index[first][1:]
    # Antes del cambio de horario el día arranca 05:00 UTC; después, 04:00 UTC.
    assert set(starts[starts < "2024-03-10"].hour) == {5}
    assert set(starts[starts > "2024-03-11"].hour) == {4}
    np.testing.assert_allclose(result[first], _typical(df)[first], rtol=1e-12)


def test_vwap_zero_volume_is_nan_until_volume_appears() -> None:
    index = pd.date_range("2024-01-01 23:00", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [11.0, 12.0, 21.0, 22.0, 23.0, 24.0],
            "low": [9.0, 10.0, 19.0, 20.0, 21.0, 22.0],
            "close": [10.0, 11.0, 20.0, 21.0, 22.0, 23.0],
            "volume": [1.0, 3.0, 0.0, 0.0, 2.0, 2.0],
        },
        index=index,
    )
    result = vwap_daily(df)
    assert result.iloc[0] == pytest.approx(10.0)
    assert result.iloc[1] == pytest.approx((10.0 * 1 + 11.0 * 3) / 4)
    assert math.isnan(result.iloc[2]) and math.isnan(result.iloc[3])  # día nuevo sin volumen
    assert result.iloc[4] == pytest.approx(22.0)
    assert result.iloc[5] == pytest.approx(22.5)


def test_vwap_bar_without_price_is_nan_and_left_out() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "high": [11.0, NAN, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.0, 11.0, 12.0],
            "volume": [1.0, 5.0, 1.0],
        },
        index=index,
    )
    result = vwap_daily(df)
    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx((10.0 + 12.0) / 2)  # la vela sin precio no pesa


def test_vwap_naive_index_is_treated_as_utc() -> None:
    df = random_walk_candles(14, n=200)
    naive = df.copy()
    naive.index = naive.index.tz_localize(None)
    result = vwap_daily(naive)
    assert result.index.equals(naive.index)
    assert_matches(result, vwap_daily(df).tolist())


# --- volumen relativo --------------------------------------------------------


def test_relative_volume_excludes_current_bar() -> None:
    volume = pd.Series([1.0] * 20 + [21.0])
    result = relative_volume(volume, 20)
    # Si la vela actual entrara en el promedio daría 21 / 2 = 10,5.
    assert result.iloc[20] == pytest.approx(21.0)
    assert leading_nans(result) == 20


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("period", [1, 20, 50])
def test_relative_volume_matches_naive_loop(seed: int, period: int) -> None:
    volume = random_walk_candles(seed)["volume"]
    result = relative_volume(volume, period)
    assert_matches(result, naive_relative_volume(volume.tolist(), period))
    assert leading_nans(result) == period
    assert result.index.equals(volume.index)


def test_relative_volume_nan_when_average_is_zero() -> None:
    volume = pd.Series([0.0] * 5 + [3.0, 6.0])
    result = relative_volume(volume, 5)
    assert math.isnan(result.iloc[5])  # promedio previo 0
    assert result.iloc[6] == pytest.approx(6.0 / (3.0 / 5))


def test_relative_volume_nan_when_window_has_missing_data() -> None:
    volume = pd.Series([1.0, NAN, 1.0, 1.0, 1.0, 2.0, 2.0])
    result = relative_volume(volume, 3)
    assert math.isnan(result.iloc[3])  # la ventana previa incluye el NaN
    assert result.iloc[5] == pytest.approx(2.0)


# --- índice preservado en todos los indicadores ------------------------------


@pytest.mark.parametrize(
    "compute",
    [
        lambda df: ema(df["close"], 20),
        lambda df: sma(df["close"], 20),
        lambda df: rma(df["close"], 14),
        lambda df: rsi(df["close"], 14),
        lambda df: macd(df["close"]),
        lambda df: true_range(df),
        lambda df: atr(df, 14),
        lambda df: bollinger(df["close"]),
        lambda df: vwap_daily(df),
        lambda df: relative_volume(df["volume"], 20),
    ],
)
def test_outputs_keep_input_index_and_do_not_mutate(compute) -> None:
    df = random_walk_candles(21, n=120)
    original = df.copy()
    result = compute(df)
    assert result.index.equals(df.index)
    assert result.index.name == "open_time"
    assert len(result) == len(df)
    pd.testing.assert_frame_equal(df, original)


# --- tendencia y posición ----------------------------------------------------


@pytest.mark.parametrize(
    ("close", "ema50", "ema200", "expected"),
    [
        (110.0, 105.0, 100.0, "alcista"),
        (90.0, 95.0, 100.0, "bajista"),
        (102.0, 105.0, 100.0, "lateral"),  # precio entre las EMAs
        (110.0, 100.0, 105.0, "lateral"),  # precio arriba pero EMA50 < EMA200
        (90.0, 100.0, 95.0, "lateral"),  # precio abajo pero EMA50 > EMA200
        (98.0, 95.0, 100.0, "lateral"),
        (105.0, 105.0, 100.0, "lateral"),  # empate precio = EMA50
        (110.0, 100.0, 100.0, "lateral"),  # empate EMA50 = EMA200
        (100.0, 100.0, 100.0, "lateral"),
        (NAN, 105.0, 100.0, "lateral"),
        (110.0, NAN, 100.0, "lateral"),
        (110.0, 105.0, NAN, "lateral"),  # sin EMA200 todavía
        (90.0, 95.0, None, "lateral"),
        (None, None, None, "lateral"),
        (np.float64(110.0), np.float64(105.0), np.float64(100.0), "alcista"),
    ],
)
def test_classify_trend_truth_table(close, ema50, ema200, expected) -> None:
    assert classify_trend(close, ema50, ema200) == expected


def test_classify_trend_series_matches_scalar_version() -> None:
    rng = np.random.default_rng(3)
    n = 500
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="open_time")
    close = pd.Series(rng.normal(100, 5, n), index=index)
    e50 = pd.Series(rng.normal(100, 5, n), index=index)
    e200 = pd.Series(rng.normal(100, 5, n), index=index)
    e200.iloc[:50] = NAN
    e50.iloc[10:20] = NAN
    result = classify_trend_series(close, e50, e200)
    assert result.index.equals(index)
    expected = [classify_trend(c, a, b) for c, a, b in zip(close, e50, e200, strict=True)]
    assert result.tolist() == expected
    assert (result.iloc[:50] == "lateral").all()
    assert set(result) == {"alcista", "bajista", "lateral"}


def test_classify_trend_series_with_real_emas() -> None:
    close = pd.Series(np.arange(1.0, 301.0))  # sube siempre
    result = classify_trend_series(close, ema(close, 50), ema(close, 200))
    assert (result.iloc[:199] == "lateral").all()  # sin EMA200 no hay tendencia
    assert (result.iloc[199:] == "alcista").all()
    falling = close[::-1].reset_index(drop=True)
    result = classify_trend_series(falling, ema(falling, 50), ema(falling, 200))
    assert (result.iloc[199:] == "bajista").all()


def test_classify_trend_series_aligns_emas_by_index() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    close = pd.Series([110.0, 90.0, 100.0], index=index)
    e50 = pd.Series([105.0, 95.0], index=index[:2])  # falta la última
    e200 = pd.Series([100.0, 100.0, 100.0], index=index[::-1])  # orden distinto
    assert classify_trend_series(close, e50, e200).tolist() == ["alcista", "bajista", "lateral"]


@pytest.mark.parametrize(
    ("price", "level", "expected"),
    [
        (101.0, 100.0, "arriba"),
        (99.0, 100.0, "abajo"),
        (100.0, 100.0, "igual"),
        (NAN, 100.0, None),
        (100.0, NAN, None),
        (None, 100.0, None),
        (100.0, None, None),
    ],
)
def test_above_below(price, level, expected) -> None:
    assert above_below(price, level) == expected


def test_distance_pct() -> None:
    assert distance_pct(105.0, 100.0) == pytest.approx(5.0)
    assert distance_pct(95.0, 100.0) == pytest.approx(-5.0)
    assert math.isnan(distance_pct(NAN, 100.0))
    assert math.isnan(distance_pct(100.0, None))
    assert math.isnan(distance_pct(100.0, 0.0))
