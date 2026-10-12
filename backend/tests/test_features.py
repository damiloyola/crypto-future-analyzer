"""Tests de las features del detector de régimen (sin red).

Chequeos:

1. Valores calculados a mano sobre un ``DataFrame`` chiquito.
2. Una implementación ingenua con bucles (``math.fsum``, sin numpy ni pandas en
   las cuentas) comparada con el código vectorizado, en velas de 1h y diarias.
3. Causalidad: calcular sobre ``df.iloc[:k]`` da exactamente lo mismo que
   calcular sobre ``df`` y cortar en ``k`` (incluidos los ``NaN``), y cambiar
   las velas posteriores a ``k`` no mueve nada antes de ``k``.
4. Calentamiento, volumen 0 / constante, ``feature_matrix`` y validaciones.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.config import AppSettings
from app.indicators.features import compute_features, feature_matrix, warmup_bars
from app.regime.types import FEATURE_COLUMNS

TOL = 1e-9
SEEDS = [1, 7, 42]


# --- datos sintéticos --------------------------------------------------------


def random_walk_candles(seed: int, n: int = 500, freq: str = "1h") -> pd.DataFrame:
    """Velas OHLCV sintéticas (caminata aleatoria) con la columna ``closed``."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate(([100.0], close[:-1])) * (1 + rng.normal(0.0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.lognormal(mean=3.0, sigma=0.8, size=n)
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "closed": np.ones(n, dtype=bool),
        },
        index=index,
    )


def candles_from(
    close: list[float],
    volume: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    n = len(close)
    high = high if high is not None else [c * 1.01 for c in close]
    low = low if low is not None else [c * 0.99 for c in close]
    index = pd.date_range("2024-05-01", periods=n, freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "closed": [True] * n,
        },
        index=index,
        dtype="float64",
    ).astype({"closed": bool})


# --- implementación ingenua (referencia independiente) -----------------------


def _naive_mean_stdev(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def naive_features(df: pd.DataFrame, rv_window: int, vz_window: int) -> list[list[float]]:
    """Las features fila por fila, con bucles y ``math`` (``NaN`` en el calentamiento)."""
    close = [float(x) for x in df["close"]]
    high = [float(x) for x in df["high"]]
    low = [float(x) for x in df["low"]]
    volume = [float(x) for x in df["volume"]]
    rows: list[list[float]] = []
    returns: list[float] = [math.nan]
    for t in range(1, len(close)):
        returns.append(math.log(close[t] / close[t - 1]))
    for t in range(len(close)):
        if t >= rv_window:
            window = returns[t - rv_window + 1 : t + 1]
            vol = 0.0 if len(set(window)) == 1 else _naive_mean_stdev(window)[1]
        else:
            vol = math.nan
        if t >= vz_window - 1:
            window = [math.log1p(v) for v in volume[t - vz_window + 1 : t + 1]]
            if len(set(window)) == 1:
                z = 0.0
            else:
                mean, sd = _naive_mean_stdev(window)
                z = (math.log1p(volume[t]) - mean) / sd
        else:
            z = math.nan
        rows.append([returns[t], vol, (high[t] - low[t]) / close[t], z])
    return rows


def leading_nan_rows(frame: pd.DataFrame) -> int:
    bad = frame.isna().any(axis=1).to_numpy()
    good = np.flatnonzero(~bad)
    return len(bad) if len(good) == 0 else int(good[0])


# --- valores a mano ----------------------------------------------------------


def test_hand_computed_values():
    e = math.e
    # log1p(volumen) = 0, 1, 2, 2, 2, 0 -> cuentas fáciles a mano.
    volume = [0.0, e - 1, e**2 - 1, e**2 - 1, e**2 - 1, 0.0]
    close = [100.0, 110.0, 99.0, 99.0, 108.9, 108.9]
    high = [101.0, 111.0, 104.0, 100.0, 110.0, 109.9]
    low = [99.0, 100.0, 98.0, 98.0, 108.0, 107.9]
    df = candles_from(close, volume, high, low)

    out = compute_features(df, realized_vol_window=2, volume_zscore_window=3, smoothing_window=1)

    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert out.index.equals(df.index)
    ln11, ln09 = math.log(1.1), math.log(0.9)
    sqrt2 = math.sqrt(2.0)
    nan = math.nan
    expected = {
        "log_return": [nan, ln11, ln09, 0.0, ln11, 0.0],
        # desvío muestral de dos datos = |a - b| / raíz(2)
        "realized_vol": [
            nan,
            nan,
            abs(ln11 - ln09) / sqrt2,
            abs(ln09) / sqrt2,
            ln11 / sqrt2,
            ln11 / sqrt2,
        ],
        "range_pct": [2 / 100, 11 / 110, 6 / 99, 2 / 99, 2 / 108.9, 2 / 108.9],
        # [0,1,2]: media 1, desvío 1 -> z = 1
        # [1,2,2]: media 5/3, desvío raíz(1/3) -> z = (1/3)/raíz(1/3) = 1/raíz(3)
        # [2,2,2]: constante -> z = 0
        # [2,2,0]: media 4/3, desvío raíz(4/3) -> z = (-4/3)/raíz(4/3) = -raíz(4/3)
        "volume_z": [nan, nan, 1.0, 1 / math.sqrt(3), 0.0, -math.sqrt(4 / 3)],
    }
    for col, values in expected.items():
        np.testing.assert_allclose(out[col].to_numpy(), values, rtol=1e-12, atol=1e-12)


def test_constant_returns_give_exactly_zero_volatility():
    close = [100.0 * 1.01**i for i in range(10)]  # retorno constante (salvo redondeo)
    flat = [50.0] * 10  # precio quieto: retorno exactamente 0
    out_flat = compute_features(candles_from(flat, [1.0] * 10), 3, 2, smoothing_window=1)
    assert (out_flat["realized_vol"].iloc[3:] == 0.0).all()
    assert (out_flat["log_return"].iloc[1:] == 0.0).all()
    out_growth = compute_features(candles_from(close, [1.0] * 10), 3, 2, smoothing_window=1)
    assert np.all(out_growth["realized_vol"].iloc[3:].to_numpy() < 1e-12)


@pytest.mark.parametrize("freq", ["1h", "1D"])
@pytest.mark.parametrize("seed", SEEDS)
def test_matches_naive_reference(seed: int, freq: str):
    df = random_walk_candles(seed, n=400, freq=freq)
    out = compute_features(df, realized_vol_window=24, volume_zscore_window=168, smoothing_window=1)
    expected = np.asarray(naive_features(df, 24, 168), dtype=float)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=TOL, atol=TOL)


# --- causalidad --------------------------------------------------------------


@pytest.mark.parametrize(("rv_window", "vz_window"), [(24, 168), (3, 5), (2, 2)])
def test_prefix_gives_exactly_the_same_values(rv_window: int, vz_window: int):
    df = random_walk_candles(3, n=300)
    full = compute_features(df, rv_window, vz_window, smoothing_window=1).to_numpy()
    for k in range(0, len(df) + 1):
        prefix = compute_features(df.iloc[:k], rv_window, vz_window, smoothing_window=1).to_numpy()
        assert prefix.shape == (k, len(FEATURE_COLUMNS))
        # Igualdad exacta (bit a bit), con los NaN en las mismas posiciones.
        assert np.array_equal(prefix, full[:k], equal_nan=True), f"cambió con k={k}"


def test_future_rows_do_not_change_the_past():
    df = random_walk_candles(11, n=400)
    base = compute_features(df, smoothing_window=1).to_numpy()
    rng = np.random.default_rng(0)
    for k in (1, 24, 25, 167, 168, 169, 250, 399):
        altered = df.copy()
        future = altered.index[k:]
        # Futuro completamente distinto: precios x5 y volumen x100.
        altered.loc[future, ["open", "high", "low", "close"]] *= 5.0 * (
            1 + rng.random((len(future), 1))
        )
        altered.loc[future, "volume"] *= 100.0
        out = compute_features(altered, smoothing_window=1).to_numpy()
        assert np.array_equal(out[:k], base[:k], equal_nan=True), f"cambió con k={k}"


def test_does_not_modify_input():
    df = random_walk_candles(5, n=200)
    snapshot = df.copy(deep=True)
    compute_features(df, smoothing_window=1)
    feature_matrix(df, smoothing_window=1)
    pd.testing.assert_frame_equal(df, snapshot)


# --- calentamiento -----------------------------------------------------------


@pytest.mark.parametrize(
    ("rv_window", "vz_window", "expected"),
    [(24, 168, 167), (2, 2, 2), (10, 5, 10), (5, 10, 9), (24, 25, 24), (24, 26, 25)],
)
def test_warmup_length(rv_window: int, vz_window: int, expected: int):
    assert warmup_bars(rv_window, vz_window, smoothing_window=1) == expected
    df = random_walk_candles(2, n=300)
    out = compute_features(df, rv_window, vz_window, smoothing_window=1)
    assert leading_nan_rows(out) == expected
    assert np.isfinite(out.iloc[expected:].to_numpy()).all()
    assert out["log_return"].isna().sum() == 1
    assert out["realized_vol"].isna().sum() == rv_window
    assert out["range_pct"].isna().sum() == 0
    assert out["volume_z"].isna().sum() == vz_window - 1


def test_warmup_matches_config(settings: AppSettings):
    cfg = settings.regime.features
    assert warmup_bars(
        cfg.realized_vol_window, cfg.volume_zscore_window, smoothing_window=1
    ) == max(cfg.realized_vol_window, cfg.volume_zscore_window - 1)


def test_shorter_than_warmup_is_all_nan():
    df = random_walk_candles(4, n=30)
    out = compute_features(df, 24, 168, smoothing_window=1)
    assert out["realized_vol"].isna().sum() == 24
    assert out["volume_z"].isna().all()
    X, index = feature_matrix(df, 24, 168, smoothing_window=1)
    assert X.shape == (0, len(FEATURE_COLUMNS))
    assert len(index) == 0


def test_empty_frame():
    df = random_walk_candles(4, n=1).iloc[:0]
    out = compute_features(df, smoothing_window=1)
    assert out.shape == (0, len(FEATURE_COLUMNS))
    assert list(out.columns) == list(FEATURE_COLUMNS)
    X, index = feature_matrix(df, smoothing_window=1)
    assert X.shape == (0, len(FEATURE_COLUMNS))
    assert isinstance(index, pd.DatetimeIndex)


# --- volumen 0 y constante ---------------------------------------------------


def test_zero_volume_window_gives_zero_z():
    n = 12
    df = candles_from([100.0 + i for i in range(n)], [0.0] * n)
    out = compute_features(df, realized_vol_window=2, volume_zscore_window=4, smoothing_window=1)
    z = out["volume_z"].to_numpy()
    assert np.isnan(z[:3]).all()
    assert (z[3:] == 0.0).all()


def test_constant_volume_window_gives_zero_z():
    n = 12
    df = candles_from([100.0] * n, [1234.5] * n)
    out = compute_features(df, realized_vol_window=2, volume_zscore_window=5, smoothing_window=1)
    assert (out["volume_z"].iloc[4:] == 0.0).all()


def test_spike_after_constant_volume():
    # V - 1 velas iguales y un pico al final: z = (V - 1) / raíz(V), el máximo posible.
    window = 6
    volume = [10.0] * 8 + [500.0]
    df = candles_from([100.0] * len(volume), volume)
    out = compute_features(
        df, realized_vol_window=2, volume_zscore_window=window, smoothing_window=1
    )
    assert out["volume_z"].iloc[-1] == pytest.approx((window - 1) / math.sqrt(window), rel=1e-12)
    assert (out["volume_z"].iloc[window - 1 : -1] == 0.0).all()
    # Y una vela sin volumen tras volumen constante da el mínimo posible.
    volume_drop = [10.0] * 8 + [0.0]
    out_drop = compute_features(
        candles_from([100.0] * 9, volume_drop), 2, window, smoothing_window=1
    )
    assert out_drop["volume_z"].iloc[-1] == pytest.approx(
        -(window - 1) / math.sqrt(window), rel=1e-12
    )


def test_trading_resumes_after_zero_volume():
    volume = [0.0] * 6 + [5.0, 7.0, 6.0, 8.0]
    df = candles_from([100.0 + i for i in range(len(volume))], volume)
    out = compute_features(df, realized_vol_window=2, volume_zscore_window=4, smoothing_window=1)
    z = out["volume_z"].to_numpy()
    assert np.isfinite(z[3:]).all()
    assert z[6] > 0  # primera vela con volumen: muy por encima de lo habitual


# --- feature_matrix ----------------------------------------------------------


@pytest.mark.parametrize("freq", ["1h", "1D"])
def test_feature_matrix_drops_exactly_the_warmup(settings: AppSettings, freq: str):
    cfg = settings.regime.features
    rv, vz = cfg.realized_vol_window, cfg.volume_zscore_window
    df = random_walk_candles(9, n=450, freq=freq)
    warm = warmup_bars(rv, vz, smoothing_window=1)

    X, index = feature_matrix(df, rv, vz, smoothing_window=1)

    assert X.dtype == np.float64
    assert X.shape == (len(df) - warm, len(FEATURE_COLUMNS))
    assert X.flags["C_CONTIGUOUS"]
    assert np.isfinite(X).all()
    assert isinstance(index, pd.DatetimeIndex)
    assert index.equals(df.index[warm:])
    full = compute_features(df, rv, vz, smoothing_window=1).to_numpy()
    assert np.array_equal(X, full[warm:])
    expected = np.asarray(naive_features(df, rv, vz), dtype=float)[warm:]
    np.testing.assert_allclose(X, expected, rtol=TOL, atol=TOL)


def test_feature_matrix_daily_index_is_daily():
    df = random_walk_candles(8, n=260, freq="1D")
    X, index = feature_matrix(df, 24, 168, smoothing_window=1)
    assert len(X) == 260 - 167
    assert (index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert str(index.tz) == "UTC"


def test_accepts_frame_without_closed_column():
    df = random_walk_candles(6, n=200).drop(columns=["closed", "open"])
    X, _ = feature_matrix(df, 24, 168, smoothing_window=1)
    assert X.shape == (200 - 167, 4)


# --- validaciones ------------------------------------------------------------


def test_missing_columns_raise():
    df = random_walk_candles(1, n=50).drop(columns=["volume", "low"])
    with pytest.raises(ValueError, match="faltan columnas.*low.*volume"):
        feature_matrix(df, smoothing_window=1)
    with pytest.raises(ValueError, match="faltan columnas"):
        compute_features(df, smoothing_window=1)


@pytest.mark.parametrize("col", ["close", "high", "low"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_prices_raise(col: str, bad: float):
    df = random_walk_candles(1, n=50)
    df.loc[df.index[10], col] = bad
    with pytest.raises(ValueError, match="precios"):
        feature_matrix(df, smoothing_window=1)


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_volume_raises(bad: float):
    df = random_walk_candles(1, n=50)
    df.loc[df.index[10], "volume"] = bad
    with pytest.raises(ValueError, match="volume"):
        compute_features(df, smoothing_window=1)


def test_unclosed_candle_raises():
    df = random_walk_candles(1, n=50)
    df.loc[df.index[-1], "closed"] = False
    with pytest.raises(ValueError, match="sin cerrar"):
        compute_features(df, smoothing_window=1)


def test_index_must_be_sorted_unique_datetime():
    df = random_walk_candles(1, n=50)
    with pytest.raises(ValueError, match="ordenadas"):
        compute_features(df.iloc[::-1], smoothing_window=1)
    with pytest.raises(ValueError, match="repetidas"):
        compute_features(pd.concat([df.iloc[:10], df.iloc[9:]]), smoothing_window=1)
    with pytest.raises(ValueError, match="índice de fechas"):
        compute_features(df.reset_index(drop=True), smoothing_window=1)


@pytest.mark.parametrize("bad", [1, 0, -3, 2.5, True, "24"])
def test_invalid_windows_raise(bad: object):
    df = random_walk_candles(1, n=50)
    with pytest.raises(ValueError, match="realized_vol_window"):
        compute_features(df, realized_vol_window=bad, smoothing_window=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="volume_zscore_window"):
        feature_matrix(df, volume_zscore_window=bad, smoothing_window=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        warmup_bars(bad, 168, smoothing_window=1)  # type: ignore[arg-type]


# --- suavizado (decisión del usuario: resumen de las últimas 24 velas) --------


@pytest.mark.parametrize("sm", [2, 5, 24])
def test_smoothed_values_match_definition(sm: int):
    df = random_walk_candles(21, n=400)
    raw = compute_features(df, 24, 168, smoothing_window=1)
    smooth = compute_features(df, 24, 168, smoothing_window=sm)
    close = df["close"].to_numpy()
    # Retorno acumulado de N velas = ln(close[t] / close[t-N]).
    expected_ret = np.full(len(df), np.nan)
    expected_ret[sm:] = np.log(close[sm:] / close[:-sm])
    np.testing.assert_allclose(smooth["log_return"], expected_ret, rtol=1e-12, atol=1e-12)
    # Rango y volumen: promedios de las últimas N velas; la volatilidad no se toca.
    np.testing.assert_allclose(
        smooth["range_pct"], raw["range_pct"].rolling(sm).mean(), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        smooth["volume_z"], raw["volume_z"].rolling(sm).mean(), rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_series_equal(smooth["realized_vol"], raw["realized_vol"])


def test_smoothing_one_is_the_literal_spec():
    df = random_walk_candles(3, n=300)
    out = compute_features(df, 24, 168, smoothing_window=1)
    close = df["close"].to_numpy()
    np.testing.assert_allclose(out["log_return"].iloc[1:], np.log(close[1:] / close[:-1]))
    raw_range = (df["high"] - df["low"]) / df["close"]
    np.testing.assert_allclose(out["range_pct"], raw_range, rtol=1e-15)


def test_smoothing_window_has_no_default():
    # Regresión: el valor por omisión era 1 (sin suavizar) mientras la config y los
    # modelos guardados usan 24. Un modelo suavizado aceptaba esas features sin
    # protestar y el estado más probable coincidía con el correcto en ~8 % de las
    # velas. Ahora hay que decir siempre el suavizado, por nombre.
    df = random_walk_candles(3, n=300)
    with pytest.raises(TypeError, match="smoothing_window"):
        compute_features(df, 24, 168)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="smoothing_window"):
        feature_matrix(df)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="smoothing_window"):
        warmup_bars(24, 168)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        warmup_bars(24, 168, 24)  # type: ignore[misc]  # ni en posición


@pytest.mark.parametrize(("rv", "vz", "sm"), [(24, 168, 24), (24, 168, 1), (5, 3, 8), (10, 4, 2)])
def test_smoothed_warmup_and_causality(rv: int, vz: int, sm: int):
    df = random_walk_candles(5, n=320)
    full = compute_features(df, rv, vz, smoothing_window=sm)
    warm = warmup_bars(rv, vz, smoothing_window=sm)
    assert leading_nan_rows(full) == warm
    assert np.isfinite(full.iloc[warm:].to_numpy()).all()
    # Causal: calcular sobre un prefijo da exactamente lo mismo, bit a bit.
    for k in (1, warm - 1, warm, warm + 1, 250, 320):
        if k < 1:
            continue
        pd.testing.assert_frame_equal(
            compute_features(df.iloc[:k], rv, vz, smoothing_window=sm), full.iloc[:k]
        )
    # Cambiar el futuro no cambia el pasado.
    changed = df.copy()
    changed.iloc[200:, changed.columns.get_loc("close")] *= 3.0
    changed.iloc[200:, changed.columns.get_loc("high")] *= 3.0
    changed.iloc[200:, changed.columns.get_loc("low")] *= 3.0
    changed.iloc[200:, changed.columns.get_loc("volume")] *= 50.0
    pd.testing.assert_frame_equal(
        compute_features(changed, rv, vz, smoothing_window=sm).iloc[:200], full.iloc[:200]
    )


def test_feature_matrix_with_config_smoothing(settings: AppSettings):
    f = settings.regime.features
    assert f.smoothing_window == 24  # decisión del usuario
    df = random_walk_candles(9, n=600)
    X, index = feature_matrix(
        df, f.realized_vol_window, f.volume_zscore_window, smoothing_window=f.smoothing_window
    )
    warm = warmup_bars(
        f.realized_vol_window, f.volume_zscore_window, smoothing_window=f.smoothing_window
    )
    assert warm == 190
    assert X.shape == (600 - warm, 4) and index.equals(df.index[warm:])


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, None])
def test_invalid_smoothing_raises(bad: object):
    df = random_walk_candles(1, n=50)
    with pytest.raises(ValueError):
        compute_features(df, 24, 168, smoothing_window=bad)  # type: ignore[arg-type]
