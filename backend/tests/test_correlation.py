"""Hito 5: correlación entre monedas y riesgo sumado (``app.safety.correlation``)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.data.synthetic import SyntheticMarketData
from app.exchanges.base import Candles, ExchangeError, FetchMeta
from app.safety.correlation import (
    DEFAULT_MIN_OVERLAP_FRACTION,
    MIN_OVERLAP_FLOOR,
    REASON_HEDGE,
    REASON_HIGH_CORRELATION,
    REASON_LOW_CORRELATION,
    REASON_OPPOSITE_SIDE,
    REASON_SAME_SYMBOL,
    REASON_UNKNOWN_CORRELATION,
    CorrelationMatrix,
    ExcludedSymbol,
    combined_risk_limit,
    correlated_exposure,
    correlation_matrix,
    default_min_overlap,
    expected_returns,
    fetch_correlation_matrix,
    returns_from_candles,
)
from tests.fakes import BINANCE_451, FIXED_NOW, FakeMarketData, ohlcv_frame

H = pd.Timedelta(hours=1)


# --- helpers -------------------------------------------------------------------


def _hours(n: int, start: str = "2026-09-01T00:00:00Z") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="1h", tz="UTC", name="open_time")


def _correlated(
    rho: float, n: int, seed: int = 0, scale: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Dos series de retornos con correlación teórica ``rho``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    y = rho * x + math.sqrt(1 - rho**2) * z
    return x * scale, y * scale


def _closes(returns: np.ndarray, base: float = 100.0) -> np.ndarray:
    return base * np.exp(np.cumsum(returns))


def _matrix(symbols: list[str], values: list[list[float]], **kw) -> CorrelationMatrix:
    """Matriz armada a mano (para valores exactos, como 0,8)."""
    n = len(symbols)
    return CorrelationMatrix(
        symbols=tuple(symbols),
        values=tuple(tuple(float(v) for v in row) for row in values),
        overlaps=tuple(tuple(720 for _ in range(n)) for _ in range(n)),
        min_overlap=360,
        as_of=FIXED_NOW,
        data_start=None,
        data_end=None,
        **kw,
    )


@dataclass(frozen=True)
class Pos:
    symbol: str
    side: str
    risk_usd: float


def _strict_json(obj) -> str:
    return json.dumps(obj, allow_nan=False)


# --- returns_from_candles --------------------------------------------------------------


def test_returns_are_log_returns_of_closed_candles_only():
    closes = [100.0, 101.0, 99.0, 102.0, 103.0]  # la última se está formando
    df = ohlcv_frame(closes, timeframe="1h", end=FIXED_NOW)
    rets = returns_from_candles(df)
    assert rets.name == "log_return"
    assert rets.index.name == "open_time"
    assert str(rets.index.tz) == "UTC"
    assert len(rets) == 3  # 4 velas cerradas → 3 retornos
    expected = np.diff(np.log(closes[:4]))
    assert rets.to_numpy() == pytest.approx(expected, rel=1e-12)
    # Cada retorno lleva la apertura de la vela en la que termina.
    assert list(rets.index) == list(df.index[1:4])


def test_returns_drop_candles_not_closed_at_now_even_if_flagged_closed():
    df = ohlcv_frame([100.0, 101.0, 102.0, 103.0, 104.0], timeframe="1h", forming_last=False)
    # Todas marcadas como cerradas, pero a "now" la última todavía no cerró.
    last_open = df.index[-1].to_pydatetime()
    now = last_open + timedelta(minutes=30)
    assert len(returns_from_candles(df)) == 4
    rets = returns_from_candles(df, now=now)
    assert len(rets) == 3
    assert rets.index[-1] == df.index[-2]
    # Exactamente al cierre ya cuenta.
    assert len(returns_from_candles(df, now=last_open + timedelta(hours=1))) == 4


def test_returns_skip_gaps_between_candles():
    closes = list(_closes(np.full(10, 0.01)))
    df = ohlcv_frame(closes, timeframe="1h", forming_last=False)
    gapped = df.drop(df.index[5])  # falta una vela
    rets = returns_from_candles(gapped)
    # 9 velas → 8 diferencias, pero la que salta el hueco se descarta.
    assert len(rets) == 7
    assert df.index[5] not in rets.index
    assert df.index[6] not in rets.index  # el retorno de 2 h no entra
    assert rets.to_numpy() == pytest.approx(np.full(7, 0.01), rel=1e-9)


def test_returns_drop_invalid_prices_and_duplicates():
    df = ohlcv_frame([100.0, 101.0, 102.0, 103.0, 104.0], timeframe="1h", forming_last=False)
    df.loc[df.index[2], "close"] = 0.0  # precio inválido: se descarta la vela
    rets = returns_from_candles(df)
    assert len(rets) == 2  # 0→1 y 3→4; los que tocan la vela 2 saltan un hueco
    assert np.isfinite(rets.to_numpy()).all()

    dup = pd.concat([df.iloc[[0, 1]], df.iloc[[1]].assign(close=150.0)])
    rets = returns_from_candles(dup)
    assert len(rets) == 1
    assert rets.iloc[0] == pytest.approx(math.log(150.0 / 100.0))  # queda la última


def test_returns_empty_and_errors():
    df = ohlcv_frame([100.0, 101.0], timeframe="1h")  # una sola cerrada
    rets = returns_from_candles(df)
    assert rets.empty and str(rets.index.tz) == "UTC" and rets.dtype == "float64"
    assert returns_from_candles(df.iloc[0:0]).empty

    naive = df.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        returns_from_candles(naive)
    with pytest.raises(ValueError, match="close"):
        returns_from_candles(df.drop(columns=["close"]))
    with pytest.raises(ValueError, match="Timeframe"):
        returns_from_candles(df, timeframe="2h")


# --- correlation_matrix ------------------------------------------------------------------


def test_matrix_recovers_known_correlations():
    n = 5000
    x, y = _correlated(0.9, n, seed=1)
    w = np.random.default_rng(2).normal(0, 0.01, n)
    idx = _hours(n)
    m = correlation_matrix(
        {"BTC": pd.Series(x, idx), "ETH": pd.Series(y, idx), "XRP": pd.Series(w, idx)},
        min_overlap=100,
    )
    assert m.symbols == ("BTC", "ETH", "XRP")
    assert m.method == "pearson"
    assert m.correlation("BTC", "ETH") == pytest.approx(0.9, abs=0.02)
    assert m.correlation("BTC", "XRP") == pytest.approx(0.0, abs=0.05)
    # Exactamente Pearson sobre los mismos datos.
    assert m.correlation("BTC", "ETH") == pytest.approx(np.corrcoef(x, y)[0, 1], rel=1e-12)
    assert m.correlation("ETH", "XRP") == pytest.approx(np.corrcoef(y, w)[0, 1], rel=1e-12)


def test_matrix_is_symmetric_with_unit_diagonal():
    n = 300
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, n)
    idx = _hours(n)
    data = {
        s: pd.Series(base * k + rng.normal(0, 0.01, n), idx)
        for s, k in [("BTC", 1.0), ("ETH", 0.8), ("SOL", 1.5), ("XRP", 0.1)]
    }
    m = correlation_matrix(data, min_overlap=50)
    values = np.asarray(m.values)
    assert values.shape == (4, 4)
    assert np.allclose(values, values.T)
    assert np.diag(values) == pytest.approx(np.ones(4))
    assert (np.abs(values) <= 1).all()
    assert m.overlaps[0][0] == n
    assert np.asarray(m.overlaps).T.tolist() == np.asarray(m.overlaps).tolist()
    frame = m.to_frame()
    assert list(frame.columns) == ["BTC", "ETH", "SOL", "XRP"]
    assert frame.loc["ETH", "SOL"] == frame.loc["SOL", "ETH"]


def test_matrix_is_pairwise_on_common_timestamps():
    n = 400
    x, y = _correlated(0.7, n, seed=4)
    _, z = _correlated(0.5, n, seed=5)
    idx = _hours(n)
    btc = pd.Series(x, idx)
    eth = pd.Series(y, idx).iloc[100:]  # empieza más tarde
    sol = pd.Series(z, idx).iloc[:250]  # termina antes
    m = correlation_matrix({"BTC": btc, "ETH": eth, "SOL": sol}, min_overlap=50)
    assert m.overlap("BTC", "ETH") == 300
    assert m.overlap("BTC", "SOL") == 250
    assert m.overlap("ETH", "SOL") == 150
    assert m.overlap("ETH", "ETH") == 300
    # BTC/ETH usa solo las horas en común (no lo recorta el hueco de SOL).
    assert m.correlation("BTC", "ETH") == pytest.approx(
        np.corrcoef(x[100:], y[100:])[0, 1], rel=1e-12
    )
    assert m.correlation("ETH", "SOL") == pytest.approx(
        np.corrcoef(y[100:250], z[100:250])[0, 1], rel=1e-12
    )
    assert m.data_start == idx[0].to_pydatetime()
    assert m.data_end == idx[-1].to_pydatetime()
    assert m.as_of == m.data_end  # sin as_of: el último retorno


def test_matrix_min_overlap_and_nan_to_none():
    n = 200
    x, y = _correlated(0.95, n, seed=6)
    idx = _hours(n)
    data = {
        "BTC": pd.Series(x, idx),
        "ETH": pd.Series(y, idx),
        "NEW": pd.Series(x[-20:], idx[-20:]),
    }
    m = correlation_matrix(data, min_overlap=21)
    assert m.overlap("BTC", "NEW") == 20
    assert m.correlation("BTC", "NEW") is None
    assert math.isnan(m.values[0][2]) and math.isnan(m.values[2][0])
    assert m.correlation("BTC", "ETH") is not None
    assert m.correlation("NEW", "NEW") == 1.0  # diagonal siempre 1
    assert any("BTC/NEW" in w and "ETH/NEW" in w for w in m.warnings)

    # Exactamente min_overlap alcanza.
    m20 = correlation_matrix(data, min_overlap=20)
    assert m20.correlation("BTC", "NEW") == pytest.approx(1.0)
    assert not m20.warnings

    d = m.to_dict()
    _strict_json(d)  # sin NaN
    assert d["values"][0][2] is None and d["values"][2][0] is None
    assert d["values"][2][2] == 1.0
    pair = next(p for p in d["pairs"] if (p["a"], p["b"]) == ("BTC", "NEW"))
    assert pair == {"a": "BTC", "b": "NEW", "correlation": None, "overlap": 20, "high": None}


def test_matrix_drops_nan_values_and_constant_series_is_unknown():
    n = 100
    x, y = _correlated(0.8, n, seed=7)
    idx = _hours(n)
    y_nan = y.copy()
    y_nan[:10] = np.nan
    y_nan[50] = np.inf
    m = correlation_matrix(
        {"BTC": pd.Series(x, idx), "ETH": pd.Series(y_nan, idx), "USDC": pd.Series(0.0, idx)},
        min_overlap=10,
    )
    assert m.overlap("BTC", "ETH") == 89
    keep = np.isfinite(y_nan)
    assert m.correlation("BTC", "ETH") == pytest.approx(
        np.corrcoef(x[keep], y[keep])[0, 1], rel=1e-12
    )
    assert m.correlation("BTC", "USDC") is None  # precio quieto: correlación indefinida
    assert m.values[2][2] == 1.0


def test_matrix_metadata_threshold_and_pairs():
    n = 500
    x, y = _correlated(0.95, n, seed=8)
    w = np.random.default_rng(9).normal(0, 0.01, n)
    idx = _hours(n)
    as_of = datetime(2026, 9, 25, 12, tzinfo=UTC)
    excluded = [ExcludedSymbol("XRP", "no se pudieron bajar sus velas")]
    m = correlation_matrix(
        {"btc": pd.Series(x, idx), " eth ": pd.Series(y, idx), "AVAX": pd.Series(w, idx)},
        min_overlap=100,
        as_of=as_of,
        timeframe="1h",
        lookback_days=30,
        high_threshold=0.8,
        excluded=excluded,
        warnings=["Datos sintéticos: no son reales"],
    )
    assert m.symbols == ("BTC", "ETH", "AVAX")  # en mayúsculas, en orden
    assert "eth" in m and "XRP" not in m and 3 not in m
    assert m.as_of == as_of
    assert [(p.a, p.b) for p in m.pairs()] == [("BTC", "ETH"), ("BTC", "AVAX"), ("ETH", "AVAX")]
    assert [(p.a, p.b) for p in m.high_pairs()] == [("BTC", "ETH")]
    assert m.high_pairs(threshold=-1.0) == m.pairs()
    assert m.correlation("BTC", "XRP") is None
    assert m.correlation("XRP", "xrp") == 1.0
    assert m.overlap("BTC", "XRP") == 0

    d = m.to_dict()
    _strict_json(d)
    assert d["timeframe"] == "1h" and d["lookback_days"] == 30 and d["min_overlap"] == 100
    assert d["as_of"] == as_of.isoformat()
    assert d["high_threshold"] == 0.8
    assert d["excluded"] == [{"symbol": "XRP", "reason": "no se pudieron bajar sus velas"}]
    assert d["warnings"] == ["Datos sintéticos: no son reales"]
    highs = {(p["a"], p["b"]): p["high"] for p in d["pairs"]}
    assert highs == {("BTC", "ETH"): True, ("BTC", "AVAX"): False, ("ETH", "AVAX"): False}
    assert d["overlaps"][0] == [500, 500, 500]


def test_matrix_validation_and_empty_input():
    idx = _hours(10)
    s = pd.Series(np.arange(10.0), idx)
    with pytest.raises(ValueError, match="min_overlap"):
        correlation_matrix({"BTC": s}, min_overlap=MIN_OVERLAP_FLOOR - 1)
    with pytest.raises(ValueError, match="min_overlap"):
        correlation_matrix({"BTC": s}, min_overlap=5.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dos veces"):
        correlation_matrix({"BTC": s, "btc": s}, min_overlap=3)
    with pytest.raises(ValueError, match="UTC"):
        correlation_matrix(
            {"BTC": pd.Series(np.arange(10.0), idx.tz_localize(None))}, min_overlap=3
        )
    with pytest.raises(ValueError, match="DatetimeIndex"):
        correlation_matrix({"BTC": pd.Series(np.arange(10.0))}, min_overlap=3)

    empty = correlation_matrix({}, min_overlap=3)
    assert empty.symbols == () and empty.values == () and empty.as_of is None
    assert empty.pairs() == ()
    assert _strict_json(empty.to_dict())


def test_default_min_overlap_and_expected_returns():
    assert expected_returns(30, "1h") == 720
    assert expected_returns(30, "4h") == 180
    assert expected_returns(1, "15m") == 96
    assert DEFAULT_MIN_OVERLAP_FRACTION == 0.5
    assert default_min_overlap(30, "1h") == 360
    assert default_min_overlap(30, "1h", 0.75) == 540
    assert default_min_overlap(1, "1d", 0.5) == MIN_OVERLAP_FLOOR  # nunca menos de 3
    with pytest.raises(ValueError):
        default_min_overlap(30, "1h", 0)
    with pytest.raises(ValueError):
        default_min_overlap(30, "1h", 1.5)
    with pytest.raises(ValueError):
        expected_returns(0, "1h")


# --- fetch_correlation_matrix --------------------------------------------------------------


class HistorySource:
    """``MarketDataSource`` mínima: solo velas históricas, por moneda, con fallas a pedido."""

    name = "binance"

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        fail: dict[str, BaseException] | None = None,
        warning: str | None = None,
    ) -> None:
        self.frames = frames
        self.fail = fail or {}
        self.warning = warning
        self.calls: list[tuple] = []

    async def get_klines_history(self, symbol, timeframe, start, end=None) -> Candles:
        self.calls.append((symbol, timeframe, start, end))
        if symbol in self.fail:
            raise self.fail[symbol]
        df = self.frames[symbol]
        mask = df.index >= pd.Timestamp(start)
        if end is not None:
            mask &= df.index <= pd.Timestamp(end)
        meta = FetchMeta("binance", FIXED_NOW, warning=self.warning)
        return Candles(symbol, timeframe, df[mask].copy(), meta)

    async def get_ticker(self, symbol):  # pragma: no cover - no se usa
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - no se usa
        pass


def _frames(n: int = 800, seed: int = 10) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Velas de 1h que terminan en ``FIXED_NOW`` (la última se está formando)."""
    bx, ey = _correlated(0.9, n, seed=seed)
    sol = np.random.default_rng(seed + 1).normal(0, 0.01, n)
    rets = {"BTC": bx, "ETH": ey, "SOL": sol}
    frames = {s: ohlcv_frame(_closes(r), timeframe="1h", end=FIXED_NOW) for s, r in rets.items()}
    return frames, rets


async def test_fetch_builds_matrix_from_closed_candles_in_the_window(settings):
    frames, rets = _frames()
    source = HistorySource(frames, warning="Se usó Bybit porque Binance no respondió")
    m = await fetch_correlation_matrix(
        ["btc", "ETH", "SOL", "BTC"], source, settings, now=FIXED_NOW
    )

    # Un pedido por moneda (sin repetir), desde 30 días + 1 vela antes de la última cerrada.
    assert [c[0] for c in source.calls] == ["BTC", "ETH", "SOL"]
    last_open = datetime(2026, 10, 12, 13, tzinfo=UTC)  # FIXED_NOW = 14:37 → cerró la de 13:00
    assert all(c[1] == settings.correlation.timeframe for c in source.calls)
    assert all(c[2] == last_open - timedelta(hours=720) for c in source.calls)
    assert all(c[3] == FIXED_NOW for c in source.calls)

    assert m.symbols == ("BTC", "ETH", "SOL")
    assert m.timeframe == "1h" and m.lookback_days == 30
    assert m.min_overlap == 360
    assert m.high_threshold == settings.correlation.high_threshold
    assert m.as_of == FIXED_NOW
    assert m.overlap("BTC", "ETH") == 720  # 30 días de 1h
    assert m.data_end == last_open
    assert m.data_start == last_open - timedelta(hours=719)
    # La vela que se forma (índice n-1) no entra: ventana = retornos n-721 .. n-2.
    n = len(rets["BTC"])
    window = slice(n - 721, n - 1)
    expected = np.corrcoef(rets["BTC"][window], rets["ETH"][window])[0, 1]
    assert m.correlation("BTC", "ETH") == pytest.approx(expected, rel=1e-9)
    assert m.correlation("BTC", "ETH") == pytest.approx(0.9, abs=0.05)
    assert abs(m.correlation("BTC", "SOL")) < 0.15
    assert m.excluded == ()
    assert m.warnings == ("Se usó Bybit porque Binance no respondió",)
    assert [(p.a, p.b) for p in m.high_pairs()] == [("BTC", "ETH")]
    _strict_json(m.to_dict())


async def test_fetch_leaves_out_failing_symbols_and_never_crashes(settings):
    frames, _ = _frames()
    frames["AVAX"] = frames["SOL"].iloc[0:0]  # sin velas
    frames["XRP"] = frames["SOL"]
    frames["BAD"] = frames["SOL"].drop(columns=["close"])  # velas con formato inválido
    source = HistorySource(
        frames,
        fail={
            "XRP": BINANCE_451,
            "DOGE": RuntimeError("se cortó todo"),
            "ADA": ExchangeError("respuesta rara"),
        },
    )
    m = await fetch_correlation_matrix(
        ["BTC", "XRP", "ETH", "DOGE", "AVAX", "ADA", "BAD"], source, settings, now=FIXED_NOW
    )
    assert m.symbols == ("BTC", "ETH")
    assert [e.symbol for e in m.excluded] == ["XRP", "DOGE", "AVAX", "ADA", "BAD"]
    reasons = {e.symbol: e.reason for e in m.excluded}
    assert "451" in reasons["XRP"]
    assert "se cortó todo" in reasons["DOGE"]
    assert "velas cerradas" in reasons["AVAX"]
    assert "con errores" in reasons["BAD"]
    for sym in ("XRP", "DOGE", "AVAX", "ADA", "BAD"):
        assert any(w.startswith(f"{sym} quedó fuera") for w in m.warnings)
    assert m.correlation("BTC", "ETH") == pytest.approx(0.9, abs=0.05)
    assert m.correlation("BTC", "XRP") is None
    d = m.to_dict()
    _strict_json(d)
    assert [e["symbol"] for e in d["excluded"]] == ["XRP", "DOGE", "AVAX", "ADA", "BAD"]


async def test_fetch_short_history_gives_unknown_correlation(settings):
    frames, _ = _frames()
    frames["NEW"] = frames["SOL"].iloc[-100:]  # moneda nueva: ~4 días de velas
    source = HistorySource(frames)
    m = await fetch_correlation_matrix(["BTC", "NEW"], source, settings, now=FIXED_NOW)
    assert m.symbols == ("BTC", "NEW")
    assert m.overlap("BTC", "NEW") == 98  # 99 cerradas → 98 retornos
    assert m.correlation("BTC", "NEW") is None
    assert any(w.startswith("De NEW hay solo 98 velas") for w in m.warnings)
    # Con un mínimo explícito más bajo sí hay dato.
    m2 = await fetch_correlation_matrix(
        ["BTC", "NEW"], source, settings, now=FIXED_NOW, min_overlap=50
    )
    assert m2.correlation("BTC", "NEW") is not None


async def test_fetch_ignores_candles_after_now(settings):
    frames, _ = _frames()
    earlier = FIXED_NOW - timedelta(hours=5)

    class Leaky(HistorySource):
        async def get_klines_history(self, symbol, timeframe, start, end=None):
            return await super().get_klines_history(symbol, timeframe, start, None)

    m = await fetch_correlation_matrix(["BTC", "ETH"], Leaky(frames), settings, now=earlier)
    assert m.data_end == datetime(2026, 10, 12, 8, tzinfo=UTC)  # cerrada a las 9:37
    assert m.overlap("BTC", "ETH") == 720


async def test_fetch_uses_correlation_settings(settings):
    cfg = settings.correlation.model_copy(update={"lookback_days": 10, "high_threshold": 0.95})
    small = settings.model_copy(update={"correlation": cfg})
    frames, _ = _frames()
    m = await fetch_correlation_matrix(["BTC", "ETH"], HistorySource(frames), small, now=FIXED_NOW)
    assert m.lookback_days == 10 and m.min_overlap == 120
    assert m.overlap("BTC", "ETH") == 240
    assert m.high_threshold == 0.95
    with pytest.raises(ValueError, match="zona horaria"):
        await fetch_correlation_matrix(
            ["BTC"], HistorySource(frames), small, now=FIXED_NOW.replace(tzinfo=None)
        )


async def test_fetch_with_synthetic_market_end_to_end(settings):
    synthetic = SyntheticMarketData(now=FIXED_NOW, history_days=35)
    source = FakeMarketData(synthetic=synthetic)
    favorites = settings.symbols.favorites
    m = await fetch_correlation_matrix(favorites, source, settings, now=FIXED_NOW)
    assert m.symbols == tuple(favorites)
    assert m.excluded == ()
    values = np.asarray(m.values)
    assert np.isfinite(values).all()
    assert np.allclose(values, values.T)
    assert all(m.overlap(a, b) == 720 for a in favorites for b in favorites)
    _strict_json(m.to_dict())


# --- correlated_exposure -----------------------------------------------------------------


MATRIX = _matrix(
    ["BTC", "ETH", "SOL", "XRP", "NEW"],
    [
        [1.0, 0.92, 0.85, 0.5, math.nan],
        [0.92, 1.0, 0.8, 0.45, math.nan],
        [0.85, 0.8, 1.0, 0.4, math.nan],
        [0.5, 0.45, 0.4, 1.0, math.nan],
        [math.nan, math.nan, math.nan, math.nan, 1.0],
    ],
    high_threshold=0.8,
)


def test_exposure_counts_same_side_correlated_positions():
    positions = [Pos("BTC", "long", 50.0), Pos("SOL", "long", 20.0)]
    exp = correlated_exposure("ETH", "long", positions, MATRIX, 0.8)
    assert [e.symbol for e in exp.counted] == ["BTC", "SOL"]
    assert all(e.reason == REASON_HIGH_CORRELATION for e in exp.counted)
    assert exp.counted[0].correlation == pytest.approx(0.92)
    assert exp.existing_risk_usd == pytest.approx(70.0)
    assert exp.skipped == ()
    assert exp.combined_risk_usd(50.0) == pytest.approx(120.0)
    assert exp.remaining_risk_usd(75.0) == 5.0
    assert exp.remaining_risk_usd(60.0) == 0.0  # nunca negativo
    assert "Suman 70 USD" in exp.explanation
    assert exp.matrix_as_of == FIXED_NOW


def test_exposure_skips_opposite_side_and_only_the_same_coin_is_a_hedge():
    positions = [Pos("BTC", "short", 50.0), Pos("ETH", "short", 30.0)]
    exp = correlated_exposure("ETH", "long", positions, MATRIX, 0.8)
    assert exp.counted == ()
    # Cobertura es solo la misma moneda al revés (CLAUDE.md); BTC va al revés pero es otra.
    assert [e.reason for e in exp.skipped] == [REASON_OPPOSITE_SIDE, REASON_HEDGE]
    assert all(not e.counted for e in exp.skipped)
    assert exp.skipped[1].explanation == (
        "ETH está en short (venta), al revés del plan en la misma moneda: funciona como "
        "cobertura y no suma riesgo."
    )
    assert "cobertura" not in exp.skipped[0].explanation
    assert exp.existing_risk_usd == 0.0
    assert exp.combined_risk_usd(50.0) == 50.0
    assert "No hay posiciones abiertas" in exp.explanation


def test_other_coin_in_the_opposite_direction_is_explained_by_its_correlation():
    """Revisión del Hito 5 (hallazgo 15): otra moneda al revés no es "cobertura"."""
    positions = [
        Pos("ETH", "short", 30.0),  # 0,92 con BTC: se mueve parecido
        Pos("XRP", "short", 10.0),  # 0,5: no se mueve junto
        Pos("NEW", "short", 5.0),  # en la matriz sin dato
        Pos("PEPE", "short", 5.0),  # fuera de la matriz
    ]
    exp = correlated_exposure("BTC", "long", positions, MATRIX, 0.8)
    assert exp.counted == () and exp.existing_risk_usd == 0.0
    assert all(e.reason == REASON_OPPOSITE_SIDE for e in exp.skipped)
    texts = [e.explanation for e in exp.skipped]
    assert texts == [
        "ETH está en short (venta): va al revés del plan y se mueve parecido a BTC "
        "(correlación 0,92): no suma riesgo.",
        "XRP está en short (venta): va al revés del plan y no se mueve junto con BTC "
        "(correlación 0,5): no suma riesgo.",
        "NEW está en short (venta): va al revés del plan (no hay datos para saber si se mueve "
        "junto con BTC): no suma riesgo.",
        "PEPE está en short (venta): va al revés del plan (no hay datos para saber si se mueve "
        "junto con BTC): no suma riesgo.",
    ]
    assert not any("cobertura" in t for t in texts)
    # Sin matriz (falló la descarga), igual: sin dato, sin "cobertura".
    exp2 = correlated_exposure("BTC", "long", [Pos("DOGE", "short", 10.0)], None, 0.8)
    assert exp2.skipped[0].reason == REASON_OPPOSITE_SIDE
    assert "no hay datos" in exp2.skipped[0].explanation
    assert exp2.to_dict()["skipped"][0]["reason"] == "opposite_side"


def test_exposure_counts_same_symbol_even_outside_matrix():
    exp = correlated_exposure("DOGE", "short", [Pos("doge", "SHORT", 40.0)], MATRIX, 0.8)
    assert len(exp.counted) == 1
    entry = exp.counted[0]
    assert entry.symbol == "DOGE" and entry.side == "short"
    assert entry.reason == REASON_SAME_SYMBOL and entry.correlation == 1.0
    assert exp.existing_risk_usd == 40.0
    # Misma moneda pero dirección contraria: cobertura.
    exp2 = correlated_exposure("BTC", "long", [Pos("BTC", "short", 40.0)], MATRIX, 0.8)
    assert exp2.counted == () and exp2.skipped[0].reason == REASON_HEDGE


def test_exposure_counts_unknown_correlation_conservatively():
    positions = [
        Pos("NEW", "long", 10.0),  # en la matriz pero sin dato
        Pos("PEPE", "long", 15.0),  # no está en la matriz
        Pos("XRP", "long", 25.0),  # correlación baja: no suma
    ]
    exp = correlated_exposure("BTC", "long", positions, MATRIX, 0.8)
    assert [e.symbol for e in exp.counted] == ["NEW", "PEPE"]
    assert all(e.reason == REASON_UNKNOWN_CORRELATION for e in exp.counted)
    assert all(e.correlation is None for e in exp.counted)
    assert exp.unknown_symbols == ("NEW", "PEPE")
    assert exp.has_unknown_correlation
    assert exp.existing_risk_usd == 25.0
    assert [e.reason for e in exp.skipped] == [REASON_LOW_CORRELATION]
    assert "por las dudas" in exp.explanation
    assert "por las dudas" in exp.counted[0].explanation

    # La moneda del plan nueva tampoco está en la matriz: todo lo de la misma dirección suma.
    exp2 = correlated_exposure("PEPE", "long", [Pos("BTC", "long", 30.0)], MATRIX, 0.8)
    assert exp2.counted[0].reason == REASON_UNKNOWN_CORRELATION

    # Sin matriz (falló la descarga): igual, salvo la misma moneda y las coberturas.
    exp3 = correlated_exposure(
        "ETH",
        "long",
        [Pos("BTC", "long", 30.0), Pos("ETH", "long", 20.0), Pos("SOL", "short", 5.0)],
        None,
        0.8,
    )
    assert [e.reason for e in exp3.counted] == [REASON_UNKNOWN_CORRELATION, REASON_SAME_SYMBOL]
    assert [e.reason for e in exp3.skipped] == [REASON_OPPOSITE_SIDE]
    assert exp3.existing_risk_usd == 50.0
    assert exp3.matrix_as_of is None


def test_exposure_explanation_never_claims_an_unknown_correlation():
    """Revisión del Hito 5 (hallazgo 8): sin dato no se dice "se mueve junto con"."""
    only_unknown = correlated_exposure("BTC", "long", [Pos("ETH", "long", 40.0)], None, 0.8)
    assert only_unknown.explanation == (
        "Ya tenés una posición en long (compra): ETH (40 USD). No hay datos para saber si ETH "
        "se mueve junto con BTC; por las dudas se cuenta. Su riesgo (40 USD) cuenta junto con "
        "el del plan nuevo porque si el mercado se da vuelta pueden perder las dos juntas."
    )
    mixed = correlated_exposure(
        "ETH",
        "long",
        [Pos("BTC", "long", 50.0), Pos("ETH", "long", 30.0), Pos("NEW", "long", 5.0)],
        MATRIX,
        0.8,
    )
    assert mixed.explanation == (
        "Ya tenés 3 posiciones en long (compra): BTC (50 USD), ETH (30 USD), NEW (5 USD). ETH "
        "es la misma moneda y BTC se mueve junto con ETH. De NEW no hay datos para saber si se "
        "mueve junto con ETH; por las dudas se cuenta. Suman 85 USD de riesgo, que cuenta junto "
        "con el del plan nuevo porque si el mercado se da vuelta pueden perder todas juntas."
    )
    # Con todas conocidas, el texto de siempre (en singular si es una sola).
    known = correlated_exposure("ETH", "long", [Pos("BTC", "long", 50.0)], MATRIX, 0.8)
    assert known.explanation == (
        "Ya tenés una posición en long (compra) que se mueve junto con ETH: BTC (50 USD). Su "
        "riesgo (50 USD) cuenta junto con el del plan nuevo porque si el mercado se da vuelta "
        "pierden las dos juntas."
    )
    same = correlated_exposure("ETH", "long", [Pos("ETH", "long", 20.0)], MATRIX, 0.8)
    assert same.explanation.startswith(
        "Ya tenés una posición en long (compra) en ETH, la misma moneda: ETH (20 USD)."
    )


def test_exposure_threshold_is_inclusive():
    # ETH/SOL = 0,8 exacto: cuenta.
    exp = correlated_exposure("ETH", "long", [Pos("SOL", "long", 20.0)], MATRIX, 0.8)
    assert exp.counted[0].reason == REASON_HIGH_CORRELATION
    # Un redondeo de punto flotante por debajo también cuenta.
    m = _matrix(["A", "B"], [[1.0, 0.8 - 1e-12], [0.8 - 1e-12, 1.0]])
    assert correlated_exposure("A", "long", [Pos("B", "long", 1.0)], m, 0.8).counted
    # Apenas por debajo, no.
    m2 = _matrix(["A", "B"], [[1.0, 0.7999], [0.7999, 1.0]])
    exp2 = correlated_exposure("A", "long", [Pos("B", "long", 1.0)], m2, 0.8)
    assert exp2.counted == () and exp2.skipped[0].reason == REASON_LOW_CORRELATION
    # Con un umbral más bajo, XRP (0,5) pasa a contar.
    exp3 = correlated_exposure("BTC", "long", [Pos("XRP", "long", 1.0)], MATRIX, 0.5)
    assert exp3.counted[0].reason == REASON_HIGH_CORRELATION


def test_exposure_mixed_portfolio_to_dict():
    positions = [
        Pos("BTC", "long", 50.0),
        Pos("ETH", "long", 30.0),
        Pos("SOL", "short", 20.0),
        Pos("XRP", "long", 10.0),
        Pos("NEW", "long", 5.0),
    ]
    exp = correlated_exposure("eth", "Long", positions, MATRIX, 0.8)
    assert exp.new_symbol == "ETH" and exp.new_side == "long"
    assert [(e.symbol, e.reason) for e in exp.counted] == [
        ("BTC", REASON_HIGH_CORRELATION),
        ("ETH", REASON_SAME_SYMBOL),
        ("NEW", REASON_UNKNOWN_CORRELATION),
    ]
    assert [(e.symbol, e.reason) for e in exp.skipped] == [
        ("SOL", REASON_OPPOSITE_SIDE),
        ("XRP", REASON_LOW_CORRELATION),
    ]
    assert exp.existing_risk_usd == 85.0
    d = exp.to_dict()
    _strict_json(d)
    assert d["new_symbol"] == "ETH" and d["threshold"] == 0.8
    assert d["existing_risk_usd"] == 85.0
    assert d["unknown_symbols"] == ["NEW"] and d["has_unknown_correlation"] is True
    assert d["matrix_as_of"] == FIXED_NOW.isoformat()
    assert d["counted"][2] == {
        "symbol": "NEW",
        "side": "long",
        "risk_usd": 5.0,
        "correlation": None,
        "counted": True,
        "reason": REASON_UNKNOWN_CORRELATION,
        "explanation": exp.counted[2].explanation,
    }
    assert d["skipped"][1]["correlation"] == 0.45
    assert isinstance(d["explanation"], str) and "85 USD" in d["explanation"]


def test_exposure_negative_risk_counts_as_zero_and_validation():
    exp = correlated_exposure("ETH", "long", [Pos("BTC", "long", -12.0)], MATRIX, 0.8)
    assert exp.counted[0].risk_usd == 0.0
    assert "stop en ganancia" in exp.counted[0].explanation
    assert exp.existing_risk_usd == 0.0

    assert correlated_exposure("ETH", "long", [], MATRIX, 0.8).counted == ()
    with pytest.raises(ValueError, match="long"):
        correlated_exposure("ETH", "comprar", [], MATRIX, 0.8)
    with pytest.raises(ValueError, match="long"):
        correlated_exposure("ETH", "long", [Pos("BTC", "up", 1.0)], MATRIX, 0.8)
    with pytest.raises(ValueError, match="finito"):
        correlated_exposure("ETH", "long", [Pos("BTC", "long", math.nan)], MATRIX, 0.8)
    with pytest.raises(ValueError, match="umbral"):
        correlated_exposure("ETH", "long", [], MATRIX, 0.0)
    with pytest.raises(ValueError, match="umbral"):
        correlated_exposure("ETH", "long", [], MATRIX, 1.2)
    with pytest.raises(ValueError, match="Símbolo"):
        correlated_exposure("  ", "long", [], MATRIX, 0.8)


def test_exposure_accepts_enum_sides():
    from enum import Enum

    class SideEnum(Enum):
        LONG = "long"
        SHORT = "short"

    exp = correlated_exposure("ETH", SideEnum.LONG, [Pos("BTC", SideEnum.LONG, 10.0)], MATRIX, 0.8)  # type: ignore[arg-type]
    assert exp.new_side == "long" and exp.existing_risk_usd == 10.0


def test_exposure_with_computed_matrix():
    n = 720
    x, y = _correlated(0.95, n, seed=11)
    w = np.random.default_rng(12).normal(0, 0.01, n)
    idx = _hours(n)
    m = correlation_matrix(
        {"BTC": pd.Series(x, idx), "ETH": pd.Series(y, idx), "XRP": pd.Series(w, idx)},
        min_overlap=360,
    )
    exp = correlated_exposure(
        "ETH", "short", [Pos("BTC", "short", 50.0), Pos("XRP", "short", 50.0)], m, 0.8
    )
    assert [e.symbol for e in exp.counted] == ["BTC"]
    assert [e.symbol for e in exp.skipped] == ["XRP"]


# --- combined_risk_limit -------------------------------------------------------------------


def test_combined_risk_limit(settings):
    assert combined_risk_limit(50, 1.5) == 75.0
    assert combined_risk_limit(50, settings.correlation.max_combined_risk_multiplier) == 75.0
    assert combined_risk_limit(0, 1.5) == 0.0
    assert combined_risk_limit(25, 1) == 25.0
    with pytest.raises(ValueError):
        combined_risk_limit(-1, 1.5)
    with pytest.raises(ValueError):
        combined_risk_limit(50, 0.9)
    with pytest.raises(ValueError):
        combined_risk_limit(math.inf, 1.5)


def test_combined_limit_with_exposure_numbers():
    """Ejemplo del pedido: long en BTC (50) + plan long en ETH (50) con tope 75 → pasa el tope."""
    exp = correlated_exposure("ETH", "long", [Pos("BTC", "long", 50.0)], MATRIX, 0.8)
    limit = combined_risk_limit(50, 1.5)
    assert exp.combined_risk_usd(50.0) == 100.0 > limit
    assert exp.remaining_risk_usd(limit) == 25.0


def test_replace_keeps_matrix_frozen():
    with pytest.raises(AttributeError):
        MATRIX.min_overlap = 1  # type: ignore[misc]
    assert replace(MATRIX, high_threshold=0.9).high_threshold == 0.9
