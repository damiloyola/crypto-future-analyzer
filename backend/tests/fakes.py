"""Fakes en memoria para los tests (sin red).

- ``FakeMarketData``: ``MarketDataSource`` que envuelve ``SyntheticMarketData`` y
  permite hacer fallar métodos o timeframes puntuales, devolver velas vacías,
  demorar cada respuesta y medir cuántas consultas hubo en vuelo a la vez (para
  probar que el snapshot pide todo en paralelo). Reescribe ``FetchMeta.source``
  con su ``name`` (por defecto ``"binance"``) y le pone ``meta_overrides``.
- ``FakePriceSource``: ``PriceSource`` con precio fijo o error (Nexo).
- ``synthetic_extras``: Fear & Greed, mercado global y noticias sintéticos, con
  fallas opcionales.
- ``make_candles`` / ``ohlcv_frame``: velas armadas a mano para casos exactos.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.data.market import GlobalMarket
from app.data.news import NewsResult
from app.data.sentiment import FearGreed
from app.data.snapshot import ExtraSources
from app.data.synthetic import SyntheticMarketData
from app.exchanges.base import (
    TIMEFRAME_MINUTES,
    Candles,
    ExchangeUnavailableError,
    FetchMeta,
    FundingHistory,
    FundingInfo,
    LongShortHistory,
    OpenInterest,
    OpenInterestHistory,
    Ticker,
    Timeframe,
    validate_candles_frame,
)

FIXED_NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)
FIXED_NOW_ISO = "2026-10-12T14:37:20Z"
BINANCE_451 = ExchangeUnavailableError("binance: HTTP 451 en https://fapi.binance.com/fapi/v1/x")


class FakeMarketData:
    """``MarketDataSource`` de mentira sobre el mercado sintético.

    ``fail`` mapea nombre de método (``"get_funding"``) a la excepción a lanzar;
    ``fail_timeframes`` hace lo mismo con ``get_klines`` por timeframe y
    ``empty_timeframes`` devuelve velas vacías para esos timeframes.
    """

    def __init__(
        self,
        *,
        name: str = "binance",
        synthetic: SyntheticMarketData | None = None,
        now: datetime = FIXED_NOW,
        fail: Mapping[str, BaseException] | None = None,
        fail_timeframes: Mapping[str, BaseException] | None = None,
        empty_timeframes: Iterable[str] = (),
        delay_s: float = 0.0,
        meta_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.synthetic = synthetic or SyntheticMarketData(now=now)
        self.fail = dict(fail or {})
        self.fail_timeframes = dict(fail_timeframes or {})
        self.empty_timeframes = set(empty_timeframes)
        self.delay_s = delay_s
        self.meta_overrides = {"warning": None, **dict(meta_overrides or {})}
        self.calls: list[tuple[Any, ...]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.closed = False

    async def _enter(self, method: str, *args: Any) -> None:
        self.calls.append((method, *args))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        finally:
            self.in_flight -= 1
        if method in self.fail:
            raise self.fail[method]

    def _meta(self, meta: FetchMeta) -> FetchMeta:
        return replace(meta, source=self.name, **self.meta_overrides)

    def _wrap(self, value: Any) -> Any:
        return replace(value, meta=self._meta(value.meta))

    async def get_ticker(self, symbol: str) -> Ticker:
        await self._enter("get_ticker", symbol)
        return self._wrap(await self.synthetic.get_ticker(symbol))

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        await self._enter("get_klines", symbol, timeframe, limit)
        if timeframe in self.fail_timeframes:
            raise self.fail_timeframes[timeframe]
        candles = await self.synthetic.get_klines(symbol, timeframe, limit, end_time)
        if timeframe in self.empty_timeframes:
            candles = replace(candles, df=candles.df.iloc[0:0])
        return self._wrap(candles)

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        await self._enter("get_klines_history", symbol, timeframe)
        return self._wrap(await self.synthetic.get_klines_history(symbol, timeframe, start, end))

    async def get_funding(self, symbol: str) -> FundingInfo:
        await self._enter("get_funding", symbol)
        return self._wrap(await self.synthetic.get_funding(symbol))

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        await self._enter("get_funding_history", symbol, limit)
        return self._wrap(await self.synthetic.get_funding_history(symbol, limit))

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        await self._enter("get_open_interest", symbol)
        return self._wrap(await self.synthetic.get_open_interest(symbol))

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        await self._enter("get_open_interest_history", symbol, period, limit)
        return self._wrap(await self.synthetic.get_open_interest_history(symbol, period, limit))

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        await self._enter("get_long_short_ratio", symbol, period, limit)
        return self._wrap(await self.synthetic.get_long_short_ratio(symbol, period, limit))

    async def aclose(self) -> None:
        self.closed = True

    def methods_called(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakePriceSource:
    """``PriceSource`` con precio fijo, o que falla (por defecto, como Nexo apagado)."""

    def __init__(
        self,
        name: str = "nexo",
        price: float | None = None,
        fail: BaseException | None = None,
        now: datetime = FIXED_NOW,
    ) -> None:
        self.name = name
        self.price = price
        self.fail = fail or ExchangeUnavailableError("nexo: sin precio público")
        self.now = now
        self.calls = 0

    async def get_ticker(self, symbol: str) -> Ticker:
        self.calls += 1
        if self.price is None:
            raise self.fail
        return Ticker(symbol.upper(), self.price, FetchMeta(self.name, self.now))

    async def aclose(self) -> None:
        """Nada que cerrar."""


def synthetic_extras(
    market: SyntheticMarketData,
    symbol: str = "BTC",
    *,
    fail: Mapping[str, BaseException] | None = None,
) -> ExtraSources:
    """Fuentes extra sintéticas; ``fail`` = {"fear_greed"|"global_market"|"news": exc}."""
    errors = dict(fail or {})

    async def fear_greed() -> FearGreed:
        if "fear_greed" in errors:
            raise errors["fear_greed"]
        return market.fear_greed(symbol)

    async def global_market() -> GlobalMarket:
        if "global_market" in errors:
            raise errors["global_market"]
        return market.global_market(symbol)

    async def news() -> NewsResult:
        if "news" in errors:
            raise errors["news"]
        return market.headlines(symbol)

    return ExtraSources(fear_greed, global_market, news)


def ohlcv_frame(
    closes: Sequence[float],
    *,
    timeframe: str = "15m",
    end: datetime = FIXED_NOW,
    forming_last: bool = True,
    spread: float = 0.002,
    volume: float | Sequence[float] = 100.0,
) -> pd.DataFrame:
    """Velas a partir de cierres: open = cierre anterior, high/low = ± ``spread``.

    La última vela empieza en el ``timeframe`` que contiene a ``end``; si
    ``forming_last`` es ``False``, todas quedan cerradas.
    """
    minutes = TIMEFRAME_MINUTES[timeframe]
    n = len(closes)
    epoch_min = int(end.timestamp() // 60)
    last_open = datetime.fromtimestamp((epoch_min - epoch_min % minutes) * 60, tz=UTC)
    if not forming_last:
        last_open -= timedelta(minutes=minutes)
    index = pd.date_range(
        end=last_open, periods=n, freq=f"{minutes}min", tz="UTC", name="open_time"
    )
    c = np.asarray(closes, dtype="float64")
    o = np.concatenate([[c[0]], c[:-1]])
    high = np.maximum(o, c) * (1 + spread)
    low = np.minimum(o, c) * (1 - spread)
    vol = np.broadcast_to(np.asarray(volume, dtype="float64"), (n,)).copy()
    closed = np.ones(n, dtype=bool)
    if forming_last and n:
        closed[-1] = False
    df = pd.DataFrame(
        {"open": o, "high": high, "low": low, "close": c, "volume": vol, "closed": closed},
        index=index,
    )
    return validate_candles_frame(df)


def make_candles(
    df: pd.DataFrame, timeframe: str = "15m", symbol: str = "BTC", source: str = "test"
) -> Candles:
    return Candles(symbol, timeframe, df, FetchMeta(source, FIXED_NOW))  # type: ignore[arg-type]
