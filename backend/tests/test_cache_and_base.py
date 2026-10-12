"""Caché con TTL y contratos de exchanges."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.data.cache import TTLCache
from app.exchanges.base import (
    DisabledOrderExecutor,
    ExchangeError,
    TradingDisabledError,
    empty_candles_frame,
    validate_candles_frame,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


async def test_cache_respects_ttl():
    clock = FakeClock()
    cache = TTLCache(clock)
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return calls

    r1 = await cache.get_or_fetch("k", 30, fetch)
    clock.advance(10)
    r2 = await cache.get_or_fetch("k", 30, fetch)
    clock.advance(25)
    r3 = await cache.get_or_fetch("k", 30, fetch)
    assert (r1.value, r2.value, r3.value) == (1, 1, 2)
    assert r2.from_cache and not r3.from_cache
    assert r2.fetched_at == r1.fetched_at


async def test_cache_returns_stale_on_error():
    clock = FakeClock()
    cache = TTLCache(clock)

    async def ok():
        return "viejo"

    async def boom():
        raise RuntimeError("sin red")

    await cache.get_or_fetch("k", 30, ok)
    clock.advance(60)
    r = await cache.get_or_fetch("k", 30, boom)
    assert r.value == "viejo" and r.stale and "sin red" in r.error


async def test_cache_propagates_error_without_previous_value():
    cache = TTLCache(FakeClock())

    async def boom():
        raise RuntimeError("sin red")

    with pytest.raises(RuntimeError):
        await cache.get_or_fetch("k", 30, boom)


async def test_disabled_order_executor_always_refuses():
    ex = DisabledOrderExecutor()
    with pytest.raises(TradingDisabledError):
        await ex.place_order("BTC", "long", 1)
    with pytest.raises(TradingDisabledError):
        await ex.set_leverage("BTC", 10)
    with pytest.raises(TradingDisabledError):
        await ex.get_positions()
    with pytest.raises(TradingDisabledError):
        await ex.close_position("BTC")


def test_validate_candles_frame():
    idx = pd.DatetimeIndex(["2026-01-01 00:15", "2026-01-01 00:00", "2026-01-01 00:15"], tz="UTC")
    df = pd.DataFrame(
        {
            "open": [2, 1, 3],
            "high": [2, 1, 3],
            "low": [2, 1, 3],
            "close": [2, 1, 3],
            "volume": [1, 1, 1],
            "closed": [True, True, False],
        },
        index=idx,
    )
    out = validate_candles_frame(df)
    assert list(out["close"]) == [1.0, 3.0]  # ordenado y sin duplicados (queda el último)
    assert out.index.name == "open_time"
    assert len(validate_candles_frame(empty_candles_frame())) == 0
    with pytest.raises(ExchangeError):
        validate_candles_frame(df.drop(columns=["closed"]))
