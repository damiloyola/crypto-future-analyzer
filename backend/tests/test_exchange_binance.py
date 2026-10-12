"""Conector de Binance: parseo de cada endpoint, velas cerradas, paginación y errores.

Sin red: respx con los fixtures de ``tests/fixtures/binance`` o respuestas
sintéticas armadas en el test.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.exchanges.base import (
    ExchangeError,
    ExchangeUnavailableError,
    MarketDataSource,
)
from app.exchanges.binance import (
    BinanceMarketData,
    history_windows,
    ms_to_utc,
    page_backward,
    pair_symbol,
    timeframe_ms,
    utc_to_ms,
)
from app.http_client import make_client

FIXTURES = Path(__file__).parent / "fixtures" / "binance"
FIXTURE_NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)  # ver fixtures/README.md
FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
M15 = 15 * 60_000


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
async def client():
    async with make_client() as c:
        yield c


def make(settings, client, now: datetime = FIXTURE_NOW, **kwargs) -> BinanceMarketData:
    kwargs.setdefault("retries", 0)
    kwargs.setdefault("backoff_s", 0)
    return BinanceMarketData(settings.exchanges, client, clock=lambda: now, **kwargs)


def kline_server(first_open_ms: int, last_open_ms: int, step_ms: int, overlap: bool = False):
    """Imita /klines: con startTime cuenta desde el principio; sin él, desde el final.

    ``overlap`` agrega la vela anterior a ``startTime`` para probar el dedup.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        limit = int(p["limit"])
        opens = list(range(first_open_ms, last_open_ms + 1, step_ms))
        start = int(p["startTime"]) if "startTime" in p else None
        end = int(p["endTime"]) if "endTime" in p else None
        if end is not None:
            opens = [o for o in opens if o <= end]
        if start is not None:
            selected = [o for o in opens if o >= start][:limit]
            if overlap and selected and selected[0] - step_ms >= first_open_ms:
                selected.insert(0, selected[0] - step_ms)
        else:
            selected = opens[-limit:]
        rows = [
            [
                o,
                f"{100 + (o - first_open_ms) / step_ms:.2f}",
                f"{101 + (o - first_open_ms) / step_ms:.2f}",
                f"{99 + (o - first_open_ms) / step_ms:.2f}",
                f"{100.5 + (o - first_open_ms) / step_ms:.2f}",
                "1.5",
                o + step_ms - 1,
                "150",
                10,
                "0.7",
                "70",
                "0",
            ]
            for o in selected
        ]
        return httpx.Response(200, json=rows)

    return handler


# --- helpers -----------------------------------------------------------------


def test_helpers():
    assert pair_symbol(" btc ", "usdt") == "BTCUSDT"
    with pytest.raises(ValueError):
        pair_symbol("BTC/USDT", "USDT")
    assert timeframe_ms("15m") == M15
    with pytest.raises(ValueError, match="Timeframe"):
        timeframe_ms("1s")
    assert ms_to_utc("1791815840000") == FIXTURE_NOW
    assert utc_to_ms(FIXTURE_NOW) == 1791815840000
    with pytest.raises(ValueError):
        utc_to_ms(datetime(2026, 1, 1))  # sin zona horaria
    assert history_windows(0, 25, 10, 1) == [(0, 9), (10, 19), (20, 25)]
    assert history_windows(10, 5, 1, 10) == []


async def test_page_backward_dedups_and_orders():
    data = list(range(0, 100, 10))  # tiempos 0..90

    async def fetch(n: int, before: int | None) -> list[int]:
        page = [t for t in data if before is None or t <= before][-n:]
        if before is not None:
            page += [t for t in data if t > before][:1]  # uno repetido de la página anterior
        return list(reversed(page))  # y en orden inverso, como Bybit

    assert await page_backward(fetch, 7, 3, lambda t: t) == [30, 40, 50, 60, 70, 80, 90]
    assert await page_backward(fetch, 50, 4, lambda t: t) == data  # se corta cuando no hay más
    with pytest.raises(ValueError):
        await page_backward(fetch, 0, 4, lambda t: t)


def test_implements_protocol(settings, client):
    assert isinstance(make(settings, client), MarketDataSource)


async def test_aclose_only_closes_own_client(settings, client):
    await make(settings, client).aclose()
    assert not client.is_closed
    owned = BinanceMarketData(settings.exchanges)
    await owned.aclose()
    assert owned._client.is_closed


# --- velas -------------------------------------------------------------------


async def test_klines_futures_parsing(settings, client, respx_mock):
    route = respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        return_value=httpx.Response(200, json=load("klines.json"))
    )
    candles = await make(settings, client).get_klines("btc", "15m", limit=8)

    params = route.calls.last.request.url.params
    assert (params["symbol"], params["interval"], params["limit"]) == ("BTCUSDT", "15m", "8")
    assert "endTime" not in params
    df = candles.df
    assert candles.symbol == "BTC" and candles.timeframe == "15m"
    assert len(df) == 8
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[0] == pd.Timestamp("2026-10-12 12:45", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2026-10-12 14:30", tz="UTC")
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "closed"]
    assert df["close"].dtype == "float64" and df["closed"].dtype == bool
    last = df.iloc[-1]
    assert (last["open"], last["high"], last["low"], last["close"]) == (
        108944.9,
        108990.0,
        108910.2,
        108962.5,
    )
    assert last["volume"] == pytest.approx(121.305)  # moneda base, no USDT
    # Solo la vela de 14:30 (todavía abierta a las 14:37:20) queda sin cerrar.
    assert df["closed"].tolist() == [True] * 7 + [False]
    assert len(candles.closed_only()) == 7
    assert candles.meta.source == "binance" and candles.meta.fetched_at == FIXTURE_NOW
    assert not candles.meta.fallback and candles.meta.warning is None


@pytest.mark.parametrize(
    ("now", "last_closed"),
    [
        (datetime(2026, 10, 12, 14, 44, 59, 999000, tzinfo=UTC), False),  # = closeTime
        (datetime(2026, 10, 12, 14, 45, tzinfo=UTC), True),
    ],
)
async def test_klines_closed_flag_edges(settings, client, respx_mock, now, last_closed):
    respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        return_value=httpx.Response(200, json=load("klines.json"))
    )
    df = (await make(settings, client, now=now).get_klines("BTC", "15m", limit=8)).df
    assert bool(df["closed"].iloc[-1]) is last_closed
    assert df["closed"].iloc[:-1].all()


async def test_klines_spot_when_not_perpetual(settings, client, respx_mock):
    binance_cfg = settings.exchanges.binance.model_copy(update={"use_perpetual_klines": False})
    exchanges = settings.exchanges.model_copy(update={"binance": binance_cfg})
    route = respx_mock.get(f"{SPOT}/api/v3/klines").mock(
        return_value=httpx.Response(200, json=load("spot_klines.json"))
    )
    md = BinanceMarketData(exchanges, client, retries=0, clock=lambda: FIXTURE_NOW)
    candles = await md.get_klines("BTC", "1h", limit=4)

    assert route.call_count == 1
    assert route.calls.last.request.url.params["interval"] == "1h"
    df = candles.df
    assert df.index[0] == pd.Timestamp("2026-10-12 11:00", tz="UTC")
    assert df["close"].iloc[-1] == pytest.approx(108958.31)
    assert df["closed"].tolist() == [True, True, True, False]  # 14:00 sigue abierta


async def test_klines_with_end_time_sends_end_and_all_closed(settings, client, respx_mock):
    end = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    route = respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        side_effect=kline_server(utc_to_ms(end) - 99 * M15, utc_to_ms(FIXTURE_NOW), M15)
    )
    candles = await make(settings, client).get_klines("BTC", "15m", limit=10, end_time=end)
    assert route.calls.last.request.url.params["endTime"] == str(utc_to_ms(end))
    assert len(candles.df) == 10
    assert candles.df.index[-1] == pd.Timestamp(end)
    assert candles.df["closed"].all()


async def test_klines_more_than_max_pages_backward(settings, client, respx_mock):
    last_open = utc_to_ms(datetime(2026, 10, 12, 14, 30, tzinfo=UTC))
    route = respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        side_effect=kline_server(last_open - 4999 * M15, last_open, M15)
    )
    df = (await make(settings, client).get_klines("BTC", "15m", limit=2000)).df

    assert route.call_count == 2
    first, second = (c.request.url.params for c in route.calls)
    assert first["limit"] == "1500" and "endTime" not in first
    assert second["limit"] == "500"
    assert int(second["endTime"]) == last_open - 1499 * M15 - 1
    assert len(df) == 2000 and df.index.is_unique and df.index.is_monotonic_increasing
    assert df.index[-1] == pd.Timestamp("2026-10-12 14:30", tz="UTC")
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert df["closed"].tolist().count(False) == 1


async def test_klines_history_pages_forward_dedups_and_stops_at_now(settings, client, respx_mock):
    current_open = utc_to_ms(datetime(2026, 10, 12, 14, 30, tzinfo=UTC))
    start_ms = current_open - 1999 * M15  # 2000 velas hasta la que se forma
    route = respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        side_effect=kline_server(start_ms - 50 * M15, current_open, M15, overlap=True)
    )
    candles = await make(settings, client).get_klines_history("BTC", "15m", ms_to_utc(start_ms))

    assert route.call_count == 2  # ventanas de 1500 velas
    first, second = (c.request.url.params for c in route.calls)
    assert int(first["startTime"]) == start_ms and first["limit"] == "1500"
    assert int(first["endTime"]) == start_ms + 1500 * M15 - 1
    assert int(second["startTime"]) == start_ms + 1500 * M15
    assert int(second["endTime"]) == utc_to_ms(FIXTURE_NOW)  # corta en "ahora"
    df = candles.df
    assert len(df) == 2000  # la vela repetida entre páginas aparece una sola vez
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp(ms_to_utc(start_ms))
    assert df["closed"].tolist() == [True] * 1999 + [False]
    # El precio sube 1 por vela en el server sintético: el orden se respeta.
    assert (df["open"].diff().dropna() == 1.0).all()


async def test_klines_history_respects_end_and_request_cap(settings, client, respx_mock):
    start = datetime(2026, 10, 12, 10, 0, tzinfo=UTC)
    end = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
    respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        side_effect=kline_server(utc_to_ms(start) - 10 * M15, utc_to_ms(FIXTURE_NOW), M15)
    )
    df = (await make(settings, client).get_klines_history("BTC", "15m", start, end)).df
    assert df.index[0] == pd.Timestamp(start) and df.index[-1] == pd.Timestamp(end)
    assert len(df) == 9 and df["closed"].all()

    md = make(settings, client, max_history_requests=1)
    with pytest.raises(ValueError, match="pedidos"):
        await md.get_klines_history("BTC", "1h", FIXTURE_NOW - timedelta(days=200))


async def test_klines_invalid_args(settings, client):
    md = make(settings, client)
    with pytest.raises(ValueError):
        await md.get_klines("BTC", "2h", limit=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await md.get_klines("BTC", "15m", limit=0)


async def test_klines_malformed_rows_raise_exchange_error(settings, client, respx_mock):
    route = respx_mock.get(f"{FAPI}/fapi/v1/klines")
    route.mock(return_value=httpx.Response(200, json=[[1791809100000, "1", "2"]]))
    with pytest.raises(ExchangeError, match="formato"):
        await make(settings, client).get_klines("BTC", "15m", limit=1)
    route.mock(return_value=httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ExchangeError, match="lista"):
        await make(settings, client).get_klines("BTC", "15m", limit=1)


# --- ticker, funding, OI, long/short ------------------------------------------


async def test_ticker(settings, client, respx_mock):
    route = respx_mock.get(f"{FAPI}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(200, json=load("ticker_24hr.json"))
    )
    t = await make(settings, client).get_ticker("BTC")
    assert route.calls.last.request.url.params["symbol"] == "BTCUSDT"
    assert t.symbol == "BTC" and t.price == 108962.5
    assert t.change_24h_pct == pytest.approx(-1.248)  # Binance ya lo manda en %
    assert (t.high_24h, t.low_24h) == (110766.0, 108401.1)
    assert t.quote_volume_24h == pytest.approx(16679422185.92)
    assert t.meta.source == "binance" and t.meta.fetched_at == FIXTURE_NOW


async def test_funding_uses_premium_index_and_default_interval(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/premiumIndex", params={"symbol": "BTCUSDT"}).mock(
        return_value=httpx.Response(200, json=load("premium_index.json"))
    )
    respx_mock.get(f"{FAPI}/fapi/v1/fundingInfo").mock(
        return_value=httpx.Response(200, json=load("funding_info.json"))
    )
    f = await make(settings, client).get_funding("BTC")
    assert f.rate == pytest.approx(0.00008234)  # fracción, no %
    assert f.mark_price == pytest.approx(108958.12345678)
    assert f.index_price == pytest.approx(108975.43921053)
    assert f.next_funding_time == datetime(2026, 10, 12, 16, 0, tzinfo=UTC)
    assert f.interval_hours == 8  # BTCUSDT no figura en fundingInfo
    assert f.meta.warning is None


async def test_funding_interval_from_funding_info(settings, client, respx_mock):
    premium = load("premium_index.json") | {"symbol": "AVAXUSDT"}
    respx_mock.get(f"{FAPI}/fapi/v1/premiumIndex", params={"symbol": "AVAXUSDT"}).mock(
        return_value=httpx.Response(200, json=premium)
    )
    respx_mock.get(f"{FAPI}/fapi/v1/fundingInfo").mock(
        return_value=httpx.Response(200, json=load("funding_info.json"))
    )
    f = await make(settings, client).get_funding("AVAX")
    assert f.symbol == "AVAX" and f.interval_hours == 4


async def test_funding_info_failure_assumes_default_with_warning(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/premiumIndex").mock(
        return_value=httpx.Response(200, json=load("premium_index.json"))
    )
    respx_mock.get(f"{FAPI}/fapi/v1/fundingInfo").mock(return_value=httpx.Response(503))
    f = await make(settings, client, default_funding_interval_hours=8).get_funding("BTC")
    assert f.interval_hours == 8
    assert "se asume 8 h" in f.meta.warning


async def test_funding_history_chronological_and_skips_special(settings, client, respx_mock):
    rows = load("funding_rate.json")
    special = rows[-1] | {"fundingTime": 1791795600000, "rateType": "Special"}
    shuffled = [*rows, special]
    random.Random(1).shuffle(shuffled)
    route = respx_mock.get(f"{FAPI}/fapi/v1/fundingRate").mock(
        return_value=httpx.Response(200, json=shuffled)
    )
    h = await make(settings, client).get_funding_history("BTC", limit=6)

    assert route.calls.last.request.url.params["limit"] == "6"
    assert [p.time for p in h.points] == sorted(p.time for p in h.points)
    assert len(h.points) == 5  # el "Special" no cuenta
    assert h.points[0].time == datetime(2026, 10, 11, 0, 0, tzinfo=UTC)
    assert h.points[1].time == datetime(2026, 10, 11, 8, 0, 0, 1000, tzinfo=UTC)  # +1 ms
    assert all(p.time.tzinfo is not None for p in h.points)
    assert [p.rate for p in h.points] == pytest.approx(
        [0.0001, 0.00007512, 0.0000412, -0.0000185, 0.00006033]
    )


async def test_open_interest(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/openInterest").mock(
        return_value=httpx.Response(200, json=load("open_interest.json"))
    )
    oi = await make(settings, client).get_open_interest("BTC")
    assert oi.contracts == pytest.approx(81534.271)
    assert oi.value_usd is None
    assert oi.time == ms_to_utc(1791815839880)


async def test_open_interest_history(settings, client, respx_mock):
    route = respx_mock.get(f"{FAPI}/futures/data/openInterestHist").mock(
        return_value=httpx.Response(200, json=load("open_interest_hist.json"))
    )
    h = await make(settings, client).get_open_interest_history("BTC", "1h", limit=5)

    params = route.calls.last.request.url.params
    assert (params["symbol"], params["period"], params["limit"]) == ("BTCUSDT", "1h", "5")
    assert h.period == "1h" and len(h.points) == 5
    assert h.points[0].time == datetime(2026, 10, 12, 10, 0, tzinfo=UTC)
    assert h.points[-1].time == datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    assert h.points[0].contracts == pytest.approx(80120.334)
    assert h.points[0].value_usd == pytest.approx(8700287099.14349937)
    assert h.points[-1].contracts / h.points[0].contracts - 1 == pytest.approx(0.0172, abs=1e-4)


async def test_open_interest_history_pages_when_over_max(settings, client, respx_mock):
    base = utc_to_ms(datetime(2026, 10, 12, 14, 0, tzinfo=UTC))

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        end = int(p.get("endTime", base))
        ts = [base - i * 3_600_000 for i in range(700) if base - i * 3_600_000 <= end]
        ts = sorted(ts)[-int(p["limit"]) :]
        rows = [
            {
                "symbol": "BTCUSDT",
                "sumOpenInterest": "1",
                "sumOpenInterestValue": "2",
                "timestamp": t,
            }
            for t in ts
        ]
        return httpx.Response(200, json=rows)

    route = respx_mock.get(f"{FAPI}/futures/data/openInterestHist").mock(side_effect=handler)
    h = await make(settings, client).get_open_interest_history("BTC", "1h", limit=600)
    assert route.call_count == 2
    assert [c.request.url.params["limit"] for c in route.calls] == ["500", "100"]
    assert len(h.points) == 600
    assert h.points[-1].time == ms_to_utc(base)


async def test_long_short_ratio(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/futures/data/globalLongShortAccountRatio").mock(
        return_value=httpx.Response(200, json=load("global_long_short_account_ratio.json"))
    )
    h = await make(settings, client).get_long_short_ratio("BTC", "1h", limit=5)
    assert len(h.points) == 5
    first = h.points[0]
    assert (first.ratio, first.long_pct, first.short_pct) == pytest.approx((1.681, 0.627, 0.373))
    assert h.points[-1].ratio == pytest.approx(1.8994)
    assert [p.time for p in h.points] == sorted(p.time for p in h.points)


# --- errores -----------------------------------------------------------------


async def test_invalid_symbol_is_exchange_error_with_msg(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(400, json=load("error_invalid_symbol.json"))
    )
    with pytest.raises(ExchangeError, match="Invalid symbol") as info:
        await make(settings, client).get_ticker("NOPE")
    assert not isinstance(info.value, ExchangeUnavailableError)
    assert "-1121" in str(info.value)


async def test_error_body_with_http_200_is_exchange_error(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/openInterest").mock(
        return_value=httpx.Response(200, json=load("error_invalid_symbol.json"))
    )
    with pytest.raises(ExchangeError, match="Invalid symbol"):
        await make(settings, client).get_open_interest("NOPE")


@pytest.mark.parametrize(
    ("status", "fixture"),
    [(451, "error_restricted_location.json"), (429, "error_rate_limit.json"), (503, None)],
)
async def test_unavailable_statuses(settings, client, respx_mock, status, fixture):
    body = load(fixture) if fixture else None
    respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(return_value=httpx.Response(status, json=body))
    with pytest.raises(ExchangeUnavailableError, match=str(status)):
        await make(settings, client).get_klines("BTC", "15m", limit=8)
