"""Conector de Bybit v5: parseo, orden cronológico, unidades, paginación y retCode.

Sin red: respx con los fixtures de ``tests/fixtures/bybit`` o respuestas
sintéticas armadas en el test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from app.exchanges.base import ExchangeError, ExchangeUnavailableError, MarketDataSource
from app.exchanges.binance import ms_to_utc, utc_to_ms
from app.exchanges.bybit import BybitMarketData
from app.http_client import make_client

FIXTURES = Path(__file__).parent / "fixtures" / "bybit"
FIXTURE_NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)  # ver fixtures/README.md
BASE = "https://api.bybit.com"
M15 = 15 * 60_000
H1 = 3_600_000


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def envelope(items: list, **result) -> dict:
    return {
        "retCode": 0,
        "retMsg": "OK",
        "result": {"category": "linear", "list": items, **result},
        "retExtInfo": {},
        "time": 1791815840000,
    }


@pytest.fixture
async def client():
    async with make_client() as c:
        yield c


def make(settings, client, now: datetime = FIXTURE_NOW, **kwargs) -> BybitMarketData:
    kwargs.setdefault("retries", 0)
    kwargs.setdefault("backoff_s", 0)
    return BybitMarketData(settings.exchanges, client, clock=lambda: now, **kwargs)


def kline_server(first_open_ms: int, last_open_ms: int, step_ms: int, overlap: bool = False):
    """Imita /v5/market/kline: la más nueva primero, cuenta desde ``end`` hacia atrás."""

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        limit = int(p["limit"])
        opens = list(range(first_open_ms, last_open_ms + 1, step_ms))
        if "start" in p:
            opens = [o for o in opens if o >= int(p["start"])]
        if "end" in p:
            opens = [o for o in opens if o <= int(p["end"])]
        selected = opens[-limit:]
        if overlap and selected and selected[-1] + step_ms <= last_open_ms:
            selected.append(selected[-1] + step_ms)  # una vela de la ventana siguiente
        rows = [
            [str(o), "100", "101", "99", f"{100 + (o - first_open_ms) / step_ms}", "2", "200"]
            for o in reversed(selected)
        ]
        return httpx.Response(200, json=envelope(rows, symbol="BTCUSDT"))

    return handler


def test_implements_protocol(settings, client):
    assert isinstance(make(settings, client), MarketDataSource)


async def test_aclose_only_closes_own_client(settings, client):
    await make(settings, client).aclose()
    assert not client.is_closed
    owned = BybitMarketData(settings.exchanges)
    await owned.aclose()
    assert owned._client.is_closed


# --- velas -------------------------------------------------------------------


async def test_klines_reversed_to_chronological(settings, client, respx_mock):
    route = respx_mock.get(f"{BASE}/v5/market/kline").mock(
        return_value=httpx.Response(200, json=load("kline.json"))
    )
    candles = await make(settings, client).get_klines("BTC", "15m", limit=8)

    p = route.calls.last.request.url.params
    assert (p["category"], p["symbol"], p["interval"], p["limit"]) == (
        "linear",
        "BTCUSDT",
        "15",
        "8",
    )
    df = candles.df
    assert len(df) == 8 and str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2026-10-12 12:45", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2026-10-12 14:30", tz="UTC")
    assert df["open"].iloc[0] == 108655.5
    assert df["close"].iloc[-1] == 108960.8
    assert df["volume"].iloc[-1] == pytest.approx(87.442)  # moneda base (no turnover)
    assert df["closed"].tolist() == [True] * 7 + [False]
    assert candles.meta.source == "bybit" and candles.meta.fetched_at == FIXTURE_NOW


async def test_klines_closed_flag_when_candle_ends(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/kline").mock(
        return_value=httpx.Response(200, json=load("kline.json"))
    )
    now = datetime(2026, 10, 12, 14, 45, tzinfo=UTC)  # open_time + 15m <= ahora
    df = (await make(settings, client, now=now).get_klines("BTC", "15m", limit=8)).df
    assert df["closed"].all()


@pytest.mark.parametrize(
    ("timeframe", "code"), [("5m", "5"), ("15m", "15"), ("1h", "60"), ("4h", "240"), ("1d", "D")]
)
async def test_kline_interval_codes(settings, client, respx_mock, timeframe, code):
    route = respx_mock.get(f"{BASE}/v5/market/kline").mock(
        return_value=httpx.Response(200, json=envelope([]))
    )
    candles = await make(settings, client).get_klines("BTC", timeframe, limit=3)
    assert route.calls.last.request.url.params["interval"] == code
    assert candles.df.empty


async def test_klines_more_than_max_pages_backward(settings, client, respx_mock):
    last_open = utc_to_ms(datetime(2026, 10, 12, 14, 30, tzinfo=UTC))
    route = respx_mock.get(f"{BASE}/v5/market/kline").mock(
        side_effect=kline_server(last_open - 2999 * M15, last_open, M15)
    )
    df = (await make(settings, client).get_klines("BTC", "15m", limit=1500)).df

    assert route.call_count == 2
    first, second = (c.request.url.params for c in route.calls)
    assert first["limit"] == "1000" and "end" not in first
    assert second["limit"] == "500" and int(second["end"]) == last_open - 999 * M15 - 1
    assert len(df) == 1500 and df.index.is_unique and df.index.is_monotonic_increasing
    assert df["closed"].tolist().count(False) == 1


async def test_klines_history_windows_dedup_and_order(settings, client, respx_mock):
    current_open = utc_to_ms(datetime(2026, 10, 12, 14, 30, tzinfo=UTC))
    start_ms = current_open - 2499 * M15  # 2500 velas
    route = respx_mock.get(f"{BASE}/v5/market/kline").mock(
        side_effect=kline_server(start_ms - 20 * M15, current_open, M15, overlap=True)
    )
    df = (await make(settings, client).get_klines_history("BTC", "15m", ms_to_utc(start_ms))).df

    assert route.call_count == 3  # ventanas de 1000 velas
    starts = [int(c.request.url.params["start"]) for c in route.calls]
    assert starts == [start_ms, start_ms + 1000 * M15, start_ms + 2000 * M15]
    assert int(route.calls.last.request.url.params["end"]) == utc_to_ms(FIXTURE_NOW)
    assert len(df) == 2500 and df.index.is_unique and df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp(ms_to_utc(start_ms))
    assert (df["close"].diff().dropna() == 1.0).all()
    assert df["closed"].tolist() == [True] * 2499 + [False]


# --- ticker y funding --------------------------------------------------------


async def test_ticker_converts_fraction_to_percent(settings, client, respx_mock):
    route = respx_mock.get(f"{BASE}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("tickers.json"))
    )
    t = await make(settings, client).get_ticker("BTC")
    p = route.calls.last.request.url.params
    assert (p["category"], p["symbol"]) == ("linear", "BTCUSDT")
    assert t.price == 108960.8
    assert t.change_24h_pct == pytest.approx(-1.2424)  # -0.012424 (fracción) → %
    assert (t.high_24h, t.low_24h) == (110770.0, 108398.5)
    assert t.quote_volume_24h == pytest.approx(9831120458.4522)  # turnover = USDT
    assert t.meta.source == "bybit"


async def test_funding_from_tickers(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("tickers.json"))
    )
    f = await make(settings, client).get_funding("BTC")
    assert f.rate == pytest.approx(0.00007891)
    assert f.mark_price == pytest.approx(108957.90)
    assert f.index_price == pytest.approx(108975.21)
    assert f.next_funding_time == datetime(2026, 10, 12, 16, 0, tzinfo=UTC)
    assert f.interval_hours == 8


@pytest.mark.parametrize(("raw", "expected"), [("4", 4.0), ("", 8.0), (None, 8.0), ("0", 8.0)])
async def test_funding_interval(settings, client, respx_mock, raw, expected):
    body = load("tickers.json")
    item = body["result"]["list"][0]
    if raw is None:
        del item["fundingIntervalHour"]
    else:
        item["fundingIntervalHour"] = raw
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(return_value=httpx.Response(200, json=body))
    f = await make(settings, client, default_funding_interval_hours=8).get_funding("BTC")
    assert f.interval_hours == expected


async def test_funding_history_sorted(settings, client, respx_mock):
    route = respx_mock.get(f"{BASE}/v5/market/funding/history").mock(
        return_value=httpx.Response(200, json=load("funding_history.json"))
    )
    h = await make(settings, client).get_funding_history("BTC", limit=5)
    assert route.calls.last.request.url.params["limit"] == "5"
    assert [p.time for p in h.points] == sorted(p.time for p in h.points)
    assert h.points[0].time == datetime(2026, 10, 11, 0, 0, tzinfo=UTC)
    assert [p.rate for p in h.points] == pytest.approx(
        [0.0001, 0.00006821, 0.00003512, -0.00002044, 0.00005317]
    )


# --- open interest y long/short ----------------------------------------------


async def test_open_interest_from_tickers(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("tickers.json"))
    )
    oi = await make(settings, client).get_open_interest("BTC")
    assert oi.contracts == pytest.approx(104682.236)  # openInterest (los dos lados)
    assert oi.value_usd == pytest.approx(11405956601.86)
    assert oi.time == ms_to_utc(1791815839750)


async def test_open_interest_history(settings, client, respx_mock):
    route = respx_mock.get(f"{BASE}/v5/market/open-interest").mock(
        return_value=httpx.Response(200, json=load("open_interest.json"))
    )
    h = await make(settings, client).get_open_interest_history("BTC", "1h", limit=5)
    p = route.calls.last.request.url.params
    assert (p["intervalTime"], p["limit"]) == ("1h", "5")
    assert len(h.points) == 5
    assert h.points[0].time == datetime(2026, 10, 12, 10, 0, tzinfo=UTC)
    assert h.points[-1].time == datetime(2026, 10, 12, 14, 0, tzinfo=UTC)
    assert h.points[0].contracts == pytest.approx(102950.118)
    assert h.points[-1].value_usd is None


async def test_open_interest_history_pages_over_200(settings, client, respx_mock):
    base = utc_to_ms(datetime(2026, 10, 12, 14, 0, tzinfo=UTC))

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        end = int(p.get("endTime", base))
        ts = [base - i * H1 for i in range(1000) if base - i * H1 <= end][: int(p["limit"])]
        items = [{"openInterest": "10", "singleOpenInterest": "5", "timestamp": str(t)} for t in ts]
        return httpx.Response(200, json=envelope(items, nextPageCursor="x"))

    route = respx_mock.get(f"{BASE}/v5/market/open-interest").mock(side_effect=handler)
    h = await make(settings, client).get_open_interest_history("BTC", "1h", limit=300)
    assert [c.request.url.params["limit"] for c in route.calls] == ["200", "100"]
    assert int(route.calls.last.request.url.params["endTime"]) == base - 199 * H1 - 1
    assert len(h.points) == 300 and h.points[-1].time == ms_to_utc(base)


async def test_long_short_ratio(settings, client, respx_mock):
    route = respx_mock.get(f"{BASE}/v5/market/account-ratio").mock(
        return_value=httpx.Response(200, json=load("account_ratio.json"))
    )
    h = await make(settings, client).get_long_short_ratio("BTC", "1h", limit=5)
    assert route.calls.last.request.url.params["period"] == "1h"
    assert [p.time for p in h.points] == sorted(p.time for p in h.points)
    first, last = h.points[0], h.points[-1]
    assert first.long_pct == pytest.approx(0.5912) and first.short_pct == pytest.approx(0.4088)
    assert first.ratio == pytest.approx(0.5912 / 0.4088)
    assert last.ratio == pytest.approx(0.6187 / 0.3813)


# --- errores -----------------------------------------------------------------


async def test_ret_code_invalid_symbol_is_exchange_error(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/kline").mock(
        return_value=httpx.Response(200, json=load("error_invalid_symbol.json"))
    )
    with pytest.raises(ExchangeError, match="Not supported symbols") as info:
        await make(settings, client).get_klines("NOPE", "15m", limit=8)
    assert not isinstance(info.value, ExchangeUnavailableError)
    assert "10001" in str(info.value)


async def test_ret_code_rate_limit_is_unavailable(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("error_rate_limit.json"))
    )
    with pytest.raises(ExchangeUnavailableError, match="10006"):
        await make(settings, client).get_ticker("BTC")


async def test_http_403_is_unavailable(settings, client, respx_mock):
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(return_value=httpx.Response(403))
    with pytest.raises(ExchangeUnavailableError, match="403"):
        await make(settings, client).get_ticker("BTC")


@pytest.mark.parametrize(
    "body",
    [
        {"retMsg": "OK", "result": {}},  # sin retCode
        {"retCode": 0, "retMsg": "OK", "result": None},  # result no es objeto
        envelope([]),  # ticker vacío
        [1, 2, 3],  # ni siquiera es un objeto
    ],
)
async def test_malformed_envelopes(settings, client, respx_mock, body):
    respx_mock.get(f"{BASE}/v5/market/tickers").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ExchangeError):
        await make(settings, client).get_ticker("BTC")
