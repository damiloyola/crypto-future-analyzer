"""Mercado global (CoinGecko /global): unidades, caché y valor viejo."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from app.data.cache import TTLCache
from app.data.market import STALE_WARNING, get_global_market, parse_global_market
from app.exchanges.base import ExchangeError, ExchangeUnavailableError, FetchMeta
from app.http_client import make_client

FIXTURE = Path(__file__).parent / "fixtures" / "coingecko" / "global.json"
BASE_URL = "https://api.coingecko.com/api/v3"
GLOBAL_URL = f"{BASE_URL}/global"
T0 = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def load() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def market_settings(settings):
    ds = settings.data_sources.model_copy(update={"coingecko_base_url": BASE_URL + "/"})
    ex = settings.exchanges.model_copy(update={"max_retries": 0})
    ttls = {**settings.cache_ttl_seconds, "global_market": 600}
    return settings.model_copy(
        update={"data_sources": ds, "exchanges": ex, "cache_ttl_seconds": ttls}
    )


def test_parse_fixture_units():
    gm = parse_global_market(load(), FetchMeta(source="x", fetched_at=T0))
    # CoinGecko manda porcentajes "humanos": se dejan igual.
    assert gm.btc_dominance_pct == pytest.approx(57.9539332566265)
    assert gm.eth_dominance_pct == pytest.approx(9.58227145398409)
    assert gm.total_market_cap_usd == pytest.approx(2621040321355.0405)
    assert gm.market_cap_change_24h_pct == pytest.approx(-1.6081983639177684)
    assert gm.updated_at == datetime.fromtimestamp(1779878351, tz=UTC)


def test_parse_optional_fields_missing():
    payload = {
        "data": {"market_cap_percentage": {"btc": "58.1"}, "total_market_cap": {"usd": 3e12}}
    }
    gm = parse_global_market(payload, FetchMeta(source="x", fetched_at=T0))
    assert gm.btc_dominance_pct == pytest.approx(58.1)
    assert gm.eth_dominance_pct is None
    assert gm.market_cap_change_24h_pct is None
    assert gm.updated_at is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": None},
        {"data": {"market_cap_percentage": {}, "total_market_cap": {"usd": 3e12}}},
        {"data": {"market_cap_percentage": {"btc": 130}, "total_market_cap": {"usd": 3e12}}},
        {"data": {"market_cap_percentage": {"btc": 58}, "total_market_cap": {"usd": 0}}},
        {"data": {"market_cap_percentage": {"btc": "NaN"}, "total_market_cap": {"usd": 3e12}}},
        [1, 2, 3],
    ],
)
def test_parse_rejects_bad_payloads(payload):
    with pytest.raises(ExchangeError):
        parse_global_market(payload, FetchMeta(source="x", fetched_at=T0))


@respx.mock
async def test_get_global_market_caches(market_settings):
    route = respx.get(GLOBAL_URL).mock(return_value=httpx.Response(200, json=load()))
    cache = TTLCache(clock=FakeClock(T0))
    async with make_client() as client:
        first = await get_global_market(client, cache, market_settings)
        second = await get_global_market(client, cache, market_settings)

    assert route.call_count == 1
    assert first.meta.source == "coingecko"
    assert first.meta.fetched_at == T0 and not first.meta.from_cache
    assert second.meta.from_cache and second.meta.fetched_at == T0
    assert second.btc_dominance_pct == first.btc_dominance_pct


@respx.mock
async def test_stale_value_when_api_fails(market_settings):
    route = respx.get(GLOBAL_URL).mock(return_value=httpx.Response(200, json=load()))
    clock = FakeClock(T0)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_global_market(client, cache, market_settings)
        clock.advance(601)
        route.mock(return_value=httpx.Response(429))
        stale = await get_global_market(client, cache, market_settings)

    assert stale.meta.stale and stale.meta.from_cache
    assert stale.meta.fetched_at == T0
    assert stale.meta.warning == STALE_WARNING
    assert stale.btc_dominance_pct == pytest.approx(57.9539332566265)


@respx.mock
async def test_error_propagates_without_cached_value(market_settings):
    respx.get(GLOBAL_URL).mock(return_value=httpx.Response(429))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError):
            await get_global_market(client, TTLCache(clock=FakeClock(T0)), market_settings)
