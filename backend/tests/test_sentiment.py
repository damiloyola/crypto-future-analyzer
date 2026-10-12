"""Fear & Greed (alternative.me): parseo, etiquetas en castellano y caché."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from app.data.cache import TTLCache
from app.data.sentiment import (
    STALE_WARNING,
    get_fear_greed,
    parse_fear_greed,
    translate_classification,
)
from app.exchanges.base import ExchangeError, ExchangeUnavailableError, FetchMeta
from app.http_client import make_client

FIXTURES = Path(__file__).parent / "fixtures" / "alternative_me"
FNG_URL = "https://api.alternative.me/fng/"
T0 = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def fng_settings(settings):
    ds = settings.data_sources.model_copy(update={"fear_greed_url": FNG_URL})
    ex = settings.exchanges.model_copy(update={"max_retries": 0})
    ttls = {**settings.cache_ttl_seconds, "fear_greed": 1800}
    return settings.model_copy(
        update={"data_sources": ds, "exchanges": ex, "cache_ttl_seconds": ttls}
    )


def test_parse_fixture_values_and_order():
    fg = parse_fear_greed(load("fng.json"), FetchMeta(source="x", fetched_at=T0))
    assert fg.value == 38
    assert isinstance(fg.value, int)
    assert fg.classification_en == "Fear"
    assert fg.classification_es == "Miedo"
    assert len(fg.history) == 30
    days = [d for d, _ in fg.history]
    assert days == sorted(days)
    assert all(d.tzinfo == UTC for d in days)
    # El primero del fixture (el más nuevo) es el 12/10 00:00 UTC y queda al final.
    assert fg.history[-1] == (datetime(2026, 10, 12, tzinfo=UTC), 38)
    assert fg.history[0] == (datetime(2026, 9, 13, tzinfo=UTC), 84)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Extreme Fear", "Miedo extremo"),
        ("Fear", "Miedo"),
        ("Neutral", "Neutral"),
        ("Greed", "Codicia"),
        ("Extreme Greed", "Codicia extrema"),
        ("extreme greed", "Codicia extrema"),
        ("Algo nuevo", "Algo nuevo"),
    ],
)
def test_translate_classification(label, expected):
    assert translate_classification(label) == expected


def test_parse_uses_api_label_for_each_class():
    payload = {
        "data": [
            {"value": "80", "value_classification": "Extreme Greed", "timestamp": "1791763200"},
            {"value": "10", "value_classification": "Extreme Fear", "timestamp": "1791676800"},
        ],
        "metadata": {"error": None},
    }
    fg = parse_fear_greed(payload, FetchMeta(source="x", fetched_at=T0))
    assert (fg.value, fg.classification_es) == (80, "Codicia extrema")
    assert [v for _, v in fg.history] == [10, 80]


@pytest.mark.parametrize(
    "payload",
    [
        load("fng_error.json"),
        {"data": [], "metadata": {"error": None}},
        {"data": [{"value": "abc", "timestamp": "1791763200"}], "metadata": {"error": None}},
        {"data": [{"value": "150", "timestamp": "1791763200"}], "metadata": {"error": None}},
        {"data": [{"value": "40"}], "metadata": {"error": None}},
        ["no", "es", "un", "dict"],
    ],
)
def test_parse_rejects_bad_payloads(payload):
    with pytest.raises(ExchangeError):
        parse_fear_greed(payload, FetchMeta(source="x", fetched_at=T0))


@respx.mock
async def test_get_fear_greed_sends_limit_and_caches(fng_settings):
    route = respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng.json")))
    cache = TTLCache(clock=FakeClock(T0))
    async with make_client() as client:
        first = await get_fear_greed(client, cache, fng_settings, limit=30)
        second = await get_fear_greed(client, cache, fng_settings, limit=30)

    assert route.call_count == 1
    params = route.calls.last.request.url.params
    assert params["limit"] == "30"
    assert "date_format" not in params
    assert first.meta.source == "alternative.me"
    assert first.meta.fetched_at == T0
    assert not first.meta.from_cache and not first.meta.stale and first.meta.warning is None
    assert second.meta.from_cache and second.meta.fetched_at == T0
    assert second.value == first.value == 38


@respx.mock
async def test_get_fear_greed_refetches_after_ttl(fng_settings):
    route = respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng.json")))
    clock = FakeClock(T0)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_fear_greed(client, cache, fng_settings)
        clock.advance(1801)
        again = await get_fear_greed(client, cache, fng_settings)
    assert route.call_count == 2
    assert not again.meta.from_cache
    assert again.meta.fetched_at == T0 + timedelta(seconds=1801)


@respx.mock
async def test_stale_value_when_api_fails(fng_settings):
    route = respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng.json")))
    clock = FakeClock(T0)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_fear_greed(client, cache, fng_settings)
        clock.advance(4000)
        route.mock(return_value=httpx.Response(503))
        stale = await get_fear_greed(client, cache, fng_settings)

    assert stale.value == 38
    assert stale.meta.stale and stale.meta.from_cache
    assert stale.meta.fetched_at == T0
    assert stale.meta.warning == STALE_WARNING


@respx.mock
async def test_api_error_body_counts_as_failure(fng_settings):
    respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng.json")))
    clock = FakeClock(T0)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_fear_greed(client, cache, fng_settings)
        clock.advance(4000)
        respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng_error.json")))
        stale = await get_fear_greed(client, cache, fng_settings)
    assert stale.meta.stale and stale.value == 38


@respx.mock
async def test_error_propagates_without_cached_value(fng_settings):
    respx.get(FNG_URL).mock(return_value=httpx.Response(503))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError):
            await get_fear_greed(client, TTLCache(clock=FakeClock(T0)), fng_settings)
    respx.get(FNG_URL).mock(return_value=httpx.Response(200, json=load("fng_error.json")))
    async with make_client() as client:
        with pytest.raises(ExchangeError, match="error"):
            await get_fear_greed(client, TTLCache(clock=FakeClock(T0)), fng_settings)


async def test_limit_must_be_positive(fng_settings):
    async with make_client() as client:
        with pytest.raises(ValueError):
            await get_fear_greed(client, TTLCache(), fng_settings, limit=0)
