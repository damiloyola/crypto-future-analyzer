"""Registry: fallback Binance → Bybit, caché con TTL, armado desde config y precio de ejecución."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from app.data.cache import TTLCache
from app.exchanges.base import (
    Candles,
    ExchangeError,
    ExchangeUnavailableError,
    FetchMeta,
    FundingHistory,
    FundingInfo,
    LongShortHistory,
    MarketDataSource,
    OpenInterest,
    OpenInterestHistory,
    Ticker,
    empty_candles_frame,
)
from app.exchanges.binance import BinanceMarketData
from app.exchanges.bybit import BybitMarketData
from app.exchanges.nexo import NexoPriceSource
from app.exchanges.registry import (
    NEXO_UNAVAILABLE_WARNING,
    CachedMarketData,
    ExecutionPrice,
    FallbackMarketData,
    build_market_data,
    build_nexo_price_source,
    failure_reason,
    fallback_warning,
    get_execution_price,
    make_market_data_source,
)
from app.http_client import make_client

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)
FAPI = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
BINANCE_451 = ExchangeUnavailableError("binance: HTTP 451 en https://fapi.binance.com/fapi/v1/x")


def load(path: str):
    return json.loads((FIXTURES / path).read_text(encoding="utf-8"))


class FakeClock:
    def __init__(self, now: datetime = FIXTURE_NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeSource:
    """``MarketDataSource`` de mentira: registra llamadas y puede fallar a pedido."""

    def __init__(
        self,
        name: str,
        price: float = 108962.5,
        fail: Exception | None = None,
        warning: str | None = None,
    ) -> None:
        self.name = name
        self.price = price
        self.fail = fail
        self.warning = warning
        self.calls: list[tuple] = []
        self.closed = False

    def _meta(self) -> FetchMeta:
        return FetchMeta(self.name, FIXTURE_NOW, warning=self.warning)

    async def _run(self, call: tuple, value):
        self.calls.append(call)
        if self.fail is not None:
            raise self.fail
        return value

    async def get_ticker(self, symbol):
        return await self._run(("get_ticker", symbol), Ticker(symbol, self.price, self._meta()))

    async def get_klines(self, symbol, timeframe, limit=500, end_time=None):
        value = Candles(symbol, timeframe, empty_candles_frame(), self._meta())
        return await self._run(("get_klines", symbol, timeframe, limit, end_time), value)

    async def get_klines_history(self, symbol, timeframe, start, end=None):
        value = Candles(symbol, timeframe, empty_candles_frame(), self._meta())
        return await self._run(("get_klines_history", symbol, timeframe, start, end), value)

    async def get_funding(self, symbol):
        value = FundingInfo(symbol, 0.0001, self.price, self._meta())
        return await self._run(("get_funding", symbol), value)

    async def get_funding_history(self, symbol, limit=21):
        value = FundingHistory(symbol, [], self._meta())
        return await self._run(("get_funding_history", symbol, limit), value)

    async def get_open_interest(self, symbol):
        value = OpenInterest(symbol, 1000.0, self._meta())
        return await self._run(("get_open_interest", symbol), value)

    async def get_open_interest_history(self, symbol, period="1h", limit=168):
        value = OpenInterestHistory(symbol, period, [], self._meta())
        return await self._run(("get_open_interest_history", symbol, period, limit), value)

    async def get_long_short_ratio(self, symbol, period="1h", limit=168):
        value = LongShortHistory(symbol, period, [], self._meta())
        return await self._run(("get_long_short_ratio", symbol, period, limit), value)

    async def aclose(self):
        self.closed = True


ALL_CALLS = [
    ("get_ticker", ("BTC",)),
    ("get_klines", ("BTC", "15m", 500, None)),
    ("get_klines_history", ("BTC", "1h", FIXTURE_NOW - timedelta(days=1), None)),
    ("get_funding", ("BTC",)),
    ("get_funding_history", ("BTC", 21)),
    ("get_open_interest", ("BTC",)),
    ("get_open_interest_history", ("BTC", "1h", 168)),
    ("get_long_short_ratio", ("BTC", "1h", 168)),
]


@pytest.fixture
async def client():
    async with make_client() as c:
        yield c


def with_exchanges(settings, **update):
    return settings.model_copy(update={"exchanges": settings.exchanges.model_copy(update=update)})


def with_nexo(settings, **update):
    return with_exchanges(settings, nexo=settings.exchanges.nexo.model_copy(update=update))


# --- avisos ------------------------------------------------------------------


def test_failure_reason_and_warning_text():
    assert failure_reason(BINANCE_451) == "HTTP 451"
    assert failure_reason(ExchangeUnavailableError("bybit: x (retCode 10006) en /v5")) == (
        "retCode 10006"
    )
    assert failure_reason(ExchangeUnavailableError("binance: sin respuesta (ConnectError)")) == (
        "sin conexión"
    )
    assert failure_reason(ExchangeError("binance: respuesta rara")) == "respuesta rara"
    invalid = "binance: Invalid symbol. (HTTP 400, código -1121) en https://fapi.binance.com/x"
    assert failure_reason(ExchangeError(invalid)) == "símbolo inválido"
    bybit_invalid = "bybit: Not supported symbols (retCode 10001) en /v5/market/kline"
    assert failure_reason(ExchangeError(bybit_invalid)) == "símbolo inválido"
    # 10001 también es "parámetro inválido" en general: sin el texto no se asume símbolo.
    assert failure_reason(ExchangeError("bybit: params error (retCode 10001) en /v5")) == (
        "retCode 10001"
    )
    assert fallback_warning("binance", "bybit", BINANCE_451) == (
        "Binance no respondió (HTTP 451): datos de Bybit"
    )
    assert fallback_warning("binance", "bybit", ExchangeError("binance: HTTP 400 en u: {}")) == (
        "Binance dio un error (HTTP 400): datos de Bybit"
    )


# --- fallback ----------------------------------------------------------------


def test_fallback_implements_protocol():
    assert isinstance(
        FallbackMarketData(FakeSource("binance"), FakeSource("bybit")), MarketDataSource
    )


@pytest.mark.parametrize(("method", "args"), ALL_CALLS)
async def test_fallback_uses_primary_when_it_works(method, args):
    primary, fallback = FakeSource("binance"), FakeSource("bybit")
    result = await getattr(FallbackMarketData(primary, fallback), method)(*args)
    assert result.meta.source == "binance" and not result.meta.fallback
    assert primary.calls == [(method, *args)] and fallback.calls == []


@pytest.mark.parametrize(("method", "args"), ALL_CALLS)
async def test_fallback_switches_on_exchange_error(method, args):
    primary, fallback = FakeSource("binance", fail=BINANCE_451), FakeSource("bybit")
    result = await getattr(FallbackMarketData(primary, fallback), method)(*args)
    assert result.meta.source == "bybit"
    assert result.meta.fallback is True
    assert result.meta.warning == "Binance no respondió (HTTP 451): datos de Bybit"
    assert fallback.calls == [(method, *args)]


async def test_fallback_keeps_existing_warning():
    primary = FakeSource("binance", fail=BINANCE_451)
    fallback = FakeSource("bybit", warning="otro aviso")
    t = await FallbackMarketData(primary, fallback).get_ticker("BTC")
    assert t.meta.warning == "Binance no respondió (HTTP 451): datos de Bybit · otro aviso"


async def test_fallback_both_fail_reports_both_reasons():
    primary = FakeSource("binance", fail=BINANCE_451)
    fallback = FakeSource(
        "bybit", fail=ExchangeError("bybit: Not supported symbols (retCode 10001)")
    )
    with pytest.raises(ExchangeUnavailableError) as info:
        await FallbackMarketData(primary, fallback).get_funding("BTC")
    msg = str(info.value)
    assert "Binance y Bybit fallaron" in msg and "HTTP 451" in msg and "10001" in msg


async def test_fallback_without_backup_reraises_original():
    with pytest.raises(ExchangeUnavailableError) as info:
        await FallbackMarketData(FakeSource("binance", fail=BINANCE_451), None).get_ticker("BTC")
    assert info.value is BINANCE_451


async def test_fallback_does_not_hide_programming_errors():
    primary = FakeSource("binance", fail=ValueError("timeframe inválido"))
    fallback = FakeSource("bybit")
    with pytest.raises(ValueError):
        await FallbackMarketData(primary, fallback).get_klines("BTC", "15m")
    assert fallback.calls == []


async def test_fallback_aclose_closes_both():
    primary, fallback = FakeSource("binance"), FakeSource("bybit")
    await FallbackMarketData(primary, fallback).aclose()
    assert primary.closed and fallback.closed


async def test_fallback_with_real_connectors_on_451(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/klines").mock(
        return_value=httpx.Response(451, json=load("binance/error_restricted_location.json"))
    )
    respx_mock.get(f"{BYBIT}/v5/market/kline").mock(
        return_value=httpx.Response(200, json=load("bybit/kline.json"))
    )
    clock = FakeClock()
    md = FallbackMarketData(
        BinanceMarketData(settings.exchanges, client, retries=0, clock=clock),
        BybitMarketData(settings.exchanges, client, retries=0, clock=clock),
    )
    candles = await md.get_klines("BTC", "15m", limit=8)
    assert candles.meta.source == "bybit" and candles.meta.fallback
    assert candles.meta.warning == "Binance no respondió (HTTP 451): datos de Bybit"
    assert len(candles.df) == 8 and candles.df["closed"].tolist() == [True] * 7 + [False]


async def test_fallback_with_real_connectors_both_invalid_symbol(settings, client, respx_mock):
    respx_mock.get(f"{FAPI}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(400, json=load("binance/error_invalid_symbol.json"))
    )
    respx_mock.get(f"{BYBIT}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("bybit/error_invalid_symbol.json"))
    )
    md = FallbackMarketData(
        BinanceMarketData(settings.exchanges, client, retries=0),
        BybitMarketData(settings.exchanges, client, retries=0),
    )
    with pytest.raises(ExchangeError) as info:
        await md.get_ticker("NOPE")
    # Las dos rechazaron el pedido: no es "no respondió", reintentar no sirve.
    assert not isinstance(info.value, ExchangeUnavailableError)
    assert "Invalid symbol" in str(info.value) and "Not supported symbols" in str(info.value)
    assert failure_reason(info.value.__cause__) == "símbolo inválido"


async def test_fallback_both_fail_is_unavailable_if_any_did_not_answer():
    primary = FakeSource("binance", fail=BINANCE_451)
    fallback = FakeSource("bybit", fail=ExchangeError("bybit: Not supported symbols (retCode 1)"))
    with pytest.raises(ExchangeUnavailableError):
        await FallbackMarketData(primary, fallback).get_ticker("BTC")
    primary = FakeSource("binance", fail=ExchangeError("binance: HTTP 400 en u: {}"))
    fallback = FakeSource("bybit", fail=ExchangeUnavailableError("bybit: HTTP 503 en u"))
    with pytest.raises(ExchangeUnavailableError):
        await FallbackMarketData(primary, fallback).get_ticker("BTC")


@pytest.mark.parametrize("kind", ["redirect_loop", "bad_gzip"])
async def test_fallback_with_real_connectors_on_invalid_response(
    settings, client, respx_mock, kind
):
    # Regresión: TooManyRedirects / DecodingError de Binance salían crudos y no se
    # probaba Bybit.
    url = f"{FAPI}/fapi/v1/ticker/24hr"
    if kind == "redirect_loop":
        respx_mock.get(url).mock(return_value=httpx.Response(302, headers={"Location": url}))
    else:
        respx_mock.get(url).mock(
            side_effect=lambda request: httpx.Response(
                200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"no es gzip")
            )
        )
    respx_mock.get(f"{BYBIT}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("bybit/tickers.json"))
    )
    md = FallbackMarketData(
        BinanceMarketData(settings.exchanges, client, retries=0, clock=FakeClock()),
        BybitMarketData(settings.exchanges, client, retries=0, clock=FakeClock()),
    )
    ticker = await md.get_ticker("BTC")
    assert ticker.meta.source == "bybit" and ticker.meta.fallback
    assert ticker.meta.warning.startswith("Binance no respondió (respuesta inválida")


# --- caché -------------------------------------------------------------------


def make_cached(settings, source=None, clock=None):
    clock = clock or FakeClock()
    source = source or FakeSource("binance")
    return CachedMarketData(source, TTLCache(clock), settings.cache_ttl_seconds), source, clock


def test_cached_implements_protocol(settings):
    assert isinstance(make_cached(settings)[0], MarketDataSource)


async def test_cached_hit_miss_by_ttl(settings):
    md, source, clock = make_cached(settings)  # price: 30 s
    first = await md.get_ticker("BTC")
    clock.advance(10)
    second = await md.get_ticker("btc")  # misma clave aunque cambie la caja
    assert not first.meta.from_cache
    assert second.meta.from_cache and not second.meta.stale
    assert second.meta.fetched_at == first.meta.fetched_at
    assert len(source.calls) == 1
    clock.advance(25)
    third = await md.get_ticker("BTC")
    assert not third.meta.from_cache and len(source.calls) == 2


async def test_cached_klines_ttl_and_key(settings):
    md, source, clock = make_cached(settings)  # klines_15m: 60 s, klines_5m: 30 s
    await md.get_klines("BTC", "15m")
    await md.get_klines("BTC", "5m")
    clock.advance(45)
    assert (await md.get_klines("BTC", "15m")).meta.from_cache
    assert not (await md.get_klines("BTC", "5m")).meta.from_cache
    assert not (await md.get_klines("BTC", "15m", limit=100)).meta.from_cache  # otra clave
    assert {"md:klines:BTC:15m:500", "md:klines:BTC:5m:500", "md:klines:BTC:15m:100"} <= set(
        md.cache.last_updated()
    )
    assert len(source.calls) == 4


@pytest.mark.parametrize(
    ("method", "args", "key", "ttl_kind"),
    [
        ("get_funding", ("BTC",), "md:funding:BTC", "funding"),
        ("get_funding_history", ("BTC", 21), "md:funding_history:BTC:21", "funding"),
        ("get_open_interest", ("BTC",), "md:open_interest:BTC", "open_interest"),
        (
            "get_open_interest_history",
            ("BTC", "1h", 168),
            "md:open_interest_history:BTC:1h:168",
            "open_interest",
        ),
        (
            "get_long_short_ratio",
            ("BTC", "1h", 168),
            "md:long_short:BTC:1h:168",
            "long_short_ratio",
        ),
    ],
)
async def test_cached_derivatives(settings, method, args, key, ttl_kind):
    md, source, clock = make_cached(settings)
    ttl = settings.cache_ttl_seconds[ttl_kind]
    await getattr(md, method)(*args)
    assert key in md.cache.last_updated()
    clock.advance(ttl - 1)
    assert (await getattr(md, method)(*args)).meta.from_cache
    clock.advance(2)
    assert not (await getattr(md, method)(*args)).meta.from_cache
    assert len(source.calls) == 2


async def test_cached_stale_on_failure(settings):
    md, source, clock = make_cached(settings)
    await md.get_ticker("BTC")
    source.fail = BINANCE_451
    clock.advance(60)
    t = await md.get_ticker("BTC")
    assert t.meta.from_cache and t.meta.stale
    assert t.meta.warning.startswith("Dato desactualizado:")
    assert "HTTP 451" in t.meta.warning
    assert t.meta.fetched_at == FIXTURE_NOW  # la hora del dato, no la de ahora


async def test_cached_propagates_error_without_previous_value(settings):
    md, _, _ = make_cached(settings, FakeSource("binance", fail=BINANCE_451))
    with pytest.raises(ExchangeUnavailableError):
        await md.get_ticker("BTC")


async def test_cached_bypasses_past_klines_and_history(settings):
    md, source, _ = make_cached(settings)
    end = FIXTURE_NOW - timedelta(hours=2)
    for _ in range(2):
        assert not (await md.get_klines("BTC", "15m", 100, end_time=end)).meta.from_cache
        h = await md.get_klines_history("BTC", "1h", FIXTURE_NOW - timedelta(days=3))
        assert not h.meta.from_cache
    assert len(source.calls) == 4
    assert md.cache.last_updated() == {}


async def test_cached_keeps_fallback_warning_on_hits(settings):
    primary, fallback = FakeSource("binance", fail=BINANCE_451), FakeSource("bybit")
    md, _, clock = make_cached(settings, FallbackMarketData(primary, fallback))
    await md.get_ticker("BTC")
    clock.advance(5)
    t = await md.get_ticker("BTC")
    assert t.meta.from_cache and t.meta.fallback and "datos de Bybit" in t.meta.warning


async def test_cached_aclose(settings):
    md, source, _ = make_cached(settings)
    await md.aclose()
    assert source.closed


# --- armado desde config -----------------------------------------------------


def test_build_market_data_wiring(settings, client):
    md = build_market_data(settings, client, TTLCache())
    assert isinstance(md, CachedMarketData) and isinstance(md, MarketDataSource)
    assert isinstance(md.source, FallbackMarketData)
    assert isinstance(md.source.primary, BinanceMarketData)
    assert isinstance(md.source.fallback, BybitMarketData)
    assert md._ttl("price") == settings.cache_ttl_seconds["price"]


def test_build_market_data_variants(settings, client):
    swapped = build_market_data(
        with_exchanges(settings, data_primary="Bybit", data_fallback="binance"), client, TTLCache()
    )
    assert isinstance(swapped.source.primary, BybitMarketData)
    assert isinstance(swapped.source.fallback, BinanceMarketData)

    alone = build_market_data(with_exchanges(settings, data_fallback=None), client, TTLCache())
    assert alone.source.fallback is None

    for bad in ("kraken", "nexo"):
        with pytest.raises(ValueError, match="desconocida"):
            build_market_data(with_exchanges(settings, data_primary=bad), client, TTLCache())
    with pytest.raises(ValueError, match="distinto"):
        build_market_data(with_exchanges(settings, data_fallback="binance"), client, TTLCache())
    with pytest.raises(ValueError):
        make_market_data_source("ftx", settings, client)


async def test_build_market_data_end_to_end(settings, client, respx_mock):
    binance = respx_mock.get(f"{FAPI}/fapi/v1/ticker/24hr").mock(
        return_value=httpx.Response(451, json=load("binance/error_restricted_location.json"))
    )
    bybit = respx_mock.get(f"{BYBIT}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("bybit/tickers.json"))
    )
    clock = FakeClock()
    settings0 = with_exchanges(settings, max_retries=0)
    md = build_market_data(settings0, client, TTLCache(clock), clock=clock)
    first = await md.get_ticker("BTC")
    second = await md.get_ticker("BTC")
    assert first.meta.source == "bybit" and first.meta.fallback
    assert first.meta.warning == "Binance no respondió (HTTP 451): datos de Bybit"
    assert first.price == 108960.8
    assert second.meta.from_cache
    assert binance.call_count == 1 and bybit.call_count == 1


def test_build_nexo_price_source(settings):
    nexo = build_nexo_price_source(settings)
    assert isinstance(nexo, NexoPriceSource) and not nexo.enabled


# --- precio de ejecución -----------------------------------------------------


class FakeNexo:
    name = "nexo"

    def __init__(self, price: float | None = 109010.0) -> None:
        self.price = price
        self.calls = 0

    async def get_ticker(self, symbol):
        self.calls += 1
        if self.price is None:
            raise ExchangeUnavailableError("Nexo: sin acceso")
        return Ticker(symbol, self.price, FetchMeta("nexo", FIXTURE_NOW))

    async def aclose(self):
        pass


async def test_execution_price_defaults_to_binance(settings):
    market, nexo = FakeSource("binance"), FakeNexo()
    ep = await get_execution_price("BTC", market, nexo, settings)
    assert isinstance(ep, ExecutionPrice)
    assert ep.source == "binance" and ep.price == 108962.5 and ep.binance_price == 108962.5
    assert ep.diff_pct is None
    assert ep.warnings == [NEXO_UNAVAILABLE_WARNING]
    assert ep.warnings[0] == (
        "Precio de Nexo no disponible: se usa el de Binance. Podés pegar el precio de Nexo a mano."
    )
    assert ep.meta.source == "binance"
    assert nexo.calls == 0  # Nexo apagado en config: ni se consulta


async def test_execution_price_with_real_disabled_nexo_makes_no_http(settings, respx_mock):
    ep = await get_execution_price(
        "BTC", FakeSource("binance"), build_nexo_price_source(settings), settings
    )
    assert ep.source == "binance"
    assert respx_mock.calls.call_count == 0


async def test_execution_price_manual_close_to_binance(settings):
    clock = FakeClock()
    ep = await get_execution_price(
        "BTC", FakeSource("binance"), FakeNexo(), settings, manual_price=109000.0, clock=clock
    )
    assert ep.source == "manual_nexo" and ep.price == 109000.0
    assert ep.binance_price == 108962.5
    assert ep.diff_pct == pytest.approx((109000 - 108962.5) / 108962.5 * 100)
    assert ep.warnings == []
    assert ep.meta.source == "manual" and ep.meta.fetched_at == FIXTURE_NOW


async def test_execution_price_manual_typo_warning(settings):
    # Le faltó un dígito: 10.896,25 en vez de 108.962,50.
    ep = await get_execution_price(
        "BTC", FakeSource("binance"), FakeNexo(), settings, manual_price=10896.25
    )
    assert ep.source == "manual_nexo" and ep.price == 10896.25
    assert ep.diff_pct == pytest.approx(-90.0)
    assert ep.warnings == [
        "El precio de Nexo que cargaste difiere 90,00 % del de Binance: "
        "revisá que esté bien escrito"
    ]


async def test_execution_price_manual_threshold_comes_from_config(settings):
    strict = with_nexo(settings, manual_price_max_diff_pct=0.01)
    ep = await get_execution_price(
        "BTC", FakeSource("binance"), FakeNexo(), strict, manual_price=109000.0
    )
    assert len(ep.warnings) == 1 and "difiere 0,03 %" in ep.warnings[0]


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf")])
async def test_execution_price_manual_must_be_positive(settings, bad):
    with pytest.raises(ValueError, match="mayor que 0"):
        await get_execution_price("BTC", FakeSource("binance"), FakeNexo(), settings, bad)


async def test_execution_price_manual_when_binance_is_down(settings):
    market = FakeSource("binance", fail=BINANCE_451)
    ep = await get_execution_price("BTC", market, FakeNexo(), settings, manual_price=109000.0)
    assert ep.source == "manual_nexo" and ep.price == 109000.0
    assert ep.binance_price is None and ep.diff_pct is None
    assert len(ep.warnings) == 1 and "HTTP 451" in ep.warnings[0]


async def test_execution_price_manual_keeps_reference_warning(settings):
    market = FakeSource("bybit", warning="Binance no respondió (HTTP 451): datos de Bybit")
    ep = await get_execution_price("BTC", market, FakeNexo(), settings, manual_price=10896.25)
    assert ep.warnings[0].startswith("El precio de Nexo que cargaste difiere")
    assert ep.warnings[1] == "Binance no respondió (HTTP 451): datos de Bybit"


async def test_execution_price_from_enabled_nexo(settings):
    nexo = FakeNexo(109010.0)
    ep = await get_execution_price(
        "BTC", FakeSource("binance"), nexo, with_nexo(settings, enabled=True)
    )
    assert ep.source == "nexo" and ep.price == 109010.0
    assert ep.binance_price == 108962.5
    assert ep.diff_pct == pytest.approx((109010 - 108962.5) / 108962.5 * 100)
    assert ep.warnings == [] and ep.meta.source == "nexo"
    assert nexo.calls == 1


async def test_execution_price_enabled_nexo_failing_uses_binance(settings):
    market = FakeSource("binance")
    enabled = with_nexo(settings, enabled=True)
    ep = await get_execution_price("BTC", market, NexoPriceSource(enabled.exchanges.nexo), enabled)
    assert ep.source == "binance" and ep.warnings == [NEXO_UNAVAILABLE_WARNING]
    assert len(market.calls) == 1  # no vuelve a pedir el ticker


async def test_execution_price_no_price_at_all_raises(settings):
    market = FakeSource("binance", fail=BINANCE_451)
    with pytest.raises(ExchangeUnavailableError):
        await get_execution_price("BTC", market, FakeNexo(), settings)
    with pytest.raises(ExchangeUnavailableError):
        await get_execution_price("BTC", market, FakeNexo(None), with_nexo(settings, enabled=True))


async def test_execution_price_appends_market_warning(settings):
    market = FakeSource("bybit", warning="Binance no respondió (HTTP 451): datos de Bybit")
    ep = await get_execution_price("BTC", market, FakeNexo(), settings)
    assert ep.warnings == [
        "Precio de Nexo no disponible: se usa el de Bybit. Podés pegar el precio de Nexo a mano.",
        "Binance no respondió (HTTP 451): datos de Bybit",
    ]
    assert ep.meta.source == "bybit"


async def test_execution_price_warnings_name_the_real_reference(settings, client, respx_mock):
    # Regresión: con Binance bloqueado el aviso de tipeo comparaba contra Bybit pero
    # decía "del de Binance".
    respx_mock.route(host="fapi.binance.com").mock(return_value=httpx.Response(451))
    respx_mock.get(f"{BYBIT}/v5/market/tickers").mock(
        return_value=httpx.Response(200, json=load("bybit/tickers.json"))
    )
    clock = FakeClock()
    md = build_market_data(with_exchanges(settings, max_retries=0), client, TTLCache(clock))
    nexo = build_nexo_price_source(settings)
    ep = await get_execution_price("BTC", md, nexo, settings, 100_000.0, clock=clock)
    assert ep.binance_price == 108960.8
    assert ep.warnings[0] == (
        "El precio de Nexo que cargaste difiere 8,22 % del de Bybit (respaldo): "
        "revisá que esté bien escrito"
    )
    auto = await get_execution_price("BTC", md, nexo, settings, clock=clock)
    assert auto.warnings[0] == (
        "Precio de Nexo no disponible: se usa el de Bybit (respaldo). "
        "Podés pegar el precio de Nexo a mano."
    )


async def test_execution_price_manual_when_nothing_answers_does_not_name_binance(settings):
    market = FakeSource("binance", fail=BINANCE_451)
    ep = await get_execution_price("BTC", market, FakeNexo(), settings, manual_price=109000.0)
    assert ep.warnings == [
        "No se pudo consultar el precio de referencia (HTTP 451) para comparar: "
        "revisá que el precio de Nexo esté bien escrito."
    ]
