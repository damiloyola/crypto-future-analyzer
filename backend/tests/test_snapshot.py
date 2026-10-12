"""Snapshot: pedidos en paralelo, fallas parciales, datos esenciales y derivados."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import respx

from app.config import AppSettings, Secrets
from app.data.cache import TTLCache
from app.data.snapshot import (
    PRICE_SOURCE_LAST_CLOSE,
    EssentialDataError,
    build_offline_snapshot,
    build_snapshot,
    essential_timeframes,
    funding_summary,
    history_change_pct,
    long_short_change_pct,
    oi_change_pct,
    snapshot_timeframes,
)
from app.data.synthetic import SYNTHETIC_SOURCE, SYNTHETIC_WARNING, SyntheticMarketData
from app.exchanges.base import (
    ExchangeError,
    ExchangeUnavailableError,
    FetchMeta,
    FundingHistory,
    FundingRatePoint,
    LongShortHistory,
    LongShortPoint,
    OpenInterestHistory,
    OpenInterestPoint,
)
from app.exchanges.registry import nexo_unavailable_warning
from tests.fakes import (
    BINANCE_451,
    FIXED_NOW,
    FakeMarketData,
    FakePriceSource,
    synthetic_extras,
)

FIXTURES = Path(__file__).parent / "fixtures"
SECRETS = Secrets.model_construct(anthropic_api_key=None, cryptopanic_api_key=None)


@pytest.fixture(scope="module")
def market() -> SyntheticMarketData:
    return SyntheticMarketData(now=FIXED_NOW)


async def _snapshot(settings: AppSettings, md: FakeMarketData, **kwargs):
    extras = kwargs.pop("extras", None) or synthetic_extras(md.synthetic, "BTC")
    return await build_snapshot(
        "BTC",
        settings,
        md,
        kwargs.pop("nexo", FakePriceSource()),
        None,
        None,
        SECRETS,
        extras=extras,
        clock=lambda: FIXED_NOW,
        **kwargs,
    )


# --- foto completa ------------------------------------------------------------------


async def test_offline_snapshot_is_complete_and_flagged(settings):
    snap = await build_offline_snapshot("btc", settings, now=FIXED_NOW)
    assert snap.symbol == "BTC"
    assert snap.offline is True
    assert snap.created_at == FIXED_NOW
    assert list(snap.candles) == ["1d", "4h", "1h", "15m", "5m"]
    for tf, c in snap.candles.items():
        assert len(c.df) == settings.timeframes.candles_per_request, tf
        assert c.meta.source == SYNTHETIC_SOURCE
    for part in (
        snap.ticker,
        snap.execution_price,
        snap.funding,
        snap.funding_history,
        snap.open_interest,
        snap.oi_history,
        snap.long_short,
        snap.fear_greed,
        snap.global_market,
        snap.news,
    ):
        assert part is not None
    assert snap.oi_change_24h_pct is not None
    assert snap.current_price == snap.ticker.price
    assert snap.price_source == "binance"
    # El aviso de datos sintéticos aparece una sola vez aunque lo traiga cada dato.
    assert snap.warnings.count(SYNTHETIC_WARNING) == 1
    assert nexo_unavailable_warning("Binance sintético") in snap.warnings
    assert all(s.ok and s.source == SYNTHETIC_SOURCE for s in snap.sources)
    names = [s.name for s in snap.sources]
    assert "Velas 15m" in names and "Fear & Greed" in names and "Precio de ejecución" in names


async def test_offline_snapshot_with_manual_nexo_price(settings):
    market = SyntheticMarketData(now=FIXED_NOW)
    reference = (await market.get_ticker("BTC")).price
    manual = reference * 1.004
    snap = await build_offline_snapshot("BTC", settings, manual, market=market)
    assert snap.current_price == manual
    assert snap.price_source == "manual_nexo"
    assert snap.execution_price.binance_price == pytest.approx(reference)
    assert snap.execution_price.diff_pct == pytest.approx(0.4)


async def test_requests_are_concurrent(settings):
    md = FakeMarketData(delay_s=0.05)
    snap = await _snapshot(settings, md)
    n_calls = len(md.calls)
    # 5 timeframes + ticker + funding x2 + OI x2 + long/short
    assert n_calls == len(snapshot_timeframes(settings)) + 6
    assert md.max_in_flight == n_calls
    assert snap.ticker is not None


async def test_request_parameters_come_from_config(settings):
    md = FakeMarketData()
    await _snapshot(settings, md)
    limit = settings.timeframes.candles_per_request
    klines = [c for c in md.calls if c[0] == "get_klines"]
    assert sorted(c[2] for c in klines) == sorted(snapshot_timeframes(settings))
    assert {c[3] for c in klines} == {limit}
    expected_funding = math.ceil(7 * 24 / settings.fees.funding_interval_hours)
    assert ("get_funding_history", "BTC", expected_funding) in md.calls
    assert ("get_open_interest_history", "BTC", "1h", 168) in md.calls
    assert ("get_long_short_ratio", "BTC", "1h", 168) in md.calls


async def test_ticker_is_requested_once_for_the_execution_price(settings):
    md = FakeMarketData()
    snap = await _snapshot(settings, md)
    assert md.methods_called().count("get_ticker") == 1
    assert snap.execution_price.price == snap.ticker.price


# --- fallas parciales -------------------------------------------------------------


async def test_optional_failures_become_warnings(settings):
    md = FakeMarketData(
        fail={
            "get_funding": BINANCE_451,
            "get_open_interest_history": ExchangeError("binance: respuesta inválida"),
            "get_long_short_ratio": BINANCE_451,
        }
    )
    snap = await _snapshot(settings, md)
    assert snap.funding is None
    assert snap.oi_history is None
    assert snap.oi_change_24h_pct is None
    assert snap.long_short is None
    assert snap.funding_history is not None and snap.open_interest is not None
    assert "No se pudo obtener el funding (HTTP 451): el análisis sigue sin ese dato." in (
        snap.warnings
    )
    assert any("el ratio long/short (HTTP 451)" in w for w in snap.warnings)
    assert any("el historial de OI (respuesta inválida)" in w for w in snap.warnings)
    failed = {s.name for s in snap.sources if not s.ok}
    assert failed == {"Funding", "Interés abierto (historial)", "Long/short"}
    for s in snap.sources:
        if not s.ok:
            assert s.source is None and s.fetched_at is None and s.warning


async def test_extra_source_failures_become_warnings(settings, caplog):
    md = FakeMarketData()
    extras = synthetic_extras(
        md.synthetic,
        fail={
            "fear_greed": ExchangeUnavailableError("alternative.me: HTTP 503 en https://x"),
            "global_market": httpx.ConnectError("sin red"),
            "news": RuntimeError("bug inesperado"),
        },
    )
    snap = await _snapshot(settings, md, extras=extras)
    assert snap.fear_greed is None and snap.global_market is None and snap.news is None
    assert any("índice de miedo y codicia (HTTP 503)" in w for w in snap.warnings)
    assert any("mercado global" in w for w in snap.warnings)
    assert any("las noticias" in w for w in snap.warnings)
    # Un error que no es de red se registra con traceback para no esconder bugs.
    assert any("Error inesperado" in r.message for r in caplog.records)


async def test_fallback_warning_appears_once(settings):
    warning = "Binance no respondió (HTTP 451): datos de Bybit"
    md = FakeMarketData(meta_overrides={"fallback": True, "warning": warning})
    snap = await _snapshot(settings, md)
    assert snap.warnings.count(warning) == 1
    market_rows = [s for s in snap.sources if s.name.startswith(("Velas", "Funding", "Precio"))]
    assert market_rows and all(s.fallback and s.warning == warning for s in market_rows)


async def test_stale_values_are_marked_in_sources(settings):
    md = FakeMarketData(meta_overrides={"stale": True, "from_cache": True, "warning": "viejo"})
    snap = await _snapshot(settings, md)
    row = next(s for s in snap.sources if s.name == "Funding")
    assert row.stale and row.from_cache and row.ok
    assert "viejo" in snap.warnings


# --- datos esenciales -------------------------------------------------------------


@pytest.mark.parametrize("tf", ["15m", "1h", "4h", "1d"])
async def test_missing_essential_candles_raise(settings, tf):
    md = FakeMarketData(fail_timeframes={tf: BINANCE_451})
    with pytest.raises(EssentialDataError) as info:
        await _snapshot(settings, md)
    msg = str(info.value)
    assert f"No se pudieron obtener las velas de {tf} de BTC" in msg
    assert "HTTP 451" in msg
    assert isinstance(info.value, ExchangeUnavailableError)
    assert info.value.__cause__ is BINANCE_451


async def test_empty_essential_candles_raise(settings):
    md = FakeMarketData(empty_timeframes={"1h"})
    with pytest.raises(EssentialDataError, match="1h"):
        await _snapshot(settings, md)


async def test_both_exchanges_down_lists_every_missing_timeframe(settings):
    down = ExchangeUnavailableError("Binance y Bybit fallaron. Binance: x. Bybit: y")
    md = FakeMarketData(fail={"get_klines": down})
    with pytest.raises(EssentialDataError) as info:
        await _snapshot(settings, md)
    for tf in essential_timeframes(settings):
        assert tf in str(info.value)


async def test_invalid_symbol_on_both_exchanges_says_so(settings):
    # Regresión: para un símbolo que no existe se mostraban códigos ("HTTP 400",
    # "retCode 10001") y se sugería probar de nuevo en unos minutos.
    both = ExchangeError(
        "Binance y Bybit fallaron. "
        "Binance: binance: Invalid symbol. (HTTP 400, código -1121) en https://fapi.binance.com/x. "
        "Bybit: bybit: Not supported symbols (retCode 10001) en /v5/market/kline"
    )
    md = FakeMarketData(fail={"get_klines": both, "get_ticker": both})
    with pytest.raises(EssentialDataError) as info:
        await _snapshot(settings, md)
    msg = str(info.value)
    assert "Binance (símbolo inválido) y Bybit (símbolo inválido) fallaron" in msg
    assert "Parece que BTC no existe como futuro en USDT" in msg
    assert "revisá que el símbolo esté bien escrito (por ejemplo BTC, ETH, SOL, XRP o AVAX)" in msg
    assert "probá de nuevo" not in msg
    assert info.value.__cause__ is both


async def test_rejected_request_does_not_suggest_retrying(settings):
    rejected = ExchangeError("binance: HTTP 400 en https://fapi.binance.com/x: {}")
    md = FakeMarketData(fail_timeframes={"1h": rejected})
    with pytest.raises(EssentialDataError) as info:
        await _snapshot(settings, md)
    msg = str(info.value)
    assert "Los exchanges rechazaron el pedido" in msg and "probá de nuevo" not in msg


async def test_mixed_failures_still_suggest_retrying(settings):
    rejected = ExchangeError("binance: HTTP 400 en https://fapi.binance.com/x: {}")
    md = FakeMarketData(fail_timeframes={"1h": rejected, "4h": BINANCE_451})
    with pytest.raises(EssentialDataError, match="probá de nuevo en unos minutos"):
        await _snapshot(settings, md)


async def test_explicit_non_essential_timeframes_all_failing_raise(settings):
    # Regresión: con timeframes=["5m"] (ninguno esencial), si fallaban las velas y el
    # ticker se rompía con "RuntimeError: coroutine raised StopIteration".
    md = FakeMarketData(fail={"get_ticker": BINANCE_451}, fail_timeframes={"5m": BINANCE_451})
    with pytest.raises(EssentialDataError) as info:
        await _snapshot(settings, md, timeframes=["5m"])
    assert "No se pudieron obtener las velas de 5m de BTC (HTTP 451)" in str(info.value)
    # Con el ticker andando tampoco devuelve una foto sin velas.
    md = FakeMarketData(fail_timeframes={"5m": BINANCE_451})
    with pytest.raises(EssentialDataError):
        await _snapshot(settings, md, timeframes=["5m"])


async def test_fine_tune_candles_are_optional(settings):
    fine = settings.timeframes.fine_tune
    assert fine and fine not in essential_timeframes(settings)
    md = FakeMarketData(fail_timeframes={fine: BINANCE_451})
    snap = await _snapshot(settings, md)
    assert fine not in snap.candles
    assert any(f"las velas de {fine} (HTTP 451)" in w for w in snap.warnings)


async def test_explicit_timeframes(settings):
    md = FakeMarketData()
    snap = await _snapshot(settings, md, timeframes=["15m"])
    assert list(snap.candles) == ["15m"]
    with pytest.raises(ValueError, match="Timeframes desconocidos"):
        await _snapshot(settings, md, timeframes=["2h"])


async def test_invalid_symbol(settings):
    md = FakeMarketData()
    with pytest.raises(ValueError, match="Símbolo inválido"):
        await build_snapshot(
            "BTC/USDT",
            settings,
            md,
            FakePriceSource(),
            None,
            None,
            SECRETS,
            extras=synthetic_extras(md.synthetic),
        )


# --- precio actual ----------------------------------------------------------------


async def test_ticker_failure_falls_back_to_last_close(settings):
    md = FakeMarketData(fail={"get_ticker": BINANCE_451})
    snap = await _snapshot(settings, md)
    entry = settings.timeframes.entry
    assert snap.ticker is None and snap.execution_price is None
    assert snap.price_source == PRICE_SOURCE_LAST_CLOSE
    assert snap.current_price == pytest.approx(snap.candles[entry].df["close"].iloc[-1])
    assert any("No se pudo obtener el precio actual (HTTP 451)" in w for w in snap.warnings)
    assert any("último cierre" in w for w in snap.warnings)
    row = next(s for s in snap.sources if s.name == "Precio de ejecución")
    assert row.ok is False


async def test_manual_price_still_works_without_ticker(settings):
    md = FakeMarketData(fail={"get_ticker": BINANCE_451})
    snap = await _snapshot(settings, md, manual_nexo_price=61_000.0)
    assert snap.current_price == 61_000.0
    assert snap.price_source == "manual_nexo"
    assert snap.execution_price.binance_price is None
    assert any("revisá que el precio de Nexo" in w for w in snap.warnings)


async def test_manual_price_typo_warning(settings):
    md = FakeMarketData()
    reference = (await md.synthetic.get_ticker("BTC")).price
    snap = await _snapshot(settings, md, manual_nexo_price=reference * 1.10)
    assert snap.execution_price.diff_pct == pytest.approx(10.0)
    assert any("difiere 10,00 %" in w for w in snap.warnings)


async def test_invalid_manual_price_raises(settings):
    md = FakeMarketData()
    with pytest.raises(ValueError, match="mayor que 0"):
        await _snapshot(settings, md, manual_nexo_price=-1.0)


async def test_enabled_nexo_source_is_used(settings):
    nexo_on = settings.model_copy(
        update={
            "exchanges": settings.exchanges.model_copy(
                update={"nexo": settings.exchanges.nexo.model_copy(update={"enabled": True})}
            )
        }
    )
    md = FakeMarketData()
    snap = await _snapshot(nexo_on, md, nexo=FakePriceSource(price=50_000.0))
    assert snap.price_source == "nexo"
    assert snap.current_price == 50_000.0


# --- derivados (funciones puras) --------------------------------------------------


def _oi(values: list[float | None], step_h: float = 1.0) -> OpenInterestHistory:
    start = FIXED_NOW - timedelta(hours=step_h * (len(values) - 1))
    points = [
        OpenInterestPoint(start + timedelta(hours=step_h * i), v if v is not None else math.nan)
        for i, v in enumerate(values)
    ]
    return OpenInterestHistory("BTC", "1h", points, FetchMeta("test", FIXED_NOW))


def test_oi_change_over_24h():
    values = [100.0] + [105.0] * 23 + [110.0]  # 25 puntos horarios: t-24h ... t
    assert oi_change_pct(_oi(values), 24) == pytest.approx(10.0)


def test_oi_change_uses_last_point_at_least_24h_old():
    values = [50.0, 100.0] + [105.0] * 23 + [90.0]  # 26 puntos: el de t-24h vale 100
    assert oi_change_pct(_oi(values), 24) == pytest.approx(-10.0)


def test_oi_change_missing_cases():
    assert oi_change_pct(None, 24) is None
    assert oi_change_pct(_oi([100.0] * 10), 24) is None  # no hay un punto de hace 24 h
    assert oi_change_pct(_oi([0.0] + [5.0] * 24), 24) is None  # base 0
    assert oi_change_pct(_oi([None] + [5.0] * 24), 24) is None  # base NaN
    empty = OpenInterestHistory("BTC", "1h", [], FetchMeta("test", FIXED_NOW))
    assert oi_change_pct(empty, 24) is None


def test_long_short_change():
    start = FIXED_NOW - timedelta(hours=24)
    points = [LongShortPoint(start, 2.0), LongShortPoint(FIXED_NOW, 1.5)]
    history = LongShortHistory("BTC", "1h", points, FetchMeta("test", FIXED_NOW))
    assert long_short_change_pct(history, 24) == pytest.approx(-25.0)
    assert long_short_change_pct(None, 24) is None


def test_history_change_pct_generic():
    class P:
        def __init__(self, h: float, v: float) -> None:
            self.time = FIXED_NOW + timedelta(hours=h)
            self.v = v

    pts = [P(0, 10.0), P(12, 11.0), P(24, 12.0)]
    assert history_change_pct(pts, 24, lambda p: p.v) == pytest.approx(20.0)
    assert history_change_pct(pts, 12, lambda p: p.v) == pytest.approx(12 / 11 * 100 - 100)
    assert history_change_pct([], 24, lambda p: p.v) is None


def test_funding_summary():
    rates = [0.0001, -0.0002, 0.0003]
    points = [FundingRatePoint(FIXED_NOW + timedelta(hours=8 * i), r) for i, r in enumerate(rates)]
    s = funding_summary(FundingHistory("BTC", points, FetchMeta("test", FIXED_NOW)))
    assert s.count == 3
    assert s.mean == pytest.approx(0.0002 / 3)
    assert s.min == -0.0002 and s.max == 0.0003
    assert s.total == pytest.approx(0.0002)
    assert s.positive_share == pytest.approx(2 / 3)
    assert s.start == points[0].time and s.end == points[-1].time
    assert funding_summary(None) is None
    assert funding_summary(FundingHistory("BTC", [], FetchMeta("test", FIXED_NOW))) is None


def test_offline_oi_change_matches_history(settings):
    import asyncio

    snap = asyncio.run(build_offline_snapshot("ETH", settings, now=FIXED_NOW))
    lookback = settings.scoring.derivatives.oi_lookback_hours
    assert snap.oi_change_24h_pct == pytest.approx(oi_change_pct(snap.oi_history, lookback))
    assert len(snap.oi_history.points) == 168


# --- fuentes reales con respx (sin red) -------------------------------------------


async def test_live_extra_sources_with_recorded_responses(settings):
    ds = settings.data_sources
    md = FakeMarketData()
    async with httpx.AsyncClient() as client:
        with respx.mock(assert_all_called=False) as router:
            router.get(url__startswith=ds.fear_greed_url).mock(
                return_value=httpx.Response(
                    200, content=(FIXTURES / "alternative_me/fng.json").read_bytes()
                )
            )
            router.get(url__startswith=ds.coingecko_base_url).mock(
                return_value=httpx.Response(
                    200, content=(FIXTURES / "coingecko/global.json").read_bytes()
                )
            )
            router.get(ds.rss_feeds[0]).mock(
                return_value=httpx.Response(
                    200, content=(FIXTURES / "rss/coindesk.xml").read_bytes()
                )
            )
            router.get(ds.rss_feeds[1]).mock(return_value=httpx.Response(451))
            snap = await build_snapshot(
                "BTC",
                settings,
                md,
                FakePriceSource(),
                client,
                TTLCache(clock=lambda: FIXED_NOW),
                SECRETS,
                clock=lambda: FIXED_NOW,
            )
    assert snap.fear_greed is not None and snap.fear_greed.value == 38
    assert snap.fear_greed.classification_es == "Miedo"
    assert snap.global_market is not None and snap.global_market.total_market_cap_usd > 1e12
    assert snap.news is not None and snap.news.provider == "rss"
    assert snap.news.headlines
    # El feed que falló deja un aviso, pero las noticias siguen.
    assert any("cointelegraph" in w for w in snap.warnings)
    row = next(s for s in snap.sources if s.name == "Fear & Greed")
    assert row.source == "alternative.me" and row.ok


async def test_live_extras_need_client_and_cache(settings):
    with pytest.raises(ValueError, match="client y cache"):
        await build_snapshot("BTC", settings, FakeMarketData(), FakePriceSource(), None, None, None)


# --- motivos de falla -------------------------------------------------------------


def test_describe_failure_names_both_exchanges():
    from app.data.snapshot import describe_failure

    both = ExchangeUnavailableError(
        "Binance y Bybit fallaron. Binance: binance: HTTP 451 en https://fapi.binance.com/x. "
        "Bybit: bybit: retCode 10006: Too many visits"
    )
    assert describe_failure(both) == "Binance (HTTP 451) y Bybit (retCode 10006) fallaron"
    assert describe_failure(BINANCE_451) == "HTTP 451"
    assert describe_failure(ExchangeError("binance: sin respuesta (ConnectError)")) == (
        "sin conexión"
    )


# --- mercado sintético ------------------------------------------------------------


def test_synthetic_timeframes_are_consistent(market):
    base = market.candles_frame("BTC", "5m")
    hourly = market.candles_frame("BTC", "1h")
    for open_time in hourly.index[-30:-1]:
        chunk = base[(base.index >= open_time) & (base.index < open_time + timedelta(hours=1))]
        assert len(chunk) == 12
        row = hourly.loc[open_time]
        assert row["open"] == chunk["open"].iloc[0]
        assert row["close"] == chunk["close"].iloc[-1]
        assert row["high"] == chunk["high"].max()
        assert row["low"] == chunk["low"].min()
        assert row["volume"] == pytest.approx(chunk["volume"].sum())
    daily = market.candles_frame("BTC", "1d")
    assert daily["close"].iloc[-1] == base["close"].iloc[-1]
    assert (daily["high"] >= daily[["open", "close"]].max(axis=1)).all()
    assert (daily["low"] <= daily[["open", "close"]].min(axis=1)).all()


@pytest.mark.parametrize("tf", ["5m", "15m", "1h", "4h", "1d"])
async def test_synthetic_only_last_candle_is_forming(market, tf):
    candles = await market.get_klines("ETH", tf, 500)
    df = candles.df
    assert len(df) == 500
    assert not df["closed"].iloc[-1]
    assert df["closed"].iloc[:-1].all()
    assert (
        df.index[-1]
        <= FIXED_NOW
        < df.index[-1]
        + timedelta(minutes={"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}[tf])
    )
    assert candles.meta.source == SYNTHETIC_SOURCE
    assert candles.meta.warning == SYNTHETIC_WARNING


async def test_synthetic_is_deterministic():
    a = SyntheticMarketData(now=FIXED_NOW, seed=3)
    b = SyntheticMarketData(now=FIXED_NOW, seed=3)
    c = SyntheticMarketData(now=FIXED_NOW, seed=4)
    fa, fb, fc = (m.candles_frame("SOL", "1h") for m in (a, b, c))
    assert fa.equals(fb)
    assert not fa["close"].equals(fc["close"])
    assert (await a.get_funding("SOL")) == (await b.get_funding("SOL"))
    assert a.headlines("SOL") == b.headlines("SOL")


@pytest.mark.parametrize(
    ("symbol", "low", "high"),
    [("BTC", 85_000, 110_000), ("ETH", 3_000, 4_200), ("XRP", 2.0, 3.2), ("AVAX", 22, 35)],
)
async def test_synthetic_prices_are_realistic(market, symbol, low, high):
    ticker = await market.get_ticker(symbol)
    assert low < ticker.price < high
    assert ticker.low_24h <= ticker.price <= ticker.high_24h
    assert -30 < ticker.change_24h_pct < 30


async def test_synthetic_derivatives(market, settings):
    funding = await market.get_funding("BTC")
    assert funding.next_funding_time > FIXED_NOW
    assert -0.01 < funding.rate < 0.01
    history = await market.get_funding_history("BTC", 21)
    times = [p.time for p in history.points]
    assert len(times) == 21 and times == sorted(times)
    assert {(b - a) for a, b in zip(times, times[1:], strict=False)} == {timedelta(hours=8)}
    oi = await market.get_open_interest_history("BTC", "1h", 168)
    assert len(oi.points) == 168 and all(p.contracts > 0 and p.value_usd > 0 for p in oi.points)
    ls = await market.get_long_short_ratio("BTC", "4h", 30)
    assert len(ls.points) == 30
    assert all(p.long_pct + p.short_pct == pytest.approx(1.0) for p in ls.points)
    fg = market.fear_greed("BTC")
    assert 0 <= fg.value <= 100 and len(fg.history) == 30
    assert fg.classification_es in {
        "Miedo extremo",
        "Miedo",
        "Neutral",
        "Codicia",
        "Codicia extrema",
    }
    gm = market.global_market("BTC")
    assert 0 < gm.btc_dominance_pct < 100
    news = market.headlines("BTC", 3)
    assert len(news.headlines) == 3
    assert all(h.title.startswith("(sintético)") for h in news.headlines)


async def test_synthetic_klines_window_and_symbols(market):
    end = FIXED_NOW - timedelta(days=2)
    c = await market.get_klines("BTC", "1h", 10, end_time=end)
    assert len(c.df) == 10 and c.df.index[-1] <= end
    h = await market.get_klines_history("BTC", "4h", FIXED_NOW - timedelta(days=3))
    assert h.df.index[0] >= FIXED_NOW - timedelta(days=3)
    unknown = await market.get_ticker("doge")
    assert unknown.symbol == "DOGE" and unknown.price > 0
    with pytest.raises(ValueError, match="Símbolo inválido"):
        await market.get_ticker("BTC/USDT")
    with pytest.raises(ValueError, match="Timeframe no soportado"):
        await market.get_klines("BTC", "2h", 10)  # type: ignore[arg-type]


# --- avisos de noticias y redacción -----------------------------------------------


def _news_extras(md: FakeMarketData, result):
    from app.data.snapshot import ExtraSources

    async def fear_greed():
        return md.synthetic.fear_greed("BTC")

    async def global_market():
        raise ExchangeUnavailableError("coingecko: HTTP 451 en https://api.coingecko.com")

    async def news():
        return result

    return ExtraSources(fear_greed, global_market, news)


async def test_news_warnings_are_not_duplicated(settings):
    from app.data.news import NewsResult

    md = FakeMarketData()
    feed_a, feed_b = "No pudimos leer el feed de a.com.", "No pudimos leer el feed de b.com."
    news = md.synthetic.headlines("BTC")
    partial = NewsResult(
        "BTC",
        news.headlines,
        "rss",
        FetchMeta("rss", FIXED_NOW, warning=f"{feed_a} {feed_b}"),
        [feed_a, feed_b],
    )
    snap = await _snapshot(settings, md, extras=_news_extras(md, partial))
    assert feed_a in snap.warnings and feed_b in snap.warnings
    assert f"{feed_a} {feed_b}" not in snap.warnings
    assert next(s for s in snap.sources if s.name == "Noticias").ok
    # Concordancia: "los datos" → "No se pudieron obtener".
    assert any(
        w.startswith("No se pudieron obtener los datos del mercado global") for w in snap.warnings
    )


async def test_news_with_every_source_down_shows_as_failed(settings):
    from app.data.news import NewsResult

    md = FakeMarketData()
    msg = "No pudimos leer ningún feed de noticias."
    empty = NewsResult(
        "BTC", [], "rss", FetchMeta("rss", FIXED_NOW, stale=True, warning=msg), [msg]
    )
    snap = await _snapshot(settings, md, extras=_news_extras(md, empty))
    row = next(s for s in snap.sources if s.name == "Noticias")
    assert row.ok is False and row.fetched_at is None
    assert snap.warnings.count(msg) == 1
    assert snap.news is not None and snap.news.headlines == []
