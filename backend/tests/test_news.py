"""Titulares: filtro por límite de palabra, relleno macro, RSS, CryptoPanic y token."""

import dataclasses
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from app.data.cache import TTLCache
from app.data.news import (
    WARN_ALL_FEEDS_FAILED,
    WARN_CRYPTOPANIC_FALLBACK,
    WARN_CRYPTOPANIC_STALE,
    RawNewsItem,
    get_headlines,
    keyword_pattern,
    leader_for,
    normalize_title,
    parse_cryptopanic,
    parse_rss,
    redact,
    sanitize_url,
    select_headlines,
)
from app.exchanges.base import ExchangeError
from app.http_client import make_client

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NOW = datetime(2026, 10, 12, 14, 37, 20, tzinfo=UTC)
COINDESK = "https://www.coindesk.com/arc/outboundfeeds/rss/"
COINTELEGRAPH = "https://cointelegraph.com/rss"
CP_BASE = "https://cryptopanic.com/api/developer/v2"
CP_POSTS = f"{CP_BASE}/posts/"
TOKEN = "tok-SECRETO-9f8e7d6c5b4a"

# Las palabras de config.yaml al escribir estos tests (README de fixtures). Se fijan
# acá para que un cambio en config.yaml no rompa las expectativas de los fixtures.
NEWS_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
    "AVAX": ["avalanche", "avax"],
}
MACRO_KEYWORDS = ["fed", "fomc", "cpi", "inflation", "sec", "etf", "rate", "powell", "tariff"]


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def rss_text(name: str) -> str:
    return (FIXTURES / "rss" / name).read_text()


def cp_payload() -> dict:
    return json.loads((FIXTURES / "cryptopanic" / "posts.json").read_text())


def mock_feeds(coindesk=None, cointelegraph=None):
    a = respx.get(COINDESK).mock(
        return_value=coindesk or httpx.Response(200, text=rss_text("coindesk.xml"))
    )
    b = respx.get(COINTELEGRAPH).mock(
        return_value=cointelegraph or httpx.Response(200, text=rss_text("cointelegraph.xml"))
    )
    return a, b


def titles(result) -> list[str]:
    return [h.title for h in result.headlines]


def assert_no_token(result, caplog=None):
    dumped = repr(dataclasses.asdict(result))
    assert TOKEN not in dumped
    assert "auth_token" not in dumped
    if caplog is not None:
        assert TOKEN not in caplog.text


@pytest.fixture
def news_settings(settings):
    ds = settings.data_sources.model_copy(
        update={
            "news_keywords": NEWS_KEYWORDS,
            "macro_keywords": MACRO_KEYWORDS,
            "rss_feeds": [COINDESK, COINTELEGRAPH],
            "cryptopanic_base_url": CP_BASE,
            "news_max_headlines": 15,
            # Los tests de abajo no dependen de la hora real: el corte por
            # antigüedad y las noticias de BTC se prueban aparte, con hora fija.
            "news_max_age_hours": None,
            "include_leader_news": False,
        }
    )
    ex = settings.exchanges.model_copy(update={"max_retries": 0})
    ttls = {**settings.cache_ttl_seconds, "news": 900}
    return settings.model_copy(
        update={"data_sources": ds, "exchanges": ex, "cache_ttl_seconds": ttls}
    )


@pytest.fixture
def all_items() -> list[RawNewsItem]:
    return parse_rss(rss_text("coindesk.xml"), COINDESK) + parse_rss(
        rss_text("cointelegraph.xml"), COINTELEGRAPH
    )


def select(items, symbol, limit=15, **kw):
    return select_headlines(items, symbol, NEWS_KEYWORDS, MACRO_KEYWORDS, limit, **kw)


# --- límites de palabra ------------------------------------------------------


@pytest.mark.parametrize(
    ("keywords", "text", "expected"),
    [
        (["sol"], "Cross-Border Solution for Merchants", False),
        (["sol"], "Console makers face tariff headwinds", False),
        (["sol"], "Daily launches on SOL-based platforms", True),
        (["solana"], "Solana's Firedancer Client Goes Live", True),
        (["sec"], "Nvidia Second-Quarter Revenue", False),
        (["sec"], "How to secure your seed phrase", False),
        (["sec"], "SEC delays decision on spot AVAX ETF", True),
        (["eth"], "Methodology matters", False),
        (["eth"], "ETH reclaimed its 50-day average", True),
        (["rate"], "Corporate Treasuries Keep Buying Gold", False),
        (["rate"], "Pirate-themed NFT game shuts down", False),
        (["rate"], "boosting rate-cut bets", True),
        (["etf"], "Bitcoin ETFs log $612M inflows", False),
        (["etf"], "XRP ETF approval", True),
        (["fed"], "Fed's Powell: Policy 'Well Positioned'", True),
        (["fed"], "The Federal Reserve chair", False),
        (["federal reserve"], "The Federal  Reserve chair", True),
        (["btc"], "img_btc_chart", False),
    ],
)
def test_keyword_pattern_word_boundaries(keywords, text, expected):
    pattern = keyword_pattern(keywords)
    assert pattern is not None
    assert bool(pattern.search(text)) is expected


def test_keyword_pattern_empty():
    assert keyword_pattern([]) is None
    assert keyword_pattern(["  "]) is None


def test_normalize_title():
    assert normalize_title("  Bitcoin  Slips, Below $109K! ") == normalize_title(
        "bitcoin slips below 109k"
    )


# --- parseo RSS --------------------------------------------------------------


def test_parse_rss_fixtures(all_items):
    assert len(all_items) == 19
    by_title = {i.title: i for i in all_items}
    # &amp; decodificado
    assert "Stocks and Bitcoin Track S&P 500 Lower as Treasury Yields Climb" in by_title
    # offset +0100 pasado a UTC
    eth = by_title["Ethereum developers set date for next hard fork"]
    assert eth.published_at == datetime(2026, 10, 12, 13, 10, 42, tzinfo=UTC)
    assert eth.source == "cointelegraph.com"
    # el HTML de la descripción se limpia
    btc = by_title["Bitcoin ETFs log $612M inflows as BTC eyes $110K"]
    assert "<" not in btc.summary and btc.summary.startswith("US spot Bitcoin ETFs")
    cd = by_title["Bitcoin Slips Below $109K as Traders Brace for U.S. CPI Data"]
    assert cd.source == "coindesk.com"
    assert cd.url.startswith("https://www.coindesk.com/markets/2026/10/12/")
    assert all(i.published_at is not None and i.published_at.tzinfo == UTC for i in all_items)


def test_parse_rss_rejects_non_feed():
    with pytest.raises(ExchangeError):
        parse_rss("<html><body>Access denied</body></html>", COINDESK)


def test_parse_rss_accepts_bytes_and_honours_declared_encoding():
    body = (
        '<?xml version="1.0" encoding="ISO-8859-1"?><rss version="2.0"><channel><title>x</title>'
        "<item><title>Bitcoin sube: año récord</title><link>https://a.b/c</link></item>"
        "</channel></rss>"
    ).encode("latin-1")
    items = parse_rss(body, COINDESK)
    assert [i.title for i in items] == ["Bitcoin sube: año récord"]
    assert parse_rss(rss_text("coindesk.xml").encode(), COINDESK) == parse_rss(
        rss_text("coindesk.xml"), COINDESK
    )


def test_parse_rss_never_opens_a_local_path_given_as_body(tmp_path):
    # Regresión: feedparser.parse(str) trata un string con forma de ruta como archivo local.
    local = tmp_path / "feed.xml"
    local.write_text(rss_text("coindesk.xml"), encoding="utf-8")
    with pytest.raises(ExchangeError):
        parse_rss(str(local), COINTELEGRAPH)
    with pytest.raises(ExchangeError):
        parse_rss(str(local).encode(), COINTELEGRAPH)


def test_parse_rss_never_fetches_a_url_given_as_body(monkeypatch):
    # Regresión: con un string con forma de URL feedparser hacía su propio pedido HTTP
    # (sin timeout de config, invisible para respx: una puerta a SSRF).
    import feedparser

    fetched: list[str] = []

    def fake_get(url, *args, **kwargs):
        fetched.append(url)
        raise AssertionError("feedparser no tiene que hacer pedidos")

    monkeypatch.setattr(feedparser.http, "get", fake_get)
    with pytest.raises(ExchangeError):
        parse_rss("http://169.254.169.254/latest/meta-data/", COINTELEGRAPH)
    assert fetched == []


def test_titles_lose_terminal_control_characters():
    # Regresión: un título armado podía mandar secuencias ANSI/OSC a la consola.
    payload = {
        "results": [
            {
                "kind": "news",
                "title": "Bitcoin \u001b]52;c;ZWNobw==\u0007 rallies \u001b[2J\u009b31m",
                "source": {"title": "Evil\u001b[31m News"},
                "published_at": "2026-10-12T11:00:00Z",
                "url": "https://cryptopanic.com/news/1/",
            }
        ]
    }
    (item,) = parse_cryptopanic(payload)
    assert item.title == "Bitcoin ]52;c;ZWNobw== rallies [2J 31m"
    assert item.source == "Evil [31m News"
    body = (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>'
        "<item><title>&#27;[31mRED&#27;[0m Bitcoin</title><link>https://a.b/c</link></item>"
        "</channel></rss>"
    )
    (rss_item,) = parse_rss(body, COINDESK)
    assert "\x1b" not in rss_item.title and rss_item.title == "[31mRED [0m Bitcoin"


# --- selección ---------------------------------------------------------------


def test_select_btc_coin_first_then_macro(all_items):
    out = select(all_items, "BTC")
    assert [h.title for h in out] == [
        "Bitcoin ETFs log $612M inflows as BTC eyes $110K",
        "Bitcoin Slips Below $109K as Traders Brace for U.S. CPI Data",
        "Stocks and Bitcoin Track S&P 500 Lower as Treasury Yields Climb",
        "Fed's Powell: Policy 'Well Positioned' Ahead of December Meeting",
        "US inflation cools in September, boosting rate-cut bets",
        "Console makers face tariff headwinds ahead of holiday season",
    ]
    by_title = {h.title: h for h in out}
    cpi = by_title["Bitcoin Slips Below $109K as Traders Brace for U.S. CPI Data"]
    assert cpi.currencies == ["BTC"] and cpi.is_macro
    assert not by_title["Bitcoin ETFs log $612M inflows as BTC eyes $110K"].is_macro  # "ETFs"
    powell = by_title["Fed's Powell: Policy 'Well Positioned' Ahead of December Meeting"]
    assert powell.currencies == [] and powell.is_macro
    for trap in ("Nvidia Second", "Payments Startup", "Corporate", "Methodology", "Pirate"):
        assert not any(t.startswith(trap) for t in by_title)


def test_select_sol_ignores_solution_and_console(all_items):
    out = select(all_items, "SOL")
    coin = [h.title for h in out if "SOL" in h.currencies]
    assert coin == [
        "Solana's Firedancer Client Goes Live for All Validators",
        "Solana memecoin launchpad volume hits new record",
    ]
    console = next(h for h in out if h.title.startswith("Console makers"))
    assert console.currencies == [] and console.is_macro  # entra por "tariff", no por "sol"
    assert not any(h.title.startswith("Payments Startup") for h in out)


def test_select_eth_uses_utc_for_order(all_items):
    out = select(all_items, "ETH", limit=2)
    # 13:20:04 UTC va antes que 14:10:42 +0100 (= 13:10:42 UTC).
    assert [h.title for h in out] == [
        "Ether Options Traders Pile Into $5K Calls for December Expiry",
        "Ethereum developers set date for next hard fork",
    ]


def test_select_avax_keeps_coin_headline_with_macro_words(all_items):
    out = select(all_items, "AVAX")
    first, second = out[0], out[1]
    assert first.title == "SEC delays decision on spot AVAX ETF to December"
    assert first.currencies == ["AVAX"] and first.is_macro
    assert second.title == "Avalanche Foundation Unveils $50M Fund for Gaming Subnets"


def test_macro_fill_skips_other_coins(all_items):
    out = select(all_items, "BTC")
    # "SEC ... AVAX ETF" y "Ripple ... XRP ETF" son macro, pero de otra moneda.
    assert not any("AVAX" in h.currencies or "XRP" in h.currencies for h in out)


@pytest.mark.parametrize(
    ("limit", "expected_len", "macro_count"), [(2, 2, 0), (3, 3, 0), (4, 4, 1), (0, 0, 0)]
)
def test_select_respects_limit(all_items, limit, expected_len, macro_count):
    out = select(all_items, "BTC", limit=limit)
    assert len(out) == expected_len
    assert sum(1 for h in out if "BTC" not in h.currencies) == macro_count


def test_select_dedupes_and_puts_undated_last():
    t = datetime(2026, 10, 12, 12, tzinfo=UTC)
    items = [
        RawNewsItem("Bitcoin hits record", "a.com", "https://a.com/1", t),
        RawNewsItem("BITCOIN hits record!", "b.com", "https://b.com/1", t + timedelta(hours=1)),
        RawNewsItem("Bitcoin miners sell", "a.com", "https://a.com/2", None),
        RawNewsItem("BTC funding flips", "a.com", "https://a.com/3", t - timedelta(hours=1)),
    ]
    out = select(items, "BTC")
    assert [(h.title, h.source) for h in out] == [
        ("BITCOIN hits record!", "b.com"),
        ("BTC funding flips", "a.com"),
        ("Bitcoin miners sell", "a.com"),
    ]


def test_select_matches_summary_but_macro_only_title():
    t = datetime(2026, 10, 12, 12, tzinfo=UTC)
    items = [
        RawNewsItem("Market wrap", "a.com", "", t, summary="Bitcoin led the move up."),
        RawNewsItem("Treasury yields jump", "a.com", "", t, summary="Fed rate cut odds fell."),
    ]
    out = select(items, "BTC")
    assert [h.title for h in out] == ["Market wrap"]


def test_select_age_cutoff(all_items):
    out = select(all_items, "BTC", now=FIXTURE_NOW, max_age_hours=6)
    assert "Console makers face tariff headwinds ahead of holiday season" not in [
        h.title for h in out
    ]
    assert len(out) == 5


def test_select_unknown_symbol_uses_symbol_as_keyword():
    t = datetime(2026, 10, 12, 12, tzinfo=UTC)
    items = [
        RawNewsItem("DOGE rallies 10%", "a.com", "", t),
        RawNewsItem("Dogecoin fans", "a.com", "", t),
    ]
    assert [h.title for h in select(items, "doge")] == ["DOGE rallies 10%"]


# --- CryptoPanic: parseo y limpieza de links ---------------------------------


def test_parse_cryptopanic_fixture():
    items = parse_cryptopanic(cp_payload(), TOKEN)
    assert len(items) == 6  # el de kind=media queda afuera
    assert not any(i.title.startswith("Analyst:") for i in items)
    first = items[0]
    assert first.source == "CoinDesk"
    assert first.instruments == ("BTC",)
    assert first.published_at == datetime(2026, 10, 12, 14, 15, 20, tzinfo=UTC)
    assert first.url.startswith("https://coindesk.com/2026/10/12/")
    sol_xrp = next(i for i in items if i.title.startswith("Spot Solana and XRP"))
    assert sol_xrp.instruments == ("SOL", "XRP")


def test_parse_cryptopanic_url_fallback_and_v1_currencies():
    payload = {
        "results": [
            {
                "kind": "news",
                "title": "Bitcoin news",
                "published_at": "2026-10-12T10:00:00Z",
                "original_url": "",
                "url": f"https://cryptopanic.com/news/1/?auth_token={TOKEN}&ref=x",
                "currencies": [{"code": "btc"}],
                "source": {"domain": "example.com"},
            },
            {"kind": "news", "title": "Bad link", "original_url": "javascript:alert(1)"},
        ]
    }
    first, second = parse_cryptopanic(payload, TOKEN)
    assert first.url == "https://cryptopanic.com/news/1/?ref=x"
    assert first.instruments == ("BTC",)
    assert first.source == "example.com"
    assert second.url == "" and second.published_at is None


def test_parse_cryptopanic_rejects_bad_payload():
    with pytest.raises(ExchangeError):
        parse_cryptopanic({"status": "Incomplete", "info": "Token not found"})


def test_redact_and_sanitize():
    assert TOKEN not in redact(f"error con {TOKEN} adentro", TOKEN)
    assert redact("https://x/?auth_token=abc&page=2", None) == "https://x/?auth_token=***&page=2"
    assert sanitize_url(f"https://x.com/a?page=2&AUTH_TOKEN={TOKEN}") == "https://x.com/a?page=2"
    assert sanitize_url(f"https://x.com/{TOKEN}/a", TOKEN) == "https://x.com/***/a"
    assert sanitize_url("ftp://x.com/a") == ""
    assert sanitize_url(None) == ""


# --- get_headlines: RSS ------------------------------------------------------


@respx.mock
async def test_rss_without_token(news_settings):
    a, b = mock_feeds()
    cache = TTLCache(clock=FakeClock(FIXTURE_NOW))
    async with make_client() as client:
        result = await get_headlines("btc", client, cache, news_settings, None)
    assert result.symbol == "BTC"
    assert result.provider == "rss"
    assert result.warnings == []
    assert result.meta.source == "rss"
    assert result.meta.fetched_at == FIXTURE_NOW
    assert not result.meta.stale and not result.meta.fallback and not result.meta.from_cache
    assert titles(result)[0] == "Bitcoin ETFs log $612M inflows as BTC eyes $110K"
    assert len(result.headlines) == 6
    assert a.call_count == b.call_count == 1
    assert "rss" in a.calls.last.request.headers["Accept"]


@respx.mock
async def test_feeds_are_cached_and_shared_between_symbols(news_settings):
    a, b = mock_feeds()
    cache = TTLCache(clock=FakeClock(FIXTURE_NOW))
    async with make_client() as client:
        await get_headlines("BTC", client, cache, news_settings, None)
        eth = await get_headlines("ETH", client, cache, news_settings, None)
    assert a.call_count == b.call_count == 1
    assert eth.meta.from_cache
    assert titles(eth)[:2] == [
        "Ether Options Traders Pile Into $5K Calls for December Expiry",
        "Ethereum developers set date for next hard fork",
    ]


@respx.mock
async def test_default_limit_from_settings(news_settings):
    mock_feeds()
    small = news_settings.model_copy(
        update={
            "data_sources": news_settings.data_sources.model_copy(update={"news_max_headlines": 2})
        }
    )
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), small, None)
    assert len(result.headlines) == 2


@respx.mock
async def test_one_feed_fails_keeps_the_other(news_settings):
    mock_feeds(cointelegraph=httpx.Response(503))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, None)
    assert result.provider == "rss"
    assert {h.source for h in result.headlines} == {"coindesk.com"}
    assert len(result.warnings) == 1 and "cointelegraph.com" in result.warnings[0]
    assert result.meta.warning == result.warnings[0]
    assert not result.meta.stale


@respx.mock
async def test_invalid_feed_body_counts_as_failure(news_settings):
    mock_feeds(coindesk=httpx.Response(200, text="<html>bloqueado</html>"))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, None)
    assert {h.source for h in result.headlines} == {"cointelegraph.com"}
    assert any("coindesk.com" in w for w in result.warnings)


@respx.mock
async def test_feed_body_that_is_a_local_path_is_not_read(news_settings, tmp_path):
    planted = tmp_path / "planted.xml"
    planted.write_text(
        '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>'
        "<item><title>Bitcoin PLANTED-FROM-LOCAL-FILE</title>"
        "<link>https://a.b/c</link></item></channel></rss>",
        encoding="utf-8",
    )
    mock_feeds(coindesk=httpx.Response(200, text=str(planted)))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, None)
    assert not any("PLANTED" in t for t in titles(result))
    assert any("coindesk.com" in w for w in result.warnings)


@respx.mock
async def test_redirect_loop_on_one_feed_keeps_the_other(news_settings):
    # Regresión: TooManyRedirects no se traducía a ExchangeError y tiraba abajo todas
    # las noticias.
    mock_feeds(coindesk=httpx.Response(302, headers={"Location": COINDESK}))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, None)
    assert {h.source for h in result.headlines} == {"cointelegraph.com"}
    assert result.warnings == ["No pudimos leer el feed de coindesk.com; seguimos con el resto."]


@respx.mock
async def test_all_feeds_fail_returns_empty(news_settings):
    mock_feeds(coindesk=httpx.Response(503), cointelegraph=httpx.Response(404))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, None)
    assert result.headlines == []
    assert result.provider == "rss"
    assert WARN_ALL_FEEDS_FAILED in result.warnings
    assert len(result.warnings) == 3
    assert result.meta.stale and result.meta.warning


@respx.mock
async def test_stale_feed_is_used_with_warning(news_settings):
    a, b = mock_feeds()
    clock = FakeClock(FIXTURE_NOW)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_headlines("BTC", client, cache, news_settings, None)
        clock.advance(901)
        b.mock(return_value=httpx.Response(503))
        result = await get_headlines("BTC", client, cache, news_settings, None)
    assert a.call_count == 2 and b.call_count == 2
    assert len(result.headlines) == 6  # cointelegraph sale de la caché
    assert result.meta.stale
    assert result.meta.fetched_at == FIXTURE_NOW  # el dato más viejo usado
    assert len(result.warnings) == 1 and "cointelegraph.com" in result.warnings[0]


# --- get_headlines: CryptoPanic ----------------------------------------------


@respx.mock
async def test_cryptopanic_with_token(news_settings, caplog):
    caplog.set_level(logging.DEBUG)
    route = respx.get(CP_POSTS).mock(return_value=httpx.Response(200, json=cp_payload()))
    a, b = mock_feeds()
    cache = TTLCache(clock=FakeClock(FIXTURE_NOW))
    async with make_client() as client:
        result = await get_headlines("BTC", client, cache, news_settings, TOKEN)
        again = await get_headlines("BTC", client, cache, news_settings, TOKEN)

    assert result.provider == "cryptopanic"
    assert result.meta.source == "cryptopanic" and not result.meta.fallback
    assert result.warnings == []
    assert titles(result) == [
        "Bitcoin reclaims $109K as spot ETFs log third straight day of inflows",
        "Fed minutes show officials split on December rate cut",
    ]
    assert result.headlines[0].url.startswith("https://coindesk.com/")
    assert result.headlines[1].is_macro and result.headlines[1].currencies == []
    params = route.calls.last.request.url.params
    assert params["currencies"] == "BTC"
    assert params["public"] == "true"
    assert params["auth_token"] == TOKEN
    assert route.call_count == 1 and again.meta.from_cache
    assert a.call_count == b.call_count == 0
    assert_no_token(result, caplog)
    assert_no_token(again, caplog)
    assert TOKEN not in repr(cache.last_updated())


@respx.mock
@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503),
        httpx.Response(403),
        httpx.Response(401, text=f'{{"info": "bad token {TOKEN}"}}'),
        httpx.Response(200, json={"status": "Incomplete", "info": f"token {TOKEN}"}),
        httpx.Response(200, text="<html>no es json</html>"),
    ],
)
async def test_cryptopanic_failure_falls_back_to_rss(news_settings, caplog, failure):
    caplog.set_level(logging.DEBUG)
    respx.get(CP_POSTS).mock(return_value=failure)
    mock_feeds()
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, TOKEN)
    assert result.provider == "rss"
    assert result.meta.fallback
    assert result.warnings == [WARN_CRYPTOPANIC_FALLBACK]
    assert len(result.headlines) == 6
    assert_no_token(result, caplog)
    assert "CryptoPanic" in caplog.text  # el fallo queda logueado, sin el token


@respx.mock
async def test_cryptopanic_and_rss_fail(news_settings, caplog):
    caplog.set_level(logging.DEBUG)
    respx.get(CP_POSTS).mock(return_value=httpx.Response(500))
    mock_feeds(coindesk=httpx.Response(500), cointelegraph=httpx.Response(500))
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, TOKEN)
    assert result.headlines == []
    assert result.warnings[0] == WARN_CRYPTOPANIC_FALLBACK
    assert WARN_ALL_FEEDS_FAILED in result.warnings
    assert_no_token(result, caplog)


@respx.mock
async def test_stale_cryptopanic_prefers_rss_then_cache(news_settings, caplog):
    caplog.set_level(logging.DEBUG)
    cp = respx.get(CP_POSTS).mock(return_value=httpx.Response(200, json=cp_payload()))
    a, b = mock_feeds()
    clock = FakeClock(FIXTURE_NOW)
    cache = TTLCache(clock=clock)
    async with make_client() as client:
        await get_headlines("BTC", client, cache, news_settings, TOKEN)
        clock.advance(901)
        cp.mock(return_value=httpx.Response(503))
        via_rss = await get_headlines("BTC", client, cache, news_settings, TOKEN)
        assert via_rss.provider == "rss" and via_rss.meta.fallback
        assert via_rss.warnings == [WARN_CRYPTOPANIC_FALLBACK]

        clock.advance(901)
        a.mock(return_value=httpx.Response(503))
        b.mock(return_value=httpx.Response(503))
        cache.invalidate("rss:")
        old = await get_headlines("BTC", client, cache, news_settings, TOKEN)

    assert old.provider == "cryptopanic"
    assert old.meta.stale and old.meta.fetched_at == FIXTURE_NOW
    assert old.warnings == [WARN_CRYPTOPANIC_STALE]
    assert titles(old)[0].startswith("Bitcoin reclaims")
    for r in (via_rss, old):
        assert_no_token(r, caplog)


@respx.mock
async def test_blank_token_uses_rss(news_settings):
    cp = respx.get(CP_POSTS).mock(return_value=httpx.Response(200, json=cp_payload()))
    mock_feeds()
    async with make_client() as client:
        result = await get_headlines("BTC", client, TTLCache(), news_settings, "   ")
    assert result.provider == "rss" and not result.meta.fallback and result.warnings == []
    assert cp.call_count == 0


async def test_negative_limit_rejected(news_settings):
    async with make_client() as client:
        with pytest.raises(ValueError):
            await get_headlines("BTC", client, TTLCache(), news_settings, None, limit=-1)


# --- noticias de BTC para altcoins y antigüedad (decisiones del usuario) ------


def _item(title: str, hours_ago: float | None) -> RawNewsItem:
    when = None if hours_ago is None else FIXTURE_NOW - timedelta(hours=hours_ago)
    return RawNewsItem(title=title, source="test", url="", published_at=when)


LEADER_ITEMS = [
    _item("Bitcoin rallies past resistance", 1),
    _item("Solana validators upgrade client", 2),
    _item("Fed holds rates steady", 3),
    _item("Ethereum gas fees drop", 4),
    _item("Solana DEX volume hits record", 5),
]


def test_altcoin_gets_leader_news_after_its_own_and_before_macro():
    got = select(LEADER_ITEMS, "SOL", leader="BTC")
    assert [h.title for h in got] == [
        "Solana validators upgrade client",
        "Solana DEX volume hits record",
        "Bitcoin rallies past resistance",
        "Fed holds rates steady",
    ]


def test_leader_is_ignored_for_the_leader_itself_and_when_disabled():
    btc = select(LEADER_ITEMS, "BTC", leader="BTC")
    assert [h.title for h in btc] == ["Bitcoin rallies past resistance", "Fed holds rates steady"]
    sol = select(LEADER_ITEMS, "SOL", leader=None)
    assert "Bitcoin rallies past resistance" not in [h.title for h in sol]


def test_max_age_drops_old_news_but_keeps_undated_ones():
    items = [
        _item("Bitcoin news from today", 2),
        _item("Bitcoin news from three days ago", 72),
        _item("Bitcoin undated news", None),
    ]
    got = select(items, "BTC", now=FIXTURE_NOW, max_age_hours=48)
    assert [h.title for h in got] == ["Bitcoin news from today", "Bitcoin undated news"]


def test_leader_for_respects_config(news_settings):
    on = news_settings.model_copy(
        update={
            "data_sources": news_settings.data_sources.model_copy(
                update={"include_leader_news": True, "leader_symbol": "BTC"}
            )
        }
    )
    assert leader_for("SOL", on) == "BTC"
    assert leader_for("btc", on) is None
    assert leader_for("SOL", news_settings) is None  # desactivado en el fixture


@respx.mock
async def test_get_headlines_uses_config_age_and_leader(news_settings):
    """Con la config por defecto (48 h y BTC para altcoins), usando la hora del fixture."""
    cfg = news_settings.model_copy(
        update={
            "data_sources": news_settings.data_sources.model_copy(
                update={"news_max_age_hours": 48, "include_leader_news": True}
            )
        }
    )
    mock_feeds()
    async with make_client() as client:
        far_future = FIXTURE_NOW + timedelta(days=30)
        stale = await get_headlines("SOL", client, TTLCache(), cfg, None, now=far_future)
        fresh = await get_headlines("SOL", client, TTLCache(), cfg, None, now=FIXTURE_NOW)
    assert [h for h in stale.headlines if h.published_at is not None] == []
    sol = [h for h in fresh.headlines if "SOL" in h.currencies]
    btc_only = [h for h in fresh.headlines if "BTC" in h.currencies and "SOL" not in h.currencies]
    assert sol and btc_only
    order = [h.title for h in fresh.headlines]
    assert max(order.index(h.title) for h in sol) < min(order.index(h.title) for h in btc_only)


@respx.mock
async def test_cryptopanic_asks_for_leader_too(news_settings):
    cfg = news_settings.model_copy(
        update={
            "data_sources": news_settings.data_sources.model_copy(
                update={"include_leader_news": True}
            )
        }
    )
    route = respx.get(f"{CP_BASE}/posts/").mock(return_value=httpx.Response(200, json=cp_payload()))
    async with make_client() as client:
        await get_headlines("SOL", client, TTLCache(), cfg, TOKEN, now=FIXTURE_NOW)
    assert route.calls.last.request.url.params["currencies"] == "SOL,BTC"
