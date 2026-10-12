"""Titulares de noticias filtrados por moneda.

``get_headlines(symbol, ...)`` devuelve un ``NewsResult`` con hasta ``limit``
titulares (por defecto ``data_sources.news_max_headlines``): título, fuente,
link, fecha en UTC, monedas que nombra y si es macro. El impacto
(positivo/negativo/neutral) lo clasifica el LLM, no este módulo.

Proveedores:

- **Con token de CryptoPanic** (``CRYPTOPANIC_API_KEY`` en ``.env``, lo pasa
  quien llama): ``GET {data_sources.cryptopanic_base_url}/posts/`` con
  ``currencies=<símbolo>``, ``public=true`` y ``kind=news``. Si falla (o solo
  queda un valor viejo en la caché), se usan los RSS con un aviso; si los RSS
  también fallan y había titulares viejos de CryptoPanic, se devuelven esos.
- **Sin token**: los feeds de ``data_sources.rss_feeds``, descargados en paralelo
  y parseados con ``feedparser``. Si un feed falla se avisa y se sigue con el
  resto; si fallan todos, se devuelve una lista vacía con un aviso (no lanza).
  El cuerpo descargado se le pasa a ``feedparser`` como stream de bytes: así
  respeta el encoding que declara el XML y nunca interpreta el contenido como
  una URL a pedir o una ruta de archivo a leer (con un string lo haría).

Los títulos, resúmenes y fuentes se limpian (``clean_text``): sin HTML, sin
entidades, sin caracteres de control (un título armado no puede meter
secuencias de escape en la terminal) y con espacios simples.

El token **nunca** sale de acá: no va en las claves de caché, ni en los avisos,
ni en los metadatos, ni en los links devueltos (se limpia ``auth_token`` de
cualquier URL), ni en los logs. Mientras dura el pedido a CryptoPanic se le
agrega al logger de ``httpx`` un filtro que tapa el token, porque ``httpx``
loguea la URL completa (con la query) a nivel INFO.

Filtrado (función pura ``select_headlines``):

1. Se descartan duplicados por título normalizado (queda el más nuevo).
2. Monedas: regex por **límite de palabra**, sin distinguir mayúsculas, con
   ``data_sources.news_keywords`` sobre título + resumen (sin HTML). Así "sol"
   no matchea "solution" ni "console", ni "eth" matchea "methodology". En
   CryptoPanic se suman las monedas de ``instruments``.
3. Macro: misma regex con ``data_sources.macro_keywords`` pero **solo sobre el
   título** (en los resúmenes palabras como "rate" aparecen en cualquier nota).
4. Primero los titulares de la moneda pedida, del más nuevo al más viejo. Si
   sobra lugar, se completa con titulares macro que no nombran ninguna moneda.
   Los que no tienen fecha van al final de su grupo.

Caché (TTL ``cache_ttl_seconds.news``): cada feed RSS se guarda una sola vez
(``"rss:<url>"``) y se reutiliza para todas las monedas; CryptoPanic se guarda
por moneda (``"cryptopanic:posts:<símbolo>"``).
"""

from __future__ import annotations

import asyncio
import calendar
import html
import io
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx

from app.config import AppSettings
from app.data.cache import CacheResult, TTLCache, ttl_for, utcnow
from app.exchanges.base import ExchangeError, FetchMeta
from app.http_client import get_bytes, get_json

logger = logging.getLogger(__name__)

Provider = Literal["cryptopanic", "rss"]

SOURCE_CRYPTOPANIC = "cryptopanic"
SOURCE_RSS = "rss"
REDACTED = "***"
RSS_ACCEPT = "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8"
# Loggers de terceros que pueden escribir la URL del pedido (con el token).
_HTTP_LOGGERS = ("httpx", "httpcore")

WARN_CRYPTOPANIC_FALLBACK = (
    "No pudimos usar CryptoPanic (no respondió o rechazó la key), así que usamos "
    "los feeds RSS como respaldo."
)
WARN_CRYPTOPANIC_STALE = (
    "CryptoPanic no respondió y los feeds RSS tampoco: te mostramos los últimos "
    "titulares que teníamos guardados."
)
WARN_ALL_FEEDS_FAILED = (
    "No pudimos leer ningún feed de noticias. Por ahora no hay titulares; probá de nuevo "
    "en unos minutos."
)


def warn_feed_failed(label: str) -> str:
    return f"No pudimos leer el feed de {label}; seguimos con el resto."


def warn_feed_stale(label: str) -> str:
    return f"El feed de {label} no respondió; te mostramos los titulares que habíamos guardado."


# --- modelos -----------------------------------------------------------------


@dataclass(frozen=True)
class Headline:
    title: str
    source: str  # medio: "CoinDesk" (CryptoPanic) o dominio del feed ("coindesk.com")
    url: str  # link a la nota (http/https), sin tokens; "" si no hay uno válido
    published_at: datetime | None  # UTC; None si el feed no trae fecha
    currencies: list[str] = field(default_factory=list)  # monedas que nombra ("BTC", ...)
    is_macro: bool = False  # el título tiene alguna palabra de macro_keywords


@dataclass(frozen=True)
class NewsResult:
    symbol: str
    headlines: list[Headline]  # primero los de la moneda, después los macro
    provider: Provider
    meta: FetchMeta
    warnings: list[str] = field(default_factory=list)  # en castellano, para la UI


@dataclass(frozen=True)
class RawNewsItem:
    """Nota tal como vino del proveedor, antes de filtrar."""

    title: str
    source: str
    url: str
    published_at: datetime | None
    summary: str = ""  # texto plano (sin HTML)
    instruments: tuple[str, ...] = ()  # monedas que informa el proveedor (CryptoPanic)


# --- filtrado (puro) ---------------------------------------------------------


def keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str] | None:
    """Regex que matchea cualquiera de las palabras como palabra entera.

    No distingue mayúsculas. Una palabra no matchea si está pegada a una letra,
    número o guion bajo ("sol" no matchea "solution"), pero sí si está pegada a
    un apóstrofo o guion ("Solana's", "SOL-based", "rate-cut"). Las frases
    ("federal reserve") aceptan cualquier espacio entre palabras. Devuelve
    ``None`` si no hay palabras.
    """
    parts = sorted(
        {r"\s+".join(map(re.escape, kw.split())) for kw in keywords if kw.strip()},
        key=len,
        reverse=True,
    )
    if not parts:
        return None
    return re.compile(rf"(?<!\w)(?:{'|'.join(parts)})(?!\w)", re.IGNORECASE)


def normalize_title(title: str) -> str:
    """Título en minúsculas, sin puntuación y con espacios simples (para deduplicar)."""
    return " ".join(re.sub(r"[\W_]+", " ", title.casefold()).split())


def select_headlines(
    items: Sequence[RawNewsItem],
    symbol: str,
    news_keywords: Mapping[str, Sequence[str]],
    macro_keywords: Sequence[str],
    limit: int,
    *,
    now: datetime | None = None,
    max_age_hours: float | None = None,
) -> list[Headline]:
    """Elige los titulares para ``symbol``: primero los de la moneda, después los macro.

    ``news_keywords`` mapea cada moneda a sus palabras. Si ``symbol`` no está, se
    usa el símbolo mismo como palabra. Con ``max_age_hours`` se descartan las
    notas publicadas antes de ``now - max_age_hours`` (las sin fecha se quedan,
    al final).
    """
    if limit <= 0:
        return []
    symbol = symbol.strip().upper()
    keywords = {code.upper(): list(words) for code, words in news_keywords.items()}
    keywords.setdefault(symbol, [symbol.lower()])
    coin_patterns = {
        code: pattern
        for code, words in keywords.items()
        if (pattern := keyword_pattern(words)) is not None
    }
    macro_pattern = keyword_pattern(macro_keywords)

    cutoff = None
    if max_age_hours is not None:
        cutoff = (now or utcnow()) - timedelta(hours=max_age_hours)

    coin: list[Headline] = []
    macro: list[Headline] = []
    seen: set[str] = set()
    for item in sorted(items, key=_newest_first):
        key = normalize_title(item.title)
        if not key or key in seen:
            continue
        seen.add(key)
        if cutoff is not None and item.published_at is not None and item.published_at < cutoff:
            continue
        text = f"{item.title} {item.summary}"
        matched = [code for code, pattern in coin_patterns.items() if pattern.search(text)]
        currencies = list(dict.fromkeys([*(c.upper() for c in item.instruments), *matched]))
        is_macro = bool(macro_pattern and macro_pattern.search(item.title))
        headline = Headline(
            title=item.title,
            source=item.source,
            url=item.url,
            published_at=item.published_at,
            currencies=currencies,
            is_macro=is_macro,
        )
        if symbol in currencies:
            coin.append(headline)
        elif is_macro and not currencies:
            macro.append(headline)
    return (coin + macro)[:limit]


def _newest_first(item: RawNewsItem) -> tuple[bool, float]:
    ts = item.published_at.timestamp() if item.published_at else 0.0
    return (item.published_at is None, -ts)


# --- parsers (puros) ---------------------------------------------------------


def parse_rss(body: bytes | str, feed_url: str) -> list[RawNewsItem]:
    """Parsea un feed RSS/Atom ya descargado (mejor en bytes, tal como llegó).

    Lanza ``ExchangeError`` si el cuerpo no es un feed. Las entradas sin título
    se descartan; las fechas se pasan a UTC. Un ``str`` se pasa a UTF-8 (es lo
    mismo que haría ``feedparser``). Siempre se le da un stream a
    ``feedparser``: con un string que parece URL haría su propio pedido HTTP y
    con uno que parece ruta leería un archivo local.
    """
    data = body.encode("utf-8") if isinstance(body, str) else bytes(body)
    parsed = feedparser.parse(io.BytesIO(data))
    if not parsed.entries and (parsed.bozo or not parsed.version):
        raise ExchangeError(f"{feed_label(feed_url)}: la respuesta no es un feed RSS válido")
    label = feed_label(feed_url)
    items: list[RawNewsItem] = []
    for entry in parsed.entries:
        title = clean_text(entry.get("title"))
        if not title:
            continue
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        published = datetime.fromtimestamp(calendar.timegm(stamp), tz=UTC) if stamp else None
        items.append(
            RawNewsItem(
                title=title,
                source=label,
                url=sanitize_url(entry.get("link")),
                published_at=published,
                summary=clean_text(entry.get("summary")),
            )
        )
    return items


def parse_cryptopanic(payload: Any, secret: str | None = None) -> list[RawNewsItem]:
    """Parsea la respuesta de ``/posts/`` de CryptoPanic (v2, acepta también v1).

    Se quedan solo las notas (``kind`` = ``news``; los videos y podcasts no).
    Las monedas salen de ``instruments`` (v2) o ``currencies`` (v1). El link es
    ``original_url`` y, si viene vacío, ``url``. Lanza ``ExchangeError`` si el
    formato no es el esperado. ``secret`` se tapa en cualquier link.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ExchangeError("cryptopanic: respuesta con formato inesperado")
    items: list[RawNewsItem] = []
    for post in results:
        if not isinstance(post, dict) or post.get("kind", "news") != "news":
            continue
        title = clean_text(post.get("title"))
        if not title:
            continue
        source_info = post.get("source") if isinstance(post.get("source"), dict) else {}
        source = clean_text(source_info.get("title") or source_info.get("domain")) or "CryptoPanic"
        coins = post.get("instruments") or post.get("currencies") or []
        codes = tuple(
            str(c["code"]).upper() for c in coins if isinstance(c, dict) and c.get("code")
        )
        items.append(
            RawNewsItem(
                title=title,
                source=source,
                url=sanitize_url(post.get("original_url") or post.get("url"), secret),
                published_at=_parse_iso_utc(post.get("published_at")),
                summary=clean_text(post.get("description")),
                instruments=codes,
            )
        )
    return items


# Caracteres de control C0 y C1 (ESC, BEL, CSI...). El tab y los saltos de
# línea también entran: igual se colapsan a un espacio.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def clean_text(value: Any) -> str:
    """Saca etiquetas HTML, decodifica entidades, quita caracteres de control y colapsa espacios.

    Los caracteres de control se sacan después de decodificar las entidades
    (``&#27;`` es ESC): así un título no puede mandar secuencias de escape a la
    terminal (cambiar colores, el título de la ventana o el portapapeles).
    """
    if not value:
        return ""
    text = html.unescape(re.sub(r"<[^>]*>", " ", str(value)))
    text = _CONTROL_CHARS.sub(" ", text)
    return " ".join(text.split())


def feed_label(url: str) -> str:
    """Nombre corto de un feed para avisos y para la fuente: su dominio sin ``www.``."""
    host = urlsplit(url).hostname or url
    return host.removeprefix("www.")


def redact(text: str, secret: str | None) -> str:
    """Tapa ``secret`` y cualquier ``auth_token=...`` dentro de ``text``."""
    if secret:
        text = text.replace(secret, REDACTED)
    return re.sub(r"(auth_token=)[^&\s\"'#]+", rf"\g<1>{REDACTED}", text, flags=re.IGNORECASE)


def sanitize_url(url: Any, secret: str | None = None) -> str:
    """Link seguro para devolver: solo http/https y sin ``auth_token`` ni ``secret``."""
    if not url:
        return ""
    try:
        parts = urlsplit(str(url).strip())
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.casefold() != "auth_token"
    ]
    clean = urlunsplit(parts._replace(query=urlencode(query)))
    return redact(clean, secret)


def _parse_iso_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


# --- consultas ---------------------------------------------------------------


async def get_headlines(
    symbol: str,
    client: httpx.AsyncClient,
    cache: TTLCache,
    settings: AppSettings,
    cryptopanic_token: str | None,
    limit: int | None = None,
    *,
    max_age_hours: float | None = None,
    now: datetime | None = None,
) -> NewsResult:
    """Titulares para ``symbol`` (``"BTC"``). Nunca lanza por errores de red.

    ``max_age_hours`` (opcional) descarta notas más viejas que eso respecto de
    ``now`` (por defecto, la hora actual).
    """
    symbol = symbol.strip().upper()
    if limit is None:
        limit = settings.data_sources.news_max_headlines
    if limit < 0:
        raise ValueError("limit no puede ser negativo")
    ds = settings.data_sources

    def select(items: Sequence[RawNewsItem]) -> list[Headline]:
        return select_headlines(
            items,
            symbol,
            ds.news_keywords,
            ds.macro_keywords,
            limit,
            now=now,
            max_age_hours=max_age_hours,
        )

    token = (cryptopanic_token or "").strip() or None
    stale_cp: CacheResult[list[RawNewsItem]] | None = None
    if token:
        try:
            cp = await _fetch_cryptopanic(symbol, client, cache, settings, token)
        except ExchangeError as exc:
            logger.warning("Noticias: CryptoPanic falló (%s)", redact(str(exc), token))
        else:
            if not cp.stale:
                meta = FetchMeta(
                    source=SOURCE_CRYPTOPANIC, fetched_at=cp.fetched_at, from_cache=cp.from_cache
                )
                return NewsResult(symbol, select(cp.value), "cryptopanic", meta)
            logger.warning(
                "Noticias: CryptoPanic falló (%s); se prueba con RSS", redact(cp.error or "", token)
            )
            stale_cp = cp

    rss_items, feed_results, rss_warnings = await _fetch_rss_feeds(client, cache, settings)
    fallback = token is not None
    warnings = ([WARN_CRYPTOPANIC_FALLBACK] if fallback else []) + rss_warnings

    if feed_results:
        meta = FetchMeta(
            source=SOURCE_RSS,
            fetched_at=min(r.fetched_at for r in feed_results),
            from_cache=all(r.from_cache for r in feed_results),
            stale=any(r.stale for r in feed_results),
            fallback=fallback,
            warning=" ".join(warnings) or None,
        )
        return NewsResult(symbol, select(rss_items), "rss", meta, warnings)

    if stale_cp is not None:
        meta = FetchMeta(
            source=SOURCE_CRYPTOPANIC,
            fetched_at=stale_cp.fetched_at,
            from_cache=True,
            stale=True,
            warning=WARN_CRYPTOPANIC_STALE,
        )
        return NewsResult(
            symbol, select(stale_cp.value), "cryptopanic", meta, [WARN_CRYPTOPANIC_STALE]
        )

    # Sin ningún dato: la UI muestra el aviso y el análisis sigue sin noticias.
    meta = FetchMeta(
        source=SOURCE_RSS,
        fetched_at=utcnow(),
        stale=True,
        fallback=fallback,
        warning=" ".join(warnings),
    )
    return NewsResult(symbol, [], "rss", meta, warnings)


async def _fetch_cryptopanic(
    symbol: str,
    client: httpx.AsyncClient,
    cache: TTLCache,
    settings: AppSettings,
    token: str,
) -> CacheResult[list[RawNewsItem]]:
    url = f"{settings.data_sources.cryptopanic_base_url.rstrip('/')}/posts/"
    params = {"auth_token": token, "currencies": symbol, "public": "true", "kind": "news"}

    async def fetch() -> list[RawNewsItem]:
        with _redacting_http_logs(token):
            payload = await get_json(
                client,
                url,
                params,
                source=SOURCE_CRYPTOPANIC,
                retries=settings.exchanges.max_retries,
            )
        return parse_cryptopanic(payload, token)

    return await cache.get_or_fetch(
        f"cryptopanic:posts:{symbol}", ttl_for(settings.cache_ttl_seconds, "news"), fetch
    )


async def _fetch_rss_feeds(
    client: httpx.AsyncClient, cache: TTLCache, settings: AppSettings
) -> tuple[list[RawNewsItem], list[CacheResult[list[RawNewsItem]]], list[str]]:
    """Baja todos los feeds en paralelo. Devuelve notas, resultados OK y avisos."""
    feeds = list(dict.fromkeys(settings.data_sources.rss_feeds))
    ttl = ttl_for(settings.cache_ttl_seconds, "news")

    async def fetch_one(feed_url: str) -> CacheResult[list[RawNewsItem]]:
        async def fetch() -> list[RawNewsItem]:
            body = await get_bytes(
                client,
                feed_url,
                source=feed_label(feed_url),
                retries=settings.exchanges.max_retries,
                headers={"Accept": RSS_ACCEPT},
            )
            return parse_rss(body, feed_url)

        return await cache.get_or_fetch(f"rss:{feed_url}", ttl, fetch)

    outcomes = await asyncio.gather(*(fetch_one(u) for u in feeds), return_exceptions=True)

    items: list[RawNewsItem] = []
    ok: list[CacheResult[list[RawNewsItem]]] = []
    warnings: list[str] = []
    for feed_url, outcome in zip(feeds, outcomes, strict=True):
        label = feed_label(feed_url)
        if isinstance(outcome, ExchangeError):
            logger.warning("Noticias: falló el feed %s (%s)", label, outcome)
            warnings.append(warn_feed_failed(label))
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome.stale:
            logger.warning("Noticias: feed %s viejo (%s)", label, outcome.error)
            warnings.append(warn_feed_stale(label))
        ok.append(outcome)
        items.extend(outcome.value)
    if feeds and not ok:
        warnings.append(WARN_ALL_FEEDS_FAILED)
    return items, ok, warnings


# --- logs sin secretos -------------------------------------------------------


class _RedactSecretFilter(logging.Filter):
    """Reescribe los registros de log que contienen el secreto."""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # un log mal formado no tiene que romper nada
            return True
        cleaned = redact(message, self._secret)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


@contextmanager
def _redacting_http_logs(secret: str) -> Iterator[None]:
    """Mientras dura el bloque, los logs de ``httpx``/``httpcore`` no muestran ``secret``."""
    flt = _RedactSecretFilter(secret)
    loggers = [logging.getLogger(name) for name in _HTTP_LOGGERS]
    for lg in loggers:
        lg.addFilter(flt)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(flt)
