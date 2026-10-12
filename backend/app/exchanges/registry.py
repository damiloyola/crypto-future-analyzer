"""Armado de la fuente de datos de mercado: fallback, caché y precio de ejecución.

- ``FallbackMarketData(primary, fallback)``: cada método prueba primero la
  fuente principal; si lanza ``ExchangeError`` (incluye
  ``ExchangeUnavailableError``: sin red, 451 por región, 5xx, límite de pedidos)
  prueba el fallback y marca el resultado con ``FetchMeta.fallback = True`` y un
  aviso como ``"Binance no respondió (HTTP 451): datos de Bybit"``. Si fallan las
  dos, lanza una excepción con los dos motivos: ``ExchangeUnavailableError`` si
  alguna de las dos no respondió (conviene reintentar o usar la caché) y
  ``ExchangeError`` si las dos rechazaron el pedido (por ejemplo, un símbolo que
  no existe: reintentar no sirve).
- ``CachedMarketData(source, cache, ttls)``: guarda cada respuesta en la
  ``TTLCache`` con el TTL de ``cache_ttl_seconds`` (``price``, ``klines_<tf>``,
  ``funding``, ``open_interest``, ``long_short_ratio``). Las claves son
  ``"md:<dato>:<símbolo>[:<tf>][:<limit>]"``, por ejemplo
  ``"md:klines:BTC:15m:500"``. Si la consulta falla y hay un valor viejo, lo
  devuelve con ``stale = True`` y el aviso ``"Dato desactualizado: ..."``.
  ``get_klines`` con ``end_time`` y ``get_klines_history`` no pasan por la caché.
- ``build_market_data(settings, client, cache)``: compone
  ``Cached(Fallback(principal, respaldo))`` según ``exchanges.data_primary`` y
  ``exchanges.data_fallback`` (``"binance"`` o ``"bybit"``).
- ``get_execution_price(...)``: precio sobre el que se calculan los niveles
  finales. Decisión del usuario: se usa Binance salvo que pegue a mano el precio
  de Nexo. Si lo pega, se compara con Binance y se avisa si difiere más que
  ``exchanges.nexo.manual_price_max_diff_pct`` (posible error de tipeo). Los
  avisos nombran la fuente real del precio de referencia: si Binance falló y el
  dato vino de Bybit, dicen "Bybit (respaldo)".
- ``failure_reason``: motivo corto para la UI (``"HTTP 451"``, ``"sin
  conexión"``, ``"símbolo inválido"``).

``TTLCache`` y ``AppSettings`` se importan solo para los tipos: este módulo no
depende en tiempo de ejecución de la capa ``data``.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx

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
    PriceSource,
    Ticker,
    Timeframe,
)
from app.exchanges.binance import BinanceMarketData, normalize_symbol, utcnow
from app.exchanges.bybit import BybitMarketData
from app.exchanges.nexo import NexoPriceSource

if TYPE_CHECKING:
    from app.config import AppSettings
    from app.data.cache import TTLCache

M = TypeVar("M")

DATA_SOURCE_NAMES = ("binance", "bybit")


def nexo_unavailable_warning(reference_label: str = "Binance") -> str:
    """Aviso de que no hay precio de Nexo y se usa el de referencia."""
    return (
        f"Precio de Nexo no disponible: se usa el de {reference_label}. "
        "Podés pegar el precio de Nexo a mano."
    )


NEXO_UNAVAILABLE_WARNING = nexo_unavailable_warning()


# --- avisos ------------------------------------------------------------------


def join_warnings(*parts: str | None) -> str | None:
    """Une avisos no vacíos en un solo texto (``None`` si no hay ninguno)."""
    texts = [p for p in parts if p]
    return " · ".join(texts) if texts else None


# Nombres para la UI de las fuentes que no se escriben solo con mayúscula inicial.
_SOURCE_LABELS = {"sintetico": "Binance sintético"}


def _label(name: str) -> str:
    return _SOURCE_LABELS.get(name, name.capitalize())


_HTTP_STATUS = re.compile(r"HTTP (\d{3})")
_RET_CODE = re.compile(r"retCode (\d+)")
# Prefijos como "ExchangeUnavailableError: binance: " al principio del mensaje.
_SOURCE_PREFIX = re.compile(r"^(?:[\w.]+: )+")
# Símbolo que el exchange no lista: Binance ``-1121 Invalid symbol.``; Bybit
# ``Not supported symbols`` / ``symbol invalid`` (retCode 10001, que también se
# usa para otros parámetros, por eso se mira el texto y no solo el código).
_INVALID_SYMBOL = re.compile(
    r"código -1121\b|invalid symbol|symbols? (?:is )?invalid|not supported symbols?",
    re.IGNORECASE,
)


def failure_reason(exc: BaseException | str) -> str:
    """Motivo corto de un error de exchange para mostrar en la UI (``"HTTP 451"``)."""
    text = str(exc)
    if _INVALID_SYMBOL.search(text):
        return "símbolo inválido"
    if m := _HTTP_STATUS.search(text):
        return f"HTTP {m.group(1)}"
    if m := _RET_CODE.search(text):
        return f"retCode {m.group(1)}"
    if "sin respuesta" in text:
        return "sin conexión"
    text = _SOURCE_PREFIX.sub("", text) or "error desconocido"
    return text if len(text) <= 80 else text[:79] + "…"


def fallback_warning(primary: str, fallback: str, exc: BaseException) -> str:
    """``"Binance no respondió (HTTP 451): datos de Bybit"``."""
    verb = "no respondió" if isinstance(exc, ExchangeUnavailableError) else "dio un error"
    return f"{_label(primary)} {verb} ({failure_reason(exc)}): datos de {_label(fallback)}"


def _stale_warning(error: str | None) -> str:
    return (
        f"Dato desactualizado: falló la última consulta ({failure_reason(error or '')}); "
        "se muestra el último valor guardado."
    )


# --- fallback ----------------------------------------------------------------


class FallbackMarketData:
    """``MarketDataSource`` que cae a una fuente de respaldo si la principal falla."""

    def __init__(self, primary: MarketDataSource, fallback: MarketDataSource | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    async def _call(self, fetch: Callable[[MarketDataSource], Awaitable[M]]) -> M:
        try:
            return await fetch(self.primary)
        except ExchangeError as primary_exc:
            if self.fallback is None:
                raise
            p, f = _label(self.primary.name), _label(self.fallback.name)
            try:
                result: Any = await fetch(self.fallback)
            except ExchangeError as fallback_exc:
                # Si alguna no respondió, puede andar más tarde; si las dos rechazaron
                # el pedido (símbolo inexistente, parámetro inválido), reintentar no sirve.
                transient = isinstance(primary_exc, ExchangeUnavailableError) or isinstance(
                    fallback_exc, ExchangeUnavailableError
                )
                error_cls = ExchangeUnavailableError if transient else ExchangeError
                raise error_cls(
                    f"{p} y {f} fallaron. {p}: {primary_exc}. {f}: {fallback_exc}"
                ) from fallback_exc
            warning = fallback_warning(self.primary.name, self.fallback.name, primary_exc)
            meta = replace(
                result.meta, fallback=True, warning=join_warnings(warning, result.meta.warning)
            )
            return replace(result, meta=meta)

    async def get_ticker(self, symbol: str) -> Ticker:
        return await self._call(lambda s: s.get_ticker(symbol))

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        return await self._call(lambda s: s.get_klines(symbol, timeframe, limit, end_time))

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        return await self._call(lambda s: s.get_klines_history(symbol, timeframe, start, end))

    async def get_funding(self, symbol: str) -> FundingInfo:
        return await self._call(lambda s: s.get_funding(symbol))

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        return await self._call(lambda s: s.get_funding_history(symbol, limit))

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        return await self._call(lambda s: s.get_open_interest(symbol))

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        return await self._call(lambda s: s.get_open_interest_history(symbol, period, limit))

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        return await self._call(lambda s: s.get_long_short_ratio(symbol, period, limit))

    async def aclose(self) -> None:
        try:
            await self.primary.aclose()
        finally:
            if self.fallback is not None:
                await self.fallback.aclose()


# --- caché -------------------------------------------------------------------


class CachedMarketData:
    """``MarketDataSource`` que guarda las respuestas en una ``TTLCache``."""

    def __init__(
        self,
        source: MarketDataSource,
        cache: TTLCache,
        ttls: Mapping[str, float],
        *,
        default_ttl_s: float = 60.0,
    ) -> None:
        self.source = source
        self.cache = cache
        self.name = source.name
        self._ttls = dict(ttls)
        self._default_ttl_s = default_ttl_s

    def _ttl(self, kind: str) -> float:
        return float(self._ttls.get(kind, self._default_ttl_s))

    async def _cached(self, key: str, kind: str, fetch: Callable[[], Awaitable[M]]) -> M:
        res = await self.cache.get_or_fetch(key, self._ttl(kind), fetch)
        value: Any = res.value
        if not res.from_cache:
            return value
        if res.stale:
            meta = replace(
                value.meta,
                from_cache=True,
                stale=True,
                warning=join_warnings(_stale_warning(res.error), value.meta.warning),
            )
        else:
            meta = replace(value.meta, from_cache=True)
        return replace(value, meta=meta)

    async def get_ticker(self, symbol: str) -> Ticker:
        s = normalize_symbol(symbol)
        return await self._cached(f"md:ticker:{s}", "price", lambda: self.source.get_ticker(s))

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        if end_time is not None:  # velas del pasado: no se cachean
            return await self.source.get_klines(symbol, timeframe, limit, end_time)
        s = normalize_symbol(symbol)
        return await self._cached(
            f"md:klines:{s}:{timeframe}:{limit}",
            f"klines_{timeframe}",
            lambda: self.source.get_klines(s, timeframe, limit),
        )

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        return await self.source.get_klines_history(symbol, timeframe, start, end)

    async def get_funding(self, symbol: str) -> FundingInfo:
        s = normalize_symbol(symbol)
        return await self._cached(f"md:funding:{s}", "funding", lambda: self.source.get_funding(s))

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        s = normalize_symbol(symbol)
        return await self._cached(
            f"md:funding_history:{s}:{limit}",
            "funding",
            lambda: self.source.get_funding_history(s, limit),
        )

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        s = normalize_symbol(symbol)
        return await self._cached(
            f"md:open_interest:{s}", "open_interest", lambda: self.source.get_open_interest(s)
        )

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        s = normalize_symbol(symbol)
        return await self._cached(
            f"md:open_interest_history:{s}:{period}:{limit}",
            "open_interest",
            lambda: self.source.get_open_interest_history(s, period, limit),
        )

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        s = normalize_symbol(symbol)
        return await self._cached(
            f"md:long_short:{s}:{period}:{limit}",
            "long_short_ratio",
            lambda: self.source.get_long_short_ratio(s, period, limit),
        )

    async def aclose(self) -> None:
        await self.source.aclose()


# --- armado ------------------------------------------------------------------


def make_market_data_source(
    name: str,
    settings: AppSettings,
    client: httpx.AsyncClient,
    *,
    clock: Callable[[], datetime] = utcnow,
) -> MarketDataSource:
    """Conector por nombre (``"binance"`` o ``"bybit"``). ``ValueError`` si no existe."""
    key = name.strip().lower()
    cls: type[BinanceMarketData] | type[BybitMarketData]
    if key == "binance":
        cls = BinanceMarketData
    elif key == "bybit":
        cls = BybitMarketData
    else:
        raise ValueError(
            f"Fuente de datos desconocida: {name!r} (opciones: {', '.join(DATA_SOURCE_NAMES)})"
        )
    return cls(
        settings.exchanges,
        client,
        quote=settings.symbols.quote,
        clock=clock,
        default_funding_interval_hours=settings.fees.funding_interval_hours,
    )


def build_market_data(
    settings: AppSettings,
    client: httpx.AsyncClient,
    cache: TTLCache,
    *,
    clock: Callable[[], datetime] = utcnow,
) -> CachedMarketData:
    """``Cached(Fallback(principal, respaldo))`` según ``config.yaml > exchanges``."""
    ex = settings.exchanges
    primary = make_market_data_source(ex.data_primary, settings, client, clock=clock)
    fallback = (
        make_market_data_source(ex.data_fallback, settings, client, clock=clock)
        if ex.data_fallback
        else None
    )
    if fallback is not None and fallback.name == primary.name:
        raise ValueError("exchanges.data_fallback tiene que ser distinto de data_primary")
    return CachedMarketData(
        FallbackMarketData(primary, fallback), cache, settings.cache_ttl_seconds
    )


def build_nexo_price_source(settings: AppSettings) -> NexoPriceSource:
    """``NexoPriceSource`` con ``exchanges.nexo`` (en la fase 1 no consulta la red)."""
    return NexoPriceSource(settings.exchanges.nexo, quote=settings.symbols.quote)


# --- precio de ejecución -----------------------------------------------------


ExecutionSource = Literal["manual_nexo", "nexo", "binance"]


@dataclass(frozen=True)
class ExecutionPrice:
    """Precio sobre el que se calculan los niveles finales del plan.

    - ``source``: ``"manual_nexo"`` (pegado a mano desde la app de Nexo),
      ``"nexo"`` (conector de Nexo, hoy nunca disponible) o ``"binance"`` (precio
      de referencia de los datos de mercado; si Binance falló puede venir de
      Bybit: lo dicen ``meta.source`` y ``warnings``).
    - ``binance_price``: precio de referencia de los datos de mercado, si se pudo
      consultar.
    - ``diff_pct``: diferencia contra ``binance_price`` en % (``None`` si
      ``source`` es ``"binance"`` o no hubo con qué comparar).
    """

    price: float
    source: ExecutionSource
    binance_price: float | None
    diff_pct: float | None
    warnings: list[str]
    meta: FetchMeta


def _fmt_pct(value: float) -> str:
    """``9.0909`` → ``"9,09"`` (coma decimal para la UI)."""
    return f"{value:.2f}".replace(".", ",")


async def _try_ticker(
    source: PriceSource, symbol: str
) -> tuple[Ticker | None, ExchangeError | None]:
    try:
        return await source.get_ticker(symbol), None
    except ExchangeError as exc:
        return None, exc


def _diff_pct(price: float, reference: float) -> float:
    return (price - reference) / reference * 100


def reference_label(meta: FetchMeta | None) -> str:
    """Nombre de la fuente del precio de referencia para los avisos.

    ``"Binance"``, ``"Bybit (respaldo)"`` si vino del fallback, o ``"referencia"``
    si no se sabe.
    """
    if meta is None or not meta.source:
        return "referencia"
    label = _label(meta.source)
    return f"{label} (respaldo)" if meta.fallback else label


async def get_execution_price(
    symbol: str,
    market_data: PriceSource,
    nexo: PriceSource,
    settings: AppSettings,
    manual_price: float | None = None,
    *,
    clock: Callable[[], datetime] = utcnow,
) -> ExecutionPrice:
    """Precio de ejecución según la decisión del usuario.

    1. ``manual_price`` (pegado desde Nexo): se valida (> 0), se usa y se compara
       con Binance; si difiere más que ``manual_price_max_diff_pct`` se avisa.
    2. Si ``exchanges.nexo.enabled`` y el conector de Nexo responde: ese precio.
    3. Si no, Binance con el aviso de que se puede pegar el precio de Nexo.

    Lanza ``ValueError`` si el precio manual no es válido y
    ``ExchangeError``/``ExchangeUnavailableError`` si no hay ningún precio.
    """
    nexo_cfg = settings.exchanges.nexo

    if manual_price is not None:
        price = float(manual_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("El precio de Nexo tiene que ser un número mayor que 0")
        ticker, exc = await _try_ticker(market_data, symbol)
        warnings: list[str] = []
        if ticker is None:
            warnings.append(
                f"No se pudo consultar el precio de referencia ({failure_reason(exc)}) para "
                "comparar: revisá que el precio de Nexo esté bien escrito."
            )
            return ExecutionPrice(
                price, "manual_nexo", None, None, warnings, FetchMeta("manual", clock())
            )
        diff = _diff_pct(price, ticker.price)
        if abs(diff) > nexo_cfg.manual_price_max_diff_pct:
            warnings.append(
                f"El precio de Nexo que cargaste difiere {_fmt_pct(abs(diff))} % del de "
                f"{reference_label(ticker.meta)}: revisá que esté bien escrito"
            )
        if ticker.meta.warning:
            warnings.append(ticker.meta.warning)
        return ExecutionPrice(
            price, "manual_nexo", ticker.price, diff, warnings, FetchMeta("manual", clock())
        )

    reference: Ticker | None = None
    reference_exc: ExchangeError | None = None
    if nexo_cfg.enabled:
        (nexo_ticker, _), (reference, reference_exc) = await asyncio.gather(
            _try_ticker(nexo, symbol), _try_ticker(market_data, symbol)
        )
        if nexo_ticker is not None:
            warnings = [nexo_ticker.meta.warning] if nexo_ticker.meta.warning else []
            if reference is None:
                warnings.append("No se pudo consultar el precio de referencia para comparar.")
            return ExecutionPrice(
                nexo_ticker.price,
                "nexo",
                reference.price if reference else None,
                _diff_pct(nexo_ticker.price, reference.price) if reference else None,
                warnings,
                nexo_ticker.meta,
            )
        if reference_exc is not None:
            raise reference_exc
    if reference is None:
        reference = await market_data.get_ticker(symbol)
    warnings = [nexo_unavailable_warning(reference_label(reference.meta))]
    if reference.meta.warning:
        warnings.append(reference.meta.warning)
    return ExecutionPrice(
        reference.price, "binance", reference.price, None, warnings, reference.meta
    )
