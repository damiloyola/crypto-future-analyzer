"""Conector de datos públicos de Binance USDⓈ-M (sin API key).

``BinanceMarketData`` implementa ``MarketDataSource`` con los endpoints
verificados en ``docs/data-sources.md``:

- Velas: ``GET /fapi/v1/klines`` (perpetuos, hasta 1500 por pedido) o, con
  ``use_perpetual_klines: false``, ``GET /api/v3/klines`` (spot, hasta 1000).
  ``closed`` se marca con ``closeTime < ahora``: solo la vela que se está
  formando queda en ``False``. ``get_klines`` pagina hacia atrás si se piden más
  velas que el máximo del endpoint; ``get_klines_history`` recorre el rango hacia
  adelante en ventanas de ``limit`` velas (``startTime``/``endTime``), elimina
  duplicados y corta en ``end`` o en "ahora".
- Ticker 24 h: ``GET /fapi/v1/ticker/24hr`` (``priceChangePercent`` ya viene en %).
- Funding actual: ``GET /fapi/v1/premiumIndex`` (``lastFundingRate`` como
  fracción, mark e index price, próximo cobro) más ``GET /fapi/v1/fundingInfo``
  para el intervalo (solo lista los símbolos con ajustes; si no figura, 8 h). Si
  ``fundingInfo`` falla se asume el intervalo por defecto con un aviso.
- Historial de funding: ``GET /fapi/v1/fundingRate`` (descarta ``rateType``
  ``"Special"``, que no es funding de cripto).
- Open interest: ``GET /fapi/v1/openInterest`` (en moneda base; sin valor en USD)
  e historial ``GET /futures/data/openInterestHist`` (solo ~30 días).
- Ratio long/short de cuentas: ``GET /futures/data/globalLongShortAccountRatio``.

Los errores HTTP los traduce ``app.http_client`` (451/403/429/5xx →
``ExchangeUnavailableError``); los cuerpos de error de Binance
(``{"code": -1121, "msg": "Invalid symbol."}``) se convierten en
``ExchangeError`` con el ``msg``.

El cliente HTTP se recibe por parámetro y lo cierra quien lo creó; si no se pasa
ninguno, el conector crea el suyo y lo cierra en ``aclose``.

Este módulo también tiene los helpers chicos que comparte con ``bybit``
(conversión de tiempos, armado del DataFrame de velas, paginación).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import pandas as pd

from app.exchanges.base import (
    TIMEFRAME_MINUTES,
    Candles,
    ExchangeError,
    ExchangeUnavailableError,
    FetchMeta,
    FundingHistory,
    FundingInfo,
    FundingRatePoint,
    LongShortHistory,
    LongShortPoint,
    OpenInterest,
    OpenInterestHistory,
    OpenInterestPoint,
    Ticker,
    Timeframe,
    empty_candles_frame,
    validate_candles_frame,
)
from app.http_client import get_json, make_client

if TYPE_CHECKING:
    from app.config import ExchangesConfig

T = TypeVar("T")

# Fila de vela ya normalizada: (open_time ms, open, high, low, close, volume, closed).
CandleRow = tuple[int, float, float, float, float, float, bool]

FUTURES_KLINES_MAX_LIMIT = 1500
SPOT_KLINES_MAX_LIMIT = 1000
FUNDING_HISTORY_MAX_LIMIT = 1000
FUTURES_DATA_MAX_LIMIT = 500  # openInterestHist y globalLongShortAccountRatio


# --- helpers compartidos (también los usa bybit) -----------------------------


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_symbol(symbol: str) -> str:
    """``" btc "`` → ``"BTC"``. Lanza ``ValueError`` si no es alfanumérico."""
    s = symbol.strip().upper()
    if not s or not s.isalnum():
        raise ValueError(f"Símbolo inválido: {symbol!r}")
    return s


def pair_symbol(symbol: str, quote: str) -> str:
    """Activo base de la app → par del exchange: ``"BTC"`` → ``"BTCUSDT"``."""
    return normalize_symbol(symbol) + normalize_symbol(quote)


def timeframe_ms(timeframe: str) -> int:
    """Duración de una vela en ms. ``ValueError`` si el timeframe no existe en la app."""
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(
            f"Timeframe no soportado: {timeframe!r} (opciones: {', '.join(TIMEFRAME_MINUTES)})"
        )
    return minutes * 60_000


def ms_to_utc(ms: int | str) -> datetime:
    """Milisegundos Unix (entero o string, como manda Bybit) → ``datetime`` UTC."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)


def utc_to_ms(dt: datetime) -> int:
    """``datetime`` con zona horaria → milisegundos Unix. Rechaza fechas sin zona."""
    if dt.tzinfo is None:
        raise ValueError("La fecha tiene que tener zona horaria (UTC)")
    return math.floor(dt.timestamp() * 1000)


def opt_float(value: Any) -> float | None:
    """``float`` o ``None`` si el campo falta o viene vacío (Bybit manda ``""``)."""
    if value is None or value == "":
        return None
    return float(value)


def candles_frame(rows: Iterable[CandleRow]) -> pd.DataFrame:
    """Arma el DataFrame de velas de ``base`` (índice UTC, ordenado, sin duplicados)."""
    rows = list(rows)
    if not rows:
        return empty_candles_frame()
    open_ms, opens, highs, lows, closes, volumes, closed = zip(*rows, strict=True)
    index = pd.DatetimeIndex(pd.to_datetime(list(open_ms), unit="ms", utc=True), name="open_time")
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "closed": closed,
        },
        index=index,
    )
    return validate_candles_frame(df)


def history_windows(
    start_ms: int, end_ms: int, step_ms: int, per_page: int
) -> list[tuple[int, int]]:
    """Ventanas ``[desde, hasta]`` (ms, inclusivas) de a lo sumo ``per_page`` velas.

    Pedir cada ventana con ``startTime`` y ``endTime`` devuelve la ventana
    completa sin importar si el exchange arranca a contar desde el principio o
    desde el final del rango.
    """
    windows: list[tuple[int, int]] = []
    cursor = start_ms
    span = step_ms * per_page
    while cursor <= end_ms:
        windows.append((cursor, min(cursor + span - 1, end_ms)))
        cursor += span
    return windows


async def page_backward(
    fetch: Callable[[int, int | None], Awaitable[Sequence[T]]],
    limit: int,
    per_page: int,
    time_of: Callable[[T], int],
    end_ms: int | None = None,
) -> list[T]:
    """Junta los últimos ``limit`` elementos pidiendo páginas hacia atrás.

    ``fetch(cantidad, hasta_ms)`` pide una página (``hasta_ms`` ``None`` = lo más
    reciente). Devuelve los elementos en orden cronológico, sin duplicados.
    """
    if limit < 1:
        raise ValueError("limit tiene que ser >= 1")
    collected: dict[int, T] = {}
    cursor = end_ms
    while len(collected) < limit:
        want = min(limit - len(collected), per_page)
        page = await fetch(want, cursor)
        fresh = [
            item
            for item in page
            if time_of(item) not in collected and (cursor is None or time_of(item) <= cursor)
        ]
        for item in fresh:
            collected[time_of(item)] = item
        if len(page) < want or not fresh:
            break
        cursor = min(time_of(item) for item in page) - 1
    return [collected[t] for t in sorted(collected)][-limit:]


@contextmanager
def parse_guard(source: str, what: str) -> Iterator[None]:
    """Convierte errores de parseo (campo faltante, tipo raro) en ``ExchangeError``."""
    try:
        yield
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ExchangeError(
            f"{source}: respuesta con formato inesperado ({what}): {exc!r}"
        ) from exc


def expect_list(data: Any, source: str, what: str) -> list[Any]:
    if not isinstance(data, list):
        raise ExchangeError(f"{source}: se esperaba una lista en {what}")
    return data


def expect_dict(data: Any, source: str, what: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ExchangeError(f"{source}: se esperaba un objeto en {what}")
    return data


def check_history_pages(n_pages: int, max_pages: int) -> None:
    """Evita que un rango gigante dispare cientos de pedidos (y un baneo por IP)."""
    if n_pages > max_pages:
        raise ValueError(
            f"El rango pedido necesita {n_pages} pedidos (máximo {max_pages}): "
            "achicá el rango o usá un timeframe más largo"
        )


# --- conector ----------------------------------------------------------------


_HTTP_ERROR_BODY = re.compile(r"HTTP (\d{3}) en \S+: (\{.*\})", re.DOTALL)


def _binance_error_text(text: str) -> str:
    """Saca el ``msg`` del cuerpo de error de Binance que viene dentro del mensaje HTTP."""
    match = _HTTP_ERROR_BODY.search(text)
    if not match:
        return text
    try:
        body = json.loads(match.group(2))
    except ValueError:
        return text
    if not isinstance(body, dict) or "msg" not in body:
        return text
    return f"binance: {body['msg']} (HTTP {match.group(1)}, código {body.get('code')})"


class BinanceMarketData:
    """Fuente principal de datos de mercado (endpoints públicos de Binance)."""

    name = "binance"

    def __init__(
        self,
        config: ExchangesConfig,
        client: httpx.AsyncClient | None = None,
        *,
        quote: str = "USDT",
        retries: int | None = None,
        backoff_s: float = 0.5,
        clock: Callable[[], datetime] = utcnow,
        default_funding_interval_hours: float = 8.0,
        max_history_requests: int = 200,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client if client is not None else make_client(config.request_timeout_s)
        self._quote = normalize_symbol(quote)
        self._retries = config.max_retries if retries is None else retries
        self._backoff_s = backoff_s
        self._clock = clock
        self._default_interval_h = default_funding_interval_hours
        self._max_history_requests = max_history_requests

    # --- infraestructura -----------------------------------------------------

    def _pair(self, symbol: str) -> str:
        return pair_symbol(symbol, self._quote)

    def _meta(self, warning: str | None = None) -> FetchMeta:
        return FetchMeta(source=self.name, fetched_at=self._clock(), warning=warning)

    def _futures_url(self, path: str) -> str:
        return self._config.binance.futures_base_url.rstrip("/") + path

    def _klines_endpoint(self) -> tuple[str, int]:
        if self._config.binance.use_perpetual_klines:
            return self._futures_url("/fapi/v1/klines"), FUTURES_KLINES_MAX_LIMIT
        url = self._config.binance.spot_base_url.rstrip("/") + "/api/v3/klines"
        return url, SPOT_KLINES_MAX_LIMIT

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        try:
            data = await get_json(
                self._client,
                url,
                params,
                source=self.name,
                retries=self._retries,
                backoff_s=self._backoff_s,
            )
        except ExchangeUnavailableError:
            raise
        except ExchangeError as exc:
            raise ExchangeError(_binance_error_text(str(exc))) from exc
        # Algunos errores llegan con HTTP 200 y solo {"code", "msg"}.
        if isinstance(data, dict) and set(data) == {"code", "msg"}:
            raise ExchangeError(f"binance: {data['msg']} (código {data['code']})")
        return data

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- velas ---------------------------------------------------------------

    def _parse_klines(self, data: Any) -> list[CandleRow]:
        now_ms = utc_to_ms(self._clock())
        rows = expect_list(data, self.name, "klines")
        with parse_guard(self.name, "klines"):
            return [
                (
                    int(r[0]),
                    float(r[1]),
                    float(r[2]),
                    float(r[3]),
                    float(r[4]),
                    float(r[5]),
                    int(r[6]) < now_ms,  # closeTime ya pasó: vela cerrada
                )
                for r in rows
            ]

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        timeframe_ms(timeframe)  # valida
        if limit < 1:
            raise ValueError("limit tiene que ser >= 1")
        url, max_limit = self._klines_endpoint()
        pair = self._pair(symbol)

        async def fetch(n: int, before_ms: int | None) -> list[CandleRow]:
            params: dict[str, Any] = {"symbol": pair, "interval": timeframe, "limit": n}
            if before_ms is not None:
                params["endTime"] = before_ms
            return self._parse_klines(await self._get(url, params))

        end_ms = utc_to_ms(end_time) if end_time is not None else None
        rows = await page_backward(fetch, limit, max_limit, lambda r: r[0], end_ms)
        return Candles(normalize_symbol(symbol), timeframe, candles_frame(rows), self._meta())

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        step = timeframe_ms(timeframe)
        url, max_limit = self._klines_endpoint()
        pair = self._pair(symbol)
        now_ms = utc_to_ms(self._clock())
        start_ms = utc_to_ms(start)
        end_ms = now_ms if end is None else min(utc_to_ms(end), now_ms)
        windows = history_windows(start_ms, end_ms, step, max_limit)
        check_history_pages(len(windows), self._max_history_requests)

        rows: list[CandleRow] = []
        for win_start, win_end in windows:
            params = {
                "symbol": pair,
                "interval": timeframe,
                "startTime": win_start,
                "endTime": win_end,
                "limit": max_limit,
            }
            page = self._parse_klines(await self._get(url, params))
            rows.extend(r for r in page if start_ms <= r[0] <= end_ms)
        return Candles(normalize_symbol(symbol), timeframe, candles_frame(rows), self._meta())

    # --- precio --------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        data = expect_dict(
            await self._get(
                self._futures_url("/fapi/v1/ticker/24hr"), {"symbol": self._pair(symbol)}
            ),
            self.name,
            "ticker/24hr",
        )
        with parse_guard(self.name, "ticker/24hr"):
            price = float(data["lastPrice"])
            ticker = Ticker(
                symbol=normalize_symbol(symbol),
                price=price,
                meta=self._meta(),
                change_24h_pct=opt_float(data.get("priceChangePercent")),  # ya en %
                high_24h=opt_float(data.get("highPrice")),
                low_24h=opt_float(data.get("lowPrice")),
                quote_volume_24h=opt_float(data.get("quoteVolume")),
            )
        if not price > 0:
            raise ExchangeError(f"binance: precio no positivo para {symbol}: {price}")
        return ticker

    # --- funding -------------------------------------------------------------

    async def _funding_intervals(self) -> dict[str, float] | None:
        """Intervalo de funding de los símbolos con ajustes, o ``None`` si falló."""
        try:
            data = expect_list(
                await self._get(self._futures_url("/fapi/v1/fundingInfo")),
                self.name,
                "fundingInfo",
            )
            with parse_guard(self.name, "fundingInfo"):
                return {
                    str(item["symbol"]): float(item["fundingIntervalHours"])
                    for item in data
                    if item.get("fundingIntervalHours")
                }
        except ExchangeError:
            return None

    async def get_funding(self, symbol: str) -> FundingInfo:
        pair = self._pair(symbol)
        premium_raw, intervals = await asyncio.gather(
            self._get(self._futures_url("/fapi/v1/premiumIndex"), {"symbol": pair}),
            self._funding_intervals(),
        )
        premium = expect_dict(premium_raw, self.name, "premiumIndex")
        warning = None
        if intervals is None:
            interval = self._default_interval_h
            warning = (
                "No se pudo confirmar cada cuántas horas se cobra el funding: "
                f"se asume {interval:g} h."
            )
        else:
            interval = intervals.get(pair, self._default_interval_h)
        with parse_guard(self.name, "premiumIndex"):
            next_ms = int(premium.get("nextFundingTime") or 0)
            return FundingInfo(
                symbol=normalize_symbol(symbol),
                rate=float(premium["lastFundingRate"]),
                mark_price=float(premium["markPrice"]),
                meta=self._meta(warning),
                index_price=opt_float(premium.get("indexPrice")),
                next_funding_time=ms_to_utc(next_ms) if next_ms > 0 else None,
                interval_hours=interval,
            )

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        url = self._futures_url("/fapi/v1/fundingRate")
        pair = self._pair(symbol)

        async def fetch(n: int, before_ms: int | None) -> list[tuple[int, float, str]]:
            params: dict[str, Any] = {"symbol": pair, "limit": n}
            if before_ms is not None:
                params["endTime"] = before_ms
            data = expect_list(await self._get(url, params), self.name, "fundingRate")
            with parse_guard(self.name, "fundingRate"):
                return [
                    (
                        int(x["fundingTime"]),
                        float(x["fundingRate"]),
                        str(x.get("rateType") or "Regular"),
                    )
                    for x in data
                ]

        items = await page_backward(fetch, limit, FUNDING_HISTORY_MAX_LIMIT, lambda x: x[0])
        points = [
            FundingRatePoint(time=ms_to_utc(t), rate=rate)
            for t, rate, kind in items
            if kind != "Special"  # funding extra por dividendos (acciones), no aplica
        ]
        return FundingHistory(normalize_symbol(symbol), points, self._meta())

    # --- open interest -------------------------------------------------------

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        data = expect_dict(
            await self._get(
                self._futures_url("/fapi/v1/openInterest"), {"symbol": self._pair(symbol)}
            ),
            self.name,
            "openInterest",
        )
        with parse_guard(self.name, "openInterest"):
            time_ms = data.get("time")
            return OpenInterest(
                symbol=normalize_symbol(symbol),
                contracts=float(data["openInterest"]),
                meta=self._meta(),
                value_usd=None,  # Binance no lo da; haría falta el mark price
                time=ms_to_utc(time_ms) if time_ms else None,
            )

    async def _futures_data(
        self, path: str, symbol: str, period: Timeframe, limit: int
    ) -> list[dict[str, Any]]:
        """Endpoints de ``/futures/data`` (ascendentes, máx. 500 por pedido)."""
        timeframe_ms(period)  # valida
        url = self._futures_url(path)
        pair = self._pair(symbol)

        async def fetch(n: int, before_ms: int | None) -> list[dict[str, Any]]:
            params: dict[str, Any] = {"symbol": pair, "period": period, "limit": n}
            if before_ms is not None:
                params["endTime"] = before_ms
            data = expect_list(await self._get(url, params), self.name, path)
            with parse_guard(self.name, path):
                return [dict(x, _ts=int(x["timestamp"])) for x in data]

        return await page_backward(fetch, limit, FUTURES_DATA_MAX_LIMIT, lambda x: x["_ts"])

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        path = "/futures/data/openInterestHist"
        items = await self._futures_data(path, symbol, period, limit)
        with parse_guard(self.name, path):
            points = [
                OpenInterestPoint(
                    time=ms_to_utc(x["_ts"]),
                    contracts=float(x["sumOpenInterest"]),
                    value_usd=opt_float(x.get("sumOpenInterestValue")),
                )
                for x in items
            ]
        return OpenInterestHistory(normalize_symbol(symbol), period, points, self._meta())

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        path = "/futures/data/globalLongShortAccountRatio"
        items = await self._futures_data(path, symbol, period, limit)
        with parse_guard(self.name, path):
            points = [
                LongShortPoint(
                    time=ms_to_utc(x["_ts"]),
                    ratio=float(x["longShortRatio"]),
                    long_pct=opt_float(x.get("longAccount")),
                    short_pct=opt_float(x.get("shortAccount")),
                )
                for x in items
            ]
        return LongShortHistory(normalize_symbol(symbol), period, points, self._meta())
