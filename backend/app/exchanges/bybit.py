"""Conector de datos públicos de Bybit v5 (fallback de Binance).

``BybitMarketData`` implementa ``MarketDataSource`` con los mismos modelos que
Binance, así el resto del sistema no sabe de dónde vinieron los datos (salvo por
``FetchMeta.source``). Endpoints (``category`` sale de ``exchanges.bybit``):

- Velas: ``GET /v5/market/kline`` (hasta 1000 por pedido; intervalos ``5``,
  ``15``, ``60``, ``240``, ``D``). Bybit las manda de la más nueva a la más vieja
  y sin ``closeTime``: se ordenan y ``closed`` es ``open_time + timeframe <= ahora``.
- Ticker y funding actual: ``GET /v5/market/tickers``. ``price24hPcnt`` viene
  como fracción y se pasa a % para ``Ticker.change_24h_pct``. El intervalo de
  funding sale de ``fundingIntervalHour`` (si falta, el valor por defecto, 8 h).
- Historial de funding: ``GET /v5/market/funding/history`` (máx. 200 por pedido).
- Open interest actual: ``tickers`` (``openInterest`` y ``openInterestValue``).
  Historial: ``GET /v5/market/open-interest`` (máx. 200 por pedido, sin valor en
  USD). Desde junio de 2026 Bybit documenta ``openInterest`` como la suma de los
  dos lados: se usa siempre ese campo para que la serie sea coherente, pero su
  valor absoluto no se compara con el de Binance.
- Ratio long/short: ``GET /v5/market/account-ratio``; ``ratio = buyRatio /
  sellRatio`` y ``long_pct = buyRatio``.

Bybit responde los errores con HTTP 200 y ``retCode != 0``: los de límite de
pedidos, servidor o región → ``ExchangeUnavailableError``; el resto (símbolo
inválido, parámetros) → ``ExchangeError``. Los listados se paginan hacia atrás
con ``end``/``endTime`` cuando se piden más elementos que el máximo por pedido.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from app.exchanges.base import (
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
)
from app.exchanges.binance import (
    CandleRow,
    candles_frame,
    check_history_pages,
    expect_dict,
    expect_list,
    history_windows,
    ms_to_utc,
    normalize_symbol,
    opt_float,
    page_backward,
    pair_symbol,
    parse_guard,
    timeframe_ms,
    utc_to_ms,
    utcnow,
)
from app.http_client import get_json, make_client

if TYPE_CHECKING:
    from app.config import ExchangesConfig

KLINE_INTERVALS: dict[str, str] = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
# open-interest.intervalTime y account-ratio.period usan los mismos códigos.
PERIOD_CODES: dict[str, str] = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}

KLINE_MAX_LIMIT = 1000
FUNDING_HISTORY_MAX_LIMIT = 200
OPEN_INTEREST_MAX_LIMIT = 200
ACCOUNT_RATIO_MAX_LIMIT = 500

# retCode que indican "probá más tarde u otra fuente": servidor (10000, 10016),
# límite de pedidos (10006, 10018) y región restringida (10009).
UNAVAILABLE_RET_CODES = frozenset({10000, 10006, 10009, 10016, 10018})


def _code(mapping: dict[str, str], timeframe: str, what: str) -> str:
    timeframe_ms(timeframe)  # mismo mensaje de error que el resto
    code = mapping.get(timeframe)
    if code is None:  # pragma: no cover - todos los timeframes de la app tienen código
        raise ValueError(f"Bybit no tiene {what} para {timeframe!r}")
    return code


class BybitMarketData:
    """Fuente de respaldo de datos de mercado (endpoints públicos de Bybit v5)."""

    name = "bybit"

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

    @property
    def _category(self) -> str:
        return self._config.bybit.category

    def _pair(self, symbol: str) -> str:
        return pair_symbol(symbol, self._quote)

    def _meta(self) -> FetchMeta:
        return FetchMeta(source=self.name, fetched_at=self._clock())

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET a la API v5; devuelve el sobre completo si ``retCode == 0``."""
        url = self._config.bybit.base_url.rstrip("/") + path
        data = expect_dict(
            await get_json(
                self._client,
                url,
                {"category": self._category, **params},
                source=self.name,
                retries=self._retries,
                backoff_s=self._backoff_s,
            ),
            self.name,
            path,
        )
        if "retCode" not in data:
            raise ExchangeError(f"bybit: respuesta sin retCode en {path}")
        try:
            code = int(data["retCode"])
        except (TypeError, ValueError) as exc:
            raise ExchangeError(f"bybit: retCode inválido en {path}") from exc
        if code != 0:
            msg = f"bybit: {data.get('retMsg') or 'error'} (retCode {code}) en {path}"
            if code in UNAVAILABLE_RET_CODES:
                raise ExchangeUnavailableError(msg)
            raise ExchangeError(msg)
        expect_dict(data.get("result"), self.name, path)
        return data

    async def _list(self, path: str, params: dict[str, Any]) -> list[Any]:
        data = await self._get(path, params)
        return expect_list(data["result"].get("list"), self.name, path)

    async def _ticker_item(self, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Primer elemento de ``tickers`` para el símbolo, junto con el sobre."""
        data = await self._get("/v5/market/tickers", {"symbol": self._pair(symbol)})
        items = expect_list(data["result"].get("list"), self.name, "tickers")
        if not items:
            raise ExchangeError(f"bybit: sin datos de ticker para {symbol}")
        return expect_dict(items[0], self.name, "tickers"), data

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- velas ---------------------------------------------------------------

    def _parse_klines(self, rows: list[Any], step_ms: int) -> list[CandleRow]:
        now_ms = utc_to_ms(self._clock())
        with parse_guard(self.name, "kline"):
            parsed = [
                (
                    int(r[0]),
                    float(r[1]),
                    float(r[2]),
                    float(r[3]),
                    float(r[4]),
                    float(r[5]),
                    int(r[0]) + step_ms <= now_ms,
                )
                for r in rows
            ]
        return sorted(parsed, key=lambda r: r[0])  # Bybit: la más nueva primero

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        interval = _code(KLINE_INTERVALS, timeframe, "velas")
        step = timeframe_ms(timeframe)
        if limit < 1:
            raise ValueError("limit tiene que ser >= 1")
        pair = self._pair(symbol)

        async def fetch(n: int, before_ms: int | None) -> list[CandleRow]:
            params: dict[str, Any] = {"symbol": pair, "interval": interval, "limit": n}
            if before_ms is not None:
                params["end"] = before_ms
            return self._parse_klines(await self._list("/v5/market/kline", params), step)

        end_ms = utc_to_ms(end_time) if end_time is not None else None
        rows = await page_backward(fetch, limit, KLINE_MAX_LIMIT, lambda r: r[0], end_ms)
        return Candles(normalize_symbol(symbol), timeframe, candles_frame(rows), self._meta())

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        interval = _code(KLINE_INTERVALS, timeframe, "velas")
        step = timeframe_ms(timeframe)
        pair = self._pair(symbol)
        now_ms = utc_to_ms(self._clock())
        start_ms = utc_to_ms(start)
        end_ms = now_ms if end is None else min(utc_to_ms(end), now_ms)
        windows = history_windows(start_ms, end_ms, step, KLINE_MAX_LIMIT)
        check_history_pages(len(windows), self._max_history_requests)

        rows: list[CandleRow] = []
        for win_start, win_end in windows:
            params = {
                "symbol": pair,
                "interval": interval,
                "start": win_start,
                "end": win_end,
                "limit": KLINE_MAX_LIMIT,
            }
            page = self._parse_klines(await self._list("/v5/market/kline", params), step)
            rows.extend(r for r in page if start_ms <= r[0] <= end_ms)
        return Candles(normalize_symbol(symbol), timeframe, candles_frame(rows), self._meta())

    # --- precio y funding ----------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        item, _ = await self._ticker_item(symbol)
        with parse_guard(self.name, "tickers"):
            price = float(item["lastPrice"])
            change = opt_float(item.get("price24hPcnt"))
            ticker = Ticker(
                symbol=normalize_symbol(symbol),
                price=price,
                meta=self._meta(),
                change_24h_pct=None if change is None else change * 100,  # fracción → %
                high_24h=opt_float(item.get("highPrice24h")),
                low_24h=opt_float(item.get("lowPrice24h")),
                quote_volume_24h=opt_float(item.get("turnover24h")),
            )
        if not price > 0:
            raise ExchangeError(f"bybit: precio no positivo para {symbol}: {price}")
        return ticker

    async def get_funding(self, symbol: str) -> FundingInfo:
        item, _ = await self._ticker_item(symbol)
        with parse_guard(self.name, "tickers"):
            interval = opt_float(item.get("fundingIntervalHour"))
            next_ms = int(item.get("nextFundingTime") or 0)
            return FundingInfo(
                symbol=normalize_symbol(symbol),
                rate=float(item["fundingRate"]),
                mark_price=float(item["markPrice"]),
                meta=self._meta(),
                index_price=opt_float(item.get("indexPrice")),
                next_funding_time=ms_to_utc(next_ms) if next_ms > 0 else None,
                interval_hours=interval if interval and interval > 0 else self._default_interval_h,
            )

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        pair = self._pair(symbol)
        path = "/v5/market/funding/history"

        async def fetch(n: int, before_ms: int | None) -> list[tuple[int, float]]:
            params: dict[str, Any] = {"symbol": pair, "limit": n}
            if before_ms is not None:
                params["endTime"] = before_ms  # solo startTime da error; solo endTime anda
            items = await self._list(path, params)
            with parse_guard(self.name, path):
                return [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in items]

        items = await page_backward(fetch, limit, FUNDING_HISTORY_MAX_LIMIT, lambda x: x[0])
        points = [FundingRatePoint(time=ms_to_utc(t), rate=rate) for t, rate in items]
        return FundingHistory(normalize_symbol(symbol), points, self._meta())

    # --- open interest y long/short ------------------------------------------

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        item, envelope = await self._ticker_item(symbol)
        with parse_guard(self.name, "tickers"):
            server_ms = int(envelope.get("time") or 0)
            return OpenInterest(
                symbol=normalize_symbol(symbol),
                contracts=float(item["openInterest"]),
                meta=self._meta(),
                value_usd=opt_float(item.get("openInterestValue")),
                time=ms_to_utc(server_ms) if server_ms > 0 else None,
            )

    async def _period_list(
        self,
        path: str,
        period_param: str,
        symbol: str,
        period: Timeframe,
        limit: int,
        per_page: int,
    ) -> list[dict[str, Any]]:
        code = _code(PERIOD_CODES, period, "períodos")
        pair = self._pair(symbol)

        async def fetch(n: int, before_ms: int | None) -> list[dict[str, Any]]:
            params: dict[str, Any] = {"symbol": pair, period_param: code, "limit": n}
            if before_ms is not None:
                params["endTime"] = before_ms
            items = await self._list(path, params)
            with parse_guard(self.name, path):
                return [dict(x, _ts=int(x["timestamp"])) for x in items]

        return await page_backward(fetch, limit, per_page, lambda x: x["_ts"])

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        path = "/v5/market/open-interest"
        items = await self._period_list(
            path, "intervalTime", symbol, period, limit, OPEN_INTEREST_MAX_LIMIT
        )
        with parse_guard(self.name, path):
            points = [
                OpenInterestPoint(time=ms_to_utc(x["_ts"]), contracts=float(x["openInterest"]))
                for x in items
            ]
        return OpenInterestHistory(normalize_symbol(symbol), period, points, self._meta())

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        path = "/v5/market/account-ratio"
        items = await self._period_list(
            path, "period", symbol, period, limit, ACCOUNT_RATIO_MAX_LIMIT
        )
        points: list[LongShortPoint] = []
        with parse_guard(self.name, path):
            for x in items:
                buy, sell = float(x["buyRatio"]), float(x["sellRatio"])
                if sell <= 0:  # sin shorts el ratio no está definido
                    continue
                points.append(
                    LongShortPoint(
                        time=ms_to_utc(x["_ts"]), ratio=buy / sell, long_pct=buy, short_pct=sell
                    )
                )
        return LongShortHistory(normalize_symbol(symbol), period, points, self._meta())
