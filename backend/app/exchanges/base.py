"""Contratos comunes para todos los exchanges.

Todo el sistema habla con los exchanges a través de estas interfaces, así en la
fase 4 se enchufa Nexo con órdenes sin tocar indicadores, scoring ni riesgo.

Convenciones (valen para todos los conectores):

- **Símbolos**: la app usa el activo base (``"BTC"``). Cada conector lo traduce
  a su formato (``"BTCUSDT"`` en Binance y Bybit) con la moneda de cotización de
  ``symbols.quote``.
- **Tiempo**: ``datetime`` con zona horaria UTC, siempre.
- **Tasas**: como fracción. ``0.0001`` = 0,01 %. (config.yaml usa porcentajes
  "humanos"; la conversión se hace donde se compara.)
- **Velas**: ``DataFrame`` con índice ``open_time`` (``DatetimeIndex`` UTC,
  ordenado y sin duplicados) y columnas ``open, high, low, close, volume``
  (float64, volumen en la moneda base) más ``closed`` (bool: ``False`` solo para
  la vela que todavía se está formando, que es siempre la última).
- **FetchMeta**: cada dato trae de dónde vino y cuándo se actualizó, para que la
  UI muestre la hora de la última actualización por fuente y los avisos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

Timeframe = Literal["5m", "15m", "1h", "4h", "1d"]

TIMEFRAME_MINUTES: dict[str, int] = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
CANDLE_COLUMNS: tuple[str, ...] = (*OHLCV_COLUMNS, "closed")


# --- errores -----------------------------------------------------------------


class ExchangeError(Exception):
    """Error al consultar un exchange: HTTP inesperado o respuesta con formato inválido."""


class ExchangeUnavailableError(ExchangeError):
    """El exchange no responde, bloquea la región (HTTP 451/403) o da 5xx.

    Es la señal para que ``registry`` pruebe con el fallback.
    """


class TradingDisabledError(RuntimeError):
    """La fase 1 no ejecuta operaciones. Cualquier intento de operar lanza esto."""


# --- metadatos ---------------------------------------------------------------


@dataclass(frozen=True)
class FetchMeta:
    """De dónde vino un dato y cuándo se actualizó."""

    source: str  # "binance", "bybit", "nexo", "manual", "alternative.me", ...
    fetched_at: datetime  # UTC, momento de la última consulta exitosa
    from_cache: bool = False
    stale: bool = False  # la consulta falló y se devolvió el último valor conocido
    fallback: bool = False  # vino de la fuente de respaldo
    warning: str | None = None  # texto para mostrar en la UI, en español


# --- modelos de mercado ------------------------------------------------------


@dataclass(frozen=True)
class Candles:
    symbol: str
    timeframe: Timeframe
    df: pd.DataFrame  # ver convenciones del módulo
    meta: FetchMeta

    def closed_only(self) -> pd.DataFrame:
        """Solo las velas cerradas (descarta la que se está formando)."""
        return self.df[self.df["closed"]]


@dataclass(frozen=True)
class Ticker:
    symbol: str
    price: float  # último precio
    meta: FetchMeta
    change_24h_pct: float | None = None  # en %, ej. -2.5
    high_24h: float | None = None
    low_24h: float | None = None
    quote_volume_24h: float | None = None  # en USDT


@dataclass(frozen=True)
class FundingInfo:
    """Funding actual (Binance ``premiumIndex``)."""

    symbol: str
    rate: float  # fracción por período, la última conocida/estimada
    mark_price: float
    meta: FetchMeta
    index_price: float | None = None
    next_funding_time: datetime | None = None
    interval_hours: float = 8


@dataclass(frozen=True)
class FundingRatePoint:
    time: datetime
    rate: float  # fracción


@dataclass(frozen=True)
class FundingHistory:
    symbol: str
    points: list[FundingRatePoint]  # orden cronológico
    meta: FetchMeta


@dataclass(frozen=True)
class OpenInterest:
    symbol: str
    contracts: float  # en moneda base
    meta: FetchMeta
    value_usd: float | None = None
    time: datetime | None = None


@dataclass(frozen=True)
class OpenInterestPoint:
    time: datetime
    contracts: float
    value_usd: float | None = None


@dataclass(frozen=True)
class OpenInterestHistory:
    symbol: str
    period: Timeframe
    points: list[OpenInterestPoint]  # orden cronológico
    meta: FetchMeta


@dataclass(frozen=True)
class LongShortPoint:
    time: datetime
    ratio: float  # cuentas long / cuentas short
    long_pct: float | None = None  # fracción 0-1
    short_pct: float | None = None


@dataclass(frozen=True)
class LongShortHistory:
    symbol: str
    period: Timeframe
    points: list[LongShortPoint]  # orden cronológico
    meta: FetchMeta


# --- interfaces --------------------------------------------------------------


@runtime_checkable
class PriceSource(Protocol):
    """Lo mínimo para dar un precio. Nexo implementa solo esto en la fase 1."""

    name: str

    async def get_ticker(self, symbol: str) -> Ticker: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class MarketDataSource(PriceSource, Protocol):
    """Fuente completa de datos de mercado (Binance, Bybit).

    Todas las funciones son async y lanzan ``ExchangeUnavailableError`` cuando
    corresponde usar el fallback, o ``ExchangeError`` ante respuestas inválidas.
    """

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        """Últimas ``limit`` velas (hasta ``end_time`` si se indica), incluida la que se forma."""
        ...

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        """Velas entre ``start`` y ``end`` paginando lo necesario (para el HMM y el backtest)."""
        ...

    async def get_funding(self, symbol: str) -> FundingInfo: ...

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        """Últimos ``limit`` funding rates (21 × 8 h = 7 días)."""
        ...

    async def get_open_interest(self, symbol: str) -> OpenInterest: ...

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory: ...

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory: ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Ejecución de órdenes. Existe para la fase 4; en la fase 1 no hay ninguna habilitada."""

    async def place_order(
        self, symbol: str, side: Literal["long", "short"], quantity: float
    ) -> str: ...

    async def set_leverage(self, symbol: str, leverage: float) -> None: ...

    async def get_positions(self) -> list[dict]: ...

    async def close_position(self, symbol: str) -> None: ...


@dataclass
class DisabledOrderExecutor:
    """Única ``OrderExecutor`` de la fase 1: todo lanza ``TradingDisabledError``."""

    reason: str = "Fase 1: la app no ejecuta operaciones."
    _calls: list[str] = field(default_factory=list, repr=False)

    def _refuse(self, what: str) -> None:
        self._calls.append(what)
        raise TradingDisabledError(f"{self.reason} (intento: {what})")

    async def place_order(
        self, symbol: str, side: Literal["long", "short"], quantity: float
    ) -> str:
        self._refuse(f"place_order {symbol} {side} {quantity}")
        raise AssertionError("unreachable")

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        self._refuse(f"set_leverage {symbol} {leverage}")

    async def get_positions(self) -> list[dict]:
        self._refuse("get_positions")
        raise AssertionError("unreachable")

    async def close_position(self, symbol: str) -> None:
        self._refuse(f"close_position {symbol}")


# --- helpers -----------------------------------------------------------------


def empty_candles_frame() -> pd.DataFrame:
    """DataFrame de velas vacío con las columnas y tipos correctos."""
    df = pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS} | {"closed": pd.Series(dtype=bool)}
    )
    df.index = pd.DatetimeIndex([], tz="UTC", name="open_time")
    return df


def validate_candles_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica las convenciones de velas y devuelve el DataFrame normalizado.

    Lanza ``ExchangeError`` si faltan columnas, el índice no es UTC o hay
    precios no positivos. Ordena y elimina duplicados de ``open_time``
    (quedándose con el último).
    """
    missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise ExchangeError(f"Velas sin columnas {missing}")
    if not isinstance(df.index, pd.DatetimeIndex) or str(df.index.tz) != "UTC":
        raise ExchangeError("El índice de velas tiene que ser DatetimeIndex en UTC")
    out = df[~df.index.duplicated(keep="last")].sort_index()
    out = out.astype({c: "float64" for c in OHLCV_COLUMNS} | {"closed": bool})
    if len(out) and (out[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        raise ExchangeError("Velas con precios no positivos")
    out.index.name = "open_time"
    return out
