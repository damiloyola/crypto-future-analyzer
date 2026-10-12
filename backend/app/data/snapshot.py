"""Foto completa del mercado para una moneda.

``build_snapshot(symbol, settings, market_data, nexo, client, cache, secrets)``
pide todo en paralelo (``asyncio.gather`` con ``return_exceptions=True``):

- velas de cada timeframe (``timeframes.context`` + ``structure`` + ``entry`` +
  ``fine_tune``), ``timeframes.candles_per_request`` velas por timeframe;
- ticker de los datos de mercado (Binance, o Bybit si Binance falla);
- funding actual e historial de ``funding_history_days`` días (con
  ``fees.funding_interval_hours`` se calcula cuántos puntos pedir: 7 días de a
  8 h son 21);
- interés abierto actual e historial horario de ``oi_history_hours`` horas;
- ratio long/short horario de ``oi_history_hours`` horas;
- Fear & Greed, mercado global (CoinGecko) y titulares.

Después arma el precio de ejecución con ``registry.get_execution_price``
(decisión del usuario: Binance, salvo que pegue a mano el precio de Nexo),
reutilizando el ticker ya pedido.

Esenciales y opcionales
    Las velas de ``timeframes.entry``, ``structure`` y ``context`` son
    **esenciales**: si falta alguna (error o vacía) se lanza
    ``EssentialDataError`` con un mensaje claro en castellano. Todo lo demás es
    opcional: si falla queda en ``None`` y se suma un aviso ("No se pudo obtener
    el funding (HTTP 451): el análisis sigue sin ese dato."). Si se pasa
    ``timeframes`` a mano, son esenciales solo los de esa lista que estén entre
    los de arriba; si ninguno lo está, alcanza con que llegue alguno. Si los
    exchanges rechazaron el pedido (por ejemplo, un símbolo que no existe), el
    mensaje lo dice y pide revisar el símbolo en vez de "probá de nuevo".

Precio actual
    ``MarketSnapshot.current_price`` es el precio de ejecución; si no se pudo
    armar (sin ticker y sin precio manual), el último cierre de las velas de
    ``timeframes.entry``, con aviso.

Avisos y fuentes
    ``warnings`` junta, sin repetir, los avisos de cada ``FetchMeta`` (fallback
    a Bybit, dato desactualizado, datos sintéticos), los de noticias y precio de
    ejecución, y los de cada dato que faltó. ``sources`` tiene una fila por dato
    (nombre, fuente, hora de actualización, si vino de la caché, si está
    desactualizado, si vino del respaldo y el aviso) para la "última
    actualización" de la UI.

Modo sin red
    ``build_offline_snapshot`` hace lo mismo con ``SyntheticMarketData`` y sus
    Fear & Greed, mercado global y titulares sintéticos: el camino de código es
    el mismo y todo sale marcado como sintético.

Además hay funciones puras para los derivados: ``history_change_pct``,
``oi_change_pct``, ``long_short_change_pct`` y ``funding_summary``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import httpx

from app.config import AppSettings, Secrets
from app.data.cache import TTLCache
from app.data.market import GlobalMarket, get_global_market
from app.data.news import NewsResult, get_headlines
from app.data.sentiment import FearGreed, get_fear_greed
from app.data.synthetic import DEFAULT_SEED, SyntheticMarketData
from app.exchanges.base import (
    TIMEFRAME_MINUTES,
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
from app.exchanges.registry import (
    ExecutionPrice,
    build_nexo_price_source,
    failure_reason,
    get_execution_price,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

PRICE_SOURCE_LAST_CLOSE = "ultimo_cierre"


class EssentialDataError(ExchangeUnavailableError):
    """Faltan datos sin los que no se puede analizar (velas de entry/structure/context)."""


# --- modelos -------------------------------------------------------------------


@dataclass(frozen=True)
class SourceStatus:
    """Una fila de "última actualización" por dato."""

    name: str  # "Velas 15m", "Funding", ... (para la UI)
    source: str | None  # "binance", "bybit", "alternative.me", "sintetico"... None si falló
    fetched_at: datetime | None
    from_cache: bool = False
    stale: bool = False
    fallback: bool = False
    warning: str | None = None
    ok: bool = True


@dataclass(frozen=True)
class FundingSummary:
    """Resumen del historial de funding (tasas como fracción por período)."""

    count: int
    mean: float
    min: float
    max: float
    total: float  # suma de las tasas del período (lo que pagó un long que estuvo todo el tiempo)
    positive_share: float  # fracción de períodos con funding > 0
    start: datetime
    end: datetime


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    created_at: datetime
    candles: dict[str, Candles]
    current_price: float
    price_source: str  # "manual_nexo", "nexo", "binance" o "ultimo_cierre"
    ticker: Ticker | None
    execution_price: ExecutionPrice | None
    funding: FundingInfo | None
    funding_history: FundingHistory | None
    open_interest: OpenInterest | None
    oi_history: OpenInterestHistory | None
    oi_change_24h_pct: float | None  # cambio del OI (contratos) en oi_lookback_hours, en %
    long_short: LongShortHistory | None
    fear_greed: FearGreed | None
    global_market: GlobalMarket | None
    news: NewsResult | None
    warnings: list[str] = field(default_factory=list)
    sources: list[SourceStatus] = field(default_factory=list)
    offline: bool = False


@dataclass(frozen=True)
class ExtraSources:
    """Cómo conseguir los datos que no vienen de un exchange (inyectable en tests)."""

    fear_greed: Callable[[], Awaitable[FearGreed]]
    global_market: Callable[[], Awaitable[GlobalMarket]]
    news: Callable[[], Awaitable[NewsResult]]


def live_extra_sources(
    symbol: str,
    settings: AppSettings,
    client: httpx.AsyncClient,
    cache: TTLCache,
    secrets: Secrets | None,
) -> ExtraSources:
    """Fear & Greed (alternative.me), CoinGecko y noticias reales."""
    token = secrets.cryptopanic_api_key if secrets is not None else None
    return ExtraSources(
        fear_greed=lambda: get_fear_greed(client, cache, settings),
        global_market=lambda: get_global_market(client, cache, settings),
        news=lambda: get_headlines(symbol, client, cache, settings, token),
    )


def synthetic_extra_sources(
    market: SyntheticMarketData, symbol: str, settings: AppSettings
) -> ExtraSources:
    """Las mismas tres fuentes, inventadas por ``SyntheticMarketData``."""

    async def fear_greed() -> FearGreed:
        return market.fear_greed(symbol)

    async def global_market() -> GlobalMarket:
        return market.global_market(symbol)

    async def news() -> NewsResult:
        return market.headlines(symbol, settings.data_sources.news_max_headlines)

    return ExtraSources(fear_greed, global_market, news)


# --- funciones puras -----------------------------------------------------------


def snapshot_timeframes(settings: AppSettings) -> list[str]:
    """Timeframes a pedir según ``timeframes``, del más largo al más corto, sin repetir."""
    tfc = settings.timeframes
    wanted = [*tfc.context, tfc.structure, tfc.entry]
    if tfc.fine_tune:
        wanted.append(tfc.fine_tune)
    unique = list(dict.fromkeys(wanted))
    return sorted(unique, key=lambda tf: -TIMEFRAME_MINUTES.get(tf, 0))


def essential_timeframes(settings: AppSettings) -> list[str]:
    """Timeframes sin los que no se analiza: ``context`` + ``structure`` + ``entry``."""
    tfc = settings.timeframes
    return list(dict.fromkeys([*tfc.context, tfc.structure, tfc.entry]))


def history_change_pct(
    points: Sequence[Any],
    hours: float,
    value_of: Callable[[Any], float | None],
) -> float | None:
    """Cambio en % entre el último punto y el último que tenga ``hours`` horas o más.

    ``points`` en orden cronológico, cada uno con ``.time``. ``None`` si no hay
    un punto lo bastante viejo o algún valor falta o es <= 0.
    """
    if not points:
        return None
    last = points[-1]
    cutoff = last.time - timedelta(hours=hours)
    base = None
    for p in points:
        if p.time <= cutoff:
            base = p
        else:
            break
    if base is None:
        return None
    now_value, base_value = value_of(last), value_of(base)
    if now_value is None or base_value is None or base_value <= 0:
        return None
    if not (math.isfinite(now_value) and math.isfinite(base_value)):
        return None
    return (now_value / base_value - 1.0) * 100.0


def oi_change_pct(history: OpenInterestHistory | None, hours: float) -> float | None:
    """Cambio del interés abierto (en contratos, no en USD) en las últimas ``hours`` horas."""
    if history is None:
        return None
    return history_change_pct(history.points, hours, lambda p: p.contracts)


def long_short_change_pct(history: LongShortHistory | None, hours: float) -> float | None:
    """Cambio del ratio long/short en las últimas ``hours`` horas, en %."""
    if history is None:
        return None
    return history_change_pct(history.points, hours, lambda p: p.ratio)


def funding_summary(history: FundingHistory | None) -> FundingSummary | None:
    """Promedio, mínimo, máximo y suma del historial de funding."""
    if history is None:
        return None
    pts = [p for p in history.points if math.isfinite(p.rate)]
    if not pts:
        return None
    rates = [p.rate for p in pts]
    return FundingSummary(
        count=len(rates),
        mean=sum(rates) / len(rates),
        min=min(rates),
        max=max(rates),
        total=sum(rates),
        positive_share=sum(1 for r in rates if r > 0) / len(rates),
        start=pts[0].time,
        end=pts[-1].time,
    )


_BOTH_FAILED = re.compile(r"^(\w+) y (\w+) fallaron\. \1: (.*)\. \2: (.*)$", re.DOTALL)


def describe_failure(exc: BaseException) -> str:
    """Motivo corto para la UI; si fallaron principal y respaldo, nombra a los dos.

    ``"Binance y Bybit fallaron. Binance: ... HTTP 451 .... Bybit: ... HTTP 403 ..."``
    queda como ``"Binance (HTTP 451) y Bybit (HTTP 403) fallaron"``.
    """
    m = _BOTH_FAILED.match(str(exc))
    if m is None:
        return failure_reason(exc)
    primary, fallback, primary_err, fallback_err = m.groups()
    return (
        f"{primary} ({failure_reason(primary_err)}) y "
        f"{fallback} ({failure_reason(fallback_err)}) fallaron"
    )


def _dedupe(texts: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    for t in texts:
        if t and t not in out:
            out.append(t)
    return out


# --- armado --------------------------------------------------------------------


class _KnownTicker:
    """``PriceSource`` que devuelve el ticker (o el error) que ya se pidió."""

    def __init__(self, name: str, ticker: Ticker | None, error: BaseException | None) -> None:
        self.name = name
        self._ticker = ticker
        self._error = error

    async def get_ticker(self, symbol: str) -> Ticker:
        if self._ticker is not None:
            return self._ticker
        if isinstance(self._error, ExchangeError):
            raise self._error
        raise ExchangeUnavailableError(
            f"sin ticker ({type(self._error).__name__ if self._error else 'sin datos'})"
        )

    async def aclose(self) -> None:
        """Nada que cerrar."""


@dataclass
class _Collector:
    """Acumula avisos y filas de fuentes mientras se arma la foto."""

    warnings: list[str] = field(default_factory=list)
    sources: list[SourceStatus] = field(default_factory=list)

    def ok(
        self,
        name: str,
        meta: FetchMeta,
        extra_warnings: Iterable[str] = (),
        *,
        use_meta_warning: bool = True,
    ) -> None:
        self.sources.append(
            SourceStatus(
                name=name,
                source=meta.source,
                fetched_at=meta.fetched_at,
                from_cache=meta.from_cache,
                stale=meta.stale,
                fallback=meta.fallback,
                warning=meta.warning,
            )
        )
        if use_meta_warning and meta.warning:
            self.warnings.append(meta.warning)
        self.warnings.extend(w for w in extra_warnings if w)

    def failed(self, name: str, label: str, exc: BaseException) -> None:
        reason = describe_failure(exc)
        verb = (
            "No se pudieron obtener" if label.startswith(("los ", "las ")) else "No se pudo obtener"
        )
        text = f"{verb} {label} ({reason}): el análisis sigue sin ese dato."
        self.sources.append(SourceStatus(name, None, None, warning=text, ok=False))
        self.warnings.append(text)


def _unwrap(result: Any, name: str, label: str, col: _Collector) -> Any | None:
    """Valor de un resultado de ``gather``; si es un error lo registra y devuelve ``None``."""
    if isinstance(result, BaseException):
        if not isinstance(result, Exception):  # CancelledError, KeyboardInterrupt...
            raise result
        if not isinstance(result, ExchangeError | httpx.HTTPError | OSError):
            logger.exception("Error inesperado al obtener %s", label, exc_info=result)
        col.failed(name, label, result)
        return None
    return result


def _join_words(words: Sequence[str]) -> str:
    """``["BTC", "ETH", "SOL"]`` → ``"BTC, ETH o SOL"``."""
    if len(words) <= 1:
        return "".join(words)
    return f"{', '.join(words[:-1])} o {words[-1]}"


def _missing_candles_error(
    sym: str,
    missing: Sequence[str],
    failed: dict[str, tuple[str, BaseException | None]],
    settings: AppSettings,
) -> EssentialDataError:
    """``EssentialDataError`` con un mensaje claro según por qué faltan las velas.

    Si los exchanges rechazaron el pedido (``ExchangeError`` que no es "no
    respondió"), reintentar no sirve: lo más probable es que el símbolo no exista.
    """
    by_reason: dict[str, list[str]] = {}
    for tf in missing:
        by_reason.setdefault(failed[tf][0], []).append(tf)
    if len(by_reason) == 1:
        detail = next(iter(by_reason))
    else:
        detail = "; ".join(f"{', '.join(t)}: {reason}" for reason, t in by_reason.items())
    errors = [failed[tf][1] for tf in missing]
    first_exc = next((e for e in errors if e is not None), None)
    rejected = all(
        isinstance(e, ExchangeError) and not isinstance(e, ExchangeUnavailableError) for e in errors
    )
    head = f"No se pudieron obtener las velas de {', '.join(missing)} de {sym} ({detail})."
    if rejected:
        examples = _join_words(list(settings.symbols.favorites))
        if all("símbolo inválido" in failed[tf][0] for tf in missing):
            advice = (
                f"Parece que {sym} no existe como futuro en {settings.symbols.quote} en esos "
                "exchanges"
            )
        else:
            advice = "Los exchanges rechazaron el pedido"
        tail = f" {advice}: revisá que el símbolo esté bien escrito (por ejemplo {examples})."
    else:
        tail = " Sin esas velas no se puede analizar: probá de nuevo en unos minutos."
    error = EssentialDataError(head + tail)
    error.__cause__ = first_exc  # como ``raise ... from first_exc``
    error.__suppress_context__ = True
    return error


async def _assemble(
    symbol: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    nexo: PriceSource,
    extras: ExtraSources,
    *,
    manual_nexo_price: float | None,
    timeframes: Sequence[str] | None,
    clock: Callable[[], datetime],
    funding_history_days: float,
    oi_history_hours: int,
    offline: bool,
) -> MarketSnapshot:
    sym = symbol.strip().upper()
    if not sym or not sym.isalnum():
        raise ValueError(f"Símbolo inválido: {symbol!r}")
    if funding_history_days <= 0 or oi_history_hours < 1:
        raise ValueError("funding_history_days y oi_history_hours tienen que ser positivos")
    tfs: list[str] = (
        list(dict.fromkeys(timeframes)) if timeframes else snapshot_timeframes(settings)
    )
    unknown = [tf for tf in tfs if tf not in TIMEFRAME_MINUTES]
    if unknown:
        raise ValueError(f"Timeframes desconocidos: {unknown}")
    essential = [tf for tf in essential_timeframes(settings) if tf in tfs]
    limit = settings.timeframes.candles_per_request
    funding_limit = max(
        1, math.ceil(funding_history_days * 24 / settings.fees.funding_interval_hours)
    )
    period: Timeframe = "1h"

    tasks: list[Awaitable[Any]] = [
        market_data.get_klines(sym, tf, limit)  # type: ignore[arg-type]
        for tf in tfs
    ]
    tasks += [
        market_data.get_ticker(sym),
        market_data.get_funding(sym),
        market_data.get_funding_history(sym, funding_limit),
        market_data.get_open_interest(sym),
        market_data.get_open_interest_history(sym, period, oi_history_hours),
        market_data.get_long_short_ratio(sym, period, oi_history_hours),
        extras.fear_greed(),
        extras.global_market(),
        extras.news(),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candle_results, rest = results[: len(tfs)], results[len(tfs) :]

    # Velas: primero las esenciales.
    candles: dict[str, Candles] = {}
    failed: dict[str, tuple[str, BaseException | None]] = {}  # tf -> (motivo, excepción)
    col = _Collector()
    for tf, res in zip(tfs, candle_results, strict=True):
        if isinstance(res, BaseException) and not isinstance(res, Exception):
            raise res
        if isinstance(res, Candles) and not res.df.empty:
            candles[tf] = res
            col.ok(f"Velas {tf}", res.meta)
            continue
        exc = res if isinstance(res, BaseException) else None
        failed[tf] = (describe_failure(exc) if exc is not None else "sin velas", exc)
        if tf not in essential:
            if exc is not None:
                _unwrap(exc, f"Velas {tf}", f"las velas de {tf}", col)
            else:
                col.failed(f"Velas {tf}", f"las velas de {tf}", ExchangeError("sin velas"))
    # Con timeframes a mano puede no haber ninguno esencial: igual hace falta alguna vela.
    required = essential if essential or candles else tfs
    missing = [tf for tf in tfs if tf in required and tf in failed]
    if missing:
        raise _missing_candles_error(sym, missing, failed, settings)

    ticker_res, funding_res, fhist_res, oi_res, oih_res, ls_res, fg_res, gm_res, news_res = rest

    ticker = _unwrap(ticker_res, "Precio", "el precio actual", col)
    if ticker is not None:
        col.ok("Precio", ticker.meta)
    funding = _unwrap(funding_res, "Funding", "el funding", col)
    if funding is not None:
        col.ok("Funding", funding.meta)
    funding_history = _unwrap(fhist_res, "Funding (historial)", "el historial de funding", col)
    if funding_history is not None:
        col.ok("Funding (historial)", funding_history.meta)
    open_interest = _unwrap(oi_res, "Interés abierto", "el interés abierto", col)
    if open_interest is not None:
        col.ok("Interés abierto", open_interest.meta)
    oi_history = _unwrap(oih_res, "Interés abierto (historial)", "el historial de OI", col)
    if oi_history is not None:
        col.ok("Interés abierto (historial)", oi_history.meta)
    long_short = _unwrap(ls_res, "Long/short", "el ratio long/short", col)
    if long_short is not None:
        col.ok("Long/short", long_short.meta)
    fear_greed = _unwrap(fg_res, "Fear & Greed", "el índice de miedo y codicia", col)
    if fear_greed is not None:
        col.ok("Fear & Greed", fear_greed.meta)
    global_market = _unwrap(gm_res, "Mercado global", "los datos del mercado global", col)
    if global_market is not None:
        col.ok("Mercado global", global_market.meta)
    news = _unwrap(news_res, "Noticias", "las noticias", col)
    if news is not None:
        if news.meta.stale and not news.headlines:
            # Fallaron todas las fuentes de noticias y no había nada guardado.
            col.sources.append(
                SourceStatus(
                    "Noticias", news.meta.source, None, warning=news.meta.warning, ok=False
                )
            )
            col.warnings.extend(news.warnings or [news.meta.warning or ""])
        else:
            # ``meta.warning`` es la unión de ``warnings``: se agregan de a uno.
            col.ok("Noticias", news.meta, news.warnings, use_meta_warning=not news.warnings)

    # Precio de ejecución: reutiliza el ticker ya pedido (no repite la consulta).
    known = _KnownTicker(
        market_data.name, ticker, ticker_res if isinstance(ticker_res, BaseException) else None
    )
    execution: ExecutionPrice | None
    try:
        execution = await get_execution_price(
            sym, known, nexo, settings, manual_nexo_price, clock=clock
        )
    except ExchangeError as exc:
        execution = None
        if ticker is not None:
            col.failed("Precio de ejecución", "el precio de ejecución", exc)
        else:  # el motivo ya se avisó con el ticker
            col.sources.append(SourceStatus("Precio de ejecución", None, None, ok=False))
    if execution is not None:
        col.ok("Precio de ejecución", execution.meta, execution.warnings)
        current_price, price_source = execution.price, execution.source
    else:
        entry_tf = settings.timeframes.entry
        base = candles.get(entry_tf) or next(iter(candles.values()))
        current_price = float(base.df["close"].iloc[-1])
        price_source = PRICE_SOURCE_LAST_CLOSE
        col.warnings.append(
            f"Sin precio actual: se usa el último cierre de las velas de {base.timeframe}."
        )

    lookback = settings.scoring.derivatives.oi_lookback_hours
    return MarketSnapshot(
        symbol=sym,
        created_at=clock(),
        candles=candles,
        current_price=current_price,
        price_source=price_source,
        ticker=ticker,
        execution_price=execution,
        funding=funding,
        funding_history=funding_history,
        open_interest=open_interest,
        oi_history=oi_history,
        oi_change_24h_pct=oi_change_pct(oi_history, lookback),
        long_short=long_short,
        fear_greed=fear_greed,
        global_market=global_market,
        news=news,
        warnings=_dedupe(col.warnings),
        sources=col.sources,
        offline=offline,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def build_snapshot(
    symbol: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    nexo: PriceSource,
    client: httpx.AsyncClient | None,
    cache: TTLCache | None,
    secrets: Secrets | None,
    manual_nexo_price: float | None = None,
    timeframes: Sequence[str] | None = None,
    *,
    extras: ExtraSources | None = None,
    clock: Callable[[], datetime] = _utcnow,
    funding_history_days: float = 7.0,
    oi_history_hours: int = 168,
    offline: bool = False,
) -> MarketSnapshot:
    """Foto del mercado de ``symbol`` (ver el docstring del módulo).

    ``extras`` reemplaza las fuentes de Fear & Greed, mercado global y
    noticias; si no se pasa, se usan las reales con ``client``, ``cache`` y la
    key de CryptoPanic de ``secrets`` (que nunca sale de ``data.news``).
    Lanza ``EssentialDataError`` si faltan velas esenciales y ``ValueError`` si
    el precio manual de Nexo no es válido.
    """
    if extras is None:
        if client is None or cache is None:
            raise ValueError("Sin extras hacen falta client y cache para las fuentes reales")
        extras = live_extra_sources(symbol.strip().upper(), settings, client, cache, secrets)
    return await _assemble(
        symbol,
        settings,
        market_data,
        nexo,
        extras,
        manual_nexo_price=manual_nexo_price,
        timeframes=timeframes,
        clock=clock,
        funding_history_days=funding_history_days,
        oi_history_hours=oi_history_hours,
        offline=offline,
    )


async def build_offline_snapshot(
    symbol: str,
    settings: AppSettings,
    manual_nexo_price: float | None = None,
    timeframes: Sequence[str] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    now: datetime | None = None,
    market: SyntheticMarketData | None = None,
    funding_history_days: float = 7.0,
    oi_history_hours: int = 168,
) -> MarketSnapshot:
    """Como ``build_snapshot`` pero con datos sintéticos (sin red). Todo sale marcado."""
    md = market or SyntheticMarketData(
        seed=seed, now=now, funding_interval_hours=settings.fees.funding_interval_hours
    )
    sym = symbol.strip().upper()
    return await _assemble(
        sym,
        settings,
        md,
        build_nexo_price_source(settings),
        synthetic_extra_sources(md, sym, settings),
        manual_nexo_price=manual_nexo_price,
        timeframes=timeframes,
        clock=lambda: md.now,
        funding_history_days=funding_history_days,
        oi_history_hours=oi_history_hours,
        offline=True,
    )
