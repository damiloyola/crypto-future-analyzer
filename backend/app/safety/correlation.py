"""Correlación entre monedas y riesgo sumado.

Para qué sirve
    Si ya hay una posición abierta y el plan nuevo va en la **misma dirección**
    en una moneda que se mueve casi igual (BTC, ETH y SOL suelen hacerlo), en
    la práctica es la misma apuesta dos veces: si el mercado se da vuelta,
    pierden las dos juntas. Por eso sus riesgos cuentan como uno solo, sumado,
    y ese total no puede pasar de ``correlation.max_combined_risk_multiplier``
    × el riesgo por operación (1,5 × 50 = 75 USD con la config actual).

Qué hay acá (todo puro salvo ``fetch_correlation_matrix``)
    - ``returns_from_candles(df)``: retornos logarítmicos de las velas
      cerradas, solo entre velas consecutivas.
    - ``correlation_matrix(returns_by_symbol, min_overlap=...)``: matriz de
      correlación de Pearson, par por par, sobre las marcas de tiempo que las
      dos monedas tienen en común.
    - ``fetch_correlation_matrix(symbols, market_data, settings, now=...)``:
      baja las velas (``correlation.timeframe`` y ``correlation.lookback_days``)
      y arma la matriz. Una moneda que falla queda afuera con un aviso; nunca
      rompe el análisis.
    - ``correlated_exposure(new_symbol, new_side, open_positions, matrix,
      threshold)``: qué posiciones abiertas suman riesgo con el plan nuevo y
      cuáles no (y por qué).
    - ``combined_risk_limit(risk_per_trade_usd, multiplier)``: tope del riesgo
      combinado.

    Qué hacer cuando el riesgo existente más el del plan nuevo pasa el tope
    (achicar el plan o bloquearlo) lo decide la capa de arriba (breakers y
    servicio). Este módulo solo da los números y los textos.

Decisiones
    - **Pearson sobre retornos log** de velas de 1 h (configurable). Los
      retornos, no los precios: dos precios que suben por su cuenta parecen
      correlacionados aunque no se muevan juntos hora a hora.
    - **Par por par** (*pairwise*): cada par usa solo las marcas de tiempo en
      las que las dos monedas tienen retorno. Una moneda con un hueco de datos
      no le quita horas a los demás pares.
    - **Solo velas consecutivas**: si falta una vela, el retorno que saltaría
      el hueco se descarta. Si no, se compararía un movimiento de 2 h de una
      moneda con uno de 1 h de la otra.
    - **Pocos datos = sin dato**: si un par tiene menos de ``min_overlap``
      retornos en común (o el precio de una no se movió), la correlación queda
      ``NaN`` (``None`` en ``to_dict``) en vez de inventar un número.
    - **Correlación desconocida se cuenta** (conservador): sin datos para saber
      si dos monedas se mueven juntas, se asume que sí y su riesgo se suma, con
      el motivo a la vista. Peor es dejar pasar dos apuestas iguales sin
      avisar que achicar de más un plan.
    - **Umbral inclusivo**: correlación >= ``correlation.high_threshold`` (0,8)
      cuenta. Se toleran ``THRESHOLD_TOLERANCE`` de redondeo, para que un 0,8
      que sale como 0,7999999999 por cuentas en punto flotante también cuente.
    - **La misma moneda** tiene correlación 1, siempre (aunque no esté en la
      matriz).
    - **Dirección contraria no suma al tope de posiciones correlacionadas**
      (sí cuenta en el límite del día del plan: pueden saltar los dos stops).
      Solo la **misma moneda** en dirección contraria se llama cobertura
      (``REASON_HEDGE``, texto en ``hedge_explanation``); otra moneda al
      revés del plan (``REASON_OPPOSITE_SIDE``) se explica según su correlación
      ("se mueve parecido", "no se mueve junto" o "no hay datos"), sin decir que
      cubre nada. Solo se miran correlaciones positivas (en cripto las
      correlaciones muy negativas entre monedas prácticamente no existen).
    - **Sin look-ahead**: solo velas cerradas a la hora ``now``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from app.data.snapshot import describe_failure
from app.exchanges.base import TIMEFRAME_MINUTES, ExchangeError, FetchMeta
from app.risk.rules import fmt_num, fmt_usd, json_safe
from app.risk.sizing import Side, ensure_finite, ensure_side

if TYPE_CHECKING:  # solo para tipos
    from app.config import AppSettings
    from app.exchanges.base import MarketDataSource

logger = logging.getLogger(__name__)

METHOD = "pearson"

# Fracción de los retornos esperados en ``lookback_days`` que dos monedas
# tienen que compartir para confiar en su correlación (30 días de 1 h = 720
# retornos → hacen falta 360 en común). Con 360 datos, el error típico de una
# correlación cerca de 0,8 es de alrededor de 0,02.
# Pedido para config: correlation.min_overlap_fraction.
DEFAULT_MIN_OVERLAP_FRACTION = 0.5

# Con menos de 3 puntos la correlación de Pearson no dice nada (con 2 da
# siempre +1 o -1).
MIN_OVERLAP_FLOOR = 3

# Tolerancia de redondeo para el umbral inclusivo (ver docstring del módulo).
THRESHOLD_TOLERANCE = 1e-9

# Motivos de ``ExposureEntry.reason``.
REASON_SAME_SYMBOL = "same_symbol"  # misma moneda, misma dirección: suma
REASON_HIGH_CORRELATION = "high_correlation"  # correlación >= umbral: suma
REASON_UNKNOWN_CORRELATION = "unknown_correlation"  # sin dato: suma por las dudas
REASON_HEDGE = "hedge"  # misma moneda en dirección contraria (cobertura): no suma
REASON_OPPOSITE_SIDE = "opposite_side"  # otra moneda en dirección contraria: no suma
REASON_LOW_CORRELATION = "low_correlation"  # correlación < umbral: no suma

_SIDE_WORD = {"long": "long (compra)", "short": "short (venta)"}


# --- helpers -------------------------------------------------------------------


def _norm_symbol(symbol: str) -> str:
    """``" eth "`` → ``"ETH"``. ``ValueError`` si queda vacío."""
    s = str(symbol).strip().upper()
    if not s:
        raise ValueError(f"Símbolo inválido: {symbol!r}")
    return s


def _norm_side(side: Any) -> Side:
    """Acepta ``"long"``/``"short"`` (sin importar mayúsculas) o un enum con ese valor."""
    raw = getattr(side, "value", side)
    return ensure_side(str(raw).strip().lower())


def _step(timeframe: str) -> pd.Timedelta:
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Timeframe desconocido: {timeframe!r}")
    return pd.Timedelta(minutes=minutes)


def _utc(dt: datetime) -> datetime:
    """Fecha con zona horaria → UTC. ``ValueError`` si no tiene zona (sería ambigua)."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("La fecha tiene que tener zona horaria (UTC).")
    return dt.astimezone(UTC)


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _empty_returns() -> pd.Series:
    index = pd.DatetimeIndex([], tz="UTC", name="open_time")
    return pd.Series([], index=index, dtype="float64", name="log_return")


def _clean_returns(series: pd.Series, name: str) -> pd.Series:
    """Índice UTC ordenado, sin duplicados (queda el último) y sin ``NaN``/infinitos."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"Los retornos de {name} tienen que tener un DatetimeIndex.")
    if series.index.tz is None:
        raise ValueError(f"Los retornos de {name} tienen que tener el índice en UTC.")
    out = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64"),
        index=series.index.tz_convert("UTC"),
    )
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[np.isfinite(out.to_numpy())]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación de Pearson; ``NaN`` si alguna serie es constante."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(np.dot(xc, xc)) * float(np.dot(yc, yc)))
    if denom == 0 or not math.isfinite(denom):
        return math.nan
    return float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def expected_returns(lookback_days: int, timeframe: str) -> int:
    """Cuántos retornos entran en ``lookback_days`` días de velas de ``timeframe``."""
    if lookback_days < 1:
        raise ValueError(f"lookback_days tiene que ser >= 1 (vino {lookback_days}).")
    minutes = int(_step(timeframe).total_seconds() // 60)
    return lookback_days * 1440 // minutes


def default_min_overlap(
    lookback_days: int, timeframe: str, fraction: float = DEFAULT_MIN_OVERLAP_FRACTION
) -> int:
    """Retornos en común que hacen falta para confiar en una correlación.

    ``fraction`` × los retornos esperados en el período, y nunca menos de
    ``MIN_OVERLAP_FLOOR``. 30 días de 1 h con 0,5 → 360.
    """
    fraction = ensure_finite(fraction, "La fracción mínima de datos en común")
    if not 0 < fraction <= 1:
        raise ValueError(f"La fracción mínima de datos en común va de 0 a 1 (vino {fraction}).")
    needed = math.ceil(fraction * expected_returns(lookback_days, timeframe))
    return max(MIN_OVERLAP_FLOOR, needed)


# --- retornos --------------------------------------------------------------------


def returns_from_candles(
    df: pd.DataFrame, *, timeframe: str = "1h", now: datetime | None = None
) -> pd.Series:
    """Retornos logarítmicos de las velas cerradas: ``ln(cierre_t / cierre_{t-1})``.

    - Solo velas cerradas: se descartan las que tienen ``closed == False`` y,
      si se pasa ``now``, las que todavía no cerraron a esa hora
      (``open_time + timeframe > now``).
    - Solo entre velas consecutivas: si entre dos velas hay un hueco (falta
      alguna), ese retorno se descarta.
    - Se descartan precios no positivos o no numéricos.

    Devuelve una ``Series`` float64 llamada ``"log_return"`` con índice
    ``open_time`` en UTC: cada retorno lleva la apertura de la vela en la que
    termina. Vacía si hay menos de dos velas cerradas consecutivas.
    ``ValueError`` si falta la columna ``close`` o el índice no es de fechas en
    UTC.
    """
    step = _step(timeframe)
    if "close" not in df.columns:
        raise ValueError("Las velas no tienen la columna 'close'.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Las velas tienen que tener un DatetimeIndex (open_time).")
    if df.index.tz is None:
        raise ValueError("El índice de las velas tiene que estar en UTC.")
    data = df
    if "closed" in data.columns:
        data = data[data["closed"].astype(bool).to_numpy()]
    if now is not None:
        data = data[np.asarray(data.index + step <= pd.Timestamp(_utc(now)))]
    close = pd.Series(
        pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype="float64"),
        index=data.index.tz_convert("UTC"),
    )
    close = close[~close.index.duplicated(keep="last")].sort_index()
    values = close.to_numpy()
    close = close[np.isfinite(values) & (values > 0)]
    if len(close) < 2:
        return _empty_returns()
    log_close = np.log(close.to_numpy())
    rets = np.diff(log_close)
    consecutive = np.asarray((close.index[1:] - close.index[:-1]) == step)
    out = pd.Series(rets[consecutive], index=close.index[1:][consecutive], dtype="float64")
    out = out[np.isfinite(out.to_numpy())]
    out.name = "log_return"
    out.index.name = "open_time"
    return out


# --- matriz ------------------------------------------------------------------------


@dataclass(frozen=True)
class ExcludedSymbol:
    """Una moneda que quedó fuera de la matriz y por qué (texto para la UI)."""

    symbol: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "reason": self.reason}


@dataclass(frozen=True)
class CorrelationPair:
    """Un par de la matriz (triángulo superior)."""

    a: str
    b: str
    correlation: float | None  # None = sin dato
    overlap: int  # retornos en común


@dataclass(frozen=True)
class CorrelationMatrix:
    """Matriz de correlación de retornos, lista para el panel y para el riesgo.

    ``values[i][j]`` es la correlación entre ``symbols[i]`` y ``symbols[j]``:
    simétrica, con 1 en la diagonal y ``NaN`` cuando no hay datos suficientes.
    ``overlaps[i][j]`` es la cantidad de retornos en común del par (en la
    diagonal, los retornos de la moneda).

    ``as_of``: hasta cuándo llegan los datos (en ``fetch_correlation_matrix``,
    el ``now`` del pedido; si no se pasa, la apertura de la vela del último
    retorno). ``data_start``/``data_end``: primer y último retorno usado (por
    apertura de vela). ``timeframe``/``lookback_days``: de qué velas y qué
    período salió (``None`` si no se indicó). ``high_threshold``: umbral de
    "muy correlacionadas", solo para marcar los pares en ``to_dict``.
    """

    symbols: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    overlaps: tuple[tuple[int, ...], ...]
    min_overlap: int
    as_of: datetime | None
    data_start: datetime | None
    data_end: datetime | None
    timeframe: str | None = None
    lookback_days: int | None = None
    high_threshold: float | None = None
    excluded: tuple[ExcludedSymbol, ...] = ()
    warnings: tuple[str, ...] = ()
    method: str = METHOD

    def index_of(self, symbol: str) -> int | None:
        """Posición de la moneda en ``symbols`` (``None`` si no está)."""
        s = _norm_symbol(symbol)
        return self.symbols.index(s) if s in self.symbols else None

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and self.index_of(symbol) is not None

    def correlation(self, a: str, b: str) -> float | None:
        """Correlación del par; 1 si es la misma moneda; ``None`` si no hay dato."""
        sa, sb = _norm_symbol(a), _norm_symbol(b)
        if sa == sb:
            return 1.0
        i, j = self.index_of(sa), self.index_of(sb)
        if i is None or j is None:
            return None
        return _finite_or_none(self.values[i][j])

    def overlap(self, a: str, b: str) -> int:
        """Retornos en común del par (0 si alguna moneda no está en la matriz)."""
        i, j = self.index_of(a), self.index_of(b)
        if i is None or j is None:
            return 0
        return self.overlaps[i][j]

    def pairs(self) -> tuple[CorrelationPair, ...]:
        """Todos los pares distintos, en el orden de ``symbols``."""
        n = len(self.symbols)
        return tuple(
            CorrelationPair(
                self.symbols[i],
                self.symbols[j],
                _finite_or_none(self.values[i][j]),
                self.overlaps[i][j],
            )
            for i in range(n)
            for j in range(i + 1, n)
        )

    def high_pairs(self, threshold: float | None = None) -> tuple[CorrelationPair, ...]:
        """Pares con correlación >= ``threshold`` (por omisión, ``high_threshold``)."""
        limit = self.high_threshold if threshold is None else threshold
        if limit is None:
            raise ValueError("Falta el umbral de correlación alta.")
        return tuple(
            p
            for p in self.pairs()
            if p.correlation is not None and p.correlation >= limit - THRESHOLD_TOLERANCE
        )

    def to_frame(self) -> pd.DataFrame:
        """La matriz como ``DataFrame`` (para la consola)."""
        return pd.DataFrame(
            np.asarray(self.values, dtype="float64").reshape(len(self.symbols), -1),
            index=list(self.symbols),
            columns=list(self.symbols),
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (``NaN`` → ``None``)."""
        threshold = self.high_threshold
        pairs = [
            {
                "a": p.a,
                "b": p.b,
                "correlation": p.correlation,
                "overlap": p.overlap,
                "high": None
                if threshold is None or p.correlation is None
                else p.correlation >= threshold - THRESHOLD_TOLERANCE,
            }
            for p in self.pairs()
        ]
        return json_safe(
            {
                "method": self.method,
                "timeframe": self.timeframe,
                "lookback_days": self.lookback_days,
                "min_overlap": self.min_overlap,
                "as_of": _iso(self.as_of),
                "data_start": _iso(self.data_start),
                "data_end": _iso(self.data_end),
                "high_threshold": threshold,
                "symbols": list(self.symbols),
                "values": [list(row) for row in self.values],
                "overlaps": [list(row) for row in self.overlaps],
                "pairs": pairs,
                "excluded": [e.to_dict() for e in self.excluded],
                "warnings": list(self.warnings),
            }
        )


def correlation_matrix(
    returns_by_symbol: Mapping[str, pd.Series],
    *,
    min_overlap: int,
    as_of: datetime | None = None,
    timeframe: str | None = None,
    lookback_days: int | None = None,
    high_threshold: float | None = None,
    excluded: Sequence[ExcludedSymbol] = (),
    warnings: Sequence[str] = (),
) -> CorrelationMatrix:
    """Matriz de correlación de Pearson, par por par.

    Cada par se calcula sobre las marcas de tiempo que las dos series tienen en
    común (*inner join*), sin ``NaN``. Si son menos de ``min_overlap`` o alguna
    de las dos no varía, la correlación del par queda ``NaN``. La diagonal es
    siempre 1. El orden de ``symbols`` es el de ``returns_by_symbol`` (con los
    símbolos en mayúsculas).

    ``excluded`` y ``warnings`` se copian al resultado (los usa
    ``fetch_correlation_matrix``); se agrega un aviso con los pares que
    quedaron sin dato. ``ValueError`` si ``min_overlap < 3``, si dos claves son
    la misma moneda o si una serie no tiene índice de fechas en UTC.
    """
    if isinstance(min_overlap, bool) or not isinstance(min_overlap, int):
        raise ValueError(f"min_overlap tiene que ser un entero (vino {min_overlap!r}).")
    if min_overlap < MIN_OVERLAP_FLOOR:
        raise ValueError(f"min_overlap tiene que ser >= {MIN_OVERLAP_FLOOR} (vino {min_overlap}).")
    if high_threshold is not None:
        high_threshold = ensure_finite(high_threshold, "El umbral de correlación alta")

    symbols: list[str] = []
    series: list[pd.Series] = []
    for key, s in returns_by_symbol.items():
        sym = _norm_symbol(key)
        if sym in symbols:
            raise ValueError(f"La moneda {sym} aparece dos veces.")
        symbols.append(sym)
        series.append(_clean_returns(s, sym))

    n = len(symbols)
    values = np.eye(n, dtype="float64")
    overlaps = np.zeros((n, n), dtype=int)
    for i in range(n):
        overlaps[i, i] = len(series[i])
        for j in range(i + 1, n):
            a, b = series[i].align(series[j], join="inner")
            k = len(a)
            rho = _pearson(a.to_numpy(), b.to_numpy()) if k >= min_overlap else math.nan
            values[i, j] = values[j, i] = rho
            overlaps[i, j] = overlaps[j, i] = k

    non_empty = [s for s in series if len(s)]
    data_start = min(s.index[0] for s in non_empty).to_pydatetime() if non_empty else None
    data_end = max(s.index[-1] for s in non_empty).to_pydatetime() if non_empty else None

    out_warnings = list(warnings)
    unknown = [
        f"{symbols[i]}/{symbols[j]}"
        for i in range(n)
        for j in range(i + 1, n)
        if not math.isfinite(values[i, j])
    ]
    if unknown:
        out_warnings.append(
            "Correlación sin dato (pocas velas en común o un precio que no se movió): "
            + ", ".join(unknown)
            + ". Si una de esas monedas tiene una posición abierta, su riesgo se suma "
            "por las dudas."
        )

    return CorrelationMatrix(
        symbols=tuple(symbols),
        values=tuple(tuple(float(v) for v in row) for row in values),
        overlaps=tuple(tuple(int(v) for v in row) for row in overlaps),
        min_overlap=min_overlap,
        as_of=_utc(as_of) if as_of is not None else data_end,
        data_start=data_start,
        data_end=data_end,
        timeframe=timeframe,
        lookback_days=lookback_days,
        high_threshold=high_threshold,
        excluded=tuple(excluded),
        warnings=tuple(out_warnings),
    )


# --- descarga ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Fetched:
    symbol: str
    returns: pd.Series | None = None
    meta: FetchMeta | None = None
    error: str | None = None  # motivo para la UI si quedó afuera


def _floor(dt: datetime, step: pd.Timedelta) -> datetime:
    ts = pd.Timestamp(dt)
    return ts.floor(step).to_pydatetime()


async def _fetch_returns(
    market_data: MarketDataSource,
    symbol: str,
    timeframe: str,
    start: datetime,
    now: datetime,
    lookback_days: int,
) -> _Fetched:
    """Velas de una moneda → retornos. Nunca lanza: un error queda en ``error``."""
    try:
        candles = await market_data.get_klines_history(
            symbol,
            timeframe,  # type: ignore[arg-type]
            start,
            now,
        )
    except ExchangeError as exc:
        return _Fetched(symbol, error=f"no se pudieron bajar sus velas ({describe_failure(exc)})")
    except Exception as exc:  # nunca rompe el análisis
        logger.warning("Correlación: falló la descarga de %s", symbol, exc_info=True)
        return _Fetched(symbol, error=f"falló la descarga de sus velas ({exc})")
    try:
        df = candles.df
        df = df[np.asarray(df.index >= pd.Timestamp(start))]
        rets = returns_from_candles(df, timeframe=timeframe, now=now)
    except Exception as exc:  # velas con formato inválido: queda afuera, no rompe
        logger.warning("Correlación: velas inválidas de %s", symbol, exc_info=True)
        return _Fetched(symbol, meta=candles.meta, error=f"sus velas llegaron con errores ({exc})")
    if rets.empty:
        return _Fetched(
            symbol,
            meta=candles.meta,
            error=f"no llegaron velas cerradas de {timeframe} de los últimos {lookback_days} días",
        )
    return _Fetched(symbol, returns=rets, meta=candles.meta)


def _unique_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for s in symbols:
        sym = _norm_symbol(s)
        if sym not in out:
            out.append(sym)
    return out


async def fetch_correlation_matrix(
    symbols: Iterable[str],
    market_data: MarketDataSource,
    settings: AppSettings,
    *,
    now: datetime,
    min_overlap: int | None = None,
    min_overlap_fraction: float = DEFAULT_MIN_OVERLAP_FRACTION,
) -> CorrelationMatrix:
    """Baja las velas de cada moneda (en paralelo) y arma la matriz de correlación.

    Usa ``settings.correlation``: velas de ``timeframe`` de los últimos
    ``lookback_days`` días cerradas a la hora ``now`` (más una vela anterior
    para el primer retorno), y ``high_threshold`` para marcar los pares. Sin
    ``min_overlap``, se usa ``default_min_overlap`` con
    ``min_overlap_fraction`` (360 retornos en común con 30 días de 1 h).

    Una moneda que falla (error del exchange, velas inválidas o vacías) queda
    en ``excluded`` con el motivo y un aviso en ``warnings``; nunca lanza por
    eso. Los avisos de las fuentes (``FetchMeta.warning``, por ejemplo "datos
    sintéticos" o "se usó el respaldo") también van a ``warnings``.
    ``ValueError`` si ``now`` no tiene zona horaria o un símbolo es inválido.
    """
    cfg = settings.correlation
    timeframe = cfg.timeframe
    step = _step(timeframe)
    now_utc = _utc(now)
    if min_overlap is None:
        min_overlap = default_min_overlap(cfg.lookback_days, timeframe, min_overlap_fraction)
    expected = expected_returns(cfg.lookback_days, timeframe)
    # Última vela cerrada a la hora ``now`` y una vela más atrás de las
    # ``expected`` para que el primer retorno caiga dentro del período.
    last_open = _floor(now_utc, step) - step.to_pytimedelta()
    start = last_open - step.to_pytimedelta() * expected

    wanted = _unique_symbols(symbols)
    fetched = await asyncio.gather(
        *(
            _fetch_returns(market_data, s, timeframe, start, now_utc, cfg.lookback_days)
            for s in wanted
        )
    )

    returns: dict[str, pd.Series] = {}
    excluded: list[ExcludedSymbol] = []
    warnings: list[str] = []
    for item in fetched:
        warning = item.meta.warning if item.meta is not None else None
        if warning and warning not in warnings:
            warnings.append(warning)
        if item.error is not None or item.returns is None:
            reason = item.error or "no llegaron datos"
            excluded.append(ExcludedSymbol(item.symbol, reason))
            warnings.append(f"{item.symbol} quedó fuera de la matriz de correlación: {reason}.")
            continue
        returns[item.symbol] = item.returns
        if len(item.returns) < min_overlap:
            warnings.append(
                f"De {item.symbol} hay solo {len(item.returns)} velas de {timeframe} con dato "
                f"de las {expected} del período: sin al menos {min_overlap} en común con otra "
                "moneda, la correlación queda sin dato."
            )

    return correlation_matrix(
        returns,
        min_overlap=min_overlap,
        as_of=now_utc,
        timeframe=timeframe,
        lookback_days=cfg.lookback_days,
        high_threshold=cfg.high_threshold,
        excluded=excluded,
        warnings=warnings,
    )


# --- riesgo sumado -------------------------------------------------------------------


class OpenPositionLike(Protocol):
    """Lo que hace falta de una posición abierta (la carga el usuario en la fase 1)."""

    symbol: str
    side: str  # "long" o "short"
    risk_usd: float  # pérdida si salta el stop


@dataclass(frozen=True)
class ExposureEntry:
    """Una posición abierta frente al plan nuevo: si suma riesgo o no, y por qué.

    ``risk_usd`` es el riesgo que aporta (un valor negativo, por ejemplo con
    el stop ya en ganancia, cuenta como 0). ``correlation`` es la correlación
    con la moneda del plan (``None`` si no hay dato). ``reason`` es uno de los
    ``REASON_*`` del módulo y ``explanation`` lo dice en palabras simples.
    """

    symbol: str
    side: Side
    risk_usd: float
    correlation: float | None
    counted: bool
    reason: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "symbol": self.symbol,
                "side": self.side,
                "risk_usd": self.risk_usd,
                "correlation": self.correlation,
                "counted": self.counted,
                "reason": self.reason,
                "explanation": self.explanation,
            }
        )


@dataclass(frozen=True)
class Exposure:
    """Las posiciones abiertas que suman riesgo con un plan nuevo.

    ``counted``: misma dirección y moneda muy correlacionada (o la misma, o
    sin dato de correlación). ``skipped``: las demás, con el motivo
    (cobertura en la misma moneda, otra moneda en dirección contraria o
    correlación baja). ``existing_risk_usd`` es la suma del riesgo
    de ``counted``: lo que hay que sumarle al riesgo del plan nuevo antes de
    compararlo con ``combined_risk_limit``.
    """

    new_symbol: str
    new_side: Side
    threshold: float
    counted: tuple[ExposureEntry, ...]
    skipped: tuple[ExposureEntry, ...]
    matrix_as_of: datetime | None = None

    @property
    def existing_risk_usd(self) -> float:
        return float(sum(e.risk_usd for e in self.counted))

    @property
    def unknown_symbols(self) -> tuple[str, ...]:
        """Monedas que se contaron sin saber su correlación (por las dudas)."""
        return tuple(e.symbol for e in self.counted if e.reason == REASON_UNKNOWN_CORRELATION)

    @property
    def has_unknown_correlation(self) -> bool:
        return bool(self.unknown_symbols)

    def combined_risk_usd(self, new_risk_usd: float) -> float:
        """Riesgo existente que suma + el del plan nuevo."""
        new = ensure_finite(new_risk_usd, "El riesgo del plan nuevo")
        return self.existing_risk_usd + max(new, 0.0)

    def remaining_risk_usd(self, limit_usd: float) -> float:
        """Cuánto riesgo le queda al plan nuevo hasta el tope (nunca negativo)."""
        limit = ensure_finite(limit_usd, "El tope de riesgo combinado")
        return max(0.0, limit - self.existing_risk_usd)

    @property
    def explanation(self) -> str:
        """Las posiciones que suman, en palabras simples.

        Las que se cuentan sin saber su correlación se nombran aparte: de esas
        no se dice que "se mueven junto" con la moneda del plan (no hay datos),
        solo que se cuentan por las dudas.
        """
        side = _SIDE_WORD[self.new_side]
        new = self.new_symbol
        if not self.counted:
            return (
                f"No hay posiciones abiertas en {side} que se muevan junto con "
                f"{new}: el plan no suma riesgo con otras posiciones."
            )
        known = [e for e in self.counted if e.reason != REASON_UNKNOWN_CORRELATION]
        unknown = [e for e in self.counted if e.reason == REASON_UNKNOWN_CORRELATION]
        n = len(self.counted)
        total = fmt_usd(self.existing_risk_usd)
        lose = "pueden perder" if unknown else "pierden"
        names = ", ".join(f"{e.symbol} ({fmt_usd(e.risk_usd)})" for e in self.counted)
        if n == 1:
            head = f"Ya tenés una posición en {side}"
            why = (
                f"Su riesgo ({total}) cuenta junto con el del plan nuevo porque si el mercado se "
                f"da vuelta {lose} las dos juntas."
            )
        else:
            head = f"Ya tenés {n} posiciones en {side}"
            why = (
                f"Suman {total} de riesgo, que cuenta junto con el del plan nuevo porque si el "
                f"mercado se da vuelta {lose} todas juntas."
            )
        same = _unique_coins(e for e in known if e.symbol == new)
        related = _unique_coins(e for e in known if e.symbol != new)
        if not unknown:
            if not related:
                relation = f"en {new}, la misma moneda"
            elif not same:
                relation = f"que {'se mueven' if n > 1 else 'se mueve'} junto con {new}"
            else:
                relation = f"en {new} o que se mueven junto con {new}"
            return f"{head} {relation}: {names}. {why}"
        # Con monedas sin dato de correlación no se afirma que "se mueven junto":
        # se dice cuáles sí (si hay) y que las otras se cuentan por las dudas.
        details: list[str] = []
        if same:
            details.append(f"{new} es la misma moneda")
        if related:
            moves = "se mueven" if len(related) > 1 else "se mueve"
            details.append(f"{_join_es(related)} {moves} junto con {new}")
        doubt_coins = _unique_coins(unknown)
        moves = "se mueven" if len(doubt_coins) > 1 else "se mueve"
        counts = "se cuentan" if len(unknown) > 1 else "se cuenta"
        if details:
            doubt = (
                f"De {_join_es(doubt_coins)} no hay datos para saber si {moves} junto con {new}; "
                f"por las dudas {counts}."
            )
            known_text = " y ".join(details)
            return f"{head}: {names}. {known_text[:1].upper()}{known_text[1:]}. {doubt} {why}"
        doubt = (
            f"No hay datos para saber si {_join_es(doubt_coins)} {moves} junto con {new}; por "
            f"las dudas {counts}."
        )
        return f"{head}: {names}. {doubt} {why}"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "new_symbol": self.new_symbol,
                "new_side": self.new_side,
                "threshold": self.threshold,
                "existing_risk_usd": self.existing_risk_usd,
                "counted": [e.to_dict() for e in self.counted],
                "skipped": [e.to_dict() for e in self.skipped],
                "unknown_symbols": list(self.unknown_symbols),
                "has_unknown_correlation": self.has_unknown_correlation,
                "matrix_as_of": _iso(self.matrix_as_of),
                "explanation": self.explanation,
            }
        )


def _unique_coins(entries: Iterable[ExposureEntry]) -> list[str]:
    """Las monedas de ``entries``, sin repetir y en orden."""
    return list(dict.fromkeys(e.symbol for e in entries))


def _join_es(items: Sequence[str]) -> str:
    """``["ETH", "SOL", "AVAX"]`` → ``"ETH, SOL y AVAX"``."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def _entry(
    position: OpenPositionLike,
    new_symbol: str,
    new_side: Side,
    matrix: CorrelationMatrix | None,
    threshold: float,
    *,
    limit_multiple: float | None = None,
    hedge_counts_in_day: bool = True,
) -> ExposureEntry:
    symbol = _norm_symbol(position.symbol)
    side = _norm_side(position.side)
    raw_risk = ensure_finite(position.risk_usd, f"El riesgo de la posición de {symbol}")
    risk = max(raw_risk, 0.0)
    if symbol == new_symbol:
        corr: float | None = 1.0
    elif matrix is None:
        corr = None
    else:
        corr = matrix.correlation(symbol, new_symbol)
    corr_text = "sin dato" if corr is None else fmt_num(corr, 2)

    def make(counted: bool, reason: str, explanation: str) -> ExposureEntry:
        if raw_risk < 0 and counted:
            explanation += " Tiene el stop en ganancia, así que suma 0."
        return ExposureEntry(symbol, side, risk, corr, counted, reason, explanation)

    if side != new_side:
        # Cobertura es solo la misma moneda al revés (decisión del usuario). Otra
        # moneda al revés no suma riesgo, pero tampoco "cubre" nada: se dice cómo
        # se mueve respecto de la del plan.
        if symbol == new_symbol:
            return make(
                False,
                REASON_HEDGE,
                hedge_explanation(
                    symbol, side, limit_multiple=limit_multiple, counts_in_day=hedge_counts_in_day
                ),
            )
        if corr is None:
            relation = f" (no hay datos para saber si se mueve junto con {new_symbol})"
        elif corr >= threshold - THRESHOLD_TOLERANCE:
            relation = f" y se mueve parecido a {new_symbol} (correlación {corr_text})"
        else:
            relation = f" y no se mueve junto con {new_symbol} (correlación {corr_text})"
        return make(
            False,
            REASON_OPPOSITE_SIDE,
            f"{symbol} está en {_SIDE_WORD[side]}: va al revés del plan{relation}: no suma riesgo.",
        )
    if symbol == new_symbol:
        return make(
            True,
            REASON_SAME_SYMBOL,
            f"Ya hay una posición en {_SIDE_WORD[side]} en {symbol}, la misma moneda: "
            "su riesgo se suma.",
        )
    if corr is None:
        return make(
            True,
            REASON_UNKNOWN_CORRELATION,
            f"No hay datos para saber si {symbol} se mueve junto con {new_symbol}: por las "
            "dudas su riesgo se suma.",
        )
    if corr >= threshold - THRESHOLD_TOLERANCE:
        return make(
            True,
            REASON_HIGH_CORRELATION,
            f"{symbol} y {new_symbol} se mueven casi igual (correlación {corr_text}, "
            f"el límite es {fmt_num(threshold, 2)}): es como la misma apuesta y su riesgo "
            "se suma.",
        )
    return make(
        False,
        REASON_LOW_CORRELATION,
        f"{symbol} no se mueve tan junto con {new_symbol} (correlación {corr_text}, por "
        f"debajo de {fmt_num(threshold, 2)}): no suma riesgo.",
    )


def hedge_explanation(
    symbol: str,
    side: str,
    *,
    limit_multiple: float | None = None,
    counts_in_day: bool = True,
) -> str:
    """Qué significa una posición en la misma moneda y dirección contraria (cobertura).

    No suma al tope de posiciones correlacionadas, pero (con
    ``trade_plan.daily_cap_includes_open_risk``, ``counts_in_day``) su riesgo
    sí cuenta en lo que queda del límite del día: pueden saltar los dos stops.
    ``limit_multiple`` (``correlation.max_combined_risk_multiplier``) agrega
    el tope al texto ("1,5 × tu riesgo").
    """
    cap = "" if limit_multiple is None else f" ({fmt_num(limit_multiple, 2)} × tu riesgo)"
    head = (
        f"{symbol} está en {_SIDE_WORD[side]}, al revés del plan en la misma moneda: funciona "
        f"como cobertura. No suma al tope de posiciones correlacionadas{cap}"
    )
    if counts_in_day:
        return (
            f"{head}, pero su riesgo sí cuenta en lo que te queda del límite del día, porque "
            "pueden saltar los dos stops."
        )
    return f"{head}. Ojo: pueden saltar los dos stops."


def correlated_exposure(
    new_symbol: str,
    new_side: str,
    open_positions: Iterable[OpenPositionLike],
    matrix: CorrelationMatrix | None,
    threshold: float,
    *,
    limit_multiple: float | None = None,
    hedge_counts_in_day: bool = True,
) -> Exposure:
    """Qué posiciones abiertas suman su riesgo al de un plan nuevo.

    Cuenta las posiciones en la **misma dirección** que el plan cuya
    correlación con ``new_symbol`` es >= ``threshold`` (inclusivo, con
    ``THRESHOLD_TOLERANCE``). La misma moneda tiene correlación 1. Si la
    correlación no se conoce (la moneda no está en la matriz, el par quedó sin
    dato o ``matrix`` es ``None``), **se cuenta igual**, por las dudas, con
    ``reason = REASON_UNKNOWN_CORRELATION``. Las de dirección contraria se
    saltean: la misma moneda como cobertura (``REASON_HEDGE``) y otra moneda
    con ``REASON_OPPOSITE_SIDE`` (su explicación dice si se mueve parecido, si
    no o si no hay datos, sin llamarla cobertura). Las de correlación baja
    también se saltean, con su motivo.

    ``threshold`` va de 0 (excluido) a 1, como ``correlation.high_threshold``.
    ``limit_multiple`` / ``hedge_counts_in_day``: solo para el texto de una
    cobertura (``hedge_explanation``; pasar
    ``correlation.max_combined_risk_multiplier`` y
    ``trade_plan.daily_cap_includes_open_risk``).
    ``ValueError`` si un lado no es long/short, un símbolo está vacío o un
    riesgo no es un número finito. Un riesgo negativo cuenta como 0.
    """
    threshold = ensure_finite(threshold, "El umbral de correlación")
    if not 0 < threshold <= 1:
        raise ValueError(f"El umbral de correlación va de 0 a 1 (vino {threshold}).")
    symbol = _norm_symbol(new_symbol)
    side = _norm_side(new_side)
    counted: list[ExposureEntry] = []
    skipped: list[ExposureEntry] = []
    for position in open_positions:
        entry = _entry(
            position,
            symbol,
            side,
            matrix,
            threshold,
            limit_multiple=limit_multiple,
            hedge_counts_in_day=hedge_counts_in_day,
        )
        (counted if entry.counted else skipped).append(entry)
    return Exposure(
        new_symbol=symbol,
        new_side=side,
        threshold=threshold,
        counted=tuple(counted),
        skipped=tuple(skipped),
        matrix_as_of=None if matrix is None else matrix.as_of,
    )


def combined_risk_limit(risk_per_trade_usd: float, multiplier: float) -> float:
    """Tope del riesgo combinado: ``risk_per_trade_usd`` × ``multiplier``.

    Con 50 USD por operación y ``correlation.max_combined_risk_multiplier``
    de 1,5 → 75 USD entre todas las posiciones muy correlacionadas en la misma
    dirección, incluido el plan nuevo. ``ValueError`` si el riesgo es negativo
    o el multiplicador es menor que 1 (el tope nunca puede quedar por debajo
    del riesgo de una operación sola).
    """
    risk = ensure_finite(risk_per_trade_usd, "El riesgo por operación")
    mult = ensure_finite(multiplier, "El multiplicador de riesgo combinado")
    if risk < 0:
        raise ValueError(f"El riesgo por operación no puede ser negativo (vino {risk}).")
    if mult < 1:
        raise ValueError(f"El multiplicador de riesgo combinado tiene que ser >= 1 (vino {mult}).")
    return risk * mult
