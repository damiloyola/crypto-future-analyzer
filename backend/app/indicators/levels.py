"""Soportes y resistencias a partir de pivotes (swing highs/lows).

``support_resistance`` arma los niveles en cuatro pasos: pivotes → máximo y
mínimo del día anterior → agrupación por cercanía (fracción del ATR) → los más
cercanos arriba y abajo del precio. Recibe solo velas **cerradas** (lo decide
quien llama) y todos los parámetros vienen de ``indicators.levels`` en
``config.yaml``; este módulo no lee configuración.

Pivotes (``find_pivots``)
    Hay *swing high* en la vela *t* si su ``high`` es **estrictamente mayor**
    que los de las ``left`` velas anteriores y **mayor o igual** que los de las
    ``right`` velas siguientes; el *swing low* es el espejo con ``low``. Con un
    techo (o piso) plato, es decir varios extremos iguales seguidos, el pivote
    es la **primera** vela: las siguientes no superan estrictamente a la de al
    lado. Una vela sin ``left`` velas antes o sin ``right`` después no puede ser
    pivote, así que el pivote de *t* recién se conoce al cerrar la vela
    *t + right* (``Pivot.confirmed_at``) y agregar velas nuevas nunca cambia un
    pivote ya confirmado: cero look-ahead. En un backtest, ``pivots_known_at``
    filtra los pivotes que se conocían en un momento dado. Una ventana con
    ``NaN`` no genera pivote. Una misma vela puede ser *swing high* y *swing
    low* a la vez (vela envolvente); en la lista sale primero el ``high``.

Agrupación (``cluster_levels``)
    Ordena los precios de menor a mayor y los recorre una vez: un precio se suma
    al grupo actual si está a ``mult × ATR`` o menos de la **media** del grupo;
    si no, abre un grupo nuevo. El nivel representativo es la media de sus
    miembros. Al comparar contra la media (y no contra el último miembro) una
    cadena de precios separados de a poco **no** se junta en un único nivel
    gigante: con tolerancia 1, los precios 100; 100,9; 101,8 dan dos niveles
    (100,45 y 101,8). La zona de un grupo (``zone_low``–``zone_high``) puede
    pasar un poco de la tolerancia porque la media se corre al sumar miembros,
    pero dos niveles consecutivos siempre quedan separados por **más** que la
    tolerancia. Si el ATR es ``NaN``, ``None`` o ``<= 0`` solo se juntan
    precios idénticos.

    ``touches`` cuenta toques distintos: un toque es un par (vela, precio). Si
    el máximo del día anterior es la misma vela y el mismo precio que un
    pivote, cuenta como un toque (con las dos fuentes). ``last_touch`` es la
    vela más reciente del grupo. Las fuentes posibles son ``pivot_high``,
    ``pivot_low``, ``prev_day_high`` y ``prev_day_low``.

Día anterior (``previous_day_high_low``)
    Cada vela va al día calendario de su ``open_time`` en ``tz`` (como el VWAP
    diario). "Hoy" es el último día con velas, o el día de ``as_of`` si se
    pasa; el día anterior es el día calendario previo a "hoy". Devuelve
    ``None`` si ese día no tiene velas (menos de dos días de datos o un hueco).
    Pasar ``as_of`` (el momento actual) evita un borde: con solo velas
    cerradas, justo después de la medianoche la última vela cerrada todavía es
    del día que terminó y sin ``as_of`` se tomaría el anterior a ese.

Selección (``nearest_levels``)
    Relativo al precio de referencia: debajo es ``soporte`` (ordenados del más
    cercano al más lejano, o sea de mayor a menor precio) y arriba es
    ``resistencia`` (de menor a mayor), hasta ``n`` de cada lado. Un pivote
    alto que quedó debajo del precio pasa a ser soporte (cambio de rol). Un
    nivel exactamente igual al precio se descarta: no sirve como stop ni como
    objetivo y así nunca aparece de los dos lados. ``distance_pct`` es
    ``(nivel − precio) / precio × 100``: negativo en soportes y positivo en
    resistencias.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, tzinfo
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

PivotKind = Literal["high", "low"]
LevelKind = Literal["soporte", "resistencia"]

SOURCE_PIVOT_HIGH = "pivot_high"
SOURCE_PIVOT_LOW = "pivot_low"
SOURCE_PREV_DAY_HIGH = "prev_day_high"
SOURCE_PREV_DAY_LOW = "prev_day_low"


# --- modelos -----------------------------------------------------------------


@dataclass(frozen=True)
class LevelPoint:
    """Un precio candidato a nivel, con de dónde salió y en qué vela."""

    price: float
    source: str  # ver las constantes SOURCE_*
    time: datetime | None = None  # open_time de la vela que marcó el precio


@dataclass(frozen=True)
class Pivot:
    """Swing high o swing low confirmado."""

    time: datetime  # open_time de la vela del pivote
    price: float  # high (swing high) o low (swing low) de esa vela
    kind: PivotKind
    index: int  # posición de la vela en el DataFrame
    confirmed_at: datetime  # open_time de la vela index + right: se conoce al cerrar esa vela

    def to_point(self) -> LevelPoint:
        """El pivote como candidato a nivel (fuente ``pivot_high`` o ``pivot_low``)."""
        source = SOURCE_PIVOT_HIGH if self.kind == "high" else SOURCE_PIVOT_LOW
        return LevelPoint(price=self.price, source=source, time=self.time)


@dataclass(frozen=True)
class LevelCluster:
    """Grupo de precios cercanos, todavía sin rol (soporte/resistencia)."""

    price: float  # media de los miembros
    touches: int  # toques distintos (vela, precio)
    sources: frozenset[str]
    last_touch: datetime | None
    zone_low: float  # precio mínimo del grupo
    zone_high: float  # precio máximo del grupo


@dataclass(frozen=True)
class Level:
    """Nivel con rol decidido respecto de un precio de referencia."""

    price: float
    kind: LevelKind
    touches: int
    sources: frozenset[str]
    last_touch: datetime | None
    distance_pct: float  # (nivel − referencia) / referencia × 100, con signo
    zone_low: float
    zone_high: float


class PreviousDayRange(NamedTuple):
    """Máximo y mínimo del día anterior (se puede desarmar como tupla)."""

    high: float
    low: float
    day: date  # día calendario en la zona horaria pedida


@dataclass(frozen=True)
class SRResult:
    """Resultado de ``support_resistance``."""

    supports: tuple[Level, ...]  # de mayor a menor precio (el más cercano primero)
    resistances: tuple[Level, ...]  # de menor a mayor precio (el más cercano primero)
    prev_day_high: float | None
    prev_day_low: float | None
    pivots_used: tuple[Pivot, ...]  # todos los pivotes confirmados de las velas recibidas
    prev_day: date | None  # qué día se tomó como "día anterior"
    reference_price: float
    cluster_tolerance: float  # mult × ATR usado al agrupar (0 = solo precios idénticos)
    # Todos los grupos de niveles (no solo los más cercanos), de menor a mayor
    # precio: el stop puede necesitar saltear los que están pegados al precio.
    clusters: tuple[LevelCluster, ...] = ()


# --- pivotes -----------------------------------------------------------------


def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Pivot]:
    """Swing highs y lows de ``df`` (reglas en el docstring del módulo).

    Devuelve los pivotes ordenados por vela; dentro de una misma vela, primero
    el ``high``. Con menos de ``left + right + 1`` velas no hay pivotes.
    """
    left = _require_int(left, "left", 1)
    right = _require_int(right, "right", 1)
    _require_columns(df, ("high", "low"))
    span = left + right + 1
    if len(df) < span:
        return []

    found: list[tuple[int, int, Pivot]] = []
    times = df.index
    for order, (kind, column) in enumerate((("high", "high"), ("low", "low"))):
        values = df[column].to_numpy(dtype="float64")
        windows = sliding_window_view(values, span)
        center = windows[:, left]
        # np.max/np.min propagan NaN y cualquier comparación con NaN da False.
        if kind == "high":
            before = windows[:, :left].max(axis=1)
            after = windows[:, left + 1 :].max(axis=1)
            mask = (center > before) & (center >= after)
        else:
            before = windows[:, :left].min(axis=1)
            after = windows[:, left + 1 :].min(axis=1)
            mask = (center < before) & (center <= after)
        for offset in np.flatnonzero(mask):
            i = int(offset) + left
            pivot = Pivot(
                time=times[i],
                price=float(values[i]),
                kind=kind,
                index=i,
                confirmed_at=times[i + right],
            )
            found.append((i, order, pivot))
    found.sort(key=lambda item: (item[0], item[1]))
    return [pivot for _, _, pivot in found]


def pivots_known_at(pivots: Iterable[Pivot], time: datetime) -> list[Pivot]:
    """Pivotes ya confirmados en ``time`` (``confirmed_at <= time``).

    ``time`` sigue la misma convención que el índice de velas: es el
    ``open_time`` de la última vela cerrada que se puede usar.
    """
    return [p for p in pivots if p.confirmed_at <= time]


# --- agrupación --------------------------------------------------------------


def cluster_tolerance(atr: float | None, mult: float) -> float:
    """Distancia máxima a la media del grupo: ``mult × atr``, o 0 si el ATR no sirve."""
    if (
        isinstance(mult, bool)
        or not isinstance(mult, (int, float, np.number))
        or not math.isfinite(mult)
        or mult < 0
    ):
        raise ValueError(f"mult tiene que ser un número >= 0 (llegó {mult!r})")
    if atr is None:
        return 0.0
    atr_value = float(atr)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return 0.0
    return float(mult) * atr_value


def cluster_levels(
    points: Iterable[LevelPoint], atr: float | None, mult: float = 0.5
) -> list[LevelCluster]:
    """Agrupa precios a ``mult × atr`` o menos de la media del grupo.

    Devuelve los grupos ordenados de menor a mayor precio. Ignora precios no
    finitos. Ver el docstring del módulo para el criterio (media, no cadena).
    """
    tolerance = cluster_tolerance(atr, mult)
    ordered = sorted(
        (p for p in points if math.isfinite(float(p.price))), key=lambda p: float(p.price)
    )
    clusters: list[LevelCluster] = []
    members: list[LevelPoint] = []
    mean = math.nan
    for point in ordered:
        price = float(point.price)
        # El chequeo de igualdad con el último miembro es matemáticamente
        # redundante, pero evita que el redondeo de la media separe duplicados.
        if members and (price == float(members[-1].price) or abs(price - mean) <= tolerance):
            members.append(point)
            mean = math.fsum(float(m.price) for m in members) / len(members)
        else:
            if members:
                clusters.append(_make_cluster(members))
            members = [point]
            mean = price
    if members:
        clusters.append(_make_cluster(members))
    return clusters


def _make_cluster(members: list[LevelPoint]) -> LevelCluster:
    prices = [float(m.price) for m in members]
    low, high = min(prices), max(prices)
    # Todos iguales: el precio exacto (la media en float podría correrse un ulp).
    price = low if low == high else min(max(math.fsum(prices) / len(prices), low), high)
    timed = {(m.time, float(m.price)) for m in members if m.time is not None}
    untimed = sum(1 for m in members if m.time is None)
    times = [m.time for m in members if m.time is not None]
    return LevelCluster(
        price=price,
        touches=len(timed) + untimed,
        sources=frozenset(m.source for m in members),
        last_touch=max(times) if times else None,
        zone_low=low,
        zone_high=high,
    )


# --- día anterior ------------------------------------------------------------


def previous_day_high_low(
    df: pd.DataFrame, tz: str | tzinfo = "UTC", *, as_of: datetime | None = None
) -> PreviousDayRange | None:
    """Máximo y mínimo del día calendario anterior a "hoy" en ``tz``.

    "Hoy" es el último día con velas, o el de ``as_of`` si se pasa. ``None`` si
    el día anterior no tiene velas.
    """
    found = _previous_day_extremes(df, tz, as_of)
    return None if found is None else found[0]


def _previous_day_extremes(
    df: pd.DataFrame, tz: str | tzinfo, as_of: datetime | None
) -> tuple[PreviousDayRange, datetime, datetime] | None:
    """Como ``previous_day_high_low`` pero también devuelve la vela de cada extremo."""
    _require_columns(df, ("high", "low"))
    if len(df) == 0:
        return None
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Las velas tienen que tener un índice de fechas (DatetimeIndex)")
    if index.tz is None:
        index = index.tz_localize("UTC")
    days = index.tz_convert(tz).tz_localize(None).normalize()
    if as_of is None:
        today = days.max()
    else:
        now = pd.Timestamp(as_of)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        today = now.tz_convert(tz).tz_localize(None).normalize()
    previous = today - pd.Timedelta(days=1)
    mask = np.asarray(days == previous)
    if not mask.any():
        return None
    highs = df["high"].to_numpy(dtype="float64")[mask]
    lows = df["low"].to_numpy(dtype="float64")[mask]
    if np.isnan(highs).all() or np.isnan(lows).all():
        return None
    times = index[mask]
    high = float(np.nanmax(highs))
    low = float(np.nanmin(lows))
    # Si el extremo se repite, la vela más reciente (es el "último toque").
    high_time = times[int(np.flatnonzero(highs == high)[-1])]
    low_time = times[int(np.flatnonzero(lows == low)[-1])]
    return PreviousDayRange(high=high, low=low, day=previous.date()), high_time, low_time


# --- selección ---------------------------------------------------------------


def nearest_levels(
    levels: Iterable[LevelCluster | Level], price: float, n: int = 3
) -> tuple[list[Level], list[Level]]:
    """Hasta ``n`` soportes debajo de ``price`` y ``n`` resistencias arriba.

    Devuelve ``(soportes, resistencias)``, cada lista con el más cercano
    primero. Un nivel exactamente en ``price`` se descarta.
    """
    reference = _require_price(price)
    n = _require_int(n, "n", 0)
    below: list[LevelCluster | Level] = []
    above: list[LevelCluster | Level] = []
    for level in levels:
        if level.price < reference:
            below.append(level)
        elif level.price > reference:
            above.append(level)
    below.sort(key=lambda lv: lv.price, reverse=True)
    above.sort(key=lambda lv: lv.price)
    supports = [_with_role(lv, "soporte", reference) for lv in below[:n]]
    resistances = [_with_role(lv, "resistencia", reference) for lv in above[:n]]
    return supports, resistances


def _with_role(level: LevelCluster | Level, kind: LevelKind, reference: float) -> Level:
    return Level(
        price=level.price,
        kind=kind,
        touches=level.touches,
        sources=level.sources,
        last_touch=level.last_touch,
        distance_pct=(level.price - reference) / reference * 100.0,
        zone_low=level.zone_low,
        zone_high=level.zone_high,
    )


# --- todo junto --------------------------------------------------------------


def support_resistance(
    df: pd.DataFrame,
    price: float,
    atr: float | None,
    left: int,
    right: int,
    cluster_atr_mult: float,
    max_levels_each_side: int,
    include_previous_day: bool = True,
    tz: str | tzinfo = "UTC",
    *,
    as_of: datetime | None = None,
) -> SRResult:
    """Soportes y resistencias de ``df`` alrededor de ``price``.

    Pivotes de todas las velas recibidas, más el máximo y mínimo del día
    anterior si ``include_previous_day``; se agrupan con ``cluster_atr_mult ×
    atr`` y se eligen los ``max_levels_each_side`` más cercanos de cada lado.
    Los parámetros salen de ``indicators.levels``; ``tz`` es
    ``app.market_timezone`` y ``as_of`` el momento actual (opcional).
    """
    reference = _require_price(price)
    pivots = find_pivots(df, left, right)
    points = [p.to_point() for p in pivots]

    prev: PreviousDayRange | None = None
    if include_previous_day:
        found = _previous_day_extremes(df, tz, as_of)
        if found is not None:
            prev, high_time, low_time = found
            points.append(LevelPoint(prev.high, SOURCE_PREV_DAY_HIGH, high_time))
            points.append(LevelPoint(prev.low, SOURCE_PREV_DAY_LOW, low_time))

    tolerance = cluster_tolerance(atr, cluster_atr_mult)
    clusters = cluster_levels(points, atr, cluster_atr_mult)
    supports, resistances = nearest_levels(clusters, reference, max_levels_each_side)
    return SRResult(
        supports=tuple(supports),
        resistances=tuple(resistances),
        prev_day_high=None if prev is None else prev.high,
        prev_day_low=None if prev is None else prev.low,
        pivots_used=tuple(pivots),
        prev_day=None if prev is None else prev.day,
        reference_price=reference,
        cluster_tolerance=tolerance,
        clusters=tuple(sorted(clusters, key=lambda c: c.price)),
    )


# --- validaciones ------------------------------------------------------------


def _require_int(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(
            f"{name} tiene que ser un entero mayor o igual a {minimum} (llegó {value!r})"
        )
    return int(value)


def _require_price(price: float) -> float:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"El precio de referencia tiene que ser un número positivo (llegó {price!r})"
        )
    return value


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en las velas: {missing}")
