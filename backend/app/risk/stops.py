"""Stop-loss por estructura (soportes y resistencias).

Regla (parámetros de ``config.yaml > trade_plan``):

- **Long**: el soporte más cercano *debajo* de la entrada entre los timeframes
  ``stop_timeframes`` (15m y 1h), menos ``stop_atr_buffer`` × ATR(15m).
- **Short**: espejado, con la resistencia más cercana *arriba* de la entrada,
  más el colchón.
- **Niveles pegados al precio** (decisión del usuario): los que están más cerca
  de la entrada que ``min_stop_distance_atr`` × ATR(15m) (lo que se mueve una
  vela normal) se saltean y se usa el siguiente. Si no, el stop queda dentro del
  ruido normal del mercado y, para arriesgar lo mismo en USD, la posición y el
  apalancamiento se disparan.

``structural_stop`` devuelve un ``StructuralStop`` (con los niveles salteados,
para explicarlo en la UI) o un ``StopNotFound`` con el motivo en castellano. La
usan el plan de operación y el ejemplo de la consola: la misma regla en los dos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.indicators.levels import SRResult

Side = Literal["long", "short"]


@dataclass(frozen=True)
class SkippedLevel:
    """Nivel descartado para el stop por estar demasiado cerca de la entrada."""

    price: float
    timeframe: str
    distance_atr: float  # distancia a la entrada medida en ATR


@dataclass(frozen=True)
class StructuralStop:
    side: Side
    entry: float
    level_price: float
    level_timeframe: str
    atr: float
    buffer_atr: float  # colchón usado, en ATR (trade_plan.stop_atr_buffer)
    min_distance_atr: float  # trade_plan.min_stop_distance_atr
    stop: float
    skipped: tuple[SkippedLevel, ...]  # del más cercano al más lejano

    @property
    def level_kind(self) -> str:
        return "soporte" if self.side == "long" else "resistencia"

    @property
    def stop_distance_pct(self) -> float:
        """Distancia entrada-stop como fracción de la entrada."""
        return abs(self.entry - self.stop) / self.entry

    @property
    def level_distance_atr(self) -> float:
        return abs(self.entry - self.level_price) / self.atr


@dataclass(frozen=True)
class StopNotFound:
    side: Side
    reason: str  # en castellano, para mostrar
    skipped: tuple[SkippedLevel, ...] = ()


def structural_stop(
    side: Side,
    entry: float,
    levels_by_timeframe: Mapping[str, SRResult | None],
    atr: float,
    buffer_atr_mult: float,
    min_distance_atr: float,
) -> StructuralStop | StopNotFound:
    """Stop por estructura para ``side`` (ver el docstring del módulo).

    ``levels_by_timeframe`` mapea timeframe → ``SRResult`` en el orden de
    preferencia (ante dos niveles al mismo precio gana el primer timeframe).
    ``atr`` es el ATR del timeframe de entrada. Lanza ``ValueError`` si
    ``entry`` o ``atr`` no son positivos o si los multiplicadores son negativos.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side tiene que ser 'long' o 'short' (llegó {side!r})")
    if not _positive(entry):
        raise ValueError(f"La entrada tiene que ser un número positivo (llegó {entry!r})")
    if not _positive(atr):
        raise ValueError(f"El ATR tiene que ser un número positivo (llegó {atr!r})")
    if buffer_atr_mult < 0 or min_distance_atr < 0:
        raise ValueError("stop_atr_buffer y min_stop_distance_atr no pueden ser negativos")

    word = "soporte" if side == "long" else "resistencia"
    where = "debajo" if side == "long" else "arriba"
    timeframes = [tf for tf, sr in levels_by_timeframe.items() if sr is not None]

    best: tuple[float, str] | None = None
    skipped: list[SkippedLevel] = []
    for tf in timeframes:
        sr = levels_by_timeframe[tf]
        assert sr is not None
        for cluster in sr.clusters:
            price = cluster.price
            if (side == "long" and price >= entry) or (side == "short" and price <= entry):
                continue
            distance_atr = abs(entry - price) / atr
            if distance_atr < min_distance_atr:
                skipped.append(SkippedLevel(price, tf, distance_atr))
                continue
            closer = best is None or (price > best[0] if side == "long" else price < best[0])
            if closer:
                best = (price, tf)

    skipped.sort(key=lambda s: s.distance_atr)
    if best is None:
        tfs = "/".join(timeframes) or "ningún timeframe"
        if skipped:
            reason = (
                f"Todos los {word}s {where} del precio en {tfs} están a menos de "
                f"{_fmt(min_distance_atr)} ATR: no hay un nivel para un stop razonable."
            )
        else:
            reason = f"No hay {word}s {where} del precio en {tfs}."
        return StopNotFound(side, reason, tuple(skipped))

    level_price, level_tf = best
    buffer = buffer_atr_mult * atr
    stop = level_price - buffer if side == "long" else level_price + buffer
    if stop <= 0:
        return StopNotFound(
            side, f"El stop calculado ({stop:.8g}) no es un precio válido.", tuple(skipped)
        )
    return StructuralStop(
        side=side,
        entry=entry,
        level_price=level_price,
        level_timeframe=level_tf,
        atr=atr,
        buffer_atr=buffer_atr_mult,
        min_distance_atr=min_distance_atr,
        stop=stop,
        skipped=tuple(skipped),
    )


def _positive(x: float) -> bool:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def _fmt(x: float) -> str:
    return f"{x:g}".replace(".", ",")
