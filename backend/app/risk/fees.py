"""Comisiones y funding de futuros.

Las comisiones se cobran sobre el TAMAÑO de la posición (nocional), nunca
sobre el margen, una vez en la entrada y otra en la salida:

- entrada: ``f_in × cantidad × precio_de_entrada``
- salida: ``f_out × cantidad × precio_de_salida``

Las tasas se manejan como fracción (``0.0006`` = 0,06 %).
``fee_rates_from_config`` las saca de la config (``fees.nexo`` en porcentaje
"humano", y ``fees.entry_order`` / ``fees.exit_order`` para elegir taker o
maker).

Funciones:

- ``trade_fees_usd``: comisiones totales de una operación en USD.
- ``gross_pnl_usd`` / ``net_pnl_usd``: resultado sin y con comisiones.
- ``gross_rr`` / ``net_rr``: relación riesgo/beneficio. La neta ("R:R con
  comisiones") es ganancia neta en el objetivo / pérdida neta en el stop,
  ambas con comisiones de entrada y salida; con comisiones > 0 siempre queda
  por debajo de la bruta.
- ``fees_pct_of``: comisiones como % de una base (el riesgo o el objetivo).
- ``funding_events`` / ``guaranteed_funding_events`` / ``funding_cost_usd``:
  costo estimado de funding si la posición queda abierta cierto tiempo.

Signo del funding: con tasa positiva los longs PAGAN y los shorts COBRAN; con
tasa negativa, al revés. ``funding_cost_usd`` devuelve el costo para quien
tiene la posición: positivo = paga, negativo = cobra. La estimación es siempre
conservadora: si la posición paga se cuentan los cobros que podrían caer en
ese tiempo (redondeo hacia arriba) y si cobra, solo los que seguro caen
(redondeo hacia abajo). Por eso el long y el short no siempre suman cero.

Todas las funciones son puras y levantan ``ValueError`` (mensaje en español)
con entradas inválidas.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from app.risk.sizing import (
    ensure_finite,
    ensure_non_negative,
    ensure_positive,
    ensure_rate,
    ensure_side,
    ensure_stop_side,
    pct_to_fraction,
    side_sign,
)

if TYPE_CHECKING:
    from app.config import FeesConfig

# Tolerancia numérica para contar eventos de funding (evita que 24.000000001 h
# cuente un evento extra por error de coma flotante). No es una regla de riesgo.
_EVENT_TOL = 1e-9


def fee_rates_from_config(fees_cfg: FeesConfig) -> tuple[float, float]:
    """Devuelve ``(tasa_entrada, tasa_salida)`` como fracción según la config.

    Usa ``fees.entry_order`` y ``fees.exit_order`` (``"taker"`` o ``"maker"``)
    para elegir la tarifa de ``fees.nexo`` (en porcentaje "humano").
    """
    venue = fees_cfg.nexo

    def rate_for(kind: str, label: str) -> float:
        if kind == "taker":
            pct = venue.taker_pct
        elif kind == "maker":
            pct = venue.maker_pct
        else:
            raise ValueError(f"{label} tiene que ser 'taker' o 'maker' (vino {kind!r}).")
        return ensure_rate(pct_to_fraction(pct), f"La comisión {kind}")

    return (
        rate_for(fees_cfg.entry_order, "El tipo de orden de entrada"),
        rate_for(fees_cfg.exit_order, "El tipo de orden de salida"),
    )


def trade_fees_usd(
    quantity: float,
    entry: float,
    exit_price: float,
    entry_rate: float,
    exit_rate: float,
) -> float:
    """Comisiones totales en USD: entrada sobre ``q·entrada`` + salida sobre ``q·salida``."""
    ensure_positive(quantity, "La cantidad")
    ensure_positive(entry, "La entrada")
    ensure_positive(exit_price, "El precio de salida")
    ensure_rate(entry_rate, "La comisión de entrada")
    ensure_rate(exit_rate, "La comisión de salida")
    return quantity * entry * entry_rate + quantity * exit_price * exit_rate


def gross_pnl_usd(side: str, quantity: float, entry: float, exit_price: float) -> float:
    """Resultado sin comisiones: ``q·(salida - entrada)`` en long, al revés en short."""
    sign = side_sign(side)
    ensure_positive(quantity, "La cantidad")
    ensure_positive(entry, "La entrada")
    ensure_positive(exit_price, "El precio de salida")
    return sign * quantity * (exit_price - entry)


def net_pnl_usd(
    side: str,
    quantity: float,
    entry: float,
    exit_price: float,
    entry_rate: float,
    exit_rate: float,
) -> float:
    """Resultado neto: bruto menos comisiones de entrada y salida."""
    gross = gross_pnl_usd(side, quantity, entry, exit_price)
    return gross - trade_fees_usd(quantity, entry, exit_price, entry_rate, exit_rate)


def _ensure_target_side(side: str, entry: float, target: float) -> None:
    ensure_positive(target, "El objetivo")
    if side == "long" and target <= entry:
        raise ValueError(
            f"En un long el objetivo va arriba de la entrada (entrada {entry}, objetivo {target})."
        )
    if side == "short" and target >= entry:
        raise ValueError(
            f"En un short el objetivo va debajo de la entrada (entrada {entry}, objetivo {target})."
        )


def gross_rr(side: str, entry: float, stop: float, target: float) -> float:
    """R:R sin comisiones: ``|objetivo - entrada| / |entrada - stop|``."""
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    _ensure_target_side(side, entry, target)
    return abs(target - entry) / abs(entry - stop)


def net_rr(
    side: str,
    entry: float,
    stop: float,
    target: float,
    entry_rate: float,
    exit_rate: float,
) -> float:
    """R:R con comisiones: ganancia neta en el objetivo / pérdida neta en el stop.

    No depende de la cantidad (todo escala con ``q``), así que se calcula por
    unidad. Puede dar cero o negativo si el objetivo está tan cerca que las
    comisiones se comen toda la ganancia.
    """
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    _ensure_target_side(side, entry, target)
    ensure_rate(entry_rate, "La comisión de entrada")
    ensure_rate(exit_rate, "La comisión de salida")
    gain = abs(target - entry) - entry_rate * entry - exit_rate * target
    loss = abs(entry - stop) + entry_rate * entry + exit_rate * stop
    return gain / loss


def fees_pct_of(fees_usd: float, base_usd: float) -> float:
    """Comisiones como PORCENTAJE (×100) de una base, p. ej. del riesgo o del objetivo."""
    ensure_non_negative(fees_usd, "Las comisiones")
    ensure_positive(base_usd, "La base")
    return fees_usd / base_usd * 100.0


def funding_events(hours: float, interval_hours: float) -> int:
    """Máximo de cobros de funding que pueden caer en ``hours`` horas: ``ceil(horas / intervalo)``.

    Es el peor caso para quien PAGA: cualquier fracción de intervalo cuenta
    como un cobro entero (no se sabe en qué momento del ciclo se abre la
    posición).
    """
    ensure_non_negative(hours, "Las horas")
    ensure_positive(interval_hours, "El intervalo de funding")
    if hours == 0:
        return 0
    return max(1, math.ceil(hours / interval_hours - _EVENT_TOL))


def guaranteed_funding_events(hours: float, interval_hours: float) -> int:
    """Cobros de funding que caen seguro en ``hours`` horas: ``floor(horas / intervalo)``.

    Es el peor caso para quien COBRA: una fracción de intervalo puede no
    cruzar ningún horario de funding, así que no suma.
    """
    ensure_non_negative(hours, "Las horas")
    ensure_positive(interval_hours, "El intervalo de funding")
    return math.floor(hours / interval_hours + _EVENT_TOL)


def funding_cost_usd(
    side: str,
    size_usd: float,
    funding_rate: float,
    hours: float,
    interval_hours: float,
) -> float:
    """Costo estimado de funding en USD para quien tiene la posición.

    ``funding_rate`` es la fracción por período (como la da el exchange,
    ``0.0001`` = 0,01 %). Signo: positivo = la posición paga, negativo =
    cobra. Con tasa positiva el long paga y el short cobra; con tasa negativa,
    al revés. Se calcula sobre el tamaño (nocional). Conservador: si paga usa
    ``funding_events`` (todos los cobros que podrían caer) y si cobra,
    ``guaranteed_funding_events`` (solo los seguros; con menos de un intervalo
    no se acredita nada).
    """
    sign = side_sign(side)
    ensure_positive(size_usd, "El tamaño")
    rate = ensure_finite(funding_rate, "La tasa de funding")
    per_event = sign * size_usd * rate
    if per_event > 0:
        events = funding_events(hours, interval_hours)
    else:
        events = guaranteed_funding_events(hours, interval_hours)
    return per_event * events
