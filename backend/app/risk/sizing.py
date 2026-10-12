"""Tamaño de posición a partir del riesgo en USD.

Principio: **riesgo en dólares primero, apalancamiento después**. Se fija
cuánto se puede perder si toca el stop (``risk_usd``) y cuánto margen se pone;
el tamaño sale del riesgo y de la distancia al stop, y el apalancamiento sale
de dividir tamaño por margen. El apalancamiento es un RESULTADO, nunca una
entrada.

Convenciones:

- Tasas y distancias como fracción (``0.01`` = 1 %, ``0.0006`` = 0,06 %). La
  config usa porcentajes "humanos" (``0.06`` = 0,06 %); ``pct_to_fraction``
  hace la conversión.
- Montos en USD (USDT). ``size_usd`` es el nocional de la posición al precio
  de entrada (cantidad × entrada), que es sobre lo que se cobran las
  comisiones (nunca sobre el margen).

Fórmulas (``q`` = cantidad en moneda base, ``E`` = entrada, ``S`` = stop,
``d = |E - S|``):

- ``stop_distance_pct = d / E`` (fracción).
- Sin comisiones: ``size = riesgo / (d / E)`` (la fórmula del usuario).
- Con comisiones dentro del riesgo: la pérdida real al tocar el stop es
  ``q·d + f_in·q·E + f_out·q·S`` (movimiento del precio + comisión de entrada
  sobre el nocional de entrada + comisión de salida sobre el nocional de
  salida). Igualándola al riesgo queda, en forma exacta,
  ``q = riesgo / (d + f_in·E + f_out·S)`` y
  ``size = q·E = riesgo / (d/E + f_in + f_out·S/E)``.
  La aproximación habitual ``riesgo / (d/E + f_in + f_out)`` ignora que la
  salida se cobra al precio del stop: en un long sobreestima un poco la
  comisión de salida (el stop está debajo de la entrada) y en un short la
  subestima. Acá se usa siempre la exacta.
- ``leverage = size / margen`` y ``margen = size / leverage``.

``size_position`` junta todo y devuelve un ``PositionSizing`` con la pérdida
real al tocar el stop (precio + comisiones). Con
``fees_included_in_risk=True`` esa pérdida es exactamente ``risk_usd``; con
``False`` el tamaño sale de la fórmula pura y la pérdida es ``risk_usd`` más
las comisiones.

Este módulo también tiene los validadores que reutilizan ``fees`` y
``liquidation``. Todas las funciones son puras y levantan ``ValueError``
(mensaje en español) con entradas inválidas: precios, tamaños o riesgo no
positivos (o no finitos), stop igual a la entrada o del lado equivocado,
apalancamiento <= 0, tasas negativas o >= 100 %.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Literal

Side = Literal["long", "short"]

_SIDES: tuple[str, ...] = ("long", "short")


# --- conversiones -------------------------------------------------------------


def pct_to_fraction(pct: float) -> float:
    """Convierte un porcentaje "humano" de la config a fracción (0.06 -> 0.0006)."""
    return pct / 100.0


def fraction_to_pct(fraction: float) -> float:
    """Convierte una fracción a porcentaje "humano" (0.0006 -> 0.06)."""
    return fraction * 100.0


# --- validadores (compartidos con fees y liquidation) ---------------------------


def ensure_finite(value: float, name: str) -> float:
    """Devuelve ``value`` si es un número finito; si no, ``ValueError``."""
    if not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} tiene que ser un número finito (vino {value!r}).")
    return float(value)


def ensure_positive(value: float, name: str) -> float:
    """Devuelve ``value`` si es finito y mayor que cero; si no, ``ValueError``."""
    value = ensure_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} tiene que ser mayor que cero (vino {value}).")
    return value


def ensure_non_negative(value: float, name: str) -> float:
    """Devuelve ``value`` si es finito y >= 0; si no, ``ValueError``."""
    value = ensure_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} no puede ser negativo (vino {value}).")
    return value


def ensure_rate(rate: float, name: str) -> float:
    """Valida una tasa como fracción: ``0 <= rate < 1`` (0,06 % = 0.0006)."""
    rate = ensure_non_negative(rate, name)
    if rate >= 1:
        raise ValueError(
            f"{name} tiene que ser una fracción menor que 1 (vino {rate}). "
            "¿Pasaste un porcentaje? Usá pct_to_fraction."
        )
    return rate


def ensure_side(side: str) -> Side:
    """Valida que el lado sea ``"long"`` o ``"short"``."""
    if side not in _SIDES:
        raise ValueError(f"El lado tiene que ser 'long' o 'short' (vino {side!r}).")
    return side  # type: ignore[return-value]


def side_sign(side: str) -> int:
    """``+1`` para long, ``-1`` para short (signo de la ganancia cuando sube el precio)."""
    return 1 if ensure_side(side) == "long" else -1


def ensure_stop_side(side: str, entry: float, stop: float) -> None:
    """Valida precios positivos y que el stop esté del lado perdedor.

    En un long el stop va debajo de la entrada; en un short, arriba.
    """
    ensure_side(side)
    ensure_positive(entry, "La entrada")
    ensure_positive(stop, "El stop")
    if stop == entry:
        raise ValueError("El stop no puede ser igual a la entrada.")
    if side == "long" and stop > entry:
        raise ValueError(
            f"En un long el stop va debajo de la entrada (entrada {entry}, stop {stop})."
        )
    if side == "short" and stop < entry:
        raise ValueError(
            f"En un short el stop va arriba de la entrada (entrada {entry}, stop {stop})."
        )


# --- fórmulas básicas -----------------------------------------------------------


def stop_distance_pct(entry: float, stop: float) -> float:
    """Distancia al stop como fracción de la entrada: ``|entrada - stop| / entrada``."""
    ensure_positive(entry, "La entrada")
    ensure_positive(stop, "El stop")
    if stop == entry:
        raise ValueError("El stop no puede ser igual a la entrada.")
    return abs(entry - stop) / entry


def position_size_usd(
    risk_usd: float,
    entry: float,
    stop: float,
    entry_fee_rate: float = 0.0,
    exit_fee_rate: float = 0.0,
) -> float:
    """Tamaño (nocional en USD) para que la pérdida al tocar el stop sea ``risk_usd``.

    Con las dos tasas en cero es la fórmula del usuario,
    ``riesgo / distancia_al_stop``. Con tasas, la pérdida incluye la comisión
    de entrada (sobre ``q·entrada``) y la de salida (sobre ``q·stop``) y se
    resuelve en forma exacta: ``riesgo / (d/E + f_in + f_out·S/E)``.
    """
    ensure_positive(risk_usd, "El riesgo")
    distance = stop_distance_pct(entry, stop)
    ensure_rate(entry_fee_rate, "La comisión de entrada")
    ensure_rate(exit_fee_rate, "La comisión de salida")
    return risk_usd / (distance + entry_fee_rate + exit_fee_rate * stop / entry)


def leverage_for(size_usd: float, margin_usd: float) -> float:
    """Apalancamiento resultante: ``tamaño / margen``."""
    ensure_positive(size_usd, "El tamaño")
    ensure_positive(margin_usd, "El margen")
    return size_usd / margin_usd


def margin_for_leverage(size_usd: float, leverage: float) -> float:
    """Margen necesario para un tamaño y un apalancamiento: ``tamaño / apalancamiento``."""
    ensure_positive(size_usd, "El tamaño")
    ensure_positive(leverage, "El apalancamiento")
    return size_usd / leverage


def quantity_for(size_usd: float, price: float) -> float:
    """Cantidad en moneda base para un nocional en USD: ``tamaño / precio``."""
    ensure_positive(size_usd, "El tamaño")
    ensure_positive(price, "El precio")
    return size_usd / price


# --- sizing completo --------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizing:
    """Resultado del sizing por riesgo en USD.

    ``stop_distance_pct`` es fracción. ``loss_at_stop_usd`` es la pérdida
    máxima real si toca el stop: movimiento del precio
    (``price_loss_at_stop_usd``) más comisiones de entrada y salida
    (``fees_at_stop_usd``).
    """

    side: Side
    entry: float
    stop: float
    risk_usd: float
    fees_included_in_risk: bool
    stop_distance_pct: float
    size_usd: float
    quantity: float
    margin_usd: float
    leverage: float
    price_loss_at_stop_usd: float
    fees_at_stop_usd: float
    loss_at_stop_usd: float


def size_position(
    entry: float,
    stop: float,
    side: str,
    risk_usd: float,
    margin_usd: float,
    entry_fee_rate: float,
    exit_fee_rate: float,
    fees_included_in_risk: bool,
) -> PositionSizing:
    """Calcula tamaño, cantidad y apalancamiento a partir del riesgo en USD.

    - ``fees_included_in_risk=True``: el tamaño se achica para que la pérdida
      al tocar el stop, comisiones incluidas, sea exactamente ``risk_usd``.
    - ``fees_included_in_risk=False``: tamaño = ``risk_usd / distancia`` y las
      comisiones se suman aparte (la pérdida real queda por encima del riesgo).

    El apalancamiento sale de ``tamaño / margen``; acá no se lo limita (los
    topes los aplican las reglas, con los valores de la config).
    """
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    ensure_positive(risk_usd, "El riesgo")
    ensure_positive(margin_usd, "El margen")
    ensure_rate(entry_fee_rate, "La comisión de entrada")
    ensure_rate(exit_fee_rate, "La comisión de salida")

    if fees_included_in_risk:
        size = position_size_usd(risk_usd, entry, stop, entry_fee_rate, exit_fee_rate)
    else:
        size = position_size_usd(risk_usd, entry, stop)

    quantity = size / entry
    price_loss = quantity * abs(entry - stop)
    fees = quantity * entry * entry_fee_rate + quantity * stop * exit_fee_rate
    return PositionSizing(
        side=side,
        entry=float(entry),
        stop=float(stop),
        risk_usd=float(risk_usd),
        fees_included_in_risk=fees_included_in_risk,
        stop_distance_pct=stop_distance_pct(entry, stop),
        size_usd=size,
        quantity=quantity,
        margin_usd=float(margin_usd),
        leverage=size / margin_usd,
        price_loss_at_stop_usd=price_loss,
        fees_at_stop_usd=fees,
        loss_at_stop_usd=price_loss + fees,
    )
