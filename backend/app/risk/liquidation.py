"""Precio de liquidación (margen aislado) y chequeo contra el stop.

Modelo aislado (el que usan las reglas, por ser el conservador). Con
``q = tamaño / entrada``, margen ``M``, margen de mantenimiento ``mmr`` y
comisión de salida ``f`` (ambos como fracción del nocional al precio ``P``),
la posición se liquida cuando el patrimonio de la posición iguala al margen
de mantenimiento:

    patrimonio = M + pnl(P) - f·q·P  ==  mmr·q·P

Despejando ``P`` en forma exacta:

- Long:  ``liq = E × (1 - M/tamaño) / (1 - mmr - f)``
- Short: ``liq = E × (1 + M/tamaño) / (1 + mmr + f)``

Con ``f = 0`` y ``M/tamaño = 1/L`` el long queda ``E × (1 - 1/L) / (1 - mmr)``,
que se parece a la aproximación ``E × (1 - 1/L + mmr)`` (la diferencia es
``E·mmr·(mmr - 1/L)/(1 - mmr)``, de orden ``mmr/L``). Si el long da
``liq <= 0`` (margen >= tamaño, o sea 1x o menos) no se puede liquidar y se
devuelve ``None``. El short siempre tiene precio de liquidación. Si
``M/tamaño <= mmr + f`` la liquidación cae del lado ganador de la entrada: la
posición nacería liquidada (``check_liquidation_vs_stop`` lo marca como no ok).

``wallet_liquidation_price`` es el modelo "estilo Nexo": en Nexo el colateral
es toda la billetera de futuros, así que se usan las mismas ecuaciones con el
saldo de la billetera (cargado a mano) en lugar del margen. Es solo
informativo: con más saldo la liquidación queda más lejos, pero las reglas
usan siempre el aislado.

Chequeo y sugerencias:

- ``check_liquidation_vs_stop``: la liquidación tiene que quedar más allá del
  stop (del lado perdedor) y a una distancia de la entrada de al menos
  ``min_ratio`` veces la distancia al stop (``trade_plan.liquidation.
  min_distance_ratio`` en la config).
- ``min_margin_for_safe_liquidation``: margen mínimo para cumplirlo sin tocar
  el tamaño ("subir el margen manteniendo el riesgo"). Con ``d = |E - S|`` y
  ``r = min_ratio``:
  long ``M >= tamaño × [1 - (1 - r·d/E)(1 - mmr - f)]``;
  short ``M >= tamaño × [(1 + r·d/E)(1 + mmr + f) - 1]``.
  Si en un long ``r·d >= E`` hace falta que no se pueda liquidar (``M >=
  tamaño``).
- ``max_size_for_safe_liquidation``: tamaño máximo para cumplirlo sin tocar
  el margen ("achicar el tamaño"), despejando lo mismo al revés.

Tasas como fracción; ``mmr_from_config`` convierte
``liquidation.maintenance_margin_pct`` de la config. Funciones puras que
levantan ``ValueError`` (mensaje en español) con entradas inválidas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.risk.sizing import (
    Side,
    ensure_positive,
    ensure_rate,
    ensure_side,
    ensure_stop_side,
    pct_to_fraction,
)

if TYPE_CHECKING:
    from app.config import LiquidationConfig

# Tolerancia numérica relativa para comparar el ratio contra el mínimo: el
# margen o tamaño que devuelven las funciones de sugerencia cae justo en el
# borde y la coma flotante no debe hacerlo fallar. No es una regla de riesgo.
_REL_TOL = 1e-9


def mmr_from_config(liquidation_cfg: LiquidationConfig) -> float:
    """Margen de mantenimiento como fracción (``0.5`` en la config -> ``0.005``)."""
    return ensure_rate(
        pct_to_fraction(liquidation_cfg.maintenance_margin_pct), "El margen de mantenimiento"
    )


def _ensure_mmr_and_fee(mmr: float, exit_fee_rate: float) -> None:
    ensure_rate(mmr, "El margen de mantenimiento")
    ensure_rate(exit_fee_rate, "La comisión de salida")
    if mmr + exit_fee_rate >= 1:
        raise ValueError(
            "El margen de mantenimiento más la comisión de salida tiene que ser menor que 1 "
            f"(vino {mmr} + {exit_fee_rate})."
        )


def _liquidation_from_ratio(
    side: str,
    entry: float,
    collateral_ratio: float,
    mmr: float,
    exit_fee_rate: float,
) -> float | None:
    """Resuelve ``colateral + pnl - f·q·P == mmr·q·P`` para ``P``.

    ``collateral_ratio`` es colateral / tamaño (``1/L`` en margen aislado).
    """
    side = ensure_side(side)
    ensure_positive(entry, "La entrada")
    _ensure_mmr_and_fee(mmr, exit_fee_rate)
    if side == "long":
        liq = entry * (1.0 - collateral_ratio) / (1.0 - mmr - exit_fee_rate)
        return liq if liq > 0 else None
    return entry * (1.0 + collateral_ratio) / (1.0 + mmr + exit_fee_rate)


def liquidation_price(
    side: str,
    entry: float,
    size_usd: float,
    margin_usd: float,
    mmr: float,
    exit_fee_rate: float = 0.0,
) -> float | None:
    """Precio de liquidación con margen aislado. ``None`` si el long no se puede liquidar."""
    ensure_positive(size_usd, "El tamaño")
    ensure_positive(margin_usd, "El margen")
    return _liquidation_from_ratio(side, entry, margin_usd / size_usd, mmr, exit_fee_rate)


def liquidation_price_for_leverage(
    side: str,
    entry: float,
    leverage: float,
    mmr: float,
    exit_fee_rate: float = 0.0,
) -> float | None:
    """Igual que ``liquidation_price`` pero a partir del apalancamiento (``M/tamaño = 1/L``)."""
    ensure_positive(leverage, "El apalancamiento")
    return _liquidation_from_ratio(side, entry, 1.0 / leverage, mmr, exit_fee_rate)


def wallet_liquidation_price(
    side: str,
    entry: float,
    size_usd: float,
    wallet_equity_usd: float,
    mmr: float,
    exit_fee_rate: float = 0.0,
) -> float | None:
    """Liquidación "estilo Nexo": toda la billetera de futuros respalda la posición.

    Mismas ecuaciones que el aislado, con el saldo de la billetera en lugar del
    margen. Es INFORMATIVO: las reglas usan el aislado, que es el conservador.
    """
    ensure_positive(size_usd, "El tamaño")
    ensure_positive(wallet_equity_usd, "El saldo de la billetera de futuros")
    return _liquidation_from_ratio(side, entry, wallet_equity_usd / size_usd, mmr, exit_fee_rate)


def liquidation_distance_pct(entry: float, liq: float | None) -> float | None:
    """Distancia a la liquidación como fracción de la entrada; ``None`` si no hay liquidación."""
    ensure_positive(entry, "La entrada")
    if liq is None:
        return None
    ensure_positive(liq, "El precio de liquidación")
    return abs(entry - liq) / entry


@dataclass(frozen=True)
class LiquidationCheck:
    """Resultado de comparar la liquidación con el stop.

    - ``stop_distance``: ``|entrada - stop|`` en precio.
    - ``liquidation_distance``: distancia de la entrada a la liquidación hacia
      el lado perdedor (en un long, ``entrada - liq``; en un short,
      ``liq - entrada``). Negativa si la liquidación quedó del lado ganador
      (la posición nacería liquidada). ``None`` si no hay liquidación.
    - ``ratio``: ``liquidation_distance / stop_distance`` (``None`` si no hay
      liquidación).
    - ``liq_before_stop``: la liquidación llega antes que el stop (o junto).
    - ``ok``: la liquidación queda más allá del stop y ``ratio >= min_ratio``.
    """

    side: Side
    entry: float
    stop: float
    liquidation_price: float | None
    min_ratio: float
    stop_distance: float
    liquidation_distance: float | None
    ratio: float | None
    liq_before_stop: bool
    ok: bool


def _ensure_min_ratio(min_ratio: float) -> float:
    ensure_positive(min_ratio, "El ratio mínimo de liquidación")
    if min_ratio <= 1:
        raise ValueError(
            "El ratio mínimo entre la distancia a liquidación y la distancia al stop tiene "
            f"que ser mayor que 1 (vino {min_ratio}); con 1 o menos la liquidación llegaría "
            "antes o junto con el stop."
        )
    return min_ratio


def check_liquidation_vs_stop(
    side: str,
    entry: float,
    stop: float,
    liq: float | None,
    min_ratio: float,
) -> LiquidationCheck:
    """Verifica que la liquidación quede más allá del stop y lo bastante lejos.

    ``min_ratio`` viene de ``trade_plan.liquidation.min_distance_ratio``. Si
    no hay liquidación (``liq is None``) el chequeo pasa.
    """
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    _ensure_min_ratio(min_ratio)
    stop_distance = abs(entry - stop)
    if liq is None:
        return LiquidationCheck(
            side=side,
            entry=float(entry),
            stop=float(stop),
            liquidation_price=None,
            min_ratio=float(min_ratio),
            stop_distance=stop_distance,
            liquidation_distance=None,
            ratio=None,
            liq_before_stop=False,
            ok=True,
        )
    ensure_positive(liq, "El precio de liquidación")
    liq_distance = entry - liq if side == "long" else liq - entry
    ratio = liq_distance / stop_distance
    liq_before_stop = liq_distance <= stop_distance
    ok = (not liq_before_stop) and ratio >= min_ratio * (1.0 - _REL_TOL)
    return LiquidationCheck(
        side=side,
        entry=float(entry),
        stop=float(stop),
        liquidation_price=float(liq),
        min_ratio=float(min_ratio),
        stop_distance=stop_distance,
        liquidation_distance=liq_distance,
        ratio=ratio,
        liq_before_stop=liq_before_stop,
        ok=ok,
    )


def _safe_margin_factor(
    side: str, entry: float, stop: float, mmr: float, exit_fee_rate: float, min_ratio: float
) -> float:
    """Factor ``k`` tal que la condición segura es ``margen >= k × tamaño``."""
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    _ensure_mmr_and_fee(mmr, exit_fee_rate)
    _ensure_min_ratio(min_ratio)
    x = min_ratio * abs(entry - stop) / entry
    if side == "long":
        if x >= 1:
            # La liquidación tendría que quedar en precio <= 0: solo sirve que
            # no se pueda liquidar, o sea margen >= tamaño.
            return 1.0
        return 1.0 - (1.0 - x) * (1.0 - mmr - exit_fee_rate)
    return (1.0 + x) * (1.0 + mmr + exit_fee_rate) - 1.0


def min_margin_for_safe_liquidation(
    side: str,
    entry: float,
    stop: float,
    size_usd: float,
    mmr: float,
    exit_fee_rate: float,
    min_ratio: float,
) -> float:
    """Margen mínimo (USD) para que la liquidación quede a ``min_ratio`` × la distancia al stop.

    El tamaño (y por lo tanto el riesgo en USD) no cambia; baja el
    apalancamiento.
    """
    ensure_positive(size_usd, "El tamaño")
    return size_usd * _safe_margin_factor(side, entry, stop, mmr, exit_fee_rate, min_ratio)


def max_size_for_safe_liquidation(
    side: str,
    entry: float,
    stop: float,
    margin_usd: float,
    mmr: float,
    exit_fee_rate: float,
    min_ratio: float,
) -> float:
    """Tamaño máximo (USD) para que la liquidación quede a ``min_ratio`` × la distancia al stop.

    El margen no cambia; al achicar el tamaño también baja el riesgo en USD.
    """
    ensure_positive(margin_usd, "El margen")
    return margin_usd / _safe_margin_factor(side, entry, stop, mmr, exit_fee_rate, min_ratio)
