"""Margen y tamaño de la posición: ajuste automático por apalancamiento y liquidación.

Implementa la **decisión del usuario** (CLAUDE.md, "Apalancamiento por encima
del tope del perfil") con la matemática de ``sizing``, ``fees`` y
``liquidation``:

1. El tamaño sale del riesgo en USD y de la distancia al stop
   (``sizing.size_position``, con las comisiones de ``fees.fee_rates_from_config``
   y ``trade_plan.fees_included_in_risk``).
2. Margen necesario = el mayor entre el margen del perfil, ``tamaño /
   apalancamiento máximo`` (para no pasar el tope) y el margen mínimo de la
   regla de liquidación (``liquidation.min_margin_for_safe_liquidation``:
   distancia a liquidación >= ``trade_plan.liquidation.min_distance_ratio`` ×
   distancia al stop).
3. Si alcanza el margen del perfil, queda ese.
4. Si no, se **sube el margen** (redondeado hacia arriba a centavos) hasta el
   máximo, ``margen del perfil × trade_plan.liquidation.max_margin_multiplier``
   y nunca más que ``available_margin_usd`` si se pasa. El tamaño y el riesgo no
   cambian.
5. Si ni con el máximo alcanza, el margen queda en el máximo y se **achica la
   posición** (redondeada hacia abajo a centavos) hasta lo que entra sin pasar
   el tope de apalancamiento y cumpliendo la regla de liquidación. El riesgo
   baja: se recalcula la pérdida real al tocar el stop (comisiones incluidas).
6. Si la pérdida real al tocar el stop queda debajo de
   ``profile.min_operable_risk_usd``, ``ok=False``: "No operar".

Cada ajuste queda en ``adjustments`` con su motivo, por ejemplo "Margen subido de
100 a 178,67 USD para no pasar de 25x; el riesgo sigue en 50 USD".

Si ``available_margin_usd`` es menor que el margen del perfil, se usa lo
disponible como margen base (con su ajuste explicado).

Si la posición queda más chica que el margen (apalancamiento menor a 1x, pasa
con stops muy lejanos), se mantiene el margen del perfil y se agrega un aviso:
alcanza con poner de margen el tamaño de la posición.

La liquidación de las reglas es la de **margen aislado** (conservadora). La del
"modelo billetera" de Nexo (``liquidation.wallet_liquidation_price``, con toda la
billetera de futuros como colateral) se informa solo como dato cuando
``liquidation.show_wallet_model`` es ``true``; por omisión la billetera es
``profile.monthly_wallet_usd`` (o ``wallet_equity_usd`` si se pasa).

Las distancias ``*_pct`` son **fracción** (``0.01`` = 1 %), igual que en
``sizing`` y ``liquidation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.risk.fees import fee_rates_from_config
from app.risk.liquidation import (
    LiquidationCheck,
    check_liquidation_vs_stop,
    liquidation_distance_pct,
    liquidation_price,
    max_size_for_safe_liquidation,
    min_margin_for_safe_liquidation,
    mmr_from_config,
    wallet_liquidation_price,
)
from app.risk.regime_adjustments import Adjustment
from app.risk.rules import (
    REL_TOL,
    RuleResult,
    check_leverage,
    check_liquidation,
    check_min_operable_risk,
    fmt_leverage,
    fmt_ratio,
    fmt_usd,
    json_safe,
)
from app.risk.sizing import (
    Side,
    ensure_positive,
    ensure_side,
    ensure_stop_side,
    size_position,
    stop_distance_pct,
)

if TYPE_CHECKING:
    from app.config import AppSettings
    from app.risk.profiles import EffectiveProfile

# Códigos de los ajustes.
MARGIN_LIMITED = "margin_limited"
MARGIN_RAISED = "margin_raised"
SIZE_REDUCED = "size_reduced"

# Qué limitó la posición (``PositionFit.limited_by``).
LIMIT_LEVERAGE = "leverage"
LIMIT_LIQUIDATION = "liquidation"
LIMIT_AVAILABLE_MARGIN = "available_margin"


def ceil_cents(value: float) -> float:
    """Redondea hacia arriba a centavos (``162.401`` -> ``162.41``; ``162.4`` queda igual)."""
    return math.ceil(round(value * 100.0, 6)) / 100.0


def floor_cents(value: float) -> float:
    """Redondea hacia abajo a centavos (``4999.999`` -> ``4999.99``; ``5000`` queda igual)."""
    return math.floor(round(value * 100.0, 6)) / 100.0


@dataclass(frozen=True)
class PositionFit:
    """Posición final después de ajustar margen y tamaño.

    - ``requested_risk_usd``: el riesgo pedido (ya con los ajustes por régimen).
    - ``initial_*``: cómo quedaba antes de ajustar, con el margen base (tamaño,
      apalancamiento y si te liquidaban antes del stop); sirve para explicar
      los ajustes.
    - ``risk_usd``: pérdida **real** si toca el stop, precio + comisiones de ida y
      vuelta (``price_loss_at_stop_usd + fees_usd``). Igual al pedido salvo que
      se haya achicado la posición (o que ``fees_included_in_risk`` sea
      ``false``: ahí es el pedido más las comisiones).
    - ``margin_usd`` / ``size_usd`` / ``quantity`` / ``leverage``: lo que hay
      que cargar en el exchange.
    - ``liquidation_*``: margen aislado (lo que usan las reglas).
      ``liquidation_ratio`` = distancia a liquidación / distancia al stop.
    - ``wallet_liquidation_*``: modelo billetera de Nexo, solo informativo
      (``None`` si ``liquidation.show_wallet_model`` es ``false`` o si no hay
      liquidación).
    - ``red_warning``: el apalancamiento pasa de
      ``risk_profiles.leverage_red_warning_above``.
    - ``limited_by``: qué obligó a ajustar (``"leverage"``, ``"liquidation"``,
      ``"available_margin"``).
    - ``ok`` / ``block_reason``: ``False`` = "No operar" (el riesgo quedó debajo
      del mínimo operable).
    """

    side: Side
    entry: float
    stop: float
    stop_distance_pct: float
    profile_name: str
    requested_risk_usd: float
    initial_size_usd: float
    initial_leverage: float
    initial_liq_before_stop: bool
    fees_included_in_risk: bool
    profile_margin_usd: float
    max_margin_usd: float
    available_margin_usd: float | None
    margin_usd: float
    size_usd: float
    quantity: float
    leverage: float
    max_leverage: float
    risk_usd: float
    price_loss_at_stop_usd: float
    fees_usd: float
    min_operable_risk_usd: float
    liquidation_price: float | None
    liquidation_distance_pct: float | None
    liquidation_ratio: float | None
    liquidation_min_ratio: float
    liq_before_stop: bool
    liquidation_ok: bool
    wallet_equity_usd: float | None
    wallet_liquidation_price: float | None
    wallet_liquidation_distance_pct: float | None
    red_warning: bool
    margin_raised: bool
    size_reduced: bool
    limited_by: tuple[str, ...]
    adjustments: tuple[Adjustment, ...]
    warnings: tuple[str, ...]
    ok: bool
    block_reason: str | None
    liquidation_check: LiquidationCheck = field(repr=False)
    profile: EffectiveProfile = field(repr=False, compare=False)

    @property
    def risk_reduced(self) -> bool:
        """La posición se achicó y ahora se arriesga menos que lo pedido."""
        return self.size_reduced

    def checks(self) -> tuple[RuleResult, ...]:
        """Reglas de bloqueo sobre la posición final: apalancamiento, liquidación y riesgo."""
        return (
            check_leverage(self.leverage, self.profile),
            check_liquidation(self.liquidation_check),
            check_min_operable_risk(self.risk_usd, self.profile),
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (sin NaN ni infinitos)."""
        return json_safe(
            {
                "side": self.side,
                "entry": self.entry,
                "stop": self.stop,
                "stop_distance_pct": self.stop_distance_pct,
                "profile": self.profile_name,
                "requested_risk_usd": self.requested_risk_usd,
                "initial_size_usd": self.initial_size_usd,
                "initial_leverage": self.initial_leverage,
                "initial_liq_before_stop": self.initial_liq_before_stop,
                "fees_included_in_risk": self.fees_included_in_risk,
                "profile_margin_usd": self.profile_margin_usd,
                "max_margin_usd": self.max_margin_usd,
                "available_margin_usd": self.available_margin_usd,
                "margin_usd": self.margin_usd,
                "size_usd": self.size_usd,
                "quantity": self.quantity,
                "leverage": self.leverage,
                "max_leverage": self.max_leverage,
                "risk_usd": self.risk_usd,
                "risk_reduced": self.risk_reduced,
                "price_loss_at_stop_usd": self.price_loss_at_stop_usd,
                "fees_usd": self.fees_usd,
                "min_operable_risk_usd": self.min_operable_risk_usd,
                "liquidation_price": self.liquidation_price,
                "liquidation_distance_pct": self.liquidation_distance_pct,
                "liquidation_ratio": self.liquidation_ratio,
                "liquidation_min_ratio": self.liquidation_min_ratio,
                "liq_before_stop": self.liq_before_stop,
                "liquidation_ok": self.liquidation_ok,
                "wallet_equity_usd": self.wallet_equity_usd,
                "wallet_liquidation_price": self.wallet_liquidation_price,
                "wallet_liquidation_distance_pct": self.wallet_liquidation_distance_pct,
                "red_warning": self.red_warning,
                "margin_raised": self.margin_raised,
                "size_reduced": self.size_reduced,
                "limited_by": list(self.limited_by),
                "adjustments": [a.to_dict() for a in self.adjustments],
                "warnings": list(self.warnings),
                "ok": self.ok,
                "block_reason": self.block_reason,
            }
        )


def _raise_reasons(
    m_lev: float,
    m_liq: float,
    base: float,
    max_lev: float,
    ratio: float,
    liq_before_stop_at_base: bool,
) -> str:
    """Por qué se subió el margen, en palabras simples."""
    parts: list[str] = []
    if m_lev > base * (1.0 + REL_TOL):
        parts.append(f"no pasar de {fmt_leverage(max_lev)}")
    if m_liq > base * (1.0 + REL_TOL):
        if liq_before_stop_at_base:
            parts.append(
                "que no te liquiden antes de llegar al stop (la liquidación tiene que quedar "
                f"al menos a {fmt_ratio(ratio)} veces la distancia al stop)"
            )
        else:
            parts.append(
                f"que la liquidación quede al menos a {fmt_ratio(ratio)} veces la distancia al stop"
            )
    return " y para ".join(parts)


def fit_position(
    side: str,
    entry: float,
    stop: float,
    risk_usd: float,
    profile: EffectiveProfile,
    settings: AppSettings,
    *,
    available_margin_usd: float | None = None,
    wallet_equity_usd: float | None = None,
) -> PositionFit:
    """Calcula margen, tamaño y apalancamiento respetando el tope del perfil y la liquidación.

    - ``risk_usd``: riesgo a usar, ya con los multiplicadores del régimen (y de
      ``safety`` si los hay).
    - ``available_margin_usd``: tope de margen por el saldo cargado a mano
      (tiene que ser > 0; con saldo cero no se llama a esta función).
    - ``wallet_equity_usd``: saldo de la billetera de futuros para la
      liquidación "modelo Nexo" (por omisión ``profile.monthly_wallet_usd``).

    Ver el docstring del módulo para el algoritmo. Levanta ``ValueError`` con
    entradas inválidas (stop del lado equivocado, riesgo no positivo, etc.).
    """
    side = ensure_side(side)
    ensure_stop_side(side, entry, stop)
    requested = ensure_positive(risk_usd, "El riesgo")
    if available_margin_usd is not None:
        available_margin_usd = ensure_positive(available_margin_usd, "El margen disponible")
    if wallet_equity_usd is not None:
        wallet_equity_usd = ensure_positive(wallet_equity_usd, "El saldo de la billetera")

    f_in, f_out = fee_rates_from_config(settings.fees)
    mmr = mmr_from_config(settings.liquidation)
    liq_cfg = settings.trade_plan.liquidation
    min_ratio = float(liq_cfg.min_distance_ratio)
    fees_included = bool(settings.trade_plan.fees_included_in_risk)
    max_lev = ensure_positive(profile.max_leverage, "El tope de apalancamiento")
    profile_margin = ensure_positive(profile.margin_per_trade_usd, "El margen del perfil")

    adjustments: list[Adjustment] = []
    warnings: list[str] = []
    limited_by: list[str] = []

    # Margen base y margen máximo (perfil × multiplicador, nunca más que lo disponible).
    base_margin = profile_margin
    max_margin = profile_margin * float(liq_cfg.max_margin_multiplier)
    available_note = ""
    if available_margin_usd is not None:
        if available_margin_usd < max_margin:
            max_margin = available_margin_usd
            available_note = " (lo que tenés disponible)"
        if available_margin_usd < profile_margin:
            base_margin = available_margin_usd
            limited_by.append(LIMIT_AVAILABLE_MARGIN)
            adjustments.append(
                Adjustment(
                    MARGIN_LIMITED,
                    f"Margen bajado de {fmt_usd(profile_margin, unit=False)} a "
                    f"{fmt_usd(base_margin)}: es lo que tenés disponible.",
                )
            )

    # 1) Tamaño por riesgo.
    sizing = size_position(entry, stop, side, requested, base_margin, f_in, f_out, fees_included)
    size = sizing.size_usd

    # Cómo quedaría sin ajustar (para explicar los ajustes).
    initial_leverage = size / base_margin
    initial_liq = liquidation_price(side, entry, size, base_margin, mmr, f_out)
    initial_liq_before_stop = check_liquidation_vs_stop(
        side, entry, stop, initial_liq, min_ratio
    ).liq_before_stop

    # 2) Margen necesario por el tope de apalancamiento y por la regla de liquidación.
    m_lev = size / max_lev
    m_liq = min_margin_for_safe_liquidation(side, entry, stop, size, mmr, f_out, min_ratio)
    needed = max(base_margin, m_lev, m_liq)

    margin_raised = False
    size_reduced = False
    if needed <= base_margin * (1.0 + REL_TOL):
        # 3) Alcanza el margen base.
        margin = base_margin
    elif needed <= max_margin * (1.0 + REL_TOL):
        # 4) Se sube el margen; tamaño y riesgo no cambian.
        margin = min(ceil_cents(needed), max_margin)
        margin_raised = True
        if m_lev > base_margin * (1.0 + REL_TOL):
            limited_by.append(LIMIT_LEVERAGE)
        if m_liq > base_margin * (1.0 + REL_TOL):
            limited_by.append(LIMIT_LIQUIDATION)
        loss_now = sizing.loss_at_stop_usd
        why = _raise_reasons(m_lev, m_liq, base_margin, max_lev, min_ratio, initial_liq_before_stop)
        adjustments.append(
            Adjustment(
                MARGIN_RAISED,
                f"Margen subido de {fmt_usd(base_margin, unit=False)} a {fmt_usd(margin)} para "
                f"{why}; el riesgo sigue en {fmt_usd(loss_now)}.",
            )
        )
    else:
        # 5) Ni con el margen máximo alcanza: margen al máximo y posición más chica.
        margin = max_margin
        if margin > base_margin * (1.0 + REL_TOL):
            margin_raised = True
            adjustments.append(
                Adjustment(
                    MARGIN_RAISED,
                    f"Margen subido de {fmt_usd(base_margin, unit=False)} a {fmt_usd(margin)}, "
                    "el máximo"
                    f"{available_note or ' permitido'}; no alcanza para mantener el riesgo.",
                )
            )
        cap_lev = max_lev * margin
        cap_liq = max_size_for_safe_liquidation(side, entry, stop, margin, mmr, f_out, min_ratio)
        new_size = min(size, cap_lev, cap_liq)
        floored = floor_cents(new_size)
        if floored > 0:
            new_size = floored
        size_reduced = new_size < size
        if size_reduced:
            binding = LIMIT_LEVERAGE if cap_lev <= cap_liq else LIMIT_LIQUIDATION
            limited_by.append(binding)
            if available_note:
                limited_by.append(LIMIT_AVAILABLE_MARGIN)
            q = new_size / entry
            new_loss = q * abs(entry - stop) + q * entry * f_in + q * stop * f_out
            if binding == LIMIT_LEVERAGE:
                why = (
                    f"con el margen máximo de {fmt_usd(margin)}{available_note} no entra más "
                    f"sin pasar de {fmt_leverage(max_lev)}"
                )
            else:
                why = (
                    f"con el margen máximo de {fmt_usd(margin)}{available_note}, una posición "
                    "más grande dejaría la liquidación a menos de "
                    f"{fmt_ratio(min_ratio)} veces la distancia al stop"
                )
            adjustments.append(
                Adjustment(
                    SIZE_REDUCED,
                    f"Posición achicada de {fmt_usd(size, unit=False)} a {fmt_usd(new_size)}: "
                    f"{why}; "
                    f"ahora arriesgás {fmt_usd(new_loss)} (en vez de "
                    f"{fmt_usd(sizing.loss_at_stop_usd)}).",
                    new_loss / sizing.loss_at_stop_usd,
                )
            )
        size = new_size

    # Resultado final: pérdida real al tocar el stop, apalancamiento y liquidación.
    quantity = size / entry
    price_loss = quantity * abs(entry - stop)
    fees_usd = quantity * entry * f_in + quantity * stop * f_out
    loss = price_loss + fees_usd
    leverage = size / margin

    liq = liquidation_price(side, entry, size, margin, mmr, f_out)
    liq_check = check_liquidation_vs_stop(side, entry, stop, liq, min_ratio)
    liq_dist = liquidation_distance_pct(entry, liq)

    wallet = None
    wallet_liq = None
    wallet_dist = None
    if settings.liquidation.show_wallet_model:
        wallet = (
            wallet_equity_usd
            if wallet_equity_usd is not None
            else float(profile.monthly_wallet_usd)
        )
        wallet_liq = wallet_liquidation_price(side, entry, size, wallet, mmr, f_out)
        wallet_dist = liquidation_distance_pct(entry, wallet_liq)

    red_above = float(settings.risk_profiles.leverage_red_warning_above)
    red = leverage > red_above * (1.0 + REL_TOL)
    if red:
        warnings.append(
            f"Advertencia: el apalancamiento queda en {fmt_leverage(leverage)}, más de "
            f"{fmt_leverage(red_above)}. Un movimiento chico en contra te puede liquidar."
        )
    if leverage < 1.0:
        warnings.append(
            f"La posición ({fmt_usd(size)}) es más chica que el margen ({fmt_usd(margin)}): "
            f"alcanza con poner {fmt_usd(ceil_cents(size))} de margen (1x)."
        )

    # Bloqueos: piso de riesgo y, como red de seguridad, las reglas de la posición.
    floor = float(profile.min_operable_risk_usd)
    block_reason: str | None = None
    if loss < floor * (1.0 - REL_TOL):
        if size_reduced:
            block_reason = (
                f"No operar: aun con el margen máximo de {fmt_usd(margin)} la posición entra "
                f"solo hasta {fmt_usd(size)} y arriesgarías {fmt_usd(loss)}, menos que el "
                f"mínimo operable de {fmt_usd(floor)}."
            )
        else:
            block_reason = (
                f"No operar: el riesgo ({fmt_usd(loss)}) queda debajo del mínimo operable "
                f"de {fmt_usd(floor)}."
            )
    elif not liq_check.ok:
        block_reason = check_liquidation(liq_check).message
    elif leverage > max_lev * (1.0 + REL_TOL):
        block_reason = check_leverage(leverage, profile).message

    return PositionFit(
        side=side,
        entry=float(entry),
        stop=float(stop),
        stop_distance_pct=stop_distance_pct(entry, stop),
        profile_name=profile.name,
        requested_risk_usd=requested,
        initial_size_usd=sizing.size_usd,
        initial_leverage=initial_leverage,
        initial_liq_before_stop=initial_liq_before_stop,
        fees_included_in_risk=fees_included,
        profile_margin_usd=profile_margin,
        max_margin_usd=max_margin,
        available_margin_usd=available_margin_usd,
        margin_usd=margin,
        size_usd=size,
        quantity=quantity,
        leverage=leverage,
        max_leverage=max_lev,
        risk_usd=loss,
        price_loss_at_stop_usd=price_loss,
        fees_usd=fees_usd,
        min_operable_risk_usd=floor,
        liquidation_price=liq,
        liquidation_distance_pct=liq_dist,
        liquidation_ratio=liq_check.ratio,
        liquidation_min_ratio=min_ratio,
        liq_before_stop=liq_check.liq_before_stop,
        liquidation_ok=liq_check.ok,
        wallet_equity_usd=wallet,
        wallet_liquidation_price=wallet_liq,
        wallet_liquidation_distance_pct=wallet_dist,
        red_warning=red,
        margin_raised=margin_raised,
        size_reduced=size_reduced,
        limited_by=tuple(dict.fromkeys(limited_by)),
        adjustments=tuple(adjustments),
        warnings=tuple(warnings),
        ok=block_reason is None,
        block_reason=block_reason,
        liquidation_check=liq_check,
        profile=profile,
    )
