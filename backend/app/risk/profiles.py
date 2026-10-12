"""Perfiles de riesgo (conservador, moderado, agresivo).

Los perfiles viven en ``config.yaml > risk_profiles``. ``get_profile`` devuelve
un ``EffectiveProfile``: los valores del perfil más el tope de apalancamiento
**efectivo**, que el usuario puede cambiar a mano desde la UI.

Reglas del tope de apalancamiento (especificación del usuario):

- se puede subir a mano hasta ``risk_profiles.leverage_hard_cap`` (50x); más
  que eso no se permite nunca (``ValueError``: "no se permite más de 50x");
- por encima de ``risk_profiles.leverage_red_warning_above`` (25x) el perfil
  queda con ``red_warning=True`` y un texto de advertencia roja permanente
  (``red_warning_text``) para la UI;
- también se puede bajar, hasta 1x (``position.MIN_EXCHANGE_LEVERAGE``: con
  menos de 1x el margen sería más grande que la posición y ningún exchange lo
  permite; un tope así haría que todo plan diera "No operar").

``get_profile(name, settings, max_leverage_override=None)``:

- ``name``: nombre del perfil, sin distinguir mayúsculas; ``None`` = el de
  ``risk_profiles.default``. Un nombre que no existe levanta ``ValueError`` con
  las opciones.
- ``max_leverage_override``: el tope elegido a mano, o ``None`` para usar el del
  perfil.

``profile_names(settings)`` lista los perfiles disponibles.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.risk.position import MIN_EXCHANGE_LEVERAGE
from app.risk.rules import fmt_leverage, fmt_num, json_safe

if TYPE_CHECKING:
    from app.config import AppSettings


@dataclass(frozen=True)
class EffectiveProfile:
    """Perfil de riesgo listo para usar, con el tope de apalancamiento efectivo.

    Los campos de plata y límites son los de ``config.RiskProfile``.
    ``max_leverage`` es el tope **efectivo** (el override manual si lo hay, si
    no el del perfil); ``profile_max_leverage`` es el que trae el perfil.
    ``warnings`` tiene, en orden, la advertencia roja (si corresponde) y el
    aviso de que el tope se cambió a mano.
    """

    name: str
    monthly_wallet_usd: float
    margin_per_trade_usd: float
    risk_per_trade_usd: float
    min_rr: float
    max_leverage: float
    max_trades_per_day: int
    max_daily_loss_usd: float
    min_operable_risk_usd: float
    profile_max_leverage: float
    leverage_overridden: bool
    leverage_hard_cap: float
    leverage_red_warning_above: float
    red_warning: bool
    red_warning_text: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "name": self.name,
                "monthly_wallet_usd": self.monthly_wallet_usd,
                "margin_per_trade_usd": self.margin_per_trade_usd,
                "risk_per_trade_usd": self.risk_per_trade_usd,
                "min_rr": self.min_rr,
                "max_leverage": self.max_leverage,
                "max_trades_per_day": self.max_trades_per_day,
                "max_daily_loss_usd": self.max_daily_loss_usd,
                "min_operable_risk_usd": self.min_operable_risk_usd,
                "profile_max_leverage": self.profile_max_leverage,
                "leverage_overridden": self.leverage_overridden,
                "leverage_hard_cap": self.leverage_hard_cap,
                "leverage_red_warning_above": self.leverage_red_warning_above,
                "red_warning": self.red_warning,
                "red_warning_text": self.red_warning_text,
                "warnings": list(self.warnings),
            }
        )


def profile_names(settings: AppSettings) -> tuple[str, ...]:
    """Nombres de los perfiles de ``config.yaml``, en el orden en que aparecen."""
    return tuple(settings.risk_profiles.profiles)


def red_warning_text(max_leverage: float, threshold: float) -> str:
    """Texto de la advertencia roja permanente cuando el tope pasa de ``threshold``."""
    return (
        f"Advertencia: el tope de apalancamiento está en {fmt_leverage(max_leverage)} "
        f"(más de {fmt_leverage(threshold)}). Con tanto apalancamiento, un movimiento "
        "chico del precio en contra te puede liquidar."
    )


def _fmt_requested(value: float) -> str:
    """Tope pedido con hasta 4 decimales (``50.04`` -> ``"50,04x"``, no ``"50x"``)."""
    return f"{fmt_num(value, 4)}x"


def _validate_override(value: Any, hard_cap: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"El apalancamiento máximo tiene que ser un número (vino {value!r}).")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"El apalancamiento máximo tiene que ser mayor que cero (vino {value}).")
    if v < MIN_EXCHANGE_LEVERAGE:
        raise ValueError(
            f"El apalancamiento máximo tiene que ser {fmt_leverage(MIN_EXCHANGE_LEVERAGE)} o más "
            f"({fmt_leverage(MIN_EXCHANGE_LEVERAGE)} = sin apalancamiento; pediste "
            f"{_fmt_requested(v)})."
        )
    if v > hard_cap:
        raise ValueError(
            f"No se permite más de {fmt_leverage(hard_cap)} de apalancamiento "
            f"(pediste {_fmt_requested(v)})."
        )
    return v


def get_profile(
    name: str | None,
    settings: AppSettings,
    max_leverage_override: float | None = None,
) -> EffectiveProfile:
    """Devuelve el perfil ``name`` con el tope de apalancamiento efectivo.

    - ``name=None`` usa ``risk_profiles.default``; un nombre desconocido levanta
      ``ValueError`` con las opciones.
    - ``max_leverage_override`` tiene que ser >= 1 (1x) y <= ``leverage_hard_cap``
      (si no, ``ValueError``).
    - ``red_warning`` = tope efectivo > ``leverage_red_warning_above``.
    """
    cfg = settings.risk_profiles
    key = cfg.default if name is None else str(name).strip().lower()
    if key not in cfg.profiles:
        options = ", ".join(cfg.profiles)
        raise ValueError(f"No existe el perfil {name!r}. Opciones: {options}.")
    p = cfg.profiles[key]

    hard_cap = float(cfg.leverage_hard_cap)
    red_above = float(cfg.leverage_red_warning_above)
    profile_max = float(p.max_leverage)
    if max_leverage_override is None:
        effective = profile_max
    else:
        effective = _validate_override(max_leverage_override, hard_cap)
    overridden = effective != profile_max

    red = effective > red_above
    red_text = red_warning_text(effective, red_above) if red else None
    warnings: list[str] = []
    if red_text:
        warnings.append(red_text)
    if overridden:
        warnings.append(
            f"Tope de apalancamiento cambiado a mano: {fmt_leverage(effective)} "
            f"(el perfil {key} trae {fmt_leverage(profile_max)})."
        )

    return EffectiveProfile(
        name=key,
        monthly_wallet_usd=float(p.monthly_wallet_usd),
        margin_per_trade_usd=float(p.margin_per_trade_usd),
        risk_per_trade_usd=float(p.risk_per_trade_usd),
        min_rr=float(p.min_rr),
        max_leverage=effective,
        max_trades_per_day=int(p.max_trades_per_day),
        max_daily_loss_usd=float(p.max_daily_loss_usd),
        min_operable_risk_usd=float(p.min_operable_risk_usd),
        profile_max_leverage=profile_max,
        leverage_overridden=overridden,
        leverage_hard_cap=hard_cap,
        leverage_red_warning_above=red_above,
        red_warning=red,
        red_warning_text=red_text,
        warnings=tuple(warnings),
    )
