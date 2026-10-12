"""Ajustes del plan según el régimen, la confianza y la estabilidad.

``apply_regime_adjustments(direction, score_total, regime, funding_rate_pct_8h,
profile, settings)`` devuelve un ``RegimeAdjustmentResult`` con:

- ``blocked`` / ``block_reason``: "No operar" por el régimen (por ejemplo, long
  en Crash) o porque, con los multiplicadores, el riesgo queda debajo del
  mínimo operable del perfil;
- ``min_abs_score``: score mínimo efectivo en la dirección del plan, el mayor
  entre el umbral de dirección (``scoring.direction_thresholds``, 30) y el del
  régimen (``min_abs_score``);
- ``min_rr``: R:R mínimo efectivo, el mayor entre el del perfil y el del
  régimen;
- ``risk_multiplier``: producto de todos los multiplicadores (se acumulan);
- ``risk_usd``: riesgo base (``profile.risk_per_trade_usd`` o
  ``base_risk_usd``) × ``risk_multiplier``;
- ``score_ok`` / ``score_check``: si el score llega al mínimo efectivo;
- ``adjustments``: cada ajuste con su motivo, por ejemplo
  "Riesgo × 0,5: el régimen está inestable". Los que tienen
  ``risk_multiplier=1`` son explicaciones (un ajuste condicional que no se
  aplicó y por qué).

Reglas (``config.yaml > regime_adjustments``):

- Por tipo de régimen y dirección: ``blocked``, ``min_abs_score``, ``min_rr`` y
  ``risk_multiplier``. Con ``only_if_funding_very_high`` el multiplicador solo
  corre si el funding (por 8 h, en % "humano") es **igual o mayor** que
  ``scoring.derivatives.funding_very_high_pct`` (el borde cuenta, del lado
  conservador). Si no hay dato de funding y
  ``assume_funding_very_high_if_missing`` (decisión del usuario), se toma como
  muy alto y el multiplicador se aplica, diciéndolo en el motivo (lado
  prudente); con ese parámetro en ``false`` no se aplica y se avisa.
- Confianza baja: ``confidence < low_confidence.threshold`` -> riesgo ×
  ``low_confidence.risk_multiplier``. Si la confianza no es un número finito se
  toma como baja.
- Régimen inestable: riesgo × ``unstable.risk_multiplier``.
- **Régimen no disponible** (``regime=None``, sin tipo, o el
  ``RegimeUnavailable`` que deja ``get_market_regimes`` en su lugar): se toma el lado
  conservador y se aplica el multiplicador de confianza baja con el motivo
  "régimen no disponible". No hay reglas por tipo (ni bloqueos ni mínimos
  extra).
- Piso de riesgo: si ``risk_usd`` queda debajo de
  ``profile.min_operable_risk_usd``, ``blocked=True`` ("No operar"). Si después
  se aplican otros multiplicadores (los circuit breakers de ``safety``), quien
  llama tiene que volver a controlar el piso con
  ``rules.check_min_operable_risk`` sobre el riesgo final, o pasar ya el riesgo
  reducido en ``base_risk_usd``.

``regime`` es cualquier objeto con ``.category``, ``.confidence`` (0 a 1) y
``.unstable``, por ejemplo ``regime.service.RegimeReport``. El régimen diario de
contexto no se usa acá (``regime_adjustments.apply_context_regime`` es
informativo por decisión del usuario).
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.risk.rules import (
    REL_TOL,
    RuleResult,
    check_min_abs_score,
    fmt_multiplier,
    fmt_percent,
    fmt_ratio,
    fmt_score,
    fmt_usd,
    json_safe,
)
from app.risk.sizing import Side, ensure_finite, ensure_positive, ensure_side

if TYPE_CHECKING:
    from app.config import AppSettings
    from app.risk.profiles import EffectiveProfile

# Nombre de cada tipo de régimen para los mensajes (texto de la UI, no es regla).
CATEGORY_LABELS: dict[str, str] = {
    "crash": "Crash (caída fuerte)",
    "bajista": "Bajista",
    "neutral": "Neutral",
    "alcista": "Alcista",
    "euforia": "Euforia (subida exagerada)",
}

# Códigos de los ajustes.
REGIME_BLOCKED = "regime_blocked"
REGIME_MIN_SCORE = "regime_min_score"
REGIME_MIN_RR = "regime_min_rr"
REGIME_RISK = "regime_risk"
REGIME_FUNDING_UNKNOWN = "regime_funding_unknown"
REGIME_FUNDING_ASSUMED_HIGH = "regime_funding_assumed_high"
REGIME_FUNDING_NORMAL = "regime_funding_normal"
LOW_CONFIDENCE = "low_confidence"
UNSTABLE = "unstable"
REGIME_UNAVAILABLE = "regime_unavailable"
MIN_OPERABLE_RISK = "min_operable_risk"


@dataclass(frozen=True)
class Adjustment:
    """Un ajuste con su motivo. ``risk_multiplier=1`` = solo explica (no cambia el riesgo)."""

    code: str
    message: str
    risk_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {"code": self.code, "message": self.message, "risk_multiplier": self.risk_multiplier}
        )


@dataclass(frozen=True)
class RegimeAdjustmentResult:
    """Resultado de aplicar los ajustes por régimen a una dirección.

    - ``blocked``: "No operar" por el régimen (``block_code="regime_blocked"``)
      o por el piso de riesgo (``"min_operable_risk"``); ``block_reasons``
      tiene todos los motivos y ``block_reason`` el primero.
    - ``score_ok``: el score llega a ``min_abs_score`` en la dirección.
    - ``ok``: ni bloqueado ni score corto. ``no_trade_reasons`` junta todos los
      motivos de "No operar" (bloqueos y score).
    """

    direction: Side
    score_total: float
    regime_available: bool
    category: str | None
    confidence: float | None
    unstable: bool
    blocked: bool
    block_code: str | None
    block_reasons: tuple[str, ...]
    min_abs_score: float
    min_abs_score_reason: str
    min_rr: float
    min_rr_reason: str
    risk_multiplier: float
    base_risk_usd: float
    risk_usd: float
    min_operable_risk_usd: float
    risk_below_min: bool
    score_ok: bool
    score_check: RuleResult
    adjustments: tuple[Adjustment, ...]

    @property
    def block_reason(self) -> str | None:
        return self.block_reasons[0] if self.block_reasons else None

    @property
    def ok(self) -> bool:
        return not self.blocked and self.score_ok

    @property
    def no_trade_reasons(self) -> tuple[str, ...]:
        extra = () if self.score_ok else (self.score_check.message,)
        return self.block_reasons + extra

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "direction": self.direction,
                "score_total": self.score_total,
                "regime_available": self.regime_available,
                "category": self.category,
                "confidence": self.confidence,
                "unstable": self.unstable,
                "ok": self.ok,
                "blocked": self.blocked,
                "block_code": self.block_code,
                "block_reason": self.block_reason,
                "block_reasons": list(self.block_reasons),
                "no_trade_reasons": list(self.no_trade_reasons),
                "min_abs_score": self.min_abs_score,
                "min_abs_score_reason": self.min_abs_score_reason,
                "min_rr": self.min_rr,
                "min_rr_reason": self.min_rr_reason,
                "risk_multiplier": self.risk_multiplier,
                "base_risk_usd": self.base_risk_usd,
                "risk_usd": self.risk_usd,
                "min_operable_risk_usd": self.min_operable_risk_usd,
                "risk_below_min": self.risk_below_min,
                "score_ok": self.score_ok,
                "score_check": self.score_check.to_dict(),
                "adjustments": [a.to_dict() for a in self.adjustments],
            }
        )


def _plural(side: Side) -> str:
    return "longs" if side == "long" else "shorts"


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def apply_regime_adjustments(
    direction: str,
    score_total: float,
    regime: Any | None,
    funding_rate_pct_8h: float | None,
    profile: EffectiveProfile,
    settings: AppSettings,
    *,
    base_risk_usd: float | None = None,
) -> RegimeAdjustmentResult:
    """Aplica las reglas del régimen, la confianza baja y la inestabilidad a ``direction``.

    ``funding_rate_pct_8h`` es el funding por período de 8 h en % "humano"
    (``0.05`` = 0,05 %), como ``scoring.derivatives``; ``None`` = sin dato.
    ``base_risk_usd`` es el riesgo antes de estos ajustes (por omisión
    ``profile.risk_per_trade_usd``). Ver el docstring del módulo.
    """
    side = ensure_side(direction)
    score = ensure_finite(score_total, "El score")
    cfg = settings.regime_adjustments
    thresholds = settings.scoring.direction_thresholds
    base_risk = (
        float(profile.risk_per_trade_usd)
        if base_risk_usd is None
        else ensure_positive(base_risk_usd, "El riesgo base")
    )

    adjustments: list[Adjustment] = []
    block_reasons: list[str] = []
    block_code: str | None = None

    min_abs = float(thresholds.long) if side == "long" else -float(thresholds.short)
    min_abs_reason = "el umbral de dirección"
    min_rr = float(profile.min_rr)
    min_rr_reason = f"el perfil {profile.name}"

    category = getattr(regime, "category", None) if regime is not None else None
    confidence = (
        _finite_or_none(getattr(regime, "confidence", None)) if regime is not None else None
    )
    unstable = bool(getattr(regime, "unstable", False)) if regime is not None else False
    available = regime is not None and category is not None
    low_cfg = cfg.low_confidence

    if not available:
        adjustments.append(
            Adjustment(
                REGIME_UNAVAILABLE,
                f"Riesgo × {fmt_multiplier(low_cfg.risk_multiplier)}: el régimen no está "
                "disponible (se toma como confianza baja).",
                low_cfg.risk_multiplier,
            )
        )
        category = None
        confidence = None
        unstable = False
    else:
        if category not in CATEGORY_LABELS:
            raise ValueError(
                f"Tipo de régimen desconocido: {category!r}. "
                f"Opciones: {', '.join(CATEGORY_LABELS)}."
            )
        label = CATEGORY_LABELS[category]
        rule = getattr(getattr(cfg, category), side)

        if rule.blocked:
            reason = f"No operar: con el régimen {label} los {_plural(side)} están bloqueados."
            block_reasons.append(reason)
            block_code = REGIME_BLOCKED
            adjustments.append(
                Adjustment(
                    REGIME_BLOCKED, f"{_plural(side).capitalize()} bloqueados: régimen {label}."
                )
            )

        if rule.min_abs_score is not None and rule.min_abs_score > min_abs:
            min_abs = float(rule.min_abs_score)
            min_abs_reason = f"el régimen {label}"
            needed = min_abs if side == "long" else -min_abs
            adjustments.append(
                Adjustment(
                    REGIME_MIN_SCORE,
                    f"Score mínimo {fmt_score(needed)} para {side}: régimen {label}.",
                )
            )

        if rule.min_rr is not None and rule.min_rr > min_rr:
            adjustments.append(
                Adjustment(
                    REGIME_MIN_RR,
                    f"R:R mínimo {fmt_ratio(rule.min_rr)}: régimen {label} "
                    f"(tu perfil pide {fmt_ratio(min_rr)}).",
                )
            )
            min_rr = float(rule.min_rr)
            min_rr_reason = f"el régimen {label}"

        if rule.risk_multiplier != 1.0:
            mult = fmt_multiplier(rule.risk_multiplier)
            if rule.only_if_funding_very_high:
                funding = _finite_or_none(funding_rate_pct_8h)
                limit = float(settings.scoring.derivatives.funding_very_high_pct)
                if funding is None and cfg.assume_funding_very_high_if_missing:
                    adjustments.append(
                        Adjustment(
                            REGIME_FUNDING_ASSUMED_HIGH,
                            f"Riesgo × {mult}: régimen {label} y no hay dato de funding, así "
                            "que se toma como muy alto (por las dudas, el lado prudente).",
                            rule.risk_multiplier,
                        )
                    )
                elif funding is None:
                    adjustments.append(
                        Adjustment(
                            REGIME_FUNDING_UNKNOWN,
                            f"Régimen {label}: no hay dato de funding, así que no se aplica "
                            f"el riesgo × {mult} que corresponde cuando el funding está muy alto.",
                        )
                    )
                elif funding >= limit:
                    adjustments.append(
                        Adjustment(
                            REGIME_RISK,
                            f"Riesgo × {mult}: régimen {label} con funding muy alto "
                            f"({fmt_percent(funding, 4)}; muy alto es desde "
                            f"{fmt_percent(limit, 4)}).",
                            rule.risk_multiplier,
                        )
                    )
                else:
                    adjustments.append(
                        Adjustment(
                            REGIME_FUNDING_NORMAL,
                            f"Régimen {label}: el funding ({fmt_percent(funding, 4)}) no está "
                            f"muy alto (desde {fmt_percent(limit, 4)}), el riesgo no se toca.",
                        )
                    )
            else:
                adjustments.append(
                    Adjustment(
                        REGIME_RISK, f"Riesgo × {mult}: régimen {label}.", rule.risk_multiplier
                    )
                )

        if confidence is None:
            adjustments.append(
                Adjustment(
                    LOW_CONFIDENCE,
                    f"Riesgo × {fmt_multiplier(low_cfg.risk_multiplier)}: no se sabe la "
                    "confianza del régimen (se toma como baja).",
                    low_cfg.risk_multiplier,
                )
            )
        else:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"La confianza del régimen tiene que estar entre 0 y 1 (vino {confidence})."
                )
            if confidence < low_cfg.threshold:
                adjustments.append(
                    Adjustment(
                        LOW_CONFIDENCE,
                        f"Riesgo × {fmt_multiplier(low_cfg.risk_multiplier)}: confianza del "
                        f"régimen baja ({fmt_percent(confidence * 100, 0)}, menos de "
                        f"{fmt_percent(low_cfg.threshold * 100, 0)}).",
                        low_cfg.risk_multiplier,
                    )
                )

        if unstable:
            m = cfg.unstable.risk_multiplier
            adjustments.append(
                Adjustment(
                    UNSTABLE,
                    f"Riesgo × {fmt_multiplier(m)}: el régimen está inestable (cambió de "
                    "tipo muchas veces en las últimas velas).",
                    m,
                )
            )

    risk_multiplier = math.prod(a.risk_multiplier for a in adjustments)
    risk_usd = base_risk * risk_multiplier
    floor = float(profile.min_operable_risk_usd)
    below = risk_usd < floor * (1.0 - REL_TOL)
    if below:
        detail = (
            f" ({fmt_usd(base_risk)} × {fmt_multiplier(risk_multiplier)})"
            if risk_multiplier != 1.0
            else ""
        )
        block_reasons.append(
            f"No operar: con los ajustes el riesgo queda en {fmt_usd(risk_usd)}{detail}, "
            f"menos que el mínimo operable de {fmt_usd(floor)}."
        )
        if block_code is None:
            block_code = MIN_OPERABLE_RISK

    score_check = check_min_abs_score(side, score, min_abs, reason=min_abs_reason)

    return RegimeAdjustmentResult(
        direction=side,
        score_total=score,
        regime_available=available,
        category=category,
        confidence=confidence,
        unstable=unstable,
        blocked=bool(block_reasons),
        block_code=block_code,
        block_reasons=tuple(block_reasons),
        min_abs_score=min_abs,
        min_abs_score_reason=min_abs_reason,
        min_rr=min_rr,
        min_rr_reason=min_rr_reason,
        risk_multiplier=risk_multiplier,
        base_risk_usd=base_risk,
        risk_usd=risk_usd,
        min_operable_risk_usd=floor,
        risk_below_min=below,
        score_ok=score_check.ok,
        score_check=score_check,
        adjustments=tuple(adjustments),
    )
