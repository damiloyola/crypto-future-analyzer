"""Reglas que bloquean la operación y fuerzan "No operar".

Cada regla es una función pura que devuelve un ``RuleResult``:

- ``code``: identificador estable para el código y la UI;
- ``ok``: ``True`` si la regla se cumple; ``False`` fuerza "No operar";
- ``message``: qué pasó, en castellano simple y con los números (cuando falla
  arranca con "No operar: ...");
- ``suggestion``: qué se puede hacer, si hay algo ("subí el margen a X USD",
  "esperá a mañana", ...); ``None`` si no hay nada útil para sugerir.

Reglas
------

- ``check_direction(score, thresholds)``: el score tiene que llegar al umbral de
  long (``>= +30``) o de short (``<= -30``); en el medio no hay dirección.
  ``direction_from_score`` devuelve la dirección candidata.
- ``check_min_abs_score(direction, score, min_abs)``: score mínimo en la
  dirección del plan (el que sube el régimen; ver ``regime_adjustments``).
- ``check_timeframe_conflict(trend_4h, trend_1h)``: 4h y 1h en direcciones
  opuestas (alcista contra bajista). "Lateral" no es conflicto.
- ``check_rr(net_rr, min_rr)``: R:R a TP2 **con comisiones** menor al mínimo.
- ``check_leverage(leverage, profile)``: apalancamiento resultante mayor al tope
  del perfil (el efectivo, con el override manual si lo hay).
- ``check_daily_limits(trades_today, pnl_today_usd, profile)``: se alcanzó el
  límite de operaciones o de pérdida del día.
- ``check_high_volatility_event(flag)``: evento de alta volatilidad marcado a
  mano por el usuario.
- ``check_min_operable_risk(risk_usd, profile)``: con los ajustes, el riesgo
  quedó debajo del mínimo operable del perfil.
- ``check_liquidation(check)``: la regla de liquidación (distancia a la
  liquidación >= ``min_distance_ratio`` × distancia al stop) sobre un
  ``LiquidationCheck`` ya calculado; se usa con la posición final (después de
  los ajustes de ``position.fit_position``), así que si falla es porque no hubo
  ajuste posible.

``failed_rules(results)`` filtra las que fallaron, en el mismo orden.

Las entradas inválidas (score no finito, tope no positivo, lado desconocido)
levantan ``ValueError`` con el mensaje en español, como el resto de ``risk``.
Los perfiles se reciben por "duck typing": alcanza con que tengan los atributos
de ``profiles.EffectiveProfile`` que use cada regla.

Formato de números
------------------

Este módulo también tiene los formateadores que usan los mensajes de ``risk``
(coma decimal y punto de miles, como se lee en Argentina): ``fmt_num``,
``fmt_usd``, ``fmt_leverage``, ``fmt_multiplier``, ``fmt_percent``,
``fmt_score``, ``fmt_price`` y ``fmt_ratio``. Y ``json_safe``, que deja un
resultado listo para JSON (NaN e infinitos pasan a ``None``).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.risk.sizing import Side, ensure_finite, ensure_positive, ensure_side

if TYPE_CHECKING:
    from app.config import DirectionThresholds
    from app.risk.liquidation import LiquidationCheck
    from app.risk.profiles import EffectiveProfile

# Tolerancia numérica relativa para comparar contra los topes (un valor que cae
# justo en el borde por coma flotante no debe fallar). No es una regla de riesgo.
REL_TOL = 1e-9

# Códigos de las reglas.
DIRECTION = "direction"
MIN_ABS_SCORE = "min_abs_score"
TIMEFRAME_CONFLICT = "timeframe_conflict"
MIN_RR = "min_rr"
LEVERAGE = "leverage"
DAILY_LIMITS = "daily_limits"
HIGH_VOLATILITY_EVENT = "high_volatility_event"
MIN_OPERABLE_RISK = "min_operable_risk"
LIQUIDATION = "liquidation"

NO_TRADE = "No operar"

_TREND_LABELS = ("alcista", "bajista", "lateral")
_SWAP = str.maketrans({",": ".", ".": ","})
_DASH = "—"


# --- formato de números --------------------------------------------------------


def fmt_num(value: float, decimals: int = 2, *, sign: bool = False, trim: bool = True) -> str:
    """Número con coma decimal y punto de miles: ``1234.5`` -> ``"1.234,5"``.

    ``trim=True`` saca los ceros de más a la derecha (``2.50`` -> ``"2,5"``,
    ``2.00`` -> ``"2"``). ``sign=True`` agrega ``+`` a los positivos. ``NaN`` o
    infinito -> ``"—"``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _DASH
    if not math.isfinite(v):
        return _DASH
    v = round(v, decimals)
    if v == 0:
        v = 0.0  # evita "-0"
    text = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    if trim and decimals > 0:
        text = text.rstrip("0").rstrip(".")
    return text.translate(_SWAP)


def fmt_usd(value: float, *, unit: bool = True) -> str:
    """Monto en USD: ``50`` -> ``"50 USD"``, ``28.4`` -> ``"28,40 USD"``.

    Sin decimales si el monto es entero (redondeado a centavos); si no, con dos.
    Punto de miles: ``5000`` -> ``"5.000 USD"``. ``unit=False`` saca el " USD"
    (para "de 100 a 162,40 USD").
    """
    try:
        v = round(float(value), 2)
    except (TypeError, ValueError):
        return _DASH
    if not math.isfinite(v):
        return _DASH
    decimals = 0 if v == int(v) else 2
    text = fmt_num(v, decimals, trim=False)
    return f"{text} USD" if unit else text


def fmt_leverage(value: float) -> str:
    """Apalancamiento: ``25`` -> ``"25x"``, ``31.24`` -> ``"31,2x"``, ``0.43`` -> ``"0,43x"``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _DASH
    decimals = 2 if abs(v) < 1 else 1
    return f"{fmt_num(v, decimals)}x"


def fmt_multiplier(value: float) -> str:
    """Multiplicador de riesgo: ``0.5`` -> ``"0,5"``, ``0.1875`` -> ``"0,1875"``."""
    return fmt_num(value, 4)


def fmt_percent(value_pct: float, decimals: int = 2) -> str:
    """Porcentaje "humano" (``55`` = 55 %): ``55`` -> ``"55 %"``, ``0.123`` -> ``"0,12 %"``."""
    text = fmt_num(value_pct, decimals)
    return text if text == _DASH else f"{text} %"


def fmt_score(value: float) -> str:
    """Score con signo: ``42`` -> ``"+42"``, ``-30`` -> ``"-30"``, ``42.55`` -> ``"+42,6"``."""
    return fmt_num(value, 1, sign=True)


def fmt_price(value: float) -> str:
    """Precio con decimales según su tamaño: BTC ``"60.000"``, XRP ``"0,5123"``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _DASH
    if not math.isfinite(v) or v <= 0:
        return fmt_num(v, 2)
    decimals = int(min(8, max(2, 4 - math.floor(math.log10(v)))))
    return fmt_num(v, decimals)


def fmt_ratio(value: float) -> str:
    """R:R o cociente: ``1.3214`` -> ``"1,32"``, ``2`` -> ``"2"``."""
    return fmt_num(value, 2)


def json_safe(obj: Any) -> Any:
    """Convierte a tipos de JSON: NaN/inf -> ``None``, tuplas -> listas, objetos con ``to_dict``."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return json_safe(to_dict())
    return str(obj)


# --- resultado -----------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    """Resultado de una regla. ``ok=False`` fuerza "No operar" y ``message`` dice por qué."""

    code: str
    ok: bool
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "ok": self.ok,
            "message": self.message,
            "suggestion": self.suggestion,
        }


def failed_rules(results: Iterable[RuleResult]) -> tuple[RuleResult, ...]:
    """Las reglas que fallaron (``ok=False``), en el mismo orden."""
    return tuple(r for r in results if not r.ok)


def _fail(code: str, reason: str, suggestion: str | None = None) -> RuleResult:
    return RuleResult(code=code, ok=False, message=f"{NO_TRADE}: {reason}", suggestion=suggestion)


def _side_word(side: Side) -> str:
    return "long" if side == "long" else "short"


# --- dirección y score -----------------------------------------------------------


def direction_from_score(score_total: float, thresholds: DirectionThresholds) -> Side | None:
    """Dirección candidata: ``"long"`` si score >= umbral long, ``"short"`` si <= umbral short.

    ``None`` si queda entre los dos umbrales (``scoring.direction_thresholds``).
    """
    score = ensure_finite(score_total, "El score")
    if score >= thresholds.long:
        return "long"
    if score <= thresholds.short:
        return "short"
    return None


def check_direction(score_total: float, thresholds: DirectionThresholds) -> RuleResult:
    """Hay dirección solo si el score llega a uno de los umbrales (long >= +30, short <= -30)."""
    score = ensure_finite(score_total, "El score")
    side = direction_from_score(score, thresholds)
    long_t, short_t = fmt_score(thresholds.long), fmt_score(thresholds.short)
    if side is None:
        return _fail(
            DIRECTION,
            f"el score ({fmt_score(score)}) está entre {short_t} y {long_t}, "
            "no hay una dirección clara.",
            f"Esperá a que el score llegue a {long_t} (long) o a {short_t} (short).",
        )
    needed = long_t if side == "long" else short_t
    cmp = "o más" if side == "long" else "o menos"
    return RuleResult(
        code=DIRECTION,
        ok=True,
        message=(
            f"Score {fmt_score(score)}: alcanza para buscar un {side} (hace falta {needed} {cmp})."
        ),
    )


def check_min_abs_score(
    direction: str, score_total: float, min_abs: float, *, reason: str | None = None
) -> RuleResult:
    """El score tiene que llegar al mínimo en la dirección del plan.

    Long: ``score >= +min_abs``. Short: ``score <= -min_abs``. ``reason`` dice
    quién pide ese mínimo (por ejemplo "el régimen Bajista") y se agrega al
    mensaje: "(lo pide el régimen Bajista)".
    """
    side = ensure_side(direction)
    score = ensure_finite(score_total, "El score")
    min_abs = ensure_positive(min_abs, "El score mínimo")
    needed = min_abs if side == "long" else -min_abs
    ok = score >= needed if side == "long" else score <= needed
    cmp = "o más" if side == "long" else "o menos"
    why = f"; lo pide {reason}" if reason else ""
    if ok:
        return RuleResult(
            code=MIN_ABS_SCORE,
            ok=True,
            message=(
                f"Score {fmt_score(score)}: cumple el mínimo para un {_side_word(side)} "
                f"({fmt_score(needed)} {cmp}{why})."
            ),
        )
    return _fail(
        MIN_ABS_SCORE,
        f"para un {_side_word(side)} el score tiene que ser {fmt_score(needed)} {cmp} "
        f"y está en {fmt_score(score)}" + (f" (lo pide {reason})." if reason else "."),
        "Esperá a que las señales se alineen más a favor.",
    )


# --- timeframes ------------------------------------------------------------------


def check_timeframe_conflict(trend_4h: str | None, trend_1h: str | None) -> RuleResult:
    """4h y 1h no pueden ir en direcciones opuestas (alcista contra bajista).

    Las tendencias son las de ``indicators.trend`` (``"alcista"``, ``"bajista"``
    o ``"lateral"``). Lateral no es conflicto. ``None`` = sin dato (no se toma
    como conflicto; se avisa en el mensaje).
    """
    for label, value in (("4h", trend_4h), ("1h", trend_1h)):
        if value is not None and value not in _TREND_LABELS:
            raise ValueError(
                f"La tendencia de {label} tiene que ser alcista, bajista o lateral "
                f"(vino {value!r})."
            )
    if trend_4h is None or trend_1h is None:
        missing = " y ".join(tf for tf, v in (("4h", trend_4h), ("1h", trend_1h)) if v is None)
        return RuleResult(
            code=TIMEFRAME_CONFLICT,
            ok=True,
            message=f"No hay dato de tendencia en {missing}: no se puede comparar 4h con 1h.",
        )
    if {trend_4h, trend_1h} == {"alcista", "bajista"}:
        return _fail(
            TIMEFRAME_CONFLICT,
            f"4h es {trend_4h} y 1h es {trend_1h}: van en direcciones opuestas.",
            "Esperá a que 4h y 1h se pongan de acuerdo.",
        )
    return RuleResult(
        code=TIMEFRAME_CONFLICT,
        ok=True,
        message=f"4h ({trend_4h}) y 1h ({trend_1h}) no se contradicen.",
    )


# --- R:R -------------------------------------------------------------------------


def check_rr(net_rr: float | None, min_rr: float, *, target: str = "TP2") -> RuleResult:
    """El R:R al objetivo (TP2), **con comisiones**, tiene que llegar al mínimo.

    ``net_rr`` es ``fees.net_rr`` (ganancia neta en el objetivo / pérdida neta
    en el stop). ``min_rr`` es el efectivo (el del perfil o el del régimen, el
    mayor). Si ``net_rr`` es ``None`` o no es finito, no se pudo calcular y la
    regla falla.
    """
    min_rr = ensure_positive(min_rr, "El R:R mínimo")
    if net_rr is None or not isinstance(net_rr, numbers.Real) or not math.isfinite(net_rr):
        return _fail(
            MIN_RR,
            f"no se pudo calcular el R:R a {target} con comisiones.",
        )
    if net_rr >= min_rr * (1.0 - REL_TOL):
        return RuleResult(
            code=MIN_RR,
            ok=True,
            message=(
                f"R:R a {target} con comisiones: {fmt_ratio(net_rr)} (mínimo {fmt_ratio(min_rr)})."
            ),
        )
    return _fail(
        MIN_RR,
        f"el R:R a {target} con comisiones es {fmt_ratio(net_rr)} y el mínimo es "
        f"{fmt_ratio(min_rr)} (por cada dólar que arriesgás ganarías "
        f"{fmt_ratio(max(net_rr, 0.0))}).",
        "Esperá una entrada mejor (más cerca del stop) o un objetivo más lejos.",
    )


# --- apalancamiento --------------------------------------------------------------


def check_leverage(leverage: float, profile: EffectiveProfile) -> RuleResult:
    """El apalancamiento resultante no puede pasar el tope efectivo del perfil."""
    leverage = ensure_positive(leverage, "El apalancamiento")
    cap = ensure_positive(profile.max_leverage, "El tope de apalancamiento")
    if leverage <= cap * (1.0 + REL_TOL):
        return RuleResult(
            code=LEVERAGE,
            ok=True,
            message=f"Apalancamiento {fmt_leverage(leverage)} (tope {fmt_leverage(cap)}).",
        )
    return _fail(
        LEVERAGE,
        f"el apalancamiento quedaría en {fmt_leverage(leverage)} y tu tope es {fmt_leverage(cap)}.",
        "Subí el margen o bajá el riesgo de la operación.",
    )


# --- límites del día -------------------------------------------------------------


def check_daily_limits(
    trades_today: int, pnl_today_usd: float, profile: EffectiveProfile
) -> RuleResult:
    """Bloquea al llegar a ``max_trades_per_day`` operaciones o a ``max_daily_loss_usd`` perdidos.

    ``pnl_today_usd`` es el resultado del día (negativo = pérdida). El "día" lo
    define quien llama (zona horaria del usuario).
    """
    if isinstance(trades_today, bool) or not isinstance(trades_today, numbers.Integral):
        raise ValueError(
            f"Las operaciones de hoy tienen que ser un entero (vino {trades_today!r})."
        )
    if trades_today < 0:
        raise ValueError(f"Las operaciones de hoy no pueden ser negativas (vino {trades_today}).")
    pnl = ensure_finite(pnl_today_usd, "El resultado del día")
    max_trades = int(profile.max_trades_per_day)
    max_loss = ensure_positive(profile.max_daily_loss_usd, "La pérdida máxima del día")
    loss = max(0.0, -pnl)

    problems: list[str] = []
    if trades_today >= max_trades:
        if max_trades == 0:
            problems.append("tu perfil no permite operaciones (máximo 0 por día)")
        else:
            problems.append(
                f"ya hiciste {trades_today} operaciones hoy y el máximo es {max_trades}"
            )
    if loss >= max_loss * (1.0 - REL_TOL):
        problems.append(f"hoy perdiste {fmt_usd(loss)} y el límite es {fmt_usd(max_loss)}")
    if problems:
        return _fail(DAILY_LIMITS, "; ".join(problems) + ".", "Esperá a mañana.")
    return RuleResult(
        code=DAILY_LIMITS,
        ok=True,
        message=(
            f"Hoy: {trades_today} de {max_trades} operaciones, "
            f"pérdida {fmt_usd(loss)} de {fmt_usd(max_loss)} permitidos."
        ),
    )


# --- evento de alta volatilidad -----------------------------------------------------


def check_high_volatility_event(flag: bool, description: str | None = None) -> RuleResult:
    """Bloquea si el usuario marcó a mano un evento de alta volatilidad (Fed, CPI, etc.)."""
    if not isinstance(flag, bool):
        raise ValueError(
            f"La marca de evento de alta volatilidad tiene que ser sí o no ({flag!r})."
        )
    if flag:
        what = f" ({description})" if description else ""
        return _fail(
            HIGH_VOLATILITY_EVENT,
            f"marcaste un evento de alta volatilidad{what}; el precio puede moverse "
            "muy rápido y saltar el stop.",
            "Esperá a que pase el evento y desmarcalo.",
        )
    return RuleResult(
        code=HIGH_VOLATILITY_EVENT,
        ok=True,
        message="No hay eventos de alta volatilidad marcados.",
    )


# --- riesgo mínimo operable ---------------------------------------------------------


def check_min_operable_risk(risk_usd: float, profile: EffectiveProfile) -> RuleResult:
    """Con los ajustes, el riesgo no puede quedar debajo de ``min_operable_risk_usd``."""
    risk = ensure_finite(risk_usd, "El riesgo")
    floor = ensure_positive(profile.min_operable_risk_usd, "El riesgo mínimo operable")
    if risk >= floor * (1.0 - REL_TOL):
        return RuleResult(
            code=MIN_OPERABLE_RISK,
            ok=True,
            message=f"Riesgo {fmt_usd(risk)} (mínimo operable {fmt_usd(floor)}).",
        )
    return _fail(
        MIN_OPERABLE_RISK,
        f"con los ajustes el riesgo queda en {fmt_usd(risk)}, menos que el mínimo "
        f"operable de {fmt_usd(floor)}.",
        "Esperá a que el mercado esté más claro (régimen estable y con confianza).",
    )


# --- liquidación ------------------------------------------------------------------


def check_liquidation(
    check: LiquidationCheck,
    *,
    suggested_margin_usd: float | None = None,
    suggested_size_usd: float | None = None,
) -> RuleResult:
    """Regla de liquidación sobre un ``LiquidationCheck`` (de ``check_liquidation_vs_stop``).

    La liquidación tiene que quedar más allá del stop y a una distancia de al
    menos ``min_ratio`` veces la distancia al stop. Si se pasan
    ``suggested_margin_usd`` (``liquidation.min_margin_for_safe_liquidation``)
    o ``suggested_size_usd`` (``liquidation.max_size_for_safe_liquidation``),
    van en la sugerencia.
    """
    min_ratio = fmt_ratio(check.min_ratio)
    if check.liquidation_price is None:
        return RuleResult(
            code=LIQUIDATION,
            ok=True,
            message="Con este margen la posición no se puede liquidar.",
        )
    liq = fmt_price(check.liquidation_price)
    if check.ok:
        return RuleResult(
            code=LIQUIDATION,
            ok=True,
            message=(
                f"La liquidación ({liq}) queda a {fmt_ratio(check.ratio or 0.0)} veces la "
                f"distancia al stop (mínimo {min_ratio})."
            ),
        )
    tips: list[str] = []
    if suggested_margin_usd is not None:
        tips.append(f"subí el margen a {fmt_usd(suggested_margin_usd)} (el riesgo no cambia)")
    if suggested_size_usd is not None:
        tips.append(f"achicá la posición a {fmt_usd(suggested_size_usd)} (el riesgo baja)")
    if tips:
        text = " o ".join(tips) + "."
        suggestion = text[0].upper() + text[1:]
    else:
        suggestion = "Subí el margen o achicá la posición."
    if check.liq_before_stop:
        reason = (
            f"te liquidan antes de llegar al stop (liquidación en {liq}, "
            f"stop en {fmt_price(check.stop)})."
        )
    else:
        reason = (
            f"la liquidación ({liq}) queda a {fmt_ratio(check.ratio or 0.0)} veces la "
            f"distancia al stop y tiene que estar al menos a {min_ratio} veces."
        )
    return _fail(LIQUIDATION, reason, suggestion)
