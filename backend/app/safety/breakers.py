"""Circuit breakers: la capa de seguridad con poder de veto (Hito 5).

Independiente del score, del régimen y del LLM: se evalúa al final y puede
bloquear cualquier plan. Trabaja con el estado cargado a mano
(``safety.state.TradingState``) y con los umbrales de ``config.yaml``
(``circuit_breakers`` y el perfil de riesgo).

Breakers (una función pura cada uno, devuelve ``BreakerStatus``)

- ``daily_loss_reduce``: pérdida del día >= 50 % del límite diario → riesgo
  × 0,5 hasta que termine el día.
- ``daily_loss_block``: pérdida del día >= 100 % del límite diario → bloqueo
  hasta el día siguiente.
- ``losing_streak``: 3 pérdidas seguidas → pausa de 4 h (enfriamiento).
- ``weekly_loss_reduce``: pérdida de la semana >= 60 % de la billetera mensual
  → riesgo × 0,5 hasta que termine la semana.
- ``monthly_drawdown_lock``: la billetera del mes cae >= 70 % desde su máximo
  → bloqueo total con archivo LOCKED.
- ``volatility_event``: hay un evento de alta volatilidad activo → bloqueo
  hasta su hora de fin.
- ``lock_file``: existe el archivo ``app.lock_file`` → bloqueo hasta que el
  usuario lo borre a mano.

Los porcentajes y multiplicadores salen de ``config.yaml``; el límite diario y
la billetera mensual, del perfil (``max_daily_loss_usd``,
``monthly_wallet_usd``). Los umbrales se comparan con ``>=`` y la misma
tolerancia de coma flotante que el resto de ``risk`` (``REL_TOL``): llegar
justo al umbral dispara.

Qué hay acá
    - ``evaluate_breakers(state, profile, settings, now, *, lock_path=None,
      lock=None) -> SafetyVerdict``: corre todos los breakers y arma el
      veredicto (verde / ámbar / rojo). La única lectura de disco es la del
      archivo LOCKED (``lockfile.read_lock``); si se pasa ``lock`` (un
      ``LockInfo`` ya leído) es pura.
    - Las funciones de cada breaker: ``daily_loss_reduce_breaker``,
      ``daily_loss_block_breaker``, ``losing_streak_breaker``,
      ``weekly_loss_reduce_breaker``, ``monthly_drawdown_lock_breaker``,
      ``volatility_event_breaker`` y ``lock_file_breaker``.
    - ``trigger_lock_if_needed(verdict, state, settings, now, *, repo=None)``:
      **con I/O**. Si la caída del mes disparó el bloqueo y todavía no hay
      archivo LOCKED, escribe el informe del incidente
      (``incident.write_incident_report``) y después el LOCKED
      (``lockfile.write_lock``), una sola vez.

Veredicto
    - ``rojo``: algún breaker bloquea (bloqueo, pausa, evento, LOCKED o la
      caída del mes): no se generan planes.
    - ``ámbar`` (``"ambar"``): no bloquea, pero algún breaker baja el riesgo.
      ``risk_multiplier`` es el producto de los multiplicadores de los que
      saltaron (día × semana: 0,5 × 0,5 = 0,25), cada uno con su motivo en
      ``adjustments``. El plan lo aplica al riesgo en USD; si así queda debajo
      del mínimo operable del perfil, "No operar".
    - ``verde``: ningún freno activo.

Pensado para el panel de riesgo (Hito 9): cada ``BreakerStatus`` trae umbral,
valor actual, progreso de 0 a 1, los textos ya formateados en castellano
(``threshold_text``, ``current_text``, ``until_text``) y un mensaje simple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, Literal

from app.config import AppSettings
from app.risk.rules import (
    REL_TOL,
    fmt_multiplier,
    fmt_num,
    fmt_percent,
    fmt_percent_beside,
    fmt_usd,
    json_safe,
)
from app.safety.incident import IncidentTrigger, fmt_datetime, write_incident_report
from app.safety.lockfile import LockInfo, is_locked, read_lock, write_lock
from app.safety.state import (
    PositionRecord,
    ProfileLimits,
    TradeRecord,
    TradingState,
    TradingStateSource,
    WalletRecord,
    resolve_user_timezone,
)

BreakerAction = Literal["reduce", "block", "lock", "none"]
VerdictStatus = Literal["verde", "ambar", "rojo"]
LockTriggerAction = Literal["not_needed", "already_locked", "locked"]

DAILY_LOSS_REDUCE = "daily_loss_reduce"
DAILY_LOSS_BLOCK = "daily_loss_block"
LOSING_STREAK = "losing_streak"
WEEKLY_LOSS_REDUCE = "weekly_loss_reduce"
MONTHLY_DRAWDOWN_LOCK = "monthly_drawdown_lock"
VOLATILITY_EVENT = "volatility_event"
LOCK_FILE = "lock_file"

#: Orden en que se evalúan y se muestran.
BREAKER_CODES: tuple[str, ...] = (
    DAILY_LOSS_REDUCE,
    DAILY_LOSS_BLOCK,
    LOSING_STREAK,
    WEEKLY_LOSS_REDUCE,
    MONTHLY_DRAWDOWN_LOCK,
    VOLATILITY_EVENT,
    LOCK_FILE,
)

STATUS_GREEN: VerdictStatus = "verde"
STATUS_AMBER: VerdictStatus = "ambar"
STATUS_RED: VerdictStatus = "rojo"

_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")


# --- resultados ----------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


@dataclass(frozen=True)
class BreakerStatus:
    """Estado de un breaker.

    - ``code`` / ``label`` / ``description``: identificador, nombre corto y qué
      hace la regla, en castellano simple.
    - ``kind``: lo que hace cuando salta (``"reduce"``, ``"block"`` o ``"lock"``).
    - ``threshold`` / ``current`` / ``unit``: umbral y valor actual (``None`` en
      los breakers de sí o no: evento y archivo LOCKED). ``threshold_text`` y
      ``current_text``: los mismos, ya formateados.
    - ``progress``: ``current / threshold`` recortado a 0-1 (para la barra).
    - ``triggered``: si saltó; ``action``: lo que hace ahora (``"none"`` si no
      saltó); ``risk_multiplier``: el multiplicador que aplica ahora (1 salvo
      un "reduce" disparado).
    - ``until``: hasta cuándo aplica (fin del día, de la semana, de la pausa o
      del evento; ``None`` si no saltó o si es el bloqueo total, que dura hasta
      que el usuario borre el LOCKED). ``resets_at``: cuándo vuelve a cero el
      contador (medianoche, lunes, día 1), aunque no haya saltado.
    - ``message``: qué está pasando, en castellano simple y con los números.
    - ``summary``: la misma idea en pocas palabras (para el banner y el
      titular del veredicto): ``"llegaste al límite de pérdida del día"``.
    """

    code: str
    label: str
    description: str
    kind: BreakerAction
    threshold: float | None
    current: float | None
    unit: str | None
    threshold_text: str
    current_text: str
    progress: float
    triggered: bool
    action: BreakerAction
    risk_multiplier: float
    until: datetime | None
    until_text: str | None
    resets_at: datetime | None
    message: str
    summary: str

    @property
    def blocks(self) -> bool:
        """``True`` si saltó y bloquea los planes (bloqueo, pausa, evento o lock)."""
        return self.triggered and self.kind in ("block", "lock")

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "code": self.code,
                "label": self.label,
                "description": self.description,
                "kind": self.kind,
                "threshold": self.threshold,
                "current": self.current,
                "unit": self.unit,
                "threshold_text": self.threshold_text,
                "current_text": self.current_text,
                "progress": self.progress,
                "triggered": self.triggered,
                "blocks": self.blocks,
                "action": self.action,
                "risk_multiplier": self.risk_multiplier,
                "until": _iso(self.until),
                "until_text": self.until_text,
                "resets_at": _iso(self.resets_at),
                "message": self.message,
                "summary": self.summary,
            }
        )


@dataclass(frozen=True)
class SafetyAdjustment:
    """Un ajuste de riesgo de la capa de seguridad, con su motivo."""

    code: str
    risk_multiplier: float
    reason: str
    until: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "code": self.code,
                "risk_multiplier": self.risk_multiplier,
                "reason": self.reason,
                "until": _iso(self.until),
            }
        )


@dataclass(frozen=True)
class SafetyVerdict:
    """Veredicto de la capa de seguridad (ver el docstring del módulo).

    - ``status``: ``"verde"``, ``"ambar"`` o ``"rojo"``; ``headline``: una
      oración para el banner.
    - ``blocked`` / ``block_reasons`` / ``block_codes``: si no se generan
      planes y por qué.
    - ``risk_multiplier`` y ``adjustments``: cuánto se achica el riesgo y por qué.
    - ``needs_lock``: la caída del mes disparó el bloqueo total y todavía no
      hay archivo LOCKED (``trigger_lock_if_needed`` lo escribe).
    - ``lock``: lo que se leyó del archivo LOCKED.
    """

    status: VerdictStatus
    headline: str
    blocked: bool
    block_reasons: tuple[str, ...]
    block_codes: tuple[str, ...]
    risk_multiplier: float
    adjustments: tuple[SafetyAdjustment, ...]
    needs_lock: bool
    locked: bool
    lock: LockInfo
    breakers: tuple[BreakerStatus, ...]
    warnings: tuple[str, ...]
    evaluated_at: datetime
    timezone: str

    def breaker(self, code: str) -> BreakerStatus:
        """El ``BreakerStatus`` con ese código (``KeyError`` si no existe)."""
        for b in self.breakers:
            if b.code == code:
                return b
        raise KeyError(code)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "status": self.status,
                "headline": self.headline,
                "blocked": self.blocked,
                "block_reasons": list(self.block_reasons),
                "block_codes": list(self.block_codes),
                "risk_multiplier": self.risk_multiplier,
                "adjustments": list(self.adjustments),
                "needs_lock": self.needs_lock,
                "locked": self.locked,
                "lock": self.lock.to_dict(tz=self.timezone),
                "breakers": list(self.breakers),
                "warnings": list(self.warnings),
                "evaluated_at": _iso(self.evaluated_at),
                "timezone": self.timezone,
            }
        )


# --- ayudas de formato y números --------------------------------------------------------


def _aware_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} tiene que ser una fecha con zona horaria (vino {value!r}).")
    return value.astimezone(UTC)


def _reached(current: float, threshold: float) -> bool:
    """``current >= threshold`` con la tolerancia de coma flotante de ``risk``."""
    return threshold > 0 and current >= threshold * (1.0 - REL_TOL)


def _progress(current: float, threshold: float) -> float:
    if threshold <= 0 or not math.isfinite(current):
        return 0.0
    return max(0.0, min(1.0, current / threshold))


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def until_phrase(until: datetime, now: datetime, tz: str | tzinfo) -> str:
    """``"hasta hoy a las 15:30"``, ``"hasta que termine el día (00:00)"``, ``"hasta el lunes..."``.

    Horas en la zona del usuario.
    """
    zone = resolve_user_timezone(tz)
    local_until = until.astimezone(zone)
    today = now.astimezone(zone).date()
    hhmm = f"{local_until:%H:%M}"
    if local_until.date() == today:
        return f"hasta hoy a las {hhmm}"
    if local_until.date() == today + timedelta(days=1):
        if hhmm == "00:00":
            return "hasta que termine el día (a las 00:00)"
        return f"hasta mañana a las {hhmm}"
    weekday = _WEEKDAYS[local_until.weekday()]
    return f"hasta el {weekday} {local_until:%d/%m} a las {hhmm}"


def _hours_text(hours: float) -> str:
    """``4`` -> ``"4 horas"``, ``1`` -> ``"1 hora"``, ``1.5`` -> ``"1,5 horas"``."""
    return f"{fmt_num(hours, 2)} {'hora' if hours == 1 else 'horas'}"


def _signed_usd(value: float) -> str:
    v = round(value, 2)
    if v == 0:
        return "0 USD"
    return f"{'+' if v > 0 else '-'}{fmt_usd(abs(v))}"


def _basis_text(basis: str, period: str) -> str:
    """Cómo se contó la pérdida: ``"neto: ganancias menos pérdidas de lo que cerraste hoy"``."""
    if basis == "gross":
        return f"sumando solo las operaciones perdidas que cerraste {period}"
    return f"neto: ganancias menos pérdidas de lo que cerraste {period}"


def _reached_word(current: float, threshold: float) -> str:
    """``"llegaste al"`` si quedó justo en el umbral (a centavos), ``"pasaste el"`` si lo superó."""
    return "llegaste al" if round(current, 2) <= round(threshold, 2) else "pasaste el"


def _result_phrase(loss: float, net: float, basis: str, when: str) -> str:
    """``"Hoy perdiste 60 USD (neto...)"`` o ``"Hoy vas ganando +30 USD"``."""
    period = "hoy" if when == "Hoy" else "en la semana"
    if loss > 0:
        return f"{when} perdiste {fmt_usd(loss)} ({_basis_text(basis, period)})"
    if net > 0:
        return f"{when} vas ganando {_signed_usd(net)}"
    return f"{when} no perdiste nada"


def _profile(profile: Any) -> ProfileLimits:
    return ProfileLimits.from_value(profile)


def _daily_limit(profile: ProfileLimits) -> float:
    limit = profile.max_daily_loss_usd
    if limit is None or not math.isfinite(limit) or limit <= 0:
        raise ValueError("El perfil no tiene una pérdida máxima del día válida (> 0).")
    return float(limit)


# --- breakers -------------------------------------------------------------------------


def daily_loss_reduce_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Pérdida del día >= ``pct_of_daily_limit`` % del límite → riesgo × ``risk_multiplier``."""
    cfg = settings.circuit_breakers.daily_loss_reduce
    prof = _profile(profile)
    moment = _aware_utc(now, "La hora")
    limit = _daily_limit(prof)
    threshold = limit * cfg.pct_of_daily_limit / 100.0
    loss = state.day_loss_usd
    triggered = _reached(loss, threshold)
    mult = fmt_multiplier(cfg.risk_multiplier)
    pct = fmt_percent(cfg.pct_of_daily_limit)
    day_end = state.periods.day_end
    until = day_end if triggered else None
    until_text = until_phrase(day_end, moment, state.timezone) if triggered else None
    result = _result_phrase(loss, state.day_pnl_usd, state.loss_basis, "Hoy")
    if triggered:
        message = (
            f"{result}: {_reached_word(loss, threshold)} {pct} de tu límite diario "
            f"({fmt_usd(threshold)} de "
            f"{fmt_usd(limit)}). El riesgo de cada operación se multiplica por {mult} "
            f"{until_text}."
        )
    else:
        message = (
            f"{result}. Si la pérdida del día llega a {fmt_usd(threshold)} ({pct} de tu "
            f"límite de {fmt_usd(limit)}), el riesgo baja (× {mult}) hasta que termine el día."
        )
    return BreakerStatus(
        code=DAILY_LOSS_REDUCE,
        label="Pérdida del día: bajar el riesgo",
        description=(
            f"Si en el día perdés el {pct} o más de tu límite diario ({fmt_usd(threshold)} de "
            f"{fmt_usd(limit)}), el riesgo de cada operación se multiplica por {mult} hasta que "
            "termine el día."
        ),
        kind="reduce",
        threshold=threshold,
        current=loss,
        unit="USD",
        threshold_text=fmt_usd(threshold),
        current_text=fmt_usd(loss),
        progress=_progress(loss, threshold),
        triggered=triggered,
        action="reduce" if triggered else "none",
        risk_multiplier=cfg.risk_multiplier if triggered else 1.0,
        until=until,
        until_text=until_text,
        resets_at=day_end,
        message=message,
        summary=(
            f"perdiste {fmt_usd(loss)} hoy: riesgo × {mult} hasta que termine el día"
            if triggered
            else f"pérdida del día {fmt_usd(loss)} de {fmt_usd(threshold)}"
        ),
    )


def daily_loss_block_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Pérdida del día >= ``pct_of_daily_limit`` % del límite (100 %) → bloqueo hasta mañana."""
    cfg = settings.circuit_breakers.daily_loss_block
    prof = _profile(profile)
    moment = _aware_utc(now, "La hora")
    limit = _daily_limit(prof)
    threshold = limit * cfg.pct_of_daily_limit / 100.0
    loss = state.day_loss_usd
    triggered = _reached(loss, threshold)
    day_end = state.periods.day_end
    until = day_end if triggered else None
    until_text = until_phrase(day_end, moment, state.timezone) if triggered else None
    result = _result_phrase(loss, state.day_pnl_usd, state.loss_basis, "Hoy")
    if triggered:
        message = (
            f"{result}: llegaste a tu límite de pérdida del día ({fmt_usd(threshold)}). "
            f"No se generan planes {until_text}. Mañana arrancás de cero."
        )
    else:
        remaining = max(0.0, threshold - loss)
        message = (
            f"{result}. Te quedan {fmt_usd(remaining)} antes de llegar al límite del día "
            f"({fmt_usd(threshold)}); si llegás, no se generan planes hasta mañana."
        )
    pct_txt = "" if cfg.pct_of_daily_limit == 100 else f" ({fmt_percent(cfg.pct_of_daily_limit)})"
    return BreakerStatus(
        code=DAILY_LOSS_BLOCK,
        label="Pérdida del día: límite",
        description=(
            f"Si en el día perdés {fmt_usd(threshold)} o más (tu límite diario{pct_txt}), no se "
            "generan planes hasta el día siguiente."
        ),
        kind="block",
        threshold=threshold,
        current=loss,
        unit="USD",
        threshold_text=fmt_usd(threshold),
        current_text=fmt_usd(loss),
        progress=_progress(loss, threshold),
        triggered=triggered,
        action="block" if triggered else "none",
        risk_multiplier=1.0,
        until=until,
        until_text=until_text,
        resets_at=day_end,
        message=message,
        summary=(
            "llegaste al límite de pérdida del día"
            if triggered
            else f"pérdida del día {fmt_usd(loss)} de {fmt_usd(threshold)}"
        ),
    )


def losing_streak_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """``consecutive_losses`` pérdidas seguidas → pausa de ``cooldown_hours`` horas."""
    cfg = settings.circuit_breakers.losing_streak
    moment = _aware_utc(now, "La hora")
    streak = state.streak
    required = int(cfg.consecutive_losses)
    hours_txt = _hours_text(cfg.cooldown_hours)
    count = streak.count_at(moment)
    active = streak.cooldown_active(moment)
    # Durante la pausa se muestran las pérdidas que la dispararon, aunque después
    # haya una ganada (la pausa sigue igual: es por tiempo).
    paused = max(count, required)
    until = streak.cooldown_until if active else None
    until_text = (
        until_phrase(streak.cooldown_until, moment, state.timezone)
        if active and streak.cooldown_until is not None
        else None
    )
    losses_txt = _plural(count, "pérdida seguida", "pérdidas seguidas")
    if active:
        last = fmt_datetime(streak.last_loss_at, state.timezone)
        message = (
            f"Perdiste {_plural(paused, 'operación seguida', 'operaciones seguidas')} "
            f"(la última cerró el {last}): pausa de {hours_txt} para enfriar la cabeza. "
            f"No se generan planes {until_text}."
        )
    elif count == 0:
        message = (
            f"No venís con pérdidas seguidas. Con {required} seguidas, la app hace una pausa "
            f"de {hours_txt}."
        )
    elif count < required:
        left = required - count
        if left == 1:
            message = (
                f"Venís con {losses_txt}: si la próxima también pierde, la app hace una pausa "
                f"de {hours_txt}."
            )
        else:
            message = (
                f"Venís con {losses_txt}; con {required} seguidas la app hace una pausa de "
                f"{hours_txt}."
            )
    else:
        message = (
            f"La pausa terminó, pero la racha sigue en {losses_txt}: si la próxima también "
            f"pierde, otra pausa de {hours_txt}."
        )
    reset_txt = (
        "Terminada la pausa, la cuenta vuelve a cero."
        if cfg.reset_after_cooldown
        else "La cuenta vuelve a cero solo con una operación ganada o empatada."
    )
    current = float(paused if active else count)
    return BreakerStatus(
        code=LOSING_STREAK,
        label="Pérdidas seguidas: pausa",
        description=(
            f"Si perdés {required} operaciones seguidas, la app hace una pausa de {hours_txt} "
            f"sin generar planes, para enfriar la cabeza. {reset_txt}"
        ),
        kind="block",
        threshold=float(required),
        current=current,
        unit="pérdidas seguidas",
        threshold_text=_plural(required, "pérdida seguida", "pérdidas seguidas"),
        current_text=_plural(int(current), "pérdida seguida", "pérdidas seguidas"),
        progress=1.0 if active else _progress(count, required),
        triggered=active,
        action="block" if active else "none",
        risk_multiplier=1.0,
        until=until,
        until_text=until_text,
        resets_at=None,
        message=message,
        summary=(
            f"pausa por {_plural(int(current), 'pérdida seguida', 'pérdidas seguidas')}"
            if active
            else f"{losses_txt} de {required}"
        ),
    )


def weekly_loss_reduce_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Pérdida de la semana >= ``pct_of_monthly_wallet`` % de la billetera mensual → riesgo × 0,5.

    La billetera mensual es la del perfil (``monthly_wallet_usd``).
    """
    cfg = settings.circuit_breakers.weekly_loss_reduce
    prof = _profile(profile)
    moment = _aware_utc(now, "La hora")
    wallet = prof.monthly_wallet_usd
    threshold = wallet * cfg.pct_of_monthly_wallet / 100.0
    loss = state.week_loss_usd
    triggered = _reached(loss, threshold)
    mult = fmt_multiplier(cfg.risk_multiplier)
    pct = fmt_percent(cfg.pct_of_monthly_wallet)
    week_end = state.periods.week_end
    until = week_end if triggered else None
    until_text = until_phrase(week_end, moment, state.timezone) if triggered else None
    result = _result_phrase(loss, state.week_pnl_usd, state.loss_basis, "En la semana")
    if triggered:
        message = (
            f"{result}: {_reached_word(loss, threshold)} {pct} de tu billetera mensual "
            f"({fmt_usd(threshold)} de "
            f"{fmt_usd(wallet)}). El riesgo de cada operación se multiplica por {mult} "
            f"{until_text}, cuando empieza la semana nueva."
        )
    else:
        message = (
            f"{result}. Si la pérdida de la semana llega a {fmt_usd(threshold)} ({pct} de tu "
            f"billetera mensual de {fmt_usd(wallet)}), el riesgo baja (× {mult}) hasta el "
            "final de la semana."
        )
    return BreakerStatus(
        code=WEEKLY_LOSS_REDUCE,
        label="Pérdida de la semana: bajar el riesgo",
        description=(
            f"Si en la semana perdés el {pct} o más de tu billetera mensual ({fmt_usd(threshold)} "
            f"de {fmt_usd(wallet)}), el riesgo de cada operación se multiplica por {mult} hasta "
            "que termine la semana."
        ),
        kind="reduce",
        threshold=threshold,
        current=loss,
        unit="USD",
        threshold_text=fmt_usd(threshold),
        current_text=fmt_usd(loss),
        progress=_progress(loss, threshold),
        triggered=triggered,
        action="reduce" if triggered else "none",
        risk_multiplier=cfg.risk_multiplier if triggered else 1.0,
        until=until,
        until_text=until_text,
        resets_at=week_end,
        message=message,
        summary=(
            f"perdiste {fmt_usd(loss)} en la semana: riesgo × {mult} hasta que termine la semana"
            if triggered
            else f"pérdida de la semana {fmt_usd(loss)} de {fmt_usd(threshold)}"
        ),
    )


def monthly_drawdown_lock_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Billetera del mes cae >= ``pct_from_peak`` % desde su máximo → bloqueo total (LOCKED)."""
    cfg = settings.circuit_breakers.monthly_drawdown_lock
    _aware_utc(now, "La hora")
    w = state.wallet
    threshold = float(cfg.pct_from_peak)
    current = w.drawdown_pct
    triggered = _reached(current, threshold)
    dd_txt = fmt_percent_beside(current, threshold)
    top = fmt_percent(threshold)
    trip = f"{fmt_usd(w.month_peak_usd)} → {fmt_usd(w.current_balance_usd)}"
    assumed = (
        " (Ojo: el saldo es supuesto porque nunca lo anotaste; anotalo para que el cálculo "
        "sea exacto.)"
        if w.assumed
        else ""
    )
    if triggered:
        message = (
            f"Tu billetera cayó {dd_txt} desde su máximo del mes ({trip}); el tope es {top}. "
            "La app se bloquea por completo: no se generan planes hasta que leas el informe "
            f"del incidente y borres a mano el archivo LOCKED.{assumed}"
        )
    elif w.drawdown_usd <= 0:
        message = (
            f"Tu billetera está en su máximo del mes ({fmt_usd(w.month_peak_usd)}). El "
            f"bloqueo total salta si cae {top} desde el máximo.{assumed}"
        )
    else:
        message = (
            f"Tu billetera está {dd_txt} abajo de su máximo del mes ({trip}). El bloqueo "
            f"total salta en {top}.{assumed}"
        )
    return BreakerStatus(
        code=MONTHLY_DRAWDOWN_LOCK,
        label="Caída del mes: bloqueo total",
        description=(
            f"Si la billetera cae {top} o más desde su máximo del mes, la app se bloquea por "
            "completo: escribe un informe de qué pasó y no genera más planes hasta que borres "
            "a mano el archivo LOCKED. Los depósitos suben el máximo; los retiros lo bajan en "
            "lo mismo que retiraste (sacar plata no es perder)."
        ),
        kind="lock",
        threshold=threshold,
        current=current,
        unit="%",
        threshold_text=top,
        current_text=dd_txt,
        progress=_progress(current, threshold),
        triggered=triggered,
        action="lock" if triggered else "none",
        risk_multiplier=1.0,
        until=None,
        until_text="hasta que borres a mano el archivo LOCKED" if triggered else None,
        resets_at=state.periods.month_end,
        message=message,
        summary=(
            f"la billetera del mes cayó {dd_txt} desde su máximo (tope {top})"
            if triggered
            else f"caída del mes {dd_txt} de {top}"
        ),
    )


def volatility_event_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Evento de alta volatilidad activo (``starts_at <= now < until``) → bloqueo hasta el fin."""
    moment = _aware_utc(now, "La hora")
    tz = state.timezone
    active = state.active_volatility_events(moment)
    upcoming = state.upcoming_volatility_events(moment)
    triggered = bool(active)
    until = max(e.until for e in active) if active else None
    until_text = until_phrase(until, moment, tz) if until is not None else None
    if triggered:
        notes = "; ".join(f"«{e.note}» (hasta el {fmt_datetime(e.until, tz)})" for e in active)
        what = "un evento" if len(active) == 1 else f"{len(active)} eventos"
        message = (
            f"Marcaste {what} de alta volatilidad: {notes}. El precio puede moverse muy rápido "
            f"y saltar el stop: no se generan planes {until_text}. Después se apaga solo."
        )
    else:
        message = "No hay eventos de alta volatilidad activos."
        if upcoming:
            nxt = upcoming[0]
            message += (
                f" Próximo: «{nxt.note}», desde el {fmt_datetime(nxt.starts_at, tz)} hasta el "
                f"{fmt_datetime(nxt.until, tz)}."
            )
    return BreakerStatus(
        code=VOLATILITY_EVENT,
        label="Evento de alta volatilidad",
        description=(
            "Mientras haya un evento de alta volatilidad marcado (dato de inflación, reunión de "
            "la Fed...), no se generan planes: el precio puede saltar el stop. Se apaga solo a "
            "la hora de fin que cargaste."
        ),
        kind="block",
        threshold=None,
        current=float(len(active)),
        unit="eventos activos",
        threshold_text="cualquier evento activo",
        current_text=(
            _plural(len(active), "evento activo", "eventos activos") if active else "ninguno"
        ),
        progress=1.0 if triggered else 0.0,
        triggered=triggered,
        action="block" if triggered else "none",
        risk_multiplier=1.0,
        until=until,
        until_text=until_text,
        resets_at=None,
        message=message,
        summary=(
            "hay un evento de alta volatilidad marcado ("
            + ", ".join(f"«{e.note}»" for e in active)
            + ")"
            if triggered
            else "sin eventos de alta volatilidad"
        ),
    )


def lock_file_breaker(lock: LockInfo, *, tz: str | tzinfo | None = None) -> BreakerStatus:
    """Existe el archivo LOCKED (aunque esté roto o no se pueda leer) → bloqueo total."""
    triggered = lock.exists
    return BreakerStatus(
        code=LOCK_FILE,
        label="Bloqueo total (archivo LOCKED)",
        description=(
            "Si existe el archivo LOCKED (lo escribe la app cuando la billetera del mes cae "
            "demasiado), no se generan planes. La app no lo borra nunca: lo borrás vos a mano "
            "después de leer el informe del incidente."
        ),
        kind="lock",
        threshold=None,
        current=None,
        unit=None,
        threshold_text="existe el archivo",
        current_text="existe" if triggered else "no existe",
        progress=1.0 if triggered else 0.0,
        triggered=triggered,
        action="lock" if triggered else "none",
        risk_multiplier=1.0,
        until=None,
        until_text="hasta que borres a mano el archivo LOCKED" if triggered else None,
        resets_at=None,
        message=lock.describe(tz) if triggered else "No hay archivo de bloqueo.",
        summary=(
            "la app está bloqueada (archivo LOCKED)" if triggered else "sin archivo de bloqueo"
        ),
    )


# --- veredicto ------------------------------------------------------------------------


def default_lock_path(settings: AppSettings) -> Path:
    """Ruta absoluta del archivo LOCKED (``app.lock_file``, relativa a la raíz del repo)."""
    return settings.resolve_path(settings.app.lock_file)


def _headline(
    status: VerdictStatus,
    blocking: list[BreakerStatus],
    reducing: list[BreakerStatus],
    multiplier: float,
) -> str:
    if status == STATUS_RED:
        # Primero el bloqueo total (es el que más dura), después el resto en orden.
        ordered = sorted(blocking, key=lambda b: 0 if b.kind == "lock" else 1)
        first = ordered[0]
        more = len(ordered) - 1
        tail = "" if more == 0 else f" (y {more} {'freno' if more == 1 else 'frenos'} más)"
        if first.kind == "lock":
            return (
                f"No se generan planes: {first.summary}{tail}. Leé el informe del incidente y "
                "borrá a mano el archivo LOCKED para seguir."
            )
        when = f", {first.until_text}" if first.until_text else ""
        return f"No se generan planes: {first.summary}{when}{tail}."
    if status == STATUS_AMBER:
        names = " y ".join(_REDUCE_NAMES.get(b.code, b.label.lower()) for b in reducing)
        return (
            f"Riesgo reducido (× {fmt_multiplier(multiplier)}) por {names}: el plan usa el "
            "riesgo achicado."
        )
    return "Ningún freno activo: el plan usa el riesgo normal de tu perfil."


_REDUCE_NAMES = {
    DAILY_LOSS_REDUCE: "la pérdida del día",
    WEEKLY_LOSS_REDUCE: "la pérdida de la semana",
}


def evaluate_breakers(
    state: TradingState,
    profile: Any,
    settings: AppSettings,
    now: datetime,
    *,
    lock_path: Path | str | None = None,
    lock: LockInfo | None = None,
) -> SafetyVerdict:
    """Corre todos los breakers y arma el veredicto (ver el docstring del módulo).

    - ``state``: de ``state.load_trading_state`` (del mismo día que ``now``:
      si ``now`` cae en otro día del usuario, ``ValueError``; hay que volver a
      cargar el estado).
    - ``profile``: el perfil de riesgo (``EffectiveProfile``).
    - ``lock_path``: dónde buscar el LOCKED (por defecto ``app.lock_file``).
    - ``lock``: un ``LockInfo`` ya leído; si viene, no se lee el disco.
    """
    moment = _aware_utc(now, "La hora")
    if not state.periods.contains_day(moment):
        raise ValueError(
            "El estado de trading es de otro día del usuario: volvé a cargarlo antes de "
            "evaluar los frenos."
        )
    tz = state.timezone
    if lock is None:
        lock = read_lock(default_lock_path(settings) if lock_path is None else lock_path)

    breakers = (
        daily_loss_reduce_breaker(state, profile, settings, moment),
        daily_loss_block_breaker(state, profile, settings, moment),
        losing_streak_breaker(state, profile, settings, moment),
        weekly_loss_reduce_breaker(state, profile, settings, moment),
        monthly_drawdown_lock_breaker(state, profile, settings, moment),
        volatility_event_breaker(state, profile, settings, moment),
        lock_file_breaker(lock, tz=tz),
    )
    reducing = [b for b in breakers if b.triggered and b.kind == "reduce"]
    blocking = [b for b in breakers if b.blocks]
    multiplier = 1.0
    for b in reducing:
        multiplier *= b.risk_multiplier
    adjustments = tuple(
        SafetyAdjustment(
            code=b.code,
            risk_multiplier=b.risk_multiplier,
            reason=f"Riesgo × {fmt_multiplier(b.risk_multiplier)}: {b.message}",
            until=b.until,
        )
        for b in reducing
    )
    blocked = bool(blocking)
    status: VerdictStatus = (
        STATUS_RED if blocked else STATUS_AMBER if multiplier < 1.0 else STATUS_GREEN
    )
    monthly = next(b for b in breakers if b.code == MONTHLY_DRAWDOWN_LOCK)

    warnings = list(state.warnings)
    if not any(e.is_active(moment) for e in state.volatility_events):
        for event in state.upcoming_volatility_events(moment):
            warnings.append(
                f"Hay un evento de alta volatilidad marcado para el "
                f"{fmt_datetime(event.starts_at, tz)} («{event.note}»): desde esa hora y hasta "
                f"el {fmt_datetime(event.until, tz)} no se van a generar planes."
            )
    if lock.exists and not lock.readable:
        warnings.append(
            "El archivo LOCKED existe pero no se pudo leer (¿se editó a mano?): la app sigue "
            "bloqueada igual."
        )
    return SafetyVerdict(
        status=status,
        headline=_headline(status, blocking, reducing, multiplier),
        blocked=blocked,
        block_reasons=tuple(b.message for b in blocking),
        block_codes=tuple(b.code for b in blocking),
        risk_multiplier=multiplier,
        adjustments=adjustments,
        needs_lock=monthly.triggered and not lock.exists,
        locked=lock.exists,
        lock=lock,
        breakers=breakers,
        warnings=tuple(warnings),
        evaluated_at=moment,
        timezone=tz,
    )


# --- bloqueo total (I/O) ---------------------------------------------------------------


@dataclass(frozen=True)
class LockTriggerResult:
    """Qué hizo ``trigger_lock_if_needed``.

    - ``action``: ``"not_needed"`` (no hacía falta), ``"already_locked"`` (ya
      había un LOCKED: no se tocó nada) o ``"locked"`` (se escribieron el
      informe y el LOCKED ahora).
    - ``report_path``: el informe escrito (``None`` si no se escribió).
    - ``report_error``: por qué no se pudo escribir el informe (el LOCKED se
      escribe igual: bloquear es más importante).
    - ``lock``: el LOCKED vigente (``None`` si no hacía falta).
    """

    action: LockTriggerAction
    message: str
    report_path: Path | None = None
    report_error: str | None = None
    lock: LockInfo | None = None

    @property
    def created(self) -> bool:
        return self.action == "locked"

    def to_dict(self, *, tz: str | tzinfo | None = None) -> dict[str, Any]:
        return json_safe(
            {
                "action": self.action,
                "created": self.created,
                "message": self.message,
                "report_path": None if self.report_path is None else str(self.report_path),
                "report_error": self.report_error,
                "lock": None if self.lock is None else self.lock.to_dict(tz=tz),
            }
        )


def _lock_reason(state: TradingState, settings: AppSettings) -> str:
    """Motivo corto del bloqueo total (va en el informe y en el archivo LOCKED)."""
    w = state.wallet
    top = float(settings.circuit_breakers.monthly_drawdown_lock.pct_from_peak)
    text = (
        f"La billetera del mes cayó {fmt_percent_beside(w.drawdown_pct, top)} desde su máximo "
        f"({fmt_usd(w.month_peak_usd)} → {fmt_usd(w.current_balance_usd)}); el tope es "
        f"{fmt_percent(top)}."
    )
    if w.assumed:
        text += (
            f" El saldo es supuesto: nunca se anotó, así que se tomó "
            f"{fmt_usd(w.assumed_start_usd or 0.0)} al empezar el mes."
        )
    return text


def _report_settings_info(state: TradingState, settings: AppSettings) -> dict[str, str]:
    """Umbrales que se listan en el informe ("Configuración en ese momento")."""
    cb = settings.circuit_breakers
    prof = state.profile
    info: dict[str, str] = {
        "Caída máxima de la billetera del mes": (
            f"{fmt_percent(cb.monthly_drawdown_lock.pct_from_peak)} desde el máximo"
        ),
    }
    if prof.max_daily_loss_usd:
        limit = prof.max_daily_loss_usd
        reduce_at = limit * cb.daily_loss_reduce.pct_of_daily_limit / 100.0
        block_at = limit * cb.daily_loss_block.pct_of_daily_limit / 100.0
        info["Pérdida del día"] = (
            f"riesgo × {fmt_multiplier(cb.daily_loss_reduce.risk_multiplier)} desde "
            f"{fmt_usd(reduce_at)}; bloqueo desde {fmt_usd(block_at)}"
        )
    info["Pérdidas seguidas"] = (
        f"{cb.losing_streak.consecutive_losses} seguidas → pausa de "
        f"{_hours_text(cb.losing_streak.cooldown_hours)}"
    )
    weekly = prof.monthly_wallet_usd * cb.weekly_loss_reduce.pct_of_monthly_wallet / 100.0
    info["Pérdida de la semana"] = (
        f"riesgo × {fmt_multiplier(cb.weekly_loss_reduce.risk_multiplier)} desde "
        f"{fmt_usd(weekly)} ({fmt_percent(cb.weekly_loss_reduce.pct_of_monthly_wallet)} de "
        f"{fmt_usd(prof.monthly_wallet_usd)})"
    )
    info["Cómo se cuenta la pérdida"] = (
        "neta (ganancias menos pérdidas)" if cb.loss_basis == "net" else "solo las pérdidas"
    )
    if state.wallet.assumed:
        info["Saldo de la billetera"] = (
            f"supuesto: {fmt_usd(state.wallet.assumed_start_usd or 0.0)} al empezar el mes "
            "(nunca se anotó un saldo)"
        )
    return info


def trigger_lock_if_needed(
    verdict: SafetyVerdict,
    state: TradingState,
    settings: AppSettings,
    now: datetime,
    *,
    repo: TradingStateSource | None = None,
    lock_path: Path | str | None = None,
    incident_dir: Path | str | None = None,
) -> LockTriggerResult:
    """Escribe el informe del incidente y el LOCKED si la caída del mes lo pide (una sola vez).

    - Si ``verdict.needs_lock`` es ``False``: no hace nada (``"not_needed"``).
    - Si ya existe el LOCKED (lo escribió otra corrida, o el usuario no lo
      borró): no escribe nada (``"already_locked"``).
    - Si no: primero el informe (``incident.write_incident_report`` en
      ``app.incident_dir``, con las operaciones del mes, los movimientos de la
      billetera, las posiciones abiertas y los números del disparo) y después
      el LOCKED (``lockfile.write_lock`` en ``app.lock_file``), que apunta al
      informe. Si el informe falla por un error de disco, el LOCKED se escribe
      igual y ``report_error`` dice qué pasó. Un error al escribir el LOCKED se
      propaga: quien llama tiene que enterarse de que el bloqueo no quedó escrito.

    ``repo`` (opcional): si viene, las operaciones del mes, los movimientos y
    las posiciones se vuelven a leer de la base en el momento del bloqueo; si
    no, se usan las del ``state``.
    """
    moment = _aware_utc(now, "La hora")
    tz = state.timezone
    path = Path(default_lock_path(settings) if lock_path is None else lock_path)
    if not verdict.needs_lock:
        return LockTriggerResult(
            action="not_needed", message="No hace falta bloquear la app.", lock=None
        )
    if is_locked(path):
        return LockTriggerResult(
            action="already_locked",
            message="La app ya estaba bloqueada: no se escribió nada nuevo.",
            lock=read_lock(path),
        )

    month_start = state.periods.month_start
    trades: tuple[Any, ...] = state.month_trades
    wallet_events: tuple[Any, ...] = state.month_wallet_events
    positions: tuple[Any, ...] = state.open_positions
    if repo is not None:
        trades = tuple(
            t
            for t in (TradeRecord.from_value(x) for x in repo.list_trades())
            if month_start <= t.closed_at <= moment
        )
        fresh_events = [
            e
            for e in (WalletRecord.from_value(x) for x in repo.list_wallet_events())
            if month_start <= e.ts <= moment
        ]
        anchor = state.wallet.anchor
        if anchor is not None and anchor.ts < month_start:
            fresh_events.insert(0, anchor)
        wallet_events = tuple(fresh_events)
        positions = tuple(PositionRecord.from_value(p) for p in repo.list_open_positions())

    w = state.wallet
    trigger = IncidentTrigger(
        code=MONTHLY_DRAWDOWN_LOCK,
        description=_lock_reason(state, settings),
        wallet_peak_usd=w.month_peak_usd,
        current_balance_usd=w.current_balance_usd,
        drawdown_pct=w.drawdown_pct,
        threshold_pct=float(settings.circuit_breakers.monthly_drawdown_lock.pct_from_peak),
    )
    folder = (
        settings.resolve_path(settings.app.incident_dir) if incident_dir is None else incident_dir
    )
    report_path: Path | None = None
    report_error: str | None = None
    try:
        report_path = write_incident_report(
            folder,
            triggered_at=moment,
            trigger=trigger,
            trades=trades,
            wallet_events=wallet_events,
            open_positions=positions,
            profile=state.profile,
            settings_info=_report_settings_info(state, settings),
            losing_streak=settings.circuit_breakers.losing_streak.consecutive_losses,
            lock_path=path,
            tz=tz,
        )
    except OSError as exc:
        report_error = f"No se pudo escribir el informe del incidente: {exc}"
    written = write_lock(
        path,
        reason=_lock_reason(state, settings),
        triggered_at=moment,
        report_path=report_path,
        trigger_code=MONTHLY_DRAWDOWN_LOCK,
        tz=tz,
    )
    if not written.created:
        return LockTriggerResult(
            action="already_locked",
            message=(
                "Otra corrida bloqueó la app al mismo tiempo: quedó su archivo LOCKED. "
                "El informe de esta corrida igual quedó guardado."
            ),
            report_path=report_path,
            report_error=report_error,
            lock=written.lock,
        )
    where = f" Informe: {report_path}." if report_path is not None else ""
    return LockTriggerResult(
        action="locked",
        message=(
            "La app quedó bloqueada: no va a generar planes hasta que leas el informe y "
            f"borres a mano el archivo {path}.{where}"
        ),
        report_path=report_path,
        report_error=report_error,
        lock=written.lock,
    )


__all__ = [
    "BREAKER_CODES",
    "DAILY_LOSS_BLOCK",
    "DAILY_LOSS_REDUCE",
    "LOCK_FILE",
    "LOSING_STREAK",
    "MONTHLY_DRAWDOWN_LOCK",
    "STATUS_AMBER",
    "STATUS_GREEN",
    "STATUS_RED",
    "VOLATILITY_EVENT",
    "WEEKLY_LOSS_REDUCE",
    "BreakerAction",
    "BreakerStatus",
    "LockTriggerAction",
    "LockTriggerResult",
    "SafetyAdjustment",
    "SafetyVerdict",
    "VerdictStatus",
    "daily_loss_block_breaker",
    "daily_loss_reduce_breaker",
    "default_lock_path",
    "evaluate_breakers",
    "lock_file_breaker",
    "losing_streak_breaker",
    "monthly_drawdown_lock_breaker",
    "trigger_lock_if_needed",
    "until_phrase",
    "volatility_event_breaker",
    "weekly_loss_reduce_breaker",
]
