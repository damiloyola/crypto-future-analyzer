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
- ``max_trades_today``: con el máximo de operaciones del día del perfil
  alcanzado → bloqueo hasta las 00:00 (``max_trades_blocks_status``: el
  estado queda en rojo, igual que lo que dice el plan).
- ``losing_streak``: 3 pérdidas seguidas → pausa de 4 h (enfriamiento);
  cualquier pérdida durante la pausa la estira (``extend_cooldown_on_loss``).
- ``weekly_loss_reduce``: pérdida de la semana >= 60 % de la billetera mensual
  → riesgo × 0,5 hasta que termine la semana.
- ``monthly_drawdown_lock``: la billetera del mes cae >= 70 % desde su máximo
  → bloqueo total con archivo LOCKED. Si el usuario ya borró el LOCKED y hay
  un bloqueo anotado en la base este mes (``restart_after_unlock``), la caída
  se mide desde el saldo del momento de ese bloqueo (``WalletState.restart``):
  solo vuelve a bloquear con una caída nueva del tope desde ahí. Mientras
  existe el LOCKED de un segundo bloqueo del mes, el freno muestra la caída
  que lo disparó (medida desde el bloqueo anterior) y la del mes calendario
  va aparte (``SafetyVerdict.drawdown_measure``).
- ``volatility_event``: hay un evento de alta volatilidad activo → bloqueo
  hasta su hora de fin; si otro evento empieza antes de que termine (o justo
  a esa hora), el bloqueo sigue hasta el fin de la cadena.
- ``lock_file``: existe el archivo ``app.lock_file`` → bloqueo hasta que el
  usuario lo borre a mano.

Los porcentajes y multiplicadores salen de ``config.yaml``; el límite diario y
la billetera mensual, del perfil (``max_daily_loss_usd``,
``monthly_wallet_usd``). Los umbrales se comparan con ``>=`` y la misma
tolerancia de coma flotante que el resto de ``risk`` (``REL_TOL``): llegar
justo al umbral dispara.

Con ``circuit_breakers.latch_until_period_end`` (decisión del usuario), los
frenos del día y de la semana miran la pérdida **más grande** a la que se llegó
en el período (``TradingState.day_max_loss_usd`` / ``week_max_loss_usd``): una
ganancia posterior no los apaga antes de las 00:00 o del lunes. El mensaje
muestra las dos cosas cuando difieren ("hoy llegaste a perder 60 USD (ahora vas
-40 USD)").

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
      (``incident.write_incident_report``; si ya hay uno idéntico de un
      intento anterior, lo reusa), después el LOCKED (``lockfile.write_lock``),
      una sola vez, y anota el bloqueo en la base (``repo.add_safety_lock``)
      para poder medir de nuevo la caída cuando el usuario lo borre. Un error
      en el informe o en la base no impide el LOCKED; un error al escribir el
      LOCKED se lanza como ``LockWriteError``.
    - ``backfill_unrecorded_lock(repo, settings, now, ...)``: **con I/O**. Si
      hay un LOCKED de este mes que no quedó anotado en la base (versión vieja
      o falla de la base), lo anota con su fecha y sus números
      (``monthly_drawdown_lock.backfill_unrecorded_lock``).
    - ``run_safety_checks(repo, settings, profile, now, ...)``: **con I/O**. Lo
      que hacen las consolas en cada corrida: anotar un LOCKED sin anotar,
      cargar el estado, evaluar, bloquear si hace falta y volver a evaluar.

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

import dataclasses
import logging
import math
from collections.abc import Sequence
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
    fmt_usd,
    json_safe,
)
from app.safety.incident import (
    CHECK_LOADED_DATA_TEXT,
    UNMEASURABLE_DRAWDOWN_TEXT,
    IncidentTrade,
    IncidentTrigger,
    analysis_test_marks,
    drawdown_unmeasurable,
    find_identical_report,
    fmt_datetime,
    fmt_drawdown_pct,
    incident_fingerprint,
    negative_balance_warning,
    write_incident_report,
)
from app.safety.lockfile import LockInfo, LockWriteError, is_locked, read_lock, write_lock
from app.safety.state import (
    DrawdownRestart,
    PositionRecord,
    ProfileLimits,
    SafetyLockRecord,
    TradeRecord,
    TradingState,
    TradingStateSource,
    VolatilityEventRecord,
    WalletRecord,
    load_trading_state,
    period_bounds,
    resolve_user_timezone,
)

logger = logging.getLogger(__name__)

BreakerAction = Literal["reduce", "block", "lock", "none"]
VerdictStatus = Literal["verde", "ambar", "rojo"]
LockTriggerAction = Literal["not_needed", "already_locked", "locked"]

DAILY_LOSS_REDUCE = "daily_loss_reduce"
DAILY_LOSS_BLOCK = "daily_loss_block"
MAX_TRADES_TODAY = "max_trades_today"
LOSING_STREAK = "losing_streak"
WEEKLY_LOSS_REDUCE = "weekly_loss_reduce"
MONTHLY_DRAWDOWN_LOCK = "monthly_drawdown_lock"
VOLATILITY_EVENT = "volatility_event"
LOCK_FILE = "lock_file"

#: Orden en que se evalúan y se muestran.
BREAKER_CODES: tuple[str, ...] = (
    DAILY_LOSS_REDUCE,
    DAILY_LOSS_BLOCK,
    MAX_TRADES_TODAY,
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
      ``current_text``: los mismos, ya formateados. En los frenos del día y de
      la semana, con ``latch_until_period_end``, ``current`` es la pérdida más
      grande a la que se llegó en el período (no la actual); en el del mes,
      después de desbloquear, la caída medida desde el último bloqueo.
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
      hay archivo LOCKED (``trigger_lock_if_needed`` lo escribe). Si el
      usuario ya borró un LOCKED anotado este mes (``restart_after_unlock``),
      cuenta la caída medida desde ese bloqueo, no la del mes calendario.
    - ``lock``: lo que se leyó del archivo LOCKED.
    - ``drawdown_measure``: la caída que miró el freno del mes (la del mes
      calendario o la medida desde un desbloqueo), con la del mes calendario
      aparte; ``drawdown_threshold_pct``: el tope.
    - ``lock_recorded``: si el LOCKED de este mes (de la caída del mes) está
      anotado en la base (``None`` si no hay LOCKED, no se puede leer, es de
      otro mes o de otro freno). En ``False`` hay un aviso en ``warnings``.
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
    drawdown_measure: DrawdownMeasure | None = None
    drawdown_threshold_pct: float | None = None
    lock_recorded: bool | None = None

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
                "drawdown_measure": (
                    None
                    if self.drawdown_measure is None
                    else self.drawdown_measure.to_dict(
                        threshold_pct=self.drawdown_threshold_pct, tz=self.timezone
                    )
                ),
                "lock_recorded": self.lock_recorded,
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


def _result_phrase(loss: float, net: float, basis: str, when: str, *, latched: bool = False) -> str:
    """``"Hoy perdiste 60 USD (neto...)"`` o ``"Hoy vas ganando +30 USD"``.

    Con ``latched`` (la pérdida más grande del período es mayor que la actual):
    ``"Hoy llegaste a perder 60 USD (ahora vas -40 USD)"``.
    """
    period = "hoy" if when == "Hoy" else "en la semana"
    if latched:
        return f"{when} llegaste a perder {fmt_usd(loss)} (ahora vas {_signed_usd(net)})"
    if loss > 0:
        return f"{when} perdiste {fmt_usd(loss)} ({_basis_text(basis, period)})"
    if net > 0:
        return f"{when} vas ganando {_signed_usd(net)}"
    return f"{when} no perdiste nada"


def _profile(profile: Any) -> ProfileLimits:
    return ProfileLimits.from_value(profile)


def _period_loss(
    state: TradingState, settings: AppSettings, period: Literal["day", "week"]
) -> tuple[float, float, bool]:
    """``(pérdida que mira el freno, pérdida actual, trabado)`` del día o de la semana.

    Con ``latch_until_period_end`` el freno mira la pérdida más grande a la que
    se llegó en el período; ``trabado`` es ``True`` si es mayor que la actual
    (hubo ganancias después). Sin eso, mira la actual.
    """
    if period == "day":
        current, largest = state.day_loss_usd, state.day_max_loss_usd
    else:
        current, largest = state.week_loss_usd, state.week_max_loss_usd
    if not settings.circuit_breakers.latch_until_period_end:
        return current, current, False
    loss = max(largest, current)
    return loss, current, round(loss, 2) > round(current, 2)


_LATCH_NOTE = " Lo que ganaste después no apaga el freno."


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
    latch = settings.circuit_breakers.latch_until_period_end
    loss, _now_loss, latched = _period_loss(state, settings, "day")
    triggered = _reached(loss, threshold)
    mult = fmt_multiplier(cfg.risk_multiplier)
    pct = fmt_percent(cfg.pct_of_daily_limit)
    day_end = state.periods.day_end
    until = day_end if triggered else None
    until_text = until_phrase(day_end, moment, state.timezone) if triggered else None
    result = _result_phrase(loss, state.day_pnl_usd, state.loss_basis, "Hoy", latched=latched)
    if triggered:
        message = (
            f"{result}: {_reached_word(loss, threshold)} {pct} de tu límite diario "
            f"({fmt_usd(threshold)} de "
            f"{fmt_usd(limit)}). El riesgo de cada operación se multiplica por {mult} "
            f"{until_text}.{_LATCH_NOTE if latched else ''}"
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
            f"Si en el día llegás a perder el {pct} o más de tu límite diario "
            f"({fmt_usd(threshold)} de {fmt_usd(limit)}), el riesgo de cada operación se "
            f"multiplica por {mult} hasta que termine el día"
            f"{', aunque después ganes' if latch else ''}."
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
            f"{'llegaste a perder' if latched else 'perdiste'} {fmt_usd(loss)} hoy: riesgo × "
            f"{mult} hasta que termine el día"
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
    latch = settings.circuit_breakers.latch_until_period_end
    loss, now_loss, latched = _period_loss(state, settings, "day")
    triggered = _reached(loss, threshold)
    day_end = state.periods.day_end
    until = day_end if triggered else None
    until_text = until_phrase(day_end, moment, state.timezone) if triggered else None
    result = _result_phrase(loss, state.day_pnl_usd, state.loss_basis, "Hoy", latched=latched)
    if triggered:
        message = (
            f"{result}: llegaste a tu límite de pérdida del día ({fmt_usd(threshold)}). "
            f"No se generan planes {until_text}.{_LATCH_NOTE if latched else ''} Mañana "
            "arrancás de cero."
        )
    else:
        # Lo que falta desde el resultado actual (el freno salta cuando la
        # pérdida acumulada llega al límite).
        remaining = max(0.0, threshold - now_loss)
        message = (
            f"{result}. Te quedan {fmt_usd(remaining)} antes de llegar al límite del día "
            f"({fmt_usd(threshold)}); si llegás, no se generan planes hasta mañana."
        )
    pct_txt = "" if cfg.pct_of_daily_limit == 100 else f" ({fmt_percent(cfg.pct_of_daily_limit)})"
    return BreakerStatus(
        code=DAILY_LOSS_BLOCK,
        label="Pérdida del día: límite",
        description=(
            f"Si en el día llegás a perder {fmt_usd(threshold)} o más (tu límite diario"
            f"{pct_txt}), no se generan planes hasta el día siguiente"
            f"{', aunque después ganes' if latch else ''}."
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
    trigger_count = len(streak.cooldown_trigger_ids) or required
    paused = max(count, trigger_count)
    until = streak.cooldown_until if active else None
    until_text = (
        until_phrase(streak.cooldown_until, moment, state.timezone)
        if active and streak.cooldown_until is not None
        else None
    )
    losses_txt = _plural(count, "pérdida seguida", "pérdidas seguidas")
    if active:
        # La racha que empezó la pausa (no la última pérdida: puede haber una
        # ganada en el medio) y, aparte, la pérdida que la estiró.
        started = fmt_datetime(streak.cooldown_started_at, state.timezone)
        message = (
            f"Perdiste {_plural(trigger_count, 'operación seguida', 'operaciones seguidas')} "
            f"(la última cerró el {started}): pausa de {hours_txt} para enfriar la cabeza."
        )
        if streak.cooldown_extended_at is not None:
            extended = fmt_datetime(streak.cooldown_extended_at, state.timezone)
            message += (
                f" Después volviste a perder durante la pausa (la operación que cerró el "
                f"{extended}), así que la pausa se estiró {hours_txt} desde esa pérdida."
            )
        elif streak.losses_during_cooldown:
            message += (
                " Volviste a perder durante la pausa, pero la pausa termina igual a la hora "
                "prevista."
            )
        message += f" No se generan planes {until_text}."
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
    extend_txt = (
        f"Si perdés otra vez durante la pausa, la pausa se estira {hours_txt} desde esa pérdida."
        if cfg.extend_cooldown_on_loss
        else "Si perdés otra vez durante la pausa, la pausa termina igual a la hora prevista."
    )
    current = float(paused if active else count)
    return BreakerStatus(
        code=LOSING_STREAK,
        label="Pérdidas seguidas: pausa",
        description=(
            f"Si perdés {required} operaciones seguidas, la app hace una pausa de {hours_txt} "
            f"sin generar planes, para enfriar la cabeza. {extend_txt} {reset_txt}"
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


def max_trades_today_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Operaciones de hoy >= ``max_trades_per_day`` del perfil → no hay planes hasta las 00:00.

    Con ``circuit_breakers.max_trades_blocks_status`` en ``false`` no salta
    (el límite lo sigue aplicando la regla de límites del día del plan) y solo
    informa cuántas van.
    """
    prof = _profile(profile)
    moment = _aware_utc(now, "La hora")
    enabled = settings.circuit_breakers.max_trades_blocks_status
    cap = prof.max_trades_per_day
    done = int(state.trades_today)
    day_end = state.periods.day_end
    reached = cap is not None and done >= cap
    triggered = bool(enabled and reached)
    until = day_end if triggered else None
    until_text = until_phrase(day_end, moment, state.timezone) if triggered else None
    ops = _plural(done, "operación", "operaciones")
    if cap is None:
        message = f"Hoy hiciste {ops}. Tu perfil no tiene un máximo de operaciones por día."
        summary = f"{ops} hoy"
        threshold_text = "sin máximo"
    else:
        threshold_text = _plural(cap, "operación", "operaciones")
        if cap == 0:
            done_txt = "tu perfil no permite operaciones (máximo 0 por día)"
        elif done > cap:
            done_txt = f"ya hiciste {ops} hoy y el máximo es {cap}"
        else:
            done_txt = f"ya hiciste {done} de {cap} operaciones permitidas hoy"
        if triggered:
            message = (
                f"{done_txt[:1].upper()}{done_txt[1:]}: no se generan planes {until_text}. "
                "Mañana arrancás de cero."
            )
            summary = done_txt
        elif reached:
            message = (
                f'{done_txt[:1].upper()}{done_txt[1:]}. El plan dice "No operar" hasta mañana.'
            )
            summary = done_txt
        else:
            left = cap - done
            message = f"Hoy hiciste {done} de {cap} operaciones permitidas."
            if left == 1:
                message += " Con una más llegás al máximo y no hay más planes hasta mañana."
            summary = f"{done} de {cap} operaciones hoy"
    return BreakerStatus(
        code=MAX_TRADES_TODAY,
        label="Operaciones del día: máximo",
        description=(
            "Si llegás al máximo de operaciones por día de tu perfil"
            + ("" if cap is None else f" ({threshold_text})")
            + ", no se generan planes hasta el día siguiente (a las 00:00). Cuentan las "
            "operaciones que abriste hoy, cerradas o abiertas."
        ),
        kind="block",
        threshold=None if cap is None else float(cap),
        current=float(done),
        unit="operaciones",
        threshold_text=threshold_text,
        # "3 de 3 operaciones" en la fila de la consola (valor "de" umbral).
        current_text=ops if cap is None else str(done),
        progress=1.0 if reached else (0.0 if not cap else _progress(done, cap)),
        triggered=triggered,
        action="block" if triggered else "none",
        risk_multiplier=1.0,
        until=until,
        until_text=until_text,
        resets_at=day_end,
        message=message,
        summary=summary,
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
    latch = settings.circuit_breakers.latch_until_period_end
    loss, _now_loss, latched = _period_loss(state, settings, "week")
    triggered = _reached(loss, threshold)
    mult = fmt_multiplier(cfg.risk_multiplier)
    pct = fmt_percent(cfg.pct_of_monthly_wallet)
    week_end = state.periods.week_end
    until = week_end if triggered else None
    until_text = until_phrase(week_end, moment, state.timezone) if triggered else None
    result = _result_phrase(
        loss, state.week_pnl_usd, state.loss_basis, "En la semana", latched=latched
    )
    if triggered:
        message = (
            f"{result}: {_reached_word(loss, threshold)} {pct} de tu billetera mensual "
            f"({fmt_usd(threshold)} de "
            f"{fmt_usd(wallet)}). El riesgo de cada operación se multiplica por {mult} "
            f"{until_text}, cuando empieza la semana nueva.{_LATCH_NOTE if latched else ''}"
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
            f"Si en la semana llegás a perder el {pct} o más de tu billetera mensual "
            f"({fmt_usd(threshold)} de {fmt_usd(wallet)}), el riesgo de cada operación se "
            f"multiplica por {mult} hasta que termine la semana"
            f"{', aunque después ganes' if latch else ''}."
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
            f"{'llegaste a perder' if latched else 'perdiste'} {fmt_usd(loss)} en la semana: "
            f"riesgo × {mult} hasta que termine la semana"
            if triggered
            else f"pérdida de la semana {fmt_usd(loss)} de {fmt_usd(threshold)}"
        ),
    )


def _short_datetime(dt: datetime, tz: str | tzinfo) -> str:
    """``"13/10 15:00"`` en la zona del usuario."""
    return f"{dt.astimezone(resolve_user_timezone(tz)):%d/%m %H:%M}"


MEASURE_MONTH = "month"
MEASURE_SINCE_UNLOCK = "since_unlock"


@dataclass(frozen=True)
class DrawdownMeasure:
    """La caída que mira el freno del mes, con la del mes calendario aparte.

    - ``basis``: ``"month"`` (desde el máximo del mes calendario) o
      ``"since_unlock"`` (desde el saldo de un bloqueo que el usuario ya
      desbloqueó: ``restart``).
    - ``peak_usd`` / ``balance_usd`` / ``drawdown_usd`` / ``drawdown_pct``: los
      números de esa medida (``drawdown_pct`` sin recortar).
    - ``while_locked``: la medida se eligió con el LOCKED puesto (es la que
      disparó el bloqueo vigente).
    - ``month_*``: los del mes calendario, siempre (para mostrarlos como dato
      aparte cuando la medida es desde un desbloqueo).
    """

    basis: str
    peak_usd: float
    balance_usd: float
    drawdown_usd: float
    drawdown_pct: float
    month_peak_usd: float
    month_drawdown_usd: float
    month_drawdown_pct: float
    restart: DrawdownRestart | None = None
    while_locked: bool = False

    @property
    def since_unlock(self) -> bool:
        return self.basis == MEASURE_SINCE_UNLOCK

    @property
    def negative_balance(self) -> bool:
        """El saldo calculado da negativo: la caída pasa el 100 % (dato mal cargado)."""
        return round(self.balance_usd, 2) < 0

    @property
    def unmeasurable(self) -> bool:
        """Saldo negativo con el máximo en 0 o menos: la caída no se puede medir.

        Pasa con un retiro más grande que el saldo o con la medida desde un
        bloqueo que arrancó con el saldo negativo. El freno del mes no salta
        hasta que se corrijan los datos (``incident.drawdown_unmeasurable``).
        """
        return drawdown_unmeasurable(self.balance_usd, self.drawdown_pct)

    @property
    def month_unmeasurable(self) -> bool:
        """Lo mismo para la caída del mes calendario."""
        return drawdown_unmeasurable(self.balance_usd, self.month_drawdown_pct)

    def drawdown_text(self, threshold_pct: float | None = None) -> str:
        """``"73,33 %"``, ``"más del 100 %"`` o ``"no se puede medir (saldo negativo)"``."""
        if self.unmeasurable:
            return f"{UNMEASURABLE_DRAWDOWN_TEXT} (saldo negativo)"
        return fmt_drawdown_pct(self.drawdown_pct, threshold_pct)

    def month_drawdown_text(self, threshold_pct: float | None = None) -> str:
        """La caída del mes calendario, escrita igual que ``drawdown_text``."""
        if self.month_unmeasurable:
            return f"{UNMEASURABLE_DRAWDOWN_TEXT} (saldo negativo)"
        return fmt_drawdown_pct(self.month_drawdown_pct, threshold_pct)

    def negative_warning(self) -> str | None:
        """El aviso de revisar los datos con el saldo negativo (``None`` si no lo es)."""
        return negative_balance_warning(
            self.balance_usd,
            self.drawdown_pct,
            month_drawdown_pct=self.month_drawdown_pct if self.since_unlock else None,
        )

    def to_dict(self, *, threshold_pct: float | None = None, tz: Any = None) -> dict[str, Any]:
        r = self.restart
        return json_safe(
            {
                "basis": self.basis,
                "since_unlock": self.since_unlock,
                "while_locked": self.while_locked,
                "peak_usd": self.peak_usd,
                "balance_usd": self.balance_usd,
                "drawdown_usd": self.drawdown_usd,
                "drawdown_pct": self.drawdown_pct,
                "drawdown_text": self.drawdown_text(threshold_pct),
                "negative_balance": self.negative_balance,
                "unmeasurable": self.unmeasurable,
                "negative_balance_warning": self.negative_warning(),
                "measured_since": None if r is None else _iso(r.locked_at),
                "measured_since_text": (
                    None if r is None or tz is None else _short_datetime(r.locked_at, tz)
                ),
                "lock_id": None if r is None else r.lock_id,
                "start_balance_usd": None if r is None else r.start_balance_usd,
                "month": {
                    "peak_usd": self.month_peak_usd,
                    "balance_usd": self.balance_usd,
                    "drawdown_usd": self.month_drawdown_usd,
                    "drawdown_pct": self.month_drawdown_pct,
                    "drawdown_text": self.month_drawdown_text(threshold_pct),
                },
            }
        )


def month_drawdown_measure(
    state: TradingState,
    settings: AppSettings,
    *,
    locked: bool,
    lock: LockInfo | None = None,
) -> DrawdownMeasure:
    """Qué caída mira el freno del mes (pura).

    - Sin LOCKED (``locked=False``), con ``restart_after_unlock`` y un bloqueo
      anotado este mes: la medida desde el último bloqueo (``wallet.restart``).
    - Con el LOCKED puesto: la medida con la que se escribió ese LOCKED. Si
      antes de él (``lock.triggered_at``) hubo otro bloqueo este mes que el
      usuario desbloqueó, la medida desde ese bloqueo anterior; si no (o si el
      LOCKED no se puede leer), la del mes calendario.
    - Si no, la del mes calendario.
    """
    w = state.wallet
    cfg = settings.circuit_breakers.monthly_drawdown_lock
    r: DrawdownRestart | None = None
    if cfg.restart_after_unlock:
        if not locked:
            r = w.restart
        elif lock is not None and lock.readable and lock.triggered_at is not None:
            r = w.restart_before(lock.triggered_at)
    month = {
        "month_peak_usd": w.month_peak_usd,
        "month_drawdown_usd": w.drawdown_usd,
        "month_drawdown_pct": w.drawdown_pct,
    }
    if r is not None:
        return DrawdownMeasure(
            MEASURE_SINCE_UNLOCK,
            r.peak_usd,
            r.current_balance_usd,
            r.drawdown_usd,
            r.drawdown_pct,
            restart=r,
            while_locked=locked,
            **month,
        )
    return DrawdownMeasure(
        MEASURE_MONTH,
        w.month_peak_usd,
        w.current_balance_usd,
        w.drawdown_usd,
        w.drawdown_pct,
        while_locked=locked,
        **month,
    )


def _month_measure(
    state: TradingState, settings: AppSettings, *, locked: bool, lock: LockInfo | None = None
) -> DrawdownMeasure:
    """Alias interno de ``month_drawdown_measure`` (lo usan los tests viejos)."""
    return month_drawdown_measure(state, settings, locked=locked, lock=lock)


def _wallet_rules_text(settings: AppSettings) -> str:
    """Cómo mueven el máximo los depósitos y los retiros, y qué pasa al borrar el LOCKED."""
    cfg = settings.circuit_breakers.monthly_drawdown_lock
    deposit = (
        "Un depósito sube el máximo en lo mismo que depositaste (la pérdida de antes sigue "
        "contando)."
        if cfg.deposit_raises_peak
        else "Un depósito sube el saldo; el máximo sube solo si el saldo lo pasa."
    )
    if cfg.deposit_refills_withdrawals:
        rest = (
            "lo que pase de eso sube el máximo en lo mismo (la pérdida de antes sigue contando)."
            if cfg.deposit_raises_peak
            else "lo que pase de eso sube el saldo, y el máximo sube solo si el saldo lo pasa."
        )
        deposit = (
            "Un depósito primero devuelve lo que retiraste en el mes, sin cambiar el % de caída "
            f"(retirar y volver a depositar no borra la caída); {rest}"
        )
    withdrawal = {
        "proportional": (
            "Un retiro baja el máximo en la misma proporción que el saldo: sacar plata no cambia "
            "el % de caída."
        ),
        "subtract": "Un retiro baja el máximo en lo mismo que retiraste.",
        "none": "Un retiro no cambia el máximo: cuenta como caída.",
    }[cfg.withdrawal_peak]
    unlock = (
        "Si borrás el LOCKED, la caída se vuelve a medir desde el saldo del momento del "
        "bloqueo, como si empezara un mes nuevo."
        if cfg.restart_after_unlock
        else "Si borrás el LOCKED y la caída del mes sigue arriba del tope, la app se vuelve a "
        "bloquear."
    )
    return f"{deposit} {withdrawal} {unlock}"


def monthly_drawdown_lock_breaker(
    state: TradingState,
    profile: Any,
    settings: AppSettings,
    now: datetime,
    *,
    locked: bool = False,
    lock: LockInfo | None = None,
) -> BreakerStatus:
    """Billetera del mes cae >= ``pct_from_peak`` % desde su máximo → bloqueo total (LOCKED).

    ``locked``: si existe el archivo LOCKED (``lock``: lo que se leyó de él).
    Sin LOCKED, con ``restart_after_unlock`` y un bloqueo anotado este mes, la
    caída se mide desde el saldo del momento de ese bloqueo; con el LOCKED de
    un segundo bloqueo, desde el bloqueo anterior (ver
    ``month_drawdown_measure``).
    """
    cfg = settings.circuit_breakers.monthly_drawdown_lock
    _aware_utc(now, "La hora")
    w = state.wallet
    tz = state.timezone
    m = month_drawdown_measure(state, settings, locked=locked, lock=lock)
    threshold = float(cfg.pct_from_peak)
    current = m.drawdown_pct
    triggered = _reached(current, threshold)
    # Más de 100 % solo pasa con el saldo calculado negativo: el texto dice
    # "más del 100 %" y pide revisar los datos; ``current`` queda sin recortar.
    dd_txt = m.drawdown_text(threshold)
    top = fmt_percent(threshold)
    trip = f"{fmt_usd(m.peak_usd)} → {fmt_usd(m.balance_usd)}"
    supposed = (
        " (Ojo: el saldo es supuesto porque nunca lo anotaste; anotalo para que el cálculo "
        "sea exacto.)"
        if w.assumed
        else ""
    )
    assumed = supposed
    negative = m.negative_warning()
    if negative is not None:
        assumed += f" {negative}"
    unmeasurable = m.unmeasurable
    r = m.restart
    if r is not None:
        since = _short_datetime(r.locked_at, tz)
        unlocked_word = "que desbloqueaste" if m.while_locked else "que ya desbloqueaste"
        after_unlock = (
            f"Después del bloqueo del {since} ({unlocked_word}), la caída se mide de nuevo "
            f"desde el saldo de ese momento ({fmt_usd(r.start_balance_usd)})"
        )
        month_txt = ""
        if round(m.month_drawdown_pct, 2) != round(current, 2) and not (
            unmeasurable and m.month_unmeasurable
        ):
            month_txt = (
                f" En todo el mes la caída es {m.month_drawdown_text()} "
                f"({fmt_usd(m.month_peak_usd)} → {fmt_usd(m.balance_usd)}), pero esa cifra ya no "
                "cuenta para el bloqueo."
            )
        assumed = month_txt + assumed
        if unmeasurable:
            # Arrancó (o quedó) con saldo negativo: no hay máximo del que medir.
            # La caída del mes va aparte y el aviso de revisar los datos, una vez.
            message = (
                f"{after_unlock}: desde entonces el saldo calculado de tu billetera es negativo "
                f"({fmt_usd(m.balance_usd)}), así que esa caída no se puede medir y el freno del "
                f"mes no salta hasta que lo corrijas.{month_txt} {CHECK_LOADED_DATA_TEXT}"
                f"{supposed}"
            )
        elif triggered:
            message = (
                f"{after_unlock} y tu billetera volvió a caer {dd_txt} desde su máximo ({trip}); "
                f"el tope es {top}. La app se vuelve a bloquear: no se generan planes hasta que "
                f"leas el informe nuevo y borres a mano el archivo LOCKED.{assumed}"
            )
        elif r.drawdown_usd <= 0:
            message = (
                f"{after_unlock}: desde entonces tu billetera está en su máximo "
                f"({fmt_usd(r.peak_usd)}). Se vuelve a bloquear solo si cae {top} desde ahí."
                f"{assumed}"
            )
        else:
            message = (
                f"{after_unlock}: desde entonces tu billetera está {dd_txt} abajo de su máximo "
                f"({trip}). Se vuelve a bloquear si la caída llega a {top}.{assumed}"
            )
        summary = (
            f"después del bloqueo del {since}, la billetera volvió a caer {dd_txt} (tope {top})"
            if triggered
            else f"caída desde el bloqueo del {since}: {dd_txt}"
            if unmeasurable
            else f"caída desde el bloqueo del {since}: {dd_txt} de {top}"
        )
    else:
        if unmeasurable:
            message = (
                f"Tu máximo del mes quedó en {fmt_usd(m.peak_usd)} y tu saldo en "
                f"{fmt_usd(m.balance_usd)}.{assumed}"
            )
        elif triggered:
            message = (
                f"Tu billetera cayó {dd_txt} desde su máximo del mes ({trip}); el tope es {top}. "
                "La app se bloquea por completo: no se generan planes hasta que leas el informe "
                f"del incidente y borres a mano el archivo LOCKED.{assumed}"
            )
        elif m.drawdown_usd <= 0:
            message = (
                f"Tu billetera está en su máximo del mes ({fmt_usd(m.peak_usd)}). El "
                f"bloqueo total salta si cae {top} desde el máximo.{assumed}"
            )
        else:
            message = (
                f"Tu billetera está {dd_txt} abajo de su máximo del mes ({trip}). El bloqueo "
                f"total salta en {top}.{assumed}"
            )
        summary = (
            f"la billetera del mes cayó {dd_txt} desde su máximo (tope {top})"
            if triggered
            else f"caída del mes: {dd_txt}"
            if unmeasurable
            else f"caída del mes {dd_txt} de {top}"
        )
    return BreakerStatus(
        code=MONTHLY_DRAWDOWN_LOCK,
        label="Caída del mes: bloqueo total",
        description=(
            f"Si la billetera cae {top} o más desde su máximo del mes, la app se bloquea por "
            "completo: escribe un informe de qué pasó y no genera más planes hasta que borres "
            f"a mano el archivo LOCKED. {_wallet_rules_text(settings)}"
        ),
        kind="lock",
        threshold=threshold,
        current=current,
        unit="%",
        threshold_text=top,
        current_text=UNMEASURABLE_DRAWDOWN_TEXT if unmeasurable else dd_txt,
        progress=_progress(current, threshold),
        triggered=triggered,
        action="lock" if triggered else "none",
        risk_multiplier=1.0,
        until=None,
        until_text="hasta que borres a mano el archivo LOCKED" if triggered else None,
        resets_at=state.periods.month_end,
        message=message,
        summary=summary,
    )


def upcoming_event_warning(event: VolatilityEventRecord, tz: str | tzinfo) -> str:
    """El aviso de un evento de alta volatilidad que todavía no empezó (va en ``warnings``)."""
    return (
        f"Hay un evento de alta volatilidad marcado para el "
        f"{fmt_datetime(event.starts_at, tz)} («{event.note}»): desde esa hora y hasta "
        f"el {fmt_datetime(event.until, tz)} no se van a generar planes."
    )


def _volatility_chain(
    active: tuple[VolatilityEventRecord, ...], upcoming: tuple[VolatilityEventRecord, ...]
) -> tuple[datetime | None, tuple[VolatilityEventRecord, ...]]:
    """Hasta cuándo bloquean los eventos activos, contando los que se encadenan.

    Un evento que todavía no empezó pero arranca antes del fin del bloqueo (o
    justo a esa hora) lo estira hasta su propio fin. Devuelve ``(fin, encadenados)``;
    ``(None, ())`` si no hay ninguno activo.
    """
    if not active:
        return None, ()
    until = max(e.until for e in active)
    chained: list[VolatilityEventRecord] = []
    for e in sorted(upcoming, key=lambda x: (x.starts_at, x.id if x.id is not None else 0)):
        if e.starts_at > until:
            break  # los que siguen empiezan todavía más tarde
        chained.append(e)
        until = max(until, e.until)
    return until, tuple(chained)


def volatility_event_breaker(
    state: TradingState, profile: Any, settings: AppSettings, now: datetime
) -> BreakerStatus:
    """Evento de alta volatilidad activo (``starts_at <= now < until``) → bloqueo hasta el fin."""
    moment = _aware_utc(now, "La hora")
    tz = state.timezone
    active = state.active_volatility_events(moment)
    upcoming = state.upcoming_volatility_events(moment)
    triggered = bool(active)
    until, chained = _volatility_chain(active, upcoming)
    until_text = until_phrase(until, moment, tz) if until is not None else None
    if triggered:
        notes = "; ".join(f"«{e.note}» (hasta el {fmt_datetime(e.until, tz)})" for e in active)
        what = "un evento" if len(active) == 1 else f"{len(active)} eventos"
        then = ""
        if chained:
            nexts = "; ".join(
                f"«{e.note}» (desde el {fmt_datetime(e.starts_at, tz)} hasta el "
                f"{fmt_datetime(e.until, tz)})"
                for e in chained
            )
            then = f" Antes de que termine (o justo a esa hora) empieza otro: {nexts}."
        message = (
            f"Marcaste {what} de alta volatilidad: {notes}.{then} El precio puede moverse muy "
            f"rápido y saltar el stop: no se generan planes {until_text}. Después se apaga solo."
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
            "la hora de fin que cargaste; si otro evento empieza antes de esa hora, sigue hasta "
            "que termine ese."
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
            + ("".join(f" y después «{e.note}»" for e in chained))
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
        rest = ordered[1:]
        if first.code == MONTHLY_DRAWDOWN_LOCK:
            # El archivo LOCKED es el mismo bloqueo, no otro freno.
            rest = [b for b in rest if b.code != LOCK_FILE]
        more = len(rest)
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

    Mientras exista el LOCKED, el freno del mes mira la caída con la que se
    escribió (la del mes calendario o, si antes hubo otro bloqueo este mes que
    el usuario desbloqueó, la medida desde ese bloqueo) y el del archivo
    bloquea. Sin LOCKED, si hay un bloqueo anotado este mes y
    ``restart_after_unlock`` está prendido, mira la caída medida desde ese
    bloqueo. Los próximos eventos de alta volatilidad se avisan en
    ``warnings`` aunque ya haya uno activo, y también un LOCKED de este mes
    que no está anotado en la base.
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
        max_trades_today_breaker(state, profile, settings, moment),
        losing_streak_breaker(state, profile, settings, moment),
        weekly_loss_reduce_breaker(state, profile, settings, moment),
        monthly_drawdown_lock_breaker(
            state, profile, settings, moment, locked=lock.exists, lock=lock
        ),
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
    # Los próximos eventos se avisan siempre, aunque ya haya uno activo.
    for event in state.upcoming_volatility_events(moment):
        warnings.append(upcoming_event_warning(event, tz))
    if lock.exists and not lock.readable:
        warnings.append(
            "El archivo LOCKED existe pero no se pudo leer (¿se editó a mano?): la app sigue "
            "bloqueada igual."
        )
    recorded = lock_is_recorded(lock, state)
    if recorded is False:
        warnings.append(unrecorded_lock_warning(lock, tz))
    measure = month_drawdown_measure(state, settings, locked=lock.exists, lock=lock)
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
        drawdown_measure=measure,
        drawdown_threshold_pct=float(settings.circuit_breakers.monthly_drawdown_lock.pct_from_peak),
        lock_recorded=recorded,
    )


#: Diferencia que se tolera entre la hora del LOCKED y la del bloqueo anotado.
_LOCK_MATCH_SECONDS = 1.0


def _month_lock_of(lock: LockInfo, month_start: datetime, month_end: datetime) -> bool:
    """El LOCKED es de la caída del mes y de este mes (se puede anotar en la base)."""
    return (
        lock.exists
        and lock.readable
        and lock.trigger_code == MONTHLY_DRAWDOWN_LOCK
        and lock.triggered_at is not None
        and month_start <= lock.triggered_at < month_end
    )


def _same_lock(when: datetime, records: Sequence[Any]) -> bool:
    for record in records:
        at = getattr(record, "triggered_at", None)
        if at is None and isinstance(record, dict):
            at = record.get("triggered_at")
        if (
            isinstance(at, datetime)
            and at.tzinfo is not None
            and abs((at - when).total_seconds()) <= _LOCK_MATCH_SECONDS
        ):
            return True
    return False


def lock_is_recorded(lock: LockInfo, state: TradingState) -> bool | None:
    """Si el LOCKED de este mes está anotado en la base (pura, con ``state.month_locks``).

    ``None`` si no hay LOCKED, si no se puede leer, si es de otro freno o de
    otro mes (no hay nada que anotar).
    """
    p = state.periods
    if not _month_lock_of(lock, p.month_start, p.month_end):
        return None
    assert lock.triggered_at is not None
    if lock.triggered_at >= state.now - timedelta(seconds=_LOCK_MATCH_SECONDS):
        # Escrito en esta misma corrida (o "en el futuro"): el estado se leyó
        # antes de anotarlo, no se puede saber todavía.
        return None
    return _same_lock(lock.triggered_at, state.month_locks)


def unrecorded_lock_warning(lock: LockInfo, tz: str | tzinfo | None) -> str:
    """El aviso de un LOCKED de este mes que no está anotado en la base."""
    when = "" if lock.triggered_at is None else f" del {fmt_datetime(lock.triggered_at, tz)}"
    return (
        f"El bloqueo{when} no está anotado en la base (no se pudo anotar). Si borrás el LOCKED "
        "así, la caída se va a medir desde el máximo del mes y la app puede volver a "
        "bloquearse enseguida. La app lo intenta anotar sola en cada corrida; si el aviso "
        "sigue, revisá que la base no esté abierta en otro programa."
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
    - ``lock_record_id``: el número del bloqueo anotado en la base (``None``
      si no se anotó: sin ``repo`` o porque falló).
    - ``db_error``: qué falló con la base (releer los datos del informe o
      anotar el bloqueo). El LOCKED queda escrito igual.
    - ``report_reused``: ``True`` si no se escribió un informe nuevo porque ya
      había uno idéntico (mismo mes, misma medida y mismos datos) de un
      intento anterior, por ejemplo uno en el que no se pudo escribir el LOCKED.
    - ``backfill``: si había un LOCKED de este mes sin anotar en la base, qué
      pasó al anotarlo (``backfill_unrecorded_lock``).
    """

    action: LockTriggerAction
    message: str
    report_path: Path | None = None
    report_error: str | None = None
    lock: LockInfo | None = None
    lock_record_id: int | None = None
    db_error: str | None = None
    report_reused: bool = False
    backfill: LockBackfillResult | None = None

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
                "lock_record_id": self.lock_record_id,
                "db_error": self.db_error,
                "report_reused": self.report_reused,
                "backfill": None if self.backfill is None else self.backfill.to_dict(),
            }
        )


LockBackfillAction = Literal["disabled", "not_applicable", "already_recorded", "recorded", "failed"]


@dataclass(frozen=True)
class LockBackfillResult:
    """Qué hizo ``backfill_unrecorded_lock``.

    - ``action``: ``"disabled"`` (``backfill_unrecorded_lock`` apagado),
      ``"not_applicable"`` (no hay LOCKED de la caída de este mes, no se puede
      leer, o la base no sabe anotar bloqueos), ``"already_recorded"`` (ya
      estaba anotado), ``"recorded"`` (se anotó ahora) o ``"failed"`` (la base
      falló: ``db_error``; se vuelve a intentar en la próxima corrida).
    - ``message``: qué pasó, en castellano simple (``""`` si no hay nada que
      contar).
    - ``lock_record_id``: el número del bloqueo anotado ahora.
    """

    action: LockBackfillAction
    message: str = ""
    lock_record_id: int | None = None
    db_error: str | None = None

    @property
    def recorded(self) -> bool:
        return self.action == "recorded"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "action": self.action,
                "recorded": self.recorded,
                "message": self.message,
                "lock_record_id": self.lock_record_id,
                "db_error": self.db_error,
            }
        )


def backfill_unrecorded_lock(
    repo: Any,
    settings: AppSettings,
    now: datetime,
    *,
    lock_path: Path | str | None = None,
    lock: LockInfo | None = None,
) -> LockBackfillResult:
    """Anota en la base un LOCKED de este mes que no quedó anotado (**con I/O**).

    Pasa cuando el LOCKED lo escribió una versión vieja de la app o cuando la
    base falló justo al anotarlo. Si existe un LOCKED que se puede leer, de la
    caída del mes (``trigger_code`` ``monthly_drawdown_lock``), con fecha de
    este mes (y no en el futuro), y la base no tiene un bloqueo con esa fecha
    (±1 segundo), lo anota con la fecha y los números del LOCKED (si los
    tiene). Así, cuando el usuario lo borre, la caída se vuelve a medir desde
    ese momento (``restart_after_unlock``) en vez de volver a bloquearse.

    ``monthly_drawdown_lock.backfill_unrecorded_lock`` en ``false`` → no hace
    nada. Nunca lanza por la base: un error queda en ``db_error`` (y el
    veredicto lo sigue avisando en cada corrida, ver ``lock_is_recorded``).
    """
    moment = _aware_utc(now, "La hora")
    cb = settings.circuit_breakers
    if not cb.monthly_drawdown_lock.backfill_unrecorded_lock:
        return LockBackfillResult(action="disabled")
    path = Path(default_lock_path(settings) if lock_path is None else lock_path)
    info = read_lock(path) if lock is None else lock
    periods = period_bounds(moment, settings.app.user_timezone, cb.week_starts_on)
    add = getattr(repo, "add_safety_lock", None)
    listing = getattr(repo, "list_safety_locks", None)
    if (
        not _month_lock_of(info, periods.month_start, periods.month_end)
        or not callable(add)
        or not callable(listing)
    ):
        return LockBackfillResult(action="not_applicable")
    when = info.triggered_at
    assert when is not None
    if when > moment:
        return LockBackfillResult(action="not_applicable")
    tz = settings.app.user_timezone
    when_txt = fmt_datetime(when, tz)
    try:
        records = [SafetyLockRecord.from_value(x) for x in listing()]
        if _same_lock(when, records):
            return LockBackfillResult(action="already_recorded")
        stored = add(
            triggered_at=when,
            trigger_code=MONTHLY_DRAWDOWN_LOCK,
            drawdown_pct=info.drawdown_pct,
            peak_usd=info.peak_usd,
            balance_usd=info.balance_usd,
            report_path=info.report_path,
            lock_path=path,
        )
    except Exception as exc:
        logger.warning("No se pudo anotar en la base el LOCKED sin anotar.", exc_info=True)
        return LockBackfillResult(
            action="failed",
            message=(
                f"No se pudo anotar en la base el bloqueo del {when_txt} "
                f"({_error_text(exc)}). La app lo vuelve a intentar en la próxima corrida."
            ),
            db_error=_error_text(exc),
        )
    return LockBackfillResult(
        action="recorded",
        message=(
            f"El bloqueo del {when_txt} no estaba anotado en la base: se anotó ahora con su "
            "fecha. Cuando borres el LOCKED, la caída se va a medir desde el saldo de ese momento."
        ),
        lock_record_id=_record_id(stored),
    )


def _lock_reason(state: TradingState, settings: AppSettings, measure: DrawdownMeasure) -> str:
    """Motivo corto del bloqueo total (va en el informe y en el archivo LOCKED)."""
    w = state.wallet
    top = float(settings.circuit_breakers.monthly_drawdown_lock.pct_from_peak)
    trip = f"({fmt_usd(measure.peak_usd)} → {fmt_usd(measure.balance_usd)})"
    dd = measure.drawdown_text(top)
    if measure.restart is not None:
        since = _short_datetime(measure.restart.locked_at, state.timezone)
        text = (
            f"Después del bloqueo del {since} (que se desbloqueó a mano), la billetera volvió a "
            f"caer {dd} desde su máximo {trip}; el tope es {fmt_percent(top)}."
        )
    else:
        text = (
            f"La billetera del mes cayó {dd} desde su máximo {trip}; el tope es {fmt_percent(top)}."
        )
    if w.assumed:
        text += (
            f" El saldo es supuesto: nunca se anotó, así que se tomó "
            f"{fmt_usd(w.assumed_start_usd or 0.0)} al empezar el mes."
        )
    negative = measure.negative_warning()
    if negative is not None:
        text += f" {negative}"
    return text


def _safe_lock_reason(state: TradingState, settings: AppSettings, measure: DrawdownMeasure) -> str:
    """``_lock_reason`` que nunca falla: el LOCKED se escribe aunque el texto no se pueda armar."""
    try:
        return _lock_reason(state, settings, measure)
    except Exception:  # bloquear es más importante que el texto
        logger.warning("No se pudo armar el motivo del bloqueo total.", exc_info=True)
        return "La billetera del mes cayó demasiado desde su máximo."


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
    info["Depósitos, retiros y desbloqueo"] = _wallet_rules_text(settings)
    if state.wallet.assumed:
        info["Saldo de la billetera"] = (
            f"supuesto: {fmt_usd(state.wallet.assumed_start_usd or 0.0)} al empezar el mes "
            "(nunca se anotó un saldo)"
        )
    return info


def _error_text(exc: BaseException) -> str:
    """``"ValueError: dato raro"`` (el tipo ayuda a encontrar un error inesperado)."""
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _mark_test_analyses(repo: Any, trades: Sequence[Any]) -> tuple[Any, ...]:
    """Las operaciones, con la marca de su análisis si fue de prueba (para el informe).

    Lee cada análisis ligado con ``repo.get_analysis`` (si la base sabe) y,
    si fue con datos sintéticos o con números de prueba
    (``incident.analysis_test_marks``), devuelve esa operación como
    ``IncidentTrade`` con ``analysis_marks``. Las demás quedan igual.
    """
    get = getattr(repo, "get_analysis", None)
    if not callable(get):
        return tuple(trades)
    marks_by_id: dict[int, tuple[str, ...]] = {}
    out: list[Any] = []
    for t in trades:
        aid = _record_id({"id": getattr(t, "analysis_id", None)})
        if aid is None:
            out.append(t)
            continue
        if aid not in marks_by_id:
            stored = get(aid)
            marks_by_id[aid] = analysis_test_marks(getattr(stored, "inputs", None))
        marks = marks_by_id[aid]
        out.append(
            dataclasses.replace(IncidentTrade.from_value(t), analysis_marks=marks) if marks else t
        )
    return tuple(out)


def _record_id(value: Any) -> int | None:
    ident = value.get("id") if isinstance(value, dict) else getattr(value, "id", None)
    return ident if isinstance(ident, int) and not isinstance(ident, bool) else None


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

    - Si ``verdict.needs_lock`` es ``False``: no escribe nada
      (``"not_needed"``).
    - Si ya existe el LOCKED (lo escribió otra corrida, o el usuario no lo
      borró): no escribe nada (``"already_locked"``).
    - En los dos casos, con ``repo``, si el LOCKED de este mes no está anotado
      en la base, lo anota (``backfill_unrecorded_lock``; queda en
      ``backfill``).
    - Si no: primero el informe (``incident.write_incident_report`` en
      ``app.incident_dir``, con las operaciones del mes, los movimientos de la
      billetera, las posiciones abiertas y los números del disparo) y después
      el LOCKED (``lockfile.write_lock`` en ``app.lock_file``), que apunta al
      informe. Si el informe falla por **cualquier** motivo (disco lleno, un
      dato raro, un error del armado), el LOCKED se escribe igual y
      ``report_error`` dice qué pasó. Si ya hay un informe idéntico de un
      intento anterior (misma huella: mismo mes, misma medida y mismos datos),
      se reusa en vez de escribir otro (``report_reused``). Solo un error al
      escribir el LOCKED se propaga, como ``LockWriteError`` (un ``OSError``
      con un mensaje claro): quien llama tiene que enterarse de que el bloqueo
      no quedó escrito. El próximo intento reusa el informe.
    - Escrito el LOCKED, si ``repo`` sabe anotar bloqueos
      (``add_safety_lock``), lo anota en la base con los números del disparo:
      con eso, cuando el usuario borre el LOCKED, la caída se mide de nuevo
      desde el saldo de este momento (``restart_after_unlock``). Si la base
      falla, ``db_error`` lo dice (el LOCKED ya quedó escrito).

    ``repo`` (opcional): si viene, las operaciones del mes, los movimientos y
    las posiciones se vuelven a leer de la base en el momento del bloqueo (si
    la lectura falla, se usan las del ``state``); si no, se usan las del
    ``state`` y el bloqueo no se anota en la base.

    Los números del disparo son los que miró el freno: la caída del mes o,
    si el usuario ya había desbloqueado un bloqueo anterior este mes, la caída
    medida desde ese bloqueo.
    """
    moment = _aware_utc(now, "La hora")
    tz = state.timezone
    path = Path(default_lock_path(settings) if lock_path is None else lock_path)
    if not verdict.needs_lock:
        backfill = (
            None
            if repo is None
            else backfill_unrecorded_lock(repo, settings, moment, lock_path=path)
        )
        return LockTriggerResult(
            action="not_needed",
            message="No hace falta bloquear la app.",
            lock=None,
            backfill=backfill,
        )
    if is_locked(path):
        current = read_lock(path)
        backfill = (
            None
            if repo is None
            else backfill_unrecorded_lock(repo, settings, moment, lock_path=path, lock=current)
        )
        return LockTriggerResult(
            action="already_locked",
            message="La app ya estaba bloqueada: no se escribió nada nuevo.",
            lock=current,
            backfill=backfill,
        )

    db_errors: list[str] = []
    month_start = state.periods.month_start
    trades: tuple[Any, ...] = state.month_trades
    wallet_events: tuple[Any, ...] = state.month_wallet_events
    positions: tuple[Any, ...] = state.open_positions
    if repo is not None:
        try:
            fresh_trades = tuple(
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
            fresh_positions = tuple(
                PositionRecord.from_value(p) for p in repo.list_open_positions()
            )
            fresh_trades = _mark_test_analyses(repo, fresh_trades)
        except Exception as exc:
            logger.warning("No se pudo releer la base para el informe.", exc_info=True)
            db_errors.append(
                f"No se pudo volver a leer la base para el informe ({_error_text(exc)}): el "
                "informe usa los datos del análisis."
            )
        else:
            trades, wallet_events, positions = fresh_trades, tuple(fresh_events), fresh_positions

    measure = month_drawdown_measure(state, settings, locked=False)
    reason = _safe_lock_reason(state, settings, measure)
    since = None if measure.restart is None else measure.restart.locked_at
    folder = (
        settings.resolve_path(settings.app.incident_dir) if incident_dir is None else incident_dir
    )
    report_path: Path | None = None
    report_error: str | None = None
    reused = False
    try:
        trigger = IncidentTrigger(
            code=MONTHLY_DRAWDOWN_LOCK,
            description=reason,
            wallet_peak_usd=measure.peak_usd,
            current_balance_usd=measure.balance_usd,
            drawdown_pct=measure.drawdown_pct,
            threshold_pct=float(settings.circuit_breakers.monthly_drawdown_lock.pct_from_peak),
            measured_since=since,
        )
        report_args: dict[str, Any] = {
            "triggered_at": moment,
            "trigger": trigger,
            "trades": trades,
            "wallet_events": wallet_events,
            "open_positions": positions,
            "profile": state.profile,
            "settings_info": _report_settings_info(state, settings),
            "losing_streak": settings.circuit_breakers.losing_streak.consecutive_losses,
            "lock_path": path,
            "tz": tz,
        }
        # Un intento anterior con los mismos datos (por ejemplo, uno en el que
        # no se pudo escribir el LOCKED) ya dejó este informe: no se repite.
        existing = find_identical_report(folder, incident_fingerprint(**report_args))
        if existing is not None:
            report_path, reused = existing, True
        else:
            report_path = write_incident_report(folder, **report_args)
    except OSError as exc:
        logger.warning("No se pudo escribir el informe del incidente.", exc_info=True)
        report_error = f"No se pudo escribir el informe del incidente: {exc}"
    except Exception as exc:  # un error del armado no puede impedir el bloqueo
        logger.warning("Error inesperado al armar el informe del incidente.", exc_info=True)
        report_error = (
            "No se pudo escribir el informe del incidente por un error inesperado "
            f"({_error_text(exc)}). La app quedó bloqueada igual."
        )
    try:
        written = write_lock(
            path,
            reason=reason,
            triggered_at=moment,
            report_path=report_path,
            trigger_code=MONTHLY_DRAWDOWN_LOCK,
            tz=tz,
            drawdown_pct=measure.drawdown_pct,
            peak_usd=measure.peak_usd,
            balance_usd=measure.balance_usd,
            measured_since=since,
        )
    except LockWriteError:
        raise
    except OSError as exc:
        raise LockWriteError(path, exc, report_path=report_path) from exc
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
            db_error=" ".join(db_errors) or None,
            report_reused=reused,
        )

    lock_record_id: int | None = None
    add_lock = getattr(repo, "add_safety_lock", None) if repo is not None else None
    if callable(add_lock):
        try:
            stored = add_lock(
                triggered_at=moment,
                trigger_code=MONTHLY_DRAWDOWN_LOCK,
                drawdown_pct=measure.drawdown_pct,
                peak_usd=measure.peak_usd,
                balance_usd=measure.balance_usd,
                report_path=report_path,
                lock_path=path,
            )
        except Exception as exc:
            logger.warning("No se pudo anotar el bloqueo total en la base.", exc_info=True)
            db_errors.append(
                f"No se pudo anotar el bloqueo en la base ({_error_text(exc)}). El archivo "
                "LOCKED quedó escrito igual y la app lo intenta anotar sola en las próximas "
                "corridas. Ojo: mientras no quede anotado, si lo borrás, la caída se va a medir "
                "desde el máximo del mes y la app puede volver a bloquearse enseguida."
            )
        else:
            lock_record_id = _record_id(stored)
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
        lock_record_id=lock_record_id,
        db_error=" ".join(db_errors) or None,
        report_reused=reused,
    )


# --- corrida completa (I/O) --------------------------------------------------------------


@dataclass(frozen=True)
class SafetyRun:
    """Resultado de ``run_safety_checks``: el estado, el veredicto y lo que se escribió."""

    state: TradingState
    verdict: SafetyVerdict
    lock_result: LockTriggerResult | None = None
    backfill: LockBackfillResult | None = None

    def to_dict(self) -> dict[str, Any]:
        tz = self.state.timezone
        return json_safe(
            {
                "verdict": self.verdict.to_dict(),
                "state": self.state.to_dict(),
                "lock_trigger": None
                if self.lock_result is None
                else self.lock_result.to_dict(tz=tz),
                "backfill": None if self.backfill is None else self.backfill.to_dict(),
            }
        )


def run_safety_checks(
    repo: Any,
    settings: AppSettings,
    profile: Any,
    now: datetime,
    *,
    lock_path: Path | str | None = None,
    incident_dir: Path | str | None = None,
    trigger_lock: bool = True,
) -> SafetyRun:
    """Lo que hacen las consolas en cada corrida (**con I/O**).

    1. Si hay un LOCKED de este mes sin anotar en la base, lo anota
       (``backfill_unrecorded_lock``).
    2. Lee el estado (``state.load_trading_state``) y evalúa los frenos.
    3. Con ``trigger_lock`` y la caída del mes pidiendo el bloqueo: escribe el
       informe y el LOCKED (``trigger_lock_if_needed``), vuelve a leer el
       estado (para que tenga el bloqueo recién anotado; si la lectura falla,
       sigue con el de antes) y vuelve a evaluar.

    Un error al escribir el LOCKED sale como ``LockWriteError``; uno de la
    base al leer el estado, como lo lance la base.
    """
    moment = _aware_utc(now, "La hora")
    path = Path(default_lock_path(settings) if lock_path is None else lock_path)
    backfill = backfill_unrecorded_lock(repo, settings, moment, lock_path=path)
    state = load_trading_state(repo, settings, profile, moment)
    verdict = evaluate_breakers(state, profile, settings, moment, lock_path=path)
    lock_result: LockTriggerResult | None = None
    if trigger_lock and verdict.needs_lock:
        lock_result = trigger_lock_if_needed(
            verdict,
            state,
            settings,
            moment,
            repo=repo,
            lock_path=path,
            incident_dir=incident_dir,
        )
        try:
            state = load_trading_state(repo, settings, profile, moment)
        except Exception:  # el LOCKED ya está escrito: alcanza con el estado de antes
            logger.warning("No se pudo releer el estado después de bloquear.", exc_info=True)
        verdict = evaluate_breakers(state, profile, settings, moment, lock_path=path)
    return SafetyRun(state=state, verdict=verdict, lock_result=lock_result, backfill=backfill)


__all__ = [
    "BREAKER_CODES",
    "DAILY_LOSS_BLOCK",
    "DAILY_LOSS_REDUCE",
    "LOCK_FILE",
    "LOSING_STREAK",
    "MAX_TRADES_TODAY",
    "MEASURE_MONTH",
    "MEASURE_SINCE_UNLOCK",
    "MONTHLY_DRAWDOWN_LOCK",
    "STATUS_AMBER",
    "STATUS_GREEN",
    "STATUS_RED",
    "VOLATILITY_EVENT",
    "WEEKLY_LOSS_REDUCE",
    "BreakerAction",
    "BreakerStatus",
    "DrawdownMeasure",
    "LockBackfillAction",
    "LockBackfillResult",
    "LockTriggerAction",
    "LockTriggerResult",
    "LockWriteError",
    "SafetyAdjustment",
    "SafetyRun",
    "SafetyVerdict",
    "VerdictStatus",
    "backfill_unrecorded_lock",
    "daily_loss_block_breaker",
    "daily_loss_reduce_breaker",
    "default_lock_path",
    "evaluate_breakers",
    "lock_file_breaker",
    "lock_is_recorded",
    "losing_streak_breaker",
    "max_trades_today_breaker",
    "month_drawdown_measure",
    "monthly_drawdown_lock_breaker",
    "run_safety_checks",
    "trigger_lock_if_needed",
    "unrecorded_lock_warning",
    "until_phrase",
    "upcoming_event_warning",
    "volatility_event_breaker",
    "weekly_loss_reduce_breaker",
]
