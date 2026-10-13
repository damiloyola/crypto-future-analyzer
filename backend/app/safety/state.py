"""Estado de trading cargado a mano (Hito 5).

Junta lo que el usuario carga en la base (operaciones cerradas, posiciones
abiertas, movimientos de la billetera, eventos de alta volatilidad y los
bloqueos totales que escribió la app) y calcula los números que miran los
circuit breakers (``safety.breakers``) y el panel de riesgo: pérdida del día y
de la semana (la actual y la más grande a la que se llegó), operaciones de hoy,
racha de pérdidas con su pausa, saldo de la billetera, su máximo del mes y
cuánto cayó.

En la fase 4 este módulo se reemplaza por datos reales de la cuenta, con la
misma interfaz (``TradingState``).

Qué hay acá
    - ``load_trading_state(repo, settings, profile, now)``: **la única función
      con I/O**: lee la base (``storage.Repository`` o cualquier objeto con los
      mismos métodos, ver ``TradingStateSource``) y llama a
      ``build_trading_state``.
    - ``build_trading_state(*, trades, open_positions, wallet_events,
      volatility_events, settings, profile, now, safety_locks=())``: arma
      ``TradingState``. Pura.
    - Funciones puras sueltas (las usa ``build_trading_state`` y se pueden
      probar por separado): ``period_bounds``, ``period_pnl``,
      ``count_opened_in``, ``losing_streak_state`` y ``wallet_state``.

Este módulo no importa ``storage`` (la capa de seguridad está debajo): lee los
registros por "duck typing" (atributos o claves de un dict con los nombres de
``storage``: ``StoredTrade``, ``StoredPosition``, ``StoredWalletEvent``,
``StoredVolatilityEvent`` y ``StoredSafetyLock``) y los copia a sus propias
dataclasses inmutables.
Un registro inválido (fecha sin zona horaria, resultado que no es número)
levanta ``ValueError``: un freno de seguridad no puede ignorar en silencio una
pérdida que no entiende.

Reglas de cálculo (todos los números de ``config.yaml``)
    - **Zona horaria**: la del usuario (``app.user_timezone``). El día es el día
      calendario local (de 00:00 a 00:00); la semana empieza el
      ``circuit_breakers.week_starts_on`` (lunes) a las 00:00; el mes es el mes
      calendario (``circuit_breakers.month_starts: calendar``). Los períodos son
      semiabiertos: una operación cerrada a las 00:00 en punto cuenta para el
      día que empieza.
    - **Pérdida del día y de la semana** (decisión del usuario,
      ``circuit_breakers.loss_basis``): con ``"net"`` es el neto de lo cerrado
      en el período (ganancias menos pérdidas, comisiones incluidas); si el neto
      es ganancia, la pérdida es 0. Con ``"gross"`` es la suma de las
      operaciones perdidas, sin descontar las ganadas. Las sumas se redondean a
      centavos.
    - **Pérdida más grande del período** (``PeriodPnl.max_loss_usd``, para
      ``circuit_breakers.latch_until_period_end``): recorre lo cerrado en el
      período en orden de cierre y se queda con el peor neto acumulado (con
      ``"net"``) o con la suma de las pérdidas (con ``"gross"``, que solo
      crece). Si hoy perdiste 60 y después ganaste 20, la pérdida actual es 40
      y la más grande, 60: los frenos del día miran la más grande (siguen hasta
      que termina el día) y el plan, la actual.
    - **Operaciones de hoy** (para ``max_trades_per_day``): las cerradas que se
      **abrieron** hoy más las posiciones abiertas que se abrieron hoy.
    - **Racha de pérdidas**: operaciones seguidas con resultado negativo (a
      centavos, como el informe del incidente), en orden de cierre; una ganada o
      un empate la cortan. Cuando llega a ``losing_streak.consecutive_losses``
      empieza una pausa de ``cooldown_hours`` desde el cierre de esa pérdida. Si
      se pierde otra vez durante la pausa (la racha sigue), la pausa se estira
      ``cooldown_hours`` desde esa pérdida (``extend_cooldown_on_loss``,
      decisión del usuario; con ``false`` termina a la hora prevista). Con
      ``reset_after_cooldown`` (decisión del usuario),
      terminada la pausa la cuenta vuelve a cero: las pérdidas cerradas antes del
      fin de la pausa ya no cuentan.
    - **Billetera**: el saldo en cada momento es el último "saldo" anotado más
      depósitos, menos retiros, más el resultado de las operaciones cerradas
      después. Un saldo anotado incluye todo lo que pasó hasta ese momento
      inclusive. Antes del primer saldo anotado, el saldo se calcula hacia atrás
      desde él (si el usuario anotó el saldo el día 15, el del día 1 sale
      restando lo que pasó en el medio). Si nunca anotó un saldo, se supone que
      empezó el mes con la billetera mensual del perfil
      (``monthly_wallet_usd``) y se avisa.
    - **Máximo del mes** (decisiones del usuario,
      ``circuit_breakers.monthly_drawdown_lock``): arranca en el saldo del
      primer instante del mes calendario y sube cuando el saldo lo pasa (una
      ganancia o un saldo anotado más alto). Un **depósito** sube el máximo en
      lo mismo que se depositó (``deposit_raises_peak``): la pérdida de antes
      sigue contando y depositar no esquiva el bloqueo (con ``false``, el
      máximo sube solo si el saldo lo pasa). Un **retiro**
      (``withdrawal_peak``): ``"proportional"`` baja el máximo en la misma
      proporción que el saldo (máximo × saldo después / saldo antes; sacar
      plata nunca cambia el % de caída; si el saldo antes del retiro ya era
      0 o negativo, el máximo no cambia); ``"subtract"`` lo baja en lo mismo
      que se retiró; ``"none"`` no lo toca (el retiro cuenta como caída). Un
      saldo anotado más bajo que el calculado cuenta como caída. La caída es
      ``(máximo - saldo actual) / máximo`` en %.
    - **Caída después de desbloquear** (``restart_after_unlock``, decisión del
      usuario): si en el mes hay un bloqueo total anotado en la base, también
      se mide la caída desde el saldo justo después del último bloqueo (la
      pérdida que lo disparó incluida), como si empezara un mes nuevo
      (``WalletState.restart``). Los frenos la usan cuando el usuario ya borró
      el LOCKED. Los bloqueos de meses anteriores no cuentan.
    - Lo que pasó después de ``now`` no cuenta (por ejemplo, con un reloj de
      prueba).

Todas las fechas que salen están en UTC con zona horaria; los ``to_dict()``
están listos para JSON (fechas en ISO 8601) y traen textos en castellano para
mostrar sin formatear nada en el frontend.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from app.config import AppSettings
from app.risk.rules import fmt_usd, json_safe
from app.safety.incident import fmt_month

LossBasis = Literal["net", "gross"]
WithdrawalPeak = Literal["proportional", "subtract", "none"]
WITHDRAWAL_PEAK_RULES: tuple[str, ...] = ("proportional", "subtract", "none")

#: Día de la semana de ``circuit_breakers.week_starts_on`` → ``date.weekday()``.
WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

WALLET_KINDS: tuple[str, ...] = ("deposit", "withdrawal", "balance")

_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")

# Tipos de punto en la historia de la billetera.
POINT_MONTH_START = "month_start"
POINT_ASSUMED_START = "assumed_start"
POINT_BALANCE = "balance"
POINT_DEPOSIT = "deposit"
POINT_WITHDRAWAL = "withdrawal"
POINT_TRADE = "trade"


# --- lectura de registros (duck typing) -------------------------------------------


class TradingStateSource(Protocol):
    """Lo que ``load_trading_state`` necesita de la base (lo cumple ``storage.Repository``)."""

    def list_trades(self) -> Sequence[Any]: ...

    def list_open_positions(self) -> Sequence[Any]: ...

    def list_wallet_events(self) -> Sequence[Any]: ...

    def list_volatility_events(self, *, ending_after: datetime | None = None) -> Sequence[Any]: ...


class SafetyLockSource(Protocol):
    """Opcional: la base sabe listar los bloqueos totales (``Repository.list_safety_locks``).

    ``load_trading_state`` la usa si el objeto tiene ese método; si no, hace
    como si no hubiera bloqueos anotados.
    """

    def list_safety_locks(self) -> Sequence[Any]: ...


def _field(obj: Any, name: str) -> Any:
    """Atributo o clave ``name`` (dict, dataclass u objeto); ``None`` si no está."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _aware_utc(value: Any, label: str) -> datetime:
    """Fecha con zona horaria pasada a UTC. Sin zona o que no es fecha: ``ValueError``."""
    if not isinstance(value, datetime):
        raise ValueError(f"{label} tiene que ser una fecha, llegó {value!r}.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{label} no tiene zona horaria ({value.isoformat()}): no se sabe a qué día cuenta."
        )
    return value.astimezone(UTC)


def _opt_aware_utc(value: Any, label: str) -> datetime | None:
    return None if value is None else _aware_utc(value, label)


def _number(value: Any, label: str) -> float:
    """Número finito (``True``/``False`` no cuentan). Si no, ``ValueError``."""
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        value = item()  # escalar de numpy
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} tiene que ser un número, llegó {value!r}.")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} tiene que ser un número finito, llegó {value!r}.")
    return v


def _opt_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cents(value: float) -> float:
    """Redondeo a centavos sin ``-0.0``."""
    v = round(float(value), 2)
    return 0.0 if v == 0 else v


def _side(value: Any) -> str | None:
    text = _opt_text(value)
    return None if text is None else text.lower()


@dataclass(frozen=True)
class TradeRecord:
    """Operación cerrada (copia de ``StoredTrade``, con el contexto del análisis si lo hubo).

    ``pnl_usd`` es el resultado neto (negativo = pérdida). Tiene los mismos
    nombres de campo que ``StoredTrade``, así ``safety.incident`` la lee igual.
    """

    id: int | None
    symbol: str
    side: str | None
    opened_at: datetime | None
    closed_at: datetime
    pnl_usd: float
    risk_usd: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    notes: str | None = None
    analysis_id: int | None = None
    regime_category: str | None = None
    regime_name: str | None = None
    regime_confidence: float | None = None
    score_total: float | None = None
    plan_status: str | None = None
    profile_name: str | None = None

    @property
    def is_loss(self) -> bool:
        """Resultado negativo a centavos (un empate o -0,004 USD no es pérdida)."""
        return round(self.pnl_usd, 2) < 0

    @classmethod
    def from_value(cls, value: Any) -> TradeRecord:
        """Desde un ``StoredTrade``, un dict o un objeto con esos atributos."""
        if isinstance(value, TradeRecord):
            return value
        ident = _opt_int(_field(value, "id"))
        label = f"La operación {ident}" if ident is not None else "Una operación"
        return cls(
            id=ident,
            symbol=(_opt_text(_field(value, "symbol")) or "?").upper(),
            side=_side(_field(value, "side")),
            opened_at=_opt_aware_utc(_field(value, "opened_at"), f"{label}: la apertura"),
            closed_at=_aware_utc(_field(value, "closed_at"), f"{label}: el cierre"),
            pnl_usd=_number(_field(value, "pnl_usd"), f"{label}: el resultado"),
            risk_usd=_opt_number(_field(value, "risk_usd"), f"{label}: el riesgo"),
            entry_price=_opt_number(_field(value, "entry_price"), f"{label}: la entrada"),
            exit_price=_opt_number(_field(value, "exit_price"), f"{label}: la salida"),
            notes=_opt_text(_field(value, "notes")),
            analysis_id=_opt_int(_field(value, "analysis_id")),
            regime_category=_opt_text(_field(value, "regime_category")),
            regime_name=_opt_text(_field(value, "regime_name")),
            regime_confidence=_opt_number(
                _field(value, "regime_confidence"), f"{label}: la confianza del régimen"
            ),
            score_total=_opt_number(_field(value, "score_total"), f"{label}: el score"),
            plan_status=_opt_text(_field(value, "plan_status")),
            profile_name=_opt_text(_field(value, "profile_name")),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "symbol": self.symbol,
                "side": self.side,
                "opened_at": _iso(self.opened_at),
                "closed_at": _iso(self.closed_at),
                "pnl_usd": self.pnl_usd,
                "is_loss": self.is_loss,
                "risk_usd": self.risk_usd,
                "entry_price": self.entry_price,
                "exit_price": self.exit_price,
                "notes": self.notes,
                "analysis_id": self.analysis_id,
                "regime_category": self.regime_category,
                "regime_name": self.regime_name,
                "regime_confidence": self.regime_confidence,
                "score_total": self.score_total,
                "plan_status": self.plan_status,
                "profile_name": self.profile_name,
            }
        )


@dataclass(frozen=True)
class PositionRecord:
    """Posición abierta (copia de ``StoredPosition``). ``risk_usd``: lo que se pierde en el stop."""

    id: int | None
    symbol: str
    side: str | None
    opened_at: datetime | None
    risk_usd: float
    entry_price: float | None = None
    notes: str | None = None
    analysis_id: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> PositionRecord:
        """Desde un ``StoredPosition``, un dict o un objeto con esos atributos."""
        if isinstance(value, PositionRecord):
            return value
        ident = _opt_int(_field(value, "id"))
        label = f"La posición {ident}" if ident is not None else "Una posición abierta"
        return cls(
            id=ident,
            symbol=(_opt_text(_field(value, "symbol")) or "?").upper(),
            side=_side(_field(value, "side")),
            opened_at=_opt_aware_utc(_field(value, "opened_at"), f"{label}: la apertura"),
            risk_usd=_number(_field(value, "risk_usd"), f"{label}: el riesgo"),
            entry_price=_opt_number(_field(value, "entry_price"), f"{label}: la entrada"),
            notes=_opt_text(_field(value, "notes")),
            analysis_id=_opt_int(_field(value, "analysis_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "symbol": self.symbol,
                "side": self.side,
                "opened_at": _iso(self.opened_at),
                "risk_usd": self.risk_usd,
                "entry_price": self.entry_price,
                "notes": self.notes,
                "analysis_id": self.analysis_id,
            }
        )


@dataclass(frozen=True)
class WalletRecord:
    """Movimiento de la billetera (copia de ``StoredWalletEvent``).

    ``kind``: ``"deposit"``, ``"withdrawal"`` (``amount_usd`` positivo, el signo
    lo da el tipo) o ``"balance"`` (el saldo que el usuario vio en ese momento).
    """

    id: int | None
    ts: datetime
    kind: str
    amount_usd: float
    notes: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> WalletRecord:
        """Desde un ``StoredWalletEvent``, un dict o un objeto con esos atributos."""
        if isinstance(value, WalletRecord):
            return value
        ident = _opt_int(_field(value, "id"))
        label = f"El movimiento {ident}" if ident is not None else "Un movimiento de la billetera"
        kind = (_opt_text(_field(value, "kind")) or "").lower()
        if kind not in WALLET_KINDS:
            raise ValueError(
                f"{label} tiene un tipo desconocido ({_field(value, 'kind')!r}): tiene que ser "
                "'deposit', 'withdrawal' o 'balance'."
            )
        amount = _number(_field(value, "amount_usd"), f"{label}: el monto")
        if amount < 0:
            raise ValueError(f"{label}: el monto no puede ser negativo ({amount:g}).")
        return cls(
            id=ident,
            ts=_aware_utc(_field(value, "ts"), f"{label}: la fecha"),
            kind=kind,
            amount_usd=amount,
            notes=_opt_text(_field(value, "notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "ts": _iso(self.ts),
                "kind": self.kind,
                "amount_usd": self.amount_usd,
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class VolatilityEventRecord:
    """Evento de alta volatilidad (copia de ``StoredVolatilityEvent``).

    Bloquea mientras ``starts_at <= now < until``.
    """

    id: int | None
    starts_at: datetime
    until: datetime
    note: str
    created_at: datetime | None = None

    def is_active(self, now: datetime) -> bool:
        """``True`` si bloquea en ``now`` (a la hora de fin ya no bloquea)."""
        moment = _aware_utc(now, "La hora")
        return self.starts_at <= moment < self.until

    def is_upcoming(self, now: datetime) -> bool:
        """``True`` si todavía no empezó en ``now``."""
        return _aware_utc(now, "La hora") < self.starts_at

    @classmethod
    def from_value(cls, value: Any) -> VolatilityEventRecord:
        """Desde un ``StoredVolatilityEvent``, un dict o un objeto con esos atributos."""
        if isinstance(value, VolatilityEventRecord):
            return value
        ident = _opt_int(_field(value, "id"))
        label = f"El evento {ident}" if ident is not None else "Un evento de alta volatilidad"
        return cls(
            id=ident,
            starts_at=_aware_utc(_field(value, "starts_at"), f"{label}: el inicio"),
            until=_aware_utc(_field(value, "until"), f"{label}: la hora de fin"),
            note=_opt_text(_field(value, "note")) or "sin nota",
            created_at=_opt_aware_utc(_field(value, "created_at"), f"{label}: la creación"),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "starts_at": _iso(self.starts_at),
                "until": _iso(self.until),
                "note": self.note,
                "created_at": _iso(self.created_at),
            }
        )


@dataclass(frozen=True)
class SafetyLockRecord:
    """Bloqueo total que escribió la app (copia de ``StoredSafetyLock``).

    ``triggered_at``: cuándo se escribió el LOCKED. Los números son los del
    disparo (``None`` si no se anotaron).
    """

    id: int | None
    triggered_at: datetime
    trigger_code: str
    drawdown_pct: float | None = None
    peak_usd: float | None = None
    balance_usd: float | None = None
    report_path: str | None = None
    lock_path: str | None = None
    created_at: datetime | None = None

    @classmethod
    def from_value(cls, value: Any) -> SafetyLockRecord:
        """Desde un ``StoredSafetyLock``, un dict o un objeto con esos atributos."""
        if isinstance(value, SafetyLockRecord):
            return value
        ident = _opt_int(_field(value, "id"))
        label = f"El bloqueo {ident}" if ident is not None else "Un bloqueo total"
        return cls(
            id=ident,
            triggered_at=_aware_utc(_field(value, "triggered_at"), f"{label}: la hora"),
            trigger_code=_opt_text(_field(value, "trigger_code")) or "?",
            drawdown_pct=_opt_number(_field(value, "drawdown_pct"), f"{label}: la caída"),
            peak_usd=_opt_number(_field(value, "peak_usd"), f"{label}: el máximo"),
            balance_usd=_opt_number(_field(value, "balance_usd"), f"{label}: el saldo"),
            report_path=_opt_text(_field(value, "report_path")),
            lock_path=_opt_text(_field(value, "lock_path")),
            created_at=_opt_aware_utc(_field(value, "created_at"), f"{label}: la creación"),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "triggered_at": _iso(self.triggered_at),
                "trigger_code": self.trigger_code,
                "drawdown_pct": self.drawdown_pct,
                "peak_usd": self.peak_usd,
                "balance_usd": self.balance_usd,
                "report_path": self.report_path,
                "lock_path": self.lock_path,
                "created_at": _iso(self.created_at),
            }
        )


@dataclass(frozen=True)
class ProfileLimits:
    """Lo que el estado y los breakers usan del perfil de riesgo (``EffectiveProfile``)."""

    name: str | None
    monthly_wallet_usd: float
    risk_per_trade_usd: float | None = None
    max_daily_loss_usd: float | None = None
    max_trades_per_day: int | None = None
    min_operable_risk_usd: float | None = None
    max_leverage: float | None = None

    @classmethod
    def from_value(cls, profile: Any) -> ProfileLimits:
        """Desde un ``EffectiveProfile``, un ``RiskProfile``, un dict o algo con esos atributos."""
        if isinstance(profile, ProfileLimits):
            return profile
        wallet = _number(_field(profile, "monthly_wallet_usd"), "La billetera mensual del perfil")
        if wallet <= 0:
            raise ValueError(f"La billetera mensual del perfil tiene que ser > 0 ({wallet:g}).")
        trades = _opt_number(_field(profile, "max_trades_per_day"), "Las operaciones por día")
        return cls(
            name=_opt_text(_field(profile, "name")),
            monthly_wallet_usd=wallet,
            risk_per_trade_usd=_opt_number(
                _field(profile, "risk_per_trade_usd"), "El riesgo por operación"
            ),
            max_daily_loss_usd=_opt_number(
                _field(profile, "max_daily_loss_usd"), "La pérdida máxima del día"
            ),
            max_trades_per_day=None if trades is None else int(trades),
            min_operable_risk_usd=_opt_number(
                _field(profile, "min_operable_risk_usd"), "El riesgo mínimo operable"
            ),
            max_leverage=_opt_number(_field(profile, "max_leverage"), "El apalancamiento máximo"),
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dict(self.__dict__))


# --- períodos ----------------------------------------------------------------------


def resolve_user_timezone(name: str | tzinfo) -> tzinfo:
    """Zona horaria IANA (``"America/Argentina/Buenos_Aires"``); desconocida → ``ValueError``.

    A diferencia del informe del incidente, acá no se cae a UTC: con la zona
    equivocada las pérdidas contarían para otro día.
    """
    if isinstance(name, tzinfo):
        return name
    text = _opt_text(name)
    if text is None:
        raise ValueError("Falta la zona horaria del usuario (app.user_timezone).")
    try:
        return ZoneInfo(text)
    except (ValueError, KeyError, OSError) as exc:  # ZoneInfoNotFoundError es KeyError
        raise ValueError(
            f"Zona horaria desconocida: {text!r} (app.user_timezone). Usá un nombre como "
            "'America/Argentina/Buenos_Aires'."
        ) from exc


def _local_midnight(day: date, zone: tzinfo) -> datetime:
    """00:00 del día ``day`` en la zona ``zone``, en UTC."""
    return datetime.combine(day, time(0), tzinfo=zone).astimezone(UTC)


def _first_of_next_month(day: date) -> date:
    return date(day.year + 1, 1, 1) if day.month == 12 else date(day.year, day.month + 1, 1)


def _short_day(day: date) -> str:
    return f"{_WEEKDAYS[day.weekday()]} {day:%d/%m}"


@dataclass(frozen=True)
class Periods:
    """Día, semana y mes del usuario que contienen ``now`` (bordes en UTC, semiabiertos)."""

    timezone: str
    local_date: date
    day_start: datetime
    day_end: datetime
    week_start: datetime
    week_end: datetime
    month_start: datetime
    month_end: datetime

    def contains_day(self, moment: datetime) -> bool:
        """``True`` si ``moment`` cae en el día de este estado."""
        return self.day_start <= _aware_utc(moment, "La hora") < self.day_end

    def labels(self) -> dict[str, str]:
        """Textos para mostrar: ``"martes 13/10"``, ``"del lunes 12/10 al domingo 18/10"``..."""
        zone = resolve_user_timezone(self.timezone)
        week_first = self.week_start.astimezone(zone).date()
        week_last = self.week_end.astimezone(zone).date() - timedelta(days=1)
        return {
            "day": _short_day(self.local_date),
            "week": f"del {_short_day(week_first)} al {_short_day(week_last)}",
            "month": fmt_month(self.month_start, zone),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "timezone": self.timezone,
            "local_date": self.local_date.isoformat(),
            "day_start": _iso(self.day_start),
            "day_end": _iso(self.day_end),
            "week_start": _iso(self.week_start),
            "week_end": _iso(self.week_end),
            "month_start": _iso(self.month_start),
            "month_end": _iso(self.month_end),
            "labels": self.labels(),
        }


def period_bounds(now: datetime, tz: str | tzinfo, week_starts_on: str = "monday") -> Periods:
    """Día, semana y mes calendario de ``now`` en la zona ``tz``.

    La semana empieza el ``week_starts_on`` (``"monday"``...) a las 00:00 local.
    """
    moment = _aware_utc(now, "La hora actual")
    zone = resolve_user_timezone(tz)
    if week_starts_on not in WEEKDAY_INDEX:
        raise ValueError(
            f"Día de inicio de semana desconocido: {week_starts_on!r} "
            f"(opciones: {', '.join(WEEKDAY_INDEX)})."
        )
    today = moment.astimezone(zone).date()
    week_first = today - timedelta(days=(today.weekday() - WEEKDAY_INDEX[week_starts_on]) % 7)
    month_first = today.replace(day=1)
    return Periods(
        timezone=str(getattr(zone, "key", None) or tz),
        local_date=today,
        day_start=_local_midnight(today, zone),
        day_end=_local_midnight(today + timedelta(days=1), zone),
        week_start=_local_midnight(week_first, zone),
        week_end=_local_midnight(week_first + timedelta(days=7), zone),
        month_start=_local_midnight(month_first, zone),
        month_end=_local_midnight(_first_of_next_month(month_first), zone),
    )


# --- resultado de un período ---------------------------------------------------------


@dataclass(frozen=True)
class PeriodPnl:
    """Resultado de las operaciones **cerradas** en ``[start, end)``.

    - ``net_pnl_usd``: neto (ganancias menos pérdidas), siempre.
    - ``gross_profit_usd`` / ``gross_loss_usd``: suma de las ganadas y de las
      perdidas (las dos positivas).
    - ``loss_usd``: la pérdida **actual** según ``loss_basis``: ``"net"`` →
      ``max(0, -neto)``; ``"gross"`` → ``gross_loss_usd``.
    - ``max_loss_usd``: la pérdida **más grande** a la que se llegó en el
      período: con ``"net"``, el peor neto acumulado recorriendo las
      operaciones en orden de cierre (si perdiste 60 y después ganaste 20, es
      60 aunque ``loss_usd`` sea 40); con ``"gross"`` es ``gross_loss_usd``
      (solo crece). Siempre ``>= loss_usd``.
    """

    start: datetime
    end: datetime
    loss_basis: LossBasis
    trades: int
    wins: int
    losses: int
    net_pnl_usd: float
    gross_profit_usd: float
    gross_loss_usd: float
    loss_usd: float
    max_loss_usd: float = 0.0

    @property
    def breakeven(self) -> int:
        return self.trades - self.wins - self.losses

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "start": _iso(self.start),
                "end": _iso(self.end),
                "loss_basis": self.loss_basis,
                "trades": self.trades,
                "wins": self.wins,
                "losses": self.losses,
                "breakeven": self.breakeven,
                "net_pnl_usd": self.net_pnl_usd,
                "gross_profit_usd": self.gross_profit_usd,
                "gross_loss_usd": self.gross_loss_usd,
                "loss_usd": self.loss_usd,
                "max_loss_usd": self.max_loss_usd,
            }
        )


def period_pnl(
    trades: Iterable[Any],
    start: datetime,
    end: datetime,
    *,
    loss_basis: LossBasis = "net",
    now: datetime | None = None,
) -> PeriodPnl:
    """Resultado de las operaciones con ``start <= closed_at < end`` (y ``<= now`` si se pasa).

    Ver ``PeriodPnl``. Las sumas se redondean a centavos.
    """
    if loss_basis not in ("net", "gross"):
        raise ValueError(f"Base de pérdida desconocida: {loss_basis!r} ('net' o 'gross').")
    lo = _aware_utc(start, "El inicio del período")
    hi = _aware_utc(end, "El fin del período")
    cutoff = None if now is None else _aware_utc(now, "La hora actual")
    selected = [
        t
        for t in _sorted_by_close(trades)
        if lo <= t.closed_at < hi and (cutoff is None or t.closed_at <= cutoff)
    ]
    wins = [t for t in selected if round(t.pnl_usd, 2) > 0]
    losses = [t for t in selected if t.is_loss]
    net = _cents(sum(t.pnl_usd for t in selected))
    gross_profit = _cents(sum(t.pnl_usd for t in wins))
    gross_loss = _cents(-sum(t.pnl_usd for t in losses))
    loss = max(0.0, -net) if loss_basis == "net" else gross_loss
    # El peor neto acumulado, en orden de cierre (empieza en 0: sin operaciones
    # no hay pérdida).
    running = worst = 0.0
    for t in selected:
        running += t.pnl_usd
        worst = min(worst, round(running, 2))
    max_loss = max(0.0, -worst) if loss_basis == "net" else gross_loss
    return PeriodPnl(
        start=lo,
        end=hi,
        loss_basis=loss_basis,
        trades=len(selected),
        wins=len(wins),
        losses=len(losses),
        net_pnl_usd=net,
        gross_profit_usd=gross_profit,
        gross_loss_usd=gross_loss,
        loss_usd=_cents(loss),
        max_loss_usd=_cents(max(max_loss, loss)),
    )


def count_opened_in(records: Iterable[Any], start: datetime, end: datetime) -> int:
    """Cuántos registros (operaciones o posiciones) se **abrieron** en ``[start, end)``."""
    lo = _aware_utc(start, "El inicio del período")
    hi = _aware_utc(end, "El fin del período")
    count = 0
    for record in records:
        opened = _opt_aware_utc(_field(record, "opened_at"), "La apertura")
        if opened is not None and lo <= opened < hi:
            count += 1
    return count


# --- racha de pérdidas -------------------------------------------------------------


@dataclass(frozen=True)
class LosingStreakState:
    """Racha de pérdidas seguidas y la pausa (enfriamiento) que dispara.

    - ``count``: pérdidas seguidas que cuentan en ``as_of`` (con
      ``reset_after_cooldown``, terminada la pausa vuelve a 0).
    - ``loss_trade_ids``: las operaciones de esa racha.
    - ``cooldown_started_at`` / ``cooldown_until``: la última pausa (aunque ya
      haya terminado; ``None`` si nunca hubo una).
    - ``extend_cooldown_on_loss``: si una pérdida durante la pausa la estira.
    """

    as_of: datetime
    count: int
    required: int
    cooldown_hours: float
    reset_after_cooldown: bool
    loss_trade_ids: tuple[int | None, ...]
    last_loss_at: datetime | None
    cooldown_started_at: datetime | None
    cooldown_until: datetime | None
    extend_cooldown_on_loss: bool = True

    def cooldown_active(self, now: datetime | None = None) -> bool:
        """``True`` si la pausa está corriendo en ``now`` (por defecto, ``as_of``)."""
        moment = self.as_of if now is None else _aware_utc(now, "La hora")
        return (
            self.cooldown_started_at is not None
            and self.cooldown_until is not None
            and self.cooldown_started_at <= moment < self.cooldown_until
        )

    def count_at(self, now: datetime) -> int:
        """Las pérdidas seguidas que cuentan en ``now``.

        Si la pausa termina entre ``as_of`` y ``now`` y la cuenta se reinicia al
        terminar, da 0 (no hay operaciones nuevas después de ``as_of``).
        """
        moment = _aware_utc(now, "La hora")
        if (
            self.reset_after_cooldown
            and self.cooldown_until is not None
            and self.as_of < self.cooldown_until <= moment
        ):
            return 0
        return self.count

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "as_of": _iso(self.as_of),
                "count": self.count,
                "required": self.required,
                "cooldown_hours": self.cooldown_hours,
                "reset_after_cooldown": self.reset_after_cooldown,
                "extend_cooldown_on_loss": self.extend_cooldown_on_loss,
                "loss_trade_ids": list(self.loss_trade_ids),
                "last_loss_at": _iso(self.last_loss_at),
                "cooldown_started_at": _iso(self.cooldown_started_at),
                "cooldown_until": _iso(self.cooldown_until),
                "cooldown_active": self.cooldown_active(),
            }
        )


def _sorted_by_close(trades: Iterable[Any]) -> list[TradeRecord]:
    items = [TradeRecord.from_value(t) for t in trades]
    return sorted(items, key=lambda t: (t.closed_at, t.id if t.id is not None else 0))


def losing_streak_state(
    trades: Iterable[Any],
    now: datetime,
    *,
    consecutive_losses: int,
    cooldown_hours: float,
    reset_after_cooldown: bool = True,
    extend_cooldown_on_loss: bool = True,
) -> LosingStreakState:
    """Racha de pérdidas en ``now`` y su pausa (ver las reglas en el docstring del módulo).

    Recorre todas las operaciones cerradas hasta ``now`` en orden de cierre:
    el estado de hoy depende de cuándo empezaron y terminaron las pausas
    anteriores. Con ``extend_cooldown_on_loss``, una pérdida durante la pausa
    la estira ``cooldown_hours`` desde esa pérdida; sin eso, la pausa termina a
    la hora prevista (la pérdida igual suma a la racha).
    """
    moment = _aware_utc(now, "La hora actual")
    required = int(consecutive_losses)
    if required < 1:
        raise ValueError("consecutive_losses tiene que ser 1 o más.")
    hours = _number(cooldown_hours, "Las horas de pausa")
    if hours <= 0:
        raise ValueError("Las horas de pausa tienen que ser > 0.")
    pause = timedelta(hours=hours)

    count = 0
    ids: list[int | None] = []
    last_loss_at: datetime | None = None
    cd_start: datetime | None = None
    cd_until: datetime | None = None
    pending_reset = False
    for trade in _sorted_by_close(trades):
        if trade.closed_at > moment:
            break
        if pending_reset and cd_until is not None and trade.closed_at >= cd_until:
            # Terminó la pausa: las pérdidas de antes ya no cuentan.
            count, ids, pending_reset = 0, [], False
        if not trade.is_loss:
            count, ids = 0, []
            continue
        count += 1
        ids.append(trade.id)
        last_loss_at = trade.closed_at
        if count >= required:
            running = cd_start is not None and cd_until is not None and trade.closed_at < cd_until
            if not running:
                cd_start = trade.closed_at
                cd_until = trade.closed_at + pause
            elif extend_cooldown_on_loss:
                cd_until = trade.closed_at + pause
            pending_reset = reset_after_cooldown
    if pending_reset and cd_until is not None and moment >= cd_until:
        count, ids = 0, []
    return LosingStreakState(
        as_of=moment,
        count=count,
        required=required,
        cooldown_hours=hours,
        reset_after_cooldown=reset_after_cooldown,
        loss_trade_ids=tuple(ids),
        last_loss_at=last_loss_at,
        cooldown_started_at=cd_start,
        cooldown_until=cd_until,
        extend_cooldown_on_loss=extend_cooldown_on_loss,
    )


# --- billetera ----------------------------------------------------------------------


@dataclass(frozen=True)
class WalletPoint:
    """Un punto de la historia de la billetera en el mes (para el gráfico del panel).

    ``kind``: ``"month_start"`` / ``"assumed_start"`` (saldo al empezar el mes,
    anotado o supuesto), ``"balance"`` (saldo anotado), ``"deposit"``,
    ``"withdrawal"`` o ``"trade"``. ``delta_usd`` es el cambio (``None`` en los
    saldos); ``balance_usd`` y ``peak_usd``, el saldo y el máximo después del punto.
    """

    ts: datetime
    kind: str
    delta_usd: float | None
    balance_usd: float
    peak_usd: float
    ref_id: int | None = None
    symbol: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "ts": _iso(self.ts),
                "kind": self.kind,
                "delta_usd": self.delta_usd,
                "balance_usd": self.balance_usd,
                "peak_usd": self.peak_usd,
                "ref_id": self.ref_id,
                "symbol": self.symbol,
            }
        )


@dataclass(frozen=True)
class DrawdownRestart:
    """Caída medida de nuevo desde un bloqueo total (``restart_after_unlock``).

    Después de que el usuario borra el LOCKED, la caída se mide como si el mes
    empezara en ``locked_at`` (la hora del último bloqueo del mes): el máximo
    arranca en el saldo de ese momento (``start_balance_usd``, con la pérdida
    que disparó el bloqueo incluida) y sigue las mismas reglas que el máximo
    del mes (depósitos, retiros, saldos anotados).
    """

    lock_id: int | None
    locked_at: datetime
    start_balance_usd: float
    peak_usd: float
    peak_at: datetime
    current_balance_usd: float
    drawdown_usd: float
    drawdown_pct: float
    deposits_usd: float = 0.0
    withdrawals_usd: float = 0.0
    trading_pnl_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "lock_id": self.lock_id,
                "locked_at": _iso(self.locked_at),
                "start_balance_usd": self.start_balance_usd,
                "peak_usd": self.peak_usd,
                "peak_at": _iso(self.peak_at),
                "current_balance_usd": self.current_balance_usd,
                "drawdown_usd": self.drawdown_usd,
                "drawdown_pct": self.drawdown_pct,
                "deposits_usd": self.deposits_usd,
                "withdrawals_usd": self.withdrawals_usd,
                "trading_pnl_usd": self.trading_pnl_usd,
            }
        )


@dataclass(frozen=True)
class WalletState:
    """Saldo de la billetera de futuros, su máximo del mes y la caída desde el máximo.

    - ``assumed``: ``True`` si el usuario nunca anotó un saldo y se supuso
      ``assumed_start_usd`` (la billetera mensual del perfil) al empezar el mes.
    - ``month_peak_usd``: el máximo del mes, con las reglas de depósitos
      (``deposit_raises_peak``) y retiros (``withdrawal_peak``) que se usaron
      (ver el docstring del módulo).
    - ``drawdown_pct``: caída desde el máximo del mes, en % (0-100+).
    - ``anchor``: el saldo anotado que se usó de base (``None`` si es supuesto).
    - ``points``: historia del mes, empezando por el saldo del primer instante.
    - ``restart``: la caída medida de nuevo desde el último bloqueo total del
      mes (``None`` si no hubo o si ``restart_after_unlock`` está apagado).
    """

    assumed: bool
    assumed_start_usd: float | None
    month_start: datetime
    month_start_balance_usd: float
    current_balance_usd: float
    month_peak_usd: float
    month_peak_at: datetime
    drawdown_usd: float
    drawdown_pct: float
    month_deposits_usd: float
    month_withdrawals_usd: float
    month_trading_pnl_usd: float
    last_balance_at: datetime | None
    last_balance_usd: float | None
    anchor: WalletRecord | None
    points: tuple[WalletPoint, ...]
    warnings: tuple[str, ...] = ()
    withdrawal_peak: WithdrawalPeak = "proportional"
    deposit_raises_peak: bool = True
    restart: DrawdownRestart | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "assumed": self.assumed,
                "assumed_start_usd": self.assumed_start_usd,
                "month_start": _iso(self.month_start),
                "month_start_balance_usd": self.month_start_balance_usd,
                "current_balance_usd": self.current_balance_usd,
                "month_peak_usd": self.month_peak_usd,
                "month_peak_at": _iso(self.month_peak_at),
                "drawdown_usd": self.drawdown_usd,
                "drawdown_pct": self.drawdown_pct,
                "month_deposits_usd": self.month_deposits_usd,
                "month_withdrawals_usd": self.month_withdrawals_usd,
                "month_trading_pnl_usd": self.month_trading_pnl_usd,
                "last_balance_at": _iso(self.last_balance_at),
                "last_balance_usd": self.last_balance_usd,
                "anchor": self.anchor,
                "points": list(self.points),
                "warnings": list(self.warnings),
                "withdrawal_peak": self.withdrawal_peak,
                "deposit_raises_peak": self.deposit_raises_peak,
                "restart": self.restart,
            }
        )


# Orden de los puntos que caen en el mismo instante: el saldo supuesto del
# mes va antes de todo (es el "primer instante"); un saldo anotado va después
# de los movimientos (incluye todo lo que pasó hasta ese momento).
_RANK_ASSUMED = -1
_RANK_MOVEMENT = 0
_RANK_SNAPSHOT = 1


@dataclass(frozen=True)
class _Point:
    ts: datetime
    rank: int
    seq: int
    kind: str
    amount: float  # cambio (movimiento) o saldo (anotado / supuesto)
    ref_id: int | None = None
    symbol: str | None = None

    @property
    def is_snapshot(self) -> bool:
        return self.rank != _RANK_MOVEMENT


@dataclass
class _PeakWalk:
    """Resultado de recorrer la billetera desde un saldo inicial (uso interno)."""

    peak: float
    peak_at: datetime
    deposits: float = 0.0
    withdrawals: float = 0.0
    trading: float = 0.0
    lowest: float = 0.0
    history: list[WalletPoint] | None = None


def _walk_peak(
    points: Sequence[_Point],
    after: Sequence[float],
    indices: Iterable[int],
    *,
    start_balance: float,
    start_ts: datetime,
    withdrawal_peak: WithdrawalPeak,
    deposit_raises_peak: bool,
    keep_history: bool,
) -> _PeakWalk:
    """Recorre los puntos ``indices`` (en orden) moviendo el máximo con las reglas del usuario."""
    walk = _PeakWalk(
        peak=start_balance,
        peak_at=start_ts,
        lowest=start_balance,
        history=[] if keep_history else None,
    )
    for i in indices:
        p = points[i]
        bal = after[i]
        if p.kind == POINT_WITHDRAWAL:
            walk.withdrawals -= p.amount  # p.amount es negativo
            if withdrawal_peak == "subtract":
                walk.peak += p.amount
            elif withdrawal_peak == "proportional":
                before = bal - p.amount
                if before > 0:
                    # El máximo baja en la misma proporción que el saldo: el %
                    # de caída no cambia. Sacar todo (o más) lo deja en 0.
                    walk.peak *= max(0.0, bal) / before
        elif p.kind == POINT_DEPOSIT:
            walk.deposits += p.amount
            if deposit_raises_peak:
                walk.peak += p.amount
        elif p.kind == POINT_TRADE:
            walk.trading += p.amount
        if bal > walk.peak:
            walk.peak, walk.peak_at = bal, p.ts
        walk.lowest = min(walk.lowest, bal)
        if walk.history is not None:
            walk.history.append(
                WalletPoint(
                    ts=p.ts,
                    kind=p.kind,
                    delta_usd=None if p.is_snapshot else _cents(p.amount),
                    balance_usd=_cents(bal),
                    peak_usd=_cents(walk.peak),
                    ref_id=p.ref_id,
                    symbol=p.symbol,
                )
            )
    return walk


def _drawdown(peak: float, current: float) -> tuple[float, float, float, float]:
    """``(máximo, saldo, caída en USD, caída en %)`` a centavos."""
    peak_c = _cents(peak)
    current_c = _cents(current)
    drawdown = _cents(max(0.0, peak_c - current_c))
    pct = drawdown * 100.0 / peak_c if peak_c > 0 else 0.0
    return peak_c, current_c, drawdown, pct


def wallet_state(
    wallet_events: Iterable[Any],
    trades: Iterable[Any],
    now: datetime,
    *,
    month_start: datetime,
    assumed_start_usd: float,
    withdrawal_peak: WithdrawalPeak = "proportional",
    deposit_raises_peak: bool = True,
    restart_at: datetime | None = None,
    restart_lock_id: int | None = None,
) -> WalletState:
    """Saldo, máximo del mes y caída (ver las reglas en el docstring del módulo).

    - ``assumed_start_usd``: el saldo que se supone al empezar el mes si el
      usuario nunca anotó un saldo (la billetera mensual del perfil).
    - ``withdrawal_peak`` / ``deposit_raises_peak``: cómo mueven el máximo los
      retiros y los depósitos (``circuit_breakers.monthly_drawdown_lock``).
    - ``restart_at``: la hora del último bloqueo total del mes; si viene (y
      cae entre el inicio del mes y ``now``), ``WalletState.restart`` trae la
      caída medida de nuevo desde el saldo de ese momento. ``restart_lock_id``
      es el número de ese bloqueo en la base (solo para mostrar).
    """
    moment = _aware_utc(now, "La hora actual")
    start = _aware_utc(month_start, "El inicio del mes")
    assumed_value = _number(assumed_start_usd, "El saldo supuesto")
    if withdrawal_peak not in WITHDRAWAL_PEAK_RULES:
        raise ValueError(
            f"Regla de retiros desconocida: {withdrawal_peak!r} "
            f"(opciones: {', '.join(WITHDRAWAL_PEAK_RULES)})."
        )
    restart_moment = _opt_aware_utc(restart_at, "La hora del bloqueo")
    events = sorted(
        (e for e in (WalletRecord.from_value(x) for x in wallet_events) if e.ts <= moment),
        key=lambda e: (e.ts, e.id if e.id is not None else 0),
    )
    closed = [t for t in _sorted_by_close(trades) if t.closed_at <= moment]

    points: list[_Point] = []
    for e in events:
        if e.kind == "balance":
            points.append(
                _Point(e.ts, _RANK_SNAPSHOT, len(points), POINT_BALANCE, e.amount_usd, e.id)
            )
        elif e.kind == "deposit":
            points.append(
                _Point(e.ts, _RANK_MOVEMENT, len(points), POINT_DEPOSIT, e.amount_usd, e.id)
            )
        else:
            points.append(
                _Point(e.ts, _RANK_MOVEMENT, len(points), POINT_WITHDRAWAL, -e.amount_usd, e.id)
            )
    for t in closed:
        points.append(
            _Point(t.closed_at, _RANK_MOVEMENT, len(points), POINT_TRADE, t.pnl_usd, t.id, t.symbol)
        )
    balances = [e for e in events if e.kind == "balance"]
    assumed = not balances
    if assumed:
        points.append(_Point(start, _RANK_ASSUMED, -1, POINT_ASSUMED_START, assumed_value))
    points.sort(key=lambda p: (p.ts, p.rank, p.seq))

    # Saldo después de cada punto: hacia adelante desde el primer saldo
    # (anotado o supuesto) y hacia atrás antes de él.
    first = next(i for i, p in enumerate(points) if p.is_snapshot)
    after = [0.0] * len(points)
    running = 0.0
    for i in range(first, len(points)):
        p = points[i]
        running = p.amount if p.is_snapshot else running + p.amount
        after[i] = running
    back = points[first].amount
    for i in range(first - 1, -1, -1):
        after[i] = back
        back -= points[i].amount
    initial = back

    pre_month = [i for i, p in enumerate(points) if p.ts < start]
    month_start_balance = after[pre_month[-1]] if pre_month else initial
    current = after[-1]

    walk = _walk_peak(
        points,
        after,
        (i for i, p in enumerate(points) if p.ts >= start and p.kind != POINT_ASSUMED_START),
        start_balance=month_start_balance,
        start_ts=start,
        keep_history=True,
        withdrawal_peak=withdrawal_peak,
        deposit_raises_peak=deposit_raises_peak,
    )
    history = [
        WalletPoint(
            ts=start,
            kind=POINT_ASSUMED_START if assumed else POINT_MONTH_START,
            delta_usd=None,
            balance_usd=_cents(month_start_balance),
            peak_usd=_cents(month_start_balance),
        ),
        *(walk.history or []),
    ]
    peak_c, current_c, drawdown, drawdown_pct = _drawdown(walk.peak, current)

    restart: DrawdownRestart | None = None
    if restart_moment is not None and start <= restart_moment <= moment:
        # Como un mes nuevo que empieza en el bloqueo: la base es el saldo
        # después de todo lo que pasó hasta esa hora (la pérdida del disparo
        # incluida).
        upto = [i for i, p in enumerate(points) if p.ts <= restart_moment]
        base = after[upto[-1]] if upto else initial
        rwalk = _walk_peak(
            points,
            after,
            (
                i
                for i, p in enumerate(points)
                if p.ts > restart_moment and p.kind != POINT_ASSUMED_START
            ),
            start_balance=base,
            start_ts=restart_moment,
            keep_history=False,
            withdrawal_peak=withdrawal_peak,
            deposit_raises_peak=deposit_raises_peak,
        )
        r_peak, r_current, r_dd, r_pct = _drawdown(rwalk.peak, current)
        restart = DrawdownRestart(
            lock_id=restart_lock_id,
            locked_at=restart_moment,
            start_balance_usd=_cents(base),
            peak_usd=r_peak,
            peak_at=rwalk.peak_at,
            current_balance_usd=r_current,
            drawdown_usd=r_dd,
            drawdown_pct=r_pct,
            deposits_usd=_cents(rwalk.deposits),
            withdrawals_usd=_cents(rwalk.withdrawals),
            trading_pnl_usd=_cents(rwalk.trading),
        )

    warnings: list[str] = []
    if assumed:
        warnings.append(
            "No anotaste nunca el saldo de tu billetera de futuros: para el freno del mes "
            f"supongo que empezaste el mes con {fmt_usd(assumed_value)} (la billetera mensual "
            "del perfil). Anotá tu saldo real para que el cálculo sea exacto."
        )
    if walk.lowest < 0 or current < 0:
        warnings.append(
            "El saldo calculado de la billetera da negativo en algún momento del mes: "
            "revisá los movimientos y el saldo anotados."
        )
    last_balance = balances[-1] if balances else None
    anchor_candidates = [e for e in balances if e.ts < start]
    anchor = anchor_candidates[-1] if anchor_candidates else (balances[0] if balances else None)
    return WalletState(
        assumed=assumed,
        assumed_start_usd=assumed_value if assumed else None,
        month_start=start,
        month_start_balance_usd=_cents(month_start_balance),
        current_balance_usd=current_c,
        month_peak_usd=peak_c,
        month_peak_at=walk.peak_at,
        drawdown_usd=drawdown,
        drawdown_pct=drawdown_pct,
        month_deposits_usd=_cents(walk.deposits),
        month_withdrawals_usd=_cents(walk.withdrawals),
        month_trading_pnl_usd=_cents(walk.trading),
        last_balance_at=None if last_balance is None else last_balance.ts,
        last_balance_usd=None if last_balance is None else last_balance.amount_usd,
        anchor=anchor,
        points=tuple(history),
        warnings=tuple(warnings),
        withdrawal_peak=withdrawal_peak,
        deposit_raises_peak=deposit_raises_peak,
        restart=restart,
    )


# --- estado completo -----------------------------------------------------------------


@dataclass(frozen=True)
class TradingState:
    """Todo lo que la capa de seguridad necesita saber del estado de trading en ``now``.

    Los números ya vienen calculados (ver el docstring del módulo). Para el
    plan (``risk.plan.PlanInputs``): ``trades_today``, ``plan_pnl_today_usd``
    (el resultado **actual** del día) y ``open_risk_usd``. Para los frenos del
    día y de la semana: ``day_max_loss_usd`` / ``week_max_loss_usd`` (la
    pérdida más grande a la que se llegó). ``month_locks``: los bloqueos
    totales del mes anotados en la base (el último, ``latest_month_lock``).
    """

    now: datetime
    timezone: str
    loss_basis: LossBasis
    periods: Periods
    profile: ProfileLimits
    day: PeriodPnl
    week: PeriodPnl
    month: PeriodPnl
    trades_today: int
    closed_trades_opened_today: int
    open_positions_opened_today: int
    streak: LosingStreakState
    wallet: WalletState
    open_positions: tuple[PositionRecord, ...]
    volatility_events: tuple[VolatilityEventRecord, ...]
    month_trades: tuple[TradeRecord, ...]
    month_wallet_events: tuple[WalletRecord, ...]
    warnings: tuple[str, ...] = ()
    month_locks: tuple[SafetyLockRecord, ...] = ()

    @property
    def day_pnl_usd(self) -> float:
        """Resultado neto de lo cerrado hoy (negativo = pérdida)."""
        return self.day.net_pnl_usd

    @property
    def day_loss_usd(self) -> float:
        """Pérdida actual del día según ``loss_basis`` (positiva; 0 si el día va ganando)."""
        return self.day.loss_usd

    @property
    def day_max_loss_usd(self) -> float:
        """La pérdida más grande a la que se llegó hoy (``PeriodPnl.max_loss_usd``)."""
        return self.day.max_loss_usd

    @property
    def week_pnl_usd(self) -> float:
        return self.week.net_pnl_usd

    @property
    def week_loss_usd(self) -> float:
        return self.week.loss_usd

    @property
    def week_max_loss_usd(self) -> float:
        """La pérdida más grande a la que se llegó en la semana."""
        return self.week.max_loss_usd

    @property
    def latest_month_lock(self) -> SafetyLockRecord | None:
        """El último bloqueo total del mes anotado en la base (``None`` si no hubo)."""
        return self.month_locks[-1] if self.month_locks else None

    @property
    def plan_pnl_today_usd(self) -> float:
        """Lo que va en ``PlanInputs.pnl_today_usd``: el plan achica el riesgo con la misma base.

        Con ``"net"`` es el neto del día; con ``"gross"``, menos la suma de las
        pérdidas (así ``max(0, -pnl)`` del plan da la pérdida que miran los breakers).
        """
        return self.day.net_pnl_usd if self.loss_basis == "net" else -self.day.gross_loss_usd

    @property
    def open_risk_usd(self) -> float:
        """Lo que se pierde si tocan todos los stops de las posiciones abiertas.

        Suma ``max(0, riesgo)`` de cada una: un stop en la entrada o en
        ganancia (riesgo 0 o negativo) cuenta 0, no resta. Una cobertura en la
        misma moneda también suma (pueden saltar los dos stops).
        """
        return _cents(sum(max(0.0, p.risk_usd) for p in self.open_positions))

    def active_volatility_events(
        self, now: datetime | None = None
    ) -> tuple[VolatilityEventRecord, ...]:
        """Los eventos que bloquean en ``now`` (por defecto, el del estado)."""
        moment = self.now if now is None else now
        return tuple(e for e in self.volatility_events if e.is_active(moment))

    def upcoming_volatility_events(
        self, now: datetime | None = None
    ) -> tuple[VolatilityEventRecord, ...]:
        """Los eventos marcados que todavía no empezaron en ``now``."""
        moment = self.now if now is None else now
        return tuple(e for e in self.volatility_events if e.is_upcoming(moment))

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (lo que muestra el panel de riesgo)."""
        return json_safe(
            {
                "now": _iso(self.now),
                "timezone": self.timezone,
                "loss_basis": self.loss_basis,
                "periods": self.periods.to_dict(),
                "profile": self.profile,
                "day": self.day,
                "week": self.week,
                "month": self.month,
                "day_pnl_usd": self.day_pnl_usd,
                "day_loss_usd": self.day_loss_usd,
                "day_max_loss_usd": self.day_max_loss_usd,
                "week_pnl_usd": self.week_pnl_usd,
                "week_loss_usd": self.week_loss_usd,
                "week_max_loss_usd": self.week_max_loss_usd,
                "plan_pnl_today_usd": self.plan_pnl_today_usd,
                "trades_today": self.trades_today,
                "closed_trades_opened_today": self.closed_trades_opened_today,
                "open_positions_opened_today": self.open_positions_opened_today,
                "max_trades_per_day": self.profile.max_trades_per_day,
                "streak": self.streak,
                "wallet": self.wallet,
                "open_positions": list(self.open_positions),
                "open_risk_usd": self.open_risk_usd,
                "active_volatility_events": list(self.active_volatility_events()),
                "upcoming_volatility_events": list(self.upcoming_volatility_events()),
                "month_trades": list(self.month_trades),
                "month_wallet_events": list(self.month_wallet_events),
                "month_locks": list(self.month_locks),
                "latest_month_lock": self.latest_month_lock,
                "warnings": list(self.warnings),
            }
        )


def build_trading_state(
    *,
    trades: Iterable[Any],
    open_positions: Iterable[Any],
    wallet_events: Iterable[Any],
    volatility_events: Iterable[Any],
    settings: AppSettings,
    profile: Any,
    now: datetime,
    safety_locks: Iterable[Any] = (),
) -> TradingState:
    """Arma ``TradingState`` con los registros ya leídos (pura: no lee nada).

    - ``trades``: todas las operaciones cerradas (hacen falta todas para la
      racha de pérdidas; las de después de ``now`` se ignoran).
    - ``open_positions``: las posiciones abiertas.
    - ``wallet_events``: todos los movimientos de la billetera.
    - ``volatility_events``: los eventos de alta volatilidad (alcanza con los
      que no terminaron en ``now``; los terminados se descartan).
    - ``profile``: el perfil de riesgo (``EffectiveProfile`` o algo con
      ``monthly_wallet_usd``, ``max_daily_loss_usd``, ...).
    - ``safety_locks``: los bloqueos totales anotados en la base
      (``StoredSafetyLock``); se usan los del mes hasta ``now``. Con
      ``monthly_drawdown_lock.restart_after_unlock``, el último define
      ``wallet.restart``.
    """
    moment = _aware_utc(now, "La hora actual")
    cb = settings.circuit_breakers
    if cb.month_starts != "calendar":  # pragma: no cover - la config solo admite "calendar"
        raise ValueError(f"circuit_breakers.month_starts desconocido: {cb.month_starts!r}.")
    limits = ProfileLimits.from_value(profile)
    periods = period_bounds(moment, settings.app.user_timezone, cb.week_starts_on)

    all_trades = [t for t in _sorted_by_close(trades) if t.closed_at <= moment]
    positions = tuple(PositionRecord.from_value(p) for p in open_positions)
    events = [WalletRecord.from_value(e) for e in wallet_events]
    vol_events = tuple(
        sorted(
            (
                e
                for e in (VolatilityEventRecord.from_value(x) for x in volatility_events)
                if e.until > moment
            ),
            key=lambda e: (e.starts_at, e.id if e.id is not None else 0),
        )
    )

    basis: LossBasis = cb.loss_basis
    day = period_pnl(all_trades, periods.day_start, periods.day_end, loss_basis=basis)
    week = period_pnl(all_trades, periods.week_start, periods.week_end, loss_basis=basis)
    month = period_pnl(all_trades, periods.month_start, periods.month_end, loss_basis=basis)
    closed_today = count_opened_in(all_trades, periods.day_start, periods.day_end)
    open_today = count_opened_in(
        [p for p in positions if p.opened_at is None or p.opened_at <= moment],
        periods.day_start,
        periods.day_end,
    )
    streak = losing_streak_state(
        all_trades,
        moment,
        consecutive_losses=cb.losing_streak.consecutive_losses,
        cooldown_hours=cb.losing_streak.cooldown_hours,
        reset_after_cooldown=cb.losing_streak.reset_after_cooldown,
        extend_cooldown_on_loss=cb.losing_streak.extend_cooldown_on_loss,
    )
    month_locks = tuple(
        sorted(
            (
                lk
                for lk in (SafetyLockRecord.from_value(x) for x in safety_locks)
                if periods.month_start <= lk.triggered_at <= moment
            ),
            key=lambda lk: (lk.triggered_at, lk.id if lk.id is not None else 0),
        )
    )
    dd_cfg = cb.monthly_drawdown_lock
    latest_lock = month_locks[-1] if month_locks else None
    restart_from = latest_lock if dd_cfg.restart_after_unlock else None
    wallet = wallet_state(
        events,
        all_trades,
        moment,
        month_start=periods.month_start,
        assumed_start_usd=limits.monthly_wallet_usd,
        withdrawal_peak=dd_cfg.withdrawal_peak,
        deposit_raises_peak=dd_cfg.deposit_raises_peak,
        restart_at=None if restart_from is None else restart_from.triggered_at,
        restart_lock_id=None if restart_from is None else restart_from.id,
    )
    month_trades = tuple(t for t in all_trades if t.closed_at >= periods.month_start)
    month_events = [e for e in events if periods.month_start <= e.ts <= moment]
    if wallet.anchor is not None and wallet.anchor.ts < periods.month_start:
        month_events.insert(0, wallet.anchor)
    month_events.sort(key=lambda e: (e.ts, e.id if e.id is not None else 0))
    return TradingState(
        now=moment,
        timezone=periods.timezone,
        loss_basis=basis,
        periods=periods,
        profile=limits,
        day=day,
        week=week,
        month=month,
        trades_today=closed_today + open_today,
        closed_trades_opened_today=closed_today,
        open_positions_opened_today=open_today,
        streak=streak,
        wallet=wallet,
        open_positions=positions,
        volatility_events=vol_events,
        month_trades=month_trades,
        month_wallet_events=tuple(month_events),
        warnings=wallet.warnings,
        month_locks=month_locks,
    )


def load_trading_state(
    repo: TradingStateSource,
    settings: AppSettings,
    profile: Any,
    now: datetime,
) -> TradingState:
    """Lee la base y arma ``TradingState`` (única función con I/O del módulo).

    Lee todas las operaciones y movimientos (una app personal tiene pocos
    cientos por año: es instantáneo), las posiciones abiertas, los eventos de
    alta volatilidad que no terminaron en ``now`` y, si la base sabe
    listarlos (``SafetyLockSource``), los bloqueos totales anotados.
    """
    moment = _aware_utc(now, "La hora actual")
    list_locks = getattr(repo, "list_safety_locks", None)
    locks: Sequence[Any] = list_locks() if callable(list_locks) else ()
    return build_trading_state(
        trades=repo.list_trades(),
        open_positions=repo.list_open_positions(),
        wallet_events=repo.list_wallet_events(),
        volatility_events=repo.list_volatility_events(ending_after=moment),
        settings=settings,
        profile=profile,
        now=moment,
        safety_locks=locks,
    )


__all__ = [
    "POINT_ASSUMED_START",
    "POINT_BALANCE",
    "POINT_DEPOSIT",
    "POINT_MONTH_START",
    "POINT_TRADE",
    "POINT_WITHDRAWAL",
    "WALLET_KINDS",
    "WEEKDAY_INDEX",
    "WITHDRAWAL_PEAK_RULES",
    "DrawdownRestart",
    "LosingStreakState",
    "LossBasis",
    "PeriodPnl",
    "Periods",
    "PositionRecord",
    "ProfileLimits",
    "SafetyLockRecord",
    "SafetyLockSource",
    "TradeRecord",
    "TradingState",
    "TradingStateSource",
    "VolatilityEventRecord",
    "WalletPoint",
    "WalletRecord",
    "WalletState",
    "WithdrawalPeak",
    "build_trading_state",
    "count_opened_in",
    "load_trading_state",
    "losing_streak_state",
    "period_bounds",
    "period_pnl",
    "resolve_user_timezone",
    "wallet_state",
]
