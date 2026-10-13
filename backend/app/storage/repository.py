"""Lectura y escritura sobre las tablas: la única puerta a la base para el resto de la app.

``Repository(engine)`` (o ``Repository.open(path)``) agrupa todas las
operaciones. Cada método abre y cierra su propia sesión (una transacción por
llamada), así que se puede usar desde varios hilos.

Qué devuelve: dataclasses inmutables (``StoredTrade``, ``StoredPosition``,
``StoredWalletEvent``, ``StoredVolatilityEvent``, ``StoredAnalysis``), nunca
los objetos de SQLModel. Se
pueden usar después de cerrada la sesión, no consultan la base de nuevo y
modificarlas no cambia nada guardado. Cada una tiene ``to_dict()`` listo para
JSON (fechas en ISO 8601 con ``+00:00``, sin NaN ni infinitos).

Fechas:

- Todas las que entran tienen que tener zona horaria (``datetime`` "aware").
  Una fecha sin zona se rechaza con ``ValueError``: no se sabe si era hora de
  Buenos Aires o UTC, y 3 h de diferencia cambian a qué día cuenta una
  pérdida para los circuit breakers. La API, que recibe la hora del usuario,
  es la que tiene que ponerle la zona.
- Se guardan en UTC y salen siempre en UTC con ``tzinfo=UTC``.
- Una fecha más de ``MAX_CLOCK_SKEW`` en el futuro se rechaza (casi siempre es
  un error de tipeo, como 2062 en vez de 2026, y contaría para otro día).
- Los filtros por rango son semiabiertos: ``since <= fecha < until``.

Validaciones (``ValueError`` con el motivo en castellano): dirección distinta
de ``"long"``/``"short"``, moneda que no es alfanumérica, cierre antes de la
apertura, riesgo <= 0, precios <= 0, montos <= 0 en depósitos y retiros,
saldo negativo, números no finitos, análisis inexistente o de otra moneda.

Eventos de alta volatilidad (``add_volatility_event``): nota obligatoria y hora
de fin posterior a ahora y al inicio (el inicio por defecto es ahora y puede
estar en el futuro, para marcar un evento con anticipación). Bloquean los
planes mientras ``starts_at <= ahora < until`` y después se apagan solos
(``active_volatility_events``); ``delete_volatility_event`` los borra antes.

Operaciones ligadas a un análisis (``analysis_id``): el análisis tiene que
existir y ser de la misma moneda; el contexto que no se pase a mano (régimen,
confianza, score, estado del plan y perfil) se copia de ese análisis. Los
valores pasados a mano tienen prioridad.
"""

from __future__ import annotations

import copy
import json
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, get_args

from sqlalchemy.engine import Engine
from sqlmodel import Session, col, select

from app.config import RegimeCategory
from app.risk.plan import PlanStatus
from app.risk.rules import json_safe
from app.storage.db import open_db, session_scope
from app.storage.models import (
    AnalysisRecord,
    HighVolatilityEvent,
    ManualTrade,
    OpenPosition,
    WalletEvent,
    utc_now,
)

#: Cuánto en el futuro se tolera una fecha (diferencias de reloj entre el
#: celular y la PC).
MAX_CLOCK_SKEW = timedelta(minutes=5)

#: Largo máximo de las notas, para que un pegado gigante no infle la base.
MAX_NOTES_LENGTH = 2000

#: Largo máximo de la nota de un evento de alta volatilidad (se muestra en el
#: aviso de bloqueo: tiene que ser corta).
MAX_EVENT_NOTE_LENGTH = 300

#: Largo máximo del símbolo de una moneda (``1000PEPE`` tiene 8).
MAX_SYMBOL_LENGTH = 20

SIDES: tuple[str, ...] = ("long", "short")
WALLET_EVENT_KINDS: tuple[str, ...] = ("deposit", "withdrawal", "balance")
PLAN_STATUSES: tuple[str, ...] = tuple(get_args(PlanStatus))
REGIME_CATEGORIES: tuple[str, ...] = tuple(get_args(RegimeCategory))

Clock = Callable[[], datetime]


# --- resultados ------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


@dataclass(frozen=True)
class StoredTrade:
    """Operación cerrada tal como quedó guardada (ver ``models.ManualTrade``)."""

    id: int
    symbol: str
    side: str
    opened_at: datetime
    closed_at: datetime
    pnl_usd: float
    risk_usd: float | None
    entry_price: float | None
    exit_price: float | None
    notes: str | None
    analysis_id: int | None
    regime_category: str | None
    regime_name: str | None
    regime_confidence: float | None
    score_total: float | None
    plan_status: str | None
    profile_name: str | None
    created_at: datetime

    @property
    def is_loss(self) -> bool:
        """``True`` si el resultado neto es negativo (un empate no es pérdida)."""
        return self.pnl_usd < 0

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
                "created_at": _iso(self.created_at),
            }
        )

    @classmethod
    def _from_row(cls, row: ManualTrade) -> StoredTrade:
        assert row.id is not None
        return cls(
            id=row.id,
            symbol=row.symbol,
            side=row.side,
            opened_at=row.opened_at,
            closed_at=row.closed_at,
            pnl_usd=row.pnl_usd,
            risk_usd=row.risk_usd,
            entry_price=row.entry_price,
            exit_price=row.exit_price,
            notes=row.notes,
            analysis_id=row.analysis_id,
            regime_category=row.regime_category,
            regime_name=row.regime_name,
            regime_confidence=row.regime_confidence,
            score_total=row.score_total,
            plan_status=row.plan_status,
            profile_name=row.profile_name,
            created_at=row.created_at,
        )


@dataclass(frozen=True)
class StoredPosition:
    """Posición abierta tal como quedó guardada (ver ``models.OpenPosition``)."""

    id: int
    symbol: str
    side: str
    opened_at: datetime
    entry_price: float | None
    risk_usd: float
    notes: str | None
    analysis_id: int | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "symbol": self.symbol,
                "side": self.side,
                "opened_at": _iso(self.opened_at),
                "entry_price": self.entry_price,
                "risk_usd": self.risk_usd,
                "notes": self.notes,
                "analysis_id": self.analysis_id,
                "created_at": _iso(self.created_at),
            }
        )

    @classmethod
    def _from_row(cls, row: OpenPosition) -> StoredPosition:
        assert row.id is not None
        return cls(
            id=row.id,
            symbol=row.symbol,
            side=row.side,
            opened_at=row.opened_at,
            entry_price=row.entry_price,
            risk_usd=row.risk_usd,
            notes=row.notes,
            analysis_id=row.analysis_id,
            created_at=row.created_at,
        )


@dataclass(frozen=True)
class StoredWalletEvent:
    """Movimiento de la billetera tal como quedó guardado (ver ``models.WalletEvent``)."""

    id: int
    ts: datetime
    kind: str
    amount_usd: float
    notes: str | None

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

    @classmethod
    def _from_row(cls, row: WalletEvent) -> StoredWalletEvent:
        assert row.id is not None
        return cls(id=row.id, ts=row.ts, kind=row.kind, amount_usd=row.amount_usd, notes=row.notes)


@dataclass(frozen=True)
class StoredVolatilityEvent:
    """Evento de alta volatilidad tal como quedó guardado (ver ``models.HighVolatilityEvent``)."""

    id: int
    created_at: datetime
    starts_at: datetime
    until: datetime
    note: str

    def is_active(self, now: datetime) -> bool:
        """``True`` si bloquea en ``now``: ``starts_at <= now < until``."""
        moment = _aware(now, "La hora")
        return self.starts_at <= moment < self.until

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "created_at": _iso(self.created_at),
                "starts_at": _iso(self.starts_at),
                "until": _iso(self.until),
                "note": self.note,
            }
        )

    @classmethod
    def _from_row(cls, row: HighVolatilityEvent) -> StoredVolatilityEvent:
        assert row.id is not None
        return cls(
            id=row.id,
            created_at=row.created_at,
            starts_at=row.starts_at,
            until=row.until,
            note=row.note,
        )


@dataclass(frozen=True)
class StoredAnalysis:
    """Análisis guardado (ver ``models.AnalysisRecord``).

    ``plan``, ``score``, ``safety``, ``llm`` e ``inputs`` son los dicts ya
    leídos del JSON (``None`` si no se guardaron). Cada lectura los arma de
    cero, así que modificarlos no afecta a otras lecturas.
    """

    id: int
    created_at: datetime
    symbol: str
    profile_name: str
    price: float
    price_source: str
    regime_category: str | None
    regime_name: str | None
    regime_confidence: float | None
    regime_unstable: bool | None
    score_total: float | None
    plan_status: str
    plan: dict[str, Any] | None
    score: dict[str, Any] | None
    safety: dict[str, Any] | None
    llm: dict[str, Any] | None = None
    inputs: dict[str, Any] | None = None

    def to_dict(self, *, include_json: bool = True) -> dict[str, Any]:
        """Dict listo para JSON. ``include_json=False`` deja afuera los dicts grandes.

        (``plan``, ``score``, ``safety``, ``llm`` e ``inputs``: para listados.)
        """
        out: dict[str, Any] = {
            "id": self.id,
            "created_at": _iso(self.created_at),
            "symbol": self.symbol,
            "profile_name": self.profile_name,
            "price": self.price,
            "price_source": self.price_source,
            "regime_category": self.regime_category,
            "regime_name": self.regime_name,
            "regime_confidence": self.regime_confidence,
            "regime_unstable": self.regime_unstable,
            "score_total": self.score_total,
            "plan_status": self.plan_status,
        }
        if include_json:
            out["plan"] = copy.deepcopy(self.plan)
            out["score"] = copy.deepcopy(self.score)
            out["safety"] = copy.deepcopy(self.safety)
            out["llm"] = copy.deepcopy(self.llm)
            out["inputs"] = copy.deepcopy(self.inputs)
        return json_safe(out)

    @classmethod
    def _from_row(cls, row: AnalysisRecord) -> StoredAnalysis:
        assert row.id is not None
        return cls(
            id=row.id,
            created_at=row.created_at,
            symbol=row.symbol,
            profile_name=row.profile_name,
            price=row.price,
            price_source=row.price_source,
            regime_category=row.regime_category,
            regime_name=row.regime_name,
            regime_confidence=row.regime_confidence,
            regime_unstable=row.regime_unstable,
            score_total=row.score_total,
            plan_status=row.plan_status,
            plan=_from_json_text(row.plan_json),
            score=_from_json_text(row.score_json),
            safety=_from_json_text(row.safety_json),
            llm=_from_json_text(row.llm_json),
            inputs=_from_json_text(row.inputs_json),
        )


# --- validación -----------------------------------------------------------------


def _symbol(value: Any) -> str:
    """Moneda en mayúsculas (``" btc "`` → ``"BTC"``); solo letras y números ASCII."""
    if not isinstance(value, str):
        raise ValueError(f"La moneda tiene que ser un texto, llegó {value!r}.")
    s = value.strip().upper()
    if not s or not s.isascii() or not s.isalnum():
        raise ValueError(f"Moneda inválida: {value!r}. Usá solo letras y números, por ejemplo BTC.")
    if len(s) > MAX_SYMBOL_LENGTH:
        raise ValueError(
            f"Moneda inválida: {value!r}. Tiene más de {MAX_SYMBOL_LENGTH} caracteres."
        )
    return s


def _side(value: Any) -> str:
    s = value.strip().lower() if isinstance(value, str) else value
    if s not in SIDES:
        raise ValueError(f"Dirección desconocida: {value!r}. Tiene que ser 'long' o 'short'.")
    return s


def _number(value: Any, label: str) -> float:
    """Número finito. ``True``/``False`` no cuentan como número."""
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


def _positive(value: Any, label: str) -> float:
    v = _number(value, label)
    if v <= 0:
        raise ValueError(f"{label} tiene que ser mayor a 0, llegó {v:g}.")
    return v


def _opt_positive(value: Any, label: str) -> float | None:
    return None if value is None else _positive(value, label)


def _opt_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        value = item()  # np.bool_
    if not isinstance(value, bool):
        raise ValueError(f"{label} tiene que ser verdadero o falso, llegó {value!r}.")
    return value


def _text(value: Any, label: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} no puede estar vacío.")
    s = value.strip()
    if len(s) > max_length:
        raise ValueError(f"{label} es demasiado largo (máximo {max_length} caracteres).")
    return s


def _opt_text(value: Any, label: str, max_length: int) -> str | None:
    """Texto opcional: ``None`` o vacío quedan en ``None``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _text(value, label, max_length)


def _notes(value: Any) -> str | None:
    return _opt_text(value, "La nota", MAX_NOTES_LENGTH)


def _confidence(value: Any) -> float | None:
    v = _opt_number(value, "La confianza del régimen")
    if v is not None and not 0.0 <= v <= 1.0:
        raise ValueError(f"La confianza del régimen va de 0 a 1, llegó {v:g}.")
    return v


def _score_total(value: Any) -> float | None:
    v = _opt_number(value, "El score")
    if v is not None and not -100.0 <= v <= 100.0:
        raise ValueError(f"El score va de -100 a +100, llegó {v:g}.")
    return v


def _regime_category(value: Any) -> str | None:
    if value is None:
        return None
    s = value.strip().lower() if isinstance(value, str) else value
    if s not in REGIME_CATEGORIES:
        raise ValueError(
            f"Tipo de régimen desconocido: {value!r}. Tiene que ser uno de "
            f"{', '.join(REGIME_CATEGORIES)}."
        )
    return s


def _plan_status(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    s = value.strip().upper() if isinstance(value, str) else value
    if s not in PLAN_STATUSES:
        raise ValueError(
            f"Estado del plan desconocido: {value!r}. Tiene que ser uno de "
            f"{', '.join(PLAN_STATUSES)}."
        )
    return s


def _wallet_kind(value: Any) -> str:
    s = value.strip().lower() if isinstance(value, str) else value
    if s not in WALLET_EVENT_KINDS:
        raise ValueError(
            f"Tipo de movimiento desconocido: {value!r}. Tiene que ser 'deposit' (depósito), "
            "'withdrawal' (retiro) o 'balance' (saldo actual)."
        )
    return s


def _record_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise ValueError(f"{label} tiene que ser un número entero positivo, llegó {value!r}.")
    return int(value)


def _opt_record_id(value: Any, label: str) -> int | None:
    return None if value is None else _record_id(value, label)


def _limit(value: Any) -> int:
    return _record_id(value, "El límite")


def _aware(value: Any, label: str) -> datetime:
    """Fecha con zona horaria, pasada a UTC. Sin zona: ``ValueError``."""
    if not isinstance(value, datetime):
        raise ValueError(f"{label} tiene que ser una fecha, llegó {value!r}.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{label} no tiene zona horaria ({value.isoformat()}). Indicá la zona "
            "(por ejemplo -03:00 para Buenos Aires) o usá UTC."
        )
    return value.astimezone(UTC)


def _opt_aware(value: Any, label: str) -> datetime | None:
    return None if value is None else _aware(value, label)


def _jsonable(obj: Any) -> Any:
    """Como ``json_safe``, pero las fechas salen en ISO 8601 (no con ``str``)."""
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict) and not isinstance(obj, str | bytes):
        return _jsonable(to_dict())
    return json_safe(obj)


def _to_json_text(value: Any, label: str) -> str | None:
    """Dict (o algo con ``to_dict()``) → texto JSON. ``None`` queda en ``None``."""
    if value is None:
        return None
    data = _jsonable(value)
    if not isinstance(data, dict):
        raise ValueError(f"{label} tiene que ser un dict o tener to_dict(); llegó {value!r}.")
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _from_json_text(text: str | None) -> dict[str, Any] | None:
    return None if text is None else json.loads(text)


# --- repositorio ----------------------------------------------------------------


class Repository:
    """Acceso a la base de la app. Ver el docstring del módulo.

    ``clock`` devuelve "ahora" (UTC, con zona): lo usan los valores por
    defecto (cierre, apertura, fecha del movimiento, ``created_at``), el
    control de fechas futuras y ``latest_analysis(within=...)``. Los tests le
    pasan un reloj fijo.
    """

    def __init__(self, engine: Engine, *, clock: Clock = utc_now) -> None:
        self._engine = engine
        self._clock = clock

    @classmethod
    def open(cls, path: Path | str, *, clock: Clock = utc_now) -> Repository:
        """Abre (o crea) la base en ``path`` con las tablas listas."""
        return cls(open_db(path), clock=clock)

    @property
    def engine(self) -> Engine:
        return self._engine

    # --- utilidades internas -----------------------------------------------------

    def _now(self) -> datetime:
        return _aware(self._clock(), "La hora del reloj")

    def _not_future(self, dt: datetime, label: str) -> datetime:
        if dt > self._now() + MAX_CLOCK_SKEW:
            raise ValueError(
                f"{label} ({dt.isoformat()}) está en el futuro. Revisá la fecha y la hora."
            )
        return dt

    def _user_date(self, value: Any, label: str, *, default_now: bool = False) -> datetime:
        if value is None and default_now:
            return self._now()
        return self._not_future(_aware(value, label), label)

    @staticmethod
    def _range(since: Any, until: Any) -> tuple[datetime | None, datetime | None]:
        start = _opt_aware(since, "El inicio del rango")
        end = _opt_aware(until, "El fin del rango")
        if start is not None and end is not None and end < start:
            raise ValueError("El fin del rango no puede ser anterior al inicio.")
        return start, end

    @staticmethod
    def _linked_analysis(
        session: Session, analysis_id: int | None, symbol: str
    ) -> AnalysisRecord | None:
        """El análisis ligado (o ``None``); tiene que existir y ser de la misma moneda."""
        if analysis_id is None:
            return None
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise ValueError(f"No existe el análisis {analysis_id}.")
        if record.symbol != symbol:
            raise ValueError(
                f"El análisis {analysis_id} es de {record.symbol}, no de {symbol}. "
                "Revisá qué análisis seguiste."
            )
        return record

    # --- operaciones cerradas -----------------------------------------------------

    def add_trade(
        self,
        *,
        symbol: str,
        side: str,
        pnl_usd: float,
        opened_at: datetime,
        closed_at: datetime | None = None,
        risk_usd: float | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        notes: str | None = None,
        analysis_id: int | None = None,
        regime_category: str | None = None,
        regime_name: str | None = None,
        regime_confidence: float | None = None,
        score_total: float | None = None,
        plan_status: str | None = None,
        profile_name: str | None = None,
    ) -> StoredTrade:
        """Guarda una operación cerrada.

        ``pnl_usd``: resultado neto realizado, comisiones incluidas (negativo =
        pérdida; 0 = empate). ``closed_at`` por defecto es ahora. Si viene
        ``analysis_id``, el contexto que falte se copia de ese análisis.
        """
        with session_scope(self._engine) as session:
            row = self._build_trade(
                session,
                symbol=symbol,
                side=side,
                pnl_usd=pnl_usd,
                opened_at=opened_at,
                closed_at=closed_at,
                risk_usd=risk_usd,
                entry_price=entry_price,
                exit_price=exit_price,
                notes=notes,
                analysis_id=analysis_id,
                regime_category=regime_category,
                regime_name=regime_name,
                regime_confidence=regime_confidence,
                score_total=score_total,
                plan_status=plan_status,
                profile_name=profile_name,
            )
            session.add(row)
            session.flush()
            return StoredTrade._from_row(row)

    def _build_trade(self, session: Session, **kw: Any) -> ManualTrade:
        symbol = _symbol(kw["symbol"])
        side = _side(kw["side"])
        pnl = _number(kw["pnl_usd"], "El resultado de la operación")
        opened_at = self._user_date(kw["opened_at"], "La fecha de apertura")
        closed_at = self._user_date(kw["closed_at"], "La fecha de cierre", default_now=True)
        if closed_at < opened_at:
            raise ValueError(
                f"La operación no puede cerrar ({closed_at.isoformat()}) antes de abrir "
                f"({opened_at.isoformat()})."
            )
        analysis_id = _opt_record_id(kw["analysis_id"], "El número de análisis")
        ctx = {
            "regime_category": _regime_category(kw["regime_category"]),
            "regime_name": _opt_text(kw["regime_name"], "El nombre del régimen", 60),
            "regime_confidence": _confidence(kw["regime_confidence"]),
            "score_total": _score_total(kw["score_total"]),
            "plan_status": _plan_status(kw["plan_status"], required=False),
            "profile_name": _opt_text(kw["profile_name"], "El perfil", 60),
        }
        record = self._linked_analysis(session, analysis_id, symbol)
        if record is not None:
            for key, value in ctx.items():
                if value is None:
                    ctx[key] = getattr(record, key)
        return ManualTrade(
            symbol=symbol,
            side=side,
            opened_at=opened_at,
            closed_at=closed_at,
            pnl_usd=pnl,
            risk_usd=_opt_positive(kw["risk_usd"], "El riesgo"),
            entry_price=_opt_positive(kw["entry_price"], "El precio de entrada"),
            exit_price=_opt_positive(kw["exit_price"], "El precio de salida"),
            notes=_notes(kw["notes"]),
            analysis_id=analysis_id,
            created_at=self._now(),
            **ctx,
        )

    def get_trade(self, trade_id: int) -> StoredTrade | None:
        """La operación ``trade_id``, o ``None`` si no existe."""
        tid = _record_id(trade_id, "El número de operación")
        with session_scope(self._engine) as session:
            row = session.get(ManualTrade, tid)
            return None if row is None else StoredTrade._from_row(row)

    def list_trades(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        symbol: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[StoredTrade]:
        """Operaciones cerradas, filtradas por fecha de **cierre** (``since <= closed_at < until``).

        Orden: por cierre, de la más vieja a la más nueva (``newest_first=True``
        al revés). Con ``limit`` y ``newest_first=True`` salen las últimas N
        (por ejemplo, para contar la racha de pérdidas).
        """
        start, end = self._range(since, until)
        stmt = select(ManualTrade)
        if start is not None:
            stmt = stmt.where(col(ManualTrade.closed_at) >= start)
        if end is not None:
            stmt = stmt.where(col(ManualTrade.closed_at) < end)
        if symbol is not None:
            stmt = stmt.where(col(ManualTrade.symbol) == _symbol(symbol))
        if newest_first:
            stmt = stmt.order_by(col(ManualTrade.closed_at).desc(), col(ManualTrade.id).desc())
        else:
            stmt = stmt.order_by(col(ManualTrade.closed_at), col(ManualTrade.id))
        if limit is not None:
            stmt = stmt.limit(_limit(limit))
        with session_scope(self._engine) as session:
            return [StoredTrade._from_row(r) for r in session.exec(stmt).all()]

    def delete_trade(self, trade_id: int) -> bool:
        """Borra la operación (para corregir una carga equivocada). ``False`` si no existía."""
        tid = _record_id(trade_id, "El número de operación")
        with session_scope(self._engine) as session:
            row = session.get(ManualTrade, tid)
            if row is None:
                return False
            session.delete(row)
            return True

    # --- posiciones abiertas ------------------------------------------------------

    def add_open_position(
        self,
        *,
        symbol: str,
        side: str,
        risk_usd: float,
        opened_at: datetime | None = None,
        entry_price: float | None = None,
        notes: str | None = None,
        analysis_id: int | None = None,
    ) -> StoredPosition:
        """Guarda una posición abierta. ``risk_usd``: USD que se pierden si toca el stop.

        ``opened_at`` por defecto es ahora. Se permiten varias posiciones en la
        misma moneda (la correlación suma sus riesgos).
        """
        sym = _symbol(symbol)
        row = OpenPosition(
            symbol=sym,
            side=_side(side),
            opened_at=self._user_date(opened_at, "La fecha de apertura", default_now=True),
            entry_price=_opt_positive(entry_price, "El precio de entrada"),
            risk_usd=_positive(risk_usd, "El riesgo"),
            notes=_notes(notes),
            analysis_id=_opt_record_id(analysis_id, "El número de análisis"),
            created_at=self._now(),
        )
        with session_scope(self._engine) as session:
            self._linked_analysis(session, row.analysis_id, sym)
            session.add(row)
            session.flush()
            return StoredPosition._from_row(row)

    def get_open_position(self, position_id: int) -> StoredPosition | None:
        """La posición ``position_id``, o ``None`` si no existe."""
        pid = _record_id(position_id, "El número de posición")
        with session_scope(self._engine) as session:
            row = session.get(OpenPosition, pid)
            return None if row is None else StoredPosition._from_row(row)

    def list_open_positions(self, *, symbol: str | None = None) -> list[StoredPosition]:
        """Posiciones abiertas, de la más vieja a la más nueva."""
        stmt = select(OpenPosition)
        if symbol is not None:
            stmt = stmt.where(col(OpenPosition.symbol) == _symbol(symbol))
        stmt = stmt.order_by(col(OpenPosition.opened_at), col(OpenPosition.id))
        with session_scope(self._engine) as session:
            return [StoredPosition._from_row(r) for r in session.exec(stmt).all()]

    def close_open_position(self, position_id: int) -> StoredPosition | None:
        """Saca la posición de la lista (sin registrar resultado). Devuelve lo que borró.

        ``None`` si no existía. Para cerrarla y anotar el resultado en un solo
        paso, ``close_position_with_result``.
        """
        pid = _record_id(position_id, "El número de posición")
        with session_scope(self._engine) as session:
            row = session.get(OpenPosition, pid)
            if row is None:
                return None
            out = StoredPosition._from_row(row)
            session.delete(row)
            return out

    def close_position_with_result(
        self,
        position_id: int,
        *,
        pnl_usd: float,
        closed_at: datetime | None = None,
        exit_price: float | None = None,
        notes: str | None = None,
    ) -> StoredTrade:
        """Cierra la posición y la guarda como operación, todo junto (o nada).

        La operación hereda moneda, dirección, apertura, precio de entrada,
        riesgo, notas (si no se pasan otras) y ``analysis_id`` (con su
        contexto). ``ValueError`` si la posición no existe.
        """
        pid = _record_id(position_id, "El número de posición")
        with session_scope(self._engine) as session:
            pos = session.get(OpenPosition, pid)
            if pos is None:
                raise ValueError(f"No existe la posición abierta {pid}.")
            row = self._build_trade(
                session,
                symbol=pos.symbol,
                side=pos.side,
                pnl_usd=pnl_usd,
                opened_at=pos.opened_at,
                closed_at=closed_at,
                risk_usd=pos.risk_usd,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                notes=pos.notes if _notes(notes) is None else notes,
                analysis_id=pos.analysis_id,
                regime_category=None,
                regime_name=None,
                regime_confidence=None,
                score_total=None,
                plan_status=None,
                profile_name=None,
            )
            session.add(row)
            session.delete(pos)
            session.flush()
            return StoredTrade._from_row(row)

    # --- billetera ----------------------------------------------------------------

    def add_wallet_event(
        self,
        *,
        kind: str,
        amount_usd: float,
        ts: datetime | None = None,
        notes: str | None = None,
    ) -> StoredWalletEvent:
        """Guarda un movimiento de la billetera de futuros (``ts`` por defecto es ahora).

        ``deposit`` y ``withdrawal`` piden ``amount_usd > 0`` (el signo lo da
        el tipo); ``balance`` (el saldo que el usuario ve ahora) pide ``>= 0``.
        """
        k = _wallet_kind(kind)
        if k == "balance":
            amount = _number(amount_usd, "El saldo")
            if amount < 0:
                raise ValueError(f"El saldo no puede ser negativo, llegó {amount:g}.")
        else:
            label = "El monto del depósito" if k == "deposit" else "El monto del retiro"
            amount = _positive(amount_usd, label)
        row = WalletEvent(
            ts=self._user_date(ts, "La fecha del movimiento", default_now=True),
            kind=k,
            amount_usd=amount,
            notes=_notes(notes),
        )
        with session_scope(self._engine) as session:
            session.add(row)
            session.flush()
            return StoredWalletEvent._from_row(row)

    def list_wallet_events(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        kind: str | None = None,
    ) -> list[StoredWalletEvent]:
        """Movimientos con ``since <= ts < until``, del más viejo al más nuevo."""
        start, end = self._range(since, until)
        stmt = select(WalletEvent)
        if start is not None:
            stmt = stmt.where(col(WalletEvent.ts) >= start)
        if end is not None:
            stmt = stmt.where(col(WalletEvent.ts) < end)
        if kind is not None:
            stmt = stmt.where(col(WalletEvent.kind) == _wallet_kind(kind))
        stmt = stmt.order_by(col(WalletEvent.ts), col(WalletEvent.id))
        with session_scope(self._engine) as session:
            return [StoredWalletEvent._from_row(r) for r in session.exec(stmt).all()]

    def delete_wallet_event(self, event_id: int) -> bool:
        """Borra un movimiento (para corregir una carga equivocada). ``False`` si no existía."""
        eid = _record_id(event_id, "El número de movimiento")
        with session_scope(self._engine) as session:
            row = session.get(WalletEvent, eid)
            if row is None:
                return False
            session.delete(row)
            return True

    # --- eventos de alta volatilidad ---------------------------------------------

    def add_volatility_event(
        self,
        note: str,
        until: datetime,
        starts_at: datetime | None = None,
    ) -> StoredVolatilityEvent:
        """Marca un evento de alta volatilidad: bloquea todos los planes hasta ``until``.

        - ``note``: qué evento es (obligatoria, se muestra en el aviso).
        - ``until``: hora de fin (con zona horaria); tiene que ser posterior a
          ahora y al inicio. A esa hora el evento deja de bloquear solo.
        - ``starts_at``: desde cuándo bloquea; por defecto, ahora. Puede estar
          en el futuro (para marcar con anticipación el dato de inflación de
          mañana, por ejemplo).
        """
        text = _text(note, "La nota del evento", MAX_EVENT_NOTE_LENGTH)
        now = self._now()
        start = now if starts_at is None else _aware(starts_at, "El inicio del evento")
        end = _aware(until, "La hora de fin del evento")
        if end <= now:
            raise ValueError(
                f"La hora de fin del evento ({end.isoformat()}) ya pasó. Poné una hora futura."
            )
        if end <= start:
            raise ValueError(
                f"La hora de fin del evento ({end.isoformat()}) tiene que ser posterior al "
                f"inicio ({start.isoformat()})."
            )
        row = HighVolatilityEvent(created_at=now, starts_at=start, until=end, note=text)
        with session_scope(self._engine) as session:
            session.add(row)
            session.flush()
            return StoredVolatilityEvent._from_row(row)

    def get_volatility_event(self, event_id: int) -> StoredVolatilityEvent | None:
        """El evento ``event_id``, o ``None`` si no existe."""
        eid = _record_id(event_id, "El número de evento")
        with session_scope(self._engine) as session:
            row = session.get(HighVolatilityEvent, eid)
            return None if row is None else StoredVolatilityEvent._from_row(row)

    def list_volatility_events(
        self, *, ending_after: datetime | None = None
    ) -> list[StoredVolatilityEvent]:
        """Eventos ordenados por inicio (y por número).

        Con ``ending_after``, solo los que todavía no terminaron en ese momento
        (``until > ending_after``): los activos y los programados para más tarde.
        """
        stmt = select(HighVolatilityEvent)
        if ending_after is not None:
            moment = _aware(ending_after, "El momento")
            stmt = stmt.where(col(HighVolatilityEvent.until) > moment)
        stmt = stmt.order_by(col(HighVolatilityEvent.starts_at), col(HighVolatilityEvent.id))
        with session_scope(self._engine) as session:
            return [StoredVolatilityEvent._from_row(r) for r in session.exec(stmt).all()]

    def active_volatility_events(self, now: datetime | None = None) -> list[StoredVolatilityEvent]:
        """Los eventos que bloquean en ``now`` (por defecto, ahora).

        Activo = ``starts_at <= now < until``: a la hora de fin ya no bloquea.
        """
        moment = self._now() if now is None else _aware(now, "La hora")
        stmt = (
            select(HighVolatilityEvent)
            .where(col(HighVolatilityEvent.starts_at) <= moment)
            .where(col(HighVolatilityEvent.until) > moment)
            .order_by(col(HighVolatilityEvent.starts_at), col(HighVolatilityEvent.id))
        )
        with session_scope(self._engine) as session:
            return [StoredVolatilityEvent._from_row(r) for r in session.exec(stmt).all()]

    def delete_volatility_event(self, event_id: int) -> bool:
        """Borra el evento (para terminarlo antes o corregirlo). ``False`` si no existía."""
        eid = _record_id(event_id, "El número de evento")
        with session_scope(self._engine) as session:
            row = session.get(HighVolatilityEvent, eid)
            if row is None:
                return False
            session.delete(row)
            return True

    # --- análisis -----------------------------------------------------------------

    def save_analysis(
        self,
        *,
        symbol: str,
        profile_name: str,
        price: float,
        price_source: str,
        plan_status: str,
        plan: Any = None,
        score: Any = None,
        safety: Any = None,
        llm: Any = None,
        inputs: Any = None,
        regime_category: str | None = None,
        regime_name: str | None = None,
        regime_confidence: float | None = None,
        regime_unstable: bool | None = None,
        score_total: float | None = None,
        created_at: datetime | None = None,
    ) -> StoredAnalysis:
        """Guarda un análisis.

        ``plan``, ``score``, ``safety``, ``llm`` e ``inputs`` pueden ser dicts u
        objetos con ``to_dict()`` (``TradePlan``, ``ConfluenceScore``, el
        veredicto de seguridad, ...); se guardan como JSON (NaN e infinitos →
        ``null``, tuplas → listas, fechas en ISO 8601). ``created_at`` por
        defecto es ahora.
        """
        row = AnalysisRecord(
            created_at=self._user_date(created_at, "La fecha del análisis", default_now=True),
            symbol=_symbol(symbol),
            profile_name=_text(profile_name, "El perfil", 60),
            price=_positive(price, "El precio"),
            price_source=_text(price_source, "La fuente del precio", 40),
            regime_category=_regime_category(regime_category),
            regime_name=_opt_text(regime_name, "El nombre del régimen", 60),
            regime_confidence=_confidence(regime_confidence),
            regime_unstable=_opt_bool(regime_unstable, "La marca de régimen inestable"),
            score_total=_score_total(score_total),
            plan_status=_plan_status(plan_status, required=True),
            plan_json=_to_json_text(plan, "El plan"),
            score_json=_to_json_text(score, "El score"),
            safety_json=_to_json_text(safety, "El veredicto de seguridad"),
            llm_json=_to_json_text(llm, "La lectura del LLM"),
            inputs_json=_to_json_text(inputs, "Los datos de entrada"),
        )
        with session_scope(self._engine) as session:
            session.add(row)
            session.flush()
            return StoredAnalysis._from_row(row)

    def set_analysis_safety(self, analysis_id: int, safety: Any) -> StoredAnalysis:
        """Reemplaza el veredicto de seguridad de un análisis ya guardado.

        ``ValueError`` si el análisis no existe.
        """
        aid = _record_id(analysis_id, "El número de análisis")
        text = _to_json_text(safety, "El veredicto de seguridad")
        with session_scope(self._engine) as session:
            row = session.get(AnalysisRecord, aid)
            if row is None:
                raise ValueError(f"No existe el análisis {aid}.")
            row.safety_json = text
            session.add(row)
            session.flush()
            return StoredAnalysis._from_row(row)

    def get_analysis(self, analysis_id: int) -> StoredAnalysis | None:
        """El análisis ``analysis_id``, o ``None`` si no existe."""
        aid = _record_id(analysis_id, "El número de análisis")
        with session_scope(self._engine) as session:
            row = session.get(AnalysisRecord, aid)
            return None if row is None else StoredAnalysis._from_row(row)

    def latest_analysis(
        self, symbol: str, *, within: timedelta | None = None
    ) -> StoredAnalysis | None:
        """El último análisis de ``symbol``; con ``within``, solo si es de ese lapso para acá.

        Ejemplo: ``latest_analysis("BTC", within=timedelta(minutes=15))`` →
        ``None`` si el último tiene más de 15 minutos.
        """
        sym = _symbol(symbol)
        stmt = select(AnalysisRecord).where(col(AnalysisRecord.symbol) == sym)
        if within is not None:
            if not isinstance(within, timedelta) or within <= timedelta(0):
                raise ValueError(f"El lapso tiene que ser positivo, llegó {within!r}.")
            stmt = stmt.where(col(AnalysisRecord.created_at) >= self._now() - within)
        stmt = stmt.order_by(col(AnalysisRecord.created_at).desc(), col(AnalysisRecord.id).desc())
        with session_scope(self._engine) as session:
            row = session.exec(stmt.limit(1)).first()
            return None if row is None else StoredAnalysis._from_row(row)

    def list_analyses(
        self,
        *,
        limit: int = 50,
        symbol: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StoredAnalysis]:
        """Los últimos ``limit`` análisis, del más nuevo al más viejo.

        Filtros opcionales: moneda y rango ``since <= created_at < until``.
        """
        start, end = self._range(since, until)
        stmt = select(AnalysisRecord)
        if symbol is not None:
            stmt = stmt.where(col(AnalysisRecord.symbol) == _symbol(symbol))
        if start is not None:
            stmt = stmt.where(col(AnalysisRecord.created_at) >= start)
        if end is not None:
            stmt = stmt.where(col(AnalysisRecord.created_at) < end)
        stmt = stmt.order_by(col(AnalysisRecord.created_at).desc(), col(AnalysisRecord.id).desc())
        stmt = stmt.limit(_limit(limit))
        with session_scope(self._engine) as session:
            return [StoredAnalysis._from_row(r) for r in session.exec(stmt).all()]


__all__ = [
    "MAX_CLOCK_SKEW",
    "MAX_EVENT_NOTE_LENGTH",
    "MAX_NOTES_LENGTH",
    "MAX_SYMBOL_LENGTH",
    "PLAN_STATUSES",
    "REGIME_CATEGORIES",
    "SIDES",
    "WALLET_EVENT_KINDS",
    "Clock",
    "Repository",
    "StoredAnalysis",
    "StoredPosition",
    "StoredTrade",
    "StoredVolatilityEvent",
    "StoredWalletEvent",
]
