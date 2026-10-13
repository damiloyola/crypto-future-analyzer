"""Informe del incidente en markdown.

Cuándo se usa
    Cuando salta el bloqueo total (la billetera del mes cae
    ``circuit_breakers.monthly_drawdown_lock.pct_from_peak`` % o más desde su
    máximo), la capa de seguridad escribe este informe y **después** el archivo
    ``LOCKED`` (``safety.lockfile``), que apunta acá. La app no genera planes
    hasta que el usuario lea el informe y borre ``LOCKED`` a mano.

Qué tiene el informe (en castellano simple, para alguien que hace mucho que no
opera futuros)
    1. **Resumen con números**: máximo de la billetera, saldo actual, cuánto
       cayó (y el tope), cuántas operaciones, cuántas ganadas y perdidas,
       resultado neto, la peor operación, la racha de pérdidas más larga y las
       posiciones que siguen abiertas.
    2. **Qué pasó, en simple**: por qué se bloqueó la app y qué significa.
    3. **Qué falló**: observaciones automáticas calculadas con los datos
       (``find_observations``); ver "Observaciones" más abajo.
    4. **Operaciones del mes**: tabla con fecha, moneda, dirección, resultado,
       riesgo previsto, régimen y score del momento, qué decía el plan y los
       avisos de cada una. Los números ``#`` son los mismos que usan las
       observaciones.
    5. **Movimientos de la billetera** y **posiciones abiertas**.
    6. **Antes de borrar LOCKED**: lista para tildar, con puntos extra según lo
       que se encontró.
    7. **Datos técnicos**: código del disparador, horas en UTC y perfil.

Observaciones (cada una con su ``code``, en el orden en que salen)
    - ``traded_on_no_trade``: operaciones hechas cuando el plan decía
      "No operar".
    - ``against_plan``: el plan decía long y se operó short, o al revés.
    - ``against_regime``: long con el mercado bajista o en crash; short con el
      mercado alcista o en euforia (neutral nunca es "en contra").
    - ``loss_above_risk``: se perdió más que el riesgo previsto (redondeado a
      centavos; el riesgo ya incluye comisiones).
    - ``losing_streak``: rachas de ``losing_streak`` pérdidas seguidas o más
      (el mismo número que el breaker de enfriamiento).
    - ``too_many_trades``: días con más operaciones que el máximo del perfil
      (necesita el perfil).
    - ``daily_loss_limit``: días en que la pérdida neta pasó el límite diario
      del perfil (necesita el perfil; llegar justo al límite no cuenta).
    - ``deposit_after_loss``: depósitos hechos el mismo día, después de una
      pérdida.
    - ``missing_risk``: operaciones sin el riesgo previsto anotado.
    - ``no_analysis_context``: operaciones sin análisis de la app asociado (no
      se sabe qué decían el régimen, el score ni el plan).

    Son hechos calculados por el código con los datos cargados, no opiniones
    del LLM. Si no hay ninguno, el informe lo dice y pide revisar igual.

Qué hay acá (todo puro salvo ``write_incident_report`` y ``create_new_file``)
    - ``write_incident_report(incident_dir, *, triggered_at, trigger, ...)``:
      arma el informe y lo guarda; devuelve la ruta.
    - ``build_incident_report(*, triggered_at, trigger, ...)``: el markdown.
    - ``find_observations(trades, ...)``, ``summarize_trades(trades)`` y
      ``losing_streaks(trades, min_length)``.
    - ``incident_file_name(triggered_at, attempt)``: nombre del archivo.
    - ``IncidentTrigger``, ``IncidentTrade``, ``IncidentWalletEvent``,
      ``IncidentPosition`` e ``IncidentProfile``: los datos ya normalizados
      (``from_value`` acepta dataclasses del ``storage``, objetos con esos
      atributos o dicts).
    - Formateadores: ``fmt_date``, ``fmt_datetime``, ``fmt_long_datetime``,
      ``fmt_month``, ``fmt_pnl``, ``fmt_side``, ``fmt_plan_status``,
      ``fmt_regime``, ``md_cell``, ``timezone_label``, ``resolve_timezone`` y
      ``to_utc`` (lectura tolerante de fechas). ``safety.lockfile`` usa los de
      fechas.
    - ``create_new_file(path, text)``: escritura atómica que nunca pisa un
      archivo existente (también la usa ``safety.lockfile``).

Decisiones
    - **Tolerante con los datos**: el informe se escribe en el peor momento y
      no puede fallar por un dato raro. Un número que falta o no es finito se
      muestra como "—" y no entra en las cuentas; un lado o estado desconocido
      se muestra tal cual. Solo un error de programación (``triggered_at`` que
      no es fecha, ``losing_streak`` < 1) lanza excepción.
    - **Horas**: los datos vienen en UTC; se muestran en la zona del usuario
      (``tz``, la de ``app.user_timezone``). Una fecha sin zona horaria se toma
      como UTC. Una zona desconocida cae a UTC (y el informe dice "UTC").
    - **Nunca pisa**: el nombre lleva fecha y hora UTC del disparo; si ya
      existe, se agrega ``_2``, ``_3``... La creación es exclusiva (si dos
      procesos escriben a la vez, cada uno queda con su archivo).
    - **Racha de pérdidas**: operaciones seguidas con resultado < 0 (a
      centavos). Un resultado 0 o positivo la corta.
    - **Pérdida del día**: neta (ganancias menos pérdidas cerradas ese día en la
      zona del usuario), igual que ``circuit_breakers.loss_basis: "net"``.
    - **Qué operaciones**: las que pasa quien llama (las del mes de la
      billetera); este módulo no filtra por fecha.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.risk.plan import STATUS_LONG, STATUS_NO_TRADE, STATUS_SHORT
from app.risk.regime_adjustments import CATEGORY_LABELS
from app.risk.rules import (
    fmt_num,
    fmt_percent,
    fmt_percent_beside,
    fmt_price,
    fmt_score,
    fmt_usd,
    json_safe,
)

# Racha de pérdidas que se marca si quien llama no pasa la de la config
# (``circuit_breakers.losing_streak.consecutive_losses``).
DEFAULT_LOSING_STREAK = 3

# Códigos de las observaciones.
OBS_TRADED_ON_NO_TRADE = "traded_on_no_trade"
OBS_AGAINST_PLAN = "against_plan"
OBS_AGAINST_REGIME = "against_regime"
OBS_LOSS_ABOVE_RISK = "loss_above_risk"
OBS_MISSING_RISK = "missing_risk"
OBS_LOSING_STREAK = "losing_streak"
OBS_NO_ANALYSIS_CONTEXT = "no_analysis_context"
OBS_TOO_MANY_TRADES = "too_many_trades"
OBS_DAILY_LOSS_LIMIT = "daily_loss_limit"
OBS_DEPOSIT_AFTER_LOSS = "deposit_after_loss"

FILE_PREFIX = "incidente_"
FILE_SUFFIX = ".md"
# Intentos de nombre (``_2``, ``_3``...) antes de rendirse. No es una regla de
# riesgo: con un informe por disparo nunca se llega.
MAX_NAME_ATTEMPTS = 1000

DASH = "—"

# Régimen "en contra" de cada dirección.
_AGAINST_LONG = frozenset({"crash", "bajista"})
_AGAINST_SHORT = frozenset({"alcista", "euforia"})

_SIDE_LABELS = {"long": "Long", "short": "Short"}
_PLAN_LABELS = {STATUS_LONG: "Long", STATUS_SHORT: "Short", STATUS_NO_TRADE: '"No operar"'}
_PLAN_ALIASES = {
    "LONG": STATUS_LONG,
    "SHORT": STATUS_SHORT,
    "NO_OPERAR": STATUS_NO_TRADE,
    "NO_TRADE": STATUS_NO_TRADE,
}
_WALLET_KIND_LABELS = {
    "deposit": "Depósito",
    "withdrawal": "Retiro",
    "balance": "Saldo anotado",
}
_UTC_KEYS = frozenset({"UTC", "Etc/UTC", "Etc/UCT", "Etc/GMT", "GMT", "Zulu", "Etc/Zulu"})
_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


# --- lectura tolerante de datos ------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    """Atributo o clave ``name`` de un dict, dataclass u objeto; ``None`` si no está."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _finite(value: Any) -> float | None:
    """Número finito o ``None`` (no lanza nunca)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _text(value: Any) -> str | None:
    """Texto sin espacios de más, o ``None`` si está vacío o no hay."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_utc(value: Any) -> datetime | None:
    """Fecha en UTC desde ``datetime`` o texto ISO; sin zona = UTC; otra cosa = ``None``.

    No lanza nunca: es la lectura tolerante de fechas del informe y del lock.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _require_datetime(value: Any, name: str) -> datetime:
    """Como ``to_utc`` pero exige un ``datetime`` (error de programación si no)."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} tiene que ser una fecha (datetime), llegó {type(value).__name__}")
    result = to_utc(value)
    assert result is not None  # un datetime siempre se puede pasar a UTC
    return result


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _one_line(text: str) -> str:
    """Junta los renglones y espacios repetidos en uno solo."""
    return re.sub(r"\s+", " ", text).strip()


# --- formato de fechas ----------------------------------------------------------


def resolve_timezone(tz: str | tzinfo | None) -> tzinfo:
    """Zona horaria a partir del nombre IANA (``"America/Argentina/Buenos_Aires"``).

    ``None``, un nombre vacío o desconocido -> UTC (el informe nunca falla por
    la zona; ``timezone_label`` va a decir "UTC").
    """
    if isinstance(tz, tzinfo):
        return tz
    name = _text(tz)
    if name is None:
        return UTC
    try:
        return ZoneInfo(name)
    except (ValueError, KeyError, OSError):  # ZoneInfoNotFoundError es KeyError
        return UTC


def timezone_label(tz: str | tzinfo | None) -> str:
    """Cómo se nombra la zona en el texto: ``"hora de Buenos Aires"`` o ``"UTC"``."""
    zone = resolve_timezone(tz)
    key = getattr(zone, "key", None)
    if zone is UTC or key in _UTC_KEYS:
        return "UTC"
    if key:
        city = str(key).rsplit("/", 1)[-1].replace("_", " ")
        return f"hora de {city}"
    return str(zone)  # zona fija sin nombre, por ejemplo "UTC-03:00"


def _local(dt: Any, tz: str | tzinfo | None) -> datetime | None:
    value = to_utc(dt)
    return None if value is None else value.astimezone(resolve_timezone(tz))


def fmt_date(dt: Any, tz: str | tzinfo | None = None) -> str:
    """Fecha corta en la zona ``tz``: ``"13/10/2026"``. Sin fecha -> ``"—"``."""
    local = _local(dt, tz)
    return DASH if local is None else local.strftime("%d/%m/%Y")


def fmt_datetime(dt: Any, tz: str | tzinfo | None = None) -> str:
    """Fecha y hora en la zona ``tz``: ``"13/10/2026 11:05"``. Sin fecha -> ``"—"``."""
    local = _local(dt, tz)
    return DASH if local is None else local.strftime("%d/%m/%Y %H:%M")


def fmt_long_datetime(dt: Any, tz: str | tzinfo | None = None) -> str:
    """Fecha larga: ``"martes 13 de octubre de 2026, 11:05"``. Sin fecha -> ``"—"``."""
    local = _local(dt, tz)
    if local is None:
        return DASH
    weekday = _WEEKDAYS[local.weekday()]
    month = _MONTHS[local.month - 1]
    return f"{weekday} {local.day} de {month} de {local.year}, {local:%H:%M}"


def fmt_month(dt: Any, tz: str | tzinfo | None = None) -> str:
    """Mes y año: ``"octubre de 2026"``. Sin fecha -> ``"—"``."""
    local = _local(dt, tz)
    return DASH if local is None else f"{_MONTHS[local.month - 1]} de {local.year}"


# --- formato de montos y textos -------------------------------------------------


def fmt_pnl(value: Any) -> str:
    """Resultado con signo: ``30`` -> ``"+30 USD"``, ``-45.5`` -> ``"-45,50 USD"``.

    ``0`` -> ``"0 USD"``; sin número -> ``"—"``.
    """
    v = _finite(value)
    if v is None:
        return DASH
    v = round(v, 2)
    if v == 0:
        return "0 USD"
    return f"{'+' if v > 0 else '-'}{fmt_usd(abs(v))}"


def fmt_side(side: Any) -> str:
    """``"long"`` -> ``"Long"``, ``"short"`` -> ``"Short"``; otro texto, tal cual."""
    text = _text(side)
    if text is None:
        return DASH
    return _SIDE_LABELS.get(text.lower(), text)


def normalize_side(side: Any) -> str | None:
    """``"LONG"``/``" long "`` -> ``"long"``; otro texto queda en minúsculas; vacío -> ``None``."""
    text = _text(side)
    return None if text is None else text.lower()


def normalize_plan_status(status: Any) -> str | None:
    """Estado del plan normalizado: ``"No operar"``/``"no_operar"`` -> ``"NO_OPERAR"``.

    ``"long"`` -> ``"LONG"``, ``"short"`` -> ``"SHORT"``. Un estado desconocido
    queda tal cual (sin espacios de más); vacío -> ``None``.
    """
    text = _text(status)
    if text is None:
        return None
    key = re.sub(r"[\s_\-]+", "_", text.upper())
    return _PLAN_ALIASES.get(key, text)


def fmt_plan_status(status: Any) -> str:
    """Qué decía el plan: ``"Long"``, ``"Short"``, ``'"No operar"'`` o ``"Sin análisis"``."""
    normalized = normalize_plan_status(status)
    if normalized is None:
        return "Sin análisis"
    return _PLAN_LABELS.get(normalized, normalized)


def fmt_regime(category: Any, name: Any = None, confidence: Any = None) -> str:
    """Régimen del momento: ``"Bajista (78 %)"``; sin datos -> ``"—"``.

    Usa el nombre si lo hay; si no, el de la categoría. ``confidence`` es la
    probabilidad 0-1 (como la da el detector); un valor mayor que 1 se toma
    como porcentaje ya hecho.
    """
    label = _text(name)
    cat = _text(category)
    if label is None and cat is not None:
        label = CATEGORY_LABELS.get(cat.lower(), cat)
    if label is None:
        return DASH
    conf = _finite(confidence)
    if conf is None or conf < 0:
        return label
    pct = conf * 100.0 if conf <= 1 else conf
    return f"{label} ({fmt_percent(pct, 0)})"


def md_cell(value: Any) -> str:
    """Texto seguro para una celda de tabla markdown (sin ``|`` sueltos ni saltos de línea)."""
    text = _text(value)
    if text is None:
        return DASH
    return _one_line(text).replace("\\", "\\\\").replace("|", "\\|")


# --- datos normalizados ---------------------------------------------------------


@dataclass(frozen=True)
class IncidentTrigger:
    """Qué disparó el bloqueo.

    - ``code``: código del breaker (por ejemplo ``"monthly_drawdown_lock"``).
    - ``description``: texto en castellano del breaker (va tal cual al informe).
    - ``wallet_peak_usd`` / ``current_balance_usd``: máximo del mes y saldo
      actual de la billetera.
    - ``drawdown_pct``: cuánto cayó desde el máximo (0-100). Si no viene y hay
      máximo y saldo, se calcula.
    - ``threshold_pct``: el tope de caída (``pct_from_peak``).
    """

    code: str
    description: str
    wallet_peak_usd: float | None = None
    current_balance_usd: float | None = None
    drawdown_pct: float | None = None
    threshold_pct: float | None = None

    def __post_init__(self) -> None:
        peak = self.wallet_peak_usd
        balance = self.current_balance_usd
        if self.drawdown_pct is None and peak is not None and peak > 0 and balance is not None:
            object.__setattr__(self, "drawdown_pct", (peak - balance) / peak * 100.0)

    @classmethod
    def from_value(cls, value: Any) -> IncidentTrigger:
        """Desde un ``IncidentTrigger``, un dict o un objeto con esos atributos."""
        if isinstance(value, IncidentTrigger):
            return value
        return cls(
            code=_text(_field(value, "code")) or "desconocido",
            description=_text(_field(value, "description")) or "sin descripción",
            wallet_peak_usd=_finite(_field(value, "wallet_peak_usd")),
            current_balance_usd=_finite(_field(value, "current_balance_usd")),
            drawdown_pct=_finite(_field(value, "drawdown_pct")),
            threshold_pct=_finite(_field(value, "threshold_pct")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "code": self.code,
                "description": self.description,
                "wallet_peak_usd": self.wallet_peak_usd,
                "current_balance_usd": self.current_balance_usd,
                "drawdown_pct": self.drawdown_pct,
                "threshold_pct": self.threshold_pct,
            }
        )


@dataclass(frozen=True)
class IncidentTrade:
    """Una operación cerrada, con el contexto del análisis si lo hubo.

    ``pnl_usd`` es el resultado neto (negativo = pérdida). ``risk_usd`` es lo
    que se iba a perder si tocaba el stop. ``plan_status`` viene normalizado
    (``"LONG"``, ``"SHORT"``, ``"NO_OPERAR"``) y ``side`` en minúsculas.
    """

    symbol: str
    side: str | None
    opened_at: datetime | None
    closed_at: datetime | None
    pnl_usd: float | None
    risk_usd: float | None = None
    regime_category: str | None = None
    regime_name: str | None = None
    regime_confidence: float | None = None
    score_total: float | None = None
    plan_status: str | None = None
    notes: str | None = None
    analysis_id: int | None = None

    @classmethod
    def from_value(cls, value: Any) -> IncidentTrade:
        """Desde un ``StoredTrade``, un dict o un objeto con esos atributos."""
        if isinstance(value, IncidentTrade):
            return value
        category = _text(_field(value, "regime_category"))
        return cls(
            symbol=(_text(_field(value, "symbol")) or DASH).upper(),
            side=normalize_side(_field(value, "side")),
            opened_at=to_utc(_field(value, "opened_at")),
            closed_at=to_utc(_field(value, "closed_at")),
            pnl_usd=_finite(_field(value, "pnl_usd")),
            risk_usd=_finite(_field(value, "risk_usd")),
            regime_category=None if category is None else category.lower(),
            regime_name=_text(_field(value, "regime_name")),
            regime_confidence=_finite(_field(value, "regime_confidence")),
            score_total=_finite(_field(value, "score_total")),
            plan_status=normalize_plan_status(_field(value, "plan_status")),
            notes=_text(_field(value, "notes")),
            analysis_id=_int_or_none(_field(value, "analysis_id")),
        )

    @property
    def is_loss(self) -> bool:
        """Resultado negativo (a centavos)."""
        return self.pnl_usd is not None and round(self.pnl_usd, 2) < 0

    @property
    def is_win(self) -> bool:
        """Resultado positivo (a centavos)."""
        return self.pnl_usd is not None and round(self.pnl_usd, 2) > 0

    @property
    def has_context(self) -> bool:
        """Tiene algo del análisis de la app (id, régimen, score o estado del plan)."""
        return self.analysis_id is not None or any(
            v is not None
            for v in (self.regime_category, self.regime_name, self.score_total, self.plan_status)
        )

    @property
    def traded_on_no_trade(self) -> bool:
        """El plan decía "No operar" y se operó igual."""
        return self.plan_status == STATUS_NO_TRADE

    @property
    def against_plan(self) -> bool:
        """El plan decía una dirección y se operó la contraria."""
        return (self.plan_status == STATUS_LONG and self.side == "short") or (
            self.plan_status == STATUS_SHORT and self.side == "long"
        )

    @property
    def against_regime(self) -> bool:
        """Long en régimen bajista/crash o short en alcista/euforia."""
        if self.regime_category is None:
            return False
        return (self.side == "long" and self.regime_category in _AGAINST_LONG) or (
            self.side == "short" and self.regime_category in _AGAINST_SHORT
        )

    @property
    def missing_risk(self) -> bool:
        """No tiene el riesgo previsto anotado (o no es positivo)."""
        return self.risk_usd is None or self.risk_usd <= 0

    @property
    def loss_beyond_risk_usd(self) -> float | None:
        """Cuánto se perdió de más sobre el riesgo previsto; ``None`` si no pasó (o no se sabe)."""
        if not self.is_loss or self.missing_risk:
            return None
        assert self.pnl_usd is not None and self.risk_usd is not None
        extra = round(-self.pnl_usd - self.risk_usd, 2)
        return extra if extra > 0 else None

    def flags(self) -> tuple[str, ...]:
        """Avisos cortos de esta operación, para la columna "Avisos" de la tabla."""
        out: list[str] = []
        if self.traded_on_no_trade:
            out.append('el plan decía "No operar"')
        if self.against_plan:
            out.append("contra la dirección del plan")
        if self.against_regime:
            out.append("contra el régimen")
        if self.loss_beyond_risk_usd is not None:
            out.append("perdió más que el riesgo previsto")
        if self.missing_risk:
            out.append("sin riesgo anotado")
        if not self.has_context:
            out.append("sin análisis de la app")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (fechas en ISO, UTC)."""
        return json_safe(
            {
                "symbol": self.symbol,
                "side": self.side,
                "opened_at": _iso(self.opened_at),
                "closed_at": _iso(self.closed_at),
                "pnl_usd": self.pnl_usd,
                "risk_usd": self.risk_usd,
                "regime_category": self.regime_category,
                "regime_name": self.regime_name,
                "regime_confidence": self.regime_confidence,
                "score_total": self.score_total,
                "plan_status": self.plan_status,
                "notes": self.notes,
                "analysis_id": self.analysis_id,
                "flags": list(self.flags()),
            }
        )


@dataclass(frozen=True)
class IncidentWalletEvent:
    """Movimiento de la billetera: ``kind`` = ``"deposit"``, ``"withdrawal"`` o ``"balance"``."""

    ts: datetime | None
    kind: str | None
    amount_usd: float | None
    notes: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> IncidentWalletEvent:
        """Desde un ``StoredWalletEvent``, un dict o un objeto con esos atributos."""
        if isinstance(value, IncidentWalletEvent):
            return value
        kind = _text(_field(value, "kind"))
        return cls(
            ts=to_utc(_field(value, "ts")),
            kind=None if kind is None else kind.lower(),
            amount_usd=_finite(_field(value, "amount_usd")),
            notes=_text(_field(value, "notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "ts": _iso(self.ts),
                "kind": self.kind,
                "amount_usd": self.amount_usd,
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class IncidentPosition:
    """Posición abierta al momento del bloqueo."""

    symbol: str
    side: str | None
    opened_at: datetime | None
    risk_usd: float | None
    entry_price: float | None = None
    notes: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> IncidentPosition:
        """Desde un ``StoredPosition``, un dict o un objeto con esos atributos."""
        if isinstance(value, IncidentPosition):
            return value
        return cls(
            symbol=(_text(_field(value, "symbol")) or DASH).upper(),
            side=normalize_side(_field(value, "side")),
            opened_at=to_utc(_field(value, "opened_at")),
            risk_usd=_finite(_field(value, "risk_usd")),
            entry_price=_finite(_field(value, "entry_price")),
            notes=_text(_field(value, "notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "symbol": self.symbol,
                "side": self.side,
                "opened_at": _iso(self.opened_at),
                "risk_usd": self.risk_usd,
                "entry_price": self.entry_price,
                "notes": self.notes,
            }
        )


@dataclass(frozen=True)
class IncidentProfile:
    """Lo que el informe usa del perfil de riesgo (todo opcional).

    ``from_value`` acepta un ``risk.profiles.EffectiveProfile``, un dict o
    cualquier objeto con esos atributos.
    """

    name: str | None = None
    monthly_wallet_usd: float | None = None
    margin_per_trade_usd: float | None = None
    risk_per_trade_usd: float | None = None
    max_daily_loss_usd: float | None = None
    max_trades_per_day: int | None = None
    max_leverage: float | None = None

    @classmethod
    def from_value(cls, value: Any) -> IncidentProfile:
        if isinstance(value, IncidentProfile):
            return value
        if value is None:
            return cls()
        max_trades = _finite(_field(value, "max_trades_per_day"))
        return cls(
            name=_text(_field(value, "name")),
            monthly_wallet_usd=_finite(_field(value, "monthly_wallet_usd")),
            margin_per_trade_usd=_finite(_field(value, "margin_per_trade_usd")),
            risk_per_trade_usd=_finite(_field(value, "risk_per_trade_usd")),
            max_daily_loss_usd=_finite(_field(value, "max_daily_loss_usd")),
            max_trades_per_day=None if max_trades is None else int(max_trades),
            max_leverage=_finite(_field(value, "max_leverage")),
        )

    @property
    def is_empty(self) -> bool:
        return all(v is None for v in self.__dict__.values())

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(dict(self.__dict__))


@dataclass(frozen=True)
class Observation:
    """Algo que "falló", calculado con los datos.

    - ``code``: uno de los ``OBS_*``.
    - ``title``: título corto.
    - ``detail``: explicación en castellano simple, con los números.
    - ``items``: las operaciones, días o movimientos involucrados (un renglón
      cada uno, con el ``#`` de la tabla cuando es una operación).
    - ``checklist_item``: punto que se agrega a la lista de "antes de borrar
      LOCKED".
    """

    code: str
    title: str
    detail: str
    items: tuple[str, ...] = ()
    checklist_item: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "code": self.code,
                "title": self.title,
                "detail": self.detail,
                "items": list(self.items),
                "checklist_item": self.checklist_item,
            }
        )


@dataclass(frozen=True)
class TradeSummary:
    """Números del resumen: cantidades, resultado y peor/mejor operación."""

    count: int
    wins: int
    losses: int
    breakeven: int
    net_pnl_usd: float
    gross_profit_usd: float  # suma de las ganadas (>= 0)
    gross_loss_usd: float  # suma de las perdidas (<= 0)
    worst_trade: IncidentTrade | None
    best_trade: IncidentTrade | None
    longest_losing_streak: int

    @property
    def win_rate_pct(self) -> float | None:
        """Porcentaje de ganadas sobre las que tienen resultado (``None`` si no hay)."""
        decided = self.wins + self.losses + self.breakeven
        return None if decided == 0 else self.wins / decided * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON."""
        return json_safe(
            {
                "count": self.count,
                "wins": self.wins,
                "losses": self.losses,
                "breakeven": self.breakeven,
                "net_pnl_usd": self.net_pnl_usd,
                "gross_profit_usd": self.gross_profit_usd,
                "gross_loss_usd": self.gross_loss_usd,
                "win_rate_pct": self.win_rate_pct,
                "worst_trade": self.worst_trade,
                "best_trade": self.best_trade,
                "longest_losing_streak": self.longest_losing_streak,
            }
        )


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


# --- cálculos puros -------------------------------------------------------------

_FAR_FUTURE = datetime.max.replace(tzinfo=UTC)


def sorted_trades(trades: Iterable[Any]) -> list[IncidentTrade]:
    """Operaciones normalizadas, en orden de cierre (sin fecha, al final).

    Es el orden de la tabla del informe: el ``#`` de cada operación es su
    posición acá, empezando en 1.
    """
    items = [IncidentTrade.from_value(t) for t in trades]
    return sorted(
        items,
        key=lambda t: (t.closed_at or _FAR_FUTURE, t.opened_at or _FAR_FUTURE),
    )


def losing_streaks(trades: Iterable[Any], min_length: int) -> list[tuple[int, ...]]:
    """Rachas de ``min_length`` pérdidas seguidas o más, como tuplas de ``#`` (desde 1).

    El orden es el de ``sorted_trades``. Un resultado 0, positivo o
    desconocido corta la racha.
    """
    if int(min_length) < 1:
        raise ValueError("min_length tiene que ser 1 o más")
    streaks: list[tuple[int, ...]] = []
    current: list[int] = []
    for n, trade in enumerate(sorted_trades(trades), start=1):
        if trade.is_loss:
            current.append(n)
            continue
        if len(current) >= min_length:
            streaks.append(tuple(current))
        current = []
    if len(current) >= min_length:
        streaks.append(tuple(current))
    return streaks


def summarize_trades(trades: Iterable[Any]) -> TradeSummary:
    """Cuántas operaciones, ganadas, perdidas, resultado neto y racha más larga."""
    items = sorted_trades(trades)
    with_pnl = [t for t in items if t.pnl_usd is not None]
    wins = [t for t in with_pnl if t.is_win]
    losses = [t for t in with_pnl if t.is_loss]
    gross_profit = sum(t.pnl_usd for t in wins if t.pnl_usd is not None)
    gross_loss = sum(t.pnl_usd for t in losses if t.pnl_usd is not None)
    longest = max((len(s) for s in losing_streaks(items, 1)), default=0)
    return TradeSummary(
        count=len(items),
        wins=len(wins),
        losses=len(losses),
        breakeven=len(with_pnl) - len(wins) - len(losses),
        net_pnl_usd=round(sum(t.pnl_usd for t in with_pnl if t.pnl_usd is not None), 2),
        gross_profit_usd=round(gross_profit, 2),
        gross_loss_usd=round(gross_loss, 2),
        worst_trade=min(losses, key=lambda t: t.pnl_usd or 0.0, default=None),
        best_trade=max(wins, key=lambda t: t.pnl_usd or 0.0, default=None),
        longest_losing_streak=longest,
    )


def _trade_label(n: int, trade: IncidentTrade, tz: str | tzinfo | None) -> str:
    """``"#3 BTC long, 10/10/2026 14:05 (-45 USD)"``."""
    side = trade.side or "?"
    when = fmt_datetime(trade.closed_at, tz)
    return f"#{n} {trade.symbol} {side}, {when} ({fmt_pnl(trade.pnl_usd)})"


def _sum_pnl(trades: Iterable[IncidentTrade]) -> float:
    return round(sum(t.pnl_usd for t in trades if t.pnl_usd is not None), 2)


def _by_local_day(
    numbered: Sequence[tuple[int, IncidentTrade]], tz: str | tzinfo | None
) -> dict[date, list[tuple[int, IncidentTrade]]]:
    zone = resolve_timezone(tz)
    days: dict[date, list[tuple[int, IncidentTrade]]] = {}
    for n, trade in numbered:
        if trade.closed_at is None:
            continue
        days.setdefault(trade.closed_at.astimezone(zone).date(), []).append((n, trade))
    return days


def _fmt_day(day: date) -> str:
    return day.strftime("%d/%m/%Y")


def find_observations(
    trades: Iterable[Any],
    *,
    wallet_events: Iterable[Any] = (),
    profile: Any | None = None,
    losing_streak: int = DEFAULT_LOSING_STREAK,
    tz: str | tzinfo | None = None,
) -> tuple[Observation, ...]:
    """Qué falló, calculado con los datos (ver la lista en el docstring del módulo).

    ``losing_streak`` es la racha que se marca (la del breaker de
    enfriamiento). Las observaciones de días (demasiadas operaciones, pérdida
    del día) necesitan el perfil; sin perfil no se calculan. Los días son los
    de la zona ``tz``. Devuelve las observaciones en orden fijo (las más graves
    primero); vacío si no se encontró nada.
    """
    if int(losing_streak) < 1:
        raise ValueError("losing_streak tiene que ser 1 o más")
    items = sorted_trades(trades)
    numbered = list(enumerate(items, start=1))
    prof = IncidentProfile.from_value(profile)
    out: list[Observation] = []

    def lines(selected: Sequence[tuple[int, IncidentTrade]]) -> tuple[str, ...]:
        return tuple(_trade_label(n, t, tz) for n, t in selected)

    # 1) Operar cuando el plan decía "No operar".
    sel = [(n, t) for n, t in numbered if t.traded_on_no_trade]
    if sel:
        total = _sum_pnl(t for _, t in sel)
        out.append(
            Observation(
                code=OBS_TRADED_ON_NO_TRADE,
                title='Operaste cuando el plan decía "No operar"',
                detail=(
                    f"{_plural(len(sel), 'operación se hizo', 'operaciones se hicieron')} "
                    'aunque el análisis de la app decía "No operar". '
                    f"Resultado de esas operaciones: {fmt_pnl(total)}."
                ),
                items=lines(sel),
                checklist_item=(
                    'Entendí por qué operé cuando el plan decía "No operar" y qué voy a hacer '
                    "distinto la próxima vez."
                ),
            )
        )

    # 2) Dirección contraria a la del plan.
    sel = [(n, t) for n, t in numbered if t.against_plan]
    if sel:
        total = _sum_pnl(t for _, t in sel)
        out.append(
            Observation(
                code=OBS_AGAINST_PLAN,
                title="Operaste en la dirección contraria a la del plan",
                detail=(
                    f"En {_plural(len(sel), 'operación', 'operaciones')} el plan decía long y "
                    "se operó short, o al revés. "
                    f"Resultado de esas operaciones: {fmt_pnl(total)}."
                ),
                items=lines(sel),
                checklist_item="Revisé por qué operé en la dirección contraria a la del plan.",
            )
        )

    # 3) Contra el régimen.
    sel = [(n, t) for n, t in numbered if t.against_regime]
    if sel:
        total = _sum_pnl(t for _, t in sel)
        out.append(
            Observation(
                code=OBS_AGAINST_REGIME,
                title="Operaste contra el régimen del mercado",
                detail=(
                    f"{_plural(len(sel), 'operación fue', 'operaciones fueron')} contra el "
                    "régimen (long con el mercado bajista o en crash, short con el mercado "
                    "alcista o en euforia). Ir contra la corriente suele salir caro. "
                    f"Resultado de esas operaciones: {fmt_pnl(total)}."
                ),
                items=lines(sel),
                checklist_item=(
                    "Revisé las operaciones contra el régimen: ¿tenía una razón de peso para "
                    "ir contra la corriente?"
                ),
            )
        )

    # 4) Pérdidas más grandes que el riesgo previsto.
    sel = [(n, t) for n, t in numbered if t.loss_beyond_risk_usd is not None]
    if sel:
        extra = round(sum(t.loss_beyond_risk_usd or 0.0 for _, t in sel), 2)
        detail_lines = []
        for n, t in sel:
            assert t.pnl_usd is not None and t.risk_usd is not None
            times = fmt_num(-t.pnl_usd / t.risk_usd, 1)
            detail_lines.append(
                f"#{n} {t.symbol} {t.side or '?'}, {fmt_datetime(t.closed_at, tz)}: perdiste "
                f"{fmt_usd(-t.pnl_usd)} con un riesgo previsto de {fmt_usd(t.risk_usd)} "
                f"({times} veces lo previsto)"
            )
        out.append(
            Observation(
                code=OBS_LOSS_ABOVE_RISK,
                title="Perdiste más de lo que ibas a arriesgar",
                detail=(
                    f"En {_plural(len(sel), 'operación', 'operaciones')} la pérdida pasó el "
                    "riesgo previsto (que ya incluye las comisiones). Suele pasar cuando se "
                    "mueve o se saca el stop, o cuando el precio salta el stop de golpe. "
                    f"Pérdida de más, en total: {fmt_usd(extra)}."
                ),
                items=tuple(detail_lines),
                checklist_item=(
                    "Revisé por qué algunas pérdidas pasaron el riesgo previsto (¿moví o saqué "
                    "el stop?)."
                ),
            )
        )

    # 5) Racha de pérdidas.
    streaks = losing_streaks(items, int(losing_streak))
    if streaks:
        streak_lines = []
        for streak in streaks:
            first, last = items[streak[0] - 1], items[streak[-1] - 1]
            total = _sum_pnl(items[n - 1] for n in streak)
            streak_lines.append(
                f"{len(streak)} pérdidas seguidas (#{streak[0]} a #{streak[-1]}), entre el "
                f"{fmt_datetime(first.closed_at, tz)} y el {fmt_datetime(last.closed_at, tz)}: "
                f"{fmt_pnl(total)} en total"
            )
        out.append(
            Observation(
                code=OBS_LOSING_STREAK,
                title="Racha de pérdidas",
                detail=(
                    f"Hubo {_plural(len(streaks), 'racha', 'rachas')} de {int(losing_streak)} "
                    "pérdidas seguidas o más. Después de varias pérdidas es común querer "
                    "recuperar rápido y arriesgar de más."
                ),
                items=tuple(streak_lines),
                checklist_item=(
                    "Pensé qué hice después de varias pérdidas seguidas: ¿paré o seguí para "
                    "recuperar?"
                ),
            )
        )

    # 6) Días con más operaciones que el máximo del perfil.
    days = _by_local_day(numbered, tz)
    if prof.max_trades_per_day is not None and prof.max_trades_per_day > 0:
        busy = [(d, v) for d, v in sorted(days.items()) if len(v) > prof.max_trades_per_day]
        if busy:
            out.append(
                Observation(
                    code=OBS_TOO_MANY_TRADES,
                    title="Días con demasiadas operaciones",
                    detail=(
                        f"Hubo {_plural(len(busy), 'día', 'días')} con más operaciones que el "
                        f"máximo del perfil ({prof.max_trades_per_day} por día)."
                    ),
                    items=tuple(
                        f"{_fmt_day(d)}: {_plural(len(v), 'operación', 'operaciones')} "
                        f"({fmt_pnl(_sum_pnl(t for _, t in v))})"
                        for d, v in busy
                    ),
                    checklist_item="Revisé los días en que operé más de lo que permite el perfil.",
                )
            )

    # 7) Días en que la pérdida pasó el límite diario.
    if prof.max_daily_loss_usd is not None and prof.max_daily_loss_usd > 0:
        limit = prof.max_daily_loss_usd
        bad = [
            (d, _sum_pnl(t for _, t in v))
            for d, v in sorted(days.items())
            if -_sum_pnl(t for _, t in v) > round(limit, 2)
        ]
        if bad:
            out.append(
                Observation(
                    code=OBS_DAILY_LOSS_LIMIT,
                    title="Días en que pasaste el límite de pérdida diario",
                    detail=(
                        f"Hubo {_plural(len(bad), 'día', 'días')} en que la pérdida del día "
                        f"pasó el límite del perfil ({fmt_usd(limit)}). Al llegar al límite hay "
                        "que dejar de operar hasta el día siguiente."
                    ),
                    items=tuple(
                        f"{_fmt_day(d)}: perdiste {fmt_usd(-pnl)} en el día" for d, pnl in bad
                    ),
                    checklist_item=(
                        "Revisé los días en que seguí operando después del límite diario."
                    ),
                )
            )

    # 8) Depósitos el mismo día, después de una pérdida.
    zone = resolve_timezone(tz)
    deposit_lines = []
    for ev in sorted(
        (IncidentWalletEvent.from_value(e) for e in wallet_events),
        key=lambda e: e.ts or _FAR_FUTURE,
    ):
        if ev.kind != "deposit" or ev.ts is None:
            continue
        day = ev.ts.astimezone(zone).date()
        earlier_losses = [
            t
            for _, t in days.get(day, [])
            if t.is_loss and t.closed_at is not None and t.closed_at <= ev.ts
        ]
        if earlier_losses:
            deposit_lines.append(
                f"{fmt_datetime(ev.ts, tz)}: depósito de {fmt_usd(ev.amount_usd or 0.0)} "
                f"(antes, ese día, perdiste {fmt_usd(-_sum_pnl(earlier_losses))})"
            )
    if deposit_lines:
        out.append(
            Observation(
                code=OBS_DEPOSIT_AFTER_LOSS,
                title="Cargaste plata después de perder",
                detail=(
                    f"{_plural(len(deposit_lines), 'depósito se hizo', 'depósitos se hicieron')} "
                    "el mismo día, después de una pérdida. A veces es para recuperar rápido lo "
                    "perdido, y eso suele terminar en más pérdidas."
                ),
                items=tuple(deposit_lines),
                checklist_item=(
                    "Pensé si los depósitos después de perder fueron para recuperar rápido."
                ),
            )
        )

    # 9) Sin riesgo anotado.
    sel = [(n, t) for n, t in numbered if t.missing_risk]
    if sel:
        out.append(
            Observation(
                code=OBS_MISSING_RISK,
                title="Operaciones sin riesgo anotado",
                detail=(
                    f"En {_plural(len(sel), 'operación', 'operaciones')} no está anotado el "
                    "riesgo previsto: no se puede saber si se respetó el stop."
                ),
                items=lines(sel),
                checklist_item="Para las próximas, anoto el riesgo previsto de cada operación.",
            )
        )

    # 10) Sin análisis de la app.
    sel = [(n, t) for n, t in numbered if not t.has_context]
    if sel:
        out.append(
            Observation(
                code=OBS_NO_ANALYSIS_CONTEXT,
                title="Operaciones sin análisis de la app",
                detail=(
                    f"{_plural(len(sel), 'operación no tiene', 'operaciones no tienen')} un "
                    "análisis de la app asociado: no se sabe qué decían el régimen, el score "
                    "ni el plan en ese momento."
                ),
                items=lines(sel),
                checklist_item=(
                    "Para las próximas, opero solo con un análisis de la app hecho antes (así "
                    "queda registrado el contexto)."
                ),
            )
        )

    return tuple(out)


# --- informe --------------------------------------------------------------------


def _fmt_value(value: Any) -> str:
    """Valor suelto de ``settings_info``: números con coma decimal, el resto tal cual."""
    if isinstance(value, bool):
        return "sí" if value else "no"
    num = _finite(value)
    if num is not None and not isinstance(value, str):
        return fmt_num(num, 4)
    return _one_line(str(value)) if value is not None else DASH


def _summary_lines(
    trigger: IncidentTrigger,
    summary: TradeSummary,
    positions: Sequence[IncidentPosition],
    events: Sequence[IncidentWalletEvent],
    prof: IncidentProfile,
    numbered: Sequence[tuple[int, IncidentTrade]],
    tz: str | tzinfo | None,
) -> list[str]:
    out: list[str] = []
    peak, balance, dd = trigger.wallet_peak_usd, trigger.current_balance_usd, trigger.drawdown_pct
    if peak is not None and balance is not None:
        line = (
            f"- La billetera del mes llegó a un máximo de **{fmt_usd(peak)}** y ahora tiene "
            f"**{fmt_usd(balance)}**"
        )
        if dd is not None:
            if trigger.threshold_pct is not None:
                dd_txt = fmt_percent_beside(dd, trigger.threshold_pct)
                line += (
                    f": cayó **{dd_txt}** desde ese máximo (el tope es "
                    f"{fmt_percent(trigger.threshold_pct)})"
                )
            else:
                line += f": cayó **{fmt_percent(dd)}** desde ese máximo"
        out.append(line + ".")
    elif dd is not None:
        out.append(f"- La billetera del mes cayó **{fmt_percent(dd)}** desde su máximo.")

    if summary.count == 0:
        out.append("- No hay operaciones cargadas en el período.")
    else:
        rate = summary.win_rate_pct
        rate_txt = "" if rate is None else f" ({fmt_percent(rate, 0)} ganadas)"
        parts = [
            _plural(summary.wins, "ganada", "ganadas"),
            _plural(summary.losses, "perdida", "perdidas"),
        ]
        if summary.breakeven:
            parts.append(f"{summary.breakeven} sin ganancia ni pérdida")
        out.append(f"- Operaciones: **{summary.count}**: {', '.join(parts)}{rate_txt}.")
        out.append(
            f"- Resultado neto: **{fmt_pnl(summary.net_pnl_usd)}** (ganado "
            f"{fmt_pnl(summary.gross_profit_usd)}, perdido {fmt_pnl(summary.gross_loss_usd)})."
        )
        worst = summary.worst_trade
        if worst is not None:
            n = next((i for i, t in numbered if t is worst or t == worst), 0)
            out.append(f"- Peor operación: {_trade_label(n, worst, tz)}.")
        if summary.longest_losing_streak >= 2:
            out.append(
                f"- Racha de pérdidas más larga: **{summary.longest_losing_streak} seguidas**."
            )

    deposits = [e for e in events if e.kind == "deposit" and e.amount_usd is not None]
    withdrawals = [e for e in events if e.kind == "withdrawal" and e.amount_usd is not None]
    if deposits or withdrawals:
        parts = []
        if deposits:
            total = sum(e.amount_usd or 0.0 for e in deposits)
            parts.append(f"depositaste {fmt_usd(total)} ({_plural(len(deposits), 'vez', 'veces')})")
        if withdrawals:
            total = sum(e.amount_usd or 0.0 for e in withdrawals)
            parts.append(
                f"retiraste {fmt_usd(total)} ({_plural(len(withdrawals), 'vez', 'veces')})"
            )
        out.append(f"- En el período {' y '.join(parts)}.")

    if positions:
        known = [p.risk_usd for p in positions if p.risk_usd is not None]
        risk_txt = f", con {fmt_usd(sum(known))} en riesgo hasta los stops" if known else ""
        names = ", ".join(f"{p.symbol} {p.side or '?'}" for p in positions)
        out.append(
            f"- Posiciones que siguen abiertas: **{len(positions)}** ({names}){risk_txt}. "
            "La app no las cierra: revisalas en el exchange."
        )
    else:
        out.append("- No había posiciones abiertas cargadas al momento del bloqueo.")

    if not prof.is_empty:
        bits = []
        if prof.monthly_wallet_usd is not None:
            bits.append(f"billetera mensual {fmt_usd(prof.monthly_wallet_usd)}")
        if prof.risk_per_trade_usd is not None:
            bits.append(f"riesgo por operación {fmt_usd(prof.risk_per_trade_usd)}")
        if prof.max_daily_loss_usd is not None:
            bits.append(f"pérdida máxima del día {fmt_usd(prof.max_daily_loss_usd)}")
        if prof.max_trades_per_day is not None:
            per_day = _plural(prof.max_trades_per_day, "operación", "operaciones")
            bits.append(f"hasta {per_day} por día")
        name = prof.name or "sin nombre"
        tail = f" ({', '.join(bits)})" if bits else ""
        out.append(f"- Perfil de riesgo: **{name}**{tail}.")
    return out


def build_incident_report(
    *,
    triggered_at: datetime,
    trigger: Any,
    trades: Iterable[Any] = (),
    wallet_events: Iterable[Any] = (),
    open_positions: Iterable[Any] = (),
    profile: Any | None = None,
    settings_info: Mapping[str, Any] | None = None,
    losing_streak: int = DEFAULT_LOSING_STREAK,
    lock_path: Path | str | None = None,
    tz: str | tzinfo | None = None,
) -> str:
    """Arma el informe en markdown (puro: no escribe nada).

    - ``trigger``: ``IncidentTrigger``, dict u objeto con ``code``,
      ``description`` y los números de la billetera.
    - ``trades``: las operaciones del mes (``StoredTrade``, dicts...).
    - ``wallet_events`` / ``open_positions``: movimientos de la billetera y
      posiciones abiertas.
    - ``profile``: el perfil de riesgo (``EffectiveProfile`` o dict).
    - ``settings_info``: pares "qué" -> valor que se listan en "Configuración
      en ese momento" (por ejemplo los umbrales de los breakers).
    - ``losing_streak``: la racha que se marca (la del breaker).
    - ``lock_path``: dónde está el archivo LOCKED, para la lista final.
    - ``tz``: zona horaria del usuario para mostrar las horas.
    """
    when = _require_datetime(triggered_at, "triggered_at")
    trig = IncidentTrigger.from_value(trigger)
    items = sorted_trades(trades)
    numbered = list(enumerate(items, start=1))
    events = sorted(
        (IncidentWalletEvent.from_value(e) for e in wallet_events),
        key=lambda e: e.ts or _FAR_FUTURE,
    )
    positions = [IncidentPosition.from_value(p) for p in open_positions]
    prof = IncidentProfile.from_value(profile)
    summary = summarize_trades(items)
    observations = find_observations(
        items, wallet_events=events, profile=prof, losing_streak=losing_streak, tz=tz
    )
    tz_txt = timezone_label(tz)
    lock_txt = f"`{lock_path}`" if lock_path is not None else "`LOCKED` (en la carpeta de datos)"
    description = _one_line(trig.description)

    md: list[str] = []
    md.append("# Informe del incidente: la app se bloqueó")
    md.append("")
    md.append(f"- **Cuándo:** {fmt_long_datetime(when, tz)} ({tz_txt})")
    md.append(f"- **Por qué:** {description}")
    md.append(f"- **Mes de la billetera:** {fmt_month(when, tz)}")
    md.append("")

    md.append("## Resumen")
    md.append("")
    md.extend(_summary_lines(trig, summary, positions, events, prof, numbered, tz))
    md.append("")

    md.append("## Qué pasó, en simple")
    md.append("")
    reason_txt = description if description.endswith((".", "!", "?")) else f"{description}."
    md.append(f"La app se bloqueó sola. Motivo: {reason_txt}")
    md.append("")
    md.append(
        f"Mientras exista el archivo {lock_txt}, la app **no va a generar planes**. No hay "
        "ningún botón para desbloquearla: es a propósito. Te obliga a frenar, leer este "
        "informe y pensar qué pasó antes de volver a operar."
    )
    md.append("")

    md.append("## Qué falló")
    md.append("")
    if observations:
        md.append(
            "Esto lo calculó la app con las operaciones que cargaste (no es una opinión). "
            "Los números `#` son los de la tabla de operaciones."
        )
        md.append("")
        for obs in observations:
            md.append(f"### {obs.title}")
            md.append("")
            md.append(obs.detail)
            md.append("")
            for item in obs.items:
                md.append(f"- {_one_line(item)}")
            if obs.items:
                md.append("")
    else:
        md.append(
            "La app no encontró ningún patrón automático (operaciones con el plan en "
            '"No operar", contra el régimen, pérdidas más grandes que el riesgo, rachas, '
            "etc.). Igual revisá cada operación de la tabla: el mercado puede haber ido en "
            "contra varias veces seguidas, o el perfil puede ser demasiado agresivo para la "
            "billetera."
        )
        md.append("")

    md.append("## Operaciones del mes")
    md.append("")
    if numbered:
        md.append(
            "| # | Fecha | Moneda | Dirección | Resultado | Riesgo previsto | "
            "Régimen en ese momento | Score | Qué decía el plan | Avisos |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for n, t in numbered:
            risk_txt = DASH if t.missing_risk else fmt_usd(t.risk_usd)
            score_txt = DASH if t.score_total is None else fmt_score(t.score_total)
            flags = "; ".join(t.flags()) or DASH
            md.append(
                "| "
                + " | ".join(
                    [
                        str(n),
                        md_cell(fmt_datetime(t.closed_at, tz)),
                        md_cell(t.symbol),
                        md_cell(fmt_side(t.side)),
                        md_cell(fmt_pnl(t.pnl_usd)),
                        md_cell(risk_txt),
                        md_cell(fmt_regime(t.regime_category, t.regime_name, t.regime_confidence)),
                        md_cell(score_txt),
                        md_cell(fmt_plan_status(t.plan_status)),
                        md_cell(flags),
                    ]
                )
                + " |"
            )
        md.append("")
        md.append(
            "Cómo leer la tabla: **Long** = ganás si el precio sube; **Short** = ganás si baja. "
            "**Fecha** es la hora de cierre. **Riesgo previsto** es lo que ibas a perder si "
            "tocaba el stop. **Score** es el puntaje de la app en ese momento (positivo = a "
            "favor de subir, negativo = a favor de bajar). El porcentaje del régimen es qué "
            "tan seguro estaba el detector."
        )
        md.append("")
        with_notes = [(n, t) for n, t in numbered if t.notes]
        if with_notes:
            md.append("Notas que cargaste:")
            md.append("")
            for n, t in with_notes:
                md.append(f"- #{n}: {_one_line(t.notes or '')}")
            md.append("")
    else:
        md.append("No hay operaciones cargadas en el período.")
        md.append("")

    md.append("## Movimientos de la billetera")
    md.append("")
    if events:
        md.append("| Fecha | Tipo | Monto | Notas |")
        md.append("|---|---|---|---|")
        for ev in events:
            if ev.kind == "deposit":
                amount = fmt_pnl(abs(ev.amount_usd)) if ev.amount_usd is not None else DASH
            elif ev.kind == "withdrawal":
                amount = fmt_pnl(-abs(ev.amount_usd)) if ev.amount_usd is not None else DASH
            else:
                amount = DASH if ev.amount_usd is None else fmt_usd(ev.amount_usd)
            kind = _WALLET_KIND_LABELS.get(ev.kind or "", ev.kind or DASH)
            md.append(
                f"| {md_cell(fmt_datetime(ev.ts, tz))} | {md_cell(kind)} | {md_cell(amount)} "
                f"| {md_cell(ev.notes)} |"
            )
    else:
        md.append("No hay movimientos de la billetera cargados en el período.")
    md.append("")

    md.append("## Posiciones abiertas al momento del bloqueo")
    md.append("")
    if positions:
        md.append(
            "| Moneda | Dirección | Abierta el | Precio de entrada | Riesgo hasta el stop | Notas |"
        )
        md.append("|---|---|---|---|---|---|")
        for p in positions:
            entry = DASH if p.entry_price is None else fmt_price(p.entry_price)
            risk = DASH if p.risk_usd is None else fmt_usd(p.risk_usd)
            md.append(
                f"| {md_cell(p.symbol)} | {md_cell(fmt_side(p.side))} | "
                f"{md_cell(fmt_datetime(p.opened_at, tz))} | {md_cell(entry)} | "
                f"{md_cell(risk)} | {md_cell(p.notes)} |"
            )
        md.append("")
        md.append(
            "La app no opera ni cierra posiciones: si siguen abiertas, decidí qué hacer con "
            "ellas en el exchange."
        )
    else:
        md.append("No había posiciones abiertas cargadas.")
    md.append("")

    md.append("## Antes de borrar el archivo LOCKED")
    md.append("")
    md.append("Tildá cada punto recién cuando lo hayas hecho de verdad:")
    md.append("")
    checklist = [
        "Leí este informe completo, con calma.",
        "Miré cada operación perdedora de la tabla y entendí qué pasó en cada una.",
    ]
    checklist.extend(o.checklist_item for o in observations if o.checklist_item)
    if positions:
        checklist.append("Revisé en el exchange las posiciones que siguen abiertas.")
    checklist.extend(
        [
            "Anoté en la app el saldo actual de la billetera.",
            "Decidí si sigo con el mismo perfil, paso a uno más conservador o dejo de operar "
            "un tiempo.",
            "Dejé pasar un tiempo con la cabeza fría antes de volver a operar.",
            f"Recién ahora, borré a mano el archivo {lock_txt}.",
        ]
    )
    md.extend(f"- [ ] {item}" for item in checklist)
    md.append("")

    md.append("## Datos técnicos")
    md.append("")
    md.append(f"- Disparador: `{trig.code}`")
    md.append(f"- Hora del disparo (UTC): {when.isoformat()}")
    if trig.drawdown_pct is not None:
        md.append(f"- Caída desde el máximo: {fmt_num(trig.drawdown_pct, 4)} %")
    if trig.threshold_pct is not None:
        md.append(f"- Tope de caída: {fmt_num(trig.threshold_pct, 4)} %")
    if settings_info:
        md.append("")
        md.append("Configuración en ese momento:")
        md.append("")
        for key, value in settings_info.items():
            md.append(f"- {_one_line(str(key))}: {_fmt_value(value)}")
    md.append("")
    return "\n".join(md)


# --- escritura ------------------------------------------------------------------


def incident_file_name(triggered_at: datetime, attempt: int = 1) -> str:
    """Nombre del informe: ``incidente_2026-10-13_140502_UTC.md`` (hora UTC del disparo).

    ``attempt`` 2, 3... agrega ``_2``, ``_3``... (para no pisar uno anterior).
    """
    when = _require_datetime(triggered_at, "triggered_at")
    suffix = "" if attempt <= 1 else f"_{int(attempt)}"
    return f"{FILE_PREFIX}{when:%Y-%m-%d_%H%M%S}_UTC{suffix}{FILE_SUFFIX}"


def _create_exclusive_fallback(target: Path, text: str) -> bool:
    """Creación exclusiva sin enlaces duros (para sistemas de archivos que no los tienen).

    No es atómica (un corte a mitad de camino puede dejar el archivo a medias),
    pero nunca pisa uno existente.
    """
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return True


def create_new_file(path: Path | str, text: str) -> bool:
    """Crea ``path`` con ``text`` de forma atómica, **sin pisar nunca** un archivo existente.

    Escribe en un temporal de la misma carpeta y lo enlaza con ``os.link``, que
    falla si el destino ya existe: el archivo aparece completo o no aparece, y
    si dos procesos lo intentan a la vez solo uno gana. El temporal se borra
    siempre. Devuelve ``True`` si lo creó y ``False`` si ya existía. Crea la
    carpeta si hace falta. Un error de disco (``OSError``) se propaga.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        return False
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError:
            return False
        except (OSError, NotImplementedError, AttributeError):
            # Sistema de archivos sin enlaces duros: creación exclusiva directa.
            return _create_exclusive_fallback(target, text)
        return True
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def write_incident_report(
    incident_dir: Path | str,
    *,
    triggered_at: datetime,
    trigger: Any,
    trades: Iterable[Any] = (),
    wallet_events: Iterable[Any] = (),
    open_positions: Iterable[Any] = (),
    profile: Any | None = None,
    settings_info: Mapping[str, Any] | None = None,
    losing_streak: int = DEFAULT_LOSING_STREAK,
    lock_path: Path | str | None = None,
    tz: str | tzinfo | None = None,
) -> Path:
    """Escribe el informe en ``incident_dir`` y devuelve la ruta del archivo.

    Los argumentos son los de ``build_incident_report``. El nombre sale de
    ``incident_file_name``; si ya existe un informe con ese nombre se usa el
    siguiente libre (``_2``, ``_3``...): nunca se pisa uno anterior. La
    escritura es atómica (``create_new_file``). ``FileExistsError`` solo si se
    agotan ``MAX_NAME_ATTEMPTS`` nombres; un error de disco se propaga.
    """
    text = build_incident_report(
        triggered_at=triggered_at,
        trigger=trigger,
        trades=trades,
        wallet_events=wallet_events,
        open_positions=open_positions,
        profile=profile,
        settings_info=settings_info,
        losing_streak=losing_streak,
        lock_path=lock_path,
        tz=tz,
    )
    folder = Path(incident_dir)
    for attempt in range(1, MAX_NAME_ATTEMPTS + 1):
        target = folder / incident_file_name(triggered_at, attempt)
        if create_new_file(target, text):
            return target
    raise FileExistsError(
        f"No se pudo elegir un nombre libre para el informe en {folder} "
        f"({MAX_NAME_ATTEMPTS} intentos)"
    )


__all__ = [
    "DEFAULT_LOSING_STREAK",
    "FILE_PREFIX",
    "FILE_SUFFIX",
    "MAX_NAME_ATTEMPTS",
    "OBS_AGAINST_PLAN",
    "OBS_AGAINST_REGIME",
    "OBS_DAILY_LOSS_LIMIT",
    "OBS_DEPOSIT_AFTER_LOSS",
    "OBS_LOSING_STREAK",
    "OBS_LOSS_ABOVE_RISK",
    "OBS_MISSING_RISK",
    "OBS_NO_ANALYSIS_CONTEXT",
    "OBS_TOO_MANY_TRADES",
    "OBS_TRADED_ON_NO_TRADE",
    "IncidentPosition",
    "IncidentProfile",
    "IncidentTrade",
    "IncidentTrigger",
    "IncidentWalletEvent",
    "Observation",
    "TradeSummary",
    "build_incident_report",
    "create_new_file",
    "find_observations",
    "fmt_date",
    "fmt_datetime",
    "fmt_long_datetime",
    "fmt_month",
    "fmt_plan_status",
    "fmt_pnl",
    "fmt_regime",
    "fmt_side",
    "incident_file_name",
    "losing_streaks",
    "md_cell",
    "normalize_plan_status",
    "normalize_side",
    "resolve_timezone",
    "sorted_trades",
    "summarize_trades",
    "timezone_label",
    "to_utc",
    "write_incident_report",
]
