"""Tablas de SQLite (SQLModel).

Estado de trading cargado a mano (lo usan los circuit breakers y la
correlación, Hito 5) y el historial de análisis:

- ``ManualTrade``: una operación cerrada con su resultado neto en USD
  (comisiones incluidas; negativo = pérdida) y, si salió de un análisis de la
  app, el contexto de ese momento (régimen, score, estado del plan, perfil).
- ``OpenPosition``: una posición abierta y cuántos USD se arriesgan hasta el
  stop (para sumar riesgo entre monedas correlacionadas).
- ``WalletEvent``: movimientos de la billetera de futuros: depósito, retiro o
  "saldo" (el usuario anota el saldo que ve en ese momento).
- ``HighVolatilityEvent``: evento de alta volatilidad marcado a mano (Fed, CPI,
  etc.) con su nota y hora de fin; mientras dura, la capa de seguridad bloquea
  todos los planes (esquema 2).
- ``AnalysisRecord``: cada análisis generado, con el plan, el score, el
  veredicto de seguridad, la lectura del LLM y los datos de entrada en JSON.

Fechas: siempre en UTC. ``UTCDateTime`` las guarda en SQLite como UTC sin zona
(SQLite no tiene tipo con zona horaria) y las devuelve con ``tzinfo=UTC``. Una
fecha sin zona horaria se rechaza al guardar: no hay forma de saber si era hora
de Buenos Aires o UTC, y 3 h de diferencia cambian a qué día cuenta una
pérdida.

El resto de la app no usa estas clases directamente: pasa por
``storage.repository``, que valida los datos y devuelve dataclasses
inmutables.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Index, Text
from sqlalchemy.engine import Dialect
from sqlalchemy.types import TypeDecorator
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    """Ahora, en UTC y con zona horaria."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """``DateTime`` que guarda en UTC y devuelve fechas con ``tzinfo=UTC``.

    - Al guardar: una fecha con zona se pasa a UTC; una sin zona lanza
      ``ValueError`` (ver docstring del módulo).
    - Al leer: SQLite devuelve la fecha sin zona; se le pone UTC.
    """

    impl = DateTime(timezone=False)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"Se esperaba una fecha (datetime), llegó {type(value).__name__}")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("La fecha no tiene zona horaria; usá una fecha en UTC o con zona")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


# --- estado cargado a mano -------------------------------------------------------


class ManualTrade(SQLModel, table=True):
    """Operación cerrada, cargada a mano.

    ``pnl_usd`` es el resultado neto realizado (comisiones y funding
    incluidos): negativo = pérdida. Los campos de contexto (``analysis_id``,
    ``regime_*``, ``score_total``, ``plan_status``, ``profile_name``) se copian
    del análisis que el usuario siguió, si siguió alguno; sirven para el
    informe del incidente ("en qué régimen estaba, qué decía el score").
    """

    __tablename__ = "manual_trades"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=20)
    side: str = Field(max_length=5)  # "long" | "short"
    opened_at: datetime = Field(sa_type=UTCDateTime)
    closed_at: datetime = Field(sa_type=UTCDateTime, index=True)
    pnl_usd: float
    risk_usd: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    notes: str | None = Field(default=None, sa_type=Text)
    analysis_id: int | None = Field(default=None, index=True)
    regime_category: str | None = Field(default=None, max_length=20)
    regime_name: str | None = Field(default=None, max_length=60)
    regime_confidence: float | None = None
    score_total: float | None = None
    plan_status: str | None = Field(default=None, max_length=20)
    profile_name: str | None = Field(default=None, max_length=60)
    created_at: datetime = Field(default_factory=utc_now, sa_type=UTCDateTime)


class OpenPosition(SQLModel, table=True):
    """Posición abierta, cargada a mano.

    ``risk_usd`` es lo que el usuario pierde si toca el stop (obligatorio:
    con eso se suma el riesgo de monedas correlacionadas).
    """

    __tablename__ = "open_positions"

    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=20)
    side: str = Field(max_length=5)  # "long" | "short"
    opened_at: datetime = Field(sa_type=UTCDateTime)
    entry_price: float | None = None
    risk_usd: float
    notes: str | None = Field(default=None, sa_type=Text)
    analysis_id: int | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=utc_now, sa_type=UTCDateTime)


class WalletEvent(SQLModel, table=True):
    """Movimiento de la billetera de futuros.

    ``kind``:

    - ``"deposit"``: entró plata (``amount_usd > 0``).
    - ``"withdrawal"``: salió plata (``amount_usd > 0``, el signo lo da el tipo).
    - ``"balance"``: el usuario anota el saldo actual (``amount_usd >= 0``).
    """

    __tablename__ = "wallet_events"

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(sa_type=UTCDateTime, index=True)
    kind: str = Field(max_length=12)
    amount_usd: float
    notes: str | None = Field(default=None, sa_type=Text)


class HighVolatilityEvent(SQLModel, table=True):
    """Evento de alta volatilidad marcado a mano (esquema 2).

    Bloquea todos los planes desde ``starts_at`` hasta ``until`` (intervalo
    semiabierto: a la hora de fin ya no bloquea) y después se apaga solo, sin
    que haya que borrarlo. ``note`` dice qué evento es ("Dato de inflación de
    EE.UU."); es obligatoria para que el aviso explique por qué no hay plan.
    """

    __tablename__ = "high_volatility_events"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now, sa_type=UTCDateTime)
    starts_at: datetime = Field(sa_type=UTCDateTime, index=True)
    until: datetime = Field(sa_type=UTCDateTime, index=True)
    note: str = Field(sa_type=Text)


# --- historial de análisis -------------------------------------------------------


class AnalysisRecord(SQLModel, table=True):
    """Un análisis generado.

    Las columnas sueltas (precio, régimen, score, estado del plan) permiten
    filtrar y listar sin abrir el JSON. Los ``*_json`` guardan dicts completos
    como texto JSON (todos opcionales):

    - ``plan_json``: ``TradePlan.to_dict()``.
    - ``score_json``: ``ConfluenceScore.to_dict()``.
    - ``safety_json``: el veredicto de la capa de seguridad (circuit breakers,
      lock y correlación).
    - ``llm_json``: la lectura del LLM, si hubo.
    - ``inputs_json``: un resumen de los datos de entrada, para poder revisar
      después con qué números se armó el análisis.
    """

    __tablename__ = "analyses"
    __table_args__ = (Index("ix_analyses_symbol_created_at", "symbol", "created_at"),)

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now, sa_type=UTCDateTime, index=True)
    symbol: str = Field(max_length=20)
    profile_name: str = Field(max_length=60)
    price: float
    price_source: str = Field(max_length=40)
    regime_category: str | None = Field(default=None, max_length=20)
    regime_name: str | None = Field(default=None, max_length=60)
    regime_confidence: float | None = None
    regime_unstable: bool | None = None
    score_total: float | None = None
    plan_status: str = Field(max_length=20)
    plan_json: str | None = Field(default=None, sa_type=Text)
    score_json: str | None = Field(default=None, sa_type=Text)
    safety_json: str | None = Field(default=None, sa_type=Text)
    llm_json: str | None = Field(default=None, sa_type=Text)
    inputs_json: str | None = Field(default=None, sa_type=Text)


__all__ = [
    "AnalysisRecord",
    "HighVolatilityEvent",
    "ManualTrade",
    "OpenPosition",
    "UTCDateTime",
    "WalletEvent",
    "utc_now",
]
