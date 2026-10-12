"""Generador del plan de operación (Hito 4).

Riesgo en dólares primero, apalancamiento después: el apalancamiento nunca se
elige, sale del cálculo. Todo es puro (sin red, sin disco, sin reloj): el
momento actual llega por parámetro (``now``), así la misma función sirve para
la app, la consola y el backtest walk-forward del Hito 10.

API
---

- ``PlanInputs``: datos sueltos que necesita el plan (precio de ejecución,
  soportes y resistencias, ATR, tendencias, rango diario, score, funding,
  régimen, perfil, estado del día). Se puede armar a mano (tests, backtest).
- ``plan_inputs_from_analysis(analysis, snapshot, score, regime, profile,
  settings, ...)``: arma ``PlanInputs`` desde ``indicators.pipeline`` y
  ``data.snapshot``.
- ``build_trade_plan(inputs, settings, *, now) -> TradePlan``: el plan.

Pasos de ``build_trade_plan`` (todos los números salen de ``config.yaml``)
------------------------------------------------------------------------

1. **Dirección** por el score (``scoring.direction_thresholds``): long si es
   >= +30, short si es <= -30; si no, "No operar" y no hay niveles.
2. **Ajustes por régimen** (``regime_adjustments.apply_regime_adjustments``):
   bloqueos (long en Crash), score mínimo, R:R mínimo (el mayor entre el del
   perfil y el del régimen) y multiplicadores de riesgo, cada uno con su motivo.
3. **Stop por estructura** (``stops.structural_stop``): el soporte (long) o la
   resistencia (short) más cercano de ``trade_plan.stop_timeframes`` ∓
   ``stop_atr_buffer`` × ATR del timeframe de entrada, salteando los que están
   a menos de ``min_stop_distance_atr`` ATR. ``R`` = distancia entrada-stop.
4. **Objetivos** (decisiones del usuario, ``trade_plan.take_profit``):

   - TP1: el siguiente soporte/resistencia a favor (mismos timeframes que el
     stop), salteando los que están a menos de ``tp1_min_distance_atr`` ATR.
     Si no hay ninguno, no hay TP1 (se dice por qué). Si queda más lejos que el
     TP2 se marca "más lejos que el TP2".
   - TP2: el más lejano entre ``tp2_r`` × R y, con ``tp2_ensure_min_rr``, el
     precio justo para que el R:R **neto de comisiones** (``fees.net_rr``) llegue
     al mínimo efectivo. Con comisiones ``f_in``/``f_out`` y
     ``L = R + f_in·E + f_out·S`` (pérdida por unidad en el stop):
     long ``T = (k·L + E·(1 + f_in)) / (1 - f_out)``;
     short ``T = (E·(1 - f_in) - k·L) / (1 + f_out)``.
   - TP3: el más lejano entre ``tp3_r`` × R y TP2 + (``tp3_r`` - ``tp2_r``) × R.
   - Cada TP más lejos que el rango típico del día (ATR diario) lleva la marca
     "poco probable intradía" (``flag_tp_beyond_daily_range``); si es el TP2 y
     ``block_if_tp2_beyond_daily_range``, no se opera.
   - En un short con el stop absurdamente lejos el TP2 daría un precio de cero
     o menos: no hay TP2 (``tp2_missing_reason`` dice por qué) y no se opera.

5. **Posición** (``position.fit_position``): tamaño por riesgo (comisiones
   dentro del riesgo si ``fees_included_in_risk``), margen del perfil, y si el
   apalancamiento pasa el tope o la liquidación queda cerca: primero sube el
   margen, después achica la posición, y "No operar" si el riesgo queda debajo
   del mínimo operable. Si la posición es más chica que el margen, el margen
   baja a 1x.
6. **Comisiones** sobre el tamaño (nunca sobre el margen), de ida y vuelta: al
   stop (como % del riesgo) y al TP2 (como % de la ganancia bruta del TP2).
7. **Liquidación** con margen aislado (la que usan las reglas) y, como dato, la
   del "modelo billetera" de Nexo.
8. **Funding** estimado hasta el stop por tiempo (si
   ``fees.include_funding_estimate``): positivo = pagás, negativo = cobrás.
9. **Stop por tiempo**: ``now + trade_plan.time_stop_hours``; si para entonces
   no tocó el TP1 ni el stop, se cierra (si no hay TP1, o queda más lejos que el
   TP2, se mira el TP2; si no hay ninguno de los dos, no hay stop por tiempo).
10. **Reglas**: se evalúan TODAS y se guardan con ``ok`` (no se corta en la
    primera): dirección, régimen, score mínimo, 4h contra 1h, límites del día,
    evento de alta volatilidad, stop, R:R a TP2 con comisiones, TP2 dentro del
    rango del día, apalancamiento, liquidación y riesgo mínimo operable (este
    último aparece aunque no haya stop si el régimen ya dejó el riesgo debajo
    del mínimo). Si alguna falla, el estado es ``NO_OPERAR`` y, si los niveles
    se pudieron calcular, se devuelven igual marcados como referencia
    (``is_reference=True``: "referencia, no operar") para que se vea por qué.

Avisos que no bloquean (``warnings``):

- Precio de entrada sospechoso: el de Nexo cargado a mano difiere del de
  referencia más que ``exchanges.nexo.manual_price_max_diff_pct`` (posible
  error de tipeo). El aviso va en el plan y el titular no dice que la entrada
  alternativa "sí cumple las reglas".
- Límite del día: si con la pérdida de hoy más la de esta operación (si toca el
  stop) se pasaría de ``max_daily_loss_usd``. Hoy solo avisa; la regla bloquea
  recién cuando la pérdida del día ya llegó al límite.

Los soportes y resistencias con precio no finito o no positivo (armados a mano)
se ignoran.

Entrada alternativa: el soporte (long) o la resistencia (short) más cercano del
timeframe de entrada (``timeframes.entry``, 15m) del lado a favor, como orden
límite. Se calcula con la misma función (su stop, objetivos y tamaño propios);
el plan principal es siempre la entrada a mercado.

Unidades: en el plan los ``*_pct`` son **porcentaje humano** (``1.5`` = 1,5 %),
como en ``config.yaml`` y la API. Ojo: ``sizing``, ``liquidation`` y
``position`` usan fracción (``0.015``). Montos en USD.

``trades_today`` y ``pnl_today_usd`` quedan en 0 hasta que el Hito 5 lleve la
cuenta del día (``TradePlan.day_state`` los devuelve como datos). Los circuit
breakers de ``safety`` (Hito 5) se van a sumar como un multiplicador de riesgo y
reglas más.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from app.indicators.levels import SRResult
from app.risk.fees import (
    fee_rates_from_config,
    fees_pct_of,
    funding_cost_usd,
    gross_pnl_usd,
    net_rr,
    trade_fees_usd,
)
from app.risk.position import LIMIT_LEVERAGE, PositionFit, fit_position
from app.risk.regime_adjustments import (
    CATEGORY_LABELS,
    REGIME_BLOCKED,
    RegimeAdjustmentResult,
    apply_regime_adjustments,
)
from app.risk.rules import (
    MIN_OPERABLE_RISK,
    MIN_RR,
    NO_TRADE,
    REL_TOL,
    RuleResult,
    check_daily_limits,
    check_direction,
    check_high_volatility_event,
    check_leverage,
    check_liquidation,
    check_min_operable_risk,
    check_rr,
    check_timeframe_conflict,
    direction_from_score,
    fmt_num,
    fmt_percent,
    fmt_price,
    fmt_ratio,
    fmt_usd,
    json_safe,
)
from app.risk.sizing import Side, ensure_finite, ensure_positive
from app.risk.stops import SkippedLevel, StopNotFound, StructuralStop, structural_stop
from app.scoring.components import FUNDING_REFERENCE_HOURS, clean_number, funding_pct_per_8h
from app.scoring.confluence import reliable_trend

if TYPE_CHECKING:
    from app.config import AppSettings
    from app.data.snapshot import MarketSnapshot
    from app.indicators.pipeline import MarketAnalysis
    from app.risk.profiles import EffectiveProfile
    from app.scoring.confluence import ConfluenceScore

# Estado del plan.
STATUS_LONG = "LONG"
STATUS_SHORT = "SHORT"
STATUS_NO_TRADE = "NO_OPERAR"
PlanStatus = Literal["LONG", "SHORT", "NO_OPERAR"]

# Tipo de entrada.
ENTRY_MARKET = "market"
ENTRY_LIMIT = "limit"

# Códigos de las reglas propias del plan (las demás están en ``rules``).
REGIME_RULE = "regime"
STOP_RULE = "stop"
TP2_DAILY_RANGE = "tp2_daily_range"

# De dónde sale cada objetivo.
TP_NEXT_LEVEL = "next_level"
TP_R_MULTIPLE = "r_multiple"
TP_MIN_NET_RR = "min_net_rr"
TP_AFTER_TP2 = "after_tp2"

# Marcas de los objetivos (texto de la UI).
FLAG_UNLIKELY_INTRADAY = "poco probable intradía"
FLAG_TP1_BEYOND_TP2 = "más lejos que el TP2"

# Origen de los ajustes.
SOURCE_REGIME = "regime"
SOURCE_POSITION = "position"

# Tendencias que se comparan para el conflicto de timeframes (la regla del
# usuario es literal: 4h contra 1h).
TREND_TIMEFRAMES = ("4h", "1h")

# Precio de ejecución que es de Nexo (los demás se avisan).
NEXO_SOURCES = ("manual_nexo", "nexo")
PRICE_SOURCE_LABELS: dict[str, str] = {
    "manual_nexo": "Nexo, cargado a mano",
    "nexo": "Nexo",
    "binance": "Binance",
    "ultimo_cierre": "el último cierre de las velas",
}

_SIDE_UPPER = {"long": "LONG", "short": "SHORT"}

# Pasos de coma flotante (de a un "ulp") para que el TP2 del R:R mínimo nunca
# quede un pelo corto. No es una regla de riesgo.
_NUDGE_STEPS = 64


# --- entradas ------------------------------------------------------------------


@dataclass(frozen=True)
class PlanInputs:
    """Datos que necesita ``build_trade_plan``; todo plano para armarlo a mano.

    - ``entry_price``: precio de ejecución (el de Nexo cargado a mano si lo hay,
      si no el de referencia). ``price_source``: de dónde salió
      (``"manual_nexo"``, ``"nexo"``, ``"binance"``, ``"ultimo_cierre"``).
    - ``levels``: timeframe → ``SRResult`` (o ``None`` si no hay velas) de
      ``trade_plan.stop_timeframes`` y del timeframe de entrada. Se usan los
      ``clusters`` (todos los niveles), relativos a ``entry_price``.
    - ``entry_atr``: ATR del timeframe de entrada (``timeframes.entry``).
    - ``trend_4h`` / ``trend_1h``: ``"alcista"``, ``"bajista"``, ``"lateral"`` o
      ``None`` (sin dato confiable).
    - ``daily_range_usd`` / ``daily_range_pct``: rango típico del día (ATR
      diario) en USD y en % del precio; si falta uno se calcula con el otro.
    - ``score``: el ``ConfluenceScore`` (se usa ``total``).
    - ``funding_pct_8h``: funding por 8 h en % humano (``0.01`` = 0,01 %);
      ``funding_interval_hours``: cada cuánto cobra el exchange (por omisión
      ``fees.funding_interval_hours``).
    - ``regime``: el ``RegimeReport`` de 1h (o algo con ``category``,
      ``confidence``, ``unstable``) o ``None`` si no está disponible.
    - ``profile``: el ``EffectiveProfile`` (con el tope de apalancamiento
      efectivo).
    - ``trades_today`` / ``pnl_today_usd``: estado del día (0 hasta el Hito 5).
    - ``high_volatility_event`` / ``event_description``: evento marcado a mano.
    - ``available_margin_usd``: tope de margen por el saldo cargado a mano.
    - ``wallet_equity_usd``: saldo de la billetera de futuros para la
      liquidación "modelo Nexo" (por omisión la billetera mensual del perfil).
    - ``price_diff_pct``: cuánto difiere el precio de entrada (el de Nexo) del
      de referencia, en % humano con signo (``None`` si no se comparó). Si pasa
      de ``exchanges.nexo.manual_price_max_diff_pct`` el plan lo avisa.
    """

    symbol: str
    entry_price: float
    price_source: str
    levels: Mapping[str, SRResult | None]
    entry_atr: float | None
    trend_4h: str | None
    trend_1h: str | None
    daily_range_usd: float | None
    daily_range_pct: float | None
    score: ConfluenceScore
    funding_pct_8h: float | None
    regime: Any | None
    profile: EffectiveProfile
    trades_today: int = 0
    pnl_today_usd: float = 0.0
    high_volatility_event: bool = False
    event_description: str | None = None
    available_margin_usd: float | None = None
    funding_interval_hours: float | None = None
    wallet_equity_usd: float | None = None
    price_diff_pct: float | None = None


def plan_inputs_from_analysis(
    analysis: MarketAnalysis,
    snapshot: MarketSnapshot,
    score: ConfluenceScore,
    regime: Any | None,
    profile: EffectiveProfile,
    settings: AppSettings,
    *,
    trades_today: int = 0,
    pnl_today_usd: float = 0.0,
    high_volatility_event: bool = False,
    event_description: str | None = None,
    available_margin_usd: float | None = None,
) -> PlanInputs:
    """``PlanInputs`` desde el análisis de indicadores y la foto del mercado.

    - Precio: ``snapshot.current_price`` (ya es el de ejecución: Nexo cargado a
      mano, o Binance, o el último cierre) y ``snapshot.price_source``.
    - Niveles: los ``SRResult`` de ``trade_plan.stop_timeframes`` y del
      timeframe de entrada (los calcula el pipeline con el mismo precio).
    - ATR: el del timeframe de entrada (última vela cerrada).
    - Tendencias de 4h y 1h con el criterio del score
      (``scoring.confluence.reliable_trend``: ``None`` si no hay EMAs).
    - Rango diario: ``analysis.daily_range_pct`` y, en USD, ese % del precio.
    - Funding: ``snapshot.funding`` pasado a % por 8 h con el intervalo que
      informa el exchange (o ``fees.funding_interval_hours``).
    - ``regime``: el ``RegimeReport`` de 1h o ``None``.
    - ``price_diff_pct``: la diferencia del precio de Nexo contra el de
      referencia (``snapshot.execution_price.diff_pct``).
    """
    entry = float(snapshot.current_price)
    timeframes = dict.fromkeys([*settings.trade_plan.stop_timeframes, settings.timeframes.entry])
    levels: dict[str, SRResult | None] = {}
    for tf in timeframes:
        ta = analysis.timeframes.get(tf)
        levels[tf] = ta.summary.levels if ta is not None else None

    entry_ta = analysis.timeframes.get(settings.timeframes.entry)
    atr = clean_number(entry_ta.summary.atr) if entry_ta is not None else None

    trends: dict[str, str | None] = {}
    for tf in TREND_TIMEFRAMES:
        ta = analysis.timeframes.get(tf)
        trends[tf] = reliable_trend(ta.summary if ta is not None else None)

    daily_pct = clean_number(analysis.daily_range_pct)
    daily_usd = None if daily_pct is None else daily_pct / 100.0 * entry

    funding_pct: float | None = None
    interval: float | None = None
    if snapshot.funding is not None:
        rate = clean_number(snapshot.funding.rate)
        interval = clean_number(snapshot.funding.interval_hours)
        if interval is None or interval <= 0:
            interval = float(settings.fees.funding_interval_hours)
        if rate is not None:
            funding_pct = funding_pct_per_8h(rate, interval)

    execution = snapshot.execution_price
    diff = None if execution is None else clean_number(execution.diff_pct)

    return PlanInputs(
        symbol=snapshot.symbol,
        entry_price=entry,
        price_source=snapshot.price_source,
        levels=levels,
        entry_atr=atr,
        trend_4h=trends["4h"],
        trend_1h=trends["1h"],
        daily_range_usd=daily_usd,
        daily_range_pct=daily_pct,
        score=score,
        funding_pct_8h=funding_pct,
        regime=regime,
        profile=profile,
        trades_today=trades_today,
        pnl_today_usd=pnl_today_usd,
        high_volatility_event=high_volatility_event,
        event_description=event_description,
        available_margin_usd=available_margin_usd,
        funding_interval_hours=interval,
        price_diff_pct=diff,
    )


# --- resultado -----------------------------------------------------------------


def _pct(fraction: float | None) -> float | None:
    """Fracción → porcentaje humano (``0.015`` → ``1.5``)."""
    return None if fraction is None else fraction * 100.0


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _skipped_dict(levels: tuple[SkippedLevel, ...]) -> list[dict[str, Any]]:
    return [
        {"price": s.price, "timeframe": s.timeframe, "distance_atr": s.distance_atr} for s in levels
    ]


@dataclass(frozen=True)
class StopPlan:
    """Stop por estructura, listo para mostrar."""

    price: float
    distance_pct: float  # % humano de la entrada
    level_price: float
    level_timeframe: str
    level_kind: str  # "soporte" o "resistencia"
    atr: float
    atr_timeframe: str
    buffer_atr: float
    min_distance_atr: float
    skipped: tuple[SkippedLevel, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "price": self.price,
                "distance_pct": self.distance_pct,
                "level_price": self.level_price,
                "level_timeframe": self.level_timeframe,
                "level_kind": self.level_kind,
                "atr": self.atr,
                "atr_timeframe": self.atr_timeframe,
                "buffer_atr": self.buffer_atr,
                "min_distance_atr": self.min_distance_atr,
                "skipped_levels": _skipped_dict(self.skipped),
                "explanation": self.explanation,
            }
        )


@dataclass(frozen=True)
class TakeProfit:
    """Un objetivo. Las ganancias son con la cantidad final de la posición."""

    name: str  # "TP1", "TP2", "TP3"
    price: float
    distance_pct: float  # % humano de la entrada
    r_multiple: float  # distancia / R
    gross_gain_usd: float
    fees_usd: float  # comisiones de ida y vuelta saliendo en este objetivo
    net_gain_usd: float
    net_rr: float  # R:R con comisiones (fees.net_rr)
    basis: str  # TP_NEXT_LEVEL, TP_R_MULTIPLE, TP_MIN_NET_RR o TP_AFTER_TP2
    explanation: str
    beyond_daily_range: bool
    flags: tuple[str, ...]
    level_timeframe: str | None = None
    skipped: tuple[SkippedLevel, ...] = ()

    @property
    def unlikely_intraday(self) -> bool:
        return FLAG_UNLIKELY_INTRADAY in self.flags

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "price": self.price,
                "distance_pct": self.distance_pct,
                "r_multiple": self.r_multiple,
                "gross_gain_usd": self.gross_gain_usd,
                "fees_usd": self.fees_usd,
                "net_gain_usd": self.net_gain_usd,
                "net_rr": self.net_rr,
                "basis": self.basis,
                "explanation": self.explanation,
                "beyond_daily_range": self.beyond_daily_range,
                "unlikely_intraday": self.unlikely_intraday,
                "flags": list(self.flags),
                "level_timeframe": self.level_timeframe,
                "skipped_levels": _skipped_dict(self.skipped),
            }
        )


@dataclass(frozen=True)
class FeesPlan:
    """Comisiones sobre el tamaño, de ida y vuelta, al stop y al objetivo (TP2)."""

    entry_rate_pct: float
    exit_rate_pct: float
    entry_fee_usd: float
    exit_fee_at_stop_usd: float
    round_trip_at_stop_usd: float
    pct_of_risk: float  # comisiones al stop / pérdida máxima × 100
    included_in_risk: bool
    target_name: str | None  # "TP2"
    exit_fee_at_target_usd: float | None
    round_trip_at_target_usd: float | None
    pct_of_target: float | None  # comisiones al TP2 / ganancia bruta del TP2 × 100

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dataclasses.asdict(self))


@dataclass(frozen=True)
class FundingEstimate:
    """Funding estimado si la posición queda abierta hasta el stop por tiempo."""

    available: bool
    hours: float
    rate_pct_8h: float | None
    interval_hours: float | None
    cost_usd: float | None  # positivo = pagás, negativo = cobrás
    text: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dataclasses.asdict(self))


@dataclass(frozen=True)
class TimeStop:
    """Stop por tiempo: cerrar si a esa hora no tocó el objetivo ni el stop."""

    hours: float
    deadline: datetime
    target_name: str  # "TP1" (o "TP2" si no hay TP1; sin ninguno no hay stop por tiempo)
    text: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "hours": self.hours,
                "deadline": _iso(self.deadline),
                "target_name": self.target_name,
                "text": self.text,
            }
        )


@dataclass(frozen=True)
class PlanAdjustment:
    """Un ajuste con su motivo; ``source`` dice si vino del régimen o de la posición."""

    source: str  # SOURCE_REGIME o SOURCE_POSITION
    code: str
    message: str
    risk_multiplier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(dataclasses.asdict(self))


@dataclass(frozen=True)
class AlternativeEntry:
    """Entrada alternativa con orden límite en el S/R más cercano del timeframe de entrada.

    ``plan`` es el plan completo calculado con la misma función (sin su propia
    alternativa); ``None`` si no hay nivel, con el motivo en ``reason``.
    """

    kind: str  # ENTRY_LIMIT
    timeframe: str
    level_kind: str  # "soporte" o "resistencia"
    level_price: float | None
    distance_pct: float | None  # % humano desde el precio actual, con signo
    plan: TradePlan | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        """Resumen compacto (el plan completo de la alternativa no se repite).

        Trae lo que hace falta para poner la orden: niveles, posición, los
        ajustes con su motivo (por ejemplo, por qué subió el margen),
        liquidación, comisiones y stop por tiempo.
        """
        out: dict[str, Any] = {
            "kind": self.kind,
            "available": self.available,
            "timeframe": self.timeframe,
            "level_kind": self.level_kind,
            "level_price": self.level_price,
            "distance_pct": self.distance_pct,
            "reason": self.reason,
        }
        p = self.plan
        if p is not None:
            tps = {tp.name: tp for tp in p.take_profits}
            tp2 = tps.get("TP2")
            pos = p.position
            out.update(
                {
                    "status": p.status,
                    "is_reference": p.is_reference,
                    "headline": p.headline,
                    "entry": p.entry_price,
                    "stop": None if p.stop is None else p.stop.price,
                    "stop_distance_pct": None if p.stop is None else p.stop.distance_pct,
                    "tp1": None if "TP1" not in tps else tps["TP1"].price,
                    "tp2": None if tp2 is None else tp2.price,
                    "tp3": None if "TP3" not in tps else tps["TP3"].price,
                    "net_rr_tp2": p.net_rr_tp2,
                    "net_gain_tp2_usd": None if tp2 is None else tp2.net_gain_usd,
                    "tp2_unlikely_intraday": None if tp2 is None else tp2.unlikely_intraday,
                    "margin_usd": None if pos is None else pos.margin_usd,
                    "size_usd": None if pos is None else pos.size_usd,
                    "leverage": None if pos is None else pos.leverage,
                    "max_loss_usd": None if pos is None else pos.risk_usd,
                    "blocking_reasons": list(p.blocking_reasons),
                    "adjustments": [a.to_dict() for a in p.adjustments],
                    "liquidation": None
                    if pos is None
                    else {
                        "model": "isolated",
                        "price": pos.liquidation_price,
                        "distance_pct": _pct(pos.liquidation_distance_pct),
                        "ratio": pos.liquidation_ratio,
                        "min_ratio": pos.liquidation_min_ratio,
                        "ok": pos.liquidation_ok,
                    },
                    "fees_round_trip_at_stop_usd": None
                    if p.fees is None
                    else p.fees.round_trip_at_stop_usd,
                    "fees_round_trip_at_tp2_usd": None
                    if p.fees is None
                    else p.fees.round_trip_at_target_usd,
                    "time_stop": None if p.time_stop is None else p.time_stop.to_dict(),
                    "warnings": list(p.warnings),
                }
            )
        return json_safe(out)


@dataclass(frozen=True)
class TradePlan:
    """Plan de operación completo. ``to_dict()`` lo deja listo para JSON.

    - ``status``: ``"LONG"``, ``"SHORT"`` o ``"NO_OPERAR"``.
    - ``direction``: dirección candidata por el score (``None`` si no llega a
      ningún umbral).
    - ``is_reference``: "No operar" pero con niveles calculados, que se muestran
      como referencia para ver por qué.
    - ``rules``: todas las reglas evaluadas, en orden, con ``ok``;
      ``blocking_rules`` / ``blocking_reasons`` son las que fallaron.
    - ``adjustments``: los ajustes del régimen y de la posición, con su motivo.
    - ``position``: el ``PositionFit`` final (margen, tamaño, apalancamiento,
      liquidación); ``regime_adjustment``: el ``RegimeAdjustmentResult``;
      ``regime_summary``: el régimen usado, listo para mostrar.
    - ``requested_risk_usd`` (el del perfil), ``adjusted_risk_usd`` (con los
      multiplicadores del régimen) y la pérdida máxima real en
      ``position.risk_usd`` (comisiones incluidas).
    - ``tp2_missing_reason``: por qué no hay TP2 (stop absurdamente lejos en
      un short), o ``None``.
    - ``day_state``: el estado del día que se usó (operaciones y resultado de
      hoy, evento marcado, margen disponible), como datos para la API.
    - ``price_diff_pct`` / ``price_suspect``: diferencia del precio de entrada
      contra el de referencia y si pasa de
      ``exchanges.nexo.manual_price_max_diff_pct`` (posible error de tipeo).
    """

    symbol: str
    status: PlanStatus
    direction: Side | None
    headline: str
    is_reference: bool
    entry_kind: str
    created_at: datetime
    entry_price: float
    price_source: str
    stop: StopPlan | None
    stop_missing_reason: str | None
    take_profits: tuple[TakeProfit, ...]
    tp1_missing_reason: str | None
    tp2_missing_reason: str | None
    net_rr_tp2: float | None
    min_rr: float | None
    min_rr_reason: str | None
    position: PositionFit | None
    requested_risk_usd: float  # riesgo del perfil, antes de ajustar
    adjusted_risk_usd: float | None  # después de los multiplicadores del régimen
    risk_multiplier: float | None
    fees: FeesPlan | None
    funding: FundingEstimate | None
    time_stop: TimeStop | None
    daily_range_usd: float | None
    daily_range_pct: float | None
    adjustments: tuple[PlanAdjustment, ...]
    rules: tuple[RuleResult, ...]
    warnings: tuple[str, ...]
    alternative: AlternativeEntry | None
    score_total: float
    score_hint: Side | None
    score_summary: str | None
    min_abs_score: float | None
    min_abs_score_reason: str | None
    regime_adjustment: RegimeAdjustmentResult | None = field(repr=False, default=None)
    regime_summary: dict[str, Any] = field(default_factory=dict)
    profile: EffectiveProfile | None = field(repr=False, default=None, compare=False)
    day_state: dict[str, Any] = field(default_factory=dict)
    price_diff_pct: float | None = None
    price_suspect: bool = False

    @property
    def ok(self) -> bool:
        return self.status != STATUS_NO_TRADE

    @property
    def blocking_rules(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.rules if not r.ok)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(r.message for r in self.blocking_rules)

    def take_profit(self, name: str) -> TakeProfit | None:
        """El objetivo ``"TP1"``, ``"TP2"`` o ``"TP3"``, o ``None`` si no hay."""
        return next((tp for tp in self.take_profits if tp.name == name), None)

    def _position_dict(self) -> dict[str, Any] | None:
        f = self.position
        if f is None:
            return None
        return json_safe(
            {
                "margin_usd": f.margin_usd,
                "size_usd": f.size_usd,
                "quantity": f.quantity,
                "leverage": f.leverage,
                "max_leverage": f.max_leverage,
                "red_warning": f.red_warning,
                "red_warning_text": f.warnings[0] if f.red_warning and f.warnings else None,
                "profile_margin_usd": f.profile_margin_usd,
                "max_margin_usd": f.max_margin_usd,
                "available_margin_usd": f.available_margin_usd,
                "initial_size_usd": f.initial_size_usd,
                "initial_leverage": f.initial_leverage,
                "margin_raised": f.margin_raised,
                "size_reduced": f.size_reduced,
                "margin_lowered_to_1x": f.margin_lowered_to_1x,
                "limited_by": list(f.limited_by),
            }
        )

    def _liquidation_dict(self) -> dict[str, Any] | None:
        f = self.position
        if f is None:
            return None
        wallet = None
        if f.wallet_equity_usd is not None:
            wallet = {
                "equity_usd": f.wallet_equity_usd,
                "price": f.wallet_liquidation_price,
                "distance_pct": _pct(f.wallet_liquidation_distance_pct),
            }
        check = f.liquidation_check
        return json_safe(
            {
                "model": "isolated",
                "price": f.liquidation_price,
                "distance_pct": _pct(f.liquidation_distance_pct),
                "stop_distance_pct": _pct(f.stop_distance_pct),
                "ratio": f.liquidation_ratio,
                "min_ratio": f.liquidation_min_ratio,
                "before_stop": f.liq_before_stop,
                # La liquidación cae del lado ganador de la entrada: nacería liquidada.
                "at_open": check.liquidation_distance is not None
                and check.liquidation_distance <= 0,
                "ok": f.liquidation_ok,
                "wallet": wallet,
            }
        )

    def _risk_dict(self) -> dict[str, Any]:
        f = self.position
        return json_safe(
            {
                "requested_usd": self.requested_risk_usd,
                "risk_multiplier": self.risk_multiplier,
                "after_adjustments_usd": self.adjusted_risk_usd,
                "max_loss_usd": None if f is None else f.risk_usd,
                "price_loss_usd": None if f is None else f.price_loss_at_stop_usd,
                "fees_at_stop_usd": None if f is None else f.fees_usd,
                "reduced_by_position": None if f is None else f.size_reduced,
                "fees_included": None if f is None else f.fees_included_in_risk,
                "min_operable_usd": None
                if self.profile is None
                else self.profile.min_operable_risk_usd,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (sin NaN ni infinitos, fechas en ISO 8601)."""
        return json_safe(
            {
                "symbol": self.symbol,
                "status": self.status,
                "direction": self.direction,
                "headline": self.headline,
                "is_reference": self.is_reference,
                "entry_kind": self.entry_kind,
                "created_at": _iso(self.created_at),
                "entry": {
                    "price": self.entry_price,
                    "kind": self.entry_kind,
                    "price_source": self.price_source,
                    "diff_vs_reference_pct": self.price_diff_pct,
                    "price_suspect": self.price_suspect,
                },
                "stop": None if self.stop is None else self.stop.to_dict(),
                "stop_missing_reason": self.stop_missing_reason,
                "take_profits": [tp.to_dict() for tp in self.take_profits],
                "tp1_missing_reason": self.tp1_missing_reason,
                "tp2_missing_reason": self.tp2_missing_reason,
                "net_rr_tp2": self.net_rr_tp2,
                "min_rr": self.min_rr,
                "min_rr_reason": self.min_rr_reason,
                "position": self._position_dict(),
                "liquidation": self._liquidation_dict(),
                "risk": self._risk_dict(),
                "fees": None if self.fees is None else self.fees.to_dict(),
                "funding": None if self.funding is None else self.funding.to_dict(),
                "time_stop": None if self.time_stop is None else self.time_stop.to_dict(),
                "daily_range": {"usd": self.daily_range_usd, "pct": self.daily_range_pct},
                "adjustments": [a.to_dict() for a in self.adjustments],
                "rules": [r.to_dict() for r in self.rules],
                "blocking_reasons": list(self.blocking_reasons),
                "warnings": list(self.warnings),
                "alternative": None if self.alternative is None else self.alternative.to_dict(),
                "score": {
                    "total": self.score_total,
                    "direction_hint": self.score_hint,
                    "summary": self.score_summary,
                    "min_abs_score": self.min_abs_score,
                    "min_abs_score_reason": self.min_abs_score_reason,
                },
                "regime": dict(self.regime_summary),
                "profile": None if self.profile is None else self.profile.to_dict(),
                "day_state": dict(self.day_state),
            }
        )


# --- piezas ----------------------------------------------------------------------


def _level_word(side: Side, *, target: bool) -> str:
    """Soporte o resistencia: el del stop (``target=False``) o el del objetivo."""
    if side == "long":
        return "resistencia" if target else "soporte"
    return "soporte" if target else "resistencia"


def _plural_word(word: str) -> str:
    return "resistencias" if word == "resistencia" else "soportes"


def _favourable(side: Side, entry: float, price: float) -> bool:
    """El precio está del lado ganador (arriba en un long, abajo en un short)."""
    return price > entry if side == "long" else price < entry


def _clean_levels(levels: Mapping[str, SRResult | None]) -> dict[str, SRResult | None]:
    """Los mismos niveles sin los que tienen un precio no finito o no positivo.

    El pipeline nunca los genera, pero ``PlanInputs`` se puede armar a mano (el
    backtest): un soporte en 0 o en NaN no puede romper el plan ni terminar
    como stop u objetivo.
    """
    out: dict[str, SRResult | None] = {}
    for tf, sr in levels.items():
        if sr is None:
            out[tf] = None
            continue
        valid = tuple(c for c in sr.clusters if (clean_number(c.price) or 0.0) > 0)
        out[tf] = sr if len(valid) == len(sr.clusters) else dataclasses.replace(sr, clusters=valid)
    return out


def _next_level(
    side: Side,
    entry: float,
    levels: Mapping[str, SRResult | None],
    atr: float,
    min_distance_atr: float,
) -> tuple[float, str, tuple[SkippedLevel, ...]] | str:
    """TP1: el S/R más cercano a favor a ``min_distance_atr`` ATR o más.

    Devuelve ``(precio, timeframe, salteados)`` o el motivo (texto) si no hay.
    Ante dos niveles al mismo precio gana el primer timeframe.
    """
    best: tuple[float, str] | None = None
    skipped: list[SkippedLevel] = []
    timeframes = [tf for tf, sr in levels.items() if sr is not None]
    for tf in timeframes:
        sr = levels[tf]
        assert sr is not None
        for cluster in sr.clusters:
            price = cluster.price
            if not _favourable(side, entry, price):
                continue
            distance_atr = abs(price - entry) / atr
            if distance_atr < min_distance_atr:
                skipped.append(SkippedLevel(price, tf, distance_atr))
                continue
            closer = best is None or (price < best[0] if side == "long" else price > best[0])
            if closer:
                best = (price, tf)
    skipped.sort(key=lambda s: s.distance_atr)
    if best is None:
        word = _plural_word(_level_word(side, target=True))
        where = "arriba" if side == "long" else "debajo"
        tfs = "/".join(timeframes) or "ningún timeframe"
        if skipped:
            article = "todas las" if word == "resistencias" else "todos los"
            return (
                f"No hay TP1: {article} {word} {where} del precio en {tfs} están a menos de "
                f"{fmt_num(min_distance_atr, 2)} ATR."
            )
        return f"No hay TP1: no hay {word} {where} del precio en {tfs}."
    return best[0], best[1], tuple(skipped)


def min_rr_target(
    side: Side, entry: float, stop: float, min_rr: float, entry_rate: float, exit_rate: float
) -> float:
    """Precio objetivo con el que el R:R **con comisiones** es exactamente ``min_rr``.

    Despeja ``fees.net_rr == min_rr`` (ver el docstring del módulo). En un short
    puede dar un precio <= 0 si el stop está absurdamente lejos (quien llama lo
    controla).
    """
    loss = abs(entry - stop) + entry_rate * entry + exit_rate * stop
    if side == "long":
        target = (min_rr * loss + entry * (1.0 + entry_rate)) / (1.0 - exit_rate)
    else:
        target = (entry * (1.0 - entry_rate) - min_rr * loss) / (1.0 + exit_rate)
    # Por la coma flotante el R:R neto puede dar 1,4999999999999: se corre el
    # objetivo de a un "ulp" hacia afuera hasta que llegue (no cambia nada visible).
    away = math.inf if side == "long" else -math.inf
    for _ in range(_NUDGE_STEPS):
        if target <= 0 or net_rr(side, entry, stop, target, entry_rate, exit_rate) >= min_rr:
            break
        target = math.nextafter(target, away)
    return target


def _daily_range(inputs: PlanInputs, entry: float) -> tuple[float | None, float | None]:
    """``(usd, pct)`` del rango típico del día; si falta uno se calcula con el otro."""
    usd = clean_number(inputs.daily_range_usd)
    pct = clean_number(inputs.daily_range_pct)
    if usd is not None and usd <= 0:
        usd = None
    if pct is not None and pct <= 0:
        pct = None
    if usd is None and pct is not None:
        usd = pct / 100.0 * entry
    if pct is None and usd is not None:
        pct = usd / entry * 100.0
    return usd, pct


def _make_tp(
    name: str,
    side: Side,
    entry: float,
    stop: float,
    price: float,
    quantity: float,
    rates: tuple[float, float],
    basis: str,
    explanation: str,
    daily_range_usd: float | None,
    flag_daily: bool,
    *,
    extra_flags: tuple[str, ...] = (),
    level_timeframe: str | None = None,
    skipped: tuple[SkippedLevel, ...] = (),
) -> TakeProfit:
    f_in, f_out = rates
    risk_distance = abs(entry - stop)
    distance = abs(price - entry)
    gross = gross_pnl_usd(side, quantity, entry, price)
    fees = trade_fees_usd(quantity, entry, price, f_in, f_out)
    beyond = daily_range_usd is not None and distance > daily_range_usd * (1.0 + REL_TOL)
    flags = list(extra_flags)
    if beyond and flag_daily:
        flags.insert(0, FLAG_UNLIKELY_INTRADAY)
    return TakeProfit(
        name=name,
        price=price,
        distance_pct=distance / entry * 100.0,
        r_multiple=distance / risk_distance,
        gross_gain_usd=gross,
        fees_usd=fees,
        net_gain_usd=gross - fees,
        net_rr=net_rr(side, entry, stop, price, f_in, f_out),
        basis=basis,
        explanation=explanation,
        beyond_daily_range=beyond,
        flags=tuple(flags),
        level_timeframe=level_timeframe,
        skipped=skipped,
    )


def _r_text(multiple: float) -> str:
    """``1.5`` → ``"1,5R (1,5 veces el riesgo)."``."""
    return f"{fmt_num(multiple, 2)}R ({fmt_num(multiple, 2)} veces el riesgo)."


def _regime_rule(adj: RegimeAdjustmentResult, side: Side) -> RuleResult:
    """Regla del régimen: bloqueo por tipo de régimen (por ejemplo, long en Crash)."""
    plural = "longs" if side == "long" else "shorts"
    if adj.block_code == REGIME_BLOCKED:
        return RuleResult(
            code=REGIME_RULE,
            ok=False,
            message=adj.block_reasons[0],
            suggestion="Esperá a que cambie el régimen de mercado.",
        )
    if not adj.regime_available:
        message = (
            f"Régimen no disponible: no hay bloqueos por régimen para los {plural} "
            "(se opera más chico)."
        )
    else:
        label = CATEGORY_LABELS.get(adj.category or "", adj.category or "")
        message = f"El régimen {label} no bloquea los {plural}."
    return RuleResult(code=REGIME_RULE, ok=True, message=message)


def _stop_rule(stop: StructuralStop) -> RuleResult:
    where = "debajo del" if stop.side == "long" else "arriba de la"
    return RuleResult(
        code=STOP_RULE,
        ok=True,
        message=(
            f"Stop en {fmt_price(stop.stop)}, {where} {stop.level_kind} de "
            f"{stop.level_timeframe} (a {fmt_percent(stop.stop_distance_pct * 100.0)} de la "
            "entrada)."
        ),
    )


def _stop_missing_rule(reason: str) -> RuleResult:
    text = reason[0].lower() + reason[1:] if reason else reason
    return RuleResult(
        code=STOP_RULE,
        ok=False,
        message=f"{NO_TRADE}: {text}",
        suggestion="Esperá a que se forme un soporte o resistencia claro para el stop.",
    )


# Sugerencia cuando no se puede fijar el TP2 (stop absurdamente lejos).
_TP2_MISSING_TIP = (
    "Esperá una entrada con el stop más cerca: con este stop no hay objetivo posible."
)


def _tp2_missing_reason(stop: StructuralStop, rr: float) -> str:
    """Por qué no hay TP2: en un short, el objetivo caería en un precio de cero o menos."""
    return (
        f"el stop está tan lejos (a {fmt_percent(stop.stop_distance_pct * 100.0)} de la entrada) "
        f"que, para ganar {fmt_ratio(rr)} veces lo que arriesgás, el precio tendría que bajar "
        "a cero o menos"
    )


def _rr_rule(net_rr_tp2: float | None, min_rr: float, tp2_missing: str | None) -> RuleResult:
    """R:R a TP2 con comisiones; sin TP2, el motivo y qué hacer."""
    if tp2_missing is None:
        return check_rr(net_rr_tp2, min_rr)
    return RuleResult(
        code=MIN_RR,
        ok=False,
        message=f"{NO_TRADE}: no hay TP2: {tp2_missing}.",
        suggestion=_TP2_MISSING_TIP,
    )


def _daily_range_rule(
    tp2: TakeProfit | None,
    daily_usd: float | None,
    entry: float,
    block_enabled: bool,
) -> RuleResult:
    """El TP2 tiene que entrar en el rango típico del día (decisión del usuario)."""
    if tp2 is None:
        return RuleResult(
            code=TP2_DAILY_RANGE,
            ok=False,
            message=(
                f"{NO_TRADE}: sin TP2 no se puede controlar si el objetivo se alcanza en el día."
            ),
            suggestion=_TP2_MISSING_TIP,
        )
    if daily_usd is None:
        return RuleResult(
            code=TP2_DAILY_RANGE,
            ok=True,
            message=(
                "No hay dato del rango típico del día: no se pudo controlar si el TP2 se "
                "alcanza en el día."
            ),
        )
    daily_pct = daily_usd / entry * 100.0
    tp_txt = fmt_percent(tp2.distance_pct)
    day_txt = fmt_percent(daily_pct)
    if not tp2.beyond_daily_range:
        return RuleResult(
            code=TP2_DAILY_RANGE,
            ok=True,
            message=f"El TP2 (a {tp_txt}) entra en el rango típico del día ({day_txt}).",
        )
    if not block_enabled:
        return RuleResult(
            code=TP2_DAILY_RANGE,
            ok=True,
            message=(
                f"El TP2 (a {tp_txt}) queda más lejos que el rango típico del día ({day_txt}): "
                "poco probable en el día (la config no bloquea por esto)."
            ),
        )
    return RuleResult(
        code=TP2_DAILY_RANGE,
        ok=False,
        message=(
            f"{NO_TRADE}: el TP2 queda a {tp_txt} de la entrada y el rango típico del día es "
            f"{day_txt} (lo que se mueve el precio en un día normal): es poco probable "
            "llegar en una operación del día."
        ),
        suggestion=(
            "Esperá una entrada con el stop más cerca (así el TP2 también se acerca) o un "
            "día con más movimiento."
        ),
    )


def _liquidation_rule(fit: PositionFit) -> RuleResult:
    """Regla de liquidación sobre la posición ya ajustada.

    ``fit_position`` ya subió el margen o achicó la posición todo lo posible, así
    que si igual falla no hay ajuste que sirva: la sugerencia lo dice.
    """
    result = check_liquidation(fit.liquidation_check)
    if result.ok:
        return result
    return dataclasses.replace(
        result,
        suggestion=(
            "No hay ajuste posible: con el stop tan lejos, ni subiendo el margen ni achicando "
            f"la posición la liquidación queda a {fmt_ratio(fit.liquidation_min_ratio)} veces "
            "la distancia al stop. Esperá una entrada con el stop más cerca."
        ),
    )


def _regime_floor_rule(adj: RegimeAdjustmentResult, profile: EffectiveProfile) -> RuleResult:
    """El riesgo con los multiplicadores del régimen ya quedó debajo del mínimo operable.

    No depende de los niveles: se lista aunque no se haya podido calcular el
    stop. El motivo es el último de ``block_reasons`` (el del piso).
    """
    suggestion = check_min_operable_risk(adj.risk_usd, profile).suggestion
    return RuleResult(MIN_OPERABLE_RISK, False, adj.block_reasons[-1], suggestion)


def _min_risk_rule(
    adj: RegimeAdjustmentResult, fit: PositionFit, profile: EffectiveProfile
) -> RuleResult:
    """Riesgo mínimo operable, con el motivo más claro.

    Falla si el régimen ya dejó el riesgo debajo del mínimo (aunque con las
    comisiones aparte la pérdida total lo pase) o si el riesgo final de la
    posición, en los términos del perfil (``PositionFit.floor_risk_usd``),
    quedó debajo.
    """
    if adj.risk_below_min:
        # El piso se cruzó por los multiplicadores del régimen.
        return _regime_floor_rule(adj, profile)
    base = check_min_operable_risk(fit.floor_risk_usd, profile)
    if base.ok:
        return base
    if fit.size_reduced and fit.block_reason:
        if LIMIT_LEVERAGE in fit.limited_by:
            tip = (
                "Esperá una entrada con el stop más lejos: con un stop tan corto la posición "
                "no entra en tu tope de apalancamiento."
            )
        else:
            tip = (
                "Esperá otra entrada: con este stop y el margen máximo, la liquidación "
                "quedaría demasiado cerca."
            )
        return RuleResult(MIN_OPERABLE_RISK, False, fit.block_reason, tip)
    return base


def _fees_plan(fit: PositionFit, tp2: TakeProfit | None, rates: tuple[float, float]) -> FeesPlan:
    f_in, f_out = rates
    q = fit.quantity
    entry_fee = q * fit.entry * f_in
    exit_stop = q * fit.stop * f_out
    round_stop = entry_fee + exit_stop
    exit_target = round_target = pct_target = None
    if tp2 is not None:
        exit_target = q * tp2.price * f_out
        round_target = entry_fee + exit_target
        if tp2.gross_gain_usd > 0:
            pct_target = fees_pct_of(round_target, tp2.gross_gain_usd)
    return FeesPlan(
        entry_rate_pct=f_in * 100.0,
        exit_rate_pct=f_out * 100.0,
        entry_fee_usd=entry_fee,
        exit_fee_at_stop_usd=exit_stop,
        round_trip_at_stop_usd=round_stop,
        pct_of_risk=fees_pct_of(round_stop, fit.risk_usd),
        included_in_risk=fit.fees_included_in_risk,
        target_name=None if tp2 is None else tp2.name,
        exit_fee_at_target_usd=exit_target,
        round_trip_at_target_usd=round_target,
        pct_of_target=pct_target,
    )


def _funding_estimate(
    side: Side, size_usd: float, inputs: PlanInputs, settings: AppSettings
) -> FundingEstimate | None:
    if not settings.fees.include_funding_estimate:
        return None
    hours = float(settings.trade_plan.time_stop_hours)
    pct = clean_number(inputs.funding_pct_8h)
    interval = clean_number(inputs.funding_interval_hours)
    if interval is None or interval <= 0:
        interval = float(settings.fees.funding_interval_hours)
    hours_txt = fmt_num(hours, 1)
    if pct is None:
        return FundingEstimate(
            available=False,
            hours=hours,
            rate_pct_8h=None,
            interval_hours=interval,
            cost_usd=None,
            text="Sin dato de funding: no se puede estimar lo que pagarías o cobrarías.",
        )
    rate = pct / 100.0 * interval / FUNDING_REFERENCE_HOURS
    cost = funding_cost_usd(side, size_usd, rate, hours, interval) if hours > 0 else 0.0
    if cost > 0:
        text = (
            f"Si la mantenés {hours_txt} h pagarías hasta {fmt_usd(cost)} de funding "
            "(aparte del riesgo)."
        )
    elif cost < 0:
        text = f"Si la mantenés {hours_txt} h cobrarías unos {fmt_usd(-cost)} de funding."
    else:
        text = f"Si la mantenés {hours_txt} h el funding no suma ni resta nada."
    return FundingEstimate(
        available=True,
        hours=hours,
        rate_pct_8h=pct,
        interval_hours=interval,
        cost_usd=cost + 0.0,
        text=text,
    )


def _time_stop(
    now: datetime, settings: AppSettings, tp1: TakeProfit | None, tp2: TakeProfit | None
) -> TimeStop | None:
    """Stop por tiempo. El objetivo que se mira es el TP1, salvo que no haya o que
    quede más lejos que el TP2 (ahí el primero que se puede tocar es el TP2).
    Sin TP1 ni TP2 no hay objetivo que mirar: ``None``."""
    hours = float(settings.trade_plan.time_stop_hours)
    if not math.isfinite(hours) or hours <= 0 or (tp1 is None and tp2 is None):
        return None
    use_tp1 = tp1 is not None and (tp2 is None or tp1.distance_pct <= tp2.distance_pct)
    target = "TP1" if use_tp1 else "TP2"
    return TimeStop(
        hours=hours,
        deadline=now + timedelta(hours=hours),
        target_name=target,
        text=(
            f"Si en {fmt_num(hours, 1)} h no tocó el {target} ni el stop, cerrá la operación "
            "(el precio no hizo lo que esperabas)."
        ),
    )


def _regime_summary(regime: Any | None, adj: RegimeAdjustmentResult | None) -> dict[str, Any]:
    category = getattr(regime, "category", None) if regime is not None else None
    available = adj.regime_available if adj is not None else category is not None
    if not available:
        return {"available": False}
    confidence = clean_number(getattr(regime, "confidence", None))
    return json_safe(
        {
            "available": True,
            "timeframe": getattr(regime, "timeframe", None),
            "name": getattr(regime, "name", None),
            "category": category,
            "label": CATEGORY_LABELS.get(str(category), category),
            "confidence": confidence,
            "unstable": bool(getattr(regime, "unstable", False)),
        }
    )


def _validate(inputs: PlanInputs, now: datetime) -> float:
    entry = ensure_positive(inputs.entry_price, "El precio de entrada")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now tiene que ser una fecha con zona horaria (vino {now!r}).")
    if inputs.available_margin_usd is not None:
        ensure_positive(inputs.available_margin_usd, "El margen disponible")
    if not isinstance(inputs.high_volatility_event, bool):
        raise ValueError(
            "La marca de evento de alta volatilidad tiene que ser sí o no "
            f"({inputs.high_volatility_event!r})."
        )
    return entry


def _headline(
    status: str,
    side: Side | None,
    entry: float,
    entry_kind: str,
    rules: list[RuleResult],
    stop: StructuralStop | None,
    tp2: TakeProfit | None,
    fit: PositionFit | None,
) -> str:
    """Una oración simple que resume el plan.

    Los precios van con los decimales de la entrada, como la tabla de niveles
    de la consola (el mismo precio se lee igual en los dos lados).
    """
    if status != STATUS_NO_TRADE and side is not None and stop and tp2 and fit:
        how = "a mercado" if entry_kind == ENTRY_MARKET else "con orden límite"
        fees = " (comisiones incluidas)" if fit.fees_included_in_risk else ""
        return (
            f"{_SIDE_UPPER[side]} {how} en {fmt_price(entry, entry)}: stop en "
            f"{fmt_price(stop.stop, entry)} y objetivo principal (TP2) en "
            f"{fmt_price(tp2.price, entry)}; arriesgás "
            f"{fmt_usd(fit.risk_usd)}{fees} para ganar {fmt_usd(tp2.net_gain_usd)} netos."
        )
    failed = [r for r in rules if not r.ok]
    if not failed:  # no debería pasar: sin dirección siempre falla la regla de dirección
        return f"{NO_TRADE}."
    first = failed[0].message
    if first.startswith(f"{NO_TRADE}: "):
        first = first[len(NO_TRADE) + 2 :]
    first = first.rstrip(".")
    more = len(failed) - 1
    tail = "" if more == 0 else f" (y {more} {'motivo' if more == 1 else 'motivos'} más)"
    return f"{NO_TRADE}: {first}{tail}."


# --- plan ----------------------------------------------------------------------


def build_trade_plan(inputs: PlanInputs, settings: AppSettings, *, now: datetime) -> TradePlan:
    """Plan de operación completo (ver el docstring del módulo).

    ``now`` es el momento del análisis (con zona horaria): de ahí sale el stop
    por tiempo. Levanta ``ValueError`` con entradas inválidas (precio no
    positivo, ``now`` sin zona horaria, margen disponible no positivo, etc.).
    """
    return _build(inputs, settings, now, entry_kind=ENTRY_MARKET, with_alternative=True)


def _build(
    inputs: PlanInputs,
    settings: AppSettings,
    now: datetime,
    *,
    entry_kind: str,
    with_alternative: bool,
) -> TradePlan:
    entry = _validate(inputs, now)
    profile = inputs.profile
    tp_cfg = settings.trade_plan
    take_cfg = tp_cfg.take_profit
    entry_tf = settings.timeframes.entry
    score_total = ensure_finite(inputs.score.total, "El score")
    thresholds = settings.scoring.direction_thresholds
    daily_usd, daily_pct = _daily_range(inputs, entry)

    rules: list[RuleResult] = []
    adjustments: list[PlanAdjustment] = []
    warnings: list[str] = list(profile.warnings)

    # Precio de entrada sospechoso (el de Nexo muy distinto del de referencia):
    # no bloquea, pero el plan lo avisa y no recomienda la entrada alternativa.
    price_diff = clean_number(inputs.price_diff_pct)
    max_diff = float(settings.exchanges.nexo.manual_price_max_diff_pct)
    price_suspect = price_diff is not None and abs(price_diff) > max_diff
    if price_suspect:
        assert price_diff is not None
        reference = entry / (1.0 + price_diff / 100.0) if price_diff > -100.0 else None
        ref_txt = "" if reference is None else f" ({fmt_price(reference)})"
        warnings.append(
            f"Revisá el precio de entrada: {fmt_price(entry)} difiere "
            f"{fmt_percent(abs(price_diff))} del precio de referencia{ref_txt}, más que el "
            f"{fmt_percent(max_diff)} que se acepta. Si está mal escrito, todo este plan (stop, "
            "objetivos y tamaño) está mal: no lo uses hasta cargarlo bien."
        )

    # 1) Dirección.
    rules.append(check_direction(score_total, thresholds))
    side = direction_from_score(score_total, thresholds)

    # 2) Ajustes por régimen (solo con dirección).
    adj: RegimeAdjustmentResult | None = None
    if side is not None:
        adj = apply_regime_adjustments(
            side, score_total, inputs.regime, inputs.funding_pct_8h, profile, settings
        )
        rules.append(_regime_rule(adj, side))
        rules.append(adj.score_check)
        adjustments += [
            PlanAdjustment(SOURCE_REGIME, a.code, a.message, a.risk_multiplier)
            for a in adj.adjustments
        ]

    # Reglas que no dependen de los niveles.
    rules.append(check_timeframe_conflict(inputs.trend_4h, inputs.trend_1h))
    daily_rule = check_daily_limits(inputs.trades_today, inputs.pnl_today_usd, profile)
    rules.append(daily_rule)
    rules.append(
        check_high_volatility_event(inputs.high_volatility_event, inputs.event_description)
    )

    stop_tfs = list(dict.fromkeys(tp_cfg.stop_timeframes))
    levels = _clean_levels(inputs.levels)
    stop_levels = {tf: levels.get(tf) for tf in stop_tfs}
    atr = clean_number(inputs.entry_atr)

    found: StructuralStop | None = None
    stop_plan: StopPlan | None = None
    stop_missing: str | None = None
    tps: list[TakeProfit] = []
    tp1_missing: str | None = None
    fit: PositionFit | None = None
    fees_plan: FeesPlan | None = None
    funding: FundingEstimate | None = None
    time_stop: TimeStop | None = None
    tp2: TakeProfit | None = None
    tp2_missing: str | None = None
    rr_tp2: float | None = None

    if side is not None and adj is not None:
        # 3) Stop por estructura.
        if atr is None or atr <= 0:
            stop_missing = f"No hay ATR de {entry_tf}: no se puede calcular el stop."
        else:
            result = structural_stop(
                side,
                entry,
                stop_levels,
                atr,
                tp_cfg.stop_atr_buffer,
                tp_cfg.min_stop_distance_atr,
            )
            if isinstance(result, StopNotFound):
                stop_missing = result.reason
            else:
                found = result
        if found is None:
            rules.append(_stop_missing_rule(stop_missing or ""))
            # El piso de riesgo del régimen no depende de los niveles: se lista igual.
            if adj.risk_below_min:
                rules.append(_regime_floor_rule(adj, profile))
        else:
            rules.append(_stop_rule(found))
            op = "menos" if side == "long" else "más"
            stop_plan = StopPlan(
                price=found.stop,
                distance_pct=found.stop_distance_pct * 100.0,
                level_price=found.level_price,
                level_timeframe=found.level_timeframe,
                level_kind=found.level_kind,
                atr=found.atr,
                atr_timeframe=entry_tf,
                buffer_atr=found.buffer_atr,
                min_distance_atr=found.min_distance_atr,
                skipped=found.skipped,
                explanation=(
                    f"{found.level_kind.capitalize()} de {found.level_timeframe} en "
                    f"{fmt_price(found.level_price)} {op} {fmt_num(found.buffer_atr, 2)} × ATR "
                    f"{entry_tf} ({fmt_price(found.atr)})."
                ),
            )

    if found is not None and side is not None and adj is not None and atr is not None:
        stop = found.stop
        risk_distance = abs(entry - stop)
        sign = 1.0 if side == "long" else -1.0
        rates = fee_rates_from_config(settings.fees)
        min_rr = adj.min_rr

        # 5) Posición (antes que los objetivos: las ganancias usan la cantidad final).
        fit = fit_position(
            side,
            entry,
            stop,
            adj.risk_usd,
            profile,
            settings,
            available_margin_usd=inputs.available_margin_usd,
            wallet_equity_usd=inputs.wallet_equity_usd,
        )
        adjustments += [
            PlanAdjustment(SOURCE_POSITION, a.code, a.message, a.risk_multiplier)
            for a in fit.adjustments
        ]
        warnings += list(fit.warnings)
        q = fit.quantity
        flag_daily = bool(tp_cfg.flag_tp_beyond_daily_range)

        # 4) Objetivos. TP2 primero (TP1 se compara contra él).
        tp2_r_distance = float(take_cfg.tp2_r) * risk_distance
        tp2_price = entry + sign * tp2_r_distance
        tp2_basis = TP_R_MULTIPLE
        if take_cfg.tp2_ensure_min_rr:
            needed = min_rr_target(side, entry, stop, min_rr, *rates)
            if abs(needed - entry) > tp2_r_distance:
                tp2_price, tp2_basis = needed, TP_MIN_NET_RR
        if tp2_price <= 0:
            # Solo en un short con el stop absurdamente lejos: no hay objetivo posible.
            target_rr = min_rr if tp2_basis == TP_MIN_NET_RR else float(take_cfg.tp2_r)
            tp2_missing = _tp2_missing_reason(found, target_rr)
        if tp2_price > 0:
            r_mult = abs(tp2_price - entry) / risk_distance
            if tp2_basis == TP_MIN_NET_RR:
                why = (
                    f"{fmt_ratio(r_mult)}R: lo justo para que el R:R con comisiones llegue a "
                    f"{fmt_ratio(min_rr)} (lo pide {adj.min_rr_reason})."
                )
            else:
                why = _r_text(float(take_cfg.tp2_r))
            tp2 = _make_tp(
                "TP2", side, entry, stop, tp2_price, q, rates, tp2_basis, why, daily_usd, flag_daily
            )
            rr_tp2 = tp2.net_rr

            # TP3: el más lejano entre tp3_r × R y TP2 + (tp3_r − tp2_r) × R.
            step = float(take_cfg.tp3_r) - float(take_cfg.tp2_r)
            d_by_r = float(take_cfg.tp3_r) * risk_distance
            d_after = abs(tp2_price - entry) + step * risk_distance
            d3 = max(d_by_r, d_after)
            tp3_price = entry + sign * d3
            if tp3_price > 0:
                if d_after > d_by_r * (1.0 + REL_TOL):
                    why3 = f"{fmt_ratio(d3 / risk_distance)}R: el TP2 más {fmt_num(step, 2)}R."
                    basis3 = TP_AFTER_TP2
                else:
                    why3 = _r_text(float(take_cfg.tp3_r))
                    basis3 = TP_R_MULTIPLE
                tp3 = _make_tp(
                    "TP3",
                    side,
                    entry,
                    stop,
                    tp3_price,
                    q,
                    rates,
                    basis3,
                    why3,
                    daily_usd,
                    flag_daily,
                )
            else:
                tp3 = None
        else:
            tp3 = None

        next_level = _next_level(
            side, entry, stop_levels, atr, float(take_cfg.tp1_min_distance_atr)
        )
        tp1: TakeProfit | None = None
        if isinstance(next_level, str):
            tp1_missing = next_level
        else:
            level_price, level_tf, skipped = next_level
            word = _level_word(side, target=True)
            why1 = f"Siguiente {word} de {level_tf}."
            if skipped:
                n = len(skipped)
                why1 = (
                    f"Siguiente {word} de {level_tf} (se {'salteó' if n == 1 else 'saltearon'} "
                    f"{n} {'nivel' if n == 1 else 'niveles'} a menos de "
                    f"{fmt_num(take_cfg.tp1_min_distance_atr, 2)} ATR)."
                )
            extra: tuple[str, ...] = ()
            if tp2 is not None and abs(level_price - entry) > abs(tp2.price - entry):
                extra = (FLAG_TP1_BEYOND_TP2,)
            tp1 = _make_tp(
                "TP1",
                side,
                entry,
                stop,
                level_price,
                q,
                rates,
                TP_NEXT_LEVEL,
                why1,
                daily_usd,
                flag_daily,
                extra_flags=extra,
                level_timeframe=level_tf,
                skipped=skipped,
            )
        tps = [tp for tp in (tp1, tp2, tp3) if tp is not None]

        # 6-9) Comisiones, reglas de la posición, funding y stop por tiempo.
        fees_plan = _fees_plan(fit, tp2, rates)
        rules.append(_rr_rule(rr_tp2, min_rr, tp2_missing))
        rules.append(
            _daily_range_rule(tp2, daily_usd, entry, bool(take_cfg.block_if_tp2_beyond_daily_range))
        )
        rules.append(check_leverage(fit.leverage, profile))
        rules.append(_liquidation_rule(fit))
        rules.append(_min_risk_rule(adj, fit, profile))
        funding = _funding_estimate(side, fit.size_usd, inputs, settings)
        time_stop = _time_stop(now, settings, tp1, tp2)

    # Límite de pérdida del día: la regla bloquea cuando ya se llegó; si todavía
    # no, pero esta operación lo pasaría si toca el stop, se avisa (no bloquea).
    loss_today = max(0.0, -float(inputs.pnl_today_usd))
    max_daily_loss = float(profile.max_daily_loss_usd)
    loss_if_stopped = None if fit is None else loss_today + fit.risk_usd
    if (
        daily_rule.ok
        and loss_if_stopped is not None
        and loss_if_stopped > max_daily_loss * (1.0 + REL_TOL)
    ):
        warnings.append(
            f"Ojo con el límite del día: hoy ya perdiste {fmt_usd(loss_today)}; si esta "
            f"operación toca el stop, el día cerraría con {fmt_usd(loss_if_stopped)} de "
            f"pérdida, más que tu límite de {fmt_usd(max_daily_loss)}."
        )

    if found is not None:
        # Avisos que solo tienen sentido si hay niveles.
        if inputs.price_source not in NEXO_SOURCES:
            used = PRICE_SOURCE_LABELS.get(inputs.price_source, inputs.price_source)
            warnings.append(
                f"Los niveles se calcularon con el precio de {used}, no con el de Nexo: si en "
                "Nexo el precio es distinto, cargalo a mano."
            )
        if daily_usd is None:
            warnings.append(
                "No hay dato del rango típico del día: no se puede marcar qué objetivos son "
                "poco probables en el día."
            )

    failed = any(not r.ok for r in rules)
    if side is None or failed:
        status: PlanStatus = STATUS_NO_TRADE
    else:
        status = STATUS_LONG if side == "long" else STATUS_SHORT
    is_reference = status == STATUS_NO_TRADE and found is not None

    alternative: AlternativeEntry | None = None
    if with_alternative and side is not None:
        alternative = _alternative(inputs, settings, now, side, entry)

    headline = _headline(status, side, entry, entry_kind, rules, found, tp2, fit)
    if (
        status == STATUS_NO_TRADE
        and not price_suspect  # con el precio dudoso no se recomienda ninguna entrada
        and alternative is not None
        and alternative.plan is not None
        and alternative.plan.ok
    ):
        headline += (
            f" Con una orden límite en {fmt_price(alternative.level_price or 0.0, entry)} el "
            "plan sí cumple las reglas (mirá la entrada alternativa)."
        )

    day_state = {
        "trades_today": inputs.trades_today,
        "max_trades_per_day": profile.max_trades_per_day,
        "pnl_today_usd": float(inputs.pnl_today_usd),
        "loss_today_usd": loss_today,
        "max_daily_loss_usd": max_daily_loss,
        "loss_if_stopped_usd": loss_if_stopped,
        "high_volatility_event": inputs.high_volatility_event,
        "event_description": inputs.event_description,
        "available_margin_usd": inputs.available_margin_usd,
        "wallet_equity_usd": inputs.wallet_equity_usd,
    }

    return TradePlan(
        symbol=inputs.symbol,
        status=status,
        direction=side,
        headline=headline,
        is_reference=is_reference,
        entry_kind=entry_kind,
        created_at=now,
        entry_price=entry,
        price_source=inputs.price_source,
        stop=stop_plan,
        stop_missing_reason=stop_missing,
        take_profits=tuple(tps),
        tp1_missing_reason=tp1_missing,
        tp2_missing_reason=tp2_missing,
        net_rr_tp2=rr_tp2,
        min_rr=None if adj is None else adj.min_rr,
        min_rr_reason=None if adj is None else adj.min_rr_reason,
        position=fit,
        requested_risk_usd=float(profile.risk_per_trade_usd),
        adjusted_risk_usd=None if adj is None else adj.risk_usd,
        risk_multiplier=None if adj is None else adj.risk_multiplier,
        fees=fees_plan,
        funding=funding,
        time_stop=time_stop,
        daily_range_usd=daily_usd,
        daily_range_pct=daily_pct,
        adjustments=tuple(adjustments),
        rules=tuple(rules),
        warnings=tuple(dict.fromkeys(warnings)),
        alternative=alternative,
        score_total=score_total,
        score_hint=side,
        score_summary=getattr(inputs.score, "summary", None),
        min_abs_score=None if adj is None else adj.min_abs_score,
        min_abs_score_reason=None if adj is None else adj.min_abs_score_reason,
        regime_adjustment=adj,
        regime_summary=_regime_summary(inputs.regime, adj),
        profile=profile,
        day_state=json_safe(day_state),
        price_diff_pct=price_diff,
        price_suspect=price_suspect,
    )


def _alternative(
    inputs: PlanInputs, settings: AppSettings, now: datetime, side: Side, entry: float
) -> AlternativeEntry:
    """Orden límite en el S/R más cercano a favor del timeframe de entrada."""
    tf = settings.timeframes.entry
    kind = "soporte" if side == "long" else "resistencia"
    where = "debajo" if side == "long" else "arriba"
    # Sin los niveles con precio no finito o no positivo (un "inf" no es una entrada).
    sr = _clean_levels({tf: inputs.levels.get(tf)})[tf]
    candidates = (
        []
        if sr is None
        else [
            c.price for c in sr.clusters if (c.price < entry if side == "long" else c.price > entry)
        ]
    )
    if not candidates:
        return AlternativeEntry(
            kind=ENTRY_LIMIT,
            timeframe=tf,
            level_kind=kind,
            level_price=None,
            distance_pct=None,
            plan=None,
            reason=f"No hay {_plural_word(kind)} de {tf} {where} del precio para una orden límite.",
        )
    level = max(candidates) if side == "long" else min(candidates)
    alt_inputs = dataclasses.replace(inputs, entry_price=level)
    plan = _build(alt_inputs, settings, now, entry_kind=ENTRY_LIMIT, with_alternative=False)
    return AlternativeEntry(
        kind=ENTRY_LIMIT,
        timeframe=tf,
        level_kind=kind,
        level_price=level,
        distance_pct=(level - entry) / entry * 100.0,
        plan=plan,
    )


__all__ = [
    "ENTRY_LIMIT",
    "ENTRY_MARKET",
    "FLAG_TP1_BEYOND_TP2",
    "FLAG_UNLIKELY_INTRADAY",
    "REGIME_RULE",
    "STATUS_LONG",
    "STATUS_NO_TRADE",
    "STATUS_SHORT",
    "STOP_RULE",
    "TP2_DAILY_RANGE",
    "AlternativeEntry",
    "FeesPlan",
    "FundingEstimate",
    "PlanAdjustment",
    "PlanInputs",
    "StopPlan",
    "TakeProfit",
    "TimeStop",
    "TradePlan",
    "build_trade_plan",
    "min_rr_target",
    "plan_inputs_from_analysis",
]
