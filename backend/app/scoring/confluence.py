"""Score de confluencia ponderado, de -100 a +100, con el aporte de cada componente.

Cómo se calcula
    Cada componente tiene un valor ``v`` en [-1, +1] (ver ``components``) y un
    peso ``w`` de ``scoring.weights`` (suman 1). Su aporte es
    ``puntos = 100 × w × v`` y el total es la suma de los puntos, así que queda
    en [-100, +100]. Con los pesos por defecto: Tendencia 4h 20 %, Tendencia 1h
    20 %, Tendencia 15m 15 %, Momentum 20 %, Derivados 15 % y Sentimiento 10 %.

    ``direction_hint`` es ``"long"`` si el total llega a
    ``scoring.direction_thresholds.long`` (+30), ``"short"`` si llega a
    ``short`` (-30) y ``None`` en el medio. Es solo una pista: el plan de riesgo
    y los ajustes por régimen deciden si se opera. La comparación es exacta y
    es la misma función que usa el plan (``rules.direction_from_score`` llama a
    ``direction_from_total``): el total ya viene redondeado a 9 decimales, así
    que no hace falta tolerancia y la pista nunca contradice al plan.

Datos faltantes
    Un componente sin datos aporta 0 puntos, queda con ``available=False`` y se
    lista en ``missing``. Los pesos **no** se reparten entre los demás: que
    falten datos empuja el total hacia 0 ("no operar"), el lado conservador.

API
    - ``ScoreInputs``: valores sueltos (tendencias, RSI, MACD, funding, OI,
      ratio long/short, Fear & Greed, balance de noticias). Así el score se
      prueba a mano y el backtest walk-forward lo puede alimentar sin snapshots.
    - ``inputs_from_analysis(analysis, snapshot, settings, news_balance=None)``:
      arma ``ScoreInputs`` desde ``indicators.pipeline.MarketAnalysis`` y
      ``data.snapshot.MarketSnapshot``.
    - ``compute_score(inputs, settings) -> ConfluenceScore``: total, aporte por
      componente (en el orden de ``scoring.weights``, para las barras del
      dashboard), faltantes, pista de dirección y avisos en castellano.
    - ``direction_from_total(total, long, short)``: la dirección por el total.
    - ``reliable_trend(summary)``: la tendencia de un ``TimeframeSummary`` o
      ``None`` si no es confiable (la usa también el plan).

Funciones puras: no leen archivos ni usan la red. ``to_dict()`` devuelve dicts
listos para JSON (sin NaN ni infinitos).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.config import AppSettings
from app.scoring.components import (
    ComponentReading,
    clean_number,
    derivatives_component,
    fmt_num,
    momentum_component,
    score_beside_threshold,
    sentiment_component,
    trend_component,
)

if TYPE_CHECKING:
    from app.data.snapshot import MarketSnapshot
    from app.indicators.pipeline import MarketAnalysis, TimeframeSummary

Direction = Literal["long", "short"]

# Componentes de tendencia y su timeframe.
TREND_COMPONENT_TIMEFRAMES: dict[str, str] = {
    "trend_4h": "4h",
    "trend_1h": "1h",
    "trend_15m": "15m",
}
COMPONENT_LABELS: dict[str, str] = {
    "trend_4h": "Tendencia 4h",
    "trend_1h": "Tendencia 1h",
    "trend_15m": "Tendencia 15m",
    "momentum": "Momentum",
    "derivatives": "Derivados",
    "sentiment": "Sentimiento",
}
# Decimales con los que se guardan puntos y total (limpia artefactos de float).
_ROUND_DIGITS = 9


def _json_float(x: float | None) -> float | None:
    """``float`` finito o ``None`` (para ``to_dict``)."""
    return clean_number(x)


# --- modelos -------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreInputs:
    """Datos crudos que usa el score; cualquier valor puede faltar (``None``).

    - ``trends``: timeframe → ``"alcista"``, ``"bajista"``, ``"lateral"`` o
      ``None`` (sin tendencia confiable: faltan velas o EMAs).
    - ``rsi``, ``macd``, ``macd_signal``: timeframe → valor de la vela usada.
    - ``funding_rate``: tasa como fracción por intervalo (0.0001 = 0,01 %), tal
      como la informa el exchange; ``funding_interval_hours``: ese intervalo
      (``None`` = ``fees.funding_interval_hours``).
    - ``oi_change_24h_pct``: cambio del interés abierto en
      ``scoring.derivatives.oi_lookback_hours`` horas, en %.
    - ``long_short_ratio``: cuentas long / cuentas short (el último dato).
    - ``fear_greed``: índice de 0 a 100.
    - ``news_balance``: -1 (todas malas) a +1 (todas buenas); ``None`` = sin
      clasificar. Con ``news_share: 0`` no se usa.
    """

    trends: Mapping[str, str | None] = field(default_factory=dict)
    rsi: Mapping[str, float | None] = field(default_factory=dict)
    macd: Mapping[str, float | None] = field(default_factory=dict)
    macd_signal: Mapping[str, float | None] = field(default_factory=dict)
    funding_rate: float | None = None
    funding_interval_hours: float | None = None
    oi_change_24h_pct: float | None = None
    long_short_ratio: float | None = None
    fear_greed: float | None = None
    news_balance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (``NaN`` e infinitos → ``None``)."""
        return {
            "trends": {str(k): v for k, v in self.trends.items()},
            "rsi": {str(k): _json_float(v) for k, v in self.rsi.items()},
            "macd": {str(k): _json_float(v) for k, v in self.macd.items()},
            "macd_signal": {str(k): _json_float(v) for k, v in self.macd_signal.items()},
            "funding_rate": _json_float(self.funding_rate),
            "funding_interval_hours": _json_float(self.funding_interval_hours),
            "oi_change_24h_pct": _json_float(self.oi_change_24h_pct),
            "long_short_ratio": _json_float(self.long_short_ratio),
            "fear_greed": _json_float(self.fear_greed),
            "news_balance": _json_float(self.news_balance),
        }


@dataclass(frozen=True)
class ComponentScore:
    """Aporte de un componente al total."""

    key: str  # "trend_4h", "momentum", ... (las claves de scoring.weights)
    label: str  # "Tendencia 4h", "Momentum", ... (para la UI)
    weight: float  # peso de scoring.weights (0 a 1)
    value: float  # en [-1, +1]; 0 si no hay datos
    points: float  # 100 × weight × value
    available: bool
    details: tuple[str, ...]  # por qué suma o resta, en castellano
    missing_parts: tuple[str, ...] = ()  # partes sin datos ("RSI 1h", "funding")

    @property
    def max_points(self) -> float:
        """Lo máximo que puede sumar o restar (100 × peso)."""
        return 100.0 * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "weight": _json_float(self.weight),
            "weight_pct": _json_float(self.weight * 100.0),
            "max_points": _json_float(self.max_points),
            "value": _json_float(self.value),
            "points": _json_float(self.points),
            "available": self.available,
            "details": list(self.details),
            "missing_parts": list(self.missing_parts),
        }


@dataclass(frozen=True)
class ConfluenceScore:
    """Resultado de ``compute_score``."""

    total: float  # suma de los puntos, en [-100, +100]
    components: tuple[ComponentScore, ...]  # en el orden de scoring.weights
    missing: tuple[str, ...]  # claves de los componentes sin datos
    direction_hint: Direction | None  # según scoring.direction_thresholds
    long_threshold: float
    short_threshold: float
    summary: str  # una línea en castellano con el total y la pista de dirección
    warnings: tuple[str, ...] = ()

    @property
    def available_weight(self) -> float:
        """Suma de los pesos de los componentes con datos (1 = no faltó nada)."""
        return sum(c.weight for c in self.components if c.available)

    def component(self, key: str) -> ComponentScore:
        """El componente con esa clave; ``KeyError`` si no existe."""
        for c in self.components:
            if c.key == key:
                return c
        raise KeyError(key)

    def points_by_key(self) -> dict[str, float]:
        """Clave → puntos, en el orden de ``components``."""
        return {c.key: c.points for c in self.components}

    def to_dict(self) -> dict[str, Any]:
        """Dict listo para JSON (sin NaN ni infinitos)."""
        return {
            "total": _json_float(self.total),
            "direction_hint": self.direction_hint,
            "thresholds": {
                "long": _json_float(self.long_threshold),
                "short": _json_float(self.short_threshold),
            },
            "summary": self.summary,
            "available_weight": _json_float(self.available_weight),
            "components": [c.to_dict() for c in self.components],
            "missing": list(self.missing),
            "warnings": list(self.warnings),
        }


# --- cálculo -------------------------------------------------------------------


def _reading(key: str, inputs: ScoreInputs, settings: AppSettings) -> ComponentReading:
    sc = settings.scoring
    if key in TREND_COMPONENT_TIMEFRAMES:
        tf = TREND_COMPONENT_TIMEFRAMES[key]
        return trend_component(inputs.trends.get(tf), tf)
    if key == "momentum":
        return momentum_component(inputs.rsi, inputs.macd, inputs.macd_signal, sc.momentum)
    if key == "derivatives":
        return derivatives_component(
            inputs.funding_rate,
            inputs.funding_interval_hours,
            inputs.oi_change_24h_pct,
            inputs.long_short_ratio,
            sc.derivatives,
            default_interval_hours=settings.fees.funding_interval_hours,
        )
    if key == "sentiment":
        return sentiment_component(inputs.fear_greed, inputs.news_balance, sc.sentiment)
    raise ValueError(f"Componente del score desconocido: {key!r}")


def direction_from_total(
    total: float, long_threshold: float, short_threshold: float
) -> Direction | None:
    """``"long"`` si ``total >= long_threshold``, ``"short"`` si ``total <= short_threshold``.

    Comparación exacta, sin tolerancia: ``compute_score`` ya redondea el total
    a 9 decimales (limpia el ruido de float) y el score mínimo de las reglas
    también es exacto, así que un total que se ve "+30" pero es 29,999999999
    no es long en ningún lado.
    """
    if total >= long_threshold:
        return "long"
    if total <= short_threshold:
        return "short"
    return None


def _summary(total: float, hint: Direction | None, long_t: float, short_t: float) -> str:
    # Si no llega a ningún umbral, el total no se redondea al más cercano
    # ("Score +29,97: no llega a long (+30 o más)" y no "Score +30: ...").
    value, decimals = total, 1
    if hint is None:
        value, decimals = score_beside_threshold(total, long_t if total >= 0 else short_t)
    score = f"Score {fmt_num(value, decimals, signed=True)}"
    long_txt = fmt_num(long_t, 1, signed=True)
    short_txt = fmt_num(short_t, 1, signed=True)
    if hint == "long":
        return f"{score}: candidato a long (desde {long_txt})."
    if hint == "short":
        return f"{score}: candidato a short (desde {short_txt})."
    return f"{score}: no llega a long ({long_txt} o más) ni a short ({short_txt} o menos)."


def compute_score(inputs: ScoreInputs, settings: AppSettings) -> ConfluenceScore:
    """Score de confluencia con el aporte de cada componente (ver el docstring del módulo).

    Lanza ``ValueError`` si algún dato es inválido (una tendencia que no es
    alcista/bajista/lateral, un balance de noticias fuera de [-1, +1]) o la
    config de sentimiento no tiene ``miedo extremo < 50 < codicia extrema``.
    """
    sc = settings.scoring
    weights = sc.weights.model_dump()  # orden de ScoringWeights (el de config.yaml)
    components: list[ComponentScore] = []
    warnings: list[str] = []
    for key, weight in weights.items():
        r = _reading(key, inputs, settings)
        label = COMPONENT_LABELS.get(key, key)
        points = round(100.0 * weight * r.value, _ROUND_DIGITS) + 0.0  # sin "-0.0"
        components.append(
            ComponentScore(
                key=key,
                label=label,
                weight=weight,
                value=r.value + 0.0,  # sin "-0.0"
                points=points,
                available=r.available,
                details=r.details,
                missing_parts=r.missing_parts,
            )
        )
        if not r.available:
            warnings.append(
                f"Sin datos de {label}: cuenta 0 y su peso ({fmt_num(weight * 100.0, 1)} %) "
                "no se reparte entre los demás."
            )
        elif r.missing_parts:
            warnings.append(f"{label}: sin {', '.join(r.missing_parts)}; se usa lo que hay.")

    raw_total = round(sum(c.points for c in components), _ROUND_DIGITS)
    total = max(-100.0, min(100.0, raw_total)) + 0.0
    long_t, short_t = sc.direction_thresholds.long, sc.direction_thresholds.short
    hint = direction_from_total(total, long_t, short_t)
    return ConfluenceScore(
        total=total,
        components=tuple(components),
        missing=tuple(c.key for c in components if not c.available),
        direction_hint=hint,
        long_threshold=long_t,
        short_threshold=short_t,
        summary=_summary(total, hint, long_t, short_t),
        warnings=tuple(warnings),
    )


# --- desde el análisis ---------------------------------------------------------


def reliable_trend(summary: TimeframeSummary | None) -> str | None:
    """Tendencia del resumen, o ``None`` si no es confiable.

    ``indicators.pipeline`` deja "lateral" cuando faltan velas o la EMA larga;
    eso no es una lectura real de tendencia, así que cuenta como faltante. Es
    el mismo criterio para el score y para el plan (``risk.plan``), así los dos
    leen igual la tendencia de 4h y 1h.
    """
    if summary is None or summary.candles_used <= 0 or summary.trend_emas is None:
        return None
    mid, long_ = summary.trend_emas
    if clean_number(summary.ema.get(mid)) is None or clean_number(summary.ema.get(long_)) is None:
        return None
    return summary.trend


def inputs_from_analysis(
    analysis: MarketAnalysis,
    snapshot: MarketSnapshot,
    settings: AppSettings,
    news_balance: float | None = None,
) -> ScoreInputs:
    """``ScoreInputs`` desde el análisis de indicadores y la foto del mercado.

    - Tendencia, RSI y MACD: del ``TimeframeSummary`` de cada timeframe
      analizado (la última vela cerrada si ``timeframes.closed_candles_only``).
      Una tendencia sin EMAs suficientes queda en ``None``.
    - Funding: ``snapshot.funding.rate`` y el intervalo que informa el
      exchange (``interval_hours``); si no es válido, el de respaldo
      ``fees.funding_interval_hours``.
    - OI: ``snapshot.oi_change_24h_pct`` (ya calculado con
      ``scoring.derivatives.oi_lookback_hours``).
    - Ratio long/short: el último punto de ``snapshot.long_short`` (si no es
      un número positivo, falta).
    - Fear & Greed: ``snapshot.fear_greed.value``.
    - ``news_balance`` se pasa tal cual (con ``news_share: 0`` no se usa).
    """
    trends: dict[str, str | None] = {}
    rsi: dict[str, float | None] = {}
    macd: dict[str, float | None] = {}
    macd_signal: dict[str, float | None] = {}
    for tf, ta in analysis.timeframes.items():
        s = ta.summary
        trends[tf] = reliable_trend(s)
        rsi[tf] = clean_number(s.rsi)
        macd[tf] = clean_number(s.macd)
        macd_signal[tf] = clean_number(s.macd_signal)

    funding_rate: float | None = None
    interval: float | None = None
    if snapshot.funding is not None:
        funding_rate = clean_number(snapshot.funding.rate)
        interval = clean_number(snapshot.funding.interval_hours)
        if interval is None or interval <= 0:
            interval = settings.fees.funding_interval_hours

    ratio: float | None = None
    if snapshot.long_short is not None and snapshot.long_short.points:
        ratio = clean_number(snapshot.long_short.points[-1].ratio)
        if ratio is not None and ratio <= 0:
            ratio = None

    fear_greed = None if snapshot.fear_greed is None else clean_number(snapshot.fear_greed.value)
    return ScoreInputs(
        trends=trends,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        funding_rate=funding_rate,
        funding_interval_hours=interval,
        oi_change_24h_pct=clean_number(snapshot.oi_change_24h_pct),
        long_short_ratio=ratio,
        fear_greed=fear_greed,
        news_balance=clean_number(news_balance),
    )


__all__ = [
    "COMPONENT_LABELS",
    "TREND_COMPONENT_TIMEFRAMES",
    "ComponentScore",
    "ConfluenceScore",
    "Direction",
    "ScoreInputs",
    "compute_score",
    "direction_from_total",
    "inputs_from_analysis",
    "reliable_trend",
]
