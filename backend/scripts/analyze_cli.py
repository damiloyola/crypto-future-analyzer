"""Script de consola: datos, indicadores, score y plan de operación de una moneda.

Uso::

    make cli SYMBOL=BTC
    make cli SYMBOL=XRP ARGS="--offline --nexo-price 2.61"
    # o: cd backend && .venv/bin/python -m scripts.analyze_cli BTC [opciones]

Opciones:

- ``--nexo-price 61234.5``: precio de Nexo pegado a mano. Se usa para los
  cálculos y se compara con Binance (aviso si difiere más que
  ``exchanges.nexo.manual_price_max_diff_pct``). Sin esto se usa Binance.
  Acepta ``96000.5``, ``96000,5``, ``96.000,50`` (formato argentino) y
  ``96,000.50`` (formato de EE. UU.). Un solo separador seguido de exactamente
  tres cifras (``96.000``, ``2,615``) es ambiguo (¿miles o decimales?) y se
  rechaza con un mensaje que explica cómo escribirlo.
- ``--profile agresivo``: perfil de riesgo del plan (por defecto
  ``risk_profiles.default``).
- ``--max-leverage 30``: tope de apalancamiento elegido a mano (override del
  perfil), de 1x para arriba (1x = sin apalancamiento; menos no existe en un
  exchange y se rechaza con código 2). Por encima de
  ``risk_profiles.leverage_red_warning_above`` muestra una advertencia roja; por
  encima de ``risk_profiles.leverage_hard_cap`` se rechaza (código 2).
- ``--high-vol-event ["CPI de EE. UU."]``: marca un evento de alta volatilidad
  (la descripción es opcional; poné la moneda antes): el plan da "No operar".
- ``--trades-today 2`` y ``--pnl-today -30``: operaciones y resultado (USD) de
  hoy, para el límite diario del perfil (hasta el Hito 5 no se llevan solos).
  Si se pasan, el plan muestra cómo va el día; el JSON los trae en
  ``plan.day_state``.
- ``--offline``: datos sintéticos, sin red (``data.synthetic``). Muestra un
  cartel grande porque los números no son reales. ``--seed`` y ``--now``
  permiten repetir exactamente la misma corrida.
- ``--json``: el mismo contenido en JSON (claves en inglés, números sin
  formatear, ``null`` donde falta un dato). El régimen va en ``"regime"``
  (``null`` con ``--no-regime``), sin el historial vela por vela; el score en
  ``"score"`` y el plan en ``"plan"`` (``TradePlan.to_dict``).
- ``--no-regime``: no calcula el régimen de mercado (más rápido). El plan lo
  toma como "régimen no disponible" (riesgo × el multiplicador de confianza
  baja).
- ``--config``: otro ``config.yaml``. ``-v``: muestra los avisos de los logs.

Qué imprime (en castellano, en tablas alineadas para fuente monoespaciada;
con colores ANSI solo si la salida es una terminal):

1. encabezado: moneda, hora (``app.user_timezone``), perfil, fuentes de datos
   con su hora de actualización y avisos (respaldo, desactualizado,
   sintético, precio de Nexo);
2. precios: referencia (Binance, o Bybit si Binance falló; "no disponible"
   si no hubo ticker), Nexo (manual o "no disponible: se usa ..." con la fuente
   real del precio usado), diferencia y cambio 24 h;
2b. régimen de mercado (Hito 3, ``app.regime.service``) de 1h (el que se usa
   para operar) y de contexto diario: régimen actual y su tipo, confianza (la
   probabilidad del estado más probable en la última vela; si ese estado no es
   el régimen actual, dice cuál es), confirmado o en espera (candidato y velas
   que lleva), "régimen inestable", hace cuánto empezó, fecha del modelo y
   cantidad de estados, una tabla corta por régimen y los demás avisos (por
   ejemplo, velas de Bybit porque Binance falló, o modelo entrenado con poca
   historia). Si hay que entrenar un modelo avisa por stderr (así el JSON queda
   limpio); en ``--offline`` usa modelos sintéticos
   (``regime.synthetic_models_dir``). Si el régimen falla, muestra "Régimen no
   disponible: <motivo>" y el resto sale igual;
3. indicadores por timeframe (1d, 4h, 1h, 15m, 5m) con la última vela cerrada;
4. soportes y resistencias de ``trade_plan.stop_timeframes`` y
   máximo/mínimo del día anterior;
5. rango diario típico (ATR diario en % del precio);
6. derivados: funding, interés abierto y ratio long/short;
7. sentimiento: Fear & Greed, dominancia de BTC y titulares;
8. score de confluencia (``app.scoring``): total, pista de dirección, una
   barra de -100 a +100 con los umbrales, y los puntos de cada componente con
   su peso y el porqué;
9. plan de operación (``app.risk.plan.build_trade_plan``, el mismo que va a
   usar la app): cartel grande LONG / SHORT / NO OPERAR, una oración que lo
   resume, la tabla de niveles (entrada a mercado y alternativa con orden
   límite, stop, TP1/TP2/TP3 con distancia, R, ganancia neta y R:R con
   comisiones), margen, tamaño, apalancamiento resultante, liquidación contra
   el stop, pérdida máxima, comisiones (en USD y como % del riesgo y del
   objetivo), funding estimado y stop por tiempo en ``app.user_timezone``; de
   dónde sale cada nivel (con una línea que explica R, R:R y ATR); los
   ajustes con su motivo; las reglas que no se cumplen (✗) con la sugerencia;
   los avisos y un resumen de la entrada alternativa (con sus propios ajustes,
   liquidación, comisiones y stop por tiempo). Usa el régimen de 1h si está
   disponible. Si no se opera pero los niveles se pudieron calcular, se
   muestran como "referencia, no operar".

Sale con código 1 y un mensaje claro si no se pueden conseguir los datos
esenciales (por ejemplo, Binance y Bybit caídos, o un símbolo que no existe),
con 2 si los argumentos o la configuración no son válidos (incluido un YAML mal
escrito) y con 130 si se cancela con Ctrl+C (sale enseguida, también durante el
entrenamiento del régimen, que no se guarda). Siempre cierra el cliente HTTP.
Nunca muestra las keys de ``.env``. Los textos de terceros (titulares, fuentes,
avisos) se imprimen sin caracteres de control, así no pueden meter secuencias de
escape en la terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from app.config import DEFAULT_CONFIG_PATH, AppSettings, get_secrets, load_settings
from app.data.cache import TTLCache
from app.data.snapshot import (
    EssentialDataError,
    MarketSnapshot,
    build_offline_snapshot,
    build_snapshot,
    funding_summary,
    long_short_change_pct,
)
from app.data.synthetic import DEFAULT_SEED, SyntheticMarketData
from app.exchanges.base import TIMEFRAME_MINUTES
from app.exchanges.registry import build_market_data, build_nexo_price_source
from app.http_client import make_client
from app.indicators.levels import Level, SRResult
from app.indicators.pipeline import MarketAnalysis, TimeframeSummary, analyze_all
from app.regime.service import (
    RegimeReport,
    get_market_regimes,
    regime_history_days,
    regimes_to_dict,
)
from app.risk.plan import build_trade_plan, plan_inputs_from_analysis
from app.risk.position import MIN_EXCHANGE_LEVERAGE
from app.risk.profiles import EffectiveProfile, get_profile
from app.risk.rules import fmt_leverage
from app.risk.rules import price_decimals as _plan_price_decimals
from app.risk.sizing import fraction_to_pct
from app.scoring import compute_score, inputs_from_analysis

OFFLINE_BANNER = "DATOS SINTÉTICOS (sin red): los números no son reales"
SCORE_TITLE = "Score de confluencia"
PLAN_TITLE = "Plan de operación"
REGIME_TITLE = "Régimen de mercado"
REGIME_UNAVAILABLE = "Régimen no disponible"
DASH = "—"
SOURCE_LABELS = {
    "binance": "Binance",
    "bybit": "Bybit, respaldo de Binance",
    "sintetico": "Binance sintético",
}
WIDTH = 100
TITLE_WIDTH = 90

EXIT_OK = 0
EXIT_DATA = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
CANCELLED = "Cancelado por el usuario."


# --- formato de números (estilo argentino: 1.234,56) ---------------------------

_SWAP = str.maketrans({",": ".", ".": ","})


def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt_num(x: Any, decimals: int = 2, *, sign: bool = False) -> str:
    """``1234.5`` → ``"1.234,50"``; ``None``/``NaN`` → ``"—"``."""
    v = _finite(x)
    if v is None:
        return DASH
    if v == 0:
        v = 0.0  # evita "-0,00"
    text = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    return text.translate(_SWAP)


def price_decimals(reference: Any) -> int:
    """Decimales para mostrar un precio según su tamaño (BTC 2, XRP 4).

    Es la misma regla que usa el plan (``risk.rules.price_decimals``), así el
    titular y la tabla muestran los precios con la misma precisión.
    """
    v = _finite(reference)
    return 2 if v is None else _plan_price_decimals(v)


def fmt_pct(x: Any, decimals: int = 2, *, sign: bool = False) -> str:
    text = fmt_num(x, decimals, sign=sign)
    return text if text == DASH else f"{text} %"


def fmt_usd(x: Any, decimals: int = 2) -> str:
    text = fmt_num(x, decimals)
    return text if text == DASH else f"{text} USD"


def fmt_compact_usd(x: Any) -> str:
    """Montos grandes en palabras: ``3.35e12`` → ``"3,35 billones USD"``."""
    v = _finite(x)
    if v is None:
        return DASH
    for size, word in ((1e12, "billones"), (1e9, "mil millones"), (1e6, "millones")):
        if abs(v) >= size:
            return f"{fmt_num(v / size, 2)} {word} USD"
    return fmt_usd(v)


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.astimezone(UTC).isoformat()


def _parse_iso(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def fmt_time(text: Any, tz: ZoneInfo, *, with_date: bool = True, seconds: bool = False) -> str:
    dt = _parse_iso(text)
    if dt is None:
        return DASH
    local = dt.astimezone(tz)
    pattern = ("%d/%m " if with_date else "") + ("%H:%M:%S" if seconds else "%H:%M")
    return local.strftime(pattern)


# --- estilos y tablas ------------------------------------------------------------

_ANSI = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "alert": "1;31",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_TREND_COLOR = {"alcista": "green", "bajista": "red", "lateral": "yellow"}
_POSITION_COLOR = {"arriba": "green", "abajo": "red", "igual": "yellow"}


@dataclass(frozen=True)
class Style:
    """Colores ANSI solo si ``enabled`` (salida a una terminal)."""

    enabled: bool = False

    def paint(self, text: str, color: str | None) -> str:
        if not self.enabled or not color or not text:
            return text
        return f"\x1b[{_ANSI[color]}m{text}\x1b[0m"


@dataclass(frozen=True)
class Cell:
    text: str
    color: str | None = None


def _cell(value: Cell | str) -> Cell:
    return value if isinstance(value, Cell) else Cell(value)


def render_table(
    rows: Sequence[Sequence[Cell | str]],
    style: Style,
    *,
    header: Sequence[str] | None = None,
    aligns: str | None = None,
    indent: int = 2,
) -> list[str]:
    """Tabla alineada. ``aligns``: una letra por columna (``<`` izquierda, ``>`` derecha)."""
    table = [[_cell(c) for c in row] for row in rows]
    head = [Cell(h, "bold") for h in header] if header else None
    all_rows = ([head] if head else []) + table
    if not all_rows:
        return []
    ncols = max(len(r) for r in all_rows)
    widths = [0] * ncols
    for r in all_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c.text))
    aligns = (aligns or "<" + ">" * (ncols - 1)).ljust(ncols, ">")
    out = []
    for r in all_rows:
        parts = []
        for i in range(ncols):
            c = r[i] if i < len(r) else Cell("")
            if aligns[i] == "<":
                # La última columna no se rellena: no deja espacios al final de la línea.
                padded = c.text if i == ncols - 1 else c.text.ljust(widths[i])
            else:
                padded = c.text.rjust(widths[i])
            parts.append(style.paint(padded, c.color))
        out.append((" " * indent + "  ".join(parts)).rstrip())
    return out


def _section(title: str, style: Style) -> list[str]:
    line = f"━━ {title} " + "━" * max(3, WIDTH - len(title) - 4)
    return ["", style.paint(line, "bold")]


def _banner(text: str, style: Style) -> list[str]:
    inner = f"  {text}  "
    top = "╔" + "═" * len(inner) + "╗"
    bottom = "╚" + "═" * len(inner) + "╝"
    return [
        style.paint(top, "alert"),
        style.paint(f"║{inner}║", "alert"),
        style.paint(bottom, "alert"),
    ]


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def printable(text: Any) -> str:
    """Texto de terceros sin caracteres de control (ESC, BEL...) para la terminal."""
    return " ".join(_CONTROL_CHARS.sub(" ", str(text)).split())


# --- reporte (dict serializable a JSON) ------------------------------------------


def _clean(obj: Any) -> Any:
    """Convierte a tipos JSON (NaN/inf → None, datetime → ISO, claves a str)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, datetime):
        return _iso(obj)
    v = _finite(obj)
    return v if v is not None else str(obj)


def _level_dict(level: Level) -> dict[str, Any]:
    return {
        "price": level.price,
        "kind": level.kind,
        "distance_pct": level.distance_pct,
        "touches": level.touches,
        "sources": sorted(level.sources),
        "last_touch": _iso(level.last_touch),
    }


def _levels_dict(levels: SRResult | None) -> dict[str, Any] | None:
    if levels is None:
        return None
    return {
        "reference_price": levels.reference_price,
        "supports": [_level_dict(lv) for lv in levels.supports],
        "resistances": [_level_dict(lv) for lv in levels.resistances],
        "prev_day": None if levels.prev_day is None else levels.prev_day.isoformat(),
        "prev_day_high": levels.prev_day_high,
        "prev_day_low": levels.prev_day_low,
        "cluster_tolerance": levels.cluster_tolerance,
    }


def _summary_dict(s: TimeframeSummary) -> dict[str, Any]:
    minutes = TIMEFRAME_MINUTES.get(s.timeframe, 0)
    close_time = None
    if s.candle_time is not None:
        close_time = s.candle_time.timestamp() + minutes * 60
    return {
        "candle_open_time": _iso(s.candle_time),
        "candle_close_time": None
        if close_time is None
        else _iso(datetime.fromtimestamp(close_time, tz=UTC)),
        "candle_closed": s.candle_closed,
        "candles_used": s.candles_used,
        "close": s.close,
        "ema": {str(p): v for p, v in s.ema.items()},
        "has_vwap": s.has_vwap,
        "vwap": s.vwap,
        "rsi": s.rsi,
        "rsi_zone": s.rsi_zone,
        "macd": s.macd,
        "macd_signal": s.macd_signal,
        "macd_hist": s.macd_hist,
        "atr": s.atr,
        "atr_pct": s.atr_pct,
        "bb_upper": s.bb_upper,
        "bb_mid": s.bb_mid,
        "bb_lower": s.bb_lower,
        "bb_width_pct": None if s.bb_width is None else s.bb_width * 100.0,
        "rel_volume": s.rel_volume,
        "trend": s.trend,
        "trend_emas": list(s.trend_emas) if s.trend_emas else None,
        "price_vs_ema": {str(p): v for p, v in s.price_vs_ema.items()},
        "price_vs_vwap": s.price_vs_vwap,
        "flags": list(s.flags),
    }


def build_report(
    snapshot: MarketSnapshot,
    analysis: MarketAnalysis,
    settings: AppSettings,
    profile: str | EffectiveProfile | None,
    regime: dict[str, Any] | None = None,
    *,
    regime_report: RegimeReport | None = None,
    trades_today: int = 0,
    pnl_today_usd: float = 0.0,
    high_volatility_event: bool = False,
    event_description: str | None = None,
) -> dict[str, Any]:
    """Todo el contenido de la salida, en un dict serializable a JSON.

    - ``profile``: nombre del perfil (``None`` = el de config) o el
      ``EffectiveProfile`` ya armado (con el tope de ``--max-leverage``).
    - ``regime``: lo que devuelve ``regimes_to_dict`` (o ``None`` si no se calculó).
    - ``regime_report``: el ``RegimeReport`` de 1h que usa el plan (``None`` si
      no está disponible: el plan lo toma como confianza baja).
    - ``trades_today``, ``pnl_today_usd``, ``high_volatility_event`` y
      ``event_description``: estado del día para las reglas del plan.

    El score (``"score"``) y el plan (``"plan"``) se calculan acá con
    ``scoring.compute_score`` y ``risk.plan.build_trade_plan``, con la hora de
    la foto del mercado como "ahora".
    """
    eff = profile if isinstance(profile, EffectiveProfile) else get_profile(profile, settings)
    score = compute_score(inputs_from_analysis(analysis, snapshot, settings), settings)
    plan_inputs = plan_inputs_from_analysis(
        analysis,
        snapshot,
        score,
        regime_report,
        eff,
        settings,
        trades_today=trades_today,
        pnl_today_usd=pnl_today_usd,
        high_volatility_event=high_volatility_event,
        event_description=event_description,
    )
    plan = build_trade_plan(plan_inputs, settings, now=snapshot.created_at)
    ex = snapshot.execution_price
    ticker = snapshot.ticker
    reference = ticker.price if ticker else (ex.binance_price if ex else None)
    nexo_price = ex.price if ex and ex.source in ("manual_nexo", "nexo") else None

    lookback = settings.scoring.derivatives.oi_lookback_hours
    funding = snapshot.funding
    fsum = funding_summary(snapshot.funding_history)
    oi = snapshot.open_interest
    oi_value = oi.value_usd if oi and oi.value_usd is not None else None
    oi_value_estimated = False
    if oi is not None and oi_value is None and reference:
        oi_value, oi_value_estimated = oi.contracts * reference, True
    ls_points = snapshot.long_short.points if snapshot.long_short else []
    ls_last = ls_points[-1] if ls_points else None
    fg = snapshot.fear_greed
    gm = snapshot.global_market
    news = snapshot.news

    stop_tfs = [tf for tf in settings.trade_plan.stop_timeframes if tf in analysis.timeframes]
    report = {
        "symbol": snapshot.symbol,
        "quote": settings.symbols.quote,
        "offline": snapshot.offline,
        "banner": OFFLINE_BANNER if snapshot.offline else None,
        "generated_at": _iso(snapshot.created_at),
        "user_timezone": settings.app.user_timezone,
        "profile": eff.name,
        "closed_candles_only": settings.timeframes.closed_candles_only,
        "sources": [
            {
                "name": s.name,
                "source": s.source,
                "fetched_at": _iso(s.fetched_at),
                "from_cache": s.from_cache,
                "stale": s.stale,
                "fallback": s.fallback,
                "ok": s.ok,
                "warning": s.warning,
            }
            for s in snapshot.sources
        ],
        "warnings": list(snapshot.warnings),
        "prices": {
            "reference_price": reference,
            "reference_source": ticker.meta.source if ticker else None,
            "nexo_price": nexo_price,
            "nexo_source": ex.source if nexo_price is not None and ex else None,
            "diff_pct": ex.diff_pct if ex else None,
            "change_24h_pct": ticker.change_24h_pct if ticker else None,
            "high_24h": ticker.high_24h if ticker else None,
            "low_24h": ticker.low_24h if ticker else None,
            "execution_price": snapshot.current_price,
            "execution_source": snapshot.price_source,
        },
        "regime": regime,
        "timeframes": {tf: _summary_dict(ta.summary) for tf, ta in analysis.timeframes.items()},
        "levels": {tf: _levels_dict(analysis.timeframes[tf].summary.levels) for tf in stop_tfs},
        "daily_range": {
            "atr_period": settings.indicators.atr_period,
            "atr_pct": analysis.daily_range_pct,
            "atr_usd": None
            if analysis.daily_range_pct is None
            else analysis.daily_range_pct / 100.0 * snapshot.current_price,
        },
        "analysis_flags": list(analysis.flags),
        "derivatives": {
            "funding": None
            if funding is None
            else {
                "rate_pct": fraction_to_pct(funding.rate),
                "interval_hours": funding.interval_hours,
                "next_funding_time": _iso(funding.next_funding_time),
                "mark_price": funding.mark_price,
                "source": funding.meta.source,
            },
            "funding_history": None
            if fsum is None
            else {
                "count": fsum.count,
                "start": _iso(fsum.start),
                "end": _iso(fsum.end),
                "mean_pct": fraction_to_pct(fsum.mean),
                "min_pct": fraction_to_pct(fsum.min),
                "max_pct": fraction_to_pct(fsum.max),
                "total_pct": fraction_to_pct(fsum.total),
                "positive_share_pct": fsum.positive_share * 100.0,
            },
            "open_interest": None
            if oi is None
            else {
                "contracts": oi.contracts,
                "value_usd": oi_value,
                "value_estimated": oi_value_estimated,
                "change_pct": snapshot.oi_change_24h_pct,
                "lookback_hours": lookback,
                "source": oi.meta.source,
            },
            "long_short": None
            if ls_last is None
            else {
                "ratio": ls_last.ratio,
                "long_pct": None if ls_last.long_pct is None else ls_last.long_pct * 100.0,
                "short_pct": None if ls_last.short_pct is None else ls_last.short_pct * 100.0,
                "change_pct": long_short_change_pct(snapshot.long_short, lookback),
                "lookback_hours": lookback,
            },
        },
        "sentiment": {
            "fear_greed": None
            if fg is None
            else {
                "value": fg.value,
                "label_es": fg.classification_es,
                "label_en": fg.classification_en,
            },
            "btc_dominance_pct": gm.btc_dominance_pct if gm else None,
            "eth_dominance_pct": gm.eth_dominance_pct if gm else None,
            "total_market_cap_usd": gm.total_market_cap_usd if gm else None,
            "market_cap_change_24h_pct": gm.market_cap_change_24h_pct if gm else None,
            "headlines": []
            if news is None
            else [
                {
                    "published_at": _iso(h.published_at),
                    "source": h.source,
                    "title": h.title,
                    "url": h.url,
                    "is_macro": h.is_macro,
                }
                for h in news.headlines[: settings.data_sources.news_max_headlines]
            ],
        },
        "score": score.to_dict(),
        "plan": plan.to_dict(),
    }
    return _clean(report)


# --- salida de texto ---------------------------------------------------------------


def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _kv(rows: Sequence[tuple[str, Cell | str]], style: Style, indent: int = 2) -> list[str]:
    return render_table([[k, v] for k, v in rows], style, aligns="<<", indent=indent)


def _signed_color(x: Any) -> str | None:
    v = _finite(x)
    if v is None or v == 0:
        return None
    return "green" if v > 0 else "red"


def _render_header(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    lines: list[str] = []
    if r["offline"]:
        lines += _banner(OFFLINE_BANNER, style)
    title = f"Crypto Future Analyzer · {r['symbol']}/{r['quote']}"
    lines.append(style.paint(title, "bold"))
    prof = (r.get("plan") or {}).get("profile") or {}
    profile_text = printable(r["profile"])
    if prof:
        profile_text += f" (tope de apalancamiento {fmt_leverage(prof['max_leverage'])}"
        if prof.get("leverage_overridden"):
            profile_text += (
                f", cambiado a mano; el perfil trae {fmt_leverage(prof['profile_max_leverage'])}"
            )
        profile_text += ")"
    lines += _kv(
        [
            ("Hora", f"{fmt_time(r['generated_at'], tz, seconds=True)} ({r['user_timezone']})"),
            ("Perfil de riesgo", profile_text),
            (
                "Velas",
                "indicadores con la última vela cerrada"
                if r["closed_candles_only"]
                else "indicadores con la vela que se está formando",
            ),
        ],
        style,
        indent=0,
    )
    if prof.get("red_warning_text"):
        lines.append(style.paint(printable(prof["red_warning_text"]), "alert"))

    lines += _section("Fuentes de datos", style)
    rows = []
    for s in r["sources"]:
        if not s["ok"]:
            state = Cell("falló", "red")
        elif s["stale"]:
            state = Cell("desactualizado", "yellow")
        elif s["fallback"]:
            state = Cell("respaldo", "yellow")
        elif s["from_cache"]:
            state = Cell("caché")
        else:
            state = Cell("ok", "green")
        rows.append(
            [
                printable(s["name"]),
                printable(s["source"] or DASH),
                fmt_time(s["fetched_at"], tz, seconds=True),
                state,
            ]
        )
    lines += render_table(
        rows, style, header=["Dato", "Fuente", "Actualizado", "Estado"], aligns="<<<<"
    )
    if r["warnings"]:
        lines.append("")
        lines.append(style.paint("  Avisos:", "yellow"))
        for w in r["warnings"]:
            lines.append(style.paint(f"  · {printable(w)}", "yellow"))
    return lines


def _render_prices(r: dict[str, Any], style: Style) -> list[str]:
    p = r["prices"]
    d = price_decimals(p["execution_price"])
    lines = _section("Precios", style)
    ref_source = p["reference_source"]
    ref_label = SOURCE_LABELS.get(ref_source, ref_source) if ref_source else None
    used = {
        "manual_nexo": "Nexo, cargado a mano",
        "nexo": "Nexo",
        "binance": ref_label or "Binance",
        "ultimo_cierre": "último cierre de las velas",
    }.get(p["execution_source"], p["execution_source"])
    if p["nexo_price"] is None:
        instead = (
            "el último cierre de las velas" if p["execution_source"] == "ultimo_cierre" else used
        )
        nexo = Cell(f"no disponible: se usa {instead}", "yellow")
    else:
        origin = "cargado a mano" if p["nexo_source"] == "manual_nexo" else "conector"
        nexo = f"{fmt_num(p['nexo_price'], d)} ({origin})"
    if ref_label is None or p["reference_price"] is None:
        reference: Cell | str = Cell("no disponible", "yellow")
    else:
        reference = f"{fmt_num(p['reference_price'], d)} ({ref_label})"
    rows: list[tuple[str, Cell | str]] = [
        ("Precio de referencia", reference),
        ("Nexo", nexo),
        (
            "Diferencia Nexo vs referencia",
            Cell(fmt_pct(p["diff_pct"], sign=True), _signed_color(p["diff_pct"])),
        ),
        (
            "Cambio 24 h",
            Cell(fmt_pct(p["change_24h_pct"], sign=True), _signed_color(p["change_24h_pct"])),
        ),
        ("Máximo / mínimo 24 h", f"{fmt_num(p['high_24h'], d)} / {fmt_num(p['low_24h'], d)}"),
        (
            "Precio usado en los cálculos",
            Cell(f"{fmt_num(p['execution_price'], d)} ({used})", "bold"),
        ),
    ]
    return lines + _kv(rows, style)


_CATEGORY_COLOR = {
    "crash": "alert",
    "bajista": "red",
    "neutral": "yellow",
    "alcista": "green",
    "euforia": "green",
}
# Avisos del régimen que ya se muestran en sus propias filas. ``data_source``
# solo se calla si las velas son sintéticas (ya lo dice el cartel); si vinieron
# de Bybit porque Binance falló, se muestra.
_REGIME_CODES_SHOWN = {"pending", "low_confidence", "unstable", "unconfirmed"}
_SYNTHETIC_SOURCE = "sintetico"


def fmt_duration(bars: Any, minutes: Any, decimals: int = 0) -> str:
    """Duración de ``bars`` velas de ``minutes`` minutos: ``"14 h"`` o ``"3 días"``."""
    b, m = _finite(bars), _finite(minutes)
    if b is None or m is None:
        return DASH
    if m < 1440:
        return f"{fmt_num(b * m / 60.0, decimals)} h"
    days = b * m / 1440.0
    text = fmt_num(days, decimals)
    return f"{text} {'día' if text in ('1', '1,0') else 'días'}"


def render_regime_states(
    rows: Sequence[dict[str, Any]], minutes: Any, style: Style, *, indent: int = 4
) -> list[str]:
    """Tabla por régimen (de ``labels.regime_stats_table``), del más bajista al más alcista."""
    table = []
    for row in rows:
        table.append(
            [
                Cell(printable(row["name"]), _CATEGORY_COLOR.get(row["category"])),
                printable(row["category"]),
                Cell(
                    fmt_pct(row["mean_return_pct"], 3, sign=True),
                    _signed_color(row["mean_return_pct"]),
                ),
                fmt_pct(row["mean_volatility_pct"], 2),
                fmt_duration(row["expected_duration_bars"], minutes, 1),
                fmt_pct(row["occupancy_pct"], 1),
                Cell("← ahora", "bold") if row.get("is_current") else "",
            ]
        )
    return render_table(
        table,
        style,
        header=[
            "Régimen",
            "Tipo",
            "Retorno medio por vela",
            "Volatilidad media",
            "Duración típica",
            "% del tiempo",
            "",
        ],
        aligns="<<>>>><",
        indent=indent,
    )


def _hours_text(hours: float) -> str:
    return f"{fmt_num(hours / 24, 1)} días" if hours >= 72 else f"{fmt_num(hours, 0)} h"


def _history_summary_text(hs: dict[str, Any], g: dict[str, Any]) -> str:
    """Texto de la fila "Últimos N días": cambios de régimen confirmados y tramos completos.

    El primer tramo y el actual quedan cortados por la ventana, así que la
    duración solo se informa con tramos completos (2 cambios o más). Con 0
    cambios el régimen actual viene de antes de la ventana (ver "Empezó"); con 1,
    ese cambio es el que dio inicio al actual.
    """
    n = hs["regime_changes"]
    if n == 0:
        text = "ningún cambio de régimen confirmado"
        if g.get("confirmed"):
            text += " (el actual ya venía de antes)"
    elif n == 1:
        text = "1 cambio de régimen confirmado (el que dio inicio al actual)"
    else:
        complete = hs["completed_stretches"]
        avg = _hours_text(hs["avg_regime_hours"])
        if complete == 1:
            detail = f"el tramo completo entre los dos duró ~{avg}"
        else:
            detail = f"los {complete} tramos completos duraron ~{avg} en promedio"
        text = f"{n} cambios de régimen confirmados ({detail})"
    return (
        f"{text}; inestable el {fmt_num(hs['unstable_pct'], 0)} % del tiempo; confianza baja "
        f"el {fmt_num(hs['low_confidence_pct'], 0)} %"
    )


def _render_one_regime(tf: str, g: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    role = "contexto diario" if g.get("is_context") else "el que se usa para operar"
    lines = ["", style.paint(f"  {tf} · {role}", "bold")]
    if not g.get("available"):
        reason = printable(g.get("reason") or "motivo desconocido")
        lines.append(style.paint(f"    {REGIME_UNAVAILABLE}: {reason}", "yellow"))
        return lines
    minutes = g["timeframe_minutes"]
    state = "confirmado" if g["confirmed"] else "todavía sin confirmar"
    current = Cell(
        f"{printable(g['name'])} (tipo {printable(g['category'])}), {state}",
        _CATEGORY_COLOR.get(g["category"]),
    )
    # Confianza = probabilidad del estado más probable en la última vela. Si ese
    # estado no es el régimen actual (por ejemplo, hay un cambio en espera), se
    # dice cuál es, así el número no se confunde con el del régimen actual.
    top = g["top_state"]
    notes = []
    if top["name"] != g["name"]:
        notes.append(f"lo más probable en la última vela es {printable(top['name'])}")
    if g["low_confidence"]:
        threshold = fmt_pct((_finite(g["low_confidence_threshold"]) or 0) * 100, 0)
        notes.append(f"baja: menos de {threshold}")
    conf_text = fmt_pct(g["confidence_pct"], 0)
    if notes:
        conf_text += f" ({'; '.join(notes)})"
    confidence = Cell(conf_text, "yellow" if g["low_confidence"] else None)
    ago = fmt_duration(g["age_bars"], minutes)
    started = "hace más de " if g["age_is_lower_bound"] else "hace "
    # Las velas diarias arrancan a las 00:00 UTC: se muestra la fecha UTC.
    when = (
        fmt_time(g["started_at"], ZoneInfo("UTC")).split()[0]
        if (_finite(minutes) or 0) >= 1440
        else fmt_time(g["started_at"], tz)
    )
    started += f"{ago} ({when})"
    rows: list[tuple[str, Cell | str]] = [
        ("Régimen actual", current),
        ("Confianza", confidence),
        ("Empezó", started),
    ]
    pending = g["pending"]
    if pending is not None:
        rows.append(
            (
                "Posible cambio",
                Cell(
                    f"a {printable(pending['name'])}: lleva {pending['bars']} de "
                    f"{pending['needed']} velas seguidas para confirmarse",
                    "yellow",
                ),
            )
        )
    window, limit = g["instability_window"], g["instability_max_changes"]
    changes = g["changes_in_window"]
    # "Estabilidad" cuenta cambios del tipo (o estado) MÁS PROBABLE vela a vela;
    # "Últimos N días", cambios del régimen CONFIRMADO. Son cosas distintas (un
    # tipo que aparece una o dos velas cuenta acá pero no se confirma), así que
    # se dicen con palabras distintas para que no parezca una contradicción.
    kind = "tipo" if g.get("change_level", "category") == "category" else "estado"
    moved = f"el {kind} más probable cambió {changes} {'vez' if changes == 1 else 'veces'}"
    if g["unstable"]:
        stability = Cell(
            f"RÉGIMEN INESTABLE: {moved} en las últimas {window} velas (más de "
            f"{limit}); conviene operar más chico",
            "alert",
        )
    else:
        stability = Cell(
            f"estable: {moved} en las últimas {window} velas (inestable si son más de {limit})"
        )
    rows.append(("Estabilidad", stability))
    hs = g.get("history_summary")
    if hs:
        rows.append((f"Últimos {fmt_num(hs['days'], 0)} días", _history_summary_text(hs, g)))
    m = g["model"]
    model_text = (
        f"del {fmt_time(m['trained_at'], tz)}, {m['n_states']} estados (elegidos por BIC), "
        f"{fmt_num(m['n_samples'], 0)} velas"
    )
    if m["synthetic"]:
        model_text += ", datos sintéticos"
    rows.append(("Modelo", model_text))
    lines += _kv(rows, style, indent=4)
    lines.append("")
    lines += render_regime_states(g["states"], minutes, style, indent=4)
    synthetic = g.get("data_source") == _SYNTHETIC_SOURCE
    others = [
        w
        for w, code in zip(g["warnings"], g["warning_codes"], strict=False)
        if code not in _REGIME_CODES_SHOWN and not (code == "data_source" and synthetic)
    ]
    for w in others:
        lines.append(style.paint(f"    · {printable(w)}", "yellow"))
    return lines


def _render_regime(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    regimes = r.get("regime")
    if regimes is None:
        return []
    lines = _section(REGIME_TITLE, style)
    note = "  Qué tipo de mercado hay según un modelo estadístico (HMM). No predice el precio."
    lines.append(style.paint(note, "dim"))
    for tf, g in regimes.items():
        lines += _render_one_regime(tf, g, style, tz)
    return lines


def _render_timeframes(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    tfs = sorted(r["timeframes"], key=lambda tf: -TIMEFRAME_MINUTES.get(tf, 0))
    lines = _section("Indicadores por timeframe", style)
    if not tfs:
        return [*lines, "  Sin velas."]
    d = price_decimals(r["prices"]["execution_price"])
    S = [r["timeframes"][tf] for tf in tfs]
    ema_periods = sorted({int(p) for s in S for p in s["ema"]})

    def row(label: str, fn: Callable[[dict[str, Any]], Cell | str]) -> list[Cell | str]:
        return [label, *[fn(s) for s in S]]

    def pos(value: str | None) -> Cell:
        return Cell(value or DASH, _POSITION_COLOR.get(value or ""))

    rows: list[list[Cell | str]] = [
        row("Vela usada (cierre)", lambda s: fmt_time(s["candle_close_time"], tz)),
        row("Cierre", lambda s: fmt_num(s["close"], d)),
    ]
    for p in ema_periods:
        rows.append(row(f"EMA {p}", lambda s, p=p: fmt_num(s["ema"].get(str(p)), d)))
    rows += [
        row("VWAP diario", lambda s: fmt_num(s["vwap"], d) if s["has_vwap"] else DASH),
        row(
            "RSI",
            lambda s: Cell(
                fmt_num(s["rsi"], 1),
                {"sobrecompra": "red", "sobreventa": "green"}.get(s["rsi_zone"] or ""),
            ),
        ),
        row("Zona RSI", lambda s: s["rsi_zone"] or DASH),
        row("MACD línea", lambda s: fmt_num(s["macd"], d)),
        row("MACD señal", lambda s: fmt_num(s["macd_signal"], d)),
        row(
            "MACD histograma",
            lambda s: Cell(fmt_num(s["macd_hist"], d, sign=True), _signed_color(s["macd_hist"])),
        ),
        row("ATR", lambda s: fmt_num(s["atr"], d)),
        row("ATR % del precio", lambda s: fmt_pct(s["atr_pct"])),
        row("Bollinger superior", lambda s: fmt_num(s["bb_upper"], d)),
        row("Bollinger media", lambda s: fmt_num(s["bb_mid"], d)),
        row("Bollinger inferior", lambda s: fmt_num(s["bb_lower"], d)),
        row("Bollinger ancho", lambda s: fmt_pct(s["bb_width_pct"])),
        row(
            "Volumen relativo",
            lambda s: DASH if s["rel_volume"] is None else f"{fmt_num(s['rel_volume'], 2)}x",
        ),
        row("Tendencia", lambda s: Cell(s["trend"], _TREND_COLOR.get(s["trend"]))),
    ]
    for p in ema_periods:
        rows.append(row(f"Precio vs EMA {p}", lambda s, p=p: pos(s["price_vs_ema"].get(str(p)))))
    rows.append(row("Precio vs VWAP", lambda s: pos(s["price_vs_vwap"]) if s["has_vwap"] else DASH))
    lines += render_table(rows, style, header=["", *tfs])
    trend_emas = next((s["trend_emas"] for s in S if s["trend_emas"]), None)
    if trend_emas:
        lines.append(
            style.paint(
                f"  Tendencia: alcista si precio > EMA {trend_emas[0]} > EMA {trend_emas[1]}, "
                "bajista si es al revés, lateral en otro caso.",
                "dim",
            )
        )
    flags = [(tf, f) for tf, s in zip(tfs, S, strict=True) for f in s["flags"]]
    if flags:
        lines.append(style.paint("  Datos insuficientes:", "yellow"))
        lines += [style.paint(f"  · {tf}: {f}", "yellow") for tf, f in flags]
    return lines


def _render_levels(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    lines = _section("Soportes y resistencias", style)
    price = r["prices"]["execution_price"]
    d = price_decimals(price)
    if not r["levels"]:
        return [*lines, "  Sin datos."]
    for tf, lv in r["levels"].items():
        lines.append(style.paint(f"  {tf}", "bold"))
        if lv is None:
            lines.append("    Sin velas cerradas para calcular niveles.")
            continue
        rows: list[list[Cell | str]] = []
        for level in sorted(lv["resistances"], key=lambda x: -x["price"]):
            rows.append(
                [
                    Cell("Resistencia", "red"),
                    fmt_num(level["price"], d),
                    fmt_pct(level["distance_pct"], sign=True),
                    str(level["touches"]),
                ]
            )
        rows.append([Cell("Precio actual", "bold"), Cell(fmt_num(price, d), "bold"), "", ""])
        for level in lv["supports"]:
            rows.append(
                [
                    Cell("Soporte", "green"),
                    fmt_num(level["price"], d),
                    fmt_pct(level["distance_pct"], sign=True),
                    str(level["touches"]),
                ]
            )
        lines += render_table(
            rows, style, header=["Nivel", "Precio", "Distancia", "Toques"], aligns="<>>>", indent=4
        )
        if not lv["resistances"]:
            lines.append("    (sin resistencias detectadas arriba del precio)")
        if not lv["supports"]:
            lines.append("    (sin soportes detectados debajo del precio)")
    first = next((lv for lv in r["levels"].values() if lv), None)
    if first is not None:
        day = first["prev_day"]
        day_txt = f" ({day[8:10]}/{day[5:7]})" if day else ""

        def dist(x: Any) -> str:
            v = _finite(x)
            return DASH if v is None or not price else fmt_pct((v - price) / price * 100, sign=True)

        lines += _kv(
            [
                (
                    f"Máximo del día anterior{day_txt}",
                    f"{fmt_num(first['prev_day_high'], d)} ({dist(first['prev_day_high'])})",
                ),
                (
                    f"Mínimo del día anterior{day_txt}",
                    f"{fmt_num(first['prev_day_low'], d)} ({dist(first['prev_day_low'])})",
                ),
            ],
            style,
        )
    dr = r["daily_range"]
    lines += _section("Rango diario típico", style)
    if dr["atr_pct"] is None:
        lines.append("  No disponible (faltan velas diarias cerradas).")
    else:
        lines.append(
            f"  ATR {dr['atr_period']} diario: {fmt_pct(dr['atr_pct'])} del precio "
            f"(≈ {fmt_num(dr['atr_usd'], d)} USD de movimiento típico por día)"
        )
    return lines


def _render_derivatives(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    lines = _section("Derivados", style)
    dv = r["derivatives"]
    rows: list[tuple[str, Cell | str]] = []

    def spct(x: Any, decimals: int = 4) -> str:
        return fmt_pct(x, decimals, sign=True)

    f = dv["funding"]
    if f is None:
        rows.append(("Funding actual", DASH))
    else:
        rate = f["rate_pct"] or 0
        if rate > 0:
            who = "los long le pagan a los short"
        elif rate < 0:
            who = "los short le pagan a los long"
        else:
            who = "nadie paga"
        every = fmt_num(f["interval_hours"], 0)
        text = f"{spct(f['rate_pct'])} cada {every} h ({who})"
        rows.append(("Funding actual", Cell(text, _signed_color(rate))))
        rows.append(("Próximo cobro de funding", fmt_time(f["next_funding_time"], tz)))
    fh = dv["funding_history"]
    if fh is None:
        rows.append(("Funding (historial)", DASH))
    else:
        stats = (
            f"promedio {spct(fh['mean_pct'])} · mín {spct(fh['min_pct'])}"
            f" · máx {spct(fh['max_pct'])}"
        )
        total = (
            f"acumulado {spct(fh['total_pct'], 3)}"
            f" · positivos {fmt_pct(fh['positive_share_pct'], 0)}"
            f" · desde {fmt_time(fh['start'], tz)}"
        )
        rows += [(f"Funding últimos {fh['count']} cobros", stats), ("", total)]
    oi = dv["open_interest"]
    if oi is None:
        rows.append(("Interés abierto", DASH))
    else:
        approx = "≈ " if oi["value_estimated"] else ""
        value = f"{approx}{fmt_compact_usd(oi['value_usd'])}"
        rows.append(("Interés abierto", f"{fmt_num(oi['contracts'], 0)} {r['symbol']} ({value})"))
        change = Cell(fmt_pct(oi["change_pct"], sign=True), _signed_color(oi["change_pct"]))
        rows.append((f"Cambio del OI en {oi['lookback_hours']} h", change))
    ls = dv["long_short"]
    if ls is None:
        rows.append(("Ratio long/short", DASH))
    else:
        split = f"{fmt_pct(ls['long_pct'], 1)} long / {fmt_pct(ls['short_pct'], 1)} short"
        rows.append(("Ratio long/short (cuentas)", f"{fmt_num(ls['ratio'], 2)} ({split})"))
        change = Cell(fmt_pct(ls["change_pct"], sign=True), _signed_color(ls["change_pct"]))
        rows.append((f"Cambio del ratio en {ls['lookback_hours']} h", change))
    return lines + _kv(rows, style)


def _render_sentiment(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    lines = _section("Sentimiento y noticias", style)
    s = r["sentiment"]
    fg = s["fear_greed"]
    fg_text = DASH if fg is None else f"{fg['value']} ({fg['label_es']})"
    cap_change = s["market_cap_change_24h_pct"]
    cap = fmt_compact_usd(s["total_market_cap_usd"])
    if cap != DASH and _finite(cap_change) is not None:
        cap += f" (24 h: {fmt_pct(cap_change, sign=True)})"
    lines += _kv(
        [
            ("Fear & Greed", fg_text),
            ("Dominancia de BTC", fmt_pct(s["btc_dominance_pct"], 1)),
            ("Capitalización total", cap),
        ],
        style,
    )
    heads = s["headlines"]
    lines.append("")
    lines.append(style.paint(f"  Titulares ({len(heads)})", "bold"))
    if not heads:
        lines.append("    Sin titulares por ahora.")
    rows = []
    for h in heads:
        rows.append(
            [
                fmt_time(h["published_at"], tz),
                _truncate(printable(h["source"] or DASH), 16),
                _truncate(printable(h["title"]), TITLE_WIDTH),
            ]
        )
    lines += render_table(rows, style, aligns="<<<", indent=4)
    return lines


_STATUS_TEXT = {"LONG": "LONG", "SHORT": "SHORT", "NO_OPERAR": "NO OPERAR"}
_STATUS_COLOR = {"LONG": "green", "SHORT": "red", "NO_OPERAR": "yellow"}
# Barra del score: una celda cada SCORE_BAR_STEP puntos de -100 a +100 (es dibujo, no regla).
SCORE_BAR_STEP = 5
SCORE_MIN, SCORE_MAX = -100.0, 100.0


def _badge(text: str, color: str, style: Style) -> list[str]:
    """Cartel grande del estado del plan (LONG, SHORT o NO OPERAR)."""
    inner = f"      {text}      "
    top = "  ╔" + "═" * len(inner) + "╗"
    bottom = "  ╚" + "═" * len(inner) + "╝"
    return [
        style.paint(top, color),
        style.paint(f"  ║{inner}║", color),
        style.paint(bottom, color),
    ]


def score_bar(total: Any, long_t: Any, short_t: Any) -> str:
    """Barra de texto de -100 a +100: ``┊`` en los umbrales, ``┼`` en 0 y ``●`` en el score."""
    cells = int((SCORE_MAX - SCORE_MIN) / SCORE_BAR_STEP) + 1

    def index(value: Any) -> int | None:
        v = _finite(value)
        if v is None:
            return None
        v = max(SCORE_MIN, min(SCORE_MAX, v))
        return int(round((v - SCORE_MIN) / SCORE_BAR_STEP))

    bar = ["─"] * cells
    for mark, char in ((0.0, "┼"), (short_t, "┊"), (long_t, "┊")):
        i = index(mark)
        if i is not None:
            bar[i] = char
    i = index(total)
    if i is not None:
        bar[i] = "●"
    return f"-100 ├{''.join(bar)}┤ +100"


def _signed_points(x: Any) -> str:
    """Puntos con signo y un decimal; el cero va sin signo (``"0,0"``)."""
    v = _finite(x)
    if v is not None and round(v, 1) == 0:
        return fmt_num(0.0, 1)
    return fmt_num(x, 1, sign=True)


def _render_score(r: dict[str, Any], style: Style) -> list[str]:
    sc = r.get("score")
    lines = _section(SCORE_TITLE, style)
    if not sc:
        return [*lines, "  No disponible."]
    th = sc["thresholds"]
    hint = sc["direction_hint"]
    color = {"long": "green", "short": "red"}.get(hint or "", "yellow")
    lines.append(style.paint(f"  {sc['summary']}", color))
    lines.append(f"  {score_bar(sc['total'], th['long'], th['short'])}")
    legend = (
        f"  ● = score actual · ┊ = umbrales: {fmt_num(th['short'], 0, sign=True)} o menos para "
        f"short, {fmt_num(th['long'], 0, sign=True)} o más para long"
    )
    lines.append(style.paint(legend, "dim"))
    lines.append("")
    rows: list[list[Cell | str]] = []
    for c in sc["components"]:
        details = [printable(d) for d in c["details"]] or [DASH]
        points = Cell(_signed_points(c["points"]), _signed_color(c["points"]))
        if not c["available"]:
            points = Cell("0 (sin datos)", "yellow")
        rows.append([c["label"], fmt_pct(c["weight_pct"], 0), points, Cell(details[0], None)])
        rows += [["", "", "", d] for d in details[1:]]
    rows.append(
        [
            Cell("Total", "bold"),
            "",
            Cell(_signed_points(sc["total"]), _signed_color(sc["total"])),
            "",
        ]
    )
    lines += render_table(
        rows, style, header=["Componente", "Peso", "Puntos", "Detalle"], aligns="<>><"
    )
    for w in sc["warnings"]:
        lines.append(style.paint(f"  · {printable(w)}", "yellow"))
    return lines


def _with_article(level_kind: str) -> str:
    """``"soporte"`` → ``"el soporte"``; ``"resistencia"`` → ``"la resistencia"``."""
    return f"{'la' if level_kind == 'resistencia' else 'el'} {level_kind}"


def _lower_first(text: str) -> str:
    """Primera letra en minúscula (después de "TP1: " o "Stop: ")."""
    return text[:1].lower() + text[1:]


def _multiplier_text(value: float) -> str:
    """Multiplicador sin ceros de más: ``0.5`` → ``"0,5"``, ``0.1875`` → ``"0,1875"``."""
    return fmt_num(value, 4).rstrip("0").rstrip(",")


def _signed_pct_from(price: Any, entry: Any) -> str:
    p, e = _finite(price), _finite(entry)
    if p is None or e is None or e == 0:
        return DASH
    return fmt_pct((p - e) / e * 100.0, sign=True)


def _plan_level_rows(p: dict[str, Any], symbol: str) -> list[list[Cell | str]]:
    """Filas de la tabla de niveles: entrada, alternativa, stop y objetivos."""
    entry = p["entry"]["price"]
    d = price_decimals(entry)
    risk = p["risk"]
    rows: list[list[Cell | str]] = [
        [
            Cell("Entrada (a mercado)", "bold"),
            Cell(fmt_num(entry, d), "bold"),
            "",
            "",
            "",
            "",
            "",
        ]
    ]
    alt = p.get("alternative")
    if alt and alt.get("level_price") is not None:
        rows.append(
            [
                "Entrada alternativa (límite)",
                fmt_num(alt["level_price"], d),
                _signed_pct_from(alt["level_price"], entry),
                "",
                "",
                "",
                f"{alt['level_kind']} de {alt['timeframe']}",
            ]
        )
    stop = p["stop"]
    loss = risk.get("max_loss_usd")
    rows.append(
        [
            Cell("Stop", "red"),
            fmt_num(stop["price"], d),
            _signed_pct_from(stop["price"], entry),
            "-1R",
            Cell("-" + fmt_usd(loss), "red") if _finite(loss) is not None else DASH,
            "",
            f"{stop['level_kind']} de {stop['level_timeframe']}",
        ]
    )
    for tp in p["take_profits"]:
        flags = ", ".join(tp["flags"])
        rows.append(
            [
                Cell(tp["name"], "green"),
                fmt_num(tp["price"], d),
                _signed_pct_from(tp["price"], entry),
                f"{fmt_num(tp['r_multiple'], 2)}R",
                Cell(fmt_usd(tp["net_gain_usd"]), _signed_color(tp["net_gain_usd"])),
                fmt_num(tp["net_rr"], 2),
                Cell(flags, "yellow") if flags else "",
            ]
        )
    return rows


def _plan_position_rows(
    p: dict[str, Any], symbol: str, tz: ZoneInfo
) -> list[tuple[str, Cell | str]]:
    pos, liq, risk, fees = p["position"], p["liquidation"], p["risk"], p["fees"]
    d = price_decimals(p["entry"]["price"])
    rows: list[tuple[str, Cell | str]] = []
    rows.append(("Margen", fmt_usd(pos["margin_usd"])))
    qty = fmt_num(pos["quantity"], max(4, d))
    rows.append(("Tamaño de la posición", f"{fmt_usd(pos['size_usd'])} ({qty} {symbol})"))
    lev_text = (
        f"{fmt_num(pos['leverage'], 1)}x (tamaño / margen; tu tope es "
        f"{fmt_leverage(pos['max_leverage'])})"
    )
    rows.append(
        ("Apalancamiento resultante", Cell(lev_text, "alert" if pos["red_warning"] else None))
    )
    if pos.get("red_warning_text"):
        rows.append(("", Cell(printable(pos["red_warning_text"]), "alert")))
    if liq["price"] is None:
        rows.append(("Liquidación (aislada)", "no se puede liquidar con este margen (1x)"))
    elif liq["at_open"]:
        rows.append(
            (
                "Liquidación (aislada)",
                Cell("nacería liquidada: el margen no cubre el mantenimiento", "alert"),
            )
        )
    else:
        text = (
            f"{fmt_num(liq['price'], d)} (a {fmt_pct(liq['distance_pct'])}; el stop está a "
            f"{fmt_pct(liq['stop_distance_pct'])}: {fmt_num(liq['ratio'], 2)} veces la "
            f"distancia, mínimo {fmt_num(liq['min_ratio'], 1)})"
        )
        ok = "cumple" if liq["ok"] else "NO cumple"
        rows.append(
            ("Liquidación (aislada)", Cell(f"{text}: {ok}", None if liq["ok"] else "alert"))
        )
    wallet = liq.get("wallet")
    if wallet:
        where = (
            "no se liquidaría"
            if wallet["price"] is None
            else f"{fmt_num(wallet['price'], d)} (a {fmt_pct(wallet['distance_pct'])})"
        )
        rows.append(
            (
                "Liquidación modelo Nexo",
                f"{where}; solo informativo, con la billetera de "
                f"{fmt_usd(wallet['equity_usd'], 0)}",
            )
        )
    if risk["fees_included"]:
        breakdown = "comisiones incluidas"
    else:
        # El monto ya trae las comisiones: se desglosa para que no se lean dos veces.
        breakdown = (
            f"{fmt_usd(risk['price_loss_usd'])} de precio + "
            f"{fmt_usd(risk['fees_at_stop_usd'])} de comisiones"
        )
    loss_text = (
        f"{fmt_usd(risk['max_loss_usd'])} si toca el stop ({breakdown}; el funding va aparte)"
    )
    rows.append(("Pérdida máxima", Cell(loss_text, "bold")))
    mult = _finite(risk["risk_multiplier"])
    if mult is not None and abs(mult - 1.0) > 1e-12:
        rows.append(
            (
                "Riesgo pedido",
                f"{fmt_usd(risk['requested_usd'])} del perfil × {_multiplier_text(mult)} = "
                f"{fmt_usd(risk['after_adjustments_usd'])} (ajustes por régimen)",
            )
        )
    if risk.get("reduced_by_position"):
        rows.append(("", Cell("La posición se achicó: arriesgás menos que lo pedido.", "yellow")))
    fee_text = (
        f"{fmt_usd(fees['round_trip_at_stop_usd'])} ida y vuelta si sale en el stop "
        f"({fmt_pct(fees['pct_of_risk'], 1)} del riesgo)"
    )
    if fees.get("round_trip_at_target_usd") is not None:
        fee_text += (
            f"; {fmt_usd(fees['round_trip_at_target_usd'])} si sale en el {fees['target_name']} "
            f"({fmt_pct(fees['pct_of_target'], 1)} de la ganancia)"
        )
    rows.append(("Comisiones", fee_text))
    rows.append(
        (
            "",
            Cell(
                f"{fmt_pct(fees['entry_rate_pct'], 3)} al entrar + "
                f"{fmt_pct(fees['exit_rate_pct'], 3)} al salir, sobre el tamaño "
                "(no sobre el margen)",
                "dim",
            ),
        )
    )
    funding = p.get("funding")
    if funding:
        rows.append(("Funding estimado", funding["text"]))
    ts = p.get("time_stop")
    if ts:
        when = fmt_time(ts["deadline"], tz)
        rows.append(
            (
                "Stop por tiempo",
                f"{when} ({fmt_num(ts['hours'], 0)} h): si para entonces no tocó el "
                f"{ts['target_name']} ni el stop, cerrá",
            )
        )
    return rows


def _render_alternative(alt: dict[str, Any], entry: Any, style: Style) -> list[str]:
    d = price_decimals(entry)
    lines = ["", style.paint("  Entrada alternativa (orden límite)", "bold")]
    if not alt.get("available"):
        lines.append(f"    {alt.get('reason') or DASH}")
        return lines
    where = (
        f"en {_with_article(alt['level_kind'])} de {alt['timeframe']}: "
        f"{fmt_num(alt['level_price'], d)} "
        f"({_signed_pct_from(alt['level_price'], entry)} del precio actual)"
    )
    lines.append(f"    Orden límite {where}. Si el precio no llega, no entrás.")
    status = alt["status"]
    state = Cell(_STATUS_TEXT.get(status, status), _STATUS_COLOR.get(status))
    rows: list[tuple[str, Cell | str]] = [("Estado", state)]
    if alt.get("stop") is not None:
        rows.append(("Stop", f"{fmt_num(alt['stop'], d)} (a {fmt_pct(alt['stop_distance_pct'])})"))
    targets = [
        f"{name.upper()} {fmt_num(alt[name], d)}" for name in ("tp1", "tp2", "tp3") if alt.get(name)
    ]
    if targets:
        rows.append(("Objetivos", " · ".join(targets)))
    if alt.get("size_usd") is not None:
        rows.append(
            (
                "Posición",
                f"{fmt_usd(alt['size_usd'])} con {fmt_usd(alt['margin_usd'])} de margen "
                f"({fmt_num(alt['leverage'], 1)}x)",
            )
        )
        rows.append(("Pérdida máxima", fmt_usd(alt["max_loss_usd"])))
    if alt.get("net_gain_tp2_usd") is not None:
        rows.append(
            (
                "Ganancia neta en el TP2",
                f"{fmt_usd(alt['net_gain_tp2_usd'])} (R:R con comisiones "
                f"{fmt_num(alt['net_rr_tp2'], 2)})",
            )
        )
    liq = alt.get("liquidation")
    if liq:
        if liq.get("price") is None:
            liq_text = "no se puede liquidar con este margen (1x)"
        else:
            ok = "cumple" if liq.get("ok") else "NO cumple"
            liq_text = (
                f"{fmt_num(liq['price'], d)} (a {fmt_pct(liq['distance_pct'])}; "
                f"{fmt_num(liq['ratio'], 2)} veces la distancia al stop, mínimo "
                f"{fmt_num(liq['min_ratio'], 1)}): {ok}"
            )
        rows.append(("Liquidación (aislada)", liq_text))
    if alt.get("fees_round_trip_at_stop_usd") is not None:
        fee_text = f"{fmt_usd(alt['fees_round_trip_at_stop_usd'])} si sale en el stop"
        if alt.get("fees_round_trip_at_tp2_usd") is not None:
            fee_text += f"; {fmt_usd(alt['fees_round_trip_at_tp2_usd'])} si sale en el TP2"
        rows.append(("Comisiones (ida y vuelta)", fee_text))
    ts = alt.get("time_stop")
    if ts:
        rows.append(
            (
                "Stop por tiempo",
                f"{fmt_num(ts['hours'], 0)} h después de entrar: si no tocó el "
                f"{ts['target_name']} ni el stop, cerrá",
            )
        )
    lines += _kv(rows, style, indent=4)
    # Los del régimen son los mismos que los del plan principal: acá solo los propios.
    adjustments = [a for a in alt.get("adjustments") or [] if a.get("source") == "position"]
    if adjustments:
        lines.append(style.paint("    Ajustes de esta entrada:", "bold"))
        for a in adjustments:
            lines.append(f"      · {printable(a['message'])}")
    for reason in alt.get("blocking_reasons") or []:
        lines.append(style.paint(f"    ✗ {printable(reason)}", "red"))
    return lines


def _render_plan(r: dict[str, Any], style: Style, tz: ZoneInfo) -> list[str]:
    p = r.get("plan")
    lines = _section(PLAN_TITLE, style)
    if not p:
        return [*lines, "  No disponible."]
    status = p["status"]
    lines += _badge(_STATUS_TEXT.get(status, status), _STATUS_COLOR.get(status, "yellow"), style)
    lines.append(style.paint(f"  {printable(p['headline'])}", "bold"))
    prof = p["profile"]
    profile_text = (
        f"  Perfil {prof['name']}: riesgo {fmt_usd(prof['risk_per_trade_usd'], 0)} por "
        f"operación, margen {fmt_usd(prof['margin_per_trade_usd'], 0)}, tope "
        f"{fmt_leverage(prof['max_leverage'])}"
    )
    if p.get("min_rr") is not None:
        profile_text += f", R:R mínimo {fmt_num(p['min_rr'], 2)} (lo pide {p['min_rr_reason']})"
    lines.append(style.paint(profile_text + ".", "dim"))

    if p["stop"] is not None:
        title = "  Niveles"
        if p["is_reference"]:
            title += " (referencia, no operar)"
        lines.append("")
        lines.append(style.paint(title, "yellow" if p["is_reference"] else "bold"))
        lines += render_table(
            _plan_level_rows(p, r["symbol"]),
            style,
            header=["", "Precio", "Distancia", "R", "Ganancia neta", "R:R con comisiones", "Nota"],
            aligns="<>>>>><",
            indent=4,
        )
        atr_tf = p["stop"].get("atr_timeframe") or "15m"
        legend = (
            "    R = lo que arriesgás hasta el stop (1R; 2R es el doble). R:R 1,5 = por cada "
            "1 USD que arriesgás ganás 1,5 (ya descontadas las comisiones). "
            f"ATR {atr_tf} = lo que se mueve el precio en una vela normal de {atr_tf}."
        )
        lines.append(style.paint(legend, "dim"))
        lines.append("")
        lines += _kv(_plan_position_rows(p, r["symbol"], tz), style, indent=4)
        why = [f"Stop: {_lower_first(p['stop']['explanation'])}"]
        word = _with_article(p["stop"]["level_kind"])
        for sk in p["stop"]["skipped_levels"]:
            why.append(
                f"Stop: se salteó {word} {fmt_num(sk['price'], price_decimals(sk['price']))} "
                f"({sk['timeframe']}) por estar a {fmt_num(sk['distance_atr'], 2)} ATR "
                "(demasiado cerca del precio)."
            )
        why += [f"{tp['name']}: {_lower_first(tp['explanation'])}" for tp in p["take_profits"]]
        if p.get("tp1_missing_reason"):
            why.append(p["tp1_missing_reason"])
        if p.get("tp2_missing_reason"):
            why.append(f"No hay TP2: {p['tp2_missing_reason']}.")
        lines.append("")
        lines.append(style.paint("  De dónde salen los niveles", "bold"))
        lines += [f"    · {printable(w)}" for w in why]
    elif p.get("stop_missing_reason"):
        lines.append("")
        lines.append(style.paint(f"  Sin stop: {printable(p['stop_missing_reason'])}", "yellow"))

    if p["adjustments"]:
        lines.append("")
        lines.append(style.paint("  Ajustes", "bold"))
        for a in p["adjustments"]:
            lines.append(f"    · {printable(a['message'])}")

    failed = [rule for rule in p["rules"] if not rule["ok"]]
    lines.append("")
    if failed:
        lines.append(style.paint("  Reglas que no se cumplen", "bold"))
        for rule in failed:
            lines.append(style.paint(f"    ✗ {printable(rule['message'])}", "red"))
            if rule.get("suggestion"):
                lines.append(f"      → {printable(rule['suggestion'])}")
    else:
        total = len(p["rules"])
        lines.append(style.paint(f"  Reglas: se cumplen las {total}.", "green"))
    # Si se cargó cómo va el día, se muestra aunque la regla se cumpla.
    day = p.get("day_state") or {}
    if day.get("trades_today") or day.get("pnl_today_usd"):
        daily = next((x for x in p["rules"] if x["code"] == "daily_limits"), None)
        if daily is not None and daily["ok"]:
            lines.append(f"    {printable(daily['message'])}")

    # La advertencia roja del apalancamiento ya va debajo de su fila.
    shown = (p.get("position") or {}).get("red_warning_text")
    warnings = [w for w in p["warnings"] if w != shown]
    if warnings:
        lines.append("")
        lines.append(style.paint("  Avisos", "yellow"))
        for w in warnings:
            color = "alert" if w.startswith("Advertencia") else "yellow"
            lines.append(style.paint(f"    · {printable(w)}", color))

    if p.get("alternative"):
        lines += _render_alternative(p["alternative"], p["entry"]["price"], style)
    return lines


def render_text(report: dict[str, Any], *, color: bool = False) -> str:
    """Salida para la consola, en castellano."""
    style = Style(enabled=color)
    tz = _tz(report["user_timezone"])
    lines: list[str] = []
    lines += _render_header(report, style, tz)
    lines += _render_prices(report, style)
    lines += _render_regime(report, style, tz)
    lines += _render_timeframes(report, style, tz)
    lines += _render_levels(report, style, tz)
    lines += _render_derivatives(report, style, tz)
    lines += _render_sentiment(report, style, tz)
    lines += _render_score(report, style)
    lines += _render_plan(report, style, tz)
    if report["offline"]:
        lines.append("")
        lines += _banner(OFFLINE_BANNER, style)
    return "\n".join(lines) + "\n"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --- ejecución ----------------------------------------------------------------------


_AR_GROUPED = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?")  # 96.000,50 / 1.234.567
_US_GROUPED = re.compile(r"[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?")  # 96,000.50 / 1,234,567
_PLAIN = re.compile(r"\d+(?:[.,]\d+)?")  # 96000 / 96000,5 / 2.6153
_AMBIGUOUS = re.compile(r"[1-9]\d{0,2}[.,]\d{3}")  # 96.000 / 2,615: ¿miles o decimales?


def parse_price(text: str) -> float:
    """Precio escrito a mano en formato argentino, de EE. UU. o sin separador de miles.

    ``"96.000,50"`` y ``"96,000.50"`` → ``96000.5``; ``"96000,5"`` y
    ``"2.6153"`` → decimal. Un solo separador seguido de exactamente tres
    cifras (``"96.000"``, ``"2,615"``) puede ser miles o decimales: lanza
    ``ValueError`` con un mensaje que explica cómo escribirlo.
    """
    raw = text.strip().replace(" ", "")
    if _AMBIGUOUS.fullmatch(raw):
        sep = raw[-4]
        sep_name = "el punto" if sep == "." else "la coma"
        raise ValueError(
            f"{text.strip()!r} es ambiguo: no se sabe si {sep_name} separa miles o decimales. "
            f"Si son miles, escribilo sin separador ({raw.replace(sep, '')}); si son "
            f"decimales, agregá un cero al final ({raw}0)."
        )
    if _AR_GROUPED.fullmatch(raw):
        normalized = raw.replace(".", "").replace(",", ".")
    elif _US_GROUPED.fullmatch(raw):
        normalized = raw.replace(",", "")
    elif _PLAIN.fullmatch(raw):
        normalized = raw.replace(",", ".")
    else:
        raise ValueError(f"no es un número: {text!r} (ejemplos: 96000,5 o 96.000,50)")
    return float(normalized)


def _positive_float(text: str) -> float:
    try:
        value = parse_price(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("tiene que ser un número mayor que 0")
    return value


def _split_sign(text: str) -> tuple[float, str]:
    """``"-30,5"`` → ``(-1.0, "30,5")``; acepta ``-``, ``−`` (signo menos) y ``+``."""
    raw = text.strip().replace(" ", "")
    sign = -1.0 if raw.startswith(("-", "−")) else 1.0
    return sign, raw[1:] if raw[:1] in ("-", "−", "+") else raw


def _signed_amount(text: str) -> float:
    """Monto con signo (``-30``, ``-30,5``, ``+12``) para ``--pnl-today``."""
    sign, body = _split_sign(text)
    try:
        value = parse_price(body)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"no es un monto en USD: {text!r} (ej. -30 o -30,5 si perdiste, 12 si ganaste)"
        ) from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("tiene que ser un número")
    return sign * value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"tiene que ser un número entero (vino {text!r})"
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError("no puede ser negativo")
    return value


def _leverage_value(text: str) -> float:
    """Tope de apalancamiento: un número de 1 para arriba (``10``, ``12,5``)."""
    sign, body = _split_sign(text)
    try:
        value = sign * parse_price(body)
    except ValueError:
        raise argparse.ArgumentTypeError(f"no es un número: {text!r} (ej. 10 o 12,5)") from None
    if not math.isfinite(value) or value < MIN_EXCHANGE_LEVERAGE:
        raise argparse.ArgumentTypeError(
            f"tiene que ser {fmt_leverage(MIN_EXCHANGE_LEVERAGE)} o más "
            f"({fmt_leverage(MIN_EXCHANGE_LEVERAGE)} = sin apalancamiento; vino {text!r})"
        )
    return value


def _aware_datetime(text: str) -> datetime:
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"fecha inválida: {text!r} (formato 2026-10-12T14:30:00Z)"
        ) from None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# Opciones que aceptan montos negativos ("-30,5"): argparse toma "-30,5" como si
# fuera otra opción, así que se pega al nombre ("--pnl-today=-30,5"). Solo si lo
# que sigue parece un número negativo: "--pnl-today --json" no se pega (es otra
# opción y argparse avisa que falta el monto).
_NEGATIVE_AMOUNT_OPTIONS = ("--pnl-today",)
_NEGATIVE_NUMBER = re.compile(r"[-−]\s*\d[\d.,]*")


def _join_negative_amounts(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if (
            a in _NEGATIVE_AMOUNT_OPTIONS
            and i + 1 < len(args)
            and _NEGATIVE_NUMBER.fullmatch(args[i + 1].strip())
        ):
            out.append(f"{a}={args[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.analyze_cli",
        description=(
            "Datos, indicadores, score y plan de una moneda (fase 1: solo análisis, no opera)."
        ),
    )
    parser.add_argument("symbol", help="moneda, por ejemplo BTC")
    parser.add_argument(
        "--nexo-price",
        type=_positive_float,
        default=None,
        help="precio de Nexo pegado a mano, ej. 96000,5 o 96.000,50 (si no, se usa Binance)",
    )
    parser.add_argument(
        "--profile", default=None, help="perfil de riesgo (por defecto el de config)"
    )
    parser.add_argument(
        "--max-leverage",
        type=_leverage_value,
        default=None,
        metavar="X",
        help="tope de apalancamiento a mano, de 1 para arriba (máximo "
        "risk_profiles.leverage_hard_cap; por encima de "
        "risk_profiles.leverage_red_warning_above muestra una advertencia roja)",
    )
    parser.add_argument(
        "--high-vol-event",
        nargs="?",
        const="",
        default=None,
        metavar="DESCRIPCIÓN",
        help="marca un evento de alta volatilidad (Fed, CPI...): el plan da No operar",
    )
    parser.add_argument(
        "--trades-today",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="operaciones que ya hiciste hoy (para el límite del día)",
    )
    parser.add_argument(
        "--pnl-today",
        type=_signed_amount,
        default=0.0,
        metavar="USD",
        help="resultado de hoy en USD, negativo si perdiste (ej. -30)",
    )
    parser.add_argument("--offline", action="store_true", help="datos sintéticos, sin red")
    parser.add_argument("--json", action="store_true", help="salida en JSON")
    parser.add_argument(
        "--no-regime", action="store_true", help="no calcula el régimen de mercado (más rápido)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="semilla del modo --offline")
    parser.add_argument(
        "--now",
        type=_aware_datetime,
        default=None,
        help="hora fija para --offline (ISO 8601, ej. 2026-10-12T14:30:00Z)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="otro config.yaml")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="muestra los avisos de los logs"
    )
    return parser


async def _regime_report(
    symbol: str,
    settings: AppSettings,
    market_data: Any,
    *,
    synthetic: bool,
    clock: Callable[[], datetime] | None,
    err: TextIO,
) -> tuple[dict[str, Any], RegimeReport | None]:
    """Régimen de 1h y diario para el reporte; nunca lanza (un error queda como motivo).

    Devuelve el dict para el reporte y el ``RegimeReport`` de
    ``regime.timeframe`` (1h, el que usa el plan) o ``None`` si no está.
    """

    def notify(message: str) -> None:
        print(message, file=err, flush=True)

    kwargs: dict[str, Any] = {"synthetic": synthetic, "notify": notify}
    if clock is not None:
        kwargs["clock"] = clock
    try:
        results = await get_market_regimes(symbol, settings, market_data, **kwargs)
        main_report = results.get(settings.regime.timeframe)
        return (
            regimes_to_dict(results, include_history=False),
            main_report if isinstance(main_report, RegimeReport) else None,
        )
    except Exception as exc:  # el régimen nunca tira abajo la consola
        logging.getLogger(__name__).exception("Falló el régimen de %s", symbol)
        reason = f"error inesperado ({type(exc).__name__}: {exc})"
        tfs = dict.fromkeys([settings.regime.timeframe, settings.regime.context_timeframe])
        return {tf: {"available": False, "timeframe": tf, "reason": reason} for tf in tfs}, None


async def _fetch_data(
    symbol: str, settings: AppSettings, args: argparse.Namespace, err: TextIO
) -> tuple[MarketSnapshot, dict[str, Any] | None, RegimeReport | None]:
    """Snapshot de mercado y, si no se pidió ``--no-regime``, el régimen (dict y reporte de 1h)."""
    want_regime = not args.no_regime
    if args.offline:
        market = SyntheticMarketData(
            seed=args.seed,
            now=args.now,
            funding_interval_hours=settings.fees.funding_interval_hours,
            # Historia larga para entrenar el régimen; las velas recientes no cambian.
            min_history_days=regime_history_days(settings.regime) if want_regime else None,
        )
        snapshot = await build_offline_snapshot(
            symbol, settings, args.nexo_price, seed=args.seed, now=args.now, market=market
        )
        regime, regime_main = None, None
        if want_regime:
            regime, regime_main = await _regime_report(
                snapshot.symbol, settings, market, synthetic=True, clock=lambda: market.now, err=err
            )
        return snapshot, regime, regime_main
    secrets = get_secrets()
    async with AsyncExitStack() as stack:
        client = make_client(settings.exchanges.request_timeout_s)
        stack.push_async_callback(client.aclose)
        cache = TTLCache()
        market_data = build_market_data(settings, client, cache)
        stack.push_async_callback(market_data.aclose)
        nexo = build_nexo_price_source(settings)
        stack.push_async_callback(nexo.aclose)
        snapshot = await build_snapshot(
            symbol, settings, market_data, nexo, client, cache, secrets, args.nexo_price
        )
        regime, regime_main = None, None
        if want_regime:
            regime, regime_main = await _regime_report(
                snapshot.symbol, settings, market_data, synthetic=False, clock=None, err=err
            )
        return snapshot, regime, regime_main


def main(
    argv: Sequence[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    """Punto de entrada. Devuelve el código de salida."""
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(_join_negative_amounts(argv))
    logging.basicConfig(
        level=logging.WARNING if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    symbol = args.symbol.strip().upper()
    if not symbol.isalnum():
        print(f"Símbolo inválido: {args.symbol!r}. Usá algo como BTC o ETH.", file=err)
        return EXIT_USAGE
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"No se pudo leer la configuración ({args.config}): {exc}", file=err)
        return EXIT_USAGE
    try:
        profile = get_profile(args.profile, settings, max_leverage_override=args.max_leverage)
    except ValueError as exc:
        print(str(exc), file=err)
        return EXIT_USAGE

    try:
        snapshot, regime, regime_main = asyncio.run(_fetch_data(symbol, settings, args, err))
    except KeyboardInterrupt:
        print(f"\n{CANCELLED}", file=err)
        return EXIT_INTERRUPTED
    except EssentialDataError as exc:
        print(f"No se pudieron conseguir los datos esenciales. {exc}", file=err)
        return EXIT_DATA
    except ValueError as exc:
        print(f"Dato inválido: {exc}", file=err)
        return EXIT_USAGE

    analysis = analyze_all(
        snapshot.candles, settings, snapshot.current_price, now=snapshot.created_at
    )
    report = build_report(
        snapshot,
        analysis,
        settings,
        profile,
        regime,
        regime_report=regime_main,
        trades_today=args.trades_today,
        pnl_today_usd=args.pnl_today,
        high_volatility_event=args.high_vol_event is not None,
        event_description=printable(args.high_vol_event) or None
        if args.high_vol_event is not None
        else None,
    )
    if args.json:
        out.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    else:
        color = bool(getattr(out, "isatty", lambda: False)())
        out.write(render_text(report, color=color))
    return EXIT_OK


def exit_process(code: int) -> None:
    """Termina el proceso con ``code``.

    Después de un Ctrl+C puede seguir corriendo, en su hilo, un entrenamiento
    del régimen que ya se descartó. Para no esperarlo (puede tardar minutos),
    se vacían las salidas y se sale con ``os._exit``, sin la finalización
    normal de Python.
    """
    if code == EXIT_INTERRUPTED:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    exit_process(main())
