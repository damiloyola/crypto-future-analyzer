"""Script de consola: todos los datos e indicadores de una moneda.

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
- ``--profile agresivo``: perfil de riesgo para el ejemplo de sizing (por
  defecto ``risk_profiles.default``).
- ``--offline``: datos sintéticos, sin red (``data.synthetic``). Muestra un
  cartel grande porque los números no son reales. ``--seed`` y ``--now``
  permiten repetir exactamente la misma corrida.
- ``--json``: el mismo contenido en JSON (claves en inglés, números sin
  formatear, ``null`` donde falta un dato).
- ``--config``: otro ``config.yaml``. ``-v``: muestra los avisos de los logs.

Qué imprime (en castellano, en tablas alineadas para fuente monoespaciada;
con colores ANSI solo si la salida es una terminal):

1. encabezado: moneda, hora (``app.user_timezone``), perfil, fuentes de datos
   con su hora de actualización y avisos (respaldo, desactualizado,
   sintético, precio de Nexo);
2. precios: referencia (Binance, o Bybit si Binance falló; "no disponible"
   si no hubo ticker), Nexo (manual o "no disponible: se usa ..." con la fuente
   real del precio usado), diferencia y cambio 24 h;
3. indicadores por timeframe (1d, 4h, 1h, 15m, 5m) con la última vela cerrada;
4. soportes y resistencias de ``trade_plan.stop_timeframes`` y
   máximo/mínimo del día anterior;
5. rango diario típico (ATR diario en % del precio);
6. derivados: funding, interés abierto y ratio long/short;
7. sentimiento: Fear & Greed, dominancia de BTC y titulares;
8. un ejemplo didáctico de sizing LONG y SHORT con el perfil elegido (el
   plan real llega en el Hito 4): stop = soporte/resistencia más cercano de
   ``trade_plan.stop_timeframes`` ∓ ``trade_plan.stop_atr_buffer`` × ATR de
   ``timeframes.entry``; tamaño, apalancamiento (con avisos si pasa el máximo
   del perfil, ``risk_profiles.leverage_red_warning_above`` o el tope duro),
   comisiones, liquidación aislada contra el stop
   (``trade_plan.liquidation.min_distance_ratio``) con el margen sugerido si
   no alcanza (y los mismos avisos de apalancamiento para ese margen y para el
   tamaño reducido), y la liquidación "modelo Nexo" informativa. Si con el
   margen del perfil la posición nacería liquidada (la liquidación cae del
   lado ganador de la entrada), lo dice así en vez de mostrar una distancia.

Sale con código 1 y un mensaje claro si no se pueden conseguir los datos
esenciales (por ejemplo, Binance y Bybit caídos, o un símbolo que no existe), y
con 2 si los argumentos o la configuración no son válidos (incluido un YAML mal
escrito). Siempre cierra el cliente HTTP. Nunca muestra las keys de ``.env``.
Los textos de terceros (titulares, fuentes, avisos) se imprimen sin caracteres
de control, así no pueden meter secuencias de escape en la terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
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
from app.data.synthetic import DEFAULT_SEED
from app.exchanges.base import TIMEFRAME_MINUTES
from app.exchanges.registry import build_market_data, build_nexo_price_source
from app.http_client import make_client
from app.indicators.levels import Level, SRResult
from app.indicators.pipeline import MarketAnalysis, TimeframeSummary, analyze_all
from app.risk.fees import fee_rates_from_config, fees_pct_of
from app.risk.liquidation import (
    check_liquidation_vs_stop,
    liquidation_distance_pct,
    liquidation_price,
    max_size_for_safe_liquidation,
    min_margin_for_safe_liquidation,
    mmr_from_config,
    wallet_liquidation_price,
)
from app.risk.sizing import fraction_to_pct, size_position
from app.risk.stops import StopNotFound, structural_stop

OFFLINE_BANNER = "DATOS SINTÉTICOS (sin red): los números no son reales"
SIZING_TITLE = "Ejemplo didáctico de sizing (el plan real llega en el Hito 4)"
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
    """Decimales para mostrar un precio según su tamaño (BTC 2, XRP 4)."""
    v = _finite(reference)
    if v is None or v <= 0:
        return 2
    return int(min(8, max(2, 4 - math.floor(math.log10(v)))))


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


def _leverage_flags(leverage: float, settings: AppSettings, profile_name: str) -> list[str]:
    rp = settings.risk_profiles
    profile = rp.profiles[profile_name]
    flags = []
    if leverage > profile.max_leverage:
        flags.append(
            f"supera el máximo del perfil {profile_name} ({fmt_num(profile.max_leverage, 0)}x)"
        )
    if leverage > rp.leverage_red_warning_above:
        flags.append(f"más de {fmt_num(rp.leverage_red_warning_above, 0)}x: advertencia roja")
    if leverage > rp.leverage_hard_cap:
        flags.append(
            f"supera el tope duro de {fmt_num(rp.leverage_hard_cap, 0)}x: la app no lo permitiría"
        )
    return flags


def _sizing_side(
    side: str,
    snapshot: MarketSnapshot,
    analysis: MarketAnalysis,
    settings: AppSettings,
    profile_name: str,
) -> dict[str, Any]:
    plan = settings.trade_plan
    profile = settings.risk_profiles.profiles[profile_name]
    entry_tf = settings.timeframes.entry
    entry = snapshot.current_price
    entry_ta = analysis.timeframes.get(entry_tf)
    atr = entry_ta.summary.atr if entry_ta else None
    if atr is None:
        return {
            "available": False,
            "side": side,
            "reason": f"Sin ATR de {entry_tf}: no se puede poner el colchón del stop.",
        }
    levels_by_tf = {
        tf: (ta.summary.levels if (ta := analysis.timeframes.get(tf)) else None)
        for tf in plan.stop_timeframes
    }
    found = structural_stop(
        side,  # type: ignore[arg-type]
        entry,
        levels_by_tf,
        atr,
        plan.stop_atr_buffer,
        plan.min_stop_distance_atr,
    )
    skipped = [
        {"price": sk.price, "timeframe": sk.timeframe, "distance_atr": sk.distance_atr}
        for sk in found.skipped
    ]
    if isinstance(found, StopNotFound):
        return {
            "available": False,
            "side": side,
            "reason": f"{found.reason} No se arma el ejemplo.",
            "skipped_levels": skipped,
        }
    stop = found.stop

    entry_rate, exit_rate = fee_rates_from_config(settings.fees)
    mmr = mmr_from_config(settings.liquidation)
    min_ratio = plan.liquidation.min_distance_ratio
    sizing = size_position(
        entry,
        stop,
        side,
        profile.risk_per_trade_usd,
        profile.margin_per_trade_usd,
        entry_rate,
        exit_rate,
        plan.fees_included_in_risk,
    )
    liq = liquidation_price(side, entry, sizing.size_usd, sizing.margin_usd, mmr, exit_rate)
    check = check_liquidation_vs_stop(side, entry, stop, liq, min_ratio)
    max_margin = profile.margin_per_trade_usd * plan.liquidation.max_margin_multiplier

    out: dict[str, Any] = {
        "available": True,
        "side": side,
        "entry": entry,
        "level_price": found.level_price,
        "level_kind": found.level_kind,
        "level_timeframe": found.level_timeframe,
        "level_distance_atr": found.level_distance_atr,
        "min_stop_distance_atr": found.min_distance_atr,
        "skipped_levels": skipped,
        "atr_timeframe": entry_tf,
        "atr": atr,
        "stop": stop,
        "stop_distance_pct": fraction_to_pct(sizing.stop_distance_pct),
        "risk_usd": sizing.risk_usd,
        "fees_included_in_risk": sizing.fees_included_in_risk,
        "size_usd": sizing.size_usd,
        "quantity": sizing.quantity,
        "margin_usd": sizing.margin_usd,
        "leverage": sizing.leverage,
        "leverage_flags": _leverage_flags(sizing.leverage, settings, profile_name),
        "fees_usd": sizing.fees_at_stop_usd,
        "fees_pct_of_risk": fees_pct_of(sizing.fees_at_stop_usd, sizing.risk_usd),
        "loss_at_stop_usd": sizing.loss_at_stop_usd,
        "liquidation_price": liq,
        "liquidation_distance_pct": None
        if liq is None
        else fraction_to_pct(liquidation_distance_pct(entry, liq) or 0.0),
        "liquidation_ratio": check.ratio,
        "liquidation_min_ratio": min_ratio,
        "liquidation_before_stop": check.liq_before_stop,
        # La liquidación cae del lado ganador de la entrada: nacería liquidada.
        "liquidation_at_open": check.liquidation_distance is not None
        and check.liquidation_distance <= 0,
        "liquidation_ok": check.ok,
        "max_margin_usd": max_margin,
        "suggested_margin_usd": None,
        "suggested_margin_fits": None,
        "suggested_leverage": None,
        "suggested_leverage_flags": [],
        "safe_size_usd": None,
        "safe_size_risk_usd": None,
        "safe_size_leverage": None,
        "safe_size_leverage_flags": [],
        "wallet_equity_usd": None,
        "wallet_liquidation_price": None,
        "wallet_liquidation_distance_pct": None,
    }
    if not check.ok:
        margin = min_margin_for_safe_liquidation(
            side, entry, stop, sizing.size_usd, mmr, exit_rate, min_ratio
        )
        safe_size = max_size_for_safe_liquidation(
            side, entry, stop, sizing.margin_usd, mmr, exit_rate, min_ratio
        )
        suggested_leverage = sizing.size_usd / margin
        safe_leverage = safe_size / sizing.margin_usd
        out["suggested_margin_usd"] = margin
        out["suggested_margin_fits"] = margin <= max_margin
        out["suggested_leverage"] = suggested_leverage
        out["suggested_leverage_flags"] = _leverage_flags(
            suggested_leverage, settings, profile_name
        )
        out["safe_size_usd"] = safe_size
        out["safe_size_risk_usd"] = sizing.loss_at_stop_usd * safe_size / sizing.size_usd
        out["safe_size_leverage"] = safe_leverage
        out["safe_size_leverage_flags"] = _leverage_flags(safe_leverage, settings, profile_name)
    if settings.liquidation.show_wallet_model:
        wallet = profile.monthly_wallet_usd
        wliq = wallet_liquidation_price(side, entry, sizing.size_usd, wallet, mmr, exit_rate)
        out["wallet_equity_usd"] = wallet
        out["wallet_liquidation_price"] = wliq
        out["wallet_liquidation_distance_pct"] = (
            None if wliq is None else fraction_to_pct(liquidation_distance_pct(entry, wliq) or 0.0)
        )
    return out


def sizing_example(
    snapshot: MarketSnapshot, analysis: MarketAnalysis, settings: AppSettings, profile_name: str
) -> dict[str, Any]:
    """Ejemplo LONG y SHORT hipotético con el perfil (no es el plan del Hito 4)."""
    profile = settings.risk_profiles.profiles[profile_name]
    entry_rate, exit_rate = fee_rates_from_config(settings.fees)
    return {
        "title": SIZING_TITLE,
        "profile": profile_name,
        "risk_usd": profile.risk_per_trade_usd,
        "margin_usd": profile.margin_per_trade_usd,
        "max_leverage": profile.max_leverage,
        "fees_included_in_risk": settings.trade_plan.fees_included_in_risk,
        "entry_fee_pct": fraction_to_pct(entry_rate),
        "exit_fee_pct": fraction_to_pct(exit_rate),
        "maintenance_margin_pct": settings.liquidation.maintenance_margin_pct,
        "stop_atr_buffer": settings.trade_plan.stop_atr_buffer,
        "long": _sizing_side("long", snapshot, analysis, settings, profile_name),
        "short": _sizing_side("short", snapshot, analysis, settings, profile_name),
    }


def build_report(
    snapshot: MarketSnapshot,
    analysis: MarketAnalysis,
    settings: AppSettings,
    profile_name: str,
) -> dict[str, Any]:
    """Todo el contenido de la salida, en un dict serializable a JSON."""
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
        "profile": profile_name,
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
        "sizing_example": sizing_example(snapshot, analysis, settings, profile_name),
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
    lines += _kv(
        [
            ("Hora", f"{fmt_time(r['generated_at'], tz, seconds=True)} ({r['user_timezone']})"),
            ("Perfil de riesgo", r["profile"]),
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


def _render_sizing_side(
    side: dict[str, Any], ex: dict[str, Any], r: dict[str, Any], style: Style
) -> list[str]:
    name = "LONG" if side["side"] == "long" else "SHORT"
    if not side["available"]:
        return [style.paint(f"  {name} hipotético", "bold"), f"    {side['reason']}"]
    d = price_decimals(side["entry"])
    is_long = side["side"] == "long"
    level_word = "soporte" if is_long else "resistencia"
    where = "debajo del soporte" if is_long else "arriba de la resistencia"
    op = "−" if is_long else "+"
    lines = [style.paint(f"  {name} hipotético: stop {where} de {side['level_timeframe']}", "bold")]

    def entry_dist(pct: Any) -> str:
        return f"a {fmt_pct(pct)} de la entrada"

    level = f"{level_word} {fmt_num(side['level_price'], d)}"
    buffer = f"{fmt_num(ex['stop_atr_buffer'], 2)} × ATR {side['atr_timeframe']}"
    stop = f"{fmt_num(side['stop'], d)} = {level} {op} {buffer} ({fmt_num(side['atr'], d)})"
    fees_mode = "con comisiones incluidas" if side["fees_included_in_risk"] else "comisiones aparte"
    qty = f"{fmt_num(side['quantity'], max(4, d))} {r['symbol']}"
    lev_color = "alert" if side["leverage_flags"] else None
    leverage = Cell(f"{fmt_num(side['leverage'], 1)}x (tamaño / margen)", lev_color)
    rows: list[tuple[str, Cell | str]] = [("Entrada", fmt_num(side["entry"], d))]
    for sk in side.get("skipped_levels", []):
        rows.append(
            (
                "Nivel salteado",
                f"{level_word} {fmt_num(sk['price'], d)} ({sk['timeframe']}): a "
                f"{fmt_num(sk['distance_atr'], 2)} ATR, menos de "
                f"{fmt_num(side['min_stop_distance_atr'], 1)} (demasiado cerca del precio)",
            )
        )
    rows += [
        ("Stop", stop),
        ("Distancia al stop", fmt_pct(side["stop_distance_pct"])),
        ("Riesgo", f"{fmt_usd(side['risk_usd'])} ({fees_mode})"),
        ("Tamaño de la posición", f"{fmt_usd(side['size_usd'])} ({qty})"),
        ("Margen", fmt_usd(side["margin_usd"])),
        ("Apalancamiento", leverage),
    ]
    rows += [("", Cell(f"ATENCIÓN: {flag}", "alert")) for flag in side["leverage_flags"]]
    fees_share = fmt_pct(side["fees_pct_of_risk"], 1)
    rows += [
        ("Comisiones ida y vuelta", f"{fmt_usd(side['fees_usd'])} ({fees_share} del riesgo)"),
        ("Pérdida total si toca el stop", fmt_usd(side["loss_at_stop_usd"])),
    ]
    if side["liquidation_price"] is None:
        rows.append(("Liquidación (aislada)", "no se puede liquidar con este margen"))
    elif side["liquidation_at_open"]:
        rows.append(
            (
                "Liquidación (aislada)",
                Cell(
                    "nacería liquidada: el margen no cubre el mantenimiento + la comisión",
                    "alert",
                ),
            )
        )
    else:
        liq = fmt_num(side["liquidation_price"], d)
        rows.append(
            ("Liquidación (aislada)", f"{liq} ({entry_dist(side['liquidation_distance_pct'])})")
        )
        ratio_txt = (
            f"{fmt_num(side['liquidation_ratio'], 2)} veces la distancia al stop "
            f"(mínimo {fmt_num(side['liquidation_min_ratio'], 1)})"
        )
        if side["liquidation_ok"]:
            rows.append(("Liquidación vs stop", Cell(f"{ratio_txt}: cumple", "green")))
        else:
            why = "llega ANTES que el stop" if side["liquidation_before_stop"] else "no cumple"
            rows.append(("Liquidación vs stop", Cell(f"{ratio_txt}: {why}", "alert")))
    if side["suggested_margin_usd"] is not None:
        fits = "entra" if side["suggested_margin_fits"] else "NO entra"
        new_lev = f"{fmt_num(side['suggested_leverage'], 1)}x"
        cap = fmt_usd(side["max_margin_usd"])
        margin = fmt_usd(side["suggested_margin_usd"])
        rows.append(("Margen sugerido", f"{margin} → {new_lev}; {fits} en el tope de {cap}"))
        # Para las alternativas alcanza con el aviso más grave (el último de la lista).
        if side["suggested_leverage_flags"]:
            flag = side["suggested_leverage_flags"][-1]
            rows.append(("", Cell(f"ATENCIÓN: {flag}", "alert")))
        risk = fmt_usd(side["safe_size_risk_usd"])
        safe_lev = f"{fmt_num(side['safe_size_leverage'], 1)}x"
        rows.append(
            (
                "o achicar el tamaño a",
                f"{fmt_usd(side['safe_size_usd'])} (riesgo ≈ {risk}; {safe_lev})",
            )
        )
        if side["safe_size_leverage_flags"]:
            flag = side["safe_size_leverage_flags"][-1]
            rows.append(("", Cell(f"ATENCIÓN: {flag}", "alert")))
    if side["wallet_equity_usd"] is not None:
        wl = side["wallet_liquidation_price"]
        where_liq = (
            "no se liquidaría"
            if wl is None
            else f"{fmt_num(wl, d)} ({entry_dist(side['wallet_liquidation_distance_pct'])})"
        )
        wallet = fmt_usd(side["wallet_equity_usd"], 0)
        rows.append(("Liquidación modelo Nexo", f"{where_liq}; informativo, billetera de {wallet}"))
    return lines + _kv(rows, style, indent=4)


def _render_sizing(r: dict[str, Any], style: Style) -> list[str]:
    ex = r["sizing_example"]
    lines = _section(ex["title"], style)
    note = "  Es un ejemplo para entender los números, no una recomendación ni un plan."
    lines.append(style.paint(note, "dim"))
    profile = (
        f"{ex['profile']}: riesgo {fmt_usd(ex['risk_usd'])}, "
        f"margen {fmt_usd(ex['margin_usd'])}, máximo {fmt_num(ex['max_leverage'], 0)}x"
    )
    fees = (
        f"entrada {fmt_pct(ex['entry_fee_pct'], 3)} + salida {fmt_pct(ex['exit_fee_pct'], 3)}"
        " sobre el tamaño"
    )
    lines += _kv(
        [
            ("Perfil", profile),
            ("Comisiones", fees),
            ("Margen de mantenimiento", fmt_pct(ex["maintenance_margin_pct"], 2)),
        ],
        style,
    )
    for key in ("long", "short"):
        lines.append("")
        lines += _render_sizing_side(ex[key], ex, r, style)
    return lines


def render_text(report: dict[str, Any], *, color: bool = False) -> str:
    """Salida para la consola, en castellano."""
    style = Style(enabled=color)
    tz = _tz(report["user_timezone"])
    lines: list[str] = []
    lines += _render_header(report, style, tz)
    lines += _render_prices(report, style)
    lines += _render_timeframes(report, style, tz)
    lines += _render_levels(report, style, tz)
    lines += _render_derivatives(report, style, tz)
    lines += _render_sentiment(report, style, tz)
    lines += _render_sizing(report, style)
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


def _aware_datetime(text: str) -> datetime:
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"fecha inválida: {text!r} (formato 2026-10-12T14:30:00Z)"
        ) from None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.analyze_cli",
        description="Datos e indicadores de una moneda (fase 1: solo análisis, no opera).",
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
    parser.add_argument("--offline", action="store_true", help="datos sintéticos, sin red")
    parser.add_argument("--json", action="store_true", help="salida en JSON")
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


async def _fetch_snapshot(
    symbol: str, settings: AppSettings, args: argparse.Namespace
) -> MarketSnapshot:
    if args.offline:
        return await build_offline_snapshot(
            symbol, settings, args.nexo_price, seed=args.seed, now=args.now
        )
    secrets = get_secrets()
    async with AsyncExitStack() as stack:
        client = make_client(settings.exchanges.request_timeout_s)
        stack.push_async_callback(client.aclose)
        cache = TTLCache()
        market_data = build_market_data(settings, client, cache)
        stack.push_async_callback(market_data.aclose)
        nexo = build_nexo_price_source(settings)
        stack.push_async_callback(nexo.aclose)
        return await build_snapshot(
            symbol, settings, market_data, nexo, client, cache, secrets, args.nexo_price
        )


def main(
    argv: Sequence[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    """Punto de entrada. Devuelve el código de salida."""
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
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
    profile = args.profile or settings.risk_profiles.default
    if profile not in settings.risk_profiles.profiles:
        options = ", ".join(settings.risk_profiles.profiles)
        print(f"No existe el perfil {profile!r}. Opciones: {options}.", file=err)
        return EXIT_USAGE

    try:
        snapshot = asyncio.run(_fetch_snapshot(symbol, settings, args))
    except EssentialDataError as exc:
        print(f"No se pudieron conseguir los datos esenciales. {exc}", file=err)
        return EXIT_DATA
    except ValueError as exc:
        print(f"Dato inválido: {exc}", file=err)
        return EXIT_USAGE

    analysis = analyze_all(
        snapshot.candles, settings, snapshot.current_price, now=snapshot.created_at
    )
    report = build_report(snapshot, analysis, settings, profile)
    if args.json:
        out.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    else:
        color = bool(getattr(out, "isatty", lambda: False)())
        out.write(render_text(report, color=color))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
