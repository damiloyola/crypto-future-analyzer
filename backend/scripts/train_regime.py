"""Entrena (o reentrena) los modelos de régimen y los guarda en disco.

Uso::

    make train-regime                              # todas las favoritas, 1h y 1d
    make train-regime SYMBOL=SOL
    make train-regime SYMBOL=BTC ARGS="--offline"  # datos sintéticos, sin red
    # o: cd backend && .venv/bin/python -m scripts.train_regime [BTC|all] [opciones]

Opciones:

- ``--timeframe 1h|1d|both``: qué modelo entrenar (``regime.timeframe``, el que
  se usa para operar; ``regime.context_timeframe``, el de contexto; o los dos).
- ``--force``: reentrena aunque el modelo guardado esté al día. Sin esto, solo
  entrena si no hay modelo, si cambió la configuración o si ya tiene
  ``regime.retrain_every_days`` días (sin importar ``regime.auto_retrain``).
- ``--offline``: mercado sintético (``data.synthetic``) con la historia larga
  que hace falta; los modelos van a ``regime.synthetic_models_dir`` y nunca
  pisan a los reales. ``--seed`` y ``--now`` repiten exactamente la corrida.
- ``--config``: otro ``config.yaml``. ``-v``: muestra los avisos de los logs.

Para cada moneda y timeframe imprime, en castellano simple: qué se hizo
(entrenado ahora o ya estaba al día, y por qué), el rango de datos y cuántas
velas se usaron, el BIC de cada cantidad de estados con la elegida marcada, la
tabla por régimen (retorno medio por vela, volatilidad media, duración típica
en horas o días y % del tiempo), la matriz de transición, cuánto tardó y dónde
quedó guardado. Los avisos de "entrenando..." y los errores van a stderr.

Usa el mismo camino que la app (``regime.service.ensure_model``): las mismas
velas cerradas, las mismas features y el mismo entrenamiento.

Si había un modelo y reentrenarlo falla (exchange caído, por ejemplo), queda
el anterior: el "Resultado" lo dice en amarillo con el motivo y la fecha del
modelo que quedó, y el aviso también sale por stderr.

Códigos de salida: 0 todo bien; 1 si alguna moneda no se pudo entrenar o
reentrenar (sin datos, exchange caído, entrenamiento fallido; las demás se
entrenan igual); 2 si los argumentos o la configuración no son válidos; 130 si
se cancela con Ctrl+C (sale enseguida con "Cancelado por el usuario." y no
guarda el modelo a medio entrenar). Siempre cierra el cliente HTTP. No usa API
keys.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

import yaml

from app.config import DEFAULT_CONFIG_PATH, REPO_ROOT, AppSettings, load_settings
from app.data.cache import TTLCache
from app.data.synthetic import DEFAULT_SEED, SyntheticMarketData
from app.exchanges.registry import build_market_data
from app.http_client import make_client
from app.indicators.features import warmup_bars
from app.regime.labels import regime_stats_table
from app.regime.service import (
    ModelStatus,
    RegimeUnavailable,
    RegimeWarning,
    ensure_model,
    regime_history_days,
    regime_timeframes,
    timeframe_minutes,
)
from scripts.analyze_cli import (
    DASH,
    Cell,
    Style,
    _aware_datetime,
    _banner,
    _section,
    _tz,
    fmt_num,
    fmt_time,
    printable,
    render_regime_states,
    render_table,
)

OFFLINE_BANNER = "DATOS SINTÉTICOS (sin red): los modelos no son reales"
BIC_HELP = "menor BIC = mejor equilibrio entre ajuste y simplicidad"

EXIT_OK = 0
EXIT_DATA = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
CANCELLED = "Cancelado por el usuario."


# --- formato ---------------------------------------------------------------------


def fmt_seconds(seconds: float | None) -> str:
    """``108.3`` → ``"1 min 48 s"``; ``14.2`` → ``"14 s"``."""
    if seconds is None:
        return DASH
    total = max(0, round(seconds))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s" if total else "menos de 1 s"


def _utc_date(dt: datetime, minutes: int) -> str:
    """Fecha (y hora si las velas son de menos de un día) en UTC."""
    pattern = "%d/%m/%Y" if minutes >= 1440 else "%d/%m/%Y %H:%M"
    return dt.astimezone(UTC).strftime(pattern)


def _show_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def retrain_failure(status: ModelStatus) -> RegimeWarning | None:
    """El aviso ``retrain_failed`` si reentrenar falló y quedó el modelo anterior."""
    return next((w for w in status.warnings if w.code == "retrain_failed"), None)


def render_status(
    symbol: str,
    timeframe: str,
    status: ModelStatus,
    settings: AppSettings,
    style: Style,
    tz: ZoneInfo,
    *,
    forced: bool = False,
) -> list[str]:
    """Bloque de texto con el resultado del entrenamiento de una moneda y un timeframe.

    ``forced``: se pidió ``--force`` (entonces no se sugiere usarlo).
    """
    cfg = settings.regime
    model = status.model
    minutes = timeframe_minutes(timeframe)
    role = "el que se usa para operar" if timeframe == cfg.timeframe else "contexto diario"
    lines = _section(f"{symbol} {timeframe} · {role}", style)

    failed = retrain_failure(status)
    if status.retrained_now:
        result = Cell(f"entrenado ahora porque {printable(status.reason)}", "green")
    elif failed is not None:
        result = Cell(printable(failed.text), "yellow")
    elif status.stale:
        result = Cell(f"se dejó el modelo anterior: {printable(status.reason)}", "yellow")
    else:
        reason = printable(status.reason)
        text = f"{reason[:1].upper()}{reason[1:]}."
        if not forced:
            text += " Usá --force para reentrenar igual."
        result = Cell(text)
    warm = warmup_bars(
        cfg.features.realized_vol_window,
        cfg.features.volume_zscore_window,
        smoothing_window=cfg.features.smoothing_window,
    )
    if status.candles_downloaded is not None:
        candles = (
            f"{fmt_num(status.candles_downloaded, 0)} velas cerradas de {timeframe}; se usaron "
            f"{fmt_num(model.n_samples, 0)} (las primeras {warm} solo sirven para calcular "
            "los indicadores)"
        )
    else:
        candles = f"{fmt_num(model.n_samples, 0)} velas de {timeframe}"
    trained = f"{fmt_time(model.trained_at.isoformat(), tz)} ({settings.app.user_timezone})"
    if status.train_seconds is not None:
        trained += f", tardó {fmt_seconds(status.train_seconds)}"
    rows: list[tuple[str, Cell | str]] = [
        ("Resultado", result),
        (
            "Datos",
            f"del {_utc_date(model.train_start, minutes)} al "
            f"{_utc_date(model.train_end, minutes)} (UTC)",
        ),
        ("Velas", candles),
        ("Entrenado", trained),
        ("Guardado en", _show_path(status.path)),
    ]
    if model.synthetic:
        rows.append(("Ojo", Cell("modelo de datos sintéticos: no sirve para operar", "yellow")))
    lines += render_table([[k, v] for k, v in rows], style, aligns="<<", indent=2)

    lines.append("")
    lines.append(
        f"  Cantidad de estados: se probó de {cfg.n_states_min} a {cfg.n_states_max} y se "
        f"eligió {model.n_states} ({BIC_HELP})."
    )
    bic_rows = []
    for n in sorted(model.bic_by_n):
        chosen = n == model.n_states
        bic_rows.append(
            [
                Cell(str(n), "bold" if chosen else None),
                Cell(fmt_num(model.bic_by_n[n], 0), "bold" if chosen else None),
                Cell("← elegido", "bold") if chosen else "",
            ]
        )
    lines += render_table(bic_rows, style, header=["Estados", "BIC", ""], aligns=">><", indent=4)

    lines.append("")
    lines.append("  Regímenes, del más bajista al más alcista:")
    lines += render_regime_states(regime_stats_table(model), minutes, style, indent=4)

    lines.append("")
    lines.append(
        "  Probabilidad de pasar de un régimen a otro en la vela siguiente "
        "(fila: el de ahora; columna: el siguiente):"
    )
    names = [printable(s.name) for s in model.states_by_rank()]
    matrix = model.transmat_by_rank()
    trans_rows = []
    for i, name in enumerate(names):
        cells: list[Cell | str] = [name]
        for j in range(len(names)):
            value = float(matrix[i, j]) * 100.0
            text = fmt_num(value, 0) + " %"
            cells.append(Cell(text, "bold") if i == j else text)
        trans_rows.append(cells)
    lines += render_table(
        trans_rows,
        style,
        header=["Desde \\ hacia", *names],
        aligns="<" + ">" * len(names),
        indent=4,
    )

    notes = [*model.notes, *(w.text for w in status.warnings if w is not failed)]
    if notes:
        lines.append("")
        lines.append(style.paint("  Avisos del entrenamiento:", "yellow"))
        for note in notes:
            lines.append(style.paint(f"  · {printable(note)}", "yellow"))
    return lines


# --- ejecución -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.train_regime",
        description="Entrena los modelos de régimen de mercado (HMM) y los guarda en disco.",
    )
    parser.add_argument(
        "symbol", nargs="?", default="all", help="moneda (BTC) o 'all' para todas las favoritas"
    )
    parser.add_argument(
        "--timeframe",
        default="both",
        help="1h (el de operar), 1d (contexto) o both (los dos, por defecto)",
    )
    parser.add_argument("--force", action="store_true", help="reentrena aunque esté al día")
    parser.add_argument("--offline", action="store_true", help="datos sintéticos, sin red")
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


async def _train_all(
    symbols: Sequence[str],
    timeframes: Sequence[str],
    settings: AppSettings,
    args: argparse.Namespace,
    out: TextIO,
    err: TextIO,
) -> int:
    style = Style(enabled=bool(getattr(out, "isatty", lambda: False)()))
    tz = _tz(settings.app.user_timezone)

    def notify(message: str) -> None:
        print(message, file=err, flush=True)

    async with AsyncExitStack() as stack:
        clock: Callable[[], datetime] | None
        if args.offline:
            market: Any = SyntheticMarketData(
                seed=args.seed,
                now=args.now,
                funding_interval_hours=settings.fees.funding_interval_hours,
                min_history_days=regime_history_days(settings.regime),
            )
            synthetic_now = market.now

            def offline_clock() -> datetime:
                return synthetic_now

            clock = offline_clock
            out.write("\n".join(_banner(OFFLINE_BANNER, style)) + "\n")
        else:
            client = make_client(settings.exchanges.request_timeout_s)
            stack.push_async_callback(client.aclose)
            market = build_market_data(settings, client, TTLCache())
            stack.push_async_callback(market.aclose)
            clock = None

        failures = 0
        for symbol in symbols:
            for tf in timeframes:
                kwargs: dict[str, Any] = {
                    "synthetic": args.offline,
                    "force_retrain": args.force,
                    "retrain_if_stale": True,
                    "notify": notify,
                }
                if clock is not None:
                    kwargs["clock"] = clock
                try:
                    status = await ensure_model(symbol, tf, settings, market, **kwargs)
                except RegimeUnavailable as exc:
                    failures += 1
                    print(f"{symbol} {tf}: no se pudo entrenar. {exc.reason}", file=err)
                    continue
                failed = retrain_failure(status)
                if failed is not None:
                    # Quedó el modelo anterior: para un reentrenamiento semanal es una falla.
                    failures += 1
                    print(f"{symbol} {tf}: {failed.text}", file=err)
                block = render_status(symbol, tf, status, settings, style, tz, forced=args.force)
                out.write("\n".join(block) + "\n")
                out.flush()
    return EXIT_DATA if failures else EXIT_OK


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
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"No se pudo leer la configuración ({args.config}): {exc}", file=err)
        return EXIT_USAGE

    cfg = settings.regime
    raw = args.symbol.strip()
    if raw.lower() == "all":
        symbols = [s.upper() for s in settings.symbols.favorites]
    elif raw.isalnum():
        symbols = [raw.upper()]
    else:
        print(f"Símbolo inválido: {args.symbol!r}. Usá algo como BTC, o 'all'.", file=err)
        return EXIT_USAGE

    allowed = regime_timeframes(cfg)
    if args.timeframe == "both":
        timeframes = allowed
    elif args.timeframe in allowed:
        timeframes = [args.timeframe]
    else:
        options = ", ".join([*allowed, "both"])
        print(f"Timeframe inválido: {args.timeframe!r}. Opciones: {options}.", file=err)
        return EXIT_USAGE

    try:
        return asyncio.run(_train_all(symbols, timeframes, settings, args, out, err))
    except KeyboardInterrupt:
        print(f"\n{CANCELLED}", file=err)
        return EXIT_INTERRUPTED


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
