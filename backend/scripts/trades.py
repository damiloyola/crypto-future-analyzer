"""Carga a mano del estado de trading (Hito 5): operaciones, posiciones, billetera y eventos.

En la fase 1 la app no ve tu cuenta: los frenos de seguridad trabajan con lo que
cargás acá (el resultado de cada operación, las posiciones abiertas, el saldo de
la billetera y los eventos de alta volatilidad). Todo va a la base de
``app.database_file`` (``data/analyzer.db``).

Uso::

    make trade ARGS="add --symbol BTC --side long --pnl -32,5"
    make trades                 # lista lo cargado
    make safety                 # cómo están los frenos de seguridad
    # o: cd backend && .venv/bin/python -m scripts.trades <comando> [opciones]

Comandos
    - ``add``: una operación cerrada. ``--symbol BTC --side long --pnl -32,5``
      (resultado neto en USD, comisiones incluidas; negativo = pérdida) y, si
      querés, ``--risk 50 --entry 61000 --exit 60600 --opened "14:00" --closed
      "14:30" --note "..."``. Sin ``--closed`` es ahora; sin ``--opened``, la
      misma hora del cierre. ``--analysis``: ``auto`` (por omisión) liga el
      último análisis de esa moneda hecho en las
      ``circuit_breakers.link_analysis_within_hours`` horas antes de abrir
      (régimen, score y plan del momento, para el informe del incidente); un
      número liga ese análisis (tiene que ser anterior a la apertura); ``none``,
      ninguno.
    - ``open``: una posición abierta. ``--symbol --side --risk`` (lo que perdés
      si toca el stop; 0 si el stop está en la entrada o en ganancia) y
      ``--entry --opened --note --analysis``.
    - ``update N``: cambia una posición abierta sin cerrarla: ``--risk`` (por
      ejemplo, después de mover el stop: 0 si quedó en la entrada o en
      ganancia, así deja de contar para el límite del día y para el tope de
      posiciones correlacionadas), ``--entry`` y ``--note`` (``""`` la borra).
      El riesgo previsto al abrir queda anotado igual (el informe del
      incidente lo usa para saber si moviste el stop) y la salida lo dice.
    - ``close [N]``: cierra la posición ``N`` (si hay una sola, no hace falta el
      número) y la guarda como operación: ``--pnl`` y ``--exit --closed --note``.
    - ``list``: últimas operaciones (``--limit``), posiciones abiertas,
      movimientos de la billetera y eventos activos o programados.
    - ``deposit`` / ``withdraw`` / ``balance``: ``--amount`` (y ``--date
      --note``). ``balance`` anota el saldo que ves en Nexo.
    - ``event``: evento de alta volatilidad. ``--note "CPI de EE. UU." --until
      "mañana 11:00"`` (y ``--from``; sin ``--from`` empieza ahora). Mientras
      dura no se generan planes.
    - ``delete trade|position|event|wallet N``: borra lo cargado por error (pide
      confirmación, salvo ``--yes``).
    - ``status``: el estado y los frenos de seguridad (si la caída del mes llegó
      al tope, escribe el informe y el archivo LOCKED, como la consola).

Confirmación con ``--yes`` (``app.manual_input``; casi siempre son un error de
tipeo): no se guardan sin ``--yes`` (se explica y sale con 1, sin guardar
nada ni crear una base nueva)

- un resultado (``add``, ``close``) más grande, en valor absoluto, que el saldo
  de la billetera o que ``confirm_pnl_over_risk_multiple`` veces el riesgo por
  operación;
- un riesgo abierto (``open``, ``update --risk``) más grande que el saldo o que
  esas mismas veces el riesgo por operación (``confirm_open_risk_over_limits``);
- un retiro más grande que el saldo calculado en su fecha
  (``confirm_withdrawal_over_balance``);
- un evento de más de ``confirm_event_longer_than_days`` días.

Opciones de todos los comandos: ``--db RUTA`` (otra base, con su propio LOCKED
y su carpeta de informes al lado, con el nombre de la base: ``prueba.db`` →
``prueba.LOCKED`` y ``prueba_incidents/``; la base real usa los de
``config.yaml``), ``--offline`` (la base de prueba de los datos sintéticos, la
misma que usa ``make cli ARGS="--offline"``), ``--profile`` y ``--config``. Una
ruta relativa es relativa a la carpeta desde donde corriste ``make`` (el
Makefile la pasa en ``ANALYZER_INVOCATION_DIR``) o, sin ``make``, a la carpeta
actual. ``list``, ``status``, ``close``, ``update`` y ``delete`` con una
``--db`` que no existe salen con 1 sin crearla (¿ruta mal escrita?); los
comandos que cargan algo la crean y lo avisan, solo si salen bien (si fallan,
la base nueva se borra).

Fechas y horas: en la zona del usuario (``app.user_timezone``), se guardan en
UTC. Se aceptan ``"2026-10-13 14:30"``, ``"13/10 14:30"``, ``"14:30"`` (hoy),
``"ayer 22:00"``, ``"mañana 11:00"``, ``"ahora"`` y fechas ISO con zona
(``2026-10-13T17:30:00Z``). Montos con coma o punto decimal (``-32,5``). Una
fecha sin año es, para aperturas, cierres y movimientos, la de este año; si esa
cae a más de medio año en el futuro, la del año anterior (el 1/1, ``"31/12
23:00"`` es de ayer); si cae menos adelante, es un error ("está en el futuro",
con el año). Para eventos, la más cercana. Un 29/02 sin año cuando esos años no
son bisiestos pide el año. Una hora hasta ``MAX_CLOCK_SKEW`` (5 min) adelante
del reloj se guarda como ahora (si no, los frenos la ignorarían hasta esa hora);
más adelante, es un error.

Después de cargar algo muestra en una línea cómo quedaron los frenos (sin
escribir el LOCKED: así podés terminar de cargar, por ejemplo, un depósito que
faltaba; el bloqueo lo escriben ``status`` y el análisis). Con las operaciones
permitidas del día ya hechas, el estado es ROJO hasta las 00:00 (freno
``max_trades_today``), igual que en la consola de análisis.

Sale con 0 si salió bien; 1 si no existe lo que se pidió, la base no existe o no
se puede usar, o falta confirmar con ``--yes``; y 2 si los argumentos no son
válidos (fecha futura, monto inválido...).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from sqlalchemy.exc import SQLAlchemyError

from app.config import DEFAULT_CONFIG_PATH, AppSettings, load_settings
from app.risk.profiles import EffectiveProfile, get_profile
from app.safety.breakers import MONTHLY_DRAWDOWN_LOCK, LockWriteError
from app.safety.incident import analysis_test_marks, fmt_analysis_marks, timezone_label
from app.safety.state import load_trading_state
from app.storage import Repository
from app.storage.repository import MAX_CLOCK_SKEW, StoredAnalysis, StoredTrade
from scripts.analyze_cli import (
    DASH,
    EXIT_DATA,
    EXIT_OK,
    EXIT_USAGE,
    INVOCATION_DIR_ENV,  # noqa: F401 (se reexporta: las dos consolas usan la misma)
    SAFETY_TITLE,
    Cell,
    SafetyContext,
    StatePaths,
    Style,
    _join_negative_amounts,
    _section,
    _signed_color,
    _signed_usd,
    _split_sign,
    _truncate,
    display_path,
    evaluate_safety,
    fmt_num,
    fmt_usd,
    lock_write_error_text,
    missing_db_message,
    parse_price,
    printable,
    render_safety_section,
    render_table,
    resolve_state_paths,
    resolve_user_path,
    safety_to_dict,
)

_SIDES = {"long": "long", "short": "short", "compra": "long", "venta": "short"}
_KIND_NAMES = {"deposit": "Depósito", "withdrawal": "Retiro", "balance": "Saldo anotado"}
_STATUS_TEXT = {"LONG": "LONG", "SHORT": "SHORT", "NO_OPERAR": "No operar"}
_VERDICT_TEXT = {"verde": "VERDE", "ambar": "ÁMBAR", "rojo": "ROJO"}
_VERDICT_COLOR = {"verde": "green", "ambar": "yellow", "rojo": "alert"}
# Qué se puede borrar (con sus nombres en castellano).
_DELETE_KINDS = {
    "trade": "trade",
    "operacion": "trade",
    "operación": "trade",
    "position": "position",
    "posicion": "position",
    "posición": "position",
    "event": "event",
    "evento": "event",
    "wallet": "wallet",
    "billetera": "wallet",
    "movimiento": "wallet",
}
_NO_LINK = {"none", "no", "ninguno", "sin"}
_YES = {"s", "si", "sí", "y", "yes"}
# Un análisis hecho hasta este rato después de la hora escrita (que no tiene
# segundos) todavía cuenta como "antes de abrir".
_LINK_GRACE = timedelta(minutes=1)


class NotFoundError(Exception):
    """Lo que se pidió no existe (sale con código 1)."""


# --- lectura de lo que escribe el usuario --------------------------------------------

_TIME = r"(?P<h>\d{1,2})[:.](?P<m>\d{2})(?::(?P<s>\d{2}))?"
_RELATIVE = {"hoy": 0, "ayer": -1, "anteayer": -2, "mañana": 1, "manana": 1}
_RE_RELATIVE = re.compile(rf"(?P<word>{'|'.join(_RELATIVE)})(?:\s+(?:a\s+las\s+)?{_TIME})?")
_RE_TIME = re.compile(_TIME)
_RE_TODAY = re.compile(rf"hoy(?:\s+(?:a\s+las\s+)?){_TIME}")
_RE_ISO = re.compile(
    rf"(?P<y>\d{{4}})-(?P<mo>\d{{1,2}})-(?P<d>\d{{1,2}})(?:[ t]+{_TIME}"
    r"(?P<tz>z|[+-]\d{2}:?\d{2})?)?"
)
_RE_DMY = re.compile(rf"(?P<d>\d{{1,2}})/(?P<mo>\d{{1,2}})(?:/(?P<y>\d{{2,4}}))?\s+{_TIME}")
_DATE_EXAMPLES = '"2026-10-13 14:30", "13/10 14:30", "14:30" (hoy) o "ayer 22:00"'


def _offset(text: str) -> tzinfo:
    if text == "z":
        return UTC
    sign = -1 if text[0] == "-" else 1
    digits = text[1:].replace(":", "")
    return timezone(sign * timedelta(hours=int(digits[:2]), minutes=int(digits[2:])))


def _at(day: date, m: re.Match[str], zone: tzinfo, text: str) -> datetime:
    try:
        return datetime(
            day.year,
            day.month,
            day.day,
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s") or 0),
            tzinfo=zone,
        )
    except ValueError:
        raise ValueError(
            f"La hora de {text!r} no existe (las horas van de 00:00 a 23:59)."
        ) from None


YearChoice = Literal["past", "nearest"]

#: Una fecha sin año de este año que cae más adelante que esto se toma del año
#: anterior (el 1/1, "31/12 23:00" es de ayer); una que cae menos adelante es un
#: error de tipeo ("está en el futuro"), no una fecha de hace casi un año. Es
#: una regla para leer fechas, no un umbral de riesgo.
YEARLESS_FUTURE_WINDOW = timedelta(days=183)


def _day_without_year(
    day: int, month: int, m: re.Match[str], tz: tzinfo, now: datetime, text: str, year: YearChoice
) -> datetime:
    """``"31/12 23:00"`` sin año: ¿de qué año es?

    - ``"past"`` (apertura, cierre, movimientos): la de este año si ya pasó
      (o está dentro de ``MAX_CLOCK_SKEW``). Si cae más adelante:
      - a más de ``YEARLESS_FUTURE_WINDOW`` (medio año), la del año anterior
        (el caso del Año Nuevo: el 1/1 a la 01:00, "31/12 23:00" es de hace dos
        horas);
      - si no, la de este año, que está en el futuro: quien llama
        (``Context.past``) la rechaza con el año a la vista ("14/10" escrito
        el 13/10 casi siempre es un error de tipeo, no algo de hace un año).
    - ``"nearest"`` (eventos): la más cercana a ``now`` entre el año anterior,
      este y el siguiente. Así, el 28/12, "05/01 10:00" es del año que viene.

    Un 29/02 sin año cuando ninguno de esos años es bisiesto: ``ValueError``
    ("no existe en estos años: poné el año").
    """
    this_year = now.astimezone(tz).year
    by_year: dict[int, datetime] = {}
    for y in (this_year - 1, this_year, this_year + 1):
        try:
            by_year[y] = _at(date(y, month, day), m, tz, text)
        except ValueError:
            continue  # 29/02 de un año no bisiesto, o una hora que no existe
    if not by_year:
        try:
            date(2000, month, day)  # 2000 fue bisiesto: si acá tampoco existe, la fecha no existe
        except ValueError:
            raise ValueError(f"La fecha {text!r} no existe.") from None
        _at(date(2000, month, day), m, tz, text)  # si la hora no existe, lanza ese error
        example = (
            _next_year_with(day, month, this_year - 1, back=True)
            if year == "past"
            else _next_year_with(day, month, this_year)
        )
        raise ValueError(
            f"La fecha {day:02d}/{month:02d} no existe en estos años: poné el año (ej. "
            f'"{day:02d}/{month:02d}/{example} {int(m.group("h")):02d}:{m.group("m")}").'
        )
    if year == "past":
        current = by_year.get(this_year)
        if current is not None and (
            current <= now + MAX_CLOCK_SKEW or current - now <= YEARLESS_FUTURE_WINDOW
        ):
            return current  # ya pasó, o está en el futuro y ``Context.past`` lo dice
        previous = by_year.get(this_year - 1)
        if previous is not None:
            return previous
        raise ValueError(
            f"La fecha {day:02d}/{month:02d} no existe en estos años: poné el año (ej. "
            f'"{day:02d}/{month:02d}/{_next_year_with(day, month, this_year - 1, back=True)} '
            f'{int(m.group("h")):02d}:{m.group("m")}").'
        )
    return min(by_year.values(), key=lambda c: abs(c - now))


def _next_year_with(day: int, month: int, start: int, *, back: bool = False) -> int:
    """El año más cercano a ``start`` (para adelante o, con ``back``, para atrás) con esa fecha."""
    step = -1 if back else 1
    year = start
    for _ in range(8):  # un 29/02 aparece cada 4 años (8 por los siglos no bisiestos)
        try:
            date(year, month, day)
            return year
        except ValueError:
            year += step
    return start


def parse_user_datetime(
    text: str, tz: tzinfo, now: datetime, *, year: YearChoice = "nearest"
) -> datetime:
    """Fecha y hora escritas por el usuario, en su zona ``tz`` (ver el docstring del módulo).

    ``now`` es el "ahora" (para "hoy", "ayer" y "14:30"). ``year``: de qué año
    es una fecha escrita sin año, como ``"13/10 14:30"`` (ver
    ``_day_without_year``). Lanza ``ValueError`` con ejemplos si no se entiende.
    """
    raw = " ".join(str(text).strip().lower().split())
    today = now.astimezone(tz).date()
    if raw in ("ahora", "now"):
        return now
    m = _RE_RELATIVE.fullmatch(raw)
    if m:
        if m.group("h") is None:
            raise ValueError(f'Falta la hora en {text!r} (ej. "{m.group("word")} 22:00").')
        return _at(today + timedelta(days=_RELATIVE[m.group("word")]), m, tz, text)
    m = _RE_TIME.fullmatch(raw)
    if m:
        return _at(today, m, tz, text)
    m = _RE_ISO.fullmatch(raw)
    if m:
        if m.group("h") is None:
            raise ValueError(f'Falta la hora en {text!r} (ej. "{raw} 14:30").')
        try:
            day = date(int(m.group("y")), int(m.group("mo")), int(m.group("d")))
        except ValueError:
            raise ValueError(f"La fecha {text!r} no existe.") from None
        zone = _offset(m.group("tz")) if m.group("tz") else tz
        return _at(day, m, zone, text)
    m = _RE_DMY.fullmatch(raw)
    if m:
        if m.group("y") is None:
            return _day_without_year(int(m.group("d")), int(m.group("mo")), m, tz, now, text, year)
        full_year = int(m.group("y"))
        if full_year < 100:
            full_year += 2000
        try:
            day = date(full_year, int(m.group("mo")), int(m.group("d")))
        except ValueError:
            raise ValueError(f"La fecha {text!r} no existe.") from None
        return _at(day, m, tz, text)
    raise ValueError(f"No entiendo la fecha {text!r}. Ejemplos: {_DATE_EXAMPLES}.")


def _amount(text: str, *, signed: bool) -> float:
    sign, body = _split_sign(text)
    if sign < 0 and not signed:
        raise argparse.ArgumentTypeError(f"tiene que ser un monto positivo (vino {text!r})")
    try:
        value = parse_price(body)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return sign * value


def signed_amount(text: str) -> float:
    """``-32,5`` → ``-32.5`` (resultado de una operación)."""
    return _amount(text, signed=True)


def positive_amount(text: str) -> float:
    """Monto mayor que 0 (riesgo, precio, depósito...)."""
    value = _amount(text, signed=False)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"tiene que ser mayor que 0 (vino {text!r})")
    return value


def non_negative_amount(text: str) -> float:
    """Monto de 0 para arriba (el saldo anotado puede ser 0)."""
    return _amount(text, signed=False)


def symbol_arg(text: str) -> str:
    """``btc`` → ``BTC``; ``BTCUSDT`` o ``BTC/USDT`` → ``BTC``."""
    raw = str(text).strip().upper().replace("/", "")
    if raw.endswith("USDT") and len(raw) > 4:
        raw = raw[:-4]
    if not raw.isalnum():
        raise argparse.ArgumentTypeError(f"moneda inválida: {text!r} (ej. BTC o ETH)")
    return raw


def side_arg(text: str) -> str:
    """``long``/``compra`` → ``"long"``; ``short``/``venta`` → ``"short"``."""
    side = _SIDES.get(str(text).strip().lower())
    if side is None:
        raise argparse.ArgumentTypeError(f"dirección inválida: {text!r} (long o short)")
    return side


def _positive_int(text: str) -> int:
    try:
        value = int(str(text).strip().lstrip("#"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"tiene que ser un número entero (vino {text!r})"
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("tiene que ser mayor que 0")
    return value


# --- contexto de cada comando ----------------------------------------------------------


@dataclass
class Context:
    """Lo que comparten los comandos: config, perfil, base, hora y salidas."""

    settings: AppSettings
    profile: EffectiveProfile
    repo: Repository
    paths: StatePaths
    tz: ZoneInfo
    now: datetime
    out: TextIO
    err: TextIO
    style: Style
    ask: Callable[[str], str]
    # Avisos de lo que se ajustó al leer las fechas (se muestran después de guardar).
    notes: list[str] = field(default_factory=list)

    def print(self, text: str = "", color: str | None = None) -> None:
        print(self.style.paint(text, color), file=self.out)

    def when(self, text: str | None, default: datetime | None = None) -> datetime:
        """Una fecha cualquiera (eventos): sin año, la más cercana a ahora."""
        if text is None:
            if default is None:
                raise ValueError("Falta la fecha.")
            return default
        return parse_user_datetime(text, self.tz, self.now, year="nearest")

    def past(self, text: str | None, label: str, default: datetime) -> datetime:
        """Una fecha que ya pasó (apertura, cierre, movimiento).

        - Sin año (``"31/12 23:00"``), la fecha pasada más reciente.
        - Más de ``MAX_CLOCK_SKEW`` en el futuro: ``ValueError`` con el año a la
          vista y, si era solo una hora (``"22:00"``), la sugerencia de "ayer".
        - Unos minutos adelante del reloj (dentro de ``MAX_CLOCK_SKEW``): se
          guarda como ahora, así los frenos la cuentan ya (si quedara en el
          futuro, la ignorarían hasta esa hora) y se avisa en ``notes``.
        """
        if text is None:
            return default
        moment = parse_user_datetime(text, self.tz, self.now, year="past")
        if moment > self.now + MAX_CLOCK_SKEW:
            hint = ""
            raw = " ".join(str(text).strip().lower().split())
            bare = _RE_TIME.fullmatch(raw) or _RE_TODAY.fullmatch(raw)
            if bare:
                hhmm = f"{int(bare.group('h')):02d}:{bare.group('m')}"
                hint = f' Si fue ayer, escribí "ayer {hhmm}".'
            raise ValueError(
                f"{label} ({self.fmt_dt(moment, year=True)}) está en el futuro: ahora son las "
                f"{self.fmt_dt(self.now, year=True)} ({timezone_label(self.tz)}). Revisá la fecha "
                f"y la hora.{hint}"
            )
        if moment > self.now:
            self.notes.append(
                f"{label} que pusiste ({self.fmt_dt(moment)}) está unos minutos adelante del reloj "
                f"de la computadora: la guardé como ahora ({self.fmt_dt(self.now)}), así los "
                "frenos ya la cuentan."
            )
            return self.now
        return moment

    def fmt_dt(self, dt: datetime | None, *, year: bool = False) -> str:
        """``"13/10 14:30"`` en tu zona (con el año si no es el actual, o siempre con ``year``)."""
        if dt is None:
            return DASH
        local = dt.astimezone(self.tz)
        same_year = local.year == self.now.astimezone(self.tz).year
        return local.strftime("%d/%m %H:%M" if same_year and not year else "%d/%m/%Y %H:%M")

    def print_notes(self) -> None:
        """Muestra (y vacía) los avisos de ``notes``."""
        for note in self.notes:
            self.print(f"  {note}", "yellow")
        self.notes.clear()


def _tz(settings: AppSettings) -> ZoneInfo:
    try:
        return ZoneInfo(settings.app.user_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _result_word(pnl: float) -> str:
    v = round(pnl, 2)
    return "ganancia" if v > 0 else "pérdida" if v < 0 else "empate"


def _analysis_text(ctx: Context, a: StoredAnalysis) -> str:
    regime = a.regime_name or "no disponible"
    score = DASH if a.score_total is None else fmt_num(a.score_total, 1, sign=True)
    status = _STATUS_TEXT.get(a.plan_status, a.plan_status)
    marks = fmt_analysis_marks(analysis_test_marks(a.inputs))
    return (
        f"análisis #{a.id} del {ctx.fmt_dt(a.created_at)}{f' {marks}' if marks else ''} "
        f"(régimen {regime}, score {score}, plan {status})"
    )


# Cuántos análisis de la ventana se miran para saltear los de prueba (una app
# personal hace pocos por hora: es solo un tope de la consulta, no una regla).
_LINK_CANDIDATES = 200


def _resolve_analysis(
    ctx: Context, symbol: str, opened_at: datetime, value: str | None
) -> tuple[int | None, str]:
    """El análisis a ligar y una línea que lo explica.

    ``auto`` liga el último análisis real de la moneda en la ventana: los de
    prueba (datos sintéticos, o con números del día puestos a mano) se
    saltean, porque no dicen nada del mercado real. Un número los liga igual,
    con un aviso.
    """
    choice = "auto" if value is None else str(value).strip().lower()
    if choice in _NO_LINK:
        return None, "Sin análisis ligado (lo pediste así)."
    if choice == "auto":
        hours = ctx.settings.circuit_breakers.link_analysis_within_hours
        window = timedelta(hours=hours)
        found = ctx.repo.list_analyses(
            limit=_LINK_CANDIDATES,
            symbol=symbol,
            since=opened_at - window,
            until=opened_at + _LINK_GRACE,
        )
        real = next((a for a in found if not analysis_test_marks(a.inputs)), None)
        if real is not None:
            return real.id, f"Ligada al {_analysis_text(ctx, real)}."
        if found:
            test = found[0]
            marks = fmt_analysis_marks(analysis_test_marks(test.inputs))
            return None, (
                f"No había un análisis real de {symbol} en las {fmt_num(hours, 0)} h antes de "
                f"abrir: el #{test.id} es de prueba {marks} y no se liga solo (si de verdad "
                f"seguiste ese, cargalo con --analysis {test.id}). Queda sin análisis ligado."
            )
        return None, (
            f"No había un análisis de {symbol} en las {fmt_num(hours, 0)} h antes de abrir: "
            "queda sin análisis ligado."
        )
    try:
        analysis_id = _positive_int(choice)
    except argparse.ArgumentTypeError:
        raise ValueError(
            f"--analysis tiene que ser un número de análisis, auto o none (vino {value!r})."
        ) from None
    found_one = ctx.repo.get_analysis(analysis_id)
    if found_one is None:
        raise NotFoundError(f"No existe el análisis #{analysis_id}.")
    if found_one.created_at > opened_at + _LINK_GRACE:
        # El informe del incidente lo mostraría como "el régimen en ese momento".
        raise ValueError(
            f"El análisis #{analysis_id} es del {ctx.fmt_dt(found_one.created_at)}, después de "
            f"que abriste ({ctx.fmt_dt(opened_at)}): no puede ser el que seguiste para entrar. "
            "Poné el número de un análisis anterior a la apertura, --analysis auto (busca solo) "
            "o --analysis none."
        )
    link = f"Ligada al {_analysis_text(ctx, found_one)}."
    marks = analysis_test_marks(found_one.inputs)
    if marks:
        link += (
            f" Ojo: ese análisis es de prueba {fmt_analysis_marks(marks)}; en el informe del "
            "incidente va a figurar marcado así."
        )
    return analysis_id, link


_PENDING_LOCK = (
    "al analizar o con make safety la app se va a bloquear y va a escribir el informe del incidente"
)
_PENDING_LOCK_HINT = (
    "Si falta cargar un depósito o el saldo real de la billetera, cargalo antes de analizar."
)


def safety_view(sc: SafetyContext) -> dict[str, Any]:
    """El dict de los frenos (``safety_to_dict``) listo para mostrar desde esta consola.

    Es el mismo de la consola de análisis, salvo un caso: si la caída del mes
    llegó al tope pero todavía no hay LOCKED (cargar no lo escribe), el
    titular dice que el bloqueo está pendiente en vez de mandar a borrar un
    LOCKED que no existe. El máximo de operaciones del día ya es un freno más
    (``max_trades_today``), con el mismo titular en las dos consolas.
    """
    data = copy.deepcopy(safety_to_dict(sc))
    v = sc.verdict
    verdict = data.setdefault("verdict", {})
    if v.needs_lock and not v.locked:
        monthly = v.breaker(MONTHLY_DRAWDOWN_LOCK)
        others = len([c for c in v.block_codes if c != MONTHLY_DRAWDOWN_LOCK])
        tail = "" if others == 0 else f" (y {others} {'freno' if others == 1 else 'frenos'} más)"
        headline = f"No se generan planes: {monthly.summary}{tail}; {_PENDING_LOCK}."
        data["headline"] = verdict["headline"] = headline
    return data


def _safety_lines(ctx: Context) -> None:
    """Cómo quedaron los frenos, en una línea (sin escribir el LOCKED)."""
    try:
        sc = evaluate_safety(
            ctx.repo, ctx.settings, ctx.profile, ctx.now, ctx.paths, trigger_lock=False
        )
        data = safety_view(sc)
    except (OSError, ValueError, SQLAlchemyError) as exc:
        ctx.print(f"No se pudieron evaluar los frenos de seguridad: {exc}", "yellow")
        return
    status = sc.verdict.status
    ctx.print(
        f"Frenos de seguridad: {_VERDICT_TEXT.get(status, status)} · {data['headline']}",
        _VERDICT_COLOR.get(status),
    )
    if sc.verdict.needs_lock and not sc.verdict.locked:
        ctx.print(_PENDING_LOCK_HINT, "alert")
    backfill = sc.backfill
    if backfill is not None and backfill.recorded and backfill.message:
        ctx.print(f"  {backfill.message}", "yellow")


# --- confirmaciones ---------------------------------------------------------------------


class NeedsConfirmationError(Exception):
    """Un dato fuera de lo normal que hay que confirmar con ``--yes`` (sale con código 1)."""


_RETYPE = "Si está bien, cargalo de nuevo agregando --yes. No se guardó nada."


def _wallet_at(ctx: Context, moment: datetime) -> Any:
    """La billetera calculada en ``moment`` (saldo anotado o supuesto y lo cargado hasta ahí)."""
    return load_trading_state(ctx.repo, ctx.settings, ctx.profile, moment).wallet


def _balance_text(wallet: Any) -> str:
    """``"200,00 USD"`` o ``"200,00 USD, supuesto: nunca lo anotaste"`` (va entre paréntesis)."""
    assumed = ", supuesto: nunca lo anotaste" if wallet.assumed else ""
    return f"{fmt_usd(wallet.current_balance_usd)}{assumed}"


def _over_limit_reasons(ctx: Context, amount: float, wallet: Any) -> list[str]:
    """Por qué ``amount`` parece un error de tipeo (``[]`` si no).

    Más grande que el saldo de la billetera o que
    ``app.manual_input.confirm_pnl_over_risk_multiple`` veces el riesgo por
    operación del perfil.
    """
    multiple = ctx.settings.app.manual_input.confirm_pnl_over_risk_multiple
    risk_cap = multiple * ctx.profile.risk_per_trade_usd
    size = round(abs(amount), 2)
    reasons: list[str] = []
    if size > round(wallet.current_balance_usd, 2):
        reasons.append(f"que el saldo de tu billetera ({_balance_text(wallet)})")
    if size > round(risk_cap, 2):
        times = fmt_num(multiple, 0 if float(multiple).is_integer() else 1)
        reasons.append(f"que {times} veces tu riesgo por operación ({fmt_usd(risk_cap)})")
    return reasons


def _check_pnl(ctx: Context, pnl: float, confirmed: bool) -> None:
    """Pide ``--yes`` si el resultado parece un error de tipeo.

    Un resultado más grande (en valor absoluto) que el saldo actual de la
    billetera o que ``app.manual_input.confirm_pnl_over_risk_multiple`` veces
    el riesgo por operación del perfil.
    """
    if confirmed:
        return
    reasons = _over_limit_reasons(ctx, pnl, _wallet_at(ctx, ctx.now))
    if reasons:
        raise NeedsConfirmationError(
            f"Ojo: el resultado {_signed_usd(pnl)} es más grande {' y '.join(reasons)}. "
            "¿No será un error de tipeo (una coma o un cero de más)? Un resultado así puede "
            f"disparar los frenos o bloquear la app. {_RETYPE}"
        )


def _check_open_risk(ctx: Context, risk: float, confirmed: bool) -> None:
    """Pide ``--yes`` si el riesgo de una posición abierta parece un error de tipeo (E9).

    ``open`` y ``update --risk``: un riesgo más grande que el saldo actual de
    la billetera o que ``confirm_pnl_over_risk_multiple`` veces el riesgo por
    operación (``app.manual_input.confirm_open_risk_over_limits``).
    """
    if confirmed or not ctx.settings.app.manual_input.confirm_open_risk_over_limits:
        return
    reasons = _over_limit_reasons(ctx, risk, _wallet_at(ctx, ctx.now))
    if not reasons:
        return
    day = (
        " Un riesgo así cuenta entero en lo que te queda del límite de pérdida del día y puede "
        'dejar todos los planes en "No operar" hasta que lo corrijas.'
        if ctx.settings.trade_plan.daily_cap_includes_open_risk
        else ""
    )
    raise NeedsConfirmationError(
        f"Ojo: el riesgo de {fmt_usd(risk)} (lo que perdés si toca el stop) es más grande "
        f"{' y '.join(reasons)}. ¿No será un error de tipeo (una coma o un cero de más)?{day} "
        f"{_RETYPE}"
    )


def _check_withdrawal(ctx: Context, amount: float, moment: datetime, confirmed: bool) -> None:
    """Pide ``--yes`` si el retiro es más grande que el saldo calculado en su fecha (E9).

    ``app.manual_input.confirm_withdrawal_over_balance``. Casi siempre es un
    error de tipeo: dejaría el saldo en negativo y los números de la caída del
    mes dejarían de tener sentido.
    """
    if confirmed or not ctx.settings.app.manual_input.confirm_withdrawal_over_balance:
        return
    wallet = _wallet_at(ctx, moment)
    if round(amount, 2) <= round(wallet.current_balance_usd, 2):
        return
    raise NeedsConfirmationError(
        f"Ojo: el retiro de {fmt_usd(amount)} es más grande que el saldo de tu billetera el "
        f"{ctx.fmt_dt(moment)} ({_balance_text(wallet)}). ¿No será un error de tipeo "
        "(un cero de más)? Un retiro así deja el saldo en negativo y desarma las cuentas del "
        "freno de la caída del mes. Si el saldo que ves en Nexo es otro, anotalo primero con "
        f'make trade ARGS="balance --amount ...". {_RETYPE}'
    )


def _check_event_length(ctx: Context, start: datetime, until: datetime, confirmed: bool) -> None:
    """Pide ``--yes`` si el evento dura más de ``manual_input.confirm_event_longer_than_days``."""
    days = ctx.settings.app.manual_input.confirm_event_longer_than_days
    length = until - start
    if confirmed or length <= timedelta(days=days):
        return
    total_days = length / timedelta(days=1)
    raise NeedsConfirmationError(
        f"Ojo: el evento duraría {fmt_num(total_days, 0 if total_days >= 10 else 1)} días "
        f"(del {ctx.fmt_dt(start, year=True)} al {ctx.fmt_dt(until, year=True)}), y mientras "
        "dure no vas a tener planes. "
        f"¿Está bien la fecha (el año, el mes)? Los eventos suelen durar horas. Si está bien, "
        "cargalo de nuevo agregando --yes. No se guardó nada."
    )


# --- comandos --------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace, ctx: Context) -> int:
    closed = ctx.past(args.closed, "La hora de cierre", ctx.now)
    opened = ctx.past(args.opened, "La hora de apertura", closed)
    if opened > closed:
        raise ValueError(
            f"La operación no puede cerrar ({ctx.fmt_dt(closed)}) antes de abrir "
            f"({ctx.fmt_dt(opened)})."
        )
    analysis_id, link = _resolve_analysis(ctx, args.symbol, opened, args.analysis)
    _check_pnl(ctx, args.pnl, args.yes)
    trade = ctx.repo.add_trade(
        symbol=args.symbol,
        side=args.side,
        pnl_usd=args.pnl,
        opened_at=opened,
        closed_at=closed,
        risk_usd=args.risk,
        entry_price=args.entry,
        exit_price=args.exit,
        notes=args.note,
        analysis_id=analysis_id,
    )
    _print_trade_saved(ctx, trade, f"Operación #{trade.id} guardada")
    ctx.print_notes()
    ctx.print(f"  {link}")
    _safety_lines(ctx)
    return EXIT_OK


def _print_trade_saved(ctx: Context, trade: StoredTrade, title: str) -> None:
    ctx.print(
        f"{title}: {trade.symbol} {trade.side}, resultado {_signed_usd(trade.pnl_usd)} "
        f"({_result_word(trade.pnl_usd)}).",
        _signed_color(trade.pnl_usd),
    )
    zone = timezone_label(ctx.tz)
    if trade.opened_at == trade.closed_at:
        ctx.print(f"  Cerrada {ctx.fmt_dt(trade.closed_at)} ({zone}).")
    else:
        ctx.print(
            f"  Abierta {ctx.fmt_dt(trade.opened_at)} · cerrada {ctx.fmt_dt(trade.closed_at)} "
            f"({zone})."
        )


def cmd_open(args: argparse.Namespace, ctx: Context) -> int:
    opened = ctx.past(args.opened, "La hora de apertura", ctx.now)
    analysis_id, link = _resolve_analysis(ctx, args.symbol, opened, args.analysis)
    _check_open_risk(ctx, args.risk, args.yes)
    pos = ctx.repo.add_open_position(
        symbol=args.symbol,
        side=args.side,
        risk_usd=args.risk,
        opened_at=opened,
        entry_price=args.entry,
        notes=args.note,
        analysis_id=analysis_id,
    )
    ctx.print(
        f"Posición #{pos.id} abierta: {pos.symbol} {pos.side}, riesgo {fmt_usd(pos.risk_usd)} "
        f"(lo que perdés si toca el stop), desde {ctx.fmt_dt(pos.opened_at)}."
    )
    if round(pos.risk_usd, 2) <= 0:
        ctx.print(f"  {_ZERO_RISK_NOTE}")
    ctx.print_notes()
    ctx.print(f"  {link}")
    ctx.print(
        f'  Si movés el stop: make trade ARGS="update {pos.id} --risk <lo que perdés ahora>". '
        f'Cuando la cierres: make trade ARGS="close {pos.id} --pnl <resultado>".'
    )
    _safety_lines(ctx)
    return EXIT_OK


_ZERO_RISK_NOTE = (
    "Riesgo 0: el stop está en la entrada o en ganancia (si salta, no perdés). Ya no cuenta "
    "para el límite de pérdida del día ni para el tope de posiciones correlacionadas."
)


def cmd_update(args: argparse.Namespace, ctx: Context) -> int:
    """Cambia el riesgo (stop movido), la entrada o la nota de una posición abierta."""
    position = ctx.repo.get_open_position(args.id)
    if position is None:
        raise NotFoundError(f"No existe la posición abierta #{args.id}.")
    if args.risk is None and args.entry is None and args.note is None:
        raise ValueError(
            "No hay nada para cambiar: poné --risk (lo que perdés ahora si toca el stop; 0 si "
            "lo moviste a la entrada o a ganancia), --entry o --note."
        )
    if args.risk is not None:
        _check_open_risk(ctx, args.risk, args.yes)
    updated = ctx.repo.update_position(
        args.id, risk_usd=args.risk, entry_price=args.entry, note=args.note
    )
    changes: list[str] = []
    if args.risk is not None:
        changes.append(f"riesgo {fmt_usd(position.risk_usd)} → {fmt_usd(updated.risk_usd)}")
    if args.entry is not None:
        changes.append(f"entrada {_price_text(position)} → {_price_text(updated)}")
    if args.note is not None:
        changes.append("nota borrada" if updated.notes is None else "nota cambiada")
    ctx.print(
        f"Posición #{updated.id} ({updated.symbol} {updated.side}) actualizada: "
        f"{'; '.join(changes)}."
    )
    if args.risk is not None:
        ctx.print(f"  {_planned_risk_note(updated)}")
    if args.risk is not None and round(updated.risk_usd, 2) <= 0:
        ctx.print(f"  {_ZERO_RISK_NOTE}")
    _safety_lines(ctx)
    return EXIT_OK


def _planned_risk_note(position: Any) -> str:
    """Qué pasa con el riesgo previsto al abrir cuando cambia el riesgo (``update --risk``).

    El previsto no cambia: el informe de un incidente lo usa para saber si se
    movió el stop a la entrada o si se agrandó.
    """
    planned = position.planned_risk_usd
    if planned is None:
        return "No hay un riesgo previsto anotado para esta posición."
    text = (
        f"El riesgo previsto al abrir ({fmt_usd(planned)}) queda anotado igual: el informe de "
        "un incidente lo usa para saber si moviste el stop."
    )
    if round(position.risk_usd, 2) > round(planned, 2):
        text += (
            " Ojo: ahora arriesgás más que lo previsto (agrandaste el stop); si fue un error al "
            "abrir, borrá la posición y cargala de nuevo."
        )
    return text


def _price_text(position: Any) -> str:
    """Precio de entrada sin ceros de más (``"61.000,5"``) o ``"—"``."""
    price = position.entry_price
    return DASH if price is None else fmt_num(price, 4).rstrip("0").rstrip(",")


def cmd_close(args: argparse.Namespace, ctx: Context) -> int:
    position_id = args.id
    if position_id is None:
        positions = ctx.repo.list_open_positions()
        if not positions:
            raise NotFoundError("No tenés posiciones abiertas cargadas.")
        if len(positions) > 1:
            names = ", ".join(f"#{p.id} {p.symbol} {p.side}" for p in positions)
            raise ValueError(f"Tenés {len(positions)} posiciones abiertas: decime cuál ({names}).")
        position_id = positions[0].id
    position = ctx.repo.get_open_position(position_id)
    if position is None:
        raise NotFoundError(f"No existe la posición abierta #{position_id}.")
    closed = ctx.past(args.closed, "La hora de cierre", ctx.now)
    if closed < position.opened_at:
        raise ValueError(
            f"La posición #{position_id} se abrió el {ctx.fmt_dt(position.opened_at)}: no puede "
            f"cerrar antes ({ctx.fmt_dt(closed)})."
        )
    _check_pnl(ctx, args.pnl, args.yes)
    trade = ctx.repo.close_position_with_result(
        position_id,
        pnl_usd=args.pnl,
        closed_at=closed,
        exit_price=args.exit,
        notes=args.note,
    )
    _print_trade_saved(
        ctx, trade, f"Posición #{position_id} cerrada y guardada como operación #{trade.id}"
    )
    ctx.print_notes()
    _safety_lines(ctx)
    return EXIT_OK


def cmd_wallet(args: argparse.Namespace, ctx: Context) -> int:
    kind = {"deposit": "deposit", "withdraw": "withdrawal", "balance": "balance"}[args.command]
    ts = ctx.past(args.date, "La fecha del movimiento", ctx.now)
    if kind == "withdrawal":
        _check_withdrawal(ctx, args.amount, ts, bool(getattr(args, "yes", False)))
    event = ctx.repo.add_wallet_event(
        kind=kind,
        amount_usd=args.amount,
        ts=ts,
        notes=args.note,
    )
    if kind == "balance":
        amount = fmt_usd(event.amount_usd)
    else:
        amount = _signed_usd(event.amount_usd if kind == "deposit" else -event.amount_usd)
    ctx.print(f"{_KIND_NAMES[kind]} #{event.id} guardado: {amount} ({ctx.fmt_dt(event.ts)}).")
    ctx.print_notes()
    _safety_lines(ctx)
    return EXIT_OK


def cmd_event(args: argparse.Namespace, ctx: Context) -> int:
    # Sin --from empieza "ahora" (la misma hora con la que se evalúan los frenos
    # después: si lo pusiera la base, con su propio reloj, quedaría unos
    # milisegundos más tarde y la línea de abajo lo vería "programado").
    start = ctx.now if args.start is None else ctx.when(args.start)
    until = ctx.when(args.until)
    if until <= ctx.now:
        raise ValueError(
            f"La hora de fin del evento ({ctx.fmt_dt(until)}) ya pasó: ahora son las "
            f'{ctx.fmt_dt(ctx.now)}. Poné una hora futura (ej. "mañana 11:00").'
        )
    if until <= start:
        raise ValueError(
            f"La hora de fin del evento ({ctx.fmt_dt(until)}) tiene que ser después del inicio "
            f"({ctx.fmt_dt(start)})."
        )
    _check_event_length(ctx, start, until, args.yes)
    event = ctx.repo.add_volatility_event(args.note, until, starts_at=start)
    ctx.print(
        f"Evento #{event.id} guardado: «{printable(event.note)}», desde "
        f"{ctx.fmt_dt(event.starts_at)} hasta {ctx.fmt_dt(event.until)}. Mientras dure no se "
        "generan planes; después se apaga solo."
    )
    _safety_lines(ctx)
    return EXIT_OK


def cmd_list(args: argparse.Namespace, ctx: Context) -> int:
    repo, style = ctx.repo, ctx.style
    lines: list[str] = []
    trades = repo.list_trades(limit=args.limit, newest_first=True)
    lines += _section(f"Operaciones cerradas (últimas {args.limit})", style)
    if trades:
        rows: list[list[Cell | str]] = [
            [
                f"#{t.id}",
                ctx.fmt_dt(t.closed_at),
                t.symbol,
                t.side,
                Cell(_signed_usd(t.pnl_usd), _signed_color(t.pnl_usd)),
                _risk_cell(t.risk_usd, t.planned_risk_usd),
                DASH if t.analysis_id is None else f"#{t.analysis_id}",
                _truncate(printable(t.notes or ""), 30),
            ]
            for t in trades
        ]
        lines += render_table(
            rows,
            style,
            header=[
                "#",
                "Cerrada",
                "Moneda",
                "Dirección",
                "Resultado",
                "Riesgo",
                "Análisis",
                "Nota",
            ],
            aligns="><<<>><<",
        )
    else:
        lines.append("  Todavía no cargaste operaciones.")

    positions = repo.list_open_positions()
    lines += _section("Posiciones abiertas", style)
    if positions:
        rows = [
            [
                f"#{p.id}",
                ctx.fmt_dt(p.opened_at),
                p.symbol,
                p.side,
                _risk_cell(p.risk_usd, p.planned_risk_usd),
                _price_text(p),
                DASH if p.analysis_id is None else f"#{p.analysis_id}",
                _truncate(printable(p.notes or ""), 30),
            ]
            for p in positions
        ]
        lines += render_table(
            rows,
            style,
            header=["#", "Abierta", "Moneda", "Dirección", "Riesgo", "Entrada", "Análisis", "Nota"],
            aligns="><<<>><<",
        )
    else:
        lines.append("  No hay posiciones abiertas.")

    wallet = repo.list_wallet_events()[-args.limit :][::-1]
    lines += _section(f"Billetera (últimos {args.limit} movimientos)", style)
    if wallet:
        rows = []
        for e in wallet:
            if e.kind == "balance":
                amount: Cell | str = fmt_usd(e.amount_usd)
            else:
                delta = e.amount_usd if e.kind == "deposit" else -e.amount_usd
                amount = Cell(_signed_usd(delta), _signed_color(delta))
            rows.append(
                [
                    f"#{e.id}",
                    ctx.fmt_dt(e.ts),
                    _KIND_NAMES.get(e.kind, e.kind),
                    amount,
                    _truncate(printable(e.notes or ""), 30),
                ]
            )
        lines += render_table(
            rows, style, header=["#", "Fecha", "Tipo", "Monto", "Nota"], aligns="><<><"
        )
    else:
        lines.append(
            '  No anotaste movimientos ni saldos (make trade ARGS="balance --amount ...").'
        )

    events = repo.list_volatility_events(ending_after=ctx.now)
    lines += _section("Eventos de alta volatilidad (activos y programados)", style)
    if events:
        rows = [
            [
                f"#{e.id}",
                ctx.fmt_dt(e.starts_at),
                ctx.fmt_dt(e.until),
                Cell("activo", "red") if e.is_active(ctx.now) else Cell("programado", "yellow"),
                _truncate(printable(e.note), 40),
            ]
            for e in events
        ]
        lines += render_table(
            rows, style, header=["#", "Desde", "Hasta", "Estado", "Nota"], aligns="><<<<"
        )
    else:
        lines.append("  Ninguno.")
    lines.append("")
    lines.append(
        style.paint(
            f"  Base: {display_path(ctx.paths.database)}. Para borrar algo cargado por error: "
            'make trade ARGS="delete trade N" (o position, event, wallet).',
            "dim",
        )
    )
    print("\n".join(lines), file=ctx.out)
    return EXIT_OK


def _risk_cell(risk: float | None, planned: float | None) -> str:
    """``"40,00 USD"`` o, si el stop se movió, ``"0,00 USD (previsto 40,00 USD)"``."""
    if risk is None:
        return DASH
    text = fmt_usd(risk)
    if planned is not None and round(planned, 2) != round(risk, 2):
        text += f" (previsto {fmt_usd(planned)})"
    return text


def _describe_for_delete(ctx: Context, kind: str, record_id: int) -> str:
    repo = ctx.repo
    if kind == "trade":
        t = repo.get_trade(record_id)
        if t is None:
            raise NotFoundError(f"No existe la operación #{record_id}.")
        return (
            f"la operación #{t.id}: {t.symbol} {t.side}, {_signed_usd(t.pnl_usd)}, cerrada el "
            f"{ctx.fmt_dt(t.closed_at)}"
        )
    if kind == "position":
        p = repo.get_open_position(record_id)
        if p is None:
            raise NotFoundError(f"No existe la posición abierta #{record_id}.")
        return (
            f"la posición abierta #{p.id}: {p.symbol} {p.side}, riesgo {fmt_usd(p.risk_usd)} "
            "(se borra sin guardar resultado; para cerrarla usá close)"
        )
    if kind == "event":
        e = repo.get_volatility_event(record_id)
        if e is None:
            raise NotFoundError(f"No existe el evento #{record_id}.")
        return f"el evento #{e.id}: «{printable(e.note)}», hasta {ctx.fmt_dt(e.until)}"
    w = next((x for x in repo.list_wallet_events() if x.id == record_id), None)
    if w is None:
        raise NotFoundError(f"No existe el movimiento de billetera #{record_id}.")
    kind_name = _KIND_NAMES.get(w.kind, w.kind).lower()
    return f"el movimiento #{w.id}: {kind_name} de {fmt_usd(w.amount_usd)}, {ctx.fmt_dt(w.ts)}"


def cmd_delete(args: argparse.Namespace, ctx: Context) -> int:
    kind = _DELETE_KINDS.get(args.kind.strip().lower())
    if kind is None:
        raise ValueError(
            f"No sé borrar {args.kind!r}: tiene que ser trade, position, event o wallet."
        )
    what = _describe_for_delete(ctx, kind, args.id)
    if not args.yes:
        try:
            answer = ctx.ask(f"Vas a borrar {what}. ¿Seguro? [s/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in _YES:
            ctx.print("No se borró nada (para confirmar sin preguntar, agregá --yes).")
            return EXIT_OK
    repo = ctx.repo
    deleted = {
        "trade": repo.delete_trade,
        "position": lambda i: repo.close_open_position(i) is not None,
        "event": repo.delete_volatility_event,
        "wallet": repo.delete_wallet_event,
    }[kind](args.id)
    if not deleted:  # lo borró otro proceso entre la pregunta y el borrado
        raise NotFoundError(f"Ya no existe: {what}.")
    ctx.print(f"Borrado: {what}.")
    _safety_lines(ctx)
    return EXIT_OK


def cmd_status(args: argparse.Namespace, ctx: Context) -> int:
    sc = evaluate_safety(ctx.repo, ctx.settings, ctx.profile, ctx.now, ctx.paths)
    lines = render_safety_section(safety_view(sc), ctx.style, ctx.tz, title=SAFETY_TITLE)
    print("\n".join(lines), file=ctx.out)
    return EXIT_OK


# --- argumentos --------------------------------------------------------------------------


def _common_options(parser: argparse.ArgumentParser) -> None:
    """Opciones de todos los comandos (sirven antes o después del comando)."""
    sup = argparse.SUPPRESS
    parser.add_argument(
        "--db",
        type=Path,
        default=sup,
        metavar="RUTA",
        help="otra base (con su propio LOCKED y sus informes al lado, con el nombre de la base)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=sup,
        help='la base de prueba de los datos sintéticos (la de make cli ARGS="--offline")',
    )
    parser.add_argument("--config", type=Path, default=sup, help="otro config.yaml")
    parser.add_argument(
        "--profile", default=sup, help="perfil de riesgo (por defecto el de config)"
    )


_YES_HELP_PNL = "confirmar un resultado muy grande (más que tu saldo o que varias veces tu riesgo)"
_YES_HELP_RISK = "confirmar un riesgo muy grande (más que tu saldo o que varias veces tu riesgo)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.trades",
        description=(
            "Carga a mano tu estado de trading (operaciones, posiciones, billetera y eventos) "
            "para los frenos de seguridad. Fechas en tu zona horaria; montos con coma o punto."
        ),
    )
    _common_options(parser)
    sub = parser.add_subparsers(dest="command", metavar="COMANDO")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        _common_options(p)
        return p

    add = command("add", "carga una operación cerrada")
    add.add_argument("--symbol", type=symbol_arg, required=True, help="moneda, ej. BTC")
    add.add_argument("--side", type=side_arg, required=True, help="long o short")
    add.add_argument(
        "--pnl",
        type=signed_amount,
        required=True,
        metavar="USD",
        help="resultado neto en USD, comisiones incluidas (negativo = pérdida, ej. -32,5)",
    )
    add.add_argument("--risk", type=positive_amount, metavar="USD", help="lo que arriesgaste")
    add.add_argument("--entry", type=positive_amount, metavar="PRECIO", help="precio de entrada")
    add.add_argument("--exit", type=positive_amount, metavar="PRECIO", help="precio de salida")
    add.add_argument("--opened", metavar="FECHA", help="cuándo abriste (por defecto, el cierre)")
    add.add_argument("--closed", metavar="FECHA", help="cuándo cerraste (por defecto, ahora)")
    add.add_argument("--note", help="nota")
    add.add_argument(
        "--analysis",
        metavar="N|auto|none",
        help="análisis que seguiste (auto, por omisión: el último de esa moneda antes de abrir)",
    )
    add.add_argument("--yes", action="store_true", help=_YES_HELP_PNL)

    op = command("open", "carga una posición abierta")
    op.add_argument("--symbol", type=symbol_arg, required=True, help="moneda, ej. BTC")
    op.add_argument("--side", type=side_arg, required=True, help="long o short")
    op.add_argument(
        "--risk",
        type=non_negative_amount,
        required=True,
        metavar="USD",
        help="lo que perdés si toca el stop (0 si el stop está en la entrada o en ganancia)",
    )
    op.add_argument("--entry", type=positive_amount, metavar="PRECIO", help="precio de entrada")
    op.add_argument("--opened", metavar="FECHA", help="cuándo abriste (por defecto, ahora)")
    op.add_argument("--note", help="nota")
    op.add_argument("--analysis", metavar="N|auto|none", help="análisis que seguiste")
    op.add_argument("--yes", action="store_true", help=_YES_HELP_RISK)

    up = command("update", "cambia una posición abierta (por ejemplo, después de mover el stop)")
    up.add_argument("id", type=_positive_int, metavar="N", help="número de la posición")
    up.add_argument(
        "--risk",
        type=non_negative_amount,
        metavar="USD",
        help="lo que perdés ahora si toca el stop (0 si lo moviste a la entrada o a ganancia)",
    )
    up.add_argument("--entry", type=positive_amount, metavar="PRECIO", help="precio de entrada")
    up.add_argument("--note", help='nota nueva ("" la borra)')
    up.add_argument("--yes", action="store_true", help=_YES_HELP_RISK)

    close = command("close", "cierra una posición abierta y la guarda como operación")
    close.add_argument(
        "id",
        nargs="?",
        type=_positive_int,
        help="número de la posición (si hay una sola, no hace falta)",
    )
    close.add_argument(
        "--pnl", type=signed_amount, required=True, metavar="USD", help="resultado neto en USD"
    )
    close.add_argument("--exit", type=positive_amount, metavar="PRECIO", help="precio de salida")
    close.add_argument("--closed", metavar="FECHA", help="cuándo cerraste (por defecto, ahora)")
    close.add_argument("--note", help="nota (si no, queda la de la posición)")
    close.add_argument("--yes", action="store_true", help=_YES_HELP_PNL)

    lst = command("list", "muestra lo cargado")
    lst.add_argument("--limit", type=_positive_int, default=10, help="cuántas operaciones mostrar")

    for name, help_text, amount_type in (
        ("deposit", "anota un depósito en la billetera de futuros", positive_amount),
        ("withdraw", "anota un retiro de la billetera de futuros", positive_amount),
        ("balance", "anota el saldo que ves en la billetera de futuros", non_negative_amount),
    ):
        w = command(name, help_text)
        w.add_argument("--amount", type=amount_type, required=True, metavar="USD", help="monto")
        w.add_argument("--date", metavar="FECHA", help="cuándo (por defecto, ahora)")
        w.add_argument("--note", help="nota")
        if name == "withdraw":
            w.add_argument(
                "--yes",
                action="store_true",
                help="confirmar un retiro más grande que tu saldo (¿un cero de más?)",
            )

    ev = command("event", "marca un evento de alta volatilidad (bloquea los planes)")
    ev.add_argument("--note", required=True, help='qué evento es, ej. "CPI de EE. UU."')
    ev.add_argument("--until", required=True, metavar="FECHA", help='hasta cuándo, ej. "11:30"')
    ev.add_argument(
        "--from", dest="start", metavar="FECHA", help="desde cuándo (por defecto, ahora)"
    )
    ev.add_argument(
        "--yes", action="store_true", help="confirmar un evento de varios días (¿año mal puesto?)"
    )

    de = command("delete", "borra algo cargado por error")
    de.add_argument("kind", metavar="QUÉ", help="trade, position, event o wallet")
    de.add_argument("id", type=_positive_int, metavar="N", help="número")
    de.add_argument("--yes", action="store_true", help="no preguntar")

    command("status", "estado de los frenos de seguridad")
    return parser


#: Comandos que solo leen (o que no tienen nada que hacer en una base vacía): con
#: una ``--db`` que no existe salen con 1 en vez de crear una base vacía (un error
#: de tipeo en la ruta mostraría todo en VERDE).
_NEEDS_EXISTING_DB = frozenset({"list", "status", "close", "update", "delete"})
# ``INVOCATION_DIR_ENV`` y ``resolve_user_path`` (rutas relativas a la carpeta
# desde donde se corrió ``make``) vienen de ``scripts.analyze_cli``: las dos
# consolas resuelven ``--db`` igual.

_HANDLERS: dict[str, Callable[[argparse.Namespace, Context], int]] = {
    "add": cmd_add,
    "open": cmd_open,
    "close": cmd_close,
    "update": cmd_update,
    "list": cmd_list,
    "deposit": cmd_wallet,
    "withdraw": cmd_wallet,
    "balance": cmd_wallet,
    "event": cmd_event,
    "delete": cmd_delete,
    "status": cmd_status,
}


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
    clock: Callable[[], datetime] | None = None,
    ask: Callable[[str], str] | None = None,
) -> int:
    """Punto de entrada. Devuelve el código de salida.

    ``clock`` (pruebas): el "ahora", con zona horaria. ``ask``: cómo preguntar
    la confirmación de ``delete`` (por defecto, ``input``).
    """
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(_join_negative_amounts(argv, ("--pnl",)))
    if args.command is None:
        parser.print_help(file=err)
        return EXIT_USAGE
    db = getattr(args, "db", None)
    offline = bool(getattr(args, "offline", False))
    config = getattr(args, "config", None)
    config = DEFAULT_CONFIG_PATH if config is None else resolve_user_path(config)
    if db is not None and offline:
        print("Usá --db o --offline, no los dos.", file=err)
        return EXIT_USAGE
    if db is not None:
        db = resolve_user_path(db)
    try:
        settings = load_settings(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"No se pudo leer la configuración ({config}): {exc}", file=err)
        return EXIT_USAGE
    try:
        profile = get_profile(getattr(args, "profile", None), settings)
    except ValueError as exc:
        print(str(exc), file=err)
        return EXIT_USAGE

    now_fn = clock or (lambda: datetime.now(UTC))
    paths = resolve_state_paths(settings, offline=offline, db=db)
    is_new = not paths.database.exists()
    if is_new and db is not None and args.command in _NEEDS_EXISTING_DB:
        print(missing_db_message(paths.database), file=err)
        return EXIT_DATA
    # Una base nueva se crea para poder validar el comando, pero solo queda si
    # el comando salió bien (si falta --yes o una fecha está mal, se borra: una
    # --db mal escrita no deja una base vacía que muestre todo en VERDE).
    new_dirs = _missing_parents(paths.database) if is_new else []
    try:
        repo = Repository.open(paths.database, clock=now_fn)
    except (OSError, ValueError, RuntimeError, SQLAlchemyError) as exc:
        print(f"No se pudo abrir la base ({display_path(paths.database)}): {exc}", file=err)
        if is_new:
            _remove_new_db(paths.database, new_dirs)
        return EXIT_DATA
    style = Style(enabled=bool(getattr(out, "isatty", lambda: False)()))
    # Con una base nueva, la salida espera a saber si el comando salió bien
    # (el aviso "Base nueva" va primero, y solo si la base queda).
    buffer = io.StringIO() if is_new else None
    ctx = Context(
        settings=settings,
        profile=profile,
        repo=repo,
        paths=paths,
        tz=_tz(settings),
        now=now_fn(),
        out=out if buffer is None else buffer,
        err=err,
        style=style,
        ask=ask or input,
    )
    code = EXIT_DATA
    try:
        code = _HANDLERS[args.command](args, ctx)
    except (NotFoundError, NeedsConfirmationError) as exc:
        print(str(exc), file=err)
        code = EXIT_DATA
    except ValueError as exc:
        print(f"Dato inválido: {exc}", file=err)
        code = EXIT_USAGE
    except LockWriteError as exc:
        # El problema es el archivo LOCKED (permisos, disco), no la base.
        print(lock_write_error_text(exc), file=err)
        code = EXIT_DATA
    except (OSError, SQLAlchemyError) as exc:
        print(f"No se pudo usar la base ({display_path(paths.database)}): {exc}", file=err)
        code = EXIT_DATA
    finally:
        # Si el comando falló, la base nueva se borra solo si de verdad quedó vacía.
        remove = buffer is not None and code != EXIT_OK and _db_is_empty(repo)
        repo.engine.dispose()
        if buffer is not None:
            if remove:
                _remove_new_db(paths.database, new_dirs)
            elif paths.database.exists():
                created = f"Base nueva: se creó {display_path(paths.database)}."
                buffer.seek(0)
                rest = buffer.read()
                buffer = io.StringIO()
                buffer.write(style.paint(created, "dim") + "\n" + rest)
            # La salida ya no puede cambiar lo guardado: si la terminal se cerró
            # (por ejemplo, con "| head"), no se hace nada más.
            with contextlib.suppress(BrokenPipeError):
                out.write(buffer.getvalue())
    return code


def _db_is_empty(repo: Repository) -> bool:
    """``True`` si la base no tiene nada cargado (``False`` ante cualquier duda: no se borra)."""
    try:
        return not (
            repo.list_trades()
            or repo.list_open_positions()
            or repo.list_wallet_events()
            or repo.list_volatility_events()
            or repo.list_safety_locks()
            or repo.list_analyses(limit=1)
        )
    except Exception:  # si no se puede leer, mejor dejarla
        return False


def _missing_parents(path: Path) -> list[Path]:
    """Las carpetas de ``path`` que todavía no existen (la base nueva las crea)."""
    missing: list[Path] = []
    folder = path.parent
    while not folder.exists() and folder != folder.parent:
        missing.append(folder)
        folder = folder.parent
    return missing


def _remove_new_db(path: Path, new_dirs: Sequence[Path]) -> None:
    """Borra una base recién creada por un comando que falló (y las carpetas que creó).

    Solo se llama si la base no existía antes de este comando y quedó vacía
    (``_db_is_empty``). Las carpetas se borran solo si quedaron vacías.
    """
    for suffix in ("", "-journal", "-wal", "-shm"):
        with contextlib.suppress(OSError):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
    for folder in new_dirs:  # de la más profunda a la más alta
        try:
            folder.rmdir()
        except OSError:
            break


if __name__ == "__main__":
    sys.exit(main())
