"""Archivo de lock: el bloqueo total de la app.

Para qué sirve
    Cuando la billetera del mes cae ``circuit_breakers.monthly_drawdown_lock``
    o más desde su máximo, la capa de seguridad escribe el informe del
    incidente (``safety.incident``) y después este archivo (``app.lock_file``,
    ``data/LOCKED``). **Mientras exista, la app no genera planes.**

    No hay función para borrarlo, y es deliberado: el usuario lo borra a mano
    después de leer el informe. Lo obliga a frenar y revisar qué pasó antes de
    seguir operando.

Qué hay acá
    - ``write_lock(path, *, reason, triggered_at, report_path)``: crea el
      archivo de forma atómica. **Nunca pisa un lock existente**: si ya hay
      uno, no lo toca (se conserva el primer incidente, sin agregarle nada) y
      devuelve el que está. ``LockWriteResult.created`` dice si lo creó. Si
      no se puede escribir (permisos, disco lleno), lanza ``LockWriteError``
      (es un ``OSError``) con un mensaje claro para el usuario.
    - ``read_lock(path) -> LockInfo``: lee el lock. Tolera un archivo editado a
      mano, roto o ilegible: si existe, **sigue bloqueado** aunque no se
      entienda lo que dice (``readable=False``).
    - ``is_locked(path) -> bool``: si existe algo en ``path``. Ante la duda
      (por ejemplo, sin permiso para mirar la carpeta) responde ``True``.
    - ``build_lock_text(...)`` y ``parse_lock_header(text)``: el contenido del
      archivo, puros.

Formato del archivo
    La primera línea es para la app: ``LOCKED `` seguido de un JSON de una
    línea con ``version``, ``reason``, ``triggered_at`` (ISO, UTC),
    ``report_path``, ``trigger_code`` y, si se pasaron, los números del
    disparo (``drawdown_pct``, ``peak_usd``, ``balance_usd`` y
    ``measured_since``: desde cuándo se midió la caída, si fue desde un
    bloqueo anterior). Un LOCKED viejo sin números se lee igual. Después viene
    el texto para el usuario, en castellano: por qué se bloqueó, cuándo (en su
    zona horaria), dónde está el informe y cómo desbloquear ("revisá el
    informe y borrá este archivo a mano"). Al leer, la app busca la primera
    línea que empiece con ``LOCKED {`` (tolera renglones agregados arriba).

Decisiones
    - **Primero el informe, después el lock**: quien llama escribe el informe y
      pasa su ruta. Si el informe no se pudo escribir, ``report_path=None``: el
      lock se escribe igual (bloquear es más importante que el informe) y el
      texto lo dice.
    - **Nunca falla por los datos**: un motivo vacío pasa a "sin motivo
      informado" y una fecha sin zona horaria se toma como UTC. Solo un error de
      disco (``OSError``) o una fecha que no es fecha (``TypeError``) lanzan.
    - **Primer incidente gana**: la creación es exclusiva (``os.link``, ver
      ``incident.create_new_file``); si dos procesos bloquean a la vez, queda
      el primero.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

from app.risk.rules import json_safe
from app.safety.incident import (
    create_new_file,
    fmt_datetime,
    fmt_long_datetime,
    timezone_label,
    to_utc,
)

LOCK_HEADER_PREFIX = "LOCKED "
LOCK_FORMAT_VERSION = 1
UNLOCK_INSTRUCTION = "revisá el informe y borrá este archivo a mano"
DEFAULT_REASON = "sin motivo informado"
# Cuánto del archivo se lee para buscar el encabezado (un lock normal ocupa ~1 KB).
_MAX_READ_BYTES = 64 * 1024


def _iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat()


def _finite(value: Any) -> float | None:
    """Número finito o ``None`` (no lanza nunca)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


@dataclass(frozen=True)
class LockInfo:
    """Estado del lock.

    - ``exists``: hay algo en ``path`` (la app está bloqueada).
    - ``reason``, ``triggered_at``, ``report_path``, ``trigger_code``: lo que
      dice el encabezado (``None`` si falta o no se pudo leer).
    - ``drawdown_pct``, ``peak_usd``, ``balance_usd``, ``measured_since``: los
      números del disparo, si el LOCKED los tiene (los de versiones viejas no).
    - ``readable``: ``False`` si el archivo existe pero no tiene un encabezado
      válido (editado a mano, roto, sin permiso de lectura...). Sigue
      bloqueado igual.
    """

    exists: bool
    path: str
    reason: str | None = None
    triggered_at: datetime | None = None
    report_path: str | None = None
    trigger_code: str | None = None
    readable: bool = True
    drawdown_pct: float | None = None
    peak_usd: float | None = None
    balance_usd: float | None = None
    measured_since: datetime | None = None

    @property
    def locked(self) -> bool:
        """La app está bloqueada (es lo mismo que ``exists``)."""
        return self.exists

    def describe(self, tz: str | tzinfo | None = None) -> str:
        """Mensaje para el usuario, en castellano simple (horas en la zona ``tz``)."""
        if not self.exists:
            return "La app no está bloqueada."
        if not self.readable:
            return (
                f"La app está bloqueada: existe el archivo {self.path}, pero no se pudo leer "
                "lo que dice (¿se editó a mano?). Igual sigue bloqueada. Revisá el informe del "
                "incidente (en la carpeta de informes) y, cuando termines, borrá ese archivo "
                "a mano."
            )
        parts = ["La app está bloqueada"]
        if self.triggered_at is not None:
            parts.append(f" desde el {fmt_datetime(self.triggered_at, tz)} ({timezone_label(tz)})")
        parts.append(".")
        if self.reason:
            reason = self.reason if self.reason.endswith((".", "!", "?")) else f"{self.reason}."
            parts.append(f" Motivo: {reason}")
        report = f"el informe ({self.report_path})" if self.report_path else "el informe"
        parts.append(
            f" No va a generar planes hasta que revises {report} y borres a mano el archivo "
            f"{self.path}."
        )
        return "".join(parts)

    def to_dict(self, *, tz: str | tzinfo | None = None) -> dict[str, Any]:
        """Dict listo para JSON (fecha en ISO, UTC; ``message`` con las horas en ``tz``)."""
        return json_safe(
            {
                "locked": self.exists,
                "exists": self.exists,
                "path": self.path,
                "reason": self.reason,
                "triggered_at": _iso(self.triggered_at),
                "report_path": self.report_path,
                "trigger_code": self.trigger_code,
                "readable": self.readable,
                "drawdown_pct": self.drawdown_pct,
                "peak_usd": self.peak_usd,
                "balance_usd": self.balance_usd,
                "measured_since": _iso(self.measured_since),
                "message": self.describe(tz),
            }
        )


class LockWriteError(OSError):
    """No se pudo escribir el archivo LOCKED (permisos, disco lleno, carpeta rara...).

    Es un ``OSError``, así que un manejo genérico de errores de disco lo
    atrapa igual; las consolas lo atrapan aparte para decir qué pasa. Trae
    ``path`` (dónde se quiso escribir), ``report_path`` (el informe del
    incidente, si quedó escrito) y ``reason`` (el error original, en texto).
    El mensaje (``str(error)``) ya está listo para mostrar.
    """

    def __init__(
        self,
        path: Path | str,
        reason: BaseException | str,
        *,
        report_path: Path | str | None = None,
    ) -> None:
        self.path = str(path)
        # De un OSError alcanza con su explicación ("Permission denied"): la
        # ruta ya va en el mensaje.
        detail = getattr(reason, "strerror", None) if isinstance(reason, OSError) else None
        self.reason = str(detail or reason).strip() or type(reason).__name__
        self.report_path = None if report_path is None else str(report_path)
        report = (
            f" El informe del incidente sí quedó guardado: {self.report_path}."
            if self.report_path
            else ""
        )
        super().__init__(
            f"No se pudo escribir el archivo LOCKED ({self.path}): {self.reason}. La app no arma "
            "planes hasta que se pueda escribir: revisá los permisos de esa carpeta y el espacio "
            f"en disco, y volvé a correr la app.{report}"
        )

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "No se pudo escribir el archivo LOCKED."


@dataclass(frozen=True)
class LockWriteResult:
    """Resultado de ``write_lock``: si se creó ahora y el lock que quedó vigente."""

    created: bool
    lock: LockInfo

    def to_dict(self, *, tz: str | tzinfo | None = None) -> dict[str, Any]:
        """Dict listo para JSON."""
        return {"created": self.created, "lock": self.lock.to_dict(tz=tz)}


# --- contenido (puro) -----------------------------------------------------------


def build_lock_text(
    *,
    reason: str | None,
    triggered_at: datetime,
    report_path: Path | str | None,
    trigger_code: str | None = None,
    tz: str | tzinfo | None = None,
    drawdown_pct: float | None = None,
    peak_usd: float | None = None,
    balance_usd: float | None = None,
    measured_since: datetime | None = None,
) -> str:
    """Contenido del archivo LOCKED: encabezado para la app + texto para el usuario.

    Los números del disparo (opcionales) van solo en el encabezado: sirven
    para anotar el bloqueo en la base si no quedó anotado.
    """
    if not isinstance(triggered_at, datetime):
        raise TypeError(
            f"triggered_at tiene que ser una fecha (datetime), llegó {type(triggered_at).__name__}"
        )
    when = to_utc(triggered_at)
    assert when is not None
    why = _clean_text(reason) or DEFAULT_REASON
    report = None if report_path is None else (str(report_path).strip() or None)
    header = {
        "version": LOCK_FORMAT_VERSION,
        "reason": why,
        "triggered_at": when.isoformat(),
        "report_path": report,
        "trigger_code": _clean_text(trigger_code),
    }
    numbers = {"drawdown_pct": drawdown_pct, "peak_usd": peak_usd, "balance_usd": balance_usd}
    for key, value in numbers.items():
        v = _finite(value)
        if v is not None:
            header[key] = round(v, 6)
    since = to_utc(measured_since)
    if since is not None:
        header["measured_since"] = since.isoformat()
    rule = "=" * 64
    why_lines = "\n".join(f"  {line.strip()}" for line in why.splitlines() if line.strip())
    report_lines = (
        f"  {report}"
        if report
        else (
            "  No se pudo escribir el informe. Revisá la carpeta de informes y el\n"
            "  historial de operaciones de la app."
        )
    )
    lines = [
        LOCK_HEADER_PREFIX + json.dumps(header, ensure_ascii=False),
        "",
        rule,
        "LA APP ESTÁ BLOQUEADA",
        rule,
        "",
        "Mientras exista este archivo, la app no genera planes de operación.",
        "",
        "Por qué se bloqueó:",
        why_lines,
        "",
        "Cuándo:",
        f"  {fmt_long_datetime(when, tz)} ({timezone_label(tz)})",
        "",
        "Informe del incidente (qué operaciones, en qué régimen, qué decía el",
        "score y qué falló):",
        report_lines,
        "",
        f"Cómo desbloquear: {UNLOCK_INSTRUCTION}.",
        "",
        "La app no tiene ningún botón ni comando para desbloquearse, y es a",
        "propósito: te obliga a frenar y revisar qué pasó antes de seguir.",
        "",
        "(La primera línea es para que la app lea los datos; no hace falta tocarla.)",
        "",
    ]
    return "\n".join(lines)


def parse_lock_header(text: str) -> dict[str, Any] | None:
    """Datos del encabezado (``reason``, ``triggered_at``, ``report_path``, ``trigger_code``...).

    Busca la primera línea que empiece con ``LOCKED {`` y lee el JSON. Cada
    campo con un tipo inválido queda en ``None``; ``triggered_at`` y
    ``measured_since`` vuelven como fecha en UTC; los números del disparo,
    como ``float`` (``None`` si faltan: un LOCKED viejo). ``None`` si no hay
    un encabezado válido.
    """
    for raw in text.splitlines():
        line = raw.lstrip("﻿").strip()
        if not line.startswith(LOCK_HEADER_PREFIX + "{"):
            continue
        try:
            data = json.loads(line[len(LOCK_HEADER_PREFIX) :])
        except (ValueError, RecursionError):
            return None
        if not isinstance(data, dict):
            return None
        return {
            "reason": _clean_text(data.get("reason")),
            "triggered_at": to_utc(data.get("triggered_at")),
            "report_path": _clean_text(data.get("report_path")),
            "trigger_code": _clean_text(data.get("trigger_code")),
            "drawdown_pct": _header_number(data.get("drawdown_pct")),
            "peak_usd": _header_number(data.get("peak_usd")),
            "balance_usd": _header_number(data.get("balance_usd")),
            "measured_since": to_utc(data.get("measured_since"))
            if isinstance(data.get("measured_since"), str)
            else None,
        }
    return None


def _header_number(value: Any) -> float | None:
    """Número del encabezado (un texto o un booleano no cuentan)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return _finite(value)


# --- disco ----------------------------------------------------------------------


def is_locked(path: Path | str) -> bool:
    """``True`` si existe algo en ``path`` (archivo, carpeta o enlace, aunque esté roto).

    Si no se puede averiguar (por ejemplo, sin permiso para mirar la carpeta),
    responde ``True``: ante la duda, bloqueado.
    """
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def read_lock(path: Path | str) -> LockInfo:
    """Lee el lock. Si existe pero no se entiende, sigue bloqueado (``readable=False``)."""
    where = str(path)
    if not is_locked(path):
        return LockInfo(exists=False, path=where)
    try:
        with open(path, "rb") as fh:
            raw = fh.read(_MAX_READ_BYTES)
    except OSError:
        return LockInfo(exists=True, path=where, readable=False)
    header = parse_lock_header(raw.decode("utf-8", errors="replace"))
    if header is None:
        return LockInfo(exists=True, path=where, readable=False)
    return LockInfo(exists=True, path=where, readable=True, **header)


def write_lock(
    path: Path | str,
    *,
    reason: str | None,
    triggered_at: datetime,
    report_path: Path | str | None,
    trigger_code: str | None = None,
    tz: str | tzinfo | None = None,
    drawdown_pct: float | None = None,
    peak_usd: float | None = None,
    balance_usd: float | None = None,
    measured_since: datetime | None = None,
) -> LockWriteResult:
    """Crea el archivo LOCKED (atómico, sin pisar nunca uno existente).

    - ``reason``: por qué se bloqueó (texto del breaker, en castellano).
    - ``triggered_at``: cuándo saltó (se guarda en UTC; sin zona = UTC).
    - ``report_path``: dónde quedó el informe del incidente (``None`` si no se
      pudo escribir; el lock se escribe igual).
    - ``trigger_code``: código del breaker (opcional, para la app).
    - ``tz``: zona horaria del usuario para la hora que se muestra en el texto
      (la de ``app.user_timezone``).

    - ``drawdown_pct`` / ``peak_usd`` / ``balance_usd`` / ``measured_since``:
      los números del disparo (opcionales, van en el encabezado).

    Si ya hay un lock, **no lo toca**: se conserva el primer incidente (sin
    agregarle nada) y se devuelve ``created=False`` con el lock que está.
    Crea la carpeta si hace falta. Un error de disco se propaga como
    ``LockWriteError`` (un ``OSError`` con un mensaje claro): quien llama
    tiene que enterarse de que el bloqueo no quedó escrito.
    """
    text = build_lock_text(
        reason=reason,
        triggered_at=triggered_at,
        report_path=report_path,
        trigger_code=trigger_code,
        tz=tz,
        drawdown_pct=drawdown_pct,
        peak_usd=peak_usd,
        balance_usd=balance_usd,
        measured_since=measured_since,
    )
    try:
        created = create_new_file(path, text)
    except LockWriteError:
        raise
    except OSError as exc:
        raise LockWriteError(path, exc, report_path=report_path) from exc
    return LockWriteResult(created=created, lock=read_lock(path))


__all__ = [
    "DEFAULT_REASON",
    "LOCK_FORMAT_VERSION",
    "LOCK_HEADER_PREFIX",
    "UNLOCK_INSTRUCTION",
    "LockInfo",
    "LockWriteError",
    "LockWriteResult",
    "build_lock_text",
    "is_locked",
    "parse_lock_header",
    "read_lock",
    "write_lock",
]
