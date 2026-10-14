"""Hito 5: archivo de lock (``app.safety.lockfile``)."""

from __future__ import annotations

import errno
import inspect
import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.safety import lockfile
from app.safety.lockfile import (
    DEFAULT_REASON,
    LOCK_HEADER_PREFIX,
    UNLOCK_INSTRUCTION,
    LockInfo,
    LockWriteError,
    build_lock_text,
    is_locked,
    parse_lock_header,
    read_lock,
    write_lock,
)

BA = "America/Argentina/Buenos_Aires"
WHEN = datetime(2026, 10, 13, 14, 5, tzinfo=UTC)
REASON = "La billetera del mes cayó 72 % desde su máximo (el tope es 70 %)."
REPORT = "data/incidents/incidente_2026-10-13_140500_UTC.md"


def _write(path: Path, **kw):
    args = {
        "reason": REASON,
        "triggered_at": WHEN,
        "report_path": REPORT,
        "trigger_code": "monthly_drawdown_lock",
        "tz": BA,
    }
    args.update(kw)
    return write_lock(path, **args)


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


# --- ida y vuelta --------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    result = _write(path)
    assert result.created is True
    assert is_locked(path)
    info = read_lock(path)
    assert info == result.lock
    assert info.exists and info.locked and info.readable
    assert info.reason == REASON
    assert info.triggered_at == WHEN
    assert info.triggered_at.tzinfo is not None
    assert info.report_path == REPORT
    assert info.trigger_code == "monthly_drawdown_lock"
    assert info.path == str(path)


def test_report_path_accepts_path_objects(tmp_path: Path) -> None:
    report = tmp_path / "incidents" / "incidente.md"
    info = _write(tmp_path / "LOCKED", report_path=report).lock
    assert info.report_path == str(report)


def test_creates_parent_folder(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sub" / "LOCKED"
    assert _write(path).created
    assert path.is_file()


def test_human_text_in_spanish(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    _write(path)
    text = path.read_text(encoding="utf-8")
    first, rest = text.split("\n", 1)
    assert first.startswith(LOCK_HEADER_PREFIX + "{")
    assert "LA APP ESTÁ BLOQUEADA" in rest
    assert REASON in rest  # por qué
    assert "martes 13 de octubre de 2026, 11:05 (hora de Buenos Aires)" in rest  # cuándo
    assert REPORT in rest  # dónde está el informe
    assert f"Cómo desbloquear: {UNLOCK_INSTRUCTION}." in rest  # cómo desbloquear
    assert UNLOCK_INSTRUCTION == "revisá el informe y borrá este archivo a mano"
    assert "no genera planes" in rest


def test_header_is_one_json_line_even_with_multiline_reason(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    _write(path, reason="Primera línea.\nSegunda línea.")
    first = path.read_text(encoding="utf-8").splitlines()[0]
    data = json.loads(first[len(LOCK_HEADER_PREFIX) :])
    assert data["version"] == 1
    assert data["reason"] == "Primera línea.\nSegunda línea."
    assert data["triggered_at"] == "2026-10-13T14:05:00+00:00"
    assert read_lock(path).reason == "Primera línea.\nSegunda línea."
    text = path.read_text(encoding="utf-8")
    assert "  Primera línea.\n  Segunda línea." in text


def test_naive_time_is_utc_and_other_zones_are_converted(tmp_path: Path) -> None:
    naive = _write(tmp_path / "a" / "LOCKED", triggered_at=datetime(2026, 10, 13, 14, 5)).lock
    assert naive.triggered_at == WHEN
    ba_time = datetime(2026, 10, 13, 11, 5, tzinfo=timezone(timedelta(hours=-3)))
    other = _write(tmp_path / "b" / "LOCKED", triggered_at=ba_time).lock
    assert other.triggered_at == WHEN
    assert other.triggered_at.utcoffset() == timedelta(0)


def test_triggered_at_must_be_a_datetime(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    with pytest.raises(TypeError, match="fecha"):
        _write(path, triggered_at="2026-10-13")
    assert not path.exists()


def test_empty_reason_gets_a_default(tmp_path: Path) -> None:
    info = _write(tmp_path / "LOCKED", reason="   ").lock
    assert info.reason == DEFAULT_REASON
    assert info.readable


def test_lock_is_written_even_without_report(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    result = _write(path, report_path=None, trigger_code=None)
    assert result.created and is_locked(path)
    assert result.lock.report_path is None
    assert result.lock.trigger_code is None
    assert "No se pudo escribir el informe" in path.read_text(encoding="utf-8")
    assert "revises el informe y borres" in result.lock.describe(BA)


def test_to_dict_is_json_safe(tmp_path: Path) -> None:
    result = _write(tmp_path / "LOCKED")
    payload = json.loads(json.dumps(result.to_dict(tz=BA)))
    assert payload["created"] is True
    lock = payload["lock"]
    assert lock["locked"] is True and lock["readable"] is True
    assert lock["triggered_at"] == "2026-10-13T14:05:00+00:00"
    assert lock["report_path"] == REPORT
    assert "13/10/2026 11:05 (hora de Buenos Aires)" in lock["message"]
    assert REASON in lock["message"]


def test_describe_messages(tmp_path: Path) -> None:
    missing = read_lock(tmp_path / "LOCKED")
    assert missing.describe() == "La app no está bloqueada."
    info = _write(tmp_path / "LOCKED").lock
    msg = info.describe(BA)
    assert msg.startswith("La app está bloqueada desde el 13/10/2026 11:05 (hora de Buenos Aires).")
    assert f"Motivo: {REASON}" in msg
    assert REPORT in msg and str(tmp_path / "LOCKED") in msg
    assert "borres a mano" in msg
    # En UTC si no se pasa zona.
    assert "13/10/2026 14:05 (UTC)" in info.describe()


# --- sin lock / lock roto ------------------------------------------------------


def test_missing_lock(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    assert is_locked(path) is False
    info = read_lock(path)
    assert info == LockInfo(exists=False, path=str(path))
    assert not info.locked
    assert info.to_dict()["locked"] is False


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"hola, lo edite a mano",
        b"LOCKED {esto no es json",
        b"LOCKED [1, 2, 3]",
        b'LOCKED "texto"',
        b"\xff\xfe\x00basura binaria\x00",
        b'LOCKED {"a": ' + b"[" * 100_000,
    ],
)
def test_corrupt_lock_is_still_locked(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "LOCKED"
    path.write_bytes(content)
    assert is_locked(path) is True
    info = read_lock(path)
    assert info.exists and info.locked
    assert info.readable is False
    assert info.reason is None and info.triggered_at is None
    msg = info.describe()
    assert "bloqueada" in msg and "no se pudo leer" in msg and "a mano" in msg
    json.dumps(info.to_dict())


def test_header_with_wrong_types_keeps_what_it_can(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    header = {"reason": 42, "triggered_at": "ayer", "report_path": ["x"], "trigger_code": "x"}
    path.write_text(LOCK_HEADER_PREFIX + json.dumps(header) + "\n", encoding="utf-8")
    info = read_lock(path)
    assert info.exists and info.readable
    assert info.reason is None and info.triggered_at is None and info.report_path is None
    assert info.trigger_code == "x"
    assert info.describe().startswith("La app está bloqueada.")


def test_hand_edited_lock_keeps_header(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    _write(path)
    text = path.read_text(encoding="utf-8")
    edited = "﻿\nNota mía: no borrar todavía\n" + text.replace("LA APP ESTÁ BLOQUEADA", "")
    path.write_text(edited, encoding="utf-8")
    info = read_lock(path)
    assert info.readable and info.reason == REASON and info.triggered_at == WHEN


def test_naive_time_in_header_is_read_as_utc(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    path.write_text(
        LOCK_HEADER_PREFIX + json.dumps({"reason": "x", "triggered_at": "2026-10-13T14:05:00"}),
        encoding="utf-8",
    )
    assert read_lock(path).triggered_at == WHEN


def test_directory_or_broken_link_counts_as_locked(tmp_path: Path) -> None:
    folder = tmp_path / "LOCKED"
    folder.mkdir()
    assert is_locked(folder)
    info = read_lock(folder)
    assert info.exists and not info.readable
    result = _write(folder)
    assert result.created is False and result.lock.exists

    link = tmp_path / "LOCKED_LINK"
    link.symlink_to(tmp_path / "no-existe")
    assert is_locked(link)
    assert read_lock(link).exists


def test_when_in_doubt_it_is_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(path, *a, **k):
        raise PermissionError(errno.EACCES, "sin permiso")

    monkeypatch.setattr(lockfile.os, "lstat", denied)
    assert is_locked(tmp_path / "LOCKED") is True


def test_unreadable_file_is_still_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "LOCKED"
    _write(path)

    def denied(*a, **k):
        raise PermissionError(errno.EACCES, "sin permiso")

    monkeypatch.setattr(lockfile, "open", denied, raising=False)
    info = read_lock(path)
    assert info.exists and not info.readable


# --- nunca pisa ----------------------------------------------------------------


def test_never_overwrites_the_first_lock(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    first = _write(path)
    before = path.read_bytes()
    second = _write(
        path,
        reason="Otro motivo",
        triggered_at=WHEN + timedelta(days=2),
        report_path="otro.md",
        trigger_code="otro",
    )
    assert first.created is True
    assert second.created is False
    assert path.read_bytes() == before  # no se agregó nada
    assert second.lock.reason == REASON
    assert second.lock.triggered_at == WHEN
    assert second.lock.report_path == REPORT
    assert _files(tmp_path) == ["LOCKED"]


def test_never_overwrites_a_corrupt_lock(tmp_path: Path) -> None:
    path = tmp_path / "LOCKED"
    path.write_text("lo escribí yo", encoding="utf-8")
    result = _write(path)
    assert result.created is False
    assert path.read_text(encoding="utf-8") == "lo escribí yo"
    assert result.lock.exists and not result.lock.readable


def test_lock_appearing_mid_write_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si otro proceso crea el lock entre el chequeo y el enlace, gana el otro."""
    path = tmp_path / "LOCKED"
    path.write_text("el otro proceso", encoding="utf-8")
    monkeypatch.setattr(os.path, "lexists", lambda p: False)
    result = _write(path)
    assert result.created is False
    assert path.read_text(encoding="utf-8") == "el otro proceso"
    assert _files(tmp_path) == ["LOCKED"]


# --- escritura atómica ---------------------------------------------------------


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    _write(tmp_path / "LOCKED")
    _write(tmp_path / "LOCKED", reason="segundo intento")
    assert _files(tmp_path) == ["LOCKED"]


def test_failed_write_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_fsync(fd):
        raise OSError(errno.EIO, "disco roto")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    with pytest.raises(OSError, match="disco roto"):
        _write(tmp_path / "LOCKED")
    assert _files(tmp_path) == []
    assert not is_locked(tmp_path / "LOCKED")


def test_works_without_hard_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_links(src, dst):
        raise OSError(errno.EPERM, "sin enlaces duros")

    monkeypatch.setattr(os, "link", no_links)
    path = tmp_path / "LOCKED"
    first = _write(path)
    assert first.created and first.lock.reason == REASON
    second = _write(path, reason="otro")
    assert second.created is False and second.lock.reason == REASON
    assert _files(tmp_path) == ["LOCKED"]


# --- puros ---------------------------------------------------------------------


def test_build_and_parse_are_inverse() -> None:
    text = build_lock_text(
        reason=REASON, triggered_at=WHEN, report_path=REPORT, trigger_code="x", tz=BA
    )
    assert parse_lock_header(text) == {
        "reason": REASON,
        "triggered_at": WHEN,
        "report_path": REPORT,
        "trigger_code": "x",
        "drawdown_pct": None,
        "peak_usd": None,
        "balance_usd": None,
        "measured_since": None,
    }
    assert parse_lock_header("sin encabezado") is None
    assert text.endswith("\n")


def test_the_header_keeps_the_numbers_of_the_trigger(tmp_path: Path) -> None:
    """Fix 7 / E2: con los números en el LOCKED, la app puede anotarlo en la base después."""
    since = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)
    text = build_lock_text(
        reason=REASON,
        triggered_at=WHEN,
        report_path=REPORT,
        trigger_code="monthly_drawdown_lock",
        drawdown_pct=73.333333333,
        peak_usd=300.0,
        balance_usd=80.0,
        measured_since=since,
    )
    header = parse_lock_header(text)
    assert header is not None
    assert header["drawdown_pct"] == pytest.approx(73.333333)
    assert (header["peak_usd"], header["balance_usd"]) == (300.0, 80.0)
    assert header["measured_since"] == since
    written = write_lock(
        tmp_path / "LOCKED",
        reason=REASON,
        triggered_at=WHEN,
        report_path=None,
        peak_usd=300.0,
        balance_usd=float("nan"),  # un número raro no rompe nada: no se guarda
    )
    assert written.lock.peak_usd == 300.0 and written.lock.balance_usd is None
    assert written.lock.to_dict()["peak_usd"] == 300.0
    bad = 'LOCKED {"reason": "x", "peak_usd": "mucho", "drawdown_pct": true}'
    assert parse_lock_header(bad)["peak_usd"] is None
    assert parse_lock_header(bad)["drawdown_pct"] is None


def test_a_disk_error_is_a_clear_lock_write_error(tmp_path: Path, monkeypatch) -> None:
    """Fix 7: un error al escribir el LOCKED dice qué pasa (y sigue siendo un OSError)."""

    def no_space(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(lockfile, "create_new_file", no_space)
    with pytest.raises(LockWriteError) as info:
        write_lock(tmp_path / "LOCKED", reason=REASON, triggered_at=WHEN, report_path=REPORT)
    err = info.value
    assert isinstance(err, OSError)
    assert err.path == str(tmp_path / "LOCKED") and err.report_path == REPORT
    text = str(err)
    assert text.startswith(f"No se pudo escribir el archivo LOCKED ({tmp_path / 'LOCKED'})")
    assert "No space left on device" in text
    assert "La app no arma planes hasta que se pueda escribir" in text
    assert f"El informe del incidente sí quedó guardado: {REPORT}." in text


def test_unknown_timezone_falls_back_to_utc() -> None:
    text = build_lock_text(reason="x", triggered_at=WHEN, report_path=None, tz="Marte/Olimpo")
    assert "martes 13 de octubre de 2026, 14:05 (UTC)" in text


# --- no hay forma de desbloquear desde el código -------------------------------


def test_module_has_no_unlock_function() -> None:
    forbidden = ("unlock", "remove", "delete", "clear", "release", "unlink", "reset", "erase")
    names = [
        name
        for name, obj in vars(lockfile).items()
        if callable(obj) and any(word in name.lower() for word in forbidden)
    ]
    assert names == []
    public_callables = [n for n in lockfile.__all__ if callable(getattr(lockfile, n))]
    assert "write_lock" in public_callables
    assert not any(any(w in n.lower() for w in forbidden) for n in public_callables)
    # Tampoco hay métodos para borrar en las clases públicas.
    for cls in (lockfile.LockInfo, lockfile.LockWriteResult):
        assert not any(any(w in name.lower() for w in forbidden) for name in dir(cls))
    # Y el código del módulo no borra archivos.
    source = inspect.getsource(lockfile)
    for call in (".unlink(", "os.remove(", "os.unlink(", "rmtree(", ".rmdir("):
        assert call not in source
