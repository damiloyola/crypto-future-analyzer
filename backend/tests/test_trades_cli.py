"""Hito 5: carga a mano del estado de trading (``scripts.trades``).

Cada test usa una base en ``tmp_path`` (``--db``) y un reloj fijo: el miércoles
14/10/2026 15:00 de Buenos Aires (18:00 UTC). El LOCKED y los informes de
``status`` quedan en la misma carpeta temporal: nunca se toca ``data/``.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from app.config import DEFAULT_CONFIG_PATH
from app.storage import Repository
from scripts import trades
from scripts.trades import main, parse_user_datetime

BA = ZoneInfo("America/Argentina/Buenos_Aires")
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=BA)


def local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=BA)


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "estado.db"


def run(db: Path, *argv: str, answer: str | None = None, now: datetime = NOW):
    """Corre el comando con la base ``db`` y el reloj fijo. Devuelve (código, salida, error)."""
    out, err = io.StringIO(), io.StringIO()

    def ask(prompt: str) -> str:
        out.write(prompt)
        if answer is None:
            raise EOFError
        return answer

    code = main([*argv, "--db", str(db)], out=out, err=err, clock=lambda: now, ask=ask)
    return code, out.getvalue(), err.getvalue()


def repo(db: Path, now: datetime = NOW) -> Repository:
    return Repository.open(db, clock=lambda: now)


# --- fechas y horas ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-13 14:30", local(13, 14, 30)),
        ("2026-10-13T14:30", local(13, 14, 30)),
        ("2026-10-13 14:30:15", local(13, 14, 30) + timedelta(seconds=15)),
        ("14:30", local(14, 14, 30)),
        ("9:05", local(14, 9, 5)),
        ("ayer 22:00", local(13, 22, 0)),
        ("Ayer a las 22:00", local(13, 22, 0)),
        ("anteayer 08:15", local(12, 8, 15)),
        ("hoy 10:00", local(14, 10, 0)),
        ("mañana 11:00", local(15, 11, 0)),
        ("13/10 14:30", local(13, 14, 30)),
        ("13/10/2026 14:30", local(13, 14, 30)),
        ("13/10/26 14:30", local(13, 14, 30)),
        ("2026-10-13T17:30:00Z", datetime(2026, 10, 13, 17, 30, tzinfo=UTC)),
        ("2026-10-13 17:30+00:00", datetime(2026, 10, 13, 17, 30, tzinfo=UTC)),
        ("ahora", NOW),
    ],
)
def test_user_dates_are_read_in_the_user_timezone(text, expected):
    parsed = parse_user_datetime(text, BA, NOW)
    assert parsed == expected
    assert parsed.utcoffset() is not None


def test_typed_hour_is_buenos_aires_not_utc():
    assert parse_user_datetime("14:30", BA, NOW).astimezone(UTC).hour == 17


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ayer", "Falta la hora en 'ayer' (ej. \"ayer 22:00\")"),
        ("2026-10-13", "Falta la hora"),
        ("25:00", "no existe"),
        ("31/02 10:00", "no existe"),
        ("el martes", "No entiendo la fecha 'el martes'. Ejemplos:"),
    ],
)
def test_bad_dates_explain_how_to_write_them(text, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse_user_datetime(text, BA, NOW)


# --- add ---------------------------------------------------------------------------------


def test_add_saves_a_closed_trade_in_utc(db):
    code, out, err = run(
        db,
        "add",
        "--symbol",
        "btcusdt",
        "--side",
        "venta",
        "--pnl",
        "-32,5",
        "--risk",
        "50",
        "--entry",
        "61.000,5",
        "--opened",
        "13:00",
        "--closed",
        "14:30",
        "--note",
        "salió mal",
    )
    assert code == 0, err
    [t] = repo(db).list_trades()
    assert (t.symbol, t.side, t.pnl_usd, t.risk_usd) == ("BTC", "short", -32.5, 50.0)
    assert t.entry_price == 61000.5 and t.notes == "salió mal"
    assert t.opened_at == datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
    assert t.closed_at == datetime(2026, 10, 14, 17, 30, tzinfo=UTC)
    assert "Operación #1 guardada: BTC short, resultado -32,50 USD (pérdida)." in out
    assert "Abierta 14/10 13:00 · cerrada 14/10 14:30 (hora de Buenos Aires)." in out
    assert "Frenos de seguridad: VERDE" in out


def test_add_defaults_close_to_now_and_open_to_the_close(db):
    code, out, _ = run(db, "add", "--symbol", "ETH", "--side", "long", "--pnl", "15")
    assert code == 0
    [t] = repo(db).list_trades()
    assert t.closed_at == NOW.astimezone(UTC) and t.opened_at == t.closed_at
    assert "resultado +15,00 USD (ganancia)" in out
    assert "Cerrada 14/10 15:00 (hora de Buenos Aires)." in out


def test_add_accepts_options_before_the_command(db, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["--db", str(db), "add", "--symbol", "SOL", "--side", "short", "--pnl=-1,25"],
        out=out,
        err=err,
        clock=lambda: NOW,
    )
    assert code == 0, err.getvalue()
    assert repo(db).list_trades()[0].pnl_usd == -1.25


def test_add_in_the_future_exits_2_with_local_times(db):
    code, out, err = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-5", "--closed", "16:00"
    )
    assert code == 2 and repo(db).list_trades() == []
    assert (
        "Dato inválido: La hora de cierre (14/10 16:00) está en el futuro: ahora son las 14/10 "
        "15:00 (hora de Buenos Aires)."
    ) in err


def test_add_closing_before_opening_exits_2(db):
    code, _, err = run(
        db,
        "add",
        "--symbol",
        "BTC",
        "--side",
        "long",
        "--pnl",
        "-5",
        "--opened",
        "12:00",
        "--closed",
        "11:00",
    )
    assert code == 2
    assert "La operación no puede cerrar (14/10 11:00) antes de abrir (14/10 12:00)." in err


@pytest.mark.parametrize(
    ("args", "where"),
    [
        (["--side", "long", "--pnl", "-5"], "--symbol"),
        (["--symbol", "BTC", "--side", "arriba", "--pnl", "-5"], "dirección inválida"),
        (["--symbol", "BTC", "--side", "long", "--pnl", "x"], "--pnl"),
        (["--symbol", "BTC", "--side", "long", "--pnl", "-5", "--risk", "-3"], "positivo"),
        (["--symbol", "B-TC", "--side", "long", "--pnl", "-5"], "moneda inválida"),
    ],
)
def test_add_bad_arguments_exit_2(db, capsys, args, where):
    with pytest.raises(SystemExit) as info:
        main(["add", *args, "--db", str(db)], clock=lambda: NOW)
    assert info.value.code == 2
    assert where in capsys.readouterr().err


# --- análisis ligado ---------------------------------------------------------------------


def _analysis(db: Path, symbol: str, created: datetime, **kw):
    r = repo(db)
    data = dict(
        symbol=symbol,
        profile_name="agresivo",
        price=60000.0,
        price_source="binance",
        plan_status="LONG",
        regime_category="alcista",
        regime_name="Alcista",
        score_total=45.0,
        created_at=created,
    )
    data.update(kw)
    return r.save_analysis(**data)


def test_add_auto_links_the_last_analysis_before_opening(db):
    _analysis(db, "BTC", local(13, 10))  # más de 24 h antes de abrir: no
    wanted = _analysis(db, "BTC", local(14, 11))
    _analysis(db, "ETH", local(14, 12, 30))  # otra moneda: no
    _analysis(db, "BTC", local(14, 13, 30))  # después de abrir: no
    code, out, err = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-20", "--opened", "13:00"
    )
    assert code == 0, err
    [t] = repo(db).list_trades()
    assert t.analysis_id == wanted.id
    assert (t.regime_category, t.regime_name, t.score_total, t.plan_status) == (
        "alcista",
        "Alcista",
        45.0,
        "LONG",
    )
    assert (
        f"Ligada al análisis #{wanted.id} del 14/10 11:00 (régimen Alcista, score +45,0, "
        "plan LONG)."
    ) in out


def test_add_auto_link_counts_an_analysis_in_the_same_minute(db):
    wanted = _analysis(db, "BTC", local(14, 13, 0) + timedelta(seconds=40))
    code, _, _ = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--opened", "13:00"
    )
    assert code == 0 and repo(db).list_trades()[0].analysis_id == wanted.id


def test_add_without_a_recent_analysis_says_so(db):
    code, out, _ = run(db, "add", "--symbol", "XRP", "--side", "long", "--pnl", "3")
    assert code == 0 and repo(db).list_trades()[0].analysis_id is None
    assert "No había un análisis de XRP en las 24 h antes de abrir" in out


def test_add_with_an_explicit_analysis_or_none(db):
    a = _analysis(db, "BTC", local(14, 11))
    code, _, _ = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--analysis", str(a.id)
    )
    assert code == 0
    code, out, _ = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--analysis", "none"
    )
    assert code == 0 and "Sin análisis ligado (lo pediste así)." in out
    assert [t.analysis_id for t in repo(db).list_trades()] == [a.id, None]


def test_add_with_a_missing_or_foreign_analysis_fails(db):
    code, _, err = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--analysis", "9"
    )
    assert code == 1 and "No existe el análisis #9." in err
    eth = _analysis(db, "ETH", local(14, 11))
    code, _, err = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--analysis", str(eth.id)
    )
    assert code == 2 and f"El análisis {eth.id} es de ETH, no de BTC." in err
    code, _, err = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "1", "--analysis", "x"
    )
    assert code == 2 and "--analysis tiene que ser un número de análisis, auto o none" in err


# --- posiciones ----------------------------------------------------------------------------


def test_open_and_close_a_position(db):
    code, out, err = run(
        db,
        "open",
        "--symbol",
        "SOL",
        "--side",
        "long",
        "--risk",
        "40",
        "--entry",
        "150,25",
        "--opened",
        "10:00",
    )
    assert code == 0, err
    [p] = repo(db).list_open_positions()
    assert (p.symbol, p.side, p.risk_usd, p.entry_price) == ("SOL", "long", 40.0, 150.25)
    assert "Posición #1 abierta: SOL long, riesgo 40,00 USD" in out
    assert 'make trade ARGS="close 1 --pnl <resultado>"' in out
    code, out, err = run(db, "close", "--pnl", "-12", "--exit", "148,5", "--closed", "12:00")
    assert code == 0, err
    assert repo(db).list_open_positions() == []
    [t] = repo(db).list_trades()
    assert (t.symbol, t.side, t.pnl_usd, t.risk_usd, t.exit_price) == (
        "SOL",
        "long",
        -12.0,
        40.0,
        148.5,
    )
    assert t.opened_at == local(14, 10).astimezone(UTC)
    assert "Posición #1 cerrada y guardada como operación #1: SOL long, resultado -12,00 USD" in out


def test_close_needs_a_number_with_several_positions(db):
    run(db, "open", "--symbol", "BTC", "--side", "long", "--risk", "30")
    run(db, "open", "--symbol", "ETH", "--side", "short", "--risk", "20")
    code, _, err = run(db, "close", "--pnl", "5")
    assert (
        code == 2 and "Tenés 2 posiciones abiertas: decime cuál (#1 BTC long, #2 ETH short)." in err
    )
    code, _, _ = run(db, "close", "2", "--pnl", "5")
    assert code == 0 and [p.id for p in repo(db).list_open_positions()] == [1]


def test_close_errors(db):
    code, _, err = run(db, "close", "--pnl", "5")
    assert code == 1 and "No tenés posiciones abiertas cargadas." in err
    run(db, "open", "--symbol", "BTC", "--side", "long", "--risk", "30", "--opened", "14:00")
    code, _, err = run(db, "close", "7", "--pnl", "5")
    assert code == 1 and "No existe la posición abierta #7." in err
    code, _, err = run(db, "close", "1", "--pnl", "5", "--closed", "13:00")
    assert code == 2 and "La posición #1 se abrió el 14/10 14:00: no puede cerrar antes" in err


# --- billetera y eventos -------------------------------------------------------------------


def test_wallet_movements(db):
    assert run(db, "balance", "--amount", "300", "--date", "2026-10-01 09:00")[0] == 0
    code, out, _ = run(db, "deposit", "--amount", "50,5", "--note", "transferencia")
    assert code == 0 and "Depósito #2 guardado: +50,50 USD (14/10 15:00)." in out
    code, out, _ = run(db, "withdraw", "--amount", "20")
    assert code == 0 and "Retiro #3 guardado: -20,00 USD (14/10 15:00)." in out
    events = repo(db).list_wallet_events()
    assert [(e.kind, e.amount_usd) for e in events] == [
        ("balance", 300.0),
        ("deposit", 50.5),
        ("withdrawal", 20.0),
    ]
    assert events[0].ts == datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def test_wallet_rejects_negative_amounts(db, capsys):
    with pytest.raises(SystemExit) as info:
        main(["deposit", "--amount=-5", "--db", str(db)], clock=lambda: NOW)
    assert info.value.code == 2
    assert "tiene que ser un monto positivo" in capsys.readouterr().err
    assert run(db, "balance", "--amount", "0")[0] == 0  # un saldo de 0 sí se puede anotar


def test_volatility_event(db):
    code, out, err = run(
        db, "event", "--note", "CPI de EE. UU.", "--until", "mañana 11:00", "--from", "mañana 09:30"
    )
    assert code == 0, err
    [e] = repo(db).list_volatility_events()
    assert e.starts_at == local(15, 9, 30).astimezone(UTC)
    assert e.until == local(15, 11).astimezone(UTC) and e.note == "CPI de EE. UU."
    assert "Evento #1 guardado: «CPI de EE. UU.», desde 15/10 09:30 hasta 15/10 11:00." in out
    code, _, err = run(db, "event", "--note", "Fed", "--until", "14:00")
    assert code == 2 and "La hora de fin del evento (14/10 14:00) ya pasó" in err
    code, _, err = run(db, "event", "--note", "Fed", "--until", "17:00", "--from", "18:00")
    assert code == 2 and "tiene que ser después del inicio" in err


def test_active_event_turns_the_brakes_red(db):
    code, out, _ = run(db, "event", "--note", "Fed", "--until", "17:00")
    assert code == 0
    assert (
        "Frenos de seguridad: ROJO · No se generan planes: hay un evento de alta volatilidad" in out
    )


# --- list y delete -----------------------------------------------------------------------


def test_list_shows_everything(db):
    run(db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-10", "--note", "una nota")
    run(db, "open", "--symbol", "ETH", "--side", "short", "--risk", "25")
    run(db, "balance", "--amount", "300")
    run(db, "event", "--note", "CPI", "--until", "mañana 10:00", "--from", "mañana 08:00")
    code, out, _ = run(db, "list")
    assert code == 0
    for title in (
        "Operaciones cerradas (últimas 10)",
        "Posiciones abiertas",
        "Billetera (últimos 10 movimientos)",
        "Eventos de alta volatilidad (activos y programados)",
    ):
        assert title in out
    assert "-10,00 USD" in out and "una nota" in out
    assert "ETH" in out and "25,00 USD" in out
    assert "Saldo anotado" in out and "300,00 USD" in out
    assert "programado" in out and "CPI" in out


def test_list_when_empty(db):
    code, out, _ = run(db, "list")
    assert code == 0
    assert "Todavía no cargaste operaciones." in out and "No hay posiciones abiertas." in out


def test_delete_asks_for_confirmation(db):
    run(db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-10")
    code, out, _ = run(db, "delete", "trade", "1", answer="n")
    assert code == 0 and "No se borró nada" in out and len(repo(db).list_trades()) == 1
    assert "Vas a borrar la operación #1: BTC long, -10,00 USD, cerrada el 14/10 15:00." in out
    code, out, _ = run(db, "delete", "trade", "1")  # sin respuesta (EOF): no borra
    assert code == 0 and len(repo(db).list_trades()) == 1
    code, out, _ = run(db, "delete", "operación", "1", answer="sí")
    assert code == 0 and "Borrado: la operación #1" in out and repo(db).list_trades() == []


def test_delete_each_kind_with_yes(db):
    run(db, "open", "--symbol", "BTC", "--side", "long", "--risk", "30")
    run(db, "event", "--note", "Fed", "--until", "17:00")
    run(db, "deposit", "--amount", "10")
    for kind in ("position", "event", "wallet"):
        code, out, err = run(db, "delete", kind, "1", "--yes")
        assert code == 0, err
        assert "Borrado:" in out
    r = repo(db)
    assert r.list_open_positions() == [] and r.list_volatility_events() == []
    assert r.list_wallet_events() == []


def test_delete_errors(db):
    code, _, err = run(db, "delete", "trade", "4", "--yes")
    assert code == 1 and "No existe la operación #4." in err
    code, _, err = run(db, "delete", "cosa", "1", "--yes")
    assert code == 2 and "No sé borrar 'cosa'" in err


# --- status y frenos ---------------------------------------------------------------------


def test_status_green(db):
    code, out, _ = run(db, "status")
    assert code == 0
    assert "Frenos de seguridad" in out and "Estado: VERDE" in out
    for label in ("Pérdida del día: bajar el riesgo", "Pérdidas seguidas: pausa", "Caída del mes"):
        assert label in out
    assert "nunca anotaste el saldo" in out or "No anotaste nunca el saldo" in out
    assert f"Base elegida con --db: {db}" in out


def test_three_losses_start_the_cooldown(db):
    for hour in ("12:00", "13:00"):
        run(db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-10", "--closed", hour)
    code, out, _ = run(
        db, "add", "--symbol", "BTC", "--side", "long", "--pnl", "-10", "--closed", "14:00"
    )
    assert code == 0
    assert (
        "Frenos de seguridad: ROJO · No se generan planes: pausa por 3 pérdidas seguidas, hasta "
        "hoy a las 18:00."
    ) in out
    code, out, _ = run(db, "status")
    assert "Estado: ROJO" in out and "BLOQUEA" in out and "3 de 3 seguidas" in out


def test_drawdown_lock_is_written_by_status_once(db):
    run(db, "balance", "--amount", "300", "--date", "2026-10-01 09:00")
    run(
        db,
        "add",
        "--symbol",
        "BTC",
        "--side",
        "long",
        "--pnl",
        "-110",
        "--closed",
        "2026-10-05 12:00",
    )
    code, out, _ = run(
        db,
        "add",
        "--symbol",
        "ETH",
        "--side",
        "long",
        "--pnl",
        "-110",
        "--closed",
        "2026-10-08 12:00",
    )
    assert code == 0
    lock = db.parent / "LOCKED"
    assert "la próxima vez que analices o mires los frenos (make safety) la app se bloquea" in out
    assert not lock.exists()  # cargar no escribe el LOCKED
    code, out, _ = run(db, "status")
    assert code == 0 and lock.exists()
    reports = list((db.parent / "incidents").glob("*.md"))
    assert len(reports) == 1
    assert "APP BLOQUEADA: no se arma ningún plan" in out
    assert "Recién se bloqueó la app" in out
    first = lock.read_text(encoding="utf-8")
    code, out, _ = run(db, "status", now=NOW + timedelta(hours=1))
    assert lock.read_text(encoding="utf-8") == first
    assert len(list((db.parent / "incidents").glob("*.md"))) == 1
    assert "Recién se bloqueó" not in out and "APP BLOQUEADA" in out


# --- opciones generales --------------------------------------------------------------------


def test_offline_uses_the_synthetic_test_database(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["regime"]["synthetic_models_dir"] = str(tmp_path / "sinteticos")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = main(
        ["deposit", "--amount", "10", "--offline", "--config", str(config)],
        out=out,
        err=err,
        clock=lambda: NOW,
    )
    assert code == 0, err.getvalue()
    assert (tmp_path / "sinteticos" / "analyzer.db").exists()


def test_db_and_offline_together_exit_2(db):
    code, _, err = run(db, "list", "--offline")
    assert code == 2 and "Usá --db o --offline, no los dos." in err


def test_no_command_prints_help(capsys):
    assert main([], clock=lambda: NOW) == 2
    assert "add" in capsys.readouterr().err


def test_unreadable_database_exits_1(tmp_path):
    bad = tmp_path / "no-es-una-base.db"
    bad.write_text("hola", encoding="utf-8")
    code, _, err = run(bad, "list")
    assert code == 1 and "No se pudo abrir la base" in err


def test_module_documents_every_command():
    for name in (
        "add",
        "open",
        "close",
        "list",
        "deposit",
        "withdraw",
        "balance",
        "event",
        "delete",
        "status",
    ):
        assert f"``{name}" in trades.__doc__
