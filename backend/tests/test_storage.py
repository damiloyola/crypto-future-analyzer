"""Hito 5: persistencia en SQLite (``app.storage``).

Cada test usa su propia base en ``tmp_path`` y un reloj fijo, así no dependen
de la hora real ni entre sí.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.exc import StatementError

from app.scoring.confluence import ScoreInputs, compute_score
from app.storage import (
    Repository,
    StoredAnalysis,
    StoredPosition,
    StoredTrade,
    StoredWalletEvent,
    default_db_path,
    init_db,
    make_engine,
    open_db,
    session_scope,
)
from app.storage.db import SCHEMA_VERSION
from app.storage.models import ManualTrade, UTCDateTime, WalletEvent
from app.storage.repository import MAX_CLOCK_SKEW, MAX_NOTES_LENGTH
from tests.test_plan import plan_for

NOW = datetime(2026, 10, 13, 15, 0, tzinfo=UTC)
BUENOS_AIRES = timezone(timedelta(hours=-3))


class FakeClock:
    """Reloj controlable: devuelve ``now`` y se puede adelantar."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(NOW)


@pytest.fixture
def repo(tmp_path: Path, clock: FakeClock) -> Repository:
    return Repository.open(tmp_path / "test.db", clock=clock)


def ago(**kwargs: float) -> datetime:
    return NOW - timedelta(**kwargs)


def add_trade(repo: Repository, **overrides) -> StoredTrade:
    data = {
        "symbol": "BTC",
        "side": "long",
        "pnl_usd": -10.0,
        "opened_at": ago(hours=2),
        "closed_at": ago(hours=1),
    }
    data.update(overrides)
    return repo.add_trade(**data)


def save_analysis(repo: Repository, **overrides) -> StoredAnalysis:
    data = {
        "symbol": "BTC",
        "profile_name": "agresivo",
        "price": 62000.5,
        "price_source": "binance",
        "plan_status": "LONG",
        "regime_category": "alcista",
        "regime_name": "Alcista",
        "regime_confidence": 0.82,
        "regime_unstable": False,
        "score_total": 41.5,
    }
    data.update(overrides)
    return repo.save_analysis(**data)


def raw_rows(path: Path, sql: str) -> list[tuple]:
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()


# --- conexión y esquema -------------------------------------------------------------


def test_default_db_path_is_inside_data_dir(settings):
    path = default_db_path(settings)
    assert path.is_absolute()
    assert path == settings.resolve_path(settings.app.data_dir) / "analyzer.db"


def test_make_engine_creates_missing_folders(tmp_path: Path):
    db_file = tmp_path / "a" / "b" / "analyzer.db"
    engine = make_engine(db_file)
    assert db_file.parent.is_dir()
    init_db(engine)
    assert db_file.is_file()
    tables = {r[0] for r in raw_rows(db_file, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"manual_trades", "open_positions", "wallet_events", "analyses"} <= tables
    assert raw_rows(db_file, "PRAGMA user_version") == [(SCHEMA_VERSION,)]


def test_in_memory_database_for_quick_tests(clock: FakeClock, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo = Repository.open(":memory:", clock=clock)
    add_trade(repo)
    assert len(repo.list_trades()) == 1  # todas las sesiones ven la misma base
    assert list(tmp_path.iterdir()) == []  # no crea un archivo llamado ":memory:"


def test_init_db_is_idempotent_and_keeps_data(tmp_path: Path, clock: FakeClock):
    db_file = tmp_path / "x.db"
    repo = Repository.open(db_file, clock=clock)
    add_trade(repo)
    init_db(repo.engine)
    again = Repository.open(db_file, clock=clock)
    assert len(again.list_trades()) == 1


def test_init_db_refuses_a_database_from_another_schema_version(tmp_path: Path):
    db_file = tmp_path / "old.db"
    open_db(db_file).dispose()
    with sqlite3.connect(db_file) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="otra versión"):
        open_db(db_file)


def test_init_db_refuses_a_foreign_sqlite_file(tmp_path: Path):
    db_file = tmp_path / "foreign.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE recetas (id INTEGER PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="recetas"):
        open_db(db_file)


def test_init_db_completes_a_half_created_database(tmp_path: Path):
    """Versión 0 con solo tablas de la app (creación cortada): se completa."""
    db_file = tmp_path / "half.db"
    open_db(db_file).dispose()
    with sqlite3.connect(db_file) as conn:
        conn.execute("DROP TABLE analyses")
        conn.execute("PRAGMA user_version = 0")
    open_db(db_file).dispose()
    tables = {r[0] for r in raw_rows(db_file, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "analyses" in tables
    assert raw_rows(db_file, "PRAGMA user_version") == [(SCHEMA_VERSION,)]


def test_session_scope_rolls_back_on_error(tmp_path: Path):
    engine = open_db(tmp_path / "s.db")
    with pytest.raises(RuntimeError, match="boom"), session_scope(engine) as session:
        session.add(WalletEvent(ts=NOW, kind="deposit", amount_usd=100.0))
        session.flush()
        raise RuntimeError("boom")
    assert Repository(engine, clock=lambda: NOW).list_wallet_events() == []


def test_each_database_is_independent(tmp_path: Path, clock: FakeClock):
    a = Repository.open(tmp_path / "a.db", clock=clock)
    b = Repository.open(tmp_path / "b.db", clock=clock)
    add_trade(a)
    assert len(a.list_trades()) == 1
    assert b.list_trades() == []


def test_repository_works_from_another_thread(repo: Repository):
    """La API usa el mismo motor desde varios hilos (``check_same_thread=False``)."""
    add_trade(repo)
    result: list[int] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            add_trade(repo, symbol="ETH")
            result.append(len(repo.list_trades()))
        except BaseException as exc:  # pragma: no cover - solo si falla
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    assert result == [2]


# --- fechas en UTC -------------------------------------------------------------------


def test_dates_are_stored_in_utc_and_come_back_aware(repo: Repository, tmp_path: Path):
    opened = datetime(2026, 10, 13, 9, 30, 15, 123456, tzinfo=BUENOS_AIRES)  # 12:30:15 UTC
    closed = datetime(2026, 10, 13, 10, 45, tzinfo=BUENOS_AIRES)  # 13:45 UTC
    trade = add_trade(repo, opened_at=opened, closed_at=closed)

    stored = repo.get_trade(trade.id)
    assert stored is not None
    assert stored.opened_at == opened  # mismo instante
    assert stored.opened_at.tzinfo is UTC
    assert stored.opened_at.utcoffset() == timedelta(0)
    assert stored.opened_at.microsecond == 123456
    assert stored.closed_at == datetime(2026, 10, 13, 13, 45, tzinfo=UTC)
    assert stored.created_at == NOW and stored.created_at.tzinfo is UTC
    # En el archivo queda la hora UTC, sin zona.
    assert raw_rows(tmp_path / "test.db", "SELECT opened_at, closed_at FROM manual_trades") == [
        ("2026-10-13 12:30:15.123456", "2026-10-13 13:45:00.000000")
    ]
    assert stored.to_dict()["opened_at"] == "2026-10-13T12:30:15.123456+00:00"


@pytest.mark.parametrize(
    "call",
    [
        lambda r: add_trade(r, opened_at=datetime(2026, 10, 13, 10, 0)),
        lambda r: add_trade(r, closed_at=datetime(2026, 10, 13, 14, 0)),
        lambda r: r.add_open_position(
            symbol="BTC", side="long", risk_usd=10, opened_at=datetime(2026, 10, 13, 10, 0)
        ),
        lambda r: r.add_wallet_event(kind="deposit", amount_usd=100, ts=datetime(2026, 10, 1)),
        lambda r: save_analysis(r, created_at=datetime(2026, 10, 13, 14, 0)),
        lambda r: r.list_trades(since=datetime(2026, 10, 13)),
        lambda r: r.list_wallet_events(until=datetime(2026, 10, 13)),
        lambda r: r.list_analyses(since=datetime(2026, 10, 13)),
    ],
)
def test_naive_dates_are_rejected(repo: Repository, call):
    """Sin zona horaria no se sabe si era hora de Buenos Aires o UTC: se rechaza."""
    with pytest.raises(ValueError, match="zona horaria"):
        call(repo)


def test_utc_datetime_type_also_rejects_naive_dates_below_the_repository(tmp_path: Path):
    engine = open_db(tmp_path / "t.db")
    with pytest.raises(StatementError, match="zona horaria"), session_scope(engine) as session:
        session.add(WalletEvent(ts=datetime(2026, 10, 1), kind="deposit", amount_usd=1.0))
        session.flush()


def test_utc_datetime_type_converts_other_zones_to_utc():
    t = UTCDateTime()
    local = datetime(2026, 10, 13, 21, 0, tzinfo=BUENOS_AIRES)
    assert t.process_bind_param(local, None) == datetime(2026, 10, 14, 0, 0)
    assert t.process_result_value(datetime(2026, 10, 14, 0, 0), None) == datetime(
        2026, 10, 14, tzinfo=UTC
    )
    assert t.process_bind_param(None, None) is None
    assert t.process_result_value(None, None) is None
    with pytest.raises(TypeError):
        t.process_bind_param("2026-10-13", None)


def test_dates_in_the_future_are_rejected(repo: Repository):
    soon = NOW + MAX_CLOCK_SKEW - timedelta(seconds=1)
    later = NOW + MAX_CLOCK_SKEW + timedelta(seconds=1)
    add_trade(repo, opened_at=ago(hours=1), closed_at=soon)  # diferencia de reloj: se tolera
    with pytest.raises(ValueError, match="futuro"):
        add_trade(repo, closed_at=later)
    with pytest.raises(ValueError, match="futuro"):
        add_trade(repo, opened_at=datetime(2062, 10, 13, tzinfo=UTC), closed_at=None)
    with pytest.raises(ValueError, match="futuro"):
        repo.add_wallet_event(kind="balance", amount_usd=100, ts=later)
    with pytest.raises(ValueError, match="futuro"):
        repo.add_open_position(symbol="BTC", side="long", risk_usd=10, opened_at=later)


# --- operaciones cerradas -------------------------------------------------------------


def test_trade_round_trip_with_every_field(repo: Repository):
    trade = repo.add_trade(
        symbol=" btc ",
        side="LONG",
        pnl_usd=-48.75,
        opened_at=ago(hours=3),
        closed_at=ago(hours=1),
        risk_usd=50.0,
        entry_price=62000.0,
        exit_price=61400.5,
        notes="  Me saltó el stop  ",
        regime_category="Bajista",
        regime_name="Bajista",
        regime_confidence=0.71,
        score_total=-35.0,
        plan_status="short",
        profile_name="agresivo",
    )
    assert isinstance(trade, StoredTrade)
    assert trade.id >= 1
    assert (trade.symbol, trade.side, trade.notes) == ("BTC", "long", "Me saltó el stop")
    assert (trade.regime_category, trade.plan_status) == ("bajista", "SHORT")
    assert trade.is_loss

    stored = repo.get_trade(trade.id)
    assert stored == trade
    d = stored.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["pnl_usd"] == -48.75 and d["is_loss"] is True and d["risk_usd"] == 50.0
    assert d["closed_at"] == "2026-10-13T14:00:00+00:00"


def test_returned_records_are_frozen_and_survive_the_session(repo: Repository):
    trade = add_trade(repo)
    with pytest.raises(dataclasses.FrozenInstanceError):
        trade.pnl_usd = 100.0  # type: ignore[misc]
    repo.engine.dispose()  # sin conexiones abiertas: los datos siguen en el objeto
    assert trade.pnl_usd == -10.0 and trade.closed_at.tzinfo is UTC


def test_trade_defaults_and_minimal_fields(repo: Repository):
    trade = repo.add_trade(symbol="SOL", side="short", pnl_usd=0, opened_at=ago(minutes=30))
    assert trade.closed_at == NOW  # cierre por defecto: ahora
    assert trade.is_loss is False  # un empate no es pérdida
    for name in ("risk_usd", "entry_price", "exit_price", "notes", "analysis_id", "score_total"):
        assert getattr(trade, name) is None
    assert (
        repo.add_trade(
            symbol="SOL", side="short", pnl_usd=1, opened_at=ago(minutes=5), notes="   "
        ).notes
        is None
    )


def test_trade_accepts_numpy_numbers(repo: Repository):
    trade = add_trade(repo, pnl_usd=np.float64(-3.5), risk_usd=np.int64(20))
    assert trade.pnl_usd == -3.5 and type(trade.pnl_usd) is float
    assert trade.risk_usd == 20.0


def test_get_and_delete_trade(repo: Repository):
    trade = add_trade(repo)
    assert repo.delete_trade(trade.id) is True
    assert repo.get_trade(trade.id) is None
    assert repo.delete_trade(trade.id) is False
    assert repo.get_trade(999) is None


def test_list_trades_filters_by_close_time_half_open(repo: Repository):
    day_start = datetime(2026, 10, 13, 3, 0, tzinfo=UTC)  # 00:00 en Buenos Aires
    t1 = add_trade(repo, opened_at=ago(hours=20), closed_at=day_start - timedelta(seconds=1))
    t2 = add_trade(repo, opened_at=ago(hours=20), closed_at=day_start)
    t3 = add_trade(repo, symbol="ETH", opened_at=ago(hours=5), closed_at=ago(hours=4))
    t4 = add_trade(repo, opened_at=ago(hours=3), closed_at=ago(hours=1))

    assert [t.id for t in repo.list_trades()] == [t1.id, t2.id, t3.id, t4.id]
    assert [t.id for t in repo.list_trades(since=day_start)] == [t2.id, t3.id, t4.id]
    assert [t.id for t in repo.list_trades(until=day_start)] == [t1.id]
    local_start = day_start.astimezone(BUENOS_AIRES)  # misma hora, otra zona
    assert [t.id for t in repo.list_trades(since=local_start, until=ago(hours=1))] == [
        t2.id,
        t3.id,
    ]
    assert [t.id for t in repo.list_trades(symbol="eth")] == [t3.id]
    assert repo.list_trades(since=day_start, until=day_start) == []
    with pytest.raises(ValueError, match="anterior al inicio"):
        repo.list_trades(since=day_start, until=day_start - timedelta(hours=1))


def test_list_trades_newest_first_with_limit(repo: Repository):
    same_close = ago(hours=1)
    a = add_trade(repo, closed_at=ago(hours=1, minutes=30))
    b = add_trade(repo, closed_at=same_close)
    c = add_trade(repo, closed_at=same_close)  # empate en el cierre: decide el id
    assert [t.id for t in repo.list_trades(newest_first=True, limit=2)] == [c.id, b.id]
    assert [t.id for t in repo.list_trades(limit=2)] == [a.id, b.id]
    with pytest.raises(ValueError, match="límite"):
        repo.list_trades(limit=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"side": "buy"}, "Dirección desconocida"),
        ({"side": None}, "Dirección desconocida"),
        ({"symbol": "BTC/USDT"}, "Moneda inválida"),
        ({"symbol": "BTC-PERP"}, "Moneda inválida"),
        ({"symbol": "  "}, "Moneda inválida"),
        ({"symbol": "ÑANDÚ"}, "Moneda inválida"),
        ({"symbol": "X" * 21}, "Moneda inválida"),
        ({"symbol": 123}, "La moneda tiene que ser un texto"),
        ({"opened_at": ago(minutes=10), "closed_at": ago(minutes=20)}, "antes de abrir"),
        ({"risk_usd": 0}, "El riesgo tiene que ser mayor a 0"),
        ({"risk_usd": -5}, "El riesgo tiene que ser mayor a 0"),
        ({"pnl_usd": float("nan")}, "número finito"),
        ({"pnl_usd": float("inf")}, "número finito"),
        ({"pnl_usd": True}, "tiene que ser un número"),
        ({"pnl_usd": "-10"}, "tiene que ser un número"),
        ({"entry_price": 0}, "El precio de entrada"),
        ({"exit_price": -1}, "El precio de salida"),
        ({"notes": "x" * (MAX_NOTES_LENGTH + 1)}, "demasiado largo"),
        ({"regime_confidence": 1.5}, "va de 0 a 1"),
        ({"score_total": 150}, "va de -100 a \\+100"),
        ({"plan_status": "COMPRAR"}, "Estado del plan desconocido"),
        ({"regime_category": "lateral"}, "Tipo de régimen desconocido"),
        ({"opened_at": "2026-10-13T10:00:00Z"}, "tiene que ser una fecha"),
        ({"analysis_id": 0}, "entero positivo"),
        ({"analysis_id": 999}, "No existe el análisis 999"),
    ],
)
def test_trade_validation_errors(repo: Repository, overrides, message):
    with pytest.raises(ValueError, match=message):
        add_trade(repo, **overrides)
    assert repo.list_trades() == []


def test_closing_exactly_at_the_opening_is_allowed(repo: Repository):
    t = add_trade(repo, opened_at=ago(hours=1), closed_at=ago(hours=1))
    assert t.opened_at == t.closed_at


def test_trade_copies_the_context_of_the_analysis_it_followed(repo: Repository):
    analysis = save_analysis(repo, plan_status="NO_OPERAR", score_total=12.0)
    trade = add_trade(repo, analysis_id=analysis.id)
    assert trade.analysis_id == analysis.id
    assert (trade.regime_category, trade.regime_name) == ("alcista", "Alcista")
    assert trade.regime_confidence == 0.82
    assert (trade.score_total, trade.plan_status) == (12.0, "NO_OPERAR")
    assert trade.profile_name == "agresivo"

    # Lo que se pasa a mano tiene prioridad sobre el análisis.
    manual = add_trade(repo, analysis_id=analysis.id, plan_status="LONG", regime_name="Neutral")
    assert (manual.plan_status, manual.regime_name) == ("LONG", "Neutral")
    assert manual.score_total == 12.0


def test_trade_linked_to_an_analysis_of_another_symbol_is_rejected(repo: Repository):
    analysis = save_analysis(repo, symbol="BTC")
    with pytest.raises(ValueError, match="es de BTC, no de ETH"):
        add_trade(repo, symbol="ETH", analysis_id=analysis.id)


# --- posiciones abiertas --------------------------------------------------------------


def test_open_position_round_trip(repo: Repository):
    pos = repo.add_open_position(
        symbol="eth",
        side="Short",
        risk_usd=25.0,
        opened_at=ago(hours=2),
        entry_price=2400.5,
        notes="Entrada en la resistencia",
    )
    assert isinstance(pos, StoredPosition)
    assert (pos.symbol, pos.side, pos.risk_usd) == ("ETH", "short", 25.0)
    assert repo.get_open_position(pos.id) == pos
    assert repo.list_open_positions() == [pos]
    d = pos.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["opened_at"] == "2026-10-13T13:00:00+00:00"


def test_open_position_defaults_and_listing(repo: Repository):
    a = repo.add_open_position(symbol="BTC", side="long", risk_usd=50)
    assert a.opened_at == NOW and a.entry_price is None and a.analysis_id is None
    b = repo.add_open_position(symbol="SOL", side="long", risk_usd=20, opened_at=ago(hours=1))
    c = repo.add_open_position(symbol="BTC", side="short", risk_usd=10, opened_at=ago(hours=3))
    assert [p.id for p in repo.list_open_positions()] == [c.id, b.id, a.id]  # por apertura
    assert [p.id for p in repo.list_open_positions(symbol="btc")] == [c.id, a.id]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"risk_usd": 0}, "El riesgo tiene que ser mayor a 0"),
        ({"risk_usd": -10}, "El riesgo tiene que ser mayor a 0"),
        ({"risk_usd": None}, "tiene que ser un número"),
        ({"side": "sell"}, "Dirección desconocida"),
        ({"symbol": "BTC USDT"}, "Moneda inválida"),
        ({"entry_price": 0}, "El precio de entrada"),
        ({"analysis_id": 42}, "No existe el análisis 42"),
    ],
)
def test_open_position_validation_errors(repo: Repository, kwargs, message):
    data = {"symbol": "BTC", "side": "long", "risk_usd": 50.0} | kwargs
    with pytest.raises(ValueError, match=message):
        repo.add_open_position(**data)
    assert repo.list_open_positions() == []


def test_close_open_position_removes_it(repo: Repository):
    pos = repo.add_open_position(symbol="BTC", side="long", risk_usd=50)
    assert repo.close_open_position(pos.id) == pos
    assert repo.list_open_positions() == []
    assert repo.get_open_position(pos.id) is None
    assert repo.close_open_position(pos.id) is None
    assert repo.list_trades() == []  # sacarla de la lista no registra resultado


def test_close_position_with_result_creates_the_trade_atomically(
    repo: Repository, clock: FakeClock
):
    analysis = save_analysis(repo, symbol="ETH", plan_status="SHORT", score_total=-40.0)
    pos = repo.add_open_position(
        symbol="ETH",
        side="short",
        risk_usd=30.0,
        opened_at=ago(hours=2),
        entry_price=2500.0,
        notes="Short en la resistencia",
        analysis_id=analysis.id,
    )
    clock.advance(minutes=30)
    trade = repo.close_position_with_result(pos.id, pnl_usd=-31.2, exit_price=2560.0)

    assert repo.list_open_positions() == []
    assert repo.list_trades() == [trade]
    assert (trade.symbol, trade.side, trade.opened_at) == ("ETH", "short", pos.opened_at)
    assert trade.closed_at == NOW + timedelta(minutes=30)
    assert (trade.risk_usd, trade.entry_price, trade.exit_price) == (30.0, 2500.0, 2560.0)
    assert trade.notes == "Short en la resistencia"
    assert trade.analysis_id == analysis.id
    assert (trade.plan_status, trade.score_total) == ("SHORT", -40.0)

    other = repo.add_open_position(symbol="BTC", side="long", risk_usd=10, notes="vieja")
    replaced = repo.close_position_with_result(other.id, pnl_usd=5, notes="nota nueva")
    assert replaced.notes == "nota nueva"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pnl_usd": float("nan")}, "número finito"),
        ({"pnl_usd": 1.0, "closed_at": ago(hours=3)}, "antes de abrir"),
        ({"pnl_usd": 1.0, "exit_price": 0}, "El precio de salida"),
    ],
)
def test_close_position_with_result_is_all_or_nothing(repo: Repository, kwargs, message):
    pos = repo.add_open_position(symbol="BTC", side="long", risk_usd=50, opened_at=ago(hours=2))
    with pytest.raises(ValueError, match=message):
        repo.close_position_with_result(pos.id, **kwargs)
    assert repo.list_open_positions() == [pos]  # la posición sigue abierta
    assert repo.list_trades() == []


def test_close_position_with_result_of_a_missing_position(repo: Repository):
    with pytest.raises(ValueError, match="No existe la posición abierta 7"):
        repo.close_position_with_result(7, pnl_usd=1.0)


# --- billetera ----------------------------------------------------------------------


def test_wallet_events_round_trip_and_filters(repo: Repository):
    dep = repo.add_wallet_event(
        kind="deposit", amount_usd=300, ts=datetime(2026, 10, 1, tzinfo=UTC)
    )
    bal = repo.add_wallet_event(kind="Balance", amount_usd=0, ts=ago(days=2), notes="Quedé en 0")
    wd = repo.add_wallet_event(kind="withdrawal", amount_usd=50.5)
    assert isinstance(dep, StoredWalletEvent)
    assert (bal.kind, bal.amount_usd, bal.notes) == ("balance", 0.0, "Quedé en 0")
    assert wd.ts == NOW and wd.amount_usd == 50.5  # el signo lo da el tipo

    assert repo.list_wallet_events() == [dep, bal, wd]
    assert repo.list_wallet_events(kind="deposit") == [dep]
    assert repo.list_wallet_events(since=ago(days=2)) == [bal, wd]
    assert repo.list_wallet_events(until=ago(days=2)) == [dep]
    d = wd.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d == {
        "id": wd.id,
        "ts": "2026-10-13T15:00:00+00:00",
        "kind": "withdrawal",
        "amount_usd": 50.5,
        "notes": None,
    }

    assert repo.delete_wallet_event(bal.id) is True
    assert repo.delete_wallet_event(bal.id) is False
    assert repo.list_wallet_events() == [dep, wd]


@pytest.mark.parametrize(
    ("kind", "amount", "message"),
    [
        ("deposit", 0, "El monto del depósito tiene que ser mayor a 0"),
        ("deposit", -100, "El monto del depósito tiene que ser mayor a 0"),
        ("withdrawal", 0, "El monto del retiro tiene que ser mayor a 0"),
        ("withdrawal", -5, "El monto del retiro tiene que ser mayor a 0"),
        ("balance", -0.01, "El saldo no puede ser negativo"),
        ("balance", float("nan"), "número finito"),
        ("transfer", 10, "Tipo de movimiento desconocido"),
    ],
)
def test_wallet_event_validation_errors(repo: Repository, kind, amount, message):
    with pytest.raises(ValueError, match=message):
        repo.add_wallet_event(kind=kind, amount_usd=amount)
    assert repo.list_wallet_events() == []


# --- análisis -------------------------------------------------------------------------


class _FakeVerdict:
    """Cualquier objeto con ``to_dict()`` se puede guardar (como el veredicto del Hito 5)."""

    def to_dict(self):
        return {"level": "rojo", "multiplier": np.float64(0.5), "blocked": np.bool_(True)}


def test_analysis_round_trip_with_json_fields(repo: Repository, settings):
    score = compute_score(ScoreInputs(), settings)
    plan = {
        "status": "LONG",
        "entry": {"price": 62000.5, "kind": "market"},
        "take_profits": ({"name": "TP1", "price": 62500.0}, {"name": "TP2", "price": 63000.0}),
        "warnings": ["Régimen inestable: confirmá en 1h", "¡Ojo con el funding!"],
        "net_rr_tp2": float("nan"),
        "created_at": datetime(2026, 10, 13, 12, 0, tzinfo=BUENOS_AIRES),
        "position": None,
    }
    stored = save_analysis(
        repo,
        plan=plan,
        score=score,
        safety=_FakeVerdict(),
        llm={"summary": "Tendencia alcista con funding neutro", "confidence": "media"},
        inputs={"symbol": "BTC", "rsi": {"15m": np.float64(55.5), "1h": None}},
    )
    assert isinstance(stored, StoredAnalysis)
    assert stored.created_at == NOW and stored.created_at.tzinfo is UTC

    again = repo.get_analysis(stored.id)
    assert again == stored
    assert again.plan == {
        "status": "LONG",
        "entry": {"price": 62000.5, "kind": "market"},
        "take_profits": [{"name": "TP1", "price": 62500.0}, {"name": "TP2", "price": 63000.0}],
        "warnings": ["Régimen inestable: confirmá en 1h", "¡Ojo con el funding!"],
        "net_rr_tp2": None,  # NaN → null
        "created_at": "2026-10-13T12:00:00-03:00",  # fechas en ISO 8601
        "position": None,
    }
    assert again.score == json.loads(json.dumps(score.to_dict()))
    assert again.safety == {"level": "rojo", "multiplier": 0.5, "blocked": True}
    assert again.llm == {"summary": "Tendencia alcista con funding neutro", "confidence": "media"}
    assert again.inputs == {"symbol": "BTC", "rsi": {"15m": 55.5, "1h": None}}
    assert (again.regime_category, again.regime_unstable, again.score_total) == (
        "alcista",
        False,
        41.5,
    )

    d = again.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["plan"]["warnings"][1] == "¡Ojo con el funding!"
    summary = again.to_dict(include_json=False)
    assert not {"plan", "score", "safety", "llm", "inputs"} & summary.keys()
    assert summary["price"] == 62000.5 and summary["created_at"] == "2026-10-13T15:00:00+00:00"


def test_analysis_stores_a_real_trade_plan(repo: Repository, settings):
    plan = plan_for(settings)
    stored = save_analysis(
        repo,
        price=plan.entry_price,
        price_source=plan.price_source,
        plan_status=plan.status,
        plan=plan,
        score_total=plan.score_total,
    )
    expected = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    assert repo.get_analysis(stored.id).plan == expected
    assert stored.plan_status == plan.status


def test_analysis_json_is_unicode_text_in_the_file(repo: Repository, tmp_path: Path):
    save_analysis(repo, plan={"headline": "No operar: régimen bajista"})
    (text,) = raw_rows(tmp_path / "test.db", "SELECT plan_json FROM analyses")[0]
    assert "régimen" in text


def test_mutating_a_returned_dict_does_not_change_the_stored_analysis(repo: Repository):
    stored = save_analysis(repo, plan={"warnings": ["a"]})
    stored.plan["warnings"].append("b")
    stored.to_dict()["plan"]["warnings"].append("c")
    assert repo.get_analysis(stored.id).plan == {"warnings": ["a"]}


def test_analysis_minimal_fields_and_numpy_values(repo: Repository):
    stored = repo.save_analysis(
        symbol="xrp",
        profile_name="moderado",
        price=np.float64(0.5321),
        price_source="manual_nexo",
        plan_status="no_operar",
        regime_unstable=np.bool_(True),
        regime_confidence=np.float32(0.5),
    )
    assert (stored.symbol, stored.plan_status, stored.regime_unstable) == ("XRP", "NO_OPERAR", True)
    assert stored.plan is None and stored.score is None and stored.safety is None
    assert stored.llm is None and stored.inputs is None
    assert stored.regime_category is None and stored.score_total is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"price": 0}, "El precio tiene que ser mayor a 0"),
        ({"price": float("inf")}, "número finito"),
        ({"profile_name": "  "}, "El perfil no puede estar vacío"),
        ({"price_source": ""}, "La fuente del precio no puede estar vacío"),
        ({"plan_status": None}, "Estado del plan desconocido"),
        ({"plan_status": "WAIT"}, "Estado del plan desconocido"),
        ({"regime_category": "lateral"}, "Tipo de régimen desconocido"),
        ({"regime_confidence": -0.1}, "va de 0 a 1"),
        ({"regime_unstable": "no"}, "verdadero o falso"),
        ({"score_total": -101}, "va de -100 a \\+100"),
        ({"symbol": "BTC_USDT"}, "Moneda inválida"),
        ({"plan": [1, 2, 3]}, "El plan tiene que ser un dict"),
        ({"score": "45"}, "El score tiene que ser un dict"),
    ],
)
def test_analysis_validation_errors(repo: Repository, overrides, message):
    with pytest.raises(ValueError, match=message):
        save_analysis(repo, **overrides)
    assert repo.list_analyses() == []


def test_get_analysis_missing_returns_none(repo: Repository):
    assert repo.get_analysis(1) is None
    with pytest.raises(ValueError, match="entero positivo"):
        repo.get_analysis(-1)


def test_latest_analysis_and_the_time_window(repo: Repository, clock: FakeClock):
    assert repo.latest_analysis("BTC") is None
    old = save_analysis(repo, created_at=ago(hours=2))
    eth = save_analysis(repo, symbol="ETH", created_at=ago(minutes=5))
    new = save_analysis(repo, created_at=ago(minutes=10))
    tie = save_analysis(repo, created_at=ago(minutes=10))  # misma hora: gana el último guardado

    assert repo.latest_analysis("btc") == tie
    assert repo.latest_analysis("BTC", within=timedelta(minutes=15)) == tie
    assert repo.latest_analysis("BTC", within=timedelta(minutes=10)) == tie  # borde incluido
    assert repo.latest_analysis("BTC", within=timedelta(minutes=9)) is None
    assert repo.latest_analysis("ETH", within=timedelta(minutes=9)) == eth
    clock.advance(hours=3)
    assert repo.latest_analysis("BTC", within=timedelta(hours=1)) is None
    assert repo.latest_analysis("BTC") == tie
    assert {old.id, new.id} < {a.id for a in repo.list_analyses()}
    with pytest.raises(ValueError, match="lapso tiene que ser positivo"):
        repo.latest_analysis("BTC", within=timedelta(0))


def test_list_analyses_newest_first_with_filters(repo: Repository):
    a = save_analysis(repo, created_at=ago(hours=3))
    b = save_analysis(repo, symbol="SOL", created_at=ago(hours=2))
    c = save_analysis(repo, created_at=ago(hours=1))
    assert [x.id for x in repo.list_analyses()] == [c.id, b.id, a.id]
    assert [x.id for x in repo.list_analyses(limit=2)] == [c.id, b.id]
    assert [x.id for x in repo.list_analyses(symbol="BTC")] == [c.id, a.id]
    assert [x.id for x in repo.list_analyses(since=ago(hours=2))] == [c.id, b.id]
    assert [x.id for x in repo.list_analyses(until=ago(hours=2))] == [a.id]
    with pytest.raises(ValueError, match="límite"):
        repo.list_analyses(limit=0)


def test_set_analysis_safety(repo: Repository):
    stored = save_analysis(repo)
    assert stored.safety is None
    updated = repo.set_analysis_safety(stored.id, {"level": "verde", "reasons": []})
    assert updated.safety == {"level": "verde", "reasons": []}
    assert repo.get_analysis(stored.id).safety == {"level": "verde", "reasons": []}
    assert repo.set_analysis_safety(stored.id, None).safety is None
    with pytest.raises(ValueError, match="No existe el análisis 99"):
        repo.set_analysis_safety(99, {"level": "rojo"})


def test_models_can_be_used_directly_with_aware_dates(tmp_path: Path):
    """La tabla acepta fechas con zona aunque no se pase por el repositorio."""
    engine = open_db(tmp_path / "m.db")
    with session_scope(engine) as session:
        session.add(
            ManualTrade(
                symbol="BTC",
                side="long",
                opened_at=datetime(2026, 10, 13, 9, 0, tzinfo=BUENOS_AIRES),
                closed_at=datetime(2026, 10, 13, 10, 0, tzinfo=BUENOS_AIRES),
                pnl_usd=1.0,
            )
        )
    (trade,) = Repository(engine, clock=lambda: NOW).list_trades()
    assert trade.opened_at == datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
    assert trade.created_at.tzinfo is UTC
