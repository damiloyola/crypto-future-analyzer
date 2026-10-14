"""Hito 5: informe del incidente (``app.safety.incident``)."""

from __future__ import annotations

import errno
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.safety import incident
from app.safety.incident import (
    OBS_AGAINST_PLAN,
    OBS_AGAINST_REGIME,
    OBS_DAILY_LOSS_LIMIT,
    OBS_DEPOSIT_AFTER_LOSS,
    OBS_LOSING_STREAK,
    OBS_LOSS_ABOVE_RISK,
    OBS_MISSING_RISK,
    OBS_NO_ANALYSIS_CONTEXT,
    OBS_TOO_MANY_TRADES,
    OBS_TRADED_ON_NO_TRADE,
    OBS_WIDENED_STOP,
    IncidentPosition,
    IncidentProfile,
    IncidentTrade,
    IncidentTrigger,
    IncidentWalletEvent,
    build_incident_report,
    create_new_file,
    drawdown_over_100,
    find_observations,
    fmt_date,
    fmt_datetime,
    fmt_drawdown_pct,
    fmt_long_datetime,
    fmt_month,
    fmt_plan_status,
    fmt_pnl,
    fmt_regime,
    fmt_side,
    incident_file_name,
    losing_streaks,
    md_cell,
    negative_balance_warning,
    normalize_plan_status,
    resolve_timezone,
    sorted_trades,
    summarize_trades,
    timezone_label,
    to_utc,
    write_incident_report,
)

BA = "America/Argentina/Buenos_Aires"
T0 = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)  # lunes 5/10, 12:00 en Buenos Aires
WHEN = datetime(2026, 10, 13, 14, 5, 2, tzinfo=UTC)

TRIGGER = {
    "code": "monthly_drawdown_lock",
    "description": "La billetera del mes cayó 72 % desde su máximo (el tope es 70 %)",
    "wallet_peak_usd": 1000.0,
    "current_balance_usd": 280.0,
    "threshold_pct": 70.0,
}

PROFILE = {
    "name": "Agresivo",
    "monthly_wallet_usd": 1000.0,
    "risk_per_trade_usd": 50.0,
    "max_daily_loss_usd": 150.0,
    "max_trades_per_day": 3,
}


def trade(**kw: Any) -> dict[str, Any]:
    """Operación "limpia" (ganadora, con análisis, a favor del plan y del régimen)."""
    closed = kw.pop("closed_at", T0)
    base: dict[str, Any] = {
        "symbol": "BTC",
        "side": "long",
        "opened_at": None if closed is None else closed - timedelta(hours=1),
        "closed_at": closed,
        "pnl_usd": 30.0,
        "risk_usd": 50.0,
        "regime_category": "alcista",
        "regime_name": "Alcista",
        "regime_confidence": 0.8,
        "score_total": 45.0,
        "plan_status": "LONG",
        "notes": None,
        "analysis_id": 1,
    }
    base.update(kw)
    return base


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def codes(observations) -> list[str]:
    return [o.code for o in observations]


def obs(observations, code: str):
    found = [o for o in observations if o.code == code]
    assert len(found) == 1, codes(observations)
    return found[0]


def report(**kw: Any) -> str:
    args: dict[str, Any] = {
        "triggered_at": WHEN,
        "trigger": TRIGGER,
        "trades": [],
        "tz": BA,
        "lock_path": "data/LOCKED",
    }
    args.update(kw)
    return build_incident_report(**args)


# --- formato --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (30, "+30 USD"),
        (-45.5, "-45,50 USD"),
        (0, "0 USD"),
        (-0.001, "0 USD"),
        (1234.5, "+1.234,50 USD"),
        (-12345.678, "-12.345,68 USD"),
        (None, "—"),
        (float("nan"), "—"),
        ("abc", "—"),
    ],
)
def test_fmt_pnl(value: Any, expected: str) -> None:
    assert fmt_pnl(value) == expected


def test_dates_in_user_timezone() -> None:
    late = datetime(2026, 10, 13, 2, 30, tzinfo=UTC)  # 23:30 del 12/10 en Buenos Aires
    assert fmt_date(late, BA) == "12/10/2026"
    assert fmt_datetime(late, BA) == "12/10/2026 23:30"
    assert fmt_long_datetime(late, BA) == "lunes 12 de octubre de 2026, 23:30"
    assert fmt_month(late, BA) == "octubre de 2026"
    assert fmt_datetime(late) == "13/10/2026 02:30"  # sin zona = UTC
    assert fmt_long_datetime(datetime(2026, 1, 3, 9, 5, tzinfo=UTC)) == (
        "sábado 3 de enero de 2026, 09:05"
    )
    assert fmt_month(datetime(2026, 9, 1, tzinfo=UTC)) == "septiembre de 2026"


def test_dates_tolerate_naive_text_and_missing() -> None:
    assert fmt_datetime(datetime(2026, 10, 13, 14, 5), BA) == "13/10/2026 11:05"
    assert fmt_datetime("2026-10-13T14:05:00Z", BA) == "13/10/2026 11:05"
    for fn in (fmt_date, fmt_datetime, fmt_long_datetime, fmt_month):
        assert fn(None, BA) == "—"
        assert fn("no es fecha", BA) == "—"


def test_to_utc() -> None:
    assert to_utc(datetime(2026, 10, 13, 14, 5)) == datetime(2026, 10, 13, 14, 5, tzinfo=UTC)
    ba = datetime(2026, 10, 13, 11, 5, tzinfo=timezone(timedelta(hours=-3)))
    assert to_utc(ba) == datetime(2026, 10, 13, 14, 5, tzinfo=UTC)
    assert to_utc(ba).utcoffset() == timedelta(0)
    assert to_utc("2026-10-13T14:05:00+00:00") == datetime(2026, 10, 13, 14, 5, tzinfo=UTC)
    assert to_utc("ayer") is None and to_utc(123) is None and to_utc(None) is None


def test_timezones() -> None:
    assert timezone_label(BA) == "hora de Buenos Aires"
    assert timezone_label("Europe/Madrid") == "hora de Madrid"
    assert timezone_label("UTC") == "UTC"
    assert timezone_label(None) == "UTC"
    assert timezone_label("Marte/Olimpo") == "UTC"
    assert timezone_label(timezone(timedelta(hours=-3))) == "UTC-03:00"
    assert resolve_timezone("Marte/Olimpo") is UTC
    assert resolve_timezone("") is UTC
    assert resolve_timezone(UTC) is UTC


def test_side_plan_and_regime_labels() -> None:
    assert fmt_side("long") == "Long"
    assert fmt_side(" SHORT ") == "Short"
    assert fmt_side(None) == "—"
    assert fmt_side("cobertura") == "cobertura"
    assert normalize_plan_status("No operar") == "NO_OPERAR"
    assert normalize_plan_status("no_operar") == "NO_OPERAR"
    assert normalize_plan_status("long") == "LONG"
    assert normalize_plan_status("  ") is None
    assert normalize_plan_status("raro") == "raro"
    assert fmt_plan_status("NO_OPERAR") == '"No operar"'
    assert fmt_plan_status("SHORT") == "Short"
    assert fmt_plan_status(None) == "Sin análisis"
    assert fmt_regime("bajista", None, 0.78) == "Bajista (78 %)"
    assert fmt_regime("crash") == "Crash (caída fuerte)"
    assert fmt_regime("alcista", "Alcista fuerte", 0.5) == "Alcista fuerte (50 %)"
    assert fmt_regime("neutral", None, 64) == "Neutral (64 %)"  # ya en porcentaje
    assert fmt_regime(None, None, 0.9) == "—"
    assert fmt_regime("raro", None, float("nan")) == "raro"


def test_md_cell() -> None:
    assert md_cell("a|b\nc") == "a\\|b c"
    assert md_cell("  x   y  ") == "x y"
    assert md_cell(None) == "—"
    assert md_cell("") == "—"
    assert md_cell("c:\\datos") == "c:\\\\datos"


# --- datos normalizados ---------------------------------------------------------


@dataclass(frozen=True)
class FakeStoredTrade:
    """Como ``storage.repository.StoredTrade`` (solo los atributos que se leen)."""

    symbol: str
    side: str
    opened_at: datetime
    closed_at: datetime
    pnl_usd: float
    risk_usd: float | None
    regime_category: str | None
    regime_name: str | None
    regime_confidence: float | None
    score_total: float | None
    plan_status: str | None
    notes: str | None
    analysis_id: int | None


def test_trade_from_object_and_dict() -> None:
    obj = FakeStoredTrade(**trade(symbol="eth", side="SHORT", plan_status="No operar"))
    t = IncidentTrade.from_value(obj)
    assert t.symbol == "ETH" and t.side == "short" and t.plan_status == "NO_OPERAR"
    assert t.closed_at == T0
    assert IncidentTrade.from_value(t) is t
    d = IncidentTrade.from_value(trade(closed_at=datetime(2026, 10, 5, 15, 0)))
    assert d.closed_at == T0  # sin zona = UTC


def test_trade_tolerates_junk() -> None:
    t = IncidentTrade.from_value(
        {
            "symbol": None,
            "side": "",
            "closed_at": "no",
            "pnl_usd": "abc",
            "risk_usd": float("inf"),
            "regime_confidence": True,
            "analysis_id": "x",
        }
    )
    assert t.symbol == "—" and t.side is None and t.closed_at is None
    assert t.pnl_usd is None and t.risk_usd is None and t.regime_confidence is None
    assert t.analysis_id is None
    assert not t.is_loss and not t.is_win and t.missing_risk and not t.has_context


def test_trigger_computes_drawdown() -> None:
    trig = IncidentTrigger.from_value(TRIGGER)
    assert trig.drawdown_pct == pytest.approx(72.0)
    explicit = IncidentTrigger("x", "y", 1000, 280, drawdown_pct=71.5)
    assert explicit.drawdown_pct == 71.5
    empty = IncidentTrigger.from_value(None)
    assert empty.code == "desconocido" and empty.drawdown_pct is None
    nan = IncidentTrigger.from_value({"code": "x", "wallet_peak_usd": float("nan")})
    assert nan.wallet_peak_usd is None


def test_to_dicts_are_json_safe() -> None:
    t = IncidentTrade.from_value(trade(pnl_usd=-60, plan_status="NO_OPERAR"))
    d = json.loads(json.dumps(t.to_dict()))
    assert d["closed_at"] == "2026-10-05T15:00:00+00:00"
    assert 'el plan decía "No operar"' in d["flags"]
    json.dumps(IncidentTrigger.from_value(TRIGGER).to_dict())
    json.dumps(IncidentWalletEvent.from_value({"ts": T0, "kind": "deposit"}).to_dict())
    json.dumps(IncidentPosition.from_value({"symbol": "sol", "risk_usd": 20}).to_dict())
    json.dumps(IncidentProfile.from_value(PROFILE).to_dict())
    summary = summarize_trades([trade(pnl_usd=-10), trade(pnl_usd=20, closed_at=at(1))])
    payload = json.loads(json.dumps(summary.to_dict()))
    assert payload["worst_trade"]["pnl_usd"] == -10
    observations = find_observations([trade(plan_status="NO_OPERAR")])
    json.dumps([o.to_dict() for o in observations])


# --- resumen y rachas -----------------------------------------------------------


def test_summarize_trades() -> None:
    trades = [
        trade(pnl_usd=-40, closed_at=at(3)),
        trade(pnl_usd=30, closed_at=at(0)),
        trade(pnl_usd=-60, closed_at=at(1)),
        trade(pnl_usd=-20, closed_at=at(2)),
        trade(pnl_usd=0, closed_at=at(4)),
        trade(pnl_usd=None, closed_at=at(5)),
    ]
    s = summarize_trades(trades)
    assert (s.count, s.wins, s.losses, s.breakeven) == (6, 1, 3, 1)
    assert s.net_pnl_usd == -90
    assert s.gross_profit_usd == 30 and s.gross_loss_usd == -120
    assert s.worst_trade is not None and s.worst_trade.pnl_usd == -60
    assert s.best_trade is not None and s.best_trade.pnl_usd == 30
    assert s.longest_losing_streak == 3
    assert s.win_rate_pct == pytest.approx(20.0)


def test_summarize_empty() -> None:
    s = summarize_trades([])
    assert s.count == 0 and s.net_pnl_usd == 0 and s.worst_trade is None
    assert s.win_rate_pct is None and s.longest_losing_streak == 0


def test_sorted_by_close_time() -> None:
    trades = [trade(closed_at=at(2)), trade(closed_at=None), trade(closed_at=at(1))]
    assert [t.closed_at for t in sorted_trades(trades)] == [at(1), at(2), None]


def test_losing_streaks() -> None:
    pnls = [-10, -10, -10, 5, -10, -10, 0, -10, -10, -10, -10]
    trades = [trade(pnl_usd=p, closed_at=at(i)) for i, p in enumerate(pnls)]
    assert losing_streaks(trades, 3) == [(1, 2, 3), (8, 9, 10, 11)]
    assert losing_streaks(trades, 4) == [(8, 9, 10, 11)]
    assert losing_streaks(trades, 5) == []
    with pytest.raises(ValueError):
        losing_streaks(trades, 0)


# --- observaciones --------------------------------------------------------------


def test_clean_trades_have_no_observations() -> None:
    trades = [trade(closed_at=at(i * 24)) for i in range(3)]
    assert find_observations(trades, profile=PROFILE) == ()
    text = report(trades=trades, profile=PROFILE)
    assert "no encontró ningún patrón automático" in text


def test_detects_trades_on_no_trade() -> None:
    trades = [
        trade(closed_at=at(0)),
        trade(plan_status="NO_OPERAR", pnl_usd=-45.5, closed_at=at(1)),
        trade(plan_status="No operar", pnl_usd=-20, closed_at=at(2)),
    ]
    o = obs(find_observations(trades), OBS_TRADED_ON_NO_TRADE)
    assert "2 operaciones se hicieron" in o.detail
    assert "-65,50 USD" in o.detail
    assert o.items[0].startswith("#2 BTC long, 05/10/2026 16:00 (-45,50 USD)")
    assert o.items[1].startswith("#3 ")
    assert o.checklist_item and "No operar" in o.checklist_item


def test_detects_against_plan() -> None:
    trades = [trade(side="short", plan_status="LONG", pnl_usd=-30), trade(closed_at=at(1))]
    o = obs(find_observations(trades), OBS_AGAINST_PLAN)
    assert o.items == ("#1 BTC short, 05/10/2026 15:00 (-30 USD)",)
    assert "1 operación el plan decía long" in o.detail


def test_detects_against_regime() -> None:
    trades = [
        trade(side="long", regime_category="crash", plan_status="LONG", closed_at=at(0)),
        trade(side="short", regime_category="euforia", plan_status="SHORT", closed_at=at(1)),
        trade(side="long", regime_category="bajista", plan_status="LONG", closed_at=at(2)),
        trade(side="short", regime_category="neutral", plan_status="SHORT", closed_at=at(3)),
        trade(side="long", regime_category="alcista", plan_status="LONG", closed_at=at(4)),
    ]
    o = obs(find_observations(trades), OBS_AGAINST_REGIME)
    assert [i.split()[0] for i in o.items] == ["#1", "#2", "#3"]
    assert "3 operaciones fueron contra el régimen" in o.detail


def test_detects_losses_bigger_than_risk() -> None:
    trades = [
        trade(
            pnl_usd=-95,
            risk_usd=50,
            side="short",
            symbol="ETH",
            plan_status="SHORT",
            regime_category="bajista",
            closed_at=at(0),
        ),
        trade(pnl_usd=-50.004, risk_usd=50, closed_at=at(1)),  # a centavos es igual
        trade(pnl_usd=-50, risk_usd=50, closed_at=at(2)),
        trade(pnl_usd=-80, risk_usd=None, closed_at=at(3)),  # sin riesgo: no se sabe
    ]
    found = find_observations(trades, tz=BA)
    o = obs(found, OBS_LOSS_ABOVE_RISK)
    assert len(o.items) == 1
    assert o.items[0] == (
        "#1 ETH short, 05/10/2026 12:00: perdiste 95 USD con un riesgo previsto de 50 USD "
        "(1,9 veces lo previsto)"
    )
    assert "Pérdida de más, en total: 45 USD." in o.detail


def test_detects_missing_risk() -> None:
    """Riesgo 0 no "falta": es el stop en la entrada (revisión 6)."""
    trades = [trade(risk_usd=None), trade(risk_usd=0, closed_at=at(1)), trade(closed_at=at(2))]
    o = obs(find_observations(trades), OBS_MISSING_RISK)
    assert [i.split()[0] for i in o.items] == ["#1"]
    at_entry = IncidentTrade.from_value(trades[1])
    assert not at_entry.missing_risk
    assert at_entry.risk_text() == "0 USD (stop en la entrada)"
    assert "sin riesgo anotado" not in at_entry.flags()


def test_a_moved_stop_keeps_the_planned_risk() -> None:
    """Revisión 6: el riesgo previsto es el de la apertura aunque después se movió el stop."""
    widened = trade(pnl_usd=-80, risk_usd=80, initial_risk_usd=50, closed_at=at(1))
    to_entry = trade(symbol="ETH", pnl_usd=-20, risk_usd=0, initial_risk_usd=50, closed_at=at(2))
    observations = find_observations([widened, to_entry], tz=BA)
    found = codes(observations)
    assert OBS_MISSING_RISK not in found
    loss = obs(observations, OBS_LOSS_ABOVE_RISK)
    assert loss.items == (
        "#1 BTC long, 05/10/2026 13:00: perdiste 80 USD con un riesgo previsto de 50 USD "
        "(1,6 veces lo previsto)",
    )
    wide = obs(observations, OBS_WIDENED_STOP)
    assert wide.items == (
        "#1 BTC long, 05/10/2026 13:00: el riesgo previsto era 50 USD y lo agrandaste a "
        "80 USD (-80 USD)",
    )
    assert "Revisé por qué agrandé el stop" in (wide.checklist_item or "")
    first, second = (IncidentTrade.from_value(t) for t in (widened, to_entry))
    assert first.flags() == ("perdió más que el riesgo previsto", "agrandaste el stop")
    assert first.risk_text() == "50 USD (lo agrandaste a 80 USD)"
    assert first.planned_risk_usd == 50 and first.widened_stop_usd == 30
    assert second.flags() == ("moviste el stop a la entrada",)
    assert second.risk_text() == "50 USD (moviste el stop a la entrada)"
    assert second.loss_beyond_risk_usd is None
    text = report(trades=[widened, to_entry])
    assert "| 50 USD (lo agrandaste a 80 USD) |" in text
    assert "| 50 USD (moviste el stop a la entrada) |" in text
    assert "### Agrandaste el stop" in text
    assert "sin riesgo anotado" not in text
    d = first.to_dict()
    assert (d["initial_risk_usd"], d["planned_risk_usd"], d["risk_usd"]) == (50, 50, 80)
    pos = IncidentPosition.from_value({"symbol": "sol", "risk_usd": 0, "initial_risk_usd": 40})
    assert pos.risk_text() == "0 USD (previsto 40 USD)"


def test_detects_losing_streak() -> None:
    trades = [trade(pnl_usd=-10, closed_at=at(i)) for i in range(3)]
    trades.append(trade(pnl_usd=20, closed_at=at(3)))
    o = obs(find_observations(trades, losing_streak=3, tz=BA), OBS_LOSING_STREAK)
    assert o.items == (
        "3 pérdidas seguidas (#1 a #3), entre el 05/10/2026 12:00 y el 05/10/2026 14:00: "
        "-30 USD en total",
    )
    assert OBS_LOSING_STREAK not in codes(find_observations(trades, losing_streak=4))
    with pytest.raises(ValueError):
        find_observations(trades, losing_streak=0)


def test_streak_is_broken_by_breakeven_and_uses_close_order() -> None:
    trades = [
        trade(pnl_usd=-10, closed_at=at(0)),
        trade(pnl_usd=-10, closed_at=at(2)),
        trade(pnl_usd=0, closed_at=at(1)),  # entre las dos pérdidas, por hora de cierre
    ]
    assert OBS_LOSING_STREAK not in codes(find_observations(trades, losing_streak=2))


def test_detects_trades_without_analysis() -> None:
    trades = [
        trade(
            analysis_id=None,
            regime_category=None,
            regime_name=None,
            score_total=None,
            plan_status=None,
        ),
        trade(analysis_id=None, closed_at=at(1)),  # tiene régimen y score: hay contexto
    ]
    o = obs(find_observations(trades), OBS_NO_ANALYSIS_CONTEXT)
    assert len(o.items) == 1 and o.items[0].startswith("#1 ")
    assert "1 operación no tiene" in o.detail


def test_detects_too_many_trades_by_local_day() -> None:
    # 01:00 y 12:00 UTC del 13/10: mismo día en UTC, distinto día en Buenos Aires.
    day = datetime(2026, 10, 13, tzinfo=UTC)
    trades = [trade(closed_at=day + timedelta(hours=1)), trade(closed_at=day + timedelta(hours=12))]
    profile = {**PROFILE, "max_trades_per_day": 1}
    o = obs(find_observations(trades, profile=profile, tz="UTC"), OBS_TOO_MANY_TRADES)
    assert o.items == ("13/10/2026: 2 operaciones (+60 USD)",)
    assert "(1 por día)" in o.detail
    assert OBS_TOO_MANY_TRADES not in codes(find_observations(trades, profile=profile, tz=BA))
    # Sin perfil no se calcula.
    assert OBS_TOO_MANY_TRADES not in codes(find_observations(trades, tz="UTC"))


def test_detects_daily_loss_limit_net() -> None:
    over = [trade(pnl_usd=-100, closed_at=at(0)), trade(pnl_usd=-55, closed_at=at(1))]
    o = obs(find_observations(over, profile=PROFILE, tz=BA), OBS_DAILY_LOSS_LIMIT)
    assert o.items == ("05/10/2026: perdiste 155 USD en el día",)
    assert "(150 USD)" in o.detail
    exact = [trade(pnl_usd=-100, closed_at=at(0)), trade(pnl_usd=-50, closed_at=at(1))]
    assert OBS_DAILY_LOSS_LIMIT not in codes(find_observations(exact, profile=PROFILE))
    netted = over + [trade(pnl_usd=20, closed_at=at(2))]  # neto -135: no pasa
    assert OBS_DAILY_LOSS_LIMIT not in codes(find_observations(netted, profile=PROFILE))
    assert OBS_DAILY_LOSS_LIMIT not in codes(find_observations(over))  # sin perfil


def test_detects_deposit_after_loss() -> None:
    trades = [trade(pnl_usd=-80, closed_at=at(2))]
    events = [
        {"ts": at(1), "kind": "deposit", "amount_usd": 100},  # antes de la pérdida
        {"ts": at(3), "kind": "deposit", "amount_usd": 200},  # después, mismo día
        {"ts": at(4), "kind": "withdrawal", "amount_usd": 50},
        {"ts": at(30), "kind": "deposit", "amount_usd": 300},  # otro día
    ]
    o = obs(find_observations(trades, wallet_events=events, tz=BA), OBS_DEPOSIT_AFTER_LOSS)
    assert o.items == ("05/10/2026 15:00: depósito de 200 USD (antes, ese día, perdiste 80 USD)",)
    clean = find_observations(trades, wallet_events=events[:1], tz=BA)
    assert OBS_DEPOSIT_AFTER_LOSS not in codes(clean)


def test_observations_have_fixed_order() -> None:
    trades = [
        trade(plan_status="NO_OPERAR", pnl_usd=-95, regime_category="bajista", closed_at=at(0)),
        trade(side="short", pnl_usd=-60, regime_category="alcista", closed_at=at(1)),
        trade(
            pnl_usd=-60,
            risk_usd=None,
            analysis_id=None,
            regime_category=None,
            regime_name=None,
            score_total=None,
            plan_status=None,
            closed_at=at(2),
        ),
        trade(pnl_usd=-10, closed_at=at(3)),
    ]
    events = [{"ts": at(5), "kind": "deposit", "amount_usd": 100}]
    found = find_observations(trades, wallet_events=events, profile=PROFILE, tz=BA)
    assert codes(found) == [
        OBS_TRADED_ON_NO_TRADE,
        OBS_AGAINST_PLAN,
        OBS_AGAINST_REGIME,
        OBS_LOSS_ABOVE_RISK,
        OBS_LOSING_STREAK,
        OBS_TOO_MANY_TRADES,
        OBS_DAILY_LOSS_LIMIT,
        OBS_DEPOSIT_AFTER_LOSS,
        OBS_MISSING_RISK,
        OBS_NO_ANALYSIS_CONTEXT,
    ]
    assert all(o.checklist_item for o in found)


# --- informe --------------------------------------------------------------------


def _full_report() -> str:
    trades = [
        trade(
            symbol="BTC",
            plan_status="NO_OPERAR",
            pnl_usd=-45.5,
            regime_category="bajista",
            regime_name=None,
            regime_confidence=0.78,
            score_total=-35,
            notes="entré igual | apurado",
            closed_at=at(0),
        ),
        trade(
            symbol="ETH",
            side="short",
            plan_status="SHORT",
            regime_category="bajista",
            regime_name=None,
            score_total=-40,
            pnl_usd=-95,
            closed_at=at(24),
        ),
        trade(symbol="SOL", pnl_usd=120, closed_at=at(48)),
    ]
    return report(
        trades=trades,
        wallet_events=[
            {"ts": at(-1), "kind": "balance", "amount_usd": 1000},
            {"ts": at(25), "kind": "deposit", "amount_usd": 200, "notes": "recarga"},
            {"ts": at(26), "kind": "withdrawal", "amount_usd": 50},
        ],
        open_positions=[
            {
                "symbol": "xrp",
                "side": "short",
                "opened_at": at(50),
                "risk_usd": 25,
                "entry_price": 0.5123,
            },
        ],
        profile=PROFILE,
        settings_info={"Caída máxima del mes (%)": 70, "Reinicio tras enfriamiento": True},
    )


def test_report_summary_has_key_numbers() -> None:
    text = _full_report()
    summary = text.split("## Resumen", 1)[1].split("## ", 1)[0]
    assert "máximo de **1.000 USD** y ahora tiene **280 USD**" in summary
    assert "cayó **72 %** desde ese máximo (el tope es 70 %)" in summary
    assert "Operaciones: **3**: 1 ganada, 2 perdidas (33 % ganadas)." in summary
    assert "Resultado neto: **-20,50 USD** (ganado +120 USD, perdido -140,50 USD)." in summary
    assert "Peor operación: #2 ETH short" in summary
    assert "Racha de pérdidas más larga: **2 seguidas**." in summary
    assert "depositaste 200 USD (1 vez) y retiraste 50 USD (1 vez)" in summary
    assert "Posiciones que siguen abiertas: **1** (XRP short), con 25 USD en riesgo" in summary
    assert "Perfil de riesgo: **Agresivo**" in summary
    assert "hasta 3 operaciones por día" in summary


def test_report_header_and_sections_in_order() -> None:
    text = _full_report()
    assert text.startswith("# Informe del incidente: la app se bloqueó\n")
    assert "- **Cuándo:** martes 13 de octubre de 2026, 11:05 (hora de Buenos Aires)" in text
    assert f"- **Por qué:** {TRIGGER['description']}" in text
    assert "- **Mes de la billetera:** octubre de 2026" in text
    sections = [
        "## Resumen",
        "## Qué pasó, en simple",
        "## Qué falló",
        "## Operaciones del mes",
        "## Movimientos de la billetera",
        "## Posiciones abiertas al momento del bloqueo",
        "## Antes de borrar el archivo LOCKED",
        "## Datos técnicos",
    ]
    positions = [text.index(s) for s in sections]
    assert positions == sorted(positions)
    assert "Mientras exista el archivo `data/LOCKED`, la app **no va a generar planes**" in text
    assert text.endswith("\n")


def test_report_trades_table() -> None:
    text = _full_report()
    table = text.split("## Operaciones del mes", 1)[1].split("\n\n")[1]
    rows = table.splitlines()
    assert rows[0].startswith("| # | Fecha | Moneda | Dirección | Resultado | Riesgo previsto |")
    assert len(rows) == 2 + 3
    first = rows[2]
    for piece in (
        "| 1 |",
        "05/10/2026 12:00",
        "BTC",
        "Long",
        "-45,50 USD",
        "50 USD",
        "Bajista (78 %)",
        "-35",
        '"No operar"',
        "contra el régimen",
    ):
        assert piece in first
    assert "Short" in rows[3] and "-95 USD" in rows[3] and "+120 USD" in rows[4]
    # Todas las filas tienen las mismas columnas (los "|" de las notas no rompen nada).
    unescaped = re.compile(r"(?<!\\)\|")
    assert len({len(unescaped.findall(r)) for r in rows}) == 1
    assert "- #1: entré igual | apurado" in text  # notas, debajo de la tabla
    assert "**Long** = ganás si el precio sube" in text


def test_report_wallet_positions_checklist_and_tech_data() -> None:
    text = _full_report()
    assert "| 05/10/2026 11:00 | Saldo anotado | 1.000 USD | — |" in text
    assert "| 06/10/2026 13:00 | Depósito | +200 USD | recarga |" in text
    assert "| 06/10/2026 14:00 | Retiro | -50 USD | — |" in text
    assert "| XRP | Short | 07/10/2026 14:00 | 0,5123 | 25 USD | — |" in text
    checklist = text.split("## Antes de borrar el archivo LOCKED", 1)[1].split("## ", 1)[0]
    items = [line for line in checklist.splitlines() if line.startswith("- [ ] ")]
    assert items[0] == "- [ ] Leí este informe completo, con calma."
    assert items[-1] == "- [ ] Recién ahora, borré a mano el archivo `data/LOCKED`."
    assert any("No operar" in i for i in items)  # punto agregado por la observación
    assert any("posiciones que siguen abiertas" in i for i in items)
    assert "- Disparador: `monthly_drawdown_lock`" in text
    assert "- Hora del disparo (UTC): 2026-10-13T14:05:02+00:00" in text
    assert "- Caída máxima del mes (%): 70" in text
    assert "- Reinicio tras enfriamiento: sí" in text


def test_report_observations_section_lists_patterns() -> None:
    text = _full_report()
    section = text.split("## Qué falló", 1)[1].split("## Operaciones del mes", 1)[0]
    assert '### Operaste cuando el plan decía "No operar"' in section
    assert "### Operaste contra el régimen del mercado" in section
    assert "### Perdiste más de lo que ibas a arriesgar" in section
    assert "- #1 BTC long, 05/10/2026 12:00 (-45,50 USD)" in section


def test_report_with_nothing_loaded() -> None:
    text = report(trigger={"code": "x", "description": "Motivo de prueba"}, lock_path=None)
    assert "No hay operaciones cargadas en el período." in text
    assert "No hay movimientos de la billetera cargados" in text
    assert "No había posiciones abiertas cargadas." in text
    assert "La app se bloqueó sola. Motivo: Motivo de prueba." in text
    assert "`LOCKED` (en la carpeta de datos)" in text
    assert "Perfil de riesgo" not in text
    assert "posiciones que siguen abiertas" not in text.split("## Antes de borrar", 1)[1]


def test_report_tolerates_junk_data() -> None:
    text = report(
        trigger={"code": None, "wallet_peak_usd": "mucho"},
        trades=[{"symbol": None, "pnl_usd": "abc", "closed_at": "nunca", "side": 3}],
        wallet_events=[{"kind": None, "amount_usd": float("nan")}],
        open_positions=[{}],
        profile={"max_trades_per_day": "x"},
    )
    assert "## Operaciones del mes" in text
    assert "desconocido" in text


def test_report_requires_a_datetime() -> None:
    with pytest.raises(TypeError):
        report(triggered_at="2026-10-13")


def test_drawdown_beside_threshold_is_not_rounded_to_it() -> None:
    trig = {**TRIGGER, "current_balance_usd": 299.96}  # cae 70,004 %
    assert "cayó **70,004 %**" in report(trigger=trig)


def test_report_says_when_the_drop_was_measured_since_an_unlocked_lock() -> None:
    """Después de desbloquear a mano, la caída se mide desde ese bloqueo, no desde el día 1."""
    trig = {**TRIGGER, "wallet_peak_usd": 80.0, "current_balance_usd": 24.0, "measured_since": T0}
    text = report(trigger=trig)
    assert (
        "- Desde el bloqueo del 05/10/2026 12:00 (que se desbloqueó a mano), la billetera llegó "
        "a un máximo de **80 USD** y ahora tiene **24 USD**: cayó **70 %**"
    ) in text
    assert "La billetera del mes llegó" not in text
    parsed = IncidentTrigger.from_value({**trig, "measured_since": "2026-10-05T15:00:00+00:00"})
    assert parsed.measured_since == T0
    assert parsed.to_dict()["measured_since"] == "2026-10-05T15:00:00+00:00"
    assert IncidentTrigger.from_value(TRIGGER).to_dict()["measured_since"] is None
    assert "La billetera del mes llegó a un máximo" in report()


# --- escritura ------------------------------------------------------------------


def test_incident_file_name() -> None:
    assert incident_file_name(WHEN) == "incidente_2026-10-13_140502_UTC.md"
    ba = WHEN.astimezone(timezone(timedelta(hours=-3)))
    assert incident_file_name(ba) == "incidente_2026-10-13_140502_UTC.md"
    assert incident_file_name(WHEN, 2) == "incidente_2026-10-13_140502_UTC_2.md"
    with pytest.raises(TypeError):
        incident_file_name("hoy")  # type: ignore[arg-type]


def test_write_incident_report(tmp_path: Path) -> None:
    folder = tmp_path / "data" / "incidents"
    kwargs = {"triggered_at": WHEN, "trigger": TRIGGER, "trades": [trade()], "tz": BA}
    path = write_incident_report(folder, **kwargs)
    assert path == folder / "incidente_2026-10-13_140502_UTC.md"
    assert path.read_text(encoding="utf-8") == build_incident_report(**kwargs)
    assert sorted(p.name for p in folder.iterdir()) == [path.name]  # sin temporales


def test_write_never_overwrites(tmp_path: Path) -> None:
    first = write_incident_report(tmp_path, triggered_at=WHEN, trigger=TRIGGER)
    original = first.read_text(encoding="utf-8")
    second = write_incident_report(
        tmp_path, triggered_at=WHEN, trigger={**TRIGGER, "description": "Segundo"}
    )
    third = write_incident_report(tmp_path, triggered_at=WHEN, trigger=TRIGGER)
    assert first.name == "incidente_2026-10-13_140502_UTC.md"
    assert second.name == "incidente_2026-10-13_140502_UTC_2.md"
    assert third.name == "incidente_2026-10-13_140502_UTC_3.md"
    assert first.read_text(encoding="utf-8") == original
    assert "Segundo" in second.read_text(encoding="utf-8")
    assert len(list(tmp_path.iterdir())) == 3


def test_write_skips_hand_made_file(tmp_path: Path) -> None:
    (tmp_path / incident_file_name(WHEN)).write_text("mío", encoding="utf-8")
    path = write_incident_report(tmp_path, triggered_at=WHEN, trigger=TRIGGER)
    assert path.name.endswith("_2.md")
    assert (tmp_path / incident_file_name(WHEN)).read_text(encoding="utf-8") == "mío"


def test_write_gives_up_after_max_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(incident, "MAX_NAME_ATTEMPTS", 2)
    for attempt in (1, 2):
        (tmp_path / incident_file_name(WHEN, attempt)).write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_incident_report(tmp_path, triggered_at=WHEN, trigger=TRIGGER)


def test_create_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "sub" / "a.md"
    assert create_new_file(target, "uno") is True
    assert create_new_file(target, "dos") is False
    assert target.read_text(encoding="utf-8") == "uno"
    assert [p.name for p in target.parent.iterdir()] == ["a.md"]

    def broken_fsync(fd):
        raise OSError(errno.EIO, "disco roto")

    monkeypatch.setattr(os, "fsync", broken_fsync)
    with pytest.raises(OSError):
        create_new_file(tmp_path / "sub" / "b.md", "tres")
    assert [p.name for p in target.parent.iterdir()] == ["a.md"]


def test_create_new_file_without_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_links(src, dst):
        raise OSError(errno.EPERM, "sin enlaces duros")

    monkeypatch.setattr(os, "link", no_links)
    target = tmp_path / "a.md"
    assert create_new_file(target, "uno") is True
    assert create_new_file(target, "dos") is False
    assert target.read_text(encoding="utf-8") == "uno"
    assert [p.name for p in tmp_path.iterdir()] == ["a.md"]


def test_no_nan_leaks_into_report() -> None:
    text = report(trades=[trade(pnl_usd=float("nan"), risk_usd=float("nan"))])
    assert not re.search(r"\b(nan|inf)\b", text, flags=re.IGNORECASE)
    assert not math.isnan(summarize_trades([trade(pnl_usd=float("nan"))]).net_pnl_usd)


# --- análisis de prueba y caída recortada (integración del Hito 5) ----------------------


def test_analysis_test_marks() -> None:
    from app.safety.incident import analysis_test_marks, fmt_analysis_marks

    assert analysis_test_marks(None) == ()
    assert analysis_test_marks({"offline": False, "day_overrides": []}) == ()
    assert analysis_test_marks({"offline": True}) == ("datos sintéticos",)
    assert analysis_test_marks({"day_overrides": [{"field": "pnl_today_usd"}]}) == (
        "números de prueba",
    )
    both = analysis_test_marks({"offline": True, "day_overrides": [1]})
    assert fmt_analysis_marks(both) == "(datos sintéticos y números de prueba)"
    assert fmt_analysis_marks(()) == ""


def test_test_analyses_are_marked_and_do_not_feed_the_plan_observations() -> None:
    """Hallazgo 18: un análisis de prueba no dice nada del mercado real."""
    fake = trade(
        closed_at=at(1),
        pnl_usd=-40.0,
        side="long",
        plan_status="NO_OPERAR",
        regime_category="bajista",
        regime_name="Bajista",
        regime_confidence=0.99,
        analysis_marks=("datos sintéticos",),
    )
    numbers = trade(
        closed_at=at(2),
        pnl_usd=-40.0,
        side="short",
        plan_status="LONG",
        regime_category="bajista",
        regime_name="Bajista",
        analysis_marks=("números de prueba",),
    )
    items = sorted_trades([fake, numbers])
    assert items[0].analysis_marks == ("datos sintéticos",)
    assert not items[0].traded_on_no_trade and not items[0].against_regime
    assert not items[1].against_plan
    found = codes(find_observations(items))
    assert OBS_TRADED_ON_NO_TRADE not in found and OBS_AGAINST_PLAN not in found
    assert OBS_AGAINST_REGIME not in found
    text = report(trades=[fake, numbers])
    assert "| Bajista (99 %) (datos sintéticos) |" in text
    assert "| Bajista (80 %) (números de prueba) |" in text
    assert "**(datos sintéticos)** = el análisis ligado se hizo con datos inventados" in text
    assert items[0].to_dict()["analysis_marks"] == ["datos sintéticos"]
    assert "(datos sintéticos)" not in report(trades=[trade()])


def test_drawdown_above_100_percent_says_more_than_100_and_asks_to_check_the_data() -> None:
    """Fix 7 / E5: con el saldo negativo se dice "más del 100 %" y se pide revisar los datos."""
    trig = {**TRIGGER, "current_balance_usd": -1500.0, "drawdown_pct": 250.0}
    text = report(trigger=trig)
    assert "cayó **más del 100 %** desde ese máximo" in text
    assert "250" not in text.split("## Datos")[0]
    assert "Ojo: el saldo calculado de tu billetera da negativo (-1.500 USD)" in text
    assert "puede haber un resultado o un retiro mal tipeado" in text
    assert (
        "- Caída desde el máximo: más del 100 % (la cuenta dio 250 % porque el saldo calculado "
        "quedó negativo: revisá lo cargado)"
    ) in text
    assert "Ojo" not in report()


def test_drawdown_helpers() -> None:
    assert fmt_drawdown_pct(71.666) == "71,67 %"
    assert fmt_drawdown_pct(69.996, 70) == "69,996 %"
    assert fmt_drawdown_pct(100.0) == "100 %"
    assert fmt_drawdown_pct(233.3, 70) == "más del 100 %"
    assert fmt_drawdown_pct(None) == "—"
    assert drawdown_over_100(100.0000001) is False and drawdown_over_100(100.01) is True
    assert negative_balance_warning(0) is None and negative_balance_warning(None) is None
    assert negative_balance_warning(-0.004) is None  # a centavos es 0
    warning = negative_balance_warning(-50)
    assert warning is not None and "(-50 USD)" in warning and "mal tipeado" in warning
    assert "pasa el 100 %" in warning
    assert "pasa el 100 %" in negative_balance_warning(-50, 133.3)
    # Un retiro más grande que el saldo dejó el máximo en 0: la caída medida es 0 %.
    zero = negative_balance_warning(-800, 0.0)
    assert "pasa el 100 %" not in zero
    assert "no se puede medir y el freno del mes no salta" in zero


def test_month_totals_are_labelled_when_measured_since_an_unlock() -> None:
    """Revisión 6: con la caída medida desde un desbloqueo, los totales son de todo el mes."""
    trig = {**TRIGGER, "measured_since": WHEN - timedelta(days=2)}
    trades = [trade(pnl_usd=-100), trade(pnl_usd=-50, closed_at=at(1))]
    text = report(trigger=trig, trades=trades)
    assert "- Operaciones en todo el mes: **2**" in text
    assert "- Resultado neto en todo el mes: **-150 USD**" in text
    assert "- Operaciones: **2**" in report(trades=trades)


def test_the_fingerprint_ignores_the_time_but_not_the_data(tmp_path: Path) -> None:
    """Fix 7: el mismo disparo con los mismos datos reusa el informe (no se apilan)."""
    trades = [trade(pnl_usd=-100)]
    first = incident.incident_fingerprint(
        triggered_at=WHEN, trigger=TRIGGER, trades=trades, tz=BA, lock_path="data/LOCKED"
    )
    later = incident.incident_fingerprint(
        triggered_at=WHEN + timedelta(minutes=5),
        trigger=TRIGGER,
        trades=trades,
        tz=BA,
        lock_path="data/LOCKED",
    )
    other = incident.incident_fingerprint(
        triggered_at=WHEN,
        trigger=TRIGGER,
        trades=[*trades, trade(pnl_usd=-1, closed_at=at(2))],
        tz=BA,
        lock_path="data/LOCKED",
    )
    assert first == later and first != other
    text = report(trades=trades)
    assert incident.report_fingerprint(text) == first
    assert text.rstrip().endswith(f"<!-- huella del informe: {first} -->")
    path = write_incident_report(
        tmp_path, triggered_at=WHEN, trigger=TRIGGER, trades=trades, tz=BA, lock_path="data/LOCKED"
    )
    assert incident.find_identical_report(tmp_path, first) == path
    assert incident.find_identical_report(tmp_path, other) is None
    assert incident.find_identical_report(tmp_path / "no-existe", first) is None
    assert incident.report_fingerprint("# informe viejo sin huella") is None
