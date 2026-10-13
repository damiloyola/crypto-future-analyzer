"""Hito 5: estado de trading cargado a mano (``app.safety.state``).

Todo en la zona del usuario (Buenos Aires, UTC-3). "Ahora" es el martes
13/10/2026 15:00 de Buenos Aires: el día empezó el 13/10 a las 00:00 (03:00
UTC), la semana el lunes 12/10 y el mes el jueves 01/10.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.config import AppSettings
from app.risk.profiles import get_profile
from app.safety.incident import IncidentTrade
from app.safety.state import (
    POINT_ASSUMED_START,
    POINT_DEPOSIT,
    POINT_MONTH_START,
    POINT_TRADE,
    POINT_WITHDRAWAL,
    PositionRecord,
    ProfileLimits,
    TradeRecord,
    TradingState,
    VolatilityEventRecord,
    WalletRecord,
    build_trading_state,
    count_opened_in,
    load_trading_state,
    losing_streak_state,
    period_bounds,
    period_pnl,
    resolve_user_timezone,
    wallet_state,
)
from app.storage import Repository

BA_NAME = "America/Argentina/Buenos_Aires"
BA = ZoneInfo(BA_NAME)


def local(month: int, day: int, hour: int = 0, minute: int = 0, year: int = 2026) -> datetime:
    """Hora de Buenos Aires."""
    return datetime(year, month, day, hour, minute, tzinfo=BA)


NOW = local(10, 13, 15, 0)
MONTH_START = local(10, 1)

_ids = iter(range(1, 100_000))


def trade(closed: datetime, pnl: float, *, opened: datetime | None = None, **kw) -> dict:
    data = {
        "id": next(_ids),
        "symbol": "BTC",
        "side": "long",
        "opened_at": opened if opened is not None else closed - timedelta(minutes=30),
        "closed_at": closed,
        "pnl_usd": pnl,
    }
    data.update(kw)
    return data


def wallet(ts: datetime, kind: str, amount: float, **kw) -> dict:
    return {"id": next(_ids), "ts": ts, "kind": kind, "amount_usd": amount, **kw}


def with_breakers(settings: AppSettings, **changes) -> AppSettings:
    cb = settings.circuit_breakers.model_copy(update=changes)
    return settings.model_copy(update={"circuit_breakers": cb})


def with_streak(settings: AppSettings, **changes) -> AppSettings:
    streak = settings.circuit_breakers.losing_streak.model_copy(update=changes)
    return with_breakers(settings, losing_streak=streak)


@pytest.fixture
def profile(settings):
    return get_profile("agresivo", settings)


def build(settings, profile, *, now=NOW, trades=(), positions=(), events=(), vol=()):
    return build_trading_state(
        trades=list(trades),
        open_positions=list(positions),
        wallet_events=list(events),
        volatility_events=list(vol),
        settings=settings,
        profile=profile,
        now=now,
    )


# --- períodos ------------------------------------------------------------------------


def test_config_uses_buenos_aires_and_monday(settings):
    assert settings.app.user_timezone == BA_NAME
    assert settings.circuit_breakers.week_starts_on == "monday"
    assert settings.circuit_breakers.loss_basis == "net"
    assert settings.circuit_breakers.month_starts == "calendar"


def test_period_bounds_in_buenos_aires():
    p = period_bounds(NOW, BA_NAME, "monday")
    assert p.timezone == BA_NAME
    assert p.local_date.isoformat() == "2026-10-13"
    assert p.day_start == datetime(2026, 10, 13, 3, 0, tzinfo=UTC)
    assert p.day_end == datetime(2026, 10, 14, 3, 0, tzinfo=UTC)
    assert p.week_start == datetime(2026, 10, 12, 3, 0, tzinfo=UTC)  # lunes
    assert p.week_end == datetime(2026, 10, 19, 3, 0, tzinfo=UTC)
    assert p.month_start == datetime(2026, 10, 1, 3, 0, tzinfo=UTC)
    assert p.month_end == datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
    assert p.labels() == {
        "day": "martes 13/10",
        "week": "del lunes 12/10 al domingo 18/10",
        "month": "octubre de 2026",
    }
    d = p.to_dict()
    assert json.loads(json.dumps(d)) == d


def test_period_bounds_late_night_local_is_still_the_same_day():
    """23:30 de Buenos Aires ya es el día siguiente en UTC, pero cuenta para hoy."""
    p = period_bounds(local(10, 13, 23, 30), BA_NAME)
    assert p.local_date.isoformat() == "2026-10-13"
    assert p.day_start == datetime(2026, 10, 13, 3, 0, tzinfo=UTC)


def test_period_bounds_other_week_start_and_year_end():
    p = period_bounds(NOW, BA_NAME, "sunday")
    assert p.week_start == datetime(2026, 10, 11, 3, 0, tzinfo=UTC)
    p = period_bounds(NOW, BA_NAME, "tuesday")
    assert p.week_start == p.day_start  # hoy es martes
    p = period_bounds(local(12, 31, 22), BA_NAME)
    assert p.month_start == datetime(2026, 12, 1, 3, 0, tzinfo=UTC)
    assert p.month_end == datetime(2027, 1, 1, 3, 0, tzinfo=UTC)


def test_period_bounds_validation():
    with pytest.raises(ValueError, match="zona horaria"):
        period_bounds(datetime(2026, 10, 13, 15, 0), BA_NAME)
    with pytest.raises(ValueError, match="Zona horaria desconocida"):
        period_bounds(NOW, "Marte/Olympus")
    with pytest.raises(ValueError, match="inicio de semana"):
        period_bounds(NOW, BA_NAME, "lunes")
    with pytest.raises(ValueError, match="Falta la zona horaria"):
        resolve_user_timezone("  ")


# --- resultado del día, la semana y el mes ----------------------------------------------


def test_trade_closed_23_30_local_counts_for_yesterday_and_00_30_for_today(settings, profile):
    yesterday = trade(local(10, 12, 23, 30), -30.0)  # 02:30 UTC del 13
    today = trade(local(10, 13, 0, 30), -20.0)  # 03:30 UTC del 13
    state = build(settings, profile, trades=[yesterday, today])
    assert state.day.trades == 1
    assert state.day_pnl_usd == -20.0
    assert state.day_loss_usd == 20.0
    assert state.week.trades == 2  # el lunes 12 también es de esta semana
    assert state.week_loss_usd == 50.0


def test_trade_closed_exactly_at_midnight_counts_for_the_new_day(settings, profile):
    state = build(settings, profile, trades=[trade(local(10, 13, 0, 0), -10.0)])
    assert state.day_loss_usd == 10.0


def test_week_boundary_sunday_night_vs_monday(settings, profile):
    sunday = trade(local(10, 11, 23, 30), -40.0)
    monday = trade(local(10, 12, 0, 30), -25.0)
    state = build(settings, profile, trades=[sunday, monday])
    assert state.week.trades == 1
    assert state.week_loss_usd == 25.0
    assert state.month.trades == 2


def test_month_boundary_last_night_of_september(settings, profile):
    september = trade(local(9, 30, 23, 30), -40.0)  # 02:30 UTC del 1/10, pero es septiembre
    october = trade(local(10, 1, 0, 30), -25.0)
    state = build(settings, profile, trades=[september, october])
    assert [t.id for t in state.month_trades] == [october["id"]]
    assert state.month.net_pnl_usd == -25.0


def test_net_basis_subtracts_the_gains_of_the_day(settings, profile):
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 40.0)]
    state = build(settings, profile, trades=trades)
    assert state.loss_basis == "net"
    assert state.day_pnl_usd == -20.0
    assert state.day_loss_usd == 20.0
    assert state.day.gross_loss_usd == 60.0
    assert state.day.gross_profit_usd == 40.0
    assert (state.day.wins, state.day.losses, state.day.breakeven) == (1, 1, 0)
    assert state.plan_pnl_today_usd == -20.0


def test_gross_basis_counts_only_the_losses(settings, profile):
    gross = with_breakers(settings, loss_basis="gross")
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 40.0)]
    state = build(gross, profile, trades=trades)
    assert state.loss_basis == "gross"
    assert state.day_pnl_usd == -20.0  # el resultado neto se muestra igual
    assert state.day_loss_usd == 60.0
    assert state.plan_pnl_today_usd == -60.0  # el plan achica el riesgo con la misma base


def test_a_winning_day_has_zero_loss(settings, profile):
    state = build(settings, profile, trades=[trade(local(10, 13, 9), 30.0)])
    assert state.day_pnl_usd == 30.0
    assert state.day_loss_usd == 0.0
    assert state.plan_pnl_today_usd == 30.0


def test_period_sums_are_rounded_to_cents():
    trades = [trade(local(10, 13, 9), -20.1), trade(local(10, 13, 10), -29.9)]
    p = period_pnl(trades, local(10, 13), local(10, 14))
    assert p.net_pnl_usd == -50.0
    assert p.loss_usd == 50.0
    breakeven = period_pnl([trade(local(10, 13, 9), -0.004)], local(10, 13), local(10, 14))
    assert (breakeven.losses, breakeven.loss_usd) == (0, 0.0)


def test_period_pnl_validation():
    with pytest.raises(ValueError, match="Base de pérdida"):
        period_pnl([], local(10, 13), local(10, 14), loss_basis="bruto")
    with pytest.raises(ValueError, match="no tiene zona horaria"):
        period_pnl([], datetime(2026, 10, 13), local(10, 14))


def test_trades_today_counts_what_was_opened_today(settings, profile):
    trades = [
        trade(local(10, 13, 1), -5.0, opened=local(10, 12, 22)),  # abierta ayer: no cuenta
        trade(local(10, 13, 10), 5.0, opened=local(10, 13, 9)),
        trade(local(10, 12, 18), 5.0, opened=local(10, 12, 17)),
    ]
    positions = [
        {
            "id": 1,
            "symbol": "ETH",
            "side": "long",
            "risk_usd": 20.0,
            "opened_at": local(10, 13, 12),
        },
        {
            "id": 2,
            "symbol": "SOL",
            "side": "short",
            "risk_usd": 15.0,
            "opened_at": local(10, 12, 9),
        },
    ]
    state = build(settings, profile, trades=trades, positions=positions)
    assert state.closed_trades_opened_today == 1
    assert state.open_positions_opened_today == 1
    assert state.trades_today == 2
    assert state.open_risk_usd == 35.0
    assert count_opened_in(positions, local(10, 12), local(10, 14)) == 2


def test_records_after_now_are_ignored(settings, profile):
    future = trade(NOW + timedelta(minutes=10), -80.0)
    state = build(settings, profile, trades=[future])
    assert state.day_loss_usd == 0.0
    assert state.month_trades == ()
    assert state.wallet.current_balance_usd == 300.0


# --- racha de pérdidas ----------------------------------------------------------------


def streak(trades, now=NOW, *, n=3, hours=4.0, reset=True):
    return losing_streak_state(
        trades, now, consecutive_losses=n, cooldown_hours=hours, reset_after_cooldown=reset
    )


def test_two_losses_build_the_streak_without_a_pause():
    s = streak([trade(local(10, 13, 9), -10.0), trade(local(10, 13, 10), -10.0)])
    assert s.count == 2
    assert s.cooldown_until is None
    assert not s.cooldown_active()


def test_third_loss_starts_a_four_hour_pause_from_its_close():
    t3 = trade(local(10, 13, 12), -10.0)
    trades = [trade(local(10, 13, 9), -10.0), trade(local(10, 13, 10), -10.0), t3]
    s = streak(trades, now=local(10, 13, 12, 5))
    assert s.count == 3
    assert s.cooldown_started_at == local(10, 13, 12)
    assert s.cooldown_until == local(10, 13, 16)
    assert s.cooldown_active()
    assert s.loss_trade_ids[-1] == t3["id"]
    assert s.cooldown_active(local(10, 13, 15, 59))
    assert not s.cooldown_active(local(10, 13, 16))  # justo al terminar ya no


def test_pause_ends_and_the_count_restarts_from_zero():
    trades = [trade(local(10, 13, h), -10.0) for h in (9, 10, 11)]
    s = streak(trades, now=local(10, 13, 15))  # pausa hasta las 15:00
    assert not s.cooldown_active()
    assert s.count == 0
    assert s.loss_trade_ids == ()
    assert s.cooldown_until == local(10, 13, 15)  # la última pausa queda como dato


def test_after_the_pause_old_losses_do_not_count():
    old = [trade(local(10, 13, h), -10.0) for h in (8, 9, 10)]  # pausa 10:00-14:00
    new = [trade(local(10, 13, 14), -10.0), trade(local(10, 13, 14, 30), -10.0)]
    s = streak(old + new, now=local(10, 13, 15))
    assert s.count == 2
    assert s.loss_trade_ids == (new[0]["id"], new[1]["id"])
    assert not s.cooldown_active()
    third = trade(local(10, 13, 14, 45), -10.0)
    s = streak(old + new + [third], now=local(10, 13, 15))
    assert s.cooldown_active()
    assert s.cooldown_started_at == local(10, 13, 14, 45)


def test_without_reset_the_streak_keeps_counting_after_the_pause():
    old = [trade(local(10, 13, h), -10.0) for h in (8, 9, 10)]
    s = streak(old, now=local(10, 13, 15), reset=False)
    assert s.count == 3 and not s.cooldown_active()
    fourth = trade(local(10, 13, 14, 30), -10.0)
    s = streak([*old, fourth], now=local(10, 13, 15), reset=False)
    assert s.count == 4
    assert s.cooldown_active()
    assert s.cooldown_started_at == local(10, 13, 14, 30)


def test_a_loss_during_the_pause_stretches_it():
    trades = [trade(local(10, 13, h), -10.0) for h in (10, 11, 12)]  # pausa hasta 16:00
    trades.append(trade(local(10, 13, 13), -10.0))  # operó igual y perdió
    s = streak(trades, now=local(10, 13, 16, 30))
    assert s.cooldown_started_at == local(10, 13, 12)
    assert s.cooldown_until == local(10, 13, 17)
    assert s.cooldown_active()
    assert s.count == 4


def test_a_win_or_breakeven_breaks_the_streak():
    trades = [
        trade(local(10, 13, 8), -10.0),
        trade(local(10, 13, 9), -10.0),
        trade(local(10, 13, 10), 0.0),  # empate
        trade(local(10, 13, 11), -10.0),
    ]
    assert streak(trades).count == 1
    trades[2]["pnl_usd"] = 5.0
    assert streak(trades).count == 1
    trades[2]["pnl_usd"] = -0.004  # a centavos es un empate
    assert streak(trades).count == 1
    trades[2]["pnl_usd"] = -0.01
    assert streak(trades, now=local(10, 13, 11, 30)).cooldown_active()


def test_a_win_during_the_pause_does_not_cancel_it():
    trades = [trade(local(10, 13, h), -10.0) for h in (10, 11, 12)]
    trades.append(trade(local(10, 13, 13), 30.0))
    s = streak(trades, now=local(10, 13, 14))
    assert s.count == 0
    assert s.cooldown_active()


def test_streak_order_is_by_close_time_not_by_input_order():
    trades = [trade(local(10, 13, 11), -10.0), trade(local(10, 13, 9), 20.0)]
    trades.append(trade(local(10, 13, 10), -10.0))
    assert streak(trades).count == 2


def test_count_at_a_later_time_resets_when_the_pause_ends():
    trades = [trade(local(10, 13, h), -10.0) for h in (10, 11, 12)]
    s = streak(trades, now=local(10, 13, 13))
    assert s.count_at(local(10, 13, 15, 59)) == 3
    assert s.count_at(local(10, 13, 16)) == 0
    no_reset = streak(trades, now=local(10, 13, 13), reset=False)
    assert no_reset.count_at(local(10, 13, 17)) == 3


def test_streak_validation():
    with pytest.raises(ValueError, match="consecutive_losses"):
        streak([], n=0)
    with pytest.raises(ValueError, match="horas de pausa"):
        streak([], hours=0)


def test_state_uses_the_streak_settings(settings, profile):
    trades = [trade(local(10, 13, h), -10.0) for h in (10, 11)]
    two = with_streak(settings, consecutive_losses=2, cooldown_hours=1.5)
    state = build(two, profile, now=local(10, 13, 12), trades=trades)
    assert state.streak.required == 2
    assert state.streak.cooldown_until == local(10, 13, 12, 30)
    assert state.streak.cooldown_active()


# --- billetera ------------------------------------------------------------------------


def wallet_of(events=(), trades=(), now=NOW, assumed=300.0):
    return wallet_state(
        list(events), list(trades), now, month_start=MONTH_START, assumed_start_usd=assumed
    )


def test_no_wallet_events_assumes_the_monthly_wallet_and_warns():
    w = wallet_of(trades=[trade(local(10, 5, 10), -50.0), trade(local(9, 20), -500.0)])
    assert w.assumed is True
    assert w.assumed_start_usd == 300.0
    assert w.month_start_balance_usd == 300.0  # lo de septiembre no cambia el supuesto
    assert w.current_balance_usd == 250.0
    assert w.month_peak_usd == 300.0
    assert w.drawdown_usd == 50.0
    assert math.isclose(w.drawdown_pct, 50 / 3)
    assert w.anchor is None and w.last_balance_at is None
    assert len(w.warnings) == 1
    assert "No anotaste nunca el saldo" in w.warnings[0]
    assert "300 USD" in w.warnings[0]
    assert w.points[0].kind == POINT_ASSUMED_START


def test_state_warns_when_the_balance_was_never_loaded(settings, profile):
    state = build(settings, profile)
    assert state.wallet.assumed
    assert any("Anotá tu saldo real" in w for w in state.warnings)
    loaded = build(settings, profile, events=[wallet(local(10, 1, 9), "balance", 280.0)])
    assert not loaded.wallet.assumed
    assert loaded.warnings == ()


def test_balance_before_the_month_plus_what_happened_until_the_first_instant():
    events = [wallet(local(9, 25), "balance", 400.0)]
    trades = [trade(local(9, 28), -100.0), trade(local(10, 2), 50.0)]
    w = wallet_of(events, trades)
    assert w.month_start_balance_usd == 300.0
    assert w.current_balance_usd == 350.0
    assert w.month_peak_usd == 350.0
    assert w.drawdown_pct == 0.0
    assert w.anchor is not None and w.anchor.amount_usd == 400.0
    assert w.month_trading_pnl_usd == 50.0


def test_balance_loaded_mid_month_is_projected_back_to_the_first_instant():
    trades = [trade(local(10, 5), -30.0), trade(local(10, 10), -20.0), trade(local(10, 12), -40.0)]
    events = [wallet(local(10, 11), "balance", 250.0)]
    w = wallet_of(events, trades)
    assert w.month_start_balance_usd == 300.0  # 250 + 30 + 20
    assert w.current_balance_usd == 210.0
    assert w.month_peak_usd == 300.0
    assert math.isclose(w.drawdown_pct, 30.0)
    assert w.last_balance_usd == 250.0


def test_deposits_raise_the_peak():
    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 6), "deposit", 200.0),
    ]
    trades = [trade(local(10, 5), -100.0), trade(local(10, 8), -50.0)]
    w = wallet_of(events, trades)
    assert w.current_balance_usd == 350.0
    assert w.month_peak_usd == 400.0
    assert w.month_peak_at == local(10, 6).astimezone(UTC)
    assert w.drawdown_usd == 50.0
    assert w.drawdown_pct == 12.5
    assert w.month_deposits_usd == 200.0
    kinds = [p.kind for p in w.points]
    assert kinds == [POINT_MONTH_START, "balance", POINT_TRADE, POINT_DEPOSIT, POINT_TRADE]
    assert [p.peak_usd for p in w.points] == [300.0, 300.0, 300.0, 400.0, 400.0]


def test_withdrawals_lower_the_peak_by_the_same_amount():
    """Sacar plata no es perder plata: no tiene que disparar el bloqueo."""
    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 7), "withdrawal", 400.0),
    ]
    trades = [trade(local(10, 5), 300.0), trade(local(10, 8), -20.0)]
    w = wallet_of(events, trades)
    assert w.current_balance_usd == 180.0
    assert w.month_peak_usd == 200.0  # 600 - 400
    assert w.drawdown_pct == 10.0
    assert w.month_withdrawals_usd == 400.0
    assert [p.kind for p in w.points][-2:] == [POINT_WITHDRAWAL, POINT_TRADE]


def test_a_lower_balance_correction_counts_as_a_drop():
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(local(10, 9), "balance", 240.0)]
    w = wallet_of(events, [trade(local(10, 5), 20.0)])
    assert w.month_peak_usd == 320.0
    assert w.current_balance_usd == 240.0
    assert w.drawdown_pct == 25.0


def test_a_balance_includes_what_happened_at_the_same_instant():
    when = local(10, 9, 12)
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(when, "balance", 270.0)]
    w = wallet_of(events, [trade(when, -30.0)])
    assert w.current_balance_usd == 270.0  # no 240: el saldo ya incluía la pérdida


def test_negative_balance_adds_a_warning():
    events = [wallet(local(10, 1, 9), "balance", 50.0)]
    w = wallet_of(events, [trade(local(10, 5), -80.0)])
    assert w.current_balance_usd == -30.0
    assert any("negativo" in x for x in w.warnings)


def test_wallet_points_are_json_ready():
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(local(10, 6), "deposit", 50.0)]
    w = wallet_of(events, [trade(local(10, 5), -10.5, symbol="ETH")])
    d = w.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    trade_point = next(p for p in d["points"] if p["kind"] == POINT_TRADE)
    assert trade_point["symbol"] == "ETH"
    assert trade_point["delta_usd"] == -10.5


# --- registros y estado completo -------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"symbol": "BTC", "closed_at": datetime(2026, 10, 13, 12), "pnl_usd": -5}, "zona horaria"),
        ({"symbol": "BTC", "closed_at": NOW, "pnl_usd": float("nan")}, "número finito"),
        ({"symbol": "BTC", "closed_at": NOW, "pnl_usd": "-5"}, "tiene que ser un número"),
        ({"symbol": "BTC", "closed_at": None, "pnl_usd": -5}, "tiene que ser una fecha"),
    ],
)
def test_invalid_trades_raise_instead_of_being_ignored(record, message):
    with pytest.raises(ValueError, match=message):
        TradeRecord.from_value(record)


def test_invalid_wallet_events_raise():
    with pytest.raises(ValueError, match="tipo desconocido"):
        WalletRecord.from_value({"ts": NOW, "kind": "transfer", "amount_usd": 5})
    with pytest.raises(ValueError, match="no puede ser negativo"):
        WalletRecord.from_value({"ts": NOW, "kind": "deposit", "amount_usd": -5})
    with pytest.raises(ValueError, match="billetera mensual"):
        ProfileLimits.from_value({"monthly_wallet_usd": 0})


def test_records_accept_objects_dicts_and_numpy():
    np = pytest.importorskip("numpy")
    t = TradeRecord.from_value(
        {"id": 3, "symbol": "eth", "side": "LONG", "closed_at": NOW, "pnl_usd": np.float64(-2.5)}
    )
    assert (t.symbol, t.side, t.pnl_usd, t.is_loss) == ("ETH", "long", -2.5, True)
    assert TradeRecord.from_value(t) is t
    p = PositionRecord.from_value({"symbol": "sol", "side": "short", "risk_usd": 10})
    assert (p.symbol, p.side, p.opened_at) == ("SOL", "short", None)


def test_trade_records_work_with_the_incident_report():
    t = TradeRecord.from_value(
        trade(local(10, 13, 9), -40.0, regime_category="bajista", score_total=35.0, risk_usd=30.0)
    )
    it = IncidentTrade.from_value(t)
    assert it.against_regime and it.loss_beyond_risk_usd == 10.0


def test_volatility_events_split_into_active_and_upcoming(settings, profile):
    vol = [
        {
            "id": 1,
            "starts_at": NOW - timedelta(hours=1),
            "until": NOW + timedelta(hours=1),
            "note": "Fed",
        },
        {
            "id": 2,
            "starts_at": NOW + timedelta(hours=2),
            "until": NOW + timedelta(hours=3),
            "note": "CPI",
        },
        {"id": 3, "starts_at": NOW - timedelta(hours=3), "until": NOW, "note": "Viejo"},
    ]
    state = build(settings, profile, vol=vol)
    assert [e.id for e in state.volatility_events] == [1, 2]  # el terminado se descarta
    assert [e.note for e in state.active_volatility_events()] == ["Fed"]
    assert [e.note for e in state.upcoming_volatility_events()] == ["CPI"]
    later = NOW + timedelta(hours=2)
    assert [e.note for e in state.active_volatility_events(later)] == ["CPI"]
    assert VolatilityEventRecord.from_value(vol[0]).is_active(NOW)


def test_state_to_dict_is_json_ready(settings, profile):
    state = build(
        settings,
        profile,
        trades=[trade(local(10, 13, 9), -12.345), trade(local(10, 2), 7.0)],
        positions=[{"id": 9, "symbol": "ETH", "side": "long", "risk_usd": 20.0, "opened_at": NOW}],
        events=[wallet(local(10, 1, 9), "balance", 300.0)],
        vol=[{"id": 4, "starts_at": NOW, "until": NOW + timedelta(hours=1), "note": "Fed"}],
    )
    d = state.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["day_loss_usd"] == 12.35
    assert d["max_trades_per_day"] == 3
    assert d["profile"]["name"] == "agresivo"
    assert d["active_volatility_events"][0]["note"] == "Fed"
    assert d["periods"]["labels"]["day"] == "martes 13/10"
    assert d["wallet"]["current_balance_usd"] == pytest.approx(294.65, abs=0.01)
    assert {"streak", "week", "month", "open_positions", "month_trades", "warnings"} <= set(d)


def test_month_wallet_events_include_the_anchor_balance(settings, profile):
    events = [wallet(local(9, 20), "balance", 500.0), wallet(local(10, 3), "deposit", 50.0)]
    state = build(settings, profile, events=events)
    assert [e.kind for e in state.month_wallet_events] == ["balance", "deposit"]
    assert state.wallet.month_start_balance_usd == 500.0


def test_state_rejects_naive_now(settings, profile):
    with pytest.raises(ValueError, match="zona horaria"):
        build(settings, profile, now=datetime(2026, 10, 13, 15, 0))


# --- lectura de la base ---------------------------------------------------------------


def test_load_trading_state_from_a_real_database(tmp_path: Path, settings, profile):
    repo = Repository.open(tmp_path / "state.db", clock=lambda: NOW)
    repo.add_wallet_event(kind="balance", amount_usd=300.0, ts=local(10, 1, 9))
    repo.add_wallet_event(kind="deposit", amount_usd=100.0, ts=local(10, 4))
    repo.add_trade(
        symbol="BTC",
        side="long",
        pnl_usd=-30.0,
        opened_at=local(10, 12, 22),
        closed_at=local(10, 12, 23, 30),
    )
    repo.add_trade(
        symbol="ETH",
        side="short",
        pnl_usd=-25.0,
        opened_at=local(10, 13, 9),
        closed_at=local(10, 13, 10),
    )
    repo.add_open_position(symbol="SOL", side="long", risk_usd=40.0, opened_at=local(10, 13, 11))
    repo.add_volatility_event("Fed", until=NOW + timedelta(hours=1))
    repo.add_volatility_event(
        "CPI", starts_at=NOW + timedelta(days=1), until=NOW + timedelta(days=1, hours=2)
    )
    expired = repo.add_volatility_event(
        "Viejo", starts_at=NOW - timedelta(hours=2), until=NOW + timedelta(minutes=1)
    )

    state = load_trading_state(repo, settings, profile, NOW + timedelta(minutes=1))
    assert isinstance(state, TradingState)
    assert state.day_loss_usd == 25.0
    assert state.week_loss_usd == 55.0
    assert state.trades_today == 2  # ETH (cerrada) + SOL (abierta)
    assert state.streak.count == 2
    assert state.wallet.current_balance_usd == 345.0
    assert state.wallet.month_peak_usd == 400.0
    assert [p.symbol for p in state.open_positions] == ["SOL"]
    assert [e.note for e in state.active_volatility_events()] == ["Fed"]
    assert expired.id not in [e.id for e in state.volatility_events]
    json.dumps(state.to_dict(), allow_nan=False)
