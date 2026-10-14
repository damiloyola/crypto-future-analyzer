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
    SafetyLockRecord,
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


def with_drawdown(settings: AppSettings, **changes) -> AppSettings:
    dd = settings.circuit_breakers.monthly_drawdown_lock.model_copy(update=changes)
    return with_breakers(settings, monthly_drawdown_lock=dd)


@pytest.fixture
def profile(settings):
    return get_profile("agresivo", settings)


def build(settings, profile, *, now=NOW, trades=(), positions=(), events=(), vol=(), locks=()):
    return build_trading_state(
        trades=list(trades),
        open_positions=list(positions),
        wallet_events=list(events),
        volatility_events=list(vol),
        settings=settings,
        profile=profile,
        now=now,
        safety_locks=list(locks),
    )


# --- períodos ------------------------------------------------------------------------


def test_config_uses_buenos_aires_and_monday(settings):
    assert settings.app.user_timezone == BA_NAME
    assert settings.circuit_breakers.week_starts_on == "monday"
    assert settings.circuit_breakers.loss_basis == "net"
    assert settings.circuit_breakers.month_starts == "calendar"


def test_config_has_the_second_round_decisions(settings):
    cb = settings.circuit_breakers
    assert cb.latch_until_period_end is True
    assert cb.losing_streak.extend_cooldown_on_loss is True
    assert cb.monthly_drawdown_lock.withdrawal_peak == "proportional"
    assert cb.monthly_drawdown_lock.deposit_raises_peak is True
    assert cb.monthly_drawdown_lock.restart_after_unlock is True


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


def test_max_loss_keeps_the_worst_running_net_of_the_period(settings, profile):
    """Decisión del usuario (latch): perdiste 60, después ganaste 20 → la más grande es 60."""
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 20.0)]
    state = build(settings, profile, trades=trades)
    assert state.day_loss_usd == 40.0  # actual
    assert state.day_max_loss_usd == 60.0  # la más grande
    assert state.plan_pnl_today_usd == -40.0  # el plan mira la actual
    assert state.week_max_loss_usd == 60.0
    d = state.to_dict()
    assert (d["day_max_loss_usd"], d["week_max_loss_usd"], d["day"]["max_loss_usd"]) == (
        60.0,
        60.0,
        60.0,
    )


def test_max_loss_walks_the_trades_in_close_order(settings, profile):
    # Cargadas desordenadas: +30 (9:00), -50 (10:00), -20 (11:00), +45 (12:00).
    trades = [
        trade(local(10, 13, 12), 45.0),
        trade(local(10, 13, 10), -50.0),
        trade(local(10, 13, 9), 30.0),
        trade(local(10, 13, 11), -20.0),
    ]
    day = build(settings, profile, trades=trades).day
    assert day.net_pnl_usd == 5.0 and day.loss_usd == 0.0
    assert day.max_loss_usd == 40.0  # 30 - 50 - 20 = -40 fue lo peor


def test_max_loss_with_a_win_first_never_goes_below_zero():
    p = period_pnl(
        [trade(local(10, 13, 9), 80.0), trade(local(10, 13, 10), -30.0)],
        local(10, 13),
        local(10, 14),
    )
    assert (p.loss_usd, p.max_loss_usd) == (0.0, 0.0)
    assert period_pnl([], local(10, 13), local(10, 14)).max_loss_usd == 0.0


def test_max_loss_with_gross_basis_is_the_sum_of_the_losses():
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 20.0)]
    p = period_pnl(trades, local(10, 13), local(10, 14), loss_basis="gross")
    assert (p.loss_usd, p.max_loss_usd) == (60.0, 60.0)


def test_max_loss_of_the_week_spans_several_days(settings, profile):
    trades = [trade(local(10, 12, 10), -150.0), trade(local(10, 13, 9), 60.0)]
    state = build(settings, profile, trades=trades)
    assert (state.week_loss_usd, state.week_max_loss_usd) == (90.0, 150.0)
    assert (state.day_loss_usd, state.day_max_loss_usd) == (0.0, 0.0)  # hoy solo ganó


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


def test_without_extension_a_loss_during_the_pause_does_not_stretch_it():
    """``extend_cooldown_on_loss: false``: la pausa termina a la hora prevista."""
    trades = [trade(local(10, 13, h), -10.0) for h in (10, 11, 12)]  # pausa hasta 16:00
    trades.append(trade(local(10, 13, 13), -10.0))
    s = losing_streak_state(
        trades,
        local(10, 13, 15, 59),
        consecutive_losses=3,
        cooldown_hours=4,
        extend_cooldown_on_loss=False,
    )
    assert s.cooldown_started_at == local(10, 13, 12)
    assert s.cooldown_until == local(10, 13, 16)
    assert s.cooldown_active()
    assert s.count == 4  # la pérdida igual suma
    assert s.extend_cooldown_on_loss is False
    later = losing_streak_state(
        trades,
        local(10, 13, 16),
        consecutive_losses=3,
        cooldown_hours=4,
        extend_cooldown_on_loss=False,
    )
    assert not later.cooldown_active()
    assert later.count == 0  # terminó la pausa: la cuenta vuelve a cero


def test_without_extension_and_without_reset_a_new_loss_after_the_pause_starts_another():
    trades = [trade(local(10, 13, h), -10.0) for h in (8, 9, 10)]  # pausa 10:00-14:00
    trades.append(trade(local(10, 13, 11), -10.0))  # durante la pausa: no la estira
    trades.append(trade(local(10, 13, 14, 30), -10.0))  # después: pausa nueva
    s = losing_streak_state(
        trades,
        local(10, 13, 15),
        consecutive_losses=3,
        cooldown_hours=4,
        reset_after_cooldown=False,
        extend_cooldown_on_loss=False,
    )
    assert s.count == 5
    assert s.cooldown_started_at == local(10, 13, 14, 30)
    assert s.cooldown_until == local(10, 13, 18, 30)


def test_state_uses_the_cooldown_extension_setting(settings, profile):
    trades = [trade(local(10, 13, h), -10.0) for h in (9, 10, 11)]  # pausa hasta 15:00
    trades.append(trade(local(10, 13, 12), -10.0))
    on = build(settings, profile, trades=trades, now=local(10, 13, 15, 30))
    assert on.streak.cooldown_until == local(10, 13, 16)
    assert on.streak.cooldown_active()
    assert on.to_dict()["streak"]["extend_cooldown_on_loss"] is True
    off_settings = with_streak(settings, extend_cooldown_on_loss=False)
    off = build(off_settings, profile, trades=trades, now=local(10, 13, 15, 30))
    assert off.streak.cooldown_until == local(10, 13, 15)
    assert not off.streak.cooldown_active()


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


def test_deposits_raise_the_peak_by_the_same_amount():
    """Decisión del usuario: la pérdida de antes sigue contando (300 - 100 + 200 - 50)."""
    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 6), "deposit", 200.0),
    ]
    trades = [trade(local(10, 5), -100.0), trade(local(10, 8), -50.0)]
    w = wallet_of(events, trades)
    assert w.current_balance_usd == 350.0
    assert w.month_peak_usd == 500.0  # 300 + 200
    assert w.month_peak_at == MONTH_START.astimezone(UTC)  # el máximo es el mismo, corrido
    assert w.drawdown_usd == 150.0
    assert w.drawdown_pct == 30.0
    assert w.month_deposits_usd == 200.0
    assert w.deposit_raises_peak is True
    kinds = [p.kind for p in w.points]
    assert kinds == [POINT_MONTH_START, "balance", POINT_TRADE, POINT_DEPOSIT, POINT_TRADE]
    assert [p.peak_usd for p in w.points] == [300.0, 300.0, 300.0, 500.0, 500.0]


def test_without_deposit_raises_peak_the_peak_only_moves_when_the_balance_passes_it():
    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 6), "deposit", 200.0),
    ]
    trades = [trade(local(10, 5), -100.0), trade(local(10, 8), -50.0)]
    w = wallet_state(
        events,
        trades,
        NOW,
        month_start=MONTH_START,
        assumed_start_usd=300.0,
        deposit_raises_peak=False,
    )
    assert w.month_peak_usd == 400.0
    assert w.month_peak_at == local(10, 6).astimezone(UTC)
    assert w.drawdown_pct == 12.5


def test_a_deposit_after_a_big_loss_does_not_erase_the_drawdown():
    """Hallazgo 3: depositar no esquiva el bloqueo (300 → 100, +200, -205 → 81 %)."""
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(local(10, 6), "deposit", 200.0)]
    only_deposit = wallet_of(events, [trade(local(10, 5), -200.0)])
    assert only_deposit.current_balance_usd == 300.0
    assert only_deposit.month_peak_usd == 500.0
    assert math.isclose(only_deposit.drawdown_pct, 40.0)  # la caída de 200 sigue ahí
    w = wallet_of(events, [trade(local(10, 5), -200.0), trade(local(10, 8), -205.0)])
    assert w.current_balance_usd == 95.0
    assert math.isclose(w.drawdown_pct, 81.0)


def test_a_withdrawal_at_the_peak_lowers_it_by_the_same_amount():
    """Sacar plata no es perder plata: en el máximo, proporcional y "restar" coinciden."""
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


def withdrawal_case(rule: str):
    """300 el 01/10, -150 el 05/10 (50 % abajo) y un retiro de 100 el 06/10 sin operar."""
    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 6), "withdrawal", 100.0),
    ]
    return wallet_state(
        events,
        [trade(local(10, 5), -150.0)],
        NOW,
        month_start=MONTH_START,
        assumed_start_usd=300.0,
        withdrawal_peak=rule,
    )


def test_a_withdrawal_below_the_peak_does_not_change_the_drawdown_percent():
    """Hallazgo 2 / decisión del usuario: el máximo baja en la misma proporción que el saldo."""
    w = withdrawal_case("proportional")
    assert w.current_balance_usd == 50.0
    assert w.month_peak_usd == 100.0  # 300 × 50 / 150
    assert w.drawdown_pct == 50.0  # igual que antes del retiro
    assert w.withdrawal_peak == "proportional"
    assert w.points[-1].peak_usd == 100.0


def test_withdrawal_rules_subtract_and_none():
    subtract = withdrawal_case("subtract")
    assert subtract.month_peak_usd == 200.0  # 300 - 100 (lo de antes)
    assert subtract.drawdown_pct == 75.0
    none = withdrawal_case("none")
    assert none.month_peak_usd == 300.0  # el retiro cuenta como caída
    assert math.isclose(none.drawdown_pct, 250 / 3)
    with pytest.raises(ValueError, match="Regla de retiros desconocida"):
        withdrawal_case("half")


def test_withdrawing_everything_leaves_no_drawdown():
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(local(10, 6), "withdrawal", 150.0)]
    w = wallet_of(events, [trade(local(10, 5), -150.0)])
    assert w.current_balance_usd == 0.0
    assert w.month_peak_usd == 0.0
    assert w.drawdown_pct == 0.0  # sacar todo no es una caída


def test_a_withdrawal_with_the_balance_already_at_zero_or_below():
    """Saldo antes del retiro <= 0 (un error de carga): el máximo no cambia y se avisa."""
    events = [wallet(local(10, 1, 9), "balance", 100.0), wallet(local(10, 6), "withdrawal", 20.0)]
    w = wallet_of(events, [trade(local(10, 5), -100.0)])
    assert w.current_balance_usd == -20.0
    assert w.month_peak_usd == 100.0
    assert w.drawdown_pct == 120.0
    assert any("negativo" in x for x in w.warnings)
    more = [wallet(local(10, 1, 9), "balance", 100.0), wallet(local(10, 6), "withdrawal", 130.0)]
    w = wallet_of(more, [trade(local(10, 5), -50.0)])  # 50 de saldo y retira 130
    assert w.current_balance_usd == -80.0
    assert w.month_peak_usd == 0.0 and w.drawdown_pct == 0.0
    assert any("negativo" in x for x in w.warnings)


def test_state_uses_the_wallet_rules_of_the_config(settings, profile):
    events = [wallet(local(10, 1, 9), "balance", 300.0), wallet(local(10, 6), "withdrawal", 100.0)]
    trades = [trade(local(10, 5), -150.0)]
    state = build(settings, profile, trades=trades, events=events)
    assert state.wallet.drawdown_pct == 50.0
    old = with_drawdown(settings, withdrawal_peak="subtract", deposit_raises_peak=False)
    assert build(old, profile, trades=trades, events=events).wallet.drawdown_pct == 75.0
    d = state.to_dict()["wallet"]
    assert (d["withdrawal_peak"], d["deposit_raises_peak"], d["restart"]) == (
        "proportional",
        True,
        None,
    )


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


# --- bloqueo total anotado: la caída se mide de nuevo (restart_after_unlock) ---------


def lock(ts: datetime, **kw) -> dict:
    return {"id": next(_ids), "triggered_at": ts, "trigger_code": "monthly_drawdown_lock", **kw}


LOCK_EVENTS = [wallet(local(10, 1, 9), "balance", 300.0)]
LOCK_TRADES = [trade(local(10, 5, 10), -220.0)]  # 300 → 80: 73 %, bloqueo el 05/10 12:00


def test_wallet_restart_measures_from_the_balance_right_after_the_lock():
    trades = [*LOCK_TRADES, trade(local(10, 6), 20.0), trade(local(10, 7), -40.0)]
    w = wallet_state(
        LOCK_EVENTS,
        trades,
        NOW,
        month_start=MONTH_START,
        assumed_start_usd=300.0,
        restart_at=local(10, 5, 12),
        restart_lock_id=7,
    )
    assert math.isclose(w.drawdown_pct, (300 - 60) / 300 * 100)  # el mes sigue igual
    r = w.restart
    assert r is not None
    assert r.lock_id == 7
    assert r.locked_at == local(10, 5, 12).astimezone(UTC)
    assert r.start_balance_usd == 80.0  # con la pérdida que disparó el bloqueo
    assert r.peak_usd == 100.0 and r.peak_at == local(10, 6).astimezone(UTC)
    assert r.current_balance_usd == 60.0
    assert (r.drawdown_usd, r.drawdown_pct) == (40.0, 40.0)
    assert r.trading_pnl_usd == -20.0
    d = r.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def test_wallet_restart_includes_a_trade_closed_exactly_at_the_lock_time():
    w = wallet_state(
        LOCK_EVENTS,
        [trade(local(10, 5, 12), -220.0)],
        NOW,
        month_start=MONTH_START,
        assumed_start_usd=300.0,
        restart_at=local(10, 5, 12),
    )
    assert w.restart is not None
    assert w.restart.start_balance_usd == 80.0 and w.restart.drawdown_pct == 0.0


def test_wallet_restart_follows_the_deposit_and_withdrawal_rules():
    events = [
        *LOCK_EVENTS,
        wallet(local(10, 6), "deposit", 120.0),  # 80 → 200, máximo 80 + 120 = 200
        wallet(local(10, 9), "withdrawal", 50.0),  # después de -100: 100 → 50, máximo × 0,5
    ]
    trades = [*LOCK_TRADES, trade(local(10, 8), -100.0)]
    w = wallet_state(
        events,
        trades,
        NOW,
        month_start=MONTH_START,
        assumed_start_usd=300.0,
        restart_at=local(10, 5, 12),
    )
    assert w.restart is not None
    assert w.restart.peak_usd == 100.0
    assert w.restart.current_balance_usd == 50.0
    assert w.restart.drawdown_pct == 50.0
    assert (w.restart.deposits_usd, w.restart.withdrawals_usd) == (120.0, 50.0)


def test_wallet_restart_outside_the_month_or_in_the_future_is_ignored():
    for when in (local(9, 28), NOW + timedelta(hours=1)):
        w = wallet_state(
            LOCK_EVENTS,
            LOCK_TRADES,
            NOW,
            month_start=MONTH_START,
            assumed_start_usd=300.0,
            restart_at=when,
        )
        assert w.restart is None


def test_state_restart_uses_the_latest_lock_of_the_month(settings, profile):
    old = lock(local(9, 20))  # mes anterior: no cuenta
    first = lock(local(10, 5, 12), drawdown_pct=73.33, peak_usd=300.0, balance_usd=80.0)
    second = lock(local(10, 10, 9))
    trades = [*LOCK_TRADES, trade(local(10, 9), 40.0), trade(local(10, 11), -10.0)]
    state = build(settings, profile, trades=trades, events=LOCK_EVENTS, locks=[second, old, first])
    assert [lk.id for lk in state.month_locks] == [first["id"], second["id"]]
    assert state.latest_month_lock is not None and state.latest_month_lock.id == second["id"]
    r = state.wallet.restart
    assert r is not None and r.lock_id == second["id"]
    assert r.start_balance_usd == 120.0  # 80 + 40 antes del segundo bloqueo
    assert r.current_balance_usd == 110.0
    d = state.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["latest_month_lock"]["id"] == second["id"]
    assert d["wallet"]["restart"]["lock_id"] == second["id"]


def test_state_ignores_locks_of_previous_months_and_after_now(settings, profile):
    locks = [lock(local(9, 30, 23)), lock(NOW + timedelta(minutes=1))]
    state = build(settings, profile, trades=LOCK_TRADES, events=LOCK_EVENTS, locks=locks)
    assert state.month_locks == ()
    assert state.latest_month_lock is None
    assert state.wallet.restart is None


def test_state_without_restart_after_unlock_has_no_restart(settings, profile):
    off = with_drawdown(settings, restart_after_unlock=False)
    state = build(
        off, profile, trades=LOCK_TRADES, events=LOCK_EVENTS, locks=[lock(local(10, 5, 12))]
    )
    assert len(state.month_locks) == 1  # se informa igual
    assert state.wallet.restart is None


def test_safety_lock_records_accept_objects_and_reject_bad_dates():
    rec = SafetyLockRecord.from_value(lock(local(10, 5, 12), report_path="  ", balance_usd=80))
    assert rec.triggered_at == local(10, 5, 12).astimezone(UTC)
    assert rec.report_path is None and rec.balance_usd == 80.0
    assert SafetyLockRecord.from_value(rec) is rec
    assert json.loads(json.dumps(rec.to_dict())) == rec.to_dict()
    with pytest.raises(ValueError, match="zona horaria"):
        SafetyLockRecord.from_value({"id": 1, "triggered_at": datetime(2026, 10, 5, 12)})


def test_open_risk_counts_zero_for_stops_at_entry_or_in_profit(settings, profile):
    positions = [
        {"id": 1, "symbol": "BTC", "side": "long", "risk_usd": 30.0, "opened_at": NOW},
        {"id": 2, "symbol": "BTC", "side": "short", "risk_usd": 20.0, "opened_at": NOW},
        {"id": 3, "symbol": "ETH", "side": "long", "risk_usd": 0.0, "opened_at": NOW},
        {"id": 4, "symbol": "SOL", "side": "long", "risk_usd": -15.0, "opened_at": NOW},
    ]
    state = build(settings, profile, positions=positions)
    assert state.open_risk_usd == 50.0  # la cobertura suma; 0 y negativo cuentan 0
    assert state.to_dict()["open_risk_usd"] == 50.0


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


def test_load_trading_state_reads_the_safety_locks(tmp_path: Path, settings, profile):
    repo = Repository.open(tmp_path / "locks.db", clock=lambda: NOW)
    repo.add_wallet_event(kind="balance", amount_usd=300.0, ts=local(10, 1, 9))
    repo.add_trade(
        symbol="BTC",
        side="long",
        pnl_usd=-220.0,
        opened_at=local(10, 5, 9),
        closed_at=local(10, 5, 10),
    )
    stored = repo.add_safety_lock(triggered_at=local(10, 5, 12), drawdown_pct=73.33)
    repo.add_safety_lock(triggered_at=local(9, 15))  # mes anterior
    state = load_trading_state(repo, settings, profile, NOW)
    assert [lk.id for lk in state.month_locks] == [stored.id]
    assert state.wallet.restart is not None
    assert state.wallet.restart.start_balance_usd == 80.0


def test_load_trading_state_works_with_a_source_without_locks(settings, profile):
    """Una fuente sin ``list_safety_locks`` (por ejemplo, un doble de prueba) sigue andando."""

    class Source:
        def list_trades(self):
            return LOCK_TRADES

        def list_open_positions(self):
            return []

        def list_wallet_events(self):
            return LOCK_EVENTS

        def list_volatility_events(self, *, ending_after=None):
            return []

    state = load_trading_state(Source(), settings, profile, NOW)
    assert state.month_locks == () and state.wallet.restart is None


# --- Fix 7: decisiones de la tercera ronda y revisión 6 ----------------------------------

#: 300 el 01/10 y -200 el 05/10: saldo 100, máximo 300 (66,67 % abajo).
REFILL_BASE = [wallet(local(10, 1, 9), "balance", 300.0)]
REFILL_TRADES = [trade(local(10, 5), -200.0)]


def refill_state(settings, profile, moves, extra_trades=()):
    events = [*REFILL_BASE, *(wallet(local(10, 6, 10 + i), k, a) for i, (k, a) in enumerate(moves))]
    return build(settings, profile, trades=[*REFILL_TRADES, *extra_trades], events=events)


def test_config_has_the_third_round_decisions(settings):
    dd = settings.circuit_breakers.monthly_drawdown_lock
    assert dd.deposit_refills_withdrawals is True
    assert dd.backfill_unrecorded_lock is True
    assert settings.circuit_breakers.max_trades_blocks_status is True


@pytest.mark.parametrize(
    ("moves", "peak", "pct"),
    [
        ([], 300.0, 200 / 3),
        ([("withdrawal", 100.0), ("deposit", 100.0)], 300.0, 200 / 3),  # retira todo y vuelve
        ([("withdrawal", 90.0), ("deposit", 90.0)], 300.0, 200 / 3),
        ([("deposit", 200.0)], 500.0, 40.0),  # sin retiros: D4 igual que antes
    ],
)
def test_withdrawing_and_depositing_back_keeps_the_drawdown(settings, profile, moves, peak, pct):
    """Tercera ronda (E1): retirar X y volver a depositar X deja el % igual."""
    w = refill_state(settings, profile, moves).wallet
    assert w.current_balance_usd == (300.0 if moves == [("deposit", 200.0)] else 100.0)
    assert w.month_peak_usd == pytest.approx(peak)
    assert w.drawdown_pct == pytest.approx(pct, abs=1e-9)
    assert w.deposit_refills_withdrawals is True
    assert w.month_withdrawn_open_usd == 0.0


def test_deposit_then_withdraw_follows_the_rules_in_order(settings, profile):
    """Tercera ronda (E1), documentado: depositar 90 y después retirar 90 da 51,28 %.

    El depósito no tiene nada que devolver (no hubo retiros): sube el máximo
    90 (300 → 390, saldo 190, 51,28 %). El retiro baja el máximo en la misma
    proporción que el saldo (390 × 100 / 190 = 205,26): el % queda en 51,28.
    """
    w = refill_state(settings, profile, [("deposit", 90.0), ("withdrawal", 90.0)]).wallet
    assert w.current_balance_usd == 100.0
    assert w.month_peak_usd == pytest.approx(205.26, abs=0.01)
    assert w.drawdown_pct == pytest.approx(200 / 390 * 100, abs=0.01)  # 51,28 %
    assert w.month_withdrawn_open_usd == 90.0  # un depósito futuro lo devuelve sin cambiar el %


def test_a_round_trip_no_longer_dodges_the_lock(settings, profile):
    """Revisión 6: con la ida y vuelta, otra pérdida de 69 tiene que bloquear (89,67 %)."""
    from app.safety.breakers import MONTHLY_DRAWDOWN_LOCK, evaluate_breakers
    from app.safety.lockfile import LockInfo

    moves = [("withdrawal", 100.0), ("deposit", 100.0)]
    state = refill_state(settings, profile, moves, [trade(local(10, 7), -69.0)])
    assert state.wallet.drawdown_pct == pytest.approx(269 / 3, abs=1e-9)
    v = evaluate_breakers(state, profile, settings, NOW, lock=LockInfo(False, "LOCKED"))
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).triggered and v.needs_lock


def test_a_bigger_deposit_refills_first_and_raises_the_peak_with_the_rest(settings, profile):
    # Retira 50 (100 → 50, máximo 150, 66,67 %) y deposita 150: 50 vuelven sin
    # cambiar el % (máximo 300, saldo 100) y los otros 100 suben el máximo (400).
    w = refill_state(settings, profile, [("withdrawal", 50.0), ("deposit", 150.0)]).wallet
    assert w.current_balance_usd == 200.0
    assert w.month_peak_usd == pytest.approx(400.0)
    assert w.drawdown_pct == pytest.approx(50.0)


def test_partial_refills_and_trades_in_between(settings, profile):
    # Retira 100 (todo), deposita 40 (vuelve con el % recordado: máximo 120,
    # saldo 40), gana 20 (60) y deposita 60: completa lo retirado con el % de
    # ese momento (120/60 = 2 → máximo 240, saldo 120: 50 %).
    moves = [("withdrawal", 100.0), ("deposit", 40.0)]
    events = [
        *REFILL_BASE,
        *(wallet(local(10, 6, 10 + i), k, a) for i, (k, a) in enumerate(moves)),
        wallet(local(10, 8), "deposit", 60.0),
    ]
    trades = [*REFILL_TRADES, trade(local(10, 7), 20.0)]
    w = build(settings, profile, trades=trades, events=events).wallet
    assert w.current_balance_usd == 120.0
    assert w.month_peak_usd == pytest.approx(240.0)
    assert w.drawdown_pct == pytest.approx(50.0)
    assert w.month_withdrawn_open_usd == 0.0


def test_refills_follow_the_inverse_of_the_withdrawal_rule():
    events = [
        *REFILL_BASE,
        wallet(local(10, 6), "withdrawal", 50.0),
        wallet(local(10, 7), "deposit", 50.0),
    ]
    for rule, peak in (("subtract", 300.0), ("none", 300.0), ("proportional", 300.0)):
        w = wallet_state(
            events,
            REFILL_TRADES,
            NOW,
            month_start=MONTH_START,
            assumed_start_usd=300.0,
            withdrawal_peak=rule,
            deposit_refills_withdrawals=True,
        )
        assert w.month_peak_usd == pytest.approx(peak), rule
        assert w.drawdown_pct == pytest.approx(200 / 3), rule
    old = wallet_state(events, REFILL_TRADES, NOW, month_start=MONTH_START, assumed_start_usd=300.0)
    assert old.month_peak_usd == pytest.approx(200.0)  # sin la regla nueva: 150 + 50


def test_state_without_refills_keeps_the_old_deposit_rule(settings, profile):
    off = with_drawdown(settings, deposit_refills_withdrawals=False)
    moves = [("withdrawal", 100.0), ("deposit", 100.0)]
    w = refill_state(off, profile, moves).wallet
    assert w.drawdown_pct == 0.0 and w.deposit_refills_withdrawals is False
    d = refill_state(settings, profile, moves).to_dict()["wallet"]
    assert d["deposit_refills_withdrawals"] is True and d["month_withdrawn_open_usd"] == 0.0


def test_restart_can_refill_a_withdrawal_made_before_the_lock(settings, profile):
    # 300 → 80 (bloqueo 05/10 12:00); antes del bloqueo retiró 0. Después del
    # desbloqueo: pierde 40 (80 → 40, 50 % desde el bloqueo), retira 20
    # (proporcional: máximo 40) y vuelve a depositar 20: sigue en 50 %.
    events = [
        *LOCK_EVENTS,
        wallet(local(10, 7), "withdrawal", 20.0),
        wallet(local(10, 8), "deposit", 20.0),
    ]
    trades = [*LOCK_TRADES, trade(local(10, 6), -40.0)]
    state = build(settings, profile, trades=trades, events=events, locks=[lock(local(10, 5, 12))])
    r = state.wallet.restart
    assert r is not None
    assert r.current_balance_usd == 40.0
    assert r.drawdown_pct == pytest.approx(50.0)
    assert r.withdrawn_open_usd == 0.0


def test_a_balance_noted_exactly_at_the_month_start_is_the_start_balance():
    """Revisión 6: un saldo anotado el 01/10 00:00 es el saldo de arranque, no una caída."""
    trades = [trade(local(9, 10), -50.0)]  # septiembre: 300 → 250 calculado
    for ts in (MONTH_START, MONTH_START - timedelta(minutes=1)):
        events = [wallet(local(9, 1, 9), "balance", 300.0), wallet(ts, "balance", 200.0)]
        w = wallet_of(events, trades)
        assert w.month_start_balance_usd == 200.0
        assert (w.month_peak_usd, w.current_balance_usd, w.drawdown_pct) == (200.0, 200.0, 0.0)
        assert w.anchor is not None and w.anchor.amount_usd == 200.0
    # Una operación del mismo instante ya está incluida en ese saldo.
    same = [wallet(local(9, 1, 9), "balance", 300.0), wallet(MONTH_START, "balance", 200.0)]
    w = wallet_of(same, [*trades, trade(MONTH_START, -30.0)])
    assert (w.month_start_balance_usd, w.current_balance_usd, w.drawdown_pct) == (200.0, 200.0, 0.0)
    # Un saldo más bajo un rato después sí es una caída del mes.
    later = [wallet(local(9, 1, 9), "balance", 300.0), wallet(local(10, 1, 0, 1), "balance", 200.0)]
    w = wallet_of(later, trades)
    assert w.month_start_balance_usd == 250.0 and w.drawdown_pct == 20.0


def test_a_loss_after_a_win_during_the_pause_stretches_it():
    """Revisión 6 (D2): cualquier pérdida durante la pausa la estira, aunque antes ganara."""
    losses = [trade(local(10, 13, h), -5.0) for h in (9, 10, 11)]  # pausa 11:00-15:00
    win = trade(local(10, 13, 12), 5.0)
    late_loss = trade(local(10, 13, 13), -5.0)
    s = streak([*losses, win, late_loss], now=local(10, 13, 14, 30))
    assert s.cooldown_started_at == local(10, 13, 11)
    assert s.cooldown_until == local(10, 13, 17)
    assert s.cooldown_active() and s.count == 1
    assert s.cooldown_trigger_ids == tuple(t["id"] for t in losses)
    assert s.cooldown_extended_at == local(10, 13, 13)
    assert s.cooldown_extended_by_id == late_loss["id"]
    assert s.losses_during_cooldown == 1
    d = s.to_dict()
    assert d["cooldown_trigger_ids"] == [t["id"] for t in losses]
    assert d["cooldown_extended_at"] == local(10, 13, 13).astimezone(UTC).isoformat()
    # A las 15:30 sigue la pausa; a las 17:00 terminó y la cuenta vuelve a cero.
    assert streak([*losses, win, late_loss], now=local(10, 13, 15, 30)).cooldown_active()
    done = streak([*losses, win, late_loss], now=local(10, 13, 17))
    assert not done.cooldown_active() and done.count == 0
    # Una pérdida justo a la hora de fin no la estira y cuenta 1.
    edge = streak([*losses, trade(local(10, 13, 15), -5.0)], now=local(10, 13, 15, 30))
    assert edge.cooldown_until == local(10, 13, 15) and edge.count == 1
    assert edge.cooldown_extended_at is None


def test_a_late_loss_closed_before_the_lock_is_part_of_that_incident(settings, profile):
    """Tercera ronda (E3): cargada tarde pero cerrada antes del bloqueo: no es caída nueva."""
    late = trade(local(10, 5, 11), -30.0)  # cerró antes del bloqueo de las 12:00
    state = build(
        settings,
        profile,
        trades=[*LOCK_TRADES, late],
        events=LOCK_EVENTS,
        locks=[lock(local(10, 5, 12))],
    )
    r = state.wallet.restart
    assert r is not None
    assert r.start_balance_usd == 50.0  # 300 - 220 - 30
    assert r.drawdown_pct == 0.0
    assert state.wallet.drawdown_pct == pytest.approx(250 / 3)  # el mes sí lo cuenta


def test_restarts_are_measured_from_every_lock_of_the_month(settings, profile):
    first = lock(local(10, 5, 12))
    second = lock(local(10, 7, 12))
    trades = [*LOCK_TRADES, trade(local(10, 7, 10), -56.0)]  # 80 → 24 (70 % desde 80)
    state = build(settings, profile, trades=trades, events=LOCK_EVENTS, locks=[second, first])
    w = state.wallet
    assert [r.lock_id for r in w.restarts] == [first["id"], second["id"]]
    assert w.restart is w.restarts[-1]
    earlier = w.restart_before(local(10, 7, 12))
    assert earlier is not None and earlier.lock_id == first["id"]
    assert (earlier.peak_usd, earlier.current_balance_usd, earlier.drawdown_pct) == (
        80.0,
        24.0,
        70.0,
    )
    assert w.restart_before(local(10, 5, 12)) is None
    d = state.to_dict()["wallet"]
    assert [r["lock_id"] for r in d["restarts"]] == [first["id"], second["id"]]


def test_negative_balance_is_flagged_for_the_texts():
    w = wallet_of([wallet(local(10, 1, 9), "balance", 50.0)], [trade(local(10, 5), -80.0)])
    assert w.negative_balance is True
    assert w.to_dict()["negative_balance"] is True
    assert any("mal tipeado" in x for x in w.warnings)
    assert wallet_of().negative_balance is False


def test_records_keep_the_planned_risk(settings, profile):
    pos = PositionRecord.from_value(
        {"id": 1, "symbol": "btc", "side": "long", "risk_usd": 0, "initial_risk_usd": 40.0}
    )
    assert (pos.risk_usd, pos.initial_risk_usd, pos.planned_risk_usd) == (0.0, 40.0, 40.0)
    assert pos.to_dict()["planned_risk_usd"] == 40.0
    legacy = PositionRecord.from_value({"id": 2, "symbol": "eth", "side": "short", "risk_usd": 30})
    assert legacy.initial_risk_usd is None and legacy.planned_risk_usd == 30.0
    t = TradeRecord.from_value(trade(local(10, 5), -80.0, risk_usd=80.0, initial_risk_usd=50.0))
    assert t.planned_risk_usd == 50.0 and t.to_dict()["initial_risk_usd"] == 50.0
    state = build(settings, profile, positions=[pos, legacy])
    assert state.open_risk_usd == 30.0  # el límite del día usa el riesgo de ahora


# --- revisión 7: billetera vaciada por un retiro ---------------------------------------


def empty_wallet_state(settings, profile, middle_events=(), middle_trades=(), deposits=(100.0,)):
    """300 → 100 (66,67 %), retira 100 (todo) el 06/10, lo del medio y los depósitos del 07/10."""
    events = [
        *REFILL_BASE,
        wallet(local(10, 6, 10), "withdrawal", 100.0),
        *middle_events,
        *(wallet(local(10, 7, 10 + i), "deposit", a) for i, a in enumerate(deposits)),
    ]
    return build(settings, profile, trades=[*REFILL_TRADES, *middle_trades], events=events)


def test_a_balance_noted_while_the_wallet_is_empty_keeps_the_remembered_drawdown(settings, profile):
    """Revisión 7: retirar todo, anotar el saldo real (5) y depositar 95 sigue en 66,67 %."""
    note = [wallet(local(10, 6, 12), "balance", 5.0)]
    w = empty_wallet_state(settings, profile, note, deposits=()).wallet
    assert (w.current_balance_usd, w.month_peak_usd) == (5.0, 15.0)
    assert w.drawdown_pct == pytest.approx(200 / 3)
    w = empty_wallet_state(settings, profile, note, deposits=(95.0,)).wallet
    assert (w.current_balance_usd, w.month_peak_usd) == (100.0, 300.0)
    assert w.drawdown_pct == pytest.approx(200 / 3)


def test_a_win_while_the_wallet_is_empty_keeps_the_remembered_drawdown(settings, profile):
    """Revisión 7: retirar todo, ganar 2 y depositar 100; después perder 60 bloquea."""
    from app.safety.breakers import MONTHLY_DRAWDOWN_LOCK, evaluate_breakers
    from app.safety.lockfile import LockInfo

    win = [trade(local(10, 6, 12), 2.0)]
    w = empty_wallet_state(settings, profile, middle_trades=win).wallet
    assert w.current_balance_usd == 102.0
    assert w.drawdown_pct == pytest.approx(200 / 3)
    state = empty_wallet_state(
        settings, profile, middle_trades=[*win, trade(local(10, 8, 12), -60.0)]
    )
    assert state.wallet.current_balance_usd == 42.0
    assert state.wallet.drawdown_pct > 70.0  # antes daba 58,82 % y no bloqueaba
    v = evaluate_breakers(state, profile, settings, NOW, lock=LockInfo(False, "LOCKED"))
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).triggered


@pytest.mark.parametrize("deposits", [(100.0,), (5.0, 95.0), (50.0, 50.0)])
def test_a_loss_while_the_wallet_is_empty_still_counts_after_the_refill(
    settings, profile, deposits
):
    """Revisión 7: retirar todo, perder 10 y volver a depositar da 70 %, como sin retirar."""
    from app.safety.breakers import MONTHLY_DRAWDOWN_LOCK, evaluate_breakers
    from app.safety.lockfile import LockInfo

    loss = [trade(local(10, 6, 12), -10.0)]
    state = empty_wallet_state(settings, profile, middle_trades=loss, deposits=deposits)
    w = state.wallet
    if deposits == (50.0, 50.0):
        # El primer depósito ya deja saldo positivo (40, máximo 150: 73,33 %) y
        # el segundo devuelve el resto sin cambiar ese % (las reglas en orden).
        assert w.drawdown_pct == pytest.approx(110 / 150 * 100)
    else:
        assert (w.current_balance_usd, w.month_peak_usd) == (90.0, 300.0)
        assert w.drawdown_pct == pytest.approx(70.0)
    v = evaluate_breakers(state, profile, settings, NOW, lock=LockInfo(False, "LOCKED"))
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).triggered
    plain = build(settings, profile, trades=[*REFILL_TRADES, *loss], events=REFILL_BASE).wallet
    assert plain.drawdown_pct == pytest.approx(70.0)


def test_withdrawing_more_than_the_balance_and_back_keeps_the_drawdown(settings, profile):
    """Retirar 120 con 100 de saldo (-20), otro retiro de 10, ganar 15 y devolver 130."""
    events = [
        *REFILL_BASE,
        wallet(local(10, 6, 10), "withdrawal", 120.0),
        wallet(local(10, 6, 11), "withdrawal", 10.0),
        wallet(local(10, 7, 10), "deposit", 130.0),
    ]
    w = build(
        settings, profile, trades=[*REFILL_TRADES, trade(local(10, 6, 12), 15.0)], events=events
    ).wallet
    # Como si no hubiera retirado nada: 300 → 100 y +15 → 115 (61,67 %).
    assert (w.current_balance_usd, w.month_peak_usd) == (115.0, 300.0)
    assert w.drawdown_pct == pytest.approx(185 / 3)
    assert w.month_withdrawn_open_usd == 0.0


def test_losses_closed_at_the_same_instant_as_the_trigger_belong_to_it():
    """Revisión 7: dos stops que saltan juntos a las 11:00 no "estiran" la pausa."""
    trades = [trade(local(10, 13, h), -5.0) for h in (9, 10, 11)]
    same = trade(local(10, 13, 11), -5.0, symbol="ETH")
    s = streak([*trades, same], now=local(10, 13, 12))
    assert s.cooldown_started_at == local(10, 13, 11)
    assert s.cooldown_until == local(10, 13, 15)
    assert s.cooldown_trigger_ids == (*(t["id"] for t in trades), same["id"])
    assert s.cooldown_extended_at is None and s.cooldown_extended_by_id is None
    assert s.losses_during_cooldown == 0
    later = trade(local(10, 13, 12), -5.0)  # esta sí la estira
    s = streak([*trades, same, later], now=local(10, 13, 12, 30))
    assert s.cooldown_until == local(10, 13, 16)
    assert s.cooldown_extended_by_id == later["id"] and s.losses_during_cooldown == 1


def test_withdrawal_effect_uses_balances_noted_later():
    """Revisión 7 (E9): un retiro con fecha anterior al primer saldo anotado sube el de antes."""
    from app.safety.state import withdrawal_effect

    events = [wallet(local(10, 10), "balance", 1000.0)]
    trades = [trade(local(10, 7), -100.0)]
    args = {"now": NOW, "month_start": MONTH_START, "assumed_start_usd": 300.0}
    e = withdrawal_effect(events, trades, 400.0, local(10, 5, 10), **args)
    # 05/10: 1.000 + 100 (la pérdida del 07/10) + 400 (el retiro) = 1.500 antes del retiro.
    assert (e.balance_before_usd, e.leaves_negative, e.assumed) == (1500.0, False, False)
    # Después del saldo anotado: baja el saldo de hoy.
    e = withdrawal_effect(events, trades, 1200.0, local(10, 12), **args)
    assert (e.balance_before_usd, e.negative_balance_usd) == (1000.0, -200.0)
    assert e.negative_at == local(10, 12).astimezone(UTC)
    # Sin saldos anotados: el supuesto al empezar el mes.
    e = withdrawal_effect([], trades, 100.0, local(10, 8), **args)
    assert (e.balance_before_usd, e.assumed) == (200.0, True)


def test_withdrawal_effect_sees_a_later_balance_go_negative():
    """Revisión 7 (E9): 300 el 01/10, -100 el 05/10, retiro de 200 hoy; otro de 250 el 02/10."""
    from app.safety.state import withdrawal_effect

    events = [
        wallet(local(10, 1, 9), "balance", 300.0),
        wallet(local(10, 13, 9), "withdrawal", 200.0),
    ]
    trades = [trade(local(10, 5, 10), -100.0)]
    args = {"now": NOW, "month_start": MONTH_START, "assumed_start_usd": 300.0}
    e = withdrawal_effect(events, trades, 250.0, local(10, 2, 10), **args)
    assert e.balance_before_usd == 300.0  # en su fecha alcanzaba...
    # ...pero con la pérdida del 05/10 queda en -50 (y hoy, con el otro retiro, en -250).
    assert e.leaves_negative and e.negative_balance_usd == -50.0
    assert e.negative_at == local(10, 5, 10).astimezone(UTC)
