"""Hito 5: circuit breakers y veredicto de seguridad (``app.safety.breakers``).

Perfil agresivo de ``config.yaml``: límite diario 100 USD (riesgo × 0,5 desde
50 USD, bloqueo desde 100), billetera mensual 300 USD (riesgo × 0,5 con 180 USD
perdidos en la semana), 3 pérdidas seguidas → pausa de 4 h y bloqueo total con
70 % de caída desde el máximo del mes. "Ahora" es el martes 13/10/2026 15:00 de
Buenos Aires.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.risk.profiles import get_profile
from app.safety import breakers as br
from app.safety.breakers import (
    BREAKER_CODES,
    DAILY_LOSS_BLOCK,
    DAILY_LOSS_REDUCE,
    LOCK_FILE,
    LOSING_STREAK,
    MONTHLY_DRAWDOWN_LOCK,
    VOLATILITY_EVENT,
    WEEKLY_LOSS_REDUCE,
    daily_loss_block_breaker,
    daily_loss_reduce_breaker,
    evaluate_breakers,
    lock_file_breaker,
    losing_streak_breaker,
    monthly_drawdown_lock_breaker,
    trigger_lock_if_needed,
    until_phrase,
    volatility_event_breaker,
    weekly_loss_reduce_breaker,
)
from app.safety.lockfile import LockInfo, read_lock, write_lock
from app.safety.state import build_trading_state, load_trading_state
from app.storage import Repository

BA_NAME = "America/Argentina/Buenos_Aires"
BA = ZoneInfo(BA_NAME)


def local(month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=BA)


NOW = local(10, 13, 15, 0)
NO_LOCK = LockInfo(exists=False, path="LOCKED")

_ids = iter(range(1, 100_000))


def trade(closed: datetime, pnl: float, **kw) -> dict:
    data = {
        "id": next(_ids),
        "symbol": "BTC",
        "side": "long",
        "opened_at": closed - timedelta(minutes=30),
        "closed_at": closed,
        "pnl_usd": pnl,
    }
    data.update(kw)
    return data


def balance(ts: datetime, amount: float) -> dict:
    return {"id": next(_ids), "ts": ts, "kind": "balance", "amount_usd": amount}


START_BALANCE = balance(local(10, 1, 9), 300.0)


@pytest.fixture
def profile(settings):
    return get_profile("agresivo", settings)


def state_for(settings, profile, *, now=NOW, trades=(), events=(START_BALANCE,), vol=()):
    return build_trading_state(
        trades=list(trades),
        open_positions=[],
        wallet_events=list(events),
        volatility_events=list(vol),
        settings=settings,
        profile=profile,
        now=now,
    )


def verdict_for(settings, profile, *, now=NOW, lock=NO_LOCK, **kw):
    state = state_for(settings, profile, now=now, **kw)
    return evaluate_breakers(state, profile, settings, now, lock=lock)


def with_breakers(settings, **changes):
    cb = settings.circuit_breakers.model_copy(update=changes)
    return settings.model_copy(update={"circuit_breakers": cb})


def test_thresholds_come_from_the_config(settings, profile):
    cb = settings.circuit_breakers
    assert cb.daily_loss_reduce.pct_of_daily_limit == 50
    assert cb.daily_loss_reduce.risk_multiplier == 0.5
    assert cb.daily_loss_block.pct_of_daily_limit == 100
    assert (cb.losing_streak.consecutive_losses, cb.losing_streak.cooldown_hours) == (3, 4)
    assert cb.weekly_loss_reduce.pct_of_monthly_wallet == 60
    assert cb.monthly_drawdown_lock.pct_from_peak == 70
    assert (profile.max_daily_loss_usd, profile.monthly_wallet_usd) == (100, 300)


# --- pérdida del día: bajar el riesgo ---------------------------------------------------


def test_daily_reduce_just_below_half_of_the_limit(settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 13, 10), -49.99)])
    b = daily_loss_reduce_breaker(state, profile, settings, NOW)
    assert not b.triggered
    assert (b.action, b.risk_multiplier, b.until) == ("none", 1.0, None)
    assert (b.threshold, b.current) == (50.0, 49.99)
    assert b.progress == pytest.approx(0.9998)
    assert (b.threshold_text, b.current_text) == ("50 USD", "49,99 USD")
    assert b.resets_at == state.periods.day_end
    assert "Hoy perdiste 49,99 USD" in b.message


def test_daily_reduce_at_exactly_half_of_the_limit(settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 13, 10), -50.0)])
    b = daily_loss_reduce_breaker(state, profile, settings, NOW)
    assert b.triggered
    assert (b.kind, b.action, b.risk_multiplier) == ("reduce", "reduce", 0.5)
    assert b.until == local(10, 14).astimezone(UTC)
    assert b.until_text == "hasta que termine el día (a las 00:00)"
    assert b.progress == 1.0
    assert "llegaste al 50 %" in b.message
    assert "se multiplica por 0,5" in b.message
    assert not b.blocks


def test_daily_reduce_uses_the_net_result_of_the_day(settings, profile):
    """Sin el freno trabado (``latch_until_period_end: false``) mira el neto actual."""
    trades = [trade(local(10, 13, 9), -40.0), trade(local(10, 13, 11), 20.0)]
    net = daily_loss_reduce_breaker(
        state_for(settings, profile, trades=trades), profile, settings, NOW
    )
    assert (net.current, net.triggered) == (40.0, False)
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 20.0)]
    unlatched = with_breakers(settings, latch_until_period_end=False)
    n = daily_loss_reduce_breaker(
        state_for(unlatched, profile, trades=trades), profile, unlatched, NOW
    )
    assert (n.current, n.triggered) == (40.0, False)
    gross = with_breakers(settings, loss_basis="gross", latch_until_period_end=False)
    g = daily_loss_reduce_breaker(state_for(gross, profile, trades=trades), profile, gross, NOW)
    assert (g.current, g.triggered) == (60.0, True)
    assert "sumando solo las operaciones perdidas" in g.message


def test_daily_reduce_on_a_winning_day(settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 13, 9), 35.0)])
    b = daily_loss_reduce_breaker(state, profile, settings, NOW)
    assert b.current == 0.0 and b.progress == 0.0
    assert "Hoy vas ganando +35 USD" in b.message


def test_yesterday_losses_do_not_count_today(settings, profile):
    trades = [trade(local(10, 12, 23, 30), -80.0), trade(local(10, 13, 0, 30), -10.0)]
    b = daily_loss_reduce_breaker(
        state_for(settings, profile, trades=trades), profile, settings, NOW
    )
    assert b.current == 10.0 and not b.triggered


def test_daily_reduce_threshold_follows_the_config(settings, profile):
    custom = with_breakers(
        settings,
        daily_loss_reduce=settings.circuit_breakers.daily_loss_reduce.model_copy(
            update={"pct_of_daily_limit": 40, "risk_multiplier": 0.25}
        ),
    )
    state = state_for(custom, profile, trades=[trade(local(10, 13, 10), -40.0)])
    b = daily_loss_reduce_breaker(state, profile, custom, NOW)
    assert (b.threshold, b.triggered, b.risk_multiplier) == (40.0, True, 0.25)


# --- pérdida del día: bloqueo ----------------------------------------------------------


def test_daily_block_just_below_the_limit(settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 13, 10), -99.99)])
    b = daily_loss_block_breaker(state, profile, settings, NOW)
    assert not b.triggered and b.action == "none"
    assert "Te quedan 0,01 USD" in b.message


def test_daily_block_at_exactly_the_limit_until_tomorrow(settings, profile):
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 10), -40.0)]
    state = state_for(settings, profile, trades=trades)
    b = daily_loss_block_breaker(state, profile, settings, NOW)
    assert b.triggered and b.blocks
    assert (b.kind, b.action, b.risk_multiplier) == ("block", "block", 1.0)
    assert b.until == state.periods.day_end
    assert "llegaste a tu límite de pérdida del día (100 USD)" in b.message
    assert "Mañana arrancás de cero" in b.message
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.status == "rojo" and v.blocked
    assert v.block_codes == (DAILY_LOSS_BLOCK,)
    assert v.breaker(DAILY_LOSS_REDUCE).triggered  # el de 50 % también saltó


# --- frenos trabados hasta el fin del período (latch_until_period_end) ------------------


def test_a_later_win_does_not_switch_off_the_daily_reduction(settings, profile):
    """Hallazgo 4 / decisión del usuario: -60 a las 10:00 y +20 a las 11:00."""
    trades = [trade(local(10, 13, 10), -60.0), trade(local(10, 13, 11), 20.0)]
    state = state_for(settings, profile, trades=trades)
    b = daily_loss_reduce_breaker(state, profile, settings, NOW)
    assert b.triggered and b.risk_multiplier == 0.5
    assert (b.current, b.current_text) == (60.0, "60 USD")
    assert b.until == state.periods.day_end
    assert b.message.startswith("Hoy llegaste a perder 60 USD (ahora vas -40 USD): pasaste el 50 %")
    assert "hasta que termine el día (a las 00:00)" in b.message
    assert "Lo que ganaste después no apaga el freno." in b.message
    assert b.summary.startswith("llegaste a perder 60 USD hoy")
    assert "aunque después ganes" in b.description
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.status == "ambar" and v.risk_multiplier == 0.5
    assert state.plan_pnl_today_usd == -40.0  # el plan sigue mirando el resultado actual


def test_a_later_win_does_not_switch_off_the_daily_block(settings, profile):
    trades = [trade(local(10, 13, 10), -100.0), trade(local(10, 13, 11), 130.0)]
    state = state_for(settings, profile, trades=trades)
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.blocked and v.block_codes == (DAILY_LOSS_BLOCK,)
    b = v.breaker(DAILY_LOSS_BLOCK)
    assert b.message.startswith("Hoy llegaste a perder 100 USD (ahora vas +30 USD)")
    assert "Lo que ganaste después no apaga el freno. Mañana arrancás de cero." in b.message
    # Al día siguiente arranca de cero.
    tomorrow = local(10, 14, 9)
    next_day = state_for(settings, profile, trades=trades, now=tomorrow)
    assert not evaluate_breakers(next_day, profile, settings, tomorrow, lock=NO_LOCK).blocked


def test_a_later_win_does_not_switch_off_the_weekly_reduction(settings, profile):
    trades = [trade(local(10, 12, 10), -180.0), trade(local(10, 13, 10), 40.0)]
    state = state_for(settings, profile, trades=trades)
    b = weekly_loss_reduce_breaker(state, profile, settings, NOW)
    assert b.triggered and b.until == local(10, 19).astimezone(UTC)
    assert b.current == 180.0
    assert b.message.startswith("En la semana llegaste a perder 180 USD (ahora vas -140 USD)")
    assert b.message.endswith("Lo que ganaste después no apaga el freno.")
    assert b.summary.startswith("llegaste a perder 180 USD en la semana")


def test_without_latch_a_later_win_switches_the_breakers_off(settings, profile):
    off = with_breakers(settings, latch_until_period_end=False)
    trades = [trade(local(10, 13, 10), -60.0), trade(local(10, 13, 11), 20.0)]
    b = daily_loss_reduce_breaker(state_for(off, profile, trades=trades), profile, off, NOW)
    assert not b.triggered and b.current == 40.0
    assert "llegaste a perder" not in b.message
    assert "aunque después ganes" not in b.description


def test_latched_but_below_the_threshold_shows_both_numbers(settings, profile):
    trades = [trade(local(10, 13, 9), -40.0), trade(local(10, 13, 10), 30.0)]
    state = state_for(settings, profile, trades=trades)
    reduce = daily_loss_reduce_breaker(state, profile, settings, NOW)
    assert not reduce.triggered and reduce.current == 40.0
    assert reduce.progress == pytest.approx(0.8)
    assert reduce.message.startswith("Hoy llegaste a perder 40 USD (ahora vas -10 USD).")
    block = daily_loss_block_breaker(state, profile, settings, NOW)
    assert "Te quedan 90 USD antes de llegar al límite del día" in block.message  # desde -10


def test_latch_with_gross_basis_counts_every_loss(settings, profile):
    gross = with_breakers(settings, loss_basis="gross")
    trades = [trade(local(10, 13, 9), -30.0), trade(local(10, 13, 10), 40.0)]
    trades.append(trade(local(10, 13, 11), -25.0))
    b = daily_loss_reduce_breaker(state_for(gross, profile, trades=trades), profile, gross, NOW)
    assert b.triggered and b.current == 55.0
    assert "llegaste a perder" not in b.message  # con "gross" la actual ya es la más grande
    assert "sumando solo las operaciones perdidas" in b.message


# --- pérdidas seguidas ------------------------------------------------------------------


def test_two_losses_in_a_row_warn_about_the_next_one(settings, profile):
    trades = [trade(local(10, 13, 9), -5.0), trade(local(10, 13, 10), -5.0)]
    b = losing_streak_breaker(state_for(settings, profile, trades=trades), profile, settings, NOW)
    assert not b.triggered
    assert (b.current, b.threshold) == (2.0, 3.0)
    assert b.progress == pytest.approx(2 / 3)
    assert b.current_text == "2 pérdidas seguidas"
    assert "si la próxima también pierde" in b.message


def opened_yesterday(closed: datetime, pnl: float, **kw) -> dict:
    """Operación abierta ayer (no cuenta para el máximo de operaciones de hoy)."""
    return trade(closed, pnl, opened_at=local(10, 12, 20), **kw)


def test_third_loss_blocks_for_four_hours(settings, profile):
    trades = [opened_yesterday(local(10, 13, h), -5.0) for h in (12, 13, 14)]
    state = state_for(settings, profile, trades=trades)
    b = losing_streak_breaker(state, profile, settings, NOW)
    assert b.triggered and b.blocks
    assert b.until == local(10, 13, 18).astimezone(UTC)
    assert b.until_text == "hasta hoy a las 18:00"
    assert b.progress == 1.0
    assert "pausa de 4 horas" in b.message
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.block_codes == (LOSING_STREAK,)
    assert "pausa por 3 pérdidas seguidas" in v.headline


def test_a_win_during_the_pause_keeps_blocking_and_the_message_makes_sense(settings, profile):
    trades = [trade(local(10, 13, h), -5.0) for h in (12, 13, 14)]
    trades.append(trade(local(10, 13, 14, 30), 40.0))  # ganó durante la pausa
    b = losing_streak_breaker(state_for(settings, profile, trades=trades), profile, settings, NOW)
    assert b.triggered
    assert b.current == 3.0
    assert "Perdiste 3 operaciones seguidas" in b.message
    assert "Perdiste 0" not in b.message


def test_pause_ends_exactly_at_the_cooldown_end_and_the_count_resets(settings, profile):
    trades = [trade(local(10, 13, h), -5.0) for h in (9, 10, 11)]  # pausa hasta las 15:00
    state = state_for(settings, profile, trades=trades, now=NOW)
    b = losing_streak_breaker(state, profile, settings, NOW)
    assert not b.triggered
    assert b.current == 0.0
    assert "No venís con pérdidas seguidas" in b.message
    before = state_for(settings, profile, trades=trades, now=NOW - timedelta(minutes=1))
    assert losing_streak_breaker(before, profile, settings, NOW - timedelta(minutes=1)).triggered
    # Evaluado más tarde con el mismo estado: la pausa terminó y la cuenta vuelve a 0.
    late = losing_streak_breaker(before, profile, settings, NOW)
    assert not late.triggered and late.current == 0.0


def test_a_loss_during_the_pause_stretches_it_four_hours(settings, profile):
    trades = [trade(local(10, 13, h), -5.0) for h in (11, 12, 13)]  # pausa hasta 17:00
    trades.append(trade(local(10, 13, 14), -5.0))  # operó igual y perdió
    b = losing_streak_breaker(state_for(settings, profile, trades=trades), profile, settings, NOW)
    assert b.triggered and b.until == local(10, 13, 18).astimezone(UTC)
    assert b.message.startswith(
        "Perdiste 3 operaciones seguidas (la última cerró el 13/10/2026 13:00): pausa de 4 horas"
    )
    assert (
        "Después volviste a perder durante la pausa (la operación que cerró el 13/10/2026 "
        "14:00), así que la pausa se estiró 4 horas desde esa pérdida."
    ) in b.message
    assert b.message.endswith("No se generan planes hasta hoy a las 18:00.")
    assert "la pausa se estira 4 horas desde esa pérdida" in b.description


def test_a_loss_during_the_pause_after_a_win_still_stretches_it(settings, profile):
    """Revisión 6 (D2): ganada y después perdida dentro de la pausa: la estira igual."""
    trades = [opened_yesterday(local(10, 13, h), -5.0) for h in (9, 10, 11)]  # hasta 15:00
    trades.append(opened_yesterday(local(10, 13, 12), 5.0))  # ganó
    trades.append(opened_yesterday(local(10, 13, 13), -5.0))  # y perdió, en la pausa
    now = local(10, 13, 14, 30)
    state = state_for(settings, profile, trades=trades, now=now)
    assert state.streak.cooldown_until == local(10, 13, 17).astimezone(UTC)
    b = losing_streak_breaker(state, profile, settings, now)
    assert b.triggered and b.until == local(10, 13, 17).astimezone(UTC)
    assert b.message == (
        "Perdiste 3 operaciones seguidas (la última cerró el 13/10/2026 11:00): pausa de 4 "
        "horas para enfriar la cabeza. Después volviste a perder durante la pausa (la operación "
        "que cerró el 13/10/2026 13:00), así que la pausa se estiró 4 horas desde esa pérdida. "
        "No se generan planes hasta hoy a las 17:00."
    )
    assert b.current == 3.0
    later = local(10, 13, 15, 30)  # antes, a esta hora ya no bloqueaba
    late_state = state_for(settings, profile, trades=trades, now=later)
    assert losing_streak_breaker(late_state, profile, settings, later).triggered
    done = local(10, 13, 17)
    end_state = state_for(settings, profile, trades=trades, now=done)
    after = losing_streak_breaker(end_state, profile, settings, done)
    assert not after.triggered and after.current == 0.0  # y la cuenta vuelve a cero


def test_without_extension_the_pause_ends_at_the_planned_time(settings, profile):
    streak_cfg = settings.circuit_breakers.losing_streak.model_copy(
        update={"extend_cooldown_on_loss": False}
    )
    custom = with_breakers(settings, losing_streak=streak_cfg)
    trades = [trade(local(10, 13, h), -5.0) for h in (11, 12, 13)]  # pausa hasta 17:00
    trades.append(trade(local(10, 13, 14), -5.0))
    b = losing_streak_breaker(state_for(custom, profile, trades=trades), profile, custom, NOW)
    assert b.triggered and b.until == local(10, 13, 17).astimezone(UTC)
    assert "Perdiste 3 operaciones seguidas (la última cerró el 13/10/2026 13:00)" in b.message
    assert (
        "Volviste a perder durante la pausa, pero la pausa termina igual a la hora prevista."
    ) in b.message
    assert "la pausa termina igual a la hora prevista" in b.description
    at_end = local(10, 13, 17)
    late = state_for(custom, profile, trades=trades, now=at_end)
    assert not losing_streak_breaker(late, profile, custom, at_end).triggered


def test_without_reset_the_message_warns_the_streak_continues(settings, profile):
    streak_cfg = settings.circuit_breakers.losing_streak.model_copy(
        update={"reset_after_cooldown": False}
    )
    custom = with_breakers(settings, losing_streak=streak_cfg)
    trades = [trade(local(10, 13, h), -5.0) for h in (9, 10, 11)]
    b = losing_streak_breaker(state_for(custom, profile, trades=trades), profile, custom, NOW)
    assert not b.triggered and b.current == 3.0
    assert "La pausa terminó, pero la racha sigue" in b.message
    assert "solo con una operación ganada" in b.description


# --- pérdida de la semana ----------------------------------------------------------------


def weekly_trades(total_loss: float) -> list[dict]:
    """La pérdida repartida entre lunes y martes (sin tocar el freno diario de hoy)."""
    return [trade(local(10, 12, 10), -(total_loss - 20.0)), trade(local(10, 13, 9), -20.0)]


def test_weekly_reduce_just_below_60_percent_of_the_monthly_wallet(settings, profile):
    state = state_for(settings, profile, trades=weekly_trades(179.99))
    b = weekly_loss_reduce_breaker(state, profile, settings, NOW)
    assert not b.triggered
    assert (b.threshold, b.current) == (180.0, 179.99)
    assert b.resets_at == local(10, 19).astimezone(UTC)


def test_weekly_reduce_at_exactly_60_percent_until_the_end_of_the_week(settings, profile):
    state = state_for(settings, profile, trades=weekly_trades(180.0))
    b = weekly_loss_reduce_breaker(state, profile, settings, NOW)
    assert b.triggered
    assert (b.action, b.risk_multiplier) == ("reduce", 0.5)
    assert b.until == local(10, 19).astimezone(UTC)
    assert b.until_text == "hasta el lunes 19/10 a las 00:00"
    assert "60 % de tu billetera mensual (180 USD de 300 USD)" in b.message


def test_weekly_loss_from_last_week_does_not_count(settings, profile):
    trades = [trade(local(10, 11, 23, 30), -250.0), trade(local(10, 12, 0, 30), -10.0)]
    b = weekly_loss_reduce_breaker(
        state_for(settings, profile, trades=trades), profile, settings, NOW
    )
    assert b.current == 10.0


# --- caída del mes ----------------------------------------------------------------------


def test_monthly_drawdown_just_below_70_percent(settings, profile):
    trades = [trade(local(10, 5), -209.99)]
    state = state_for(settings, profile, trades=trades)
    b = monthly_drawdown_lock_breaker(state, profile, settings, NOW)
    assert not b.triggered
    assert b.current == pytest.approx(69.9966667)
    assert b.current_text != "70 %"  # no se redondea al umbral
    assert b.resets_at == local(11, 1).astimezone(UTC)
    assert "abajo de su máximo del mes (300 USD → 90,01 USD)" in b.message


def test_monthly_drawdown_at_exactly_70_percent_locks(settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 5), -210.0)])
    b = monthly_drawdown_lock_breaker(state, profile, settings, NOW)
    assert b.triggered and b.blocks
    assert (b.kind, b.action, b.until) == ("lock", "lock", None)
    assert b.until_text == "hasta que borres a mano el archivo LOCKED"
    assert b.current == 70.0
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.blocked and v.needs_lock
    assert v.status == "rojo"
    assert "la billetera del mes cayó 70 %" in v.headline
    assert "borrá a mano el archivo LOCKED" in v.headline


def test_monthly_drawdown_message_at_the_peak_and_with_assumed_balance(settings, profile):
    b = monthly_drawdown_lock_breaker(state_for(settings, profile), profile, settings, NOW)
    assert "está en su máximo del mes (300 USD)" in b.message
    assumed = state_for(settings, profile, events=())
    b = monthly_drawdown_lock_breaker(assumed, profile, settings, NOW)
    assert "el saldo es supuesto" in b.message


def test_a_withdrawal_during_a_drawdown_does_not_lock(settings, profile):
    """Hallazgo 2: 300 → 150 (50 %) y un retiro de 100 sin operar: sigue en 50 %."""
    events = [
        START_BALANCE,
        {"id": 8, "ts": local(10, 6), "kind": "withdrawal", "amount_usd": 100.0},
    ]
    v = verdict_for(settings, profile, trades=[trade(local(10, 5), -150.0)], events=events)
    b = v.breaker(MONTHLY_DRAWDOWN_LOCK)
    assert not b.triggered and b.current == 50.0
    assert not v.needs_lock and not v.blocked
    assert "misma proporción que el saldo" in b.description
    old = with_breakers(
        settings,
        monthly_drawdown_lock=settings.circuit_breakers.monthly_drawdown_lock.model_copy(
            update={"withdrawal_peak": "subtract"}
        ),
    )
    v_old = verdict_for(old, profile, trades=[trade(local(10, 5), -150.0)], events=events)
    assert v_old.needs_lock  # la regla vieja subía la caída a 75 %
    assert (
        "baja el máximo en lo mismo que retiraste"
        in v_old.breaker(MONTHLY_DRAWDOWN_LOCK).description
    )


def test_depositing_after_a_big_loss_does_not_dodge_the_lock(settings, profile):
    """Hallazgo 3: 300 → 100, depósito de 200 y otra pérdida de 205: 81 % → bloqueo."""
    events = [START_BALANCE, {"id": 9, "ts": local(10, 6), "kind": "deposit", "amount_usd": 200.0}]
    trades = [trade(local(10, 5), -200.0), trade(local(10, 8), -205.0)]
    v = verdict_for(settings, profile, trades=trades, events=events)
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).triggered
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).current == pytest.approx(81.0)
    assert v.needs_lock


def test_a_deposit_raises_the_peak_for_the_drawdown(settings, profile):
    events = [START_BALANCE, {"id": 7, "ts": local(10, 6), "kind": "deposit", "amount_usd": 300.0}]
    trades = [trade(local(10, 8), -400.0)]  # 600 → 200: 66,7 %
    b = monthly_drawdown_lock_breaker(
        state_for(settings, profile, trades=trades, events=events), profile, settings, NOW
    )
    assert not b.triggered
    assert b.current == pytest.approx(200 / 600 * 200)


# --- evento de alta volatilidad ----------------------------------------------------------


def event(start: datetime, until: datetime, note: str = "Dato de inflación de EE.UU.") -> dict:
    return {"id": next(_ids), "starts_at": start, "until": until, "note": note}


def test_active_volatility_event_blocks_until_its_end(settings, profile):
    vol = [event(NOW - timedelta(minutes=30), local(10, 13, 16, 30))]
    state = state_for(settings, profile, vol=vol)
    b = volatility_event_breaker(state, profile, settings, NOW)
    assert b.triggered and b.blocks
    assert b.until == local(10, 13, 16, 30).astimezone(UTC)
    assert b.until_text == "hasta hoy a las 16:30"
    assert "«Dato de inflación de EE.UU.»" in b.message
    assert "13/10/2026 16:30" in b.message
    assert b.current_text == "1 evento activo"
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.block_codes == (VOLATILITY_EVENT,)


def test_volatility_event_switches_off_by_itself_at_the_end(settings, profile):
    vol = [event(NOW - timedelta(hours=2), NOW)]
    state = state_for(settings, profile, vol=vol)
    b = volatility_event_breaker(state, profile, settings, NOW)
    assert not b.triggered and b.progress == 0.0
    assert b.message == "No hay eventos de alta volatilidad activos."


def test_volatility_event_evaluated_after_its_end_with_an_older_state(settings, profile):
    vol = [event(NOW - timedelta(hours=1), NOW + timedelta(minutes=30))]
    state = state_for(settings, profile, vol=vol)
    later = NOW + timedelta(minutes=30)
    assert not volatility_event_breaker(state, profile, settings, later).triggered


def test_upcoming_volatility_event_only_warns(settings, profile):
    vol = [event(local(10, 13, 16), local(10, 13, 18), note="Reunión de la Fed")]
    v = verdict_for(settings, profile, vol=vol)
    assert not v.blocked and v.status == "verde"
    b = v.breaker(VOLATILITY_EVENT)
    assert "Próximo: «Reunión de la Fed»" in b.message
    assert any("Reunión de la Fed" in w and "no se van a generar planes" in w for w in v.warnings)


def test_several_active_events_use_the_latest_end(settings, profile):
    vol = [
        event(NOW - timedelta(hours=1), NOW + timedelta(hours=1), note="Fed"),
        event(NOW - timedelta(minutes=5), NOW + timedelta(hours=3), note="CPI"),
    ]
    b = volatility_event_breaker(state_for(settings, profile, vol=vol), profile, settings, NOW)
    assert b.until == (NOW + timedelta(hours=3)).astimezone(UTC)
    assert b.current_text == "2 eventos activos"
    assert "«Fed»" in b.message and "«CPI»" in b.message


def test_chained_volatility_events_block_until_the_end_of_the_chain(settings, profile):
    """Hallazgo 6: CPI de 9:00 a 12:00 y Fed de 11:30 a 14:00; son las 11:00."""
    now = local(10, 13, 11)
    vol = [
        event(local(10, 13, 9), local(10, 13, 12), note="Dato de CPI"),
        event(local(10, 13, 11, 30), local(10, 13, 14), note="Fed"),
    ]
    v = verdict_for(settings, profile, now=now, vol=vol)
    b = v.breaker(VOLATILITY_EVENT)
    assert b.triggered and b.until == local(10, 13, 14).astimezone(UTC)
    assert b.until_text == "hasta hoy a las 14:00"
    assert "«Dato de CPI»" in b.message and "«Fed»" in b.message
    assert "no se generan planes hasta hoy a las 14:00" in b.message
    assert b.current_text == "1 evento activo"
    assert "y después «Fed»" in b.summary
    assert any("«Fed»" in w for w in v.warnings)  # se avisa aunque haya uno activo
    assert any("Fed" in r for r in v.block_reasons)


def test_back_to_back_volatility_events_are_chained(settings, profile):
    vol = [
        event(NOW - timedelta(hours=1), local(10, 13, 16), note="A"),
        event(local(10, 13, 16), local(10, 13, 17), note="B"),  # empieza justo al terminar A
        event(local(10, 13, 16, 30), local(10, 13, 19), note="C"),  # se pisa con B
        event(local(10, 13, 21), local(10, 13, 22), note="D"),  # más tarde: no se encadena
    ]
    v = verdict_for(settings, profile, vol=vol)
    b = v.breaker(VOLATILITY_EVENT)
    assert b.until == local(10, 13, 19).astimezone(UTC)
    assert "«B»" in b.message and "«C»" in b.message and "«D»" not in b.message
    assert sum("«D»" in w for w in v.warnings) == 1
    assert sum(any(f"«{n}»" in w for w in v.warnings) for n in "BCD") == 3


def test_an_upcoming_event_with_a_gap_is_not_chained(settings, profile):
    vol = [
        event(NOW - timedelta(hours=1), local(10, 13, 16), note="A"),
        event(local(10, 13, 16, 1), local(10, 13, 17), note="B"),
    ]
    b = volatility_event_breaker(state_for(settings, profile, vol=vol), profile, settings, NOW)
    assert b.until == local(10, 13, 16).astimezone(UTC)
    assert "«B»" not in b.message


# --- archivo LOCKED ----------------------------------------------------------------------


def test_lock_file_present_blocks(tmp_path: Path, settings, profile):
    path = tmp_path / "LOCKED"
    write_lock(path, reason="Prueba", triggered_at=NOW, report_path=None, tz=BA_NAME)
    state = state_for(settings, profile)
    v = evaluate_breakers(state, profile, settings, NOW, lock_path=path)
    assert v.blocked and v.locked
    assert v.block_codes == (LOCK_FILE,)
    assert not v.needs_lock
    assert v.breaker(LOCK_FILE).current_text == "existe"
    assert "La app está bloqueada" in v.breaker(LOCK_FILE).message


def test_corrupt_lock_file_still_blocks(tmp_path: Path, settings, profile):
    path = tmp_path / "LOCKED"
    path.write_text("lo edité a mano\n", encoding="utf-8")
    v = evaluate_breakers(state_for(settings, profile), profile, settings, NOW, lock_path=path)
    assert v.blocked and v.lock.exists and not v.lock.readable
    assert "no se pudo leer" in v.breaker(LOCK_FILE).message
    assert any("no se pudo leer" in w for w in v.warnings)


def test_lock_file_absent_and_default_path(tmp_path: Path, settings, profile, monkeypatch):
    seen: list[Path] = []
    monkeypatch.setattr(
        br, "read_lock", lambda p: seen.append(Path(p)) or read_lock(tmp_path / "x")
    )
    v = evaluate_breakers(state_for(settings, profile), profile, settings, NOW)
    assert seen == [settings.resolve_path(settings.app.lock_file)]
    assert not v.locked
    assert lock_file_breaker(NO_LOCK).message == "No hay archivo de bloqueo."


def test_passing_the_lock_info_does_not_read_the_disk(settings, profile, monkeypatch):
    def boom(_path):
        raise AssertionError("no tenía que leer el disco")

    monkeypatch.setattr(br, "read_lock", boom)
    v = evaluate_breakers(state_for(settings, profile), profile, settings, NOW, lock=NO_LOCK)
    assert v.status == "verde"


def test_monthly_trigger_with_an_existing_lock_does_not_need_a_new_one(settings, profile):
    lock = LockInfo(exists=True, path="LOCKED", reason="anterior")
    v = verdict_for(settings, profile, trades=[trade(local(10, 5), -250.0)], lock=lock)
    assert v.breaker(MONTHLY_DRAWDOWN_LOCK).triggered
    assert not v.needs_lock
    assert set(v.block_codes) == {MONTHLY_DRAWDOWN_LOCK, LOCK_FILE}


# --- veredicto --------------------------------------------------------------------------


def test_green_verdict(settings, profile):
    v = verdict_for(settings, profile)
    assert v.status == "verde"
    assert not v.blocked and v.risk_multiplier == 1.0
    assert v.adjustments == () and v.block_reasons == ()
    assert v.headline == "Ningún freno activo: el plan usa el riesgo normal de tu perfil."
    assert tuple(b.code for b in v.breakers) == BREAKER_CODES
    assert all(not b.triggered for b in v.breakers)


def test_amber_verdict_with_one_reduction(settings, profile):
    v = verdict_for(settings, profile, trades=[trade(local(10, 13, 10), -60.0)])
    assert v.status == "ambar"
    assert not v.blocked
    assert v.risk_multiplier == 0.5
    (adj,) = v.adjustments
    assert (adj.code, adj.risk_multiplier) == (DAILY_LOSS_REDUCE, 0.5)
    assert adj.reason.startswith("Riesgo × 0,5: Hoy perdiste 60 USD")
    assert "pasaste el 50 %" in adj.reason
    assert adj.until == local(10, 14).astimezone(UTC)
    assert v.headline == (
        "Riesgo reducido (× 0,5) por la pérdida del día: el plan usa el riesgo achicado."
    )


def test_multipliers_of_the_day_and_the_week_multiply(settings, profile):
    trades = [trade(local(10, 12, 10), -130.0), trade(local(10, 13, 10), -60.0)]
    v = verdict_for(settings, profile, trades=trades)
    assert v.status == "ambar"
    assert v.risk_multiplier == 0.25
    assert [a.code for a in v.adjustments] == [DAILY_LOSS_REDUCE, WEEKLY_LOSS_REDUCE]
    assert "por la pérdida del día y la pérdida de la semana" in v.headline


def test_red_verdict_lists_every_block_and_puts_the_lock_first(settings, profile):
    trades = [trade(local(10, 5), -150.0)] + [
        opened_yesterday(local(10, 13, h), -25.0) for h in (12, 13, 14)
    ]
    v = verdict_for(settings, profile, trades=trades)  # día -75, racha 3, mes -225 (75 %)
    assert v.status == "rojo"
    assert set(v.block_codes) == {LOSING_STREAK, MONTHLY_DRAWDOWN_LOCK}
    assert len(v.block_reasons) == 2
    assert v.headline.startswith("No se generan planes: la billetera del mes cayó 75 %")
    assert "(y 1 freno más)" in v.headline
    assert v.risk_multiplier == 0.5  # el ajuste del día se informa igual


def test_verdict_to_dict_is_json_ready(settings, profile):
    trades = [trade(local(10, 13, 10), -60.0)]
    vol = [event(NOW + timedelta(hours=1), NOW + timedelta(hours=2))]
    v = verdict_for(settings, profile, trades=trades, vol=vol)
    d = v.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["status"] == "ambar"
    assert d["risk_multiplier"] == 0.5
    assert d["lock"]["locked"] is False
    assert [b["code"] for b in d["breakers"]] == list(BREAKER_CODES)
    daily = d["breakers"][0]
    assert {
        "label",
        "description",
        "threshold",
        "current",
        "threshold_text",
        "current_text",
        "progress",
        "triggered",
        "action",
        "risk_multiplier",
        "until",
        "until_text",
        "resets_at",
        "message",
        "summary",
    } <= set(daily)
    assert daily["until"] == "2026-10-14T03:00:00+00:00"
    assert d["evaluated_at"] == "2026-10-13T18:00:00+00:00"


def test_verdict_refuses_a_state_from_another_day(settings, profile):
    state = state_for(settings, profile)
    with pytest.raises(ValueError, match="otro día"):
        evaluate_breakers(state, profile, settings, local(10, 14, 0, 1), lock=NO_LOCK)
    with pytest.raises(ValueError, match="zona horaria"):
        evaluate_breakers(state, profile, settings, datetime(2026, 10, 13, 15), lock=NO_LOCK)


def test_until_phrase_variants():
    assert until_phrase(local(10, 13, 18, 5), NOW, BA_NAME) == "hasta hoy a las 18:05"
    assert until_phrase(local(10, 14), NOW, BA_NAME) == "hasta que termine el día (a las 00:00)"
    assert until_phrase(local(10, 14, 2), NOW, BA_NAME) == "hasta mañana a las 02:00"
    assert until_phrase(local(10, 19), NOW, BA_NAME) == "hasta el lunes 19/10 a las 00:00"


# --- bloqueo total: informe + LOCKED ------------------------------------------------------


def locked_case(settings, profile):
    trades = [
        trade(
            local(10, 5), -120.0, regime_category="bajista", score_total=32.0, plan_status="LONG"
        ),
        trade(local(10, 9), -100.0, symbol="ETH", side="short", risk_usd=50.0),
    ]
    state = state_for(settings, profile, trades=trades)
    verdict = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    return state, verdict


def test_trigger_lock_not_needed_writes_nothing(tmp_path: Path, settings, profile):
    state = state_for(settings, profile)
    verdict = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=tmp_path / "inc"
    )
    assert result.action == "not_needed" and not result.created
    assert list(tmp_path.iterdir()) == []


def test_trigger_lock_writes_the_report_and_the_lock_once(tmp_path: Path, settings, profile):
    state, verdict = locked_case(settings, profile)
    assert verdict.needs_lock
    lock_path = tmp_path / "LOCKED"
    incidents = tmp_path / "inc"
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=lock_path, incident_dir=incidents
    )
    assert result.action == "locked" and result.created
    assert result.report_error is None
    assert result.report_path is not None and result.report_path.parent == incidents
    assert result.report_path.name == "incidente_2026-10-13_180000_UTC.md"

    lock = read_lock(lock_path)
    assert lock.exists and lock.readable
    assert lock.trigger_code == MONTHLY_DRAWDOWN_LOCK
    assert lock.report_path == str(result.report_path)
    assert lock.triggered_at == NOW.astimezone(UTC)
    assert lock.reason == (
        "La billetera del mes cayó 73,33 % desde su máximo (300 USD → 80 USD); el tope es 70 %."
    )

    report = result.report_path.read_text(encoding="utf-8")
    assert "# Informe del incidente" in report
    assert "73,33 %" in report
    assert "Bajista" in report and "+32" in report  # régimen y score de la operación
    assert "Operaste contra el régimen" in report
    assert "Caída máxima de la billetera del mes" in report
    assert str(lock_path) in report

    again = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=lock_path, incident_dir=incidents
    )
    assert again.action == "already_locked" and not again.created
    assert again.lock is not None and again.lock.report_path == str(result.report_path)
    assert len(list(incidents.iterdir())) == 1  # un solo informe

    after = evaluate_breakers(state, profile, settings, NOW, lock_path=lock_path)
    assert after.locked and not after.needs_lock and after.blocked
    d = result.to_dict(tz=BA_NAME)
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def test_trigger_lock_still_locks_if_the_report_fails(tmp_path: Path, settings, profile):
    state, verdict = locked_case(settings, profile)
    not_a_folder = tmp_path / "inc"
    not_a_folder.write_text("soy un archivo", encoding="utf-8")
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=not_a_folder
    )
    assert result.action == "locked"
    assert result.report_path is None
    assert (
        result.report_error is not None and "No se pudo escribir el informe" in result.report_error
    )
    lock = read_lock(tmp_path / "LOCKED")
    assert lock.exists and lock.report_path is None


def test_trigger_lock_still_locks_if_the_report_builder_raises(
    tmp_path: Path, settings, profile, monkeypatch
):
    """Hallazgo 5: un error que no es de disco en el informe no puede impedir el LOCKED."""
    state, verdict = locked_case(settings, profile)

    def broken_report(*_args, **_kwargs):
        raise ValueError("dato raro en el informe")

    monkeypatch.setattr(br, "write_incident_report", broken_report)
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=tmp_path / "i"
    )
    assert result.action == "locked" and result.report_path is None
    assert result.report_error is not None
    assert "ValueError: dato raro en el informe" in result.report_error
    assert "La app quedó bloqueada igual" in result.report_error
    lock = read_lock(tmp_path / "LOCKED")
    assert lock.exists and lock.readable and lock.report_path is None


def test_trigger_lock_still_locks_if_the_reason_text_fails(
    tmp_path: Path, settings, profile, monkeypatch
):
    state, verdict = locked_case(settings, profile)
    monkeypatch.setattr(br, "_lock_reason", lambda *a, **k: 1 / 0)
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=tmp_path / "i"
    )
    assert result.action == "locked"
    assert read_lock(tmp_path / "LOCKED").reason == (
        "La billetera del mes cayó demasiado desde su máximo."
    )


def test_trigger_lock_propagates_only_an_error_writing_the_lock(
    tmp_path: Path, settings, profile, monkeypatch
):
    state, verdict = locked_case(settings, profile)

    def no_disk(*_args, **_kwargs):
        raise OSError("disco lleno")

    monkeypatch.setattr(br, "write_lock", no_disk)
    with pytest.raises(OSError, match="disco lleno"):
        trigger_lock_if_needed(
            verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=tmp_path
        )


def test_trigger_lock_with_assumed_balance_says_so(tmp_path: Path, settings, profile):
    state = state_for(settings, profile, trades=[trade(local(10, 5), -250.0)], events=())
    verdict = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=tmp_path / "LOCKED", incident_dir=tmp_path / "i"
    )
    reason = read_lock(tmp_path / "LOCKED").reason
    assert reason is not None and "El saldo es supuesto" in reason


def test_trigger_lock_rereads_the_month_from_the_repo(tmp_path: Path, settings, profile):
    repo = Repository.open(tmp_path / "db.sqlite", clock=lambda: NOW)
    repo.add_wallet_event(kind="balance", amount_usd=300.0, ts=local(10, 1, 9))
    repo.add_trade(
        symbol="BTC", side="long", pnl_usd=-230.0, opened_at=local(10, 5), closed_at=local(10, 5, 1)
    )
    repo.add_trade(
        symbol="XRP", side="short", pnl_usd=-1.0, opened_at=local(9, 20), closed_at=local(9, 20, 1)
    )  # del mes anterior: no va al informe
    state = state_for(
        settings, profile, trades=repo.list_trades(), events=repo.list_wallet_events()
    )
    verdict = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    repo.add_trade(
        symbol="SOL",
        side="long",
        pnl_usd=-5.0,
        opened_at=local(10, 13, 13),
        closed_at=local(10, 13, 14),
    )  # cargada después de armar el estado
    repo.add_open_position(symbol="AVAX", side="long", risk_usd=10.0, opened_at=local(10, 13, 14))
    result = trigger_lock_if_needed(
        verdict,
        state,
        settings,
        NOW,
        repo=repo,
        lock_path=tmp_path / "LOCKED",
        incident_dir=tmp_path / "inc",
    )
    report = result.report_path.read_text(encoding="utf-8")
    assert "SOL" in report and "AVAX" in report
    assert "XRP" not in report


def test_trigger_lock_does_nothing_if_someone_else_locked_first(tmp_path: Path, settings, profile):
    state, verdict = locked_case(settings, profile)
    lock_path = tmp_path / "LOCKED"
    write_lock(lock_path, reason="El primero", triggered_at=NOW, report_path=None)
    result = trigger_lock_if_needed(
        verdict, state, settings, NOW, lock_path=lock_path, incident_dir=tmp_path / "inc"
    )
    assert result.action == "already_locked"
    assert read_lock(lock_path).reason == "El primero"
    assert not (tmp_path / "inc").exists()


# --- bloqueo anotado en la base y desbloqueo a mano (restart_after_unlock) --------------


class LockFlow:
    """Una base real, el LOCKED y los informes en ``tmp_path``, como en la consola."""

    def __init__(self, tmp_path: Path, settings, profile) -> None:
        self.settings = settings
        self.profile = profile
        self.lock_path = tmp_path / "LOCKED"
        self.incidents = tmp_path / "inc"
        self.repo = Repository.open(tmp_path / "flow.db", clock=lambda: NOW)
        self.repo.add_wallet_event(kind="balance", amount_usd=300.0, ts=local(10, 1, 9))

    def loss(self, when: datetime, pnl: float) -> None:
        self.repo.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=pnl,
            opened_at=when - timedelta(minutes=30),
            closed_at=when,
        )

    def run(self, now: datetime):
        state = load_trading_state(self.repo, self.settings, self.profile, now)
        verdict = evaluate_breakers(
            state, self.profile, self.settings, now, lock_path=self.lock_path
        )
        result = trigger_lock_if_needed(
            verdict,
            state,
            self.settings,
            now,
            repo=self.repo,
            lock_path=self.lock_path,
            incident_dir=self.incidents,
        )
        after = evaluate_breakers(state, self.profile, self.settings, now, lock_path=self.lock_path)
        return state, verdict, result, after

    def reports(self) -> list[str]:
        return sorted(p.name for p in self.incidents.iterdir()) if self.incidents.exists() else []


def test_lock_is_recorded_in_the_database_with_its_numbers(tmp_path: Path, settings, profile):
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    _state, verdict, result, after = flow.run(local(10, 5, 12))
    assert verdict.needs_lock and result.action == "locked"
    (stored,) = flow.repo.list_safety_locks()
    assert result.lock_record_id == stored.id
    assert result.db_error is None
    assert stored.triggered_at == local(10, 5, 12).astimezone(UTC)
    assert stored.trigger_code == MONTHLY_DRAWDOWN_LOCK
    assert (stored.peak_usd, stored.balance_usd) == (300.0, 80.0)
    assert stored.drawdown_pct == pytest.approx(73.333, abs=0.001)
    assert stored.report_path == str(result.report_path)
    assert stored.lock_path == str(flow.lock_path)
    assert after.locked and after.blocked
    d = result.to_dict(tz=BA_NAME)
    assert d["lock_record_id"] == stored.id and d["db_error"] is None


def test_unlock_by_hand_then_small_loss_does_not_relock_then_new_drop_relocks_once(
    tmp_path: Path, settings, profile
):
    """Hallazgo 1 / decisión del usuario: borrar el LOCKED reinicia la medición del mes."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)  # 300 → 80: 73 %
    flow.run(local(10, 5, 12))
    assert flow.lock_path.exists() and len(flow.reports()) == 1

    # Mientras el LOCKED existe: bloqueado y con los números del mes calendario.
    state, verdict, result, _ = flow.run(local(10, 5, 13))
    assert result.action == "not_needed" and verdict.blocked
    assert {MONTHLY_DRAWDOWN_LOCK, LOCK_FILE} <= set(verdict.block_codes)  # y el del día
    assert verdict.breaker(MONTHLY_DRAWDOWN_LOCK).current == pytest.approx(73.333, abs=0.001)

    # El usuario lee el informe y borra el LOCKED: se vuelve a operar.
    flow.lock_path.unlink()
    state, verdict, result, _ = flow.run(local(10, 6, 9))
    assert not verdict.blocked and not verdict.needs_lock and result.action == "not_needed"
    b = verdict.breaker(MONTHLY_DRAWDOWN_LOCK)
    assert b.current == 0.0
    assert "Después del bloqueo del 05/10 12:00 (que ya desbloqueaste)" in b.message
    assert "desde el saldo de ese momento (80 USD)" in b.message
    assert b.summary == "caída desde el bloqueo del 05/10 12:00: 0 % de 70 %"
    assert state.wallet.drawdown_pct == pytest.approx(73.333, abs=0.001)  # el mes sigue igual

    # Una pérdida chica (80 → 60, 25 %) no vuelve a bloquear ni escribe nada.
    flow.loss(local(10, 6, 10), -20.0)
    _, verdict, result, _ = flow.run(local(10, 6, 11))
    assert not verdict.blocked and result.action == "not_needed"
    assert verdict.breaker(MONTHLY_DRAWDOWN_LOCK).current == 25.0
    assert len(flow.reports()) == 1 and not flow.lock_path.exists()

    # Una caída nueva de 70 % desde 80 (80 → 24) vuelve a bloquear, una sola vez.
    flow.loss(local(10, 7, 10), -36.0)
    _, verdict, result, after = flow.run(local(10, 7, 12))
    assert verdict.needs_lock and result.action == "locked"
    b = verdict.breaker(MONTHLY_DRAWDOWN_LOCK)
    assert b.current == 70.0 and b.triggered
    assert "volvió a caer 70 % desde su máximo (80 USD → 24 USD)" in b.message
    assert read_lock(flow.lock_path).reason.startswith(
        "Después del bloqueo del 05/10 12:00 (que se desbloqueó a mano), la billetera volvió a "
        "caer 70 %"
    )
    assert after.blocked and after.locked
    locks = flow.repo.list_safety_locks()
    assert len(locks) == 2 and locks[-1].balance_usd == 24.0 and locks[-1].peak_usd == 80.0
    second_report = result.report_path.read_text(encoding="utf-8")
    assert "Desde el bloqueo del 05/10/2026 12:00 (que se desbloqueó a mano)" in second_report
    assert "máximo de **80 USD** y ahora tiene **24 USD**" in second_report
    _, verdict, result, _ = flow.run(local(10, 7, 13))
    assert result.action == "not_needed" and verdict.blocked
    assert len(flow.reports()) == 2 and len(flow.repo.list_safety_locks()) == 2

    # Y si lo vuelve a borrar, se mide desde el segundo bloqueo (24 USD).
    flow.lock_path.unlink()
    _, verdict, _, _ = flow.run(local(10, 8, 9))
    assert not verdict.blocked
    assert "Después del bloqueo del 07/10 12:00" in verdict.breaker(MONTHLY_DRAWDOWN_LOCK).message


def test_restart_after_unlock_with_exactly_70_percent_from_the_lock_balance(settings, profile):
    trades = [trade(local(10, 5, 10), -200.0), trade(local(10, 6, 10), -70.0)]  # 100 → 30
    locks = [{"id": 1, "triggered_at": local(10, 5, 12), "trigger_code": MONTHLY_DRAWDOWN_LOCK}]
    state = build_trading_state(
        trades=trades,
        open_positions=[],
        wallet_events=[START_BALANCE],
        volatility_events=[],
        settings=settings,
        profile=profile,
        now=NOW,
        safety_locks=locks,
    )
    b = monthly_drawdown_lock_breaker(state, profile, settings, NOW)
    assert b.triggered and b.current == 70.0
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    assert v.needs_lock
    # Con el LOCKED puesto se miran los números del mes (90 %) y no hace falta otro.
    locked = evaluate_breakers(state, profile, settings, NOW, lock=LockInfo(True, "LOCKED"))
    assert locked.breaker(MONTHLY_DRAWDOWN_LOCK).current == 90.0
    assert not locked.needs_lock


def test_a_lock_from_a_previous_month_is_ignored(tmp_path: Path, settings, profile):
    flow = LockFlow(tmp_path, settings, profile)
    flow.repo.add_safety_lock(triggered_at=local(9, 25, 12))  # septiembre: no cuenta
    flow.loss(local(10, 5, 10), -220.0)
    _, verdict, result, _ = flow.run(local(10, 5, 12))
    assert verdict.needs_lock and result.action == "locked"  # se mide el mes calendario


def test_without_restart_after_unlock_deleting_the_lock_relocks(tmp_path: Path, settings, profile):
    off = with_breakers(
        settings,
        monthly_drawdown_lock=settings.circuit_breakers.monthly_drawdown_lock.model_copy(
            update={"restart_after_unlock": False}
        ),
    )
    flow = LockFlow(tmp_path, off, profile)
    flow.loss(local(10, 5, 10), -220.0)
    flow.run(local(10, 5, 12))
    flow.lock_path.unlink()
    _, verdict, result, _ = flow.run(local(10, 6, 9))
    assert verdict.needs_lock and result.action == "locked"  # comportamiento anterior
    assert "la app se vuelve a bloquear" in verdict.breaker(MONTHLY_DRAWDOWN_LOCK).description


def test_a_database_error_recording_the_lock_still_leaves_the_lock(
    tmp_path: Path, settings, profile, monkeypatch
):
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)

    def broken(**_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(flow.repo, "add_safety_lock", broken)
    _, _, result, after = flow.run(local(10, 5, 12))
    assert result.action == "locked" and flow.lock_path.exists()
    assert result.lock_record_id is None
    assert result.db_error is not None
    assert "No se pudo anotar el bloqueo en la base (RuntimeError: database is locked)" in (
        result.db_error
    )
    assert after.blocked


def test_a_database_error_rereading_the_month_uses_the_state(
    tmp_path: Path, settings, profile, monkeypatch
):
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    state = load_trading_state(flow.repo, settings, profile, local(10, 5, 12))
    verdict = evaluate_breakers(state, profile, settings, local(10, 5, 12), lock=NO_LOCK)

    def broken():
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(flow.repo, "list_trades", broken)
    result = trigger_lock_if_needed(
        verdict,
        state,
        settings,
        local(10, 5, 12),
        repo=flow.repo,
        lock_path=flow.lock_path,
        incident_dir=flow.incidents,
    )
    assert result.action == "locked" and result.report_path is not None
    assert "No se pudo volver a leer la base para el informe" in (result.db_error or "")
    assert result.lock_record_id is not None  # anotarlo sí funcionó


# --- integración del Hito 5: análisis de prueba, avisos de eventos y caída recortada ------


def test_trigger_lock_marks_trades_linked_to_test_analyses(tmp_path: Path, settings, profile):
    """Hallazgo 18: el informe marca las operaciones ligadas a un análisis de prueba."""
    repo = Repository.open(tmp_path / "db.sqlite", clock=lambda: NOW)
    repo.add_wallet_event(kind="balance", amount_usd=300.0, ts=local(10, 1, 9))
    fake = repo.save_analysis(
        symbol="BTC",
        profile_name="agresivo",
        price=60000.0,
        price_source="sintetico",
        plan_status="NO_OPERAR",
        regime_category="bajista",
        regime_name="Bajista",
        regime_confidence=0.99,
        inputs={"offline": True, "day_overrides": []},
        created_at=local(10, 5, 9),
    )
    repo.add_trade(
        symbol="BTC",
        side="long",
        pnl_usd=-230.0,
        opened_at=local(10, 5, 10),
        closed_at=local(10, 5, 11),
        analysis_id=fake.id,
    )
    state = state_for(
        settings, profile, trades=repo.list_trades(), events=repo.list_wallet_events()
    )
    verdict = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    result = trigger_lock_if_needed(
        verdict,
        state,
        settings,
        NOW,
        repo=repo,
        lock_path=tmp_path / "LOCKED",
        incident_dir=tmp_path / "inc",
    )
    assert result.created and result.db_error is None
    text = result.report_path.read_text(encoding="utf-8")
    assert "| Bajista (99 %) (datos sintéticos) |" in text
    assert 'el plan decía "No operar"' not in text  # un plan con datos inventados no cuenta


def test_upcoming_event_warning_is_the_same_text_in_the_verdict(settings, profile):
    from app.safety.breakers import upcoming_event_warning
    from app.safety.state import VolatilityEventRecord

    ev = event(local(10, 13, 18), local(10, 13, 20), note="Fed")
    v = verdict_for(settings, profile, vol=[ev])
    record = VolatilityEventRecord.from_value(ev)
    assert upcoming_event_warning(record, BA_NAME) in v.warnings
    assert upcoming_event_warning(record, BA_NAME) == (
        "Hay un evento de alta volatilidad marcado para el 13/10/2026 18:00 («Fed»): desde esa "
        "hora y hasta el 13/10/2026 20:00 no se van a generar planes."
    )


def test_monthly_drawdown_above_100_percent_says_so_and_asks_to_check_the_data(settings, profile):
    """Fix 7 / E5: con el saldo calculado negativo, "más del 100 %" y revisar los datos."""
    state = state_for(settings, profile, trades=[trade(local(10, 5), -700.0)])
    v = evaluate_breakers(state, profile, settings, NOW, lock=NO_LOCK)
    monthly = v.breaker(MONTHLY_DRAWDOWN_LOCK)
    assert monthly.current == pytest.approx(700 / 3)  # 300 → -400: 233 %, sin recortar
    assert monthly.current_text == "más del 100 %"
    assert "cayó más del 100 % desde su máximo del mes (300 USD → -400 USD)" in monthly.message
    assert "Ojo: el saldo calculado de tu billetera da negativo (-400 USD)" in monthly.message
    assert "un resultado o un retiro mal tipeado" in monthly.message
    assert monthly.summary == "la billetera del mes cayó más del 100 % desde su máximo (tope 70 %)"
    assert v.headline.startswith("No se generan planes: la billetera del mes cayó más del 100 %")
    assert "233" not in monthly.message + monthly.summary + v.headline
    reason = br._lock_reason(state, settings, br._month_measure(state, settings, locked=False))
    assert "cayó más del 100 % desde su máximo" in reason and "233" not in reason
    assert "mal tipeado" in reason
    d = v.to_dict()["drawdown_measure"]
    assert d["negative_balance"] is True and d["drawdown_text"] == "más del 100 %"
    assert "mal tipeado" in d["negative_balance_warning"]


# --- Fix 7: tercera ronda y revisión 6 --------------------------------------------------


def opened_today(hour: int, pnl: float = 5.0) -> dict:
    return trade(local(10, 13, hour), pnl)


def test_max_trades_of_the_day_blocks_until_midnight(settings, profile):
    """Tercera ronda (E4): con el máximo de operaciones del día, el estado queda en ROJO."""
    v = verdict_for(settings, profile, trades=[opened_today(h) for h in (9, 10, 11)])
    b = v.breaker(br.MAX_TRADES_TODAY)
    assert b.triggered and b.blocks and b.kind == "block"
    assert (b.threshold, b.current, b.progress) == (3.0, 3.0, 1.0)
    assert (b.current_text, b.threshold_text) == ("3", "3 operaciones")  # "3 de 3 operaciones"
    assert b.until == local(10, 14).astimezone(UTC)
    assert b.message == (
        "Ya hiciste 3 de 3 operaciones permitidas hoy: no se generan planes hasta que termine "
        "el día (a las 00:00). Mañana arrancás de cero."
    )
    assert b.summary == "ya hiciste 3 de 3 operaciones permitidas hoy"
    assert v.status == "rojo" and v.blocked
    assert v.block_codes == (br.MAX_TRADES_TODAY,)
    assert v.headline == (
        "No se generan planes: ya hiciste 3 de 3 operaciones permitidas hoy, hasta que "
        "termine el día (a las 00:00)."
    )
    assert v.to_dict()["breakers"][2]["code"] == br.MAX_TRADES_TODAY


def test_max_trades_below_the_limit_and_switched_off(settings, profile):
    two = verdict_for(settings, profile, trades=[opened_today(h) for h in (9, 10)])
    b = two.breaker(br.MAX_TRADES_TODAY)
    assert not b.triggered and two.status == "verde"
    assert b.message == (
        "Hoy hiciste 2 de 3 operaciones permitidas. Con una más llegás al máximo y no hay más "
        "planes hasta mañana."
    )
    off = with_breakers(settings, max_trades_blocks_status=False)
    v = verdict_for(off, profile, trades=[opened_today(h) for h in (9, 10, 11, 12)])
    b = v.breaker(br.MAX_TRADES_TODAY)
    assert not b.triggered and v.status == "verde"
    assert b.message.startswith("Ya hiciste 4 operaciones hoy y el máximo es 3.")
    zero = dataclasses.replace(get_profile("agresivo", settings), max_trades_per_day=0)
    b = verdict_for(settings, zero).breaker(br.MAX_TRADES_TODAY)
    assert b.triggered and b.message.startswith("Tu perfil no permite operaciones")


def test_the_lock_file_does_not_count_as_one_more_breaker(settings, profile):
    """Revisión 6: el LOCKED es el mismo bloqueo que la caída del mes, no "1 freno más"."""
    lock = LockInfo(exists=True, path="LOCKED", reason="anterior")
    v = verdict_for(settings, profile, trades=[trade(local(10, 5), -250.0)], lock=lock)
    assert set(v.block_codes) == {MONTHLY_DRAWDOWN_LOCK, LOCK_FILE}
    assert "freno más" not in v.headline
    assert v.headline.startswith("No se generan planes: la billetera del mes cayó 83,33 %")


def test_wallet_rules_text_explains_the_refill(settings, profile):
    b = verdict_for(settings, profile).breaker(MONTHLY_DRAWDOWN_LOCK)
    assert (
        "Un depósito primero devuelve lo que retiraste en el mes, sin cambiar el % de caída"
    ) in b.description
    off = with_breakers(
        settings,
        monthly_drawdown_lock=settings.circuit_breakers.monthly_drawdown_lock.model_copy(
            update={"deposit_refills_withdrawals": False}
        ),
    )
    assert "devuelve" not in verdict_for(off, profile).breaker(MONTHLY_DRAWDOWN_LOCK).description


def second_lock_flow(tmp_path: Path, settings, profile) -> LockFlow:
    """300 → 80 (bloqueo 05/10 12:00), desbloqueo, 80 → 24 (segundo bloqueo 07/10 12:00)."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    flow.run(local(10, 5, 12))
    flow.lock_path.unlink()
    flow.loss(local(10, 7, 10), -56.0)
    _, verdict, result, _ = flow.run(local(10, 7, 12))
    assert result.action == "locked"
    return flow


def test_while_the_second_lock_exists_the_breaker_shows_the_drop_that_fired_it(
    tmp_path: Path, settings, profile
):
    """Revisión 6: con el LOCKED del segundo bloqueo, la caída es la medida desde el desbloqueo."""
    flow = second_lock_flow(tmp_path, settings, profile)
    state, verdict, result, _ = flow.run(local(10, 7, 13))
    assert result.action == "not_needed" and verdict.locked
    b = verdict.breaker(MONTHLY_DRAWDOWN_LOCK)
    assert b.current == 70.0 and b.triggered  # no el 92 % del mes calendario
    assert "Después del bloqueo del 05/10 12:00 (que desbloqueaste)" in b.message
    assert "volvió a caer 70 % desde su máximo (80 USD → 24 USD)" in b.message
    assert (
        "En todo el mes la caída es 92 % (300 USD → 24 USD), pero esa cifra ya no cuenta para "
        "el bloqueo."
    ) in b.message
    assert verdict.headline.startswith(
        "No se generan planes: después del bloqueo del 05/10 12:00, la billetera volvió a caer "
        "70 % (tope 70 %)"
    )
    assert "freno más" not in verdict.headline
    m = verdict.drawdown_measure
    assert m is not None and m.since_unlock and m.while_locked
    assert (m.peak_usd, m.balance_usd, m.month_peak_usd) == (80.0, 24.0, 300.0)
    assert m.month_drawdown_pct == pytest.approx(92.0)
    d = verdict.to_dict()["drawdown_measure"]
    assert d["basis"] == "since_unlock" and d["measured_since_text"] == "05/10 12:00"
    assert d["drawdown_text"] == "70 %" and d["month"]["drawdown_text"] == "92 %"
    lock = read_lock(flow.lock_path)
    assert lock.reason is not None and "volvió a caer 70 %" in lock.reason
    assert lock.measured_since == local(10, 5, 12).astimezone(UTC)
    assert (lock.peak_usd, lock.balance_usd) == (80.0, 24.0)
    assert verdict.lock_recorded is True
    # El primer bloqueo (sin uno anterior en el mes) sigue mostrando el mes calendario.
    assert state.wallet.drawdown_pct == pytest.approx(92.0)


def test_a_lock_that_could_not_be_recorded_is_backfilled_and_unlock_does_not_relock(
    tmp_path: Path, settings, profile, monkeypatch
):
    """Tercera ronda (E2): el LOCKED sin anotar se anota solo; borrarlo no vuelve a bloquear."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    real_add = flow.repo.add_safety_lock

    def broken(**_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(flow.repo, "add_safety_lock", broken)
    _, _, result, after = flow.run(local(10, 5, 12))
    assert result.action == "locked" and result.db_error is not None
    assert flow.repo.list_safety_locks() == []
    # La base sigue fallando: el aviso se repite en cada corrida.
    _, verdict, result, _ = flow.run(local(10, 5, 13))
    assert result.backfill is not None and result.backfill.action == "failed"
    assert "database is locked" in (result.backfill.db_error or "")
    assert verdict.lock_recorded is False
    assert any("no está anotado en la base" in w for w in verdict.warnings)
    # La base vuelve a andar: la próxima corrida lo anota con la fecha y los números del LOCKED.
    monkeypatch.setattr(flow.repo, "add_safety_lock", real_add)
    _, verdict, result, _ = flow.run(local(10, 5, 14))
    assert result.backfill is not None and result.backfill.action == "recorded"
    assert "se anotó ahora con su fecha" in result.backfill.message
    (stored,) = flow.repo.list_safety_locks()
    assert stored.triggered_at == local(10, 5, 12).astimezone(UTC)
    assert (stored.peak_usd, stored.balance_usd) == (300.0, 80.0)
    assert stored.drawdown_pct == pytest.approx(73.333, abs=0.001)
    assert stored.lock_path == str(flow.lock_path)
    # Ya anotado: no se duplica y no hay aviso.
    state, verdict, result, _ = flow.run(local(10, 5, 15))
    assert result.backfill is not None and result.backfill.action == "already_recorded"
    assert len(flow.repo.list_safety_locks()) == 1
    assert not any("no está anotado" in w for w in verdict.warnings)
    # El usuario borra el LOCKED al día siguiente: no se vuelve a bloquear ni hay otro informe.
    flow.lock_path.unlink()
    _, verdict, result, _ = flow.run(local(10, 6, 13))
    assert not verdict.blocked and result.action == "not_needed"
    assert len(flow.reports()) == 1
    assert "Después del bloqueo del 05/10 12:00" in verdict.breaker(MONTHLY_DRAWDOWN_LOCK).message


def test_an_old_lock_without_numbers_is_backfilled_too(tmp_path: Path, settings, profile):
    """Un LOCKED de la versión anterior (sin números en el encabezado) se anota con su fecha."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    write_lock(
        flow.lock_path,
        reason="La billetera del mes cayó 73,33 %",
        triggered_at=local(10, 5, 12),
        report_path=None,
        trigger_code=MONTHLY_DRAWDOWN_LOCK,
    )
    run = br.run_safety_checks(
        flow.repo,
        settings,
        profile,
        local(10, 5, 13),
        lock_path=flow.lock_path,
        incident_dir=flow.incidents,
    )
    assert run.backfill is not None and run.backfill.recorded
    (stored,) = flow.repo.list_safety_locks()
    assert stored.peak_usd is None and stored.balance_usd is None
    assert run.verdict.lock_recorded is True and run.lock_result is None
    assert [lk.id for lk in run.state.month_locks] == [stored.id]
    d = run.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    flow.lock_path.unlink()
    after = br.run_safety_checks(
        flow.repo, settings, profile, local(10, 6, 9), lock_path=flow.lock_path
    )
    assert not after.verdict.blocked  # se mide desde el bloqueo anotado


def test_backfill_ignores_locks_of_other_months_other_breakers_or_when_disabled(
    tmp_path: Path, settings, profile
):
    repo = Repository.open(tmp_path / "db.sqlite", clock=lambda: NOW)
    path = tmp_path / "LOCKED"
    assert br.backfill_unrecorded_lock(repo, settings, NOW, lock_path=path).action == (
        "not_applicable"
    )
    write_lock(
        path,
        reason="x",
        triggered_at=local(9, 25),
        report_path=None,
        trigger_code=MONTHLY_DRAWDOWN_LOCK,
    )
    assert br.backfill_unrecorded_lock(repo, settings, NOW, lock_path=path).action == (
        "not_applicable"
    )  # septiembre
    path.unlink()
    write_lock(path, reason="x", triggered_at=local(10, 5), report_path=None, trigger_code="otro")
    assert br.backfill_unrecorded_lock(repo, settings, NOW, lock_path=path).action == (
        "not_applicable"
    )
    off = with_breakers(
        settings,
        monthly_drawdown_lock=settings.circuit_breakers.monthly_drawdown_lock.model_copy(
            update={"backfill_unrecorded_lock": False}
        ),
    )
    assert br.backfill_unrecorded_lock(repo, off, NOW, lock_path=path).action == "disabled"
    assert repo.list_safety_locks() == []


def test_a_lock_write_error_is_clear_and_does_not_pile_up_reports(
    tmp_path: Path, settings, profile, monkeypatch
):
    """Revisión 6: sin poder escribir el LOCKED, error claro y un solo informe."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    real_write = br.write_lock

    def no_permission(path, **_kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(br, "write_lock", no_permission)
    for minute in (0, 1, 2):
        with pytest.raises(br.LockWriteError) as info:
            br.run_safety_checks(
                flow.repo,
                settings,
                profile,
                local(10, 5, 12, minute),
                lock_path=flow.lock_path,
                incident_dir=flow.incidents,
            )
        assert isinstance(info.value, OSError)
        assert str(info.value).startswith(
            f"No se pudo escribir el archivo LOCKED ({flow.lock_path})"
        )
        assert "Permission denied" in str(info.value)
        assert info.value.report_path is not None
    assert len(flow.reports()) == 1  # el mismo informe en los tres intentos
    assert not flow.lock_path.exists()
    monkeypatch.setattr(br, "write_lock", real_write)
    run = br.run_safety_checks(
        flow.repo,
        settings,
        profile,
        local(10, 5, 12, 3),
        lock_path=flow.lock_path,
        incident_dir=flow.incidents,
    )
    assert run.lock_result is not None and run.lock_result.action == "locked"
    assert run.lock_result.report_reused is True
    assert flow.reports() == [Path(run.lock_result.report_path).name]
    assert read_lock(flow.lock_path).report_path == str(run.lock_result.report_path)
    assert run.lock_result.to_dict()["report_reused"] is True
    assert run.verdict.locked and run.verdict.lock_recorded is None  # recién escrito


def test_a_new_loss_makes_a_new_report_after_a_failed_lock(
    tmp_path: Path, settings, profile, monkeypatch
):
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    monkeypatch.setattr(br, "write_lock", lambda path, **_k: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(br.LockWriteError):
        br.run_safety_checks(
            flow.repo,
            settings,
            profile,
            local(10, 5, 12),
            lock_path=flow.lock_path,
            incident_dir=flow.incidents,
        )
    flow.loss(local(10, 5, 12, 30), -5.0)  # datos nuevos: el informe cambia
    with pytest.raises(br.LockWriteError):
        br.run_safety_checks(
            flow.repo,
            settings,
            profile,
            local(10, 5, 13),
            lock_path=flow.lock_path,
            incident_dir=flow.incidents,
        )
    assert len(flow.reports()) == 2


def test_a_late_loss_before_the_lock_does_not_relock_after_unlocking(
    tmp_path: Path, settings, profile
):
    """Tercera ronda (E3): una pérdida cargada tarde que cerró antes del bloqueo es parte de él."""
    flow = LockFlow(tmp_path, settings, profile)
    flow.loss(local(10, 5, 10), -220.0)
    flow.run(local(10, 5, 12))
    flow.loss(local(10, 5, 11), -60.0)  # cargada después, cerró antes del bloqueo (80 → 20)
    flow.lock_path.unlink()
    state, verdict, result, _ = flow.run(local(10, 6, 9))
    assert not verdict.blocked and result.action == "not_needed"
    r = state.wallet.restart
    assert r is not None and r.start_balance_usd == 20.0 and r.drawdown_pct == 0.0
    assert len(flow.reports()) == 1
