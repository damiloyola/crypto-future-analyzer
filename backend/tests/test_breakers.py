"""Hito 5: circuit breakers y veredicto de seguridad (``app.safety.breakers``).

Perfil agresivo de ``config.yaml``: límite diario 100 USD (riesgo × 0,5 desde
50 USD, bloqueo desde 100), billetera mensual 300 USD (riesgo × 0,5 con 180 USD
perdidos en la semana), 3 pérdidas seguidas → pausa de 4 h y bloqueo total con
70 % de caída desde el máximo del mes. "Ahora" es el martes 13/10/2026 15:00 de
Buenos Aires.
"""

from __future__ import annotations

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
from app.safety.state import build_trading_state
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
    trades = [trade(local(10, 13, 9), -60.0), trade(local(10, 13, 11), 20.0)]
    net = daily_loss_reduce_breaker(
        state_for(settings, profile, trades=trades), profile, settings, NOW
    )
    assert (net.current, net.triggered) == (40.0, False)
    gross = with_breakers(settings, loss_basis="gross")
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


# --- pérdidas seguidas ------------------------------------------------------------------


def test_two_losses_in_a_row_warn_about_the_next_one(settings, profile):
    trades = [trade(local(10, 13, 9), -5.0), trade(local(10, 13, 10), -5.0)]
    b = losing_streak_breaker(state_for(settings, profile, trades=trades), profile, settings, NOW)
    assert not b.triggered
    assert (b.current, b.threshold) == (2.0, 3.0)
    assert b.progress == pytest.approx(2 / 3)
    assert b.current_text == "2 pérdidas seguidas"
    assert "si la próxima también pierde" in b.message


def test_third_loss_blocks_for_four_hours(settings, profile):
    trades = [trade(local(10, 13, h), -5.0) for h in (12, 13, 14)]
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
    trades = [trade(local(10, 5), -150.0)] + [trade(local(10, 13, h), -25.0) for h in (12, 13, 14)]
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
