"""Servicio del régimen de punta a punta: modelo en disco, reentrenamiento, semántica y JSON.

Sin red: el mercado es ``SyntheticMarketData`` (o ``FakeMarketData`` para hacer
fallar consultas). Las configs del régimen son chicas (un número de estados, una
inicialización, pocas iteraciones, 60 días de 1h) para que cada entrenamiento
tarde menos de un segundo.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from app.config import AppSettings
from app.data.synthetic import SyntheticMarketData
from app.exchanges.base import Timeframe
from app.indicators.features import feature_matrix
from app.regime import service
from app.regime.service import (
    CATEGORY_NAMES,
    UNCONFIRMED_NAME,
    ModelStatus,
    RegimeReport,
    RegimeUnavailable,
    build_report,
    classify_bars,
    ensure_model,
    get_market_regimes,
    get_regime,
    regime_history_days,
    regimes_to_dict,
)
from app.regime.store import load_model
from app.regime.types import StateInfo
from tests.fakes import BINANCE_451, FIXED_NOW, FakeMarketData


def small_settings(settings: AppSettings, root: Path, **regime_changes) -> AppSettings:
    """Config del régimen chica y rápida, con las carpetas de modelos en ``root``."""
    regime = settings.regime.model_copy(
        update={
            "n_states_min": 4,
            "n_states_max": 4,
            "n_restarts": 1,
            "max_iter": 25,
            "training_days": 60,
            "context_training_days": 400,
            "min_training_samples": 100,
            "models_dir": str(root / "reales"),
            "synthetic_models_dir": str(root / "sinteticos"),
            **regime_changes,
        }
    )
    return settings.model_copy(update={"regime": regime})


@pytest.fixture(scope="module")
def market() -> SyntheticMarketData:
    return SyntheticMarketData(now=FIXED_NOW)


def fixed_clock(when=FIXED_NOW):
    return lambda: when


def regime(symbol, timeframe, settings, md, **kwargs) -> RegimeReport:
    kwargs.setdefault("clock", fixed_clock())
    return asyncio.run(get_regime(symbol, timeframe, settings, md, **kwargs))


def history_calls(md: FakeMarketData) -> int:
    return md.methods_called().count("get_klines_history")


# --- modelo en disco y reentrenamiento ------------------------------------------------


def test_train_then_reuse_saved_model(settings, market, tmp_path):
    s = small_settings(settings, tmp_path)
    md = FakeMarketData(synthetic=market)
    notes: list[str] = []
    first = regime("BTC", "1h", s, md, notify=notes.append)
    path = tmp_path / "reales" / "BTC_1h.joblib"
    assert path.is_file()
    assert first.model.retrained_now and first.model.retrain_reason == "no hay modelo guardado"
    assert first.model.path == str(path)
    assert first.model.train_seconds is not None and first.model.train_seconds > 0
    assert len(notes) == 1 and notes[0].startswith("Entrenando el modelo de régimen de BTC 1h")
    assert history_calls(md) == 1  # las velas del entrenamiento se reusan para el cálculo

    second = regime("BTC", "1h", s, md, notify=notes.append)
    assert not second.model.retrained_now
    assert "al día" in second.model.retrain_reason
    assert second.model.trained_at == first.model.trained_at
    assert len(notes) == 1  # no avisó de ningún entrenamiento
    assert history_calls(md) == 2  # solo las velas para calcular
    assert (second.name, second.confidence) == (first.name, first.confidence)
    assert [b.state_prob for b in second.history] == [b.state_prob for b in first.history]


def test_stale_model_retrains_only_with_auto_retrain(settings, market, tmp_path):
    s = small_settings(settings, tmp_path, auto_retrain=False)
    md = FakeMarketData(synthetic=market)
    regime("BTC", "1h", s, md)
    path = tmp_path / "reales" / "BTC_1h.joblib"
    before = path.read_bytes()
    later = fixed_clock(FIXED_NOW + timedelta(days=8))

    kept = regime("BTC", "1h", s, md, clock=later)
    assert not kept.model.retrained_now and kept.model.stale
    stale = [w for w in kept.warnings if w.code == "stale_model"]
    assert len(stale) == 1
    assert "8 días" in stale[0].text and "auto_retrain" in stale[0].text
    assert path.read_bytes() == before

    s_auto = small_settings(settings, tmp_path, auto_retrain=True)
    not_allowed = regime("BTC", "1h", s_auto, md, clock=later, allow_train=False)
    assert not not_allowed.model.retrained_now
    assert any(w.code == "stale_model" for w in not_allowed.warnings)
    assert path.read_bytes() == before

    retrained = regime("BTC", "1h", s_auto, md, clock=later)
    assert retrained.model.retrained_now and not retrained.model.stale
    assert "8 días" in retrained.model.retrain_reason
    assert load_model(path).trained_at == FIXED_NOW + timedelta(days=8)


def test_force_retrain_and_allow_train(settings, market, tmp_path):
    s = small_settings(settings, tmp_path)
    md = FakeMarketData(synthetic=market)
    with pytest.raises(RegimeUnavailable, match="make train-regime SYMBOL=BTC") as info:
        regime("BTC", "1h", s, md, allow_train=False)
    assert "no hay modelo guardado" in info.value.reason
    assert history_calls(md) == 0  # ni siquiera bajó velas

    regime("BTC", "1h", s, md)
    forced = regime("BTC", "1h", s, md, force_retrain=True)
    assert forced.model.retrained_now and forced.model.retrain_reason == "se pidió reentrenar"


def test_config_change_forces_retrain(settings, market, tmp_path):
    md = FakeMarketData(synthetic=market)
    regime("BTC", "1h", small_settings(settings, tmp_path), md)
    other = regime("BTC", "1h", small_settings(settings, tmp_path, max_iter=26), md)
    assert other.model.retrained_now
    assert other.model.retrain_reason == "cambió la configuración del régimen"


def test_synthetic_and_real_models_never_mix(settings, market, tmp_path):
    s = small_settings(settings, tmp_path)
    md = FakeMarketData(synthetic=market)
    real_path = tmp_path / "reales" / "ETH_1h.joblib"
    synth_path = tmp_path / "sinteticos" / "ETH_1h.joblib"

    fake = regime("ETH", "1h", s, md, synthetic=True)
    assert fake.model.synthetic and fake.model.path == str(synth_path)
    assert synth_path.is_file() and not real_path.exists()
    synth_bytes = synth_path.read_bytes()

    real = regime("ETH", "1h", s, md, synthetic=False)
    assert real.model.retrained_now and not real.model.synthetic
    assert real_path.is_file()
    assert synth_path.read_bytes() == synth_bytes  # el sintético quedó intacto
    assert load_model(real_path).synthetic is False
    assert load_model(synth_path).synthetic is True

    again = regime("ETH", "1h", s, md, synthetic=True)
    assert not again.model.retrained_now and again.model.synthetic


def test_same_folder_for_synthetic_and_real_is_refused(settings, market, tmp_path):
    s = small_settings(
        settings,
        tmp_path,
        models_dir=str(tmp_path / "modelos"),
        synthetic_models_dir=str(tmp_path / "modelos"),
    )
    with pytest.raises(RegimeUnavailable, match="carpeta distinta"):
        regime("BTC", "1h", s, FakeMarketData(synthetic=market), synthetic=True)
    assert not (tmp_path / "modelos").exists()


def test_model_of_the_other_kind_is_never_overwritten(settings, market, tmp_path):
    s = small_settings(settings, tmp_path)
    md = FakeMarketData(synthetic=market)
    shared = tmp_path / "compartida"
    regime("BTC", "1d", s, md, models_dir=shared, synthetic=False)
    path = shared / "BTC_1d.joblib"
    before = path.read_bytes()
    with pytest.raises(RegimeUnavailable, match="carpetas distintas"):
        regime("BTC", "1d", s, md, models_dir=shared, synthetic=True)
    assert path.read_bytes() == before


def test_training_runs_in_a_worker_thread(settings, market, tmp_path, monkeypatch):
    threads: list[threading.Thread] = []
    real_train = service.train_regime_model

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real_train(*args, **kwargs)

    monkeypatch.setattr(service, "train_regime_model", spy)
    regime("BTC", "1h", small_settings(settings, tmp_path), FakeMarketData(synthetic=market))
    assert len(threads) == 1 and threads[0] is not threading.main_thread()


def test_never_uses_lookahead_methods(settings, market, tmp_path, monkeypatch):
    def forbidden(self, *args, **kwargs):
        raise AssertionError("método con look-ahead")

    for name in ("predict", "predict_proba", "decode", "score_samples"):
        monkeypatch.setattr(GaussianHMM, name, forbidden)
    report = regime("SOL", "1h", small_settings(settings, tmp_path), market)
    assert report.bars_analyzed > 0


def test_service_regime_of_bar_t_never_changes_with_later_bars(settings, market, tmp_path):
    """Cero look-ahead de punta a punta (el test obligatorio, a nivel servicio).

    Con el mismo modelo guardado, el historial calculado "a las 14 h" y el
    calculado horas o días después tienen que coincidir en todas las velas que
    comparten: el régimen de la vela *t* no cambia al agregar velas posteriores.
    Esto agarra look-ahead metido en cualquier paso del servicio (por ejemplo
    estandarizar con la media de toda la ventana, o suavizar hacia atrás las
    probabilidades), no solo en ``forward``.
    """
    s = small_settings(settings, tmp_path)
    for timeframe, hours_later in (("1h", (1, 5, 24, 48)), ("1d", (24, 72))):
        late = regime("BTC", timeframe, s, market, synthetic=True)
        late_by_time = {b.time: b for b in late.history}
        for hours in hours_later:
            early = regime(
                "BTC",
                timeframe,
                s,
                market,
                synthetic=True,
                allow_train=False,
                clock=fixed_clock(FIXED_NOW - timedelta(hours=hours)),
            )
            assert not early.model.retrained_now
            assert early.model.trained_at == late.model.trained_at  # el mismo modelo
            common = [b for b in early.history if b.time in late_by_time]
            bars_later = hours * 60 // late.timeframe_minutes
            assert len(common) == len(early.history) - bars_later > 0
            for b in common:
                other = late_by_time[b.time]
                assert (b.state, b.state_category, b.regime, b.category, b.unstable) == (
                    other.state,
                    other.state_category,
                    other.regime,
                    other.category,
                    other.unstable,
                ), b.time
                assert b.state_prob == pytest.approx(other.state_prob, abs=1e-9)
                assert b.confidence == pytest.approx(other.confidence, abs=1e-9)
                if b.regime_prob is None:
                    assert other.regime_prob is None
                else:
                    assert b.regime_prob == pytest.approx(other.regime_prob, abs=1e-9)


@pytest.mark.parametrize("symbol", ["../../outside/evil", "BTC/USDT", "", "..", "BTC USDT"])
def test_invalid_symbol_is_refused_before_touching_the_disk(
    settings, market, tmp_path, monkeypatch, symbol
):
    # Regresión: ensure_model armaba la ruta con el símbolo tal cual y cargaba con
    # joblib (pickle) un archivo fuera de la carpeta de modelos antes de validar.
    loads: list[object] = []
    monkeypatch.setattr(service, "load_model", lambda path: loads.append(path))
    s = small_settings(settings, tmp_path)
    (tmp_path / "reales").mkdir()
    with pytest.raises(RegimeUnavailable, match="símbolo inválido"):
        asyncio.run(ensure_model(symbol, "1h", s, market, clock=fixed_clock()))
    with pytest.raises(RegimeUnavailable, match="símbolo inválido"):
        regime(symbol, "1h", s, market)
    assert loads == []


def test_offline_hints_say_offline(settings, market, tmp_path):
    # Regresión: en --offline sugería "make train-regime SYMBOL=ETH", que entrena
    # modelos reales y deja el sintético viejo.
    s = small_settings(settings, tmp_path, auto_retrain=False)
    regime("ETH", "1h", s, market, synthetic=True)
    later = fixed_clock(FIXED_NOW + timedelta(days=9))
    kept = regime("ETH", "1h", s, market, synthetic=True, clock=later)
    stale = [w for w in kept.warnings if w.code == "stale_model"]
    assert len(stale) == 1
    assert stale[0].text.endswith('make train-regime SYMBOL=ETH ARGS="--offline"')
    real = regime("ETH", "1h", s, FakeMarketData(synthetic=market))
    real_later = regime("ETH", "1h", s, FakeMarketData(synthetic=market), clock=later)
    assert real.model.retrained_now
    text = next(w.text for w in real_later.warnings if w.code == "stale_model")
    assert text.endswith("make train-regime SYMBOL=ETH") and "--offline" not in text
    with pytest.raises(RegimeUnavailable) as info:
        regime("SOL", "1h", s, market, synthetic=True, allow_train=False)
    assert info.value.reason.endswith('make train-regime SYMBOL=SOL ARGS="--offline"')
    assert "((" not in info.value.reason


def test_short_training_history_is_warned(settings, market, tmp_path):
    # Moneda listada hace poco: 20 días de historia cuando se piden 60.
    young = SyntheticMarketData(now=FIXED_NOW, history_days=20)
    s = small_settings(settings, tmp_path)
    r = regime("AVAX", "1h", s, young)
    short = [w for w in r.warnings if w.code == "short_training"]
    assert len(short) == 1
    assert "se entrenó con solo 20 días de historia y se piden 60" in short[0].text
    # Con la historia completa no hay aviso.
    full = regime("AVAX", "1h", small_settings(settings, tmp_path / "otro"), market)
    assert "short_training" not in [w.code for w in full.warnings]


def test_short_training_warning_threshold(four_state_status):
    s, status = four_state_status
    model = status.model
    expected = s.regime.training_days * 24 - 167
    enough = dataclasses.replace(model, n_samples=int(0.8 * expected) + 1)
    few = dataclasses.replace(model, n_samples=int(0.8 * expected) - 1)
    assert service.short_training_warning(enough, s.regime, "1h") is None
    assert service.short_training_warning(few, s.regime, "1h") is not None
    assert service.short_training_warning(few, s.regime, "1h", min_coverage=0.5) is None


def test_cancelling_during_training_does_not_wait_for_it(settings, market, tmp_path, monkeypatch):
    # Regresión: Ctrl+C durante el entrenamiento tardaba lo que faltara de
    # entrenamiento (asyncio.run esperaba al hilo del ejecutor por defecto).
    started, release = threading.Event(), threading.Event()

    def slow_train(*args, **kwargs):
        started.set()
        release.wait(20)
        raise RuntimeError("entrenamiento abandonado")

    monkeypatch.setattr(service, "train_regime_model", slow_train)
    s = small_settings(settings, tmp_path)

    async def cancel_while_training() -> None:
        task = asyncio.create_task(get_regime("BTC", "1h", s, market, clock=fixed_clock()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    t0 = time.perf_counter()
    try:
        asyncio.run(cancel_while_training())
        elapsed = time.perf_counter() - t0
    finally:
        release.set()
        with service._TRAIN_LOCK:  # el hilo abandonado termina y suelta el candado
            pass
    assert elapsed < 5
    assert not (tmp_path / "reales" / "BTC_1h.joblib").exists()  # no se guardó nada


# --- historial, tablas y JSON ----------------------------------------------------------


@pytest.fixture(scope="module")
def reports(settings, market, tmp_path_factory) -> dict[str, RegimeReport]:
    s = small_settings(settings, tmp_path_factory.mktemp("modelos"))
    results = asyncio.run(get_market_regimes("BTC", s, market, synthetic=True, clock=fixed_clock()))
    assert all(isinstance(r, RegimeReport) for r in results.values()), results
    return results  # type: ignore[return-value]


def test_market_regimes_have_main_and_context(reports, settings):
    assert list(reports) == [settings.regime.timeframe, settings.regime.context_timeframe]
    main, ctx = reports["1h"], reports["1d"]
    assert not main.is_context and main.role == "main"
    assert ctx.is_context and ctx.role == "context"


@pytest.mark.parametrize(("timeframe", "bars_per_day"), [("1h", 24), ("1d", 1)])
def test_history_covers_display_days_after_burn_in(
    reports, settings, market, timeframe, bars_per_day
):
    cfg = settings.regime
    r = reports[timeframe]
    assert len(r.history) == cfg.display_days * bars_per_day
    times = [b.time for b in r.history]
    step = timedelta(minutes=r.timeframe_minutes)
    assert all(b - a == step for a, b in zip(times, times[1:], strict=False))
    assert times[-1] == r.as_of
    # Velas cerradas de la ventana de cálculo, menos calentamiento y arranque del filtro.
    days = cfg.inference_days if timeframe == "1h" else cfg.context_inference_days
    df = market.candles_frame("BTC", timeframe)
    df = df[df["closed"] & (df.index >= pd.Timestamp(FIXED_NOW - timedelta(days=days)))]
    X, _ = feature_matrix(df, cfg.features.realized_vol_window, cfg.features.volume_zscore_window)
    assert r.bars_analyzed == len(X) - cfg.burn_in_bars
    for b in r.history:
        assert 0.0 <= b.state_prob <= 1.0
        assert b.confidence == b.state_prob  # confianza = probabilidad del estado más probable
        assert b.regime_prob is None or 0.0 <= b.regime_prob <= 1.0 + 1e-12


def test_report_numbers_are_consistent(reports, settings):
    for r in reports.values():
        assert 0.0 <= r.confidence <= 1.0 + 1e-12
        assert r.confidence == r.top_state_prob
        assert 0.0 <= r.regime_prob <= 1.0 + 1e-12
        assert r.confidence_pct == pytest.approx(r.confidence * 100)
        assert r.low_confidence == (
            r.confidence < settings.regime_adjustments.low_confidence.threshold
        )
        assert r.low_confidence == any(w.code == "low_confidence" for w in r.warnings)
        assert r.unstable == any(w.code == "unstable" for w in r.warnings)
        assert r.unstable == (r.changes_in_window > settings.regime.instability_max_changes)
        assert sum(p["prob"] for p in r.probabilities) == pytest.approx(1.0)
        assert [s["rank"] for s in r.states] == list(range(r.model.n_states))
        returns = [s["mean_return_pct"] for s in r.states]
        assert returns == sorted(returns)  # del más bajista al más alcista
        assert sum(s["is_current"] for s in r.states) == 1
        assert sum(d["pct"] for d in r.distribution) == pytest.approx(100.0)
        assert sum(d["bars"] for d in r.distribution) == len(r.history)
        matrix = np.array(r.transition_matrix)
        assert matrix.shape == (r.model.n_states, r.model.n_states)
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert list(r.transition_names) == [s["name"] for s in r.states]
        assert 1 <= r.age_bars <= r.bars_analyzed
        assert r.started_at <= r.as_of
        assert r.model.synthetic and r.data_source == "sintetico"
        assert any(w.code == "data_source" for w in r.warnings)


def test_to_dict_is_json_serializable(reports):
    for r in reports.values():
        full = r.to_dict()
        text = json.dumps(full, allow_nan=False, ensure_ascii=False)
        back = json.loads(text)
        assert back["available"] is True and back["symbol"] == "BTC"
        assert len(back["history"]) == back["history_bars"] == len(r.history)
        assert back["warnings"] and len(back["warnings"]) == len(back["warning_codes"])
        assert set(back["model"]["bic_by_n"]) == {"4"}
        assert back["confidence_pct"] == pytest.approx(r.confidence_pct)
        short = r.to_dict(include_history=False)
        assert "history" not in short and short["history_bars"] == len(r.history)


def test_to_dict_turns_nan_and_inf_into_none(reports):
    r = reports["1h"]
    states = list(r.states)
    states[0] = {**states[0], "expected_duration_bars": math.inf, "mean_return_pct": math.nan}
    broken = dataclasses.replace(r, confidence=math.nan, states=tuple(states))
    out = broken.to_dict()
    json.dumps(out, allow_nan=False)
    assert out["confidence"] is None and out["confidence_pct"] is None
    assert out["states"][0]["expected_duration_bars"] is None
    assert out["states"][0]["mean_return_pct"] is None


def test_regimes_to_dict_includes_unavailable():
    report_less = {"1h": RegimeUnavailable("sin velas"), "1d": RegimeUnavailable("otra cosa")}
    out = regimes_to_dict(report_less)
    assert out == {
        "1h": {"available": False, "timeframe": "1h", "reason": "sin velas"},
        "1d": {"available": False, "timeframe": "1d", "reason": "otra cosa"},
    }


# --- errores ----------------------------------------------------------------------------


def test_insufficient_history_says_so(settings, tmp_path):
    short = SyntheticMarketData(now=FIXED_NOW, history_days=10)
    with pytest.raises(RegimeUnavailable) as info:
        regime("BTC", "1h", small_settings(settings, tmp_path), short)
    reason = info.value.reason
    assert "no hay suficiente historia" in reason and "velas de 1 hora" in reason
    # 100 para entrenar + 167 de calentamiento, dicho en palabras simples.
    assert "hacen falta al menos 267" in reason
    assert "las primeras 167 solo sirven para calcular los indicadores" in reason
    assert "calentamiento" not in reason and "((" not in reason


def test_exchange_down_gives_simple_reason(settings, market, tmp_path):
    md = FakeMarketData(synthetic=market, fail={"get_klines_history": BINANCE_451})
    with pytest.raises(RegimeUnavailable) as info:
        regime("BTC", "1h", small_settings(settings, tmp_path), md)
    assert info.value.reason == "no se pudieron bajar las velas de 1h de BTC (HTTP 451)"


def test_unknown_timeframe_is_a_programming_error(settings, market, tmp_path):
    with pytest.raises(ValueError, match="se pidió '4h'"):
        regime("BTC", "4h", small_settings(settings, tmp_path), market)


class _DailyDown(FakeMarketData):
    """Falla solo el historial de velas diarias."""

    async def get_klines_history(self, symbol, timeframe: Timeframe, start, end=None):
        if timeframe == "1d":
            raise BINANCE_451
        return await super().get_klines_history(symbol, timeframe, start, end)


def test_market_regimes_partial_failure(settings, market, tmp_path):
    s = small_settings(settings, tmp_path)
    results = asyncio.run(
        get_market_regimes("BTC", s, _DailyDown(synthetic=market), clock=fixed_clock())
    )
    assert isinstance(results["1h"], RegimeReport)
    assert isinstance(results["1d"], RegimeUnavailable)
    assert "velas de 1d" in results["1d"].reason
    out = regimes_to_dict(results, include_history=False)
    json.dumps(out, allow_nan=False)
    assert out["1h"]["available"] is True and out["1d"]["available"] is False


def test_market_regimes_contain_unexpected_errors(settings, market, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("se rompió")

    monkeypatch.setattr(service, "filtered_from_model", boom)
    results = asyncio.run(
        get_market_regimes("BTC", small_settings(settings, tmp_path), market, clock=fixed_clock())
    )
    for value in results.values():
        assert isinstance(value, RegimeUnavailable)
        assert "error inesperado" in value.reason and "se rompió" in value.reason


def test_regime_history_days(settings):
    cfg = settings.regime
    expected = max(
        cfg.training_days, cfg.context_training_days, cfg.inference_days, cfg.context_inference_days
    )
    assert regime_history_days(cfg) == expected + 1


# --- semántica de change_level (función pura) -----------------------------------------

SEVEN = [
    ("Crash", "crash"),
    ("Bajista fuerte", "bajista"),
    ("Bajista", "bajista"),
    ("Neutral", "neutral"),
    ("Alcista", "alcista"),
    ("Alcista fuerte", "alcista"),
    ("Euforia", "euforia"),
]


def seven_states() -> list[StateInfo]:
    return [
        StateInfo(i, i, name, cat, (i - 3) * 0.001, 0.01, 0.01, 5.0, 1 / 7)  # type: ignore[arg-type]
        for i, (name, cat) in enumerate(SEVEN)
    ]


ROW_A = [0, 0, 0, 0.1, 0.5, 0.4, 0]  # alcista 0,9; gana "Alcista"
ROW_B = [0, 0, 0, 0.1, 0.4, 0.5, 0]  # alcista 0,9; gana "Alcista fuerte"
ROW_C = [0, 0, 0, 0.4, 0.3, 0.3, 0]  # el estado top es Neutral, pero alcista suma 0,6


def classify(rows, level, **kw):
    params = {"confirmation_bars": 3, "instability_window": 20, "instability_max_changes": 4}
    params.update(kw)
    return classify_bars(np.array(rows, dtype=float), seven_states(), change_level=level, **params)


def test_category_level_ignores_moves_inside_a_category():
    bc = classify([ROW_A, ROW_B] * 3, "category")
    assert bc.raw_label == ["alcista"] * 6
    assert bc.confirmed_label == [None, None] + ["alcista"] * 4
    assert bc.regime_prob[:2] == [None, None]
    assert bc.regime_prob[2:] == pytest.approx([0.9] * 4)  # suma de los dos estados alcistas
    assert bc.confidence == pytest.approx([0.5] * 6)  # la del estado más probable
    assert bc.display_state == [None, None, 4, 5, 4, 5]  # el más probable de la categoría
    assert bc.unstable == [False] * 6


def test_state_level_counts_every_state_change():
    bc = classify([ROW_A, ROW_B] * 3, "state")
    assert bc.raw_label == [4, 5, 4, 5, 4, 5]
    assert bc.confirmed_label == [None] * 6  # nunca 3 velas seguidas
    assert bc.unstable == [False] * 5 + [True]  # 5 cambios > 4
    bc = classify([ROW_A] * 4, "state")
    assert bc.confirmed_label == [None, None, 4, 4]
    assert bc.regime_prob[2:] == pytest.approx([0.5, 0.5])  # solo la del estado
    assert bc.confidence == pytest.approx([0.5] * 4)


def test_category_wins_by_summed_probability():
    cat = classify([ROW_C] * 3, "category")
    assert cat.raw_state == [3, 3, 3] and cat.raw_label == ["alcista"] * 3
    assert cat.raw_label_prob == pytest.approx([0.6] * 3)
    assert cat.confirmed_label[-1] == "alcista" and cat.regime_prob[-1] == pytest.approx(0.6)
    # La confianza es la del estado más probable (Neutral, 0,4), no la de la categoría.
    assert cat.confidence[-1] == pytest.approx(0.4)
    assert cat.display_state[-1] == 4  # empate 0,3/0,3: el primero
    state = classify([ROW_C] * 3, "state")
    assert state.confirmed_label[-1] == 3 and state.regime_prob[-1] == pytest.approx(0.4)
    assert state.confidence[-1] == pytest.approx(0.4)


def test_classify_bars_rejects_bad_input():
    with pytest.raises(ValueError, match="forma"):
        classify_bars(
            np.ones((3, 2)) / 2,
            seven_states(),
            change_level="state",
            confirmation_bars=3,
            instability_window=20,
            instability_max_changes=4,
        )
    with pytest.raises(ValueError, match="change_level"):
        classify([ROW_A], "otro")


# --- build_report con probabilidades armadas a mano -------------------------------------


@pytest.fixture(scope="module")
def four_state_status(settings, market, tmp_path_factory) -> tuple[AppSettings, ModelStatus]:
    s = small_settings(settings, tmp_path_factory.mktemp("cuatro"))
    status = asyncio.run(ensure_model("BTC", "1h", s, market, clock=fixed_clock()))
    assert status.model.n_states == 4
    return s, status


def one_hot_by_rank(status: ModelStatus, ranks: list[int], p: float = 0.97) -> np.ndarray:
    """Probabilidades con el estado de cada rango casi seguro (``p``) en cada vela."""
    by_rank = {s.rank: s.index for s in status.model.states}
    k = status.model.n_states
    probs = np.full((len(ranks), k), (1 - p) / (k - 1))
    for t, r in enumerate(ranks):
        probs[t, by_rank[r]] = p
    return probs


def report_for(s, status, ranks, **kw):
    times = pd.date_range("2026-10-01", periods=len(ranks), freq="h", tz="UTC")
    now = times[-1].to_pydatetime() + timedelta(minutes=90)
    probs = kw.pop("probs", None)
    if probs is None:
        probs = one_hot_by_rank(status, ranks)
    return build_report(
        symbol="BTC",
        timeframe="1h",
        settings=s,
        status=status,
        probs=probs,
        times=times,
        now=now,
        **kw,
    )


def test_age_counts_from_the_start_of_the_run(four_state_status):
    s, status = four_state_status
    r = report_for(s, status, [1] * 20 + [2] * 7)
    by_rank = status.model.states_by_rank()
    assert r.confirmed and r.name == by_rank[2].name and r.category == by_rank[2].category
    assert r.confirmed_bars == 5 and r.age_bars == 7 and not r.age_is_lower_bound
    assert r.started_at == pd.Timestamp("2026-10-01 20:00", tz="UTC").to_pydatetime()
    assert r.confidence == pytest.approx(0.97) and not r.low_confidence
    assert r.pending is None and not r.unstable
    assert [w.code for w in r.warnings] == ["short_history"]  # 27 velas < 30 días

    whole = report_for(s, status, [2] * 27)
    assert whole.age_is_lower_bound and whole.age_bars == 27


def test_flicker_marks_unstable_and_pending_but_confidence_is_the_top_state(four_state_status):
    s, status = four_state_status
    ranks = [2] * 30 + [0, 1] * 4 + [3, 3]
    r = report_for(s, status, ranks)
    by_rank = status.model.states_by_rank()
    assert r.name == by_rank[2].name  # sigue vigente el último confirmado
    assert r.pending is not None
    assert (r.pending.name, r.pending.bars, r.pending.needed) == (by_rank[3].name, 2, 3)
    assert r.unstable and r.changes_in_window > s.regime.instability_max_changes
    # Regresión: la confianza era la del régimen confirmado (1 %) y decía "el modelo
    # no está seguro" con el modelo 97 % seguro del candidato. Ahora es la del
    # estado más probable, como pide la especificación.
    assert r.top_state_name == by_rank[3].name
    assert r.confidence == pytest.approx(0.97) and r.confidence == r.top_state_prob
    assert not r.low_confidence
    assert r.regime_prob == pytest.approx(0.01)  # el confirmado, solo para mostrar
    codes = [w.code for w in r.warnings]
    assert {"pending", "unstable"} <= set(codes) and "low_confidence" not in codes
    assert r.history[-1].unstable and r.history[-1].regime == by_rank[2].name
    assert r.history[-1].state == by_rank[3].name
    assert r.history[-1].confidence == pytest.approx(0.97)
    assert r.history[-1].regime_prob == pytest.approx(0.01)


def test_low_confidence_when_the_top_state_is_unsure(four_state_status):
    s, status = four_state_status
    ranks = [1] * 30
    probs = one_hot_by_rank(status, ranks, p=0.97)
    by_rank = {st.rank: st.index for st in status.model.states}
    probs[-1] = 0.0
    probs[-1, by_rank[1]] = 0.40  # el más probable, pero con 40 %
    probs[-1, by_rank[0]] = 0.35
    probs[-1, by_rank[2]] = 0.25
    r = report_for(s, status, ranks, probs=probs)
    threshold = s.regime_adjustments.low_confidence.threshold
    assert r.confidence == pytest.approx(0.40) and r.confidence < threshold
    assert r.low_confidence
    low = [w for w in r.warnings if w.code == "low_confidence"]
    assert len(low) == 1 and low[0].text.startswith("Confianza baja (40 %, menos de 60 %)")


def test_unconfirmed_start_shows_the_most_likely(four_state_status):
    s, status = four_state_status
    r = report_for(s, status, [0, 1, 0, 1, 2, 2])
    by_rank = status.model.states_by_rank()
    assert not r.confirmed and r.name == by_rank[2].name
    assert r.age_bars == 2 and r.confirmed_bars == 0 and r.pending is None
    assert "unconfirmed" in [w.code for w in r.warnings]
    assert r.distribution[-1]["name"] == UNCONFIRMED_NAME
    assert r.distribution[-1]["bars"] == 6
    assert "short_history" in [w.code for w in r.warnings]


def test_distribution_uses_category_names(four_state_status):
    s, status = four_state_status
    r = report_for(s, status, [0] * 10 + [3] * 30)
    names = [d["name"] for d in r.distribution]
    expected = [CATEGORY_NAMES[st.category] for st in status.model.states_by_rank()]
    # Las dos primeras velas todavía no tienen régimen confirmado, y las dos
    # primeras del rango 3 siguen contando como el régimen anterior (en espera).
    assert names == [*expected, UNCONFIRMED_NAME]
    by_category = {d["category"]: d["bars"] for d in r.distribution}
    first, last = status.model.states_by_rank()[0], status.model.states_by_rank()[-1]
    assert by_category[first.category] == 10 and by_category[last.category] == 28
    assert by_category[None] == 2


def test_old_data_and_outliers_are_warned(four_state_status):
    s, status = four_state_status
    times = pd.date_range("2026-10-01", periods=10, freq="h", tz="UTC")
    r = build_report(
        symbol="BTC",
        timeframe="1h",
        settings=s,
        status=status,
        probs=one_hot_by_rank(status, [1] * 10),
        times=times,
        now=times[-1].to_pydatetime() + timedelta(hours=6),
        n_degenerate=2,
    )
    codes = [w.code for w in r.warnings]
    assert "data_lag" in codes and "outliers" in codes


def test_state_level_confidence_is_the_state_probability(four_state_status):
    s, status = four_state_status
    s_state = s.model_copy(update={"regime": s.regime.model_copy(update={"change_level": "state"})})
    r = report_for(s_state, status, [1] * 10)
    assert r.change_level == "state" and r.confidence == pytest.approx(0.97)
    assert r.distribution[1]["name"] == status.model.states_by_rank()[1].name


# --- semántica de punta a punta con estados que comparten categoría -----------------------


def test_end_to_end_category_regime_prob_sums_its_states(settings, market, tmp_path):
    s = small_settings(settings, tmp_path, n_states_min=6, n_states_max=6)
    r = regime("XRP", "1h", s, market, synthetic=True)
    assert r.model.n_states == 6
    in_category = [p for p in r.probabilities if p["category"] == r.category]
    if r.confirmed:
        assert r.regime_prob == pytest.approx(sum(p["prob"] for p in in_category))
    assert r.name == max(in_category, key=lambda p: p["prob"])["name"]
    top = max(r.probabilities, key=lambda p: p["prob"])
    assert r.confidence == pytest.approx(top["prob"]) == r.top_state_prob

    s_state = s.model_copy(update={"regime": s.regime.model_copy(update={"change_level": "state"})})
    r2 = regime("XRP", "1h", s_state, market, synthetic=True)
    assert not r2.model.retrained_now  # change_level no cambia el modelo
    probs = {p["name"]: p["prob"] for p in r2.probabilities}
    if r2.confirmed:
        assert r2.regime_prob == pytest.approx(probs[r2.name])
    assert r2.confidence == pytest.approx(max(probs.values()))
    assert r2.changes_in_window >= r.changes_in_window


# --- mercado sintético con historia larga (modo --offline) ------------------------------


def test_synthetic_long_history_keeps_recent_candles(settings):
    days = regime_history_days(settings.regime)
    started = time.perf_counter()
    long = SyntheticMarketData(now=FIXED_NOW, min_history_days=days)
    hourly = long.candles_frame("AVAX", "1h")
    daily = long.candles_frame("AVAX", "1d")
    assert time.perf_counter() - started < 5  # segundos, no minutos
    short = SyntheticMarketData(now=FIXED_NOW)
    recent = short.candles_frame("AVAX", "1h")
    assert hourly.iloc[-len(recent) :].equals(recent)  # las recientes no cambian
    assert short.candles_frame("AVAX", "1d").equals(
        daily.iloc[-len(short.candles_frame("AVAX", "1d")) :]
    )
    closed_daily = daily[daily["closed"]]
    assert closed_daily.index[0] <= pd.Timestamp(
        FIXED_NOW - timedelta(days=settings.regime.context_training_days)
    )
    closed_hourly = hourly[hourly["closed"]]
    assert closed_hourly.index[0] <= pd.Timestamp(
        FIXED_NOW - timedelta(days=settings.regime.training_days)
    )
    # Continuidad: el empalme no deja saltos raros en los cierres diarios.
    jumps = np.abs(np.diff(np.log(closed_daily["close"].to_numpy())))
    assert jumps.max() < 0.3
    again = SyntheticMarketData(now=FIXED_NOW, min_history_days=days).candles_frame("AVAX", "1d")
    assert again.equals(daily)  # determinista
