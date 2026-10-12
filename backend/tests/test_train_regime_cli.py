"""Script ``train_regime``: salida en castellano, carpetas de modelos y códigos de salida.

Sin red: ``--offline`` usa el mercado sintético y el modo "en vivo" usa un
``FakeMarketData``. La config del régimen es chica (entrena en menos de un
segundo) y los modelos van a un directorio temporal.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
import yaml

from app.config import DEFAULT_CONFIG_PATH, load_settings
from app.regime.store import load_model, save_model
from scripts import train_regime
from scripts.train_regime import BIC_HELP, OFFLINE_BANNER, fmt_seconds, main
from tests.fakes import BINANCE_451, FIXED_NOW_ISO, FakeMarketData

LEAKS = re.compile(r"\b(nan|none|inf|null)\b", re.IGNORECASE)


def write_small_config(root: Path, **regime_changes) -> Path:
    """``config.yaml`` real con un régimen chico y rápido, y modelos dentro de ``root``."""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["regime"].update(
        {
            "n_states_min": 3,
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
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def offline(cfg: Path, *extra: str) -> tuple[int, str, str]:
    return run(*extra, "--offline", "--now", FIXED_NOW_ISO, "--config", str(cfg))


def test_offline_trains_and_prints_everything(tmp_path):
    cfg = write_small_config(tmp_path)
    code, out, err = offline(cfg, "BTC")
    assert code == 0, err
    assert OFFLINE_BANNER in out
    assert "Entrenando el modelo de régimen de BTC 1h porque no hay modelo guardado." in err
    assert "Entrenando el modelo de régimen de BTC 1d porque no hay modelo guardado." in err
    for title in ("BTC 1h · el que se usa para operar", "BTC 1d · contexto diario"):
        assert title in out
    assert out.count("entrenado ahora porque no hay modelo guardado") == 2
    assert out.count(BIC_HELP) == 2
    assert out.count("← elegido") == 2
    assert out.count("Regímenes, del más bajista al más alcista:") == 2
    assert out.count("Probabilidad de pasar de un régimen a otro") == 2
    assert "Retorno medio por vela" in out and "Duración típica" in out
    assert "las primeras 190 solo sirven para calcular los indicadores" in out
    assert "tardó" in out and "Guardado en" in out
    assert " h " in out and " días " in out  # duraciones en horas (1h) y días (1d)
    assert LEAKS.search(out) is None
    synth = tmp_path / "sinteticos"
    assert (synth / "BTC_1h.joblib").is_file() and (synth / "BTC_1d.joblib").is_file()
    assert not (tmp_path / "reales").exists()  # nunca pisa los modelos reales
    assert load_model(synth / "BTC_1h.joblib").synthetic is True


def test_bic_table_marks_the_chosen_number_of_states(tmp_path):
    cfg = write_small_config(tmp_path)
    code, out, _ = offline(cfg, "SOL", "--timeframe", "1h")
    assert code == 0
    model = load_model(tmp_path / "sinteticos" / "SOL_1h.joblib")
    rows = [line.split() for line in out.splitlines() if re.match(r"^\s+[34]\s", line)]
    assert [r[0] for r in rows] == ["3", "4"]
    chosen = [r for r in rows if "elegido" in r]
    assert len(chosen) == 1 and chosen[0][0] == str(model.n_states)
    assert f"se eligió {model.n_states}" in out
    # Los regímenes salen ordenados del más bajista al más alcista.
    names = [s.name for s in model.states_by_rank()]
    positions = [out.index(f"    {name} ") for name in names]
    assert positions == sorted(positions)


def test_up_to_date_model_is_not_retrained_without_force(tmp_path):
    cfg = write_small_config(tmp_path)
    assert offline(cfg, "ETH", "--timeframe", "1d")[0] == 0
    path = tmp_path / "sinteticos" / "ETH_1d.joblib"
    before = path.read_bytes()
    code, out, err = offline(cfg, "ETH", "--timeframe", "1d")
    assert code == 0 and "Entrenando" not in err
    assert "El modelo está al día: tiene menos de 1 día y se reentrena cada 7." in out
    assert "Usá --force para reentrenar igual" in out
    assert "((" not in out and "))" not in out  # sin paréntesis anidados
    assert path.read_bytes() == before
    code, out, err = offline(cfg, "ETH", "--timeframe", "1d", "--force")
    assert (
        code == 0 and "Entrenando el modelo de régimen de ETH 1d porque se pidió reentrenar" in err
    )
    assert "entrenado ahora porque se pidió reentrenar" in out
    assert "BTC 1h" not in out and "ETH 1h" not in out  # solo el timeframe pedido


def test_all_trains_every_favorite(tmp_path):
    cfg = write_small_config(tmp_path)
    code, out, err = offline(cfg, "all", "--timeframe", "1d")
    assert code == 0, err
    favorites = load_settings(cfg).symbols.favorites
    for symbol in favorites:
        assert f"{symbol} 1d · contexto diario" in out
        assert (tmp_path / "sinteticos" / f"{symbol}_1d.joblib").is_file()


def test_training_failure_exits_1_and_explains(tmp_path):
    cfg = write_small_config(tmp_path, min_training_samples=50_000)
    code, out, err = offline(cfg, "BTC")
    assert code == 1
    assert "BTC 1h: no se pudo entrenar. no hay suficiente historia" in err
    assert "BTC 1d: no se pudo entrenar." in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("BTC", "--timeframe", "4h"), "Timeframe inválido: '4h'"),
        (("BTC/USDT",), "Símbolo inválido"),
    ],
)
def test_bad_arguments_exit_2(tmp_path, argv, message):
    cfg = write_small_config(tmp_path)
    code, out, err = run(*argv, "--offline", "--config", str(cfg))
    assert code == 2 and out == ""
    assert message in err


def test_bad_config_exits_2(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("regime: [roto\n", encoding="utf-8")
    code, _, err = run("BTC", "--offline", "--config", str(bad))
    assert code == 2 and "No se pudo leer la configuración" in err


def test_unknown_option_exits_2(capsys):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--rapido"])
    assert info.value.code == 2


def test_live_mode_uses_real_folder_and_closes_the_client(tmp_path, monkeypatch):
    cfg = write_small_config(tmp_path)
    fake = FakeMarketData()
    clients = []
    real_make_client = train_regime.make_client

    def make_client(timeout_s: float = 10.0):
        client = real_make_client(timeout_s)
        clients.append(client)
        return client

    monkeypatch.setattr(train_regime, "make_client", make_client)
    monkeypatch.setattr(train_regime, "build_market_data", lambda settings, client, cache: fake)
    code, out, err = run("XRP", "--timeframe", "1d", "--config", str(cfg))
    assert code == 0, err
    assert OFFLINE_BANNER not in out
    assert (tmp_path / "reales" / "XRP_1d.joblib").is_file()
    assert not (tmp_path / "sinteticos").exists()
    assert fake.closed and len(clients) == 1 and clients[0].is_closed


def test_live_mode_exchange_down_exits_1(tmp_path, monkeypatch):
    cfg = write_small_config(tmp_path)
    fake = FakeMarketData(fail={"get_klines_history": BINANCE_451})
    monkeypatch.setattr(train_regime, "build_market_data", lambda settings, client, cache: fake)
    code, _, err = run("BTC", "--config", str(cfg))
    assert code == 1
    assert "no se pudieron bajar las velas de 1h de BTC (HTTP 451)" in err
    assert "no se pudieron bajar las velas de 1d de BTC (HTTP 451)" in err
    assert fake.closed


def test_fmt_seconds():
    assert fmt_seconds(108.3) == "1 min 48 s"
    assert fmt_seconds(14.2) == "14 s"
    assert fmt_seconds(0.2) == "menos de 1 s"
    assert fmt_seconds(None) == "—"


@pytest.mark.parametrize("extra", [(), ("--force",)])
def test_failed_retrain_keeps_old_model_and_exits_1(tmp_path, monkeypatch, extra):
    # Regresión: si reentrenar fallaba quedaba el modelo anterior, salía con 0 y
    # decía "El modelo está al día ... Usá --force" (aun después de --force).
    cfg = write_small_config(tmp_path, retrain_every_days=1)
    ok = FakeMarketData()
    monkeypatch.setattr(train_regime, "build_market_data", lambda settings, client, cache: ok)
    assert run("XRP", "--timeframe", "1d", "--config", str(cfg))[0] == 0
    path = tmp_path / "reales" / "XRP_1d.joblib"
    before = path.read_bytes()

    down = FakeMarketData(fail={"get_klines_history": BINANCE_451})
    monkeypatch.setattr(train_regime, "build_market_data", lambda settings, client, cache: down)
    if not extra:  # sin --force solo reentrena si está viejo: se lo envejece
        old = load_model(path)
        old.trained_at = old.trained_at.replace(year=old.trained_at.year - 1)
        save_model(old, path)
        before = path.read_bytes()
    code, out, err = run("XRP", "--timeframe", "1d", "--config", str(cfg), *extra)
    assert code == 1
    assert "XRP 1d: No se pudo reentrenar el modelo de régimen de 1d" in err
    assert "(HTTP 451)" in err and "Traceback" not in err
    assert re.search(r"Resultado +No se pudo reentrenar el modelo de régimen de 1d", out)
    assert "Se usa el anterior, del" in out
    assert "al día" not in out and "Usá --force" not in out
    assert path.read_bytes() == before  # el modelo anterior queda intacto


def test_ctrl_c_exits_130_without_traceback(tmp_path, monkeypatch):
    cfg = write_small_config(tmp_path)

    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train_regime, "_train_all", interrupted)
    code, out, err = offline(cfg, "BTC")
    assert code == 130
    assert err.strip() == "Cancelado por el usuario." and "Traceback" not in err
