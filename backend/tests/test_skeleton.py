"""Hito 0: el esqueleto importa, la configuración es válida y la API responde."""

import copy
import importlib
import pkgutil

import pytest
import yaml
from fastapi.testclient import TestClient

import app
from app.config import DEFAULT_CONFIG_PATH, AppSettings


def test_all_modules_import():
    """Cada módulo del paquete se puede importar (no hay errores de sintaxis)."""
    names = [m.name for m in pkgutil.walk_packages(app.__path__, prefix="app.")]
    assert len(names) > 50
    for name in names:
        importlib.import_module(name)


def test_every_module_has_docstring():
    for m in pkgutil.walk_packages(app.__path__, prefix="app."):
        module = importlib.import_module(m.name)
        assert module.__doc__ and module.__doc__.strip(), f"{m.name} no tiene docstring"


def test_config_yaml_is_valid(settings: AppSettings):
    assert settings.risk_profiles.default == "agresivo"
    assert set(settings.risk_profiles.profiles) == {"conservador", "moderado", "agresivo"}
    assert settings.symbols.favorites == ["BTC", "ETH", "SOL", "XRP", "AVAX"]
    assert settings.exchanges.nexo.allow_orders is False


def test_aggressive_profile_matches_spec(settings: AppSettings):
    p = settings.risk_profiles.profiles["agresivo"]
    assert (p.monthly_wallet_usd, p.margin_per_trade_usd, p.risk_per_trade_usd) == (300, 100, 50)
    assert (p.min_rr, p.max_leverage, p.max_trades_per_day, p.max_daily_loss_usd) == (
        1.5,
        25,
        3,
        100,
    )


def _raw_config() -> dict:
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["scoring"]["weights"].update(trend_4h=0.5), "sumar 1.0"),
        (lambda c: c["risk_profiles"].update(leverage_hard_cap=60), "50x"),
        (
            lambda c: c["risk_profiles"]["profiles"]["agresivo"].update(max_leverage=55),
            "tope duro",
        ),
        (lambda c: c["exchanges"]["nexo"].update(allow_orders=True), "fase 1"),
        (lambda c: c["regime"]["labels"].pop(7), "7 estados"),
        (lambda c: c["indicators"].update(rsi_peroid=14), "Extra inputs"),
    ],
)
def test_invalid_config_is_rejected(mutate, message):
    raw = copy.deepcopy(_raw_config())
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        AppSettings.model_validate(raw)


def test_health_endpoint():
    from app.main import app as fastapi_app

    client = TestClient(fastapi_app)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["trading_enabled"] is False
