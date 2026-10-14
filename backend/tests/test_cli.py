"""Script de consola: salida de texto y JSON en modo sin red, errores y cierre del cliente.

El régimen de mercado se calcula con una config chica (entrena en menos de un
segundo) y con las carpetas de modelos en un directorio temporal: los tests
nunca escriben en ``data/models``.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
import yaml

from app.config import DEFAULT_CONFIG_PATH, Secrets
from app.data.synthetic import DEFAULT_SEED, SyntheticMarketData
from app.storage import Repository
from scripts import analyze_cli
from scripts.analyze_cli import (
    CORRELATION_TITLE,
    DASH,
    LOCKED_TITLE,
    OFFLINE_BANNER,
    PLAN_TITLE,
    REGIME_TITLE,
    REGIME_UNAVAILABLE,
    SAFETY_TITLE,
    SCORE_TITLE,
    build_report,
    fmt_compact_usd,
    fmt_duration,
    fmt_num,
    fmt_pct,
    fmt_usd,
    main,
    parse_price,
    price_decimals,
    printable,
    progress_bar,
    render_text,
    resolve_state_paths,
    score_bar,
)
from tests.fakes import FIXED_NOW, FIXED_NOW_ISO, FakeMarketData

SECTIONS = [
    "Fuentes de datos",
    SAFETY_TITLE,
    "Precios",
    REGIME_TITLE,
    "Indicadores por timeframe",
    "Soportes y resistencias",
    "Rango diario típico",
    "Derivados",
    "Sentimiento y noticias",
    SCORE_TITLE,
    CORRELATION_TITLE,
    PLAN_TITLE,
]
LEAKS = re.compile(r"\b(nan|none|inf|null)\b", re.IGNORECASE)


def write_small_config(root: Path, **regime_changes) -> Path:
    """``config.yaml`` real con un régimen chico y rápido, y modelos dentro de ``root``.

    La base del estado de trading, el LOCKED y los informes también van dentro
    de ``root``: los tests nunca tocan ``data/``.
    """
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["app"].update(
        {
            "database_file": str(root / "estado" / "analyzer.db"),
            "lock_file": str(root / "estado" / "LOCKED"),
            "incident_dir": str(root / "estado" / "incidents"),
        }
    )
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


_CONFIG: dict[str, Path] = {}


@pytest.fixture(scope="module", autouse=True)
def small_config(tmp_path_factory) -> Path:
    path = write_small_config(tmp_path_factory.mktemp("cli"))
    _CONFIG["path"] = path
    yield path
    _CONFIG.clear()


def run(*argv: str) -> tuple[int, str, str]:
    """Corre la consola; sin ``--config`` usa la config chica del módulo."""
    args = list(argv)
    if "--config" not in args and "path" in _CONFIG:
        args += ["--config", str(_CONFIG["path"])]
    out, err = io.StringIO(), io.StringIO()
    code = main(args, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def synthetic_price(symbol: str) -> float:
    market = SyntheticMarketData(seed=DEFAULT_SEED, now=FIXED_NOW)
    return asyncio.run(market.get_ticker(symbol)).price


@pytest.fixture(scope="module")
def btc_text(small_config) -> str:
    code, out, err = run("BTC", "--offline", "--now", FIXED_NOW_ISO)
    assert code == 0, err
    return out


@pytest.fixture(scope="module")
def btc_json(small_config) -> dict:
    code, out, err = run("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO)
    assert code == 0, err
    return json.loads(out)


# --- texto ------------------------------------------------------------------------


def test_text_output_has_banner_and_sections(btc_text):
    assert btc_text.count(OFFLINE_BANNER) == 2  # arriba y abajo
    for title in SECTIONS:
        assert title in btc_text, title
    for word in ("Componente", "Reglas", "Fear & Greed", "Funding actual"):
        assert word in btc_text
    assert "BTC/USDT" in btc_text
    assert "\x1b[" not in btc_text  # sin colores si no es una terminal


def test_text_output_never_leaks_raw_missing_values(btc_text):
    assert LEAKS.search(btc_text) is None


def test_text_shows_every_timeframe_and_indicator(btc_text):
    header = next(line for line in btc_text.splitlines() if "1d" in line and "5m" in line)
    assert header.split() == ["1d", "4h", "1h", "15m", "5m"]
    for label in ("EMA 20", "EMA 200", "VWAP diario", "RSI", "MACD histograma", "ATR % del precio"):
        assert label in btc_text
    assert "Bollinger ancho" in btc_text and "Volumen relativo" in btc_text
    assert "Precio vs EMA 200" in btc_text


def test_nexo_not_available_note(btc_text):
    assert "no disponible: se usa Binance" in btc_text


def test_manual_nexo_price_and_diff_are_shown():
    reference = synthetic_price("BTC")
    manual = round(reference * 1.003, 2)
    code, out, err = run("BTC", "--offline", "--now", FIXED_NOW_ISO, "--nexo-price", str(manual))
    assert code == 0, err
    assert f"{fmt_num(manual, 2)} (cargado a mano)" in out
    expected_diff = (manual / reference - 1) * 100
    assert fmt_pct(expected_diff, sign=True) in out
    assert "no disponible: se usa Binance" not in out
    assert "difiere" not in out  # 0,3 % está por debajo del umbral de config


def test_manual_nexo_price_typo_warning():
    manual = synthetic_price("BTC") * 0.9
    code, out, _ = run("BTC", "--offline", "--now", FIXED_NOW_ISO, "--nexo-price", f"{manual:.2f}")
    assert code == 0
    assert "El precio de Nexo que cargaste difiere" in out


def test_xrp_prices_use_more_decimals():
    code, out, err = run("XRP", "--offline", "--now", FIXED_NOW_ISO)
    assert code == 0, err
    price = synthetic_price("XRP")
    assert price_decimals(price) == 4
    assert fmt_num(price, 4) in out
    assert LEAKS.search(out) is None


def test_color_only_when_requested(btc_json):
    colored = render_text(btc_json, color=True)
    assert "\x1b[" in colored
    assert analyze_cli.strip_ansi(colored) == render_text(btc_json, color=False)


# --- JSON -------------------------------------------------------------------------


def test_json_output_structure(btc_json):
    r = btc_json
    assert r["symbol"] == "BTC" and r["offline"] is True and r["banner"] == OFFLINE_BANNER
    assert r["profile"] == "agresivo"
    assert list(r["timeframes"]) == ["1d", "4h", "1h", "15m", "5m"]
    assert set(r["levels"]) == {"15m", "1h"}
    for key in ("prices", "daily_range", "derivatives", "sentiment", "score", "plan", "sources"):
        assert key in r
    assert "sizing_example" not in r
    assert r["prices"]["execution_source"] == "binance"
    assert r["prices"]["nexo_price"] is None
    assert len(r["sentiment"]["headlines"]) > 0
    assert list(r["regime"]) == ["1h", "1d"]


def test_json_numbers_are_plausible(btc_json):
    r = btc_json
    price = r["prices"]["execution_price"]
    assert 10_000 < price < 1_000_000
    for tf, s in r["timeframes"].items():
        assert 0 <= s["rsi"] <= 100, tf
        assert s["atr"] > 0 and s["bb_upper"] > s["bb_mid"] > s["bb_lower"]
        assert s["trend"] in {"alcista", "bajista", "lateral"}
    for tf, lv in r["levels"].items():
        assert all(x["price"] < price for x in lv["supports"]), tf
        assert all(x["price"] > price for x in lv["resistances"]), tf
    assert 0 < r["daily_range"]["atr_pct"] < 20
    fg = r["sentiment"]["fear_greed"]["value"]
    assert 0 <= fg <= 100


# --- score y plan (Hito 4) --------------------------------------------------------

DEADLINE_UTC = "2026-10-13T14:37:20+00:00"  # FIXED_NOW + trade_plan.time_stop_hours (24 h)
BADGES = {"LONG": "║      LONG      ║", "SHORT": "║      SHORT      ║"}
BADGES["NO_OPERAR"] = "║      NO OPERAR      ║"


@pytest.fixture(scope="module")
def sol_json(small_config) -> dict:
    # Con la semilla por omisión y FIXED_NOW, SOL da un SHORT que cumple todas las reglas.
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO)
    assert code == 0, err
    return json.loads(out)


def assert_plan_invariants(plan: dict) -> None:
    """Invariantes de cualquier plan del JSON de la consola."""
    assert plan["status"] in {"LONG", "SHORT", "NO_OPERAR"}
    failing = [r for r in plan["rules"] if not r["ok"]]
    assert plan["blocking_reasons"] == [r["message"] for r in failing]
    assert (plan["status"] == "NO_OPERAR") == bool(failing)
    assert all(r["message"].startswith("No operar: ") for r in failing)
    if plan["status"] != "NO_OPERAR":
        assert plan["status"].lower() == plan["direction"]
    assert plan["is_reference"] == (plan["status"] == "NO_OPERAR" and plan["stop"] is not None)
    if plan["stop"] is None:
        assert plan["take_profits"] == [] and plan["position"] is None
        return
    entry, stop = plan["entry"]["price"], plan["stop"]["price"]
    is_long = plan["direction"] == "long"
    assert stop < entry if is_long else stop > entry
    pos, risk = plan["position"], plan["risk"]
    assert pos["leverage"] == pytest.approx(pos["size_usd"] / pos["margin_usd"])
    assert 1.0 <= pos["leverage"] <= pos["max_leverage"] * (1 + 1e-9)
    assert risk["max_loss_usd"] <= risk["after_adjustments_usd"] * (1 + 1e-9)
    tps = {tp["name"]: tp for tp in plan["take_profits"]}
    assert {"TP2", "TP3"} <= set(tps)
    assert tps["TP2"]["net_rr"] >= plan["min_rr"] * (1 - 1e-9)
    assert tps["TP3"]["r_multiple"] > tps["TP2"]["r_multiple"]
    for tp in tps.values():
        assert tp["price"] > entry if is_long else tp["price"] < entry
    assert plan["fees"]["round_trip_at_stop_usd"] > 0
    assert plan["time_stop"]["deadline"] == DEADLINE_UTC


def _plan_block(text: str) -> str:
    return text[text.index(f"━━ {PLAN_TITLE}") :]


def test_json_plan_is_consistent(btc_json, sol_json):
    for report in (btc_json, sol_json):
        assert_plan_invariants(report["plan"])
        json.dumps(report["plan"], allow_nan=False)


def test_plan_uses_the_1h_regime(btc_json):
    plan_regime, regime_1h = btc_json["plan"]["regime"], btc_json["regime"]["1h"]
    assert plan_regime["available"] is True and plan_regime["timeframe"] == "1h"
    assert plan_regime["name"] == regime_1h["name"]
    assert plan_regime["category"] == regime_1h["category"]
    assert plan_regime["confidence"] == pytest.approx(regime_1h["confidence"])


@pytest.mark.parametrize("symbol", ["BTC", "ETH", "SOL", "XRP"])
def test_plan_for_each_symbol_offline(symbol):
    code, out, err = run(symbol, "--offline", "--json", "--now", FIXED_NOW_ISO)
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert_plan_invariants(plan)
    assert plan["direction"] == report["score"]["direction_hint"]
    text = render_text(report)
    block = _plan_block(text)
    assert BADGES[plan["status"]] in block
    assert plan["headline"] in block
    assert LEAKS.search(text) is None
    for rule in plan["rules"]:
        if not rule["ok"]:
            assert f"✗ {rule['message']}" in block
            if rule["suggestion"]:
                assert f"→ {rule['suggestion']}" in block


def test_score_section(btc_json, btc_text):
    sc = btc_json["score"]
    block = btc_text[btc_text.index(f"━━ {SCORE_TITLE}") : btc_text.index(f"━━ {PLAN_TITLE}")]
    assert sc["summary"] in block
    assert score_bar(sc["total"], 30, -30) in block
    for c in sc["components"]:
        assert c["label"] in block
        assert c["details"][0] in block
    total_line = next(line for line in block.splitlines() if line.strip().startswith("Total"))
    assert fmt_num(sc["total"], 1, sign=True) in total_line
    assert sum(c["points"] for c in sc["components"]) == pytest.approx(sc["total"])
    assert LEAKS.search(block) is None


def test_score_bar():
    assert score_bar(0, 30, -30) == "-100 ├──────────────┊─────●─────┊──────────────┤ +100"
    assert score_bar(100, 30, -30).endswith("●┤ +100")
    assert score_bar(-250, 30, -30).startswith("-100 ├●")  # se recorta a -100
    assert score_bar(32, 30, -30).count("●") == 1
    assert "●" not in score_bar(None, 30, -30)


def test_reference_plan_shows_levels_and_why_not(btc_json, btc_text):
    plan = btc_json["plan"]
    block = _plan_block(btc_text)
    # Con la config chica del régimen, BTC da long bloqueado: niveles como referencia.
    assert plan["status"] == "NO_OPERAR" and plan["is_reference"] is True
    assert "Niveles (referencia, no operar)" in block
    for label in ("Entrada (a mercado)", "Stop", "TP2", "TP3", "Pérdida máxima", "Comisiones"):
        assert label in block, label
    assert "Reglas que no se cumplen" in block
    if plan["stop"]["skipped_levels"]:
        assert "se salteó" in block and "demasiado cerca del precio" in block


def test_tradeable_plan_text(sol_json):
    plan = sol_json["plan"]
    assert plan["status"] == "SHORT" and plan["is_reference"] is False
    assert all(r["ok"] for r in plan["rules"])
    block = _plan_block(render_text(sol_json))
    assert BADGES["SHORT"] in block
    assert plan["headline"].startswith("SHORT a mercado en ")
    assert f"Reglas: se cumplen las {len(plan['rules'])}." in block
    for label in (
        "Entrada (a mercado)",
        "Margen",
        "Tamaño de la posición",
        "Apalancamiento resultante",
        "Liquidación (aislada)",
        "Liquidación modelo Nexo",
        "Pérdida máxima",
        "Comisiones",
        "Funding estimado",
        "De dónde salen los niveles",
        "Ajustes",
    ):
        assert label in block, label
    # El stop por tiempo va en la hora del usuario (Buenos Aires = UTC-3).
    assert plan["time_stop"]["deadline"] == DEADLINE_UTC
    assert "13/10 11:37 (24 h)" in block
    fees = plan["fees"]
    assert f"({fmt_pct(fees['pct_of_risk'], 1)} del riesgo)" in block
    assert f"({fmt_pct(fees['pct_of_target'], 1)} de la ganancia)" in block
    tp2 = next(tp for tp in plan["take_profits"] if tp["name"] == "TP2")
    row = next(line for line in block.splitlines() if line.strip().startswith("TP2"))
    assert fmt_usd(tp2["net_gain_usd"]) in row and fmt_num(tp2["net_rr"], 2) in row


def test_alternative_entry_is_summarized(sol_json):
    alt = sol_json["plan"]["alternative"]
    entry = sol_json["plan"]["entry"]["price"]
    assert alt["kind"] == "limit" and alt["timeframe"] == "15m"
    if alt["available"]:
        assert alt["level_kind"] == "resistencia" and alt["level_price"] > entry  # short
        assert alt["entry"] == alt["level_price"]
        block = _plan_block(render_text(sol_json))
        assert "Entrada alternativa (orden límite)" in block
        assert "Entrada alternativa (límite)" in block  # fila de la tabla
    else:
        assert alt["reason"]


def test_no_regime_plan_takes_it_as_low_confidence():
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--no-regime")
    assert code == 0, err
    plan = json.loads(out)["plan"]
    assert plan["regime"] == {"available": False}
    assert "regime_unavailable" in [a["code"] for a in plan["adjustments"]]
    assert plan["risk"]["after_adjustments_usd"] == pytest.approx(25.0)  # 50 × 0,5


def test_max_leverage_override_shows_red_warning():
    code, out, err = run(
        "BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--max-leverage", "30"
    )
    assert code == 0, err
    report = json.loads(out)
    prof = report["plan"]["profile"]
    assert prof["max_leverage"] == 30 and prof["leverage_overridden"] is True
    assert prof["red_warning"] is True
    text = render_text(report)
    header = text[: text.index("━━ Fuentes de datos")]
    assert "tope de apalancamiento 30x, cambiado a mano; el perfil trae 25x" in header
    assert "Advertencia: el tope de apalancamiento está en 30x" in header
    assert "Advertencia: el tope de apalancamiento está en 30x" in _plan_block(text)


def test_max_leverage_above_hard_cap_exits_2():
    code, out, err = run("BTC", "--offline", "--max-leverage", "60")
    assert code == 2 and out == ""
    assert "No se permite más de 50x de apalancamiento (pediste 60x)." in err


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "0,5", "0.0001"])
def test_max_leverage_invalid_exits_2(capsys, bad):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", f"--max-leverage={bad}"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "--max-leverage" in err
    # Mensajes propios del apalancamiento (no los ejemplos de precios).
    assert "96000,5" not in err
    if bad != "abc":
        assert "tiene que ser 1x o más (1x = sin apalancamiento" in err


def test_high_volatility_event_blocks_the_plan():
    code, out, err = run(
        "SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--high-vol-event", "CPI de EE. UU."
    )
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert plan["status"] == "NO_OPERAR" and plan["is_reference"] is True
    failing = [r for r in plan["rules"] if not r["ok"]]
    assert [r["code"] for r in failing] == ["high_volatility_event"]
    assert failing[0]["message"].startswith(
        "No operar: marcaste un evento de alta volatilidad (CPI de EE. UU.)"
    )
    block = _plan_block(render_text(report))
    assert "✗ No operar: marcaste un evento de alta volatilidad (CPI de EE. UU.)" in block
    assert "→ Esperá a que pase el evento y desmarcalo." in block


def test_high_volatility_event_without_description():
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--high-vol-event")
    assert code == 0, err
    plan = json.loads(out)["plan"]
    rule = next(r for r in plan["rules"] if r["code"] == "high_volatility_event")
    assert rule["ok"] is False
    assert rule["message"].startswith("No operar: marcaste un evento de alta volatilidad;")


def test_daily_limits_from_the_command_line():
    base = ("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO)
    code, out, _ = run(*base, "--trades-today", "3")
    rule = next(r for r in json.loads(out)["plan"]["rules"] if r["code"] == "daily_limits")
    assert code == 0 and rule["ok"] is False
    assert "ya hiciste 3 operaciones hoy y el máximo es 3" in rule["message"]
    code, out, _ = run(*base, "--pnl-today", "-100")
    rule = next(r for r in json.loads(out)["plan"]["rules"] if r["code"] == "daily_limits")
    assert code == 0 and rule["ok"] is False
    assert "hoy perdiste 100 USD y el límite es 100 USD" in rule["message"]
    code, out, _ = run(*base, "--pnl-today", "-30,5", "--trades-today", "1")
    plan = json.loads(out)["plan"]
    rule = next(r for r in plan["rules"] if r["code"] == "daily_limits")
    assert code == 0 and rule["ok"] is True and plan["status"] == "SHORT"
    assert "1 de 3 operaciones, pérdida 30,50 USD" in rule["message"]


@pytest.mark.parametrize(
    ("flag", "value"), [("--trades-today", "-1"), ("--trades-today", "2.5"), ("--pnl-today", "x")]
)
def test_bad_day_state_exits_2(capsys, flag, value):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", f"{flag}={value}"])
    assert info.value.code == 2
    assert flag in capsys.readouterr().err


def test_json_never_contains_nan():
    code, out, _ = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO)
    assert code == 0
    assert "NaN" not in out and "Infinity" not in out
    json.loads(out)


def test_text_and_json_have_the_same_content(btc_json, btc_text):
    assert render_text(btc_json) == btc_text


def test_other_profile(btc_json):
    code, out, _ = run(
        "BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--profile", "conservador"
    )
    assert code == 0
    report = json.loads(out)
    assert report["profile"] == "conservador"
    plan = report["plan"]
    assert plan["profile"]["name"] == "conservador"
    assert plan["profile"]["risk_per_trade_usd"] == 15
    assert plan["risk"]["requested_usd"] == 15


# --- errores ----------------------------------------------------------------------


def test_unknown_profile_exits_2():
    code, out, err = run("BTC", "--offline", "--profile", "kamikaze")
    assert code == 2 and out == ""
    assert "No existe el perfil 'kamikaze'" in err


def test_invalid_symbol_exits_2():
    code, _, err = run("BTC-USDT", "--offline")
    assert code == 2
    assert "Símbolo inválido" in err


def test_invalid_nexo_price_exits_2(capsys):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", "--nexo-price", "0"])
    assert info.value.code == 2
    assert "mayor que 0" in capsys.readouterr().err


def test_yaml_syntax_error_exits_2_with_message(tmp_path):
    # Regresión: un YAML mal escrito salía con traceback y código 1.
    bad = tmp_path / "config.yaml"
    bad.write_text('app:\n  user_timezone: "America/Argentina\n  host: [sin cerrar\n')
    code, out, err = run("BTC", "--offline", "--config", str(bad))
    assert code == 2 and out == ""
    assert err.startswith(f"No se pudo leer la configuración ({bad}):")
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("96000", 96000.0),
        ("96000,5", 96000.5),
        ("96000.50", 96000.5),
        ("96.000,50", 96000.5),  # formato argentino, el mismo que imprime la CLI
        ("96,000.50", 96000.5),  # formato de EE. UU.
        ("1.234.567", 1234567.0),
        ("1,234,567.8", 1234567.8),
        ("2,6153", 2.6153),
        ("2.6153", 2.6153),
        ("0.512", 0.512),
        ("0,512", 0.512),
        (" 96 000,5 ", 96000.5),
    ],
)
def test_parse_price_formats(text, expected):
    assert parse_price(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["96.000", "96,000", "2.615", "1,234"])
def test_parse_price_rejects_ambiguous_thousands(text):
    # Regresión: "96.000" se leía como 96 y todo el análisis se hacía a 96 USD.
    with pytest.raises(ValueError, match="ambiguo"):
        parse_price(text)


@pytest.mark.parametrize("text", ["abc", "96.000.5", "1.2.3", "96,000,50", "", "-5"])
def test_parse_price_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_price(text)


def test_nexo_price_in_argentine_format_is_accepted():
    reference = synthetic_price("BTC")
    manual = round(reference * 1.003, 2)
    code, out, err = run(
        "BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--nexo-price", fmt_num(manual, 2)
    )
    assert code == 0, err
    assert json.loads(out)["prices"]["nexo_price"] == pytest.approx(manual)


def test_ambiguous_nexo_price_exits_2(capsys):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", "--nexo-price", "96.000"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "ambiguo" in err and "96000" in err


def test_born_liquidated_is_said_plainly(sol_json):
    # Regresión (del ejemplo de sizing del Hito 2): si la liquidación cae del lado
    # ganador de la entrada no se muestra "a 0,05 %" ni "-0,39 veces la distancia".
    # Con el ajuste de margen del plan no debería pasar nunca, pero si llega a pasar
    # se dice así.
    report = json.loads(json.dumps(sol_json))
    report["plan"]["liquidation"].update(at_open=True, ok=False, ratio=-0.39)
    block = _plan_block(render_text(report))
    line = next(x for x in block.splitlines() if "Liquidación (aislada)" in x)
    assert "nacería liquidada" in line
    assert "-0,39" not in line and "veces la distancia" not in line


def test_red_leverage_warning_is_shown_once(sol_json):
    report = json.loads(json.dumps(sol_json))
    text = "Advertencia: el apalancamiento queda en 34,8x, más de 25x. Un movimiento chico..."
    report["plan"]["position"].update(red_warning=True, red_warning_text=text)
    report["plan"]["warnings"].append(text)
    block = _plan_block(render_text(report))
    assert block.count(text) == 1
    line = next(x for x in block.splitlines() if "Apalancamiento resultante" in x)
    after = block.splitlines()[block.splitlines().index(line) + 1]
    assert after.strip() == text  # justo debajo de su fila, sin "ATENCIÓN:" de más


def test_plan_liquidation_is_never_born_liquidated(btc_json, sol_json):
    for report in (btc_json, sol_json):
        assert report["plan"]["liquidation"]["at_open"] is False


def test_printable_strips_terminal_escapes():
    assert printable("Bitcoin \x1b]52;c;ZWNobw==\x07 rallies\x1b[2J") == (
        "Bitcoin ]52;c;ZWNobw== rallies [2J"
    )


def test_bad_config_exits_2(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("app: {}\n", encoding="utf-8")
    code, _, err = run("BTC", "--offline", "--config", str(bad))
    assert code == 2
    assert "No se pudo leer la configuración" in err


@pytest.fixture
def no_secrets(monkeypatch):
    monkeypatch.setattr(
        analyze_cli,
        "get_secrets",
        lambda: Secrets.model_construct(anthropic_api_key=None, cryptopanic_api_key=None),
    )


@pytest.fixture
def tracked_clients(monkeypatch):
    created: list[httpx.AsyncClient] = []
    real = analyze_cli.make_client

    def factory(timeout_s: float = 10.0) -> httpx.AsyncClient:
        client = real(timeout_s)
        created.append(client)
        return client

    monkeypatch.setattr(analyze_cli, "make_client", factory)
    return created


def test_live_mode_exits_1_when_both_exchanges_fail(no_secrets, tracked_clients):
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(451))
        code, out, err = run("BTC")
    assert code == 1
    assert out == ""
    assert "No se pudieron conseguir los datos esenciales" in err
    assert "Binance (HTTP 451) y Bybit (HTTP 451) fallaron" in err
    assert len(tracked_clients) == 1 and tracked_clients[0].is_closed


def test_live_mode_invalid_symbol_says_so(no_secrets, tracked_clients):
    # Regresión: decía "Binance (HTTP 400) y Bybit (retCode 10001) fallaron ... probá de
    # nuevo en unos minutos" para un símbolo que no existe.
    fixtures = Path(__file__).parent / "fixtures"
    binance = json.loads((fixtures / "binance/error_invalid_symbol.json").read_text())
    bybit = json.loads((fixtures / "bybit/error_invalid_symbol.json").read_text())
    with respx.mock(assert_all_called=False) as router:
        router.route(host="fapi.binance.com").mock(return_value=httpx.Response(400, json=binance))
        router.route(host="api.bybit.com").mock(return_value=httpx.Response(200, json=bybit))
        router.route().mock(return_value=httpx.Response(503))
        code, out, err = run("BTCX")
    assert code == 1 and out == ""
    assert "Binance (símbolo inválido) y Bybit (símbolo inválido) fallaron" in err
    assert "Parece que BTCX no existe como futuro en USDT" in err
    assert "probá de nuevo" not in err
    assert len(tracked_clients) == 1 and tracked_clients[0].is_closed


def test_live_mode_prints_with_fake_market_and_closes_everything(
    monkeypatch, no_secrets, tracked_clients
):
    fake = FakeMarketData()
    monkeypatch.setattr(analyze_cli, "build_market_data", lambda settings, client, cache: fake)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(451))  # F&G, CoinGecko y RSS caídos
        code, out, err = run("BTC")
    assert code == 0, err
    assert OFFLINE_BANNER not in out
    assert "(Binance)" in out
    assert "No se pudo obtener el índice de miedo y codicia (HTTP 451)" in out
    assert "No pudimos leer ningún feed de noticias" in out
    assert "Sin titulares por ahora." in out
    assert REGIME_TITLE in out and "datos sintéticos" not in out  # modelo "real" (en tmp)
    assert LEAKS.search(out) is None
    assert fake.closed
    assert len(tracked_clients) == 1 and tracked_clients[0].is_closed


# --- reporte con datos faltantes y formato ---------------------------------------


def test_report_with_missing_parts_renders_dashes(settings):
    from app.data.snapshot import build_snapshot
    from app.indicators.pipeline import analyze_all
    from tests.fakes import BINANCE_451, FakePriceSource, synthetic_extras

    md = FakeMarketData(
        fail={
            "get_ticker": BINANCE_451,
            "get_funding": BINANCE_451,
            "get_funding_history": BINANCE_451,
            "get_open_interest": BINANCE_451,
            "get_open_interest_history": BINANCE_451,
            "get_long_short_ratio": BINANCE_451,
        },
        fail_timeframes={"5m": BINANCE_451},
    )
    extras = synthetic_extras(
        md.synthetic, fail={k: BINANCE_451 for k in ("fear_greed", "global_market", "news")}
    )
    snap = asyncio.run(
        build_snapshot(
            "BTC",
            settings,
            md,
            FakePriceSource(),
            None,
            None,
            None,
            extras=extras,
            clock=lambda: FIXED_NOW,
        )
    )
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    report = build_report(snap, analysis, settings, "agresivo")
    json.dumps(report, allow_nan=False)
    text = render_text(report)
    assert LEAKS.search(text) is None
    assert DASH in text
    assert "falló" in text
    assert "último cierre de las velas" in text
    assert "5m" not in report["timeframes"]
    assert report["derivatives"]["funding"] is None
    assert report["sentiment"]["fear_greed"] is None


def test_prices_without_ticker_do_not_name_binance(settings):
    # Regresión: sin ticker decía "Precio de referencia — (Binance)" y
    # "Nexo no disponible: se usa Binance" aunque se usaba el último cierre.
    from app.data.snapshot import build_snapshot
    from app.indicators.pipeline import analyze_all
    from tests.fakes import BINANCE_451, FakePriceSource, synthetic_extras

    md = FakeMarketData(fail={"get_ticker": BINANCE_451})
    snap = asyncio.run(
        build_snapshot(
            "BTC",
            settings,
            md,
            FakePriceSource(),
            None,
            None,
            None,
            extras=synthetic_extras(md.synthetic),
            clock=lambda: FIXED_NOW,
        )
    )
    analysis = analyze_all(snap.candles, settings, snap.current_price, now=snap.created_at)
    text = render_text(build_report(snap, analysis, settings, "agresivo"))
    prices = text[text.index("━━ Precios") : text.index("━━ Indicadores")]
    assert "Binance" not in prices
    assert re.search(r"Precio de referencia\s+no disponible", prices)
    assert "no disponible: se usa el último cierre de las velas" in prices
    assert "(último cierre de las velas)" in prices


def test_headline_escapes_are_not_printed(btc_json):
    report = json.loads(json.dumps(btc_json))
    report["sentiment"]["headlines"] = [
        {
            "published_at": FIXED_NOW_ISO,
            "source": "Evil\x1b[31m",
            "title": "Bitcoin \x1b]52;c;ZWNobw==\x07 rallies",
            "url": "",
            "is_macro": False,
        }
    ]
    report["warnings"].append("aviso con \x1b[2J escape")
    text = render_text(report)
    assert "\x1b" not in text and "\x07" not in text
    assert "Bitcoin ]52;c;ZWNobw== rallies" in text


def test_number_formatting():
    assert fmt_num(1234.5) == "1.234,50"
    assert fmt_num(-0.0001, 2) == "-0,00" or fmt_num(-0.0001, 2) == "0,00"
    assert fmt_num(0.0) == "0,00"
    assert fmt_num(12.3456, 1, sign=True) == "+12,3"
    assert fmt_num(None) == DASH
    assert fmt_num(float("nan")) == DASH
    assert fmt_num(float("inf")) == DASH
    assert fmt_pct(1.5) == "1,50 %"
    assert fmt_pct(None) == DASH
    assert price_decimals(98_000) == 2
    assert price_decimals(28) == 3
    assert price_decimals(2.6) == 4
    assert price_decimals(0.5) == 5
    assert price_decimals(None) == 2
    assert fmt_compact_usd(3.35e12) == "3,35 billones USD"
    assert fmt_compact_usd(8.49e9) == "8,49 mil millones USD"
    assert fmt_compact_usd(512e6) == "512,00 millones USD"
    assert fmt_compact_usd(1234.0) == "1.234,00 USD"
    assert fmt_compact_usd(None) == DASH


# --- régimen de mercado -------------------------------------------------------------


def _regime_block(text: str) -> str:
    return text[text.index(f"━━ {REGIME_TITLE}") : text.index("━━ Indicadores por timeframe")]


def test_regime_section_comes_after_prices(btc_text):
    assert btc_text.index("━━ Precios") < btc_text.index(f"━━ {REGIME_TITLE}")
    assert btc_text.index(f"━━ {REGIME_TITLE}") < btc_text.index("━━ Indicadores por timeframe")


def test_regime_section_shows_both_timeframes(btc_text, btc_json):
    block = _regime_block(btc_text)
    assert "1h · el que se usa para operar" in block
    assert "1d · contexto diario" in block
    # Regresión: la aserción era ``A and B or C`` y pasaba aunque faltara "Últimos".
    assert block.count("Últimos 30 días") == 2
    assert block.count("de régimen confirmado") == 2
    for label in ("Régimen actual", "Confianza", "Empezó", "Estabilidad", "Modelo"):
        assert block.count(label) == 2, label
    assert "Retorno medio por vela" in block and "% del tiempo" in block
    assert block.count("← ahora") == 2
    assert "elegidos por BIC" in block and "datos sintéticos" in block
    for tf in ("1h", "1d"):
        g = btc_json["regime"][tf]
        assert g["name"] in block
        assert f"{fmt_num(g['confidence_pct'], 0)} %" in block
        span = fmt_duration(g["age_bars"], g["timeframe_minutes"])
        assert span in block
        assert ("RÉGIMEN INESTABLE" in block) or not g["unstable"]
        if g["pending"]:
            assert f"lleva {g['pending']['bars']} de {g['pending']['needed']} velas" in block
    assert " h (" in block and " días (" in block  # 1h en horas, 1d en días
    assert LEAKS.search(block) is None


def test_regime_json_structure(btc_json):
    for tf, minutes in (("1h", 60), ("1d", 1440)):
        g = btc_json["regime"][tf]
        assert g["available"] is True and g["timeframe"] == tf
        assert g["timeframe_minutes"] == minutes
        assert g["symbol"] == "BTC" and g["model"]["synthetic"] is True
        assert 0 <= g["confidence_pct"] <= 100
        assert g["category"] in {"crash", "bajista", "neutral", "alcista", "euforia"}
        assert "history" not in g and g["history_bars"] > 0
        assert set(g["model"]["bic_by_n"]) == {"3", "4"}
        assert len(g["states"]) == g["model"]["n_states"]
        assert isinstance(g["confirmed"], bool) and isinstance(g["unstable"], bool)
    assert btc_json["regime"]["1h"]["role"] == "main"
    for tf in ("1h", "1d"):
        hs = btc_json["regime"][tf]["history_summary"]
        assert set(hs) == {
            "bars",
            "days",
            "regime_changes",
            "completed_stretches",
            "avg_regime_hours",
            "unstable_pct",
            "low_confidence_pct",
        }
    assert btc_json["regime"]["1d"]["role"] == "context"


def test_no_regime_hides_the_section():
    code, out, err = run("BTC", "--offline", "--now", FIXED_NOW_ISO, "--no-regime")
    assert code == 0, err
    assert REGIME_TITLE not in out and "Entrenando" not in err
    code, out, _ = run("BTC", "--offline", "--now", FIXED_NOW_ISO, "--no-regime", "--json")
    assert code == 0 and json.loads(out)["regime"] is None


def test_training_note_goes_to_stderr_and_json_stays_clean(tmp_path):
    cfg = write_small_config(tmp_path)
    argv = ("ETH", "--offline", "--json", "--now", FIXED_NOW_ISO, "--config", str(cfg))
    code, out, err = run(*argv)
    assert code == 0, err
    assert "Entrenando el modelo de régimen de ETH 1h" in err
    assert "Entrenando el modelo de régimen de ETH 1d" in err
    report = json.loads(out)  # stdout es solo el JSON
    assert report["regime"]["1h"]["model"]["retrained_now"] is True
    assert (tmp_path / "sinteticos" / "ETH_1h.joblib").is_file()
    assert (tmp_path / "sinteticos" / "ETH_1d.joblib").is_file()
    assert not (tmp_path / "reales").exists()  # --offline nunca toca los modelos reales
    code, out, err = run(*argv)
    assert code == 0 and "Entrenando" not in err
    assert json.loads(out)["regime"]["1h"]["model"]["retrained_now"] is False


def test_regime_unavailable_does_not_break_the_cli(tmp_path):
    cfg = write_small_config(tmp_path, min_training_samples=50_000)
    code, out, err = run("BTC", "--offline", "--now", FIXED_NOW_ISO, "--config", str(cfg))
    assert code == 0, err
    block = _regime_block(out)
    assert block.count(f"{REGIME_UNAVAILABLE}: no hay suficiente historia") == 2
    assert PLAN_TITLE in out
    assert "el régimen no está disponible" in out  # el plan lo toma como confianza baja
    assert LEAKS.search(out) is None
    code, out, _ = run("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--config", str(cfg))
    report = json.loads(out)
    regime = report["regime"]
    assert regime["1h"]["available"] is False and "suficiente historia" in regime["1h"]["reason"]
    assert report["plan"]["regime"] == {"available": False}


def test_offline_regime_does_not_change_market_numbers(tmp_path):
    # context_training_days=900 obliga a agregar ~380 días de prehistoria sintética.
    cfg = write_small_config(tmp_path, context_training_days=900)
    argv = ("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--config", str(cfg))
    code, out, err = run(*argv)
    assert code == 0, err
    with_regime = json.loads(out)
    assert with_regime["regime"]["1d"]["model"]["n_samples"] > 700
    code, out, _ = run(*argv, "--no-regime")
    without = json.loads(out)
    # La historia larga para el régimen agrega velas viejas pero no cambia las recientes.
    for key in ("prices", "timeframes", "levels", "daily_range", "derivatives", "score"):
        assert without[key] == with_regime[key], key


def test_fmt_duration():
    assert fmt_duration(14, 60) == "14 h"
    assert fmt_duration(1.94, 60, 1) == "1,9 h"
    assert fmt_duration(3, 1440) == "3 días"
    assert fmt_duration(1, 1440) == "1 día"
    assert fmt_duration(2.25, 1440, 1) == "2,2 días" or fmt_duration(2.25, 1440, 1) == "2,3 días"
    assert fmt_duration(None, 60) == DASH


def test_confidence_row_names_the_top_state_when_it_is_not_the_regime(btc_json):
    # Regresión: decía "Confianza 0 %" (la del régimen confirmado) y escondía que
    # el modelo veía otro estado al 97 %. Ahora la confianza es la del estado más
    # probable y, si no es el régimen actual, se dice cuál es.
    g = dict(btc_json["regime"]["1h"])
    style = analyze_cli.Style(enabled=False)
    tz = analyze_cli._tz("UTC")
    other = next(s["name"] for s in g["states"] if s["name"] != g["name"])
    g.update(
        confidence=0.97,
        confidence_pct=97.0,
        low_confidence=False,
        top_state={"name": other, "category": "euforia", "prob": 0.97},
    )
    text = "\n".join(analyze_cli._render_one_regime("1h", g, style, tz))
    assert f"97 % (lo más probable en la última vela es {other})" in text
    assert "baja" not in text.split("Confianza", 1)[1].splitlines()[0]
    g.update(
        confidence=0.4,
        confidence_pct=40.0,
        low_confidence=True,
        top_state={"name": g["name"], "category": g["category"], "prob": 0.4},
    )
    text = "\n".join(analyze_cli._render_one_regime("1h", g, style, tz))
    line = next(x for x in text.splitlines() if "Confianza" in x)
    assert line.rstrip().endswith("40 % (baja: menos de 60 %)")


_UTC = analyze_cli._tz("UTC")


def _summary_row(g: dict, **summary) -> str:
    g = dict(g)
    g["history_summary"] = {
        "bars": 720,
        "days": 30.0,
        "unstable_pct": 0.0,
        "low_confidence_pct": 0.0,
        **summary,
    }
    text = "\n".join(analyze_cli._render_one_regime("1h", g, analyze_cli.Style(False), _UTC))
    return next(x for x in text.splitlines() if "Últimos 30 días" in x)


def test_history_summary_row_never_presents_the_window_as_a_duration(btc_json):
    # Regresión: sin cambios decía "cada tramo dura ~30,0 días" (la ventana misma)
    # al lado de "Empezó hace 43 días"; con 1 cambio promediaba dos tramos cortados.
    g = dict(btc_json["regime"]["1h"], confirmed=True)
    none = _summary_row(g, regime_changes=0, completed_stretches=0, avg_regime_hours=None)
    assert "ningún cambio de régimen confirmado (el actual ya venía de antes)" in none
    assert "dura" not in none and "duró" not in none
    one = _summary_row(g, regime_changes=1, completed_stretches=0, avg_regime_hours=None)
    assert "1 cambio de régimen confirmado (el que dio inicio al actual)" in one
    assert "dura" not in one and "duró" not in one
    two = _summary_row(g, regime_changes=2, completed_stretches=1, avg_regime_hours=30.0)
    assert "2 cambios de régimen confirmados (el tramo completo entre los dos duró ~30 h)" in two
    many = _summary_row(g, regime_changes=4, completed_stretches=3, avg_regime_hours=96.0)
    assert "4 cambios de régimen confirmados (los 3 tramos completos duraron ~4,0 días" in many
    assert LEAKS.search(none + one + two + many) is None


def test_stability_row_does_not_read_like_the_confirmed_changes(btc_json):
    # Regresión: "Estabilidad: 4 cambios en las últimas 20 velas" y justo abajo
    # "Últimos 30 días: 0 cambios de régimen" parecía una contradicción. Una fila
    # cuenta el tipo más probable vela a vela y la otra el régimen confirmado.
    g = dict(btc_json["regime"]["1h"], changes_in_window=4, unstable=False)
    text = "\n".join(analyze_cli._render_one_regime("1h", g, analyze_cli.Style(False), _UTC))
    row = next(x for x in text.splitlines() if "Estabilidad" in x)
    assert "el tipo más probable cambió 4 veces en las últimas 20 velas" in row
    g.update(changes_in_window=1)
    text = "\n".join(analyze_cli._render_one_regime("1h", g, analyze_cli.Style(False), _UTC))
    assert "cambió 1 vez en las últimas" in text
    g.update(change_level="state", changes_in_window=6, unstable=True)
    text = "\n".join(analyze_cli._render_one_regime("1h", g, analyze_cli.Style(False), _UTC))
    assert "RÉGIMEN INESTABLE: el estado más probable cambió 6 veces" in text


def test_regime_candles_from_bybit_are_warned(monkeypatch, no_secrets, tracked_clients):
    # Regresión: el aviso de que las velas del régimen vinieron de Bybit (porque
    # Binance falló) no aparecía en ningún lado del texto.
    warning = "Binance no respondió (HTTP 451); se usó Bybit"
    fake = FakeMarketData(name="bybit", meta_overrides={"warning": warning})
    monkeypatch.setattr(analyze_cli, "build_market_data", lambda settings, client, cache: fake)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(451))
        code, out, err = run("SOL")
    assert code == 0, err
    block = _regime_block(out)
    assert block.count(f"· {warning}") == 2  # 1h y 1d


def test_offline_regime_block_does_not_repeat_the_synthetic_warning(btc_text):
    assert "Datos sintéticos: no son reales" not in _regime_block(btc_text)


def test_ctrl_c_exits_130_without_traceback(monkeypatch):
    async def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(analyze_cli, "_fetch_data", interrupted)
    code, out, err = run("BTC", "--offline", "--now", FIXED_NOW_ISO)
    assert code == 130 and out == ""
    assert err.strip() == "Cancelado por el usuario." and "Traceback" not in err


# --- revisión final del Hito 4 ------------------------------------------------------


def test_fractional_leverage_cap_is_not_rounded():
    # Antes: "tope de apalancamiento 26x" en una línea y "está en 25,5x" en la siguiente.
    code, out, err = run("BTC", "--offline", "--no-regime", "--max-leverage", "25,5")
    assert code == 0, err
    header = out[: out.index("━━ Fuentes de datos")]
    assert "tope de apalancamiento 25,5x, cambiado a mano; el perfil trae 25x" in header
    assert "tope 25,5x" in _plan_block(out)
    assert "26x" not in out


def test_pnl_today_followed_by_another_option_is_not_eaten(capsys):
    # "--pnl-today --json" pegaba "--json" como monto ("no es un número: '-json'").
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", "--pnl-today", "--json"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "--pnl-today" in err and "-json" not in err.replace("--json", "")


def test_pnl_today_bad_amount_message(capsys):
    with pytest.raises(SystemExit) as info:
        main(["BTC", "--offline", "--pnl-today=-"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "no es un monto en USD: '-' (ej. -30 o -30,5" in err


def test_max_leverage_help_does_not_hardcode_the_red_threshold():
    action = next(
        a for a in analyze_cli.build_parser()._actions if "--max-leverage" in a.option_strings
    )
    assert "25x" not in action.help
    assert "leverage_red_warning_above" in action.help


def test_fees_outside_the_risk_are_broken_down(tmp_path):
    # Con fees_included_in_risk: false la pérdida máxima ya trae las comisiones:
    # antes decía "(más comisiones)" y se leían dos veces.
    data = yaml.safe_load(_CONFIG["path"].read_text(encoding="utf-8"))
    data["trade_plan"]["fees_included_in_risk"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    code, out, err = run(
        "SOL", "--offline", "--json", "--no-regime", "--now", FIXED_NOW_ISO, "--config", str(path)
    )
    assert code == 0, err
    report = json.loads(out)
    risk = report["plan"]["risk"]
    assert risk["fees_included"] is False
    assert risk["max_loss_usd"] == pytest.approx(risk["price_loss_usd"] + risk["fees_at_stop_usd"])
    line = next(x for x in render_text(report).splitlines() if "Pérdida máxima" in x)
    assert "más comisiones" not in line
    assert (
        f"{fmt_usd(risk['max_loss_usd'])} si toca el stop ({fmt_usd(risk['price_loss_usd'])} de "
        f"precio + {fmt_usd(risk['fees_at_stop_usd'])} de comisiones; el funding va aparte)"
    ) in line


def test_day_state_is_shown_and_in_the_json():
    base = ("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO)
    code, out, err = run(*base, "--trades-today", "1", "--pnl-today", "-30,5")
    assert code == 0, err
    report = json.loads(out)
    day = report["plan"]["day_state"]
    assert day["trades_today"] == 1 and day["pnl_today_usd"] == -30.5
    assert day["max_daily_loss_usd"] == 100.0 and day["high_volatility_event"] is False
    block = _plan_block(render_text(report))
    assert "Hoy: 1 de 3 operaciones, pérdida 30,50 USD de 100 USD permitidos." in block
    # Sin cargar el día no se muestra esa línea.
    code, out, _ = run(*base)
    assert "Hoy: " not in _plan_block(render_text(json.loads(out)))


def test_daily_loss_limit_caps_the_risk_in_the_plan_block():
    # Perdiste 80 de 100: el riesgo baja de 50 a 20 y se dice por qué.
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--pnl-today", "-80")
    assert code == 0, err
    report = json.loads(out)
    risk = report["plan"]["risk"]
    assert risk["capped_by_daily_limit"] is True
    assert risk["after_adjustments_usd"] == pytest.approx(20.0)
    assert risk["max_loss_usd"] <= 20.0 + 1e-9
    block = _plan_block(render_text(report))
    assert (
        "Riesgo bajado de 50 a 20 USD: hoy ya perdiste 80 de tus 100 USD de límite del día"
    ) in block
    assert "→ 20,00 USD (lo que te queda del límite de pérdida del día)" in block
    assert "Ojo con el límite del día" not in block


def test_daily_loss_limit_almost_reached_blocks_the_plan():
    code, out, err = run("SOL", "--offline", "--now", FIXED_NOW_ISO, "--pnl-today", "-99")
    assert code == 0, err
    block = _plan_block(out)
    assert (
        "✗ No operar: hoy ya perdiste 99 de tus 100 USD de límite del día; lo que te queda "
        "(1 USD) es menos que el mínimo operable de 15 USD."
    ) in block
    assert "→ Esperá a mañana." in block


def test_levels_table_explains_r_rr_and_atr(sol_json):
    block = _plan_block(render_text(sol_json))
    assert "R = lo que arriesgás hasta el stop" in block
    assert "R:R 1,5 = por cada 1 USD que arriesgás ganás 1,5" in block
    assert "ATR 15m = lo que se mueve el precio en una vela normal de 15m" in block


def test_alternative_block_shows_its_adjustments_liquidation_fees_and_time_stop(sol_json):
    report = json.loads(json.dumps(sol_json))
    alt = report["plan"]["alternative"]
    if not alt["available"]:
        pytest.skip("con estos datos no hay entrada alternativa")
    alt["adjustments"].append(
        {
            "source": "position",
            "code": "margin_raised",
            "message": "Margen subido de 100 a 166,20 USD para no pasar de 25x.",
            "risk_multiplier": 1.0,
        }
    )
    block = _plan_block(render_text(report))
    alt_block = block[block.index("Entrada alternativa (orden límite)") :]
    assert "Ajustes de esta entrada:" in alt_block
    assert "Margen subido de 100 a 166,20 USD para no pasar de 25x." in alt_block
    assert "Liquidación (aislada)" in alt_block
    assert "Comisiones (ida y vuelta)" in alt_block
    assert "Stop por tiempo" in alt_block
    # Los ajustes del régimen ya están arriba: no se repiten en la alternativa.
    assert "el régimen" not in alt_block.split("Ajustes de esta entrada:")[1]


def test_alternative_block_shows_the_skipped_levels(sol_json):
    report = json.loads(json.dumps(sol_json))
    alt = report["plan"]["alternative"]
    if not alt["available"]:
        pytest.skip("con estos datos no hay entrada alternativa")
    near = alt["level_price"] * 0.999
    alt["skipped_levels"] = [{"price": near, "timeframe": "15m", "distance_atr": 0.25}]
    block = _plan_block(render_text(report))
    alt_block = block[block.index("Entrada alternativa (orden límite)") :]
    d = price_decimals(report["plan"]["entry"]["price"])
    assert (
        f"Se salteó {'la' if alt['level_kind'] == 'resistencia' else 'el'} {alt['level_kind']} "
        f"{fmt_num(near, d)} (15m) por estar a 0,25 ATR: tan cerca del precio que sería casi lo "
        "mismo que entrar a mercado."
    ) in alt_block


def test_requested_risk_row_shows_regime_and_daily_cap(sol_json):
    report = json.loads(json.dumps(sol_json))
    risk = report["plan"]["risk"]
    risk.update(
        {
            "requested_usd": 50.0,
            "risk_multiplier": 0.5,
            "after_regime_usd": 25.0,
            "capped_by_daily_limit": True,
            "after_adjustments_usd": 20.0,
        }
    )
    line = next(x for x in render_text(report).splitlines() if "Riesgo pedido" in x)
    assert line.split("Riesgo pedido", 1)[1].strip() == (
        "50,00 USD del perfil × 0,5 = 25,00 USD (ajustes por régimen) → 20,00 USD (lo que te "
        "queda del límite de pérdida del día)"
    )


def test_mistyped_nexo_price_blocks_the_plan():
    price = synthetic_price("XRP") * 10  # un cero de más
    code, out, err = run(
        "XRP", "--offline", "--json", "--now", FIXED_NOW_ISO, "--nexo-price", f"{price:.4f}"
    )
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert plan["status"] == "NO_OPERAR"
    assert plan["entry"]["price_suspect"] is True
    assert plan["entry"]["reference_price"] == pytest.approx(report["prices"]["reference_price"])
    assert plan["rules"][0]["code"] == "price_check" and plan["rules"][0]["ok"] is False
    assert plan["headline"].startswith("No operar: el precio de Nexo que cargaste (")
    assert "difiere 900 % del de referencia" in plan["headline"]
    assert "sí cumple las reglas" not in plan["headline"]
    block = _plan_block(render_text(report))
    assert "✗ No operar: el precio de Nexo que cargaste" in block
    assert "→ Revisá el precio en Nexo y volvé a cargarlo bien" in block


def test_plan_headline_prices_match_the_table(sol_json):
    # El titular usa los mismos decimales que la tabla (sin los ceros de más).
    plan = sol_json["plan"]
    assert plan["status"] == "SHORT"
    d = price_decimals(plan["entry"]["price"])

    def shown(x: float) -> str:
        text = fmt_num(x, d)
        return text.rstrip("0").rstrip(",") if "," in text else text

    tp2 = next(tp for tp in plan["take_profits"] if tp["name"] == "TP2")
    assert plan["headline"].startswith(
        f"SHORT a mercado en {shown(plan['entry']['price'])}: stop en "
        f"{shown(plan['stop']['price'])} y objetivo principal (TP2) en {shown(tp2['price'])};"
    )


# --- Hito 5: frenos de seguridad, correlación y estado cargado a mano ----------------------

FIXED_BA = "America/Argentina/Buenos_Aires"


def _state_repo(db: Path) -> Repository:
    return Repository.open(db, clock=lambda: FIXED_NOW)


def _lock_of(db: Path) -> Path:
    """El LOCKED de una base de ``--db``: con el nombre de la base (``estado.LOCKED``)."""
    return db.parent / f"{db.stem}.LOCKED"


def _incidents_of(db: Path) -> Path:
    """La carpeta de informes de una base de ``--db`` (``estado_incidents/``)."""
    return db.parent / f"{db.stem}_incidents"


def _new_db(tmp_path: Path) -> Path:
    """Una base vacía (``--db`` tiene que existir: la consola no la crea)."""
    db = tmp_path / "estado.db"
    _state_repo(db).engine.dispose()
    return db


def _local(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=ZoneInfo(FIXED_BA))


def test_sections_order_safety_near_the_top(btc_text):
    assert btc_text.index("━━ Fuentes de datos") < btc_text.index(f"━━ {SAFETY_TITLE}")
    assert btc_text.index(f"━━ {SAFETY_TITLE}") < btc_text.index("━━ Precios")
    assert btc_text.index(f"━━ {SCORE_TITLE}") < btc_text.index(f"━━ {CORRELATION_TITLE}")
    assert btc_text.index(f"━━ {CORRELATION_TITLE}") < btc_text.index(f"━━ {PLAN_TITLE}")


def test_safety_section_lists_every_breaker(btc_text):
    block = btc_text[btc_text.index(f"━━ {SAFETY_TITLE}") : btc_text.index("━━ Precios")]
    assert "Estado: VERDE · Ningún freno activo" in block
    for label in (
        "Pérdida del día: bajar el riesgo",
        "Pérdida del día: límite",
        "Pérdidas seguidas: pausa",
        "Pérdida de la semana: bajar el riesgo",
        "Caída del mes: bloqueo total",
        "Evento de alta volatilidad",
        "Bloqueo total (archivo LOCKED)",
    ):
        assert label in block
    assert "0 USD de 50 USD" in block and "0 de 3 seguidas" in block
    assert progress_bar(0) in block
    assert "Billetera del mes" in block and "(supuesto)" in block
    assert "Base de prueba de los datos sintéticos:" in block


def test_progress_bar():
    assert progress_bar(0.5) == "████░░░░  50 %"
    assert progress_bar(1.7) == "████████ 100 %"
    assert progress_bar(None) == "░░░░░░░░   0 %"


def test_json_has_safety_and_correlation(btc_json):
    safety = btc_json["safety"]
    assert safety["status"] == "verde" and safety["blocked"] is False
    assert [b["code"] for b in safety["verdict"]["breakers"]] == [
        "daily_loss_reduce",
        "daily_loss_block",
        "max_trades_today",
        "losing_streak",
        "weekly_loss_reduce",
        "monthly_drawdown_lock",
        "volatility_event",
        "lock_file",
    ]
    assert safety["state"]["trades_today"] == 0 and safety["day_state_source"] == "state"
    assert safety["paths"]["kind"] == "offline"
    corr = btc_json["correlation"]
    assert corr["symbols"][0] == "BTC"
    assert set(corr["symbols"]) >= {"BTC", "ETH", "SOL", "XRP", "AVAX"}
    assert corr["matrix"]["symbols"] == corr["symbols"]
    assert corr["limit_usd"] == 75.0 and set(corr["exposure"]) == {"long", "short"}
    assert btc_json["locked"] is False and btc_json["analysis_id"] is None
    assert btc_json["plan"]["safety"]["evaluated"] is True


def test_offline_uses_its_own_database(tmp_path):
    from app.config import load_settings

    path = write_small_config(tmp_path)
    settings = load_settings(path)
    offline = resolve_state_paths(settings, offline=True)
    real = resolve_state_paths(settings)
    assert offline.database.parent == tmp_path / "sinteticos"
    assert offline.lock_file.parent == offline.database.parent
    assert offline.database != real.database and offline.lock_file != real.lock_file
    code, out, err = run(
        "BTC", "--offline", "--json", "--no-regime", "--now", FIXED_NOW_ISO, "--config", str(path)
    )
    assert code == 0, err
    report = json.loads(out)
    assert report["safety"]["paths"]["database"] == str(offline.database)
    assert offline.database.exists() and not real.database.exists()
    assert report["analysis_id"] is None  # con datos sintéticos no se guarda


def test_loaded_losses_start_the_cooldown_and_veto_the_plan(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    for hour in (9, 10, 11):
        closed = _local(12, hour)
        r.add_trade(symbol="BTC", side="long", pnl_usd=-10.0, opened_at=closed, closed_at=closed)
    code, out, err = run(
        "SOL", "--offline", "--json", "--no-regime", "--now", FIXED_NOW_ISO, "--db", str(db)
    )
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert plan["status"] == "NO_OPERAR"
    assert plan["rules"][0]["code"] == "safety_losing_streak"
    assert "pausa de 4 horas" in plan["rules"][0]["message"]
    assert plan["headline"].startswith("No operar: perdiste 3 operaciones seguidas")
    assert report["safety"]["status"] == "rojo"
    assert plan["day_state"]["trades_today"] == 3
    assert plan["day_state"]["pnl_today_usd"] == -30.0
    # Con --db el análisis se guarda aunque los datos sean sintéticos.
    stored = _state_repo(db).get_analysis(report["analysis_id"])
    assert stored is not None and stored.symbol == "SOL" and stored.plan_status == "NO_OPERAR"
    assert stored.safety["status"] == "rojo" and stored.plan["status"] == "NO_OPERAR"
    text = render_text(report)
    block = text[text.index(f"━━ {SAFETY_TITLE}") : text.index("━━ Precios")]
    assert "Estado: ROJO" in block and "BLOQUEA" in block
    assert f"Análisis guardado con el número {report['analysis_id']}." in text


def test_daily_loss_halves_the_risk_from_the_loaded_state(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=1000.0, ts=_local(1, 9))
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-60.0, opened_at=_local(12, 8), closed_at=_local(12, 9)
    )
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert report["safety"]["status"] == "ambar"
    assert plan["risk"]["safety_multiplier"] == 0.5
    assert plan["risk"]["after_safety_usd"] == pytest.approx(plan["risk"]["after_regime_usd"] / 2)
    assert any(a["source"] == "safety" for a in plan["adjustments"])
    line = next(x for x in render_text(report).splitlines() if "Riesgo pedido" in x)
    assert "× 0,5 = " in line and "(frenos de seguridad)" in line
    # El saldo anotado se usa para la liquidación "modelo Nexo" (940 USD).
    assert plan["liquidation"]["wallet"]["equity_usd"] == pytest.approx(940.0)


def test_same_coin_open_position_shrinks_the_plan_and_hedge_warns(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_open_position(symbol="SOL", side="short", risk_usd=40.0, opened_at=_local(11, 20))
    r.add_open_position(symbol="SOL", side="long", risk_usd=15.0, opened_at=_local(11, 21))
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    report = json.loads(out)
    plan = report["plan"]
    assert plan["status"] == "SHORT"
    assert plan["risk"]["capped_by_correlation"] is True
    assert plan["risk"]["after_adjustments_usd"] == pytest.approx(35.0)
    assert plan["risk"]["max_loss_usd"] <= 35.0 + 1e-9
    assert any("funciona como cobertura" in w for w in plan["warnings"])
    assert report["correlation"]["plan_side"] == "short"
    assert report["correlation"]["exposure"]["short"]["existing_risk_usd"] == 40.0
    text = render_text(report)
    corr = text[text.index(f"━━ {CORRELATION_TITLE}") : text.index(f"━━ {PLAN_TITLE}")]
    assert "Riesgo sumado para el plan (short)" in corr
    assert "Ya usado: 40,00 USD. Le queda al plan: 35,00 USD." in corr
    # D6: la cobertura no suma al tope de correlación, pero sí al límite del día.
    assert (
        "No suma al tope de posiciones correlacionadas (1,5 × tu riesgo), pero su riesgo sí "
        "cuenta en lo que te queda del límite del día"
    ) in corr
    assert "así que no suma riesgo" not in text
    assert "(lo que entra en el tope con tus posiciones correlacionadas)" in text


def test_lock_file_means_no_plan(tmp_path):
    from app.safety.lockfile import write_lock

    db = _new_db(tmp_path)
    write_lock(
        _lock_of(db),
        reason="La billetera del mes cayó 72 % desde su máximo.",
        triggered_at=FIXED_NOW,
        report_path=_incidents_of(db) / "informe.md",
    )
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    report = json.loads(out)
    assert report["plan"] is None and report["locked"] is True
    assert report["score"] is not None  # el resto del análisis sale igual
    text = render_text(report)
    assert LEAKS.search(text) is None
    assert LOCKED_TITLE in text
    assert "La billetera del mes cayó 72 % desde su máximo." in text
    assert f"Revisá el informe y borrá {_lock_of(db)} a mano" in text
    plan_block = _plan_block(text)
    assert "No hay plan: la app está bloqueada por el archivo LOCKED" in plan_block
    assert "NO OPERAR" not in plan_block and "Niveles" not in plan_block
    assert "Si operás" not in text


def test_drawdown_writes_the_lock_once_and_then_no_plan(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    for day in (5, 8):
        r.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=-110.0,
            opened_at=_local(day, 10),
            closed_at=_local(day, 12),
        )
    lock = _lock_of(db)
    code, out, err = run("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    report = json.loads(out)
    assert lock.exists() and report["plan"] is None and report["locked"] is True
    assert report["safety"]["lock_trigger"]["action"] == "locked"
    reports = list(_incidents_of(db).glob("*.md"))
    assert len(reports) == 1
    assert "Recién se bloqueó la app" in render_text(report)
    first = lock.read_text(encoding="utf-8")
    code, out, err = run("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    again = json.loads(out)
    assert again["safety"]["lock_trigger"] is None and again["plan"] is None
    assert lock.read_text(encoding="utf-8") == first
    assert len(list(_incidents_of(db).glob("*.md"))) == 1


def test_day_overrides_warn_and_say_where_the_numbers_came_from(tmp_path):
    db = _new_db(tmp_path)
    code, out, err = run(
        "SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db), "--pnl-today", "-80"
    )
    assert code == 0, err
    report = json.loads(out)
    assert report["safety"]["day_state_source"] == "manual"
    assert report["plan"]["day_overrides"] == [
        {"field": "pnl_today_usd", "value": -80.0, "loaded": 0.0}
    ]
    assert any(
        w.startswith("Números del día puestos a mano para probar: resultado de hoy = -80 USD")
        for w in report["plan"]["warnings"]
    )
    assert report["safety"]["status"] == "verde"  # los frenos siguen mirando lo cargado


def test_live_mode_saves_the_analysis_in_the_configured_database(
    monkeypatch, no_secrets, tracked_clients
):
    fake = FakeMarketData()
    monkeypatch.setattr(analyze_cli, "build_market_data", lambda settings, client, cache: fake)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(return_value=httpx.Response(451))
        code, out, err = run("BTC", "--json", "--no-regime")
    assert code == 0, err
    report = json.loads(out)
    config = yaml.safe_load(_CONFIG["path"].read_text(encoding="utf-8"))
    database = Path(config["app"]["database_file"])
    assert report["safety"]["paths"] == {
        "kind": "real",
        "database": str(database),
        "lock_file": config["app"]["lock_file"],
        "incident_dir": config["app"]["incident_dir"],
        "database_display": str(database),
        "lock_file_display": config["app"]["lock_file"],
    }
    stored = Repository.open(database).get_analysis(report["analysis_id"])
    assert stored is not None and stored.symbol == "BTC"


def test_unusable_database_exits_1(tmp_path):
    bad = tmp_path / "roto.db"
    bad.write_text("no soy una base", encoding="utf-8")
    code, out, err = run("BTC", "--offline", "--no-regime", "--db", str(bad))
    assert code == 1 and out == ""
    assert "No se pudo abrir la base de tu estado de trading" in err


def _trades(db: Path, *argv: str, now: datetime) -> tuple[int, str, str]:
    from scripts import trades

    out, err = io.StringIO(), io.StringIO()
    code = trades.main([*argv, "--db", str(db)], out=out, err=err, clock=lambda: now)
    return code, out.getvalue(), err.getvalue()


def test_synthetic_analysis_is_marked_and_only_linked_on_purpose(tmp_path):
    """Hallazgo 18: un análisis con datos sintéticos queda marcado y no se liga solo."""
    db = _new_db(tmp_path)
    code, out, err = run("SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db))
    assert code == 0, err
    report = json.loads(out)
    analysis_id = report["analysis_id"]
    stored = _state_repo(db).get_analysis(analysis_id)
    assert stored.inputs["offline"] is True and stored.inputs["day_overrides"] == []
    assert (
        f"Análisis guardado con el número {analysis_id}. Queda marcado como de prueba "
        "(datos sintéticos): no se liga solo a una operación." in render_text(report)
    )
    later = FIXED_NOW + timedelta(hours=2)
    code, t_out, t_err = _trades(
        db, "add", "--symbol", "SOL", "--side", "short", "--pnl", "12,5", now=later
    )
    assert code == 0, t_err
    [auto] = _state_repo(db).list_trades()
    assert auto.analysis_id is None and auto.plan_status is None
    assert (
        f"el #{analysis_id} es de prueba (datos sintéticos) y no se liga solo (si de verdad "
        f"seguiste ese, cargalo con --analysis {analysis_id})" in t_out
    )
    code, t_out, t_err = _trades(
        db,
        "add",
        "--symbol",
        "SOL",
        "--side",
        "short",
        "--pnl",
        "5",
        "--analysis",
        str(analysis_id),
        now=later,
    )
    assert code == 0, t_err
    linked = _state_repo(db).list_trades()[-1]
    assert linked.analysis_id == analysis_id and linked.plan_status == "SHORT"
    assert f"Ligada al análisis #{analysis_id} del 12/10 11:37 (datos sintéticos)" in t_out
    assert "Ojo: ese análisis es de prueba (datos sintéticos)" in t_out


def test_analysis_with_test_numbers_is_marked(tmp_path):
    db = _new_db(tmp_path)
    code, out, err = run(
        "SOL", "--offline", "--json", "--now", FIXED_NOW_ISO, "--db", str(db), "--trades-today", "1"
    )
    assert code == 0, err
    report = json.loads(out)
    stored = _state_repo(db).get_analysis(report["analysis_id"])
    assert stored.inputs["day_overrides"] == [{"field": "trades_today", "value": 1, "loaded": 0}]
    assert "(datos sintéticos y números de prueba)" in render_text(report)


# --- integración del Hito 5 (segunda ronda) ----------------------------------------------


def _run_db(db: Path, symbol: str = "SOL", *extra: str, now: str = FIXED_NOW_ISO) -> dict:
    code, out, err = run(symbol, "--offline", "--json", "--now", now, "--db", str(db), *extra)
    assert code == 0, err
    return json.loads(out)


def _safety_block(text: str) -> str:
    return text[text.index(f"━━ {SAFETY_TITLE}") : text.index("━━ Precios")]


def test_lock_then_manual_unlock_brings_plans_back_until_a_new_drop(tmp_path):
    """Hallazgo 10: bloqueo → borrar LOCKED → plan → rebloqueo solo con otra caída."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    for day, pnl in ((5, -100.0), (7, -100.0), (8, 10.0), (11, -25.0)):
        r.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=pnl,
            opened_at=_local(day, 11),
            closed_at=_local(day, 12),
        )
    lock = _lock_of(db)
    incidents = _incidents_of(db)

    # 1) 300 → 85 USD: 71,67 % abajo. Se escribe el informe y el LOCKED, una vez.
    first = _run_db(db, "BTC", "--no-regime")
    assert first["plan"] is None and first["locked"] is True and lock.exists()
    assert first["safety"]["lock_trigger"]["action"] == "locked"
    assert first["safety"]["lock_trigger"]["lock_record_id"] is not None
    assert len(list(incidents.glob("*.md"))) == 1
    assert len(_state_repo(db).list_safety_locks()) == 1
    again = _run_db(db, "BTC", "--no-regime")
    assert again["safety"]["lock_trigger"] is None and len(list(incidents.glob("*.md"))) == 1

    # 2) El usuario lee el informe y borra el LOCKED: vuelve a haber plan.
    lock.unlink()
    unlocked = _run_db(db, "BTC", "--no-regime", now="2026-10-12T15:37:20Z")
    assert unlocked["locked"] is False and unlocked["plan"] is not None
    assert unlocked["safety"]["lock_trigger"] is None
    assert not lock.exists() and len(list(incidents.glob("*.md"))) == 1
    monthly = next(
        b for b in unlocked["safety"]["verdict"]["breakers"] if b["code"] == "monthly_drawdown_lock"
    )
    assert monthly["triggered"] is False and monthly["current"] == 0.0
    block = _safety_block(render_text(unlocked))
    assert (
        "saldo 85,00 USD · la caída se mide de nuevo desde el bloqueo del 12/10 11:37 (que ya "
        "desbloqueaste): máximo 85,00 USD · caída 0 %" in block
    )
    assert "En todo el mes: máximo del mes 300,00 USD · caída 71,67 % (ya no cuenta" in block

    # 3) Una pérdida chica después (85 → 65, 23,5 %) no vuelve a bloquear.
    r = Repository.open(db, clock=lambda: FIXED_NOW + timedelta(days=3))
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-20.0, opened_at=_local(13, 9), closed_at=_local(13, 10)
    )
    small = _run_db(db, "BTC", "--no-regime", now="2026-10-13T15:00:00Z")
    assert small["locked"] is False and small["plan"] is not None
    assert not lock.exists() and len(list(incidents.glob("*.md"))) == 1

    # 4) Otra caída de 70 % desde el saldo del bloqueo (85 → 25, 70,6 %): bloquea una vez.
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-40.0, opened_at=_local(14, 9), closed_at=_local(14, 10)
    )
    relock = _run_db(db, "BTC", "--no-regime", now="2026-10-14T15:00:00Z")
    assert relock["locked"] is True and relock["plan"] is None and lock.exists()
    assert relock["safety"]["lock_trigger"]["action"] == "locked"
    assert "Después del bloqueo del 12/10 11:37 (que se desbloqueó a mano)" in lock.read_text(
        encoding="utf-8"
    )
    reports = sorted(incidents.glob("*.md"))
    assert len(reports) == 2 and len(_state_repo(db).list_safety_locks()) == 2
    assert "Desde el bloqueo del 12/10/2026 11:37" in reports[-1].read_text(encoding="utf-8")
    _run_db(db, "BTC", "--no-regime", now="2026-10-14T15:30:00Z")
    assert len(list(incidents.glob("*.md"))) == 2


def test_missing_db_exits_1_without_creating_it(tmp_path):
    """Hallazgo 19: una --db mal escrita no crea una base vacía que diga todo VERDE."""
    missing = tmp_path / "no_existe" / "x.db"
    code, out, err = run("BTC", "--offline", "--no-regime", "--db", str(missing))
    assert code == 1 and out == ""
    assert "No existe la base" in err and "¿está bien escrita la ruta de --db?" in err
    assert not missing.exists() and not missing.parent.exists()


def test_relative_db_is_relative_to_where_make_was_run(tmp_path, monkeypatch):
    db = _new_db(tmp_path)
    monkeypatch.setenv(analyze_cli.INVOCATION_DIR_ENV, str(tmp_path))
    code, out, err = run(
        "BTC", "--offline", "--json", "--no-regime", "--now", FIXED_NOW_ISO, "--db", "estado.db"
    )
    assert code == 0, err
    assert json.loads(out)["safety"]["paths"]["database"] == str(db)


def test_open_positions_count_in_what_is_left_of_the_day(tmp_path):
    """D6: lo que queda del día descuenta el riesgo de las posiciones abiertas."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-20.0, opened_at=_local(12, 8), closed_at=_local(12, 9)
    )
    r.add_open_position(symbol="XRP", side="short", risk_usd=50.0, opened_at=_local(12, 10))
    report = _run_db(db)
    plan = report["plan"]
    assert report["safety"]["state"]["open_risk_usd"] == 50.0
    assert plan["risk"]["open_risk_usd"] == 50.0 and plan["risk"]["daily_open_risk_usd"] == 50.0
    assert plan["day_state"]["daily_remaining_usd"] == 30.0
    assert plan["risk"]["capped_by_daily_limit"] is True
    assert plan["risk"]["after_adjustments_usd"] == pytest.approx(30.0)
    text = render_text(report)
    assert (
        "→ 30,00 USD (lo que te queda del límite de pérdida del día, descontando lo que tenés en "
        "juego en posiciones abiertas)" in text
    )
    assert (
        "Límite del día: hoy perdiste 20,00 USD y tenés 50,00 USD en juego en posiciones abiertas "
        "(lo que perdés si saltan sus stops): te quedan 30,00 USD de 100,00 USD." in text
    )


def test_open_positions_can_leave_no_room_for_a_new_plan(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-40.0, opened_at=_local(12, 8), closed_at=_local(12, 9)
    )
    r.add_open_position(symbol="XRP", side="short", risk_usd=50.0, opened_at=_local(12, 10))
    report = _run_db(db)
    plan = report["plan"]
    assert plan["status"] == "NO_OPERAR"
    rule = next(x for x in plan["rules"] if x["code"] == "min_operable_risk")
    assert not rule["ok"]
    assert (
        "hoy ya perdiste 40 USD y tenés 50 USD en juego en posiciones abiertas" in rule["message"]
    )
    assert "quedan 10 USD, menos que el mínimo operable de 15 USD" in rule["message"]
    # La regla ya lo dice: la línea del límite del día no se repite.
    assert "Límite del día:" not in render_text(report)


def test_latched_day_loss_is_shown_with_the_current_result(tmp_path):
    """D1: el freno del día mira la pérdida más grande; la consola muestra las dos cosas."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=1000.0, ts=_local(1, 9))
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-60.0, opened_at=_local(12, 8), closed_at=_local(12, 9)
    )
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=20.0, opened_at=_local(12, 9), closed_at=_local(12, 10)
    )
    report = _run_db(db, "SOL", "--no-regime")
    safety = report["safety"]
    assert safety["status"] == "ambar"
    assert safety["state"]["day_max_loss_usd"] == 60.0 and safety["state"]["day_loss_usd"] == 40.0
    block = _safety_block(render_text(report))
    assert (
        "-40,00 USD neto · 2 cerradas (1 ganada, 1 perdida) · llegaste a perder 60,00 USD: los "
        "frenos del día cuentan eso hasta las 00:00" in block
    )
    assert "Hoy llegaste a perder 60 USD (ahora vas -40 USD)" in block
    assert "Lo que ganaste después no apaga el freno." in block


def test_upcoming_event_is_listed_once(tmp_path):
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_volatility_event("Fed", _local(12, 17), starts_at=_local(12, 15))
    report = _run_db(db, "SOL", "--no-regime")
    safety = report["safety"]
    assert safety["status"] == "verde"
    assert len(safety["upcoming_event_warnings"]) == 1
    assert safety["upcoming_event_warnings"][0] in safety["verdict"]["warnings"]
    block = _safety_block(render_text(report))
    assert block.count("«Fed»") == 1
    assert (
        "«Fed», programado desde 12/10 15:00 hasta 12/10 17:00 (en ese rato no se generan "
        "planes)" in block
    )


def test_lock_db_error_is_shown(btc_json):
    report = json.loads(json.dumps(btc_json))
    report["safety"]["lock_trigger"] = {
        "action": "locked",
        "report_path": None,
        "report_error": None,
        "db_error": "No se pudo anotar el bloqueo en la base (OperationalError: disco).",
    }
    block = _safety_block(render_text(report))
    assert "Recién se bloqueó la app" in block
    assert "No se pudo anotar el bloqueo en la base (OperationalError: disco)." in block


def test_assumed_wallet_caps_the_margin_like_the_safety_section(tmp_path):
    """Hallazgo 14: el plan usa el mismo saldo (supuesto) que muestran los frenos."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    for day, pnl in ((6, -90.0), (8, -95.0)):
        r.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=pnl,
            opened_at=_local(day, 11),
            closed_at=_local(day, 12),
        )
    report = _run_db(db)
    wallet = report["safety"]["state"]["wallet"]
    assert wallet["assumed"] is True and wallet["current_balance_usd"] == 115.0
    plan = report["plan"]
    assert plan["position"]["margin_usd"] <= 115.0
    assert plan["liquidation"]["wallet"]["equity_usd"] == 115.0
    assert plan["day_state"]["wallet_balance_assumed"] is True
    text = render_text(report)
    assert "con la billetera de 115 USD (saldo supuesto: nunca lo anotaste)" in text
    assert "saldo 115,00 USD (supuesto) · máximo del mes 300,00 USD · caída 61,67 %" in text


def test_wallet_noted_at_zero_means_no_trade(tmp_path):
    db = tmp_path / "estado.db"
    _state_repo(db).add_wallet_event(kind="balance", amount_usd=0.0, ts=_local(1, 0))
    report = _run_db(db)
    plan = report["plan"]
    assert plan["status"] == "NO_OPERAR"
    rule = next(x for x in plan["rules"] if x["code"] == "wallet_balance")
    assert rule["message"].startswith("No operar: tu billetera de futuros tiene 0 USD")
    assert plan["day_state"]["wallet_balance_usd"] == 0.0
    assert "no hay plata para poner de margen" in render_text(report)


def test_json_has_the_new_safety_fields(btc_json):
    state = btc_json["safety"]["state"]
    for key in ("day_max_loss_usd", "week_max_loss_usd", "open_risk_usd", "latest_month_lock"):
        assert key in state
    assert "restart" in state["wallet"] and state["wallet"]["restart"] is None
    assert btc_json["safety"]["upcoming_event_warnings"] == []
    plan = btc_json["plan"]
    assert plan["risk"]["open_risk_usd"] == 0.0
    assert plan["day_state"]["wallet_balance_usd"] == 300.0
    assert plan["day_state"]["wallet_balance_assumed"] is True


# --- tercera ronda del Hito 5: lo que muestra la consola -----------------------------------


def test_custom_db_has_its_own_lock_named_after_it(tmp_path):
    """Una --db de prueba en data/ no comparte el LOCKED ni los informes de la base real."""
    from app.config import load_settings

    settings = load_settings(write_small_config(tmp_path))
    real = resolve_state_paths(settings)
    test_db = real.database.parent / "prueba.db"
    custom = resolve_state_paths(settings, db=test_db)
    assert custom.kind == "custom"
    assert custom.lock_file == real.database.parent / "prueba.LOCKED"
    assert custom.incident_dir == real.database.parent / "prueba_incidents"
    assert custom.lock_file != real.lock_file and custom.incident_dir != real.incident_dir
    # La base real elegida con --db usa sus rutas de siempre.
    assert resolve_state_paths(settings, db=real.database) == real
    offline = resolve_state_paths(settings, offline=True)
    assert resolve_state_paths(settings, db=offline.database) == offline


def test_max_trades_turns_the_status_red_like_make_safety(tmp_path):
    """E4: con 3 de 3 operaciones hoy, la consola dice ROJO, igual que make safety."""
    from scripts import trades

    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    for hour in (8, 9, 10):
        closed = _local(12, hour)
        r.add_trade(symbol="BTC", side="long", pnl_usd=5.0, opened_at=closed, closed_at=closed)
    report = _run_db(db, "SOL", "--no-regime")
    assert report["safety"]["status"] == "rojo"
    block = _safety_block(render_text(report))
    expected = (
        "Estado: ROJO · No se generan planes: ya hiciste 3 de 3 operaciones permitidas hoy, "
        "hasta que termine el día (a las 00:00)."
    )
    assert expected in block and "Ningún freno activo" not in block
    out = io.StringIO()
    code = trades.main(
        ["status", "--db", str(db), "--config", str(_CONFIG["path"])],
        out=out,
        err=io.StringIO(),
        clock=lambda: FIXED_NOW,
    )
    assert code == 0 and expected in out.getvalue()


def test_second_lock_shows_one_drop_everywhere(tmp_path):
    """D5: con el LOCKED de un segundo bloqueo, la sección muestra la misma caída que el motivo."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    for day, pnl in ((5, -100.0), (7, -100.0), (8, 10.0), (11, -25.0)):
        r.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=pnl,
            opened_at=_local(day, 11),
            closed_at=_local(day, 12),
        )
    _run_db(db, "BTC", "--no-regime")  # primer bloqueo (300 → 85)
    _lock_of(db).unlink()
    r = Repository.open(db, clock=lambda: FIXED_NOW + timedelta(days=2))
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-65.0, opened_at=_local(13, 9), closed_at=_local(13, 10)
    )
    relock = _run_db(db, "BTC", "--no-regime", now="2026-10-14T15:00:00Z")
    assert relock["locked"] is True
    measure = relock["safety"]["verdict"]["drawdown_measure"]
    assert measure["basis"] == "since_unlock" and measure["while_locked"] is True
    block = _safety_block(render_text(relock))
    # 85 → 20: 76,47 % (lo que dicen el motivo, la fila del freno y la billetera).
    assert "volvió a caer 76,47 % desde su máximo (85 USD → 20 USD)" in block
    assert "76,47 % de 70 %" in block
    assert (
        "la caída que bloqueó la app se midió desde el bloqueo del 12/10 11:37 (que "
        "desbloqueaste): máximo 85,00 USD · caída 76,47 %"
    ) in block
    # La del mes entero (300 → 20) va aparte y dice que ya no cuenta.
    assert "En todo el mes: máximo del mes 300,00 USD · caída 93,33 % (ya no cuenta" in block
    lines_with_93 = [x for x in block.splitlines() if "93,33 %" in x]
    assert all("todo el mes" in x for x in lines_with_93)


def test_negative_balance_says_more_than_100_and_asks_to_check(tmp_path):
    """E5: saldo calculado negativo → "más del 100 %" y el aviso de revisar los datos."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=0.0, ts=_local(1, 0))
    r.add_trade(
        symbol="BTC", side="long", pnl_usd=-50.0, opened_at=_local(5, 11), closed_at=_local(5, 12)
    )
    report = _run_db(db, "SOL", "--no-regime")
    block = _safety_block(render_text(report))
    row = next(x for x in block.splitlines() if "Billetera del mes" in x)
    assert "saldo -50,00 USD" in row and "100,0 %" not in row
    assert "da negativo (-50 USD)" in block
    assert "un resultado o un retiro mal tipeado" in block


def test_empty_wallet_reference_rows_do_not_invent_a_wallet(tmp_path):
    """Hallazgo 14: con saldo 0 el plan de referencia dice "sin saldo" (no 300 ni un margen)."""
    db = tmp_path / "estado.db"
    _state_repo(db).add_wallet_event(kind="balance", amount_usd=0.0, ts=_local(1, 0))
    report = _run_db(db, "BTC", "--no-regime")
    plan = _plan_block(render_text(report))
    assert "No operar: tu billetera de futuros tiene 0 USD" in plan
    assert "sin saldo: tu billetera tiene 0,00 USD, no hay plata para poner de margen" in plan
    assert "billetera de 300 USD" not in plan and "Margen subido" not in plan
    margin = next(x for x in plan.splitlines() if x.strip().startswith("Margen "))
    assert "sin saldo" in margin


def _report_with_day(day_loss_block: dict, day_state: dict, failed: list[str]) -> dict:
    """Un reporte mínimo para probar la línea "Límite del día" del plan."""
    from scripts.analyze_cli import _plan_latched_day_loss, day_room_text

    report = {
        "safety": {
            "day_state_source": "state",
            "verdict": {"breakers": [day_loss_block]},
            "state": {"day_loss_usd": day_state["loss_today_usd"]},
        }
    }
    said = bool(set(failed) & {"daily_remaining", "daily_limits", "safety_daily_loss_block"})
    return {
        "latched": _plan_latched_day_loss(report),
        "room": None if said else day_room_text(day_state, _plan_latched_day_loss(report)),
    }


def test_day_room_line_with_a_latched_loss():
    """D1: la pérdida trabada se aclara; con el límite alcanzado no se dice "te quedan"."""
    day = {
        "daily_open_risk_usd": 20.0,
        "loss_today_usd": 30.0,
        "daily_remaining_usd": 50.0,
        "max_daily_loss_usd": 100.0,
    }
    reduce_only = _report_with_day({"code": "daily_loss_block", "current": 60.0}, day, [])
    assert reduce_only["latched"] == 60.0
    assert reduce_only["room"].startswith(
        "Límite del día: hoy perdiste 30,00 USD (antes llegaste a perder 60,00 USD: eso mantiene "
        "trabados los frenos del día que ya saltaron, pero lo que queda del límite se cuenta "
        "desde lo que perdés ahora)"
    )
    assert reduce_only["room"].endswith("te quedan 50,00 USD de 100,00 USD.")
    blocked = _report_with_day(
        {"code": "daily_loss_block", "current": 100.0}, day, ["safety_daily_loss_block"]
    )
    assert blocked["room"] is None


def test_latched_daily_block_plan_does_not_say_there_is_room_left(tmp_path):
    """D1: -100 y después +50 con una posición abierta: el plan no dice "te quedan 40 USD"."""
    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    r.add_trade(
        symbol="BTC",
        side="long",
        pnl_usd=-100.0,
        opened_at=_local(12, 0, 5),
        closed_at=_local(12, 0, 10),
    )
    r.add_trade(
        symbol="SOL",
        side="long",
        pnl_usd=50.0,
        opened_at=_local(12, 0, 7),
        closed_at=_local(12, 0, 20),
    )
    r.add_open_position(symbol="ETH", side="long", risk_usd=10.0, opened_at=_local(11, 20))
    report = _run_db(db, "BTC", "--no-regime")
    failed = {x["code"] for x in report["plan"]["rules"] if not x["ok"]}
    assert "safety_daily_loss_block" in failed
    plan = _plan_block(render_text(report))
    assert "llegaste a tu límite de pérdida del día" in plan
    assert "te quedan" not in plan


def test_unrecorded_lock_is_recorded_and_said(tmp_path):
    """E2: un LOCKED de este mes que no está en la base se anota (y la consola lo dice)."""
    from app.safety.lockfile import write_lock

    db = _new_db(tmp_path)
    write_lock(
        _lock_of(db),
        reason="La billetera del mes cayó 72 % desde su máximo.",
        triggered_at=FIXED_NOW - timedelta(days=1),
        report_path=None,
        trigger_code="monthly_drawdown_lock",
    )
    report = _run_db(db, "BTC", "--no-regime")
    assert report["safety"]["lock_backfill"]["action"] == "recorded"
    assert len(_state_repo(db).list_safety_locks()) == 1
    assert "no estaba anotado en la base: se anotó ahora" in _safety_block(render_text(report))
    again = _run_db(db, "BTC", "--no-regime")
    assert again["safety"]["lock_backfill"]["action"] == "already_recorded"
    assert "se anotó ahora" not in _safety_block(render_text(again))


def test_lock_write_error_is_explained_and_reports_do_not_pile_up(tmp_path, monkeypatch):
    """Si no se puede escribir el LOCKED: exit 1, mensaje claro y un solo informe."""
    import app.safety.breakers as breakers

    db = tmp_path / "estado.db"
    r = _state_repo(db)
    r.add_wallet_event(kind="balance", amount_usd=300.0, ts=_local(1, 9))
    for day in (5, 8):
        r.add_trade(
            symbol="BTC",
            side="long",
            pnl_usd=-110.0,
            opened_at=_local(day, 10),
            closed_at=_local(day, 12),
        )

    def denied(path, **_kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(breakers, "write_lock", denied)
    for _ in range(2):
        code, out, err = run(
            "BTC", "--offline", "--no-regime", "--now", FIXED_NOW_ISO, "--db", str(db)
        )
        assert code == 1 and out == ""
        assert f"No se pudo escribir el archivo LOCKED ({_lock_of(db)}): Permission denied." in err
        assert "revisá los permisos de esa carpeta y el espacio en disco" in err
        assert "No se pudieron evaluar los frenos" not in err
    assert not _lock_of(db).exists()
    assert len(list(_incidents_of(db).glob("*.md"))) == 1
