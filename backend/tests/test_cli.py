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
from pathlib import Path

import httpx
import pytest
import respx
import yaml

from app.config import DEFAULT_CONFIG_PATH, Secrets
from app.data.synthetic import DEFAULT_SEED, SyntheticMarketData
from scripts import analyze_cli
from scripts.analyze_cli import (
    DASH,
    OFFLINE_BANNER,
    REGIME_TITLE,
    REGIME_UNAVAILABLE,
    SIZING_TITLE,
    build_report,
    fmt_compact_usd,
    fmt_duration,
    fmt_num,
    fmt_pct,
    main,
    parse_price,
    price_decimals,
    printable,
    render_text,
)
from tests.fakes import FIXED_NOW, FIXED_NOW_ISO, FakeMarketData

SECTIONS = [
    "Fuentes de datos",
    "Precios",
    REGIME_TITLE,
    "Indicadores por timeframe",
    "Soportes y resistencias",
    "Rango diario típico",
    "Derivados",
    "Sentimiento y noticias",
    SIZING_TITLE,
]
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
    for word in ("LONG hipotético", "SHORT hipotético", "Fear & Greed", "Funding actual"):
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
    for key in ("prices", "daily_range", "derivatives", "sentiment", "sizing_example", "sources"):
        assert key in r
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


def test_json_sizing_example_is_consistent(btc_json):
    ex = btc_json["sizing_example"]
    assert ex["profile"] == "agresivo"
    for key in ("long", "short"):
        side = ex[key]
        assert side["available"], side
        assert side["leverage"] == pytest.approx(side["size_usd"] / side["margin_usd"])
        assert side["fees_usd"] > 0
        assert side["loss_at_stop_usd"] == pytest.approx(side["risk_usd"])  # comisiones incluidas
        entry, stop, liq = side["entry"], side["stop"], side["liquidation_price"]
        if key == "long":
            assert stop < entry and liq < entry
        else:
            assert stop > entry and liq > entry
        assert side["liquidation_ok"] == (side["liquidation_ratio"] >= 2.0)
        if not side["liquidation_ok"]:
            assert side["suggested_margin_usd"] > side["margin_usd"]
        if side["leverage"] > 50:
            assert any("tope duro" in f for f in side["leverage_flags"])
        # Decisión del usuario: el nivel del stop está a 1 ATR(15m) o más de la entrada
        # y los más cercanos aparecen como salteados.
        min_atr = side["min_stop_distance_atr"]
        assert min_atr == 1.0
        assert side["level_distance_atr"] >= min_atr
        for sk in side["skipped_levels"]:
            assert sk["distance_atr"] < min_atr
            if key == "long":
                assert side["level_price"] < sk["price"] < entry
            else:
                assert entry < sk["price"] < side["level_price"]


def test_text_explains_skipped_levels(btc_json, btc_text):
    sides = [btc_json["sizing_example"][k] for k in ("long", "short")]
    if any(side["skipped_levels"] for side in sides):
        assert "Nivel salteado" in btc_text
        assert "demasiado cerca del precio" in btc_text


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
    ex = json.loads(out)["sizing_example"]
    assert ex["profile"] == "conservador" and ex["risk_usd"] == 15


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


def test_short_born_liquidated_is_said_plainly(tmp_path):
    # Regresión: con esta hora y SIN saltear niveles pegados al precio
    # (min_stop_distance_atr: 0), el SHORT tiene la liquidación del lado ganador
    # de la entrada; se mostraba "a 0,05 % de la entrada" y "-0,39 veces la
    # distancia". Con la regla del usuario (1 ATR) ese stop ya no se elige.
    cfg = tmp_path / "config.yaml"
    original = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert "min_stop_distance_atr: 1.0" in original
    cfg.write_text(
        original.replace("min_stop_distance_atr: 1.0", "min_stop_distance_atr: 0"),
        encoding="utf-8",
    )
    argv = (
        "BTC",
        "--offline",
        "--now",
        "2026-10-12T04:36:39Z",
        "--config",
        str(cfg),
        "--no-regime",
    )
    code, out, err = run(*argv, "--json")
    assert code == 0, err
    short = json.loads(out)["sizing_example"]["short"]
    assert short["liquidation_price"] < short["entry"]
    assert short["liquidation_at_open"] is True and short["liquidation_ok"] is False
    assert short["suggested_leverage"] > 50
    assert any("tope duro" in f for f in short["suggested_leverage_flags"])
    assert any("tope duro" in f for f in short["safe_size_leverage_flags"])
    code, text, _ = run(*argv)
    block = text[text.index("SHORT hipotético") :]
    assert "nacería liquidada" in block
    assert "-0,39" not in block and "veces la distancia al stop" not in block
    margin_line = next(line for line in block.splitlines() if "Margen sugerido" in line)
    after = block.splitlines()[block.splitlines().index(margin_line) + 1]
    assert "ATENCIÓN" in after and "tope duro" in after


def test_long_liquidation_is_not_flagged_as_born_liquidated(btc_json):
    for key in ("long", "short"):
        assert btc_json["sizing_example"][key]["liquidation_at_open"] is False


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
    assert SIZING_TITLE in out
    assert LEAKS.search(out) is None
    code, out, _ = run("BTC", "--offline", "--json", "--now", FIXED_NOW_ISO, "--config", str(cfg))
    regime = json.loads(out)["regime"]
    assert regime["1h"]["available"] is False and "suficiente historia" in regime["1h"]["reason"]


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
    for key in ("prices", "timeframes", "levels", "daily_range", "derivatives", "sizing_example"):
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
