"""Hito 4: perfiles de riesgo y tope de apalancamiento (override hasta 50x)."""

import json

import pytest

from app.risk.profiles import EffectiveProfile, get_profile, profile_names, red_warning_text


def _with_profiles(settings, **update):
    return settings.model_copy(
        update={"risk_profiles": settings.risk_profiles.model_copy(update=update)}
    )


def test_profile_names_follow_config(settings):
    assert profile_names(settings) == tuple(settings.risk_profiles.profiles)
    assert {"conservador", "moderado", "agresivo"} <= set(profile_names(settings))


@pytest.mark.parametrize("name", ["conservador", "moderado", "agresivo"])
def test_profile_values_come_from_config(settings, name):
    p = get_profile(name, settings)
    cfg = settings.risk_profiles.profiles[name]
    assert isinstance(p, EffectiveProfile)
    assert p.name == name
    assert p.monthly_wallet_usd == cfg.monthly_wallet_usd
    assert p.margin_per_trade_usd == cfg.margin_per_trade_usd
    assert p.risk_per_trade_usd == cfg.risk_per_trade_usd
    assert p.min_rr == cfg.min_rr
    assert p.max_leverage == cfg.max_leverage
    assert p.profile_max_leverage == cfg.max_leverage
    assert p.max_trades_per_day == cfg.max_trades_per_day
    assert p.max_daily_loss_usd == cfg.max_daily_loss_usd
    assert p.min_operable_risk_usd == cfg.min_operable_risk_usd
    assert p.leverage_overridden is False
    assert p.leverage_hard_cap == settings.risk_profiles.leverage_hard_cap
    assert p.leverage_red_warning_above == settings.risk_profiles.leverage_red_warning_above


def test_none_uses_default_profile(settings):
    assert get_profile(None, settings).name == settings.risk_profiles.default


def test_name_is_case_insensitive(settings):
    assert get_profile("  Agresivo ", settings).name == "agresivo"


def test_unknown_profile_lists_the_options(settings):
    with pytest.raises(ValueError) as exc:
        get_profile("kamikaze", settings)
    msg = str(exc.value)
    assert "kamikaze" in msg
    for name in ("conservador", "moderado", "agresivo"):
        assert name in msg


def test_profiles_without_override_have_no_red_warning(settings):
    # Ningún perfil de la config pasa de 25x.
    for name in profile_names(settings):
        p = get_profile(name, settings)
        assert p.red_warning is False
        assert p.red_warning_text is None
        assert p.warnings == ()


# --- override manual del tope -------------------------------------------------------


def test_override_up_to_50x_with_permanent_red_warning(settings):
    p = get_profile("agresivo", settings, max_leverage_override=50)
    assert p.max_leverage == 50
    assert p.profile_max_leverage == 25
    assert p.leverage_overridden is True
    assert p.red_warning is True
    assert "50x" in p.red_warning_text
    assert "25x" in p.red_warning_text
    assert p.warnings[0] == p.red_warning_text
    assert any("cambiado a mano" in w and "25x" in w for w in p.warnings)


def test_red_warning_only_strictly_above_25x(settings):
    assert get_profile("agresivo", settings, 25).red_warning is False
    assert get_profile("moderado", settings, 25).red_warning is False
    p = get_profile("moderado", settings, 25.5)
    assert p.red_warning is True
    assert "25,5x" in p.red_warning_text  # coma decimal


def test_override_above_hard_cap_is_rejected(settings):
    with pytest.raises(ValueError, match="No se permite más de 50x"):
        get_profile("agresivo", settings, 50.5)
    with pytest.raises(ValueError, match="pediste 100x"):
        get_profile("agresivo", settings, 100)


def test_hard_cap_comes_from_config(settings):
    custom = _with_profiles(settings, leverage_hard_cap=40, leverage_red_warning_above=20)
    with pytest.raises(ValueError, match="No se permite más de 40x"):
        get_profile("agresivo", custom, 45)
    p = get_profile("agresivo", custom)
    assert p.red_warning is True  # 25x > 20x
    assert "20x" in p.red_warning_text


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), True, "30", None])
def test_invalid_override_values(settings, bad):
    if bad is None:
        # None = sin override (no es un error).
        assert get_profile("agresivo", settings, bad).max_leverage == 25
        return
    with pytest.raises(ValueError):
        get_profile("agresivo", settings, bad)


def test_override_can_lower_the_cap(settings):
    p = get_profile("conservador", settings, 3)
    assert p.max_leverage == 3
    assert p.leverage_overridden is True
    assert p.red_warning is False
    assert p.warnings == (
        "Tope de apalancamiento cambiado a mano: 3x (el perfil conservador trae 5x).",
    )


def test_override_equal_to_profile_is_not_marked(settings):
    p = get_profile("agresivo", settings, 25)
    assert p.leverage_overridden is False
    assert p.warnings == ()


def test_red_warning_text_uses_spanish_numbers():
    text = red_warning_text(30.5, 25)
    assert "30,5x" in text and "25x" in text
    assert text.startswith("Advertencia")


def test_to_dict_is_json_safe(settings):
    d = get_profile("agresivo", settings, 30).to_dict()
    json.dumps(d, allow_nan=False)
    assert d["max_leverage"] == 30
    assert d["red_warning"] is True
    assert isinstance(d["warnings"], list) and len(d["warnings"]) == 2


# --- revisión final del Hito 4 ------------------------------------------------------


@pytest.mark.parametrize("bad", [0.5, 0.9999])
def test_override_below_1x_is_rejected(settings, bad):
    # Con menos de 1x el margen sería más grande que la posición: todo plan daba
    # "No operar" con sugerencias imposibles ("subí el margen").
    with pytest.raises(ValueError, match="tiene que ser 1x o más"):
        get_profile("agresivo", settings, bad)


def test_override_of_exactly_1x_is_accepted(settings):
    assert get_profile("agresivo", settings, 1).max_leverage == 1.0


def test_hard_cap_message_shows_enough_decimals(settings):
    # Antes: "No se permite más de 50x de apalancamiento (pediste 50x)".
    with pytest.raises(ValueError, match=r"pediste 50,04x"):
        get_profile("agresivo", settings, 50.04)
