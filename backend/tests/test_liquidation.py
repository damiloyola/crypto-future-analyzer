"""Hito 2: precio de liquidación (aislado y "estilo Nexo") y chequeo contra el stop."""

import math

import numpy as np
import pytest

from app.risk.liquidation import (
    LiquidationCheck,
    check_liquidation_vs_stop,
    liquidation_distance_pct,
    liquidation_price,
    liquidation_price_for_leverage,
    max_size_for_safe_liquidation,
    min_margin_for_safe_liquidation,
    mmr_from_config,
    wallet_liquidation_price,
)
from app.risk.sizing import size_position

MMR = 0.005  # 0,5 %
FEE = 0.0006


def _equity_and_maintenance(
    side: str, entry: float, size: float, collateral: float, price: float, mmr: float, f: float
) -> tuple[float, float]:
    """Patrimonio (colateral + pnl - comisión de salida) y margen de mantenimiento a ``price``."""
    q = size / entry
    pnl = q * (price - entry) if side == "long" else q * (entry - price)
    equity = collateral + pnl - f * q * price
    return equity, mmr * q * price


def _random_position(rng: np.random.Generator):
    side = "long" if rng.random() < 0.5 else "short"
    entry = float(10 ** rng.uniform(-1, 5))
    size = float(rng.uniform(10, 1e5))
    leverage = float(rng.uniform(1.05, 100))
    mmr = float(rng.uniform(0, 0.02))
    f = float(rng.uniform(0, 0.002))
    return side, entry, size, size / leverage, mmr, f


def _random_stop(rng: np.random.Generator, side: str, entry: float) -> float:
    d = float(rng.uniform(0.001, 0.08))
    return entry * (1 - d) if side == "long" else entry * (1 + d)


# --- config ----------------------------------------------------------------------


def test_mmr_from_repo_config(settings):
    assert mmr_from_config(settings.liquidation) == pytest.approx(0.005)


# --- fórmulas -------------------------------------------------------------------------


def test_hand_example_long_50x():
    # Entrada 60 000, tamaño 5000, margen 100 (50x), mmr 0,5 %, sin comisión.
    liq = liquidation_price("long", 60000, 5000, 100, MMR)
    assert liq == pytest.approx(60000 * 0.98 / 0.995)
    assert liq == pytest.approx(59095.4774, rel=1e-9)
    # Cerca de la aproximación del usuario E × (1 - 1/L + mmr) = 59 100.
    assert liq == pytest.approx(59100, rel=1e-4)


def test_hand_example_short_50x():
    liq = liquidation_price("short", 60000, 5000, 100, MMR)
    assert liq == pytest.approx(60000 * 1.02 / 1.005)
    assert liq == pytest.approx(60895.5224, rel=1e-9)
    # Aproximación E × (1 + 1/L - mmr) = 60 900.
    assert liq == pytest.approx(60900, rel=1e-4)


def test_hand_example_with_exit_fee():
    assert liquidation_price("long", 60000, 5000, 100, MMR, FEE) == pytest.approx(
        60000 * 0.98 / (1 - MMR - FEE)
    )
    assert liquidation_price("short", 60000, 5000, 100, MMR, FEE) == pytest.approx(
        60000 * 1.02 / (1 + MMR + FEE)
    )
    # La comisión de salida acerca la liquidación a la entrada.
    assert liquidation_price("long", 60000, 5000, 100, MMR, FEE) > liquidation_price(
        "long", 60000, 5000, 100, MMR
    )
    assert liquidation_price("short", 60000, 5000, 100, MMR, FEE) < liquidation_price(
        "short", 60000, 5000, 100, MMR
    )


def test_difference_with_user_approximation_identity():
    # exacta - aproximada = E·mmr·(mmr - 1/L)/(1 - mmr) para el long sin comisión.
    for lev in (2, 5, 10, 25, 50, 100):
        exact = liquidation_price_for_leverage("long", 60000, lev, MMR)
        approx = 60000 * (1 - 1 / lev + MMR)
        assert exact - approx == pytest.approx(60000 * MMR * (MMR - 1 / lev) / (1 - MMR))


def test_equity_equals_maintenance_at_liquidation_random():
    rng = np.random.default_rng(21)
    for _ in range(1000):
        side, entry, size, margin, mmr, f = _random_position(rng)
        liq = liquidation_price(side, entry, size, margin, mmr, f)
        assert liq is not None
        equity, maintenance = _equity_and_maintenance(side, entry, size, margin, liq, mmr, f)
        assert math.isclose(equity, maintenance, rel_tol=1e-9, abs_tol=1e-9)
        # Del lado perdedor de la entrada, salvo que 1/L no alcance a cubrir
        # mantenimiento + comisión (la posición nacería liquidada).
        if margin / size > mmr + f:
            assert liq < entry if side == "long" else liq > entry


def test_position_born_liquidated_when_margin_below_maintenance():
    # 1/L = 0,4 % < mmr 0,5 %: la liquidación del long queda por encima de la entrada.
    liq = liquidation_price_for_leverage("long", 60000, 250, MMR)
    assert liq > 60000
    check = check_liquidation_vs_stop("long", 60000, 59400, liq, 2.0)
    assert check.liq_before_stop is True
    assert check.ok is False


def test_wallet_model_equity_equals_maintenance_random():
    rng = np.random.default_rng(22)
    for _ in range(500):
        side, entry, size, margin, mmr, f = _random_position(rng)
        wallet = margin * float(rng.uniform(1, 5))
        liq = wallet_liquidation_price(side, entry, size, wallet, mmr, f)
        if liq is None:
            assert side == "long" and wallet >= size
            continue
        equity, maintenance = _equity_and_maintenance(side, entry, size, wallet, liq, mmr, f)
        assert math.isclose(equity, maintenance, rel_tol=1e-9, abs_tol=1e-9)
        isolated = liquidation_price(side, entry, size, margin, mmr, f)
        # Con más colateral la liquidación queda igual o más lejos.
        if side == "long":
            assert liq <= isolated * (1 + 1e-12)
        else:
            assert liq >= isolated * (1 - 1e-12)


def test_wallet_model_equals_isolated_when_wallet_is_margin():
    for side in ("long", "short"):
        assert wallet_liquidation_price(side, 60000, 5000, 100, MMR, FEE) == pytest.approx(
            liquidation_price(side, 60000, 5000, 100, MMR, FEE)
        )


def test_wallet_model_hand_example():
    # Billetera de 300 USD respaldando 5000 USD: liquidación bastante más lejos.
    liq = wallet_liquidation_price("long", 60000, 5000, 300, MMR)
    assert liq == pytest.approx(60000 * (1 - 0.06) / (1 - MMR))
    assert liq < liquidation_price("long", 60000, 5000, 100, MMR)


def test_long_without_liquidation_returns_none():
    assert liquidation_price("long", 60000, 5000, 5000, MMR) is None  # 1x
    assert liquidation_price("long", 60000, 5000, 6000, MMR, FEE) is None
    assert liquidation_price_for_leverage("long", 60000, 1, MMR) is None
    assert liquidation_price_for_leverage("long", 60000, 0.5, MMR) is None
    assert wallet_liquidation_price("long", 60000, 5000, 10000, MMR) is None
    # El short siempre tiene liquidación (aunque esté lejísimos).
    assert liquidation_price("short", 60000, 5000, 5000, MMR) == pytest.approx(
        60000 * 2 / (1 + MMR)
    )


def test_leverage_variant_matches_size_margin_variant_random():
    rng = np.random.default_rng(23)
    for _ in range(300):
        side, entry, size, margin, mmr, f = _random_position(rng)
        a = liquidation_price(side, entry, size, margin, mmr, f)
        b = liquidation_price_for_leverage(side, entry, size / margin, mmr, f)
        assert a == pytest.approx(b, rel=1e-12)


def test_more_leverage_brings_liquidation_closer():
    previous = {"long": 0.0, "short": math.inf}
    for lev in (2, 5, 10, 25, 50, 100):
        long_liq = liquidation_price_for_leverage("long", 60000, lev, MMR, FEE)
        short_liq = liquidation_price_for_leverage("short", 60000, lev, MMR, FEE)
        assert long_liq > previous["long"]
        assert short_liq < previous["short"]
        previous = {"long": long_liq, "short": short_liq}


def test_liquidation_distance_pct():
    liq = liquidation_price("long", 60000, 5000, 100, MMR)
    assert liquidation_distance_pct(60000, liq) == pytest.approx((60000 - liq) / 60000)
    assert liquidation_distance_pct(60000, liq) == pytest.approx(0.01507538, rel=1e-6)
    assert liquidation_distance_pct(100, 110) == pytest.approx(0.1)
    assert liquidation_distance_pct(60000, None) is None
    with pytest.raises(ValueError):
        liquidation_distance_pct(60000, 0)
    with pytest.raises(ValueError):
        liquidation_distance_pct(0, 100)


# --- chequeo contra el stop ------------------------------------------------------------


def test_check_50x_with_1pct_stop_is_too_close():
    liq = liquidation_price("long", 60000, 5000, 100, MMR)
    check = check_liquidation_vs_stop("long", 60000, 59400, liq, 2.0)
    assert isinstance(check, LiquidationCheck)
    assert check.stop_distance == pytest.approx(600)
    assert check.liquidation_distance == pytest.approx(60000 - liq)
    assert check.ratio == pytest.approx((60000 - liq) / 600)
    assert 1.5 < check.ratio < 1.51
    assert check.liq_before_stop is False
    assert check.ok is False


def test_check_liquidation_before_stop():
    # 50x con stop al 2 %: la liquidación (≈1,5 %) llega antes que el stop.
    liq = liquidation_price("long", 60000, 5000, 100, MMR)
    check = check_liquidation_vs_stop("long", 60000, 58800, liq, 2.0)
    assert check.liq_before_stop is True
    assert check.ratio < 1
    assert check.ok is False
    liq_s = liquidation_price("short", 60000, 5000, 100, MMR)
    check_s = check_liquidation_vs_stop("short", 60000, 61200, liq_s, 2.0)
    assert check_s.liq_before_stop is True
    assert check_s.ok is False


def test_check_10x_is_ok():
    liq = liquidation_price("long", 60000, 5000, 500, MMR, FEE)
    check = check_liquidation_vs_stop("long", 60000, 59400, liq, 2.0)
    assert check.ok is True
    assert check.ratio > 9
    liq_s = liquidation_price("short", 60000, 5000, 500, MMR, FEE)
    check_s = check_liquidation_vs_stop("short", 60000, 60600, liq_s, 2.0)
    assert check_s.ok is True
    assert check_s.liq_before_stop is False


def test_check_without_liquidation_passes():
    check = check_liquidation_vs_stop("long", 60000, 59400, None, 2.0)
    assert check.ok is True
    assert check.ratio is None
    assert check.liquidation_distance is None
    assert check.liq_before_stop is False


def test_check_liquidation_on_winning_side_fails():
    # Liquidación por encima de la entrada en un long: nacería liquidada.
    check = check_liquidation_vs_stop("long", 60000, 59400, 60100, 2.0)
    assert check.liquidation_distance == pytest.approx(-100)
    assert check.ratio < 0
    assert check.liq_before_stop is True
    assert check.ok is False
    check_s = check_liquidation_vs_stop("short", 60000, 60600, 59900, 2.0)
    assert check_s.ratio < 0
    assert check_s.ok is False


def test_check_liquidation_exactly_at_stop_is_before():
    check = check_liquidation_vs_stop("long", 60000, 59400, 59400, 2.0)
    assert check.liq_before_stop is True
    assert check.ok is False


def test_check_respects_min_ratio_parameter():
    # Ratio ≈ 2,5: pasa con 2, no pasa con 3.
    assert check_liquidation_vs_stop("long", 60000, 59400, 58500, 2.0).ok is True
    assert check_liquidation_vs_stop("long", 60000, 59400, 58500, 3.0).ok is False
    assert check_liquidation_vs_stop("short", 60000, 60600, 61500, 2.0).ok is True
    assert check_liquidation_vs_stop("short", 60000, 60600, 61500, 3.0).ok is False


def test_check_invalid_inputs():
    with pytest.raises(ValueError):
        check_liquidation_vs_stop("long", 60000, 60600, 58000, 2.0)  # stop del lado ganador
    with pytest.raises(ValueError):
        check_liquidation_vs_stop("long", 60000, 59400, -1, 2.0)
    with pytest.raises(ValueError, match="mayor que 1"):
        check_liquidation_vs_stop("long", 60000, 59400, 58000, 1.0)
    with pytest.raises(ValueError):
        check_liquidation_vs_stop("long", 60000, 59400, 58000, 0)
    with pytest.raises(ValueError):
        check_liquidation_vs_stop("up", 60000, 59400, 58000, 2.0)


# --- sugerencias: subir margen / achicar tamaño ----------------------------------------


def test_min_margin_hand_example():
    # x = 2 × 600 / 60000 = 0.02; k = 1 - 0.98 × (1 - 0.005 - 0.0006) = 0.025488.
    margin = min_margin_for_safe_liquidation("long", 60000, 59400, 5000, MMR, FEE, 2.0)
    assert margin == pytest.approx(5000 * 0.025488)
    assert margin == pytest.approx(127.44)
    liq = liquidation_price("long", 60000, 5000, margin, MMR, FEE)
    assert liq == pytest.approx(58800)  # justo a 2 × la distancia al stop


def test_min_margin_hand_example_short():
    # k = 1.02 × 1.0056 - 1 = 0.025712.
    margin = min_margin_for_safe_liquidation("short", 60000, 60600, 5000, MMR, FEE, 2.0)
    assert margin == pytest.approx(5000 * 0.025712)
    assert liquidation_price("short", 60000, 5000, margin, MMR, FEE) == pytest.approx(61200)


def test_min_margin_makes_check_pass_and_less_fails_random():
    rng = np.random.default_rng(24)
    for _ in range(1000):
        side, entry, size, _margin, mmr, f = _random_position(rng)
        stop = _random_stop(rng, side, entry)
        ratio = float(rng.uniform(1.1, 5))
        m = min_margin_for_safe_liquidation(side, entry, stop, size, mmr, f, ratio)
        liq = liquidation_price(side, entry, size, m, mmr, f)
        assert check_liquidation_vs_stop(side, entry, stop, liq, ratio).ok, (side, entry, stop)
        liq_less = liquidation_price(side, entry, size, m * (1 - 1e-6), mmr, f)
        assert not check_liquidation_vs_stop(side, entry, stop, liq_less, ratio).ok
        if liq is not None:
            # Cae justo en el borde: distancia a liquidación == ratio × distancia al stop.
            assert abs(entry - liq) == pytest.approx(ratio * abs(entry - stop), rel=1e-9)


def test_max_size_makes_check_pass_and_more_fails_random():
    rng = np.random.default_rng(25)
    for _ in range(1000):
        side, entry, _size, margin, mmr, f = _random_position(rng)
        stop = _random_stop(rng, side, entry)
        ratio = float(rng.uniform(1.1, 5))
        s = max_size_for_safe_liquidation(side, entry, stop, margin, mmr, f, ratio)
        liq = liquidation_price(side, entry, s, margin, mmr, f)
        assert check_liquidation_vs_stop(side, entry, stop, liq, ratio).ok
        liq_more = liquidation_price(side, entry, s * (1 + 1e-6), margin, mmr, f)
        assert not check_liquidation_vs_stop(side, entry, stop, liq_more, ratio).ok
        # Las dos sugerencias son inversas entre sí.
        assert min_margin_for_safe_liquidation(
            side, entry, stop, s, mmr, f, ratio
        ) == pytest.approx(margin, rel=1e-9)


def test_long_with_very_wide_ratio_needs_no_liquidation():
    # Stop al 50 % con ratio 2: la liquidación tendría que estar en precio <= 0,
    # así que hace falta margen >= tamaño (1x).
    m = min_margin_for_safe_liquidation("long", 100, 50, 1000, MMR, FEE, 2.0)
    assert m == pytest.approx(1000)
    assert liquidation_price("long", 100, 1000, m, MMR, FEE) is None
    assert check_liquidation_vs_stop("long", 100, 50, None, 2.0).ok
    assert max_size_for_safe_liquidation("long", 100, 50, 300, MMR, FEE, 2.0) == pytest.approx(300)


def test_raise_margin_keeping_risk():
    """Flujo "subir el margen manteniendo el riesgo": mismo tamaño, menos apalancamiento."""
    s = size_position(60000, 59400, "long", 50, 100, FEE, FEE, True)
    liq = liquidation_price("long", 60000, s.size_usd, s.margin_usd, MMR, FEE)
    assert not check_liquidation_vs_stop("long", 60000, 59400, liq, 2.0).ok
    new_margin = min_margin_for_safe_liquidation("long", 60000, 59400, s.size_usd, MMR, FEE, 2.0)
    assert new_margin > s.margin_usd
    s2 = size_position(60000, 59400, "long", 50, new_margin, FEE, FEE, True)
    assert s2.size_usd == pytest.approx(s.size_usd)
    assert s2.loss_at_stop_usd == pytest.approx(50)
    assert s2.leverage < s.leverage
    liq2 = liquidation_price("long", 60000, s2.size_usd, s2.margin_usd, MMR, FEE)
    assert check_liquidation_vs_stop("long", 60000, 59400, liq2, 2.0).ok


def test_shrink_size_keeping_margin():
    """Flujo "achicar el tamaño": el margen no cambia y el riesgo en USD baja."""
    s = size_position(60000, 59400, "long", 50, 100, FEE, FEE, True)
    max_size = max_size_for_safe_liquidation("long", 60000, 59400, 100, MMR, FEE, 2.0)
    assert max_size < s.size_usd
    liq = liquidation_price("long", 60000, max_size, 100, MMR, FEE)
    assert check_liquidation_vs_stop("long", 60000, 59400, liq, 2.0).ok


# --- validaciones -----------------------------------------------------------------------


def test_liquidation_invalid_inputs():
    with pytest.raises(ValueError):
        liquidation_price("long", 0, 5000, 100, MMR)
    with pytest.raises(ValueError):
        liquidation_price("long", 60000, 0, 100, MMR)
    with pytest.raises(ValueError):
        liquidation_price("long", 60000, 5000, 0, MMR)
    with pytest.raises(ValueError):
        liquidation_price("long", 60000, 5000, 100, -0.001)
    with pytest.raises(ValueError):
        liquidation_price("long", 60000, 5000, 100, 0.5, 0.5)  # mmr + f >= 1
    with pytest.raises(ValueError):
        liquidation_price("long", 60000, 5000, 100, 0.5 * 100)  # porcentaje en vez de fracción
    with pytest.raises(ValueError):
        liquidation_price("middle", 60000, 5000, 100, MMR)
    with pytest.raises(ValueError):
        liquidation_price_for_leverage("long", 60000, 0, MMR)
    with pytest.raises(ValueError):
        liquidation_price_for_leverage("short", 60000, -5, MMR)
    with pytest.raises(ValueError):
        wallet_liquidation_price("long", 60000, 5000, 0, MMR)


def test_safe_liquidation_helpers_invalid_inputs():
    with pytest.raises(ValueError):
        min_margin_for_safe_liquidation("long", 60000, 60600, 5000, MMR, FEE, 2.0)
    with pytest.raises(ValueError):
        min_margin_for_safe_liquidation("short", 60000, 59400, 5000, MMR, FEE, 2.0)
    with pytest.raises(ValueError):
        min_margin_for_safe_liquidation("long", 60000, 59400, 0, MMR, FEE, 2.0)
    with pytest.raises(ValueError, match="mayor que 1"):
        min_margin_for_safe_liquidation("long", 60000, 59400, 5000, MMR, FEE, 0.8)
    with pytest.raises(ValueError):
        max_size_for_safe_liquidation("long", 60000, 59400, 0, MMR, FEE, 2.0)
    with pytest.raises(ValueError):
        max_size_for_safe_liquidation("long", 60000, 60000, 100, MMR, FEE, 2.0)
