"""Stop por estructura: niveles pegados al precio se saltean (decisión del usuario)."""

from datetime import UTC, datetime

import pytest

from app.indicators.levels import LevelCluster, SRResult
from app.risk.stops import StopNotFound, StructuralStop, structural_stop

T0 = datetime(2026, 10, 12, tzinfo=UTC)


def _cluster(price: float) -> LevelCluster:
    return LevelCluster(
        price=price,
        touches=2,
        sources=frozenset({"pivot_low"}),
        last_touch=T0,
        zone_low=price,
        zone_high=price,
    )


def _sr(*prices: float, reference: float = 100.0) -> SRResult:
    return SRResult(
        supports=(),
        resistances=(),
        prev_day_high=None,
        prev_day_low=None,
        pivots_used=(),
        prev_day=None,
        reference_price=reference,
        cluster_tolerance=0.0,
        clusters=tuple(_cluster(p) for p in sorted(prices)),
    )


def test_long_skips_levels_closer_than_one_atr():
    # ATR 1.0: el soporte en 99.6 está a 0,4 ATR -> se saltea; se usa 98.5.
    levels = {"15m": _sr(99.6, 98.5, 97.0, 101.0), "1h": None}
    got = structural_stop("long", 100.0, levels, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=1.0)
    assert isinstance(got, StructuralStop)
    assert got.level_price == 98.5 and got.level_timeframe == "15m"
    assert got.stop == pytest.approx(98.0)
    assert [s.price for s in got.skipped] == [99.6]
    assert got.skipped[0].distance_atr == pytest.approx(0.4)
    assert got.stop_distance_pct == pytest.approx(0.02)
    assert got.level_kind == "soporte"


def test_short_is_mirrored_and_picks_nearest_across_timeframes():
    levels = {"15m": _sr(103.0, 99.0), "1h": _sr(101.5, 100.5)}
    got = structural_stop(
        "short", 100.0, levels, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=1.0
    )
    assert isinstance(got, StructuralStop)
    assert (got.level_price, got.level_timeframe) == (101.5, "1h")  # 100.5 está a 0,5 ATR
    assert got.stop == pytest.approx(102.0)
    assert got.level_kind == "resistencia"


def test_same_price_prefers_first_timeframe():
    levels = {"15m": _sr(98.0), "1h": _sr(98.0)}
    got = structural_stop("long", 100.0, levels, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=1.0)
    assert isinstance(got, StructuralStop) and got.level_timeframe == "15m"


def test_min_distance_zero_uses_the_nearest_level():
    levels = {"15m": _sr(99.9, 98.0)}
    got = structural_stop("long", 100.0, levels, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=0.0)
    assert isinstance(got, StructuralStop) and got.level_price == 99.9 and not got.skipped


def test_level_exactly_at_threshold_is_kept():
    levels = {"15m": _sr(99.0)}
    got = structural_stop("long", 100.0, levels, atr=1.0, buffer_atr_mult=0.0, min_distance_atr=1.0)
    assert isinstance(got, StructuralStop) and got.stop == pytest.approx(99.0)


def test_not_found_when_everything_is_too_close():
    levels = {"15m": _sr(99.7, 99.5), "1h": _sr(99.8)}
    got = structural_stop("long", 100.0, levels, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=1.0)
    assert isinstance(got, StopNotFound)
    assert "menos de 1 ATR" in got.reason
    assert [s.price for s in got.skipped] == [99.8, 99.7, 99.5]


def test_not_found_without_levels_on_that_side():
    got = structural_stop(
        "short", 100.0, {"15m": _sr(95.0)}, atr=1.0, buffer_atr_mult=0.5, min_distance_atr=1.0
    )
    assert isinstance(got, StopNotFound) and "No hay resistencias" in got.reason


def test_stop_never_crosses_zero():
    got = structural_stop(
        "long", 1.0, {"15m": _sr(0.2)}, atr=0.5, buffer_atr_mult=1.0, min_distance_atr=1.0
    )
    assert isinstance(got, StopNotFound)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entry": 0},
        {"atr": 0},
        {"atr": float("nan")},
        {"buffer_atr_mult": -1},
        {"min_distance_atr": -0.1},
        {"side": "sideways"},
    ],
)
def test_invalid_inputs(kwargs):
    args = {
        "side": "long",
        "entry": 100.0,
        "levels_by_timeframe": {"15m": _sr(98.0)},
        "atr": 1.0,
        "buffer_atr_mult": 0.5,
        "min_distance_atr": 1.0,
    } | kwargs
    with pytest.raises(ValueError):
        structural_stop(**args)


def test_support_resistance_exposes_all_clusters():
    """El stop necesita todos los niveles, no solo los 3 más cercanos."""
    import numpy as np
    import pandas as pd

    from app.indicators.levels import support_resistance

    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 600))
    idx = pd.date_range("2026-01-01", periods=600, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0},
        index=idx,
    )
    sr = support_resistance(df, float(close[-1]), 0.5, 3, 3, 0.5, 3)
    assert len(sr.clusters) >= len(sr.supports) + len(sr.resistances)
    prices = [c.price for c in sr.clusters]
    assert prices == sorted(prices)
    for lv in (*sr.supports, *sr.resistances):
        assert any(c.price == lv.price for c in sr.clusters)
