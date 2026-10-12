"""Tests de soportes y resistencias (sin red, todo con datos armados acá)."""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from app.indicators.levels import (
    SOURCE_PIVOT_HIGH,
    SOURCE_PIVOT_LOW,
    SOURCE_PREV_DAY_HIGH,
    SOURCE_PREV_DAY_LOW,
    Level,
    LevelCluster,
    LevelPoint,
    Pivot,
    PreviousDayRange,
    SRResult,
    cluster_levels,
    cluster_tolerance,
    find_pivots,
    nearest_levels,
    pivots_known_at,
    previous_day_high_low,
    support_resistance,
)

NAN = float("nan")
SEEDS = [1, 7, 42]


# --- datos sintéticos --------------------------------------------------------


def candles(
    highs: list[float],
    lows: list[float] | None = None,
    start: str = "2024-01-01 00:00",
    freq: str = "1h",
) -> pd.DataFrame:
    """Velas con los máximos (y mínimos) dados; por defecto low = high − 1."""
    high = np.asarray(highs, dtype="float64")
    low = high - 1.0 if lows is None else np.asarray(lows, dtype="float64")
    mid = (high + low) / 2.0
    index = pd.date_range(start, periods=len(high), freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": mid,
            "high": high,
            "low": low,
            "close": mid,
            "volume": np.ones(len(high)),
            "closed": np.ones(len(high), dtype=bool),
        },
        index=index,
    )


def random_walk_candles(seed: int, n: int = 600, freq: str = "15min") -> pd.DataFrame:
    """Caminata aleatoria OHLCV que cruza varios días UTC."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    open_ = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.004, n)))
    index = pd.date_range("2024-03-01 20:00", periods=n, freq=freq, tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.lognormal(3.0, 0.6, n),
            "closed": np.ones(n, dtype=bool),
        },
        index=index,
    )


def simple_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Media simple del rango de las últimas velas (suficiente para estos tests)."""
    return float((df["high"] - df["low"]).iloc[-period:].mean())


def naive_pivots(df: pd.DataFrame, left: int, right: int) -> list[tuple[int, str, float]]:
    """Versión con bucles, escrita sin reutilizar nada del módulo."""
    out: list[tuple[int, str, float]] = []
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    for t in range(left, len(df) - right):
        before_h = highs[t - left : t]
        after_h = highs[t + 1 : t + right + 1]
        if all(highs[t] > h for h in before_h) and all(highs[t] >= h for h in after_h):
            out.append((t, "high", highs[t]))
        before_l = lows[t - left : t]
        after_l = lows[t + 1 : t + right + 1]
        if all(lows[t] < x for x in before_l) and all(lows[t] <= x for x in after_l):
            out.append((t, "low", lows[t]))
    return out


def as_tuples(pivots: list[Pivot]) -> list[tuple[int, str, float]]:
    return [(p.index, p.kind, p.price) for p in pivots]


def point(price: float, source: str = SOURCE_PIVOT_HIGH, hour: int | None = None) -> LevelPoint:
    time = None if hour is None else pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=hour)
    return LevelPoint(price=price, source=source, time=time)


def cluster_at(price: float, touches: int = 1) -> LevelCluster:
    return LevelCluster(
        price=price,
        touches=touches,
        sources=frozenset({SOURCE_PIVOT_HIGH}),
        last_touch=None,
        zone_low=price,
        zone_high=price,
    )


# --- pivotes -----------------------------------------------------------------


class TestFindPivots:
    def test_single_peak_and_trough(self) -> None:
        # Pico en la vela 3 y valle en la 7, con 3 velas de cada lado.
        highs = [1, 2, 3, 9, 3, 2, 1, 0.5, 1, 2, 3]
        lows = [h - 0.5 for h in highs]
        df = candles(highs, lows)
        pivots = find_pivots(df, left=3, right=3)
        assert as_tuples(pivots) == [(3, "high", 9.0), (7, "low", 0.0)]
        peak, trough = pivots
        assert peak.time == df.index[3]
        assert peak.confirmed_at == df.index[6]
        assert trough.time == df.index[7]
        assert trough.confirmed_at == df.index[10]

    def test_edges_without_enough_bars_are_not_pivots(self) -> None:
        # Máximo absoluto en la vela 1 (le faltan velas a la izquierda) y otro
        # en la anteúltima (le faltan a la derecha): ninguno es pivote.
        highs = [5, 20, 5, 4, 3, 4, 5, 6, 7, 30, 6]
        df = candles(highs)
        assert [p for p in find_pivots(df, 3, 3) if p.kind == "high"] == []
        # Con right=1 la anteúltima ya tiene la vela que necesita.
        assert [(p.index, p.price) for p in find_pivots(df, 3, 1) if p.kind == "high"] == [
            (9, 30.0)
        ]

    def test_too_few_bars_returns_empty(self) -> None:
        assert find_pivots(candles([1, 2, 3, 9, 3, 2]), 3, 3) == []
        assert find_pivots(candles([]), 3, 3) == []

    def test_plateau_pivot_is_first_bar(self) -> None:
        highs = [1, 2, 3, 5, 5, 3, 2, 1]
        lows = [10 - h for h in highs]  # piso plato espejado: [9, 8, 7, 5, 5, 7, 8, 9]
        pivots = find_pivots(candles(highs, lows), 3, 3)
        assert as_tuples(pivots) == [(3, "high", 5.0), (3, "low", 5.0)]

    def test_long_plateau_still_one_pivot(self) -> None:
        highs = [1, 2, 3, 5, 5, 5, 5, 5, 4, 3, 2, 1]
        highs_pivots = [p for p in find_pivots(candles(highs), 3, 3) if p.kind == "high"]
        assert as_tuples(highs_pivots) == [(3, "high", 5.0)]

    def test_plateau_followed_by_higher_bar(self) -> None:
        # El techo en 3-4 no es pivote porque la vela 5 lo supera; la 5 sí.
        highs = [1, 2, 3, 5, 5, 6, 2, 1, 0.5]
        highs_pivots = [p for p in find_pivots(candles(highs), 3, 3) if p.kind == "high"]
        assert as_tuples(highs_pivots) == [(5, "high", 6.0)]

    def test_tie_on_the_left_blocks_pivot(self) -> None:
        # La vela 4 iguala a la 1 (dentro de las 3 anteriores): no es estrictamente mayor.
        highs = [1, 5, 2, 3, 5, 3, 2, 1]
        assert [p for p in find_pivots(candles(highs), 3, 3) if p.kind == "high"] == []

    def test_outside_bar_is_high_and_low(self) -> None:
        highs = [5, 5.1, 5.2, 9, 5.2, 5.1, 5]
        lows = [4, 3.9, 3.8, 1, 3.8, 3.9, 4]
        pivots = find_pivots(candles(highs, lows), 3, 3)
        assert [(p.index, p.kind) for p in pivots] == [(3, "high"), (3, "low")]

    def test_nan_in_window_blocks_pivot(self) -> None:
        highs = [1, NAN, 3, 9, 3, 2, 1]
        assert [p for p in find_pivots(candles(highs), 3, 3) if p.kind == "high"] == []
        highs = [1, 2, 3, NAN, 3, 2, 1]
        assert [p for p in find_pivots(candles(highs), 3, 3) if p.kind == "high"] == []

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(("left", "right"), [(3, 3), (2, 5), (5, 1)])
    def test_matches_naive_loop(self, seed: int, left: int, right: int) -> None:
        df = random_walk_candles(seed, n=400)
        # Redondeo a 1 decimal para que aparezcan empates y techos planos.
        df["high"] = df["high"].round(1)
        df["low"] = df["low"].round(1)
        assert as_tuples(find_pivots(df, left, right)) == naive_pivots(df, left, right)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "3", None])
    def test_invalid_window(self, bad: object) -> None:
        df = candles([1, 2, 3, 9, 3, 2, 1])
        with pytest.raises(ValueError):
            find_pivots(df, left=bad, right=3)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            find_pivots(df, left=3, right=bad)  # type: ignore[arg-type]

    def test_missing_columns(self) -> None:
        df = candles([1, 2, 3]).drop(columns=["low"])
        with pytest.raises(ValueError, match="low"):
            find_pivots(df, 1, 1)

    def test_input_not_modified(self) -> None:
        df = random_walk_candles(3, n=100)
        copy = df.copy()
        find_pivots(df, 3, 3)
        pd.testing.assert_frame_equal(df, copy)


# --- cero look-ahead ---------------------------------------------------------


class TestNoLookAhead:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(("left", "right"), [(3, 3), (2, 4)])
    def test_prefix_pivots_are_the_confirmed_ones(self, seed: int, left: int, right: int) -> None:
        df = random_walk_candles(seed, n=300)
        full = find_pivots(df, left, right)
        for k in range(1, len(df) + 1, 7):
            prefix = find_pivots(df.iloc[:k], left, right)
            expected = [p for p in full if p.index + right < k]
            assert prefix == expected
            assert pivots_known_at(full, df.index[k - 1]) == expected

    @pytest.mark.parametrize("seed", SEEDS)
    def test_future_bars_never_change_confirmed_pivots(self, seed: int) -> None:
        df = random_walk_candles(seed, n=250)
        confirmed = find_pivots(df, 3, 3)
        rng = np.random.default_rng(seed + 100)
        # Velas futuras extremas: un máximo enorme y un mínimo minúsculo pegados al final.
        future = random_walk_candles(seed + 1, n=40)
        future.index = pd.date_range(
            df.index[-1] + pd.Timedelta(minutes=15), periods=40, freq="15min", tz="UTC"
        )
        future.iloc[0, future.columns.get_loc("high")] = 1e6
        future.iloc[1, future.columns.get_loc("low")] = 1e-6
        future["high"] *= rng.uniform(0.5, 2.0, len(future))
        extended = pd.concat([df, future])
        new = find_pivots(extended, 3, 3)
        assert new[: len(confirmed)] == confirmed

    def test_pivot_needs_right_bars_to_appear(self) -> None:
        highs = [1, 2, 3, 9, 3, 2, 1]
        df = candles(highs)
        assert find_pivots(df.iloc[:6], 3, 3) == []  # falta la tercera vela de la derecha
        (pivot,) = [p for p in find_pivots(df, 3, 3) if p.kind == "high"]
        assert pivots_known_at([pivot], df.index[5]) == []
        assert pivots_known_at([pivot], df.index[6]) == [pivot]


# --- agrupación --------------------------------------------------------------


class TestClusterLevels:
    def test_mean_not_chain(self) -> None:
        # Tolerancia 1: cada precio está a 0,9 del anterior (una cadena los
        # juntaría a todos), pero 101,8 queda a 1,35 de la media 100,45.
        clusters = cluster_levels([point(100.0), point(100.9), point(101.8)], atr=2.0, mult=0.5)
        assert [c.price for c in clusters] == pytest.approx([100.45, 101.8])
        assert [c.touches for c in clusters] == [2, 1]
        assert (clusters[0].zone_low, clusters[0].zone_high) == (100.0, 100.9)

    def test_tolerance_is_inclusive(self) -> None:
        clusters = cluster_levels([point(10.0), point(11.0)], atr=2.0, mult=0.5)
        assert len(clusters) == 1
        assert clusters[0].price == 10.5

    def test_tolerance_scales_with_atr_and_mult(self) -> None:
        pts = [point(100.0), point(100.8)]
        assert len(cluster_levels(pts, atr=2.0, mult=0.5)) == 1
        assert len(cluster_levels(pts, atr=1.0, mult=0.5)) == 2
        assert len(cluster_levels(pts, atr=1.0, mult=1.0)) == 1

    def test_merges_sources_and_last_touch(self) -> None:
        pts = [
            point(100.0, SOURCE_PIVOT_HIGH, hour=1),
            point(100.2, SOURCE_PIVOT_LOW, hour=9),
            point(100.1, SOURCE_PREV_DAY_HIGH, hour=5),
            point(150.0, SOURCE_PIVOT_LOW, hour=3),
        ]
        low_cluster, high_cluster = cluster_levels(pts, atr=1.0, mult=0.5)
        assert low_cluster.price == pytest.approx(100.1)
        assert low_cluster.touches == 3
        assert low_cluster.sources == {SOURCE_PIVOT_HIGH, SOURCE_PIVOT_LOW, SOURCE_PREV_DAY_HIGH}
        assert low_cluster.last_touch == pd.Timestamp("2024-01-01 09:00", tz="UTC")
        assert high_cluster.sources == {SOURCE_PIVOT_LOW}
        assert high_cluster.touches == 1

    def test_same_candle_counts_once(self) -> None:
        # El máximo del día anterior es la misma vela que un pivote: un toque, dos fuentes.
        pts = [point(100.0, SOURCE_PIVOT_HIGH, hour=4), point(100.0, SOURCE_PREV_DAY_HIGH, hour=4)]
        (cluster,) = cluster_levels(pts, atr=1.0, mult=0.5)
        assert cluster.touches == 1
        assert cluster.sources == {SOURCE_PIVOT_HIGH, SOURCE_PREV_DAY_HIGH}

    def test_points_without_time_count_individually(self) -> None:
        (cluster,) = cluster_levels([point(100.0), point(100.0)], atr=1.0, mult=0.5)
        assert cluster.touches == 2
        assert cluster.last_touch is None

    @pytest.mark.parametrize("bad_atr", [NAN, 0.0, -1.0, None, math.inf])
    def test_unusable_atr_only_merges_exact_duplicates(self, bad_atr: float | None) -> None:
        pts = [point(100.0), point(100.0), point(100.0001), point(99.9999)]
        clusters = cluster_levels(pts, atr=bad_atr, mult=0.5)
        assert [(c.price, c.touches) for c in clusters] == [
            (99.9999, 1),
            (100.0, 2),
            (100.0001, 1),
        ]

    def test_exact_duplicates_survive_float_rounding(self) -> None:
        # 0,1 × 3 / 3 no da exactamente 0,1 en float: igual tienen que juntarse.
        clusters = cluster_levels([point(0.1)] * 7, atr=NAN)
        assert len(clusters) == 1
        assert clusters[0].price == 0.1
        assert clusters[0].touches == 7

    def test_order_independent(self) -> None:
        rnd = random.Random(5)
        pts = [point(100 + rnd.uniform(-5, 5), hour=i) for i in range(60)]
        base = cluster_levels(pts, atr=1.3, mult=0.5)
        for _ in range(5):
            shuffled = pts[:]
            rnd.shuffle(shuffled)
            assert cluster_levels(shuffled, atr=1.3, mult=0.5) == base

    def test_consecutive_levels_separated_by_more_than_tolerance(self) -> None:
        rnd = random.Random(11)
        pts = [point(rnd.uniform(90, 110)) for _ in range(300)]
        tolerance = cluster_tolerance(2.0, 0.5)
        clusters = cluster_levels(pts, atr=2.0, mult=0.5)
        prices = [c.price for c in clusters]
        assert prices == sorted(prices)
        assert all(b - a > tolerance for a, b in zip(prices, prices[1:], strict=False))
        assert sum(c.touches for c in clusters) == len(pts)
        assert all(c.zone_low <= c.price <= c.zone_high for c in clusters)

    def test_skips_non_finite_and_empty(self) -> None:
        assert cluster_levels([], atr=1.0) == []
        clusters = cluster_levels([point(NAN), point(math.inf), point(5.0)], atr=1.0)
        assert [c.price for c in clusters] == [5.0]

    @pytest.mark.parametrize("bad_mult", [-0.1, NAN, math.inf, None, True])
    def test_invalid_mult(self, bad_mult: object) -> None:
        with pytest.raises(ValueError):
            cluster_levels([point(1.0)], atr=1.0, mult=bad_mult)  # type: ignore[arg-type]

    def test_zero_mult_only_duplicates(self) -> None:
        clusters = cluster_levels([point(1.0), point(1.0), point(1.01)], atr=5.0, mult=0)
        assert [c.touches for c in clusters] == [2, 1]


# --- día anterior ------------------------------------------------------------


class TestPreviousDay:
    def test_across_utc_midnight(self) -> None:
        df = candles([10, 12, 11], [9, 8, 10.5], start="2024-03-01 22:00")
        result = previous_day_high_low(df)
        assert result == PreviousDayRange(high=12.0, low=8.0, day=date(2024, 3, 1))
        high, low, day = result  # se desarma como tupla
        assert (high, low, day) == (12.0, 8.0, date(2024, 3, 1))

    def test_takes_only_the_previous_day(self) -> None:
        # 20:00 del 1/3 → 01:00 del 3/3: días 1 (parcial), 2 (completo) y 3 (hoy).
        n = 30
        highs = [100.0] * n
        lows = [90.0] * n
        highs[1] = 500.0  # 1/3, no cuenta
        lows[2] = 1.0  # 1/3, no cuenta
        highs[29] = 600.0  # 3/3 (hoy), no cuenta
        lows[28] = 2.0  # 3/3 (hoy), no cuenta
        highs[10] = 130.0  # 2/3 06:00
        lows[20] = 70.0  # 2/3 16:00
        df = candles(highs, lows, start="2024-03-01 20:00")
        assert previous_day_high_low(df) == (130.0, 70.0, date(2024, 3, 2))

    def test_non_utc_timezone(self) -> None:
        # 1h desde el 1/3 00:00 UTC hasta el 2/3 05:00 UTC. En Buenos Aires
        # (UTC−3) el día local arranca a las 03:00 UTC.
        n = 30
        highs = [100.0] * n
        lows = [90.0] * n
        highs[1] = 200.0  # 1/3 01:00 UTC = 29/2 22:00 ART
        lows[25] = 50.0  # 2/3 01:00 UTC = 1/3 22:00 ART
        df = candles(highs, lows, start="2024-03-01 00:00")
        assert previous_day_high_low(df, "UTC") == (200.0, 90.0, date(2024, 3, 1))
        assert previous_day_high_low(df, "America/Argentina/Buenos_Aires") == (
            100.0,
            50.0,
            date(2024, 3, 1),
        )

    def test_candle_goes_to_the_day_of_its_open_time(self) -> None:
        # Velas de 4h: la de las 22:00 UTC del 1/3 cierra el 2/3 pero es del 1/3.
        df = candles([10, 20, 15], [9, 19, 14], start="2024-03-01 18:00", freq="4h")
        assert previous_day_high_low(df) == (20.0, 9.0, date(2024, 3, 1))

    def test_fewer_than_two_days(self) -> None:
        assert previous_day_high_low(candles([1, 2, 3], start="2024-03-01 00:00")) is None
        assert previous_day_high_low(candles([])) is None

    def test_gap_returns_none(self) -> None:
        # Hay velas el 1/3 y el 3/3, pero no el 2/3: el día anterior no tiene datos.
        df = pd.concat(
            [candles([5, 6], start="2024-03-01 10:00"), candles([7, 8], start="2024-03-03 10:00")]
        )
        assert previous_day_high_low(df) is None

    def test_as_of_fixes_the_midnight_border(self) -> None:
        # Solo velas cerradas: la última es la de las 23:00 del 2/3, pero ya es el 3/3.
        df = candles([10, 11, 50, 12], [9, 8, 7, 11], start="2024-03-01 22:00", freq="1h")
        df = pd.concat([df, candles([30] * 22, [20] * 22, start="2024-03-02 02:00")])
        assert df.index[-1] == pd.Timestamp("2024-03-02 23:00", tz="UTC")
        without = previous_day_high_low(df)
        assert without is not None and without.day == date(2024, 3, 1)
        now = datetime(2024, 3, 3, 0, 5, tzinfo=UTC)
        assert previous_day_high_low(df, as_of=now) == (50.0, 7.0, date(2024, 3, 2))
        # as_of sin zona horaria se toma como UTC.
        assert previous_day_high_low(df, as_of=datetime(2024, 3, 3, 0, 5)) == (
            50.0,
            7.0,
            date(2024, 3, 2),
        )

    def test_naive_index_is_utc(self) -> None:
        df = candles([10, 12, 11], [9, 8, 10.5], start="2024-03-01 22:00")
        df.index = df.index.tz_localize(None)
        assert previous_day_high_low(df) == (12.0, 8.0, date(2024, 3, 1))

    def test_requires_datetime_index(self) -> None:
        df = candles([1, 2, 3]).reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            previous_day_high_low(df)


# --- selección ---------------------------------------------------------------


class TestNearestLevels:
    def test_ordering_limit_and_roles(self) -> None:
        prices = [110, 90, 101, 99, 105, 95, 98, 102, 100]
        supports, resistances = nearest_levels([cluster_at(p) for p in prices], 100.0, n=3)
        assert [lv.price for lv in supports] == [99, 98, 95]
        assert [lv.price for lv in resistances] == [101, 102, 105]
        assert all(lv.kind == "soporte" for lv in supports)
        assert all(lv.kind == "resistencia" for lv in resistances)
        assert supports[0].distance_pct == pytest.approx(-1.0)
        assert resistances[0].distance_pct == pytest.approx(1.0)
        assert resistances[2].distance_pct == pytest.approx(5.0)

    def test_level_at_price_is_excluded(self) -> None:
        supports, resistances = nearest_levels([cluster_at(100.0)], 100.0, n=3)
        assert supports == [] and resistances == []

    def test_n_bounds(self) -> None:
        clusters = [cluster_at(p) for p in (97, 98, 103)]
        supports, resistances = nearest_levels(clusters, 100.0, n=10)
        assert [lv.price for lv in supports] == [98, 97]
        assert [lv.price for lv in resistances] == [103]
        assert nearest_levels(clusters, 100.0, n=0) == ([], [])
        with pytest.raises(ValueError):
            nearest_levels(clusters, 100.0, n=-1)

    @pytest.mark.parametrize("bad_price", [0.0, -5.0, NAN, math.inf, None])
    def test_invalid_price(self, bad_price: object) -> None:
        with pytest.raises(ValueError):
            nearest_levels([cluster_at(1.0)], bad_price, n=3)  # type: ignore[arg-type]

    def test_no_double_counting(self) -> None:
        rnd = random.Random(3)
        clusters = [cluster_at(rnd.uniform(90, 110)) for _ in range(50)]
        clusters.append(cluster_at(100.0))
        supports, resistances = nearest_levels(clusters, 100.0, n=50)
        assert len(supports) + len(resistances) == 50  # el de 100 queda afuera
        assert {lv.price for lv in supports}.isdisjoint({lv.price for lv in resistances})

    def test_keeps_metadata_and_accepts_levels(self) -> None:
        when = pd.Timestamp("2024-01-02 03:00", tz="UTC")
        cluster = LevelCluster(
            price=95.0,
            touches=4,
            sources=frozenset({SOURCE_PIVOT_HIGH, SOURCE_PREV_DAY_LOW}),
            last_touch=when,
            zone_low=94.5,
            zone_high=95.4,
        )
        (support,), _ = nearest_levels([cluster], 100.0)
        assert support == Level(
            price=95.0,
            kind="soporte",
            touches=4,
            sources=frozenset({SOURCE_PIVOT_HIGH, SOURCE_PREV_DAY_LOW}),
            last_touch=when,
            distance_pct=pytest.approx(-5.0),
            zone_low=94.5,
            zone_high=95.4,
        )
        # Reevaluar contra otro precio (ej. el de Nexo cargado a mano) cambia el rol.
        _, (resistance,) = nearest_levels([support], 90.0)
        assert resistance.kind == "resistencia"
        assert resistance.distance_pct == pytest.approx(95.0 / 90.0 * 100 - 100)


# --- todo junto --------------------------------------------------------------


class TestSupportResistance:
    def test_hand_built_series(self) -> None:
        # Día 1/3: pico 110 (vela 3) y valle 95 (vela 9). Día 2/3: pico 108
        # (vela 15), valle 97 (vela 21) y la última vela en 102.
        highs = [100, 101, 102, 110, 102, 101, 100, 99, 98, 96, 98, 99]
        highs += [100, 101, 102, 108, 102, 101, 100, 99, 98, 97.5, 99, 100, 101, 102]
        lows = [h - 1 for h in highs]
        lows[9] = 95.0
        lows[21] = 97.0
        df = candles(highs, lows, start="2024-03-01 12:00")
        price = 102.0
        result = support_resistance(
            df,
            price=price,
            atr=2.0,
            left=3,
            right=3,
            cluster_atr_mult=0.5,
            max_levels_each_side=3,
        )
        assert isinstance(result, SRResult)
        assert result.reference_price == price
        assert result.cluster_tolerance == 1.0
        assert result.prev_day == date(2024, 3, 1)
        assert (result.prev_day_high, result.prev_day_low) == (110.0, 95.0)
        assert result.pivots_used == tuple(find_pivots(df, 3, 3))

        res_prices = [lv.price for lv in result.resistances]
        assert res_prices == [108.0, 110.0]
        top = result.resistances[1]
        # El pico de 110 es pivote y máximo del día anterior: mismo toque, dos fuentes.
        assert top.sources == {SOURCE_PIVOT_HIGH, SOURCE_PREV_DAY_HIGH}
        assert top.touches == 1

        sup_prices = [lv.price for lv in result.supports]
        assert sup_prices == sorted(sup_prices, reverse=True)
        assert 97.0 in sup_prices
        assert all(p < price for p in sup_prices)
        prev_low = [lv for lv in result.supports if SOURCE_PREV_DAY_LOW in lv.sources]
        assert len(prev_low) == 1 and prev_low[0].price == 95.0

    def test_without_previous_day(self) -> None:
        df = random_walk_candles(1, n=300)
        price = float(df["close"].iloc[-1])
        result = support_resistance(df, price, simple_atr(df), 3, 3, 0.5, 3, False)
        assert result.prev_day_high is None and result.prev_day_low is None
        assert result.prev_day is None
        for level in (*result.supports, *result.resistances):
            assert level.sources <= {SOURCE_PIVOT_HIGH, SOURCE_PIVOT_LOW}

    def test_pivot_high_below_price_becomes_support(self) -> None:
        highs = [1, 2, 3, 9, 3, 2, 1]
        df = candles(highs, start="2024-03-01 00:00")
        result = support_resistance(df, 20.0, NAN, 3, 3, 0.5, 3)
        (support,) = result.supports
        assert support.price == 9.0
        assert support.kind == "soporte"
        assert support.sources == {SOURCE_PIVOT_HIGH}
        assert result.resistances == ()
        assert result.cluster_tolerance == 0.0

    def test_empty_frame(self) -> None:
        result = support_resistance(candles([]), 100.0, 1.0, 3, 3, 0.5, 3)
        assert result.supports == () and result.resistances == ()
        assert result.pivots_used == () and result.prev_day_high is None

    def test_invalid_price(self) -> None:
        with pytest.raises(ValueError):
            support_resistance(candles([1, 2, 3]), 0.0, 1.0, 3, 3, 0.5, 3)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_walk_with_config(self, seed: int, settings) -> None:
        cfg = settings.indicators.levels
        df = random_walk_candles(seed, n=600)
        price = float(df["close"].iloc[-1])
        atr_value = simple_atr(df)
        result = support_resistance(
            df,
            price,
            atr_value,
            cfg.pivot_left,
            cfg.pivot_right,
            cfg.cluster_atr_mult,
            cfg.max_levels_each_side,
            cfg.include_previous_day,
            settings.app.market_timezone,
        )
        assert 0 < len(result.supports) <= cfg.max_levels_each_side
        assert 0 < len(result.resistances) <= cfg.max_levels_each_side
        assert all(
            lv.price < price < rv.price for lv in result.supports for rv in result.resistances
        )
        sup = [lv.price for lv in result.supports]
        res = [lv.price for lv in result.resistances]
        assert sup == sorted(sup, reverse=True)
        assert res == sorted(res)
        assert all(lv.distance_pct < 0 for lv in result.supports)
        assert all(lv.distance_pct > 0 for lv in result.resistances)
        assert all(lv.touches >= 1 for lv in (*result.supports, *result.resistances))
        # Niveles vecinos separados por más que la tolerancia.
        ladder = sup[::-1] + res
        tolerance = cfg.cluster_atr_mult * atr_value
        assert result.cluster_tolerance == pytest.approx(tolerance)
        assert all(b - a > tolerance for a, b in zip(ladder, ladder[1:], strict=False))

        # El día anterior coincide con un cálculo directo por día UTC.
        days = df.index.normalize()
        prev_mask = days == days.max() - pd.Timedelta(days=1)
        assert result.prev_day_high == df["high"][prev_mask].max()
        assert result.prev_day_low == df["low"][prev_mask].min()

    def test_historical_slice_uses_only_known_pivots(self) -> None:
        df = random_walk_candles(42, n=400)
        k = 250
        past = df.iloc[: k + 1]
        price = float(past["close"].iloc[-1])
        result = support_resistance(past, price, simple_atr(past), 3, 3, 0.5, 3)
        full = find_pivots(df, 3, 3)
        assert list(result.pivots_used) == pivots_known_at(full, df.index[k])
