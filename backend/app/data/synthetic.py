"""Mercado sintético determinista para demos y tests sin red.

Sirve para correr la app entera sin conexión (``analyze_cli --offline``) y para
los tests de punta a punta. **Nada de esto es real**: cada dato sale con
``FetchMeta(source="sintetico", warning="Datos sintéticos: no son reales")``.

Velas
    Se genera una sola serie base de velas de 5m por moneda y las de 15m, 1h,
    4h y 1d se arman agregándola (``aggregate_candles``), así todos los
    timeframes son coherentes entre sí: el máximo de una vela de 1h es el
    máximo de sus doce velas de 5m, el cierre diario es el último cierre de 5m
    del día, etc. La serie arranca a las 00:00 UTC de ``history_days`` días
    atrás y termina en la vela de 5m que se está formando (``closed=False``).

    Con ``min_history_days`` (o ``prehistory_days``) se agregan velas más
    viejas antes de ese inicio, con otra semilla y el mismo generador, para
    entrenar el detector de régimen (1,5 años de 1h, 4 años de 1d) sin cambiar
    ni una vela de la serie reciente. Generar 1.600 días de velas de 5m tarda
    menos de medio segundo.

    La serie base cambia de régimen por tramos de entre 6 horas y 5 días:

    - ``up`` / ``down``: tendencia con deriva diaria de 0,4 a 1,2 veces la
      volatilidad diaria de la moneda;
    - ``range``: lateral. El camino aleatorio se "pliega" dentro de una banda
      alrededor del precio de entrada al tramo (reflexión en los bordes), así
      el precio rebota varias veces en los mismos niveles y aparecen soportes y
      resistencias;
    - ``volatile``: más volatilidad y deriva al azar.

    Los retornos tienen colas gruesas (t de Student con 4 grados de libertad),
    la volatilidad y el volumen siguen un ciclo diario (más movimiento cerca de
    la apertura de EE. UU.) y el volumen crece con el tamaño de la vela. Al
    final se escala la serie para que el último precio quede cerca del precio
    de referencia de la moneda (``COIN_PROFILES``: BTC en decenas de miles, XRP
    en pocos dólares, etc.).

Resto de los datos
    Ticker (del último cierre y las últimas 24 h de velas de 5m), funding
    (actual e historial cada ``funding_interval_hours``, ligado a cómo viene el
    precio), interés abierto e historial, ratio long/short e historial
    (series suaves en una grilla de 5m de los últimos 30 días que se muestrean
    según el período pedido), Fear & Greed, mercado global y algunos
    titulares. Todo depende solo de ``seed``, del símbolo y de ``now``: con los
    mismos valores sale exactamente lo mismo.

``SyntheticMarketData`` implementa ``MarketDataSource`` (el mismo camino de
código que Binance o Bybit) y suma ``fear_greed``, ``global_market`` y
``headlines`` para los datos que no vienen de un exchange.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import numpy as np
import pandas as pd

from app.data.market import GlobalMarket
from app.data.news import Headline, NewsResult
from app.data.sentiment import FearGreed, translate_classification
from app.exchanges.base import (
    TIMEFRAME_MINUTES,
    Candles,
    FetchMeta,
    FundingHistory,
    FundingInfo,
    FundingRatePoint,
    LongShortHistory,
    LongShortPoint,
    OpenInterest,
    OpenInterestHistory,
    OpenInterestPoint,
    Ticker,
    Timeframe,
    validate_candles_frame,
)

SYNTHETIC_SOURCE = "sintetico"
SYNTHETIC_WARNING = "Datos sintéticos: no son reales"
DEFAULT_SEED = 7
DEFAULT_HISTORY_DAYS = 520  # alcanza para 500 velas diarias

BASE_MINUTES = 5
BARS_PER_DAY = 24 * 60 // BASE_MINUTES
DERIVATIVES_GRID_DAYS = 30  # grilla de 5m para OI y long/short


@dataclass(frozen=True)
class CoinProfile:
    """Parámetros de generación de una moneda (no son reglas de riesgo)."""

    name: str  # nombre para los titulares
    base_price: float  # precio aproximado del final de la serie, en USDT
    daily_vol: float  # volatilidad diaria típica, fracción (0.03 = 3 %)
    volume_5m: float  # volumen típico de una vela de 5m, en moneda base
    open_interest: float  # interés abierto típico, en moneda base
    long_short: float  # ratio de cuentas long/short típico


COIN_PROFILES: MappingProxyType[str, CoinProfile] = MappingProxyType(
    {
        "BTC": CoinProfile("Bitcoin", 98_000.0, 0.020, 450.0, 85_000.0, 1.3),
        "ETH": CoinProfile("Ethereum", 3_600.0, 0.027, 12_000.0, 2_200_000.0, 1.9),
        "SOL": CoinProfile("Solana", 175.0, 0.036, 87_000.0, 9_000_000.0, 2.6),
        "XRP": CoinProfile("XRP", 2.6, 0.034, 3_500_000.0, 700_000_000.0, 3.0),
        "AVAX": CoinProfile("Avalanche", 28.0, 0.038, 21_000.0, 18_000_000.0, 2.4),
    }
)

# Bandas de alternative.me para etiquetar el Fear & Greed sintético.
_FEAR_GREED_BANDS: tuple[tuple[int, str], ...] = (
    (24, "Extreme Fear"),
    (46, "Fear"),
    (54, "Neutral"),
    (75, "Greed"),
    (100, "Extreme Greed"),
)

_SEGMENT_KINDS = ("up", "down", "range", "volatile")
_SEGMENT_BASE_PROBS = np.array([0.25, 0.25, 0.40, 0.10])


# --- utilidades ----------------------------------------------------------------


def synthetic_meta(now: datetime) -> FetchMeta:
    """``FetchMeta`` de un dato sintético (fuente ``"sintetico"`` y aviso)."""
    return FetchMeta(source=SYNTHETIC_SOURCE, fetched_at=now, warning=SYNTHETIC_WARNING)


def _stable_seed(*parts: object) -> int:
    """Semilla estable entre procesos (``hash()`` de Python cambia en cada corrida)."""
    return zlib.crc32(":".join(str(p) for p in parts).encode("utf-8"))


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(*parts))


def _norm_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s or not s.isalnum():
        raise ValueError(f"Símbolo inválido: {symbol!r}")
    return s


def coin_profile(symbol: str) -> CoinProfile:
    """Perfil de la moneda; para una desconocida se inventa uno estable a partir del nombre."""
    s = _norm_symbol(symbol)
    known = COIN_PROFILES.get(s)
    if known is not None:
        return known
    rng = _rng("profile", s)
    price = float(10 ** rng.uniform(-1, 2))
    return CoinProfile(
        name=s,
        base_price=price,
        daily_vol=float(rng.uniform(0.03, 0.05)),
        volume_5m=50_000.0 / price,
        open_interest=20_000_000.0 / price,
        long_short=float(rng.uniform(1.2, 2.8)),
    )


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("La fecha tiene que tener zona horaria")
    return dt.astimezone(UTC)


def _floor(dt: datetime, minutes: int) -> datetime:
    epoch_min = int(dt.timestamp() // 60)
    return datetime.fromtimestamp((epoch_min - epoch_min % minutes) * 60, tz=UTC)


def _fold(x: np.ndarray, half_width: float) -> np.ndarray:
    """Pliega ``x`` dentro de ``[-half_width, half_width]`` reflejando en los bordes."""
    w = half_width
    return w - np.abs(np.mod(x + w, 4 * w) - 2 * w)


def _timeframe_minutes(timeframe: str) -> int:
    try:
        return TIMEFRAME_MINUTES[timeframe]
    except KeyError:
        raise ValueError(
            f"Timeframe no soportado: {timeframe!r} (opciones: {', '.join(TIMEFRAME_MINUTES)})"
        ) from None


# --- serie base y agregación ---------------------------------------------------


def _simulate_returns(
    rng: np.random.Generator, n: int, profile: CoinProfile
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retornos logarítmicos de ``n`` velas de 5m que arrancan a las 00:00 UTC.

    Devuelve ``(log_ret, vol_mult, z, vol_season, volume_season)``: los
    retornos por tramos de régimen, el multiplicador de volatilidad de cada
    vela, el shock estandarizado y los ciclos diarios de volatilidad y volumen.
    """
    # Ciclo diario: más movimiento y volumen cerca de las 14:30 UTC.
    minute_of_day = (np.arange(n) * BASE_MINUTES) % (24 * 60)
    phase = np.cos(2 * np.pi * (minute_of_day / 60.0 - 14.5) / 24.0)
    vol_season = 1.0 + 0.35 * phase
    volume_season = 1.0 + 0.5 * phase

    z = np.clip(rng.standard_t(4, n) / math.sqrt(2.0), -8.0, 8.0)
    bar_vol = profile.daily_vol / math.sqrt(BARS_PER_DAY)

    log_ret = np.empty(n)
    vol_mult = np.empty(n)
    band = profile.daily_vol * math.sqrt(60.0)  # cuánto se deja alejar el precio del inicio
    level = 0.0
    i = 0
    while i < n:
        # Cuanto más se alejó el precio del inicio, menos probable es seguir
        # en esa dirección: mantiene la serie en un rango creíble.
        pull = float(np.clip(level / band, -3.0, 3.0))
        probs = _SEGMENT_BASE_PROBS * np.array([math.exp(-2 * pull), math.exp(2 * pull), 1, 1])
        kind = _SEGMENT_KINDS[int(rng.choice(4, p=probs / probs.sum()))]
        length = int(rng.integers(BARS_PER_DAY // 4, BARS_PER_DAY * 5))
        if kind == "volatile":
            length //= 2
        j = min(n, i + length)
        seg = slice(i, j)
        sigma = bar_vol * vol_season[seg]
        if kind in ("up", "down"):
            mult = float(rng.uniform(0.8, 1.1))
            drift = float(rng.uniform(0.4, 1.2)) * profile.daily_vol / BARS_PER_DAY
            drift = drift if kind == "up" else -drift
            r = drift + sigma * mult * z[seg]
        elif kind == "volatile":
            mult = float(rng.uniform(1.6, 2.2))
            drift = float(rng.uniform(-1.0, 1.0)) * profile.daily_vol / BARS_PER_DAY
            if abs(pull) > 0.5:  # lejos del inicio, el tramo volátil vuelve hacia el centro
                drift = -math.copysign(abs(drift), pull)
            r = drift + sigma * mult * z[seg]
        else:
            mult = float(rng.uniform(0.6, 1.0))
            half_width = float(rng.uniform(0.6, 1.8)) * profile.daily_vol
            path = _fold(np.cumsum(sigma * mult * z[seg]), half_width)
            r = np.diff(path, prepend=0.0)
        log_ret[seg] = r
        vol_mult[seg] = mult
        level += float(r.sum())
        i = j
    return log_ret, vol_mult, z, vol_season, volume_season


def _ohlcv_from_returns(
    rng: np.random.Generator,
    simulated: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    profile: CoinProfile,
    end_price: float,
) -> dict[str, np.ndarray]:
    """Arma open/high/low/close/volume a partir de los retornos simulados.

    La serie se escala para que el último cierre sea ``end_price``.
    """
    log_ret, vol_mult, z, vol_season, volume_season = simulated
    n = len(log_ret)
    bar_vol = profile.daily_vol / math.sqrt(BARS_PER_DAY)
    log_close = np.cumsum(log_ret)
    close = np.exp(log_close - log_close[-1]) * end_price
    open_ = np.empty(n)
    open_[0] = close[0] / math.exp(log_ret[0])
    open_[1:] = close[:-1]
    sigma_all = bar_vol * vol_season * vol_mult
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.7, n)) * sigma_all)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.7, n)) * sigma_all)
    volume = (
        profile.volume_5m
        * volume_season
        * vol_mult
        * np.exp(rng.normal(0, 0.3, n) - 0.045)
        * (0.6 + 0.4 * np.abs(z))
    )
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def generate_base_candles(
    symbol: str,
    *,
    now: datetime,
    seed: int = DEFAULT_SEED,
    history_days: int = DEFAULT_HISTORY_DAYS,
    prehistory_days: int = 0,
) -> pd.DataFrame:
    """Velas de 5m desde las 00:00 UTC de ``history_days`` días atrás hasta ``now``.

    Devuelve el DataFrame con las convenciones de ``exchanges.base`` (índice
    ``open_time`` UTC, OHLCV y ``closed``). La última vela es la que se forma.

    ``prehistory_days`` agrega velas **más viejas** antes de ese inicio (para
    entrenar el detector de régimen con años de historia). Salen de otra
    semilla, con el mismo generador de tramos, y empalman con la primera vela
    de la serie: las velas de los últimos ``history_days`` días quedan
    exactamente iguales que sin prehistoria.
    """
    if history_days < 1:
        raise ValueError("history_days tiene que ser >= 1")
    if prehistory_days < 0:
        raise ValueError("prehistory_days tiene que ser >= 0")
    now = _to_utc(now)
    s = _norm_symbol(symbol)
    profile = coin_profile(s)
    start = _floor(now, 24 * 60) - timedelta(days=history_days)
    last_open = _floor(now, BASE_MINUTES)
    n = int((last_open - start).total_seconds() // (BASE_MINUTES * 60)) + 1
    rng = _rng("candles", seed, s)

    simulated = _simulate_returns(rng, n, profile)
    end_price = profile.base_price * math.exp(float(rng.uniform(-0.04, 0.04)))
    columns = _ohlcv_from_returns(rng, simulated, profile, end_price)

    if prehistory_days:
        n_pre = prehistory_days * BARS_PER_DAY
        rng_pre = _rng("candles-prehistory", seed, s, prehistory_days)
        simulated_pre = _simulate_returns(rng_pre, n_pre, profile)
        # El último cierre de la prehistoria es la apertura de la primera vela.
        pre = _ohlcv_from_returns(rng_pre, simulated_pre, profile, float(columns["open"][0]))
        columns = {k: np.concatenate([pre[k], columns[k]]) for k in columns}
        start -= timedelta(days=prehistory_days)
        n += n_pre

    index = pd.date_range(start, periods=n, freq=f"{BASE_MINUTES}min", tz="UTC", name="open_time")
    closed = (index + pd.Timedelta(minutes=BASE_MINUTES)) <= pd.Timestamp(now)
    df = pd.DataFrame({**columns, "closed": closed}, index=index)
    return validate_candles_frame(df)


def aggregate_candles(base: pd.DataFrame, timeframe: str, now: datetime) -> pd.DataFrame:
    """Agrega velas de 5m (que arrancan a las 00:00 UTC) al ``timeframe`` pedido.

    Una vela agregada está cerrada si ``open_time + timeframe <= now``.
    """
    minutes = _timeframe_minutes(timeframe)
    if minutes == BASE_MINUTES:
        return base.copy()
    if minutes % BASE_MINUTES:
        raise ValueError(f"No se puede armar {timeframe} con velas de {BASE_MINUTES}m")
    if base.empty:
        return base.copy()
    factor = minutes // BASE_MINUTES
    starts = np.arange(0, len(base), factor)
    open_ = base["open"].to_numpy()[starts]
    high = np.maximum.reduceat(base["high"].to_numpy(), starts)
    low = np.minimum.reduceat(base["low"].to_numpy(), starts)
    ends = np.append(starts[1:], len(base)) - 1
    close = base["close"].to_numpy()[ends]
    volume = np.add.reduceat(base["volume"].to_numpy(), starts)
    index = base.index[starts]
    closed = (index + pd.Timedelta(minutes=minutes)) <= pd.Timestamp(_to_utc(now))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df["closed"] = closed
    return validate_candles_frame(df)


# --- fuente de datos -----------------------------------------------------------


class SyntheticMarketData:
    """``MarketDataSource`` sin red, con datos inventados pero coherentes.

    ``now`` queda fijo al crear la instancia (por defecto, la hora actual), así
    todas las consultas ven el mismo mercado. Las series se generan la primera
    vez que se piden y se guardan en la instancia.

    ``min_history_days``: días de historia que tienen que estar disponibles
    (por ejemplo ``regime.context_training_days`` para entrenar el régimen
    diario). Si es mayor que ``history_days``, la diferencia se agrega como
    prehistoria (ver ``generate_base_candles``): las velas recientes, el
    ticker y los derivados quedan idénticos a los de una instancia sin ella.
    """

    name = SYNTHETIC_SOURCE

    def __init__(
        self,
        *,
        seed: int = DEFAULT_SEED,
        now: datetime | None = None,
        history_days: int = DEFAULT_HISTORY_DAYS,
        funding_interval_hours: float = 8.0,
        min_history_days: int | None = None,
    ) -> None:
        if funding_interval_hours <= 0:
            raise ValueError("funding_interval_hours tiene que ser > 0")
        if min_history_days is not None and min_history_days < 1:
            raise ValueError("min_history_days tiene que ser >= 1")
        self.seed = seed
        self.now = _to_utc(now) if now is not None else datetime.now(UTC)
        self.history_days = history_days
        self.prehistory_days = max(0, (min_history_days or 0) - history_days)
        self.funding_interval_hours = float(funding_interval_hours)
        self._base: dict[str, pd.DataFrame] = {}
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}
        self._grids: dict[str, pd.DataFrame] = {}
        self.calls: list[str] = []  # para inspección en tests y demos

    @property
    def meta(self) -> FetchMeta:
        return synthetic_meta(self.now)

    # --- series internas ---------------------------------------------------

    def base_candles(self, symbol: str) -> pd.DataFrame:
        """Serie base de 5m completa de la moneda (generada una vez por instancia)."""
        s = _norm_symbol(symbol)
        if s not in self._base:
            self._base[s] = generate_base_candles(
                s,
                now=self.now,
                seed=self.seed,
                history_days=self.history_days,
                prehistory_days=self.prehistory_days,
            )
        return self._base[s]

    def candles_frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Todas las velas de ``timeframe`` disponibles (sin copiar: no modificar)."""
        s = _norm_symbol(symbol)
        key = (s, timeframe)
        if key not in self._frames:
            self._frames[key] = aggregate_candles(self.base_candles(s), timeframe, self.now)
        return self._frames[key]

    def _closes_5m(self, symbol: str) -> np.ndarray:
        return self.base_candles(symbol)["close"].to_numpy()

    def _price_at(self, symbol: str, when: datetime) -> float:
        """Último cierre de 5m conocido en ``when`` (o el primero si es anterior a la serie)."""
        base = self.base_candles(symbol)
        pos = int(base.index.searchsorted(pd.Timestamp(when), side="right")) - 1
        return float(base["close"].iloc[max(pos, 0)])

    def _return_over(self, symbol: str, hours: float, until: datetime | None = None) -> float:
        until = until or self.now
        start = until - timedelta(hours=hours)
        return self._price_at(symbol, until) / self._price_at(symbol, start) - 1.0

    def _derivatives_grid(self, symbol: str) -> pd.DataFrame:
        """OI y long/short en una grilla de 5m de los últimos ``DERIVATIVES_GRID_DAYS`` días."""
        s = _norm_symbol(symbol)
        if s not in self._grids:
            self._grids[s] = self._build_derivatives_grid(s)
        return self._grids[s]

    def _build_derivatives_grid(self, s: str) -> pd.DataFrame:
        base = self.base_candles(s)
        n = min(len(base), DERIVATIVES_GRID_DAYS * BARS_PER_DAY)
        closes = base["close"].to_numpy()[-n:]
        index = base.index[-n:]
        profile = coin_profile(s)
        rng = _rng("derivatives", self.seed, s)
        log_ret = np.diff(np.log(closes), prepend=math.log(closes[0]))
        oi_log = np.cumsum(0.4 * log_ret + rng.normal(0, 0.0015, n))
        oi_end = profile.open_interest * math.exp(float(rng.uniform(-0.15, 0.15)))
        oi = oi_end * np.exp(oi_log - oi_log[-1])
        day_ret = pd.Series(closes).pct_change(BARS_PER_DAY).fillna(0.0).to_numpy()
        ls_noise = _fold(np.cumsum(rng.normal(0, 0.01, n)), 0.25)
        ratio = profile.long_short * np.exp(ls_noise - 1.5 * day_ret)
        return pd.DataFrame({"oi": oi, "close": closes, "ratio": ratio}, index=index)

    def _sample_times(self, period: str, limit: int) -> list[datetime]:
        minutes = _timeframe_minutes(period)
        last = _floor(self.now, minutes)
        return [last - timedelta(minutes=minutes * k) for k in range(limit - 1, -1, -1)]

    # --- MarketDataSource ----------------------------------------------------

    async def get_ticker(self, symbol: str) -> Ticker:
        s = _norm_symbol(symbol)
        self.calls.append(f"ticker:{s}")
        base = self.base_candles(s)
        last_day = base.iloc[-BARS_PER_DAY:]
        price = float(base["close"].iloc[-1])
        ref = float(base["close"].iloc[-BARS_PER_DAY - 1]) if len(base) > BARS_PER_DAY else None
        return Ticker(
            symbol=s,
            price=price,
            meta=self.meta,
            change_24h_pct=None if ref is None else (price / ref - 1.0) * 100.0,
            high_24h=float(last_day["high"].max()),
            low_24h=float(last_day["low"].min()),
            quote_volume_24h=float((last_day["volume"] * last_day["close"]).sum()),
        )

    async def get_klines(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 500,
        end_time: datetime | None = None,
    ) -> Candles:
        s = _norm_symbol(symbol)
        if limit < 1:
            raise ValueError("limit tiene que ser >= 1")
        self.calls.append(f"klines:{s}:{timeframe}:{limit}")
        df = self.candles_frame(s, timeframe)
        if end_time is not None:
            df = df[df.index <= pd.Timestamp(_to_utc(end_time))]
        return Candles(s, timeframe, df.iloc[-limit:].copy(), self.meta)

    async def get_klines_history(
        self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime | None = None
    ) -> Candles:
        s = _norm_symbol(symbol)
        self.calls.append(f"klines_history:{s}:{timeframe}")
        df = self.candles_frame(s, timeframe)
        mask = df.index >= pd.Timestamp(_to_utc(start))
        if end is not None:
            mask &= df.index <= pd.Timestamp(_to_utc(end))
        return Candles(s, timeframe, df[mask].copy(), self.meta)

    def _funding_rate_at(self, symbol: str, when: datetime) -> float:
        """Funding (fracción) ligado a cómo vino el precio las 24 h previas, con ruido."""
        trend = self._return_over(symbol, 24, when)
        noise = _rng("funding", self.seed, symbol, int(when.timestamp())).normal(0, 0.00003)
        rate = 0.0001 + 0.004 * trend + noise
        return round(float(np.clip(rate, -0.00075, 0.0015)), 8)

    async def get_funding(self, symbol: str) -> FundingInfo:
        s = _norm_symbol(symbol)
        self.calls.append(f"funding:{s}")
        price = self._price_at(s, self.now)
        interval_min = int(round(self.funding_interval_hours * 60))
        next_time = _floor(self.now, interval_min) + timedelta(minutes=interval_min)
        return FundingInfo(
            symbol=s,
            rate=self._funding_rate_at(s, self.now),
            mark_price=price * 1.0001,
            meta=self.meta,
            index_price=price,
            next_funding_time=next_time,
            interval_hours=self.funding_interval_hours,
        )

    async def get_funding_history(self, symbol: str, limit: int = 21) -> FundingHistory:
        s = _norm_symbol(symbol)
        self.calls.append(f"funding_history:{s}:{limit}")
        interval_min = int(round(self.funding_interval_hours * 60))
        last = _floor(self.now, interval_min)
        times = [last - timedelta(minutes=interval_min * k) for k in range(limit - 1, -1, -1)]
        points = [FundingRatePoint(t, self._funding_rate_at(s, t)) for t in times]
        return FundingHistory(s, points, self.meta)

    async def get_open_interest(self, symbol: str) -> OpenInterest:
        s = _norm_symbol(symbol)
        self.calls.append(f"open_interest:{s}")
        grid = self._derivatives_grid(s)
        contracts = float(grid["oi"].iloc[-1])
        return OpenInterest(
            symbol=s,
            contracts=contracts,
            meta=self.meta,
            value_usd=contracts * float(grid["close"].iloc[-1]),
            time=self.now,
        )

    async def get_open_interest_history(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> OpenInterestHistory:
        s = _norm_symbol(symbol)
        self.calls.append(f"open_interest_history:{s}:{period}:{limit}")
        grid = self._derivatives_grid(s)
        points = []
        for t in self._sample_times(period, limit):
            pos = int(grid.index.searchsorted(pd.Timestamp(t), side="right")) - 1
            if pos < 0:
                continue
            oi = float(grid["oi"].iloc[pos])
            points.append(OpenInterestPoint(t, oi, oi * float(grid["close"].iloc[pos])))
        return OpenInterestHistory(s, period, points, self.meta)

    async def get_long_short_ratio(
        self, symbol: str, period: Timeframe = "1h", limit: int = 168
    ) -> LongShortHistory:
        s = _norm_symbol(symbol)
        self.calls.append(f"long_short:{s}:{period}:{limit}")
        grid = self._derivatives_grid(s)
        points = []
        for t in self._sample_times(period, limit):
            pos = int(grid.index.searchsorted(pd.Timestamp(t), side="right")) - 1
            if pos < 0:
                continue
            ratio = float(grid["ratio"].iloc[pos])
            long_pct = ratio / (1.0 + ratio)
            points.append(LongShortPoint(t, ratio, long_pct, 1.0 - long_pct))
        return LongShortHistory(s, period, points, self.meta)

    async def aclose(self) -> None:
        """No tiene recursos abiertos."""

    # --- datos que no vienen de un exchange ----------------------------------

    def _fear_greed_value(self, symbol: str, when: datetime) -> int:
        trend = self._return_over(symbol, 24 * 14, when)
        noise = _rng("fng", self.seed, symbol, when.date().isoformat()).normal(0, 4)
        return int(np.clip(round(50 + 250 * trend + noise), 3, 97))

    def fear_greed(self, symbol: str = "BTC", days: int = 30) -> FearGreed:
        """Fear & Greed sintético según cómo vino ``symbol`` las últimas 2 semanas."""
        s = _norm_symbol(symbol)
        if days < 1:
            raise ValueError("days tiene que ser >= 1")
        today = _floor(self.now, 24 * 60)
        history = [
            (today - timedelta(days=k), self._fear_greed_value(s, today - timedelta(days=k)))
            for k in range(days - 1, -1, -1)
        ]
        value = history[-1][1]
        label = next(lbl for top, lbl in _FEAR_GREED_BANDS if value <= top)
        return FearGreed(value, translate_classification(label), label, history, self.meta)

    def global_market(self, symbol: str = "BTC") -> GlobalMarket:
        """Dominancia y capitalización sintéticas (se mueven con ``symbol``)."""
        s = _norm_symbol(symbol)
        rng = _rng("global", self.seed, self.now.date().isoformat())
        month = self._return_over(s, 24 * 30)
        day = self._return_over(s, 24)
        return GlobalMarket(
            btc_dominance_pct=round(float(57.0 + rng.normal(0, 0.6)), 2),
            eth_dominance_pct=round(float(12.5 + rng.normal(0, 0.4)), 2),
            total_market_cap_usd=float(3.3e12 * (1.0 + 0.8 * month)),
            market_cap_change_24h_pct=float(day * 100.0 * 0.9),
            meta=self.meta,
            updated_at=self.now,
        )

    def headlines(self, symbol: str, limit: int = 15) -> NewsResult:
        """Titulares inventados (marcados como sintéticos) sobre la moneda y la macro."""
        s = _norm_symbol(symbol)
        name = coin_profile(s).name
        day = self._return_over(s, 24) * 100.0
        move = "sube" if day >= 0 else "cae"
        day_txt = f"{abs(day):.1f}".replace(".", ",")
        rng = _rng("news", self.seed, s, self.now.date().isoformat())
        templates: list[tuple[str, bool, float]] = [
            (
                f"{name} {move} {day_txt} % en 24 horas y el mercado mira los niveles",
                False,
                0.6,
            ),
            (f"Ballenas de {name} mueven fondos a exchanges, según datos on-chain", False, 2.0),
            (
                "La Fed deja las tasas sin cambios y el mercado espera el próximo dato de CPI",
                True,
                3.5,
            ),
            (f"Analistas discuten si {name} puede romper la resistencia de la semana", False, 5.0),
            ("Los ETF de cripto registran flujos mixtos en la última sesión", True, 7.5),
            (f"Sube el interés abierto en los futuros de {name}", False, 10.0),
            ("Powell habla de inflación: qué puede pasar con los activos de riesgo", True, 14.0),
            (
                f"Actualización de la red de {name}: desarrolladores fijan fecha tentativa",
                False,
                20.0,
            ),
        ]
        items = []
        for title, is_macro, hours in templates[: max(limit, 0)]:
            jitter = float(rng.uniform(-0.3, 0.3))
            items.append(
                Headline(
                    title=f"(sintético) {title}",
                    source=SYNTHETIC_SOURCE,
                    url="",
                    published_at=self.now - timedelta(hours=hours + jitter),
                    currencies=[] if is_macro else [s],
                    is_macro=is_macro,
                )
            )
        return NewsResult(s, items, "rss", self.meta, [SYNTHETIC_WARNING])
