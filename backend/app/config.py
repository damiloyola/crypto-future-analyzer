"""Carga y valida la configuración de la app.

Dos fuentes:

- ``config.yaml`` (raíz del repo): reglas de riesgo, umbrales, pesos y
  parámetros de cada módulo. Se valida con los modelos Pydantic de abajo. Si
  algo no cierra (pesos que no suman 1, apalancamiento por encima del tope duro,
  etc.) la app no arranca y explica por qué.
- ``.env`` (raíz del repo, fuera de git): secretos. ``ANTHROPIC_API_KEY`` y,
  opcional, ``CRYPTOPANIC_API_KEY``. Nunca se loguean ni se mandan al frontend.

El resto de los módulos recibe estos objetos por parámetro (no leen archivos
por su cuenta), así se pueden testear con configuraciones armadas a mano.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

RegimeCategory = Literal["crash", "bajista", "neutral", "alcista", "euforia"]
Timeframe = Literal["5m", "15m", "1h", "4h", "1d"]


class _Strict(BaseModel):
    """Base que rechaza claves desconocidas para detectar typos en config.yaml."""

    model_config = ConfigDict(extra="forbid")


# --- app / símbolos / exchanges ----------------------------------------------


class AppConfig(_Strict):
    user_timezone: str
    market_timezone: str = "UTC"
    host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 5173
    data_dir: str = "data"
    lock_file: str = "data/LOCKED"
    incident_dir: str = "data/incidents"


class SymbolsConfig(_Strict):
    favorites: list[str]
    quote: str = "USDT"
    default: str = "BTC"


class BinanceConfig(_Strict):
    spot_base_url: str
    futures_base_url: str
    use_perpetual_klines: bool = True


class BybitConfig(_Strict):
    base_url: str
    category: str = "linear"


class NexoConfig(_Strict):
    enabled: bool = False
    base_url: str
    allow_orders: bool = False
    instruments: dict[str, str] = Field(default_factory=dict)
    manual_price_max_diff_pct: float = Field(2.0, gt=0)


class ExchangesConfig(_Strict):
    data_primary: str
    data_fallback: str | None = None
    execution_venue: str
    request_timeout_s: float = 10
    max_retries: int = 2
    binance: BinanceConfig
    bybit: BybitConfig
    nexo: NexoConfig


class DataSourcesConfig(_Strict):
    fear_greed_url: str
    coingecko_base_url: str
    cryptopanic_base_url: str
    rss_feeds: list[str]
    news_max_headlines: int = 15
    news_keywords: dict[str, list[str]]
    macro_keywords: list[str] = Field(default_factory=list)


class TimeframesConfig(_Strict):
    context: list[Timeframe]
    structure: Timeframe
    entry: Timeframe
    fine_tune: Timeframe | None = None
    candles_per_request: int = 500
    closed_candles_only: bool = True


# --- indicadores -------------------------------------------------------------


class MacdConfig(_Strict):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class BollingerConfig(_Strict):
    period: int = 20
    std: float = 2.0


class LevelsConfig(_Strict):
    pivot_left: int = 3
    pivot_right: int = 3
    cluster_atr_mult: float = 0.5
    max_levels_each_side: int = 3
    include_previous_day: bool = True


class IndicatorsConfig(_Strict):
    ema_periods: list[int]
    vwap_timeframes: list[Timeframe]
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd: MacdConfig
    atr_period: int = 14
    bollinger: BollingerConfig
    relative_volume_period: int = 20
    levels: LevelsConfig


# --- régimen -----------------------------------------------------------------


class RegimeFeaturesConfig(_Strict):
    realized_vol_window: int = 24
    volume_zscore_window: int = 168


class RegimeLabel(_Strict):
    name: str
    category: RegimeCategory


class RegimeConfig(_Strict):
    timeframe: Timeframe = "1h"
    context_timeframe: Timeframe = "1d"
    training_days: int
    context_training_days: int
    features: RegimeFeaturesConfig
    n_states_min: int = 3
    n_states_max: int = 7
    selection_criterion: Literal["bic"] = "bic"
    covariance_type: Literal["full", "diag", "spherical", "tied"] = "full"
    n_restarts: int = 5
    max_iter: int = 200
    random_seed: int = 42
    retrain_every_days: int = 7
    models_dir: str = "data/models"
    confirmation_bars: int = 3
    instability_window: int = 20
    instability_max_changes: int = 4
    labels: dict[int, list[RegimeLabel]]

    @model_validator(mode="after")
    def _labels_cover_all_state_counts(self) -> RegimeConfig:
        if not 2 <= self.n_states_min <= self.n_states_max:
            raise ValueError("regime: n_states_min tiene que ser >= 2 y <= n_states_max")
        for n in range(self.n_states_min, self.n_states_max + 1):
            names = self.labels.get(n)
            if names is None:
                raise ValueError(f"regime.labels: falta la lista de nombres para {n} estados")
            if len(names) != n:
                raise ValueError(
                    f"regime.labels[{n}]: tiene {len(names)} nombres, se esperaban {n}"
                )
        return self


# --- scoring -----------------------------------------------------------------


class ScoringWeights(_Strict):
    trend_4h: float
    trend_1h: float
    trend_15m: float
    momentum: float
    derivatives: float
    sentiment: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoringWeights:
        total = sum(self.model_dump().values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"scoring.weights tiene que sumar 1.0 (suma {total:.4f})")
        if any(w < 0 for w in self.model_dump().values()):
            raise ValueError("scoring.weights no admite pesos negativos")
        return self


class MomentumScoringConfig(_Strict):
    timeframes: list[Timeframe]
    rsi_neutral: float = 50
    rsi_overbought: float = 70
    rsi_oversold: float = 30


class DerivativesScoringConfig(_Strict):
    funding_high_pct: float
    funding_very_high_pct: float
    funding_low_pct: float
    oi_lookback_hours: int = 24
    oi_rising_pct: float
    long_short_crowded_long: float
    long_short_crowded_short: float


class SentimentScoringConfig(_Strict):
    fear_greed_extreme_fear: float = 25
    fear_greed_extreme_greed: float = 75
    fear_greed_mode: Literal["contrarian", "trend"] = "contrarian"
    news_share: float = Field(0.5, ge=0, le=1)


class DirectionThresholds(_Strict):
    long: float = 30
    short: float = -30

    @model_validator(mode="after")
    def _signs(self) -> DirectionThresholds:
        if self.long <= 0 or self.short >= 0:
            raise ValueError("scoring.direction_thresholds: long tiene que ser > 0 y short < 0")
        return self


class ScoringConfig(_Strict):
    weights: ScoringWeights
    momentum: MomentumScoringConfig
    derivatives: DerivativesScoringConfig
    sentiment: SentimentScoringConfig
    direction_thresholds: DirectionThresholds


# --- plan, comisiones, liquidación ------------------------------------------


class TakeProfitConfig(_Strict):
    tp1: Literal["next_level"] = "next_level"
    tp2_r: float = 1.5
    tp3_r: float = 2.5


class PlanLiquidationConfig(_Strict):
    min_distance_ratio: float = 2.0
    max_margin_multiplier: float = 2.0


class TradePlanConfig(_Strict):
    stop_timeframes: list[Timeframe]
    stop_atr_buffer: float = 0.5
    take_profit: TakeProfitConfig
    flag_tp_beyond_daily_range: bool = True
    time_stop_hours: float = 24
    liquidation: PlanLiquidationConfig


class VenueFees(_Strict):
    taker_pct: float = Field(ge=0)
    maker_pct: float = Field(ge=0)


class FeesConfig(_Strict):
    nexo: VenueFees
    entry_order: Literal["taker", "maker"] = "taker"
    exit_order: Literal["taker", "maker"] = "taker"
    include_funding_estimate: bool = True
    funding_interval_hours: float = 8


class LiquidationConfig(_Strict):
    model: Literal["isolated"] = "isolated"
    maintenance_margin_pct: float = Field(gt=0, lt=100)
    show_wallet_model: bool = True


# --- perfiles ----------------------------------------------------------------


class RiskProfile(_Strict):
    monthly_wallet_usd: float = Field(gt=0)
    margin_per_trade_usd: float = Field(gt=0)
    risk_per_trade_usd: float = Field(gt=0)
    min_rr: float = Field(gt=0)
    max_leverage: float = Field(gt=0)
    max_trades_per_day: int = Field(ge=0)
    max_daily_loss_usd: float = Field(gt=0)
    min_operable_risk_usd: float = Field(gt=0)


class RiskProfilesConfig(_Strict):
    default: str
    leverage_hard_cap: float = 50
    leverage_red_warning_above: float = 25
    profiles: dict[str, RiskProfile]

    @model_validator(mode="after")
    def _check(self) -> RiskProfilesConfig:
        if self.default not in self.profiles:
            raise ValueError(f"risk_profiles.default='{self.default}' no existe en profiles")
        if self.leverage_hard_cap > 50:
            raise ValueError("risk_profiles.leverage_hard_cap no puede superar 50x")
        for name, p in self.profiles.items():
            if p.max_leverage > self.leverage_hard_cap:
                raise ValueError(
                    f"perfil '{name}': max_leverage {p.max_leverage}x supera el tope duro "
                    f"de {self.leverage_hard_cap}x"
                )
            if p.min_operable_risk_usd > p.risk_per_trade_usd:
                raise ValueError(f"perfil '{name}': min_operable_risk_usd > risk_per_trade_usd")
        return self


# --- ajustes por régimen -----------------------------------------------------


class DirectionAdjustment(_Strict):
    blocked: bool = False
    min_abs_score: float | None = None
    min_rr: float | None = None
    risk_multiplier: float = Field(1.0, gt=0, le=1)
    only_if_funding_very_high: bool = False


class RegimeDirectionAdjustments(_Strict):
    long: DirectionAdjustment = Field(default_factory=DirectionAdjustment)
    short: DirectionAdjustment = Field(default_factory=DirectionAdjustment)


class LowConfidenceAdjustment(_Strict):
    threshold: float = Field(0.60, gt=0, lt=1)
    risk_multiplier: float = Field(0.5, gt=0, le=1)


class UnstableAdjustment(_Strict):
    risk_multiplier: float = Field(0.5, gt=0, le=1)


class RegimeAdjustmentsConfig(_Strict):
    crash: RegimeDirectionAdjustments
    bajista: RegimeDirectionAdjustments
    neutral: RegimeDirectionAdjustments
    alcista: RegimeDirectionAdjustments
    euforia: RegimeDirectionAdjustments
    low_confidence: LowConfidenceAdjustment
    unstable: UnstableAdjustment
    apply_context_regime: bool = False


# --- seguridad ---------------------------------------------------------------


class ReduceBreaker(_Strict):
    risk_multiplier: float = Field(0.5, gt=0, le=1)


class DailyLossReduce(ReduceBreaker):
    pct_of_daily_limit: float = Field(gt=0, le=100)


class DailyLossBlock(_Strict):
    pct_of_daily_limit: float = Field(gt=0)


class LosingStreak(_Strict):
    consecutive_losses: int = Field(ge=1)
    cooldown_hours: float = Field(gt=0)


class WeeklyLossReduce(ReduceBreaker):
    pct_of_monthly_wallet: float = Field(gt=0)


class MonthlyDrawdownLock(_Strict):
    pct_from_peak: float = Field(gt=0, le=100)


class CircuitBreakersConfig(_Strict):
    daily_loss_reduce: DailyLossReduce
    daily_loss_block: DailyLossBlock
    losing_streak: LosingStreak
    weekly_loss_reduce: WeeklyLossReduce
    monthly_drawdown_lock: MonthlyDrawdownLock
    week_starts_on: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ] = "monday"


class CorrelationConfig(_Strict):
    timeframe: Timeframe = "1h"
    lookback_days: int = 30
    high_threshold: float = Field(0.8, gt=0, le=1)
    max_combined_risk_multiplier: float = Field(1.5, ge=1)


# --- paneles educativos ------------------------------------------------------


class MonteCarloConfig(_Strict):
    simulations: int = 200
    trades: int = 30
    default_win_rate: float = Field(0.45, ge=0, le=1)
    default_rr: float = Field(1.5, gt=0)
    percentiles: list[int] = Field(default_factory=lambda: [10, 50, 90])
    random_seed: int | None = None


class WalletMathConfig(_Strict):
    monte_carlo: MonteCarloConfig


class LeverageSimulatorConfig(_Strict):
    levels: list[float]
    noise_atr_mult: float = 1.0


# --- LLM, historial, validación ----------------------------------------------


class LlmConfig(_Strict):
    enabled: bool = True
    model: str
    max_tokens: int = 4000
    timeout_s: float = 90
    invalid_json_retries: int = Field(1, ge=0, le=3)
    headlines_in_prompt: int = 15
    horizon_hours: int = 24


class HistoryConfig(_Strict):
    outcome_window_hours: float = 24
    price_checkpoints_hours: list[float] = Field(default_factory=lambda: [1, 4, 24])


class StressTestConfig(_Strict):
    daily_shocks_pct: list[float]


class ValidationConfig(_Strict):
    train_months: int = 6
    test_months: int = 1
    total_months: int = 18
    entry_timeframe: Timeframe = "15m"
    slippage_pct_per_side: float = Field(0.05, ge=0)
    max_open_per_symbol: int = 1
    random_benchmark_runs: int = 100
    confidence_buckets: list[float] = Field(default_factory=lambda: [0.60, 0.80])
    stress_test: StressTestConfig


# --- raíz --------------------------------------------------------------------


class AppSettings(_Strict):
    """Espejo tipado de config.yaml."""

    app: AppConfig
    symbols: SymbolsConfig
    exchanges: ExchangesConfig
    cache_ttl_seconds: dict[str, float]
    data_sources: DataSourcesConfig
    timeframes: TimeframesConfig
    indicators: IndicatorsConfig
    regime: RegimeConfig
    scoring: ScoringConfig
    trade_plan: TradePlanConfig
    fees: FeesConfig
    liquidation: LiquidationConfig
    risk_profiles: RiskProfilesConfig
    regime_adjustments: RegimeAdjustmentsConfig
    circuit_breakers: CircuitBreakersConfig
    correlation: CorrelationConfig
    wallet_math: WalletMathConfig
    leverage_simulator: LeverageSimulatorConfig
    llm: LlmConfig
    history: HistoryConfig
    validation: ValidationConfig

    @model_validator(mode="after")
    def _cross_checks(self) -> AppSettings:
        missing = [s for s in self.symbols.favorites if s not in self.data_sources.news_keywords]
        if missing:
            raise ValueError(f"data_sources.news_keywords: faltan palabras para {missing}")
        if self.exchanges.nexo.allow_orders:
            # Fase 1: la app no puede operar. El conector igual se niega, pero
            # preferimos que la configuración tampoco lo sugiera.
            raise ValueError("exchanges.nexo.allow_orders tiene que ser false en la fase 1")
        return self

    def resolve_path(self, relative: str) -> Path:
        """Convierte una ruta de config.yaml (relativa a la raíz del repo) en absoluta."""
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


class Secrets(BaseSettings):
    """Secretos leídos de .env o del entorno. Nunca se exponen por la API."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_PATH, env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str | None = None
    cryptopanic_api_key: str | None = None


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> AppSettings:
    """Lee y valida config.yaml. Lanza ``ValueError`` con un mensaje claro si es inválido."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppSettings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Configuración global (cacheada) para la app FastAPI."""
    return load_settings()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()
