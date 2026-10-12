"""Régimen de mercado de una moneda, de punta a punta (Hito 3).

Junta las piezas del detector: baja las velas, carga o entrena el modelo,
calcula las probabilidades **filtradas** (algoritmo forward, cero look-ahead),
aplica el filtro de estabilidad y arma un ``RegimeReport`` con todo lo que
necesitan el panel, el plan de riesgo y el LLM.

Funciones principales
---------------------

- ``get_regime(symbol, timeframe, settings, market_data, ...) -> RegimeReport``:
  régimen de un timeframe (``regime.timeframe`` = 1h, el que se usa para
  operar, o ``regime.context_timeframe`` = 1d, el de contexto). Lanza
  ``RegimeUnavailable`` con el motivo en palabras simples si no se puede (no hay
  historia suficiente, el exchange no responde, el entrenamiento falló).
- ``get_market_regimes(symbol, settings, market_data, ...)``: los dos
  timeframes a la vez. Si uno falla, en su lugar queda su ``RegimeUnavailable``
  y el otro sale igual.
- ``ensure_model(...)``: solo la parte del modelo (cargar, decidir si
  reentrenar, entrenar y guardar). La usa ``scripts/train_regime.py``.
- ``short_training_warning(model, regime_cfg, timeframe)``: el aviso de "se
  entrenó con poca historia" (lo agrega ``ensure_model``).
- ``regimes_to_dict(results)``: el resultado de ``get_market_regimes`` listo
  para JSON.
- ``regime_history_days(regime_cfg)``: cuántos días de velas necesita el
  régimen (para el mercado sintético del modo ``--offline``).
- ``classify_bars(probs, states, ...)``: la parte pura (sin red ni disco) que
  pasa de probabilidades por vela a régimen crudo, confirmado y confianza.

Modelo en disco
---------------

Un archivo por moneda y timeframe en ``regime.models_dir``
(``data/models/BTC_1h.joblib``). En modo ``--offline`` (``synthetic=True``)
van a ``regime.synthetic_models_dir``: un modelo sintético nunca pisa ni se usa
como uno real (y al revés). Si las dos carpetas coinciden, o si en la carpeta
pedida hay un modelo del otro tipo, se corta con ``RegimeUnavailable`` antes de
entrenar.

Cuándo se entrena: si no hay modelo o no sirve (otra configuración, formato
viejo, otro tipo), siempre que ``allow_train`` lo permita; si está viejo
(``retrain_every_days``), solo con ``regime.auto_retrain``; con
``force_retrain``, siempre. Si está viejo y no se reentrena, se usa igual con un
aviso. Si reentrenar falla y el modelo anterior todavía sirve, se usa el
anterior con un aviso (``retrain_failed``). Si el modelo se entrenó con mucha
menos historia que la pedida (moneda nueva: menos de
``min_training_coverage``, 80 % por omisión, de las velas útiles esperadas) se
avisa con ``short_training``. El entrenamiento corre en un hilo aparte para no
frenar el servidor, y de a uno por vez en todo el proceso (el ajuste limita los
hilos de BLAS para ser reproducible, y dos ajustes simultáneos se pisarían ese
límite). Ese hilo es propio, no el ejecutor de ``asyncio``: si se cancela la
espera (Ctrl+C en la consola), ``asyncio.run`` no se queda esperando a que
termine el entrenamiento (que se descarta) y la consola sale enseguida con
``os._exit``. No es *daemon* a propósito: si el proceso termina normalmente
(por ejemplo, se apaga el servidor), Python espera a que el ajuste termine en
vez de cortarlo en medio del código C++ de hmmlearn, que aborta el proceso
("terminate called without an active exception").

La moneda tiene que ser solo letras y números (``RegimeUnavailable`` si no),
así la ruta del modelo nunca sale de la carpeta de modelos. Los comandos que se
sugieren para reentrenar llevan ``ARGS="--offline"`` cuando el modelo es
sintético.

Datos
-----

Siempre velas **cerradas** de ``market_data.get_klines_history``. Entrenamiento:
los últimos ``training_days`` (``context_training_days`` para el diario).
Cálculo del régimen: los últimos ``inference_days`` (``context_inference_days``);
si se acaba de entrenar se reusan las velas ya bajadas. De esas velas se sacan
las de calentamiento de las features, se filtra y se descartan las primeras
``burn_in_bars`` (el arranque del filtro, que todavía depende de la
probabilidad inicial). El régimen de cada vela usa solo esa vela y las
anteriores. Ojo: el modelo se entrena con la historia hasta el último
reentrenamiento, así que el fondo de los últimos días es "dentro de muestra"
para los parámetros (no para las probabilidades, que nunca miran velas
futuras). La validación fuera de muestra es del Hito 10.

Qué cuenta como "cambio" (``regime.change_level``)
--------------------------------------------------

- ``"category"`` (por defecto): la confirmación (``confirmation_bars`` velas
  seguidas) y la inestabilidad (más de ``instability_max_changes`` cambios en
  ``instability_window`` velas) miran la **categoría** (crash, bajista,
  neutral, alcista, euforia). La categoría cruda de cada vela es la de mayor
  probabilidad sumada entre sus estados. El nombre que se muestra es el del
  estado más probable dentro de la categoría vigente (por ejemplo "Alcista
  fuerte" dentro de "alcista"). Pasar de "Alcista" a "Alcista fuerte" no es un
  cambio.
- ``"state"``: todo mira el estado del HMM.

Mientras no hay ningún régimen confirmado (arranque), se informa el crudo con
``confirmed=False``.

Confianza
---------

``confidence`` es lo que pide la especificación: la **probabilidad filtrada del
estado más probable** en la vela (0-1; la UI la muestra como 0-100 %), sea o no
el régimen vigente. Es igual a ``top_state_prob`` y, vela por vela, a
``RegimeBar.state_prob``. ``low_confidence`` es ``confidence <
regime_adjustments.low_confidence.threshold`` (los ajustes de riesgo del Hito 4
lo usan junto con ``unstable`` y ``category``).

Aparte, ``regime_prob`` es la probabilidad del **régimen vigente** (en
``"category"``, la suma de los estados de su categoría; en ``"state"``, la del
estado; en el arranque, la del crudo que se muestra). Es solo un dato para
mostrar: baja mientras hay un candidato en espera aunque el modelo esté seguro
de ese candidato.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import math
import threading
import time
import weakref
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from app.data.snapshot import describe_failure
from app.exchanges.base import TIMEFRAME_MINUTES, ExchangeError, FetchMeta
from app.indicators.features import feature_matrix, warmup_bars
from app.regime.forward import filtered_from_model
from app.regime.labels import regime_stats_table
from app.regime.model import RegimeTrainingError, config_fingerprint, train_regime_model, utcnow
from app.regime.stability import confirmed_series, summarize_stability, unstable_series
from app.regime.store import load_model, model_path, needs_retrain, save_model
from app.regime.types import RegimeModel, StateInfo

if TYPE_CHECKING:  # solo para tipos
    from app.config import AppSettings, RegimeConfig
    from app.exchanges.base import MarketDataSource

logger = logging.getLogger(__name__)

# Nombre de cada categoría para el panel (texto de la UI, no es una regla).
CATEGORY_NAMES: dict[str, str] = {
    "crash": "Crash",
    "bajista": "Bajista",
    "neutral": "Neutral",
    "alcista": "Alcista",
    "euforia": "Euforia",
}
UNCONFIRMED_NAME = "Sin confirmar"

# Para preguntarle a ``needs_retrain`` solo por compatibilidad (sin mirar la edad).
_NEVER_STALE_DAYS = 10**9

# Un solo entrenamiento a la vez en todo el proceso (ver docstring del módulo).
_TRAIN_LOCK = threading.Lock()

# Si el modelo se entrenó con menos de esta fracción de las velas útiles que
# pide la config (training_days menos el calentamiento), se avisa
# (``short_training``). Es solo un aviso, no cambia ninguna regla de riesgo.
# Pedido para config: regime.min_training_coverage.
DEFAULT_MIN_TRAINING_COVERAGE = 0.8

_T = TypeVar("_T")

# Un candado por archivo de modelo y por event loop: si dos pedidos llegan
# juntos, el segundo espera y usa el modelo que guardó el primero.
_FILE_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]] = (
    weakref.WeakKeyDictionary()
)

Notify = Callable[[str], None]


class RegimeUnavailable(Exception):
    """No se pudo calcular el régimen. ``reason`` explica por qué, en palabras simples."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RegimeWarning:
    """Un aviso para el usuario: ``code`` para el código y ``text`` en español simple.

    Códigos: ``stale_model``, ``retrain_failed``, ``save_failed``,
    ``short_training``, ``unconfirmed``, ``pending``, ``low_confidence``,
    ``unstable``, ``outliers``, ``data_lag``, ``data_source``,
    ``short_history``.
    """

    code: str
    text: str


# --- utilidades ----------------------------------------------------------------


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _timestamp_utc(ts: Any) -> datetime:
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _velas(n: int) -> str:
    return _plural(n, "vela", "velas")


def _pct_text(fraction: float) -> str:
    return f"{fraction * 100:.0f} %"


def _candles_text(n: int, timeframe: str) -> str:
    """``"250 velas diarias"``, ``"327 velas de 1 hora"`` o ``"40 velas de 4h"``."""
    words = {"1h": ("vela de 1 hora", "velas de 1 hora"), "1d": ("vela diaria", "velas diarias")}
    singular, plural = words.get(timeframe, (f"vela de {timeframe}", f"velas de {timeframe}"))
    return _plural(n, singular, plural)


def _train_hint(symbol: str, synthetic: bool) -> str:
    """Comando para reentrenar; con ``--offline`` si el modelo es sintético."""
    extra = ' ARGS="--offline"' if synthetic else ""
    return f"make train-regime SYMBOL={symbol}{extra}"


def _valid_symbol(symbol: str) -> str:
    """Moneda en mayúsculas si es solo letras y números; si no, ``RegimeUnavailable``."""
    sym = str(symbol).strip().upper()
    if not (sym and sym.isascii() and sym.isalnum()):
        raise RegimeUnavailable(f"símbolo inválido: {symbol!r} (usá algo como BTC)")
    return sym


def _json_safe(obj: Any) -> Any:
    """Tipos de JSON: NaN/inf → None, datetime → ISO UTC, numpy → Python, claves a str."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, datetime):
        return _as_utc(obj).isoformat()
    if isinstance(obj, pd.Timestamp):
        return _timestamp_utc(obj).isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def timeframe_minutes(timeframe: str) -> int:
    """Minutos de una vela del timeframe (``ValueError`` si no se conoce)."""
    try:
        return TIMEFRAME_MINUTES[timeframe]
    except KeyError:
        raise ValueError(f"timeframe desconocido: {timeframe!r}") from None


def _is_context(regime_cfg: RegimeConfig, timeframe: str) -> bool:
    """True si ``timeframe`` es el de contexto; ``ValueError`` si no es ninguno de los dos."""
    if timeframe == regime_cfg.timeframe:
        return False
    if timeframe == regime_cfg.context_timeframe:
        return True
    raise ValueError(
        f"el régimen se calcula en {regime_cfg.timeframe} (para operar) y "
        f"{regime_cfg.context_timeframe} (contexto); se pidió {timeframe!r}"
    )


def _training_days(regime_cfg: RegimeConfig, timeframe: str) -> int:
    ctx = _is_context(regime_cfg, timeframe)
    return regime_cfg.context_training_days if ctx else regime_cfg.training_days


def _inference_days(regime_cfg: RegimeConfig, timeframe: str) -> int:
    ctx = _is_context(regime_cfg, timeframe)
    return regime_cfg.context_inference_days if ctx else regime_cfg.inference_days


def regime_history_days(regime_cfg: RegimeConfig) -> int:
    """Días de velas que hacen falta para entrenar y calcular los dos regímenes.

    Sirve para armar el mercado sintético del modo ``--offline``
    (``SyntheticMarketData(min_history_days=...)``).
    """
    return (
        max(
            regime_cfg.training_days,
            regime_cfg.context_training_days,
            regime_cfg.inference_days,
            regime_cfg.context_inference_days,
        )
        + 1
    )


def resolve_models_dir(
    settings: AppSettings, *, synthetic: bool, models_dir: Path | str | None = None
) -> Path:
    """Carpeta de modelos: la pasada a mano, o la de config según real/sintético.

    Lanza ``RegimeUnavailable`` si la carpeta sintética es la misma que la real.
    """
    if models_dir is not None:
        return Path(models_dir)
    cfg = settings.regime
    real = settings.resolve_path(cfg.models_dir)
    if not synthetic:
        return real
    synth = settings.resolve_path(cfg.synthetic_models_dir)
    if synth.resolve() == real.resolve():
        raise RegimeUnavailable(
            "regime.synthetic_models_dir tiene que ser una carpeta distinta de "
            "regime.models_dir: los modelos de prueba no pueden pisar a los reales"
        )
    return synth


def _lock_for(path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    per_loop = _FILE_LOCKS.get(loop)
    if per_loop is None:
        per_loop = {}
        _FILE_LOCKS[loop] = per_loop
    key = path.resolve()
    if key not in per_loop:
        per_loop[key] = asyncio.Lock()
    return per_loop[key]


# --- datos -----------------------------------------------------------------------


@dataclass(frozen=True)
class _History:
    df: pd.DataFrame  # solo velas cerradas hasta ``now``
    meta: FetchMeta | None


async def _fetch_closed(
    market_data: MarketDataSource,
    symbol: str,
    timeframe: str,
    start: datetime,
    now: datetime,
) -> _History:
    """Velas cerradas entre ``start`` y ``now``; errores → ``RegimeUnavailable``."""
    try:
        candles = await market_data.get_klines_history(symbol, timeframe, start)  # type: ignore[arg-type]
    except ExchangeError as exc:
        raise RegimeUnavailable(
            f"no se pudieron bajar las velas de {timeframe} de {symbol} ({describe_failure(exc)})"
        ) from exc
    except ValueError as exc:
        raise RegimeUnavailable(
            f"no se pudieron pedir las velas de {timeframe} de {symbol}: {exc}"
        ) from exc
    df = candles.df
    if "closed" in df.columns:
        df = df[df["closed"].astype(bool)]
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    df = df[df.index + step <= pd.Timestamp(now)]  # cerradas a la hora ``now``
    return _History(df=df, meta=candles.meta)


def _features(df: pd.DataFrame, regime_cfg: RegimeConfig) -> tuple[np.ndarray, pd.DatetimeIndex]:
    f = regime_cfg.features
    try:
        return feature_matrix(df, f.realized_vol_window, f.volume_zscore_window, f.smoothing_window)
    except ValueError as exc:
        raise RegimeUnavailable(f"las velas recibidas tienen datos inválidos: {exc}") from exc


def _warmup(regime_cfg: RegimeConfig) -> int:
    f = regime_cfg.features
    return warmup_bars(f.realized_vol_window, f.volume_zscore_window, f.smoothing_window)


# --- modelo ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStatus:
    """Resultado de ``ensure_model``: el modelo a usar y qué se hizo con él."""

    model: RegimeModel
    path: Path
    retrained_now: bool
    reason: str  # por qué se entrenó, o en qué estado está el guardado
    stale: bool  # quedó viejo y se usa igual
    train_seconds: float | None = None  # solo si se entrenó ahora
    candles_downloaded: int | None = None  # velas cerradas bajadas para entrenar
    warnings: tuple[RegimeWarning, ...] = ()
    history: _History | None = field(default=None, repr=False)  # velas del entrenamiento


def _expected_wait(regime_cfg: RegimeConfig, timeframe: str) -> str:
    bars = _training_days(regime_cfg, timeframe) * 1440 // timeframe_minutes(timeframe)
    return "un par de minutos" if bars > 5000 else "unos segundos"


def _train_sync(
    history: _History,
    symbol: str,
    timeframe: str,
    regime_cfg: RegimeConfig,
    synthetic: bool,
    now: datetime,
) -> RegimeModel:
    """Features + entrenamiento (corre en un hilo; de a uno por vez)."""
    X, index = _features(history.df, regime_cfg)
    if len(X) < regime_cfg.min_training_samples:
        warm = _warmup(regime_cfg)
        need = regime_cfg.min_training_samples
        raise RegimeUnavailable(
            f"no hay suficiente historia para entrenar el régimen de {symbol} {timeframe}: "
            f"hay {_candles_text(len(history.df), timeframe)} y hacen falta al menos "
            f"{need + warm} (las primeras {warm} solo sirven para calcular los indicadores "
            f"y después se necesitan {need} para entrenar)"
        )
    with _TRAIN_LOCK:
        try:
            return train_regime_model(
                X, index, symbol, timeframe, regime_cfg, synthetic=synthetic, clock=lambda: now
            )
        except RegimeTrainingError as exc:
            raise RegimeUnavailable(
                f"no se pudo entrenar el modelo de régimen de {symbol} {timeframe}: {exc}"
            ) from exc


def _model_date(model: RegimeModel) -> str:
    return _as_utc(model.trained_at).strftime("%d/%m/%Y")


async def _run_in_own_thread(fn: Callable[..., _T], *args: Any) -> _T:
    """Corre ``fn(*args)`` en un hilo propio y espera el resultado sin frenar el loop.

    Es como ``asyncio.to_thread``, pero sin el ejecutor por defecto: si se
    cancela la espera (Ctrl+C en la consola), ``asyncio.run`` cierra sin
    esperar a que el hilo termine. El hilo sigue solo hasta terminar y su
    resultado se descarta. No es *daemon* (ver el docstring del módulo).
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def deliver(ok: bool, value: Any) -> None:
        if future.done():  # la espera se canceló mientras entrenaba
            return
        if ok:
            future.set_result(value)
        else:
            future.set_exception(value)

    def target() -> None:
        try:
            outcome: tuple[bool, Any] = (True, fn(*args))
        except BaseException as exc:  # se le entrega al que espera
            outcome = (False, exc)
        with contextlib.suppress(RuntimeError):  # el loop ya cerró: nadie espera
            loop.call_soon_threadsafe(deliver, *outcome)

    threading.Thread(target=target, name="regime-train", daemon=False).start()
    return await future


def short_training_warning(
    model: RegimeModel,
    regime_cfg: RegimeConfig,
    timeframe: str,
    min_coverage: float = DEFAULT_MIN_TRAINING_COVERAGE,
) -> RegimeWarning | None:
    """Aviso ``short_training`` si el modelo se entrenó con mucha menos historia que la pedida.

    Velas útiles esperadas: ``training_days`` (o ``context_training_days``) por
    las velas de un día, menos el calentamiento de las features. Si el modelo
    tiene menos de ``min_coverage`` de eso (moneda listada hace poco), avisa.
    """
    bars_per_day = max(1, 1440 // timeframe_minutes(timeframe))
    wanted_days = _training_days(regime_cfg, timeframe)
    warm = _warmup(regime_cfg)
    expected = wanted_days * bars_per_day - warm
    if expected <= 0 or model.n_samples >= min_coverage * expected:
        return None
    used_days = int((model.n_samples + warm) / bars_per_day)
    return RegimeWarning(
        "short_training",
        f"El modelo de régimen de {timeframe} se entrenó con solo "
        f"{_plural(used_days, 'día', 'días')} de historia y se piden {wanted_days}: "
        "puede ser poco para reconocer bien los regímenes, tomalo con cuidado.",
    )


async def ensure_model(
    symbol: str,
    timeframe: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    *,
    models_dir: Path | str | None = None,
    synthetic: bool = False,
    clock: Callable[[], datetime] = utcnow,
    allow_train: bool = True,
    force_retrain: bool = False,
    retrain_if_stale: bool | None = None,
    notify: Notify | None = None,
    min_training_coverage: float = DEFAULT_MIN_TRAINING_COVERAGE,
) -> ModelStatus:
    """Carga el modelo de ``symbol``/``timeframe`` y lo (re)entrena si corresponde.

    ``retrain_if_stale``: si un modelo viejo se reentrena (por omisión,
    ``regime.auto_retrain``). ``notify`` recibe un aviso corto antes de entrenar
    (la consola lo imprime en stderr). ``min_training_coverage``: ver
    ``short_training_warning``. Lanza ``RegimeUnavailable`` (también si la
    moneda no es solo letras y números: se valida antes de tocar el disco).
    """
    sym = _valid_symbol(symbol)
    status = await _ensure_model(
        sym,
        timeframe,
        settings,
        market_data,
        models_dir=models_dir,
        synthetic=synthetic,
        clock=clock,
        allow_train=allow_train,
        force_retrain=force_retrain,
        retrain_if_stale=retrain_if_stale,
        notify=notify,
    )
    extra = short_training_warning(status.model, settings.regime, timeframe, min_training_coverage)
    if extra is None:
        return status
    return dataclasses.replace(status, warnings=(*status.warnings, extra))


async def _ensure_model(
    sym: str,
    timeframe: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    *,
    models_dir: Path | str | None,
    synthetic: bool,
    clock: Callable[[], datetime],
    allow_train: bool,
    force_retrain: bool,
    retrain_if_stale: bool | None,
    notify: Notify | None,
) -> ModelStatus:
    """``ensure_model`` sin la validación de la moneda ni el aviso de poca historia."""
    cfg = settings.regime
    _is_context(cfg, timeframe)  # ValueError si el timeframe no es del régimen
    now = _as_utc(clock())
    fingerprint = config_fingerprint(cfg, timeframe)
    directory = resolve_models_dir(settings, synthetic=synthetic, models_dir=models_dir)
    path = model_path(directory, sym, timeframe)
    stale_policy = cfg.auto_retrain if retrain_if_stale is None else retrain_if_stale

    async with _lock_for(path):
        existing = load_model(path)
        if existing is not None and bool(existing.synthetic) != bool(synthetic):
            kind = "sintético (de prueba)" if existing.synthetic else "real"
            raise RegimeUnavailable(
                f"en {directory} hay un modelo {kind} de {sym} {timeframe}; los modelos reales "
                "y los sintéticos tienen que ir en carpetas distintas (regime.models_dir y "
                "regime.synthetic_models_dir), así uno nunca pisa al otro"
            )
        check = {"symbol": sym, "timeframe": timeframe, "synthetic": synthetic}
        incompatible, why_incompatible = needs_retrain(
            existing, now, _NEVER_STALE_DAYS, fingerprint, **check
        )
        stale, why_stale = needs_retrain(
            existing, now, cfg.retrain_every_days, fingerprint, **check
        )

        if force_retrain:
            reason = "se pidió reentrenar"
        elif incompatible:
            if not allow_train:
                raise RegimeUnavailable(
                    f"no hay un modelo de régimen usable para {sym} {timeframe} porque "
                    f"{why_incompatible}; entrenalo con: {_train_hint(sym, synthetic)}"
                )
            reason = why_incompatible
        elif stale and stale_policy and allow_train:
            reason = why_stale
        else:
            assert existing is not None
            warnings: list[RegimeWarning] = []
            if stale:
                why_not = (
                    "regime.auto_retrain está apagado"
                    if not stale_policy
                    else "no se permitió entrenar ahora"
                )
                warnings.append(
                    RegimeWarning(
                        "stale_model",
                        f"El modelo de régimen de {timeframe} está viejo: {why_stale}. Se usa "
                        f"igual porque {why_not}; reentrenalo con: {_train_hint(sym, synthetic)}",
                    )
                )
            return ModelStatus(
                model=existing,
                path=path,
                retrained_now=False,
                reason=why_stale,
                stale=stale,
                warnings=tuple(warnings),
            )

        if notify is not None:
            notify(
                f"Entrenando el modelo de régimen de {sym} {timeframe} porque {reason}. "
                f"Puede tardar {_expected_wait(cfg, timeframe)}."
            )
        usable_old = existing if (existing is not None and not incompatible) else None
        started = time.perf_counter()
        try:
            start = now - timedelta(days=_training_days(cfg, timeframe))
            history = await _fetch_closed(market_data, sym, timeframe, start, now)
            model = await _run_in_own_thread(
                _train_sync, history, sym, timeframe, cfg, synthetic, now
            )
        except RegimeUnavailable as exc:
            if usable_old is None:
                raise
            logger.warning("No se pudo reentrenar %s %s: %s", sym, timeframe, exc.reason)
            return ModelStatus(
                model=usable_old,
                path=path,
                retrained_now=False,
                reason=why_stale,
                stale=stale,
                warnings=(
                    RegimeWarning(
                        "retrain_failed",
                        f"No se pudo reentrenar el modelo de régimen de {timeframe}: "
                        f"{exc.reason}. Se usa el anterior, del {_model_date(usable_old)}.",
                    ),
                ),
            )
        seconds = time.perf_counter() - started
        warnings = []
        try:
            save_model(model, path)
        except OSError as exc:
            logger.warning("No se pudo guardar el modelo %s: %s", path, exc)
            warnings.append(
                RegimeWarning(
                    "save_failed",
                    f"El modelo de régimen de {timeframe} se entrenó pero no se pudo guardar en "
                    f"{path} ({exc}); se va a volver a entrenar la próxima vez.",
                )
            )
        return ModelStatus(
            model=model,
            path=path,
            retrained_now=True,
            reason=reason,
            stale=False,
            train_seconds=seconds,
            candles_downloaded=len(history.df),
            warnings=tuple(warnings),
            history=history,
        )


# --- de probabilidades a régimen ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class BarClassification:
    """Régimen de cada vela según ``change_level`` (todas las listas miden T).

    - ``raw_state`` / ``raw_state_prob``: estado más probable y su probabilidad.
    - ``confidence`` (propiedad): la confianza de la especificación, la
      probabilidad del estado más probable (igual a ``raw_state_prob``).
    - ``raw_label`` / ``raw_label_prob``: lo que mira el filtro de estabilidad
      (categoría o índice de estado) y su probabilidad (sumada si es categoría).
    - ``confirmed_label``: régimen vigente (``None`` en el arranque).
    - ``regime_prob``: probabilidad del régimen vigente (sumada si es
      categoría; ``None`` si no hay). Solo para mostrar.
    - ``display_state``: estado que le da nombre al régimen vigente (el más
      probable dentro de la categoría; ``None`` si no hay).
    - ``unstable``: más de ``instability_max_changes`` cambios en la ventana.
    """

    change_level: str
    raw_state: list[int]
    raw_state_prob: list[float]
    raw_label: list[Hashable]
    raw_label_prob: list[float]
    confirmed_label: list[Hashable | None]
    regime_prob: list[float | None]
    display_state: list[int | None]
    unstable: list[bool]

    @property
    def confidence(self) -> list[float]:
        """Confianza por vela: probabilidad del estado más probable (fracción 0-1)."""
        return self.raw_state_prob


def _category_order(states: Sequence[StateInfo]) -> list[str]:
    """Categorías presentes, de la más bajista a la más alcista (por rango)."""
    order: list[str] = []
    for s in sorted(states, key=lambda s: s.rank):
        if s.category not in order:
            order.append(s.category)
    return order


def _best_state_in(probs_row: np.ndarray, members: Sequence[int]) -> int:
    """Estado más probable entre ``members`` (empate: el primero)."""
    return int(members[int(np.argmax(probs_row[list(members)]))])


def classify_bars(
    probs: np.ndarray,
    states: Sequence[StateInfo],
    *,
    change_level: str,
    confirmation_bars: int,
    instability_window: int,
    instability_max_changes: int,
) -> BarClassification:
    """Pasa de probabilidades filtradas ``(T, K)`` a régimen por vela (sin mirar el futuro).

    ``states`` indexados por índice del HMM (``RegimeModel.states``). Cada vela
    depende solo de su fila y de las anteriores (lo hereda del filtro de
    estabilidad), así que agregar velas no cambia lo ya calculado.
    """
    P = np.asarray(probs, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(states):
        raise ValueError(f"probs tiene forma {P.shape} y hay {len(states)} estados")
    if change_level not in ("category", "state"):
        raise ValueError(f"change_level desconocido: {change_level!r}")
    T = P.shape[0]
    raw_state = np.argmax(P, axis=1) if T else np.zeros(0, dtype=int)
    raw_state_prob = P[np.arange(T), raw_state] if T else np.zeros(0)

    if change_level == "category":
        cats = _category_order(states)
        members = {c: [s.index for s in states if s.category == c] for c in cats}
        cat_probs = np.column_stack([P[:, members[c]].sum(axis=1) for c in cats])
        raw_cat_idx = np.argmax(cat_probs, axis=1) if T else np.zeros(0, dtype=int)
        raw_label: list[Hashable] = [cats[i] for i in raw_cat_idx]
        raw_label_prob = cat_probs[np.arange(T), raw_cat_idx] if T else np.zeros(0)
        confirmed = confirmed_series(raw_label, confirmation_bars)
        regime_prob: list[float | None] = []
        display: list[int | None] = []
        for t, label in enumerate(confirmed):
            if label is None:
                regime_prob.append(None)
                display.append(None)
            else:
                regime_prob.append(float(cat_probs[t, cats.index(label)]))
                display.append(_best_state_in(P[t], members[label]))
    else:
        raw_label = [int(k) for k in raw_state]
        raw_label_prob = raw_state_prob
        confirmed = confirmed_series(raw_label, confirmation_bars)
        regime_prob = [None if k is None else float(P[t, k]) for t, k in enumerate(confirmed)]
        display = [None if k is None else int(k) for k in confirmed]

    return BarClassification(
        change_level=change_level,
        raw_state=[int(k) for k in raw_state],
        raw_state_prob=[float(p) for p in raw_state_prob],
        raw_label=raw_label,
        raw_label_prob=[float(p) for p in raw_label_prob],
        confirmed_label=confirmed,
        regime_prob=regime_prob,
        display_state=display,
        unstable=unstable_series(raw_label, instability_window, instability_max_changes),
    )


def _label_name(label: Hashable, change_level: str, states: Sequence[StateInfo]) -> str:
    if change_level == "category":
        return CATEGORY_NAMES.get(str(label), str(label).capitalize())
    return states[int(label)].name  # type: ignore[call-overload]


def _label_category(label: Hashable, change_level: str, states: Sequence[StateInfo]) -> str:
    if change_level == "category":
        return str(label)
    return states[int(label)].category  # type: ignore[call-overload]


def _label_display_state(
    label: Hashable, change_level: str, states: Sequence[StateInfo], probs_row: np.ndarray
) -> int:
    """Estado que da nombre a ``label`` en una vela (el más probable de la categoría)."""
    if change_level == "category":
        members = [s.index for s in states if s.category == label]
        return _best_state_in(probs_row, members)
    return int(label)  # type: ignore[call-overload]


# --- reporte ---------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeModelInfo:
    """Datos del modelo usado, para mostrar."""

    trained_at: datetime
    age_days: float
    retrain_every_days: int
    n_states: int
    selection_criterion: str
    bic_by_n: dict[int, float]
    train_start: datetime
    train_end: datetime
    n_samples: int
    synthetic: bool
    retrained_now: bool
    retrain_reason: str
    stale: bool
    path: str
    notes: tuple[str, ...] = ()
    train_seconds: float | None = None


@dataclass(frozen=True)
class RegimePending:
    """Un régimen nuevo que todavía no se confirmó."""

    name: str
    category: str
    bars: int  # velas seguidas que lleva
    needed: int  # velas seguidas que necesita (confirmation_bars)


@dataclass(frozen=True)
class RegimeBar:
    """Una vela del historial (fondo del gráfico y línea de confianza)."""

    time: datetime  # apertura de la vela, UTC
    state: str  # estado más probable (crudo)
    state_category: str
    state_prob: float  # su probabilidad filtrada (0-1)
    regime: str | None  # régimen vigente (confirmado), None en el arranque
    category: str | None
    confidence: float  # confianza: probabilidad del estado más probable (= state_prob)
    regime_prob: float | None  # probabilidad del régimen vigente (None en el arranque)
    unstable: bool


@dataclass(frozen=True)
class RegimeReport:
    """Todo sobre el régimen de una moneda en un timeframe. ``to_dict()`` lo pasa a JSON."""

    symbol: str
    timeframe: str
    timeframe_minutes: int
    is_context: bool  # True = régimen diario de contexto
    change_level: str
    as_of: datetime  # apertura de la última vela cerrada usada (UTC)
    bars_analyzed: int  # velas después del calentamiento y del arranque del filtro
    model: RegimeModelInfo
    # régimen actual
    name: str
    category: str
    confidence: float  # 0-1: probabilidad del estado más probable (= top_state_prob)
    regime_prob: float  # 0-1: probabilidad del régimen actual (solo para mostrar)
    confirmed: bool  # False solo en el arranque (se muestra el crudo)
    pending: RegimePending | None
    unstable: bool
    changes_in_window: int
    instability_window: int
    instability_max_changes: int
    confirmation_bars: int
    age_bars: int  # velas desde que empezó (incluye las de confirmación)
    started_at: datetime | None  # apertura de la primera vela del régimen actual
    age_is_lower_bound: bool  # empezó antes de las velas analizadas
    confirmed_bars: int  # velas desde que quedó confirmado
    top_state_name: str  # estado más probable en la última vela
    top_state_category: str
    top_state_prob: float
    low_confidence: bool
    low_confidence_threshold: float
    # tablas
    probabilities: tuple[dict[str, Any], ...]  # por estado (rango), última vela
    states: tuple[dict[str, Any], ...]  # labels.regime_stats_table + duración en horas
    transition_names: tuple[str, ...]
    transition_matrix: tuple[tuple[float, ...], ...]  # filas/columnas por rango
    distribution: tuple[dict[str, Any], ...]  # % de velas por régimen vigente
    history: tuple[RegimeBar, ...]
    warnings: tuple[RegimeWarning, ...]
    data_source: str | None = None

    @property
    def confidence_pct(self) -> float:
        return self.confidence * 100.0

    @property
    def role(self) -> str:
        return "context" if self.is_context else "main"

    def history_summary(self) -> dict[str, Any] | None:
        """Resumen de la historia mostrada: cuántas veces cambió y qué tan estable fue.

        ``regime_changes`` cuenta cambios del régimen vigente (de tipo o de estado
        según ``change_level``), ``avg_regime_hours`` es la duración media de cada
        tramo, ``unstable_pct`` y ``low_confidence_pct`` el % de velas inestables y
        con confianza debajo del umbral. Sirve para ver de un vistazo si el
        régimen se comporta bien con datos reales. ``None`` sin historia.
        """
        bars = self.history
        if not bars:
            return None
        labels = [b.category if self.change_level == "category" else b.regime for b in bars]
        known = [lb for lb in labels if lb is not None]
        changes = sum(1 for a, b in zip(known, known[1:], strict=False) if a != b)
        hours = len(bars) * self.timeframe_minutes / 60.0
        return {
            "bars": len(bars),
            "days": hours / 24.0,
            "regime_changes": changes,
            "avg_regime_hours": hours / (changes + 1),
            "unstable_pct": 100.0 * sum(b.unstable for b in bars) / len(bars),
            "low_confidence_pct": 100.0
            * sum(b.confidence < self.low_confidence_threshold for b in bars)
            / len(bars),
        }

    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        """Dict listo para JSON (sin NaN ni infinitos: se vuelven ``None``)."""
        m = self.model
        out: dict[str, Any] = {
            "available": True,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timeframe_minutes": self.timeframe_minutes,
            "role": self.role,
            "is_context": self.is_context,
            "change_level": self.change_level,
            "as_of": self.as_of,
            "bars_analyzed": self.bars_analyzed,
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
            "confidence_pct": self.confidence_pct,
            "regime_prob": self.regime_prob,
            "confirmed": self.confirmed,
            "pending": None
            if self.pending is None
            else {
                "name": self.pending.name,
                "category": self.pending.category,
                "bars": self.pending.bars,
                "needed": self.pending.needed,
            },
            "unstable": self.unstable,
            "changes_in_window": self.changes_in_window,
            "instability_window": self.instability_window,
            "instability_max_changes": self.instability_max_changes,
            "confirmation_bars": self.confirmation_bars,
            "age_bars": self.age_bars,
            "started_at": self.started_at,
            "age_is_lower_bound": self.age_is_lower_bound,
            "confirmed_bars": self.confirmed_bars,
            "top_state": {
                "name": self.top_state_name,
                "category": self.top_state_category,
                "prob": self.top_state_prob,
            },
            "low_confidence": self.low_confidence,
            "low_confidence_threshold": self.low_confidence_threshold,
            "model": {
                "trained_at": m.trained_at,
                "age_days": m.age_days,
                "retrain_every_days": m.retrain_every_days,
                "n_states": m.n_states,
                "selection_criterion": m.selection_criterion,
                "bic_by_n": {int(k): v for k, v in sorted(m.bic_by_n.items())},
                "train_start": m.train_start,
                "train_end": m.train_end,
                "n_samples": m.n_samples,
                "synthetic": m.synthetic,
                "retrained_now": m.retrained_now,
                "retrain_reason": m.retrain_reason,
                "stale": m.stale,
                "path": m.path,
                "notes": list(m.notes),
                "train_seconds": m.train_seconds,
            },
            "probabilities": list(self.probabilities),
            "states": list(self.states),
            "transition_names": list(self.transition_names),
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "distribution": list(self.distribution),
            "history_bars": len(self.history),
            "history_summary": self.history_summary(),
            "warnings": [w.text for w in self.warnings],
            "warning_codes": [w.code for w in self.warnings],
            "data_source": self.data_source,
        }
        if include_history:
            out["history"] = [
                {
                    "time": b.time,
                    "state": b.state,
                    "state_category": b.state_category,
                    "state_prob": b.state_prob,
                    "regime": b.regime,
                    "category": b.category,
                    "confidence": b.confidence,
                    "regime_prob": b.regime_prob,
                    "unstable": b.unstable,
                }
                for b in self.history
            ]
        return _json_safe(out)


def _distribution(
    labels: Sequence[Hashable | None], change_level: str, states: Sequence[StateInfo]
) -> tuple[dict[str, Any], ...]:
    """% de velas en cada régimen vigente, del más bajista al más alcista."""
    total = len(labels)
    if change_level == "category":
        keys: list[Hashable] = list(_category_order(states))
    else:
        keys = [s.index for s in sorted(states, key=lambda s: s.rank)]
    counts = {k: 0 for k in keys}
    unconfirmed = 0
    for label in labels:
        if label is None:
            unconfirmed += 1
        else:
            counts[label] = counts.get(label, 0) + 1
    rows: list[dict[str, Any]] = [
        {
            "name": _label_name(k, change_level, states),
            "category": _label_category(k, change_level, states),
            "bars": counts[k],
            "pct": counts[k] / total * 100.0 if total else 0.0,
        }
        for k in keys
    ]
    if unconfirmed:
        rows.append(
            {
                "name": UNCONFIRMED_NAME,
                "category": None,
                "bars": unconfirmed,
                "pct": unconfirmed / total * 100.0,
            }
        )
    return tuple(rows)


def _states_table(
    model: RegimeModel, tf_minutes: int, current_state: int
) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in regime_stats_table(model):
        bars = row["expected_duration_bars"]
        rows.append(
            {
                **row,
                "expected_duration_hours": None if bars is None else bars * tf_minutes / 60.0,
                "is_current": row["index"] == current_state,
            }
        )
    return tuple(rows)


def build_report(
    *,
    symbol: str,
    timeframe: str,
    settings: AppSettings,
    status: ModelStatus,
    probs: np.ndarray,
    times: pd.DatetimeIndex,
    now: datetime,
    n_degenerate: int = 0,
    data_meta: FetchMeta | None = None,
    extra_warnings: Sequence[RegimeWarning] = (),
) -> RegimeReport:
    """Arma el ``RegimeReport`` con las probabilidades filtradas ya sin el arranque."""
    cfg = settings.regime
    model = status.model
    states = model.states
    tf_minutes = timeframe_minutes(timeframe)
    bars_per_day = max(1, 1440 // tf_minutes)
    n_bars = len(times)
    if n_bars == 0:
        raise RegimeUnavailable(f"no quedaron velas para calcular el régimen de {timeframe}")
    level = cfg.change_level
    bc = classify_bars(
        probs,
        states,
        change_level=level,
        confirmation_bars=cfg.confirmation_bars,
        instability_window=cfg.instability_window,
        instability_max_changes=cfg.instability_max_changes,
    )
    summary = summarize_stability(
        bc.raw_label,
        bars=cfg.confirmation_bars,
        window=cfg.instability_window,
        max_changes=cfg.instability_max_changes,
    )
    last = np.asarray(probs[-1], dtype=float)
    top = states[bc.raw_state[-1]]
    confidence = float(bc.confidence[-1])  # la del estado más probable (especificación)
    warnings: list[RegimeWarning] = list(status.warnings) + list(extra_warnings)

    # Régimen actual: el confirmado; en el arranque, el crudo.
    confirmed = summary.confirmed is not None
    if confirmed:
        label = summary.confirmed
        display = states[bc.display_state[-1]]  # type: ignore[index]
        regime_prob = float(bc.regime_prob[-1])  # type: ignore[arg-type]
        confirmed_bars = summary.age_bars
        start_idx = n_bars - confirmed_bars - (cfg.confirmation_bars - 1)
    else:
        label = bc.raw_label[-1]
        display = states[_label_display_state(label, level, states, last)]
        regime_prob = float(bc.raw_label_prob[-1])
        confirmed_bars = 0
        run = summary.pending[1] if summary.pending else 1
        start_idx = n_bars - run
        warnings.append(
            RegimeWarning(
                "unconfirmed",
                "Todavía no hay un régimen confirmado (ninguno se mantuvo "
                f"{_velas(cfg.confirmation_bars)} seguidas); se muestra el más probable.",
            )
        )
    age_is_lower_bound = start_idx <= 0
    start_idx = max(start_idx, 0)
    age_bars = n_bars - start_idx
    name = display.name
    category = _label_category(label, level, states)

    pending = None
    if confirmed and summary.pending is not None:
        cand_label, held = summary.pending
        cand_state = states[_label_display_state(cand_label, level, states, last)]
        pending = RegimePending(
            name=cand_state.name,
            category=_label_category(cand_label, level, states),
            bars=int(held),
            needed=cfg.confirmation_bars,
        )
        warnings.append(
            RegimeWarning(
                "pending",
                f"Posible cambio de régimen: hace {_velas(pending.bars)} que el modelo ve "
                f"'{pending.name}' (se confirma con {_velas(pending.needed)} seguidas).",
            )
        )

    threshold = settings.regime_adjustments.low_confidence.threshold
    low_confidence = confidence < threshold
    if low_confidence:
        warnings.append(
            RegimeWarning(
                "low_confidence",
                f"Confianza baja ({_pct_text(confidence)}, menos de {_pct_text(threshold)}): "
                "el modelo no tiene claro qué tipo de mercado hay ahora.",
            )
        )
    if summary.unstable:
        warnings.append(
            RegimeWarning(
                "unstable",
                f"Régimen inestable: {summary.changes_in_window} cambios en las últimas "
                f"{_velas(cfg.instability_window)} (el límite es "
                f"{cfg.instability_max_changes}). Conviene operar más chico.",
            )
        )
    if n_degenerate:
        warnings.append(
            RegimeWarning(
                "outliers",
                f"{_velas(n_degenerate)} con valores muy fuera de lo normal para el modelo; "
                "tomá el régimen con cuidado.",
            )
        )
    last_close = _timestamp_utc(times[-1]) + timedelta(minutes=tf_minutes)
    lag = now - last_close
    if lag > timedelta(minutes=2 * tf_minutes):
        hours = lag.total_seconds() / 3600.0
        warnings.append(
            RegimeWarning(
                "data_lag",
                f"La última vela cerrada que llegó terminó hace {hours:.0f} h: "
                "los datos pueden estar atrasados.",
            )
        )
    if data_meta is not None and data_meta.warning:
        warnings.append(RegimeWarning("data_source", data_meta.warning))

    want = cfg.display_days * bars_per_day
    if n_bars < want:
        warnings.append(
            RegimeWarning(
                "short_history",
                f"Solo hay {_velas(n_bars)} de historia del régimen para mostrar "
                f"(se pedían {want}).",
            )
        )
    window = slice(max(0, n_bars - want), n_bars)
    history = []
    for t in range(window.start, window.stop):
        raw = states[bc.raw_state[t]]
        conf_label = bc.confirmed_label[t]
        shown = bc.display_state[t]
        history.append(
            RegimeBar(
                time=_timestamp_utc(times[t]),
                state=raw.name,
                state_category=raw.category,
                state_prob=bc.raw_state_prob[t],
                regime=None if shown is None else states[shown].name,
                category=None if conf_label is None else _label_category(conf_label, level, states),
                confidence=bc.confidence[t],
                regime_prob=bc.regime_prob[t],
                unstable=bc.unstable[t],
            )
        )

    by_rank = model.states_by_rank()
    model_age = (now - _as_utc(model.trained_at)).total_seconds() / 86400.0
    info = RegimeModelInfo(
        trained_at=_as_utc(model.trained_at),
        age_days=model_age,
        retrain_every_days=cfg.retrain_every_days,
        n_states=model.n_states,
        selection_criterion=cfg.selection_criterion,
        bic_by_n=dict(model.bic_by_n),
        train_start=_as_utc(model.train_start),
        train_end=_as_utc(model.train_end),
        n_samples=model.n_samples,
        synthetic=bool(model.synthetic),
        retrained_now=status.retrained_now,
        retrain_reason=status.reason,
        stale=status.stale,
        path=str(status.path),
        notes=tuple(model.notes),
        train_seconds=status.train_seconds,
    )
    return RegimeReport(
        symbol=symbol,
        timeframe=timeframe,
        timeframe_minutes=tf_minutes,
        is_context=_is_context(cfg, timeframe),
        change_level=level,
        as_of=_timestamp_utc(times[-1]),
        bars_analyzed=n_bars,
        model=info,
        name=name,
        category=category,
        confidence=confidence,
        regime_prob=regime_prob,
        confirmed=confirmed,
        pending=pending,
        unstable=summary.unstable,
        changes_in_window=summary.changes_in_window,
        instability_window=cfg.instability_window,
        instability_max_changes=cfg.instability_max_changes,
        confirmation_bars=cfg.confirmation_bars,
        age_bars=age_bars,
        started_at=_timestamp_utc(times[start_idx]),
        age_is_lower_bound=age_is_lower_bound,
        confirmed_bars=confirmed_bars,
        top_state_name=top.name,
        top_state_category=top.category,
        top_state_prob=bc.raw_state_prob[-1],
        low_confidence=low_confidence,
        low_confidence_threshold=threshold,
        probabilities=tuple(
            {
                "rank": s.rank,
                "name": s.name,
                "category": s.category,
                "prob": float(last[s.index]),
            }
            for s in by_rank
        ),
        states=_states_table(model, tf_minutes, display.index),
        transition_names=tuple(s.name for s in by_rank),
        transition_matrix=tuple(
            tuple(float(x) for x in row) for row in model.transmat_by_rank().tolist()
        ),
        distribution=_distribution([b for b in bc.confirmed_label[window]], level, states),
        history=tuple(history),
        warnings=tuple(dict.fromkeys(warnings)),  # sin repetidos, en orden
        data_source=None if data_meta is None else data_meta.source,
    )


# --- API -------------------------------------------------------------------------


def _covers(history: _History | None, start: datetime, now: datetime, timeframe: str) -> bool:
    """True si las velas del entrenamiento sirven para calcular el régimen a la hora ``now``.

    Tienen que arrancar antes de ``start`` y llegar a la última vela cerrada a
    esa hora (si mientras se entrenaba cerró otra vela, se vuelven a pedir).
    """
    if history is None or history.df.empty:
        return False
    minutes = timeframe_minutes(timeframe)
    epoch_min = int(now.timestamp() // 60)
    last_open = datetime.fromtimestamp((epoch_min - epoch_min % minutes - minutes) * 60, tz=UTC)
    index = history.df.index
    return bool(index[0] <= pd.Timestamp(start) and index[-1] >= pd.Timestamp(last_open))


async def get_regime(
    symbol: str,
    timeframe: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    *,
    models_dir: Path | str | None = None,
    synthetic: bool = False,
    clock: Callable[[], datetime] = utcnow,
    allow_train: bool = True,
    force_retrain: bool = False,
    notify: Notify | None = None,
    min_training_coverage: float = DEFAULT_MIN_TRAINING_COVERAGE,
) -> RegimeReport:
    """Régimen actual e historial de ``symbol`` en ``timeframe`` (ver docstring del módulo).

    ``synthetic=True`` (modo ``--offline``) usa ``regime.synthetic_models_dir``.
    ``models_dir`` fuerza otra carpeta (tests). ``clock`` da la hora "actual"
    (en ``--offline``, la del mercado sintético). Lanza ``RegimeUnavailable``.
    """
    cfg = settings.regime
    sym = _valid_symbol(symbol)
    status = await ensure_model(
        sym,
        timeframe,
        settings,
        market_data,
        models_dir=models_dir,
        synthetic=synthetic,
        clock=clock,
        allow_train=allow_train,
        force_retrain=force_retrain,
        notify=notify,
        min_training_coverage=min_training_coverage,
    )
    now = _as_utc(clock())
    start = now - timedelta(days=_inference_days(cfg, timeframe))
    if _covers(status.history, start, now, timeframe):
        assert status.history is not None
        df = status.history.df
        history = _History(df=df[df.index >= pd.Timestamp(start)], meta=status.history.meta)
    else:
        history = await _fetch_closed(market_data, sym, timeframe, start, now)
    X, index = _features(history.df, cfg)
    burn = cfg.burn_in_bars
    if len(X) <= burn:
        warm = _warmup(cfg)
        raise RegimeUnavailable(
            f"hay pocas velas para calcular el régimen de {sym} {timeframe}: hay "
            f"{_candles_text(len(history.df), timeframe)} y hacen falta al menos "
            f"{warm + burn + 1} (las primeras {warm} solo sirven para calcular los "
            f"indicadores y las {burn} siguientes, para que el cálculo arranque)"
        )
    try:
        result = await asyncio.to_thread(filtered_from_model, status.model, X)
    except ValueError as exc:
        raise RegimeUnavailable(f"no se pudo calcular el régimen de {timeframe}: {exc}") from exc
    return build_report(
        symbol=sym,
        timeframe=timeframe,
        settings=settings,
        status=status,
        probs=result.probs[burn:],
        times=index[burn:],
        now=now,
        n_degenerate=result.n_degenerate,
        data_meta=history.meta,
    )


def regime_timeframes(regime_cfg: RegimeConfig) -> list[str]:
    """Timeframes del régimen: primero el de operar, después el de contexto."""
    return list(dict.fromkeys([regime_cfg.timeframe, regime_cfg.context_timeframe]))


async def get_market_regimes(
    symbol: str,
    settings: AppSettings,
    market_data: MarketDataSource,
    **kwargs: Any,
) -> dict[str, RegimeReport | RegimeUnavailable]:
    """Régimen de 1h y de contexto 1d (las claves son los timeframes de config).

    Si uno falla, su valor es el ``RegimeUnavailable`` con el motivo y el otro
    sale igual. Nunca lanza por un error de un timeframe (un error inesperado
    se registra en el log y se informa como no disponible). ``kwargs`` van a
    ``get_regime``.
    """
    timeframes = regime_timeframes(settings.regime)

    async def one(tf: str) -> RegimeReport | RegimeUnavailable:
        try:
            return await get_regime(symbol, tf, settings, market_data, **kwargs)
        except RegimeUnavailable as exc:
            return exc
        except Exception as exc:  # el régimen nunca tira abajo al resto
            logger.exception("Error inesperado calculando el régimen de %s %s", symbol, tf)
            return RegimeUnavailable(
                f"error inesperado al calcular el régimen de {tf} ({type(exc).__name__}: {exc})"
            )

    results = await asyncio.gather(*(one(tf) for tf in timeframes))
    return dict(zip(timeframes, results, strict=True))


def regimes_to_dict(
    results: dict[str, RegimeReport | RegimeUnavailable], *, include_history: bool = True
) -> dict[str, dict[str, Any]]:
    """Resultado de ``get_market_regimes`` para JSON: el reporte o ``available: false``."""
    out: dict[str, dict[str, Any]] = {}
    for tf, value in results.items():
        if isinstance(value, RegimeReport):
            out[tf] = value.to_dict(include_history=include_history)
        else:
            out[tf] = {"available": False, "timeframe": tf, "reason": value.reason}
    return out
