"""Entrenamiento del HMM con selección automática del número de estados.

- ``train_regime_model(X_raw, index, symbol, timeframe, regime_cfg)``:
  1. Valida los datos (sin NaN/inf, al menos ``min_training_samples`` filas).
  2. Ajusta el ``FeatureScaler`` **solo con los datos de entrenamiento** y
     estandariza.
  3. Para cada cantidad de estados entre ``n_states_min`` y ``n_states_max``
     entrena ``n_restarts`` veces un ``GaussianHMM`` (semillas distintas pero
     fijas) y se queda con el de mayor log-verosimilitud. Saltea las cantidades
     con más parámetros libres que velas.
  4. Elige la cantidad de estados con **menor BIC** (nada hardcodeado) y guarda
     el BIC y la log-verosimilitud de cada cantidad para mostrarlos.
  5. Etiqueta los estados con ``labels.label_states`` (ocupación = distribución
     estacionaria de la matriz de transición).
  El resultado es determinístico: mismos datos + misma config = mismo modelo
  (el ajuste corre en un solo hilo para que las sumas no cambien de orden).
  Los avisos (inicializaciones que no convergieron o fallaron, cantidades
  salteadas) quedan en ``RegimeModel.notes``, en español simple.

- ``config_fingerprint(regime_cfg, timeframe)``: hash corto de todo lo que
  cambia el modelo. Si cambia, el modelo guardado ya no sirve y se reentrena.

- ``n_free_parameters(n_states, n_features, covariance_type)``: cantidad de
  parámetros libres del HMM (la misma cuenta que usa hmmlearn para el BIC).

Acá no se calcula el régimen de ninguna vela: ``hmm.score`` (algoritmo forward,
solo la verosimilitud total) se usa únicamente para comparar ajustes. El
régimen por vela sale de ``regime.forward`` (probabilidades filtradas).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from threadpoolctl import threadpool_limits

from app.regime.labels import LabelingError, label_states
from app.regime.types import FEATURE_COLUMNS, MODEL_FORMAT_VERSION, FeatureScaler, RegimeModel

if TYPE_CHECKING:  # solo para tipos
    from app.config import RegimeConfig

logger = logging.getLogger(__name__)

# Separación entre semillas: la inicialización r de n estados usa
# random_seed + _SEED_STRIDE * n + r, así cada combinación es reproducible.
_SEED_STRIDE = 1000


class RegimeTrainingError(Exception):
    """No se pudo entrenar el modelo de régimen (el mensaje explica por qué)."""


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- huella de la configuración ----------------------------------------------


def config_fingerprint(regime_cfg: RegimeConfig, timeframe: str) -> str:
    """Hash corto y estable (sha256 de un JSON canónico) de la config que afecta al modelo.

    Incluye: ventanas de las features, rango de estados, criterio de selección,
    tipo de covarianza, inicializaciones, iteraciones, tolerancia, semilla, los
    nombres de los estados para cada cantidad posible, el período de
    entrenamiento del timeframe (``training_days`` para ``regime.timeframe``,
    ``context_training_days`` para ``regime.context_timeframe``), el timeframe,
    ``FEATURE_COLUMNS`` y ``MODEL_FORMAT_VERSION``. Lo que no cambia el modelo
    (confirmación, inestabilidad, carpetas, etc.) no entra. Lanza ``ValueError``
    si ``timeframe`` no es ninguno de los dos del régimen.
    """
    if timeframe == regime_cfg.timeframe:
        training_days = regime_cfg.training_days
    elif timeframe == regime_cfg.context_timeframe:
        training_days = regime_cfg.context_training_days
    else:
        raise ValueError(
            f"timeframe '{timeframe}' no es el del régimen ({regime_cfg.timeframe}) "
            f"ni el de contexto ({regime_cfg.context_timeframe})"
        )
    labels = {
        str(n): [
            {"name": label.name, "category": label.category}
            for label in regime_cfg.labels.get(n, [])
        ]
        for n in range(regime_cfg.n_states_min, regime_cfg.n_states_max + 1)
    }
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "timeframe": timeframe,
        "training_days": training_days,
        "features": regime_cfg.features.model_dump(mode="json"),
        "n_states_min": regime_cfg.n_states_min,
        "n_states_max": regime_cfg.n_states_max,
        "selection_criterion": regime_cfg.selection_criterion,
        "covariance_type": regime_cfg.covariance_type,
        "n_restarts": regime_cfg.n_restarts,
        "max_iter": regime_cfg.max_iter,
        "tol": regime_cfg.tol,
        "random_seed": regime_cfg.random_seed,
        "labels": labels,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --- entrenamiento -----------------------------------------------------------


def n_free_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    """Parámetros libres de un ``GaussianHMM`` (misma cuenta que ``hmm.bic``)."""
    cov = {
        "spherical": n_states,
        "diag": n_states * n_features,
        "full": n_states * n_features * (n_features + 1) // 2,
        "tied": n_features * (n_features + 1) // 2,
    }
    if covariance_type not in cov:
        raise ValueError(f"tipo de covarianza desconocido: {covariance_type}")
    return (
        (n_states - 1)  # probabilidades iniciales
        + n_states * (n_states - 1)  # matriz de transición
        + n_states * n_features  # medias
        + cov[covariance_type]
    )


@contextmanager
def _fit_context() -> Iterator[None]:
    """Entorno del ajuste: un solo hilo y sin avisos de hmmlearn/sklearn.

    Con varios hilos (OpenBLAS/OpenMP) las sumas cambian de orden entre
    corridas y el resultado varía en el último decimal; eso alcanza para que
    KMeans elija otra inicialización "empatada" y los estados salgan
    permutados. Con un hilo el entrenamiento es reproducible (y con estos
    tamaños, hasta más rápido). La no convergencia se detecta aparte y queda en
    ``RegimeModel.notes``.
    """
    hmm_logger = logging.getLogger("hmmlearn")
    previous = hmm_logger.level
    hmm_logger.setLevel(logging.ERROR)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        hmm_logger.setLevel(previous)


def _converged(hmm: GaussianHMM, tol: float) -> bool:
    """True si EM paró porque la mejora fue menor a ``tol``.

    ``hmm.monitor_.converged`` también da True cuando se agotan las
    iteraciones, así que se mira la historia directamente.
    """
    history = list(hmm.monitor_.history)
    return len(history) >= 2 and (history[-1] - history[-2]) < tol


def _params_finite(hmm: GaussianHMM) -> bool:
    arrays = (hmm.startprob_, hmm.transmat_, hmm.means_, hmm.covars_)
    return all(np.isfinite(np.asarray(a, dtype=float)).all() for a in arrays)


@dataclass
class _FitResult:
    hmm: GaussianHMM
    loglik: float
    converged: bool


def _fit_one(Z: np.ndarray, n_states: int, seed: int, regime_cfg: RegimeConfig) -> _FitResult:
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=regime_cfg.covariance_type,
        n_iter=regime_cfg.max_iter,
        tol=regime_cfg.tol,
        random_state=seed,
        verbose=False,
    )
    with _fit_context():
        hmm.fit(Z)
        loglik = float(hmm.score(Z))
    return _FitResult(hmm=hmm, loglik=loglik, converged=_converged(hmm, regime_cfg.tol))


def _to_utc(ts: Any) -> datetime:
    stamp = pd.Timestamp(ts)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def _as_matrix(X_raw: Any) -> np.ndarray:
    if isinstance(X_raw, pd.DataFrame):
        missing = [c for c in FEATURE_COLUMNS if c not in X_raw.columns]
        if missing:
            raise RegimeTrainingError(f"faltan columnas de features: {', '.join(missing)}")
        X_raw = X_raw.loc[:, list(FEATURE_COLUMNS)]
    X = np.ascontiguousarray(X_raw, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
        raise RegimeTrainingError(
            f"las features tienen que ser una matriz (velas × {len(FEATURE_COLUMNS)}) "
            f"en el orden {', '.join(FEATURE_COLUMNS)}; llegó una de forma {X.shape}"
        )
    return X


def train_regime_model(
    X_raw: np.ndarray | pd.DataFrame,
    index: pd.DatetimeIndex | Sequence[Any],
    symbol: str,
    timeframe: str,
    regime_cfg: RegimeConfig,
    *,
    synthetic: bool = False,
    clock: Callable[[], datetime] = utcnow,
) -> RegimeModel:
    """Entrena el modelo de régimen de una moneda y un timeframe.

    ``X_raw``: features crudas (sin estandarizar), una fila por vela cerrada y
    columnas en el orden de ``FEATURE_COLUMNS`` (o un DataFrame con esas
    columnas). ``index``: la hora de apertura de cada fila (UTC, creciente).
    Lanza ``RegimeTrainingError`` si los datos no alcanzan o ningún ajuste sirve,
    y ``ValueError`` si ``timeframe`` no es ni el del régimen ni el de contexto.
    """
    fingerprint = config_fingerprint(regime_cfg, timeframe)  # falla rápido si el tf no va
    X = _as_matrix(X_raw)
    times = pd.DatetimeIndex(index)
    n_samples = X.shape[0]
    if len(times) != n_samples:
        raise RegimeTrainingError(
            f"el índice tiene {len(times)} fechas y las features {n_samples} filas"
        )
    bad_rows = int((~np.isfinite(X).all(axis=1)).sum())
    if bad_rows:
        raise RegimeTrainingError(
            f"hay {bad_rows} filas con valores vacíos o infinitos (NaN/inf); "
            "sacá el calentamiento de las features antes de entrenar"
        )
    if n_samples < regime_cfg.min_training_samples:
        raise RegimeTrainingError(
            f"hay {n_samples} velas útiles y se necesitan al menos "
            f"{regime_cfg.min_training_samples} para entrenar"
        )
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise RegimeTrainingError("las fechas de las velas tienen que ser crecientes y sin repetir")

    scaler = FeatureScaler.fit(X)
    Z = scaler.transform(X)
    n_features = Z.shape[1]

    best_by_n: dict[int, _FitResult] = {}
    bic_by_n: dict[int, float] = {}
    loglik_by_n: dict[int, float] = {}
    notes: list[str] = []
    skipped_too_many_params: list[int] = []
    last_error = ""

    for n in range(regime_cfg.n_states_min, regime_cfg.n_states_max + 1):
        n_params = n_free_parameters(n, n_features, regime_cfg.covariance_type)
        if n_params >= n_samples:
            skipped_too_many_params.append(n)
            continue
        best: _FitResult | None = None
        failed = 0
        not_converged = 0
        for r in range(regime_cfg.n_restarts):
            seed = regime_cfg.random_seed + _SEED_STRIDE * n + r
            try:
                result = _fit_one(Z, n, seed, regime_cfg)
            except Exception as exc:  # un ajuste fallido no corta el resto
                logger.debug("ajuste HMM falló (n=%s, semilla=%s): %s", n, seed, exc)
                last_error = f"{type(exc).__name__}: {exc}"
                failed += 1
                continue
            if not math.isfinite(result.loglik) or not _params_finite(result.hmm):
                failed += 1
                continue
            if not result.converged:
                not_converged += 1
            if best is None or result.loglik > best.loglik:
                best = result
        total = regime_cfg.n_restarts
        if failed:
            notes.append(f"{n} estados: {failed} de {total} intentos de entrenamiento fallaron.")
        if not_converged:
            notes.append(
                f"{n} estados: {not_converged} de {total} intentos no terminaron de "
                f"ajustarse en {regime_cfg.max_iter} iteraciones."
            )
        if best is None:
            continue
        with _fit_context():
            bic = float(best.hmm.bic(Z))
        if not math.isfinite(bic):
            notes.append(f"{n} estados: el BIC no dio un número válido; se descarta.")
            continue
        best_by_n[n] = best
        bic_by_n[n] = bic
        loglik_by_n[n] = best.loglik

    if skipped_too_many_params:
        listed = ", ".join(str(n) for n in skipped_too_many_params)
        notes.append(
            f"No se probaron {listed} estados: tienen más parámetros que velas "
            f"para entrenar ({n_samples})."
        )
    if not best_by_n:
        detail = f"; último error: {last_error}" if last_error else ""
        raise RegimeTrainingError(
            "ningún entrenamiento del HMM dio un resultado válido "
            f"(se probaron de {regime_cfg.n_states_min} a {regime_cfg.n_states_max} "
            f"estados{detail})"
        )

    chosen_n = min(bic_by_n, key=lambda n: (bic_by_n[n], n))
    chosen = best_by_n[chosen_n]
    if not chosen.converged:
        notes.append(
            f"El modelo elegido ({chosen_n} estados) no terminó de ajustarse en "
            f"{regime_cfg.max_iter} iteraciones; puede ser algo menos preciso."
        )
    if chosen_n == regime_cfg.n_states_max and regime_cfg.n_states_max > regime_cfg.n_states_min:
        notes.append(
            f"El BIC eligió el máximo de estados permitido ({chosen_n}); es lo habitual con "
            "tanta historia y no es un problema."
        )

    labels_for_n = regime_cfg.labels.get(chosen_n)
    if labels_for_n is None or len(labels_for_n) != chosen_n:
        raise RegimeTrainingError(f"regime.labels no tiene los {chosen_n} nombres necesarios")
    try:
        states = label_states(
            chosen.hmm, scaler, labels_for_n, return_window=regime_cfg.features.smoothing_window
        )
    except LabelingError as exc:
        raise RegimeTrainingError(f"no se pudieron etiquetar los estados: {exc}") from exc

    return RegimeModel(
        symbol=symbol,
        timeframe=timeframe,
        hmm=chosen.hmm,
        scaler=scaler,
        states=states,
        n_states=chosen_n,
        bic_by_n=bic_by_n,
        loglik_by_n=loglik_by_n,
        trained_at=_to_utc(clock()),
        train_start=_to_utc(times[0]),
        train_end=_to_utc(times[-1]),
        n_samples=n_samples,
        config_fingerprint=fingerprint,
        synthetic=synthetic,
        format_version=MODEL_FORMAT_VERSION,
        notes=notes,
    )
