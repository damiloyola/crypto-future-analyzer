"""Tipos compartidos del detector de régimen.

- ``FEATURE_COLUMNS``: las cuatro features del HMM, en este orden.
- ``FeatureScaler``: estandarización (media 0, desvío 1) ajustada **solo con
  datos de entrenamiento** y aplicada igual hacia adelante. Es propia (no la de
  scikit-learn) para que el modelo guardado no dependa de la versión de sklearn.
- ``StateInfo``: un estado del HMM ya etiquetado (nombre, categoría, rango de
  más bajista a más alcista y estadísticas en unidades reales).
- ``RegimeModel``: el modelo entrenado completo (HMM + escalador + etiquetas +
  metadatos), que es lo que se guarda en disco.

Los parámetros del HMM (``hmm.means_``, ``hmm.covars_``, etc.) están en el
espacio **estandarizado**. Las estadísticas de ``StateInfo`` están en unidades
reales (retorno logarítmico por vela, volatilidad realizada, rango).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np

RegimeCategory = Literal["crash", "bajista", "neutral", "alcista", "euforia"]

FEATURE_COLUMNS: tuple[str, ...] = ("log_return", "realized_vol", "range_pct", "volume_z")

# Versión del formato del modelo guardado. Cambiarla invalida los modelos viejos.
MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FeatureScaler:
    """Estandarización por columna: ``(x - mean) / scale``."""

    mean: np.ndarray  # (F,)
    scale: np.ndarray  # (F,), desvío poblacional; un desvío 0 se reemplaza por 1
    feature_names: tuple[str, ...] = FEATURE_COLUMNS

    @classmethod
    def fit(cls, X: np.ndarray, feature_names: tuple[str, ...] = FEATURE_COLUMNS) -> FeatureScaler:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("FeatureScaler.fit necesita una matriz (T, F) con T >= 2")
        if not np.isfinite(X).all():
            raise ValueError("FeatureScaler.fit: hay valores no finitos (sacá el calentamiento)")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        return cls(mean=mean, scale=scale, feature_names=tuple(feature_names))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) * self.scale + self.mean


@dataclass(frozen=True)
class StateInfo:
    """Un estado del HMM etiquetado."""

    index: int  # índice interno del HMM (0..K-1)
    rank: int  # 0 = el más bajista ... K-1 = el más alcista
    name: str  # "Crash", "Bajista", "Neutral bajista", ...
    category: RegimeCategory
    mean_return: float  # retorno logarítmico medio por vela (unidades reales)
    mean_volatility: float  # volatilidad realizada media (unidades reales)
    mean_range: float  # rango medio (high - low) / close
    expected_duration_bars: float  # 1 / (1 - P(quedarse)), en velas
    occupancy: float  # fracción del tiempo de entrenamiento en este estado (0-1)


@dataclass
class RegimeModel:
    """Modelo de régimen entrenado para una moneda y un timeframe."""

    symbol: str
    timeframe: str
    hmm: Any  # hmmlearn.hmm.GaussianHMM, parámetros en espacio estandarizado
    scaler: FeatureScaler
    states: tuple[StateInfo, ...]  # indexado por índice interno del HMM
    n_states: int
    bic_by_n: dict[int, float]  # BIC del mejor ajuste para cada cantidad de estados
    loglik_by_n: dict[int, float]
    trained_at: datetime  # UTC
    train_start: datetime  # primera vela usada (UTC)
    train_end: datetime  # última vela usada (UTC)
    n_samples: int
    config_fingerprint: str  # hash de la config que afecta al modelo
    synthetic: bool = False  # entrenado con datos sintéticos (modo --offline)
    format_version: int = MODEL_FORMAT_VERSION
    notes: list[str] = field(default_factory=list)  # avisos del entrenamiento

    def states_by_rank(self) -> tuple[StateInfo, ...]:
        """Estados ordenados del más bajista al más alcista."""
        return tuple(sorted(self.states, key=lambda s: s.rank))

    def transmat_by_rank(self) -> np.ndarray:
        """Matriz de transición con filas/columnas ordenadas por ``rank``."""
        order = [s.index for s in self.states_by_rank()]
        return np.asarray(self.hmm.transmat_)[np.ix_(order, order)]
