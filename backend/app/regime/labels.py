"""Ordena los estados del HMM, les pone nombre y arma la tabla por régimen.

- ``label_states(hmm, scaler, labels_for_n)``: pasa las medias de cada estado a
  unidades reales (``scaler.inverse_transform(hmm.means_)``) y ordena los
  estados por retorno medio, del más bajista (rango 0) al más alcista. Cada
  estado recibe el nombre y la categoría de ``labels_for_n[rango]`` (la lista
  de ``regime.labels[n]`` en config.yaml). Devuelve los ``StateInfo`` en el
  orden del índice interno del HMM.

  Empates de retorno (diferencia relativa menor a ``RETURN_TIE_REL_TOL``): se
  desempata por volatilidad y el estado más volátil se va al extremo de su lado.
  Si el retorno es <= 0 queda más abajo (más "crash"); si es > 0, más arriba
  (más "euforia"). Esa tolerancia es numérica (1e-12): con modelos entrenados
  los retornos medios casi nunca empatan, así que en la práctica el orden (y
  el nombre) sale solo del retorno medio. Como la feature de retorno es el
  retorno acumulado de las últimas ``regime.features.smoothing_window`` velas
  (decisión del usuario), ese orden refleja la tendencia del día y no una vela
  suelta; ``return_window`` lo vuelve a pasar a retorno por vela para mostrarlo.

- ``stationary_distribution(transmat)``: distribución estacionaria de la cadena
  (la fracción de tiempo que, a la larga, se pasa en cada estado). Es la
  "ocupación" de cada régimen. Sale solo de la matriz de transición, no de
  probabilidades calculadas sobre la serie (que mirarían el futuro).

- ``regime_stats_table(model)``: una fila por régimen, ordenada del más bajista
  al más alcista, con retorno medio por vela, volatilidad media, rango medio
  (todo en %), duración esperada en velas y % de tiempo. Es lo que muestran la
  consola y el panel. La matriz de transición ya ordenada por rango está en
  ``RegimeModel.transmat_by_rank()``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from app.regime.types import FeatureScaler, RegimeModel, StateInfo

if TYPE_CHECKING:  # solo para tipos: evita importar config en tiempo de ejecución
    from app.config import RegimeLabel

# Dos retornos medios se consideran iguales si su diferencia relativa es menor
# a esto (es una tolerancia numérica, no una regla de trading).
RETURN_TIE_REL_TOL = 1e-12

# Probabilidades negativas más chicas que esto (en valor absoluto) son ruido
# numérico de la resolución del sistema lineal y se llevan a 0.
_NEGATIVE_NOISE_TOL = 1e-9


class LabelingError(ValueError):
    """No se pudieron etiquetar los estados (datos del modelo inconsistentes)."""


def _feature_index(scaler: FeatureScaler, name: str) -> int:
    try:
        return list(scaler.feature_names).index(name)
    except ValueError as exc:
        raise LabelingError(
            f"el escalador no tiene la feature '{name}' (tiene {scaler.feature_names})"
        ) from exc


def _returns_tied(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RETURN_TIE_REL_TOL, abs_tol=0.0)


def rank_order(mean_return: np.ndarray, mean_volatility: np.ndarray) -> list[int]:
    """Índices de estado ordenados del más bajista al más alcista.

    Orden principal: retorno medio ascendente. Los grupos de retornos empatados
    (cadenas de valores consecutivos con diferencia relativa < 1e-12) se ordenan
    por volatilidad: si el retorno del grupo es <= 0, el más volátil primero
    (rango más bajo); si es > 0, el más volátil último (rango más alto). Último
    desempate: el índice interno, para que el orden sea determinístico.
    """
    ret = np.asarray(mean_return, dtype=float)
    vol = np.asarray(mean_volatility, dtype=float)
    if ret.shape != vol.shape or ret.ndim != 1:
        raise LabelingError("retornos y volatilidades tienen que ser vectores del mismo largo")
    if not (np.isfinite(ret).all() and np.isfinite(vol).all()):
        raise LabelingError("hay medias no finitas en los estados del HMM")

    by_return = sorted(range(len(ret)), key=lambda i: (ret[i], i))

    groups: list[list[int]] = []
    for i in by_return:
        if groups and _returns_tied(ret[groups[-1][-1]], ret[i]):
            groups[-1].append(i)
        else:
            groups.append([i])

    order: list[int] = []
    for group in groups:
        if len(group) == 1:
            order.extend(group)
            continue
        if ret[group[0]] <= 0:
            # Lado bajista: el más volátil es el más extremo (va primero).
            order.extend(sorted(group, key=lambda i: (-vol[i], i)))
        else:
            # Lado alcista: el más volátil es el más extremo (va último).
            order.extend(sorted(group, key=lambda i: (vol[i], i)))
    return order


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Distribución estacionaria ``pi`` de una matriz de transición (``pi A = pi``).

    Resuelve por mínimos cuadrados el sistema ``(Aᵀ - I) pi = 0`` junto con
    ``sum(pi) = 1``. Si la cadena tiene más de una distribución estacionaria
    (por ejemplo, estados que nunca se comunican) devuelve la de norma mínima,
    que igual es válida. Los negativos diminutos por redondeo se llevan a 0 y se
    renormaliza. Si el sistema no da una solución usable, usa iteración de
    potencias desde la distribución uniforme.
    """
    A = np.asarray(transmat, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise LabelingError("la matriz de transición tiene que ser cuadrada y no vacía")
    if not np.isfinite(A).all() or (A < 0).any():
        raise LabelingError("la matriz de transición tiene valores negativos o no finitos")
    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise LabelingError("las filas de la matriz de transición tienen que sumar 1")
    A = A / row_sums[:, None]
    k = A.shape[0]

    system = np.vstack([A.T - np.eye(k), np.ones((1, k))])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(system, rhs, rcond=None)

    if np.isfinite(pi).all() and (pi > -_NEGATIVE_NOISE_TOL).all() and pi.sum() > 0:
        pi = np.clip(pi, 0.0, None)
        pi = pi / pi.sum()
        if np.allclose(pi @ A, pi, atol=1e-8):
            return pi

    # Respaldo: iteración de potencias (siempre converge a una distribución
    # válida o se queda oscilando; en ese caso se promedia).
    pi = np.full(k, 1.0 / k)
    acc = np.zeros(k)
    for _ in range(10_000):
        pi = pi @ A
        acc += pi
    pi = acc / acc.sum()
    return pi


def label_states(
    hmm: Any,
    scaler: FeatureScaler,
    labels_for_n: Sequence[RegimeLabel],
    return_window: int = 1,
) -> tuple[StateInfo, ...]:
    """Etiqueta los estados de un ``GaussianHMM`` entrenado.

    ``hmm.means_`` está en espacio estandarizado; se pasa a unidades reales con
    el escalador para tener retorno logarítmico medio por vela, volatilidad
    realizada media y rango medio de cada estado. ``return_window`` es la
    cantidad de velas que suma la feature de retorno
    (``regime.features.smoothing_window``): el retorno medio se divide por eso
    para que ``StateInfo.mean_return`` sea siempre **por vela** (el orden de los
    estados no cambia). Devuelve una tupla de ``StateInfo`` indexada por el
    índice interno del HMM.
    """
    if isinstance(return_window, bool) or int(return_window) != return_window or return_window < 1:
        raise LabelingError(f"return_window tiene que ser un entero >= 1 (llegó {return_window!r})")
    means_z = np.asarray(hmm.means_, dtype=float)
    transmat = np.asarray(hmm.transmat_, dtype=float)
    n_states = means_z.shape[0]
    if len(labels_for_n) != n_states:
        raise LabelingError(
            f"hay {n_states} estados pero {len(labels_for_n)} nombres en regime.labels"
        )
    if transmat.shape != (n_states, n_states):
        raise LabelingError("la matriz de transición no coincide con la cantidad de estados")

    raw = scaler.inverse_transform(means_z)
    mean_return = raw[:, _feature_index(scaler, "log_return")] / int(return_window)
    mean_volatility = raw[:, _feature_index(scaler, "realized_vol")]
    mean_range = raw[:, _feature_index(scaler, "range_pct")]

    order = rank_order(mean_return, mean_volatility)
    rank_of = {state: rank for rank, state in enumerate(order)}
    occupancy = stationary_distribution(transmat)

    states: list[StateInfo] = []
    for i in range(n_states):
        stay = float(transmat[i, i])
        duration = math.inf if stay >= 1.0 else 1.0 / (1.0 - stay)
        label = labels_for_n[rank_of[i]]
        states.append(
            StateInfo(
                index=i,
                rank=rank_of[i],
                name=label.name,
                category=label.category,
                mean_return=float(mean_return[i]),
                mean_volatility=float(mean_volatility[i]),
                mean_range=float(mean_range[i]),
                expected_duration_bars=float(duration),
                occupancy=float(occupancy[i]),
            )
        )
    return tuple(states)


def regime_stats_table(model: RegimeModel) -> list[dict[str, Any]]:
    """Tabla por régimen para la consola y el panel, del más bajista al más alcista.

    Columnas: ``rank``, ``index`` (estado interno del HMM), ``name``,
    ``category``, ``mean_return_pct`` (retorno logarítmico medio por vela × 100),
    ``mean_volatility_pct`` y ``mean_range_pct`` (× 100),
    ``expected_duration_bars`` (velas que dura en promedio; ``None`` si el
    estado nunca se abandona, para que se pueda pasar a JSON) y
    ``occupancy_pct`` (% del tiempo, de la distribución estacionaria).
    """
    rows: list[dict[str, Any]] = []
    for s in model.states_by_rank():
        duration = s.expected_duration_bars
        rows.append(
            {
                "rank": s.rank,
                "index": s.index,
                "name": s.name,
                "category": s.category,
                "mean_return_pct": s.mean_return * 100.0,
                "mean_volatility_pct": s.mean_volatility * 100.0,
                "mean_range_pct": s.mean_range * 100.0,
                "expected_duration_bars": duration if math.isfinite(duration) else None,
                "occupancy_pct": s.occupancy * 100.0,
            }
        )
    return rows
