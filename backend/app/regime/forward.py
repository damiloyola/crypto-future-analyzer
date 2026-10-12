"""Forward filtering: régimen de cada vela usando solo datos hasta esa vela.

Es el corazón del "cero look-ahead". Trabaja con numpy/scipy puros, sobre
features ya estandarizadas (el espacio en el que viven los parámetros del HMM).

- ``log_emission(Z, means, covars) -> ndarray (T, K)``: log-densidad gaussiana
  de cada vela en cada estado, con matrices de covarianza **completas**
  ``(K, F, F)``. Usa Cholesky: nunca calcula la densidad en escala lineal, así
  que no hay underflow. Si una covarianza no es definida positiva por redondeo,
  le suma a la diagonal un "jitter" mínimo y creciente.
- ``model_covariances(hmm) -> ndarray (K, F, F)``: las covarianzas completas de
  un ``GaussianHMM`` para cualquier ``covariance_type`` (``full``, ``diag``,
  ``tied`` o ``spherical``). Hace falta porque con ``spherical`` hmmlearn 0.3.3
  devuelve en ``covars_`` una matriz por estado **y por feature** (K·F en vez
  de K); acá se arma una por estado desde su varianza.
- ``filtered_probabilities(Z, startprob, transmat, means, covars) ->
  FilterResult``: algoritmo forward normalizado en espacio log (logsumexp).
  ``probs[t]`` es P(estado | Z[0..t]): la fila *t* depende **solo** de
  ``Z[:t+1]``, así que agregar o cambiar velas posteriores nunca la modifica.
  Las filas suman 1 y nunca dan NaN, aun con outliers extremos.
- ``filtered_from_model(model, X_raw) -> FilterResult``: estandariza las
  features crudas con ``model.scaler`` (ajustado solo con el entrenamiento) y
  filtra con los parámetros de ``model.hmm`` (covarianzas de
  ``model_covariances``).

``FilterResult`` trae además ``states`` (estado más probable por vela, el
"régimen crudo") y ``confidence`` (su probabilidad, fracción 0-1; la UI la
muestra como 0-100 %).

Casos degenerados (``FilterResult.n_degenerate``): si una vela es tan absurda
que su densidad cae fuera del rango de un float en **todos** los estados, se
usa el límite matemático: gana el estado con menor distancia de Mahalanobis
(el más "ancho" en esa dirección). Si la observación solo es compatible con
estados a los que la matriz de transición no deja llegar, se confía en la
observación. En ambos casos ``log_likelihood`` pasa a ``-inf`` (el modelo le
da probabilidad prácticamente cero a esos datos). Con datos reales no pasa.

Prohibido acá y en toda la app: ``predict``, ``decode``, ``predict_proba`` y
``score_samples`` de hmmlearn para clasificar velas (usan forward-backward o
Viterbi, que miran el futuro). ``tests/test_regime_no_lookahead_guard.py``
revisa el código fuente de ``app/`` y falla si aparecen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import solve_triangular

if TYPE_CHECKING:  # solo para tipos
    from app.regime.types import RegimeModel

# Tolerancia para aceptar que startprob y cada fila de transmat suman 1
# (después se renormalizan exacto).
_PROB_SUM_ATOL = 1e-6

# Jitter para covarianzas casi singulares: arranca en 1e-12 × (varianza media)
# y se multiplica por 10 hasta 12 veces (como mucho 0,1 × varianza media).
_JITTER_START = 1e-12
_JITTER_STEPS = 12

# Por encima de este piso, la predicción se calcula como alpha @ A en
# probabilidades (lo que se pierde por underflow es < 1e-300, despreciable);
# por debajo, con logsumexp en espacio log.
_PRED_PROB_FLOOR = 1e-200

# Dos estados empatan en el límite degenerado si sus log-distancias de
# Mahalanobis difieren menos que esto (relativo).
_DEGENERATE_TIE_RTOL = 1e-12

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, eq=False)  # eq=False: comparar arrays con == no tiene sentido
class FilterResult:
    """Resultado del forward filtering.

    ``probs[t, k]`` = P(estado k | velas 0..t), fracción 0-1; cada fila suma 1.
    ``log_likelihood`` = log P(Z[0..T-1]) según el modelo (igual a
    ``hmm.score``); ``-inf`` si hubo velas degeneradas.
    ``n_degenerate`` = velas que necesitaron el tratamiento de casos extremos.
    """

    probs: np.ndarray  # (T, K)
    log_likelihood: float
    n_degenerate: int = 0

    @property
    def n_bars(self) -> int:
        return int(self.probs.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.probs.shape[1])

    @property
    def states(self) -> np.ndarray:
        """Estado más probable de cada vela (índice interno del HMM). En empate, el menor."""
        return np.argmax(self.probs, axis=1)

    @property
    def confidence(self) -> np.ndarray:
        """Probabilidad del estado más probable de cada vela (fracción 0-1)."""
        return np.max(self.probs, axis=1)


# --- validación ----------------------------------------------------------------


def _as_prob_vector(p: Any, name: str, k: int | None = None) -> np.ndarray:
    v = np.asarray(p, dtype=float)
    if v.ndim != 1 or (k is not None and v.shape[0] != k):
        raise ValueError(f"{name}: se esperaba un vector de largo {k}, llegó forma {v.shape}")
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f"{name}: tiene valores negativos o no finitos")
    total = v.sum()
    if not abs(total - 1.0) <= _PROB_SUM_ATOL:
        raise ValueError(f"{name}: las probabilidades suman {total:.6g}, no 1")
    return v / total


def _validate_gaussians(means: Any, covars: Any) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(means, dtype=float)
    if mu.ndim != 2 or mu.shape[0] < 1 or mu.shape[1] < 1:
        raise ValueError(f"means: se esperaba forma (K, F), llegó {mu.shape}")
    k, f = mu.shape
    if not np.isfinite(mu).all():
        raise ValueError("means: hay valores no finitos")
    cv = np.asarray(covars, dtype=float)
    if cv.shape != (k, f, f):
        raise ValueError(
            f"covars: se esperaban matrices completas de forma {(k, f, f)}, llegó {cv.shape}"
        )
    if not np.isfinite(cv).all():
        raise ValueError("covars: hay valores no finitos")
    return mu, cv


def _validate_params(
    startprob: Any, transmat: Any, means: Any, covars: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu, cv = _validate_gaussians(means, covars)
    k = mu.shape[0]
    pi = _as_prob_vector(startprob, "startprob", k)
    a = np.asarray(transmat, dtype=float)
    if a.shape != (k, k):
        raise ValueError(f"transmat: se esperaba forma {(k, k)}, llegó {a.shape}")
    a = np.stack([_as_prob_vector(row, f"transmat fila {i}") for i, row in enumerate(a)])
    return pi, a, mu, cv


def _validate_observations(Z: Any, n_features: int) -> np.ndarray:
    z = np.asarray(Z, dtype=float)
    if z.ndim != 2 or z.shape[1] != n_features:
        raise ValueError(f"Z: se esperaba una matriz (T, {n_features}), llegó forma {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError(
            "Z: hay valores no finitos (sacá las primeras velas, las del calentamiento "
            "de las features)"
        )
    return z


# --- emisiones -----------------------------------------------------------------


def _robust_cholesky(cov: np.ndarray, state: int) -> np.ndarray:
    """Cholesky inferior; si la matriz no es definida positiva, suma un jitter mínimo."""
    sym = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        pass
    f = sym.shape[0]
    mean_var = float(np.trace(sym)) / f
    scale = mean_var if math.isfinite(mean_var) and mean_var > 0 else 1.0
    jitter = _JITTER_START * scale
    eye = np.eye(f)
    for _ in range(_JITTER_STEPS):
        try:
            return np.linalg.cholesky(sym + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise ValueError(f"covars: la matriz del estado {state} no es definida positiva")


def _emission_terms(z: np.ndarray, mu: np.ndarray, cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(log_b, log_q)``, ambos ``(T, K)``.

    ``log_b`` es la log-densidad gaussiana y ``log_q`` el logaritmo de la
    distancia de Mahalanobis al cuadrado (sirve para el límite degenerado
    cuando ``log_b`` es ``-inf`` en todos los estados).

    Cada fila se divide por su mayor desvío absoluto antes de resolver el
    sistema triangular, así no hay overflow ni con observaciones enormes.
    """
    t_len, f = z.shape
    k = mu.shape[0]
    log_b = np.empty((t_len, k))
    log_q = np.empty((t_len, k))
    for j in range(k):
        chol = _robust_cholesky(cv[j], j)
        log_norm = 0.5 * f * _LOG_2PI + float(np.sum(np.log(np.diag(chol))))
        diff = z - mu[j]
        s = np.max(np.abs(diff), axis=1)
        s = np.where(s > 0, s, 1.0)
        y = solve_triangular(chol, (diff / s[:, None]).T, lower=True, check_finite=False)
        r = np.sum(y * y, axis=0)
        with np.errstate(over="ignore", divide="ignore"):
            q = np.where(r > 0, (s * s) * r, 0.0)
            log_q[:, j] = 2.0 * np.log(s) + np.log(r)
        log_b[:, j] = -0.5 * q - log_norm
    return log_b, log_q


def log_emission(Z: Any, means: Any, covars: Any) -> np.ndarray:
    """Log-densidad gaussiana ``(T, K)`` de cada vela en cada estado.

    ``Z`` es ``(T, F)``, ``means`` es ``(K, F)`` y ``covars`` ``(K, F, F)``
    (covarianzas completas). Calculada con Cholesky, en espacio log.
    """
    mu, cv = _validate_gaussians(means, covars)
    z = _validate_observations(Z, mu.shape[1])
    return _emission_terms(z, mu, cv)[0]


# --- forward filtering -----------------------------------------------------------


def _logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    if not math.isfinite(m):
        return m  # todo -inf (o algún +inf, que no debería pasar)
    return m + math.log(float(np.sum(np.exp(v - m))))


def _log_predict(log_alpha: np.ndarray, log_a: np.ndarray) -> np.ndarray:
    """log P(estado_t = j | Z[0..t-1]) = logsumexp_i(log_alpha_i + log A_ij)."""
    m = log_alpha[:, None] + log_a  # (K, K)
    mx = np.max(m, axis=0)
    safe = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return np.where(np.isfinite(mx), safe + np.log(np.sum(np.exp(m - safe), axis=0)), -np.inf)


def _degenerate_step(
    log_pred: np.ndarray, log_b_t: np.ndarray, log_q_t: np.ndarray
) -> tuple[np.ndarray, float]:
    """Paso del filtro cuando ``logsumexp(log_pred + log_b_t)`` no es finito.

    1. Si la densidad es ``-inf`` en todos los estados (observación fuera de
       rango), se toma el límite: solo cuentan los estados con menor distancia
       de Mahalanobis.
    2. Si aun así ningún estado alcanzable explica la vela, se confía en la
       observación (se ignora la predicción de la matriz de transición).
    3. Último recurso: probabilidades uniformes.
    """
    lb = log_b_t
    if not np.isfinite(lb).any():
        finite_q = log_q_t[np.isfinite(log_q_t)]
        if finite_q.size:
            m = float(finite_q.min())
            near = np.isclose(log_q_t, m, rtol=_DEGENERATE_TIE_RTOL, atol=0.0)
            lb = np.where(near, 0.0, -np.inf)
        else:
            lb = np.zeros_like(log_b_t)
        joint = log_pred + lb
        c = _logsumexp(joint)
        if math.isfinite(c):
            return joint, c
    c = _logsumexp(lb)
    if math.isfinite(c):
        return lb, c
    k = log_b_t.shape[0]
    return np.full(k, -math.log(k)), 0.0


def filtered_probabilities(
    Z: Any, startprob: Any, transmat: Any, means: Any, covars: Any
) -> FilterResult:
    """Probabilidades filtradas P(estado | Z[0..t]) para cada vela *t*.

    Recursión forward normalizada en espacio log:

    - ``log_pred_0 = log startprob``;
      ``log_pred_t = logsumexp_i(log_alpha_{t-1,i} + log A_ij)``. Como
      ``alpha_{t-1}`` está normalizado, esto es ``log(alpha_{t-1} @ A)`` y se
      calcula así (más rápido) mientras toda probabilidad predicha sea mayor que
      ``1e-200``; si no, con logsumexp exacto.
    - ``log_joint_t = log_pred_t + log_emission(Z[t])``;
      ``c_t = logsumexp(log_joint_t)``; ``log_alpha_t = log_joint_t - c_t``.
    - ``log_likelihood = Σ c_t``.

    Cada paso usa solo la vela *t* y el resultado del paso anterior, así que la
    fila *t* depende únicamente de ``Z[:t+1]``.
    """
    pi, a, mu, cv = _validate_params(startprob, transmat, means, covars)
    z = _validate_observations(Z, mu.shape[1])
    log_b, log_q = _emission_terms(z, mu, cv)
    t_len, k = log_b.shape
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
        log_a = np.log(a)

    probs = np.empty((t_len, k))
    log_likelihood = 0.0
    n_degenerate = 0
    log_alpha = log_pi
    alpha = pi
    for t in range(t_len):
        if t == 0:
            log_pred = log_pi
        else:
            # alpha ya está normalizado (suma 1): el producto no puede desbordar.
            # Si alguna probabilidad predicha es diminuta, se usa logsumexp exacto.
            pred = alpha @ a
            if pred.min() > _PRED_PROB_FLOOR:
                log_pred = np.log(pred)
            else:
                log_pred = _log_predict(log_alpha, log_a)
        log_joint = log_pred + log_b[t]
        m = log_joint.max()
        if m > -math.inf:
            e = np.exp(log_joint - m)  # el máximo vale 1: total >= 1
            total = e.sum()
            c = m + math.log(total)
            alpha = e / total
        else:
            log_joint, c = _degenerate_step(log_pred, log_b[t], log_q[t])
            n_degenerate += 1
            log_likelihood = -math.inf
            e = np.exp(log_joint - c)
            alpha = e / e.sum()
        log_alpha = log_joint - c
        log_likelihood += c
        probs[t] = alpha
    probs.setflags(write=False)
    return FilterResult(
        probs=probs, log_likelihood=float(log_likelihood), n_degenerate=n_degenerate
    )


# --- desde un modelo entrenado ------------------------------------------------------


def _feature_matrix(X_raw: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Matriz (T, F) en el orden de ``feature_names``; acepta ndarray o DataFrame."""
    names = list(feature_names)
    if hasattr(X_raw, "columns"):  # DataFrame: se eligen las columnas por nombre
        missing = [c for c in names if c not in X_raw.columns]
        if missing:
            raise ValueError(f"faltan las columnas de features: {missing}")
        return np.asarray(X_raw[names], dtype=float)
    x = np.asarray(X_raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(f"X_raw: se esperaba una matriz (T, {len(names)}), llegó forma {x.shape}")
    return x


def model_covariances(hmm: Any) -> np.ndarray:
    """Covarianzas completas ``(K, F, F)`` de un ``GaussianHMM``, para cualquier tipo.

    Con ``full``, ``diag`` y ``tied``, ``hmm.covars_`` ya viene así. Con
    ``spherical`` (una sola varianza por estado), hmmlearn 0.3.3 guarda
    ``_covars_`` como ``(K, F)`` con la varianza repetida (o ``(K,)``) y
    ``covars_`` lo aplana en K·F matrices; acá se arma ``varianza × identidad``
    para cada estado.
    """
    means = np.asarray(hmm.means_, dtype=float)
    k, f = means.shape
    if getattr(hmm, "covariance_type", "full") == "spherical":
        var = np.asarray(hmm._covars_, dtype=float).reshape(k, -1).mean(axis=1)
        return var[:, None, None] * np.eye(f)[None, :, :]
    return np.asarray(hmm.covars_, dtype=float)


def filtered_from_model(model: RegimeModel, X_raw: Any) -> FilterResult:
    """Filtra features crudas (unidades reales) con un ``RegimeModel`` entrenado.

    Aplica ``model.scaler.transform`` (medias y desvíos del entrenamiento, nunca
    de estos datos) y usa ``startprob_``, ``transmat_`` y ``means_`` de
    ``model.hmm`` y sus covarianzas completas (``model_covariances``).
    ``X_raw`` puede ser un ndarray ``(T, F)`` en el orden de
    ``model.scaler.feature_names`` o un DataFrame con esas columnas.
    """
    x = _feature_matrix(X_raw, model.scaler.feature_names)
    if not np.isfinite(x).all():
        raise ValueError(
            "X_raw: hay valores no finitos (sacá las primeras velas, las del calentamiento "
            "de las features)"
        )
    z = model.scaler.transform(x)
    hmm = model.hmm
    return filtered_probabilities(
        z, hmm.startprob_, hmm.transmat_, hmm.means_, model_covariances(hmm)
    )
