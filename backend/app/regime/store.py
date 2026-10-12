"""Persistencia de modelos de régimen en disco y regla de reentrenamiento.

- ``model_path(models_dir, symbol, timeframe)``: ``{models_dir}/{SYMBOL}_{tf}.joblib``
  (por ejemplo ``data/models/BTC_1h.joblib``). El modo ``--offline`` usa otra
  carpeta (``regime.synthetic_models_dir``) para no pisar los modelos reales.
  La moneda y el timeframe tienen que ser solo letras y números ASCII
  (``ValueError`` si no): así la ruta nunca sale de la carpeta de modelos y
  nunca se carga (con pickle) un archivo de otro lado.
- ``save_model(model, path)``: guarda con joblib de forma **atómica**: escribe
  un archivo temporal en la misma carpeta y después lo renombra con
  ``os.replace``. Si algo falla a mitad de camino, el modelo anterior queda
  intacto y no quedan temporales.
- ``load_model(path)``: devuelve el ``RegimeModel`` o ``None`` si no existe, si
  el archivo está roto, si no es un modelo de régimen o si su
  ``format_version`` no es ``MODEL_FORMAT_VERSION`` (en esos casos deja un
  aviso en el log). Atención: joblib usa pickle, que puede ejecutar código al
  cargar; solo se cargan archivos locales que generó la propia app.
- ``needs_retrain(model, now, retrain_every_days, expected_fingerprint)``:
  dice si hay que reentrenar y por qué, en palabras simples y sin paréntesis
  (el motivo se muestra dentro de otras frases): no hay modelo, es viejo,
  cambió la configuración, es de otra moneda/timeframe, es sintético cuando se
  piden datos reales o al revés.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib

from app.regime.types import MODEL_FORMAT_VERSION, RegimeModel

logger = logging.getLogger(__name__)

_SUFFIX = ".joblib"


def _safe_part(value: str, what: str) -> str:
    """``value`` sin espacios, si es solo letras y números ASCII; si no, ``ValueError``."""
    text = str(value).strip()
    if not (text and text.isascii() and text.isalnum()):
        raise ValueError(f"{what} inválido para el modelo de régimen: {value!r}")
    return text


def model_path(models_dir: Path | str, symbol: str, timeframe: str) -> Path:
    """Ruta del modelo de una moneda y un timeframe: ``{SYMBOL}_{tf}.joblib``.

    ``ValueError`` si la moneda o el timeframe tienen algo que no sea letras y
    números (por ejemplo ``../`` o ``/``).
    """
    sym = _safe_part(symbol, "símbolo").upper()
    tf = _safe_part(timeframe, "timeframe")
    return Path(models_dir) / f"{sym}_{tf}{_SUFFIX}"


def save_model(model: RegimeModel, path: Path | str) -> Path:
    """Guarda el modelo con joblib, creando la carpeta si hace falta (escritura atómica)."""
    if not isinstance(model, RegimeModel):
        raise TypeError("save_model espera un RegimeModel")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            joblib.dump(model, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return target


def _consistent(model: RegimeModel) -> bool:
    """Chequeo liviano de que el objeto cargado tiene sentido."""
    try:
        n = int(model.n_states)
        return (
            len(model.states) == n
            and int(model.hmm.n_components) == n
            and all(s.index == i for i, s in enumerate(model.states))
            and sorted(s.rank for s in model.states) == list(range(n))
        )
    except Exception:  # atributos faltantes o de otro tipo
        return False


def load_model(path: Path | str) -> RegimeModel | None:
    """Carga un modelo guardado; ``None`` si no hay uno usable (con aviso en el log).

    Solo para archivos locales de confianza: joblib/pickle puede ejecutar código.
    """
    source = Path(path)
    if not source.is_file():
        return None
    try:
        obj = joblib.load(source)
    except Exception as exc:  # archivo roto, truncado o de otra versión de librerías
        logger.warning("No se pudo leer el modelo de régimen %s (%s); se reentrena.", source, exc)
        return None
    if not isinstance(obj, RegimeModel):
        logger.warning(
            "El archivo %s no es un modelo de régimen (es %s); se ignora.",
            source,
            type(obj).__name__,
        )
        return None
    version = getattr(obj, "format_version", None)
    if version != MODEL_FORMAT_VERSION:
        logger.warning(
            "El modelo %s tiene formato %s y la app usa %s; se reentrena.",
            source,
            version,
            MODEL_FORMAT_VERSION,
        )
        return None
    if not _consistent(obj):
        logger.warning("El modelo %s está incompleto o inconsistente; se reentrena.", source)
        return None
    return obj


def _as_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _days_text(days: float) -> str:
    whole = int(days)
    if whole < 1:
        return "menos de 1 día"
    return "1 día" if whole == 1 else f"{whole} días"


def needs_retrain(
    model: RegimeModel | None,
    now: datetime,
    retrain_every_days: int,
    expected_fingerprint: str,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    synthetic: bool | None = None,
) -> tuple[bool, str]:
    """¿Hay que reentrenar? Devuelve ``(sí/no, motivo en palabras simples)``.

    Orden de los chequeos: que exista, que sea de la moneda y el timeframe
    pedidos, que sea real/sintético según lo pedido, que tenga el formato y la
    configuración actuales y, por último, la antigüedad (se reentrena cuando
    tiene ``retrain_every_days`` días o más). ``symbol``, ``timeframe`` y
    ``synthetic`` son opcionales: si vienen en ``None`` no se chequean.
    """
    if model is None:
        return True, "no hay modelo guardado"
    if symbol is not None and model.symbol.upper() != symbol.upper():
        return True, f"el modelo es de otra moneda: es de {model.symbol} y se pidió {symbol}"
    if timeframe is not None and model.timeframe != timeframe:
        return True, (
            f"el modelo es de otro timeframe: es de {model.timeframe} y se pidió {timeframe}"
        )
    if synthetic is not None and bool(model.synthetic) != bool(synthetic):
        if model.synthetic:
            return (
                True,
                "el modelo guardado es sintético, de datos de prueba, y se piden datos reales",
            )
        return True, "el modelo guardado es de datos reales y se pidió uno sintético"
    if getattr(model, "format_version", None) != MODEL_FORMAT_VERSION:
        return True, "el modelo guardado tiene un formato viejo"
    if model.config_fingerprint != expected_fingerprint:
        return True, "cambió la configuración del régimen"

    age_days = (_as_utc(now) - _as_utc(model.trained_at)).total_seconds() / 86400.0
    if age_days >= retrain_every_days:
        return True, (
            f"el modelo tiene {_days_text(age_days)} y se reentrena cada {retrain_every_days}"
        )
    return False, (
        f"el modelo está al día: tiene {_days_text(max(age_days, 0.0))} y "
        f"se reentrena cada {retrain_every_days}"
    )
