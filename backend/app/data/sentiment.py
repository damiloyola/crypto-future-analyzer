"""Fear & Greed Index de alternative.me.

``get_fear_greed`` devuelve el valor de hoy (0 a 100), la clasificación que
manda la API (en inglés y traducida) y el histórico de los últimos ``limit``
días, en orden cronológico, para el sparkline.

- Endpoint: ``GET {data_sources.fear_greed_url}?limit=N``. No se manda
  ``date_format`` (convierte el timestamp en una fecha en texto).
- La API manda todo como texto y de la más nueva a la más vieja; acá se pasa a
  ``int`` y ``datetime`` UTC y se ordena de la más vieja a la más nueva.
- Si ``metadata.error`` trae algo, o los datos no tienen el formato esperado, se
  lanza ``ExchangeError``. La caché (TTL ``cache_ttl_seconds.fear_greed``)
  devuelve entonces el último valor conocido marcado como ``stale`` y con un
  aviso en castellano; si no hay nada guardado, el error se propaga.
- La clasificación es la de alternative.me: no se recalcula con umbrales
  propios. La interpretación (contrarian o tendencia) la hace ``scoring``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import AppSettings
from app.data.cache import CacheResult, TTLCache, ttl_for, utcnow
from app.exchanges.base import ExchangeError, FetchMeta
from app.http_client import get_json

logger = logging.getLogger(__name__)

SOURCE = "alternative.me"
CACHE_KEY_PREFIX = "alternative.me:fear_greed"

CLASSIFICATION_ES: dict[str, str] = {
    "extreme fear": "Miedo extremo",
    "fear": "Miedo",
    "neutral": "Neutral",
    "greed": "Codicia",
    "extreme greed": "Codicia extrema",
}

STALE_WARNING = (
    "No pudimos actualizar el índice de miedo y codicia (alternative.me). "
    "Te mostramos el último dato que teníamos."
)


@dataclass(frozen=True)
class FearGreed:
    """Índice de miedo y codicia con su histórico."""

    value: int  # 0 a 100, el de hoy
    classification_es: str  # "Miedo extremo", "Miedo", "Neutral", "Codicia", "Codicia extrema"
    classification_en: str  # el texto tal cual lo manda la API
    history: list[tuple[datetime, int]]  # (día UTC, valor), de la más vieja a la más nueva
    meta: FetchMeta


def translate_classification(label: str) -> str:
    """Traduce la clasificación de alternative.me. Si no la conoce, la deja como vino."""
    return CLASSIFICATION_ES.get(label.strip().casefold(), label.strip())


def parse_fear_greed(payload: Any, meta: FetchMeta) -> FearGreed:
    """Convierte la respuesta de ``/fng/`` en ``FearGreed``.

    Lanza ``ExchangeError`` si la API informa un error o el formato no es el
    esperado. Si hay dos datos para el mismo día se queda con el último leído.
    """
    if not isinstance(payload, dict):
        raise ExchangeError(f"{SOURCE}: respuesta con formato inesperado")
    metadata = payload.get("metadata")
    error = metadata.get("error") if isinstance(metadata, dict) else None
    if error:
        raise ExchangeError(f"{SOURCE}: la API devolvió un error: {error}")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ExchangeError(f"{SOURCE}: la respuesta no trae datos")

    points: dict[datetime, tuple[int, str]] = {}
    for row in rows:
        try:
            value = int(row["value"])
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=UTC)
            label = str(row.get("value_classification") or "")
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
            raise ExchangeError(f"{SOURCE}: dato con formato inválido: {row!r}") from exc
        if not 0 <= value <= 100:
            raise ExchangeError(f"{SOURCE}: valor fuera de rango (0-100): {value}")
        points[day] = (value, label)

    ordered = sorted(points.items())
    _, (current, label_en) = ordered[-1]
    return FearGreed(
        value=current,
        classification_es=translate_classification(label_en),
        classification_en=label_en,
        history=[(day, value) for day, (value, _) in ordered],
        meta=meta,
    )


async def get_fear_greed(
    client: httpx.AsyncClient,
    cache: TTLCache,
    settings: AppSettings,
    limit: int = 30,
) -> FearGreed:
    """Fear & Greed actual y de los últimos ``limit`` días (cacheado).

    Lanza ``ExchangeError`` (o ``ExchangeUnavailableError``) solo si la consulta
    falla y no hay ningún valor guardado en la caché.
    """
    if limit < 1:
        raise ValueError("limit tiene que ser >= 1")
    url = settings.data_sources.fear_greed_url

    async def fetch() -> FearGreed:
        payload = await get_json(
            client,
            url,
            {"limit": limit},
            source=SOURCE,
            retries=settings.exchanges.max_retries,
        )
        return parse_fear_greed(payload, FetchMeta(source=SOURCE, fetched_at=utcnow()))

    result = await cache.get_or_fetch(
        f"{CACHE_KEY_PREFIX}:{limit}", ttl_for(settings.cache_ttl_seconds, "fear_greed"), fetch
    )
    return replace(result.value, meta=_meta_from_cache(result))


def _meta_from_cache(result: CacheResult[Any]) -> FetchMeta:
    if result.stale:
        logger.warning("%s: se usa el último valor guardado (%s)", SOURCE, result.error)
    return FetchMeta(
        source=SOURCE,
        fetched_at=result.fetched_at,
        from_cache=result.from_cache,
        stale=result.stale,
        warning=STALE_WARNING if result.stale else None,
    )
