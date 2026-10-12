"""Datos del mercado cripto global (CoinGecko ``/global``).

``get_global_market`` devuelve la dominancia de BTC y ETH, la capitalización
total en USD y su variación en 24 h. Se usa como contexto para el LLM y en el
header de la UI.

- Endpoint: ``GET {data_sources.coingecko_base_url}/global``, sin key (con el
  TTL de ``cache_ttl_seconds.global_market`` alcanza el límite gratuito).
- Unidades: CoinGecko manda la dominancia y la variación 24 h en **porcentaje**
  (``57.95`` = 57,95 %) y ``updated_at`` en segundos. Se dejan así.
- Si la respuesta no trae los campos obligatorios (dominancia de BTC y
  capitalización total) se lanza ``ExchangeError``. La caché devuelve entonces
  el último valor conocido marcado como ``stale`` y con un aviso en castellano;
  si no hay nada guardado, el error se propaga.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import AppSettings
from app.data.cache import CacheResult, TTLCache, ttl_for, utcnow
from app.exchanges.base import ExchangeError, FetchMeta
from app.http_client import get_json

logger = logging.getLogger(__name__)

SOURCE = "coingecko"
CACHE_KEY = "coingecko:global"

STALE_WARNING = (
    "No pudimos actualizar los datos del mercado global (CoinGecko). "
    "Te mostramos los últimos que teníamos."
)


@dataclass(frozen=True)
class GlobalMarket:
    """Foto del mercado cripto completo."""

    btc_dominance_pct: float  # en %, 57.95 = 57,95 %
    eth_dominance_pct: float | None  # en %
    total_market_cap_usd: float
    market_cap_change_24h_pct: float | None  # en %, -1.6 = -1,6 %
    meta: FetchMeta
    updated_at: datetime | None = None  # cuándo calculó CoinGecko el dato (UTC)


def parse_global_market(payload: Any, meta: FetchMeta) -> GlobalMarket:
    """Convierte la respuesta de ``/global`` en ``GlobalMarket``.

    Lanza ``ExchangeError`` si faltan la dominancia de BTC o la capitalización
    total, o si tienen valores imposibles. Los campos opcionales inválidos
    quedan en ``None``.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ExchangeError(f"{SOURCE}: respuesta con formato inesperado")
    dominance = data.get("market_cap_percentage")
    market_cap = data.get("total_market_cap")
    dominance = dominance if isinstance(dominance, dict) else {}
    market_cap = market_cap if isinstance(market_cap, dict) else {}

    btc = _number(dominance.get("btc"))
    total = _number(market_cap.get("usd"))
    if btc is None or not 0 < btc <= 100:
        raise ExchangeError(f"{SOURCE}: dominancia de BTC inválida: {dominance.get('btc')!r}")
    if total is None or total <= 0:
        raise ExchangeError(f"{SOURCE}: capitalización total inválida: {market_cap.get('usd')!r}")
    eth = _number(dominance.get("eth"))
    if eth is not None and not 0 <= eth <= 100:
        eth = None

    updated_at = None
    updated_s = _number(data.get("updated_at"))
    if updated_s is not None and updated_s > 0:
        updated_at = datetime.fromtimestamp(updated_s, tz=UTC)

    return GlobalMarket(
        btc_dominance_pct=btc,
        eth_dominance_pct=eth,
        total_market_cap_usd=total,
        market_cap_change_24h_pct=_number(data.get("market_cap_change_percentage_24h_usd")),
        meta=meta,
        updated_at=updated_at,
    )


async def get_global_market(
    client: httpx.AsyncClient, cache: TTLCache, settings: AppSettings
) -> GlobalMarket:
    """Dominancia y capitalización del mercado (cacheado).

    Lanza ``ExchangeError`` (o ``ExchangeUnavailableError``) solo si la consulta
    falla y no hay ningún valor guardado en la caché.
    """
    url = f"{settings.data_sources.coingecko_base_url.rstrip('/')}/global"

    async def fetch() -> GlobalMarket:
        payload = await get_json(client, url, source=SOURCE, retries=settings.exchanges.max_retries)
        return parse_global_market(payload, FetchMeta(source=SOURCE, fetched_at=utcnow()))

    result = await cache.get_or_fetch(
        CACHE_KEY, ttl_for(settings.cache_ttl_seconds, "global_market"), fetch
    )
    return replace(result.value, meta=_meta_from_cache(result))


def _number(value: Any) -> float | None:
    """``float`` finito o ``None`` (acepta números y strings numéricos)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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
