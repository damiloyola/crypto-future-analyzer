"""Cliente HTTP compartido por los conectores de exchanges y las fuentes de datos.

Solo hace pedidos GET públicos (la fase 1 no usa endpoints privados) y traduce
los errores a las excepciones de ``app.exchanges.base``:

- sin conexión, timeout, HTTP 5xx, 451 (bloqueo por región), 403 y 429
  (límite de pedidos) → ``ExchangeUnavailableError``: corresponde probar el
  fallback o devolver el último valor cacheado;
- otros 4xx o una respuesta que no es JSON → ``ExchangeError``.

Reintenta con backoff exponencial solo los errores transitorios.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.exchanges.base import ExchangeError, ExchangeUnavailableError

USER_AGENT = "crypto-future-analyzer/0.1 (+uso personal, solo lectura)"
UNAVAILABLE_STATUS = {403, 418, 429, 451}


def make_client(timeout_s: float = 10.0) -> httpx.AsyncClient:
    """``AsyncClient`` con timeout y User-Agent. Quien lo crea lo cierra (``aclose``)."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    source: str,
    retries: int = 2,
    backoff_s: float = 0.5,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET que devuelve el JSON decodificado o lanza la excepción que corresponde."""
    resp = await _get(
        client, url, params, source=source, retries=retries, backoff_s=backoff_s, headers=headers
    )
    try:
        return resp.json()
    except ValueError as exc:
        raise ExchangeError(f"{source}: respuesta no es JSON ({url})") from exc


async def get_text(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    source: str,
    retries: int = 2,
    backoff_s: float = 0.5,
    headers: dict[str, str] | None = None,
) -> str:
    """GET que devuelve el cuerpo como texto (RSS, HTML)."""
    resp = await _get(
        client, url, params, source=source, retries=retries, backoff_s=backoff_s, headers=headers
    )
    return resp.text


async def _get(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None,
    *,
    source: str,
    retries: int,
    backoff_s: float,
    headers: dict[str, str] | None,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            await asyncio.sleep(backoff_s * 2 ** (attempt - 1))
        try:
            resp = await client.get(url, params=params, headers=headers)
        except httpx.TransportError as exc:  # sin red, DNS, timeout, conexión cortada
            last_exc = ExchangeUnavailableError(f"{source}: sin respuesta ({type(exc).__name__})")
            continue
        status = resp.status_code
        if status < 400:
            return resp
        if status >= 500 or status in UNAVAILABLE_STATUS:
            last_exc = ExchangeUnavailableError(f"{source}: HTTP {status} en {url}")
            if status in (403, 451):  # bloqueo por región: reintentar no sirve
                break
            continue
        raise ExchangeError(f"{source}: HTTP {status} en {url}: {resp.text[:200]}")
    assert last_exc is not None
    raise last_exc
