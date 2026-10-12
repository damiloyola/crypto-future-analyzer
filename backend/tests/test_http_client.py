"""Traducción de errores HTTP a las excepciones de exchanges."""

import httpx
import pytest
import respx

from app.exchanges.base import ExchangeError, ExchangeUnavailableError
from app.http_client import get_bytes, get_json, get_text, make_client

URL = "https://api.example.com/x"


@respx.mock
async def test_get_json_ok():
    respx.get(URL).mock(return_value=httpx.Response(200, json={"a": 1}))
    async with make_client() as client:
        assert await get_json(client, URL, source="test") == {"a": 1}


@respx.mock
@pytest.mark.parametrize("status", [451, 403, 429, 500, 503])
async def test_unavailable_statuses(status):
    respx.get(URL).mock(return_value=httpx.Response(status))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError):
            await get_json(client, URL, source="test", retries=1, backoff_s=0)


@respx.mock
async def test_region_block_is_not_retried():
    route = respx.get(URL).mock(return_value=httpx.Response(451))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError):
            await get_json(client, URL, source="test", retries=3, backoff_s=0)
    assert route.call_count == 1


@respx.mock
async def test_retries_transient_then_succeeds():
    route = respx.get(URL).mock(
        side_effect=[httpx.ConnectError("x"), httpx.Response(200, json=[1])]
    )
    async with make_client() as client:
        assert await get_json(client, URL, source="test", retries=2, backoff_s=0) == [1]
    assert route.call_count == 2


@respx.mock
async def test_client_error_and_invalid_json():
    respx.get(URL).mock(return_value=httpx.Response(400, text="bad symbol"))
    async with make_client() as client:
        with pytest.raises(ExchangeError, match="400"):
            await get_json(client, URL, source="test")
    respx.get(URL).mock(return_value=httpx.Response(200, text="<html>"))
    async with make_client() as client:
        with pytest.raises(ExchangeError, match="JSON"):
            await get_json(client, URL, source="test")


@respx.mock
@pytest.mark.parametrize("status", [418, 429])
async def test_rate_limit_and_ip_ban_are_not_retried(status):
    # Binance: seguir pidiendo después de un 429 termina en baneo (418), y con 418
    # cada pedido alarga el baneo. Se corta enseguida y el mensaje dice cuánto esperar.
    route = respx.get(URL).mock(return_value=httpx.Response(status, headers={"Retry-After": "60"}))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError, match=r"pide esperar 60 s"):
            await get_json(client, URL, source="test", retries=2, backoff_s=0)
    assert route.call_count == 1


@respx.mock
async def test_server_errors_are_still_retried():
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
    )
    async with make_client() as client:
        assert await get_json(client, URL, source="test", retries=2, backoff_s=0) == {"ok": True}
    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize("kind", ["redirect_loop", "bad_gzip"])
async def test_request_errors_outside_transport_are_mapped(kind):
    # Regresión: TooManyRedirects y DecodingError no son TransportError y salían crudos,
    # así que no había fallback a Bybit ni aviso en castellano.
    if kind == "redirect_loop":
        route = respx.get(URL).mock(return_value=httpx.Response(302, headers={"Location": URL}))
    else:
        # Con ``stream`` el cuerpo se descomprime recién al leerlo dentro de ``client.get``,
        # como pasa con un servidor real.
        route = respx.get(URL).mock(
            side_effect=lambda request: httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=httpx.ByteStream(b"esto no es gzip"),
            )
        )
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError, match="respuesta inválida"):
            await get_json(client, URL, source="test", retries=2, backoff_s=0)
        with pytest.raises(ExchangeUnavailableError, match="respuesta inválida"):
            await get_bytes(client, URL, source="test", retries=2, backoff_s=0)
    # No se reintenta (retries=2): no es un problema pasajero de la red. Con el bucle de
    # redirecciones cada pedido son max_redirects + 1 llamadas.
    per_call = 1 if kind == "bad_gzip" else client.max_redirects + 1
    assert route.call_count == 2 * per_call


@respx.mock
async def test_get_bytes_and_get_text():
    body = '<?xml version="1.0" encoding="ISO-8859-1"?><rss>ñ</rss>'.encode("latin-1")
    respx.get(URL).mock(return_value=httpx.Response(200, content=body))
    async with make_client() as client:
        assert await get_bytes(client, URL, source="test") == body
        assert "rss" in await get_text(client, URL, source="test")
