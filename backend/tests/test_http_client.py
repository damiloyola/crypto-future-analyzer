"""Traducción de errores HTTP a las excepciones de exchanges."""

import httpx
import pytest
import respx

from app.exchanges.base import ExchangeError, ExchangeUnavailableError
from app.http_client import get_json, make_client

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
