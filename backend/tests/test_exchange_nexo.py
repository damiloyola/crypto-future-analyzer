"""Nexo en la fase 1: el precio nunca sale a la red y las órdenes siempre se rechazan."""

from __future__ import annotations

import httpx
import pytest

from app.config import NexoConfig
from app.exchanges.base import (
    DisabledOrderExecutor,
    ExchangeUnavailableError,
    OrderExecutor,
    PriceSource,
    TradingDisabledError,
)
from app.exchanges.nexo import (
    NEXO_DISABLED_MESSAGE,
    NEXO_NO_PUBLIC_ENDPOINT_MESSAGE,
    NexoOrderExecutor,
    NexoPriceSource,
)
from app.http_client import make_client


def nexo_config(settings, **update) -> NexoConfig:
    return settings.exchanges.nexo.model_copy(update=update)


def test_config_default_is_disabled(settings):
    assert settings.exchanges.nexo.enabled is False
    assert settings.exchanges.nexo.allow_orders is False


def test_price_source_protocol(settings):
    assert isinstance(NexoPriceSource(settings.exchanges.nexo), PriceSource)


async def test_disabled_raises_without_http_call(settings, respx_mock):
    # Sin rutas: cualquier pedido HTTP haría fallar el test.
    source = NexoPriceSource(nexo_config(settings, enabled=False))
    async with make_client() as client:
        with pytest.raises(ExchangeUnavailableError) as info:
            await source.get_ticker("BTC")
        assert not client.is_closed
    assert str(info.value) == NEXO_DISABLED_MESSAGE
    assert str(info.value) == "Nexo: sin acceso público a precios; se usa Binance"
    assert respx_mock.calls.call_count == 0


async def test_enabled_still_makes_no_http_call(settings, respx_mock):
    route = respx_mock.get(url__startswith="https://pro-api.nexo.io").mock(
        return_value=httpx.Response(200, json={})
    )
    source = NexoPriceSource(nexo_config(settings, enabled=True))
    with pytest.raises(ExchangeUnavailableError, match="endpoint público") as info:
        await source.get_ticker("BTC")
    assert str(info.value) == NEXO_NO_PUBLIC_ENDPOINT_MESSAGE
    assert route.call_count == 0
    assert respx_mock.calls.call_count == 0


async def test_invalid_symbol_is_value_error(settings):
    with pytest.raises(ValueError):
        await NexoPriceSource(settings.exchanges.nexo).get_ticker("BTC/USDT")


def test_instrument_mapping(settings):
    cfg = nexo_config(settings, instruments={"BTC": "BTCUSDT-PERP"})
    source = NexoPriceSource(cfg)
    assert source.instrument("btc") == "BTCUSDT-PERP"
    assert source.instrument("DOGE") == "DOGEUSDT"  # sin mapeo: base + quote


async def test_aclose_is_noop(settings):
    await NexoPriceSource(settings.exchanges.nexo).aclose()


@pytest.mark.parametrize("allow_orders", [False, True])
async def test_order_executor_refuses_everything(settings, allow_orders):
    # Aunque la config dijera allow_orders=true, en la fase 1 se niega igual.
    ex = NexoOrderExecutor(config=nexo_config(settings, allow_orders=allow_orders))
    assert isinstance(ex, OrderExecutor)
    assert isinstance(ex, DisabledOrderExecutor)
    with pytest.raises(TradingDisabledError, match="Nexo"):
        await ex.place_order("BTC", "long", 0.01)
    with pytest.raises(TradingDisabledError):
        await ex.set_leverage("BTC", 10)
    with pytest.raises(TradingDisabledError):
        await ex.get_positions()
    with pytest.raises(TradingDisabledError):
        await ex.close_position("BTC")


async def test_order_executor_without_config(respx_mock):
    with pytest.raises(TradingDisabledError):
        await NexoOrderExecutor().place_order("ETH", "short", 1)
    assert respx_mock.calls.call_count == 0
