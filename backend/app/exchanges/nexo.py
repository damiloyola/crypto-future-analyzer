"""Nexo: precio de ejecución (cuando haya acceso) y órdenes (fase 4, bloqueadas).

Ver ``docs/nexo-api-research.md``. Resumen de por qué esto no consulta nada:

- La Nexo Pro API (``https://pro-api.nexo.io/api/v1``) firma TODAS las llamadas
  con API key, y no hay ningún endpoint público de precios de futuros
  confirmado.
- La fase 1 no usa API keys de exchanges.

Por eso:

- ``NexoPriceSource`` implementa ``PriceSource`` pero **nunca hace pedidos
  HTTP**. Con ``exchanges.nexo.enabled: false`` (lo normal) ``get_ticker`` lanza
  ``ExchangeUnavailableError("Nexo: sin acceso público a precios; se usa
  Binance")``; con ``enabled: true`` también lanza ``ExchangeUnavailableError``,
  explicando que todavía no hay un endpoint público confirmado. En los dos casos
  ``registry.get_execution_price`` usa el precio de Binance (o el que el usuario
  pegó a mano desde la app de Nexo).
- ``NexoOrderExecutor`` rechaza todo con ``TradingDisabledError``, sin importar
  lo que diga ``exchanges.nexo.allow_orders``.

---------------------------------------------------------------------------
NOTA PARA LA FASE 4 (NO IMPLEMENTADO; NO FUNCIONA; NO COPIAR A PRODUCCIÓN
SIN REVISAR LA DOC OFICIAL)
---------------------------------------------------------------------------
Según el código de la librería no oficial ``nexo-pro`` (2023), cada pedido a la
Nexo Pro API lleva tres headers:

- ``X-API-KEY``: la API key.
- ``X-NONCE``: timestamp en milisegundos (string), distinto en cada pedido.
- ``X-SIGNATURE``: ``base64(HMAC-SHA256(key=api_secret, msg=nonce))``.

Endpoints que interesarían: ``GET /futures/instruments``,
``GET /futures/positions?status=active`` y ``POST /futures/order`` (solo órdenes
``market``, sin parámetro de apalancamiento). Las keys irían en ``.env`` (las
escribe el usuario) y nunca se mostrarían ni se loguearían. Nada de esto se
implementa en la fase 1.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.exchanges.base import DisabledOrderExecutor, ExchangeUnavailableError, Ticker
from app.exchanges.binance import normalize_symbol

if TYPE_CHECKING:
    from app.config import NexoConfig

NEXO_DISABLED_MESSAGE = "Nexo: sin acceso público a precios; se usa Binance"
NEXO_NO_PUBLIC_ENDPOINT_MESSAGE = (
    "Nexo está habilitado en config.yaml, pero todavía no hay un endpoint público "
    "de precios confirmado (ver docs/nexo-api-research.md); se usa Binance"
)


class NexoPriceSource:
    """``PriceSource`` de Nexo. En la fase 1 nunca consulta la red (ver el módulo)."""

    name = "nexo"

    def __init__(self, config: NexoConfig, *, quote: str = "USDT") -> None:
        self._config = config
        self._quote = normalize_symbol(quote)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def instrument(self, symbol: str) -> str:
        """Instrumento de futuros de Nexo para el símbolo (``exchanges.nexo.instruments``)."""
        s = normalize_symbol(symbol)
        return self._config.instruments.get(s, s + self._quote)

    async def get_ticker(self, symbol: str) -> Ticker:
        normalize_symbol(symbol)  # valida igual que los otros conectores
        if not self._config.enabled:
            raise ExchangeUnavailableError(NEXO_DISABLED_MESSAGE)
        raise ExchangeUnavailableError(NEXO_NO_PUBLIC_ENDPOINT_MESSAGE)

    async def aclose(self) -> None:
        """No tiene recursos abiertos."""


@dataclass
class NexoOrderExecutor(DisabledOrderExecutor):
    """Órdenes en Nexo. En la fase 1 todo lanza ``TradingDisabledError``.

    Recibe la config solo para dejar claro que **no** la mira: aunque
    ``allow_orders`` fuera ``true``, se niega igual.
    """

    reason: str = "Fase 1: la app no ejecuta operaciones en Nexo."
    config: NexoConfig | None = field(default=None, repr=False)
