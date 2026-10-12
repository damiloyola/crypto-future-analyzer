# Fixtures

Respuestas de las APIs públicas para los tests. Los tests nunca usan la red.
Detalle de cada endpoint (parámetros, unidades, rarezas) en
[`docs/data-sources.md`](../../../docs/data-sources.md).

## "Ahora" de los fixtures

Todos los fixtures de mercado están armados alrededor del mismo instante:

- **`FIXTURE_NOW = 2026-10-12T14:37:20Z`** (`1791815840000` ms, `1791815840` s).
- Próximo funding: `2026-10-12T16:00:00Z` (`1791820800000`).

Con ese "ahora", en las velas de 15m la de las **14:30 sigue abierta** y las
anteriores están cerradas; en las velas spot de 1h la de las **14:00 sigue
abierta**. Los precios de BTC son coherentes entre fuentes (≈ 108.960 USDT).

## Origen

- **docs**: copiado tal cual de la documentación oficial.
- **synthetic**: armado a mano con el formato de la documentación oficial
  (nombres de campos, tipos y orden verificados el 12/10/2026). Los números son
  inventados pero coherentes (OHLC válidos, tiempos alineados).
- **real**: respuesta grabada en vivo. Por ahora no hay ninguna: desde el
  entorno de desarrollo no se llega a las APIs.

## Índice

| Archivo | Endpoint | Origen | Notas |
|---|---|---|---|
| `binance/klines.json` | `GET /fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=8` | synthetic | 8 velas de 12:45 a 14:30, ascendentes. La última (14:30) está abierta: `closeTime` 1791816299999 > ahora. |
| `binance/spot_klines.json` | `GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=4` | synthetic | 4 velas de 11:00 a 14:00; la de 14:00 está abierta. Precios con 8 decimales, como spot. |
| `binance/ticker_24hr.json` | `GET /fapi/v1/ticker/24hr?symbol=BTCUSDT` | synthetic | `lastPrice` = cierre de la última vela de 15m. `priceChangePercent` en % (−1,248). |
| `binance/premium_index.json` | `GET /fapi/v1/premiumIndex?symbol=BTCUSDT` | synthetic | `lastFundingRate` 0.00008234 (fracción); `nextFundingTime` 16:00 UTC. |
| `binance/funding_rate.json` | `GET /fapi/v1/fundingRate?symbol=BTCUSDT&limit=5` | synthetic | 5 registros cada 8 h (11/10 00:00 a 12/10 08:00), ascendentes. Algunos `fundingTime` con 1-3 ms de más, como en la API real. Incluye `rateType`. Una tasa negativa. |
| `binance/funding_info.json` | `GET /fapi/v1/fundingInfo` | synthetic | Solo símbolos con ajustes. **AVAXUSDT con 4 h** (inventado, para probar el caso ≠ 8 h); BTCUSDT no aparece (= 8 h). |
| `binance/open_interest.json` | `GET /fapi/v1/openInterest?symbol=BTCUSDT` | synthetic | 81534.271 BTC. |
| `binance/open_interest_hist.json` | `GET /futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=5` | synthetic | 10:00 a 14:00, ascendente. OI subiendo +1,72 % en 4 h. Incluye `CMCCirculatingSupply`. |
| `binance/global_long_short_account_ratio.json` | `GET /futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=5` | synthetic | 10:00 a 14:00, ascendente. `longAccount` 0.6270 a 0.6551; `longShortRatio` = long/short. |
| `binance/error_invalid_symbol.json` | cualquier endpoint con símbolo inválido (HTTP 400) | docs | `{"code": -1121, "msg": "Invalid symbol."}` |
| `binance/error_restricted_location.json` | cualquier endpoint desde región bloqueada (HTTP 451) | synthetic | Texto tomado de reportes públicos. |
| `binance/error_rate_limit.json` | cualquier endpoint al pasarse del peso (HTTP 429) | synthetic | Código −1003. |
| `bybit/kline.json` | `GET /v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=8` | synthetic | Mismas 8 velas que Binance pero **la más nueva primero** y todo en strings. La primera (14:30) está abierta. |
| `bybit/tickers.json` | `GET /v5/market/tickers?category=linear&symbol=BTCUSDT` | synthetic | `price24hPcnt` −0.012424 (**fracción**); funding 0.00007891; `fundingIntervalHour` "8"; `openInterest` = 2 × `singleOpenInterest`. |
| `bybit/funding_history.json` | `GET /v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=5` | synthetic | Mismas horas que Binance, la más nueva primero. |
| `bybit/open_interest.json` | `GET /v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1h&limit=5` | synthetic | 14:00 a 10:00 (la más nueva primero), con `nextPageCursor`. |
| `bybit/account_ratio.json` | `GET /v5/market/account-ratio?category=linear&symbol=BTCUSDT&period=1h&limit=5` | synthetic | La más nueva primero; `buyRatio` 0.6187 a 0.5912. |
| `bybit/error_invalid_symbol.json` | kline con símbolo inválido (HTTP 200) | synthetic | `retCode` 10001. |
| `bybit/error_rate_limit.json` | cualquier endpoint (HTTP 200) | synthetic | `retCode` 10006. |
| `alternative_me/fng.json` | `GET https://api.alternative.me/fng/?limit=30` | synthetic | 30 días, el más nuevo primero (12/10 00:00 UTC). Valores como string, cubre las 5 clasificaciones. `time_until_update` solo en el primero. |
| `alternative_me/fng_error.json` | ídem, con error | synthetic | `data` vacío y `metadata.error` con texto. |
| `coingecko/global.json` | `GET /api/v3/global` | docs | Ejemplo verbatim de la spec OpenAPI oficial. `updated_at` = 1779878351 (27/05/2026; no sigue el "ahora" de los fixtures). Dominancia BTC 57,95 %. |
| `cryptopanic/posts.json` | `GET /api/developer/v2/posts/?auth_token=…` | synthetic | 7 posts: BTC ×2 (uno es `kind: media`), SOL ×2 (uno junto con XRP), ETH, AVAX y uno macro (Fed) sin monedas. Formato v2 (`instruments`, `original_url`). El más nuevo a las 14:15:20. |
| `rss/coindesk.xml` | `https://www.coindesk.com/arc/outboundfeeds/rss/` | synthetic | 10 notas (ver tabla de abajo). Una del 09/10 para probar el corte por antigüedad. Un título con `&amp;`. |
| `rss/cointelegraph.xml` | `https://cointelegraph.com/rss` | synthetic | 9 notas. Dos `pubDate` con offset `+0100` (14:10 +0100 = 13:10 UTC). Descripciones con HTML. |

### Qué tiene que matchear cada titular de los RSS

Con las palabras de `config.yaml` y **límites de palabra** (sin distinguir
mayúsculas). Las columnas "trampa" muestran lo que matchearía, mal, buscando por
"contiene".

| Feed | Titular | Monedas | Macro | Trampa por "contiene" |
|---|---|---|---|---|
| coindesk | Bitcoin Slips Below $109K as Traders Brace for U.S. CPI Data | BTC | cpi | |
| coindesk | Ether Options Traders Pile Into $5K Calls for December Expiry | ETH | | |
| coindesk | Stocks and Bitcoin Track S&P 500 Lower as Treasury Yields Climb | BTC | | |
| coindesk | Solana's Firedancer Client Goes Live for All Validators | SOL | | |
| coindesk | XRP Ledger Adds Native Lending; Ripple Says Banks Are Testing It | XRP | | |
| coindesk | Fed's Powell: Policy 'Well Positioned' Ahead of December Meeting | | fed, powell | |
| coindesk | Nvidia Second-Quarter Revenue Tops Estimates on AI Demand | | | sec |
| coindesk | Payments Startup Raises $40M to Build a Cross-Border Solution for Merchants | | | SOL |
| coindesk | Avalanche Foundation Unveils $50M Fund for Gaming Subnets | AVAX | | |
| coindesk | Corporate Treasuries Keep Buying Gold, Survey Shows (09/10) | | | rate |
| cointelegraph | Bitcoin ETFs log $612M inflows as BTC eyes $110K | BTC | (ninguna: "ETFs" en plural) | etf (acá "contiene" acierta) |
| cointelegraph | Ethereum developers set date for next hard fork | ETH | | |
| cointelegraph | SEC delays decision on spot AVAX ETF to December | AVAX | sec, etf | |
| cointelegraph | Solana memecoin launchpad volume hits new record | SOL | | |
| cointelegraph | Ripple CEO says XRP ETF approval is 'a matter of when' | XRP | etf | |
| cointelegraph | US inflation cools in September, boosting rate-cut bets | | inflation, rate | |
| cointelegraph | Methodology matters: How to secure your seed phrase | | | ETH, sec |
| cointelegraph | Console makers face tariff headwinds ahead of holiday season | | tariff | SOL |
| cointelegraph | Pirate-themed NFT game shuts down after two years (11/10) | | | rate |

Si editás un fixture a mano, mantené el "ahora" y la coherencia de precios y
horarios entre fuentes.
