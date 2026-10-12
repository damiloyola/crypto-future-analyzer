# Fuentes de datos públicas (Hito 2)

Fecha de verificación: **12/10/2026**. Referencia para los conectores
(`exchanges/binance.py`, `exchanges/bybit.py`) y las fuentes de datos
(`data/sentiment.py`, `data/market.py`, `data/news.py`).

## Cómo se verificó y qué tan confiable es

Desde el entorno de desarrollo **no se puede llegar a las APIs ni a sus páginas de
documentación** (los dominios no resuelven o el proxy los rechaza). Por eso no hay
respuestas grabadas en vivo: todos los fixtures son de la documentación o
sintéticos. Lo que sí se pudo leer:

| Fuente | Qué se leyó | Confianza |
|---|---|---|
| Binance USDⓈ-M | `binance/binance-connector-python`, SDK oficial generado de la especificación OpenAPI de Binance (commit del 09/10/2026): rutas, pesos, notas, enums, campos y tipos de respuesta, changelog. Resultados de búsqueda de developers.binance.com para los `limit`. | Alta |
| Binance spot | `binance/binance-spot-api-docs` (repo oficial de la doc, commit del 07/10/2026): `rest-api.md` y `errors.md`. | Alta |
| Bybit v5 | `bybit-exchange/docs`, el repo de donde sale bybit-exchange.github.io (commit del 09/10/2026): páginas de cada endpoint, códigos de error, límites, enums y changelog. | Alta |
| CoinGecko | `coingecko/coingecko-api-oas`, especificación OpenAPI oficial (commit del 24/09/2026), más resultados de búsqueda de docs.coingecko.com para los límites. | Alta |
| alternative.me | No se pudo abrir la doc oficial. Formato conocido más fuentes secundarias de 2026 que lo usan igual. | Media |
| CryptoPanic | No se pudo abrir la doc oficial. Fuentes secundarias de 2026 (SDK PHP `tigusigalpa/cryptopanic-php`, guía de dltHub). | Baja |
| RSS | Fuentes secundarias de octubre de 2026 (Feeder.co y otras) y el artículo de CoinDesk con la URL de su feed. | Media |

**Antes de dar por cerrado el Hito 2 conviene correr una vez los conectores contra
las APIs reales desde tu máquina** (ahí sí hay red) y, si algo difiere, actualizar
este documento y los fixtures.

## Convenciones que valen para todo

- La app trabaja con fracciones para las tasas (`0.0001` = 0,01 %) y con
  `datetime` UTC (ver `exchanges/base.py`). Cada fuente usa unidades distintas:
  la tabla siguiente dice qué convertir.
- `http_client.get_json` ya convierte HTTP 403/418/429/451/5xx, los errores de
  red y las respuestas rotas (redirecciones infinitas, cuerpo mal comprimido) en
  `ExchangeUnavailableError`, y los demás 4xx en `ExchangeError`. Solo reintenta
  lo pasajero (sin red, timeout, 5xx): 403/451 son bloqueo por región y 418/429
  son límite de pedidos, así que se pasa enseguida al respaldo o a la caché (el
  mensaje incluye el `Retry-After` si viene). **Bybit responde los errores con
  HTTP 200** y `retCode != 0`, así que eso lo tiene que revisar su conector.
- Si Binance y Bybit **rechazan** el pedido (por ejemplo un símbolo que no
  existe: Binance `-1121`, Bybit `10001` "Not supported symbols"), el error es
  `ExchangeError` y no "no respondió": la app dice que revises el símbolo en vez
  de "probá de nuevo".

| Dato | Binance | Bybit | Qué hacer |
|---|---|---|---|
| Funding rate | fracción (`"0.00010000"`) | fracción (`"0.0001"`) | usar tal cual |
| Variación 24 h | **porcentaje** (`priceChangePercent: "-1.248"` = −1,248 %) | **fracción** (`price24hPcnt: "-0.012424"` = −1,2424 %) | `Ticker.change_24h_pct` va en %: Bybit × 100 |
| Long/short | `longAccount` fracción 0-1 y `longShortRatio` | `buyRatio`/`sellRatio` fracción 0-1, sin ratio | en Bybit, ratio = `buyRatio / sellRatio` |
| Open interest | en moneda base (BTC) | en moneda base, **suma de los dos lados** | no comparar valores absolutos entre exchanges (ver Bybit) |
| Tiempos | enteros en ms | **strings** en ms | `int(...)` y después a UTC |
| Precios y volúmenes | strings | strings | `float(...)` |

---

## Binance USDⓈ-M futures (fuente principal)

Base: `https://fapi.binance.com`. Sin API key para todo lo de abajo.

### Generales

- **Límite por IP**: se mide en "peso" por minuto (`REQUEST_WEIGHT`). El valor
  vigente sale de `GET /fapi/v1/exchangeInfo` → `rateLimits`; históricamente es
  2400 por minuto (no se pudo confirmar el número actual). Cada respuesta trae el
  header `X-MBX-USED-WEIGHT-1M` con lo consumido.
- **429**: te pasaste del límite; viene el header `Retry-After` en segundos. Si
  seguís pidiendo, **418** (IP baneada por un rato, también con `Retry-After`).
- **403**: lo bloqueó el firewall (WAF), por límite o por seguridad.
- **451**: Binance bloquea la región o el datacenter de donde sale el pedido. El
  cuerpo es `{"code":0,"msg":"Service unavailable from a restricted location
  according to 'b. Eligibility' in https://www.binance.com/en/terms. ..."}`. Pasa
  con IPs de EE. UU. y con muchos proveedores de nube. Desde tu casa en Argentina
  no debería pasar; si pasa, se usa Bybit.
- **Errores de la API**: HTTP 4xx con cuerpo `{"code": -1121, "msg": "Invalid
  symbol."}`. Los códigos son fijos; los mensajes pueden cambiar.
- **5xx**: estado desconocido; reintentar y después caer al fallback.
- Todos los tiempos vienen en **milisegundos** (enteros).

### `GET /fapi/v1/klines`: velas de perpetuos

| Parámetro | Obligatorio | Notas |
|---|---|---|
| `symbol` | sí | `BTCUSDT` |
| `interval` | sí | `1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M` (en futuros **no** hay `1s`) |
| `startTime`, `endTime` | no | ms. Sin ninguno, devuelve las más recientes |
| `limit` | no | por defecto 500, **máximo 1500** |

Peso según `limit`: `[1,100)` → 1, `[100,500)` → 2, `[500,1000]` → 5, `>1000` → 10.
Ojo: **`limit=500` cuesta 5 y `limit=499` cuesta 2**.

Respuesta: lista de listas, de la más vieja a la más nueva:

```
[openTime(ms int), open, high, low, close, volume(base), closeTime(ms int),
 quoteVolume, trades(int), takerBuyBase, takerBuyQuote, "0"]
```

- Las velas se identifican por `openTime`. `closeTime = openTime + intervalo − 1 ms`.
- **La última vela es la que se está formando** si no pasás `endTime`. Para
  marcar `closed`: `closeTime < ahora` (o `openTime + intervalo <= ahora`).
- Paginación hacia atrás: pedir con `endTime = openTime_más_viejo − 1`.
- `volume` está en la moneda base (BTC), que es lo que pide `base.py`.

### `GET /api/v3/klines`: velas spot (si `use_perpetual_klines: false`)

Base `https://api.binance.com`. Mismo formato de respuesta. Peso fijo **2**.
`limit` por defecto 500, **máximo 1000**. Intervalos: los de arriba más `1s`.
Parámetro extra `timeZone` (por defecto `0` = UTC; cambia cómo se arman las velas
de 1d, no hay que usarlo). Los tiempos vienen en ms salvo que mandes el header
`X-MBX-TIME-UNIT: MICROSECOND` (no mandarlo).

### `GET /fapi/v1/ticker/24hr`: ticker 24 h

`symbol` opcional. Peso **1** con símbolo, **40** sin símbolo (devuelve todos en
una lista; no hace falta). Campos (todos strings salvo los marcados):
`symbol, priceChange, priceChangePercent (en %), weightedAvgPrice, lastPrice,
lastQty, openPrice, highPrice, lowPrice, volume (base), quoteVolume (USDT),
openTime (ms int), closeTime (ms int), firstId, lastId, count (int)`.

> `GET /fapi/v1/ticker/price` figura como **deprecado** en la spec. No usarlo.

### `GET /fapi/v1/premiumIndex`: funding actual y mark price

`symbol` opcional. Peso **1** con símbolo, **10** sin símbolo. Campos:
`symbol, markPrice, indexPrice, estimatedSettlePrice, lastFundingRate (fracción),
interestRate, nextFundingTime (ms int), time (ms int)`.

`lastFundingRate` es el funding estimado del período en curso, que se cobra en
`nextFundingTime`. Es lo que va a `FundingInfo.rate`.

### `GET /fapi/v1/fundingRate`: historial de funding

| Parámetro | Notas |
|---|---|
| `symbol` | opcional en la API; mandarlo siempre |
| `startTime`, `endTime` | ms, inclusivos |
| `limit` | por defecto 100, máximo 1000 |

- Comparte un límite propio de **500 pedidos cada 5 minutos por IP** con
  `/fapi/v1/fundingInfo`.
- Orden **ascendente** (más viejo primero).
- La doc se contradice: dice "por defecto 100" y también "sin `startTime` ni
  `endTime` devuelve los últimos 200". **Mandar siempre `limit`**.
- Campos: `symbol, fundingTime (ms int), fundingRate (fracción), markPrice,
  rateType`. `fundingTime` puede venir con algunos ms de más (ej.
  `1791705600001`); no asumir que cae justo en la hora.
- **Nuevo (julio de 2026)**: `rateType` = `"Regular"` o `"Special"` (funding extra
  por dividendos en contratos de acciones). Para cripto siempre es `Regular`; si
  aparece `Special`, ignorarlo en el historial.

### `GET /fapi/v1/fundingInfo`: intervalos y topes de funding

Sin parámetros. Peso 0 (comparte los 500/5 min con `fundingRate`). Devuelve
**solo los símbolos con algún ajuste** (tope, piso o intervalo distinto de 8 h):
`symbol, adjustedFundingRateCap, adjustedFundingRateFloor, fundingIntervalHours
(int), disclaimer (bool, ignorar)`. Si el símbolo no está en la lista, su
intervalo es 8 h. Sirve para llenar `FundingInfo.interval_hours`.

### `GET /fapi/v1/openInterest`: open interest actual

`symbol` obligatorio. Peso 1. Campos: `openInterest` (string, moneda base),
`symbol`, `time` (ms int). No trae valor en USD: multiplicar por el mark price.

### `GET /futures/data/openInterestHist`: historial de open interest

| Parámetro | Notas |
|---|---|
| `symbol`, `period` | obligatorios. `period`: `5m 15m 30m 1h 2h 4h 6h 12h 1d` |
| `limit` | por defecto 30, máximo 500 |
| `startTime`, `endTime` | opcionales |

- Peso 0, pero límite propio de **1000 pedidos cada 5 minutos por IP**.
- **Solo guarda el último mes (~30 días)**. No sirve para el backtest de 18 meses.
- Orden ascendente. Campos: `symbol, sumOpenInterest (base), sumOpenInterestValue
  (USD), CMCCirculatingSupply (nuevo, ignorar), timestamp (ms int)`.

### `GET /futures/data/globalLongShortAccountRatio`: ratio long/short de cuentas

Mismos parámetros, límites y retención (**30 días**) que `openInterestHist`.
Orden ascendente. Campos: `symbol, longShortRatio, longAccount (fracción),
shortAccount (fracción), timestamp (ms int)`.

---

## Bybit v5 (fallback)

Base: `https://api.bybit.com` (dominio alternativo oficial: `api.bybick.com`).
Siempre `category=linear` para los perpetuos USDT.

### Generales

- **Todas las respuestas vienen envueltas** y con HTTP 200, aunque haya error:

  ```json
  {"retCode": 0, "retMsg": "OK", "result": {...}, "retExtInfo": {}, "time": 1791815840000}
  ```

  `retCode != 0` es error. Los más comunes en datos públicos:

  | `retCode` | Qué es | Excepción sugerida |
  |---|---|---|
  | 10001 | parámetro inválido (incluye símbolo inexistente) | `ExchangeError` |
  | 10006 | demasiados pedidos | `ExchangeUnavailableError` |
  | 10000, 10016 | timeout o error del servidor | `ExchangeUnavailableError` |
  | 10009 | servicio restringido para tu región | `ExchangeUnavailableError` |

- **Límite por IP**: 600 pedidos cada 5 segundos. Si te pasás, **HTTP 403**
  ("access too frequent") y hay que esperar al menos 10 minutos. También da 403 si
  la IP es de EE. UU. 429 = protección general del sistema (la app no reintenta
  enseguida: usa la caché).
- **Todo viene como string**, incluidos los tiempos (ms).
- **Las listas vienen de la más nueva a la más vieja** (kline seguro, según la
  doc; open-interest y account-ratio según los ejemplos; funding/history en la
  práctica). **Ordenar siempre por tiempo** antes de devolver.

Equivalencias de timeframes:

| App | `kline.interval` | `open-interest.intervalTime` y `account-ratio.period` |
|---|---|---|
| 5m | `5` | `5min` |
| 15m | `15` | `15min` |
| 1h | `60` | `1h` |
| 4h | `240` | `4h` |
| 1d | `D` | `1d` |

### `GET /v5/market/kline`

Parámetros: `category` (por defecto `linear`), `symbol` (mayúsculas), `interval`
(`1 3 5 15 30 60 120 240 360 720 D W M`), `start`, `end` (ms), `limit` (1 a 1000,
**por defecto 200**: mandar siempre el `limit`).

`result.list`: listas de 7 strings `[startTime, open, high, low, close, volume,
turnover]`, **la más nueva primero**. En USDT, `volume` está en la moneda base y
`turnover` en USDT. Mientras la vela está abierta, `close` es el último precio. No
trae `closeTime`: la vela está cerrada si `startTime + intervalo <= ahora`.

### `GET /v5/market/tickers`

`category` obligatorio, `symbol` opcional. `result.list[0]` trae, entre otros:
`lastPrice, indexPrice, markPrice, prevPrice24h, price24hPcnt (fracción),
highPrice24h, lowPrice24h, prevPrice1h, openInterest, openInterestValue,
singleOpenInterest, singleOpenInterestValue, turnover24h (USDT), volume24h (base),
fundingRate (fracción), nextFundingTime (ms string), fundingIntervalHour (string,
horas enteras), fundingCap`.

Sirve a la vez para `get_ticker` y `get_funding` (no hay un `premiumIndex`
aparte).

### `GET /v5/market/funding/history`

Parámetros: `category=linear`, `symbol`, `startTime`, `endTime`, `limit` (1 a 200,
por defecto 200). Ojo: **mandar solo `startTime` da error**; solo `endTime`
devuelve 200 registros hasta ese momento. Campos: `symbol, fundingRate
(fracción), fundingRateTimestamp (ms string)`.

### `GET /v5/market/open-interest`

Parámetros: `category=linear`, `symbol`, `intervalTime` (obligatorio: `5min 15min
30min 1h 4h 1d`), `startTime`, `endTime`, `limit` (1 a 200, **por defecto 50**),
`cursor`. Campos: `openInterest`, `singleOpenInterest`, `timestamp`, más
`nextPageCursor` para paginar. A diferencia de Binance, guarda historia desde que
se listó el contrato.

**Cambio de junio de 2026**: la doc ahora dice que `openInterest` es la **suma de
los dos lados** y agregó `singleOpenInterest` (un lado, la mitad). Como el scoring
usa la variación porcentual del OI, alcanza con usar siempre el mismo campo
(`openInterest`) dentro de una serie. Lo que no hay que hacer es comparar el valor
absoluto de Bybit con el de Binance ni mezclarlos en una misma serie.

### `GET /v5/market/account-ratio`

Parámetros: `category=linear`, `symbol`, `period` (`5min 15min 30min 1h 4h 1d`;
la página de enums dice `4d`, es un error de tipeo de la doc), `startTime`,
`endTime`, `limit` (1 a 500, **por defecto 50**), `cursor`. Campos: `symbol,
buyRatio, sellRatio (fracciones), timestamp (ms string)`. Datos desde el
20/07/2020 (más historia que Binance).

---

## alternative.me: Fear & Greed Index

`GET https://api.alternative.me/fng/?limit=N`

- `limit`: cuántos días (por defecto 1; `0` = todo el historial).
- `format`: `json` (por defecto) o `csv`. `date_format`: `us`, `cn`, `kr` o
  `world`; si se manda, `timestamp` pasa a ser una fecha en texto. **No mandar
  ninguno de los dos.**
- Respuesta:

  ```json
  {"name": "Fear and Greed Index",
   "data": [{"value": "38", "value_classification": "Fear",
             "timestamp": "1791763200", "time_until_update": "33760"}, ...],
   "metadata": {"error": null}}
  ```

- `value` es un **string** (0 a 100). `timestamp` es un **string en segundos**
  (00:00 UTC de cada día). `time_until_update` (segundos) viene **solo en el
  primer elemento**. Lista de la más nueva a la más vieja.
- `value_classification`: `Extreme Fear`, `Fear`, `Neutral`, `Greed`,
  `Extreme Greed`. Usar el texto que manda la API, no recalcularlo.
- Si algo falla, `metadata.error` trae un texto y `data` puede venir vacío: tratarlo
  como `ExchangeError`.
- Se actualiza una vez por día; el TTL de 1800 s de `config.yaml` sobra.
- Límite publicado (no reverificado): 60 pedidos por minuto.

## CoinGecko: `GET /api/v3/global`

Base pública: `https://api.coingecko.com/api/v3`.

- **Sin key** funciona, con límite por IP compartido y variable (unos 10-30
  pedidos por minuto). **Con key Demo** (gratis): header `x-cg-demo-api-key` o
  parámetro `x_cg_demo_api_key`, 100 pedidos por minuto desde el 17/05/2026 y
  10.000 por mes. La key Pro usa otra base (`pro-api.coingecko.com`). Con el TTL de
  600 s alcanza sin key.
- Los pedidos con error también cuentan para el límite. Si te pasás: HTTP 429.
- Campos que usa la app (dentro de `data`):
  - `total_market_cap.usd`: capitalización total en USD (número).
  - `market_cap_percentage.btc`: dominancia de BTC en **porcentaje** (`57.95` =
    57,95 %). Las claves son los símbolos en minúscula.
  - `market_cap_change_percentage_24h_usd`: variación 24 h en **porcentaje**.
  - `updated_at`: **segundos** Unix (entero).
- El dato se actualiza más o menos cada 10 minutos (fuente secundaria).

## CryptoPanic (solo si hay `CRYPTOPANIC_API_KEY`)

**Estado: no verificado contra la doc oficial.**

- Base de la v2: `https://cryptopanic.com/api/{plan}/v2`, con `{plan}` =
  `developer`, `growth` o `enterprise` según el plan de **tu** key. La URL de
  `config.yaml` (`.../api/developer/v2`) sigue ese formato, pero solo sirve si tu
  key es del plan Developer.
- Un SDK actualizado en septiembre de 2026 y una guía de dltHub dicen que el
  **plan gratis Developer se discontinuó** a principios de 2026. No se pudo
  confirmar. Si no tenés key, la app usa los RSS y no pasa nada.
- Endpoint: `GET {base}/posts/` (con la barra final). Parámetros:
  `auth_token` (obligatorio), `currencies=BTC,ETH`, `kind=news|media|all`,
  `filter=rising|hot|bullish|bearish|important|saved|lol`, `regions=en` (por
  defecto), `public=true`, `page`. Algunos (`size`, `search`, `panic_period`,
  `with_content`, `last_pull`) son solo para planes pagos.
- Respuesta: `{"next": url|null, "previous": url|null, "results": [...]}`. Cada post
  trae `id, slug, title, description, published_at (ISO 8601), created_at, kind,
  source {title, region, domain, type}, original_url, url (página de CryptoPanic),
  image, instruments [{code, title, slug}], votes {positive, negative, important,
  liked, disliked, lol, toxic, saved, comments}, panic_score, author`.
- **Cambios respecto de la v1**: las monedas vienen en `instruments` (antes
  `currencies`); hay `original_url` (link al medio, puede venir vacío según el
  plan). Para el link usar `original_url` y, si está vacío, `url`. Conviene que el
  parser acepte también `currencies` por si acaso.
- `next` trae tu `auth_token` en la URL: no loguearla.

## RSS (sin key; fuente de noticias por defecto)

| Feed | URL | Estado |
|---|---|---|
| CoinDesk | `https://www.coindesk.com/arc/outboundfeeds/rss/` | Vigente según fuentes de octubre de 2026. Un solo feed con las últimas ~25 notas. La forma "oficial" que usa el botón del sitio también acepta `?outputType=xml`. La vieja `https://www.coindesk.com/feed/` dejó de andar en 2024. |
| Cointelegraph | `https://cointelegraph.com/rss` | Vigente según fuentes de octubre de 2026. Últimas ~30 notas. Alternativas: `/feed` y feeds por tema `/rss/tag/bitcoin`. |

- Ambos son RSS 2.0. Títulos dentro de `CDATA`; puede haber entidades (`&amp;`).
  Fecha en `pubDate` (RFC 822, con offset: convertir a UTC; `feedparser` lo hace en
  `published_parsed`).
- La descripción de Cointelegraph trae HTML (`<p>`, `<img>`): limpiarlo si se
  muestra.
- **Filtrar con límites de palabra**, no con "contiene". Con "contiene", `sol`
  matchea "solution" y "console", `eth` matchea "methodology", `sec` matchea
  "second" y "secure", y `rate` matchea "corporate" y "pirate". Los fixtures
  traen esos casos. Con límites de palabra aparece el problema inverso: `etf` no
  matchea "ETFs" y `fed` no matchea "Federal Reserve" (ver abajo).

---

## Qué cambió respecto de `config.yaml`

1. **`data_sources.cryptopanic_base_url`**: el formato `/api/{plan}/v2` está bien,
   pero el plan depende de tu key (`developer`, `growth` o `enterprise`) y el
   gratis Developer estaría discontinuado. El endpoint es `/posts/` con barra
   final, y la respuesta v2 usa `instruments` en lugar de `currencies`.
2. **`data_sources.macro_keywords`**: si se filtra por palabra completa (lo
   correcto), faltan plurales y formas largas: `etfs`, `federal reserve`,
   `rates`. Si se filtrara por "contiene", `sec` y `rate` dan falsos positivos.
3. **`fees.funding_interval_hours: 8`** y el "funding por período de 8 h" de
   `scoring.derivatives`: no todos los perpetuos cobran cada 8 h. Binance lo
   informa en `fundingInfo.fundingIntervalHours` y Bybit en
   `tickers.fundingIntervalHour`. El valor de config tendría que ser solo el de
   respaldo, y los umbrales de funding normalizarse a 8 h si el intervalo real es
   otro.
4. **Retención de derivados**: Binance guarda solo ~30 días de
   `openInterestHist` y `globalLongShortAccountRatio`. La validación
   walk-forward de 18 meses no puede reconstruir el componente de derivados con
   Binance; Bybit sí tiene historia larga (`account-ratio` desde 2020).
5. **`timeframes.candles_per_request: 500`**: entra en todos los límites (Binance
   futuros 1500, spot 1000, Bybit 1000), pero Bybit devuelve 200 si no se manda
   `limit`, y en Binance `limit=500` pesa 5 contra 2 de `limit=499`.
6. Sin cambios: `exchanges.binance.*_base_url`, `exchanges.bybit.base_url`,
   `fear_greed_url`, `coingecko_base_url` y los dos `rss_feeds` siguen vigentes.
