# Investigación: Nexo Pro API (Hito 1)

Fecha: 12/10/2026.

## Cómo se investigó y qué tan confiable es

- **Documentación oficial**: las páginas oficiales son
  `https://pro.nexo.com/apiDocPro.html` y `https://pro.nexo.io/api-doc-pro`.
  Ninguna se pudo abrir desde el entorno donde se desarrolla esto: los dominios
  de Nexo no resuelven. Lo que viene de ahí está **citado por resultados de
  búsqueda**, no leído de primera mano.
- **Código fuente de la librería no oficial `nexo-pro`** (npm, v1.2.2, agosto de
  2023): leído completo. Es la evidencia más concreta sobre endpoints,
  parámetros y firma, pero tiene más de 3 años y puede estar desactualizada.
- **Páginas y blog de Nexo** (nexo.com/futures y posts del blog), vistos a
  través de resultados de búsqueda.

Conviene que confirmes los puntos marcados como **no confirmado** abriendo la
documentación oficial desde tu Mac. Si me pegás el contenido, actualizo este
informe.

## Lo que muestra el código de `nexo-pro`

- Base URL: `https://pro-api.nexo.io/api/v1`
- **Todas** las llamadas van firmadas, incluso las de "info":
  - `X-API-KEY`: la key
  - `X-NONCE`: timestamp en ms
  - `X-SIGNATURE`: HMAC-SHA256 del nonce con el secret, en base64

| Método | Endpoint | Para qué |
|---|---|---|
| GET | `/accountSummary` | Saldos |
| GET | `/pairs` | Pares spot con mínimos y máximos |
| GET | `/quote?pair=BTC/USDT&amount=..&side=buy` | Cotización **spot** (precio para un monto) |
| POST | `/orders`, `/orders/cancel`, `/orders/cancel/all` | Órdenes spot |
| POST | `/orders/trigger`, `/orders/advanced`, `/orders/twap` | Stop-loss, take-profit, trailing y TWAP (spot) |
| GET | `/orders`, `/orderDetails`, `/trades`, `/transaction` | Historial |
| GET | `/futures/instruments` | Instrumentos de futuros (nombre y precisiones) |
| GET | `/futures/positions?status=active` | Posiciones: entrada, mark price, **precio de liquidación**, PnL, colateral, **apalancamiento** |
| POST | `/futures/order` | Abrir o cerrar: `instrument`, `positionAction` (open/close), `positionSide` (long/short), `type` (**solo `market`**), `quantity` |

La librería Python `python-nexo` (2022) lista los mismos endpoints de futuros,
más un "cerrar todas las posiciones" que marca como "no funciona".

## Respuestas

### 1. ¿Qué endpoints públicos de mercado tiene?

**No encontré ninguno confirmado.** En el código, hasta `GET /pairs`,
`GET /quote` y `GET /futures/instruments` van firmados con API key. Además:

- `GET /quote` da un precio **spot** agregado, no el mark price de futuros.
- No aparece ningún endpoint de ticker, mark price, libro de órdenes ni velas
  de futuros. El único lugar con mark price es `/futures/positions`, y solo
  muestra las posiciones abiertas.

**Consecuencia para la fase 1:** no se puede obtener el precio de futuros de
Nexo sin API key. Aun con una key de solo lectura, lo único confirmado es el
precio spot. El conector queda preparado (`backend/app/exchanges/nexo.py`) y
apagado (`exchanges.nexo.enabled: false`), y la app usa Binance con un aviso
visible. Hay una decisión que tomar sobre esto (ver "Decisiones pendientes").

### 2. ¿La API cubre los mismos futuros que uso desde la app de Nexo, o solo los de Nexo Pro?

**No confirmado.** Lo que se sabe:

- Es la misma cuenta y el mismo login. Nexo Pro usa "la misma cuenta y los
  mismos fondos" que Nexo.
- Los futuros de la app usan la **Futures Wallet** como colateral. Según la
  página de futuros de Nexo: hasta 100x (entre 10x y 100x según el contrato) y
  más de 100 contratos perpetuos "en la app o en la web".
- Fuentes más viejas describen los perpetuos de Nexo Pro con apalancamiento de
  hasta 2x. Eso sugiere que en su momento eran productos distintos.
- La API se llama "Nexo Pro API" y nada de lo que encontré dice que muestre las
  posiciones de la Futures Wallet de la app.

**Cómo confirmarlo en la fase 4:** con una key de solo lectura, abrir una
posición chica desde la app y ver si aparece en `GET /futures/positions`.

### 3. ¿Se puede configurar el apalancamiento por API?

**Todo indica que no, pero no está confirmado.**

- `POST /futures/order` no tiene parámetro de apalancamiento: solo
  instrumento, abrir/cerrar, long/short, tipo `market` y cantidad.
- Las posiciones **devuelven** `leverage`, pero no hay endpoint para fijarlo.
- La web de Nexo Pro dice que el apalancamiento de futuros se ajusta por par
  desde la plataforma web.

**Otro hallazgo que importa para la fase 4:** por API solo hay órdenes de
futuros **a mercado**. No aparecen stop-loss ni take-profit para futuros (sí
existen para spot). Si se confirma, el stop tendría que vigilarlo el bot desde
afuera, y eso es frágil: si el bot o la conexión se caen, la posición queda
sin stop. Hay que resolverlo antes de la fase 4.

### 4. Comisiones de futuros y fórmula de liquidación

**Comisiones:**

- Nexo publica una "tarifa estándar de 0,06 % para todos los usuarios" en
  futuros. Reseñas externas la describen como taker. El maker no está
  publicado con claridad. Hay descuentos por volumen de 30 días.
- Son perpetuos, así que además hay **funding** cada 8 h (Nexo no publica su
  propio histórico; se usa el de Binance como referencia).
- En `config.yaml`: `taker_pct: 0.06` y `maker_pct: 0.06` (conservador, porque
  el maker no está confirmado), ambos editables.

**Liquidación:** Nexo **no** usa margen aislado por posición. Usa la billetera
de futuros entera como colateral:

- Desde el 26/03/2025: **Margin Risk = Maintenance Margin / Equity de la
  Futures Wallet**. La liquidación se dispara cuando el Margin Risk llega a
  **100 %** (antes era 83,33 %). Se calcula con el **mark price**.
- Alertas: push al 65 % y email al 85 %.
- No encontré la tabla de maintenance margin por contrato ni una fórmula de
  precio de liquidación por posición.

**Qué hace la app:**

- Calcula la liquidación con **margen aislado**, como pide el diseño, con
  maintenance margin configurable (`liquidation.maintenance_margin_pct`, 0,5 %
  por defecto):
  - Long: `liq ≈ entrada × (1 − 1/apalancamiento + mmr)`
  - Short: `liq ≈ entrada × (1 + 1/apalancamiento − mmr)`
- Con más saldo en la Futures Wallet que el margen de la operación, la
  liquidación real queda **más lejos** que la calculada. Por eso el cálculo
  aislado es el conservador y es el que se usa para la regla "liquidación
  ≥ 2 × stop".
- La otra cara: en Nexo, una operación que sale mal puede comerse **toda** la
  billetera de futuros, no solo el margen. Ese riesgo lo vigilan el stop y los
  circuit breakers.

## Decisiones pendientes (para el usuario)

1. **Precio de Nexo:** sin endpoints públicos, ¿cómo seguimos? Opciones:
   Binance con aviso, cargar el precio de Nexo a mano, o una key de solo
   lectura (que rompe la regla de la fase 1).
2. **Modelo de liquidación:** ¿solo aislado (conservador) o también el modelo
   de billetera de Nexo como dato informativo?

## Fuentes

- [nexo-pro en npm](https://www.npmjs.com/package/nexo-pro) y
  [repositorio](https://github.com/aussedatlo/nexo-pro). Código fuente v1.2.2.
- [python-nexo](https://github.com/guilyx/python-nexo) y su
  [lista de endpoints](https://github.com/guilyx/python-nexo/blob/master/docs/endpoints.md).
- [Documentación oficial de Nexo Pro API](https://pro.nexo.com/apiDocPro.html)
  (no accesible desde el entorno de desarrollo).
- [Nexo Pro](https://pro.nexo.com/): ajuste de apalancamiento por par desde la web.
- [Nexo: futuros](https://nexo.com/futures): apalancamiento, tarifa 0,06 %,
  Futures Wallet y alertas de Margin Risk.
- [Nexo blog: nueva fórmula de Margin Risk](https://nexo.com/blog/enhanced-futures-margin-formula):
  Margin Risk = Maintenance Margin / Equity, liquidación al 100 % desde el 26/03/2025.
- [Nexo blog: guía de futuros](https://nexo.com/blog/futures-on-the-nexo-app-a-beginner-s-guide):
  liquidación por mark price.
- [Reseña de Nexo Pro (CaptainAltcoin)](https://captainaltcoin.com/nexo-pro-review/)
  y [comisiones de Nexo (Traders Union)](https://tradersunion.com/brokers/crypto/view/nexo/fees/):
  fuentes secundarias sobre comisiones.
