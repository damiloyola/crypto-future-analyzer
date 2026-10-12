# Crypto Future Analyzer

Aplicación web local para analizar criptomonedas antes de abrir operaciones de
futuros intradiarias (long/short). Corre en la Mac y se puede usar desde el
celular en la misma red wifi.

> **Herramienta de análisis personal. No es asesoramiento financiero.**
> La app no predice el futuro y **no ejecuta operaciones**: reúne datos
> objetivos, calcula niveles y riesgo con reglas fijas y presenta escenarios
> con probabilidades **estimadas**.

## Principios

1. Las reglas de riesgo las aplica **el código**, nunca el LLM. El LLM
   interpreta y redacta; no decide tamaños ni puede saltarse reglas.
2. **Riesgo en dólares primero, apalancamiento después**: el apalancamiento
   sale del cálculo, no se elige a mano.
3. **Cero look-ahead**: el detector de régimen usa solo datos hasta cada vela
   (algoritmo forward), con un test que lo verifica.
4. Los circuit breakers tienen **poder de veto** sobre cualquier plan.
5. Datos, indicadores, scoring y riesgo son **módulos reutilizables**, sin
   lógica en la UI, para montar encima las fases 2 a 4.

## Fases

| Fase | Qué es | Estado |
|---|---|---|
| 1 | Analizador visual (sin órdenes) | En construcción |
| 2 | Diario de operaciones con estadísticas | Pendiente |
| 3 | Bot en simulación con backtesting | Pendiente |
| 4 | Bot real conectado por API a Nexo | Pendiente |

### Hitos de la fase 1

| # | Hito | Estado |
|---|---|---|
| 0 | Esqueleto del proyecto | ✅ |
| 1 | Investigación de la Nexo Pro API ([informe](docs/nexo-api-research.md)) | ✅ |
| 2 | Datos e indicadores + script de consola ([fuentes](docs/data-sources.md)) | ✅ (falta correrlo en vivo en la Mac) |
| 3 | Detector de régimen (HMM) + `make train-regime` | ✅ (falta entrenarlo con datos reales en la Mac) |
| 4 | Score, plan de riesgo, ajustes por régimen y perfiles | ✅ (se ve en la consola; falta probarlo con datos reales en la Mac) |
| 5 | Circuit breakers y correlación | ⏳ |
| 6 | API FastAPI | ⏳ |
| 7 | Dashboard: régimen, KPIs y gráfico principal | ⏳ |
| 8 | Integración del LLM | ⏳ |
| 9 | Simulador, matemática de la billetera, paneles e historial | ⏳ |
| 10 | Validación histórica walk-forward | ⏳ |
| 11 | README y documentación completa | ⏳ |

## Requisitos

- macOS (probado para Apple Silicon) o Linux.
- **Python 3.11 a 3.13** (recomendado 3.12). Lo más simple: `brew install uv`;
  el `Makefile` usa `uv` si está instalado y se baja el Python correcto solo.
  Sin `uv` hace falta `python3` 3.11+ (`brew install python@3.12`).
- **Node.js 20.19+ o 22+** (`brew install node`).

## Uso

```bash
make setup   # instala backend y frontend; crea .env a partir de .env.example
# completá a mano ANTHROPIC_API_KEY (y opcional CRYPTOPANIC_API_KEY) en .env
make dev     # levanta backend y frontend juntos
```

- Dashboard: http://localhost:5173
- Desde el celular: `http://<ip-de-la-mac>:5173` (`make dev` muestra la IP).
  La primera vez, macOS puede pedir permiso para aceptar conexiones entrantes.
- API y documentación interactiva: http://localhost:8000/docs

Otros comandos: `make help`.

```bash
make test    # tests del backend (pytest) y del frontend (typecheck + vitest)
make lint    # ruff
```

### Script de consola (Hito 2)

Imprime todo lo que la app sabe de una moneda: precios, indicadores por
timeframe, soportes y resistencias, derivados, sentimiento, noticias, el score
de confluencia y el plan de operación (ver Hito 4).

```bash
make cli SYMBOL=BTC                          # datos reales (Binance, o Bybit si falla)
make cli SYMBOL=SOL ARGS="--nexo-price 187,45"   # con el precio que ves en Nexo
make cli SYMBOL=ETH ARGS="--profile moderado"
make cli SYMBOL=BTC ARGS="--offline"         # datos sintéticos, sin red (para probar)
make cli SYMBOL=BTC ARGS="--json"            # salida en JSON
```

Si Binance y Bybit no responden, termina con un mensaje claro (código 1). Si
falla una fuente secundaria (noticias, Fear & Greed), sigue y lo avisa.

### Detector de régimen (Hito 3)

Antes de cualquier señal, la app clasifica qué tipo de mercado hay (crash,
bajista, neutral, alcista, euforia) con un modelo estadístico (Gaussian HMM),
uno por moneda. **No predice el precio**: detecta el entorno. Usa velas de 1
hora (el régimen para operar) y velas diarias (el de contexto), siempre
cerradas y sin mirar el futuro: el régimen de cada vela se calcula solo con
las velas hasta ella (algoritmo forward; hay tests que lo verifican).

Mira el **resumen de las últimas 24 velas** (cuánto subió o bajó, rango y
volumen promedio, más la volatilidad), no cada vela suelta: así una vela
grande no alcanza para declarar un "Crash" y, en el régimen de 1 hora, cada
régimen dura del orden de un día (decisión tuya;
`regime.features.smoothing_window`, 1 = vela por vela). Ojo: el ajuste es uno
solo para los dos timeframes, así que en el régimen diario de contexto las 24
velas son **24 días**: mira la tendencia de las últimas tres semanas y media,
cada régimen dura de 2 a 4 semanas y hacen falta 190 días de calentamiento
(así lo elegiste: el diario da la foto grande y es solo informativo).

```bash
make train-regime                              # todas las favoritas, 1 hora y diario
make train-regime SYMBOL=SOL                   # una sola moneda
make train-regime SYMBOL=SOL ARGS="--force"    # reentrena aunque esté al día
make train-regime SYMBOL=BTC ARGS="--offline"  # datos sintéticos, sin red (para probar)
make cli SYMBOL=BTC                            # la consola ya muestra el régimen
make cli SYMBOL=BTC ARGS="--no-regime"         # sin régimen (más rápido)
```

- La cantidad de estados (de 3 a 5: Crash, Bajista, Neutral, Alcista y
  Euforia) la elige sola el BIC. Los modelos se guardan en `data/models/`
  (los de prueba, en `data/models/sintetico/`, y nunca se mezclan) y se
  reentrenan solos cuando tienen 7 días (`regime.auto_retrain`). Entrenar una
  moneda tarda alrededor de 1 minuto y cuarto (casi todo es el modelo de 1
  hora); si hace falta, la consola avisa antes.
- Qué muestra: régimen actual y si está confirmado (se mantuvo 3 velas
  seguidas) o hay un cambio en espera; confianza (la probabilidad del estado
  más probable en la última vela, "baja" si es menos de 60 %); "régimen
  inestable" si el tipo más probable cambió más de 4 veces en las últimas 20
  velas; hace cuánto empezó; y una tabla por régimen con retorno medio por
  vela, volatilidad, duración típica y % del tiempo. Avisa si el modelo es
  viejo, si se entrenó con poca historia o si las velas vinieron de Bybit.
- Resumen de los últimos 30 días: cuántas veces cambió el régimen confirmado,
  cuánto duraron en promedio los tramos completos (los que empezaron y
  terminaron dentro de los 30 días; con 0 o 1 cambio no hay ninguno y no se
  muestra duración), % del tiempo "inestable" y % con confianza baja. Con
  datos de prueba: un cambio cada ~33 h, inestable menos del 1 % del tiempo.
  Si con datos reales sale muy distinto, avisame.
- La confirmación y la inestabilidad cuentan cambios de tipo (Neutral →
  Bajista; `regime.change_level: category`). Con hasta 5 estados cada tipo es
  un solo estado, así que es lo mismo que contar cambios de estado (los
  nombres para 6 y 7 estados de `regime.labels` hoy no se usan).
- `make train-regime` termina con código 1 si alguna moneda no se pudo
  entrenar o reentrenar (en ese caso queda el modelo anterior y lo dice), y
  con 130 si lo cortás con Ctrl+C (sale enseguida y no guarda nada a medias).

### Score y plan de operación (Hito 4)

La consola termina con el **score de confluencia** (de -100 a +100, con lo que
suma o resta cada parte) y el **plan de operación**: un cartel grande LONG,
SHORT o NO OPERAR y, abajo, entrada, stop, objetivos, margen, tamaño,
apalancamiento, liquidación, comisiones y stop por tiempo. Es la misma función
que va a usar la app (`risk.plan.build_trade_plan`). Nunca opera: solo calcula.

```bash
make cli SYMBOL=BTC                                         # plan con el perfil por omisión (agresivo)
make cli SYMBOL=SOL ARGS="--nexo-price 187,45"              # niveles sobre el precio que ves en Nexo
make cli SYMBOL=ETH ARGS="--max-leverage 30"                # tope a mano, de 1x a 50x (más de 25x: advertencia roja)
make cli SYMBOL=BTC ARGS='--high-vol-event "Dato de CPI"'   # evento de alta volatilidad: No operar
make cli SYMBOL=BTC ARGS="--trades-today 2 --pnl-today -30" # cómo te fue hoy (el riesgo baja a lo que queda del límite)
make cli SYMBOL=BTC ARGS="--json"                           # el plan va en "plan" y el score en "score"
```

Cómo se arma, en pocas palabras (todos los números están en `config.yaml`):

- **Dirección**: long si el score es +30 o más, short si es -30 o menos; en el
  medio, No operar.
- **Stop**: debajo del soporte (o arriba de la resistencia) más cercano de 15m
  o 1h, con un colchón de media vela (0,5 ATR de 15m). Los niveles a menos de
  una vela normal (1 ATR) se saltean.
- **Objetivos**: TP1 en el siguiente soporte o resistencia (también salteando
  los que están a menos de 1 ATR; si ese nivel queda en el TP2 o más lejos, no
  hay TP1 y el primer objetivo es el TP2); TP2 en 1,5 veces lo que arriesgás,
  corrido lo justo para que, pagadas las comisiones, ganes al menos lo que pide
  tu perfil o el régimen (1,5 o 2 por cada 1 que arriesgás); TP3, un "R" más
  allá. Si el TP2 queda más lejos de lo que se mueve el precio en un día normal
  (ATR diario), No operar; los objetivos así se marcan "poco probable
  intradía". Si faltan las velas diarias para saberlo, también No operar.
- **Tamaño**: primero el riesgo en dólares (50 USD en el perfil agresivo, con
  las comisiones adentro), de ahí el tamaño, y el apalancamiento sale solo
  (tamaño / margen). Si pasa tu tope o la liquidación queda cerca del stop, la
  app sube el margen; si no alcanza, achica la posición; si el riesgo queda
  debajo del mínimo, No operar. Si la posición queda más chica que el margen
  (menos de 1x, pasa con stops muy lejos), el margen baja solo a 1x. Cada
  ajuste (también los del régimen) se muestra con su motivo. En Euforia, si no
  hay dato de funding, se toma como muy alto (riesgo × 0,5 para los longs).
- **Límite de pérdida del día**: el riesgo se achica a lo que te queda del
  límite (por ejemplo, si perdiste 80 de tus 100 USD, la operación arriesga 20
  en vez de 50), así un stop no te hace pasar el límite. Si lo que queda es
  menos que el mínimo operable (15 USD en el perfil agresivo), No operar hasta
  mañana. El tope vale también para los niveles de referencia de un plan
  bloqueado (por ejemplo, si ya hiciste todas las operaciones del día).
- **No operar** también si: 4h y 1h van en direcciones opuestas, el régimen lo
  bloquea (por ejemplo, long en Crash), el score no llega al mínimo que pide el
  régimen (por ejemplo, +50 para un long con el régimen Bajista), llegaste al
  límite de operaciones o de pérdida del día, marcaste un evento de alta
  volatilidad, o el precio de Nexo que cargaste difiere más de 2 % del de
  Binance (¿error de tipeo?: se muestran los dos precios y hay que corregirlo).
  Se listan todas las reglas que fallan, cada una con una sugerencia, y los
  niveles se muestran igual como "referencia, no operar" para que veas por qué.
- **Entrada alternativa**: una orden límite en el soporte (long) o la
  resistencia (short) más cercano de 15m, salteando los que están a menos de
  1 ATR del precio (pegados al precio sería casi lo mismo que entrar a
  mercado), con su propio stop, tamaño, ajustes, liquidación y comisiones.
- **Stop por tiempo**: si en 24 h no tocó el TP1 (o el TP2, si no hay TP1) ni
  el stop, cerrá (la hora se muestra en tu zona horaria). También estima el
  funding de esas 24 h (aparte del riesgo).

## Configuración

- **`config.yaml`**: todas las reglas, umbrales y pesos (perfiles de riesgo,
  ajustes por régimen, circuit breakers, comisiones, etc.), con comentarios. Se
  valida al arrancar: si algo no cierra, la app no levanta y dice por qué.
- **`.env`**: solo secretos (`ANTHROPIC_API_KEY`, `CRYPTOPANIC_API_KEY`). Está
  en `.gitignore`. En la fase 1 **no** se usan API keys de exchanges.

## Estructura

```
├── backend/
│   ├── app/
│   │   ├── config.py      # carga y valida config.yaml y .env
│   │   ├── main.py        # app FastAPI
│   │   ├── exchanges/     # interfaz común + Binance, Bybit (fallback), Nexo
│   │   ├── data/          # caché, Fear & Greed, CoinGecko, noticias, snapshot
│   │   ├── indicators/    # EMA, VWAP, RSI, MACD, ATR, Bollinger, S/R, tendencia, features
│   │   ├── regime/        # HMM: BIC, forward filtering, etiquetas, estabilidad, disco
│   │   ├── scoring/       # score de confluencia
│   │   ├── risk/          # plan, sizing, comisiones, liquidación, perfiles, simuladores
│   │   ├── safety/        # circuit breakers, lock file, incidentes, correlación
│   │   ├── validation/    # walk-forward, benchmarks, métricas, stress test
│   │   ├── llm/           # prompt, esquema y validación
│   │   ├── storage/       # SQLite
│   │   ├── services/      # orquesta un análisis completo
│   │   └── api/           # endpoints (sin lógica de negocio)
│   ├── scripts/           # CLI, entrenamiento del régimen, walk-forward
│   └── tests/
├── frontend/              # React + Vite + TypeScript + Tailwind
│   └── src/ (components, charts, hooks, pages, lib)
├── docs/                  # investigación y documentación
├── data/                  # SQLite, modelos, lock e informes (no se versiona)
├── config.yaml
├── .env.example
└── Makefile
```

Cada módulo tiene un docstring que explica qué hace y en qué hito se
implementa.

## Fuentes de datos

Públicas, sin cuenta de exchange: Binance (velas, funding, open interest, ratio
long/short), Bybit como fallback, alternative.me (Fear & Greed), CoinGecko
(dominancia), CryptoPanic o RSS de CoinDesk/Cointelegraph (noticias). El precio
de Nexo no tiene acceso público confirmado; ver
[docs/nexo-api-research.md](docs/nexo-api-research.md).
