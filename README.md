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
| 4 | Score, plan de riesgo, ajustes por régimen y perfiles | ⏳ |
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
timeframe, soportes y resistencias, derivados, sentimiento, noticias y un
ejemplo didáctico de tamaño de posición, comisiones y liquidación.

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
