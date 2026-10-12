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
| 2 | Datos e indicadores + script de consola | ⏳ |
| 3 | Detector de régimen (HMM) | ⏳ |
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
make cli SYMBOL=BTC   # (Hito 2) imprime datos e indicadores por consola
```

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
