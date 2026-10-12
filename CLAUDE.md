# CLAUDE.md

Guía para retomar el proyecto. El README tiene el contexto general y el estado
de cada hito.

## Reglas que no se negocian

- **La app no opera en la fase 1.** Nada de endpoints privados ni API keys de
  exchanges. `OrderExecutor` existe solo como interfaz; cualquier llamada lanza
  `TradingDisabledError`. `exchanges.nexo.allow_orders` tiene que ser `false`
  (la validación de config lo exige).
- **Las reglas de riesgo las aplica el código, nunca el LLM.** El LLM recibe
  números ya calculados y devuelve texto en JSON validado; no puede cambiar
  niveles, tamaños ni bloqueos.
- **Cero look-ahead en el régimen:** solo algoritmo forward. Nunca
  `model.predict`, `decode` ni `predict_proba` sobre la serie completa. Los
  pivotes de S/R se confirman con `pivot_right` velas de retraso.
- **Todos los umbrales van en `config.yaml`**, con su modelo en
  `backend/app/config.py` (`extra="forbid"`). No hardcodear números de riesgo.
- **Las API keys las escribe el usuario en `.env`.** Nunca pedirlas ni
  mostrarlas.
- La UI está en **español rioplatense**. El código y las claves de config en
  inglés; docstrings y comentarios en español.
- Cada hito termina con sus tests pasando. Los tests no usan la red (fixtures en
  `backend/tests/fixtures/`).
- Si hay una decisión de diseño importante, preguntarle al usuario antes.

## Comandos

```bash
make setup         # dependencias
make dev           # backend :8000 + frontend :5173 (red local)
make test          # pytest + typecheck + vitest
make lint          # ruff check + format --check
make train-regime SYMBOL=BTC ARGS="--offline"   # modelos de régimen (sin SYMBOL: todas)
```

## Arquitectura

Capas en `backend/app/`, cada una depende solo de las de abajo:
`exchanges` → `data` → `indicators` → `regime` → `scoring` → `risk` →
`safety` → `llm` → `storage` → `services` → `api`. `validation` reutiliza
`indicators`, `regime`, `scoring`, `risk` y `safety` (la misma función
`risk.plan.build_trade_plan` que usa la app).

El frontend solo muestra lo que devuelve la API; no calcula riesgo.

## Decisiones del usuario (no cambiarlas sin preguntar)

- **Precio de ejecución:** Binance, salvo que el usuario pegue a mano el precio
  de Nexo (`registry.get_execution_price`). Nexo no tiene endpoints públicos.
- **Liquidación:** las reglas usan margen aislado (conservador); el "modelo
  billetera" de Nexo se muestra solo como dato.
- **Velas:** indicadores solo con velas cerradas (`timeframes.closed_candles_only`).
- **Riesgo con comisiones:** los 50 USD de riesgo incluyen las comisiones de
  entrada y salida (`trade_plan.fees_included_in_risk`).
- **Stop:** se saltean los soportes/resistencias a menos de 1 ATR(15m) de la
  entrada (`trade_plan.min_stop_distance_atr`, `risk.stops.structural_stop`).
- **Noticias:** a las altcoins se les suman las de BTC; corte de 48 h.
- **Apalancamiento por encima del tope del perfil (Hito 4):** ajustar solo.
  Primero subir el margen hasta `margen × trade_plan.liquidation.max_margin_multiplier`
  manteniendo el riesgo; si no alcanza, achicar la posición hasta el tope de
  apalancamiento (el riesgo baja); si el riesgo queda debajo de
  `min_operable_risk_usd`, "No operar". Mostrar cada ajuste con su motivo.
- **Régimen (Hito 3):** features resumidas en las últimas 24 velas
  (`regime.features.smoothing_window: 24`; vela por vela daba Crash/Euforia de
  2 h e "inestable" la mitad del tiempo), máximo 5 estados (Crash, Bajista,
  Neutral, Alcista, Euforia), reentrenamiento automático semanal al analizar,
  inestabilidad y confirmación por cambio de **tipo** (`change_level:
  category`). Confianza = probabilidad del estado más probable en la última
  vela (con 5 estados cada tipo es un solo estado).
- Al usuario hay que explicarle todo en palabras simples (hace mucho que no
  opera futuros) y hacerle las preguntas con opciones (AskUserQuestion).

## Cómo probar sin red

El contenedor de desarrollo no llega a las APIs. `make cli SYMBOL=BTC
ARGS="--offline"` usa datos sintéticos (`app/data/synthetic.py`) por el mismo
camino de código; los tests usan fixtures (`backend/tests/fixtures/README.md`).
En `--offline` los modelos de régimen van a `regime.synthetic_models_dir` y
nunca pisan a los reales; la primera corrida entrena (alrededor de 1 minuto y
cuarto por moneda con la config real). Los tests del régimen usan configs chicas
(`settings.regime.model_copy(update=...)`) para entrenar en menos de un
segundo.
