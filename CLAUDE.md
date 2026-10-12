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
```

## Arquitectura

Capas en `backend/app/`, cada una depende solo de las de abajo:
`exchanges` → `data` → `indicators` → `regime` → `scoring` → `risk` →
`safety` → `llm` → `storage` → `services` → `api`. `validation` reutiliza
`indicators`, `regime`, `scoring`, `risk` y `safety` (la misma función
`risk.plan.build_trade_plan` que usa la app).

El frontend solo muestra lo que devuelve la API; no calcula riesgo.
