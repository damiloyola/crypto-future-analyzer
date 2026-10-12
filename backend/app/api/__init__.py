"""Endpoints FastAPI. Delgados: validan la entrada y llaman a ``services``.

Routers previstos (Hito 6):

- ``GET  /api/health``: estado del backend (ya existe).
- ``GET  /api/config/public``: perfiles, favoritos y parámetros visibles.
- ``POST /api/analysis``: corre un análisis completo.
- ``GET  /api/market/{symbol}``: velas e indicadores para el gráfico.
- ``GET  /api/regime/{symbol}``: régimen y estadísticas.
- ``GET/POST /api/safety/...``: estado de breakers, cargar operaciones.
- ``GET  /api/history``: análisis guardados y "¿qué pasó después?".
- ``POST /api/simulate/...``: simulador de apalancamiento y Monte Carlo.
- ``GET  /api/validation/...``: resultados del walk-forward.
"""

from fastapi import APIRouter

from app.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)
