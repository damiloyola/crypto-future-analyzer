"""Punto de entrada FastAPI.

``uvicorn app.main:app`` levanta la API. Acá solo se arma la aplicación:
CORS para la red local, routers de ``app.api`` y el endpoint de salud. Nada de
lógica de negocio.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import router as api_router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()  # falla al arrancar si config.yaml es inválido
    app = FastAPI(
        title="Crypto Future Analyzer",
        version=__version__,
        description=(
            "Analizador de criptomonedas para futuros intradiarios (fase 1). "
            "Herramienta de análisis personal: no ejecuta operaciones y no es "
            "asesoramiento financiero."
        ),
    )
    # La UI se sirve por el proxy de Vite (mismo origen), pero dejamos CORS
    # abierto a la red local para poder pegarle a la API desde otras herramientas.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    app.state.settings = settings
    return app


app = create_app()
