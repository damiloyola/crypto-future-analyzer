"""``GET /api/health``: el backend está vivo y la configuración es válida."""

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from app import __version__

router = APIRouter(tags=["sistema"])


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "version": __version__,
        "phase": 1,
        "trading_enabled": False,
        "server_time": datetime.now(UTC).isoformat(),
        "default_profile": settings.risk_profiles.default,
        "favorites": settings.symbols.favorites,
    }
