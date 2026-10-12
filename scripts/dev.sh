#!/usr/bin/env bash
# Levanta backend (FastAPI) y frontend (Vite) juntos. Ctrl+C corta los dos.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PY="$ROOT/backend/.venv/bin/python"

if [[ ! -x "$PY" || ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Falta instalar dependencias. Corré: make setup" >&2
  exit 1
fi

cleanup() {
  trap - INT TERM EXIT
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(cd "$ROOT/backend" && "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload) &
(cd "$ROOT/frontend" && BACKEND_PORT="$BACKEND_PORT" FRONTEND_PORT="$FRONTEND_PORT" npm run dev --silent) &

LAN_IP="$( (ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') || true)"
sleep 2
echo
echo "  Dashboard en la Mac:     http://localhost:$FRONTEND_PORT"
[[ -n "${LAN_IP:-}" ]] && echo "  Desde el celular (wifi): http://$LAN_IP:$FRONTEND_PORT"
echo "  API y documentación:     http://localhost:$BACKEND_PORT/docs"
echo

wait
