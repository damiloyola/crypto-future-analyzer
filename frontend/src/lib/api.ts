/**
 * Cliente HTTP de la API. Todas las llamadas van a /api (mismo origen; Vite
 * las reenvía al backend). La UI no calcula nada de riesgo: solo muestra lo que
 * devuelve el backend.
 */

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api${path}`, { signal });
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export interface Health {
  status: "ok";
  version: string;
  phase: number;
  trading_enabled: boolean;
  server_time: string;
  default_profile: string;
  favorites: string[];
}
