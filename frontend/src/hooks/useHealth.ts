import { useEffect, useState } from "react";

import { apiGet, type Health } from "../lib/api";

export type HealthState =
  | { kind: "loading" }
  | { kind: "ok"; data: Health; checkedAt: Date }
  | { kind: "error"; message: string; checkedAt: Date };

/** Consulta /api/health cada `intervalMs` para mostrar si el backend está vivo. */
export function useHealth(intervalMs = 15_000): HealthState {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    const ctrl = new AbortController();
    const check = () =>
      apiGet<Health>("/health", ctrl.signal)
        .then((data) => setState({ kind: "ok", data, checkedAt: new Date() }))
        .catch((err: unknown) => {
          if (ctrl.signal.aborted) return;
          setState({ kind: "error", message: String(err), checkedAt: new Date() });
        });
    check();
    const id = setInterval(check, intervalMs);
    return () => {
      ctrl.abort();
      clearInterval(id);
    };
  }, [intervalMs]);

  return state;
}
