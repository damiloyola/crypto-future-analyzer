import type { HealthState } from "../hooks/useHealth";

/** Indicador de conexión con el backend y hora de la última verificación. */
export function HealthBadge({ health }: { health: HealthState }) {
  if (health.kind === "loading") {
    return <span className="text-xs text-muted">Conectando con el backend…</span>;
  }
  const ok = health.kind === "ok";
  const time = health.checkedAt.toLocaleTimeString("es-AR");
  return (
    <span className="flex items-center gap-2 text-xs text-muted">
      <span className={`size-2 rounded-full ${ok ? "bg-bull" : "bg-bear"}`} />
      {ok ? `Backend v${health.data.version}` : "Backend sin respuesta"}
      <span className="num">· {time}</span>
    </span>
  );
}
