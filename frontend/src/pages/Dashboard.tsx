import { Disclaimer } from "../components/Disclaimer";
import { HealthBadge } from "../components/HealthBadge";
import { SectionPlaceholder } from "../components/SectionPlaceholder";
import { useHealth } from "../hooks/useHealth";
import { SECTIONS } from "../lib/sections";

/**
 * Dashboard principal. En el Hito 0 es solo el esqueleto: header, disclaimer y
 * una tarjeta por sección indicando en qué hito se construye.
 */
export default function Dashboard() {
  const health = useHealth();
  const favorites = health.kind === "ok" ? health.data.favorites : [];

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Crypto Future Analyzer</h1>
          <p className="text-xs text-muted">Fase 1 · analizador intradiario · sin ejecución de órdenes</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {favorites.map((s) => (
            <span key={s} className="num rounded border border-border px-2 py-1 text-xs">
              {s}
            </span>
          ))}
          <button
            type="button"
            disabled
            className="rounded bg-entry/30 px-3 py-1 text-xs font-semibold text-text/60"
            title="Se habilita en el Hito 7"
          >
            Analizar
          </button>
        </div>
        <HealthBadge health={health} />
      </header>

      <Disclaimer />

      <main className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {SECTIONS.map((s) => (
          <SectionPlaceholder key={s.id} section={s} />
        ))}
      </main>
    </div>
  );
}
