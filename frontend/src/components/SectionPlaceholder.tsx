import type { SectionInfo } from "../lib/sections";

/** Tarjeta vacía para las secciones que todavía no se construyeron. */
export function SectionPlaceholder({ section }: { section: SectionInfo }) {
  return (
    <section
      className={`rounded-lg border border-dashed border-border bg-panel p-4 ${
        section.wide ? "md:col-span-2" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-sm font-semibold">{section.title}</h2>
        <span className="num shrink-0 rounded bg-panel-2 px-2 py-0.5 text-[10px] text-muted">
          Hito {section.milestone}
        </span>
      </div>
      <p className="mt-2 text-xs text-muted">{section.description}</p>
    </section>
  );
}
