/**
 * Colores compartidos por los gráficos (Lightweight Charts y Recharts no leen
 * clases de Tailwind, así que necesitan los valores en JS). Mantener en
 * sincronía con los tokens de index.css.
 */

export type RegimeCategory = "crash" | "bajista" | "neutral" | "alcista" | "euforia";

export const REGIME_COLORS: Record<RegimeCategory, string> = {
  crash: "#7f1d1d",
  bajista: "#dc2626",
  neutral: "#6b7280",
  alcista: "#16a34a",
  euforia: "#8b5cf6",
};

export const COLORS = {
  bg: "#0b0e14",
  panel: "#121722",
  border: "#243044",
  text: "#e6edf6",
  muted: "#8a97ab",
  bull: "#22c55e",
  bear: "#ef4444",
  neutral: "#f59e0b",
  entry: "#3b82f6",
  stop: "#ef4444",
  takeProfit: "#22c55e",
  liquidation: "#ef4444",
} as const;
