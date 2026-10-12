/**
 * Mapa de las secciones del dashboard y el hito en el que se construye cada
 * una. Lo usa el esqueleto para mostrar qué falta; se va achicando a medida
 * que las secciones se implementan.
 */

export interface SectionInfo {
  id: string;
  title: string;
  description: string;
  milestone: number;
  wide?: boolean;
}

export const SECTIONS: SectionInfo[] = [
  { id: "regime", title: "Régimen de mercado", description: "Régimen 1h y diario, confianza, estabilidad y estado de los circuit breakers.", milestone: 7, wide: true },
  { id: "kpis", title: "KPIs", description: "Precio Nexo vs Binance, score, RSI, funding, OI, Fear & Greed, rango del día y estado del día.", milestone: 7, wide: true },
  { id: "chart", title: "Gráfico principal", description: "Velas, EMAs, VWAP, S/R, plan, liquidación y fondo por régimen. Volumen, RSI y MACD abajo.", milestone: 7, wide: true },
  { id: "plan", title: "Plan de operación", description: "LONG / SHORT / NO OPERAR, niveles, tamaño, apalancamiento resultante y reglas.", milestone: 7 },
  { id: "llm", title: "Análisis del LLM", description: "Resumen, puntos clave y riesgos. Estimación, no predicción.", milestone: 8 },
  { id: "scenarios", title: "Escenarios", description: "Probabilidades estimadas, disparadores, objetivos e invalidación.", milestone: 8 },
  { id: "leverage", title: "Simulador de apalancamiento", description: "5x a 100x con el mismo riesgo en USD.", milestone: 9 },
  { id: "wallet", title: "Matemática de la billetera", description: "Win rate de equilibrio, rachas y Monte Carlo.", milestone: 9 },
  { id: "regime-panel", title: "Panel de régimen", description: "Confianza en el tiempo, distribución y matriz de transición.", milestone: 9 },
  { id: "risk-controls", title: "Controles de riesgo", description: "Circuit breakers, pérdidas, drawdown y correlación.", milestone: 9 },
  { id: "score", title: "Desglose del score", description: "Aporte de cada componente.", milestone: 7 },
  { id: "mtf", title: "Alineación multi-timeframe", description: "15m, 1h, 4h y 1d: tendencia, RSI, EMAs y VWAP.", milestone: 7 },
  { id: "derivatives", title: "Derivados", description: "Funding, open interest y ratio long/short de 7 días.", milestone: 7 },
  { id: "news", title: "Noticias y eventos", description: "Titulares con impacto y eventos de las próximas 24 h.", milestone: 8 },
  { id: "history", title: "Historial", description: "Análisis guardados y ¿qué pasó después?", milestone: 9, wide: true },
  { id: "validation", title: "Validación walk-forward", description: "¿El score le gana al azar?", milestone: 10, wide: true },
];
