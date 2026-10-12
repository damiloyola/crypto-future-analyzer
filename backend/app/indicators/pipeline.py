"""Cálculo de todos los indicadores para un timeframe.

Se implementa en el Hito 2.

``analyze_timeframe(df, tf, settings) -> TimeframeAnalysis`` devuelve el
DataFrame con columnas de indicadores (para el gráfico) y un resumen de la
última vela: precio, EMAs, VWAP, RSI, MACD, ATR, Bollinger, volumen relativo,
tendencia y niveles de S/R. ``analyze_all(snapshot, settings)`` lo corre para
todos los timeframes.
"""
