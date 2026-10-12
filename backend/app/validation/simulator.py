"""Simulación de resultados de un plan sobre velas históricas.

Se implementa en el Hito 9 (historial) y se reutiliza en el Hito 10.

- ``first_touch(plan, candles_after)``: qué tocó primero el precio, el stop o
  el TP1 (u otros TPs), dentro de la ventana. Si una misma vela toca ambos, se
  asume el stop (conservador).
- Aplica comisiones y slippage (``validation.slippage_pct_per_side``).
"""
