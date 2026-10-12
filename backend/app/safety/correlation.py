"""Correlación entre monedas y riesgo combinado.

Se implementa en el Hito 5.

- ``correlation_matrix(returns_1h, days=30)``: matriz de correlación de
  retornos de 1h.
- ``combined_risk_check(new_plan, open_positions, matrix, settings)``: si hay
  una posición abierta en la misma dirección en una moneda con correlación
  >= 0,8, los riesgos se suman y no pueden superar 1,5 × el riesgo por
  operación.
"""
