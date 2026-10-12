"""Ajustes del plan según el régimen, la confianza y la estabilidad.

Se implementa en el Hito 4.

``apply_regime_adjustments(direction, score, regime, funding, profile, settings)``
devuelve una lista de ``Adjustment`` (multiplicador de riesgo, score mínimo,
R:R mínimo o bloqueo), cada uno con su motivo legible, por ejemplo
"Riesgo × 0,5: régimen inestable". Los multiplicadores se acumulan; si el
riesgo final queda por debajo de ``min_operable_risk_usd`` el resultado es
"No operar".
"""
