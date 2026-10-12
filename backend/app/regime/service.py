"""API de alto nivel del régimen para una moneda.

Se implementa en el Hito 3.

``get_regime(symbol, candles_1h, candles_1d, settings) -> RegimeReport``:
carga o entrena los modelos (1h y diario), calcula probabilidades filtradas,
aplica el filtro de estabilidad y devuelve régimen actual, confianza,
antigüedad, si está confirmado o inestable, la serie histórica (para el fondo
del gráfico y la línea de confianza) y las estadísticas por régimen.
"""
