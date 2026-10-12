"""Score ponderado total.

Se implementa en el Hito 4.

``compute_score(inputs, settings) -> ConfluenceScore`` multiplica cada
componente por su peso (``scoring.weights``) y escala a [-100, +100].
Devuelve el total y el aporte de cada componente (para las barras divergentes
del dashboard).
"""
