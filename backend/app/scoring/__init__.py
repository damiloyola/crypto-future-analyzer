"""Score de confluencia de -100 a +100 con el aporte de cada componente.

- ``components``: cada componente normalizado a [-1, +1] (tendencia, momentum,
  derivados, sentimiento) con su explicación en castellano.
- ``confluence``: ``ScoreInputs``, ``inputs_from_analysis`` y
  ``compute_score`` → ``ConfluenceScore`` (total, puntos por componente,
  faltantes y pista de dirección según ``scoring.direction_thresholds``).

Todos los pesos y umbrales salen de ``config.yaml > scoring``.
"""

from app.scoring.components import ComponentReading
from app.scoring.confluence import (
    ComponentScore,
    ConfluenceScore,
    ScoreInputs,
    compute_score,
    inputs_from_analysis,
)

__all__ = [
    "ComponentReading",
    "ComponentScore",
    "ConfluenceScore",
    "ScoreInputs",
    "compute_score",
    "inputs_from_analysis",
]
